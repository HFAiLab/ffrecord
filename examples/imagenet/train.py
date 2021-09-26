import hf_env

hf_env.set_env('202105')

import os
import time
import pickle
from pathlib import Path
import torch
import torchvision
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models

from ffrecord.torch import Dataset, DataLoader

import hfai
import hfai.nccl.distributed as dist
from torch.multiprocessing import Process

hfai.client.bind_hf_except_hook(Process)


class FireFlyerImageNet(Dataset):
    def __init__(self, fname, transform=None):
        super(FireFlyerImageNet, self).__init__(fname, check_data=True)
        self.transform = transform

    def process(self, indexes, data):
        samples = []

        for bytes_ in data:
            img, label = pickle.loads(bytes_)
            if self.transform:
                img = self.transform(img)
            samples.append((img, label))

        # default collate_fn would handle them
        return samples


def train(dataloader, model, criterion, optimizer, scheduler, epoch, local_rank,
          start_step, best_acc, save_path):
    model.train()
    for step, batch in enumerate(dataloader):
        if step < start_step:
            continue

        samples, labels = [x.cuda(non_blocking=True) for x in batch]
        outputs = model(samples)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if local_rank == 0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

        # 保存
        rank = torch.distributed.get_rank()
        if rank == 0 and hfai.receive_suspend_command():
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'step': step + 1
            }
            torch.save(state, os.path.join(save_path, 'latest.pt'))
            time.sleep(5)
            hfai.go_suspend()


def validate(dataloader, model, criterion, epoch, local_rank):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            samples, labels = [x.cuda(non_blocking=True) for x in batch]
            outputs = model(samples)
            loss += criterion(outputs, labels)
            _, preds = outputs.topk(5, -1, True, True)
            correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
            correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
            total += samples.size(0)

    for x in [loss, correct1, correct5, total]:
        dist.reduce(x, 0)

    if local_rank == 0:
        loss_val = loss.item() / dist.get_world_size() / len(dataloader)
        acc1 = 100 * correct1.item() / total.item()
        acc5 = 100 * correct5.item() / total.item()
        print(f'Epoch: {epoch}, Loss: {loss_val}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)

    return correct1.item() / total.item()


def main(local_rank):
    # 超参数设置
    epochs = 100
    batch_size = 50
    num_workers = 4
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    save_path = 'output/resnet'
    Path(save_path).mkdir(exist_ok=True, parents=True)

    train_data = '/private_dataset/ImageNet/train.ffr'
    val_data = '/private_dataset/ImageNet/val.ffr'

    # 多机通信
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # 机器个数
    rank = int(os.environ['RANK'])  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的GPU个数

    # world_size是全局GPU个数，rank是当前GPU全局编号
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://{ip}:{port}',
                            world_size=hosts * gpus,
                            rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    # 模型、数据、优化
    model = models.resnet50()
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])  # 定义训练集变换
    train_dataset = FireFlyerImageNet(train_data, transform=train_transform)
    train_datasampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size,
                                  sampler=train_datasampler,
                                  num_workers=num_workers,
                                  pin_memory=True)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])  # 定义测试集变换
    val_dataset = FireFlyerImageNet(val_data, transform=val_transform)
    val_datasampler = DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,
                                batch_size,
                                sampler=val_datasampler,
                                num_workers=num_workers,
                                pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # 加载
    best_acc, start_epoch, start_step = 0, 0, 0
    if os.path.exists(os.path.join(save_path, 'latest.pt')):
        ckpt = torch.load(os.path.join(save_path, 'latest.pt'),
                          map_location='cpu')
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_acc, start_epoch = ckpt['acc'], ckpt['epoch']
        start_step = ckpt['step']

    # 训练、验证
    for epoch in range(start_epoch, epochs):
        train_datasampler.set_epoch(epoch)
        train(train_dataloader, model, criterion, optimizer, scheduler, epoch, local_rank,
              start_step, best_acc, save_path)
        start_step = 0  # reset
        scheduler.step()
        acc = validate(val_dataloader, model, criterion, epoch, local_rank)
        torch.cuda.empty_cache()
        # 保存
        if rank == 0 and local_rank == 0:
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch + 1,
                'step': 0
            }
            torch.save(state, os.path.join(save_path, 'latest.pt'))
            if acc > best_acc:
                best_acc = acc
                print(f'New Best Acc: {100*acc:.2f}%!')
                torch.save(model.module.state_dict(),
                           os.path.join(save_path, 'best.pt'))


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
