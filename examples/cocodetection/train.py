import hf_env
hf_env.set_env('202105')
import hfai

import datetime
import builtins
import os
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process

import hfai
import hfai.nccl.distributed as dist
hfai.client.bind_hf_except_hook(Process)

from net.ssd import SSD300, MultiBoxLoss
from ffdataloader import CoCoDataLoader


def init_dist_process(args):
    ip = os.environ['MASTER_ADDR']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # 机器个数
    rank = int(os.environ['RANK'])  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的GPU个数, 8
    # assert gpus == 8

    # multi-GPU job (local or multi-node)
    args.rank = rank * gpus + args.gpu  # 全局GPU编号
    args.world_size = hosts * gpus  # 全局GPU数量
    args.dist_url = f'tcp://{ip}:{port}'

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    torch.cuda.set_device(args.gpu)
    print('rank', args.rank, 'pid', os.getpid(), flush=True)

    # only gpu0 would print
    if args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass
    else:
        builtins.print_c = print

        # 设置flush=True
        def print_and_flush(*args):
            builtins.print_c(*args, flush=True)

        builtins.print = print_and_flush

    return args


def reduce_scalar(x):
    if isinstance(x, torch.Tensor):
        x = x.item()
    x = torch.tensor(x).cuda()
    torch.distributed.all_reduce(x)
    world_size = torch.distributed.get_world_size()
    x = x / world_size
    return x.item()


class Timer(object):
    def __init__(self):
        self.total_cost = 0

    def time(self, f):
        start = time.time()
        r = f()
        torch.cuda.synchronize()
        fin = time.time()
        self.total_cost += fin - start
        return r

    def __float__(self):
        return self.total_cost

    def __format__(self, format_spec):
        #print('formatting timer with', format_spec, flush=True)
        return ('{:' + format_spec + '}').format(float(self))

    def reset(self):
        self.total_cost = 0


def sync_worker():
    # 同步worker
    dist.all_reduce(torch.randn(1).cuda())
    torch.cuda.synchronize()


timers = defaultdict(Timer)


def train_for_one_step(model, criterion, optimizer, inputs, b_boxes, b_labels,
                       num_objs):
    global timers
    optimizer.zero_grad()

    inputs = inputs.cuda()
    boxes = [b.cuda() for b in b_boxes]
    labels = [b.cuda() for b in b_labels]

    # forward + backward + optimize
    # 用timer记录每一步的时间
    pred_locs, pred_cls_prob = timers['forward'].time(lambda: model(inputs))
    loss = timers['loss'].time(lambda: criterion(pred_locs, pred_cls_prob, boxes, labels))
    timers['backward'].time(lambda: loss.backward())
    timers['gap'].time(lambda: sync_worker())
    timers['optimize'].time(lambda: optimizer.step())

    loss = timers['reduce_loss'].time(lambda: reduce_scalar(loss))

    return loss


def train_for_one_epoch(model,
                        criterion,
                        optimizer,
                        loader,
                        epoch,
                        step_per_epoch,
                        args,
                        writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(loader), [batch_time, data_time, losses],
                             prefix="Epoch: [{}]".format(epoch))

    step = 0
    print_freq = 10

    global timers
    end = time.time()
    for images, boxes, labels, num_obj_in_images in loader:
        # 记录加载一个batch数据的时间
        data_time.update(time.time() - end)

        if step < args.start_step:
            step += 1
            continue

        loss = train_for_one_step(model, criterion, optimizer, images, boxes,
                                  labels, num_obj_in_images)

        losses.update(loss, images.size(0))

        # 记录训练一个batch的时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 保存模型参数
        save_checkpoint_step(epoch, step + 1, 0, model, optimizer, args)

        if writer is not None:
            writer.add_scalar("Loss/train", loss,
                              step + epoch * step_per_epoch)
        if step % print_freq == 0:
            progress.display(step)
            print('forward: {forward:.2f} loss: {loss:.2f} backward: {backward:.2f} gap: {gap:.2f} ' \
                  'optimize: {optimize:.2f}　reduce_loss: {reduce_loss:.2f}'.format(**timers))
            for t in timers.values():
                t.reset()

        step += 1
    
    # reset
    args.start_step = 0


@torch.no_grad()
def eval_for_one_epoch(model,
                       criterion,
                       loader,
                       epoch,
                       step_per_epoch,
                       writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Val Loss', ':.4e')
    progress = ProgressMeter(len(loader), [batch_time, data_time, losses],
                             prefix="Epoch: [{}]".format(epoch))

    step = 0
    print_freq = 10

    end = time.time()
    for images, boxes, labels, num_obj_in_images in loader:
        data_time.update(time.time() - end)

        pred_locs, pred_cls_prob = model(images.cuda())

        boxes = [b.cuda() for b in boxes]
        labels = [l.cuda() for l in labels]

        loss = criterion(pred_locs.cuda(), pred_cls_prob.cuda(), boxes, labels)

        # 把每个worker的loss值取平均并记录下来
        loss = reduce_scalar(loss)
        losses.update(loss, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end) 
        end = time.time()

        if step % print_freq == 0:
            progress.display(step)

        step += 1

    loss = losses.avg

    if writer is not None:
        writer.add_scalar("Loss/avg_val", loss, epoch)
    print(f"Val Loss: {loss:.4f}")

    return loss


def train_loop(training_setup, args, num_epochs):
    model = training_setup['model']
    criterion = training_setup['criterion']
    optimizer = training_setup['optimizer']
    train_loader = training_setup['loader'].train_loader
    test_loader = training_setup['loader'].test_loader
    writer = training_setup['writer']
    args.steps_per_epoch = len(train_loader)

    for epoch in range(args.start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)

        # 训练一个epoch
        model.train()
        train_for_one_epoch(model,
                            criterion,
                            optimizer,
                            train_loader,
                            epoch,
                            len(train_loader),
                            args,
                            writer=writer)

        # 在验证集上评估
        model.eval()
        eval_loss = eval_for_one_epoch(model,
                                       criterion,
                                       test_loader,
                                       epoch,
                                       len(test_loader),
                                       writer=writer)

        # 保存模型
        save_checkpoint(epoch, 0, eval_loss, model, optimizer, args)


def save_checkpoint(epoch, step, eval_loss, model, optimizer, args):
    if args.rank != 0:
        return

    state = {
        'epoch': epoch + 1,
        'step': step,
        'eval_loss': eval_loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    # in case of corruption
    ckpt_path = args.ckpt_dir / f'latest_new.pt'
    torch.save(state, ckpt_path)
    latest_path = args.ckpt_dir / f'latest.pt'
    ckpt_path.rename(str(latest_path))

    if epoch % args.save_per_epochs == 0:
        ckpt_path = args.ckpt_dir / f'{epoch:03d}.pt'
        torch.save(state, ckpt_path)


def save_checkpoint_step(epoch, step, eval_loss, model, optimizer, args):
    if args.rank != 0 or not hfai.receive_suspend_command():
        return

    state = {
        'epoch': epoch,
        'step': step,
        'eval_loss': eval_loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    steps = args.steps_per_epoch * epoch + step
    hfai.set_whole_life_state(steps)

    # 先保存到latest_new.pt，然后再重命名为latest.pt
    # 防止中途挂掉导致原来的checkpoint被污染
    ckpt_path = args.ckpt_dir / f'latest_new.pt'
    torch.save(state, ckpt_path)
    latest_path = args.ckpt_dir / f'latest.pt'
    ckpt_path.rename(str(latest_path))

    time.sleep(5)
    hfai.go_suspend()


def setting_up(args):
    setup = {}
    model = SSD300(args.backbone, args.num_classes)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.gpu])
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    ckpt_path = args.ckpt_dir / f'latest.pt'
    if ckpt_path.exists():
        print("=> loading checkpoint '{}'".format(ckpt_path))
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        state = torch.load(ckpt_path, map_location=loc)
        args.start_epoch = state['epoch']  # 开始的epoch
        args.start_step = state['step']    # 开始的step
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print(f"=> loaded checkpoint '{ckpt_path}' (epoch {args.start_epoch}, step {args.start_step})")
    else:
        args.start_epoch = 0
        args.start_step = 0
        args.best_ap = 0
        print(f"=> no checkpoint found at {ckpt_path}")

    # 模型
    setup['model'] = model

    # 优化器
    setup['optimizer'] = optimizer

    # dataloader
    setup['loader'] = CoCoDataLoader(args.data_dir, args.batch_size,
                                     args.num_workers)

    # 损失函数
    prior_boxes = model.module.priors_cxcy
    setup['criterion'] = MultiBoxLoss(prior_boxes)

    if args.rank == 0:
        setup['writer'] = SummaryWriter(args.log_dir)
    else:
        setup['writer'] = None

    return setup


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Config(dict):
    pass


def get_args():
    """
    模型参数、训练参数等等
    """
    args = Config()
    args.data_dir = Path('data/coco')
    args.log_dir = Path('log/run')
    args.ckpt_dir = Path('log/checkpoint')
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # args.backbone = 'data/vgg16-397923af.pth'
    args.backbone = None

    args.num_classes = 91
    args.num_epochs = 100
    args.lr = 0.01
    args.batch_size = 64
    args.num_workers = 8
    args.save_per_epochs = 10

    return args


def main_worker(gpu, args):
    args.gpu = gpu

    # 初始化分布式环境
    args = init_dist_process(args)

    # 创建模型、优化器、dataloader
    setup = setting_up(args)

    # 开始训练
    train_loop(setup, args, num_epochs=args.num_epochs)


def main():
    args = get_args()
    ngpus_per_node = torch.cuda.device_count()
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, ))


if __name__ == '__main__':
    main()
