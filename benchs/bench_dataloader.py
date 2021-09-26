# import hf_env
# hf_env.set_env('202105')

import time
import os
import pickle
from pathlib import Path
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms

from ffrecord.torch import Dataset, DataLoader


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


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def bench_speed(dataloader, name, bs):
    it = iter(dataloader)
    next(it)

    num_batches = 1000

    t0 = time.time()
    for i in range(num_batches):
        samples = next(it)
    t1 = time.time()

    t = (t1 - t0) / num_batches
    print(f'batch size {bs}, {name}: {t} s per batch')
    return t


def bench(batch_size):
    num_workers = 8

    train_data = '/private_dataset/ImageNet/train.ffr'
    ffdataset = FireFlyerImageNet(train_data, transform=transform)
    ffdataloader = DataLoader(ffdataset,
                              batch_size,
                              num_workers=num_workers,
                              pin_memory=True)

    data_dir = '/public_dataset/2/ImageNet'
    torchdataset = datasets.ImageNet(data_dir, split='train', transform=transform)
    torchdataloader = torch.utils.data.DataLoader(torchdataset,
                                               batch_size,
                                               num_workers=num_workers,
                                               pin_memory=True)

    t0 = bench_speed(ffdataloader, 'FFRecord', batch_size)
    t1 = bench_speed(torchdataloader, 'PyTorch', batch_size)
    return t0, t1


def main():

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    times = []
    times0, times1 = [], []

    for batch_size in batch_sizes:
        t0, t1 = bench(batch_size)
        times0.append(t0)
        times1.append(t1)

    plt.plot(batch_sizes, times0, label='FFRecord DataLoader')
    plt.plot(batch_sizes, times1, label='PyTorch DataLoader')
    plt.title('DataLoaders on ImageNet')
    plt.xlabel('batch size')
    plt.ylabel('seconds per batch')
    plt.legend()
    plt.savefig('bench.png')


if __name__ == '__main__':
    main()
