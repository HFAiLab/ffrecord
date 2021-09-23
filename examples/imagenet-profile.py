# import hf_env
# hf_env.set_env('202105')
import sys

# sys.path.insert(
#     0,
#     "/nfs-jd/prod/nfs1/zsy/pylibs/lib/python3.6/site-packages/ffrecord-1.1.0+fdc638b-py3.8.egg"
# )

# from viztracer import VizTracer

# tracer = VizTracer()
# tracer.start()

from time import perf_counter
from PIL import Image
from pathlib import Path

from numpy.core.fromnumeric import trace
import torchvision
import torchvision.transforms as transforms
import pickle
import torch
import os

from ffrecord import FileWriter
from ffrecord.torch import Dataset, DataLoader

file_path = "/fs-jd/prod/private/zsy/dataloader_test/imagenet_train.ffr"


class FireFlyerImageNet(Dataset):
    def __init__(self, fname, transform=None):
        super(FireFlyerImageNet, self).__init__(fname, check_data=False)
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


def load_imagenet():

    mean = [0.4753, 0.4495, 0.4115]
    std = [0.2638, 0.2566, 0.2720]
    transofrm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = FireFlyerImageNet(file_path, transofrm)
    print(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=True,
        num_workers=8,
    )

    file_size = os.stat(file_path).st_size
    i = 0
    for epoch in range(1):
        t = perf_counter()
        for img, label in dataloader:
            print(img.is_shared(), label.is_shared())
            i += 100
            if i == 6400:
                break

        print(
            f"read time: {perf_counter() - t}s, bw: {file_size / 5 / (perf_counter() - t) / (1 << 30):.2f}GBps"
        )
    del dataloader


if __name__ == '__main__':
    load_imagenet()

    # tracer.stop()
    # tracer.save()
