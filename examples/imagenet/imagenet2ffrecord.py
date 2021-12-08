import os
from PIL import Image
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import Subset

from ffrecord import FileWriter
from ffrecord.torch import Dataset, DataLoader


class DumpDataset(torchvision.datasets.ImageNet):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        data = pickle.dumps(sample)
        return data


def dump_imagenet(split, out_dir, nfiles):
    # we recommend users to split data into >= 16 files
    batch_size = 32
    data_dir = '/public_dataset/2/ImageNet'
    dataset = DumpDataset(data_dir, split=split)

    # split data int into multiple files
    n = len(dataset)
    chunk_size = (n + nfiles - 1) // nfiles

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cnt = 0
    for i0 in range(0, n, chunk_size):
        ni = min(n - i0, chunk_size)
        indices = list(range(i0, i0 + ni))
        subdataset = Subset(dataset, indices)
        assert len(subdataset) == ni

        loader = torch.utils.data.DataLoader(
            subdataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: x,
            num_workers=64,
        )

        out_file = os.path.join(out_dir, f'{cnt:05d}.ffr')
        print(f'dumping {ni} samples to {out_file}')
        writer = FileWriter(out_file, ni)

        for samples in tqdm(loader):
            for sample in samples:
                writer.write_one(sample)

        writer.close()
        cnt += 1


if __name__ == '__main__':
    dump_imagenet('val', '/private_dataset/ImageNet/val.ffr', nfiles=50)
    dump_imagenet('train', '/private_dataset/ImageNet/train.ffr', nfiles=50)
