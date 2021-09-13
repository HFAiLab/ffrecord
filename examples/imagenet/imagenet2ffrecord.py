from PIL import Image
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import pickle
import torch
from tqdm import tqdm

from ffrecord import FileWriter
from ffrecord.torch import Dataset, DataLoader


class DumpDataset(torchvision.datasets.ImageNet):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        data = pickle.dumps(sample)
        return data


def dump_imagenet(split, out_file):
    batch_size = 32
    data_dir = '/public_dataset/2/ImageNet'
    dataset = DumpDataset(data_dir, split=split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x,
        num_workers=64,
    )

    n = len(dataset)
    print(f'dumping {n} samples to {out_file}')
    writer = FileWriter(out_file, n)


    for samples in tqdm(loader):
        for sample in samples:
            writer.write_one(sample)

    writer.close()


if __name__ == '__main__':
    import ffrecord
    print(ffrecord.FileReader.validate)
    dump_imagenet('val', '/private_dataset/ImageNet/val.ffr')
    dump_imagenet('train', '/private_dataset/ImageNet/train.ffr')
