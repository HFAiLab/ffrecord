from PIL import Image
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import pickle
import torch

from ffrecord import FileWriter
from ffrecord.torch import Dataset, DataLoader


class DumpDataset(torchvision.datasets.ImageNet):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        data = pickle.dumps(sample)
        return data

def dump_imagenet():
    batch_size = 32
    data_dir = '/public_dataset/2/ImageNet'
    dataset = DumpDataset(data_dir, split='train')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x,
        num_workers=8,
    )

    num_batches = 10
    n = num_batches * batch_size

    writer = FileWriter('imagenet_train.ffr', n)

    for batch, samples in enumerate(loader):
        for sample in samples:
            writer.write_one(sample)

        if batch == num_batches - 1:
            break

    writer.close()


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


def load_imagenet():

    mean = [0.4753, 0.4495, 0.4115]
    std = [0.2638, 0.2566, 0.2720]
    transofrm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = FireFlyerImageNet('imagenet_train.ffr',
                                transofrm)
    print(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
        num_workers=2,
    )

    for img, label in dataloader:
        # train model
        print(img.shape, label.shape)


if __name__ == '__main__':
    dump_imagenet()
    load_imagenet()
