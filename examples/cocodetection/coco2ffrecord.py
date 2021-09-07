import sys
import os
import pickle
from typing import Union
from tqdm import tqdm, trange
import torch
import torchvision.datasets as datasets

from ffrecord import FileWriter


class DumpDataset(datasets.coco.CocoDetection):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        data = pickle.dumps(sample)  # 序列化数据
        return data


def convert(dataset, out_file: Union[str, os.PathLike]):
    """
    把CocoDetection数据集转换为FFrecord格式
    """
    assert not os.path.exists(out_file)
    n = len(dataset)
    print(f'writing {n} samples to {out_file}')

    # 采用DataLoader的多进程加速数据的处理
    batch_size = 64
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x,
        num_workers=16,
        multiprocessing_context='fork',
    )

    # 创建FileWriter
    writer = FileWriter(out_file, n)

    # 把每条smaple写入FFRecord
    for samples in tqdm(loader, total=len(loader)):
        for sample in samples:
            writer.write_one(sample)

    # 关闭FileWriter
    writer.close()
    print(f'writing {writer.count} samples to {out_file}')


def main():
    coco_dir = 'data/coco/'
    out_dir = 'data/coco/'

    train_dir = coco_dir + 'train2017/'
    train_dataset = DumpDataset(
        root=train_dir,
        annFile=coco_dir + 'annotations/instances_train2017.json')

    val_dir = coco_dir + 'val2017/'
    val_dataset = DumpDataset(
        root=val_dir, annFile=coco_dir + 'annotations/instances_val2017.json')

    convert(train_dataset, out_dir + 'train2017_2.ffr')
    convert(val_dataset, out_dir + 'val2017_2.ffr')


if __name__ == '__main__':
    main()
