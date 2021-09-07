import sys
import pickle
import os
from typing import List, Tuple, Any
from pathlib import Path

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ffrecord.torch import DataLoader, Dataset


class CoCoDataset(Dataset):
    """
    Coco detection数据集
    """
    def __init__(self, fname, transform=None):
        """
        fname: FFRecord file name 
        """
        super().__init__(fname)

        self.transform = transform

    def process(self, indexes: List[int], data: List[bytes]) -> List[Any]:
        """
        indexes: index of each sample
        data:    bytes data of each sample
        """

        samples = []
        for bytes_ in data:
            # 对字节数据做反序列化
            # image是一个Pillow Image对象
            # anno是一个list，包含一张图片中每个物体的label
            image, anno = pickle.loads(bytes_)
            if self.transform is not None:
                # 对图像做变换
                image = self.transform(image)

            samples.append((image, anno))

        # samples后续会被送入collate_fn
        return samples


class CoCoDataLoader(object):
    def __init__(self, data_dir, batch_size, num_workers):
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.train_loader, self.test_loader = self.__get_coco(
            data_dir, transform, batch_size, num_workers)

    def __get_coco(self, coco_root, transform, batch_size, num_workers):

        # 创建train、test dataloader
        # 每个ffr文件对应一个split
        coco_root = Path(coco_root)
        train_set = CoCoDataset(coco_root / 'train2017.ffr', transform)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, shuffle=True)
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  sampler=train_sampler,
                                  collate_fn=collate_fn_coco)

        test_set = CoCoDataset(coco_root / 'val2017.ffr', transform)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set, shuffle=False)
        test_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 sampler=test_sampler,
                                 collate_fn=collate_fn_coco)

        return train_loader, test_loader


def collate_fn_coco(batch):
    images, annos = tuple(zip(*batch))
    num_objs = torch.empty((len(images)), dtype=torch.int64)

    t_images = []
    b_bboxes, b_labels = [], []

    for i, image in enumerate(images):
        r_width = 1 / image.shape[0]
        r_height = 1 / image.shape[1]
        t_image = torch.unsqueeze(image, dim=0)
        t_images.append(t_image)

        num_objs[i] = len(annos[i])
        boxes = torch.empty((len(annos[i]), 4), dtype=torch.float32)
        labels = torch.empty((len(annos[i])), dtype=torch.int64)
        for obj, anno in enumerate(annos[i]):
            boxes[obj][0] = anno['bbox'][0] * r_width
            boxes[obj][1] = anno['bbox'][1] * r_height
            boxes[obj][2] = (anno['bbox'][0] + anno['bbox'][2]) * r_width
            boxes[obj][3] = (anno['bbox'][1] + anno['bbox'][3]) * r_height
            labels[obj] = anno['category_id']
        b_bboxes.append(boxes)
        b_labels.append(labels)

    t_images = torch.cat(t_images)

    return t_images, b_bboxes, b_labels, num_objs
