import unittest
import random
import pickle
import tempfile
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

import sys

sys.path.insert(
    0,
    "/nfs-jd/prod/nfs1/zsy/pylibs/lib/python3.6/site-packages/ffrecord-1.1.0+e792bf8-py3.6.egg"
)
from ffrecord.fileio import FileWriter, FileReader
from ffrecord.torch import Dataset as FFDataset, Dataloader as FFDataloader


class DummyDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return (index * 2, index * 3)


class DummyFireFlyerDataset(FFDataset):
    def process(self, indexes, data):
        processed_data = []
        for bytes_ in data:
            sample = pickle.loads(bytes_)
            processed_data.append(sample)

        return processed_data


class TestDataLoader(unittest.TestCase):
    def subtest_dataloader(self, num_workers):
        _, file = tempfile.mkstemp()
        n = 100

        dataset = DummyDataset(n)
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
        )

        # dump dataset
        writer = FileWriter(file, len(dataset))
        for i in range(len(dataset)):
            data = pickle.dumps(dataset[i])
            data = bytearray(data)
            writer.write_one(data)
        writer.close()
        # load dataset
        aiodataset = DummyFireFlyerDataset(file)
        aiodataloader = FFDataloader(
            aiodataset,
            batch_size=8,
            shuffle=False,
            num_workers=num_workers,
        )

        assert len(dataloader) == len(aiodataloader)
        for batch1, batch2 in zip(dataloader, aiodataloader):
            print(batch1[0], batch2[0])
            assert torch.equal(batch1[0], batch2[0])
            assert torch.equal(batch1[1], batch2[1])

        Path(file).unlink()

    def test_single_process(self):
        self.subtest_dataloader(0)

    def test_multiple_processes(self):
        self.subtest_dataloader(8)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    unittest.main()
