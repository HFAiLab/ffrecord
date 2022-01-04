import unittest
import pickle
import tempfile
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from ffrecord import FileWriter
from ffrecord.torch import Dataset as FFDataset, DataLoader as FFDataLoader


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
        _, file = tempfile.mkstemp(suffix='.ffr')
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
        ffdataset = DummyFireFlyerDataset(file)
        ffdataloader = FFDataLoader(
            ffdataset,
            batch_size=8,
            shuffle=False,
            num_workers=num_workers,
        )

        assert len(dataloader) == len(ffdataloader)
        for batch1, batch2 in zip(dataloader, ffdataloader):
            print(batch1[0], batch2[0])
            assert torch.equal(batch1[0], batch2[0])
            assert torch.equal(batch1[1], batch2[1])

        Path(file).unlink()

    def test_single_process(self):
        self.subtest_dataloader(0)

    def test_multiple_processes(self):
        self.subtest_dataloader(8)


    def subtest_set_step(self, shuffle, num_workers, batch_size):
        print(f"subtest_set_step: shuffle {shuffle}, "
              f"num_workers {num_workers}, batch_size {batch_size}")

        _, file = tempfile.mkstemp(suffix='.ffr')
        nbatches = 125
        n = batch_size * nbatches

        generator = torch.Generator()
        generator = generator.manual_seed(12345)

        dataset = DummyDataset(n)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
        )
        assert len(dataloader) == nbatches

        # dump dataset
        writer = FileWriter(file, len(dataset))
        for i in range(len(dataset)):
            data = pickle.dumps(dataset[i])
            data = bytearray(data)
            writer.write_one(data)
        writer.close()

        # load dataset
        generator = torch.Generator()
        generator = generator.manual_seed(12345)
        ffdataset = DummyFireFlyerDataset(file)
        ffdataloader = FFDataLoader(
            ffdataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
        )

        step = 5
        ffdataloader.set_step(step)
        assert len(ffdataloader) == nbatches - step

        # skip steps
        it = iter(dataloader)
        for i in range(step):
            _ = next(it)

        for batch1, batch2 in zip(it, ffdataloader):
            assert torch.equal(batch1[0], batch2[0])
            assert torch.equal(batch1[1], batch2[1])

        ffdataloader.set_step(0)
        assert len(ffdataloader) == nbatches

        Path(file).unlink()

    def test_set_step(self):
        for shuffle in True, False:
            for num_workers in 0, 4:
                for batch_size in 1, 4:
                    self.subtest_set_step(shuffle, num_workers, batch_size)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    unittest.main()
