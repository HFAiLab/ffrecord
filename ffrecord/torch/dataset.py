import bisect
from collections import defaultdict
from typing import Union, List, Tuple, Any, Sequence, Iterable
from torch.utils.data import Dataset as TorchDataset

from ffrecord import FileReader


class ReaderRegistry():

    def ffreaders(self):
        readers = []
        if hasattr(self, "_readers"):
            readers += list(self._readers.values())

        if hasattr(self, "_resitries"):
            for r in self._resitries.values():
                readers += r.ffreaders()

        return readers

    def __setattr__(self, name: str, value: Any) -> None:
        # we register all FileReader objects
        if '_readers' not in self.__dict__:
            self.__dict__['_readers'] = {}
        if '_resitries' not in self.__dict__:
            self.__dict__['_resitries'] = {}

        if isinstance(value, FileReader):
            self._readers[name] = value
        elif isinstance(value, ReaderRegistry):
            self._resitries[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> None:
        if '_readers' in self.__dict__ and name in self.__dict__['_readers']:
            return self.__dict__['_readers'][name]

        if '_resitries' in self.__dict__ and name in self.__dict__['_resitries']:
            return self.__dict__['_resitries'][name]

        return object.__getattribute__(self, name)

    def __delattr__(self, name: str) -> None:
        if name in self._readers:
            del self._readers[name]
        if name in self._resitries:
            del self._resitries[name]
        else:
            object.__delattr__(self, name)


class Dataset(TorchDataset, ReaderRegistry):
    """
    Different from ``torch.utils.data.Dataset`` which accepts an index as input and returns one sample,
    ``ffrecord.torch.Dataset`` accepts a batch of indices as input and returns a batch of samples.
    One advantage of :code:`ffrecord.torch.Dataset` is that it could read a batch of data at a time using Linux AIO.

    Users need to inherit from ``ffrecord.torch.Dataset`` and define their custom ``__getitem__()`` and ``__len__()`` function.

    For example:

    .. code-block:: python

        class CustomDataset(ffrecord.torch.Dataset):

            def __init__(self, fname, check_data=True, transform=None):
                self.reader = FileReader(fname, check_data)
                self.transform = transform

            def __len__(self):
                return self.reader.n

            def __getitem__(self, indices):
                # we read a batch of samples at once
                assert isintance(indices, list)
                data = self.reader.read(indices)

                # deserialize data
                samples = [pickle.loads(b) for b in data]

                # transform data
                if self.transform:
                    samples = [self.transform(s) for s in samples]
                return samples

        dataset = CustomDataset('train.ffr')
        indices = [3, 4, 1, 0]
        samples = dataset[indices]

    Args:
        fnames:     FFrecord file names
        check_data: validate checksum or not

    """
    def __init__(self, fnames: Union[str, List, Tuple], check_data: bool = True):
        self.fnames = fnames
        self.reader = FileReader(fnames, check_data)

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indices):
        if not isinstance(indices, (list, tuple)):
            raise TypeError("indices must be a list of index")

        # read raw bytes data form the FFRecord file
        data = self.reader.read(indices)

        # pass the raw bytes data into the user-defined process function
        return self.process(indices, data)

    def process(self, indices: List[int], data: List[bytearray]) -> List[Any]:
        """ process the raw bytes data

        Args:
            indices: indices of each sample in one batch
            data:    raw bytes of each sample in one batch

        Return:
            A list of samples.
            It will be passed into collate_fn in Dataloader
        """
        # user-defined method
        raise NotImplementedError

    def __getattr__(self, name: str) -> None:
        try:
            return TorchDataset.__getattr__(self, name)
        except AttributeError:
            return ReaderRegistry.__getattr__(self, name)


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset: The whole Dataset
        indices: Indices in the whole set selected for subset
    """
    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, indices):
        indices = [self.indices[idx] for idx in indices]
        return self.dataset[indices]

    def __len__(self):
        return len(self.indices)


class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets: List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert isinstance(d, Dataset)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _convert_index(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitem__(self, indices):
        m = defaultdict(list)

        reorder = {}
        for i, idx in enumerate(indices):
            dataset_idx, sample_idx = self._convert_index(idx)
            m[dataset_idx].append(sample_idx)

            reorder[(dataset_idx, sample_idx)] = i

        samples = [0] * len(indices)
        for dataset_idx, sample_indices in sorted(m.items()):
            data = self.datasets[dataset_idx][sample_indices]

            # reorder
            for i, sid in enumerate(sample_indices):
                idx = reorder[(dataset_idx, sid)]
                samples[idx] = data[i]

        return samples
