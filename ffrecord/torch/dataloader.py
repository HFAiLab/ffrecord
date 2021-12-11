from typing import Union, List, Tuple, Any

import torch
import torch.multiprocessing as mp
import torch.utils.data._utils as _utils
from torch.utils.data import (
    Dataset as TorchDataset,
    DataLoader as TorchDataLoader,
)
from torch.utils.data.dataloader import _DatasetKind

from ffrecord import FileReader

################################################################################
# public methods and classes
################################################################################


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

    def __hasattr(self, name):
        return name in self.__dict__

    def __getattr__(self, name: str) -> None:
        if '_readers' in self.__dict__ and name in self.__dict__['_readers']:
            return self.__dict__['_readers'][name]

        if '_resitries' in self.__dict__ and name in self.__dict__['_resitries']:
            return self.__dict__['_resitries'][name]

        return getattr(self, name)

    def __delattr__(self, name: str) -> None:
        if name in self._readers:
            del self._readers[name]
        if name in self._resitries:
            del self._resitries[name]
        else:
            object.__delattr__(self, name)


class Dataset(TorchDataset, ReaderRegistry):
    """
    Different from `torch.utils.data.Dataset` which accepts an index as input and returns one sample,
    `ffrecord.torch.Dataset` accepts a batch of indexes as input and returns a batch of samples.
    One advantage of `ffrecord.torch.Dataset` is that it could read a batch of data at a time using Linux AIO.

    We first read a batch of bytes data from FFReocrd file and then pass the bytes data to process()
    function. Users need to inherit from `ffrecord.torch.Dataset` and define their custom process function.

    ```
    Pipline:   indexes ----------------------------> bytes -------------> samples
                        reader.read(indexes)               process()
    ```

    For example:
    ```python
    class CustomDataset(ffrecord.torch.Dataset):

        def __init__(self, fname, transform=None):
            super().__init__(fname)
            self.transform = transform

        def process(self, indexes, data):
            # deserialize data
            samples = [pickle.loads(b) for b in data]

            # transform data
            if self.transform:
                samples = [self.transform(s) for s in samples]
            return samples

    dataset = CustomDataset('train.ffr')
    indexes = [3, 4, 1, 0]
    samples = dataset[indexes]
    ```
    """
    def __init__(self, fnames: Union[str, List, Tuple], check_data: bool = True):
        """
        Args:
            fnames:     FFrecord file names
            check_data: validate checksum or not
        """
        self.fnames = fnames
        self.reader = FileReader(fnames, check_data)

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indexes):
        if not isinstance(indexes, (list, tuple)):
            raise TypeError("indexes must be a list of index")

        # read raw bytes data form the FFRecord file
        data = self.reader.read(indexes)

        # pass the raw bytes data into the user-defined process function
        return self.process(indexes, data)

    def process(self, indexes: List[int], data: List[bytearray]) -> List[Any]:
        """ process the raw bytes data

        Args:
            indexes: indexes of each sample in one batch
            data:    raw bytes of each sample in one batch

        Return:
            A list of samples.
            It will be passed into collate_fn in Dataloader
        """
        # user-defined method
        raise NotImplementedError


class DataLoader(TorchDataLoader):
    """
    This DataLoader class acts like Pytorch DataLoader except that
    multiprocessing_context is fixed to `fork`.

    ffrecord.torch.Dataset and torch.utils.data.Dataset are both supported.
    """
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle: bool = False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers: int = 0,
                 collate_fn=None,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0,
                 worker_init_fn=None,
                 generator=None,
                 *,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False):

        # use fork to create subprocesses
        if num_workers == 0:
            multiprocessing_context = None
        else:
            multiprocessing_context = 'fork'

        super(DataLoader,
              self).__init__(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=sampler,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             pin_memory=pin_memory,
                             drop_last=drop_last,
                             timeout=timeout,
                             worker_init_fn=worker_init_fn,
                             multiprocessing_context=multiprocessing_context,
                             generator=generator,
                             prefetch_factor=prefetch_factor,
                             persistent_workers=persistent_workers)

        if isinstance(dataset, Dataset):
            self._dataset_kind = _DatasetKind.SliceMap


################################################################################
# private methods and classes
################################################################################


class _SliceMapDatasetFetcher(_utils.fetch._BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_SliceMapDatasetFetcher, self).__init__(dataset, auto_collation,
                                                      collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            indexes = possibly_batched_index
        else:
            indexes = [possibly_batched_index]

        data = self.dataset[indexes]
        return self.collate_fn(data)


def _create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
    if kind == _DatasetKind.Map:
        return _utils.fetch._MapDatasetFetcher(dataset, auto_collation,
                                               collate_fn, drop_last)
    elif kind == _DatasetKind.Iterable:
        return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation,
                                                    collate_fn, drop_last)
    elif kind == _DatasetKind.SliceMap:
        return _SliceMapDatasetFetcher(dataset, auto_collation, collate_fn,
                                       drop_last)


_DatasetKind.SliceMap = 2
_DatasetKind.create_fetcher = _create_fetcher
