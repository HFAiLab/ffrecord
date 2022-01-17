from typing import List, Iterator
import torch
from torch.utils.data import Sampler
import torch.utils.data._utils as _utils
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import _DatasetKind

from .dataset import Dataset


################################################################################
# public methods and classes
################################################################################

class DataLoader(TorchDataLoader):
    """
    This DataLoader class acts like Pytorch DataLoader except that
    multiprocessing_context is fixed to ``fork``.

    ``ffrecord.torch.Dataset`` and ``torch.utils.data.Dataset`` are both supported.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of sample loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers samples prefetched across all workers. (default: ``2``)
        persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)

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

        assert self.batch_sampler is not None
        batch_sampler = SkipableSampler(self.batch_sampler)
        object.__setattr__(self, 'batch_sampler', batch_sampler)

    def set_step(self, step: int) -> None:
        assert 0 <= step < len(self.batch_sampler.sampler)
        self.batch_sampler.set_step(step)


################################################################################
# private methods and classes
################################################################################

class SkipableSampler(Sampler[List[int]]):

    def __init__(self, sampler) -> None:
        self.sampler = sampler
        self.step = 0

    def __iter__(self) -> Iterator[List[int]]:
        for i, index in enumerate(self.sampler):
            if i < self.step:
                continue
            yield index

    def __len__(self) -> int:
        return self._len()

    def _len(self) -> int:
        return len(self.sampler) - self.step

    def set_step(self, step: int) -> None:
        self.step = step


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
