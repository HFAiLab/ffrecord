from typing import Union, List, Tuple, Any
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

    def __getattr__(self, name: str) -> None:
        try:
            return TorchDataset.__getattr__(self, name)
        except AttributeError:
            return ReaderRegistry.__getattr__(self, name)
