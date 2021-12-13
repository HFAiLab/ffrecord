from _ffrecord_cpp import (
    FileReader as _FileReader,
    FileWriter as _FileWriter,
    checkFsAlign,
)
import os
from pathlib import Path


class FileReader(_FileReader):

    def __init__(self, fname, check_data=True):
        if isinstance(fname, (str, os.PathLike)):
            fname = str(fname)
            if os.path.isdir(fname):
                fnames = [str(p) for p in Path(fname).glob('*.ffr')]
                fnames.sort()
                fname = fnames
        elif isinstance(fname, (list, tuple)):
            fname = [str(i) for i in fname]
        else:
            raise TypeError("fname must be str, os.PathLike or list")

        super().__init__(fname, check_data)


class FileWriter(_FileWriter):

    def __init__(self, fname, n):
        assert Path(fname).suffix == '.ffr'
        super().__init__(fname, n)


__all__ = ['FileReader', 'FileWriter', 'checkFsAlign']

# Please keep this list sorted
assert __all__ == sorted(__all__)
