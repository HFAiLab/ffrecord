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
            if Path(fname).is_dir():
                fnames = [str(p) for p in Path(fname).glob('*.ffr')]
                fnames.sort()
                fname = fnames
            else:
                fname = [fname]
        elif isinstance(fname, (list, tuple)):
            fname = [str(i) for i in fname]
        else:
            raise TypeError("fname must be str, os.PathLike, list or tuple")

        assert len(fname) >= 1, "At least one .ffr file as input"
        for f in fname:
            f = Path(f)
            assert f.is_file(), f'{f} is not a file'
            assert f.suffix == '.ffr', f'{f} does not end with .ffr'

        super().__init__(fname, check_data)

    def __reduce__(self):
        return (FileReader, (self.fnames, self.check_data))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FileWriter(_FileWriter):

    def __init__(self, fname, n):
        assert isinstance(fname, (str, os.PathLike))
        assert Path(fname).suffix == '.ffr'
        fname = str(fname)
        super().__init__(fname, n)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
