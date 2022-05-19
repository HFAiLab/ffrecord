from ._ffrecord_cpp import (
    FileReader as _FileReader,
    FileWriter as _FileWriter,
    checkFsAlign,
)
import os
from pathlib import Path


class FileReader(_FileReader):
    """
    FFRecord file reader.

    Args:
        fname (str, list or os.PathLike): Input file name.
        check_data (bool, optional): validate crc32 checksum or not (default: ``True``)

    """

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
            assert f.exists(), f'{f} does not exist'
            assert f.is_file(), f'{f} is not a file'
            assert f.suffix == '.ffr', f'{f} does not end with .ffr'

        super().__init__(fname, check_data)

    def __reduce__(self):
        return (FileReader, (self.fnames, self.check_data))

    def __enter__(self):
        return self

    def __exit__(self, *unsed):
        self.close()


class FileWriter(_FileWriter):
    """
    FFRecord file writer.

    Args:
        fname (str or os.PathLike): Output file name.
        n (int): number of samples to be written

    """
    def __init__(self, fname, n):
        assert isinstance(fname, (str, os.PathLike))
        assert Path(fname).suffix == '.ffr'
        fname = str(fname)
        super().__init__(fname, n)

    def __enter__(self):
        return self

    def __exit__(self, *unsed):
        self.close()
