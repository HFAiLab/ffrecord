from pkg_resources import get_distribution
from .fileio import (
    FileReader,
    FileWriter,
    checkFsAlign,
)
from . import utils


__version__ = get_distribution('ffrecord').version

__all__ = ['FileReader', 'FileWriter', 'checkFsAlign']

# Please keep this list sorted
assert __all__ == sorted(__all__)
