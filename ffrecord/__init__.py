from pkg_resources import get_distribution
from .fileio import (
    FileReader,
    FileWriter,
    checkFsAlign,
)
from .package import PackedFolder, pack_folder
from . import utils


__version__ = get_distribution('ffrecord').version

__all__ = [
    'FileReader',
    'FileWriter',
    'PackedFolder',
    'checkFsAlign',
    'pack_folder',
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
