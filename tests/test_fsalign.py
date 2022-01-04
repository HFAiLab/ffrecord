import os
import unittest
import tempfile
import fcntl
import struct

from ffrecord import checkFsAlign


FS_IOCNUM_CHECK_FS_ALIGN = 2147772004


def checkFsAlign2(fd):
    buf = bytearray(4)
    try:
        fcntl.ioctl(fd, FS_IOCNUM_CHECK_FS_ALIGN, buf)
    except OSError as err:
        return False

    fsAlign = struct.unpack("i", buf)
    return fsAlign[0] == 1


class TestFsAlign(unittest.TestCase):

    def subtest_fsalign(self, fname, is_aligned):
        if not os.path.exists(fname):
            print(f'{fname} does not exist, skip...')
            return

        fd = os.open(fname, os.O_RDONLY | os.O_DIRECT)
        assert checkFsAlign(fd) == checkFsAlign2(fd) == is_aligned

    def test_fs(self):
        fname = "/public_dataset/1/ImageNet/train.ffr/PART_00000.ffr"
        self.subtest_fsalign(fname, True)

    def test_tmp(self):
        with tempfile.NamedTemporaryFile() as tmp:
            self.subtest_fsalign(tmp.name, False)


if __name__ == '__main__':
    unittest.main()
