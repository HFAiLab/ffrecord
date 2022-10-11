import unittest
from pathlib import Path
import numpy as np

import ffrecord
from ffrecord import pack_folder, PackedFolder
print(ffrecord.__version__)


class TestPackFolder(unittest.TestCase):

    def setUp(self):
        # create dummy files
        self.ffr_name = Path("packed.ffr")
        self.tmp_dir = Path("test_pack_folder")
        self.tmp_dir.mkdir(exist_ok=True)
        for i in range(20):
            with open(self.tmp_dir / f"{i:03d}", "wb") as fp:
                data = bytearray([(i + j) % 256 for j in range(1000 + i)])
                fp.write(data)

        pack_folder(self.tmp_dir, self.ffr_name, verbose=True)

    def tearDown(self) -> None:
        # delete dummy files
        files = list(self.tmp_dir.iterdir())
        for file in files:
            file.unlink()
        self.tmp_dir.rmdir()
        self.ffr_name.unlink()

    def test_packfolder(self):
        folder = PackedFolder(self.ffr_name)
        assert folder.num_files() == 20
        assert folder.is_file("000")
        assert not folder.is_dir("000")
        assert not folder.is_file("100")

        files = folder.list()
        assert len(files) == folder.num_files()
        assert set(files) == set(f"{i:03d}" for i in range(20))

        for i in range(20):
            data = folder.read_one(f"{i:03d}")
            expected = bytearray([(i + j) % 256 for j in range(1000 + i)])
            expected = np.frombuffer(expected, dtype=np.uint8)
            np.testing.assert_array_equal(data, expected)


if __name__ == '__main__':
    unittest.main()
