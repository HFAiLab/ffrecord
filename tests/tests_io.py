import unittest
import random
import tempfile
from pathlib import Path
from ffrecord.fileio import FileWriter, FileReader


class TestIO(unittest.TestCase):
    def test_posix_io(self):
        print("test_posix_io")
        _, file = tempfile.mkstemp()
        n = 100

        writer = FileWriter(file, n)
        offsets, sizes = [], []

        ofs = 12 + 12 * n
        for i in range(n):
            bytes = bytearray([(x**2 % 256) for x in range(i + 5)])
            writer.write_one(bytes)
            offsets.append(ofs)
            sizes.append(i + 5)
            ofs += i + 5
        writer.close()

        reader = FileReader(file, check_data=True)
        reader.validate()
        indexes = list(range(n))
        random.shuffle(indexes)
        for i in indexes:
            bytes = reader.read_one(i)
            gt_bytes = bytearray([(x**2 % 256) for x in range(i + 5)])
            self.assertEqual(bytes, gt_bytes)

        # test offset and size
        offsets1, sizes1 = reader.loc(list(range(n)))
        self.assertEqual(offsets1, offsets)
        self.assertEqual(sizes1, sizes)

        reader.close()
        Path(file).unlink()

    def test_libaio(self):
        print("test_libaio")
        _, file = tempfile.mkstemp()
        n = 100

        writer = FileWriter(file, n)
        offsets, sizes = [], []

        ofs = 12 + 12 * n
        for i in range(n):
            bytes = bytearray([(x**2 % 256) for x in range(i + 5)])
            writer.write_one(bytes)
            offsets.append(ofs)
            sizes.append(i + 5)
            ofs += i + 5
        writer.close()

        reader = FileReader(file, check_data=True)
        reader.validate()
        indexes = list(range(n))
        random.shuffle(indexes)
        batch_size = 10
        for j in range(n // batch_size):
            batch_indices = indexes[j * batch_size:(j + 1) * batch_size]
            results = reader.read(batch_indices)
            for i in range(batch_size):
                bytes = results[i]
                gt_bytes = bytearray([(x**2 % 256)
                                      for x in range(batch_indices[i] + 5)])
                self.assertEqual(bytes, gt_bytes)

        # test offset and size
        offsets1, sizes1 = reader.loc(list(range(n)))
        self.assertEqual(offsets1, offsets)
        self.assertEqual(sizes1, sizes)

        reader.close()
        Path(file).unlink()


if __name__ == '__main__':
    unittest.main()
