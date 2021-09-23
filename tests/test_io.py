import unittest
import random
import tempfile
from pathlib import Path

from ffrecord import FileWriter, FileReader, checkFsAlign


class TestMultiFileIO(unittest.TestCase):

    def subtest_posix_io(self, n, num_files):
        print("test_posix_io", n, num_files)

        files = []
        cnt = 0
        for file_no in range(num_files):
            _, file = tempfile.mkstemp()
            files.append(file)

            writer = FileWriter(file, n)

            for i in range(n):
                bytes_ = bytearray([(x**2 % 256) for x in range(cnt + 5)])
                writer.write_one(bytes_)
                cnt += 1
            writer.close()

        reader = FileReader(files, check_data=True)
        assert reader.n == num_files * n
        indexes = list(range(n * num_files))
        random.shuffle(indexes)
        for i in indexes:
            bytes_ = reader.read_one(i).tobytes()
            gt_bytes = bytearray([(x**2 % 256) for x in range(i + 5)])
            self.assertEqual(bytes_, gt_bytes)

        reader.close()
        for file in files:
            Path(file).unlink()

    def subtest_libaio(self, n, num_files):
        print("test_libaio", n, num_files)

        files = []
        cnt = 0
        for file_no in range(num_files):
            _, file = tempfile.mkstemp()
            files.append(file)

            writer = FileWriter(file, n)

            for i in range(n):
                bytes_ = bytearray([(x**2 % 256) for x in range(cnt + 5)])
                writer.write_one(bytes_)
                cnt += 1
            writer.close()

        reader = FileReader(files, check_data=True)
        indexes = list(range(n * num_files))
        random.shuffle(indexes)
        batch_size = 10
        for j in range(n // batch_size):
            batch_indices = indexes[j * batch_size:(j + 1) * batch_size]
            results = reader.read(batch_indices)
            for i in range(batch_size):
                bytes_ = results[i].tobytes()
                gt_bytes = bytearray([(x**2 % 256)
                                      for x in range(batch_indices[i] + 5)])
                self.assertEqual(bytes_, gt_bytes)

        reader.close()
        for file in files:
            Path(file).unlink()

    def test_posix_io(self):
        self.subtest_posix_io(100, 1)
        self.subtest_posix_io(100, 10)

    def test_libaio(self):
        self.subtest_libaio(100, 1)
        self.subtest_libaio(100, 10)


if __name__ == '__main__':
    unittest.main()

