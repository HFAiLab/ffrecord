import os
import unittest
import random
import tempfile
from pathlib import Path
import numpy as np

import ffrecord
from ffrecord import FileWriter, FileReader
print(ffrecord.__version__)


class TestMultiFilesIO(unittest.TestCase):

    def subtest_posix_io(self, n, num_files):
        print("test_posix_io", n, num_files)

        files = []
        cnt = 0
        for file_no in range(num_files):
            _, file = tempfile.mkstemp(suffix='.ffr')
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
            _, file = tempfile.mkstemp(suffix='.ffr')
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

    def test_opendir(self):
        n, num_files = 100, 5
        print("test_opendir", n, num_files)

        files = []
        cnt = 0
        dir = 'test_files/'
        os.makedirs(dir, exist_ok=True)
        for file_no in range(num_files):
            file = dir + f'{file_no:05d}.ffr'
            files.append(file)

            writer = FileWriter(file, n)

            for i in range(n):
                bytes_ = bytearray([(x**2 % 256) for x in range(cnt + 5)])
                writer.write_one(bytes_)
                cnt += 1
            writer.close()

        reader = FileReader(dir, check_data=True)
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
        Path(dir).rmdir()

    def test_pathlike_and_with(self):
        print('test_pathlike_and_with')
        _, file = tempfile.mkstemp(suffix='.ffr')
        with FileWriter(Path(file), 10) as w:
            for i in range(10):
                w.write_one(b'12345')

        with FileReader(Path(file)) as r:
            r.read(list(range(10)))

        Path(file).unlink()


@unittest.skipIf(os.getenv('DISABLE_TEST_LARGE_SAMPLE'), 'skip TestLargeSample')
class TestLargeSample(unittest.TestCase):

    def test_libaio(self):
        n = 100
        sample_size = 4 * (1 << 30)  # 4GB
        _, file = tempfile.mkstemp(suffix='.ffr')

        writer = FileWriter(file, n)
        for i in range(n):
            size = sample_size if i == 0 else 1024
            bytes_ = np.full((size,), i % 256, dtype=np.uint8)
            writer.write_one(bytes_)
        writer.close()

        reader = FileReader(file, check_data=True)
        indexes = list(range(n))
        random.shuffle(indexes)

        results = reader.read(indexes)
        for i in range(n):
            bytes_ = results[i]
            size = sample_size if indexes[i] == 0 else 1024
            gt_bytes = np.full((size,), indexes[i] % 256, dtype=np.uint8)
            assert np.array_equal(bytes_, gt_bytes)

        reader.close()
        Path(file).unlink()

    def test_posix_io(self):
        n = 2
        sample_size = 4 * (1 << 30)  # 4GB
        _, file = tempfile.mkstemp(suffix='.ffr')

        writer = FileWriter(file, n)
        for i in range(n):
            bytes_ = np.full((sample_size,), i % 256, dtype=np.uint8)
            writer.write_one(bytes_)
        writer.close()

        reader = FileReader(file, check_data=True)
        indexes = list(range(n))
        random.shuffle(indexes)

        for i in range(n):
            bytes_ = reader.read_one(indexes[i])
            gt_bytes = np.full((sample_size,), indexes[i] % 256, dtype=np.uint8)
            assert np.array_equal(bytes_, gt_bytes)

        reader.close()
        Path(file).unlink()


if __name__ == '__main__':
    unittest.main()

