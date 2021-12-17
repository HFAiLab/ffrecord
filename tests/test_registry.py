import unittest
import tempfile

from ffrecord import FileWriter, FileReader
from ffrecord.torch import (
    Dataset as FFDataset,
    ReaderRegistry,
)


class A(ReaderRegistry):

    def __init__(self, file1, file2):
        self.r1 = FileReader(file1)
        self.r2 = FileReader(file2)


class B(FFDataset):

    def __init__(self, file1, file2):
        self.reader1 = FileReader(file1)
        self.reader2 = FileReader(file2)
        self.another = A(file1, file2)


class TestReaderRegistry(unittest.TestCase):

    def test_registry(self):
        _, file = tempfile.mkstemp(suffix='.ffr')
        n = 100
        writer = FileWriter(file, n)
        for i in range(n):
            writer.write_one(b'12345')
        writer.close()

        a = B(file, file)
        readers = a.ffreaders()
        print(readers)
        assert len(readers) == 4


if __name__ == '__main__':
    unittest.main()
