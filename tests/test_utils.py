import unittest
import tempfile
import pickle
from pathlib import Path

from ffrecord.utils import dump
from ffrecord import FileReader


class TestUtils(unittest.TestCase):

    def subtest_dump_to_multiple_files(self, n, nfiles):
        with tempfile.TemporaryDirectory(suffix='.ffr') as tmp:
            dataset = [i * 2 for i in range(n)]
            dump(dataset, tmp, nfiles=nfiles)

            # check # of files
            ffrs = list(Path(tmp).glob('*.ffr'))
            assert len(ffrs) <= nfiles

            r = FileReader(tmp, True)
            assert r.n == n

            samples = r.read(list(range(r.n)))
            samples = [pickle.loads(x) for x in samples]

            assert dataset == samples

    def subtest_dump_single_file(self, n):
        tmp = tempfile.mktemp(suffix='.ffr', prefix='tmp-test')
        dataset = [i * 2 for i in range(n)]
        dump(dataset, tmp, nfiles=1)

        r = FileReader(tmp, True)
        assert r.n == n

        samples = r.read(list(range(r.n)))
        samples = [pickle.loads(x) for x in samples]

        assert dataset == samples
        Path(tmp).unlink()

    def test_dump(self):
        self.subtest_dump_single_file(200)
        self.subtest_dump_to_multiple_files(200, 8)
        self.subtest_dump_to_multiple_files(200, 64)
        self.subtest_dump_to_multiple_files(10, 16)


if __name__ == '__main__':
    unittest.main()
