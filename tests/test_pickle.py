import unittest
import tempfile
import pickle
import numpy as np

from ffrecord import FileWriter, FileReader


class TestFileReaderPickle(unittest.TestCase):

    def test_pickle(self):
        with tempfile.NamedTemporaryFile(suffix='.ffr') as tmp:
            file = tmp.name
            with FileWriter(file, 2) as w:
                w.write_one(bytearray([1, 2, 3, 4]))
                w.write_one(bytearray([5, 6, 7, 8]))

            r1 = FileReader(file, True)
            buf = pickle.dumps(r1)
            r2 = pickle.loads(buf)

            bytes1 = r1.read([0, 1])
            bytes2 = r2.read([0, 1])
            np.testing.assert_array_equal(bytes1[0], bytes2[0])
            np.testing.assert_array_equal(bytes1[1], bytes2[1])

            r1.close()
            r2.close()


if __name__ == '__main__':
    unittest.main()
