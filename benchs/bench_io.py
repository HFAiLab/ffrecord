import time
import unittest
import random
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

from ffrecord import FileWriter, FileReader
from ffrecord.fileio import (
    FileWriter as FFFileWriter,
    FileReader as FFFileReader,
)

data = bytearray([i % 256 for i in range(1000)])
tmp_dir = '/private_dataset'


def bench_cpp(n, bs):
    with tempfile.NamedTemporaryFile(dir=tmp_dir) as tmp:
        writer = FileWriter(tmp.name, n)

        t0 = time.time()
        for i in range(n):
            writer.write_one(data)
        t_write = time.time() - t0
        writer.close()
        print('cpp write: ', t_write)

        reader = FileReader(tmp.name)
        t0 = time.time()
        for i0 in range(0, n, bs):
            xn = min(bs, n - i0)
            indexes = list(range(i0, i0 + xn))
            reader.read(indexes)
        t_read = time.time() - t0
        reader.close()
        print('cpp read: ', t_read)

        return t_write, t_read

def bench_python(n, bs):
    with tempfile.NamedTemporaryFile(dir=tmp_dir) as tmp:
        writer = FFFileWriter(tmp.name, n)

        t0 = time.time()
        for i in range(n):
            writer.write_one(data)
        t_write = time.time() - t0
        writer.close()
        print('python write: ', t_write)

        reader = FFFileReader(tmp.name)
        t0 = time.time()
        for i0 in range(0, n, bs):
            xn = min(bs, n - i0)
            indexes = list(range(i0, i0 + xn))
            reader.read(indexes)
        t_read = time.time() - t0
        reader.close()
        print('python read: ', t_read)

        return t_write, t_read


def main():
    n = 100000
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    size = n * len(data) / (1 << 20)

    t_cpp, t_python = [], []
    for bs in batch_sizes:
        t_cpp.append(bench_cpp(n, bs))
        t_python.append(bench_python(n, bs))

    write_cpp = sum([size / a for a, b in t_cpp]) / len(t_cpp)
    write_python = sum([size / a for a, b in t_python]) / len(t_python)
    plt.bar(1.0, write_cpp, label='C++ write')
    plt.bar(2.0, write_python, label='Python write')
    plt.title('Write')
    plt.ylabel('MB/s')
    plt.legend()
    plt.savefig('bench_write.png')

    plt.clf()
    plt.plot(batch_sizes, [size / b for a, b in t_cpp], label='C++ read')
    plt.plot(batch_sizes, [size / b for a, b in t_python], label='Python read')
    plt.title('Read')
    plt.xlabel('batch size')
    plt.ylabel('MB/s')
    plt.legend()
    plt.savefig('bench_read.png')


if __name__ == '__main__':
    main()
