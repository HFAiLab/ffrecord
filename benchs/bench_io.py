import hf_env
hf_env.set_env('202105')

import time
from pathlib import Path
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Process
from tqdm import trange

from ffrecord import FileWriter, FileReader


def read_chunk(reader, q):
    t0 = time.time()
    while True:
        r = q.get()
        if r is None:
            break

        i0, xn = r
        indexes = list(range(i0, i0 + xn))
        bytes_ = reader.read(indexes)
    t_read = time.time() - t0
    reader.close()

    return t_read


def bench_read(implem, Reader, fname, n, bs, nprocs):
    reader = Reader(fname)

    q = mp.Queue()
    for i0 in range(0, n, bs):
        ni = min(bs, n - i0)
        q.put([i0, ni])

    # use None as sentinel
    for i in range(nprocs):
        q.put(None)

    t0 = time.time()
    procs = []
    for i in range(nprocs):
        p = Process(target=read_chunk, args=(reader, q))
        p.start()
        procs.append(p)
    for i in range(nprocs):
        procs[i].join()

    t_read = time.time() - t0
    print(f'{implem} read: {t_read} s')

    return t_read


def main():
    sample_size = 1 * (1 << 20)
    data = bytearray([i % 256 for i in range(sample_size)])
    tmp_dir = '/private_dataset'
    nprocs = 128
    n = 100000
    fname = tmp_dir + f'/test_ss_{sample_size}.ffr'

    if not Path(fname).exists():
        writer = FileWriter(fname, n)

        t0 = time.time()
        for i in trange(n):
            writer.write_one(data)
        t_write = time.time() - t0
        writer.close()
        print('cpp write: ', t_write)

    n = 100000
    size = n * len(data) / (1 << 30)
    print(f'Reading {size} GB from {fname}')
    batch_sizes = [64, 80, 96, 112, 128]

    t_cpp = []
    for bs in batch_sizes:
        t_cpp.append(bench_read('cpp', FileReader, fname, n, bs, nprocs))

    plt.plot(batch_sizes, [size / b for b in t_cpp], label='read')
    plt.title(f'Read, nprocs {nprocs}, sample_size {sample_size}')
    plt.xlabel('batch size')
    plt.ylabel('GB/s')
    plt.legend()
    plt.savefig(f'bench_read_mp_{nprocs}_{sample_size}.png')


if __name__ == '__main__':
    main()
