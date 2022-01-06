
from typing import Union, Any, Mapping, Callable
import multiprocessing as mp
import pickle
from pathlib import Path
import os
import warnings
from tqdm import trange

from .fileio import FileWriter


def dump(
        dataset: Mapping[int, Any],
        fname: Union[str, os.PathLike],
        nfiles: int,
        serializer: Callable = pickle.dumps,
        verbose: bool = False,
    ) -> None:
    r"""
    Dump an indexable object to ffrecord files.

    Args:
        dataset:    an indexable object which implements `__getitem__` and `__len__`
        fname:      output folder (nfiles > 1) or file name (nfiles = 1)
        nfiles:     number of output files
        serializer: serialize function, it will be called if datasets[i] does not
                    return bytes or bytearray
        verbose:    show dumping progress or not
    """

    n = len(dataset)

    if nfiles == 1:
        _write_to_ffr(0, n, dataset, fname, serializer, verbose)
        return

    out_dir = Path(fname)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = (n + nfiles - 1) // nfiles
    tasks = []

    fid = 0
    for i0 in range(0, n, bs):
        ni = min(bs, n - i0)
        fname = out_dir / f"PART_{fid:05d}.ffr"
        tasks.append([i0, ni, dataset, fname, serializer, verbose])
        fid += 1

    if len(tasks) != nfiles:
        warnings.warn(f"Split into {len(tasks)} files rather than {nfiles} files")

    nprocs = min(16, len(tasks))
    for i in range(0, len(tasks), nprocs):
        procs = []
        for task in tasks[i:(i + nprocs)]:
            p = mp.Process(target=_write_to_ffr, args=task)
            p.start()
            procs.append(p)

        for p in procs:
            p.join()


def _write_to_ffr(i0, ni, dataset, fname, serializer, verbose):
    rng = trange if verbose else range

    with FileWriter(fname, ni) as w:
        for i in rng(i0, i0 + ni):
            item = dataset[i]

            isbuffer = isinstance(item, (bytes, bytearray))
            assert isbuffer or serializer is not None
            if not isbuffer:
                item = serializer(item)

            w.write_one(item)
    return
