import pickle
from pathlib import Path
from collections import OrderedDict
from ffrecord import FileWriter, FileReader


MAGIC_BYTES = b"piouhjgbr"


class PackedFolder():

    def __init__(self, ffr_file, check_data=True):
        self.reader = FileReader(ffr_file, check_data)
        meta = self.reader.read_one(self.reader.n - 1).tobytes()

        if len(meta) <= len(MAGIC_BYTES) or meta[:len(MAGIC_BYTES)] != MAGIC_BYTES:
            raise ValueError(f"{ffr_file} is not a packed folder")

        self.structure = pickle.loads(meta[len(MAGIC_BYTES):])

    def num_files(self):
        return self.reader.n - 1

    def read_one(self, fname):
        index = self._fname_to_index(fname)
        data = self.reader.read_one(index)

        return data

    def read(self, fnames):
        assert isinstance(fnames, (list, tuple))
        indices = [self._fname_to_index(p) for p in fnames]
        data = self.reader.read(indices)

        return data

    def _fname_to_index(self, fname):
        struct = self.structure
        for level_name in Path(fname).parts:
            assert level_name in struct, f"{fname} does not exist"
            struct = struct[level_name]
        assert isinstance(struct, int), f"{fname} is not a file but a directory"
        return struct

    def list(self, path=None, list_file=True, list_dir=True, recursive=False):
        if path is not None:
            assert self.exists(path), f"{path} does not exist"
            path = Path(path)
            struct = self.structure
            for level_name in path.parts:
                struct = struct[level_name]
        else:
            struct = self.structure
            path = Path("")

        assert isinstance(struct, OrderedDict), f"{path} is not a directory but a file"
        fnames = self._list(struct, list_file, list_dir, recursive)

        return fnames

    def exists(self, path):
        struct = self.structure
        for level_name in Path(path).parts:
            if level_name not in struct:
                return False
            struct = struct[level_name]
        return True

    def is_file(self, path):
        return self._is_file_or_dir(path, True)

    def is_dir(self, path):
        return self._is_file_or_dir(path, False)

    def _is_file_or_dir(self, path, is_file=True):
        struct = self.structure
        for level_name in Path(path).parts:
            if level_name not in struct:
                return False
            struct = struct[level_name]

        return not (is_file ^ isinstance(struct, int))

    def _list(self, struct, list_file, list_dir, recursive):
        fnames = []
        for child in struct:
            is_file = isinstance(struct[child], int)
            if is_file and list_file:
                fnames.append(child)
            if (not is_file) and list_dir:
                fnames.append(child)

            if recursive and (not is_file):
                # this is a folder
                files = self._list(struct[child], list_file, list_dir, recursive)
                fnames += [Path(child) / p for p in files]

        return fnames


def pack_folder(folder, out_file, verbose=False):
    """
    Pack a folder into ffrecord file format. The packed file could be read by :class:`PackedFolder`.

    Args:
        folder (str): path to the folder to be packed
        out_file (str): output file name
        verbose (bool): print progress or not
    """
    folder = Path(folder).absolute()
    assert folder.is_dir(), "'folder' must be a directory"

    # extract structure
    structure = OrderedDict()
    counter = dump_structure(folder, structure, 0)
    meta = MAGIC_BYTES + pickle.dumps(structure)

    # dump files
    writer = FileWriter(out_file, counter + 1)
    counter = dump_files(writer, folder, structure, 0, verbose, counter)
    writer.write_one(meta)  # write meta into ffr
    writer.close()


def dump_structure(folder, structure, counter):
    children = sorted(list(folder.iterdir()))
    for child in children:
        if child.is_dir():
            structure[child.name] = OrderedDict()
            counter = dump_structure(child, structure[child.name], counter)
        else:
            structure[child.name] = counter
            counter += 1

    return counter


def dump_files(writer, prefix, structure, counter, verbose, n):
    width = len(str(n))

    for path in structure:
        if isinstance(structure[path], int):
            index = structure[path]
            assert index == counter
            counter += 1

            path = prefix / path
            with open(path, "rb") as fp:
                writer.write_one(fp.read())
                if verbose:
                    print(f"{index:0{width}d}/{n} {path}", flush=True)
        else:
            counter = dump_files(writer, prefix / path, structure[path], counter, verbose, n)

    return counter
