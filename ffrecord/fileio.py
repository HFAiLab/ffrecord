import os
from typing import Union, List, Tuple
import numpy as np
import libaio
import fcntl
import struct
import zlib
"""

File Format:

+-----------------------------------+---------------------------------------+
|         checksum                  |             N                         |
+-----------------------------------+---------------------------------------+
|         checksums                 |           offsets                     |
+---------------------+---------------------+--------+----------------------+
|      sample 1       |      sample 2       | ....   |      sample N        |
+---------------------+---------------------+--------+----------------------+

Fields:
    checksum:     checksum of metadata
    N:            number of samples
    checksums:    checksums of each samples
    offsets:      offsets of each samples
    sample i:     data of the i-th sample

"""

AIO_MAX_EVENTS = 1024
FS_IOCNUM_CHECK_FS_ALIGN = 2147772004
OFFSET_TYPE = np.uint64
CHECKSUM_TYPE = np.uint32


class FileWriter():
    """
    fd:         opened file descriptor
    count:      number of written samples.
    n:          number of samples.
    sample_pos: current data position
    offsets:    offsets of each samples
    checksums:  checksums of each samples
    """
    def __init__(self, fname: Union[str, os.PathLike], n: int):
        """ open a file to write

        Args:
            fname: file to be opened
            n:     number of samples
        """
        if not isinstance(fname, (str, os.PathLike)):
            raise TypeError("fname must be str or os.PathLike")

        self.fd = os.open(fname, os.O_RDWR | os.O_CREAT)
        self.count = 0
        self.n = n

        self.sample_pos = 8 + CHECKSUM_TYPE(
        ).itemsize + CHECKSUM_TYPE().itemsize * n + OFFSET_TYPE().itemsize * n
        os.lseek(self.fd, self.sample_pos, os.SEEK_SET)

        self.offsets = np.empty(n, dtype=OFFSET_TYPE)
        self.checksums = np.empty(n, dtype=CHECKSUM_TYPE)

    async def write(self, fd, pos):
        pass

    def write_one(self, data: Union[bytes, bytearray]) -> None:
        """ write one sample

        Args:
            data: data of one sample
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes or bytearray")
        assert self.count < self.n

        # write offset
        self.offsets[self.count] = self.sample_pos
        # write checksum
        self.checksums[self.count] = zlib.crc32(data)

        # write sample
        os.write(self.fd, data)
        self.sample_pos += len(data)
        self.count += 1

    def finish(self):
        if self.n != self.count:
            raise ValueError("number of sample written does not match the input number")

        os.lseek(self.fd, 0, os.SEEK_SET)
        # fill header
        os.write(self.fd, struct.pack("<IQ", 0, self.n))
        # checksums and offsets
        os.write(self.fd, self.checksums.tobytes())
        os.write(self.fd, self.offsets.tobytes())

        # compute checksum for metadata
        chunk_size = 64 * (1 << 20)  # 64 MB
        start_pos = CHECKSUM_TYPE().itemsize
        nbytes = 8 + CHECKSUM_TYPE().itemsize * self.n + OFFSET_TYPE().itemsize * self.n
        end_pos = start_pos + nbytes

        checksum = 0
        for offset in range(start_pos, end_pos, chunk_size):
            size = min(chunk_size, end_pos - offset)
            data = os.pread(self.fd, size, offset)
            checksum = zlib.crc32(data, checksum)
        os.pwrite(self.fd, struct.pack("<I", checksum), 0)
        os.close(self.fd)

    def close(self):
        if hasattr(self, 'fd') and self.fd:
            self.finish()
            self.fd = None

    def __del__(self):
        self.close()


def checkFsAlign(fd):
    buf = bytearray(4)
    try:
        fcntl.ioctl(fd, FS_IOCNUM_CHECK_FS_ALIGN, buf)
    except:
        return False

    fsAlign = struct.unpack("i", buf)
    return fsAlign[0] == 1


class FileReader():
    """
    fd:           opened file descriptor
    aio_fd:       opened file descriptor for aio read
    n:            number of samples
    filesize:     file size
    offsets:      offsets of each samples
    checksums:    checksums of each samples
    checksum_all: checksum of all data (disabled)
    """
    def __init__(self,
                 fname: Union[str, os.PathLike],
                 check_data: bool = True):
        """ open a file to read

        Args:
            fname:      file to be opened
            check_data: validate checksum or not
        """
        if not isinstance(fname, (str, os.PathLike)):
            raise TypeError("fname must be str or os.PathLike")

        self.fd = os.open(fname, os.O_RDONLY)
        if checkFsAlign(self.fd):
            self.aio_fd = os.open(fname, os.O_RDONLY | os.O_DIRECT)
        else:
            print("The underlying filesystem doesn't align the IO requests, "
                  "aio read will use the fd without O_DIRECT.")
            self.aio_fd = self.fd

        self.checksum_all, self.n = struct.unpack("<IQ",
                                                  os.pread(self.fd, 12, 0))
        self.filesize = os.stat(fname).st_size
        metadata = os.pread(
            self.fd,
            self.n * (CHECKSUM_TYPE().itemsize + OFFSET_TYPE().itemsize), 12)

        self.check_data = check_data
        if check_data:
            self.checksums = np.frombuffer(metadata[:self.n *
                                                    CHECKSUM_TYPE().itemsize],
                                           dtype=CHECKSUM_TYPE)
        self.offsets = np.frombuffer(metadata[self.n *
                                              CHECKSUM_TYPE().itemsize:],
                                     dtype=OFFSET_TYPE)
        self.io_context = None

    def validate(self):
        """ validate checksum, this may take a long while
        """
        chunk_size = 64 * (1 << 20)  # 64 MB
        start_pos = CHECKSUM_TYPE().itemsize
        nbytes = 8 + CHECKSUM_TYPE().itemsize * self.n + OFFSET_TYPE().itemsize * self.n
        end_pos = start_pos + nbytes

        checksum = 0
        for offset in range(start_pos, end_pos, chunk_size):
            size = min(chunk_size, end_pos - offset)
            data = os.pread(self.fd, size, offset)
            checksum = zlib.crc32(data, checksum)

        if checksum != self.checksum_all:
            raise ValueError("checksum mismatched!")

        
    def read(self, indices: List[int]) -> List[bytearray]:
        """ read a batch of samples

        Args:
            indices: sample indices
        
        Return:
            return a list of bytearray, each bytearray is the raw data of one sample
        """
        batch_size = len(indices)

        if self.io_context == None:
            self.io_context = libaio.AIOContext(AIO_MAX_EVENTS)

        offsets = [self.offsets[index] for index in indices]
        offsets2idx = {offsets[i]: i for i in range(batch_size)}
        next_offsets = [
            OFFSET_TYPE(self.filesize) if index == self.n -
            1 else self.offsets[index + 1] for index in indices
        ]
        lengthes = [next_offsets[i] - offsets[i] for i in range(batch_size)]

        buffers = [bytearray(length) for length in lengthes]

        read_blocks = [
            libaio.AIOBlock(
                mode=libaio.AIOBLOCK_MODE_READ,
                target_file=self.aio_fd if batch_size > 2 else self.fd,
                buffer_list=[buffers[i]],
                offset=offsets[i]) for i in range(batch_size)
        ]

        # submit & complete
        submitted = 0
        completed = 0
        events = []
        while completed < len(read_blocks):
            if submitted < len(read_blocks):
                res = self.io_context.submit(read_blocks[submitted:])
                if res >= 0:
                    submitted += res
                else:
                    raise RuntimeError("libaio failed to submit io requests.")

            res = self.io_context.getEvents(min_nr=submitted - completed)
            completed += len(res)
            events.extend(res)

        # check
        for block, res, res2 in events:
            index = offsets2idx[block.offset]
            if res != lengthes[index]:
                raise IOError(
                    f"Sample {indices[index]}: libaio read error, read res: {res}."
                )
            if self.check_data:
                checksum = zlib.crc32(buffers[index])
                if checksum != self.checksums[indices[index]]:
                    raise ValueError(
                        f"Sample {indices[index]}: checksum mismatched!")

        return buffers

    def read_one(self, index: int) -> bytearray:
        """ read one sample given index

        Args:
            index: sample index

        Return:
            return the raw bytes data of one sample

        """
        assert index < self.n

        offset = self.offsets[index]
        if index == self.n - 1:
            size = OFFSET_TYPE(self.filesize) - offset
        else:
            size = self.offsets[index + 1] - offset

        data = os.pread(self.fd, size, offset)

        # compute checksum
        if self.check_data:
            checksum = zlib.crc32(data)
            if checksum != self.checksums[index]:
                raise ValueError(f"Sample {index}: checksum mismatched!")

        return data

    def loc(self, indexes: List[int]) -> Tuple[List[int], List[int]]:
        offsets, sizes = [], []
        for index in indexes:
            offset = self.offsets[index]
            if index == self.n - 1:
                size = OFFSET_TYPE(self.filesize) - offset
            else:
                size = self.offsets[index + 1] - offset
            offsets.append(offset)
            sizes.append(size)

        return offsets, sizes

    def close(self):
        if hasattr(self, 'fd') and self.fd:
            os.close(self.fd)
            if self.aio_fd != self.fd:
                os.close(self.aio_fd)
            self.fd = None
            self.aio_fd = None

    def __del__(self):
        self.close()
