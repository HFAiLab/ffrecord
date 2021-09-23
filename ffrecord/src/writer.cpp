#include <fileio.h>

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include "zlib.h"

namespace py = pybind11;

namespace ffrecord {

using byte = unsigned char;


FileWriter::FileWriter(const std::string &fname, int64_t n) : n(n), count(0) {
    fd = open(fname.data(), O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
    offsets.resize(n);
    checksums.resize(n);

    sample_pos = 4 + 8 + (4 + 8) * n;
    lseek(fd, sample_pos, SEEK_SET);
}

FileWriter::~FileWriter() { close_fd(); }

void FileWriter::write_one(const pybind11::buffer &buf) {
    py::buffer_info info = buf.request();

    auto data = (const byte*)info.ptr;
    auto len = info.shape[0];

    checksums[count] = crc32(0, data, len);
    offsets[count] = sample_pos;

    write(fd, data, len);
    sample_pos += len;
    count += 1;
}

void FileWriter::finish() {
    uint32_t checksum = 0;

    assert(n == count);
    lseek(fd, 0, SEEK_SET);

    // write checksum and n
    write(fd, &checksum, sizeof(checksum));
    write(fd, &n, sizeof(n));

    // write checksums and offsets
    write(fd, checksums.data(), sizeof(checksums[0]) * n);
    write(fd, offsets.data(), sizeof(offsets[0]) * n);

    // checksum of metadata
    checksum = crc32(checksum, (const byte*)&n, sizeof(n));
    checksum = crc32(checksum, (const byte*)checksums.data(), sizeof(checksums[0]) * n);
    checksum = crc32(checksum, (const byte*)offsets.data(), sizeof(offsets[0]) * n);
    pwrite(fd, &checksum, sizeof(uint32_t), 0);

    close(fd);
    fd = -1;
}

void FileWriter::close_fd() {
    if (fd >= 0) {
        finish();
    }
}

}  // ffrecord
