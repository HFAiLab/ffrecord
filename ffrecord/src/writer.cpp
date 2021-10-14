
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <fileio.h>
#include <utils.h>


namespace ffrecord {

FileWriter::FileWriter(const std::string &fname, int64_t n) : n(n), count(0) {
    fd = open(fname.data(), O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
    offsets.resize(n);
    checksums.resize(n);

    sample_pos = 4 + 8 + (4 + 8) * n;
    lseek(fd, sample_pos, SEEK_SET);
}

FileWriter::~FileWriter() { close_fd(); }

void FileWriter::write_one(const uint8_t *data, int64_t len) {
    checksums[count] = ffcrc32(0, data, len);
    offsets[count] = sample_pos;

    for (int64_t start = 0; start < len; start += MAX_SIZE) {
        int64_t len_i = std::min(MAX_SIZE, len - start);
        int64_t res = write(fd, data + start, len_i);
        FFRECORD_ASSERT(res == len_i,
                "Sample %" PRId64 ": length %" PRId64 " but wrote %" PRId64 " bytes",
                count, len_i, res);
    }
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
    checksum = ffcrc32(checksum, &n, sizeof(n));
    checksum = ffcrc32(checksum, checksums.data(), sizeof(checksums[0]) * n);
    checksum = ffcrc32(checksum, offsets.data(), sizeof(offsets[0]) * n);
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
