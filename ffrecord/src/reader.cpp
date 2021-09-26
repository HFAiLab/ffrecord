#include <fileio.h>

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <libaio.h>
#include <errno.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "zlib.h"


namespace py = pybind11;
using byte = unsigned char;

constexpr unsigned long FS_IOCNUM_CHECK_FS_ALIGN = 2147772004;
constexpr int MAX_EVENTS = 4096;
constexpr int MIN_NR = 1;


std::string format_msg(const std::string &msg, const std::string &func,
        const std::string &file, int line) {
    std::string out = msg;
    out.back() = ' ';
    out += ", Error in " + func + " at " + file + " line " + std::to_string(line);
    return out;
}


#define FFRECORD_THROW_FMT(FMT, ...)                           \
    do {                                                       \
        std::string __s;                                       \
        int __size = snprintf(nullptr, 0, FMT, __VA_ARGS__);   \
        __s.resize(__size + 1);                                \
        snprintf(&__s[0], __s.size(), FMT, __VA_ARGS__);       \
        throw std::runtime_error(format_msg(                   \
            __s, __PRETTY_FUNCTION__, __FILE__, __LINE__));    \
    } while (false)


#define FFRECORD_ASSERT(X, FMT, ...)                                         \
    do {                                                                     \
        if (!(X)) {                                                          \
            FFRECORD_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
        }                                                                    \
    } while (false)


namespace ffrecord {

bool checkFsAlign(int fd) {
    uint32_t buf;
    int ret = ioctl(fd, FS_IOCNUM_CHECK_FS_ALIGN, &buf);
    return ret == 0 && buf == 1;
}


void free_buffer(void *buf) {
    delete[] reinterpret_cast<uint8_t*>(buf);
}


/***************************************************
 * FileHeader
 **************************************************/

FileHeader::FileHeader(const std::string &fname, bool check_data) {
    this->fname = fname;
    aiofd = fd = open(fname.data(), O_RDONLY);
    if (checkFsAlign(fd)) {
        aiofd = open(fname.data(), O_RDONLY | O_DIRECT);
    }

    // checksum of metadata, number of samples
    read(fd, &checksum_meta, sizeof(checksum_meta));
    read(fd, &n, sizeof(n));

    // offsets and checksums
    offsets.resize(n + 1);
    checksums.resize(n);
    read(fd, checksums.data(), sizeof(checksums[0]) * n);
    read(fd, offsets.data(), sizeof(offsets[0]) * n);

    // file length
    struct stat buf;
    stat(fname.data(), &buf);
    offsets[n] = buf.st_size;

    // validate checksum
    if (check_data) {
        validate();
    }
}

FileHeader::~FileHeader() {}

void FileHeader::close_fd() {
    if (fd >= 0) {
        if (aiofd != fd) {
            close(aiofd);
            aiofd = -1;
        }
        close(fd);
        fd = -1;
    }
}

void FileHeader::validate() const {
    uint32_t checksum = 0;
    checksum = crc32(checksum, (const byte*)&n, sizeof(n));
    checksum = crc32(checksum, (const byte*)checksums.data(), sizeof(checksums[0]) * n);
    checksum = crc32(checksum, (const byte*)offsets.data(), sizeof(offsets[0]) * n);

    FFRECORD_ASSERT(checksum == checksum_meta,
            "%s: checksum of metadata mismached!", fname.data());
}

void FileHeader::access(int64_t index, int *pfd, int64_t *offset, int64_t *len,
                        uint32_t *checksum, bool use_aio) const {
    *pfd = use_aio ? this->aiofd : this->fd;
    *offset = offsets[index];
    *len = offsets[index + 1] - offsets[index];
    *checksum = checksums[index];
}

/***************************************************
 * FileReader
 **************************************************/

FileReader::FileReader(const std::vector<std::string> &fnames, bool check_data) {
    this->check_data = check_data;
    nfiles = fnames.size();
    n = 0;

    nsamples.push_back(0);
    for (const auto &fname : fnames) {
        headers.emplace_back(fname, check_data);
        n += headers.back().n;
        nsamples.push_back(n);
    }
}

FileReader::FileReader(const std::string &fname, bool check_data)
        : FileReader(std::vector<std::string>({fname}), check_data) {}

FileReader::~FileReader() {
    close_fd();
    if (pctx != nullptr) {
        io_destroy(*pctx);
        delete pctx;
    }
}

void FileReader::close_fd() {
    for (auto &header : headers) {
        header.close_fd();
    }
}

void FileReader::validate() {
    for (const auto &header : headers) {
        header.validate();
    }
}

void FileReader::validate_sample(int64_t index, uint8_t *buf, int64_t len, uint32_t checksum) {
    // crc
    if (check_data) {
        uint32_t checksum2 = crc32(0, (const byte*)buf, len);
        FFRECORD_ASSERT(checksum2 == checksum,
                "sample %zd: checksum mismached!", size_t(index));
    }
}

std::vector<pybind11::array> FileReader::read_batch(const std::vector<int64_t> &indices) {
    if (indices.empty()) {
        return {};
    } else if (indices.size() < 3) {
        std::vector<py::array> results;
        for (auto index : indices) {
            results.push_back(read_one(index));
        }
        return results;
    }
    assert(indices.size() <= MAX_EVENTS);

    if (pctx == nullptr) {
        pctx = new io_context_t;
        memset(pctx, 0, sizeof(*pctx));
        io_setup(MAX_EVENTS, pctx);
    }
    auto &ctx = *pctx;

    int nr = indices.size();
    std::vector<iocb *> ios(nr);

    int aiofd;
    int64_t offset, len;
    std::vector<uint32_t> checksums(nr);

    // prepare iocbs
    for (int i = 0; i < nr; i++) {
        int64_t index = indices[i];
        int fid = 0;
        while (index >= nsamples[fid + 1]) {
            fid++;
        }
        assert(fid < nfiles);

        index = index - nsamples[fid];
        headers[fid].access(index, &aiofd, &offset, &len, &checksums[i], true);

        uint8_t *buf = new uint8_t[len];
        ios[i] = new iocb();
        io_prep_pread(ios[i], aiofd, buf, len, offset);
        ios[i]->data = (void *)(int64_t)i;
    }

    int min_nr = MIN_NR;
    int nr_completed = 0;
    int nr_submitted = 0;
    std::vector<io_event> events(nr);
    std::vector<py::array> results(nr);

    // submit & wait
    while (nr_completed < nr) {

        // submit jobs
        if (nr_submitted < nr) {
            int ns = io_submit(ctx, nr, &ios[nr_submitted]);
            assert(ns > 0);
            nr_submitted += ns;
        }

        // wait until min_nr jobs are completed
        min_nr = std::min(min_nr, nr_submitted - nr_completed);
        int ne = io_getevents(ctx, min_nr, nr_submitted, events.data(), nullptr);
        assert(ne > 0);
        nr_completed += ne;

        // postprocess
        for (int i = 0; i < ne; i++) {
            auto obj = events[i].obj;
            auto buf = (uint8_t *)(obj->u.c.buf);
            auto len = obj->u.c.nbytes;
            int64_t idx = (int64_t)obj->data;

            validate_sample(indices[idx], buf, len, checksums[idx]);
            auto capsule = py::capsule(buf, free_buffer);
            auto arr = py::array(len, buf, capsule);
            results[idx] = arr;

            delete obj;
        }
    }

    return results;
}

py::array FileReader::read_one(int64_t index) {
    int fd;
    int64_t offset, len;
    int fid = 0;
    while (index >= nsamples[fid + 1]) {
        fid++;
    }
    assert(fid < nfiles);

    uint32_t checksum;
    headers[fid].access(index - nsamples[fid], &fd, &offset, &len, &checksum, false);

    uint8_t *buf = new uint8_t[len];
    pread(fd, buf, len, offset);
    validate_sample(index, buf, len, checksum);

    auto capsule = py::capsule(buf, free_buffer);
    return py::array(len, buf, capsule);
}


}  // ffrecord
