
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cinttypes>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <libaio.h>
#include <errno.h>

#include <fileio.h>
#include <utils.h>


constexpr int MAX_EVENTS = 4096;
constexpr int MIN_NR = 1;


namespace ffrecord {

/***************************************************
 * FileHeader
 **************************************************/

FileHeader::FileHeader(const std::string &fname, bool check_data) {
    this->fname = fname;
    aiofd = fd = open(fname.data(), O_RDONLY);
    FFRECORD_ASSERT(fd > 0, "Failed to open file %s: %s", fname.data(), strerror(errno));
    if (checkFsAlign(fd)) {
        aiofd = open(fname.data(), O_RDONLY | O_DIRECT);
        FFRECORD_ASSERT(aiofd > 0,
                "Failed to open file with O_DIRECT %s: %s",
                fname.data(), strerror(errno));
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

    // for barckward compatibility
    if (checksum_meta == 0) {
        fprintf(stderr, "warning: you are using an old version ffrecord file, please update the file\n");
        return;
    }

    uint32_t checksum = 0;
    checksum = ffcrc32(checksum, &n, sizeof(n));
    checksum = ffcrc32(checksum, checksums.data(), sizeof(checksums[0]) * n);
    checksum = ffcrc32(checksum, offsets.data(), sizeof(offsets[0]) * n);

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
        uint32_t checksum2 = ffcrc32(0, buf, len);
        FFRECORD_ASSERT(checksum2 == checksum,
                "Sample %" PRId64 ": checksum mismached!", index);
    }
}

std::vector<MemBlock> FileReader::read_batch(const std::vector<int64_t> &indices) {
    if (indices.empty()) {
        return {};
    } else if (indices.size() < 3) {
        std::vector<MemBlock> results;
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

    int nr = indices.size();  // number of reads
    std::vector<iocb *> ios;
    ios.reserve(nr);

    int aiofd;
    int64_t offset, len;
    std::vector<uint32_t> checksums(nr);  // checksum of each sample
    std::vector<int> nblocks(nr, 0);      // number of blocks for each sample
    std::vector<MemBlock> buffers(nr);    // results to be returned

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

        // allocate memory
        auto &buf = buffers[i];
        buf.data = new uint8_t[len];
        buf.len = len;

        // split data into multiple blocks if too large
        for (int64_t start = 0; start < len; start += MAX_SIZE) {
            int64_t len_i = std::min(MAX_SIZE, len - start);
            ios.push_back(new iocb());
            io_prep_pread(ios.back(), aiofd, buf.data + start, len_i, offset + start);
            ios.back()->data = (void *)(int64_t)i;
            nblocks[i] += 1;
        }
    }

    int min_nr = MIN_NR;
    int nr_completed = 0;  // number of requests completed
    int nr_submitted = 0;  // number of requests submitted
    std::vector<io_event> events(ios.size());
    std::vector<int> nblocks_completed(nr, 0);
    nr = ios.size();  // number of requests in total

    // submit & wait
    while (nr_completed < nr) {

        // submit jobs
        if (nr_submitted < nr) {
            int ns = io_submit(ctx, nr - nr_submitted, &ios[nr_submitted]);
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
            auto len = obj->u.c.nbytes;
            int64_t idx = (int64_t)obj->data;

            FFRECORD_ASSERT(events[i].res == len,
                    "Sample %" PRId64 ": length %lu but read %lu bytes",
                    indices[idx], len, events[i].res);

            nblocks_completed[idx] += 1;
            if (nblocks_completed[idx] == nblocks[idx]) {
                validate_sample(indices[idx], buffers[idx].data, buffers[idx].len, checksums[idx]);
            }

            delete obj;
        }
    }

    return buffers;
}

MemBlock FileReader::read_one(int64_t index) {
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
    for (int64_t start = 0; start < len; start += MAX_SIZE) {
        int64_t len_i = std::min(MAX_SIZE, len - start);
        int64_t res = pread(fd, buf + start, len_i, offset + start);
        FFRECORD_ASSERT(res == len_i,
                "Sample %" PRId64 ": length %" PRId64 " but read %" PRId64 " bytes",
                index, len_i, res);
    }
    validate_sample(index, buf, len, checksum);

    return MemBlock(buf, len);
}

}  // ffrecord
