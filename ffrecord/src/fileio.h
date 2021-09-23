
#pragma once

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


/****************************************************

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

****************************************************/


namespace ffrecord {

bool checkFsAlign(int fd);


class FileWriter {

  public:

    /** Open a file and write to it
     *
     * @param fname opened file name
     * @param n     number of samples to be written
     */
    FileWriter(const std::string &fname, int64_t n);

    ~FileWriter();

    /** Write one sample
     *
     * @param buf binary data of one sample
     */
    void write_one(const pybind11::buffer &buf);

    ///< Close opened file
    void close_fd();

  private:

    void finish();

    int fd;
    int64_t n;
    int64_t count;
    int64_t sample_pos;
    std::vector<int64_t> offsets;
    std::vector<uint32_t> checksums;

};


class FileHeader {
  public:
    /** Open a file and read header from it
     *
     * @param fname       opened file name
     * @param check_data  validate checksum or not
     */
    FileHeader(const std::string &fname, bool check_data);

    ~FileHeader();

    ///< Validate metadata
    void validate() const;

    /** Read metadata of one sample
     *
     * @param index     index of the sample
     * @param fd        file descriptor
     * @param offset    file offset of the sample
     * @param len       length of the sample
     * @param checksum  checksum of the sample
     */
    void access(int64_t index, int *fd, int64_t *offset, int64_t *len,
                uint32_t *checksum, bool use_aio) const;

    ///< Close opened file
    void close_fd();

    ///< number of samples
    int64_t n;

  private:
    std::string fname;
    int fd;
    int aiofd;

    uint32_t checksum_meta;
    std::vector<int64_t> offsets;
    std::vector<uint32_t> checksums;
};

class FileReader {

  public:

    /** Open multiple files and read from them
     *
     * @param fnames      opened files
     * @param check_data  validate checksum or not
     */
    FileReader(const std::vector<std::string> &fnames, bool check_data = true);

    /** Open a file and read from it
     *
     * @param fname       opened file name
     * @param check_data  validate checksum or not
     */
    FileReader(const std::string &fname, bool check_data = true);

    ~FileReader();

    ///< Validate metadata
    void validate();

    /** Read a batch of samples
     *
     * @param indices  index of each sample
     */
    std::vector<pybind11::array> read_batch(const std::vector<int64_t> &indices);

    /** Read one sample
     *
     * @param index  index of one sample
     */
    pybind11::array read_one(int64_t index);

    ///< Close opened file
    void close_fd();

    ///< number of samples
    int64_t n;

  private:

    inline void validate_sample(int64_t index, uint8_t *buf, int64_t len, uint32_t checksum);

    bool check_data;
    int nfiles;
    std::vector<FileHeader> headers;
    std::vector<int64_t> nsamples;
};


}  // ffrecord