
#pragma once

#include <stdexcept>
#include <cstdint>
#include <libaio.h>


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

constexpr int64_t MAX_SIZE = 512 * (1 << 20);

bool checkFsAlign(int fd);

/** compute crc32 checksum, zlib's crc32() may not work if data size >= 4G
 *
 * @param code  starting value of the checksum
 * @param data  input data
 * @param len   data size
 */
uint32_t ffcrc32(uint32_t code, const void *data, int64_t len);

std::string format_msg(const std::string &msg, const std::string &func,
        const std::string &file, int line);

}  // ffrecord
