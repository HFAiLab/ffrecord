#include <algorithm>
#include <string>

#include <sys/ioctl.h>
#include <fcntl.h>
#include <zlib.h>

#include "utils.h"

namespace ffrecord {

constexpr unsigned long FS_IOCNUM_CHECK_FS_ALIGN = 2147772004;

bool checkFsAlign(int fd) {
    uint32_t buf;
    int ret = ioctl(fd, FS_IOCNUM_CHECK_FS_ALIGN, &buf);
    return ret == 0 && buf == 1;
}

uint32_t ffcrc32(uint32_t code, const void *data, int64_t len) {
    for (int64_t start = 0; start < len; start += MAX_SIZE) {
        int64_t len_i = std::min(MAX_SIZE, len - start);
        code = crc32(code, (const unsigned char *)data + start, len_i);
    }
    return code;
}

std::string format_msg(const std::string &msg, const std::string &func,
        const std::string &file, int line) {
    std::string out = msg;
    out.back() = ' ';
    out += ", Error in " + func + " at " + file + " line " + std::to_string(line);
    return out;
}

}  // ffrecord
