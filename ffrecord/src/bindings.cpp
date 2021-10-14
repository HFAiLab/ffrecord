#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>

#include <fileio.h>
#include <utils.h>

namespace py = pybind11;

namespace ffrecord {

void free_buffer(void *buf) {
    delete[] reinterpret_cast<uint8_t*>(buf);
}

class WriterWrapper : public FileWriter {
public:
    WriterWrapper(const std::string &fname, int64_t n) : FileWriter(fname, n) {}

    void write_one_wrapper(const pybind11::buffer &buf) {
        py::buffer_info info = buf.request();
        auto data = (const uint8_t*)info.ptr;
        auto len = info.shape[0];
        write_one(data, len);
    }
};

class ReaderWrapper : public FileReader {
public:
    ReaderWrapper(const std::vector<std::string> &fnames, bool check_data = true)
            : FileReader(fnames, check_data) {}

    ReaderWrapper(const std::string &fname, bool check_data = true)
            : FileReader(fname, check_data) {}

    std::vector<py::array> read_batch_wrapper(const std::vector<int64_t> &indices) {
        auto blocks = read_batch(indices);
        std::vector<py::array> results;
        for (const auto &b : blocks) {
            auto capsule = py::capsule(b.data, free_buffer);
            results.emplace_back(b.len, b.data, capsule);
        }
        return results;
    }

    py::array read_one_wrapper(int64_t index) {
        auto b = read_one(index);
        auto capsule = py::capsule(b.data, free_buffer);
        return py::array(b.len, b.data, capsule);
    }

    py::array_t<int64_t> get_offsets(int file_index) {
        auto &v = headers[file_index].offsets;
        auto capsule = py::capsule(v.data(), [](void*) {});
        return py::array(v.size(), v.data(), capsule);
    }

    py::array_t<uint32_t> get_checksums(int file_index) {
        auto &v = headers[file_index].checksums;
        auto capsule = py::capsule(v.data(), [](void*) {});
        return py::array(v.size(), v.data(), capsule);
    }
};


PYBIND11_MODULE(_ffrecord_cpp, m) {
    m.doc() = "_ffrecord_cpp";

    m.def("checkFsAlign", &checkFsAlign);

    py::class_<WriterWrapper>(m, "FileWriter")
        .def(py::init<std::string, int64_t>())
        .def("write_one", &WriterWrapper::write_one_wrapper)
        .def("close", &WriterWrapper::close_fd);

    py::class_<ReaderWrapper>(m, "FileReader")
        .def(py::init<std::vector<std::string>, bool>(), py::arg("fnames"), py::arg("check_data") = true)
        .def(py::init<std::string, bool>(), py::arg("fname"), py::arg("check_data") = true)
        .def_readonly("n", &ReaderWrapper::n)
        .def("read", &ReaderWrapper::read_batch_wrapper)
        .def("read_one", &ReaderWrapper::read_one_wrapper)
        .def("validate", &ReaderWrapper::validate)
        .def("get_offsets", &ReaderWrapper::get_offsets)
        .def("get_checksums", &ReaderWrapper::get_checksums)
        .def("close", &ReaderWrapper::close_fd);

}

}