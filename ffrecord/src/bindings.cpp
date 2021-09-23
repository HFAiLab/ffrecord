#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fileio.h>
#include <cstdint>


namespace py = pybind11;

namespace ffrecord {

PYBIND11_MODULE(_ffrecord_cpp, m) {
    m.doc() = "_ffrecord_cpp";

    m.def("checkFsAlign", &checkFsAlign);

    py::class_<FileWriter>(m, "FileWriter")
        .def(py::init<std::string, int64_t>())
        .def("write_one", &FileWriter::write_one)
        .def("close", &FileWriter::close_fd);

    py::class_<FileReader>(m, "FileReader")
        .def(py::init<std::vector<std::string>, bool>(), py::arg("fnames"), py::arg("check_data") = true)
        .def(py::init<std::string, bool>(), py::arg("fname"), py::arg("check_data") = true)
        .def_readonly("n", &FileReader::n)
        .def("read", &FileReader::read_batch)
        .def("read_one", &FileReader::read_one)
        .def("validate", &FileReader::validate)
        .def("close", &FileReader::close_fd);

}

}