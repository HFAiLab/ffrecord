cmake_minimum_required(VERSION 3.10)

project(ffrecord)

if( "${PYBIND11_CMAKE_DIR}" STREQUAL "" )
    message("PYBIND11_CMAKE_DIR is not set")
else()
    message("PYBIND11_CMAKE_DIR is set to: ${PYBIND11_CMAKE_DIR}")
endif()

set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

##### PYBIND11 WRAPPER #####
pybind11_add_module(_ffrecord_cpp SHARED
    reader.cpp
    writer.cpp
    utils.cpp
    bindings.cpp
)
target_include_directories(_ffrecord_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(_ffrecord_cpp PRIVATE aio z)
target_compile_options(_ffrecord_cpp PRIVATE -Wall -Wno-unused-result -O3)
