cmake_minimum_required(VERSION 3.10)

project(ffrecord)

find_package(pybind11)

##### FFRECORD #####
add_library(ffrecord SHARED
    reader.cpp
    writer.cpp
    utils.cpp
)
target_include_directories(ffrecord PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(ffrecord PRIVATE aio z)
target_compile_options(ffrecord PRIVATE -Wno-unused-result)


##### PYBIND11 WRAPPER #####
pybind11_add_module(_ffrecord_cpp SHARED
    bindings.cpp
)
target_include_directories(_ffrecord_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(_ffrecord_cpp PRIVATE ffrecord)

