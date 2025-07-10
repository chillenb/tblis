cmake_minimum_required(VERSION 3.20)

project(stl_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
include(GNUInstallDirs)

add_library(stl_ext INTERFACE)
target_include_directories(stl_ext INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(stl_ext
    INTERFACE FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
    stl_ext/algorithm.hpp
    stl_ext/any.hpp
    stl_ext/array.hpp
    stl_ext/bounded_vector.hpp
    stl_ext/complex.hpp
    stl_ext/cosort.hpp
    stl_ext/fill_iterator.hpp
    stl_ext/global_ptr.hpp
    stl_ext/iostream.hpp
    stl_ext/ptr_list.hpp
    stl_ext/ptr_vector.hpp
    stl_ext/string.hpp
    stl_ext/type_traits.hpp
    stl_ext/vector.hpp
    stl_ext/zip.hpp
)
    

install(TARGETS stl_ext EXPORT stl_ext-targets
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)