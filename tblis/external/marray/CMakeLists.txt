cmake_minimum_required(VERSION 3.20)

project(marray LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
include(GNUInstallDirs)

add_library(marray INTERFACE)
target_include_directories(marray INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(marray
    INTERFACE FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
    marray/array_1d.hpp
    marray/array_2d.hpp
    marray/detail/utility.hpp
    marray/detail/vector_avx512.hpp
    marray/detail/vector_avx.hpp
    marray/detail/vector.hpp
    marray/detail/vector_neon.hpp
    marray/detail/vector_sse41.hpp
    marray/dpd/dpd_marray_base.hpp
    marray/dpd/dpd_marray.hpp
    marray/dpd/dpd_marray_view.hpp
    marray/dpd/dpd_range.hpp
    marray/expression.hpp
    marray/fwd/expression_fwd.hpp
    marray/fwd/marray_fwd.hpp
    marray/indexed/indexed_marray_base.hpp
    marray/indexed/indexed_marray.hpp
    marray/indexed/indexed_marray_view.hpp
    marray/indexed_dpd/indexed_dpd_marray_base.hpp
    marray/indexed_dpd/indexed_dpd_marray.hpp
    marray/indexed_dpd/indexed_dpd_marray_view.hpp
    marray/index_iterator.hpp
    marray/marray_base.hpp
    marray/marray.hpp
    marray/marray_iterator.hpp
    marray/marray_slice.hpp
    marray/marray_view.hpp
    marray/range.hpp
    marray/rotate.hpp
    marray/short_vector.hpp
    marray/types.hpp)

install(TARGETS marray EXPORT marray-targets
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})