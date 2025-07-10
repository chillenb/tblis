cmake_minimum_required(VERSION 3.25)

project(tblis LANGUAGES C CXX VERSION 2.0 HOMEPAGE_URL "http://www.github.com/devinamatthews/tblis")
set(CMAKE_CXX_STANDARD 20)

include(GNUInstallDirs)
include(ExternalProject)

include(CheckIncludeFile)
include(CheckIncludeFiles)
include(CheckCompilerFlag)

include(CheckFunctionExists)
include(CheckSymbolExists)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")


set(TBLIS_SOURCES
    tblis/frame/0/add.cxx
    tblis/frame/0/mult.cxx
    tblis/frame/0/reduce.cxx
    tblis/frame/1m/packm/packm_blk_bsmtc.cxx
    tblis/frame/1m/packm/packm_blk_dpd.cxx
    tblis/frame/1t/dense/add.cxx
    tblis/frame/1t/dense/dot.cxx
    tblis/frame/1t/dense/reduce.cxx
    tblis/frame/1t/dense/scale.cxx
    tblis/frame/1t/dense/set.cxx
    tblis/frame/1t/dense/shift.cxx
    tblis/frame/1t/dpd/add.cxx
    tblis/frame/1t/dpd/dot.cxx
    tblis/frame/1t/dpd/reduce.cxx
    tblis/frame/1t/dpd/scale.cxx
    tblis/frame/1t/dpd/set.cxx
    tblis/frame/1t/dpd/shift.cxx
    tblis/frame/1t/indexed/add.cxx
    tblis/frame/1t/indexed/dot.cxx
    tblis/frame/1t/indexed/reduce.cxx
    tblis/frame/1t/indexed/scale.cxx
    tblis/frame/1t/indexed/set.cxx
    tblis/frame/1t/indexed/shift.cxx
    tblis/frame/1t/indexed_dpd/add.cxx
    tblis/frame/1t/indexed_dpd/dot.cxx
    tblis/frame/1t/indexed_dpd/reduce.cxx
    tblis/frame/1t/indexed_dpd/scale.cxx
    tblis/frame/1t/indexed_dpd/set.cxx
    tblis/frame/1t/indexed_dpd/shift.cxx
    tblis/frame/1t/add.cxx
    tblis/frame/1t/dot.cxx
    tblis/frame/1t/reduce.cxx
    tblis/frame/1t/scale.cxx
    tblis/frame/1t/set.cxx
    tblis/frame/1t/shift.cxx
    tblis/frame/3m/gemm/gemm_ker_bsmtc.cxx
    tblis/frame/3m/gemm/gemm_ker_dpd.cxx
    tblis/frame/3t/dense/mult.cxx
    tblis/frame/3t/dpd/mult.cxx
    tblis/frame/3t/indexed/mult.cxx
    tblis/frame/3t/indexed_dpd/mult.cxx
    tblis/frame/3t/mult.cxx
    tblis/frame/base/basic_types.cxx
    tblis/frame/base/block_scatter.cxx
    tblis/frame/base/dpd_block_scatter.cxx
    tblis/frame/base/env.cxx
    tblis/frame/base/tensor.cxx
    tblis/frame/base/thread.cxx
)



add_library(tblis SHARED ${TBLIS_SOURCES})



# Read the BLIS git tag
file(READ blis-git-tag BLIS_GIT_TAG)
string(STRIP ${BLIS_GIT_TAG} BLIS_GIT_TAG)

# find Make
find_program(MAKE_EXECUTABLE NAMES gmake make mingw32-make REQUIRED)


# download and build BLIS
set(BLIS_CONFIG_FAMILY "auto" CACHE STRING "Set the configuration family for which the BLIS library will be built.")
set(BLIS_CONFIG_FLAGS -tomp,pthreads;--disable-blas;--disable-cblas;--enable-static;--disable-shared CACHE STRING "Set the configuration flags for the BLIS library.")


set(BLIS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/blis CACHE PATH "Set the installation prefix for the BLIS library.")
set(BLIS_CONFIGURE_PLUGIN_SCRIPT ${BLIS_INSTALL_PREFIX}/share/blis/configure-plugin)
set(BLIS_STATIC_LIBRARY ${BLIS_INSTALL_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}blis${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Add(blis_external
GIT_REPOSITORY https://github.com/flame/blis.git
GIT_TAG ${BLIS_GIT_TAG}
UPDATE_DISCONNECTED ON
CONFIGURE_COMMAND <SOURCE_DIR>/configure ${BLIS_CONFIG_FLAGS} --prefix=${CMAKE_CURRENT_BINARY_DIR}/blis ${BLIS_CONFIG_FAMILY}
INSTALL_COMMAND ${MAKE_EXECUTABLE} install
TEST_COMMAND ""
BUILD_BYPRODUCTS ${BLIS_CONFIGURE_PLUGIN_SCRIPT}
    ${BLIS_STATIC_LIBRARY}
    ${BLIS_INSTALL_PREFIX}/include/blis/blis.h
)

add_library(blis STATIC IMPORTED GLOBAL)

set(BLIS_INCLUDE_DIRS ${BLIS_INSTALL_PREFIX}/include)
file(MAKE_DIRECTORY ${BLIS_INCLUDE_DIRS})


set_target_properties(blis PROPERTIES
    IMPORTED_LOCATION ${BLIS_STATIC_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/blis/include
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}/blis/include/blis
)


add_dependencies(tblis blis_external)


set(TBLIS_LABEL_TYPE "char" CACHE STRING "Default index label type (default: char)")
set(TBLIS_LEN_TYPE "ptrdiff_t" CACHE STRING "Default length type (default: ptrdiff_t)")
set(TBLIS_STRIDE_TYPE "ptrdiff_t" CACHE STRING "Default stride type (default: ptrdiff_t)")
set(TBLIS_RESTRICT "__restrict__" CACHE STRING "Default restrict keyword (default: __restrict__)")

# Hardcode this for now, since we set CMAKE_CXX_STANDARD to 20
set(TBLIS_HAVE_CXX20 ON)

check_include_file(dlfcn.h TBLIS_HAVE_DLFCN_H)
check_include_file(hbwmalloc.h TBLIS_HAVE_HBWMALLOC_H)
check_include_file(hwloc.h TBLIS_HAVE_HWLOC_H)
check_include_file(memkind.h TBLIS_HAVE_MEMKIND_H)
check_include_file(inttypes.h TBLIS_HAVE_INTTYPES_H)
check_include_file(stdint.h TBLIS_HAVE_STDINT_H)
check_include_file(stdio.h TBLIS_HAVE_STDIO_H)
check_include_file(stdlib.h TBLIS_HAVE_STDLIB_H)
check_include_file(string.h TBLIS_HAVE_STRING_H)
check_include_file(strings.h TBLIS_HAVE_STRINGS_H)
check_function_exists(sysconf TBLIS_HAVE_SYSCONF)
check_function_exists(sysctl TBLIS_HAVE_SYSCTL)
check_function_exists(sysctlbyname TBLIS_HAVE_SYSCTLBYNAME)
check_include_file(sys/stat.h TBLIS_HAVE_SYS_STAT_H)
check_include_file(sys/types.h TBLIS_HAVE_SYS_TYPES_H)
check_include_file(unistd.h TBLIS_HAVE_UNISTD_H)
check_symbol_exists(_SC_NPROCESSORS_CONF unistd.h TBLIS_HAVE__SC_NPROCESSORS_CONF)
check_symbol_exists(_SC_NPROCESSORS_ONLN unistd.h TBLIS_HAVE__SC_NPROCESSORS_ONLN)

find_program(LSCPU_EXECUTABLE NAMES lscpu)
if(${LSCPU_EXECUTABLE} STREQUAL "LSCPU_EXECUTABLE-NOTFOUND")
    set(TBLIS_HAVE_LSCPU OFF)
else()
    set(TBLIS_HAVE_LSCPU ON)
endif()

check_compiler_flag(CXX -fopenmp-simd HAVE_FOPENMP_SIMD)
if(HAVE_FOPENMP_SIMD)
    list(APPEND CMAKE_CXX_FLAGS -fopenmp-simd)
    list(APPEND CMAKE_C_FLAGS -fopenmp-simd)
endif()

check_compiler_flag(CXX -qopenmp-simd HAVE_QOPENMP_SIMD)
if(HAVE_QOPENMP_SIMD)
    list(APPEND CMAKE_CXX_FLAGS -qopenmp-simd)
    list(APPEND CMAKE_C_FLAGS -qopenmp-simd)
endif()


# Mimic autotools tblis_config.h. Not sure what actually gets used.
set(TBLIS_PACKAGE ${PROJECT_NAME})
set(TBLIS_PACKAGE_NAME ${PROJECT_NAME})
set(TBLIS_PACKAGE_TARNAME ${PROJECT_NAME})
set(TBLIS_TOPDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TBLIS_PACKAGE_URL ${PROJECT_HOMEPAGE_URL})
set(TBLIS_VERSION "${PROJECT_VERSION}")
set(TBLIS_PACKAGE_VERSION "${PROJECT_VERSION}")
set(TBLIS_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(TBLIS_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(TBLIS_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
if(TBLIS_VERSION_PATCH STREQUAL "")
    set(TBLIS_VERSION_PATCH "0")
endif()


set(TBLIS_PACKAGE_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")

configure_file(tblis_config_h_cmake.in tblis/tblis_config.h)

target_include_directories(tblis PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tblis>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(tblis PUBLIC marray tci stl_ext)



set(PLUGIN_FLAGS_LIST
    ${CMAKE_CXX_FLAGS}
    -std=c++20
    -I${CMAKE_CURRENT_SOURCE_DIR}/tblis
    -I${CMAKE_CURRENT_BINARY_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/stl_ext
    -I${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/marray
    -I${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/tci
)

string(REPLACE ";" " " PLUGIN_FLAGS "${PLUGIN_FLAGS_LIST}")

# Build the BLIS plugin for TBLIS
ExternalProject_Add_Step(blis_external
    build_tblis_plugin
    DEPENDEES install
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tblis/plugin
    JOB_SERVER_AWARE 1
    COMMAND ${BLIS_CONFIGURE_PLUGIN_SCRIPT} CXX=${CMAKE_CXX_COMPILER} CXXFLAGS=${PLUGIN_FLAGS} -f --build --path=${CMAKE_CURRENT_SOURCE_DIR}/tblis/plugin tblis
    COMMAND ${MAKE_EXECUTABLE}
)

# the plugin archive name (e.g., libblis_tblis.a)
set(TBLIS_PLUGIN_ARCHIVE ${CMAKE_STATIC_LIBRARY_PREFIX}blis_tblis${CMAKE_STATIC_LIBRARY_SUFFIX})

# copy the plugin archive to the build directory
# workaround because CMake doesn't know the BLIS confname at configure time
add_custom_target(tblis_plugin_tgt
    COMMENT "Copy BLIS plugin to build directory"
    DEPENDS blis_external
    COMMAND cat ${BLIS_INSTALL_PREFIX}/share/blis/config.mk | grep '^CONFIG_NAME' | awk '{print $$3}' > BLIS_CONFNAME.tmp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/tblis/plugin/lib/`cat BLIS_CONFNAME.tmp`/${TBLIS_PLUGIN_ARCHIVE} ${CMAKE_CURRENT_BINARY_DIR}
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/${TBLIS_PLUGIN_ARCHIVE}
    SOURCES tblis/tblis_config.h
)

add_library(tblis_plugin_lib STATIC IMPORTED GLOBAL)
set_target_properties(tblis_plugin_lib PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${TBLIS_PLUGIN_ARCHIVE}
)

add_dependencies(tblis tblis_plugin_tgt)

# finally link everything together
target_link_libraries(tblis PRIVATE blis tblis_plugin_lib)

add_subdirectory(tblis/external/marray)
add_subdirectory(tblis/external/tci)
add_subdirectory(tblis/external/stl_ext)

option(TBLIS_BUILD_TESTS "Build tblis tests" OFF)
if(TBLIS_BUILD_TESTS)
  add_executable(tblis_test
    test/test.cxx
    test/random.cxx
    test/1t/dot.cxx
    test/1t/reduce.cxx
    test/1t/replicate.cxx
    test/1t/scale.cxx
    test/1t/trace.cxx
    test/1t/transpose.cxx
    test/3m/gemm.cxx
    test/3m/gemv.cxx
    test/3m/ger.cxx
    test/3t/contract.cxx
    test/3t/mult.cxx
    test/3t/outer_prod.cxx
    test/3t/weight.cxx
  )
  target_link_libraries(tblis_test tblis)
  target_include_directories(tblis_test PRIVATE tblis/external/catch)
endif()


# Select header files to be installed:

target_sources(tblis
    INTERFACE FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
    tblis/frame/base/aligned_allocator.hpp
    tblis/frame/base/alignment.hpp
    tblis/frame/base/basic_types.h
    tblis/frame/base/thread.h
    tblis/frame/1t/add.h
    tblis/frame/1t/reduce.h
    tblis/frame/1t/scale.h
    tblis/frame/1t/set.h
    tblis/frame/1t/shift.h
    tblis/frame/3t/mult.h
    tblis/tblis.h
    tblis.h
)

target_sources(tblis
    PUBLIC FILE_SET config_header
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/tblis/tblis_config.h
)


install(TARGETS tblis
    EXPORT tblis-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILE_SET config_header DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)