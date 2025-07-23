# Keep empty arguments instead of undefining the variable
set(CMAKE_POLICY_DEFAULT_CMP0174 NEW)

cmake_minimum_required(VERSION 3.23)

project(tblis-external-test LANGUAGES C CXX)

find_package(TBLIS)

add_executable(shared-test test_install.cxx)
target_link_libraries(shared-test PUBLIC TBLIS::tblis)
set_target_properties(shared-test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_executable(static-test test_install.cxx)
target_link_libraries(static-test PUBLIC TBLIS::tblis-static)
set_target_properties(static-test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
