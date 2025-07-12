find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Python 3.8 COMPONENTS Development.SABIModule)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

find_package(nanobind REQUIRED)

nanobind_add_module(pytblis STABLE_ABI
    pytblis.cxx
)

set_target_properties(pytblis PROPERTIES CXX_STANDARD 20)

target_link_libraries(pytblis PRIVATE tblis)


install(TARGETS pytblis 
    LIBRARY DESTINATION "${Python_SITEARCH}"
)