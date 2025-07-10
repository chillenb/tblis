cmake_minimum_required(VERSION 3.20)

project(tci LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
include(GNUInstallDirs)
include(CheckFunctionExists)
include(CheckIncludeFile)


add_library(tci SHARED
tci/barrier.c
tci/communicator.c
tci/task_set.c
tci/context.c
tci/mutex.c
tci/parallel.c
tci/slot.c
tci/work_item.c
)


set(TCI_BARRIER_TYPE "auto" CACHE STRING "use a barrier of the specified type, one of auto, pthread_barrier, spin_barrier (default: auto).")
set_property(CACHE TCI_BARRIER_TYPE PROPERTY STRINGS "pthread_barrier" "spin_barrier", "auto")

set(TCI_THREAD_MODEL "auto" CACHE STRING "enable threading with the specified model, one of none, auto, task, openmp, pthreads, tbb, omptask, winthreads, ppl, or dispatch (default: auto).")
set_property(CACHE TCI_THREAD_MODEL PROPERTY STRINGS "openmp" "pthread" "windows" "tbb" "ppl" "dispatch" "omptask" "auto")

set(TCI_MUTEX_TYPE "auto" CACHE STRING "use a mutex of the specified type, one of auto, pthread_mutex, pthread_spinlock, osx_spinlock, os_unfair_lock, omp_lock, atomic_spinlock (default: auto)")



find_package(OpenMP)
find_package(Threads)
check_function_exists(pthread_barrier_init HAVE_PTHREAD_BARRIER_INIT)
check_function_exists(pthread_spin_init HAVE_PTHREAD_SPIN_INIT)
check_function_exists(pthread_mutex_init HAVE_PTHREAD_MUTEX_INIT)
find_package(TBB)
check_include_file("ppl.h" HAVE_PPL)

find_package(dispatch)


check_include_file("os/lock.h" HAVE_OS_LOCK_H)
check_function_exists(os_unfair_lock_lock HAVE_OS_UNFAIR_LOCK_LOCK)
if(HAVE_OS_LOCK_H AND HAVE_OS_UNFAIR_LOCK_LOCK)
  set(HAVE_OS_UNFAIR_LOCK ON)
endif()


check_include_file("libkern/OSAtomic.h" HAVE_LIBKERN_OSATOMIC_H)
check_function_exists(OSSpinLockLock HAVE_OSSPINLOCK_LOCK)
if(HAVE_LIBKERN_OSATOMIC_H AND HAVE_OSSPINLOCK_LOCK)
  set(HAVE_OSX_SPINLOCK ON)
endif()


# Check for GCC atomic builtin functions
include(CheckCSourceCompiles)

check_c_source_compiles("
int main() {
    // Test all atomic builtin functions
    __atomic_load_n((int*)0, __ATOMIC_SEQ_CST);
    __atomic_load((int*)0, (int*)0, __ATOMIC_SEQ_CST);
    __atomic_store_n((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_store((int*)0, (int*)0, __ATOMIC_SEQ_CST);
    __atomic_exchange_n((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_exchange((int*)0, (int*)0, (int*)0, __ATOMIC_SEQ_CST);
    __atomic_compare_exchange_n((int*)0, (int*)0, 0, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    __atomic_compare_exchange((int*)0, (int*)0, (int*)0, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    __atomic_add_fetch((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_sub_fetch((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_and_fetch((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_xor_fetch((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_or_fetch((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_nand_fetch((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_fetch_add((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_fetch_sub((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_fetch_and((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_fetch_xor((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_fetch_or((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_fetch_nand((int*)0, 1, __ATOMIC_SEQ_CST);
    __atomic_test_and_set((char*)0, __ATOMIC_SEQ_CST);
    __atomic_clear((_Bool*)0, __ATOMIC_SEQ_CST);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __atomic_always_lock_free(4, 0);
    __atomic_is_lock_free(4, 0);
    return 0;
}" HAVE_GCC_ATOMIC_BUILTINS)

if(NOT HAVE_GCC_ATOMIC_BUILTINS)
  message(FATAL_ERROR "GCC atomic builtins are required but not available.")
endif()

include(CheckLibraryExists)
check_library_exists(atomic __atomic_compare_exchange_16 "" HAVE_ATOMIC_LIB)


# determine mutex type

# Determine mutex type automatically if set to "auto"
if(TCI_MUTEX_TYPE STREQUAL "auto")
    if(HAVE_OS_UNFAIR_LOCK)
        set(TCI_MUTEX_TYPE "os_unfair_lock")
    elseif(HAVE_OSX_SPINLOCK)
        set(TCI_MUTEX_TYPE "osx_spinlock")
    elseif(HAVE_PTHREAD_SPIN_INIT)
        set(TCI_MUTEX_TYPE "pthread_spinlock")
    else()
        set(TCI_MUTEX_TYPE "atomic_spinlock")
    endif()
endif()

# Initialize all mutex type flags to 0
set(USE_PTHREAD_MUTEX 0)
set(USE_PTHREAD_SPINLOCK 0)
set(USE_OS_UNFAIR_LOCK 0)
set(USE_OSX_SPINLOCK 0)
set(USE_OMP_LOCK 0)
set(USE_ATOMIC_SPINLOCK 0)

# Set the appropriate flag based on selected mutex type
message(STATUS "Checking mutex type")
if(TCI_MUTEX_TYPE STREQUAL "pthread_mutex")
    if(NOT HAVE_PTHREAD_MUTEX_INIT)
        message(FATAL_ERROR "pthread mutex requested but not available")
    endif()
    set(USE_PTHREAD_MUTEX 1)
    target_link_libraries(tci PRIVATE Threads::Threads)
elseif(TCI_MUTEX_TYPE STREQUAL "pthread_spinlock")
    if(NOT HAVE_PTHREAD_SPIN_INIT)
        message(FATAL_ERROR "pthread spinlock requested but not available")
    endif()
    set(USE_PTHREAD_SPINLOCK 1)
    target_link_libraries(tci PRIVATE Threads::Threads)
elseif(TCI_MUTEX_TYPE STREQUAL "os_unfair_lock")
    if(NOT HAVE_OS_UNFAIR_LOCK)
        message(FATAL_ERROR "macOS os_unfair_lock requested but not available")
    endif()
    set(USE_OS_UNFAIR_LOCK 1)
elseif(TCI_MUTEX_TYPE STREQUAL "osx_spinlock")
    if(NOT HAVE_OSX_SPINLOCK)
        message(FATAL_ERROR "OSX OSSpinLock requested but not available")
    endif()
    set(USE_OSX_SPINLOCK 1)
elseif(TCI_MUTEX_TYPE STREQUAL "omp_lock")
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "omp_lock requested but not available")
    endif()
    set(USE_OMP_LOCK 1)
elseif(TCI_MUTEX_TYPE STREQUAL "atomic_spinlock")
    set(USE_ATOMIC_SPINLOCK 1)
    target_link_libraries(tci PRIVATE atomic)
else()
    message(FATAL_ERROR "invalid mutex type specified: ${TCI_MUTEX_TYPE}")
endif()
message(STATUS "Checking mutex type - ${TCI_MUTEX_TYPE}")



# determine barrier type
if(TCI_BARRIER_TYPE STREQUAL "auto")
    set(TCI_BARRIER_TYPE "spin_barrier")
endif()

set(USE_PTHREAD_BARRIER 0)
set(USE_SPIN_BARRIER 0)

message(STATUS "Checking barrier type")
if(TCI_BARRIER_TYPE STREQUAL "pthread_barrier")
    if(NOT HAVE_PTHREAD_BARRIER_INIT)
        message(FATAL_ERROR "pthread barrier requested but not available")
    endif()
    set(USE_PTHREAD_BARRIER 1)
    target_link_libraries(tci PRIVATE Threads::Threads)
elseif(TCI_BARRIER_TYPE STREQUAL "spin_barrier")
    set(USE_SPIN_BARRIER 1)
    target_link_libraries(tci PRIVATE atomic)
else()
    message(FATAL_ERROR "invalid barrier type specified: ${TCI_BARRIER_TYPE}")
endif()
message(STATUS "Checking barrier type - ${TCI_BARRIER_TYPE}")

# determine thread model
if(TCI_THREAD_MODEL STREQUAL "auto")
    if(OpenMP_FOUND)
        set(TCI_THREAD_MODEL "openmp")
    elseif(Threads_FOUND)
        set(TCI_THREAD_MODEL "pthread")
    elseif(WIN32)
        set(TCI_THREAD_MODEL "windows")
    elseif(TBB_FOUND)
        set(TCI_THREAD_MODEL "tbb")
    elseif(HAVE_PPL)
        set(TCI_THREAD_MODEL "ppl")
    elseif(dispatch_FOUND)
        set(TCI_THREAD_MODEL "dispatch")
    else()
        set(TCI_THREAD_MODEL "none")
    endif()
elseif(TCI_THREAD_MODEL STREQUAL "task")
    if(TBB_FOUND)
        set(TCI_THREAD_MODEL "tbb")
    elseif(HAVE_PPL)
        set(TCI_THREAD_MODEL "ppl")
    elseif(dispatch_FOUND)
        set(TCI_THREAD_MODEL "dispatch")
    elseif(HAVE_OPENMP)
        set(TCI_THREAD_MODEL "omptask")
    else()
        set(TCI_THREAD_MODEL "none")
    endif()
endif()

# Initialize all thread model flags to 0
set(USE_OPENMP_THREADS 0)
set(USE_PTHREADS_THREADS 0)
set(USE_TBB_THREADS 0)
set(USE_DISPATCH_THREADS 0)
set(USE_WINDOWS_THREADS 0)
set(USE_PPL_THREADS 0)
set(USE_OMPTASK_THREADS 0)

# Set the appropriate flag and configure libraries based on selected thread model
message(STATUS "Checking thread model")
if(TCI_THREAD_MODEL STREQUAL "openmp")
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "openmp requested but not available")
    endif()
    set(USE_OPENMP_THREADS 1)
    target_link_libraries(tci PUBLIC OpenMP::OpenMP_C)
elseif(TCI_THREAD_MODEL STREQUAL "omptask")
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "omptask requested but not available")
    endif()
    set(USE_OMPTASK_THREADS 1)
    target_link_libraries(tci PUBLIC OpenMP::OpenMP_C)
elseif(TCI_THREAD_MODEL STREQUAL "pthread" OR TCI_THREAD_MODEL STREQUAL "pthreads")
    if(NOT Threads_FOUND)
        message(FATAL_ERROR "pthreads requested but not available")
    endif()
    set(USE_PTHREADS_THREADS 1)
    target_link_libraries(tci PUBLIC Threads::Threads)
elseif(TCI_THREAD_MODEL STREQUAL "tbb")
    if(NOT TBB_FOUND)
        message(FATAL_ERROR "tbb requested but not available")
    endif()
    set(USE_TBB_THREADS 1)
    target_link_libraries(tci PUBLIC TBB::tbb)
elseif(TCI_THREAD_MODEL STREQUAL "dispatch" OR TCI_THREAD_MODEL STREQUAL "libdispatch" OR TCI_THREAD_MODEL STREQUAL "gcd")
    if(NOT dispatch_FOUND)
        message(FATAL_ERROR "dispatch requested but not available")
    endif()
    set(USE_DISPATCH_THREADS 1)
    target_link_libraries(tci PUBLIC dispatch::dispatch)
elseif(TCI_THREAD_MODEL STREQUAL "windows" OR TCI_THREAD_MODEL STREQUAL "winthreads")
    if(NOT WIN32)
        message(FATAL_ERROR "winthreads requested but not available")
    endif()
    set(USE_WINDOWS_THREADS 1)
elseif(TCI_THREAD_MODEL STREQUAL "ppl")
    if(NOT WIN32 OR NOT HAVE_PPL)
        message(FATAL_ERROR "ppl requested but not available")
    endif()
    set(USE_PPL_THREADS 1)
elseif(TCI_THREAD_MODEL STREQUAL "none")
    # No threading - all flags remain 0
else()
    message(FATAL_ERROR "Invalid threading model specified: ${TCI_THREAD_MODEL}")
endif()
message(STATUS "Checking thread model - ${TCI_THREAD_MODEL}")

# Set C++ standard to 17 if using TBB or PPL (which require C++)
if(USE_TBB_THREADS OR USE_PPL_THREADS)
    set_property(TARGET tci PROPERTY CXX_STANDARD 17)
    set_property(TARGET tci PROPERTY CXX_STANDARD_REQUIRED ON)
endif()



configure_file(tci/tci_config.h.in tci/tci_config.h @ONLY)

# Have to compile communicator and task_set with C++ if using TBB or PPL
if(USE_PPL_THREADS OR USE_TBB_THREADS)
    set(TCI_CXX ON)
endif()

if(TCI_CXX)
    set_source_files_properties(tci/communicator.c PROPERTIES LANGUAGE CXX)
    set_source_files_properties(tci/task_set.c PROPERTIES LANGUAGE CXX)
endif()

target_include_directories(tci PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tci>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/tci>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(tci
    INTERFACE FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    FILES
    tci/barrier.h
    tci/communicator.h
    tci/context.h
    tci/mutex.h
    tci/parallel.h
    tci/pipeline.h
    tci/slot.h
    tci/task_set.h
    tci/tci_config.h
    tci/tci_global.h
    tci/work_item.h
    tci/yield.h
    tci.h
)
    

install(TARGETS tci EXPORT tci-targets
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)