# ########################################################################
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Only define out-of-src-tree build if not defined previously
if(NOT DEFINED ENABLE_OUTOFSRCTREE_TESTS)
    SET(ENABLE_OUTOFSRCTREE_TESTS ON)
endif()

## Common config
OPTION(BUILD_SPARSE_TESTS_OUTOFSRCTREE "Allow out-of-source tree build for Sparse tests" ${ENABLE_OUTOFSRCTREE_TESTS})

## Setup the pre-requisites for out-of-source-tree build
if (BUILD_SPARSE_TESTS_OUTOFSRCTREE)
    cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
    #sparse tests project
    project(aoclsparse_tests LANGUAGES CXX C)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Set a default build type if none was specified
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
      message(STATUS "Setting build type to 'Release' as none was specified")
      set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
    endif()
    OPTION(BUILD_SHARED_LIBS "Build aocl-sparse as a shared library" ON)
    OPTION(BUILD_ILP64 "ILP64 Support" OFF)
    OPTION(SUPPORT_OMP "Compile with OpenMP support." ON)
    set(SUPPORT_OMP ON)
    OPTION(USE_AVX512 "AVX512 Support" ON)
    OPTION(COVERAGE "Do source code coverage (only GCC/Debug build)" OFF)
    OPTION(ASAN "Enable Address SANitizer (only Debug/Linux build)" OFF)

    set(USE_EXTERNAL_OMP_LIB OFF CACHE BOOL "Use OpenMP target generated from find_package(OpenMP) by default")
    set(EXTERNAL_OMP_LIBRARY "" CACHE STRING "Provide external OMP library")

    # Set paths to AOCLUTILS, BLAS and LAPACK installations.
    set(CMAKE_AOCL_ROOT
        $ENV{AOCL_ROOT}
        CACHE
        STRING
        "AOCL_ROOT directory to be used to find BLAS/LAPACK/AOCLUTILS libraries")
    if(CMAKE_AOCL_ROOT STREQUAL "")
        message(FATAL_ERROR "CMAKE_AOCL_ROOT is empty. Either set environment variable AOCL_ROOT or set -DCMAKE_AOCL_ROOT=<path_to_AOCL_libs>.")
    endif()
    set(CMAKE_SPARSE_ROOT $ENV{SPARSE_ROOT} CACHE STRING "SPARSE_ROOT directory to be used to find Sparse artifacts")
    if(CMAKE_SPARSE_ROOT STREQUAL "")
        message(WARNING "SPARSE_ROOT was not set. Will search for it in main CMAKE_AOCL_ROOT directory.")
        #Inherit AOCL_ROOT since we assume Sparse artifacts would be present as part of AOCL installation
        set(CMAKE_SPARSE_ROOT ${CMAKE_AOCL_ROOT})
    endif()

    if (NOT (${ASAN} AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND CMAKE_BUILD_TYPE MATCHES "Debug"))
        # Switch off ASAN: target not supported
        message(STATUS "ASAN is supported only for Linux-Debug environemnt. Disabling ASAN to proceed further.")
        set(ASAN OFF)
    endif()
    if (NOT (${COVERAGE} AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND CMAKE_BUILD_TYPE MATCHES "Debug"))
        # Switch off GCOV: target not supported
        message(STATUS "COVERAGE is supported only for Linux-Debug environemnt. Disabling COVERAGE to proceed further.")
        set(COVERAGE OFF)
    endif()

    # Debug flags
    set(COMPILER_FLAGS_DEBUG "-O0;-gdwarf-5;-g3;-Wall;-Wextra;")
    if(ASAN)
        set(CMAKE_CXX_FLAGS_DEBUG "-fsanitize=address")
        set(COMPILER_FLAGS_DEBUG "${COMPILER_FLAGS_DEBUG};${CMAKE_CXX_FLAGS_DEBUG}")
    endif()

    ## Change default installation path
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
        CACHE PATH "New default Intallation Path" FORCE)
    endif ()
    enable_testing()

    IF(BUILD_ILP64)
        SET(ILP_DIR "ILP64")
        SET(AOCLSPARSE_ILP64 -Daoclsparse_ILP64)
    ELSE(BUILD_ILP64)
        SET(ILP_DIR "LP64")
    ENDIF(BUILD_ILP64)
    ## During out-of-source-tree build of examples, Blis, Libflame, openmp and utils dependecies need to be found explicitly.
    ## So we call Dependencies.cmake
    include(Dependencies.cmake)

    # Add empty "cleanall" target
    # for coverage target to add dependencies to it
    add_custom_target( cleanall
    COMMAND ${CMAKE_MAKE_PROGRAM} clean
    )
    if(COVERAGE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        include(Coverage.cmake)
        set(COMPILER_FLAGS_DEBUG "${COMPILER_FLAGS_DEBUG};--coverage")
        add_link_options("$<$<CONFIG:Debug>:--coverage>")
    else()
        set(COVERAGE OFF)
    endif()

    find_library(SPARSE_LIBRARY
    NAMES "aoclsparse" "aoclsparse-d"
    HINTS  ${CMAKE_SPARSE_ROOT} ${CMAKE_SPARSE_ROOT}/sparse ${CMAKE_SPARSE_ROOT}/amd-sparse
    PATH_SUFFIXES  "lib/${ILP_DIR}" "lib_${ILP_DIR}" "lib" "lib/${ILP_DIR}/shared" "lib/${ILP_DIR}/static"
    DOC "AOCL Sparse library")

    find_path(SPARSE_INCLUDE_DIR
    NAMES "aoclsparse.h"
    HINTS  ${CMAKE_SPARSE_ROOT} ${CMAKE_SPARSE_ROOT}/sparse ${CMAKE_SPARSE_ROOT}/amd-sparse
    PATH_SUFFIXES  "include/${ILP_DIR}" "include_${ILP_DIR}" "include"
    DOC "AOCL Sparse headers")

    if(NOT SPARSE_LIBRARY)
        message (FATAL_ERROR "Error: could not find a suitable installation of AOCL-Sparse in \$CMAKE_SPARSE_ROOT=${CMAKE_SPARSE_ROOT}")
    endif()
    if(NOT SPARSE_INCLUDE_DIR)
        message (FATAL_ERROR "Error: could not find AOCL-Sparse header files in \$CMAKE_SPARSE_ROOT=${CMAKE_SPARSE_ROOT}/include")
    endif()

    message(STATUS "  \$SPARSE_INCLUDE_DIR......${SPARSE_INCLUDE_DIR}")
else (BUILD_SPARSE_TESTS_OUTOFSRCTREE)
    set(SPARSE_LIBRARY "aoclsparse")
endif (BUILD_SPARSE_TESTS_OUTOFSRCTREE)
message(STATUS "  \$SPARSE_LIBRARY......${SPARSE_LIBRARY}")

include(FetchContent)
set(FETCHCONTENT_QUIET off)

# by default, the third party sources (google test) are located inside <build_directory>/external.
if(NOT CMAKE_EXTERNAL_BUILD_DIR)
    set(CMAKE_EXTERNAL_BUILD_DIR "./external")
endif()
get_filename_component(fc_base ${CMAKE_EXTERNAL_BUILD_DIR}
                       REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  # Avoid installing gtest libs/headers by default
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(GTEST_EXECUTABLES
    dispatcher_tests
    gmres_tests
    cg_tests
    csrmv_tests
    ellmv_tests
    blkcsrmv_tests
    csrmm_tests
    csr2m_tests
    sp2m_tests
    spmm_tests
    hint_tests
    ilu_tests
    trsv_tests
    mv_tests
    set_value_test
    csr_convert_tests
    diamv_tests
    bsrmv_tests
    csr_add_test
    createcsc_tests
    optimize_tests
    coo_tests
    update_values_test
    gthr_tests
    dotp_tests
    sctr_tests
    roti_tests
    createcsr_tests
    axpyi_tests
    dotmv_tests
    sorv_tests
    conversion_tests
    order_tests
    copy_tests
    export_mat_tests
    trsm_tests
    spmmd_tests
    syrk_tests
    syrkd_tests
    sp2md_tests
    sypr_tests
    syprd_tests
    symgs_tests
    rng_tests
    context_tests
    extreme_value_tests
    createtcsr_tests
    tcsrmv_tests
)

set(AOCLSPARSE_TEST_DEPENDENCIES
${SPARSE_LIBRARY}
${OpenMP_Library}
${LAPACK_LIBRARIES}
GTest::gtest_main
)

set(AOCLSPARSE_TEST_HEADER_DEPENDENCIES
${SPARSE_INCLUDE_DIR}
${LAPACK_INCLUDE_DIRS}
${BLIS_INCLUDE_DIRS}
${UTILS_INCLUDE_DIRS}
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)

if(NOT BUILD_SPARSE_TESTS_OUTOFSRCTREE)
    set(GTEST_INTERNAL_EXECUTABLES
        kt_tests
        ktlvl2_tests
    )
    set(TESTS_COMMON_DIR "${CMAKE_SOURCE_DIR}/tests/common")
    # -----------------------------------------------------------------------
    # Build KT tests (Note: this section should not be build for out of source)
    # -----------------------------------------------------------------------
    set(KT_KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/kt_kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ktlvl2_kernels.cpp
    )
    set(AOCLSPARSE_INTERNAL_HEADER_DEPENDENCIES
    ${CMAKE_SOURCE_DIR}/library/src/include
    ${CMAKE_SOURCE_DIR}/library/src/extra
    ${CMAKE_SOURCE_DIR}/tests/include
    )
    list(APPEND AOCLSPARSE_TEST_HEADER_DEPENDENCIES ${AOCLSPARSE_INTERNAL_HEADER_DEPENDENCIES})
    # Double compilation of KT test kernels
    add_library(KT_KERNELS_AVX512 OBJECT ${KT_KERNELS})
    target_compile_options(KT_KERNELS_AVX512 PUBLIC ${AOCLSPARSE_AVX512_FLAGS})
    target_link_libraries(KT_KERNELS_AVX512 PUBLIC aoclsparse GTest::gtest_main)
    target_include_directories(KT_KERNELS_AVX512 PUBLIC ${AOCLSPARSE_TEST_HEADER_DEPENDENCIES})

    add_library(KT_KERNELS_AVX2 OBJECT ${KT_KERNELS})
    target_compile_options(KT_KERNELS_AVX2 PUBLIC "-mno-avx512f;" -DAOCLSPARSE_AVX2_INSTANT)
    target_link_libraries(KT_KERNELS_AVX2 PUBLIC aoclsparse GTest::gtest_main)
    target_include_directories(KT_KERNELS_AVX2 PUBLIC ${AOCLSPARSE_TEST_HEADER_DEPENDENCIES})
    # -----------------------------------------------------------------------

    list(APPEND AOCLSPARSE_TEST_DEPENDENCIES $<TARGET_OBJECTS:KT_KERNELS_AVX512>)
    list(APPEND AOCLSPARSE_TEST_DEPENDENCIES $<TARGET_OBJECTS:KT_KERNELS_AVX2>)
    list(APPEND GTEST_EXECUTABLES ${GTEST_INTERNAL_EXECUTABLES})
else()
    list(APPEND AOCLSPARSE_TEST_HEADER_DEPENDENCIES ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set(TESTS_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")
endif()

## Compile common sources only once and then link
add_library(TEST_COMMONS OBJECT
                common_data_utils.cpp
                ${TESTS_COMMON_DIR}/aoclsparse_template_specialization.cpp
                ${TESTS_COMMON_DIR}/aoclsparse_utility.cpp)
target_link_libraries(TEST_COMMONS PUBLIC aoclsparse GTest::gtest_main)
target_include_directories(TEST_COMMONS PUBLIC ${AOCLSPARSE_TEST_HEADER_DEPENDENCIES}
                                               ${TESTS_COMMON_DIR})
# Add definitions that depend on the CMake configuration.
target_compile_definitions(TEST_COMMONS PUBLIC ${AOCLSPARSE_DEFS})
# Define -Daoclsparse_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
target_compile_definitions(TEST_COMMONS PUBLIC ${AOCLSPARSE_ILP64})
target_compile_options(TEST_COMMONS PUBLIC $<$<CONFIG:Debug>:${COMPILER_FLAGS_DEBUG}>)

list(APPEND AOCLSPARSE_TEST_DEPENDENCIES $<TARGET_OBJECTS:TEST_COMMONS>)

#copy the raw data files (mtx inputs) to build directory
add_subdirectory(data)

set(TESTS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/unit_tests")
foreach(TNAME ${GTEST_EXECUTABLES})
    add_executable(${TNAME} ${TNAME}.cpp)
    target_link_libraries(${TNAME} PUBLIC ${AOCLSPARSE_TEST_DEPENDENCIES})
    # For Windows shared library builds copying some relevant DLLs into the
    # unit_tests bin folder enables ctest to run without needing to update PATH
    # gtest_discover_tests does not work on Windows shared library builds - this
    # is the best we can do for now
    if(WIN32 AND BUILD_SHARED_LIBS)
      add_custom_command(
        TARGET ${TNAME}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_RUNTIME_DLLS:${TNAME}> $<TARGET_FILE_DIR:${TNAME}>
        COMMAND_EXPAND_LISTS)
    endif()
    target_include_directories(${TNAME} PUBLIC ${AOCLSPARSE_TEST_HEADER_DEPENDENCIES})

    # Add definitions that depend on the CMake configuration.
    target_compile_definitions(${TNAME} PUBLIC ${AOCLSPARSE_DEFS})
    # Define -Daoclsparse_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
    target_compile_definitions(${TNAME} PUBLIC ${AOCLSPARSE_ILP64})

    target_compile_options(${TNAME} PUBLIC $<$<CONFIG:Debug>:${COMPILER_FLAGS_DEBUG}>)
    if(NOT BUILD_SPARSE_TESTS_OUTOFSRCTREE)
        install(FILES ${TNAME}.cpp DESTINATION ${TESTS_INSTALL_DIR})
    endif()
endforeach()

if(NOT BUILD_SPARSE_TESTS_OUTOFSRCTREE)
    install(FILES CMakeLists.txt DESTINATION ${TESTS_INSTALL_DIR})
    install(FILES common_data_utils.cpp DESTINATION ${TESTS_INSTALL_DIR})
    install(FILES common_data_utils.h DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES createcsc_ut_functions.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES cg_ut_functions.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES gmres_ut_functions.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES data/CMakeLists.txt DESTINATION ${TESTS_INSTALL_DIR}/data)
    install(FILES data/tinyc.mtx DESTINATION ${TESTS_INSTALL_DIR}/data)
    install(FILES data/tinyr.mtx DESTINATION ${TESTS_INSTALL_DIR}/data)
    install(FILES ../common/aoclsparse_template_specialization.cpp DESTINATION ${TESTS_INSTALL_DIR}/common)
    install(FILES ../common/aoclsparse_utility.cpp DESTINATION ${TESTS_INSTALL_DIR}/common)
    install(FILES ../include/aoclsparse.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../include/aoclsparse_reference.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../include/aoclsparse_random.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../include/aoclsparse_arguments.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../include/aoclsparse_datatype2string.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../include/aoclsparse_init.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../include/aoclsparse_test.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../include/aoclsparse_utility.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../include/aoclsparse_check.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../../library/src/include/aoclsparse_descr.h DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../../library/src/include/aoclsparse_mat_structures.h DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ../../library/src/extra/aoclsparse_utils.hpp DESTINATION ${TESTS_INSTALL_DIR}/include)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake DESTINATION ${TESTS_INSTALL_DIR})
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/Coverage.cmake DESTINATION ${TESTS_INSTALL_DIR})
    install(FILES ../benchmarks/bench_tests.cmake DESTINATION ${TESTS_INSTALL_DIR})
endif()

include(GoogleTest)
foreach(TNAME ${GTEST_EXECUTABLES})
  gtest_discover_tests(${TNAME})
endforeach()

if(BUILD_SPARSE_TESTS_OUTOFSRCTREE)
    # Include bench tests
    set(AOCLSPARSE_BENCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../bin")
    include(bench_tests.cmake)

    ## Configured Info
    message (STATUS "Filename: ${CMAKE_CURRENT_LIST_FILE}")
    message (STATUS "Build Configuration for AOCL-Sparse-Tests")
    message (STATUS "  Build Type <CMAKE_BUILD_TYPE>....................${CMAKE_BUILD_TYPE}")
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "  CXX Compiler flags (Debug).......................${COMPILER_FLAGS_DEBUG}")
    endif()
    message (STATUS "  Current Generator <CMAKE_GENERATOR>..............${CMAKE_GENERATOR}")
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message (STATUS "  Address SANitization (only Linux/Debug) <ASAN>...${ASAN}")
    message (STATUS "  Source code coverage (only Debug/GCC) <COVERAGE>.${COVERAGE}")
    endif()
    message (STATUS "  Verbose flag <CMAKE_VERBOSE_MAKEFILE>............${CMAKE_VERBOSE_MAKEFILE}")
    message (STATUS "  Static/Dynamic Build <BUILD_SHARED_LIBS>.........${BUILD_SHARED_LIBS}")
    message (STATUS "  ILP64 Build Type <BUILD_ILP64>...................${BUILD_ILP64}")
    message (STATUS "  Multi-Threading using OpenMP <SUPPORT_OMP>.......${SUPPORT_OMP}")
    message (STATUS "  Use AVX512 instructions <USE_AVX512>.............${USE_AVX512}")
    message (STATUS  "  \$AOCL_ROOT (environment Variable)................$ENV{AOCL_ROOT}")
    message (STATUS  "  \$CMAKE_AOCL_ROOT.................................${CMAKE_AOCL_ROOT}")

endif(BUILD_SPARSE_TESTS_OUTOFSRCTREE)

