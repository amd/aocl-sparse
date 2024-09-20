# ########################################################################
# Copyright (c) 2020-2024 Advanced Micro Devices, Inc.
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

set(Boost_USE_MULTITHREADED ON)
set(Boost_DETAILED_FAILURE_MSG ON)
set(Boost_ADDITIONAL_VERSIONS 1.65.1 1.65)
set(Boost_USE_STATIC_LIBS OFF)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Boost)

if(NOT Boost_FOUND)
    message(STATUS "Dynamic boost libraries not found. Attempting to find static libraries...")
    set(Boost_USE_STATIC_LIBS ON)
find_package(Boost)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "boost is a required dependency and is not found; try adding boost path to CMAKE_PREFIX_PATH")
    endif()
endif()

# fetch external benchmarking libraries, etc. if enabled
if(EXTERNAL_BENCHMARKING)
  if (NOT DEFINED EXT_BENCH_INC)
    message(FATAL_ERROR "EXTERNAL_BENCHMARKING is turned ON but no EXT_BENCH_INC is provided. Set EXT_BENCH_INC to the include directory containing ext_benchmarking.hpp.")
  endif()
  if (NOT DEFINED EXT_BENCH_DEF OR NOT DEFINED EXT_BENCH_LIB)
    message(WARNING "EXTERNAL_BENCHMARKING is turned ON but not all EXT_BENCH_DEF, EXT_BENCH_INC or EXT_BENCH_LIB are set.")
  endif()

  message(STATUS "EXTERNAL_BENCHMARKING......${EXTERNAL_BENCHMARKING}")
  message(STATUS "EXT_BENCH_DEF..............${EXT_BENCH_DEF}")
  message(STATUS "EXT_BENCH_INC..............${EXT_BENCH_INC}")
  message(STATUS "EXT_BENCH_LIB..............${EXT_BENCH_LIB}")
endif()


set(AOCLSPARSE_BENCHMARK_SOURCES
  aoclsparse_test.cpp
)

set(AOCLSPARSE_CLIENTS_COMMON
  ../common/aoclsparse_utility.cpp
  ../common/aoclsparse_template_specialization.cpp
  ../common/aoclsparse_stats.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/library/include
                    ${CMAKE_SOURCE_DIR}/library/src/include
                    ${CMAKE_SOURCE_DIR}/library/src/solvers
                    ${CMAKE_SOURCE_DIR}/library/src/analysis
                    ${CMAKE_SOURCE_DIR}/library/src/conversion
                    ${CMAKE_SOURCE_DIR}/library/src/extra
                    ${CMAKE_SOURCE_DIR}/library/src/level1
                    ${CMAKE_SOURCE_DIR}/library/src/level2
                    ${CMAKE_SOURCE_DIR}/tests/include)

add_executable(aoclsparse-bench ${AOCLSPARSE_BENCHMARK_SOURCES} ${AOCLSPARSE_CLIENTS_COMMON})

# Internal header includes
target_include_directories(aoclsparse-bench
  PRIVATE
    ${LAPACK_INCLUDE_DIRS}
    ${BLIS_INCLUDE_DIRS}
    ${UTILS_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)
# Add extra info into definition
execute_process(COMMAND ${GIT_EXECUTABLE} log -n 1 --format=%H RESULT_VARIABLE status OUTPUT_VARIABLE HASH ERROR_QUIET WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_STRIP_TRAILING_WHITESPACE)
if(status EQUAL 0)
    string(REPLACE "\n" "" HASH ${HASH})
endif()
set(HID "\"${HASH}\"")
set(COMPILER_ID "\"${CMAKE_CXX_COMPILER_ID} ${COMPILER_BASENAME} (${CMAKE_CXX_COMPILER_VERSION}) ${CMAKE_CXX_PLATFORM_ID}\"")
target_compile_definitions(aoclsparse-bench PRIVATE CID=${COMPILER_ID} HASH=${HID})

# Add definitions that depend on the CMake configuration.
target_compile_definitions(aoclsparse-bench PRIVATE ${AOCLSPARSE_DEFS})
# Define -Daoclsparse_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
target_compile_definitions(aoclsparse-bench PRIVATE ${AOCLSPARSE_ILP64})

# Add aoclsparse target
if(NOT TARGET aoclsparse)
  target_link_libraries(aoclsparse-bench PRIVATE ${AOCLSPARSE_LIBRARIES})
else()
  target_link_libraries(aoclsparse-bench PRIVATE aocl::aoclsparse )
endif()

# Add OpenMP
target_link_libraries(aoclsparse-bench PRIVATE ${OpenMP_Library})
target_link_libraries(aoclsparse-bench PUBLIC ${LAPACK_LIBRARIES})

# fetch external benchmarking libraries, etc. if enabled
if(EXTERNAL_BENCHMARKING)
  # Define -DEXT_BENCHMARKING and if any external deinitions
  target_compile_definitions(aoclsparse-bench PRIVATE -DEXT_BENCHMARKING ${EXT_BENCH_DEF})
  target_link_libraries(aoclsparse-bench PRIVATE ${EXT_BENCH_LIB})
  target_include_directories(aoclsparse-bench PRIVATE ${EXT_BENCH_INC})
endif()

# Set benchmark output directory
set_target_properties(aoclsparse-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

install(TARGETS aoclsparse-bench DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Include bench tests
set(AOCLSPARSE_BENCH_PATH "../staging")
include(bench_tests.cmake)