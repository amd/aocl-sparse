# ########################################################################
# Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(NOT CMAKE_INSTALL_PREFIX)
  if(WIN32)
    set(CMAKE_INSTALL_PREFIX
        "${PROJECT_BINARY_DIR}/package"
        CACHE PATH "Install path prefix, prepended onto install directories")
  else()
    set(CMAKE_INSTALL_PREFIX
        "/opt/aoclsparse"
        CACHE PATH "Install path prefix, prepended onto install directories")
  endif()
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# aocl-sparse project
project(aoclsparse LANGUAGES CXX)

# Force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Build flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build aocl-sparse as a shared library" ON)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks (requires Boost) " OFF)
option(BUILD_ILP64 "ILP64 Support" OFF)
option(USE_AVX512 "AVX512 Support" OFF)
set(ENABLE_SET_BUILD_DATE OFF CACHE STRING "Set library build date")
option(BUILD_CLIENTS_SAMPLES "Build sparse examples" ON)
option(BUILD_CLIENTS_TESTS "Build sparse unit tests" OFF)
option(BUILD_DOCS "Build the documentation" OFF)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
set(CMAKE_VERBOSE_MAKEFILE "True" CACHE STRING "" FORCE)

# Dependencies
include(cmake/Dependencies.cmake)

# Define a version for the aocl-sparse code.
if(NOT DEFINED AOCLSPARSE_VERSION_MAJOR)
  set(AOCLSPARSE_VERSION_MAJOR 4)
endif()

if(NOT DEFINED AOCLSPARSE_VERSION_MINOR)
  set(AOCLSPARSE_VERSION_MINOR 0)
endif()

if(NOT DEFINED AOCLSPARSE_VERSION_PATCH)
  set(AOCLSPARSE_VERSION_PATCH 1)
endif()

if(NOT DEFINED AOCLSPARSE_VERSION_TWEAK)
  set(AOCLSPARSE_VERSION_TWEAK 0)
endif()

if(NOT DEFINED AOCL_SPARSE_BUILD_DATE)
  set(AOCL_SPARSE_BUILD_DATE 0)
endif()

set(AOCLSPARSE_VERSION "${AOCLSPARSE_VERSION_MAJOR}.${AOCLSPARSE_VERSION_MINOR}.${AOCLSPARSE_VERSION_PATCH}.${AOCLSPARSE_VERSION_TWEAK}")
string(TIMESTAMP BUILD_DATE "%Y%m%d")

set(aoclsparse_SOVERSION ${AOCLSPARSE_VERSION})

if(ENABLE_SET_BUILD_DATE)
  add_definitions(-DAOCL_SPARSE_BUILD_DATE=${ENABLE_SET_BUILD_DATE})
else()
  add_definitions(-DAOCL_SPARSE_BUILD_DATE=${BUILD_DATE})
endif()

# AMD targets
set(AMDCPU_TARGETS CACHE STRING "List of specific machine types for library to target")

if(BUILD_ILP64)
  message(STATUS "BUILD_ILP64 = ON")
  add_definitions(-Daoclsparse_ILP64)
endif(BUILD_ILP64)

# Set explicit compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_DEBUG "")
# Check for CL compatibility layer of clang
get_filename_component(COMPILER_BASENAME "${CMAKE_CXX_COMPILER}" NAME)
if(COMPILER_BASENAME MATCHES "^(.+-)?(clang-cl)(-[0-9]+(\\.[0-9]+)*)?(-[^.]+)?(\\.exe)?$")
  set(CLANG_OPTIONS_STYLE "MSVC")
else()
  set(CLANG_OPTIONS_STYLE "GNU")
endif()
message(STATUS "CXX Compiler ID= ${CMAKE_CXX_COMPILER_ID} (${COMPILER_BASENAME}) platform id: ${CMAKE_CXX_PLATFORM_ID}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE STREQUAL "GNU"))
  # Common flags
  set(COMPILER_FLAGS_COMMON "-mtune=native;-mavx2;-mfma")
  # Release flags
  set(COMPILER_FLAGS_RELEASE "-DNDEBUG;-O3;-ftree-vectorize;-ffp-contract=fast")
  # Debug flags
  set(COMPILER_FLAGS_DEBUG "-O0;-gdwarf-5;-g3;-Wall;-Wextra")
  if(COVERAGE)
    # Add flags for gcov
    # TODO add dependecies and targets
    set(COMPILER_FLAGS_DEBUG "${COMPILER_FLAGS_DEBUG};-fprofile-arcs;-ftest-coverage")
  endif()
  if(USE_AVX512)
    set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};-mavx512f;-mavx512dq;-mavx512vl")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CLANG_OPTIONS_STYLE STREQUAL "MSVC")
  # Clang with MSVC style compiler options
  # Common flags
  if(USE_AVX512)
    set(COMPILER_FLAGS_COMMON "/arch:AVX512")
  else()
    set(COMPILER_FLAGS_COMMON "/arch:AVX2")
  endif()
  # Release flags
  set(COMPILER_FLAGS_RELEASE "/Ox;/Qpar;/fp:fast")
  # Debug flags
  set(COMPILER_FLAGS_DEBUG "/Od;/Zi")
else()
  message(WARNING "Explicit flags for ${CMAKE_CXX_COMPILER_ID} have NOT been set. Update the top-level CMakeList.txt!")
  message(WARNING "Continuing compilation using tentative CXX flags...")
  set(COMPILER_FLAGS_DEBUG "-g")
  set(COMPILER_FLAGS_RELEASE "-O3;-NDEBUG")
  set(COMPILER_FLAGS_COMMON "-mtune=native;-mavx2;-mfma")
  if(USE_AVX512)
    set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};-mavx512f;-mavx512dq;-mavx512vl")
  endif()
endif()
if(SUPPORT_OMP)
  set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};${OpenMP_CXX_FLAGS}")
endif()
add_compile_options("$<$<CONFIG:Release>:${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_RELEASE}>")
add_compile_options("$<$<CONFIG:Debug>:${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_DEBUG}>")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  message(STATUS "CXX Compiler flags (Debug): ${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
  message(STATUS "CXX Compiler flags (Release): ${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_RELEASE}")
endif()

if(USE_AVX512)
  message(STATUS "USE_AVX512 = ON")
  add_definitions(-DUSE_AVX512)
endif(USE_AVX512)

# aoclsparse library
add_subdirectory(library)

if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  # Internal state variable indicating that tests are to be build inside the
  # source tree and not as an independent project
  set(ENABLE_OUTOFSRCTREE OFF)
  add_subdirectory(tests)
endif()

if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

## Configured Info
message (STATUS "Filename: ${CMAKE_CURRENT_LIST_FILE}\n")

message (STATUS "Build Configuration for AOCL-Sparse")
message (STATUS "  Current Generator <CMAKE_GENERATOR>..............${CMAKE_GENERATOR}")
message (STATUS "  Install Prefix <CMAKE_INSTALL_PREFIX>............${CMAKE_INSTALL_PREFIX}")
message (STATUS "  Build Type <CMAKE_BUILD_TYPE>....................${CMAKE_BUILD_TYPE}")
message (STATUS "  Verbose flag <CMAKE_VERBOSE_MAKEFILE>....................${CMAKE_VERBOSE_MAKEFILE}")
message (STATUS "  AOCL-Sparse Version ...............${AOCLSPARSE_VERSION_MAJOR}.${AOCLSPARSE_VERSION_MINOR}.${AOCLSPARSE_VERSION_PATCH}.${AOCLSPARSE_VERSION_TWEAK}")
message (STATUS "  Static/Dynamic Build <BUILD_SHARED_LIBS>...................${BUILD_SHARED_LIBS}")
message (STATUS "  ILP64 Build Type <BUILD_ILP64>...................${BUILD_ILP64}")
message (STATUS "  Sparse Performance Benchmark tests <BUILD_CLIENTS_BENCHMARKS>.............${BUILD_CLIENTS_BENCHMARKS}")
message (STATUS "  Sparse examples <BUILD_CLIENTS_SAMPLES>.............${BUILD_CLIENTS_SAMPLES}")
message (STATUS "  Sparse Functionality tests <BUILD_CLIENTS_TESTS>.............${BUILD_CLIENTS_TESTS}")
message (STATUS "  Sparse Doxygen configuration and API guide build target <BUILD_DOCS>.............${BUILD_DOCS}")
