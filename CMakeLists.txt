# ########################################################################
# Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/aoclsparse" CACHE PATH "Install path prefix, prepended onto install directories")
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
OPTION(BUILD_ILP64 "ILP64 Support" OFF)

# Dependencies
include(cmake/Dependencies.cmake)

# Define a version for the aocl-sparse code.
if( NOT DEFINED AOCLSPARSE_VERSION_MAJOR )
	set( AOCLSPARSE_VERSION_MAJOR 3 )
endif( )

if( NOT DEFINED AOCLSPARSE_VERSION_MINOR )
	set( AOCLSPARSE_VERSION_MINOR 1 )
endif( )

if( NOT DEFINED AOCLSPARSE_VERSION_PATCH )
	set( AOCLSPARSE_VERSION_PATCH 0 )
endif( )

if( NOT DEFINED AOCLSPARSE_VERSION_TWEAK )
	set( AOCLSPARSE_VERSION_TWEAK 0 )
endif( )

set( AOCLSPARSE_VERSION "${AOCLSPARSE_VERSION_MAJOR}.${AOCLSPARSE_VERSION_MINOR}.${AOCLSPARSE_VERSION_PATCH}.${AOCLSPARSE_VERSION_TWEAK}")

message( STATUS "AOCLSPARSE_VERSION= ${AOCLSPARSE_VERSION}" )
message( STATUS "CMAKE_BUILD_TYPE= ${CMAKE_BUILD_TYPE}" )
set(aoclsparse_SOVERSION ${AOCLSPARSE_VERSION})

# AMD targets
set(AMDCPU_TARGETS  CACHE STRING "List of specific machine types for library to target")

IF(BUILD_ILP64)
    message( STATUS "BUILD_ILP64 = ON" )
    ADD_DEFINITIONS(-Daoclsparse_ILP64)
ENDIF(BUILD_ILP64)

SET(AMD_ARCH "native" CACHE STRING "select AMD zen version for Clang toolchain")

if ("${AMD_ARCH}" STREQUAL "")
  message(FATAL_ERROR "Machine arch missing! Select one of znver1, znver2 or znver3")
elseif (${AMD_ARCH} STREQUAL "znver1")
  add_compile_options("-march=znver1")
elseif (${AMD_ARCH} STREQUAL "znver2")
  add_compile_options("-march=znver2")
elseif (${AMD_ARCH} STREQUAL "znver3")
  add_compile_options("-march=znver3")
else ()
  add_compile_options("-march=native")
  message(STATUS "Using -march=native as default")
endif ()

add_compile_options ( -shared -fpermissive )
# aoclsparse library
add_subdirectory(library)

if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
add_subdirectory(tests)
endif()
