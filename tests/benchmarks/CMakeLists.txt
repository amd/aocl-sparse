# ########################################################################
# Copyright (c) 2019 Advanced Micro Devices, Inc.
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

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

set(AOCLSPARSE_BENCHMARK_SOURCES
  aoclsparse_test.cpp
)

set(AOCLSPARSE_CLIENTS_COMMON
  ../common/aoclsparse_utility.cpp
  ../common/aoclsparse_template_specialization.cpp
)

add_executable(aoclsparse-bench ${AOCLSPARSE_BENCHMARK_SOURCES} ${AOCLSPARSE_CLIENTS_COMMON})

# Internal header includes
target_include_directories(aoclsparse-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# OpenMP cmake fix for cmake <= 3.9
if(OPENMP_FOUND)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
  endif()
endif()

# Add aoclsparse target
if(NOT TARGET aoclsparse)
  target_link_libraries(aoclsparse-bench PRIVATE ${AOCLSPARSE_LIBRARIES})
else()
  target_link_libraries(aoclsparse-bench PRIVATE aocl::aoclsparse)
endif()


# Add OpenMP
if(OPENMP_FOUND)
  target_link_libraries(aoclsparse-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set benchmark output directory
set_target_properties(aoclsparse-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
