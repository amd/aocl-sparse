# ########################################################################
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  # Avoid installing gtest libs/headers by default
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(utils_source common_data_utils.cpp ../common/aoclsparse_template_specialization.cpp)
include_directories(${CMAKE_SOURCE_DIR}/library/src/include
                    ${CMAKE_SOURCE_DIR}/library/src/solvers
                    ${CMAKE_SOURCE_DIR}/library/src/analysis
                    ${CMAKE_SOURCE_DIR}/library/src/level2
                    ${CMAKE_SOURCE_DIR}/library/src/level1
                    ${CMAKE_SOURCE_DIR}/library/src/extra
                    ${CMAKE_SOURCE_DIR}/tests/include)

set(MT_LIBRARIES "OpenMP::OpenMP_CXX")
set(PTHREAD_LIBRARY "pthread")

if(WIN32)
ELSE()
    list(APPEND MT_LIBRARIES ${PTHREAD_LIBRARY})
ENDIF()

set(GTEST_EXECUTABLES
    gmres_tests
    cg_tests
    csrmv_tests
    ellmv_tests
    blkcsrmv_tests
    csrmm_tests
    csr2m_tests
    hint_tests
    ilu_tests
    trsv_tests
    kt_tests
    ktlvl2_tests
    mv_tests
    diamv_tests
    bsrmv_tests
    createcsc_tests
    optimize_tests
    coo_tests
    gthr_tests
    dotp_tests
    sctr_tests
    roti_tests
    createcsr_tests
    utils_tests
    axpyi_tests
    conversion_tests
    order_tests
    copy_tests
)
foreach(TNAME ${GTEST_EXECUTABLES})
    add_executable(${TNAME} ${TNAME}.cpp ${utils_source})
    target_link_libraries(${TNAME} aoclsparse ${MT_LIBRARIES} ${LAPACK_LIBRARIES} GTest::gtest_main)
    target_include_directories(${TNAME} PUBLIC ${LAPACK_INCLUDE_DIRS} ${BLIS_INCLUDE_DIRS} ${UTILS_INCLUDE_DIRS})
endforeach()

include(GoogleTest)
foreach(TNAME ${GTEST_EXECUTABLES})
  gtest_discover_tests(${TNAME})
endforeach()
