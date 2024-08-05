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
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  # Avoid installing gtest libs/headers by default
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(utils_source common_data_utils.cpp ../common/aoclsparse_template_specialization.cpp ../common/aoclsparse_utility.cpp)
include_directories(${CMAKE_SOURCE_DIR}/library/src/include
                    ${CMAKE_SOURCE_DIR}/library/src/solvers
                    ${CMAKE_SOURCE_DIR}/library/src/conversion
                    ${CMAKE_SOURCE_DIR}/library/src/analysis
                    ${CMAKE_SOURCE_DIR}/library/src/level2
                    ${CMAKE_SOURCE_DIR}/library/src/level1
                    ${CMAKE_SOURCE_DIR}/library/src/extra
                    ${CMAKE_SOURCE_DIR}/tests/include)

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
    kt_tests
    ktlvl2_tests
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
)

#copy the raw data files (mtx inputs) to build directory
add_subdirectory(data)

foreach(TNAME ${GTEST_EXECUTABLES})
    add_executable(${TNAME} ${TNAME}.cpp ${utils_source})
    target_link_libraries(${TNAME} aoclsparse ${OpenMP_Library} ${LAPACK_LIBRARIES} GTest::gtest_main)
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
    target_include_directories(${TNAME} PUBLIC ${LAPACK_INCLUDE_DIRS} ${BLIS_INCLUDE_DIRS} ${UTILS_INCLUDE_DIRS})
    # Add definitions that depend on the CMake configuration.
    target_compile_definitions(${TNAME} PUBLIC ${AOCLSPARSE_DEFS})
    # Define -Daoclsparse_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
    target_compile_definitions(${TNAME} PUBLIC ${AOCLSPARSE_ILP64})
endforeach()

include(GoogleTest)
foreach(TNAME ${GTEST_EXECUTABLES})
  gtest_discover_tests(${TNAME})
endforeach()
