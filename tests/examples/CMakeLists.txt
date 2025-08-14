# ########################################################################
# Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Only define out-of-src-tree build if not defined previously
if(NOT DEFINED ENABLE_OUTOFSRCTREE)
    SET(ENABLE_OUTOFSRCTREE ON)
endif()

## Common config
option (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE "Allow out-of-source tree build for Sparse examples" ${ENABLE_OUTOFSRCTREE})
set(COMPILER_OPTIONS "")

## Setup the pre-requisites for out-of-source-tree build
if (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)
    cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
    #sparse examples project
    project(aoclsparse_examples LANGUAGES CXX C)
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
    set(USE_EXTERNAL_OMP_LIB OFF CACHE BOOL "Use OpenMP target generated from find_package(OpenMP) by default")
    set(EXTERNAL_OMP_LIBRARY "" CACHE STRING "Provide external OMP library")

    # Set paths to AOCLUTILS, BLAS and LAPACK installations.
    set(CMAKE_AOCL_ROOT $ENV{AOCL_ROOT} CACHE STRING "AOCL_ROOT directory to be used to find BLAS/LAPACK/AOCLUTILS libraries")

    set(CMAKE_SPARSE_ROOT $ENV{SPARSE_ROOT} CACHE STRING "SPARSE_ROOT directory to be used to find Sparse artifacts")
    if(CMAKE_SPARSE_ROOT STREQUAL "")
        message(WARNING "SPARSE_ROOT was not set. Will search for it in main CMAKE_AOCL_ROOT directory.")
        #Inherit AOCL_ROOT since we assume Sparse artifacts would be present as part of AOCL installation
        set(CMAKE_SPARSE_ROOT ${CMAKE_AOCL_ROOT})
    endif()

    ## Change default installation path
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
        CACHE PATH "New default Intallation Path" FORCE)
    endif ()

    IF(BUILD_ILP64)
        SET(ILP_DIR "ILP64")
        SET(AOCLSPARSE_ILP64 -Daoclsparse_ILP64)
    ELSE(BUILD_ILP64)
        SET(ILP_DIR "LP64")
    ENDIF(BUILD_ILP64)

    ## During out-of-source-tree build of examples, Blis, Libflame, openmp and utils dependecies need to be found explicitly.
    ## So we call Dependencies.cmake
    include(cmake/Dependencies.cmake)

    set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};${COMPILER_OPTIONS}")

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
    message(STATUS "AOCL-Sparse header and library locations")
    message(STATUS "Headers: ${SPARSE_INCLUDE_DIR}")
    message(STATUS "Library: ${SPARSE_LIBRARY}")
    message(STATUS "Compiler flags ${COMPILER_FLAGS_COMMON}")
else (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)
    set(SPARSE_LIBRARY "aoclsparse")
    set(SPARSE_INCLUDE_DIR "")
    message(STATUS "Compiler flags ${COMPILER_FLAGS_COMMON}")
endif (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)

file(GLOB_RECURSE SPARSE_SAMPLES sample_*.cpp)
foreach( sample_example_source ${SPARSE_SAMPLES})
    string( REPLACE ".cpp" "" new_name ${sample_example_source})
    get_filename_component(sample_target ${new_name} NAME)
    add_executable( ${sample_target} ${sample_example_source})
    target_include_directories(${sample_target} PUBLIC ${SPARSE_INCLUDE_DIR} ${LAPACK_INCLUDE_DIRS} ${BLIS_INCLUDE_DIRS} ${UTILS_INCLUDE_DIRS})
    # Define -Daoclsparse_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
    target_compile_definitions(${sample_target} PUBLIC ${AOCLSPARSE_ILP64})

    target_link_libraries(${sample_target} ${SPARSE_LIBRARY} ${LAPACK_LIBRARIES} ${OpenMP_Library})
    #define compile time options explicitly to handle out-of-source tree scenario where options would not be defined by a top-level CMakeLists file
    target_compile_options(${sample_target} PUBLIC "$<$<CONFIG:Release>:${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_RELEASE}>;-mno-avx512f;")
    target_compile_options(${sample_target} PUBLIC "$<$<CONFIG:Debug>:${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_DEBUG}>;-mno-avx512f;")

    if(NOT BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)
        install(FILES ${sample_example_source} DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)
    endif()
endforeach( sample_example_source ${SPARSE_SAMPLES} )

# C compatibility test - compile with C compiler and link with C++ compiler
set_source_files_properties(sample_spmv_c.c sample_spmv_multi_instance.c PROPERTIES LANGUAGE C)

add_executable(sample_spmv_c sample_spmv_c.c)
add_executable(sample_spmv_multi_instance sample_spmv_multi_instance.c)

set_target_properties(sample_spmv_multi_instance sample_spmv_c PROPERTIES LINKER_LANGUAGE CXX)
# Define -Daoclsparse_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
target_compile_definitions(sample_spmv_multi_instance PUBLIC ${AOCLSPARSE_ILP64})
target_include_directories(sample_spmv_multi_instance PUBLIC ${SPARSE_INCLUDE_DIR} ${LAPACK_INCLUDE_DIRS} ${BLIS_INCLUDE_DIRS} ${UTILS_INCLUDE_DIRS})
target_link_libraries(sample_spmv_multi_instance ${SPARSE_LIBRARY} ${LAPACK_LIBRARIES} ${OpenMP_Library})

set_target_properties(sample_spmv_c PROPERTIES LINKER_LANGUAGE CXX)
# Define -Daoclsparse_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
target_compile_definitions(sample_spmv_c PUBLIC ${AOCLSPARSE_ILP64})
target_include_directories(sample_spmv_c PUBLIC ${SPARSE_INCLUDE_DIR} ${LAPACK_INCLUDE_DIRS} ${BLIS_INCLUDE_DIRS} ${UTILS_INCLUDE_DIRS})
target_link_libraries(sample_spmv_c ${SPARSE_LIBRARY} ${LAPACK_LIBRARIES} ${OpenMP_Library})

#define compile time options explicitly to handle out-of-source tree scenario where options would not be defined by a top-level CMakeLists file
target_compile_options(sample_spmv_c PUBLIC "${COMPILER_FLAGS_COMMON};-mno-avx512f;")
target_compile_options(sample_spmv_multi_instance PUBLIC "${COMPILER_FLAGS_COMMON};-mno-avx512f;")

#target_compile_options(sample_spmv_c PUBLIC "$<$<CONFIG:Release>:${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_RELEASE}>")
#target_compile_options(sample_spmv_c PUBLIC "$<$<CONFIG:Debug>:${COMPILER_FLAGS_COMMON};${COMPILER_FLAGS_DEBUG}>")
if(NOT BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)
    install(FILES sample_spmv_c.c DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)
    install(FILES sample_spmv_multi_instance.c DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)
endif()

if(NOT BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)
    set(EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/examples")
    install(FILES README.md DESTINATION ${EXAMPLES_INSTALL_DIR})
    install(FILES CMakeLists.txt DESTINATION ${EXAMPLES_INSTALL_DIR})
    install(FILES ${CMAKE_SOURCE_DIR}/CMakePresets.json DESTINATION ${EXAMPLES_INSTALL_DIR})
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake DESTINATION ${EXAMPLES_INSTALL_DIR}/cmake)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/presets/base.json DESTINATION ${EXAMPLES_INSTALL_DIR}/cmake/presets)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/presets/linux-make.json DESTINATION ${EXAMPLES_INSTALL_DIR}/cmake/presets)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/presets/linux-ninja.json DESTINATION ${EXAMPLES_INSTALL_DIR}/cmake/presets)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/presets/win-msvc.json DESTINATION ${EXAMPLES_INSTALL_DIR}/cmake/presets)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/presets/win-ninja.json DESTINATION ${EXAMPLES_INSTALL_DIR}/cmake/presets)
    install(FILES ${CMAKE_SOURCE_DIR}/cmake/presets/README.MD DESTINATION ${EXAMPLES_INSTALL_DIR}/cmake/presets)
endif()
set(EXAMPLE_EXECUTABLES
    sample_spmv_multi_instance
    sample_zgthr
    sample_zsp2m
    sample_csr2m
    sample_dtrsv
    sample_itsol_d_cg_rci
    sample_itsol_d_cg
    sample_itsol_s_cg_rci
    sample_itsol_s_cg
    sample_itsol_d_gmres_rci
    sample_itsol_d_gmres
    sample_itsol_s_gmres_rci
    sample_itsol_s_gmres
    sample_dsorv
    sample_dotp
    sample_sctr
    sample_roti
    sample_dotmv
    sample_axpyi
    sample_csrmm
    sample_dtrsm
    sample_ztrsm
    sample_spmv_c
    sample_spmmd
    sample_dsyrk
    sample_sp2md
    sample_zsypr
    sample_dsyrkd
    sample_syprd
    sample_dsymgs
    sample_dsymgs_mv
    sample_zsymgs
    sample_zsymgs_mv
    sample_tcsr_dspmv
    sample_tcsr_dtrsv
    sample_tcsr_ztrsv
    sample_itsol_z_gmres_rci
    sample_itsol_z_gmres
    # CPP examples
    sample_mv_cpp
    sample_trsv_cpp
    sample_csr2m_cpp
)
# Take all sample executables also as test targets
foreach(SNAME ${EXAMPLE_EXECUTABLES})
    string(REPLACE "sample_" "" SNAMESHORT ${SNAME})
    add_test(Sample.${SNAMESHORT} ${SNAME})
    # set_property(TEST Sample.${SNAMESHORT} PROPERTY ENVIRONMENT_MODIFICATION ASAN_OPTIONS="log_path=ASANlogger.${SNAME}")
endforeach()
