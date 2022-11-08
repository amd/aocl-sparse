# ########################################################################
#Copyright(c) 2020-2023 Advanced Micro Devices, Inc.
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files(the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following                       conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the                               Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.
#
# ########################################################################

#Only define out - of - src - tree build if not defined previously
if(NOT DEFINED ENABLE_OUTOFSRCTREE)
    SET(ENABLE_OUTOFSRCTREE ON)
endif()
## Common config
option (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE "Allow out-of-source tree build for Sparse examples" ${ENABLE_OUTOFSRCTREE})
set(MT_LIBRARIES "OpenMP::OpenMP_CXX")
set(PTHREAD_LIBRARY "pthread")
set(COMPILER_OPTIONS)

## Setup the pre-requisites for out-of-source-tree build 
if (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)
    cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
#sparse examples project
    project(aoclsparse_examples LANGUAGES CXX)
    
    OPTION(BUILD_SHARED_LIBS "Build aocl-sparse as a shared library" ON)
    OPTION(BUILD_ILP64 "ILP64 Support" OFF)
    OPTION(SUPPORT_OMP "Compile with OpenMP support." ON)
    
    if(DEFINED ENV{AOCL_ROOT})        
        set(SPARSE_ARTIFACTS_DIR $ENV{AOCL_ROOT})
    ELSE()
        set(SPARSE_ARTIFACTS_DIR ${CMAKE_INSTALL_PREFIX})
    ENDIF()

    if(SUPPORT_OMP)
        find_package(OpenMP)
        if (NOT OPENMP_FOUND)
            message("-- OpenMP not found.")
        endif()    
    endif()
    set(COMPILER_OPTIONS ${OpenMP_CXX_FLAGS})

    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    set(CMAKE_VERBOSE_MAKEFILE "True" CACHE STRING "" FORCE) 

    ## Change default installation path
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
        CACHE PATH "New default Intallation Path" FORCE)
    endif ()
endif (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)

IF(WIN32)
    SET(CMAKE_FIND_LIBRARY_PREFIXES "")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
ELSE(WIN32)
    SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
    list(APPEND MT_LIBRARIES ${PTHREAD_LIBRARY})
ENDIF(WIN32)

if (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)
    IF(BUILD_ILP64) 
        list(APPEND COMPILER_OPTIONS "-Daoclsparse_ILP64")            
        find_library(SPARSE_LIBRARY
                    NAMES aoclsparse
                    HINTS ${SPARSE_ARTIFACTS_DIR}
                    PATH_SUFFIXES "lib" "lib/ILP64")       

        find_path(SPARSE_INCLUDE_DIR
                NAMES "aoclsparse.h"
                PATHS ${SPARSE_ARTIFACTS_DIR}
                PATH_SUFFIXES "include")                  

    ELSE(BUILD_ILP64)    
        find_library(SPARSE_LIBRARY
                    NAMES aoclsparse
                    HINTS ${SPARSE_ARTIFACTS_DIR}
                    PATH_SUFFIXES "lib" "lib/LP64")   

        find_path(SPARSE_INCLUDE_DIR
                NAMES "aoclsparse.h"
                PATHS ${SPARSE_ARTIFACTS_DIR}
                PATH_SUFFIXES "include")                  
    ENDIF(BUILD_ILP64)  
    
    if(NOT SPARSE_LIBRARY)        
        message (FATAL_ERROR "Error: could not find a suitable installation of AOCL-Sparse in \$AOCL_ROOT=${AOCL_ROOT}")
    endif()
    if(NOT SPARSE_INCLUDE_DIR)
        message (FATAL_ERROR "Error: could not find AOCL-Sparse header files in \$AOCL_ROOT=${AOCL_ROOT}/include")
    endif()
    message(STATUS "Headers and library linkages defined")
    message(STATUS "Headers = ${SPARSE_INCLUDE_DIR}")
    message(STATUS "Library = ${SPARSE_LIBRARY}")    
else (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)  
    set(SPARSE_LIBRARY "aoclsparse")
    set(SPARSE_INCLUDE_DIR "")  
endif (BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)

file(GLOB_RECURSE SPARSE_SAMPLES sample_*.cpp)
foreach( sample_example_source ${SPARSE_SAMPLES})
    string( REPLACE ".cpp" "" new_name ${sample_example_source})
    get_filename_component(sample_target ${new_name} NAME)
    add_executable( ${sample_target} ${sample_example_source})
    target_include_directories(${sample_target} PUBLIC ${SPARSE_INCLUDE_DIR})
    target_link_libraries(${sample_target} ${SPARSE_LIBRARY} ${MT_LIBRARIES})
    target_compile_options(${sample_target} PUBLIC ${COMPILER_OPTIONS})
    if(NOT BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)    
        install(FILES ${sample_example_source} DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)
    endif()   
endforeach( sample_example_source ${SPARSE_SAMPLES} )

if(NOT BUILD_SPARSE_EXAMPLE_OUTOFSRCTREE)
    set(EXAMPLES_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/examples")
    install(FILES README.md DESTINATION ${EXAMPLES_INSTALL_DIR})
    install(FILES CMakeLists.txt DESTINATION ${EXAMPLES_INSTALL_DIR})    
endif()
set(EXAMPLE_EXECUTABLES
    sample_spmv
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
)
# Take all sample executables also as test targets
foreach(SNAME ${EXAMPLE_EXECUTABLES})
    string(REPLACE "sample_" "" SNAMESHORT ${SNAME})
    add_test(Sample.${SNAMESHORT} ${SNAME})
endforeach()
