# ##############################################################################
# Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################

# ============= aocl function ================
function(aocl_libs)

  if(BUILD_ILP64)
    set(ILP_DIR "ILP64")
  else(BUILD_ILP64)
    set(ILP_DIR "LP64")
  endif(BUILD_ILP64)

  if(WIN32)
    set(CMAKE_FIND_LIBRARY_PREFIXES "")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
    set(_blas_library "AOCL-LibBlis-Win")
    set(_flame_library "AOCL-LibFlame-Win")
    set(_utils_library "libaoclutils")
    set(_static_tag "static")
    set(_mt_tag "MT")

    if(SUPPORT_OMP)
      set(_blas_static_library "${_blas_library}-${_mt_tag}")
      set(_blas_dyn_library "${_blas_static_library}-dll")
      set(_flame_static_library "${_flame_library}-${_mt_tag}")
      set(_flame_dyn_library "${_flame_static_library}-dll")
    else(SUPPORT_OMP)
      set(_blas_static_library "${_blas_library}")
      set(_blas_dyn_library "${_blas_static_library}-dll")
      set(_flame_static_library "${_flame_library}")
      set(_flame_dyn_library "${_flame_static_library}-dll")
    endif(SUPPORT_OMP)

    set(_utils_static_library "${_utils_library}_${_static_tag}")
    set(_utils_dyn_library "${_utils_library}")

  else(WIN32)
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    if(BUILD_SHARED_LIBS)
      set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
    else(BUILD_SHARED_LIBS)
      set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    endif(BUILD_SHARED_LIBS)
    set(_blas_library "blis")
    set(_flame_library "flame")
    set(_mt_tag "mt")
    set(_utils_static_library "aoclutils")
    set(_utils_dyn_library "aoclutils")

    if(SUPPORT_OMP)
      set(_blas_static_library "${_blas_library}-${_mt_tag}")
      set(_blas_dyn_library "${_blas_library}-${_mt_tag}")
      set(_flame_static_library "${_flame_library}")
      set(_flame_dyn_library "${_flame_library}")
    else(SUPPORT_OMP)
      set(_blas_static_library "${_blas_library}")
      set(_blas_dyn_library "${_blas_library}")
      set(_flame_static_library "${_flame_library}")
      set(_flame_dyn_library "${_flame_library}")
    endif(SUPPORT_OMP)
  endif(WIN32)

  # Link against dynamic library by default
  find_library(
    AOCL_UTILS_LIB
    NAMES ${_utils_dyn_library} ${_utils_static_library} NAMES_PER_DIR
    HINTS ${AOCL_ROOT}/utils ${AOCL_ROOT}/amd-utils ${AOCL_ROOT}
    PATH_SUFFIXES "lib/${ILP_DIR}" "lib_${ILP_DIR}" "lib"
    DOC "AOCL Utils library")

  find_library(
    AOCL_BLIS_LIB
    NAMES ${_blas_dyn_library} ${_blas_static_library}
    HINTS ${AOCL_ROOT}/blis ${AOCL_ROOT}/amd-blis ${AOCL_ROOT}
    PATH_SUFFIXES "lib/${ILP_DIR}" "lib_${ILP_DIR}" "lib"
    DOC "AOCL Blis library")

  find_library(
    AOCL_LIBFLAME
    NAMES ${_flame_dyn_library} ${_flame_static_library} NAMES_PER_DIR
    HINTS ${AOCL_ROOT}/libflame ${AOCL_ROOT}/amd-libflame ${AOCL_ROOT}
    PATH_SUFFIXES "lib/${ILP_DIR}" "lib_${ILP_DIR}" "lib"
    DOC "AOCL LIBFLAME library")
  # ====Headers
  find_path(
    AOCL_UTILS_INCLUDE_DIR
    NAMES alci_c.h alci.h arch.h cache.h enum.h macros.h
    HINTS ${AOCL_ROOT}/amd-utils ${AOCL_ROOT}/utils ${AOCL_ROOT}
    PATH_SUFFIXES "include/${ILP_DIR}/alci" "include_${ILP_DIR}/alci"
                  "include/alci"
    DOC "AOCL Utils headers")

  find_path(
    AOCL_BLIS_INCLUDE_DIR
    NAMES blis.h cblas.h blis.hh cblas.hh
    HINTS ${AOCL_ROOT}/amd-blis ${AOCL_ROOT}/blis ${AOCL_ROOT}
    PATH_SUFFIXES "include/${ILP_DIR}" "include_${ILP_DIR}" "include"
                  "include/blis"
    DOC "AOCL Blis headers")
  find_path(
    AOCL_LIBFLAME_INCLUDE_DIR
    NAMES FLAME.h lapacke.h lapacke_mangling.h lapack.h libflame.hh
          libflame_interface.hh
    HINTS ${AOCL_ROOT}/libflame ${AOCL_ROOT}/amd-libflame ${AOCL_ROOT}
    PATH_SUFFIXES "include/${ILP_DIR}" "include_${ILP_DIR}" "include"
    DOC "AOCL Libflame headers")

  # ===========
  if(AOCL_BLIS_LIB
     AND AOCL_LIBFLAME
     AND AOCL_BLIS_INCLUDE_DIR
     AND AOCL_LIBFLAME_INCLUDE_DIR
     AND AOCL_UTILS_LIB
     AND AOCL_UTILS_INCLUDE_DIR)
    set(LAPACK_AOCL_FOUND
        true
        PARENT_SCOPE)
    set(LAPACK_LIBRARY ${AOCL_LIBFLAME} ${AOCL_BLIS_LIB} ${AOCL_UTILS_LIB})
  else()
    message(
      FATAL_ERROR
        "Error: could not find a suitable installation of Blas/Lapack/Utils Libraries in \$AOCL_ROOT=${AOCL_ROOT}"
    )
  endif()

  set(LAPACK_LIBRARY
      ${LAPACK_LIBRARY}
      PARENT_SCOPE)

endfunction(aocl_libs)

# ============= openmp function ================
function(openmp_libs)

  get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
  # Find OpenMP package
  find_package(OpenMP)
  get_property(OpenMP_Library DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
  list(REMOVE_ITEM OpenMP_Library ${importTargets})

  if(NOT OPENMP_FOUND)
    message(
      FATAL_ERROR
        "Error: could not find a suitable installation of openMP for the requested multi-threaded build"
    )    
  else()
    option(SUPPORT_OMP "Compile WITH OpenMP support." ON)    

    # OpenMP cmake fix for cmake <= 3.9
    if(NOT TARGET OpenMP::OpenMP_CXX)
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
        set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
    endif()

    if(WIN32)    
      set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};/openmp")
    else()  
      set(COMPILER_FLAGS_COMMON "${COMPILER_FLAGS_COMMON};${OpenMP_CXX_FLAGS}")
    endif()

    set(OPENMP_FOUND ${OPENMP_FOUND} PARENT_SCOPE)
    set(OpenMP_Library ${OpenMP_Library} PARENT_SCOPE)
    set(COMPILER_FLAGS_COMMON ${COMPILER_FLAGS_COMMON} PARENT_SCOPE)

  endif()

endfunction(openmp_libs)
# ==================main=================

# Add external dependencies such as Git, OpenMP Git
find_package(Git REQUIRED)

if(SUPPORT_OMP)
  openmp_libs()
endif(SUPPORT_OMP)

# fetch pthread library for Linux builds, irrespective of ST/MT modes
if(NOT WIN32)
  get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
  find_package(Threads REQUIRED)
  get_property(Threads_Library DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
  list(REMOVE_ITEM Threads_Library ${importTargets})
  #collect all threading libraries into a single variable for linking later  
  set(OpenMP_Library "${OpenMP_Library};${Threads_Library}")
endif()

# clear to avoid endless appending on subsequent calls
set(LAPACK_LIBRARY)
set(BLAS_LIBRARY)
unset(LAPACK_INCLUDE_DIR)

if(DEFINED ENV{AOCL_ROOT})
  set(AOCL_ROOT $ENV{AOCL_ROOT})
else()
  set(AOCL_ROOT ${CMAKE_INSTALL_PREFIX})
endif()

# find AOCL dependencies such as Blis, Libflame
aocl_libs()

set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})
set(BLIS_INCLUDE_DIRS ${AOCL_BLIS_INCLUDE_DIR})
set(LAPACK_INCLUDE_DIRS ${AOCL_LIBFLAME_INCLUDE_DIR})
# Utils package is built with relative header path inclusion. Finding absolute
# path results in compilation errors in windows builds. So path is adjusted to a
# folder under "include" that contains "alci" folder. Therefore, the below path
# to Utils Headers is adjusted. If we can get the right path using find_path( ),
# then we can remove this adjustment.
set(UTILS_INCLUDE_DIRS ${AOCL_UTILS_INCLUDE_DIR}/..)

message(STATUS "Dependencies (libraries and includes)")
message(STATUS "  \$LAPACK LIBRARIES......${LAPACK_LIBRARIES}")
message(STATUS "  \$LAPACK_INCLUDE_DIRS...${LAPACK_INCLUDE_DIRS}")
message(STATUS "  \$BLIS_INCLUDE_DIRS.....${BLIS_INCLUDE_DIRS}")
message(STATUS "  \$UTILS_INCLUDE_DIRS....${UTILS_INCLUDE_DIRS}")
if(SUPPORT_OMP)
  message(STATUS "  \$OpenMP_Library....${OpenMP_Library}")
  message(STATUS "  \$OpenMP_Flags....${COMPILER_FLAGS_COMMON}")
else(SUPPORT_OMP)
  message(STATUS "  \$Threads Library....${Threads_Library}")
endif(SUPPORT_OMP)

mark_as_advanced(LAPACK_LIBRARIES LAPACK_INCLUDE_DIRS BLIS_INCLUDE_DIRS)
