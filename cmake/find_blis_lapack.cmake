# ########################################################################
#Copyright(c) 2023 Advanced Micro Devices, Inc.
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

# ============= aocl function ================
function(aocl_libs)

  IF(BUILD_ILP64) 
    SET(ILP_DIR "ILP64")
  ELSE(BUILD_ILP64)
    SET(ILP_DIR "LP64")
  ENDIF(BUILD_ILP64)

  IF(WIN32)
    SET(CMAKE_FIND_LIBRARY_PREFIXES "")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib") 
    find_library(AOCL_BLIS_LIB
    NAMES AOCL-LibBlis-Win-MT AOCL-LibBlis-Win-MT-dll AOCL-LibBlis-Win AOCL-LibBlis-Win-dll
    HINTS ${AOCL_ROOT}/blis ${AOCL_ROOT}/amd-blis ${AOCL_ROOT}
    PATH_SUFFIXES "lib/${ILP_DIR}" "lib"
    DOC "AOCL Blis library"
    )

    find_library(AOCL_LIBFLAME
    NAMES AOCL-LibFlame-Win-MT AOCL-LibFlame-Win-MT-dll AOCL-LibFlame-Win AOCL-LibFlame-Win-dll
    NAMES_PER_DIR
    HINTS ${AOCL_ROOT}/libflame ${AOCL_ROOT}/amd-libflame ${AOCL_ROOT}
    PATH_SUFFIXES "lib/${ILP_DIR}" "lib"
    DOC "AOCL LIBFLAME library"
    )
    #====Headers
    find_path(AOCL_BLIS_INCLUDE_DIR
    NAMES blis.h  cblas.h
    HINTS ${AOCL_ROOT}/amd-blis ${AOCL_ROOT}/blis ${AOCL_ROOT}
    PATH_SUFFIXES "include/${ILP_DIR}" "include" "include/blis"
    DOC "AOCL Blis headers"
    )

    find_path(AOCL_LIBFLAME_INCLUDE_DIR
    NAMES FLAME.h  lapacke.h  lapacke_mangling.h  lapack.h  lapacke_config.h lapacke_utils.h
    HINTS ${AOCL_ROOT}/libflame ${AOCL_ROOT}/amd-libflame ${AOCL_ROOT}
    PATH_SUFFIXES "include/${ILP_DIR}" "include"
    DOC "AOCL Libflame headers" 
    )
  ELSE(WIN32)   
    SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    IF(BUILD_SHARED_LIBS)
      SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
    ELSE(BUILD_SHARED_LIBS)
      SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    ENDIF(BUILD_SHARED_LIBS) 

    find_library(AOCL_BLIS_LIB
    NAMES blis-mt blis
    HINTS ${AOCL_ROOT}/blis ${AOCL_ROOT}/amd-blis ${AOCL_ROOT}
    PATH_SUFFIXES "lib/${ILP_DIR}" "lib"
    DOC "AOCL Blis library"
    )
    
    find_library(AOCL_LIBFLAME
    NAMES flame
    NAMES_PER_DIR
    HINTS ${AOCL_ROOT}/libflame ${AOCL_ROOT}/amd-libflame ${AOCL_ROOT}
    PATH_SUFFIXES "lib/${ILP_DIR}" "lib"
    DOC "AOCL LIBFLAME library"
    )
    #====Headers
    find_path(AOCL_BLIS_INCLUDE_DIR
    NAMES blis.h  blis.hh  cblas.h  cblas.hh
    HINTS ${AOCL_ROOT}/blis ${AOCL_ROOT}/amd-blis ${AOCL_ROOT}
    PATH_SUFFIXES "include/${ILP_DIR}" "include" "include/blis"
    DOC "AOCL Blis headers"
    )

    find_path(AOCL_LIBFLAME_INCLUDE_DIR
    NAMES FLAME.h  lapacke.h  lapacke_mangling.h  lapack.h  libflame.hh  libflame_interface.hh
    HINTS ${AOCL_ROOT}/libflame ${AOCL_ROOT}/amd-libflame ${AOCL_ROOT}
    PATH_SUFFIXES "include/${ILP_DIR}" "include"
    DOC "AOCL Libflame headers" 
    )
  ENDIF(WIN32)

  #===========
  if(AOCL_BLIS_LIB AND AOCL_LIBFLAME AND AOCL_BLIS_INCLUDE_DIR AND AOCL_LIBFLAME_INCLUDE_DIR)
    set(LAPACK_AOCL_FOUND true PARENT_SCOPE)
    set(LAPACK_LIBRARY ${AOCL_LIBFLAME} ${AOCL_BLIS_LIB})  
  else()    
    message (FATAL_ERROR "Error: could not find a suitable installation of Blas/Lapack Libraries in \$AOCL_ROOT=${AOCL_ROOT}")
  endif()

  set(LAPACK_LIBRARY ${LAPACK_LIBRARY} PARENT_SCOPE)

endfunction(aocl_libs)

#==================main=================
# clear to avoid endless appending on subsequent calls
set(LAPACK_LIBRARY)
set(BLAS_LIBRARY)
unset(LAPACK_INCLUDE_DIR)

if(DEFINED ENV{AOCL_ROOT})            
    SET(AOCL_ROOT $ENV{AOCL_ROOT})
ELSE()      
    set(AOCL_ROOT ${CMAKE_INSTALL_PREFIX})
ENDIF()

aocl_libs()

set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})
set(BLIS_INCLUDE_DIRS ${AOCL_BLIS_INCLUDE_DIR})
set(LAPACK_INCLUDE_DIRS ${AOCL_LIBFLAME_INCLUDE_DIR})

message(STATUS "LAPACK_LIBRARIES= ${LAPACK_LIBRARIES}")
message(STATUS "LAPACK_INCLUDE_DIRS= ${LAPACK_INCLUDE_DIRS}")
message(STATUS "BLIS_INCLUDE_DIRS= ${BLIS_INCLUDE_DIRS}")

mark_as_advanced(LAPACK_LIBRARIES LAPACK_INCLUDE_DIRS BLIS_INCLUDE_DIRS)
