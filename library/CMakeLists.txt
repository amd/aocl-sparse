# ########################################################################
# Copyright (c) 2020-2026 Advanced Micro Devices, Inc.
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

# Configure a header file to pass the aoclsparse version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/aoclsparse_version.h.in"
               "${PROJECT_BINARY_DIR}/include/aoclsparse_version.h"
)

# Public aoclsparse headers
set(PUBLIC_HEADERS
    include/aoclsparse_types.h
    include/aoclsparse_functions.h
    include/aoclsparse_convert.h
    include/aoclsparse_analysis.h
    include/aoclsparse_auxiliary.h
    include/aoclsparse_solvers.h
    include/aoclsparse.h
    include/aoclsparse.hpp
    ${PROJECT_BINARY_DIR}/include/aoclsparse_version.h
)

source_group("Header Files\\Public" FILES ${PUBLIC_HEADERS})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)

function(aoclsparse_add_dependency OBJ_NAME)
  target_include_directories(${OBJ_NAME}
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/analysis>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/level2>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/level1>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/extra>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/conversion>
                           PUBLIC  $<BUILD_INTERFACE:${AOCLSPARSE_ROOT}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   # NOTE: the dependency include dirs are listed
                                   # Dependency include dirs are listed WITHOUT a wrapping $<BUILD_INTERFACE:...>:
                                   # each may be a ';'-separated list (e.g. AOCL-Utils splits headers across folders
                                   # in a target-based build), and a list inside a single genex collapses into one
                                   # malformed path. The target isn't exported, so plain build-tree dirs are correct.
                                   ${LAPACK_INCLUDE_DIRS}
                                   ${BLIS_INCLUDE_DIRS}
                                   ${UTILS_INCLUDE_DIRS}
                                   $<INSTALL_INTERFACE:include>
  )

 # Add definitions that depend on the CMake configuration.
 target_compile_definitions(${OBJ_NAME} PRIVATE ${AOCLSPARSE_DEFS})

 # Define -Daoclsparse_ILP64 when ILP64 libraries are built. For LP64, this is an empty string.
 target_compile_definitions(${OBJ_NAME} PRIVATE ${AOCLSPARSE_ILP64})

 # Sets -fPIC flag, symbol visibility to hidden
 set_target_properties(${OBJ_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

 # Propagate OpenMP usage requirements to the target. This is essential for
 # the AVX2/AVX512 OBJECT libraries whose kernels (kt / avx512) have openmp pragmas.
 if(SUPPORT_OMP)
   target_link_libraries(${OBJ_NAME} PRIVATE ${OpenMP_Library})
 endif()
endfunction()

# ----------------- Double compilation files with AVX2 flags -------------------------------
add_library(DOUBLE_COMPILATION_AVX2 OBJECT ${DOUBLE_COMPILATION})
target_compile_definitions(DOUBLE_COMPILATION_AVX2 PRIVATE -DAOCLSPARSE_AVX2_INSTANT)
target_compile_options(DOUBLE_COMPILATION_AVX2 PRIVATE "-mno-avx512f;" ${AOCLSPARSE_GENERIC_FLAGS})
aoclsparse_add_dependency(DOUBLE_COMPILATION_AVX2)  # also propagates OpenMP (see helper)
# ==========================================================================================

# List of all targets in the library
set(AOCLSPARSE_TARGETS $<TARGET_OBJECTS:DOUBLE_COMPILATION_AVX2>)

if(USE_AVX512)
 # ---------------- Double compilation files with AVX 512 flags -----------------------------
 add_library(DOUBLE_COMPILATION_AVX512 OBJECT ${DOUBLE_COMPILATION})
 target_compile_options(DOUBLE_COMPILATION_AVX512 PRIVATE ${AOCLSPARSE_AVX512_FLAGS}${AOCLSPARSE_GENERIC_FLAGS})
 aoclsparse_add_dependency(DOUBLE_COMPILATION_AVX512)  # also propagates OpenMP (see helper)
 # ==========================================================================================

 # ---------------- Compile files with AVX512 intrinsics ------------------------------------
 add_library(AVX512_FILES OBJECT ${AVX512_SOURCES})
 target_compile_options(AVX512_FILES PRIVATE ${AOCLSPARSE_AVX512_FLAGS}${AOCLSPARSE_GENERIC_FLAGS})
 aoclsparse_add_dependency(AVX512_FILES)  # also propagates OpenMP (see helper)
 # ==========================================================================================

 # If the target is AVX512, append all the AVX512 objects
 list(APPEND AOCLSPARSE_TARGETS $<TARGET_OBJECTS:AVX512_FILES> $<TARGET_OBJECTS:DOUBLE_COMPILATION_AVX512>)

endif()

add_library(aoclsparse ${AOCLSPARSE_SOURCES} ${PUBLIC_HEADERS} ${AOCLSPARSE_TARGETS})
target_compile_options(aoclsparse PRIVATE "-mno-avx512f;" ${AOCLSPARSE_GENERIC_FLAGS})
# TODO plug-ins

add_library(aocl::aoclsparse ALIAS aoclsparse)
aoclsparse_add_dependency(aoclsparse)

# AOCL Build-It-Yourself unified build: contribute sparse's objects to libaocl.
# The direct sources become a sibling OBJECT library (via objectify, which copies
# the "-mno-avx512f" / generic flags and the LAPACK/OpenMP include usage); the
# AVX2/AVX512 OBJECT libraries already referenced in AOCLSPARSE_TARGETS are
# registered directly as their $<TARGET_OBJECTS:...> genexes.
if(AOCL_TB_UNIFIED_BUILD)
    aocl_tb_objectify(aoclsparse)
    aocl_tb_add_objects(${AOCLSPARSE_TARGETS})
endif()

# Target link libraries and compile options
if(SUPPORT_OMP)
  target_link_libraries(aoclsparse PRIVATE ${OpenMP_Library}
                                  PUBLIC ${LAPACK_LIBRARIES})
else()
  target_link_libraries(aoclsparse PUBLIC ${LAPACK_LIBRARIES})
endif()

# Target properties
set_target_properties(aoclsparse PROPERTIES SOVERSION ${AOCLSPARSE_VERSION} )
set_target_properties(aoclsparse PROPERTIES VERSION ${AOCLSPARSE_VERSION})
set_target_properties(aoclsparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/library")
set_target_properties(aoclsparse PROPERTIES DEBUG_POSTFIX "-d")

install(TARGETS aoclsparse DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

# Package name
set(package_name aoclsparse)

set(AOCLSPARSE_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

# Install the kernel-templates directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/include/kernel-templates/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/kernel-templates
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
