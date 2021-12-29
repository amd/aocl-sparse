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

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configure a header file to pass the aoclsparse version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/aoclsparse_version.h.in"
               "${PROJECT_BINARY_DIR}/include/aoclsparse_version.h"
)

# Public aoclsparse headers
set(PUBLIC_HEADERS
    include/aoclsparse_types.h
    include/aoclsparse_functions.h
    include/aoclsparse_convert.h
    include/aoclsparse_auxiliary.h
    include/aoclsparse.h
    ${PROJECT_BINARY_DIR}/include/aoclsparse_version.h
)

source_group("Header Files\\Public" FILES ${PUBLIC_HEADERS})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)


# TODO plug-ins

# Create aoclsparse library
add_library(aoclsparse ${AOCLSPARSE_SOURCES} ${PUBLIC_HEADERS})
add_library(aocl::aoclsparse ALIAS aoclsparse)

# Target link libraries
if(SUPPORT_OMP)
target_link_libraries(aoclsparse PRIVATE OpenMP::OpenMP_CXX)
endif()

# Target include directories
target_include_directories(aoclsparse
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<INSTALL_INTERFACE:include>
)

# Target compile options
if(SUPPORT_OMP)
  target_compile_options(aoclsparse PUBLIC ${OpenMP_CXX_FLAGS})
endif()

# Target properties
set_target_properties(aoclsparse PROPERTIES SOVERSION ${aoclsparse_SOVERSION} )
set_target_properties(aoclsparse PROPERTIES VERSION ${aoclsparse_SOVERSION})
set_target_properties(aoclsparse PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
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
