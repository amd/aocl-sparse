# ########################################################################
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
project(aoclsparse-doc)
message( STATUS "Building documentation for version ${AOCLSPARSE_VERSION}" )

string(TIMESTAMP BUILD_DATE "%d %b %Y")

# This target does the following
# 1. Bump version in configuration file: replace PORJECT_NAME = * for the string "version ${AOCLSPARSE_VERSION} (date)" OR (git commit hash?)
# 2. Run doxygen on the sources
# 3. Run pdflatex
# 4. Move and rename the pdf file

add_custom_target( AOCL-Sparse_API_Guide.pdf ALL
  COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && pwd &&
  sed -i -e "s/^\\(PROJECT_NUMBER[[:space:]]*=\\).*$/\\1 version ${AOCLSPARSE_VERSION} (${BUILD_DATE})/" -e "s+^\\(OUTPUT_DIRECTORY[[:space:]]*=\\).*$+\\1 ${PROJECT_BINARY_DIR}+" Doxyfile &&
  cd ../library && doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile && cd ${PROJECT_BINARY_DIR}/latex/ && make && cp -v refman.pdf ${CMAKE_CURRENT_SOURCE_DIR}/AOCL-Sparse_API_Guide.pdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Building documentation (PDF) from source" VERBATIM
  SOURCES Doxyfile
)

add_custom_target( clean-docs
  COMMAND cd ${PROJECT_BINARY_DIR} && rm -vrf html/ latex/ rtf/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Cleaning temporary documentation files" VERBATIM
)
