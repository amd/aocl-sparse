################################################################################
# Copyright(c) 2023 Advanced Micro Devices, Inc.
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
################################################################################

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(aoclsparse-lcov-coverage)

find_program(LCOV NAMES lcov HINTS "/usr" PATH_SUFFIXES "bin" DOC "lcov - a graphical GCOV front-end" REQUIRED)
find_program(GCOV NAMES $ENV{GCOV_NAME} gcov HINTS "/usr" PATH_SUFFIXES "bin" DOC "GNU gcov binary" REQUIRED)
find_program(GENHTML NAMES genhtml HINTS "/usr" PATH_SUFFIXES "bin" DOC "genhtml - Generate HTML view from LCOV coverage data files" REQUIRED)

message( STATUS "Code Coverage Module (LCOV)" )

add_custom_target( coverage-clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf coverage/
  COMMAND find . -name *.gcda -exec rm -v {} \;
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Cleaning coverage related files" VERBATIM
)

add_custom_target( coverage-run
  COMMAND ${CMAKE_MAKE_PROGRAM} all
  COMMAND ${CMAKE_MAKE_PROGRAM} coverage-clean
  COMMAND CLICOLOR=0 ASAN_OPTIONS="log_path=ASANlogger" ctest --timeout 20 --output-junit Testing/Temporary/LastTest_JUnit.xml || true
  COMMAND ${CMAKE_SOURCE_DIR}/tools/collate_asan.sh
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Running Code Coverage (ctest + collate_asan)"
)

add_custom_target( coverage-report
  COMMAND ${CMAKE_COMMAND} -E make_directory coverage/
  COMMAND ${LCOV} -d . -c -o coverage/coverage.info --gcov-tool ${GCOV}
  COMMAND ${LCOV} --remove   coverage/coverage.info --gcov-tool ${GCOV} -o coverage/coverage_filtered.info '/usr/*' '/*/_deps/*'
  COMMAND ${GENHTML} coverage/coverage_filtered.info --output coverage/html --title "AOCL-Sparse Code Coverage Report" --legend
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Building Code Coverage Report (LCOV)"
)

# Alias (only Makefile/Linux)
add_custom_target( coverage
  DEPENDS coverage-report
)

add_custom_target( cleanall
  COMMAND ${CMAKE_MAKE_PROGRAM} clean
  DEPENDS coverage-clean
)
