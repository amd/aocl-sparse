# ##############################################################################
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(aoclsparse-doc)

find_package(Doxygen REQUIRED)
find_package(LATEX REQUIRED)
find_program(SED NAMES sed REQUIRED)
find_program(GREP NAMES grep REQUIRED)
find_program(SORT NAMES sort REQUIRED)

# Check for python and the documentation packages:
# - rocm_docs
# - breathe
# - sphinxcontrib.bibtex
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from rocm_docs import ROCmDocs"
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET)
if(NOT ${EXIT_CODE} EQUAL 0)
  message(
    FATAL_ERROR
      "Python3 package rocm_docs is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install rocm-docs-core"
  )
endif()
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import breathe"
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET)
if(NOT ${EXIT_CODE} EQUAL 0)
  message(
    FATAL_ERROR
      "Python3 package breathe is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install breathe"
  )
endif()
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sphinxcontrib.bibtex"
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET)
if(NOT ${EXIT_CODE} EQUAL 0)
  message(
    FATAL_ERROR
      "Python3 package sphinxcontrib.bibtex is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install sphinxcontrib.bibtex"
  )
endif()
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sphinx_collapse"
  RESULT_VARIABLE EXIT_CODE
  OUTPUT_QUIET)
if(NOT ${EXIT_CODE} EQUAL 0)
  message(
    FATAL_ERROR
      "Python3 package sphinx_collapse is required to build the documentation\n   Use: ${Python3_EXECUTABLE} -m pip install sphinx_collapse"
  )
endif()

message(STATUS "Building AOCL-Sparse documentation version ${AOCLSPARSE_VERSION}")

string(TIMESTAMP BUILD_DATE "%d %b %Y")

# This target makes sure all latex packaged are available prior to launching the
# full build build a here-document from the main doc which takes all required
# packages NOTE: this here-document needs to be kept in sync with refman.tex
# TODO: Automate the generation of the here-document
add_custom_target(
  latexdeps_sphinx.pdf
  COMMAND
    bash -c
    [[ echo -e '\\documentclass[twoside]{book}\n\\usepackage{ifthen}\n\\ifx\\requestedLaTeXdate\\undefined\n\\usepackage{array}\n\\else\n\\usepackage{array}[=2016-10-06]\n\\fi\n\\usepackage{fixltx2e}\n%\\usepackage{doxygen}\n\\usepackage{graphicx}\n\\usepackage[utf8]{inputenc}\n\\usepackage{makeidx}\n\\PassOptionsToPackage{warn}{textcomp}\n\\usepackage{textcomp}\n\\usepackage[nointegrals]{wasysym}\n\\usepackage{ifxetex}\n\\usepackage[T1]{fontenc}\n\\usepackage[scaled=.90]{helvet}\n\\usepackage{courier}\n\\usepackage{sectsty}\n\\usepackage{geometry}\n\\usepackage{fancyhdr}\n\\usepackage{natbib}\n\\usepackage[titles]{tocloft}\n\\usepackage{amsmath}\n\\usepackage{newunicodechar}\n\\ifpdf\n\\usepackage[pdftex,pagebackref=true]{hyperref}\n\\else\n\\ifxetex\n\\usepackage[pagebackref=true]{hyperref}\n\\else\n\\usepackage[ps2pdf,pagebackref=true]{hyperref}\n\\fi\n\\fi\n\\usepackage{etoc}\n\\begin{document}\n\\end{document}' > latexdeps_sphinx.tex ]]
  COMMAND
    bash -c
    [[ pdflatex -halt-on-error -interaction nonstopmode latexdeps_sphinx.tex ; if [ $? -ne 0 ] ; then echo -e '\n! LaTeX Error: Could not check for the required dependencies. Check latexdeps_sphinx.log for\nmore details and make sure your LaTeX distribution has the following packages:' ; grep usepackage latexdeps_sphinx.tex | sed 's/^ *\\usepackage.*{\(.*\)}.*$/\1/' | sort -u  ; echo ; exit 1 ; fi ]]
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Checking for LaTeX dependencies..."
  VERBATIM
  SOURCES Doxyfile)

add_custom_target(
  clean-doc
  COMMAND cd ${PROJECT_BINARY_DIR} && rm -vrf html/ latex/ rtf/ xml/ sphinx/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Cleaning temporary documentation files"
  VERBATIM)

add_dependencies(cleanall clean-doc)

# Build sphinx Documentation DOC_TYPE controls what document to produce and can
# be set to either pdf or html, default is html
#
# Inspired from Sphinx in CMake section of MSFT blog post
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Add Sphinx doc target
find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(PDF_DOC_NAME AOCL-Sparse_Documentation_${AOCLSPARSE_VERSION})
option(INTERNAL_DOC "Build internal documentation" ON)
if(INTERNAL_DOC)
  set(SPHINX_TAG "-t" "internal")
else()
  set(SPHINX_TAG "")
endif()

# Target building the doxygen XML Also modifies the output directory and the
# version number in the Doxyfile: 1. replace PROJECT_NAME = .* for the string
# "version ${AOCLSPARSE_VERSION}" 2. Run doxygen on the sources
add_custom_target(
  doxygen_xml
  COMMAND
  # Version bump and set the correct output directory
  # "s/^\\(PROJECT_NUMBER[[:space:]]*=\\).*$/\\1 \"version ${AOCLSPARSE_VERSION} (${BUILD_DATE})\"/"
    ${SED} -i -e
    "s/^\\(PROJECT_NUMBER[[:space:]]*=\\).*$/\\1 \"Release ${AOCLSPARSE_VERSION}\"/"
    -e "s+^\\(OUTPUT_DIRECTORY[[:space:]]*=\\).*$+\\1 ${PROJECT_BINARY_DIR}+"
    Doxyfile
  # Run doxygen THEN reset the OUTPUT_DIRECTORY
  COMMAND doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile &&
    ${SED} -i -e
    "s+^\\(OUTPUT_DIRECTORY[[:space:]]*=\\).*$+\\1 \"auto-generated-by-cmake\"+"
    Doxyfile
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  VERBATIM
  SOURCES Doxyfile
  COMMENT "Building doxygen XML files")


# This target updates some of the documentation source/config files copyright years and the version number to
# match with ${AOCLSPARSE_VERSION}
# Project repo reports start
set(INITIAL_YEAR "2020")
string(TIMESTAMP YEAR "%Y")
add_custom_target(
  update_doc_dates
  # update copyright years
  COMMAND ${SED} -i -e
  "s/\\(^[[:space:]]*copyright[[:space:]]*=[[:space:]]*'\\)\\([0-9]\\{4\\}-\\)\\([0-9]\\{4\\}\\)/\\1${INITIAL_YEAR}-${YEAR}/" conf.py
  # update release / version number
  COMMAND ${SED} -i -e "s/\\(^[[:space:]]*release[[:space:]]*=\\)\\(.*\\)/\\1\'${AOCLSPARSE_VERSION}\'/" conf.py
  # update copyright years in footer
  COMMAND ${SED} -i -e "s/[0-9]\\{4\\}-[0-9]\\{4\\}/${INITIAL_YEAR}-${YEAR}/" _template/flavors/local/footer.jinja
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  VERBATIM
  SOURCES conf.py _template/flavors/local/footer.jinja
  COMMENT "Updating version number and copyright years in conf.py")

# Build the html documentation in doc/sphinx/html
add_custom_target(
  doc_html
  COMMAND
    ${SPHINX_EXECUTABLE} ${SPHINX_TAG} -b html
    -Dbreathe_projects.aocl-sparse=${DOXYGEN_OUTPUT_DIR}/xml ${SPHINX_SOURCE}
    ${SPHINX_BUILD}/html
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS doxygen_xml update_doc_dates
  COMMENT "Generating HTML documentation")

# Build a latex documentation in doc/sphinx/latex
add_custom_target(
  doc_pdf
  COMMAND
    ${SPHINX_EXECUTABLE} ${SPHINX_TAG} -b latex
    # Tell Breathe where to find the Doxygen output
    -Dbreathe_projects.aocl-sparse=${DOXYGEN_OUTPUT_DIR}/xml ${SPHINX_SOURCE}
    ${SPHINX_BUILD}/latex/
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND
    cd ${SPHINX_BUILD}/latex && make
    && cp -v ${SPHINX_BUILD}/latex/aocl-sparse.pdf ${CMAKE_CURRENT_SOURCE_DIR}/AOCL-Sparse_API_Guide.pdf
    && mv ${SPHINX_BUILD}/latex/aocl-sparse.pdf ${SPHINX_BUILD}/latex/${PDF_DOC_NAME}.pdf
  DEPENDS doxygen_xml latexdeps_sphinx.pdf
  COMMENT "Generating PDF documentation")

# Main target doc, builds both html and pdf
add_custom_target(doc DEPENDS doc_pdf doc_html)
# Add alias to legacy target
add_custom_target(docs DEPENDS doc)

