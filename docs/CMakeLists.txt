# ########################################################################
# Copyright (c) 2022 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(aoclsparse-doc)

find_package(Doxygen REQUIRED)
find_package(LATEX REQUIRED)
find_program(SED NAMES sed HINTS "/usr" PATH_SUFFIXES "bin" DOC "sed - stream editor for filtering and transforming text")
if (NOT SED)
    message(FATAL_ERROR "sed: program not found. Document building requires this program.")
endif()
find_program(GREP NAMES sed HINTS "/usr" PATH_SUFFIXES "bin" DOC "grep - print lines that match patterns")
if (NOT GREP)
    message(FATAL_ERROR "grep: program not found. Document building requires this program.")
endif()
find_program(SORT NAMES sed HINTS "/usr" PATH_SUFFIXES "bin" DOC "sort - sort lines")
if (NOT GREP)
    message(FATAL_ERROR "sort: program not found. Document building requires this program.")
endif()

message( STATUS "Building documentation for version ${AOCLSPARSE_VERSION}" )

string(TIMESTAMP BUILD_DATE "%d %b %Y")


# This target makes sure all latex packaged are available
# prior to launching the full build
# build a here-document from the main doc which takes all required packages
# NOTE: this here-document needs to be kept in sync with refman.tex
# TODO: Automate the generation of the here-document
add_custom_target( latexdeps.pdf
  COMMAND bash -c [[ echo -e '\\documentclass[twoside]{book}\n\\usepackage{ifthen}\n\\ifx\\requestedLaTeXdate\\undefined\n\\usepackage{array}\n\\else\n\\usepackage{array}[=2016-10-06]\n\\fi\n\\usepackage{fixltx2e}\n%\\usepackage{doxygen}\n\\usepackage{graphicx}\n\\usepackage[utf8]{inputenc}\n\\usepackage{makeidx}\n\\PassOptionsToPackage{warn}{textcomp}\n\\usepackage{textcomp}\n\\usepackage[nointegrals]{wasysym}\n\\usepackage{ifxetex}\n\\usepackage[T1]{fontenc}\n\\usepackage[scaled=.90]{helvet}\n\\usepackage{courier}\n\\usepackage{sectsty}\n\\usepackage{geometry}\n\\usepackage{fancyhdr}\n\\usepackage{natbib}\n\\usepackage[titles]{tocloft}\n\\usepackage{amsmath}\n\\usepackage{newunicodechar}\n\\ifpdf\n\\usepackage[pdftex,pagebackref=true]{hyperref}\n\\else\n\\ifxetex\n\\usepackage[pagebackref=true]{hyperref}\n\\else\n\\usepackage[ps2pdf,pagebackref=true]{hyperref}\n\\fi\n\\fi\n\\usepackage{etoc}\n\\begin{document}\n\\end{document}' > latexdeps.tex ]]
  COMMAND bash -c [[ pdflatex -halt-on-error -interaction nonstopmode latexdeps.tex ; if [ $? -ne 0 ] ; then echo -e '\n! LaTeX Error: Could not check for the required dependencies. Check latexdeps.log for\nmore details and make sure your LaTeX distribution has the following packages:' ; grep usepackage latexdeps.tex | sed 's/^ *\\usepackage.*{\(.*\)}.*$/\1/' | sort -u  ; echo ; exit 1 ; fi ]]
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Checking for LaTeX dependencies..." VERBATIM
  SOURCES Doxyfile
)

# This target does the following
# 1. Bump version in configuration file: replace PORJECT_NAME = * for the string "version ${AOCLSPARSE_VERSION} (date)" OR (git commit hash?)
# 2. Run doxygen on the sources
# 3. Run pdflatex
# 4. Move and rename the pdf file

add_custom_target( docs
  COMMAND sed -i -e "s/^\\(PROJECT_NUMBER[[:space:]]*=\\).*$/\\1 \"version ${AOCLSPARSE_VERSION} (${BUILD_DATE})\"/" -e "s+^\\(OUTPUT_DIRECTORY[[:space:]]*=\\).*$+\\1 ${PROJECT_BINARY_DIR}+" Doxyfile &&
  cd ../library && doxygen ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile && cd ${PROJECT_BINARY_DIR}/latex/ && make && cp -v refman.pdf ${CMAKE_CURRENT_SOURCE_DIR}/AOCL-Sparse_API_Guide.pdf
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Building documentation (PDF) from source" VERBATIM
  SOURCES Doxyfile
  DEPENDS latexdeps.pdf
)

add_custom_target( clean-docs
  COMMAND cd ${PROJECT_BINARY_DIR} && rm -vrf html/ latex/ rtf/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Cleaning temporary documentation files" VERBATIM
)
