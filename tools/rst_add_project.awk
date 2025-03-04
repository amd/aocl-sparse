#!/bin/gawk -f
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
############################################################

# This script extends any breathe's ".. doxygen*" directive
# with :project: directive including the project name
# This is useful when multiple API docs from different projects
# get merged into a single documentation.

BEGIN {
    project_name = "sparse"
    indent = 0
}

{
    if (indent>0) {
        # count number of spaces before the first non-space character
        loc_indent = match($0, /[^ ]/) - 1
        if (loc_indent>=indent) {
            # indentation block of the same directive, match the indentation
            indent = loc_indent
        }
        # print project directive
        printf "%*s:project: %s\n", indent, "", project_name
        indent = 0
    }
    print
}

# doxygen directive -> need to extend it
/^ *\.\. doxygen/ {
    # find number of spaces before "doxygen" for minimal indenting
    indent = match($0, /doxygen/) - 1
    next
}

END {
    if (indent>0) {
        # print project directive
        printf "%*s:project: %s\n", indent, "", project_name
    }
}

