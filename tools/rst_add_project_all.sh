#!/bin/bash
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

# Use this script to add :project: directive (via rst_add_project.awk)
# to any rst file passed as an argument
#
# Usage examples:
#   tools/rst_add_project_all.sh docs/*.rst
#   find docs -name '*.rst' | xargs tools/rst_add_project_all.sh

help() {
  echo "Usage: $0 [-p project_name] file1 [file2 ...]"
}

if [ $# -eq 0 ]; then
  help
  exit 1
fi

# retrieve the project_name if present
project_name=
case $1 in
  -p)
    if [ -n "$2" ]; then
      project_name=$2
      shift 2
    else
      help
      exit 1
    fi
  ;;
esac

mydir=$(dirname "$0")

for f in "$@"; do
  echo Working on $f ...
  mv $f $f~
  gawk -f "$mydir/rst_add_project.awk" -v project_name=$project_name $f~ > $f
done

