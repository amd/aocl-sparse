#!/bin/env bash
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

# SYNOPSYS
# warningtable.sh -- build a table of warnings generated during
# compilation. Its taylored for GCC (g++) or Clang with flags such 
# as -Wall and -Wextra

BIN="$0"

function print_usage (){
  echo -e "\nParse a compiler log for warnings and pretty-print a table\n"
  echo -e "Usage:\n"
  echo -e "    $BIN /tmp/make-sup.log\n"
  echo "Where make-sup.log file is produced with a command similar to:"
  echo -e "    cd \$AOCL_ROOT/build && cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug \\"
  echo "    -DBUILD_CLIENTS_TESTS=On -DBUILD_CLIENTS_SAMPLES=On && make clean && make -j 1 \\"
  echo "    VERBOSE=1 2>&1 | tee /tmp/make-sup.log"
  exit 1
}

if [ $# -lt 1 ] ; then
  print_usage
fi

INPUT="$1"

if [ ! -r "$INPUT" ] ; then
  echo -e "\nError: file \`$INPUT' cannot be read."
  print_usage
fi

TMP=`mktemp`
echo "warning%filepath%line%(col)%details" > "$TMP"
# grep 'warning:' "$INPUT" | sed 's/\(^.\+aocl-sparse\/\)\(.*\):\([0-9]\+\):\([0-9]\+\):.warning:.\(.*\)\[\(.\+\)\]/\6%\2%\3%(\4)%\5/' >> "$TMP"
grep 'warning:' "$INPUT" | sed 's/\(^.\+\):\([0-9]\+\):\([0-9]\+\):.warning:.\(.*\)\[\(.\+\)\]/\5%\1%\2%(\3)%\4/' >> "$TMP"
column -t -s % "$TMP"

echo -e "\n\n Statistics:\n"
echo '   count warning'
cut -f 1 -d %  "$TMP" | sort | uniq -c | grep -- -W | sort -nr
rm "$TMP"

