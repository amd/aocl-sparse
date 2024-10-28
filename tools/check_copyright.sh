#!/bin/bash
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#
# Check if all files have adjusted current copyright year
# and return number of non-matching files.
# It is assumed that the copyright label should on the top of the files
# in the form: 'Copyright (c) 2020-2024' or 'Copyright (c) 2024'

# Whitelist of files which are considered always OK
whitelist=(README.md tests/examples/README.md docs/Doxyfile docs/AOCL-Sparse_API_Guide.pdf LICENSE.md NOTICES CMakePresets.json cmake/presets/README.MD cmake/presets/base.json cmake/presets/linux-make.json cmake/presets/linux-ninja.json cmake/presets/win-msvc.json cmake/presets/win-ninja.json)

print_help() {
  echo $0 [-p] [file1 [file2 ...]]
  echo Checks if files have adjusted copyright to the current year
  echo and return the number of non-matching files.
  echo If no file is given, all staged files are checked.
  echo Only regular files are checked, whitelisted files
  echo are skipped: ${whitelist[@]}
  echo
  echo '-p flag indicates to print the copyright header for ANY year or'
  echo a message indicating no copyright info found. The flag NEEDS to
  echo be the first argument.
}

PRINT=

if [ "$#" -gt 0 ]; then
  if [ "$1" = "-p" ]; then
    PRINT='-p'
    shift
  fi
fi

if [ "$#" -eq 0 ]; then
  print_help
  echo ""
  echo "Testing all staged files"
  git status --porcelain | grep '^[AM]' | cut -c4- | xargs --no-run-if-empty bash $0 $PRINT
  exit
fi

nfailed=0

if [ ! -z "$PRINT" ]; then
  current_year='\([0-9]\{4\}\)'
  message=" does NOT include copyright banner"
  function prn () { echo "$1:" \"$2\" ; }
else
  current_year=$(date '+%Y')
  message=" does NOT include $current_year in the copyright"
  function prn () { echo "$1" is ok. ; }
fi

for f in "$@"; do

  if [ ! -f "$f" ]; then
    echo "$f" is not a regular file, skipping
    continue
  fi

  # Check if the first argument matches any of the whitelisted names
  wlisted=0
  for i in "${whitelist[@]}"; do
    if [ "$i" = "$f" ]; then
      wlisted=1
      break
    fi
  done
  if [ "$wlisted" = 1 ]; then
    echo "$f" is whitelisted and not checked
    continue
  fi

  # Check copyright header
  CP=`head -3 "$f" | grep "Copyright (c) \([0-9]\{4\}-\)\?$current_year"`
  if [ "$?" -ne 0 ]; then
    # Notify problem
    echo \>\>\>"$f" $message
    let nfailed++
  else
    # print OK
    prn "$f" "$CP"
  fi

done

exit $nfailed
