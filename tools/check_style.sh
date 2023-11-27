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
# Check if files have the correct style
# Style guide
# * examples should not have #if 0 directives

# Whitelist of files which are considered always OK
whitelist=()

if [ "$#" -eq 0 ]; then
  print_help
  echo ""
  echo "Testing all staged files"
  git status --porcelain | grep '^[AM]' | cut -c4- | xargs --no-run-if-empty bash $0 $PRINT
  exit
fi

nfailed=0

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

  # If example, check style
  BASE=`basename $f`
  if [ ${BASE%%_*} == "sample" ] ; then
    LOG=`grep -n '^[[:blank:]]*#if[[:blank:]]*[[:digit:]]' $f`
    if [ ! -z "$LOG" ] ; then
      echo \>\>\>"STYLE_ERROR:01:illegal directive found in $f:" $LOG
      let nfailed++
    fi
    LOG=`grep -n '^[[:blank:]]*//.*cout.*<<' $f`
    if [ ! -z "$LOG" ] ; then
      echo \>\>\>"STYLE_ERROR:02:commented out cout directives found in $f:" $LOG
      let nfailed++
    fi
  fi

  # Check for comments that break style:
  # a = b - 10; // very long comment will get formated as:
  # a
  #   = b
  #     - 10; // very long comment will get formated as:
  LOG=`grep -n '^[[:blank:]]*[=+-].*/[/\*]' $f`
  if [ ! -z "$LOG" ] ; then
    echo \>\>\>"STYLE_ERROR:03:line-comment-breaks-polisher found in $f:" $LOG
    let nfailed++
  fi

  # Check for DOS line ending
  LOG=`dos2unix --info=c $f`
  if [ ! -z "$LOG" ] ; then
    echo \>\>\>"STYLE_ERROR:04:DOS-line-ending found in $f"
    let nfailed++
  fi

done

exit $nfailed

