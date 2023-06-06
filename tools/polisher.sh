#!/bin/bash
# Polish all [hc][++] files uses the current directory for locating
# the style file, .clang-format. So this script should be invoked
# from the root of the project.

if [ ! -f .clang-format ] ; then
  echo Can not find in `pwd` the style file .clang-format.
  echo This script should be called from the project root.
  exit 1
fi

find library tests \( -iname "*.c" -o -iname "*.cpp" -o -iname "*.h" -o -iname "*.hpp" \) -type f -print -exec clang-format --style="file:.clang-format" -i '{}' \;

