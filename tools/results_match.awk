#!/usr/bin/gawk -f
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
# call results_match.awk ref_result new_results and it will collect the matching lines
BEGIN {
  FS = ","
  OFS = ","
  }

/^\*/ && FNR==NR { # first file is reference, only * lines have valid data
  key=substr($1,3) # skip "* "
  $1=""
  #print "REF key " key "line" $0;
  ref[key]=", " $3 ", " $4 ", " $5  # mean stdev n
  next;
}
/^\*/ { # only these lines have valid data
  key=substr($1,3) # skip "* "
  if (key in ref) {
    #print "FOUND  " key
    #print "   ref " ref[key]
    #print "   now " $0
    print key ref[key] ", " $3 ", " $4 ", " $5
  }
  else
    print key ", ref results NOT FOUND"
}
