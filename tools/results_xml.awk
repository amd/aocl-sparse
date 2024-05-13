#!/usr/bin/gawk -f
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# merged results passed through python to compute stats
# this will generate XML which can be digested by Jenkins
#call results_xml.awk [file or pipe it]
BEGIN {
  FS = ","
  print "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<testsuite name=\"(empty)\" >"
  inprogress = 0
  }
END {
  if (inprogress) {
    print "    </system-out>"
    print "  </testcase>"
  }
  print "</testsuite>"
  }

/^\*/ {
  if (inprogress) {
    print "    </system-out>"
    print "  </testcase>"
  }
  inprogress = 1

   key = $1
   sub(/^\* /,"",key)
   sub(/^aocl:/,"",key)
   gsub(/ /,"",key)
   gsub(/:/,".",key)
   if ($2 == "same")
     result="run"
   else
     result="fail"
   #print "   " key " result " result
   print "  <testcase name=\"" key "\" classname=\"" key "\" time=\"0.\" status=\"" result "\">"
   if (result == "fail")
     print "    <failure message=\"\"/>"
   print "    <system-out>"
   next;
   #print $0

  }
  {
    print
 }
