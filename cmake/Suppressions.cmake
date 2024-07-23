# ########################################################################
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

message(STATUS "Adding list of suppressions.")

# Suppressions list
# ================
# This file adds a list of source files with required suppression flags

# Suppressions
# ============

# warning: self-comparison always evaluates to false [-Wtautological-compare]
# Tautological-compares are correct for these instances: the comparison is to
# check input for NaNs, since if x=NaN, then x!=x always.
set_source_files_properties(src/solvers/aoclsparse_itsol_functions.cpp PROPERTIES COMPILE_FLAGS -Wno-tautological-compare)

# warning: unused parameter context [-Wunused-parameter]
# These parameters in the interface are warranted for reasons
# such has compatibility or similar valid reasons.

# warning: this statement may fall through [-Wimplicit-fallthrough=]
# Since there are occasions where a switch case fall through is desirable,
# GCC provides an attribute, "__attribute__ ((fallthrough))", that is to be
# used along with a null statement to suppress this warning that would
# normally occur:
#                    switch (cond)
#                      {
#                      case 1:
#                        bar (0);
#                        __attribute__ ((fallthrough));
#                      default:
#                        ...
#                      }

