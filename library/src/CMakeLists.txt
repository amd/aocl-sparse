# ########################################################################
# Copyright (c) 2020-2024 Advanced Micro Devices, Inc.
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

set(AOCLSPARSE_SOURCES
  src/level1/aoclsparse_roti.cpp
  src/level1/aoclsparse_dot.cpp
  src/level1/aoclsparse_gthr.cpp
  src/level1/aoclsparse_sctr.cpp
  src/level1/aoclsparse_axpyi.cpp
  src/level2/aoclsparse_csrmv.cpp
  src/level2/aoclsparse_blkcsrmv_api.cpp
  src/level2/aoclsparse_blkcsrmv.cpp
  src/level2/aoclsparse_ellmv.cpp
  src/level2/aoclsparse_mv.cpp
  src/level2/aoclsparse_diamv.cpp
  src/level2/aoclsparse_bsrmv.cpp
  src/level2/aoclsparse_csrsv.cpp
  src/level2/aoclsparse_trsv.cpp
  src/level2/aoclsparse_dotmv.cpp
  src/level3/aoclsparse_csrmm.cpp
  src/level3/aoclsparse_csr2m.cpp
  src/level3/aoclsparse_sp2m.cpp
  src/level3/aoclsparse_spmm.cpp
  src/level3/aoclsparse_csradd.cpp
  src/level3/aoclsparse_trsm.cpp
  src/level3/aoclsparse_spmmd.cpp
  src/level3/aoclsparse_sp2md.cpp
  src/level3/aoclsparse_syprd.cpp

# Conversion
  src/conversion/aoclsparse_convert.cpp

# Extra
  src/extra/aoclsparse_auxiliary.cpp
  src/extra/aoclsparse_context.cpp

# Solvers
  src/solvers/aoclsparse_ilu.cpp
  src/solvers/aoclsparse_itsol_functions.cpp

# Analysis
  src/analysis/aoclsparse_analysis.cpp
  src/analysis/aoclsparse_csr_util.cpp
)

include(Suppressions)
