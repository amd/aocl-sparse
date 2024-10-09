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


# List of sources that need to compiled only once with the
# library's base compilation flag (AVX2)
# ----------------------------------------------------------
set(AOCLSPARSE_SOURCES
# Level 1 APIs
  src/level1/aoclsparse_roti.cpp
  src/level1/aoclsparse_axpyi.cpp
  src/level1/aoclsparse_dot.cpp
  src/level1/aoclsparse_gthr.cpp
  src/level1/aoclsparse_sctr.cpp

  src/level2/aoclsparse_trsv.cpp
  src/level2/aoclsparse_csrmv.cpp
  src/level2/aoclsparse_blkcsrmv.cpp
  src/level2/aoclsparse_ellmv.cpp
  src/level2/aoclsparse_mv.cpp
  src/level2/aoclsparse_diamv.cpp
  src/level2/aoclsparse_bsrmv.cpp
  src/level2/aoclsparse_csrsv.cpp
  src/level2/aoclsparse_dotmv.cpp
  src/level3/aoclsparse_csradd.cpp
  src/level2/aoclsparse_tcsrmv.cpp
  src/level3/aoclsparse_csrmm.cpp
  src/level3/aoclsparse_csr2m.cpp
  src/level3/aoclsparse_sp2m.cpp
  src/level3/aoclsparse_spmm.cpp
  src/level3/aoclsparse_spmmd.cpp
  src/level3/aoclsparse_sp2md.cpp
  src/level3/aoclsparse_syrkd.cpp
  src/level3/aoclsparse_syprd.cpp
  src/level3/aoclsparse_sypr.cpp
  src/level3/aoclsparse_syrk.cpp
  src/level3/aoclsparse_trsm.cpp

# Solvers
  src/solvers/aoclsparse_ilu.cpp
  src/solvers/aoclsparse_sorv.cpp
  src/solvers/aoclsparse_itsol_functions.cpp
  src/solvers/aoclsparse_symgs.cpp

# Conversion
  src/conversion/aoclsparse_convert.cpp

# Extra
  src/extra/aoclsparse_auxiliary.cpp
  src/extra/aoclsparse_context.cpp

# Analysis
  src/analysis/aoclsparse_analysis.cpp
  src/analysis/aoclsparse_csr_util.cpp
)

# List of sources that need to compiled with AVX512
# -------------------------------------------------
set(AVX512_SOURCES
  src/level2/aoclsparse_blkcsrmv_avx512.cpp
  src/level2/aoclsparse_csrmv_avx512.cpp
  src/level2/aoclsparse_ellmv_avx512.cpp
)

set(DOUBLE_COMPILATION
  src/level1/aoclsparse_dense_dot_kt.cpp
  src/level1/aoclsparse_roti_kt.cpp
  src/level1/aoclsparse_axpyi_kt.cpp
  src/level1/aoclsparse_dot_kt.cpp
  src/level1/aoclsparse_gthr_kt.cpp
  src/level1/aoclsparse_sctr_kt.cpp

  src/level2/aoclsparse_csrmv_kt.cpp
  src/level2/aoclsparse_trsv_kt.cpp
)

include(Suppressions)
