.. 
   Copyright (c) 2023 Advanced Micro Devices, Inc.
..
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
..
   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
..
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

AOCL-Sparse Level 1,2,3 Functions
*********************************

``aoclsparse_functions.h`` provides AMD CPU hardware optimized level 1, 2, and 3 Sparse Linear Algebra Subprograms (Sparse BLAS).

Level 1
=======

.. doxygenfunction:: aoclsparse_saxpyi
.. doxygenfunction:: aoclsparse_daxpyi
.. doxygenfunction:: aoclsparse_caxpyi
.. doxygenfunction:: aoclsparse_zaxpyi
.. doxygenfunction:: aoclsparse_cdotci
.. doxygenfunction:: aoclsparse_zdotci
.. doxygenfunction:: aoclsparse_cdotui
.. doxygenfunction:: aoclsparse_zdotui
.. doxygenfunction:: aoclsparse_sdoti
.. doxygenfunction:: aoclsparse_ddoti
.. doxygenfunction:: aoclsparse_ssctr
.. doxygenfunction:: aoclsparse_dsctr
.. doxygenfunction:: aoclsparse_csctr
.. doxygenfunction:: aoclsparse_zsctr
.. doxygenfunction:: aoclsparse_ssctrs
.. doxygenfunction:: aoclsparse_dsctrs
.. doxygenfunction:: aoclsparse_csctrs
.. doxygenfunction:: aoclsparse_zsctrs
.. doxygenfunction:: aoclsparse_sroti
.. doxygenfunction:: aoclsparse_droti
.. doxygenfunction:: aoclsparse_sgthr
.. doxygenfunction:: aoclsparse_dgthr
.. doxygenfunction:: aoclsparse_cgthr
.. doxygenfunction:: aoclsparse_zgthr
.. doxygenfunction:: aoclsparse_sgthrz
.. doxygenfunction:: aoclsparse_dgthrz
.. doxygenfunction:: aoclsparse_cgthrz
.. doxygenfunction:: aoclsparse_zgthrz
.. doxygenfunction:: aoclsparse_sgthrs
.. doxygenfunction:: aoclsparse_dgthrs
.. doxygenfunction:: aoclsparse_cgthrs
.. doxygenfunction:: aoclsparse_zgthrs

Level 2
=======

.. doxygenfunction:: aoclsparse_scsrmv
.. doxygenfunction:: aoclsparse_dcsrmv
.. doxygenfunction:: aoclsparse_sellmv
.. doxygenfunction:: aoclsparse_dellmv
.. doxygenfunction:: aoclsparse_sdiamv
.. doxygenfunction:: aoclsparse_ddiamv
.. doxygenfunction:: aoclsparse_sbsrmv
.. doxygenfunction:: aoclsparse_dbsrmv
.. doxygenfunction:: aoclsparse_smv
.. doxygenfunction:: aoclsparse_dmv
.. doxygenfunction:: aoclsparse_cmv
.. doxygenfunction:: aoclsparse_zmv
.. doxygenfunction:: aoclsparse_scsrsv
.. doxygenfunction:: aoclsparse_dcsrsv
.. doxygenfunction:: aoclsparse_strsv
.. doxygenfunction:: aoclsparse_dtrsv
.. doxygenfunction:: aoclsparse_ctrsv
.. doxygenfunction:: aoclsparse_ztrsv
.. doxygenfunction:: aoclsparse_strsv_kid
.. doxygenfunction:: aoclsparse_dtrsv_kid
.. doxygenfunction:: aoclsparse_ctrsv_kid
.. doxygenfunction:: aoclsparse_ztrsv_kid
.. doxygenfunction:: aoclsparse_sdotmv
.. doxygenfunction:: aoclsparse_ddotmv
.. doxygenfunction:: aoclsparse_cdotmv
.. doxygenfunction:: aoclsparse_zdotmv

Level 3
=======

.. doxygenfunction:: aoclsparse_strsm
.. doxygenfunction:: aoclsparse_dtrsm
.. doxygenfunction:: aoclsparse_ctrsm
.. doxygenfunction:: aoclsparse_ztrsm
.. doxygenfunction:: aoclsparse_strsm_kid
.. doxygenfunction:: aoclsparse_dtrsm_kid
.. doxygenfunction:: aoclsparse_ctrsm_kid
.. doxygenfunction:: aoclsparse_ztrsm_kid
.. doxygenfunction:: aoclsparse_sp2m
.. doxygenfunction:: aoclsparse_spmm
.. doxygenfunction:: aoclsparse_scsrmm
.. doxygenfunction:: aoclsparse_dcsrmm
.. doxygenfunction:: aoclsparse_ccsrmm
.. doxygenfunction:: aoclsparse_zcsrmm
.. doxygenfunction:: aoclsparse_dcsr2m
.. doxygenfunction:: aoclsparse_scsr2m
.. doxygenfunction:: aoclsparse_sadd
.. doxygenfunction:: aoclsparse_dadd
.. doxygenfunction:: aoclsparse_cadd
.. doxygenfunction:: aoclsparse_zadd

Miscellaneous 
=============

.. doxygenfunction:: aoclsparse_dilu_smoother
.. doxygenfunction:: aoclsparse_silu_smoother

..
   removed from doc in 4.2
   .. doxygenfunction:: aoclsparse_selltmv
   .. doxygenfunction:: aoclsparse_delltmv
   .. doxygenfunction:: aoclsparse_sellthybmv
   .. doxygenfunction:: aoclsparse_dellthybmv
   .. doxygenfunction:: aoclsparse_dblkcsrmv

