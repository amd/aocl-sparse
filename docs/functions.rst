..
   Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

Sparse BLAS level 1, 2, and 3 functions
***************************************

``aoclsparse_functions.h`` provides AMD CPU hardware optimized level 1, 2, and 3
Sparse Linear Algebra Subprograms (Sparse BLAS).

Level 1
=======

The sparse level 1 routines describe operations between a vector in sparse
format and a vector in dense format.

This section describes all provided level 1 sparse linear algebra functions.

aoclsparse\_?axpyi()
--------------------

.. doxygenfunction:: aoclsparse_saxpyi
    :outline:
.. doxygenfunction:: aoclsparse_daxpyi
    :outline:
.. doxygenfunction:: aoclsparse_caxpyi
    :outline:
.. doxygenfunction:: aoclsparse_zaxpyi

aoclsparse\_?dotci()
--------------------

.. doxygenfunction:: aoclsparse_cdotci
    :outline:
.. doxygenfunction:: aoclsparse_zdotci

aoclsparse\_?dotui()
--------------------

.. doxygenfunction:: aoclsparse_cdotui
    :outline:
.. doxygenfunction:: aoclsparse_zdotui

aoclsparse\_?doti()
-------------------

.. doxygenfunction:: aoclsparse_sdoti
    :outline:
.. doxygenfunction:: aoclsparse_ddoti

aoclsparse\_?sctr()
-------------------

.. doxygenfunction:: aoclsparse_ssctr
    :outline:
.. doxygenfunction:: aoclsparse_dsctr
    :outline:
.. doxygenfunction:: aoclsparse_csctr
    :outline:
.. doxygenfunction:: aoclsparse_zsctr

sparse\_?sctrs()
----------------

.. doxygenfunction:: aoclsparse_ssctrs
    :outline:
.. doxygenfunction:: aoclsparse_dsctrs
    :outline:
.. doxygenfunction:: aoclsparse_csctrs
    :outline:
.. doxygenfunction:: aoclsparse_zsctrs

aoclsparse\_?roti()
-------------------

.. doxygenfunction:: aoclsparse_sroti
    :outline:
.. doxygenfunction:: aoclsparse_droti

aoclsparse\_?gthr()
-------------------

.. doxygenfunction:: aoclsparse_sgthr
    :outline:
.. doxygenfunction:: aoclsparse_dgthr
    :outline:
.. doxygenfunction:: aoclsparse_cgthr
    :outline:
.. doxygenfunction:: aoclsparse_zgthr

aoclsparse\_?gthrz()
--------------------

.. doxygenfunction:: aoclsparse_sgthrz
    :outline:
.. doxygenfunction:: aoclsparse_dgthrz
    :outline:
.. doxygenfunction:: aoclsparse_cgthrz
    :outline:
.. doxygenfunction:: aoclsparse_zgthrz

aoclsparse\_?gthrs()
--------------------

.. doxygenfunction:: aoclsparse_sgthrs
    :outline:
.. doxygenfunction:: aoclsparse_dgthrs
    :outline:
.. doxygenfunction:: aoclsparse_cgthrs
    :outline:
.. doxygenfunction:: aoclsparse_zgthrs

Level 2
=======

This module holds all sparse level 2 routines.

The sparse level 2 routines describe operations between a matrix in sparse
format and a vector in dense or sparse format.

aoclsparse\_?mv()
-----------------

.. doxygenfunction:: aoclsparse_smv
    :outline:
.. doxygenfunction:: aoclsparse_dmv
    :outline:
.. doxygenfunction:: aoclsparse_cmv
    :outline:
.. doxygenfunction:: aoclsparse_zmv

aoclsparse\_?trsv()
-------------------

.. doxygenfunction:: aoclsparse_strsv
    :outline:
.. doxygenfunction:: aoclsparse_dtrsv
    :outline:
.. doxygenfunction:: aoclsparse_ctrsv
    :outline:
.. doxygenfunction:: aoclsparse_ztrsv

.. doxygenfunction:: aoclsparse_strsv_kid
    :outline:
.. doxygenfunction:: aoclsparse_dtrsv_kid
    :outline:
.. doxygenfunction:: aoclsparse_ctrsv_kid
    :outline:
.. doxygenfunction:: aoclsparse_ztrsv_kid

aoclsparse\_?dotmv()
--------------------

.. doxygenfunction:: aoclsparse_sdotmv
    :outline:
.. doxygenfunction:: aoclsparse_ddotmv
    :outline:
.. doxygenfunction:: aoclsparse_cdotmv
    :outline:
.. doxygenfunction:: aoclsparse_zdotmv


aoclsparse\_?ellmv()
--------------------

.. doxygenfunction:: aoclsparse_sellmv
    :outline:
.. doxygenfunction:: aoclsparse_dellmv

aoclsparse\_?diamv()
--------------------

.. doxygenfunction:: aoclsparse_sdiamv
    :outline:
.. doxygenfunction:: aoclsparse_ddiamv

aoclsparse\_?bsrmv()
--------------------

.. doxygenfunction:: aoclsparse_sbsrmv
    :outline:
.. doxygenfunction:: aoclsparse_dbsrmv

.. Mark for deprecation

aoclsparse\_?csrmv()
--------------------

.. doxygenfunction:: aoclsparse_scsrmv
    :outline:
.. doxygenfunction:: aoclsparse_dcsrmv

aoclsparse\_?csrsv()
--------------------

.. doxygenfunction:: aoclsparse_scsrsv
    :outline:
.. doxygenfunction:: aoclsparse_dcsrsv


Level 3
=======

This module holds all sparse level 3 routines.

The sparse level 3 routines describe operations between matrices.

aoclsparse\_?trsm()
-------------------

.. doxygenfunction:: aoclsparse_strsm
    :outline:
.. doxygenfunction:: aoclsparse_dtrsm
    :outline:
.. doxygenfunction:: aoclsparse_ctrsm
    :outline:
.. doxygenfunction:: aoclsparse_ztrsm

.. doxygenfunction:: aoclsparse_strsm_kid
    :outline:
.. doxygenfunction:: aoclsparse_dtrsm_kid
    :outline:
.. doxygenfunction:: aoclsparse_ctrsm_kid
    :outline:
.. doxygenfunction:: aoclsparse_ztrsm_kid

aoclsparse_sp2m()
-----------------

.. doxygenfunction:: aoclsparse_sp2m

aoclsparse_spmm()
-----------------

.. doxygenfunction:: aoclsparse_spmm

aoclsparse\_?csrmm()
--------------------

.. doxygenfunction:: aoclsparse_scsrmm
    :outline:
.. doxygenfunction:: aoclsparse_dcsrmm
    :outline:
.. doxygenfunction:: aoclsparse_ccsrmm
    :outline:
.. doxygenfunction:: aoclsparse_zcsrmm

aoclsparse\_?csr2m()
--------------------

.. doxygenfunction:: aoclsparse_dcsr2m
    :outline:
.. doxygenfunction:: aoclsparse_scsr2m

aoclsparse\_?add()
------------------

.. doxygenfunction:: aoclsparse_sadd
    :outline:
.. doxygenfunction:: aoclsparse_dadd
    :outline:
.. doxygenfunction:: aoclsparse_cadd
    :outline:
.. doxygenfunction:: aoclsparse_zadd

aoclsparse\_?spmmd()
--------------------

.. doxygenfunction:: aoclsparse_sspmmd
    :outline:
.. doxygenfunction:: aoclsparse_dspmmd
    :outline:
.. doxygenfunction:: aoclsparse_cspmmd
    :outline:
.. doxygenfunction:: aoclsparse_zspmmd

aoclsparse\_?sp2md()
--------------------

.. doxygenfunction:: aoclsparse_ssp2md
    :outline:
.. doxygenfunction:: aoclsparse_dsp2md
    :outline:
.. doxygenfunction:: aoclsparse_csp2md
    :outline:
.. doxygenfunction:: aoclsparse_zsp2md

aoclsparse_syrk()
--------------------

.. doxygenfunction:: aoclsparse_syrk

aoclsparse\_?syrkd()
--------------------

.. doxygenfunction:: aoclsparse_ssyrkd
    :outline:
.. doxygenfunction:: aoclsparse_dsyrkd
    :outline:
.. doxygenfunction:: aoclsparse_csyrkd
    :outline:
.. doxygenfunction:: aoclsparse_zsyrkd

aoclsparse\_?sypr()
--------------------

.. doxygenfunction:: aoclsparse_sypr

aoclsparse\_?syprd()
--------------------

.. doxygenfunction:: aoclsparse_ssyprd
    :outline:
.. doxygenfunction:: aoclsparse_dsyprd
    :outline:
.. doxygenfunction:: aoclsparse_csyprd
    :outline:
.. doxygenfunction:: aoclsparse_zsyprd

Miscellaneous
=============

aoclsparse_ilu\_?smoother()
---------------------------

.. doxygenfunction:: aoclsparse_silu_smoother
    :outline:
.. doxygenfunction:: aoclsparse_dilu_smoother

..
   removed from doc in 4.2
   .. doxygenfunction:: aoclsparse_selltmv
   .. doxygenfunction:: aoclsparse_delltmv
   .. doxygenfunction:: aoclsparse_sellthybmv
   .. doxygenfunction:: aoclsparse_dellthybmv
   .. doxygenfunction:: aoclsparse_dblkcsrmv
