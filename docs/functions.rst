..
   Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_daxpyi
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_caxpyi
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zaxpyi
   :project: sparse

aoclsparse\_?dotci()
--------------------

.. doxygenfunction:: aoclsparse_cdotci
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zdotci
   :project: sparse

aoclsparse\_?dotui()
--------------------

.. doxygenfunction:: aoclsparse_cdotui
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zdotui
   :project: sparse

aoclsparse\_?doti()
-------------------

.. doxygenfunction:: aoclsparse_sdoti
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ddoti
   :project: sparse

aoclsparse\_?sctr()
-------------------

.. doxygenfunction:: aoclsparse_ssctr
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dsctr
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_csctr
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zsctr
   :project: sparse

sparse\_?sctrs()
----------------

.. doxygenfunction:: aoclsparse_ssctrs
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dsctrs
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_csctrs
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zsctrs
   :project: sparse

aoclsparse\_?roti()
-------------------

.. doxygenfunction:: aoclsparse_sroti
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_droti
   :project: sparse

aoclsparse\_?gthr()
-------------------

.. doxygenfunction:: aoclsparse_sgthr
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dgthr
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_cgthr
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zgthr
   :project: sparse

aoclsparse\_?gthrz()
--------------------

.. doxygenfunction:: aoclsparse_sgthrz
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dgthrz
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_cgthrz
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zgthrz
   :project: sparse

aoclsparse\_?gthrs()
--------------------

.. doxygenfunction:: aoclsparse_sgthrs
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dgthrs
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_cgthrs
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zgthrs
   :project: sparse

Level 2
=======

This module holds all sparse level 2 routines.

The sparse level 2 routines describe operations between a matrix in sparse
format and a vector in dense or sparse format.

aoclsparse\_?mv()
-----------------

.. doxygenfunction:: aoclsparse_smv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dmv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_cmv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zmv
   :project: sparse

aoclsparse::mv()
-----------------

.. doxygenfunction:: aoclsparse::mv
    :project: sparse

aoclsparse\_?trsv()
-------------------

.. doxygenfunction:: aoclsparse_strsv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dtrsv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ctrsv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ztrsv
   :project: sparse

.. doxygenfunction:: aoclsparse_strsv_strided
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dtrsv_strided
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ctrsv_strided
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ztrsv_strided
   :project: sparse

.. doxygenfunction:: aoclsparse_strsv_kid
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dtrsv_kid
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ctrsv_kid
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ztrsv_kid
   :project: sparse

aoclsparse::trsv()
------------------

.. doxygenfunction:: aoclsparse::trsv
    :project: sparse

aoclsparse\_?dotmv()
--------------------

.. doxygenfunction:: aoclsparse_sdotmv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ddotmv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_cdotmv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zdotmv
   :project: sparse

aoclsparse\_?ellmv()
--------------------

.. doxygenfunction:: aoclsparse_sellmv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dellmv
   :project: sparse

aoclsparse\_?diamv()
--------------------

.. doxygenfunction:: aoclsparse_sdiamv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ddiamv
   :project: sparse

aoclsparse\_?bsrmv()
--------------------

.. doxygenfunction:: aoclsparse_sbsrmv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dbsrmv
   :project: sparse

.. Mark for deprecation

aoclsparse\_?csrmv()
--------------------

.. doxygenfunction:: aoclsparse_scsrmv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dcsrmv
   :project: sparse

aoclsparse\_?csrsv()
--------------------

.. doxygenfunction:: aoclsparse_scsrsv
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dcsrsv
   :project: sparse


Level 3
=======

This module holds all sparse level 3 routines.

The sparse level 3 routines describe operations between matrices.

aoclsparse\_?trsm()
-------------------

.. doxygenfunction:: aoclsparse_strsm
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dtrsm
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ctrsm
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ztrsm
   :project: sparse

.. doxygenfunction:: aoclsparse_strsm_kid
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dtrsm_kid
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ctrsm_kid
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ztrsm_kid
   :project: sparse

aoclsparse_sp2m()
-----------------

.. doxygenfunction:: aoclsparse_sp2m
   :project: sparse

aoclsparse::sp2m()
------------------

.. doxygenfunction:: aoclsparse::sp2m
    :project: sparse

aoclsparse_spmm()
-----------------

.. doxygenfunction:: aoclsparse_spmm
   :project: sparse

aoclsparse\_?csrmm()
--------------------

.. doxygenfunction:: aoclsparse_scsrmm
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dcsrmm
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ccsrmm
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zcsrmm
   :project: sparse

aoclsparse\_?csr2m()
--------------------

.. doxygenfunction:: aoclsparse_dcsr2m
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_scsr2m
   :project: sparse

aoclsparse\_?add()
------------------

.. doxygenfunction:: aoclsparse_sadd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dadd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_cadd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zadd
   :project: sparse

aoclsparse\_?spmmd()
--------------------

.. doxygenfunction:: aoclsparse_sspmmd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dspmmd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_cspmmd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zspmmd
   :project: sparse

aoclsparse\_?sp2md()
--------------------

.. doxygenfunction:: aoclsparse_ssp2md
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dsp2md
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_csp2md
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zsp2md
   :project: sparse

aoclsparse_syrk()
--------------------

.. doxygenfunction:: aoclsparse_syrk
   :project: sparse

aoclsparse\_?syrkd()
--------------------

.. doxygenfunction:: aoclsparse_ssyrkd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dsyrkd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_csyrkd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zsyrkd
   :project: sparse

aoclsparse\_?sypr()
--------------------

.. doxygenfunction:: aoclsparse_sypr
   :project: sparse

aoclsparse\_?syprd()
--------------------

.. doxygenfunction:: aoclsparse_ssyprd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dsyprd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_csyprd
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zsyprd
   :project: sparse

