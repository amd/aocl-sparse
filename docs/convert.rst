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

Conversion Functions
********************

``aoclsparse_convert.h`` provides sparse format conversion functions.

aoclsparse_csr2ell_width()
--------------------------

.. doxygenfunction:: aoclsparse_csr2ell_width
   :project: sparse

aoclsparse\_?csr2ell()
----------------------

.. doxygenfunction:: aoclsparse_scsr2ell
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2ell
   :project: sparse

aoclsparse_csr2dia_ndiag()
--------------------------

.. doxygenfunction:: aoclsparse_csr2dia_ndiag
   :project: sparse

aoclsparse\_?csr2dia()
----------------------

.. doxygenfunction:: aoclsparse_scsr2dia
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2dia
   :project: sparse

aoclsparse_csr2bsr_nnz()
------------------------

.. doxygenfunction:: aoclsparse_csr2bsr_nnz
   :project: sparse

aoclsparse\_?csr2bsr()
----------------------

.. doxygenfunction:: aoclsparse_scsr2bsr
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2bsr
   :project: sparse

aoclsparse\_?csr2csc()
----------------------

.. doxygenfunction:: aoclsparse_scsr2csc
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2csc
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ccsr2csc
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zcsr2csc
   :project: sparse

aoclsparse\_?csr2dense()
------------------------

.. doxygenfunction:: aoclsparse_scsr2dense
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2dense
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_ccsr2dense
    :project: sparse
    :outline:
.. doxygenfunction:: aoclsparse_zcsr2dense
   :project: sparse

aoclsparse_convert_csr()
------------------------

.. doxygenfunction:: aoclsparse_convert_csr
   :project: sparse

aoclsparse_convert_bsr()
------------------------

.. doxygenfunction:: aoclsparse_convert_bsr
   :project: sparse
..
   removed from doc in 4.2
    .. doxygenfunction:: aoclsparse_csr2ellthyb_width
       :project: sparse
    .. doxygenfunction:: aoclsparse_scsr2ellt
       :project: sparse
    .. doxygenfunction:: aoclsparse_dcsr2ellt
       :project: sparse
    .. doxygenfunction:: aoclsparse_scsr2ellthyb
       :project: sparse
    .. doxygenfunction:: aoclsparse_dcsr2ellthyb
       :project: sparse
    .. doxygenfunction:: aoclsparse_opt_blksize
       :project: sparse
    .. doxygenfunction:: aoclsparse_csr2blkcsr
       :project: sparse
