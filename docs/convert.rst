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

Conversion Functions
********************

``aoclsparse_convert.h`` provides sparse format conversion functions.

aoclsparse_csr2ell_width()
--------------------------

.. doxygenfunction:: aoclsparse_csr2ell_width

aoclsparse\_?csr2ell()
----------------------

.. doxygenfunction:: aoclsparse_scsr2ell
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2ell

aoclsparse_csr2dia_ndiag()
--------------------------

.. doxygenfunction:: aoclsparse_csr2dia_ndiag

aoclsparse\_?csr2dia()
----------------------

.. doxygenfunction:: aoclsparse_scsr2dia
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2dia

aoclsparse_csr2bsr_nnz()
------------------------

.. doxygenfunction:: aoclsparse_csr2bsr_nnz

aoclsparse\_?csr2bsr()
----------------------

.. doxygenfunction:: aoclsparse_scsr2bsr
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2bsr

aoclsparse\_?csr2csc()
----------------------

.. doxygenfunction:: aoclsparse_scsr2csc
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2csc
    :outline:
.. doxygenfunction:: aoclsparse_ccsr2csc
    :outline:
.. doxygenfunction:: aoclsparse_zcsr2csc

aoclsparse\_?csr2dense()
------------------------

.. doxygenfunction:: aoclsparse_scsr2dense
    :outline:
.. doxygenfunction:: aoclsparse_dcsr2dense
    :outline:
.. doxygenfunction:: aoclsparse_ccsr2dense
    :outline:
.. doxygenfunction:: aoclsparse_zcsr2dense

aoclsparse_convert_csr()
------------------------

.. doxygenfunction:: aoclsparse_convert_csr

..
   removed from doc in 4.2
    .. doxygenfunction:: aoclsparse_csr2ellthyb_width
    .. doxygenfunction:: aoclsparse_scsr2ellt
    .. doxygenfunction:: aoclsparse_dcsr2ellt
    .. doxygenfunction:: aoclsparse_scsr2ellthyb
    .. doxygenfunction:: aoclsparse_dcsr2ellthyb
    .. doxygenfunction:: aoclsparse_opt_blksize
    .. doxygenfunction:: aoclsparse_csr2blkcsr
