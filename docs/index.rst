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

.. AOCL-Sparse documentation master file

AOCL-Sparse
*************************************

.. _MainIntro:

Introduction
------------

The AMD Optimized CPU Library AOCL-Sparse is a library that contains Basic Linear
Algebra Subroutines for sparse matrices and vectors (Sparse BLAS) and is
optimized for AMD EPYC and RYZEN family of CPU processors. It implements numerical
algorithms in C++ while providing a public-facing C interface so it is can be used
with C, C++ and compatible languages.

The current functionality of AOCL-Sparse is organized in the following categories:

* **Sparse level 1** functions perform vector operations such as dot product, vector
  additions on sparse vectors, gather, scatter, and other similar operations.
* **Sparse level 2** functions describe the operations between a matrix in a sparse
  format and a vector in the dense format, including matrix-vector product (SpMV),
  triangular solve (TRSV) and similar.
* **Sparse level 3** functions describe the operations between a matrix in a sparse
  format and one or more dense/sparse matrices. The operations comprise of matrix
  additions (SpADD), matrix-matrix product (SpMM, Sp2M), and triangular solver with
  multiple right-hand sides (TRSM).
* **Iterative sparse solvers** based on Krylov subspace methods (CGM, GMRES) and
  preconditioners (such as, SymGS, ILU0).
* Sparse format conversion functions for translating matrices in a variety of
  sparse storage formats.
* Auxiliary functions to allow basic operations, including create, copy, destroy
  and modify matrix handles and descriptors.

Additional highlights:

* Supported data types: single, double, and the complex variants
* 0-based and 1-based indexing of sparse formats
* **Hint and optimize framework** to accelerate supported functions by a prior matrix
  analysis based on users' hints of expected operations.

.. _NamingConvention:

Naming convention
-----------------

API's in the library are formed by three sections: :code:`aoclsparse` prefix,
:code:`P` data type precision, followed by an abbreviated form of the functionality.
Data type precion :code:`P` is a single letter indicating:
:code:`s` single,
:code:`d` double,
:code:`c` complex single, and
:code:`z` complex double floating point.
Some illustrative examples follow.

.. csv-table:: API naming convention examples
    :header: "API", "Precision" :code:`P`, "Functionality"
    :widths: auto
    :align: left
    :escape: @

    :cpp:func:`aoclsparse_strsv()`,  :code:`s`, :code:`TRSV` single precision linear system of equations TRiangular SolVer@,
    :cpp:func:`aoclsparse_daxpyi()`, :code:`d`, :code:`AXPY` perform a variant of the operation :math:`a\@,x+y` in double precision@,
    :cpp:func:`aoclsparse_cmv()`,    :code:`c`, :code:`SPMV` sparse matrix-vector product using complex single precision@,
    :cpp:func:`aoclsparse_ztrsm()`,  :code:`z`, :code:`TRSM` complex double precision linear system of equations TRiangular Solver with Multiple right-hand sides.


Throughout this document and where not ambiguous, if an API supports two or more data types described above, then it will be
indicated by :code:`?` (question mark) in place of the data type single-letter abbreviation.
As an example, :cpp:func:`aoclsparse_?trsv() <aoclsparse_strsv>` references all supported data types for the TRSV solver, that is,
:cpp:func:`aoclsparse_strsv`, :cpp:func:`aoclsparse_dtrsv`, :cpp:func:`aoclsparse_ctrsv`, and :cpp:func:`aoclsparse_ztrsv`;
while :cpp:func:`aoclsparse_?dotci() <aoclsparse_cdotci>` references only :cpp:func:`aoclsparse_cdotci`, and :cpp:func:`aoclsparse_zdotci`.

.. toctree::
   :maxdepth: 2
   :caption: Functionality API

   analysis
   auxiliary
   convert
   functions
   solvers
   types

.. toctree::
   :maxdepth: 2
   :caption: Details

   storage

.. a section on parallelization (OMP?)

Search the documentation
========================

* :ref:`genindex`
* :ref:`search`



