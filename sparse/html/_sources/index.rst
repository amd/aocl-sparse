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

.. AOCL-Sparse documentation master file

AMD AOCL-Sparse Library Documentation
*************************************

The AMD Optimized CPU Library AOCL-Sparse is a library that contains Basic Linear
Algebra Subroutines for sparse matrices and vectors (Sparse BLAS) and is
optimized for AMD EPYC and RYZEN family of CPU processors. It implements numerical
algorithms in C++ while providing a public-facing C interface so it is can be used
with C, C++ and compatible languages.

The current functionality of AOCL-Sparse is organized in the following categories:

* **Sparse Level 1** functions perform vector operations such as dot product, vector
  additions on sparse vectors, gather, scatter, and other similar operations.
* **Sparse Level 2** functions describe the operations between a matrix in a sparse
  format and a vector in the dense format, including matrix-vector product (SpMV),
  triangular solve (TRSV) and similar.
* **Sparse Level 3** functions describe the operations between a matrix in a sparse
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
* **Hint & Optimize framework** to accelerate supported functions by a prior matrix
  analysis based on users' hints of expected operations.


.. toctree::
   :maxdepth: 1
   :caption: AOCL APIs

   analysis
   auxiliary
   convert
   functions
   solvers
   types



Search the documentation
========================

* :ref:`genindex`
* :ref:`search`



