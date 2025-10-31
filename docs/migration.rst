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

Migration Guide
***************

This section provides step-by-step instructions for migrating from Intel MKL Sparse BLAS [#intel_copyright]_
to AOCL-Sparse. It covers both traditional and inspector-executor interfaces with
mapping tables and examples.

The migration process involves understanding key AOCL-Sparse concepts:

* :cpp:type:`aoclsparse_matrix` - Matrix handle created via APIs like :cpp:func:`aoclsparse_create_?csr() <aoclsparse_create_scsr>`,
  :cpp:func:`aoclsparse_create_?csc() <aoclsparse_create_scsc>`, or :cpp:func:`aoclsparse_create_?coo() <aoclsparse_create_scoo>`.
  See :doc:`storage` for supported formats.

* :cpp:type:`aoclsparse_mat_descr` - Matrix descriptor created via :cpp:func:`aoclsparse_create_mat_descr`
  and configured using :cpp:func:`aoclsparse_set_mat_type`, :cpp:func:`aoclsparse_set_mat_fill_mode`,
  :cpp:func:`aoclsparse_set_mat_diag_type`, and :cpp:func:`aoclsparse_set_mat_index_base`.

* :cpp:type:`aoclsparse_operation` - Operation type (none, transpose, conjugate transpose).

* :cpp:type:`aoclsparse_index_base` - Index base (0-based or 1-based).

Porting MKL's Inspector-Executor APIs to AOCL-Sparse
-----------------------------------------------------

:numref:`table-mkl-ie` illustrates the mapping between MKL's inspector-executor (I/E)
and AOCL-Sparse's APIs. Most of these APIs have similar signatures.
Apart from minor differences like the one in create_csr (4 pointers vs.
3 pointers in the function arguments), porting from MKL to AOCL-Sparse
should be straightforward provided the entire MKL Sparse APIs are
replaced with the corresponding AOCL-Sparse APIs.

.. table:: Mapping MKL's inspector-executor APIs to AOCL-Sparse
   :name: table-mkl-ie

   .. csv-table::
      :header-rows: 1
      :widths: 4 4 5

      MKL,                         AOCL-Sparse,                Comments
      ``mkl_sparse_?_create_csr``, :cpp:func:`aoclsparse_create_?csr() <aoclsparse_create_scsr>`, "4 vs. 3 pointers, matrix structure"
      ``mkl_sparse_convert_csr``,  :cpp:func:`aoclsparse_convert_csr`,
      ``mkl_sparse_?_export_csr``, :cpp:func:`aoclsparse_export_?csr() <aoclsparse_export_scsr>`,
      ``mkl_sparse_order``,        :cpp:func:`aoclsparse_order_mat`,
      ``mkl_sparse_?_mv``,         :cpp:func:`aoclsparse_?mv() <aoclsparse_smv>`,
      ``mkl_sparse_?_trsv``,       :cpp:func:`aoclsparse_?trsv() <aoclsparse_strsv>`,
      ``mkl_sparse_sp2m``,         :cpp:func:`aoclsparse_sp2m`,
      ``mkl_sparse_?_mm``,         :cpp:func:`aoclsparse_?csrmm() <aoclsparse_scsrmm>`,     Works for general & symmetric matrices
      ``mkl_sparse_?_trsm``,       :cpp:func:`aoclsparse_?trsm() <aoclsparse_strsm>`,
      ``mkl_sparse_?_add``,        :cpp:func:`aoclsparse_?add() <aoclsparse_sadd>`,


Porting MKL's Traditional SPARSE BLAS APIs to AOCL-Sparse
----------------------------------------------------------

:numref:`table-mkl-traditional` shows the mapping between MKL's traditional Sparse BLAS and
AOCL-Sparse's APIs. Unlike the MKL's I/E APIs, the traditional APIs work
with pointers, without utilizing any abstract data structures. Some of
the APIs of interest are listed in the table with the corresponding
AOCL-Sparse API. Additionally, :numref:`table-mkl-params` lists the mappings of various
types (value of pointers) within MKL's traditional interface with
AOCL-Sparse types. This section describes the porting process with a few
illustrations.

.. table:: Mapping MKL's traditional APIs to AOCL-Sparse
   :name: table-mkl-traditional

   .. csv-table::
      :header-rows: 1
      :widths: 4 4 5

      MKL Old,                      AOCL-Sparse,                Comments
      ``mkl_?csrcsc``,               :cpp:func:`aoclsparse_convert_csr`,
      "| ``mkl_cspblas_dcsrgemv``
      | ``mkl_?csrmv``
      | ``mkl_?csrgemv``
      | ``mkl_?csrsymv``",               :cpp:func:`aoclsparse_?mv() <aoclsparse_smv>`,
      "| ``mkl_?csrtrsv``
      | ``mkl_?csrsv``",                 :cpp:func:`aoclsparse_?trsv() <aoclsparse_strsv>`,
      ``mkl_dcsrmultcsr``,           :cpp:func:`aoclsparse_sp2m`,
      ``mkl_dcsrmm``,                :cpp:func:`aoclsparse_?csrmm() <aoclsparse_scsrmm>`, "General, symmetric and Hermitian matrices supported"
      ``mkl_?csrsm``,                    :cpp:func:`aoclsparse_?trsm() <aoclsparse_strsm>`,
      ``mkl_dcsradd``,               :cpp:func:`aoclsparse_?add() <aoclsparse_sadd>`,
      ``mkl_sparse_destroy``,        :cpp:func:`aoclsparse_destroy`,

.. _sparse-mkl-struct:

.. table:: Mapping MKL Parameters to AOCL-Sparse
   :name: table-mkl-params

   .. csv-table::
      :header-rows: 1
      :widths: 3 2 5 5

      MKL,,AOCL-Sparse,Comments
      ``transa``,       ``N`` or ``n``,   op = :cpp:enumerator:`aoclsparse_operation_none`,
      ``transa``,       ``T`` or ``t``,   op = :cpp:enumerator:`aoclsparse_operation_transpose`,
      ``transa``,       ``C`` or ``c``,   op = :cpp:enumerator:`aoclsparse_operation_conjugate_transpose`,
      ``matdescra[b]``, ``G``,            mat_type = :cpp:enumerator:`aoclsparse_matrix_type_general`,   b is 0/1 based on zero/one-based indexing
      ``matdescra[b]``, ``S``,            mat_type = :cpp:enumerator:`aoclsparse_matrix_type_symmetric`,
      ``matdescra[b]``, ``H``,            mat_type = :cpp:enumerator:`aoclsparse_matrix_type_hermitian`,
      ``matdescra[b]``, ``T``,            mat_type = :cpp:enumerator:`aoclsparse_matrix_type_triangular`,
      "| ``matdescra[b+1]``
      | ``*uplo``",     ``L``,            fill = :cpp:enumerator:`aoclsparse_fill_mode_lower`,           matdescra[b+1] is ``S`` or ``H`` or ``T``
      "| ``matdescra[b+1]``
      | ``*uplo``",     ``U``,            fill = :cpp:enumerator:`aoclsparse_fill_mode_upper`,           matdescra[b+1] is ``S`` or ``H`` or ``T``
      ``matdescra[b+2]``,``N``,           diag = :cpp:enumerator:`aoclsparse_diag_type_non_unit`,        matdescra[b+1] is ``S`` or ``H`` or ``T``
      ``matdescra[b+2]``,``U``,           diag = :cpp:enumerator:`aoclsparse_diag_type_unit`,            matdescra[b+1] is ``S`` or ``H`` or ``T``
      ``matdescra[b+3]``,``C``,           base = :cpp:enumerator:`aoclsparse_index_base_zero`,
      ``matdescra[b+3]``,``F``,           base = :cpp:enumerator:`aoclsparse_index_base_one`,


Example: Convert MKL Sparse Matrix-Vector Multiplication with CSR Matrices to AOCL-Sparse
------------------------------------------------------------------------------------------

**MKL call**

.. code-block:: cpp

   mkl_?csrmv(const char *transa, const MKL_INT *m, const MKL_INT *k, const float *alpha,
     const char *matdescra, const float *val, const MKL_INT *indx, const MKL_INT *pntrb,
     const MKL_INT *pntre, const float *x, const float *beta, float *y)

**Steps to convert into an AOCL-Sparse call**

.. code-block:: cpp

   aoclsparse_?mv(aoclsparse_operation op, const float *alpha, aoclsparse_matrix A,
     const aoclsparse_mat_descr descr, const float *x, const float *beta, float *y)

1. Create a :cpp:type:`aoclsparse_matrix` handle, ``A``, using :cpp:func:`aoclsparse_create_?csr() <aoclsparse_create_scsr>`

   .. code-block:: cpp

      // Create matrix A from m, k, val, indx, pntrb, pntre
      aoclsparse_matrix A;
      aoclsparse_index_base base = /* refer to Table 3 */;
      aoclsparse_int nnz;
      aoclsparse_int *row_ptr;
      aoclsparse_int *col_idx;
      aoclsparse_int i;
      aoclsparse_status status;

      nnz = pntre[*m-1] - pntrb[0];
      col_idx = (aoclsparse_int *) indx;

      // convert pntrb and pntre to row_ptr
      // assuming there is no gap between pntre[i] and pntrb[i+1]
      row_ptr[0] = base;
      for (i = 0; i < m; i++)
        row_ptr[i+1] = row_ptr[i] + pntre[i] - pntrb[i];

      status = aoclsparse_create_scsr(&A, base, *m, *n, nnz, row_ptr, col_idx, val);
      if (status != aoclsparse_status_success)
        return status;

2. Create a :cpp:type:`aoclsparse_mat_descr`, ``descr``, and configure it using
   :cpp:func:`aoclsparse_set_mat_type`, :cpp:func:`aoclsparse_set_mat_fill_mode`,
   :cpp:func:`aoclsparse_set_mat_diag_type`, and :cpp:func:`aoclsparse_set_mat_index_base`.
   Refer to :numref:`table-mkl-params` for parameter mappings.

   .. code-block:: cpp

      // create descr
      aoclsparse_mat_descr descr;
      status = aoclsparse_create_mat_descr(&descr);
      if (status != aoclsparse_status_success)
        return status;

      // fill in descr based on the values of the matdescra pointers
      aoclsparse_matrix_type mat_type = /* refer to Table 3 */;
      aoclsparse_fill_mode fill = /* refer to Table 3 */;
      aoclsparse_diag_type diag = /* refer to Table 3 */;
      aoclsparse_set_mat_index_base(descr, base);
      aoclsparse_set_mat_type(descr, mat_type);
      aoclsparse_set_mat_fill_mode(descr, fill);
      aoclsparse_set_mat_diag_type(descr, diag);

3. Call :cpp:func:`aoclsparse_?mv() <aoclsparse_smv>` for sparse matrix-vector multiplication

   .. code-block:: cpp

      // Map op to an appropriate value
      aoclsparse_operation op = /* refer to Table 3 */;

      // Call sparse matrix vector multiplication
      status = aoclsparse_smv(op, alpha, A, descr, x, beta, y);
      if (status != aoclsparse_status_success)
        return status;

4. Cleanup using :cpp:func:`aoclsparse_destroy_mat_descr` and :cpp:func:`aoclsparse_destroy`

   .. code-block:: cpp

      // descriptor
      aoclsparse_destroy_mat_descr(descr);

      // matrix
      aoclsparse_destroy(&A);

Example: Convert MKL Symmetric Sparse Matrix-Vector Multiplication with CSR Matrices to AOCL-Sparse
----------------------------------------------------------------------------------------------------

**MKL call**

.. code-block:: cpp

   void mkl_scsrsymv(const char *uplo, const MKL_INT *m, const float *a,
     const MKL_INT *ia, const MKL_INT *ja, const float *x, float *y);

**Steps to convert into an AOCL-Sparse call**

.. code-block:: cpp

   aoclsparse_?mv(aoclsparse_operation op, const float *alpha, aoclsparse_matrix A,
     const aoclsparse_mat_descr descr, const float *x, const float *beta, float *y)

1. Create a :cpp:type:`aoclsparse_matrix` handle, ``A``, using :cpp:func:`aoclsparse_create_?csr() <aoclsparse_create_scsr>`

   .. code-block:: cpp

      // Create matrix A from m, k, val, indx, pntrb, pntre
      aoclsparse_matrix A;
      aoclsparse_index_base base = /* refer to Table 3 */;
      aoclsparse_int nnz;
      aoclsparse_int *row_ptr;
      aoclsparse_int *col_idx;
      aoclsparse_int i;
      aoclsparse_status status;

      nnz = pntre[*m-1] - pntrb[0];
      col_idx = (aoclsparse_int *) indx;

      // convert pntrb and pntre to row_ptr
      // assuming there is no gap between pntre[i] and pntrb[i+1]
      row_ptr[0] = base;
      for (i = 0; i < m; i++)
        row_ptr[i+1] = row_ptr[i] + pntre[i] - pntrb[i];

      status = aoclsparse_create_scsr(&A, base, *m, *n, nnz, row_ptr, col_idx, val);
      if (status != aoclsparse_status_success)
        return status;

2. Create a :cpp:type:`aoclsparse_mat_descr`, ``descr``, and configure it using
   :cpp:func:`aoclsparse_set_mat_type`, :cpp:func:`aoclsparse_set_mat_fill_mode`,
   :cpp:func:`aoclsparse_set_mat_diag_type`, and :cpp:func:`aoclsparse_set_mat_index_base`.
   Refer to :numref:`table-mkl-params` for parameter mappings.

   .. code-block:: cpp

      // create descr
      aoclsparse_mat_descr descr;
      status = aoclsparse_create_mat_descr(&descr);
      if (status != aoclsparse_status_success)
        return status;

      // fill in descr based on the values of the matdescra pointers
      aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_symmetric;
      aoclsparse_fill_mode fill = /* refer to Table 3 */;
      aoclsparse_diag_type diag = /* refer to Table 3 */;
      aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
      aoclsparse_set_mat_type(descr, mat_type);
      aoclsparse_set_mat_fill_mode(descr, fill);
      aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

3. Call :cpp:func:`aoclsparse_?mv() <aoclsparse_smv>` for sparse matrix-vector multiplication

   .. code-block:: cpp

      // Map op to an appropriate value
      aoclsparse_operation op = aoclsparse_operation_none;

      // Call sparse matrix vector multiplication
      status = aoclsparse_smv(op, 1, A, descr, x, 0, y);
      if (status != aoclsparse_status_success)
        return status;

4. Cleanup using :cpp:func:`aoclsparse_destroy_mat_descr` and :cpp:func:`aoclsparse_destroy`

   .. code-block:: cpp

      // descriptor
      aoclsparse_destroy_mat_descr(descr);

      // matrix
      aoclsparse_destroy(&A);


.. rubric:: Footnotes

.. [#intel_copyright]
   The oneMKL contents in this document are copyrighted by Intel
   Corporation and licensed under the terms of the Apache v2:
   `<https://github.com/oneapi-src/oneMKL/blob/develop/LICENSE>`__ and are
   not from files in the "third party programs" in oneMKL.
