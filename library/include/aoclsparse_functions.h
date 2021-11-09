/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
/*! \file
 *  \brief aoclsparse_functions.h provides Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3, for AMD CPU hardware.
 */
#ifndef AOCLSPARSE_FUNCTIONS_H_
#define AOCLSPARSE_FUNCTIONS_H_

#include "aoclsparse_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level2_module
 *  \brief Single & Double precision sparse matrix vector multiplication using CSR storage format
 *
 *  \details
 *  \p aoclsparse_csrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == aoclsparse_operation_none} \\
 *        A^T, & \text{if trans == aoclsparse_operation_transpose} \\
 *        A^H, & \text{if trans == aoclsparse_operation_conjugate_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < m; ++i)
 *      {
 *          y[i] = beta * y[i];
 *
 *          for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
 *          {
 *              y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
 *          }
 *      }
 *  \endcode
 *
 *  \note
 *  Currently, only \p trans == \ref aoclsparse_operation_none is supported.
 *  Currently, for \ref aoclsparse_matrix_type == \ref aoclsparse_matrix_type_symmetric,
 *  only lower triangular matrices are supported.
 *
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix. Currently, only
 *              \ref aoclsparse_matrix_type_general and
 *              \ref aoclsparse_matrix_type_symmetric is supported.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p x, \p beta or \p y pointer is
 *              invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \p trans != \ref aoclsparse_operation_none or
 *              \ref aoclsparse_matrix_type != \ref aoclsparse_matrix_type_general.
 *              \ref aoclsparse_matrix_type != \ref aoclsparse_matrix_type_symmetric.
 *
 *  \par Example
 *  This example performs a sparse matrix vector multiplication in CSR format
 *  using additional meta data to improve performance.
 *  \code{.c}
 *      // Compute y = Ax
 *      aoclsparse_scsrmv(aoclsparse_operation_none,
 *                       &alpha,
 *                       m,
 *                       n,
 *                       nnz,
 *                       csr_val,
 *                       csr_col_ind,
 *                       csr_row_ptr,
 *                       descr,
 *                       x,
 *                       &beta,
 *                       y);
 *
 *      // Do more work
 *      // ...
 *
 *  \endcode
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_scsrmv(aoclsparse_operation       trans,
                                  const float*              alpha,
                                  aoclsparse_int             m,
                                  aoclsparse_int             n,
                                  aoclsparse_int             nnz,
                                  const float*              csr_val,
                                  const aoclsparse_int*      csr_col_ind,
                                  const aoclsparse_int*      csr_row_ptr,
                                  const aoclsparse_mat_descr descr,
                                  const float*             x,
                                  const float*             beta,
                                  float*                   y);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsrmv(aoclsparse_operation       trans,
                                  const double*              alpha,
                                  aoclsparse_int             m,
                                  aoclsparse_int             n,
                                  aoclsparse_int             nnz,
                                  const double*              csr_val,
                                  const aoclsparse_int*      csr_col_ind,
                                  const aoclsparse_int*      csr_row_ptr,
                                  const aoclsparse_mat_descr descr,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);
/**@}*/

/*! \ingroup level2_module
 *  \brief Single & Double precision sparse matrix vector multiplication using ELL storage format
 *
 *  \details
 *  \p aoclsparse_ellmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in ELL storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == aoclsparse_operation_none} \\
 *        A^T, & \text{if trans == aoclsparse_operation_transpose} \\
 *        A^H, & \text{if trans == aoclsparse_operation_conjugate_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < m; ++i)
 *      {
 *          y[i] = beta * y[i];
 *
 *          for(p = 0; p < ell_width; ++p)
 *          {
 *              idx = p * m + i;
 *
 *              if((ell_col_ind[idx] >= 0) && (ell_col_ind[idx] < n))
 *              {
 *                  y[i] = y[i] + alpha * ell_val[idx] * x[ell_col_ind[idx]];
 *              }
 *          }
 *      }
 *  \endcode
 *
 *  \note
 *  Currently, only \p trans == \ref aoclsparse_operation_none is supported.
 *
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  m           number of rows of the sparse ELL matrix.
 *  @param[in]
 *  n           number of columns of the sparse ELL matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse ELL matrix.
 *  @param[in]
 *  descr       descriptor of the sparse ELL matrix. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  ell_val     array that contains the elements of the sparse ELL matrix. Padded
 *              elements should be zero.
 *  @param[in]
 *  ell_col_ind array that contains the column indices of the sparse ELL matrix.
 *              Padded column indices should be -1.
 *  @param[in]
 *  ell_width   number of non-zero elements per row of the sparse ELL matrix.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p ell_val,
 *              \p ell_col_ind, \p x, \p beta or \p y pointer is invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \p trans != \ref aoclsparse_operation_none or
 *              \ref aoclsparse_matrix_type != \ref aoclsparse_matrix_type_general.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_sellmv(aoclsparse_operation       trans,
                                   const float*              alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const float*              ell_val,
                                   const aoclsparse_int*      ell_col_ind,
                                   aoclsparse_int      ell_width,
                                   const aoclsparse_mat_descr descr,
                                   const float*             x,
                                   const float*            beta,
                                   float*                   y );

DLL_PUBLIC
aoclsparse_status aoclsparse_dellmv(aoclsparse_operation       trans,
                                   const double*              alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const double*              ell_val,
                                   const aoclsparse_int*      ell_col_ind,
                                   aoclsparse_int      ell_width,
                                   const aoclsparse_mat_descr descr,
                                   const double*             x,
                                   const double*            beta,
                                   double*                   y );
/**@}*/


/*! \ingroup level2_module
 *  \brief Single & Double precision sparse matrix vector multiplication using DIA storage format
 *
 *  \details
 *  \p aoclsparse_diamv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in DIA storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == aoclsparse_operation_none} \\
 *        A^T, & \text{if trans == aoclsparse_operation_transpose} \\
 *        A^H, & \text{if trans == aoclsparse_operation_conjugate_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  Currently, only \p trans == \ref aoclsparse_operation_none is supported.
 *
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  m           number of rows of the sparse DIA matrix.
 *  @param[in]
 *  n           number of columns of the sparse DIA matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse DIA matrix.
 *  @param[in]
 *  descr       descriptor of the sparse DIA matrix. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  dia_val     array that contains the elements of the sparse DIA matrix. Padded
 *              elements should be zero.
 *  @param[in]
 *  dia_offset  array that contains the offsets of each diagonal of the sparse DIAL matrix.
 *
 *  @param[in]
 *  dia_num_diag  number of diagonals in the sparse DIA matrix.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) == A\f$) or \p m elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) == A\f$) or \p n elements
 *              (\f$op(A) == A^T\f$ or \f$op(A) == A^H\f$).
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p ell_val,
 *              \p ell_col_ind, \p x, \p beta or \p y pointer is invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \p trans != \ref aoclsparse_operation_none or
 *              \ref aoclsparse_matrix_type != \ref aoclsparse_matrix_type_general.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_sdiamv(aoclsparse_operation       trans,
                                 const float*              alpha,
                                 aoclsparse_int             m,
                                 aoclsparse_int             n,
                                 aoclsparse_int             nnz,
                                 const float*              dia_val,
                                 const aoclsparse_int*      dia_offset,
                                 aoclsparse_int      dia_num_diag,
                                 const aoclsparse_mat_descr descr,
                                 const float*             x,
                                 const float*            beta,
                                 float*                   y );

DLL_PUBLIC
aoclsparse_status aoclsparse_ddiamv(aoclsparse_operation       trans,
                                 const double*              alpha,
                                 aoclsparse_int             m,
                                 aoclsparse_int             n,
                                 aoclsparse_int             nnz,
                                 const double*              dia_val,
                                 const aoclsparse_int*      dia_offset,
                                 aoclsparse_int      dia_num_diag,
                                 const aoclsparse_mat_descr descr,
                                 const double*             x,
                                 const double*            beta,
                                 double*                   y );
/**@}*/

/*! \ingroup level2_module
*  \brief Single & Double precision Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p aoclsparse_bsrmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \cdot \text{bsr_dim}) \times (nb \cdot \text{bsr_dim})\f$
*  matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == aoclsparse_operation_none} \\
*        A^T, & \text{if trans == aoclsparse_operation_transpose} \\
*        A^H, & \text{if trans == aoclsparse_operation_conjugate_transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  Currently, only \p trans == \ref aoclsparse_operation_none is supported.
*
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix. Currently, only
*              \ref aoclsparse_matrix_type_general is supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  bsr_dim     block dimension of the sparse BSR matrix.
*  @param[in]
*  x           array of \p nb*bsr_dim elements (\f$op(A) = A\f$) or \p mb*bsr_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb*bsr_dim elements (\f$op(A) = A\f$) or \p nb*bsr_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_handle the library context was not initialized.
*  \retval     aoclsparse_status_invalid_size \p mb, \p nb, \p nnzb or \p bsr_dim is
*              invalid.
*  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
*              \p bsr_row_ind, \p bsr_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     aoclsparse_status_arch_mismatch the device is not supported.
*  \retval     aoclsparse_status_not_implemented
*              \p trans != \ref aoclsparse_operation_none or
*              \ref aoclsparse_matrix_type != \ref aoclsparse_matrix_type_general.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_sbsrmv(aoclsparse_operation       trans,
                                    const float*              alpha,
                                    aoclsparse_int             mb,
                                    aoclsparse_int             nb,
                                    aoclsparse_int             bsr_dim,
                                    const float*              bsr_val,
                                    const aoclsparse_int*      bsr_col_ind,
                                    const aoclsparse_int*      bsr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const float*             x,
                                    const float*             beta,
                                    float*                   y
                                    );

DLL_PUBLIC
aoclsparse_status aoclsparse_dbsrmv(aoclsparse_operation       trans,
                                    const double*              alpha,
                                    aoclsparse_int             mb,
                                    aoclsparse_int             nb,
                                    aoclsparse_int             bsr_dim,
                                    const double*              bsr_val,
                                    const aoclsparse_int*      bsr_col_ind,
                                    const aoclsparse_int*      bsr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const double*             x,
                                    const double*             beta,
                                    double*                   y
                                    );

/**@}*/

/*! \ingroup level2_module
 *  \brief Sparse triangular solve using CSR storage format for single and double
 *      data precisions.
 *
 *  \details
 *  \p aoclsparse_csrsv solves a sparse triangular linear system of a sparse
 *  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
 *  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
 *  \f[
 *    op(A) \cdot y = \alpha \cdot x,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == aoclsparse_operation_none} \\
 *        A^T, & \text{if trans == aoclsparse_operation_transpose} \\
 *        A^H, & \text{if trans == aoclsparse_operation_conjugate_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  Currently, only \p trans == \ref aoclsparse_operation_none is supported.
 *
 *  \note
 *  The input matrix has to be sparse upper or lower triangular matrix
 *  with unit or non-unit main diagonal. Matrix has to be sorted.
 *  No diagonal element can be omitted from a sparse storage
 *  if the solver is called with the non-unit indicator.
 *
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  x           array of \p m elements, holding the right-hand side.
 *  @param[out]
 *  y           array of \p m elements, holding the solution.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p x or \p y pointer is invalid.
 *  \retval     aoclsparse_status_internal_error an internal error occurred.
 *  \retval     aoclsparse_status_not_implemented
 *              \p trans == \ref aoclsparse_operation_conjugate_transpose or
 *              \p trans == \ref aoclsparse_operation_transpose or
 *              \ref aoclsparse_matrix_type != \ref aoclsparse_matrix_type_general.
 *
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_scsrsv(aoclsparse_operation       trans,
                                  const float*              alpha,
                                  aoclsparse_int             m,
                                  const float*              csr_val,
                                  const aoclsparse_int*      csr_col_ind,
                                  const aoclsparse_int*      csr_row_ptr,
                                  const aoclsparse_mat_descr descr,
                                  const float*             x,
                                  float*                  y
                               );


DLL_PUBLIC
aoclsparse_status aoclsparse_dcsrsv(aoclsparse_operation       trans,
                                  const double*              alpha,
                                  aoclsparse_int             m,
                                  const double*              csr_val,
                                  const aoclsparse_int*      csr_col_ind,
                                  const aoclsparse_int*      csr_row_ptr,
                                  const aoclsparse_mat_descr descr,
                                  const double*             x,
                                  double*                  y
                               );
/**@}*/

/*! \ingroup level3_module
 *  \brief Sparse matrix dense matrix multiplication using CSR storage format
 *
 *  \details
 *  \p aoclsparse_csrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
 *  matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
 *  matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot B + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans_A == aoclsparse_operation_none} \\
 *        A^T, & \text{if trans_A == aoclsparse_operation_transpose} \\
 *        A^H, & \text{if trans_A == aoclsparse_operation_conjugate_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \code{.c}
 *      for(i = 0; i < ldc; ++i)
 *      {
 *          for(j = 0; j < n; ++j)
 *          {
 *              C[i][j] = beta * C[i][j];
 *
 *              for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
 *              {
 *                  C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
 *              }
 *          }
 *      }
 *  \endcode
 *
 *
 *  @param[in]
 *  trans_A     matrix \f$A\f$ operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  csr         sparse CSR matrix \f$A\f$ structure.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix \f$A\f$. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  order       aoclsparse_order_row/aoclsparse_order_column for dense matrix
 *  @param[in]
 *  B           array of dimension \f$ldb \times n\f$ or
 *              \f$ldb \times k\f$ .
 *  @param[in]
 *  n           number of columns of the dense matrix \f$B\f$ and \f$C\f$.
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$
 *              (\f$op(A) == A\f$) or \f$\max{(1, m)}\f$ (\f$op(A) == A^T\f$ or
 *              \f$op(A) == A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  C           array of dimension \f$ldc \times n\f$.
 *  @param[in]
 *  ldc         leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$
 *              (\f$op(A) == A\f$) or \f$\max{(1, k)}\f$ (\f$op(A) == A^T\f$ or
 *              \f$op(A) == A^H\f$).
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n, \p k, \p nnz, \p ldb or \p ldc
 *              is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p csr,
 *              \p B, \p beta or \p C pointer is invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \ref aoclsparse_matrix_type != \ref aoclsparse_matrix_type_general.
 *
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_scsrmm(aoclsparse_operation     trans_A,
                                  const float*               alpha,
                                  const aoclsparse_mat_csr   csr,
                                  const aoclsparse_mat_descr descr,
                                  aoclsparse_order           order,
                                  const float*               B,
                                  aoclsparse_int             n,
                                  aoclsparse_int             ldb,
                                  const float*               beta,
                                  float*                     C,
                                  aoclsparse_int             ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsrmm(aoclsparse_operation     trans_A,
                                  const double*              alpha,
                                  const aoclsparse_mat_csr   csr,
                                  const aoclsparse_mat_descr descr,
                                  aoclsparse_order           order,
                                  const double*              B,
                                  aoclsparse_int             n,
                                  aoclsparse_int             ldb,
                                  const double*              beta,
                                  double*                    C,
                                  aoclsparse_int             ldc);
/**@}*/

/*! \ingroup level3_module
 *  \brief Sparse matrix Sparse matrix multiplication using CSR storage format
 *
 *  \details
 *  \p aoclsparse_csr2m multiplies a sparse \f$m \times k\f$
 *  matrix \f$A\f$, defined in CSR storage format, and the sparse \f$k \times n\f$
 *  matrix \f$B\f$, defined in CSR storage format and stores the result to the sparse
 *  \f$m \times n\f$ matrix \f$C\f$, such that
 *  \f[
 *    C :=  op(A) \cdot op(B),
 *  \f]
 *  with
 *  \f[
 *     op(A) = \left\{
 *     \begin{array}{ll}
 *         A,   & \text{if trans_A == aoclsparse_operation_none} \\
 *         A^T, & \text{if trans_A == aoclsparse_operation_transpose} \\
 *         A^H, & \text{if trans_A == aoclsparse_operation_conjugate_transpose}
 *     \end{array}
 *     \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if trans_B == aoclsparse_operation_none} \\
 *        B^T, & \text{if trans_B == aoclsparse_operation_transpose} \\
 *        B^H, & \text{if trans_B == aoclsparse_operation_conjugate_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  @param[in]
 *  trans_A     matrix \f$A\f$ operation type.
 *  @param[in]
 *  descrA      descriptor of the sparse CSR matrix \f$A\f$. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  csrA        sparse CSR matrix \f$A\f$ structure.
 *  @param[in]
 *  trans_B     matrix \f$B\f$ operation type.
 *  @param[in]
 *  descrB      descriptor of the sparse CSR matrix \f$B\f$. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  csrB        sparse CSR matrix \f$B\f$ structure.
 *  @param[in]
 *  request     Specifies full computation or two-stage algorithm
 *  		\ref aoclsparse_stage_nnz_count , Only rowIndex array of the
 *  		CSR matrix is computed internally. The output sparse CSR matrix
 *  		can be extracted to measure the memory required for full operation.
 *  		\ref aoclsparse_stage_finalize . Finalize computation of remaining
 *  		output arrays ( column indices and values of output matrix entries) .
 *  		Has to be called only after aoclsparse_dcsr2m call with
 *  		aoclsparse_stage_nnz_count parameter.
 *  		\ref aoclsparse_stage_full_computation . Perform the entire
 *  		computation in a single step.
 *
 *  @param[out]
 *  *csrC        Pointer to sparse CSR matrix \f$C\f$ structure.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size input parameters contain an invalid value.
 *  \retval     aoclsparse_status_invalid_pointer \p descrA,  \p csr,
 *              \p descrB,  \p csrB, \p csrC is invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \ref aoclsparse_matrix_type != \ref aoclsparse_matrix_type_general.
 *
 *  \par Example
 *  Shows multiplication of 2 sparse matrices to give a newly allocated sparse matrix
 *  \code{.c}
 *  	aoclsparse_mat_csr csrA;
 *  	aoclsparse_create_mat_csr(csrA, base, M, K, nnz_A, csr_row_ptr_A.data(), csr_col_ind_A.data(), csr_val_A.data());
 *  	aoclsparse_mat_csr csrB;
 *  	aoclsparse_create_mat_csr(csrB, base, K, N, nnz_B, csr_row_ptr_B.data(), csr_col_ind_B.data(), csr_val_B.data());
 *
 * 	aoclsparse_mat_csr csrC = NULL;
 * 	aoclsparse_int *csr_row_ptr_C = NULL;
 * 	aoclsparse_int *csr_col_ind_C = NULL;
 * 	double             *csr_val_C = NULL;
 * 	aoclsparse_int C_M, C_N;
 * 	request =  aoclsparse_stage_nnz_count;
 * 	CHECK_AOCLSPARSE_ERROR(aoclsparse_dcsr2m(transA,
 * 		descrA,
 * 		csrA,
 * 		transB,
 * 		descrB,
 * 		csrB,
 * 		request,
 * 		&csrC));
 *
 * 	request =  aoclsparse_stage_finalize;
 * 	CHECK_AOCLSPARSE_ERROR(aoclsparse_dcsr2m(transA,
 * 		descrA,
 * 		csrA,
 * 		transB,
 * 		descrB,
 * 		csrB,
 * 		request,
 * 		&csrC));
 * 	aoclsparse_export_mat_csr(csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, (void **)&csr_val_C);
 *
 *  \endcode
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_dcsr2m(aoclsparse_operation     trans_A,
	    const aoclsparse_mat_descr descrA,
	    const aoclsparse_mat_csr   csrA,
	    aoclsparse_operation       trans_B,
	    const aoclsparse_mat_descr descrB,
	    const aoclsparse_mat_csr   csrB,
	    const aoclsparse_request   request,
	    aoclsparse_mat_csr         *csrC);
/**@}*/

#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_FUNCTIONS_H_

