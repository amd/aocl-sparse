/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
 *  \brief Double Precision Sparse matrix vector multiplication using CSR storage format
 *
 *  \details
 *  \p aoclsparse_dcsrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref aoclsparse_operation_none is supported.
 *
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
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
 *
 *  \par Example
 *  This example performs a sparse matrix vector multiplication in CSR format
 *  using additional meta data to improve performance.
 *
 *  \Code{.c}
 *      aoclsparse_dcsrmv(m,
 *                       n,
 *                       nnz,
 *                       &alpha,
 *                       csr_val,
 *                       csr_row_ptr,
 *                       csr_col_ind,
 *                       x,
 *                       &beta,
 *                       y);
 *  \endcode
 */
/**@{*/
__attribute__((__visibility__("default"))) aoclsparse_status aoclsparse_dcsrmv(aoclsparse_int             m,
                                  aoclsparse_int             n,
                                  aoclsparse_int             nnz,
                                  const double*             alpha,
                                  const double*             csr_val,
                                  const aoclsparse_int*      csr_row_ptr,
                                  const aoclsparse_int*      csr_col_ind,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);

/*! \ingroup level2_module
 *  \brief Single Precision Sparse matrix vector multiplication using CSR storage format
 *
 *  \details
 *  \p aoclsparse_scsrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in CSR storage format, and the sparse vector \f$x\f$ and adds the
 *  result to the sparse vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref aoclsparse_operation_none is supported.
 *
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
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
 *
 *  \par Example
 *  This example performs a sparse matrix vector multiplication in CSR format
 *  using additional meta data to improve performance.
 *
 *  \Code{.c}
 *      aoclsparse_scsrmv(m,
 *                       n,
 *                       nnz,
 *                       &alpha,
 *                       csr_val,
 *                       csr_row_ptr,
 *                       csr_col_ind,
 *                       x,
 *                       &beta,
 *                       y);
 *  \endcode
 */
/**@{*/
__attribute__((__visibility__("default"))) aoclsparse_status aoclsparse_scsrmv(aoclsparse_int             m,
                                  aoclsparse_int             n,
                                  aoclsparse_int             nnz,
                                  const float*             alpha,
                                  const float*             csr_val,
                                  const aoclsparse_int*      csr_row_ptr,
                                  const aoclsparse_int*      csr_col_ind,
                                  const float*             x,
                                  const float*             beta,
                                  float*                   y);

/******************************************************************************** 
* \brief Get aoclsparse version 
* version % 100        = patch level 
* version / 100 % 1000 = minor version 
* version / 100000     = major version 
*******************************************************************************/ 
__attribute__((__visibility__("default"))) aoclsparse_status aoclsparse_get_version(int* version);

#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_FUNCTIONS_H_

