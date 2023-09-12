/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 *  \brief aoclsparse_analysis.h provides sparse format analysis subprograms
 */
#ifndef AOCLSPARSE_ANALYSIS_H_
#define AOCLSPARSE_ANALYSIS_H_

#include "aoclsparse_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup analysis_module
*  \brief Performs data allocations and restructuring operations related to sparse matrices
*
*  \details
*  \p aoclsparse_optimize Sparse matrices are restructured based on matrix analysis, 
*   into different storage formats to improve data access and thus performance. 
*
*  @param[in]
*  mat         sparse matrix in CSR format and sparse format information inside
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size \p m is invalid.
*  \retval     aoclsparse_status_invalid_pointer  
*  \retval     aoclsparse_status_internal_error an internal error occurred.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_optimize(aoclsparse_matrix mat);
/**@}*/

/*! \ingroup analysis_module
*  \brief Provides any hints such as the type of routine, expected no of calls etc
*
*  \details
*  \p aoclsparse_set_mv_hint sets a hint id for analysis and execute phases of the program
*     to analyse and perform ILU factorization and Solution
*
*  @param[in]
*  mat         sparse matrix in CSR format and sparse format information inside
*  @param[in]
*  trans       Whether in transposed state or not. Transpose operation is not yet supported.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix. Currently, only
 *              \ref aoclsparse_matrix_type_general and
 *              \ref aoclsparse_matrix_type_symmetric is supported.
*  @param[in]
*  expected_no_of_calls   unused parameter
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size \p m is invalid.
*  \retval     aoclsparse_status_invalid_pointer  
*  \retval     aoclsparse_status_internal_error an internal error occurred.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_mv_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls);
/**@}*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_sv_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls);
DLL_PUBLIC
aoclsparse_status aoclsparse_set_mm_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls);

DLL_PUBLIC
aoclsparse_status aoclsparse_set_2m_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls);
/*! \ingroup analysis_module
*  \brief Provides any hints such as the type of routine, expected no of calls etc
*
*  \details
*  \p aoclsparse_set_lu_smoother_hint sets a hint id for analysis and execute phases of the program
*     to analyse and perform ILU factorization and Solution
*
*  @param[in]
*  mat         A sparse matrix and ILU related information inside
*  @param[in]
*  trans       Whether in transposed state or not. Transpose operation is not yet supported.
*  @param[in]
*  descr       Descriptor of the sparse matrix. 
*  @param[in]
*  expected_no_of_calls   unused parameter
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size indicates that \p m is invalid, expecting m>=0.
*  \retval     aoclsparse_status_invalid_pointer.  
*  \retval     aoclsparse_status_internal_error Indicates that an internal error occurred.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_lu_smoother_hint(aoclsparse_matrix          mat,
                                                  aoclsparse_operation       trans,
                                                  const aoclsparse_mat_descr descr,
                                                  aoclsparse_int             expected_no_of_calls);
/**@}*/

/*! \ingroup analysis_module
*  \brief Store user-hints to accelerate the \f$ \verb+aoclsparse_?trsm+\f$ triangular-solvers.
*
*  \details
*  This function stores user-provided hints related to the structures of the matrices involved 
*  in a triangular linear system of equations and its solvers. The hints are for the problem
*  \f[
*    op(A) \cdot X = \alpha \cdot B,
*  \f]
*  where \f$A\f$ is a sparse matrix, \f$op()\f$ is a linear operator, \f$X\f$ and \f$B\f$ are 
*  dense matrices, while \f$alpha\f$ is a scalar.
*  The hints are used in order to perform certain optimizations over the input data that can 
*  potentially accelerate the solve operation. The hints include, expected number of calls to 
*  the API, matrix layout, dimension of dense right-hand-side matrix, etc.
*
*  @param[in]
*  A           A sparse matrix \f$A\f$.
*  @param[in]
*  trans       Operation to perform on the sparse matrix \f$A\f$, valid options are
*  \ref aoclsparse_operation_none, aoclsparse_operation_transpose, and aoclsparse_operation_conjugate_transpose.
*  @param[in]
*  descr       Descriptor of the sparse matrix \f$A\f$.
*  @param[in]
*  order       Layout of the right-hand-side matrix \f$B\f$, valid options are \ref aoclsparse_order_row 
*  and \ref aoclsparse_order_column.
*  @param[in]
*  dense_matrix_dim number of columns of the dense matrix \f$B\f$.
*  @param[in]
*  expected_no_of_calls   Hint on the potential number of calls to the solver API, e.g., calls to \ref aoclsparse_strsm().
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size \p m, \p n, \p nnz, \p ldb or \p ldx is invalid. 
*              Expecting m>0, n>0, m==n, nnz>0, ldb>=n, ldx>=n
*  \retval     aoclsparse_status_invalid_value Sparse matrix is not square, or 
               \p expected_no_of_calls or \p dense_matrix_dim or \p matrix type are invalid.
*  \retval     aoclsparse_status_invalid_pointer Pointers to sparse matrix \f$A\f$ or dense matrices 
*                                                \f$B\f$ or \f$X\f$ or descriptor are null
*  \retval     aoclsparse_status_internal_error Indicates that an internal error occurred.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_sm_hint(aoclsparse_matrix          A,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         const aoclsparse_order     order,
                                         const aoclsparse_int       dense_matrix_dim,
                                         const aoclsparse_int       expected_no_of_calls);
/**@}*/
#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_ANALYSIS_H_
