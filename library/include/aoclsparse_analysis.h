/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 *  \brief These functions provides analysis and optiomization functionality
 */
#ifndef AOCLSPARSE_ANALYSIS_H_
#define AOCLSPARSE_ANALYSIS_H_

#include "aoclsparse_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup analysis_module
*  \brief Performs analysis and possible data allocations and matrix restructuring operations
*  related to accelerate sparse operations involving matrices
*
*  \details
*  In aoclsparse_optimize() sparse matrices are restructured based on matrix analysis,
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
*  \brief Record hints of the expected number and types of calls to optimize the input matrix for.
*
*  \details
*  Any of the \p aoclsparse_set_*_hint functions may be used to indicate that a given number of calls to the same
*  Sparse BLAS API will be performed. When aoclsparse_optimize() is invoked, the input matrix might be
*  tuned to accelerate the hinted calls.
*
*  @param[in]
*  mat         Input sparse matrix to be tuned.
*  @param[in]
*  trans       Matrix operation to perform during the calls.
*  @param[in]
*  descr       Descriptor of the sparse matrix used during the calls.
*  @param[in]
*  expected_no_of_calls   A rough estimate of the number of the calls.
*
*  \retval  aoclsparse_status_success           the operation completed successfully.
*  \retval  aoclsparse_status_invalid_value     \p mat, \p trans, \p descr or \p expected_no_of_calls is invalid.
*                                               Expecting \p expected_no_of_calls > 0.
*  \retval  aoclsparse_status_invalid_pointer   \p mat or \p descr is invalid.
*  \retval  aoclsparse_status_memory_error      internal memory allocation failure.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_mv_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls);
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
/**@}*/

/*! \ingroup analysis_module
*  \brief Provides hints to optimize preconditioning matrices
*
*  \details
*  Set hints for analysis and optimization of preconditioning-related factorizations and/or
*  accelerate the application of such preconditioner,
*  this can also include hints for "fused" operations that accelerate two operations in a
*  single call.
*
*  @param[in]
*  mat         A sparse matrix
*  @param[in]
*  trans       Whether in transposed state or not. Transpose operation is not yet supported.
*  @param[in]
*  descr       Descriptor of the sparse matrix.
*  @param[in]
*  expected_no_of_calls   Expected number of call to an API that uses matrix \p mat.
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size indicates that \p mat is invalid.
*  \retval     aoclsparse_status_invalid_pointer at least one of the input pointers is invalid.
*  \retval     aoclsparse_status_internal_error Indicates that an internal error occurred.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_lu_smoother_hint(aoclsparse_matrix          mat,
                                                  aoclsparse_operation       trans,
                                                  const aoclsparse_mat_descr descr,
                                                  aoclsparse_int             expected_no_of_calls);
DLL_PUBLIC
aoclsparse_status aoclsparse_set_symgs_hint(aoclsparse_matrix          mat,
                                            aoclsparse_operation       trans,
                                            const aoclsparse_mat_descr descr,
                                            aoclsparse_int             expected_no_of_calls);
DLL_PUBLIC
aoclsparse_status aoclsparse_set_dotmv_hint(aoclsparse_matrix          mat,
                                            aoclsparse_operation       trans,
                                            const aoclsparse_mat_descr descr,
                                            aoclsparse_int             expected_no_of_calls);
/**@}*/

/*! \ingroup analysis_module
*  \brief Record a hint of the expected number of calls to aoclsparse_strsm() and variants
*  to optimize the input matrix for.
*
*  \details
*  aoclsparse_set_sm_hint() may be used to indicate that a given number
*  of calls to the triangular solver aoclsparse_strsm() or other variant
*  will be performed. When aoclsparse_optimize() is invoked,
*  the input matrix might be tuned to accelerate the hinted calls.
*  The hints include not only the estimated number of calls to the API solver,
*  but also other (matrix) parameters. The hinted matrix should not be modified
*  after the call to optimize and before the call to the solver.
*
*  @param[in]
*  mat         Input sparse matrix to be tuned.
*  @param[in]
*  trans       Matrix operation to perform during the calls.
*  @param[in]
*  descr       Descriptor of the sparse matrix used during the calls.
*  @param[in]
*  order       Layout of the right-hand-side input matrix used during the calls,
*              valid options are \ref aoclsparse_order_row
*              and \ref aoclsparse_order_column.
*  @param[in]
*  expected_no_of_calls   A rough estimate of the number of the calls.
*
*  \retval  aoclsparse_status_success           the operation completed successfully.
*  \retval  aoclsparse_status_invalid_value     \p expected_no_of_calls, \p order, \p mat,
*                                               \p trans or \p descr is invalid.
*  \retval  aoclsparse_status_invalid_pointer   \p mat or \p descr is invalid.
*  \retval  aoclsparse_status_memory_error      internal memory allocation failure.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_sm_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         const aoclsparse_order     order,
                                         const aoclsparse_int       expected_no_of_calls);
/**@}*/

/*! \ingroup analysis_module
*  \brief Record a hint of the expected number of aoclsparse_sorv()
*  calls to optimize the input matrix for.
*
*  \details
*  \P{aoclsparse_set_sorv_hint} may be used to indicate that a given number
*  of calls to the SOR preconditioner aoclsparse_sorv()
*  will be performed. When aoclsparse_optimize() is invoked,
*  the input matrix might be tuned to accelerate the hinted calls.
*  The hints include not only the estimated number of the API calls
*  but also their other parameters which should match the actual calls.
*
*  @param[in]
*  mat         Input sparse matrix to be tuned.
*  @param[in]
*  descr       Descriptor of the sparse matrix used during the calls.
*  @param[in]
*  type        The operation to perform by the SOR preconditioner.
*  @param[in]
*  expected_no_of_calls     A rough estimate of the number of the calls.
*
*  \retval  aoclsparse_status_success           the operation completed successfully.
*  \retval  aoclsparse_status_invalid_value     \p expected_no_of_calls, \p descr, \p type  or
                                                \p mat type is invalid.
*  \retval  aoclsparse_status_invalid_pointer   \p mat or \p descr is NULL.
*  \retval  aoclsparse_status_memory_error      internal memory allocation failure.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_sorv_hint(aoclsparse_matrix          mat,
                                           const aoclsparse_mat_descr descr,
                                           const aoclsparse_sor_type  type,
                                           const aoclsparse_int       expected_no_of_calls);
/**@}*/

/*! \ingroup analysis_module
*  \brief Record user's attitude to the memory consumption while optimizing
*  the input matrix for the hinted operations.
*
*  \details
*  \P{aoclsparse_set_memory_hint} may be used to indicate how much memory can
*  be allocated during the optimization process of the input matrix for
*  the previously hinted operations. In particular, \ref aoclsparse_memory_usage_minimal
*  suggests that the new memory should be only of order of vectors, whereas
*  \ref aoclsparse_memory_usage_unrestricted allows even new copies of
*  the whole matrix. The unrestricted memory policy is the default. Any change
*  to the memory policy applies only to any new optimizations for the new hints
*  which have not been processed by aoclsparse_optimize() yet.
*  The optimizations from any previous calls are unaffected. Note that
*  the memory policy is only an indication rather than rule.
*
*  @param[in]
*  mat         Input sparse matrix to be tuned.
*  @param[in]
*  policy      Memory usage policy for future optimizations.
*
*  \retval  aoclsparse_status_success           the operation completed successfully.
*  \retval  aoclsparse_status_invalid_value     \p policy type is invalid.
*  \retval  aoclsparse_status_invalid_pointer   pointer \p mat is invalid.
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_set_memory_hint(aoclsparse_matrix             mat,
                                             const aoclsparse_memory_usage policy);
/**@}*/

#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_ANALYSIS_H_
