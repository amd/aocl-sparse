/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
 *  \brief aoclsparse_analysis.h provides Sparse Format analysis Subprograms
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
*  mat         sparse matrix in CSR format and ILU related information inside
*  @param[in]
*  trans       Whether in transposed state or not. Transpose operation is not yet supported.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix. Currently, only
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
aoclsparse_status aoclsparse_set_lu_smoother_hint(aoclsparse_matrix          mat,
                                                  aoclsparse_operation       trans,
                                                  const aoclsparse_mat_descr descr,
                                                  aoclsparse_int             expected_no_of_calls);
/**@}*/

#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_ANALYSIS_H_
