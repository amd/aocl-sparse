/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
 *  \brief aoclsparse_auxiliary.h provides auxilary functions in aoclsparse
 */
#ifndef AOCLSPARSE_AUXILIARY_H_
#define AOCLSPARSE_AUXILIARY_H_

#include "aoclsparse_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Get AOCL-Sparse version
 *
 *  \details
 *  \p aoclsparse_get_version gets the aoclsparse library version number.
 *    in the format "AOCL-Sparse <major>.<minor>.<patch>
 *
 *  @param[out]
 *  version the version string of the aoclsparse library.
 *
 */
DLL_PUBLIC
char* aoclsparse_get_version();

/*! \ingroup aux_module
 *  \brief Create a matrix descriptor
 *  \details
 *  \p aoclsparse_create_mat_descr creates a matrix descriptor. It initializes
 *  \ref aoclsparse_matrix_type to \ref aoclsparse_matrix_type_general and
 *  \ref aoclsparse_index_base to \ref aoclsparse_index_base_zero. It should be destroyed
 *  at the end using aoclsparse_destroy_mat_descr().
 *
 *  @param[out]
 *  descr   the pointer to the matrix descriptor.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p descr pointer is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_create_mat_descr(aoclsparse_mat_descr* descr);

/*! \ingroup aux_module
 *  \brief Copy a matrix descriptor
 *  \details
 *  \p aoclsparse_copy_mat_descr copies a matrix descriptor. Both, source and destination
 *  matrix descriptors must be initialized prior to calling \p aoclsparse_copy_mat_descr.
 *
 *  @param[out]
 *  dest    the pointer to the destination matrix descriptor.
 *  @param[in]
 *  src     the pointer to the source matrix descriptor.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p src or \p dest pointer is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_copy_mat_descr(aoclsparse_mat_descr dest, const aoclsparse_mat_descr src);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_destroy_mat_descr destroys a matrix descriptor and releases all
 *  resources used by the descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p descr is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_destroy_mat_descr(aoclsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the index base of a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_set_mat_index_base sets the index base of a matrix descriptor. Valid
 *  options are \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  base    \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval aoclsparse_status_invalid_value \p base is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_set_mat_index_base(aoclsparse_mat_descr descr, aoclsparse_index_base base);

/*! \ingroup aux_module
 *  \brief Get the index base of a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_get_mat_index_base returns the index base of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 */
DLL_PUBLIC
aoclsparse_index_base aoclsparse_get_mat_index_base(const aoclsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix type of a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_set_mat_type sets the matrix type of a matrix descriptor. Valid
 *  matrix types are \ref aoclsparse_matrix_type_general,
 *  \ref aoclsparse_matrix_type_symmetric, \ref aoclsparse_matrix_type_hermitian or
 *  \ref aoclsparse_matrix_type_triangular.
 *
 *  @param[inout]
 *  descr   the matrix descriptor.
 *  @param[in]
 *  type    \ref aoclsparse_matrix_type_general, \ref aoclsparse_matrix_type_symmetric,
 *          \ref aoclsparse_matrix_type_hermitian or
 *          \ref aoclsparse_matrix_type_triangular.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval aoclsparse_status_invalid_value \p type is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_set_mat_type(aoclsparse_mat_descr descr, aoclsparse_matrix_type type);

/*! \ingroup aux_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_get_mat_type returns the matrix type of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref aoclsparse_matrix_type_general, \ref aoclsparse_matrix_type_symmetric,
 *              \ref aoclsparse_matrix_type_hermitian or
 *              \ref aoclsparse_matrix_type_triangular.
 */
DLL_PUBLIC
aoclsparse_matrix_type aoclsparse_get_mat_type(const aoclsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_set_mat_fill_mode sets the matrix fill mode of a matrix descriptor.
 *  Valid fill modes are \ref aoclsparse_fill_mode_lower or
 *  \ref aoclsparse_fill_mode_upper.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  fill_mode   \ref aoclsparse_fill_mode_lower or \ref aoclsparse_fill_mode_upper.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval aoclsparse_status_invalid_value \p fill_mode is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_set_mat_fill_mode(aoclsparse_mat_descr descr,
                                             aoclsparse_fill_mode fill_mode);

/*! \ingroup aux_module
 *  \brief Get the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_get_mat_fill_mode returns the matrix fill mode of a matrix descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns    \ref aoclsparse_fill_mode_lower or \ref aoclsparse_fill_mode_upper.
 */
DLL_PUBLIC
aoclsparse_fill_mode aoclsparse_get_mat_fill_mode(const aoclsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Specify the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_set_mat_diag_type sets the matrix diagonal type of a matrix
 *  descriptor. Valid diagonal types are \ref aoclsparse_diag_type_unit or
 *  \ref aoclsparse_diag_type_non_unit.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  diag_type   \ref aoclsparse_diag_type_unit or \ref aoclsparse_diag_type_non_unit.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p descr pointer is invalid.
 *  \retval aoclsparse_status_invalid_value \p diag_type is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_set_mat_diag_type(aoclsparse_mat_descr descr,
                                             aoclsparse_diag_type diag_type);

/*! \ingroup aux_module
 *  \brief Get the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p aoclsparse_get_mat_diag_type returns the matrix diagonal type of a matrix
 *  descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref aoclsparse_diag_type_unit or \ref aoclsparse_diag_type_non_unit.
 */
DLL_PUBLIC
aoclsparse_diag_type aoclsparse_get_mat_diag_type(const aoclsparse_mat_descr descr);

/*! \ingroup aux_module
 *  \brief Update a \p CSR matrix structure
 *
 *  \details
 *  \p aoclsparse_create_mat_csr updates a structure that holds the matrix in \p CSR
 *  storage format. It should be destroyed at the end using aoclsparse_destroy_mat_csr().
 *
 *  @param[inout]
 *  csr the pointer to the CSR sparse matrix.
 *  @param[in]
 *  base    \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  csr_nnz     number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p csr pointer is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_create_mat_csr(aoclsparse_mat_csr &csr,
                                         aoclsparse_index_base   base,
                                         aoclsparse_int          m,
                                         aoclsparse_int          n,
                                         aoclsparse_int          csr_nnz,
                                         aoclsparse_int*         csr_row_ptr,
                                         aoclsparse_int*         csr_col_ind,
                                         void*                   csr_val);


DLL_PUBLIC
aoclsparse_status aoclsparse_create_dcsr(aoclsparse_matrix &mat,
                    aoclsparse_index_base   base,
                    aoclsparse_int          M,
                    aoclsparse_int          N,
                    aoclsparse_int          csr_nnz,
                    aoclsparse_int*         csr_row_ptr,
                    aoclsparse_int*         csr_col_ptr,
                    void*                   csr_val);


DLL_PUBLIC
aoclsparse_status aoclsparse_create_ell(aoclsparse_matrix mat,
		            aoclsparse_int          ell_width,
                    aoclsparse_int*         ell_cold_ind,
                    void*                   ell_val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_ell_csr_hyb(aoclsparse_matrix mat,
                    aoclsparse_int          ell_width,
                    aoclsparse_int          ell_m,
                    aoclsparse_int*         ell_cold_ind,
                    aoclsparse_int*         csr_row_id_map,
                    void*                   ell_val);                    


/*! \ingroup aux_module
 *  \brief Export a \p CSR matrix structure
 *
 *  \details
 *  \p aoclsparse_export_mat_csr exports a structure that holds the matrix in \p CSR
 *  storage format.
 *
 *  @param[in]
 *  csr the pointer to the CSR sparse matrix.
 *  @param[out]
 *  base    \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *  @param[out]
 *  M           number of rows of the sparse CSR matrix.
 *  @param[out]
 *  N           number of columns of the sparse CSR matrix.
 *  @param[out]
 *  csr_nnz     number of non-zero entries of the sparse CSR matrix.
 *  @param[out]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[out]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[out]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p csr pointer is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_export_mat_csr(aoclsparse_mat_csr &csr,
           aoclsparse_index_base   *base,
           aoclsparse_int          *M,
           aoclsparse_int          *N,
           aoclsparse_int          *csr_nnz,
           aoclsparse_int*         *csr_row_ptr,
           aoclsparse_int*         *csr_col_ind,
           void*                   *csr_val);


/*! \ingroup aux_module
 *  \brief Destroy a \p CSR matrix structure
 *
 *  \details
 *  \p aoclsparse_destroy_mat_csr destroys a structure that holds the matrix in \p CSR
 *  storage format.
 *
 *  @param[in]
 *  csr the pointer to the CSR sparse matrix.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p csr pointer is invalid.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_destroy_mat_csr(aoclsparse_mat_csr csr);
#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_AUXILIARY_H_

