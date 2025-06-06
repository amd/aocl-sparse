/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.
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

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Get AOCL-Sparse Library version
 *
 *  \details
 *  \returns AOCL-Sparse Library version number
 *  in the format "AOCL-Sparse <major>.<minor>.<patch>"
 */
DLL_PUBLIC
const char *aoclsparse_get_version();

/*! \ingroup aux_module
 *  \brief Set ISA code-path preference
 *  \details
 *  This auxiliary function sets the ISA code-path preference to run
 *  kernels that have more than a single path implementation.
 *  ISA code-path preference set using this API will influence the kernel
 *  selection logic.
 *
 *  @param[in] isa_preference pointer to a const char array specifying the ISA
 *  preference, valid values are: "AVX2", "AVX512", "GENERIC", "ENV". The first
 *  three values are also valid for the \p AOCL_ENABLE_INSTRUCTIONS environmental
 *  variable. While "ENV" is a special value to request to read again the environmental
 *  variable and set the ISA path preference accordingly.
 *
 *  \returns \ref aoclsparse_status.
 *  \retval aoclsparse_status_success ISA preference set successfully.
 *  \retval aoclsparse_status_invalid_input signals that either the \p isa_preference
 *  contains an invalid string or that "ENV" was passed and the environmental
 *  variable \p AOCL_ENABLE_INSTRUCTIONS contained an invalid string.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_enable_instructions(const char isa_preference[]);

/*! \ingroup aux_module
 *  \brief Auxiliary debug utility
 *  \details
 *  This auxiliary function is intended for debug purposes, it reports the global and thread-level
 *  ISA code-path preference and the number of threads to use.
 *
 *  @param[inout] isa_preference pointer to a char array of at least size 20 specifying the global ISA
 *  preference, valid values are: "AVX2", "AVX512", "GENERIC".
 *  @param[inout] num_threads pointer to an integer specifying the number of threads.
 *  @param[inout] tl_isa_preference pointer to a char array of at least size 20 specifying the thread local
 *  ISA preference, valid values are: "AVX2", "AVX512", "GENERIC".
 *  @param[inout] is_isa_updated pointer to a bool specifying if the current ISA preference is different from
 *  the last ISA preference.
  *  @param[inout] arch pointer to a char array of at least size 20 specifying the architecture of the machine
 *
 *  \returns \ref aoclsparse_status.
 *  \retval aoclsparse_status_success ISA preference retrieved successfully.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_debug_get(char            isa_preference[],
                                       aoclsparse_int *num_threads,
                                       char            tl_isa_preference[],
                                       bool           *is_isa_updated,
                                       char            arch[]);

/*! \ingroup aux_module
 *  \brief Create a matrix descriptor
 *  \details
 *  \P{aoclsparse_create_mat_descr} creates a matrix descriptor. It initializes
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
aoclsparse_status aoclsparse_create_mat_descr(aoclsparse_mat_descr *descr);

/*! \ingroup aux_module
 *  \brief Copy a matrix descriptor
 *  \details
 *  \P{aoclsparse_copy_mat_descr} copies a matrix descriptor. Both, source and destination
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
aoclsparse_status aoclsparse_copy_mat_descr(aoclsparse_mat_descr       dest,
                                            const aoclsparse_mat_descr src);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \P{aoclsparse_destroy_mat_descr} destroys a matrix descriptor and releases all
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
 *  \P{aoclsparse_set_mat_index_base} sets the index base of a matrix descriptor. Valid
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
aoclsparse_status aoclsparse_set_mat_index_base(aoclsparse_mat_descr  descr,
                                                aoclsparse_index_base base);

/*! \ingroup aux_module
 *  \brief Get the index base of a matrix descriptor
 *
 *  \details
 *  \P{aoclsparse_get_mat_index_base} returns the index base of a matrix descriptor.
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
 *  \P{aoclsparse_set_mat_type} sets the matrix type of a matrix descriptor. Valid
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
 *  \P{aoclsparse_get_mat_type} returns the matrix type of a matrix descriptor.
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
 *  \P{aoclsparse_set_mat_fill_mode} sets the matrix fill mode of a matrix descriptor.
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
 *  \P{aoclsparse_get_mat_fill_mode} returns the matrix fill mode of a matrix descriptor.
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
 *  \P{aoclsparse_set_mat_diag_type} sets the matrix diagonal type of a matrix
 *  descriptor. Valid diagonal types are \ref aoclsparse_diag_type_unit,
 *  \ref aoclsparse_diag_type_non_unit or \ref aoclsparse_diag_type_zero.
 *
 *  @param[inout]
 *  descr       the matrix descriptor.
 *  @param[in]
 *  diag_type   \ref aoclsparse_diag_type_unit or \ref aoclsparse_diag_type_non_unit or \ref aoclsparse_diag_type_zero.
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
 *  \P{aoclsparse_get_mat_diag_type} returns the matrix diagonal type of a matrix
 *  descriptor.
 *
 *  @param[in]
 *  descr   the matrix descriptor.
 *
 *  \returns \ref aoclsparse_diag_type_unit or \ref aoclsparse_diag_type_non_unit or \ref aoclsparse_diag_type_zero.
 */
DLL_PUBLIC
aoclsparse_diag_type aoclsparse_get_mat_diag_type(const aoclsparse_mat_descr descr);

/*! \brief Set a new value to an existing nonzero in the matrix.
 *
 *  \details
 *  \P{aoclsparse_?set_value} modifies the value of an existing nonzero element specified by its coordinates.
 *  The row and column coordinates need to match the base (0 or 1-base) of the matrix.
 *  The change directly affects user's arrays if the matrix was created using aoclsparse_create_scsr(),
 *  aoclsparse_create_scsc(), aoclsparse_create_scoo() or other variants.
 *
 *  \note   The successful modification invalidates existing optimized data so it is desirable to call
 *          aoclsparse_optimize() once all modifications are performed.
 *
 *  \param[inout] A             The sparse matrix to be modified.
 *  \param[in]     row_idx      The row index of the element to be updated.
 *  \param[in]     col_idx      The column index of the element to be updated.
 *  \param[in]     val          The value to be updated.
 *
 *  \retval aoclsparse_status_success               The operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer       The matrix handler \p A is invalid
 *  \retval aoclsparse_status_invalid_value         The cooridante \p row_idx or \p col_idx is out of matrix bound
 *  \retval aoclsparse_status_wrong_type            Matrix has different data type then the one used in API
 *  \retval aoclsparse_status_not_implemented       Matrix format is not supported for this operation
 *  \retval aoclsparse_status_invalid_index_value   The specified element does not exist in the matrix
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_zset_value(aoclsparse_matrix         A,
                                        aoclsparse_int            row_idx,
                                        aoclsparse_int            col_idx,
                                        aoclsparse_double_complex val);
DLL_PUBLIC
aoclsparse_status aoclsparse_cset_value(aoclsparse_matrix        A,
                                        aoclsparse_int           row_idx,
                                        aoclsparse_int           col_idx,
                                        aoclsparse_float_complex val);
DLL_PUBLIC
aoclsparse_status aoclsparse_dset_value(aoclsparse_matrix A,
                                        aoclsparse_int    row_idx,
                                        aoclsparse_int    col_idx,
                                        double            val);
DLL_PUBLIC
aoclsparse_status aoclsparse_sset_value(aoclsparse_matrix A,
                                        aoclsparse_int    row_idx,
                                        aoclsparse_int    col_idx,
                                        float             val);
/**@}*/

/*! \ingroup aux_module
 *  \brief Creates a new \ref aoclsparse_matrix based on CSR (Compressed Sparse Row) format.
 *
 *  \details
 *  \P{aoclsparse_create_?csr} creates \ref aoclsparse_matrix and initializes it with
 *  input parameters passed. The input arrays are left unchanged by the library except for the call to
 *  aoclsparse_order_mat(), which performs ordering of column indices of the matrix,
 *  or aoclsparse_sset_value(), aoclsparse_supdate_values() and variants, which modify
 *  the values of a nonzero element. To avoid any
 *  changes to the input data, aoclsparse_copy() can be used. To convert any other format to CSR,
 *  aoclsparse_convert_csr() can be used. Matrix should be destroyed at the end using aoclsparse_destroy().
 *
 *  @param[out]
 *  mat the pointer to the CSR sparse matrix allocated in the API.
 *  @param[in]
 *  base    \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *  @param[in]
 *  M           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  N           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz     number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  col_idx array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  val     array of \p nnz elements of the sparse CSR matrix.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer at least one of \p row_ptr, \p col_idx or \p val pointer is NULL.
 *  \retval aoclsparse_status_invalid_size    at least one  of \p M, \p N or \p nnz has a negative value.
 *  \retval aoclsparse_status_invalid_index_value  any \p col_idx value is not within N.
 *  \retval aoclsparse_status_memory_error         memory allocation for matrix failed.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_create_scsr(aoclsparse_matrix    *mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *row_ptr,
                                         aoclsparse_int       *col_idx,
                                         float                *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_dcsr(aoclsparse_matrix    *mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *row_ptr,
                                         aoclsparse_int       *col_idx,
                                         double               *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_ccsr(aoclsparse_matrix        *mat,
                                         aoclsparse_index_base     base,
                                         aoclsparse_int            M,
                                         aoclsparse_int            N,
                                         aoclsparse_int            nnz,
                                         aoclsparse_int           *row_ptr,
                                         aoclsparse_int           *col_idx,
                                         aoclsparse_float_complex *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_zcsr(aoclsparse_matrix         *mat,
                                         aoclsparse_index_base      base,
                                         aoclsparse_int             M,
                                         aoclsparse_int             N,
                                         aoclsparse_int             nnz,
                                         aoclsparse_int            *row_ptr,
                                         aoclsparse_int            *col_idx,
                                         aoclsparse_double_complex *val);

/**@}*/

/*! \ingroup aux_module
 *  \brief Creates a new \ref aoclsparse_matrix based on TCSR (Triangular Compressed Sparse Row) format.
 *
 *  \details
 *  \P{aoclsparse_create_?tcsr} creates \ref aoclsparse_matrix and initializes it with input
 *  parameters passed. Array data must not be modified by the user while matrix is being used
 *  as the pointers are copied, not the data. The input arrays are not modified by the library and
 *  the matrix should be destroyed at the end using aoclsparse_destroy().
 *
 *  TCSR matrix structure holds lower triangular (L) and upper triangular (U) part of the matrix
 *  separately with diagonal (D) elements stored in both the parts. Both triangles (L+D and D+U)
 *  are stored like CSR and assumes partial sorting (L+D and D+U order is followed, but the indices
 *  within L or U group may not be sorted)
 *   - One array with L elements potentially unsorted, followed by D elements in the L+D part for
 *     each row of the matrix.
 *   - Another array with D elements, followed by U elements potentially unsorted in the D+U part
 *     for each row of the matrix.
 *   - Currently TCSR storage format supports only square matrices with full(non-zero) diagonals.
 *
 *  @param[out]   mat         The pointer to the TCSR sparse matrix.
 *  @param[in]    base        \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *  @param[in]    M           Total number of rows in the \p mat.
 *  @param[in]    N           Total number of columns in the \p mat.
 *  @param[in]    nnz         Number of non-zero entries in the \p mat.
 *  @param[in]    row_ptr_L   Array of lower triangular elements that point to the start of every row of
 *                            the \p mat in \p col_idx_L and \p val_L.
 *  @param[in]    row_ptr_U   Array of upper triangular elements that point to the start of every row of
 *                            the \p mat in \p col_idx_U and \p val_U.
 *  @param[in]    col_idx_L   Array of lower triangular elements containing column indices of the \p mat.
 *  @param[in]    col_idx_U   Array of upper triangular elements containing column indices of the \p mat.
 *  @param[in]    val_L       Array of lower triangular elements of the \p mat.
 *  @param[in]    val_U       Array of upper triangular elements of the \p mat.
 *
 *  \retval aoclsparse_status_success               The operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer       Pointer given to API is invalid or nullptr.
 *  \retval aoclsparse_status_invalid_size          M, N, nnz is invalid.
 *  \retval aoclsparse_status_invalid_index_value   Index given for \p mat is out of matrix bounds depending on base given.
 *  \retval aoclsparse_status_invalid_value         The cooridante \p row_ptr or \p col_idx is out of matrix bound or \p mat
 *                                                  has duplicate diagonals or \p mat does not have full diagonals.
 *  \retval aoclsparse_status_unsorted_input        The \p mat is unsorted. It supports only fully sorted and partially sorted
 *                                                  matrix as input. The lower triangular part must not contain U elements, the
 *                                                  upper triangular part must not contain L elements, and the position of the
 *                                                  diagonal element must not be altered.
 *  \retval aoclsparse_status_memory_error          Memory allocation for matrix failed.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_create_stcsr(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr_L,
                                          aoclsparse_int             *row_ptr_U,
                                          aoclsparse_int             *col_idx_L,
                                          aoclsparse_int             *col_idx_U,
                                          float                      *val_L,
                                          float                      *val_U);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_dtcsr(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr_L,
                                          aoclsparse_int             *row_ptr_U,
                                          aoclsparse_int             *col_idx_L,
                                          aoclsparse_int             *col_idx_U,
                                          double                     *val_L,
                                          double                     *val_U);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_ctcsr(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr_L,
                                          aoclsparse_int             *row_ptr_U,
                                          aoclsparse_int             *col_idx_L,
                                          aoclsparse_int             *col_idx_U,
                                          aoclsparse_float_complex   *val_L,
                                          aoclsparse_float_complex   *val_U);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_ztcsr(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr_L,
                                          aoclsparse_int             *row_ptr_U,
                                          aoclsparse_int             *col_idx_L,
                                          aoclsparse_int             *col_idx_U,
                                          aoclsparse_double_complex  *val_L,
                                          aoclsparse_double_complex  *val_U);

/**@}*/

/*! \ingroup aux_module
 *  \brief Creates a new \ref aoclsparse_matrix based on COO (Co-ordinate format).
 *
 *  \details
 *  \P{aoclsparse_create_?coo} creates \ref aoclsparse_matrix and initializes it with
 *  input parameters passed. Array data must not be modified by the user while matrix is alive as
 *  the pointers are copied, not the data. The input arrays are left unchanged
 *  by the library except for the call to aoclsparse_sset_value(),
 *  aoclsparse_supdate_values() and variants, which modify the value of
 *  a nonzero element. Matrix should be destroyed at the end using aoclsparse_destroy().
 *
 *  @param[inout] mat       the pointer to the COO sparse matrix.
 *  @param[in]    base      \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one
 *                          depending on whether the index first element starts from 0 or 1.
 *  @param[in]    M         total number of rows of the sparse COO matrix.
 *  @param[in]    N         total number of columns of the sparse COO matrix.
 *  @param[in]    nnz       number of non-zero entries of the sparse COO matrix.
 *  @param[in]    row_ind   array of \p nnz elements that point to the row of the element in co-ordinate Format.
 *  @param[in]    col_ind   array of \p nnz elements that point to the column of the element in co-ordinate Format.
 *  @param[in]    val       array of \p nnz elements of the sparse COO matrix.
 *
 *  \retval aoclsparse_status_success               the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer       pointer given to API is invalid or nullptr.
 *  \retval aoclsparse_status_invalid_size          \p coo dimension of matrix or non-zero elements is invalid.
 *  \retval aoclsparse_status_invalid_index_value   index given for \p coo is out of matrix bounds depending on base given
 *  \retval aoclsparse_status_memory_error          memory allocation for matrix failed.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_create_scoo(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         float                      *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_dcoo(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         double                     *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_ccoo(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         aoclsparse_float_complex   *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_zcoo(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         aoclsparse_double_complex  *val);
/**@}*/

/*! \brief Set new values to all existing nonzero element in the matrix.
 *
 *  \details
 *  \P{aoclsparse_?update_values} overwrites all existing nonzeros in the matrix
 *  with the new values provided in \p val array. The order of elements must
 *  match the order in the matrix. That would be either the order at the
 *  creation of the matrix or the sorted order if aoclsparse_order_mat() has been
 *  called.
 *  The change directly affects user's arrays if the matrix was created using aoclsparse_create_scsr(),
 *  aoclsparse_create_scsc(), aoclsparse_create_scoo() or other variants.
 *
 *  \note   The successful update invalidates existing optimized data so it is desirable to call
 *          aoclsparse_optimize() once all modifications are performed.
 *
 *  \param[inout]  A         The sparse matrix to be modified.
 *  \param[in]     len       Length of the \p val array and the number of nonzeros in the matrix.
 *  \param[in]     val       Array with the values to be copied.
 *
 *  \retval aoclsparse_status_success           The operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer   The matrix \p A is invalid or \p val in NULL
 *  \retval aoclsparse_status_invalid_size      \p len is not equal to nnz of matrix
 *  \retval aoclsparse_status_wrong_type        Matrix has different data type then the one used in API
 *  \retval aoclsparse_status_not_implemented   Matrix format is not supported for this operation
 *
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_zupdate_values(aoclsparse_matrix          A,
                                            aoclsparse_int             len,
                                            aoclsparse_double_complex *val);
DLL_PUBLIC
aoclsparse_status aoclsparse_cupdate_values(aoclsparse_matrix         A,
                                            aoclsparse_int            len,
                                            aoclsparse_float_complex *val);
DLL_PUBLIC
aoclsparse_status aoclsparse_dupdate_values(aoclsparse_matrix A, aoclsparse_int len, double *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_supdate_values(aoclsparse_matrix A, aoclsparse_int len, float *val);
/**@}*/

/*! \ingroup aux_module
 *  \brief Export a \p CSR matrix
 *
 *  \details
 *  \P{aoclsparse_export_?csr} exposes the components defining the
 *  \p CSR matrix in \p mat structure by copying out the data pointers. No additional
 *  memory is allocated. User should not modify the arrays and once aoclsparse_destroy()
 *  is called to free \p mat, these arrays will become inaccessible. If the matrix is
 *  not in \p CSR format, an error is obtained. aoclsparse_convert_csr() can be used
 *  to convert non-CSR format to \p CSR format.
 *
 *  @param[in]
 *  mat     the pointer to the CSR sparse matrix.
 *  @param[out]
 *  base    \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *  @param[out]
 *  m       number of rows of the sparse CSR matrix.
 *  @param[out]
 *  n       number of columns of the sparse CSR matrix.
 *  @param[out]
 *  nnz     number of non-zero entries of the sparse CSR matrix.
 *  @param[out]
 *  row_ptr array of \p m +1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[out]
 *  col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[out]
 *  val     array of \p nnz elements of the sparse CSR matrix.
 *
 *  \retval aoclsparse_status_success           the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer   \p mat or any of the output arguments are NULL.
 *  \retval aoclsparse_status_invalid_value     \p mat is not in CSR format.
 *  \retval aoclsparse_status_wrong_type        data type of \p mat does not match the function.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_export_scsr(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **row_ptr,
                                         aoclsparse_int        **col_ind,
                                         float                 **val);
DLL_PUBLIC
aoclsparse_status aoclsparse_export_dcsr(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **row_ptr,
                                         aoclsparse_int        **col_ind,
                                         double                **val);
DLL_PUBLIC
aoclsparse_status aoclsparse_export_ccsr(const aoclsparse_matrix    mat,
                                         aoclsparse_index_base     *base,
                                         aoclsparse_int            *m,
                                         aoclsparse_int            *n,
                                         aoclsparse_int            *nnz,
                                         aoclsparse_int           **row_ptr,
                                         aoclsparse_int           **col_ind,
                                         aoclsparse_float_complex **val);
DLL_PUBLIC
aoclsparse_status aoclsparse_export_zcsr(const aoclsparse_matrix     mat,
                                         aoclsparse_index_base      *base,
                                         aoclsparse_int             *m,
                                         aoclsparse_int             *n,
                                         aoclsparse_int             *nnz,
                                         aoclsparse_int            **row_ptr,
                                         aoclsparse_int            **col_ind,
                                         aoclsparse_double_complex **val);
/**@}*/

/*! \ingroup aux_module
 *  \brief Destroy a sparse matrix structure
 *
 *  \details
 *  \p aoclsparse_destroy destroys a structure that holds matrix \p mat.
 *  @param[in]
 *  mat the pointer to the sparse matrix.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p matrix structure pointer is invalid.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_destroy(aoclsparse_matrix *mat);
/**@}*/

/*! \ingroup aux_module
 *  \brief Creates a new \ref aoclsparse_matrix based on CSC (Compressed Sparse Column) format.
 *
 *  \details
 *  \P{aoclsparse_create_?csc} creates \ref aoclsparse_matrix and initializes it with
 *  input parameters passed. The input arrays are left unchanged by the library except for the call to
 *  aoclsparse_order_mat(), which performs ordering of row indices of the matrix,
 *  or aoclsparse_sset_value(), aoclsparse_supdate_values() and variants, which
 *  modify the value of a nonzero element. To avoid any
 *  changes to the input data, aoclsparse_copy() can be used. Matrix should be destroyed at the end
 *  using aoclsparse_destroy().
 *
 *  @param[inout]
 *  mat         the pointer to the CSC sparse matrix allocated in the API.
 *  @param[in]
 *  base        \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *  @param[in]
 *  M           number of rows of the sparse CSC matrix.
 *  @param[in]
 *  N           number of columns of the sparse CSC matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSC matrix.
 *  @param[in]
 *  col_ptr     array of \p n +1 elements that points to the start of every column
 *              in row_idx array of the sparse CSC matrix.
 *  @param[in]
 *  row_idx     array of \p nnz elements containing the row indices of the sparse
 *              CSC matrix.
 *  @param[in]
 *  val         array of \p nnz elements of the sparse CSC matrix.
 *
 *  \retval aoclsparse_status_success              the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer      \p col_ptr, \p row_idx or \p val pointer is NULL.
 *  \retval aoclsparse_status_invalid_size         \p M, \p N or \p nnz are negative values.
 *  \retval aoclsparse_status_invalid_index_value  any \p row_idx value is not within M.
 *  \retval aoclsparse_status_memory_error         memory allocation for matrix failed.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_create_scsc(aoclsparse_matrix    *mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *col_ptr,
                                         aoclsparse_int       *row_idx,
                                         float                *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_dcsc(aoclsparse_matrix    *mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *col_ptr,
                                         aoclsparse_int       *row_idx,
                                         double               *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_ccsc(aoclsparse_matrix        *mat,
                                         aoclsparse_index_base     base,
                                         aoclsparse_int            M,
                                         aoclsparse_int            N,
                                         aoclsparse_int            nnz,
                                         aoclsparse_int           *col_ptr,
                                         aoclsparse_int           *row_idx,
                                         aoclsparse_float_complex *val);

DLL_PUBLIC
aoclsparse_status aoclsparse_create_zcsc(aoclsparse_matrix         *mat,
                                         aoclsparse_index_base      base,
                                         aoclsparse_int             M,
                                         aoclsparse_int             N,
                                         aoclsparse_int             nnz,
                                         aoclsparse_int            *col_ptr,
                                         aoclsparse_int            *row_idx,
                                         aoclsparse_double_complex *val);
/**@}*/

/*! \brief Export a \p COO matrix.
 *
 *  \details
 *  \P{aoclsparse_export_?coo} exposes the components defining the
 *  \p COO matrix in \p mat structure by copying out the data pointers. No additional
 *  memory is allocated. User should not modify the arrays and once aoclsparse_destroy()
 *  is called to free \p mat, these arrays will become inaccessible. If the matrix is
 *  not in \p COO format, an error is obtained.
 *
 *  @param[in] mat          the pointer to the COO sparse matrix.
 *  @param[out] base        \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *  @param[out] m           number of rows of the sparse COO matrix.
 *  @param[out] n           number of columns of the sparse COO matrix.
 *  @param[out] nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[out] row_ptr     array of \p nnz elements containing the row indices of the sparse
 *                          COO matrix.
 *  @param[out] col_ptr     array of \p nnz elements containing the column indices of the sparse
 *                          COO matrix.
 *  @param[out] val         array of \p nnz elements of the sparse COO matrix.
 *
 *  \retval aoclsparse_status_success           the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer   \p mat or any of the output arguments are NULL.
 *  \retval aoclsparse_status_invalid_value     \p mat is not in COO format.
 *  \retval aoclsparse_status_wrong_type        data type of \p mat does not match the function.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_export_zcoo(const aoclsparse_matrix     mat,
                                         aoclsparse_index_base      *base,
                                         aoclsparse_int             *m,
                                         aoclsparse_int             *n,
                                         aoclsparse_int             *nnz,
                                         aoclsparse_int            **row_ptr,
                                         aoclsparse_int            **col_ptr,
                                         aoclsparse_double_complex **val);

DLL_PUBLIC
aoclsparse_status aoclsparse_export_ccoo(const aoclsparse_matrix    mat,
                                         aoclsparse_index_base     *base,
                                         aoclsparse_int            *m,
                                         aoclsparse_int            *n,
                                         aoclsparse_int            *nnz,
                                         aoclsparse_int           **row_ptr,
                                         aoclsparse_int           **col_ptr,
                                         aoclsparse_float_complex **val);

DLL_PUBLIC
aoclsparse_status aoclsparse_export_dcoo(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **row_ptr,
                                         aoclsparse_int        **col_ptr,
                                         double                **val);

DLL_PUBLIC
aoclsparse_status aoclsparse_export_scoo(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **row_ptr,
                                         aoclsparse_int        **col_ptr,
                                         float                 **val);
/**@}*/

/*! \ingroup aux_module
 *  \brief Creates a copy of source \ref aoclsparse_matrix
 *
 *  \details
 *  \P{aoclsparse_copy} creates a deep copy of source \ref aoclsparse_matrix (hints and optimized data
 *  are not copied). Matrix should be destroyed using aoclsparse_destroy(). aoclsparse_convert_csr()
 *  can also be used to create a copy of the source matrix while converting it in CSR format.
 *
 *  @param[in]
 *  src         the source \ref aoclsparse_matrix to copy.
 *  @param[in]
 *  descr       the source matrix descriptor, this argument is reserved for future
 *              releases and it will not be referenced.
 *  @param[out]
 *  dest        pointer to the newly allocated copied \ref aoclsparse_matrix.
 *
 *  \retval aoclsparse_status_success           the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer   \p src, \p dest or internal pointers are invalid.
 *                                              or dest points to src.
 *  \retval aoclsparse_status_memory_error      memory allocation for matrix failed.
 *  \retval aoclsparse_status_invalid_value     \p src matrix type is invalid.
 *  \retval aoclsparse_status_wrong_type        \p src matrix data type is invalid.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_copy(const aoclsparse_matrix    src,
                                  const aoclsparse_mat_descr descr,
                                  aoclsparse_matrix         *dest);
/**@}*/

/*! \ingroup aux_module
 *  \brief Performs ordering of index array of the matrix
 *
 *  \details
 *  \P{aoclsparse_order} orders column indices within a row for matrix in CSR format and row indices
 *  within a column for CSC format. It also adjusts value array accordingly. Ordering is implemented
 *  only for CSR and CSC format. aoclsparse_copy() can be used to get exact copy of data
 *  aoclsparse_convert_csr() can be used to convert any format to CSR. Matrix should be destroyed
 *  at the end using aoclsparse_destroy().
 *
 *  @param[inout]
 *  mat     pointer to matrix in either CSR or CSC format
 *
 *  \retval aoclsparse_status_success              the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer      \p mat pointer is invalid.
 *  \retval aoclsparse_status_memory_error         internal memory allocation failed.
 *  \retval aoclsparse_status_not_implemented      matrix is not in CSR format.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_order_mat(aoclsparse_matrix mat);
/**@}*/

/*! \ingroup aux_module
 *  \brief Export \p CSC matrix
 *
 *  \details
 *  \P{aoclsparse_export_?csc} exposes the components defining the
 *  \p CSC matrix in \p mat structure by copying out the data pointers. No additional
 *  memory is allocated. User should not modify the arrays and once aoclsparse_destroy()
 *  is called to free \p mat, these arrays will become inaccessible. If the matrix is
 *  not in \p CSC format, an error is obtained.
 *
 *  @param[in]
 *  mat         the pointer to the CSC sparse matrix.
 *  @param[out]
 *  base        \ref aoclsparse_index_base_zero or \ref aoclsparse_index_base_one.
 *  @param[out]
 *  m           number of rows of the sparse CSC matrix.
 *  @param[out]
 *  n           number of columns of the sparse CSC matrix.
 *  @param[out]
 *  nnz         number of non-zero entries of the sparse CSC matrix.
 *  @param[out]
 *  col_ptr     array of \p n+1 elements that point to the start
 *              of every col of the sparse CSC matrix.
 *  @param[out]
 *  row_ind     array of \p nnz elements containing the row indices of the sparse
 *              CSC matrix.
 *  @param[out]
 *  val         array of \p nnz elements of the sparse CSC matrix.
 *
 *  \retval aoclsparse_status_success           the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer   \p mat or any of the output arguments are invalid.
 *  \retval aoclsparse_status_invalid_value     \p mat is not in CSC format.
 *  \retval aoclsparse_status_wrong_type        data type of \p mat does not match the function data type.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_export_scsc(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **col_ptr,
                                         aoclsparse_int        **row_ind,
                                         float                 **val);
DLL_PUBLIC
aoclsparse_status aoclsparse_export_dcsc(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **col_ptr,
                                         aoclsparse_int        **row_ind,
                                         double                **val);
DLL_PUBLIC
aoclsparse_status aoclsparse_export_ccsc(const aoclsparse_matrix    mat,
                                         aoclsparse_index_base     *base,
                                         aoclsparse_int            *m,
                                         aoclsparse_int            *n,
                                         aoclsparse_int            *nnz,
                                         aoclsparse_int           **col_ptr,
                                         aoclsparse_int           **row_ind,
                                         aoclsparse_float_complex **val);
DLL_PUBLIC
aoclsparse_status aoclsparse_export_zcsc(const aoclsparse_matrix     mat,
                                         aoclsparse_index_base      *base,
                                         aoclsparse_int             *m,
                                         aoclsparse_int             *n,
                                         aoclsparse_int             *nnz,
                                         aoclsparse_int            **col_ptr,
                                         aoclsparse_int            **row_ind,
                                         aoclsparse_double_complex **val);

/**@}*/

DLL_PUBLIC
aoclsparse_int aoclsparse_debug_dispatcher(const char                  name[],
                                           aoclsparse_matrix_data_type dt,
                                           aoclsparse_int              kid);

DLL_PUBLIC
aoclsparse_int aoclsparse_is_avx512_build();

#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_AUXILIARY_H_
