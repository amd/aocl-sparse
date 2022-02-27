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
 *  \brief aoclsparse_convert.h provides Sparse Format conversion Subprograms
 */
#ifndef AOCLSPARSE_CONVERT_H_
#define AOCLSPARSE_CONVERT_H_

#include "aoclsparse_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse ELL matrix
*
*  \details
*  \p aoclsparse_csr2ell_width computes the maximum of the per row non-zero elements
*  over all rows, the ELL \p width, for a given CSR matrix.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[out]
*  ell_width   pointer to the number of non-zero elements per row in ELL storage
*              format.
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size \p m is invalid.
*  \retval     aoclsparse_status_invalid_pointer  \p csr_row_ptr, or
*              \p ell_width pointer is invalid.
*  \retval     aoclsparse_status_internal_error an internal error occurred.
*/
DLL_PUBLIC
aoclsparse_status aoclsparse_csr2ell_width(
                aoclsparse_int       m,
                aoclsparse_int       nnz,
                const aoclsparse_int *csr_row_ptr,
                aoclsparse_int       *ell_width);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse ELLPACK matrix
*
*  \details
*  \p aoclsparse_csr2ell converts a CSR matrix into an ELL matrix. It is assumed,
*  that \p ell_val and \p ell_col_ind are allocated. Allocation size is computed by the
*  number of rows times the number of ELL non-zero elements per row, such that
*  \f$\text{nnz}_{\text{ELL}} = m \cdot \text{ell_width}\f$. The number of ELL
*  non-zero elements per row is obtained by aoclsparse_csr2ell_width().
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  csr_val     array containing the values of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array containing the column indices of the sparse CSR matrix.
*  @param[in]
*  ell_width   number of non-zero elements per row in ELL storage format.
*  @param[out]
*  ell_val     array of \p m times \p ell_width elements of the sparse ELL matrix.
*  @param[out]
*  ell_col_ind array of \p m times \p ell_width elements containing the column indices
*              of the sparse ELL matrix.
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_handle the library context was not initialized.
*  \retval     aoclsparse_status_invalid_size \p m or \p ell_width is invalid.
*  \retval     aoclsparse_status_invalid_pointer \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind,  \p ell_val or
*              \p ell_col_ind pointer is invalid.
*
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_scsr2ell(
                aoclsparse_int       m,
                const aoclsparse_int *csr_row_ptr,
                const aoclsparse_int *csr_col_ind,
                const float          *csr_val,
                aoclsparse_int       *ell_col_ind,
                float                *ell_val,
                aoclsparse_int       ell_width);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsr2ell(
                aoclsparse_int       m,
                const aoclsparse_int *csr_row_ptr,
                const aoclsparse_int *csr_col_ind,
                const double         *csr_val,
                aoclsparse_int       *ell_col_ind,
                double               *ell_val,
                aoclsparse_int       ell_width);
/**@}*/


DLL_PUBLIC
aoclsparse_status aoclsparse_scsr2ellt(
                aoclsparse_int       m,
                const aoclsparse_int *csr_row_ptr,
                const aoclsparse_int *csr_col_ind,
                const float          *csr_val,
                aoclsparse_int       *ell_col_ind,
                float                *ell_val,
                aoclsparse_int       ell_width);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsr2ellt(
                aoclsparse_int       m,
                const aoclsparse_int *csr_row_ptr,
                const aoclsparse_int *csr_col_ind,
                const double         *csr_val,
                aoclsparse_int       *ell_col_ind,
                double               *ell_val,
                aoclsparse_int       ell_width);



/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse DIA matrix
*
*  \details
*  \p aoclsparse_csr2dia_ndiag computes the number of the diagonals
*  for a given CSR matrix.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of cols of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array containing the column indices of the sparse CSR matrix.
*  @param[out]
*  dia_num_diag   pointer to the number of diagonals with non-zeroes in DIA storage
*              format.
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size \p m is invalid.
*  \retval     aoclsparse_status_invalid_pointer  \p csr_row_ptr, or
*              \p ell_width pointer is invalid.
*  \retval     aoclsparse_status_internal_error an internal error occurred.
*/
DLL_PUBLIC
aoclsparse_status aoclsparse_csr2dia_ndiag(
                aoclsparse_int       m,
                aoclsparse_int       n,
                aoclsparse_int       nnz,
                const aoclsparse_int *csr_row_ptr,
                const aoclsparse_int *csr_col_ind,
                aoclsparse_int       *dia_num_diag);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse DIA matrix
*
*  \details
*  \p aoclsparse_csr2dia converts a CSR matrix into an DIA matrix. It is assumed,
*  that \p dia_val and \p dia_offset are allocated. Allocation size is computed by the
*  number of rows times the number of diagonals. The number of DIA
*  diagonals is obtained by aoclsparse_csr2dia_ndiag().
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of cols of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array containing the column indices of the sparse CSR matrix.
*  @param[in]
*  csr_val     array containing the values of the sparse CSR matrix.
*  @param[in]
*  dia_num_diag number of diagoanls in ELL storage format.
*  @param[out]
*  dia_offset  array of \p dia_num_diag elements containing the diagonal offsets from
*              main diagonal.
*  @param[out]
*  dia_val     array of \p m times \p dia_num_diag elements of the sparse DIA matrix.
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_handle the library context was not initialized.
*  \retval     aoclsparse_status_invalid_size \p m or \p ell_width is invalid.
*  \retval     aoclsparse_status_invalid_pointer \p csr_val,
*              \p csr_row_ptr, \p csr_col_ind,  \p ell_val or
*              \p ell_col_ind pointer is invalid.
*
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_scsr2dia(
            aoclsparse_int       m,
            aoclsparse_int       n,
            const aoclsparse_int *csr_row_ptr,
            const aoclsparse_int *csr_col_ind,
            const float          *csr_val,
            aoclsparse_int       dia_num_diag,
            aoclsparse_int       *dia_offset,
            float                *dia_val);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsr2dia(
            aoclsparse_int       m,
            aoclsparse_int       n,
            const aoclsparse_int *csr_row_ptr,
            const aoclsparse_int *csr_col_ind,
            const double         *csr_val,
            aoclsparse_int       dia_num_diag,
            aoclsparse_int       *dia_offset,
            double               *dia_val);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  aoclsparse_csr2bsr_nnz computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  BSR matrix given a sparse CSR matrix as input.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*
*  @param[in]
*  csr_row_ptr integer array containing \p m+1 elements that point to the start of each row of the CSR matrix
*
*  @param[in]
*  csr_col_ind integer array of the column indices for each non-zero element in the CSR matrix
*
*  @param[in]
*  block_dim   the block dimension of the BSR matrix. Between 1 and min(m, n)
*
*  @param[out]
*  bsr_row_ptr integer array containing \p mb+1 elements that point to the start of each block row of the BSR matrix
*
*  @param[out]
*  bsr_nnz     total number of nonzero elements in device or host memory.
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size \p m or \p n or \p block_dim is invalid.
*  \retval     aoclsparse_status_invalid_pointer \p csr_row_ptr or \p csr_col_ind or \p bsr_row_ptr or \p bsr_nnz
*              pointer is invalid.
*/
DLL_PUBLIC
aoclsparse_status aoclsparse_csr2bsr_nnz(
            aoclsparse_int       m,
            aoclsparse_int       n,
            const aoclsparse_int *csr_row_ptr,
            const aoclsparse_int *csr_col_ind,
            aoclsparse_int       block_dim,
            aoclsparse_int       *bsr_row_ptr,
            aoclsparse_int       *bsr_nnz);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse BSR matrix
*
*  \details
*  \p aoclsparse_csr2bsr converts a CSR matrix into a BSR matrix. It is assumed,
*  that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
*  for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
*  the BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
*  \p csr2bsr_nnz() which also fills in \p bsr_row_ptr.
*
*  @param[in]
*  m            number of rows in the sparse CSR matrix.
*  @param[in]
*  n            number of columns in the sparse CSR matrix.
*  @param[in]
*  csr_val      array of \p nnz elements containing the values of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr  array of \p m+1 elements that point to the start of every row of the
*               sparse CSR matrix.
*  @param[in]
*  csr_col_ind  array of \p nnz elements containing the column indices of the sparse CSR matrix.
*  @param[in]
*  block_dim    size of the blocks in the sparse BSR matrix.
*  @param[out]
*  bsr_val      array of \p nnzb*block_dim*block_dim containing the values of the sparse BSR matrix.
*  @param[out]
*  bsr_row_ptr  array of \p mb+1 elements that point to the start of every block row of the
*               sparse BSR matrix.
*  @param[out]
*  bsr_col_ind  array of \p nnzb elements containing the block column indices of the sparse BSR matrix.
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size \p m or \p n or \p block_dim is invalid.
*  \retval     aoclsparse_status_invalid_pointer \p bsr_val,
*              \p bsr_row_ptr, \p bsr_col_ind, \p csr_val, \p csr_row_ptr or
*              \p csr_col_ind pointer is invalid.
*
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_scsr2bsr(
            aoclsparse_int       m,
            aoclsparse_int       n,
            const float          *csr_val,
            const aoclsparse_int *csr_row_ptr,
            const aoclsparse_int *csr_col_ind,
            aoclsparse_int       block_dim,
            float                *bsr_val,
            aoclsparse_int       *bsr_row_ptr,
            aoclsparse_int       *bsr_col_ind);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsr2bsr(
            aoclsparse_int       m,
            aoclsparse_int       n,
            const double         *csr_val,
            const aoclsparse_int *csr_row_ptr,
            const aoclsparse_int *csr_col_ind,
            aoclsparse_int       block_dim,
            double               *bsr_val,
            aoclsparse_int       *bsr_row_ptr,
            aoclsparse_int       *bsr_col_ind);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse CSC matrix
*
*  \details
*  \p aoclsparse_csr2csc converts a CSR matrix into a CSC matrix. \p aoclsparse_csr2csc
*  can also be used to convert a CSC matrix into a CSR matrix.
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  @param[in]
*  m           number of rows of the sparse CSR matrix.
*  @param[in]
*  n           number of columns of the sparse CSR matrix.
*  @param[in]
*  nnz         number of non-zero entries of the sparse CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements of the sparse CSR matrix.
*  @param[in]
*  csr_row_ptr array of \p m+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[out]
*  csc_val     array of \p nnz elements of the sparse CSC matrix.
*  @param[out]
*  csc_row_ind array of \p nnz elements containing the row indices of the sparse CSC
*              matrix.
*  @param[out]
*  csc_col_ptr array of \p n+1 elements that point to the start of every column of the
*              sparse CSC matrix.
*              aoclsparse_csr2csc_buffer_size().
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
*  \retval     aoclsparse_status_invalid_pointer \p csr_val, \p csr_row_ptr,
*              \p csr_col_ind, \p csc_val, \p csc_row_ind, \p csc_col_ptr
*              is invalid.
*
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_scsr2csc(
            aoclsparse_int       m,
            aoclsparse_int       n,
            aoclsparse_int       nnz,
            const aoclsparse_int *csr_row_ptr,
            const aoclsparse_int *csr_col_ind,
            const float          *csr_val,
            aoclsparse_int       *csc_row_ind,
            aoclsparse_int       *csc_col_ptr,
            float                *csc_val);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsr2csc(
            aoclsparse_int       m,
            aoclsparse_int       n,
            aoclsparse_int       nnz,
            const aoclsparse_int *csr_row_ptr,
            const aoclsparse_int *csr_col_ind,
            const double         *csr_val,
            aoclsparse_int       *csc_row_ind,
            aoclsparse_int       *csc_col_ptr,
            double               *csc_val);
/**@}*/

/*! \ingroup conv_module
 *  \brief
 *  This function converts the sparse matrix in CSR format into a dense matrix.
 *
 *  @param[in]
 *  m           number of rows of the dense matrix \p A.
 *
 *  @param[in]
 *  n           number of columns of the dense matrix \p A.
 *
 *  @param[in]
 *  descr       the descriptor of the dense matrix \p A, the supported matrix type is \ref aoclsparse_matrix_type_general and also any valid value of the \ref aoclsparse_index_base.
 *
 *  @param[in]
 *  csr_val     array of nnz ( = \p csr_row_ptr[m] - \p csr_row_ptr[0] ) nonzero elements of matrix \p A.
 *  @param[in]
 *  csr_row_ptr integer array of m+1 elements that contains the start of every row and the end of the last row plus one.
 *  @param[in]
 *  csr_col_ind integer array of nnz ( = \p csr_row_ptr[m] - csr_row_ptr[0] ) column indices of the non-zero elements of matrix \p A.
 *
 *  @param[out]
 *  A           array of dimensions (\p ld, \p n)
 *
 *  @param[out]
 *  ld          leading dimension of dense array \p A.
 *  @param[in]
 *  order       memory layout of a dense matrix \p A.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m or \p n or \p ld is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p A or \p csr_val \p csr_row_ptr or \p csr_col_ind
 *              pointer is invalid.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_scsr2dense(
            aoclsparse_int             m,
            aoclsparse_int             n,
            const aoclsparse_mat_descr descr,
            const float*               csr_val,
            const aoclsparse_int*      csr_row_ptr,
            const aoclsparse_int*      csr_col_ind,
            float*                     A,
            aoclsparse_int             ld,
            aoclsparse_order           order);


DLL_PUBLIC
aoclsparse_status aoclsparse_dcsr2dense(
            aoclsparse_int             m,
            aoclsparse_int             n,
            const aoclsparse_mat_descr descr,
            const double*              csr_val,
            const aoclsparse_int*      csr_row_ptr,
            const aoclsparse_int*      csr_col_ind,
            double*                    A,
            aoclsparse_int             ld,
            aoclsparse_order           order);
/**@}*/
#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_CONVERT_H_

