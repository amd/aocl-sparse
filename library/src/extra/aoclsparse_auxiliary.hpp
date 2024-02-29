/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#ifndef AOCLSPARSE_AUXILIARY_HPP
#define AOCLSPARSE_AUXILIARY_HPP

#include "aoclsparse_mat_structures.h"
#include "aoclsparse_analysis.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_utils.hpp"

#include <cmath>
#include <limits>

// Ignore compiler warning from BLIS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wunused-function"
// The fix order of BLIS/Flame headers
// clang-format off
#include "blis.h"
#include "cblas.hh"
#include "FLAME.h"
// clang-format on
// Restore
#pragma GCC diagnostic pop

/* Check that the size of integers in the used libraries is OK. */
static_assert(
    sizeof(f77_int) == sizeof(aoclsparse_int),
    "Error: Incompatible size of ints in blis. Using wrong header or compilation of the library?");
static_assert(
    sizeof(integer) == sizeof(aoclsparse_int),
    "Error: Incompatible size of ints in flame. Using wrong header or compilation of the library?");

aoclsparse_status aoclsparse_destroy_mv(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_2m(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_ilu(_aoclsparse_ilu *ilu_info);
aoclsparse_status aoclsparse_destroy_opt_csr(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_csc(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_coo(aoclsparse_matrix A);

void aoclsparse_init_mat(aoclsparse_matrix             A,
                         aoclsparse_index_base         base,
                         aoclsparse_int                M,
                         aoclsparse_int                N,
                         aoclsparse_int                nnz,
                         aoclsparse_matrix_format_type matrix_type);

/********************************************************************************
 * \brief aoclsparse_create_csr_t sets the sparse matrix in the CSR format
 * for any data type
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_create_csr_t(aoclsparse_matrix    *mat,
                                          aoclsparse_index_base base,
                                          aoclsparse_int        M,
                                          aoclsparse_int        N,
                                          aoclsparse_int        nnz,
                                          aoclsparse_int       *row_ptr,
                                          aoclsparse_int       *col_idx,
                                          T                    *val)
{
    aoclsparse_status status;
    if(!mat)
        return aoclsparse_status_invalid_pointer;
    *mat = nullptr;
    // Validate the input parameters
    aoclsparse_matrix_sort mat_sort;
    bool                   mat_fulldiag;
    if((status = aoclsparse_mat_check_internal(
            M, N, nnz, row_ptr, col_idx, val, shape_general, base, mat_sort, mat_fulldiag, nullptr))
       != aoclsparse_status_success)
    {
        return status;
    }
    try
    {
        *mat = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(*mat, base, M, N, nnz, aoclsparse_csr_mat);
    (*mat)->val_type            = get_data_type<T>();
    (*mat)->mat_type            = aoclsparse_csr_mat;
    (*mat)->csr_mat.csr_row_ptr = row_ptr;
    (*mat)->csr_mat.csr_col_ptr = col_idx;
    (*mat)->csr_mat.csr_val     = val;
    (*mat)->csr_mat_is_users    = true;
    (*mat)->sort                = mat_sort;
    (*mat)->fulldiag            = mat_fulldiag;

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_create_csc_t(aoclsparse_matrix    *mat,
                                          aoclsparse_index_base base,
                                          aoclsparse_int        M,
                                          aoclsparse_int        N,
                                          aoclsparse_int        nnz,
                                          aoclsparse_int       *col_ptr,
                                          aoclsparse_int       *row_idx,
                                          T                    *val);

template <typename T>
aoclsparse_status aoclsparse_create_coo_t(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr,
                                          aoclsparse_int             *col_ptr,
                                          T                          *val);

template <typename T>
aoclsparse_status aoclsparse_update_values_t(aoclsparse_matrix A, aoclsparse_int len, T *val)
{
    if(A == nullptr || val == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(len != A->nnz)
        return aoclsparse_status_invalid_size;

    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    T *A_val = nullptr;

    switch(A->input_format)
    {
    case aoclsparse_csr_mat:
        A_val = reinterpret_cast<T *>(A->csr_mat.csr_val);
        memcpy(A_val, val, len * sizeof(T));
        break;
    case aoclsparse_csc_mat:
        A_val = reinterpret_cast<T *>(A->csc_mat.val);
        memcpy(A_val, val, len * sizeof(T));
        break;
    case aoclsparse_coo_mat:
        A_val = reinterpret_cast<T *>(A->coo_mat.val);
        memcpy(A_val, val, len * sizeof(T));
        break;
    default:
        return aoclsparse_status_not_implemented;
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_copy_csc(aoclsparse_int                n,
                                      aoclsparse_int                nnz,
                                      const struct _aoclsparse_csc *src,
                                      struct _aoclsparse_csc       *dest);

template <typename T>
aoclsparse_status aoclsparse_copy_coo(aoclsparse_int                nnz,
                                      const struct _aoclsparse_coo *src,
                                      struct _aoclsparse_coo       *dest);

template <typename T>
aoclsparse_status aoclsparse_copy_mat(const aoclsparse_matrix src, aoclsparse_matrix dest);

template <typename T>
aoclsparse_status aoclsparse_sort_mat(aoclsparse_matrix mat);

template <typename T>
aoclsparse_status aoclsparse_export_csr_t(const aoclsparse_matrix csr,
                                          aoclsparse_index_base  *base,
                                          aoclsparse_int         *m,
                                          aoclsparse_int         *n,
                                          aoclsparse_int         *nnz,
                                          aoclsparse_int        **row_ptr,
                                          aoclsparse_int        **col_ind,
                                          T                     **val);

template <typename T>
aoclsparse_status aoclsparse_export_csc_t(const aoclsparse_matrix mat,
                                          aoclsparse_index_base  *base,
                                          aoclsparse_int         *m,
                                          aoclsparse_int         *n,
                                          aoclsparse_int         *nnz,
                                          aoclsparse_int        **col_ptr,
                                          aoclsparse_int        **row_idx,
                                          T                     **val);

/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the aoclsparse coo matrix.
 * Use this routine to export the contents of this structure
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_export_coo_t(const aoclsparse_matrix mat,
                                          aoclsparse_index_base  *base,
                                          aoclsparse_int         *m,
                                          aoclsparse_int         *n,
                                          aoclsparse_int         *nnz,
                                          aoclsparse_int        **row_ptr,
                                          aoclsparse_int        **col_ptr,
                                          T                     **val)
{
    if((mat == nullptr) || (base == nullptr) || (m == nullptr) || (n == nullptr) || (nnz == nullptr)
       || (row_ptr == nullptr) || (col_ptr == nullptr) || (val == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    // check if data type of matrix is same as requested
    if(mat->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }
    if((mat->coo_mat.row_ind != nullptr) && (mat->coo_mat.col_ind != nullptr)
       && (mat->coo_mat.val != nullptr))
    {
        *row_ptr = mat->coo_mat.row_ind;
        *col_ptr = mat->coo_mat.col_ind;
        *val     = static_cast<T *>(mat->coo_mat.val);
    }
    else
    {
        return aoclsparse_status_invalid_value;
    }

    *m    = mat->m;
    *n    = mat->n;
    *nnz  = mat->nnz;
    *base = mat->base;
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief generates a plane rotation with cosine and sine. Slower and more accurate
 * version of BLAS's DROTG performs the Givens Rotation. The mathematical formulas
 * used for C and S are
        hv = sqrt(rr^2 + hh^2)
        c = rr/hv
        s = hh/hv
        h_mj_j = hv
 *
 *******************************************************************************/
inline void aoclsparse_givens_rotation(double &rr, double &hh, double &c, double &s, double &h_mj_j)
{
    dlartg_(&rr, &hh, &c, &s, &h_mj_j);
}
inline void aoclsparse_givens_rotation(float &rr, float &hh, float &c, float &s, float &h_mj_j)
{
    slartg_(&rr, &hh, &c, &s, &h_mj_j);
}

template <typename T>
aoclsparse_status aoclsparse_set_coo_value(aoclsparse_matrix A,
                                           aoclsparse_int    row_idx,
                                           aoclsparse_int    col_idx,
                                           T                 val)
{
    if(A == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->coo_mat.row_ind == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->coo_mat.col_ind == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->coo_mat.val == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->input_format != aoclsparse_coo_mat)
        return aoclsparse_status_internal_error;

    T *temp_val = reinterpret_cast<T *>(A->coo_mat.val);

    for(aoclsparse_int i = 0; i < A->nnz; i++)
    {
        if(A->coo_mat.row_ind[i] == row_idx && A->coo_mat.col_ind[i] == col_idx)
        {
            temp_val[i] = val;
            return aoclsparse_status_success;
        }
    }
    return aoclsparse_status_invalid_index_value;
}

template <typename T>
aoclsparse_status aoclsparse_set_csr_value(aoclsparse_index_base base,
                                           const aoclsparse_int *row_ptr,
                                           const aoclsparse_int *col_ptr,
                                           T                    *val_ptr,
                                           aoclsparse_int        row_idx,
                                           aoclsparse_int        col_idx,
                                           T                     val)
{
    if(base != aoclsparse_index_base_one && base != aoclsparse_index_base_zero)
        return aoclsparse_status_invalid_value;

    if(row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(col_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(val_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;

    aoclsparse_int row   = row_idx - base;
    aoclsparse_int begin = row_ptr[row] - base;
    aoclsparse_int end   = row_ptr[row + 1] - base;

    for(aoclsparse_int i = begin; i < end; i++)
    {
        if(col_ptr[i] == col_idx)
        {
            val_ptr[i] = val;
            return aoclsparse_status_success;
        }
    }
    return aoclsparse_status_invalid_index_value;
}

template <typename T>
aoclsparse_status aoclsparse_set_value_t(aoclsparse_matrix A,
                                         aoclsparse_int    row_idx,
                                         aoclsparse_int    col_idx,
                                         T                 val)
{
    if(A == nullptr)
        return aoclsparse_status_invalid_pointer;

    // check if coordinate given by user is within matrix bounds
    if((A->m + A->base <= row_idx || row_idx < A->base)
       || (A->n + A->base <= col_idx || col_idx < A->base))
        return aoclsparse_status_invalid_value;

    // if matrix type is same as T
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    aoclsparse_status status;
    T                *val_ptr = nullptr;

    // different method to set value for different types
    switch(A->input_format)
    {
    case aoclsparse_csr_mat:
        val_ptr = reinterpret_cast<T *>(A->csr_mat.csr_val);
        status  = aoclsparse_set_csr_value(A->base,
                                          A->csr_mat.csr_row_ptr,
                                          A->csr_mat.csr_col_ptr,
                                          val_ptr,
                                          row_idx,
                                          col_idx,
                                          val);
        break;
    case aoclsparse_csc_mat:
        val_ptr = reinterpret_cast<T *>(A->csc_mat.val);
        status  = aoclsparse_set_csr_value(
            A->base, A->csc_mat.col_ptr, A->csc_mat.row_idx, val_ptr, col_idx, row_idx, val);
        break;
    case aoclsparse_coo_mat:
        status = aoclsparse_set_coo_value(A, row_idx, col_idx, val);
        break;
    default:
        return aoclsparse_status_not_implemented;
    }
    if(status != aoclsparse_status_success)
        return status;

    // destroy the previously optimized data
    return aoclsparse_destroy_opt_csr(A);
}

#endif
