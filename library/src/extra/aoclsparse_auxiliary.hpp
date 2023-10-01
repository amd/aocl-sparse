/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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
    if((status = aoclsparse_mat_check_internal(
            M, N, nnz, row_ptr, col_idx, val, shape_general, base, nullptr))
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

#endif
