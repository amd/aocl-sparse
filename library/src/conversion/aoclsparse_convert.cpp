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

#include "aoclsparse.h"
#include "aoclsparse_convert.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
aoclsparse_status aoclsparse_csr2ell_width(aoclsparse_int        m,
                                           aoclsparse_int        nnz,
                                           const aoclsparse_int *csr_row_ptr,
                                           aoclsparse_int       *ell_width)
{
    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Check ell_width pointer
    if(ell_width == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Determine ELL width
    *ell_width = 0;

    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        *ell_width             = std::max(row_nnz, *ell_width);
    }

    // Compute ELL non-zeros
    aoclsparse_int ell_nnz = *ell_width * m;

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csr2ellthyb_width(aoclsparse_int        m,
                                               aoclsparse_int        nnz,
                                               const aoclsparse_int *csr_row_ptr,
                                               aoclsparse_int       *ell_m,
                                               aoclsparse_int       *ell_width)
{
    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Check ell_width pointer
    if(ell_width == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Determine ELL width
    *ell_width = 0;

    aoclsparse_int mx_nnz_lt_nnza = 0, mn_nnz_gt_nnza = nnz, cmn = 0, cmx = 0, nnza = nnz / m;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int nnzi = csr_row_ptr[i + 1] - csr_row_ptr[i];
        if((nnzi > mx_nnz_lt_nnza) && (nnzi <= nnza))
        {
            mx_nnz_lt_nnza = nnzi;
        }
        if((nnzi < mn_nnz_gt_nnza) && (nnzi > nnza))
        {
            mn_nnz_gt_nnza = nnzi;
        }

        if((nnzi <= nnza))
            cmx++;
        else
            cmn++;
    }
    if(cmx >= cmn)
        *ell_width = mx_nnz_lt_nnza;
    else
        *ell_width = mn_nnz_gt_nnza;

    *ell_m = 0;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        if(row_nnz <= *ell_width)
            (*ell_m)++;
    }

    return aoclsparse_status_success;
}

extern "C" aoclsparse_status aoclsparse_scsr2ell(aoclsparse_int        m,
                                                 const aoclsparse_int *csr_row_ptr,
                                                 const aoclsparse_int *csr_col_ind,
                                                 const float          *csr_val,
                                                 aoclsparse_int       *ell_col_ind,
                                                 float                *ell_val,
                                                 aoclsparse_int        ell_width)
{
    return aoclsparse_csr2ell_template(
        m, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ell(aoclsparse_int        m,
                                                 const aoclsparse_int *csr_row_ptr,
                                                 const aoclsparse_int *csr_col_ind,
                                                 const double         *csr_val,
                                                 aoclsparse_int       *ell_col_ind,
                                                 double               *ell_val,
                                                 aoclsparse_int        ell_width)
{
    return aoclsparse_csr2ell_template(
        m, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

extern "C" aoclsparse_status aoclsparse_scsr2ellt(aoclsparse_int        m,
                                                  const aoclsparse_int *csr_row_ptr,
                                                  const aoclsparse_int *csr_col_ind,
                                                  const float          *csr_val,
                                                  aoclsparse_int       *ell_col_ind,
                                                  float                *ell_val,
                                                  aoclsparse_int        ell_width)
{
    return aoclsparse_csr2ellt_template(
        m, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ellt(aoclsparse_int        m,
                                                  const aoclsparse_int *csr_row_ptr,
                                                  const aoclsparse_int *csr_col_ind,
                                                  const double         *csr_val,
                                                  aoclsparse_int       *ell_col_ind,
                                                  double               *ell_val,
                                                  aoclsparse_int        ell_width)
{
    return aoclsparse_csr2ellt_template(
        m, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

extern "C" aoclsparse_status aoclsparse_scsr2ellthyb(aoclsparse_int        m,
                                                     aoclsparse_int       *ell_m,
                                                     const aoclsparse_int *csr_row_ptr,
                                                     const aoclsparse_int *csr_col_ind,
                                                     const float          *csr_val,
                                                     aoclsparse_int       *row_idx_map,
                                                     aoclsparse_int       *csr_row_idx_map,
                                                     aoclsparse_int       *ell_col_ind,
                                                     float                *ell_val,
                                                     aoclsparse_int        ell_width)
{
    return aoclsparse_csr2ellthybrid_template(m,
                                              ell_m,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              csr_val,
                                              row_idx_map,
                                              csr_row_idx_map,
                                              ell_col_ind,
                                              ell_val,
                                              ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ellthyb(aoclsparse_int        m,
                                                     aoclsparse_int       *ell_m,
                                                     const aoclsparse_int *csr_row_ptr,
                                                     const aoclsparse_int *csr_col_ind,
                                                     const double         *csr_val,
                                                     aoclsparse_int       *row_idx_map,
                                                     aoclsparse_int       *csr_row_idx_map,
                                                     aoclsparse_int       *ell_col_ind,
                                                     double               *ell_val,
                                                     aoclsparse_int        ell_width)
{
    return aoclsparse_csr2ellthybrid_template(m,
                                              ell_m,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              csr_val,
                                              row_idx_map,
                                              csr_row_idx_map,
                                              ell_col_ind,
                                              ell_val,
                                              ell_width);
}

extern "C" aoclsparse_status aoclsparse_csr2dia_ndiag(aoclsparse_int        m,
                                                      aoclsparse_int        n,
                                                      aoclsparse_int        nnz,
                                                      const aoclsparse_int *csr_row_ptr,
                                                      const aoclsparse_int *csr_col_ind,
                                                      aoclsparse_int       *dia_num_diag)
{

    // Check sizes
    if((m < 0) || (n < 0) || (nnz < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Check dia_num_diag pointer
    if(dia_num_diag == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check pointer arguments
    if((csr_row_ptr == nullptr) || (csr_col_ind == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Determine number of populated diagonals
    *dia_num_diag = 0;

    std::vector<aoclsparse_int> diag_idx(m + n, 0);

    // Loop over rows and increment ndiag counter if diag offset has not been visited yet
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        for(aoclsparse_int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
        {
            // Diagonal offset the current entry belongs to
            aoclsparse_int offset = csr_col_ind[j] - i + m;
            if(diag_idx[offset] == 0)
            {
                diag_idx[offset] = 1;
                ++*dia_num_diag;
            }
        }
    }
    aoclsparse_int size    = (m > n) ? m : n;
    aoclsparse_int nnz_dia = size * *dia_num_diag;

    return aoclsparse_status_success;
}

extern "C" aoclsparse_status aoclsparse_scsr2dia(aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 const aoclsparse_int *csr_row_ptr,
                                                 const aoclsparse_int *csr_col_ind,
                                                 const float          *csr_val,
                                                 aoclsparse_int        dia_num_diag,
                                                 aoclsparse_int       *dia_offset,
                                                 float                *dia_val)
{
    return aoclsparse_csr2dia_template(
        m, n, csr_row_ptr, csr_col_ind, csr_val, dia_num_diag, dia_offset, dia_val);
}

extern "C" aoclsparse_status aoclsparse_dcsr2dia(aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 const aoclsparse_int *csr_row_ptr,
                                                 const aoclsparse_int *csr_col_ind,
                                                 const double         *csr_val,
                                                 aoclsparse_int        dia_num_diag,
                                                 aoclsparse_int       *dia_offset,
                                                 double               *dia_val)
{
    return aoclsparse_csr2dia_template(
        m, n, csr_row_ptr, csr_col_ind, csr_val, dia_num_diag, dia_offset, dia_val);
}

extern "C" aoclsparse_status aoclsparse_csr2bsr_nnz(aoclsparse_int        m,
                                                    aoclsparse_int        n,
                                                    const aoclsparse_int *csr_row_ptr,
                                                    const aoclsparse_int *csr_col_ind,
                                                    aoclsparse_int        block_dim,
                                                    aoclsparse_int       *bsr_row_ptr,
                                                    aoclsparse_int       *bsr_nnz)
{
    // Check sizes
    if(m < 0 || n < 0 || block_dim < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || block_dim == 0)
    {
        *bsr_nnz = 0;
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_nnz == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_int              mb = (m + block_dim - 1) / block_dim;
    aoclsparse_int              nb = (n + block_dim - 1) / block_dim;
    std::vector<bool>           blockcol(nb, false);
    std::vector<aoclsparse_int> erase(nb);
    // Loop over blocked rows
    for(aoclsparse_int bcsr_i = 0; bcsr_i < mb; ++bcsr_i)
    {
        // CSR row index
        aoclsparse_int csr_i = bcsr_i * block_dim;

        // number of blocks required in the blocked row
        aoclsparse_int nblocks = 0;

        // Loop over rows inside the current block
        for(aoclsparse_int i = 0; i < block_dim; ++i)
        {
            // Do not exceed CSR rows
            if(i >= m - csr_i)
            {
                break;
            }

            aoclsparse_int csr_row_begin = csr_row_ptr[csr_i + i];
            aoclsparse_int csr_row_end   = csr_row_ptr[csr_i + i + 1];

            // Loop over CSR columns for each of the rows in the block
            for(aoclsparse_int csr_j = csr_row_begin; csr_j < csr_row_end; ++csr_j)
            {
                // Block column index
                aoclsparse_int bcsr_j = csr_col_ind[csr_j] / block_dim;

                // Increment block counter for current blocked row if this column
                // creates a new block
                if(blockcol[bcsr_j] == false)
                {
                    blockcol[bcsr_j] = true;
                    erase[nblocks++] = bcsr_j;
                }
            }
        }

        // Store number of blocks of the current blocked row
        bsr_row_ptr[bcsr_i + 1] = nblocks;

        // Clear block buffer
        for(aoclsparse_int i = 0; i < nblocks; ++i)
        {
            blockcol[erase[i]] = false;
        }
    }

    // Exclusive sum to obtain BCSR row pointers
    bsr_row_ptr[0] = 0;
    for(aoclsparse_int i = 0; i < mb; ++i)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    // Extract BCSR nnz
    *bsr_nnz = bsr_row_ptr[mb];
    return aoclsparse_status_success;
}

extern "C" aoclsparse_status aoclsparse_scsr2bsr(aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 const float          *csr_val,
                                                 const aoclsparse_int *csr_row_ptr,
                                                 const aoclsparse_int *csr_col_ind,
                                                 aoclsparse_int        block_dim,
                                                 float                *bsr_val,
                                                 aoclsparse_int       *bsr_row_ptr,
                                                 aoclsparse_int       *bsr_col_ind)
{
    return aoclsparse_csr2bsr_template(
        m, n, csr_val, csr_row_ptr, csr_col_ind, block_dim, bsr_val, bsr_row_ptr, bsr_col_ind);
}

extern "C" aoclsparse_status aoclsparse_dcsr2bsr(aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 const double         *csr_val,
                                                 const aoclsparse_int *csr_row_ptr,
                                                 const aoclsparse_int *csr_col_ind,
                                                 aoclsparse_int        block_dim,
                                                 double               *bsr_val,
                                                 aoclsparse_int       *bsr_row_ptr,
                                                 aoclsparse_int       *bsr_col_ind)
{
    return aoclsparse_csr2bsr_template(
        m, n, csr_val, csr_row_ptr, csr_col_ind, block_dim, bsr_val, bsr_row_ptr, bsr_col_ind);
}

extern "C" aoclsparse_status aoclsparse_scsr2csc(aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 aoclsparse_int        nnz,
                                                 const aoclsparse_int *csr_row_ptr,
                                                 const aoclsparse_int *csr_col_ind,
                                                 const float          *csr_val,
                                                 aoclsparse_int       *csc_row_ind,
                                                 aoclsparse_int       *csc_col_ptr,
                                                 float                *csc_val)
{
    return aoclsparse_csr2csc_template(
        m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, csc_row_ind, csc_col_ptr, csc_val);
}

extern "C" aoclsparse_status aoclsparse_dcsr2csc(aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 aoclsparse_int        nnz,
                                                 const aoclsparse_int *csr_row_ptr,
                                                 const aoclsparse_int *csr_col_ind,
                                                 const double         *csr_val,
                                                 aoclsparse_int       *csc_row_ind,
                                                 aoclsparse_int       *csc_col_ptr,
                                                 double               *csc_val)
{
    return aoclsparse_csr2csc_template(
        m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, csc_row_ind, csc_col_ptr, csc_val);
}

extern "C" aoclsparse_status aoclsparse_scsr2dense(aoclsparse_int             m,
                                                   aoclsparse_int             n,
                                                   const aoclsparse_mat_descr descr,
                                                   const float               *csr_val,
                                                   const aoclsparse_int      *csr_row_ptr,
                                                   const aoclsparse_int      *csr_col_ind,
                                                   float                     *A,
                                                   aoclsparse_int             ld,
                                                   aoclsparse_order           order)
{
    return aoclsparse_csr2dense_template(
        m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld, order);
}

extern "C" aoclsparse_status aoclsparse_dcsr2dense(aoclsparse_int             m,
                                                   aoclsparse_int             n,
                                                   const aoclsparse_mat_descr descr,
                                                   const double              *csr_val,
                                                   const aoclsparse_int      *csr_row_ptr,
                                                   const aoclsparse_int      *csr_col_ind,
                                                   double                    *A,
                                                   aoclsparse_int             ld,
                                                   aoclsparse_order           order)
{
    return aoclsparse_csr2dense_template(
        m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld, order);
}
