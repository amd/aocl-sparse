/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_CONVERT_HPP
#define AOCLSPARSE_CONVERT_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits.h>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

template <typename T>
aoclsparse_status aoclsparse_csr2ell_template(aoclsparse_int             m,
                                              const aoclsparse_mat_descr descr,
                                              const aoclsparse_int      *csr_row_ptr,
                                              const aoclsparse_int      *csr_col_ind,
                                              const T                   *csr_val,
                                              aoclsparse_int            *ell_col_ind,
                                              T                         *ell_val,
                                              aoclsparse_int             ell_width)
{
    // Check sizes
    if((m < 0) || (ell_width < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if((m == 0) || (ell_width == 0))
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_index_base base = descr->base;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int p         = i * ell_width;
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        aoclsparse_int row_nnz   = row_end - row_begin;

        // Fill ELL matrix with data
        for(aoclsparse_int j = row_begin; j < row_end; ++j, ++p)
        {
            //preserve base-index of ell column index array for ellmv
            ell_col_ind[p] = csr_col_ind[j];
            ell_val[p]     = csr_val[j];
        }

        // Add padding to ELL structures
        for(aoclsparse_int j = row_nnz; j < ell_width; ++j, ++p)
        {
            ell_col_ind[p] = -1;
            ell_val[p]     = static_cast<T>(0);
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csr2ellt_template(aoclsparse_int             m,
                                               const aoclsparse_mat_descr descr,
                                               const aoclsparse_int      *csr_row_ptr,
                                               const aoclsparse_int      *csr_col_ind,
                                               const T                   *csr_val,
                                               aoclsparse_int            *ell_col_ind,
                                               T                         *ell_val,
                                               aoclsparse_int             ell_width)
{
    // Check sizes
    if((m < 0) || (ell_width < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if((m == 0) || (ell_width == 0))
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_index_base base = descr->base;
    // Transpose the ell matrix as we populate
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        aoclsparse_int k         = 0;
        // Fill ELL matrix with data
        for(aoclsparse_int j = row_begin; j < row_end; ++j, ++k)
        {
            ell_col_ind[k * m + i] = csr_col_ind[j];
            ell_val[k * m + i]     = csr_val[j];
        }

        // Add padding to ELL structures
        for(; k < ell_width; ++k)
        {
            // padding the col_ind with the value of last col ind for reuse
            ell_col_ind[k * m + i] = csr_col_ind[row_end - 1];
            ell_val[k * m + i]     = static_cast<T>(0);
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csr2ellthybrid_template(
    aoclsparse_int        m,
    aoclsparse_index_base base,
    aoclsparse_int       *ell_m,
    const aoclsparse_int *csr_row_ptr,
    const aoclsparse_int *csr_col_ind,
    const T              *csr_val,
    aoclsparse_int *csr_row_idx_map, // mapping of rows that need to be accessed in the csr format
    aoclsparse_int *ell_col_ind,
    T              *ell_val,
    aoclsparse_int  ell_width)
{
    // Check sizes
    if((m < 0) || (ell_width < 0))
    {
        return aoclsparse_status_invalid_size;
    }
    // Quick return if possible
    if(m == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_idx_map == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_m == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Find the number of rows that will be in the ell format
    aoclsparse_int m_ell = 0;
    for(aoclsparse_int k = 0; k < m; ++k)
    {
        if((csr_row_ptr[k + 1] - csr_row_ptr[k]) <= ell_width)
            m_ell++;
    }
    *ell_m = m_ell;
    // Transpose the ell matrix as we populate
    aoclsparse_int t_csr_ridx = 0;
    aoclsparse_int t_ridx     = 0;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        aoclsparse_int k         = 0;
        aoclsparse_int flag      = 0;

        // split rows based on the "computed" ell_width
        if((row_end - row_begin) > ell_width)
        {
            // populate the csr_row_idx_map. csr_row_idx_map array is computed to be
            // 0-based irrespective of the input base-index format
            csr_row_idx_map[t_csr_ridx++] = i;
            //	   continue;
            flag = 1;
        }

        // Fill ELL matrix with data
        m_ell = m;
        if(flag)
        {
            for(aoclsparse_int j = 0; j < ell_width; ++j)
            {
                ell_col_ind[j * m_ell + t_ridx] = csr_col_ind[row_end - 1];
                ell_val[j * m_ell + t_ridx]     = static_cast<T>(0);
            }
        }
        else
        {
            for(aoclsparse_int j = row_begin; j < row_end; ++j, ++k)
            {
                ell_col_ind[k * m_ell + t_ridx] = csr_col_ind[j];
                ell_val[k * m_ell + t_ridx]     = csr_val[j];
            }

            // Add padding to ELL structures
            for(; k < ell_width; ++k)
            {
                // padding the col_ind with the value of last col ind for reuse
                ell_col_ind[k * m_ell + t_ridx] = csr_col_ind[row_end - 1];
                ell_val[k * m_ell + t_ridx]     = static_cast<T>(0);
            }
        }
        t_ridx++;
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csr2dia_template(aoclsparse_int             m,
                                              aoclsparse_int             n,
                                              const aoclsparse_mat_descr descr,
                                              const aoclsparse_int      *csr_row_ptr,
                                              const aoclsparse_int      *csr_col_ind,
                                              const T                   *csr_val,
                                              aoclsparse_int             dia_num_diag,
                                              aoclsparse_int            *dia_offset,
                                              T                         *dia_val)
{
    // Check sizes
    if((m < 0) || (n < 0) || (dia_num_diag < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if((m == 0) || (n == 0) || (dia_num_diag == 0))
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(dia_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(dia_offset == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    std::vector<aoclsparse_int> diag_idx;
    aoclsparse_index_base       base = descr->base;

    try
    {
        diag_idx.resize(m + n, 0);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    // Loop over rows and increment ndiag counter if diag offset has not been visited yet
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        for(aoclsparse_int j = (csr_row_ptr[i] - base); j < (csr_row_ptr[i + 1] - base); ++j)
        {
            // Diagonal offset the current entry belongs to
            aoclsparse_int offset = csr_col_ind[j] - base - i + m;
            if(diag_idx[offset] == 0)
            {
                diag_idx[offset] = 1;
            }
        }
    }
    for(aoclsparse_int i = 0, d = 0; i < (m + n); ++i)
    {
        // Fill DIA offset, if i-th diagonal is populated
        if(diag_idx[i])
        {

            // Store offset index for reverse index access
            diag_idx[i] = d;
            // Store diagonals offset, where the diagonal is offset 0
            // Left from diagonal offsets are decreasing
            // Right from diagonal offsets are increasing
            dia_offset[d++] = i - m;
        }
    }
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        for(aoclsparse_int j = (csr_row_ptr[i] - base); j < (csr_row_ptr[i + 1] - base); ++j)
        {
            // Diagonal offset the current entry belongs to
            aoclsparse_int offset = csr_col_ind[j] - base - i + m;

            dia_val[i + m * diag_idx[offset]] = csr_val[j];
        }
    }

    return aoclsparse_status_success;
}

#define BCSR_IND(j, bi, bj, dim) ((j) + (bi) + (bj) * (dim))
template <typename T>
aoclsparse_status aoclsparse_csr2bsr_template(aoclsparse_int             m,
                                              aoclsparse_int             n,
                                              const aoclsparse_mat_descr descr,
                                              const T *__restrict__ csr_val,
                                              const aoclsparse_int *__restrict__ csr_row_ptr,
                                              const aoclsparse_int *__restrict__ csr_col_ind,
                                              aoclsparse_int block_dim,
                                              T *__restrict__ bsr_val,
                                              aoclsparse_int *__restrict__ bsr_row_ptr,
                                              aoclsparse_int *__restrict__ bsr_col_ind)
{
    // Check sizes
    if(m < 0 || n < 0 || block_dim < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || block_dim == 0)
    {
        return aoclsparse_status_success;
    }

    if((descr->base != aoclsparse_index_base_zero) && (descr->base != aoclsparse_index_base_one))
    {
        return aoclsparse_status_invalid_value;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_int        mb   = (m + block_dim - 1) / block_dim;
    aoclsparse_int        nb   = (n + block_dim - 1) / block_dim;
    aoclsparse_index_base base = descr->base;

    // Fill BCSR structure
    {
        std::vector<aoclsparse_int> blockcol;
        try
        {
            blockcol.resize(nb, -1);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        // Loop over blocked rows
        for(aoclsparse_int bcsr_i = 0; bcsr_i < mb; ++bcsr_i)
        {
            // CSR row index
            aoclsparse_int csr_i = bcsr_i * block_dim;

            // Offset into BCSR row
            aoclsparse_int bcsr_row_begin = bsr_row_ptr[bcsr_i] - base;
            aoclsparse_int bcsr_row_end   = bsr_row_ptr[bcsr_i + 1] - base;
            aoclsparse_int bcsr_idx       = bcsr_row_begin;

            // Loop over rows inside the current block
            for(aoclsparse_int i = 0; i < block_dim; ++i)
            {
                // Do not exceed CSR rows
                if(i >= m - csr_i)
                {
                    break;
                }

                aoclsparse_int csr_row_begin = csr_row_ptr[csr_i + i] - base;
                aoclsparse_int csr_row_end   = csr_row_ptr[csr_i + i + 1] - base;

                // Loop over CSR columns for each of the rows in the block
                for(aoclsparse_int csr_j = csr_row_begin; csr_j < csr_row_end; ++csr_j)
                {
                    // CSR column index
                    aoclsparse_int csr_col = csr_col_ind[csr_j] - base;

                    // Block column index
                    aoclsparse_int bcsr_col = csr_col / block_dim;

                    // Intra block column
                    aoclsparse_int j = csr_col % block_dim;

                    // Fill the block's column index
                    if(blockcol[bcsr_col] == -1)
                    {
                        // Keep block value offset for filling
                        blockcol[bcsr_col] = bcsr_idx * block_dim * block_dim;

                        // Write BCSR column index
                        bsr_col_ind[bcsr_idx++] = bcsr_col + base;
                    }

                    // Write BCSR value
                    bsr_val[BCSR_IND(blockcol[bcsr_col], i, j, block_dim)] = csr_val[csr_j];
                }
            }

            // Clear block buffer
            for(aoclsparse_int i = bcsr_row_begin; i < bcsr_row_end; ++i)
            {
                blockcol[bsr_col_ind[i] - base] = -1;
            }
        }

        // Sort
        for(aoclsparse_int i = 0; i < mb; ++i)
        {
            aoclsparse_int row_begin = bsr_row_ptr[i] - base;
            aoclsparse_int row_end   = bsr_row_ptr[i + 1] - base;

            for(aoclsparse_int j = row_begin; j < row_end; ++j)
            {
                for(aoclsparse_int k = row_begin; k < row_end - 1; ++k)
                {
                    if(bsr_col_ind[k] > bsr_col_ind[k + 1])
                    {
                        // Swap values
                        for(aoclsparse_int b = 0; b < block_dim * block_dim; ++b)
                        {
                            std::swap(bsr_val[block_dim * block_dim * k + b],
                                      bsr_val[block_dim * block_dim * (k + 1) + b]);
                        }

                        // Swap column index
                        std::swap(bsr_col_ind[k], bsr_col_ind[k + 1]);
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csr2csc_template(aoclsparse_int             m,
                                              aoclsparse_int             n,
                                              aoclsparse_int             nnz,
                                              const aoclsparse_mat_descr descr,
                                              aoclsparse_index_base      baseCSC,
                                              const aoclsparse_int      *csr_row_ptr,
                                              const aoclsparse_int      *csr_col_ind,
                                              const T                   *csr_val,
                                              aoclsparse_int            *csc_row_ind,
                                              aoclsparse_int            *csc_col_ptr,
                                              T                         *csc_val)
{
    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        for(aoclsparse_int i = 0; i < n + 1; i++)
            csc_col_ptr[i] = baseCSC;
        return aoclsparse_status_success;
    }

    //Check base index for validity
    if((descr->base != aoclsparse_index_base_zero) && (descr->base != aoclsparse_index_base_one))
    {
        return aoclsparse_status_invalid_value;
    }
    if((baseCSC != aoclsparse_index_base_zero) && (baseCSC != aoclsparse_index_base_one))
    {
        return aoclsparse_status_invalid_value;
    }
    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csc_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csc_row_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csc_col_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    aoclsparse_index_base base = descr->base;
    // csc_col_ptr comes from the user; initialize it to 0
    for(aoclsparse_int i = 0; i < n + 1; ++i)
    {
        csc_col_ptr[i] = 0;
    }

    // Determine nnz per column
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        ++csc_col_ptr[csr_col_ind[i] - base + 1];
    }

    // Scan
    for(aoclsparse_int i = 0; i < n; ++i)
    {
        csc_col_ptr[i + 1] += csc_col_ptr[i];
    }

    // Fill row indices and values
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int j = row_begin; j < row_end; ++j)
        {
            aoclsparse_int col = csr_col_ind[j] - base;
            aoclsparse_int idx = csc_col_ptr[col];
            csc_row_ind[idx]   = i + baseCSC;
            csc_val[idx]       = csr_val[j];

            ++csc_col_ptr[col];
        }
    }
    // Shift column pointer array
    for(aoclsparse_int i = n; i > 0; --i)
    {
        csc_col_ptr[i] = csc_col_ptr[i - 1] + baseCSC;
    }

    csc_col_ptr[0] = baseCSC;
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csr2dense_template(aoclsparse_int             m,
                                                aoclsparse_int             n,
                                                const aoclsparse_mat_descr descr,
                                                const T                   *csr_val,
                                                const aoclsparse_int      *csr_row_ptr,
                                                const aoclsparse_int      *csr_col_ind,
                                                T                         *A,
                                                aoclsparse_int             ld,
                                                aoclsparse_order           order)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    //Check base index for validity
    if((descr->base != aoclsparse_index_base_zero) && (descr->base != aoclsparse_index_base_one))
    {
        return aoclsparse_status_invalid_value;
    }

    // Support General matrices.
    // Return for any other matrix type
    if(descr->type
       != aoclsparse_matrix_type_general) //&& (descr->type != aoclsparse_matrix_type_symmetric))
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(A == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_index_base base = descr->base;

    if(order == aoclsparse_order_column)
    {
        for(aoclsparse_int col = 0; col < n; ++col)
        {
            for(aoclsparse_int row = 0; row < m; ++row)
            {
                A[row + ld * col] = static_cast<T>(0);
            }
        }
        for(aoclsparse_int row = 0; row < m; ++row)
        {
            aoclsparse_int start = csr_row_ptr[row] - base;
            aoclsparse_int end   = csr_row_ptr[row + 1] - base;

            for(aoclsparse_int at = start; at < end; ++at)
            {
                aoclsparse_int col_idx = csr_col_ind[at] - base;
                A[col_idx * ld + row]  = csr_val[at];
            }
        }
    }
    else
    {
        for(aoclsparse_int row = 0; row < m; ++row)
        {
            for(aoclsparse_int col = 0; col < n; ++col)
            {
                A[col + ld * row] = static_cast<T>(0);
            }
        }
        for(aoclsparse_int row = 0; row < m; ++row)
        {
            aoclsparse_int start = csr_row_ptr[row] - base;
            aoclsparse_int end   = csr_row_ptr[row + 1] - base;

            for(aoclsparse_int at = start; at < end; ++at)
            {
                aoclsparse_int col_idx = csr_col_ind[at] - base;
                A[row * ld + col_idx]  = csr_val[at];
            }
        }
    }

    return aoclsparse_status_success;
}

/**
 *  this function will convert coo format to csr format with same base as input matrix 
 *  NOTE: all the input should from the same matrix. 
 *  Validation of values present in \p coo_row_ind and \p coo_col_ind is not done prior to use
*/
template <typename T>
aoclsparse_status aoclsparse_coo2csr_template(aoclsparse_int        M,
                                              aoclsparse_int        N,
                                              aoclsparse_int        nnz,
                                              aoclsparse_index_base base,
                                              const aoclsparse_int *coo_row_ind,
                                              const aoclsparse_int *coo_col_ind,
                                              const T              *coo_val,
                                              aoclsparse_int       *csr_row_ptr,
                                              aoclsparse_int       *csr_col_ind,
                                              T                    *csr_val)
{
    // Check sizes
    if(M < 0 || N < 0 || nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(coo_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(coo_row_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(coo_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(M == 0 || N == 0 || nnz == 0)
    {
        for(aoclsparse_int i = 0; i < M + 1; i++)
            csr_row_ptr[i] = base;
        return aoclsparse_status_success;
    }

    // csr_row_ptr comes from the user; initialize it to 0
    for(aoclsparse_int i = 0; i < M + 1; ++i)
    {
        csr_row_ptr[i] = 0;
    }

    // Determine nnz per row
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr[coo_row_ind[i] + 1 - base];
    }

    // Scan
    for(aoclsparse_int i = 0; i < M; ++i)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    // Fill col indices and values
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        aoclsparse_int row_idx = coo_row_ind[i] - base;
        aoclsparse_int idx     = csr_row_ptr[row_idx];
        csr_col_ind[idx]       = coo_col_ind[i];
        csr_val[idx]           = coo_val[i];
        ++csr_row_ptr[row_idx];
    }

    // Shift row pointer array
    for(aoclsparse_int i = M; i > 0; --i)
    {
        csr_row_ptr[i] = csr_row_ptr[i - 1] + base;
    }

    csr_row_ptr[0] = base;

    return aoclsparse_status_success;
}

// aoclsparse_ell2csr_template has been test and does not support 1 base indexing. Currently we don't have need for this
#if 0
template <typename T>
aoclsparse_status aoclsparse_ell2csr_template(aoclsparse_int        M,
                                              aoclsparse_int        N,
                                              aoclsparse_int        nnz,
                                              const aoclsparse_int *ell_col_ind,
                                              const aoclsparse_int  ell_width,
                                              const T              *ell_val,
                                              aoclsparse_int       *csr_row_ptr,
                                              aoclsparse_int       *csr_col_ind,
                                              T                    *csr_val)
{
    // Check sizes
    if(M < 0 || N < 0 || nnz < 0 || ell_width < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(M == 0 || N == 0 || nnz == 0 || ell_width == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    aoclsparse_index_base base = aoclsparse_index_base_zero;

    // csc_col_ptr comes from the user; initialize it to 0
    for(aoclsparse_int i = 0; i < M + 1; ++i)
    {
        csr_row_ptr[i] = 0;
    }

    // Fill col indices and values
    int col_ind = 0;
    for(aoclsparse_int i = 0; i < M; ++i)
    {
        int row_ind = i * ell_width;
        for(aoclsparse_int j = 0; j < ell_width; j++)
        {
            if(ell_col_ind[row_ind + j] < base)
                break;
            csr_col_ind[col_ind] = ell_col_ind[row_ind + j];
            csr_val[col_ind]     = ell_val[row_ind + j];
            col_ind++;
        }
        csr_row_ptr[i + 1] = col_ind;
    }

    return aoclsparse_status_success;
}
#endif

#endif // AOCLSPARSE_CONVERT_HPP
