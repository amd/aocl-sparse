/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#ifndef AOCLSPARSE_CONVERT_HPP
#define AOCLSPARSE_CONVERT_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <aoclsparse.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
inline void csr_to_ell(aoclsparse_int                     M,
                            aoclsparse_int&                    nnz,
                            const std::vector<aoclsparse_int>& csr_row_ptr,
                            const std::vector<aoclsparse_int>& csr_col_ind,
                            const std::vector<T>&             csr_val,
                            std::vector<aoclsparse_int>&       ell_col_ind,
                            std::vector<T>&                   ell_val,
                            aoclsparse_int&                    ell_width,
                            aoclsparse_index_base              csr_base,
                            aoclsparse_index_base              ell_base)
{
    // Determine ELL width
    ell_width = 0;

    for(aoclsparse_int i = 0; i < M; ++i)
    {
        aoclsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        ell_width             = std::max(row_nnz, ell_width);
    }

    // Compute ELL non-zeros
    aoclsparse_int ell_nnz = ell_width * M;

    // Limit ELL size to 5 times CSR nnz
    if(ell_nnz > ( 5 * nnz ))
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }
    ell_col_ind.resize(ell_nnz);
    ell_val.resize(ell_nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(aoclsparse_int i = 0; i < M; ++i)
    {
        aoclsparse_int p = i * ell_width ;
        aoclsparse_int row_begin = csr_row_ptr[i] - csr_base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - csr_base;
        aoclsparse_int row_nnz   = row_end - row_begin;

        // Fill ELL matrix with data
        for(aoclsparse_int j = row_begin; j < row_end; ++j,++p)
        {
            ell_col_ind[p] = csr_col_ind[j] - csr_base + ell_base;
            ell_val[p]     = csr_val[j];
        }

        // Add padding to ELL structures
        for(aoclsparse_int j = row_nnz; j < ell_width; ++j, ++p)
        {
            ell_col_ind[p] = -1;
            ell_val[p]     = static_cast<T>(0);

        }
    }
}

template <typename T>
inline void csr_to_dia(aoclsparse_int                     M,
                       aoclsparse_int                     N,
                       aoclsparse_int&                    nnz,
                       const std::vector<aoclsparse_int>& csr_row_ptr,
                       const std::vector<aoclsparse_int>& csr_col_ind,
                       const std::vector<T>&              csr_val,
                       std::vector<aoclsparse_int>&       dia_offset,
                       std::vector<T>&                    dia_val,
                       aoclsparse_int&                    dia_num_diag,
                       aoclsparse_index_base              csr_base)
{
    // Determine number of populated diagonals
    dia_num_diag = 0;

    std::vector<aoclsparse_int> diag_idx(M + N, 0);

    // Loop over rows and increment ndiag counter if diag offset has not been visited yet
    for(aoclsparse_int i = 0; i < M; ++i)
    {
        for(aoclsparse_int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
        {
            // Diagonal offset the current entry belongs to
            aoclsparse_int offset = csr_col_ind[j] - i + M;
            if(diag_idx[offset] == 0)
            {
                diag_idx[offset] = 1;
                ++dia_num_diag;
            }
        }
    }

    aoclsparse_int size = (M > N) ? M : N;
    aoclsparse_int nnz_dia = size * dia_num_diag;
    // Conversion fails if DIA nnz exceeds 5 times CSR nnz
    if(nnz_dia > (5 * nnz))
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }
    // Allocate DIA matrix
    dia_offset.resize(dia_num_diag);
    dia_val.resize(nnz_dia);

    for(aoclsparse_int i = 0, d = 0; i < (M + N); ++i)
    {
        // Fill DIA offset, if i-th diagonal is populated
        if(diag_idx[i])
        {
            // Store offset index for reverse index access
            diag_idx[i] = d;
            // Store diagonals offset, where the diagonal is offset 0
            // Left from diagonal offsets are decreasing
            // Right from diagonal offsets are increasing
            dia_offset[d++] = i - M;
        }
    }

    for(aoclsparse_int i = 0; i < M; ++i)
    {
        for(aoclsparse_int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
        {
            // Diagonal offset the current entry belongs to
            aoclsparse_int offset = csr_col_ind[j] - i + M;

            dia_val[i + M * diag_idx[offset]] = csr_val[j];
        }
    }
}
void csr_to_bsr_nnz(aoclsparse_int block_dim,
                aoclsparse_int        m,
                aoclsparse_int        n,
                aoclsparse_int        mb,
                aoclsparse_int        nb,
                aoclsparse_index_base csr_base,
                const aoclsparse_int* __restrict__ csr_row_ptr,
                const aoclsparse_int* __restrict__ csr_col_ind,
                aoclsparse_index_base bsr_base,
                aoclsparse_int* __restrict__ bsr_row_ptr,
                aoclsparse_int*            bsr_nnz)
{
    std::vector<bool>      blockcol(nb, false);
    std::vector<aoclsparse_int> erase(nb);
    // Loop over blocked rows
    for(aoclsparse_int bcsr_i = 0; bcsr_i < mb; ++bcsr_i)
    {
        // CSR row index
        aoclsparse_int csr_i = bcsr_i * block_dim;

        // Number of blocks required in the blocked row
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
}
// BCSR indexing
#define BCSR_IND(j, bi, bj, dim) ((j) + (bi) + (bj) * (dim))
template <typename T>
void csr_to_bsr(aoclsparse_int              m,
                aoclsparse_int              n,
                aoclsparse_int              mb,
                aoclsparse_int              nb,
                aoclsparse_int              block_dim,
                const aoclsparse_index_base csr_base,
                const T* __restrict__ csr_val,
                const aoclsparse_int* __restrict__ csr_row_ptr,
                const aoclsparse_int* __restrict__ csr_col_ind,
                const aoclsparse_index_base bsr_base,
                T* __restrict__ bsr_val,
                aoclsparse_int* __restrict__ bsr_row_ptr,
                aoclsparse_int* __restrict__ bsr_col_ind)
{
    // Fill BCSR structure
    {
        std::vector<aoclsparse_int> blockcol(nb, -1);

        // Loop over blocked rows
        for(aoclsparse_int bcsr_i = 0; bcsr_i < mb; ++bcsr_i)
        {
            // CSR row index
            aoclsparse_int csr_i = bcsr_i * block_dim;

            // Offset into BCSR row
            aoclsparse_int bcsr_row_begin = bsr_row_ptr[bcsr_i];
            aoclsparse_int bcsr_row_end   = bsr_row_ptr[bcsr_i + 1];
            aoclsparse_int bcsr_idx       = bcsr_row_begin;

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
                    // CSR column index
                    aoclsparse_int csr_col = csr_col_ind[csr_j];

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
                        bsr_col_ind[bcsr_idx++] = bcsr_col;
                    }

                    // Write BCSR value
                    bsr_val[BCSR_IND(blockcol[bcsr_col], i, j, block_dim)] = csr_val[csr_j];
                }
            }

            // Clear block buffer
            for(aoclsparse_int i = bcsr_row_begin; i < bcsr_row_end; ++i)
            {
                blockcol[bsr_col_ind[i]] = -1;
            }
        }

        // Sort
        for(aoclsparse_int i = 0; i < mb; ++i)
        {
            aoclsparse_int row_begin = bsr_row_ptr[i];
            aoclsparse_int row_end   = bsr_row_ptr[i + 1];

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
}

template <typename T>
void csr_to_csc(aoclsparse_int                     M,
                aoclsparse_int                     N,
                aoclsparse_int                     nnz,
                const std::vector<aoclsparse_int>& csr_row_ptr,
                const std::vector<aoclsparse_int>& csr_col_ind,
                const std::vector<T>&              csr_val,
                std::vector<aoclsparse_int>&       csc_row_ind,
                std::vector<aoclsparse_int>&       csc_col_ptr,
                std::vector<T>&                    csc_val,
                aoclsparse_index_base              base)
{
    csc_row_ind.resize(nnz);
    csc_col_ptr.resize(N + 1, 0);
    csc_val.resize(nnz);

    // Determine nnz per column
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        ++csc_col_ptr[csr_col_ind[i] + 1 - base];
    }

    // Scan
    for(aoclsparse_int i = 0; i < N; ++i)
    {
        csc_col_ptr[i + 1] += csc_col_ptr[i];
    }

    // Fill row indices and values
    for(aoclsparse_int i = 0; i < M; ++i)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

        for(aoclsparse_int j = row_begin; j < row_end; ++j)
        {
            aoclsparse_int col = csr_col_ind[j] - base;
            aoclsparse_int idx = csc_col_ptr[col];

            csc_row_ind[idx] = i + base;
            csc_val[idx]     = csr_val[j];

            ++csc_col_ptr[col];
        }
    }
    // Shift column pointer array
    for(aoclsparse_int i = N; i > 0; --i)
    {
        csc_col_ptr[i] = csc_col_ptr[i - 1] + base;
    }

    csc_col_ptr[0] = base;
}
#endif // AOCLSPARSE_CONVERT_HPP
