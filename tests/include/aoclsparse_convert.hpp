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

#endif // AOCLSPARSE_CONVERT_HPP
