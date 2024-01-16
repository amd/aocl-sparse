/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */

#include "aoclsparse.h"
#include "aoclsparse_syrk.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Computes the product of a sparse matrix with its transpose and stores the result in a
 * newly allocated sparse matrix. The sparse matrices are in CSR storage format and supports s/d/c/z data types.
 */
extern "C" aoclsparse_status
    aoclsparse_syrk(const aoclsparse_operation op, const aoclsparse_matrix A, aoclsparse_matrix *C)
{
    const aoclsparse_int kid = -1; /* auto */

    if(A == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(A->val_type == aoclsparse_smat)
        return aoclsparse_syrk_t<float>(op, A, C, kid);
    else if(A->val_type == aoclsparse_dmat)
        return aoclsparse_syrk_t<double>(op, A, C, kid);
    else if(A->val_type == aoclsparse_cmat)
        return aoclsparse_syrk_t<std::complex<float>>(op, A, C, kid);
    else if(A->val_type == aoclsparse_zmat)
        return aoclsparse_syrk_t<std::complex<double>>(op, A, C, kid);
    else
        return aoclsparse_status_wrong_type;
}

// Overestimates the number of nonzeros in the upper triangle of AA' or A'A
aoclsparse_status estimate_nnz(const aoclsparse_operation op,
                               aoclsparse_index_base      base,
                               aoclsparse_int             m,
                               aoclsparse_int             n,
                               const aoclsparse_int      *csr_row_ptr_A,
                               const aoclsparse_int      *csr_col_ind_A,
                               aoclsparse_int            &nnz)
{
    /* Imagine building i-th row of C when multiplying C=A*B:
     * It is the sum of all sparse j-th B rows where a_ij is nonzero.
     * Now given that B=A' in our case, there will be always diagonal
     * element for each nonempty row, so count the off-diagonal elements
     * 'row_length[i]-1' and add +1 at the end. Also the matrix is symmetric
     * so half the estimate.
     */
    nnz = 0;

    csr_col_ind_A = csr_col_ind_A - base;
    if(op == aoclsparse_operation_none) // AA'
    {
        std::vector<aoclsparse_int> nnz_count; // stores #nnz per A column (A' row)
        try
        {
            nnz_count.resize(n, 0);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        aoclsparse_int i, idx;

        for(i = 0; i < m; i++)
        {
            for(idx = csr_row_ptr_A[i]; idx < csr_row_ptr_A[i + 1]; ++idx)
            {
                nnz_count[csr_col_ind_A[idx] - base]++;
            }
        }

        for(i = 0; i < m; i++)
        {
            aoclsparse_int nnz_row = 0;
            for(idx = csr_row_ptr_A[i]; idx < csr_row_ptr_A[i + 1]; ++idx)
                nnz_row += nnz_count[csr_col_ind_A[idx] - base] - 1;
            if(nnz_row > m - 1)
                nnz_row = m - 1;
            nnz += (nnz_row + 1) / 2 + 1; // only 1 triangle and add the diagonal
        }
    }
    else // A'A
    {
        std::vector<aoclsparse_int> nnz_count; // stores nnz count per A row
        std::vector<aoclsparse_int> nnz_row; // stores est. nnz per row A'A
        try
        {
            nnz_count.resize(m, 0);
            nnz_row.resize(n, 0);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        aoclsparse_int i, idx, j;

        for(i = 0; i < m; i++)
            nnz_count[i] = csr_row_ptr_A[i + 1] - csr_row_ptr_A[i];

        for(i = 0; i < m; i++)
        {
            for(idx = csr_row_ptr_A[i]; idx < csr_row_ptr_A[i + 1]; ++idx)
            {
                j = csr_col_ind_A[idx] - base;
                nnz_row[j] += nnz_count[i] - 1;
            }
        }
        for(j = 0; j < n; ++j)
        {
            if(nnz_row[j] > n - 1)
                nnz_row[j] = n - 1;
            nnz += (nnz_row[j] + 1) / 2 + 1; // only 1 triangle and add the diagonal
        }
    }

    return aoclsparse_status_success;
}
