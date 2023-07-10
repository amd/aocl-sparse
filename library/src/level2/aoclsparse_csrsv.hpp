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
#ifndef AOCLSPARSE_CSRSV_HPP
#define AOCLSPARSE_CSRSV_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"

#include "iostream"

template <typename T>
aoclsparse_status aoclsparse_csrsv_template(const T                    alpha,
                                            aoclsparse_int             m,
                                            const T                   *csr_val,
                                            const aoclsparse_int      *csr_col_ind,
                                            const aoclsparse_int      *csr_row_ptr,
                                            const aoclsparse_mat_descr descr,
                                            const T                   *x,
                                            T                         *y)
{
    if(descr->fill_mode == aoclsparse_fill_mode_lower)
    {
        aoclsparse_csr_lsolve(
            alpha, m, csr_val, csr_col_ind, csr_row_ptr, x, y, descr->diag_type, descr->base);
    }
    else
    {
        aoclsparse_csr_usolve(
            alpha, m, csr_val, csr_col_ind, csr_row_ptr, x, y, descr->diag_type, descr->base);
    }
    return aoclsparse_status_success;
}

template <typename T>
static inline void aoclsparse_csr_lsolve(const T               alpha,
                                         aoclsparse_int        m,
                                         const T              *csr_val,
                                         const aoclsparse_int *csr_col_ind,
                                         const aoclsparse_int *csr_row_ptr,
                                         const T              *x,
                                         T                    *y,
                                         aoclsparse_diag_type  diag_type,
                                         aoclsparse_index_base base)
{
    aoclsparse_int diag_j = 0;

    // Solve L
    for(aoclsparse_int row = 0; row < m; ++row)
    {
        y[row] = alpha * x[row];

        for(aoclsparse_int j = (csr_row_ptr[row] - base); j < (csr_row_ptr[row + 1] - base); ++j)
        {
            // Ignore all entries that are above the diagonal
            if((csr_col_ind[j] - base) < row)
            {
                // under the diagonal
                y[row] -= csr_val[j] * y[csr_col_ind[j] - base];
            }
            else
            {
                if(diag_type == aoclsparse_diag_type_non_unit)
                {
                    if((csr_col_ind[j] - base) == row)
                        diag_j = j;
                }
                break;
            }
        }
        // If diagonal type is non unit, do division by diagonal entry
        // This is not required for unit diagonal for obvious reasons
        if(diag_type == aoclsparse_diag_type_non_unit)
        {
            y[row] /= csr_val[diag_j];
        }
    }
}

template <typename T>
static inline void aoclsparse_csr_usolve(const T               alpha,
                                         aoclsparse_int        m,
                                         const T              *csr_val,
                                         const aoclsparse_int *csr_col_ind,
                                         const aoclsparse_int *csr_row_ptr,
                                         const T              *x,
                                         T                    *y,
                                         aoclsparse_diag_type  diag_type,
                                         aoclsparse_index_base base)
{
    aoclsparse_int diag_j = 0;

    // Solve L
    for(aoclsparse_int row = m - 1; row >= 0; --row)
    {
        y[row] = alpha * x[row];

        for(aoclsparse_int j = (csr_row_ptr[row] - base); j < (csr_row_ptr[row + 1] - base); ++j)
        {
            // Ignore all entries that are below the diagonal
            if((csr_col_ind[j] - base) > row)
            {
                // under the diagonal
                y[row] -= csr_val[j] * y[csr_col_ind[j] - base];
            }
            if(diag_type == aoclsparse_diag_type_non_unit)
            {
                if((csr_col_ind[j] - base) == row)
                    diag_j = j;
            }
        }
        // If diagonal type is non unit, do division by diagonal entry
        // This is not required for unit diagonal for obvious reasons
        if(diag_type == aoclsparse_diag_type_non_unit)
        {
            y[row] /= csr_val[diag_j];
        }
    }
}

#endif // AOCLSPARSE_CSRSV_HPP
