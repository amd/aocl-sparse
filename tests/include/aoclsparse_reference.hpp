/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_REFERENCE_HPP
#define AOCLSPARSE_REFERENCE_HPP

#include "aoclsparse.h"

/* This file contains reference implementations of matrix operations
 * which are used for comparisons of the results while testing
 * in aoclsparse-bench and in gtests.
 */

/* Symmetric SPMV with a mxm matrix stored as CSR: y = alpha*A*x + beta*y
 * This is a safe & slow implementation, checking for out of bounds indices.
 * Supported:
 * - zero/one-based indexing
 * - L/U triangle
 * - non-unit/unit diagonal/zero diagonal
 * CSR doesn't need to be sorted in rows and doesn't need to contain only
 * the given triangle, the unuseful parts (and out of bounds indices)
 * are ignored. Duplicate entries are summed.
 */
template <typename T>
aoclsparse_status ref_csrmvsym(T                     alpha,
                               aoclsparse_int        m,
                               const T              *csr_val,
                               const aoclsparse_int *csr_col_ind,
                               const aoclsparse_int *csr_row_ptr,
                               aoclsparse_fill_mode  fill,
                               aoclsparse_diag_type  diag,
                               aoclsparse_index_base base,
                               const T              *x,
                               T                     beta,
                               T                    *y)
{

    if(csr_val == nullptr || csr_col_ind == nullptr || csr_row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(m < 0)
        return aoclsparse_status_invalid_size;

    // nothing to do case
    if(m == 0)
        return aoclsparse_status_success;

    // check validity of csr_row_ptr
    if(csr_row_ptr[0] < base)
        return aoclsparse_status_invalid_value;
    for(aoclsparse_int i = 0; i < m; i++)
        if(csr_row_ptr[i] > csr_row_ptr[i + 1])
            return aoclsparse_status_invalid_value;

    bool is_lower  = fill == aoclsparse_fill_mode_lower;
    bool keep_diag = diag == aoclsparse_diag_type_non_unit;
    bool unit_diag = diag == aoclsparse_diag_type_unit;

    if(beta == 0.)
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    else
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int idxend = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int idx = csr_row_ptr[i] - base; idx < idxend; idx++)
        {
            // valid indices given the specified triangle
            aoclsparse_int j = csr_col_ind[idx] - base;
            if(j >= 0 && j < i)
            {
                if(is_lower)
                {
                    y[i] += alpha * csr_val[idx] * x[j];
                    y[j] += alpha * csr_val[idx] * x[i];
                }
            }
            else if(j == i)
            {
                if(keep_diag)
                    y[i] += alpha * csr_val[idx] * x[i];
            }
            else if(j > i && i < m)
            {
                if(!is_lower)
                {
                    y[i] += alpha * csr_val[idx] * x[j];
                    y[j] += alpha * csr_val[idx] * x[i];
                }
            }
        }
        if(unit_diag)
        {
            // add unit diagonal
            y[i] += alpha * x[i];
        }
    }

    return aoclsparse_status_success;
}

/* SPMV with a L or U triangular submatrix of a mxn matrix stored as CSR:
 *   y = alpha*op(A)*x + beta*y   where op=tril or triu
 * As symmetric version above but uses only the triangle (doesn't symmetrize).
 *
 * This is a safe & slow implementation, checking for out of bounds indices.
 * Supported:
 * - zero/one-based indexing
 * - L/U triangle
 * - non-unit/unit diagonal/zero diagonal
 * CSR doesn't need to be sorted in rows and doesn't need to contain only
 * the given triangle, the unuseful parts (and out of bounds indices)
 * are ignored. Duplicate entries are summed.
 */
template <typename T>
aoclsparse_status ref_csrmvtrg(T                     alpha,
                               aoclsparse_int        m,
                               aoclsparse_int        n,
                               const T              *csr_val,
                               const aoclsparse_int *csr_col_ind,
                               const aoclsparse_int *csr_row_ptr,
                               aoclsparse_fill_mode  fill,
                               aoclsparse_diag_type  diag,
                               aoclsparse_index_base base,
                               const T              *x,
                               T                     beta,
                               T                    *y)
{

    if(csr_val == nullptr || csr_col_ind == nullptr || csr_row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(m < 0 || n < 0)
        return aoclsparse_status_invalid_size;

    // nothing to do case
    if(m == 0 || n == 0)
        return aoclsparse_status_success;

    // check validity of csr_row_ptr
    if(csr_row_ptr[0] < base)
        return aoclsparse_status_invalid_value;
    for(aoclsparse_int i = 0; i < m; i++)
        if(csr_row_ptr[i] > csr_row_ptr[i + 1])
            return aoclsparse_status_invalid_value;

    bool is_lower  = fill == aoclsparse_fill_mode_lower;
    bool keep_diag = diag == aoclsparse_diag_type_non_unit;
    bool unit_diag = diag == aoclsparse_diag_type_unit;

    if(beta == 0.)
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    else
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int idxend = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int idx = csr_row_ptr[i] - base; idx < idxend; idx++)
        {
            // valid indices given the specified triangle
            aoclsparse_int j = csr_col_ind[idx] - base;
            if(j >= 0 && j < i)
            {
                if(is_lower)
                    y[i] += alpha * csr_val[idx] * x[j];
            }
            else if(j == i)
            {
                if(keep_diag)
                    y[i] += alpha * csr_val[idx] * x[i];
            }
            else if(j > i && i < m)
            {
                if(!is_lower)
                    y[i] += alpha * csr_val[idx] * x[j];
            }
        }
        // if diag is aoclsparse_diag_type_zero, then we do not add the diagonal
        // leading to strict triangular (lower/upper) spmv
        if(unit_diag)
        {
            // add unit diagonal
            y[i] += alpha * x[i];
        }
    }

    return aoclsparse_status_success;
}

/* Generic SPMV with a mxn matrix stored as CSR: y = alpha*A*x + beta*y
 * This is a safe & slow implementation, indices out of bounds are ignored,
 * duplicate entries are summed. Zero/one-based indexing is supported.
 */
template <typename T>
aoclsparse_status ref_csrmv(T                     alpha,
                            aoclsparse_int        m,
                            aoclsparse_int        n,
                            const T              *csr_val,
                            const aoclsparse_int *csr_col_ind,
                            const aoclsparse_int *csr_row_ptr,
                            aoclsparse_index_base base,
                            const T              *x,
                            T                     beta,
                            T                    *y)
{

    if(csr_val == nullptr || csr_col_ind == nullptr || csr_row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(m < 0 || n < 0)
        return aoclsparse_status_invalid_size;

    // nothing to do case
    if(m == 0 || n == 0)
        return aoclsparse_status_success;

    // check validity of csr_row_ptr
    if(csr_row_ptr[0] < base)
        return aoclsparse_status_invalid_value;
    for(aoclsparse_int i = 0; i < m; i++)
        if(csr_row_ptr[i] > csr_row_ptr[i + 1])
            return aoclsparse_status_invalid_value;

    if(beta == 0.)
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    else
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int idxend = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int idx = csr_row_ptr[i] - base; idx < idxend; idx++)
        {
            // valid indices given the specified triangle
            aoclsparse_int j = csr_col_ind[idx] - base;
            if(j >= 0 && j < n)
                y[i] += alpha * csr_val[idx] * x[j];
        }
    }

    return aoclsparse_status_success;
}

/* Generic SPMV transposed with a mxn matrix stored as CSR:
 *   y = alpha*A'*x + beta*y
 * This is a safe & slow implementation, indices out of bounds are ignored,
 * duplicate entries are summed. Zero/one-based indexing is supported.
 */
template <typename T>
aoclsparse_status ref_csrmvt(T                     alpha,
                             aoclsparse_int        m,
                             aoclsparse_int        n,
                             const T              *csr_val,
                             const aoclsparse_int *csr_col_ind,
                             const aoclsparse_int *csr_row_ptr,
                             aoclsparse_index_base base,
                             const T              *x,
                             T                     beta,
                             T                    *y)
{

    if(csr_val == nullptr || csr_col_ind == nullptr || csr_row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(m < 0 || n < 0)
        return aoclsparse_status_invalid_size;

    // nothing to do case
    if(m == 0 || n == 0)
        return aoclsparse_status_success;

    // check validity of csr_row_ptr
    if(csr_row_ptr[0] < base)
        return aoclsparse_status_invalid_value;
    for(aoclsparse_int i = 0; i < m; i++)
        if(csr_row_ptr[i] > csr_row_ptr[i + 1])
            return aoclsparse_status_invalid_value;

    if(beta == 0.)
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = 0.;
    else
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = beta * y[i];

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int idxend = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int idx = csr_row_ptr[i] - base; idx < idxend; idx++)
        {
            // valid indices given the specified triangle
            aoclsparse_int j = csr_col_ind[idx] - base;
            if(j >= 0 && j < n)
                y[j] += alpha * csr_val[idx] * x[i];
        }
    }

    return aoclsparse_status_success;
}
/* Transposed-SPMV with a L or U triangular submatrix of a m x n matrix stored as CSR:
 *   y = alpha*[op(A)]^T*x + beta*y   where op=tril or triu
 * As symmetric version above but uses only the triangle (doesn't symmetrize).
 *
 * This is a safe & slow implementation, checking for out of bounds indices.
 * Supported:
 * - zero/one-based indexing
 * - L/U triangle
 * - non-unit/unit diagonal/zero diagonal
 * CSR doesn't need to be sorted in rows and doesn't need to contain only
 * the given triangle, the unuseful parts (and out of bounds indices)
 * are ignored. Duplicate entries are summed.
 */
template <typename T>
aoclsparse_status ref_csrmvtrgt(T                     alpha,
                                aoclsparse_int        m,
                                aoclsparse_int        n,
                                const T              *csr_val,
                                const aoclsparse_int *csr_col_ind,
                                const aoclsparse_int *csr_row_ptr,
                                aoclsparse_fill_mode  fill,
                                aoclsparse_diag_type  diag,
                                aoclsparse_index_base base,
                                const T              *x,
                                T                     beta,
                                T                    *y)
{

    if(csr_val == nullptr || csr_col_ind == nullptr || csr_row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(m < 0 || n < 0)
        return aoclsparse_status_invalid_size;

    // nothing to do case
    if(m == 0 || n == 0)
        return aoclsparse_status_success;

    // check validity of csr_row_ptr
    if(csr_row_ptr[0] < base)
        return aoclsparse_status_invalid_value;
    for(aoclsparse_int i = 0; i < m; i++)
        if(csr_row_ptr[i] > csr_row_ptr[i + 1])
            return aoclsparse_status_invalid_value;

    bool is_lower  = fill == aoclsparse_fill_mode_lower;
    bool keep_diag = diag == aoclsparse_diag_type_non_unit;
    bool unit_diag = diag == aoclsparse_diag_type_unit;

    if(beta == 0.)
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    else
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int idxend = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int idx = csr_row_ptr[i] - base; idx < idxend; idx++)
        {
            // valid indices given the specified triangle
            aoclsparse_int j = csr_col_ind[idx] - base;
            if(j >= 0 && j < i)
            {
                if(is_lower)
                {
                    y[j] += alpha * csr_val[idx] * x[i];
                }
            }
            else if(j == i)
            {
                if(keep_diag)
                {
                    y[i] += alpha * csr_val[idx] * x[i];
                }
            }
            else if(j > i && i < m)
            {
                if(!is_lower)
                {
                    y[j] += alpha * csr_val[idx] * x[i];
                }
            }
        }
        // if diag is aoclsparse_diag_type_zero, then we do not add the diagonal
        // leading to strict triangular (lower/upper) spmv
        if(unit_diag)
        {
            // add unit diagonal
            y[i] += alpha * x[i];
        }
    }
    return aoclsparse_status_success;
}
#endif // AOCLSPARSE_REFERENCE_HPP
