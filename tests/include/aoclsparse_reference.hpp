/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 * Portions of this file consist of AI-generated content.
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
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_utils.hpp"

#include <map>

template <typename T>
inline aoclsparse_status ref_csr2csc(aoclsparse_index_base base,
                                     aoclsparse_int        M,
                                     aoclsparse_int        N,
                                     aoclsparse_int       *row_ptr_in,
                                     aoclsparse_int       *col_ind_in,
                                     T                    *val_in,
                                     aoclsparse_int       *row_out,
                                     aoclsparse_int       *col_out,
                                     T                    *val_out)
{
    std::map<aoclsparse_int, std::map<aoclsparse_int, T>> out_map;

    for(aoclsparse_int i = 0; i < M; ++i)
    {
        aoclsparse_int row = i + base;
        for(aoclsparse_int j = row_ptr_in[i] - base; j < row_ptr_in[i + 1] - base; ++j)
        {
            aoclsparse_int col = col_ind_in[j];
            T              val = val_in[j];
            out_map[col][row]  = val;
        }
    }

    int idx = 0;

    for(auto [col, row_val_map] : out_map)
    {
        col_out[col - base + 1] = row_val_map.size();
        for(auto [row, val] : row_val_map)
        {
            row_out[idx] = row;
            val_out[idx] = val;
            idx++;
        }
    }

    col_out[0] = base;
    for(int i = 0; i < N; i++)
    {
        col_out[i + 1] += col_out[i];
    }

    return aoclsparse_status_success;
}

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

    if(base != aoclsparse_index_base_zero && base != aoclsparse_index_base_one)
        return aoclsparse_status_invalid_value;
    if(fill != aoclsparse_fill_mode_lower && fill != aoclsparse_fill_mode_upper)
        return aoclsparse_status_invalid_value;
    if(diag != aoclsparse_diag_type_non_unit && diag != aoclsparse_diag_type_unit
       && diag != aoclsparse_diag_type_zero)
        return aoclsparse_status_invalid_value;

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

    if(beta == aoclsparse_numeric::zero<T>())
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = aoclsparse_numeric::zero<T>();
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
            else if(j > i && j < m)
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
 *   y = alpha*op2(op1(A))*x + beta*y
 * where op1=tril or triu and op2 = none/transpose/conjugate_transpose
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
aoclsparse_status ref_csrmvtrg(aoclsparse_operation  op,
                               T                     alpha,
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

    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;
    if(base != aoclsparse_index_base_zero && base != aoclsparse_index_base_one)
        return aoclsparse_status_invalid_value;
    if(fill != aoclsparse_fill_mode_lower && fill != aoclsparse_fill_mode_upper)
        return aoclsparse_status_invalid_value;
    if(diag != aoclsparse_diag_type_non_unit && diag != aoclsparse_diag_type_unit
       && diag != aoclsparse_diag_type_zero)
        return aoclsparse_status_invalid_value;

    if(m < 0 || n < 0)
        return aoclsparse_status_invalid_size;

    // check validity of csr_row_ptr
    if(csr_row_ptr[0] < base)
        return aoclsparse_status_invalid_value;
    for(aoclsparse_int i = 0; i < m; i++)
        if(csr_row_ptr[i] > csr_row_ptr[i + 1])
            return aoclsparse_status_invalid_value;

    bool is_lower  = fill == aoclsparse_fill_mode_lower;
    bool keep_diag = diag == aoclsparse_diag_type_non_unit;
    bool unit_diag = diag == aoclsparse_diag_type_unit;

    aoclsparse_int y_dim = op == aoclsparse_operation_none ? m : n;
    if(beta == aoclsparse_numeric::zero<T>())
        for(aoclsparse_int i = 0; i < y_dim; i++)
            y[i] = aoclsparse_numeric::zero<T>();
    else
        for(aoclsparse_int i = 0; i < y_dim; i++)
            y[i] = beta * y[i];

    // nothing to do case
    if(m == 0 || n == 0)
        return aoclsparse_status_success;

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int idxend = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int idx = csr_row_ptr[i] - base; idx < idxend; idx++)
        {
            // valid indices given the specified triangle
            aoclsparse_int j = csr_col_ind[idx] - base;
            if(j >= n)
                continue;
            if(j >= 0 && j < i)
            {
                if(is_lower)
                {
                    if(op == aoclsparse_operation_none)
                        y[i] += alpha * csr_val[idx] * x[j];
                    else if(op == aoclsparse_operation_transpose)
                        y[j] += alpha * csr_val[idx] * x[i];
                    else // op==aoclsparse_operation_conjugate_transpose
                        y[j] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
                }
            }
            else if(j == i)
            {
                if(keep_diag)
                {
                    if(op == aoclsparse_operation_conjugate_transpose)
                        y[i] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
                    else
                        y[i] += alpha * csr_val[idx] * x[i];
                }
            }
            else if(j > i)
            {
                if(!is_lower)
                {
                    if(op == aoclsparse_operation_none)
                        y[i] += alpha * csr_val[idx] * x[j];
                    else if(op == aoclsparse_operation_transpose)
                        y[j] += alpha * csr_val[idx] * x[i];
                    else // op==aoclsparse_operation_conjugate_transpose
                        y[j] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
                }
            }
        }
        // if diag is aoclsparse_diag_type_zero, then we do not add the diagonal
        // leading to strict triangular (lower/upper) spmv
        if(unit_diag)
        {
            // add unit diagonal
            if(i < n)
                y[i] += alpha * x[i];
        }
    }

    return aoclsparse_status_success;
}

/* SPMV with a general mxn matrix stored as CSR: y = alpha*op(A)*x + beta*y
 * where op is either none, transpose or conjugate_transpose.
 * This is a safe & slow implementation, indices out of bounds are ignored,
 * duplicate entries are summed. Zero/one-based indexing is supported.
 */
template <typename T>
aoclsparse_status ref_csrmvgen(aoclsparse_operation  op,
                               T                     alpha,
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

    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;
    if(base != aoclsparse_index_base_zero && base != aoclsparse_index_base_one)
        return aoclsparse_status_invalid_value;

    if(m < 0 || n < 0)
        return aoclsparse_status_invalid_size;

    // check validity of csr_row_ptr
    if(csr_row_ptr[0] < base)
        return aoclsparse_status_invalid_value;
    for(aoclsparse_int i = 0; i < m; i++)
        if(csr_row_ptr[i] > csr_row_ptr[i + 1])
            return aoclsparse_status_invalid_value;

    aoclsparse_int y_dim = op == aoclsparse_operation_none ? m : n;
    if(beta == aoclsparse_numeric::zero<T>())
        for(aoclsparse_int i = 0; i < y_dim; i++)
            y[i] = aoclsparse_numeric::zero<T>();
    else
        for(aoclsparse_int i = 0; i < y_dim; i++)
            y[i] = beta * y[i];

    // nothing to do case
    if(m == 0 || n == 0)
        return aoclsparse_status_success;

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int idxend = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int idx = csr_row_ptr[i] - base; idx < idxend; idx++)
        {
            // valid indices given the specified triangle
            aoclsparse_int j = csr_col_ind[idx] - base;
            if(j >= 0 && j < n)
            {
                if(op == aoclsparse_operation_none)
                    y[i] += alpha * csr_val[idx] * x[j];
                else if(op == aoclsparse_operation_transpose)
                    y[j] += alpha * csr_val[idx] * x[i];
                else // op==aoclsparse_operation_conjugate_transpose
                    y[j] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
            }
        }
    }

    return aoclsparse_status_success;
}

/* Hermitian SPMV with a mxm matrix stored as CSR: y = alpha*A*x + beta*y
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
aoclsparse_status ref_csrmvher(T                     alpha,
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

    if(base != aoclsparse_index_base_zero && base != aoclsparse_index_base_one)
        return aoclsparse_status_invalid_value;
    if(fill != aoclsparse_fill_mode_lower && fill != aoclsparse_fill_mode_upper)
        return aoclsparse_status_invalid_value;
    if(diag != aoclsparse_diag_type_non_unit && diag != aoclsparse_diag_type_unit
       && diag != aoclsparse_diag_type_zero)
        return aoclsparse_status_invalid_value;

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

    if(beta == aoclsparse_numeric::zero<T>())
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = aoclsparse_numeric::zero<T>();
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
                    y[j] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
                }
            }
            else if(j == i)
            {
                // assumption is that the diagonal is real but we don't check it here
                if(keep_diag)
                    y[i] += alpha * csr_val[idx] * x[i];
            }
            else if(j > i && j < m)
            {
                if(!is_lower)
                {
                    y[i] += alpha * csr_val[idx] * x[j];
                    y[j] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
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

/* One generic SPMV wrapper encapsulating all operations and matrix 'descriptors'
 * calling the reference implementations above.
 * Assumes a mxn matrix stored as CSR: y = alpha*op(A)*x + beta*y
 * where op is either none, transpose or conjugate_transpose and the matrix
 * interpretation can be modified via mattype, fill and diag.
 * This is a safe & slow implementation, indices out of bounds are ignored,
 * duplicate entries are summed. Zero/one-based indexing is supported.
 */
template <typename T>
aoclsparse_status ref_csrmv(aoclsparse_operation   op,
                            T                      alpha,
                            aoclsparse_int         m,
                            aoclsparse_int         n,
                            const T               *csr_val,
                            const aoclsparse_int  *csr_col_ind,
                            const aoclsparse_int  *csr_row_ptr,
                            aoclsparse_matrix_type mattype,
                            aoclsparse_fill_mode   fill,
                            aoclsparse_diag_type   diag,
                            aoclsparse_index_base  base,
                            const T               *x,
                            T                      beta,
                            T                     *y)
{

    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;

    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        if(op == aoclsparse_operation_conjugate_transpose)
            op = aoclsparse_operation_transpose;

    if(mattype == aoclsparse_matrix_type_general)
        return ref_csrmvgen(op, alpha, m, n, csr_val, csr_col_ind, csr_row_ptr, base, x, beta, y);
    else if(mattype == aoclsparse_matrix_type_symmetric)
    {
        // conjugate_transpose would require conjugate kernel which is
        // probably not needed
        if(op == aoclsparse_operation_conjugate_transpose)
            return aoclsparse_status_not_implemented;
        if(m != n)
            return aoclsparse_status_invalid_value;

        return ref_csrmvsym(
            alpha, m, csr_val, csr_col_ind, csr_row_ptr, fill, diag, base, x, beta, y);
    }
    else if(mattype == aoclsparse_matrix_type_triangular)
    {
        return ref_csrmvtrg(
            op, alpha, m, n, csr_val, csr_col_ind, csr_row_ptr, fill, diag, base, x, beta, y);
    }
    else if(mattype == aoclsparse_matrix_type_hermitian)
    {
        // transpose on hermitian matrix would require conjugate kernel which
        // is probably not needed
        if(op == aoclsparse_operation_transpose)
            return aoclsparse_status_not_implemented;
        if(m != n)
            return aoclsparse_status_invalid_value;

        return ref_csrmvher(
            alpha, m, csr_val, csr_col_ind, csr_row_ptr, fill, diag, base, x, beta, y);
    }
    else
        return aoclsparse_status_invalid_value;
}

/* Specializations to allow working with aoclsparse_*_complex types
 * by casting them to std::complex<> */
template <>
inline aoclsparse_status ref_csrmv(aoclsparse_operation            op,
                                   aoclsparse_float_complex        alpha,
                                   aoclsparse_int                  m,
                                   aoclsparse_int                  n,
                                   const aoclsparse_float_complex *csr_val,
                                   const aoclsparse_int           *csr_col_ind,
                                   const aoclsparse_int           *csr_row_ptr,
                                   aoclsparse_matrix_type          mattype,
                                   aoclsparse_fill_mode            fill,
                                   aoclsparse_diag_type            diag,
                                   aoclsparse_index_base           base,
                                   const aoclsparse_float_complex *x,
                                   aoclsparse_float_complex        beta,
                                   aoclsparse_float_complex       *y)
{
    std::complex<float> *alphap = reinterpret_cast<std::complex<float> *>(&alpha);
    std::complex<float> *betap  = reinterpret_cast<std::complex<float> *>(&beta);
    return ref_csrmv<std::complex<float>>(op,
                                          *alphap,
                                          m,
                                          n,
                                          reinterpret_cast<const std::complex<float> *>(csr_val),
                                          csr_col_ind,
                                          csr_row_ptr,
                                          mattype,
                                          fill,
                                          diag,
                                          base,
                                          reinterpret_cast<const std::complex<float> *>(x),
                                          *betap,
                                          reinterpret_cast<std::complex<float> *>(y));
}

template <>
inline aoclsparse_status ref_csrmv(aoclsparse_operation             op,
                                   aoclsparse_double_complex        alpha,
                                   aoclsparse_int                   m,
                                   aoclsparse_int                   n,
                                   const aoclsparse_double_complex *csr_val,
                                   const aoclsparse_int            *csr_col_ind,
                                   const aoclsparse_int            *csr_row_ptr,
                                   aoclsparse_matrix_type           mattype,
                                   aoclsparse_fill_mode             fill,
                                   aoclsparse_diag_type             diag,
                                   aoclsparse_index_base            base,
                                   const aoclsparse_double_complex *x,
                                   aoclsparse_double_complex        beta,
                                   aoclsparse_double_complex       *y)
{
    std::complex<double> *alphap = reinterpret_cast<std::complex<double> *>(&alpha);
    std::complex<double> *betap  = reinterpret_cast<std::complex<double> *>(&beta);
    return ref_csrmv<std::complex<double>>(op,
                                           *alphap,
                                           m,
                                           n,
                                           reinterpret_cast<const std::complex<double> *>(csr_val),
                                           csr_col_ind,
                                           csr_row_ptr,
                                           mattype,
                                           fill,
                                           diag,
                                           base,
                                           reinterpret_cast<const std::complex<double> *>(x),
                                           *betap,
                                           reinterpret_cast<std::complex<double> *>(y));
}

/* Gather entries from dense y[] vector to a a sparse vector(x, nnz, indx) based indices from indx array
 */
template <typename T>
inline aoclsparse_status
    ref_gather(const aoclsparse_int nnz, const T *y, T *x, const aoclsparse_int *indx)
{
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        if(indx[i] < 0)
            return aoclsparse_status_invalid_index_value;
        x[i] = y[indx[i]];
    }
    return aoclsparse_status_success;
}
/* Gather entries from dense y[] vector to a a sparse vector(x, nnz, indx) based indices from indx array.
   Also set the y[] indices addressed by indx[] to zero
 */
template <typename T>
inline aoclsparse_status
    ref_gatherz(const aoclsparse_int nnz, const T *y, T *y_gold, T *x, const aoclsparse_int *indx)
{
    if(x == nullptr || y == nullptr || y_gold == nullptr)
        return aoclsparse_status_invalid_pointer;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        if(indx[i] < 0)
            return aoclsparse_status_invalid_index_value;
        x[i]            = y[indx[i]];
        y_gold[indx[i]] = aoclsparse_numeric::zero<T>();
    }
    return aoclsparse_status_success;
}
/* Gather entries from dense y[] vector using strided loads and save them to a sparse vector(x, nnz, indx) based on stride applied to y[]
 */
template <typename T>
inline aoclsparse_status
    ref_gathers(const aoclsparse_int nnz, const T *y, T *x, const aoclsparse_int stride)
{
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        x[i] = y[i * stride];
    }
    return aoclsparse_status_success;
}
/* scatter entries from sparse x[] vector(x, nnz, indx) to a dense vector y[] at indices defined by indx[]
 */
template <typename T>
inline aoclsparse_status
    ref_scatter(const aoclsparse_int nnz, const T *x, const aoclsparse_int *indx, T *y)
{
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        if(indx[i] < 0)
            return aoclsparse_status_invalid_index_value;
        y[indx[i]] = x[i];
    }
    return aoclsparse_status_success;
}
/* scatter entries from sparse x[] vector(x, nnz, indx) to a dense vector y[] with strided stores
 */
template <typename T>
inline aoclsparse_status
    ref_scatters(const aoclsparse_int nnz, const T *x, const aoclsparse_int stride, T *y)
{
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        y[i * stride] = x[i];
    }
    return aoclsparse_status_success;
}
/*  Accumulate product of constant scalar(real/complex) and sparse x[] vector(x, nnz, indx) to a dense vector y[]
    at indices defined by indx[]
 */
template <typename T>
inline aoclsparse_status
    ref_axpyi(const aoclsparse_int nnz, const T a, const T *x, const aoclsparse_int *indx, T *y)
{
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        if(indx[i] < 0)
            return aoclsparse_status_invalid_index_value;
        y[indx[i]] = a * x[i] + y[indx[i]];
    }
    return aoclsparse_status_success;
}
template <>
inline aoclsparse_status ref_axpyi(const aoclsparse_int            nnz,
                                   aoclsparse_float_complex        alpha,
                                   const aoclsparse_float_complex *x,
                                   const aoclsparse_int           *indx,
                                   aoclsparse_float_complex       *y)
{
    std::complex<float> *alphap = reinterpret_cast<std::complex<float> *>(&alpha);
    return ref_axpyi<std::complex<float>>(nnz,
                                          *alphap,
                                          reinterpret_cast<const std::complex<float> *>(x),
                                          indx,
                                          reinterpret_cast<std::complex<float> *>(y));
}
template <>
inline aoclsparse_status ref_axpyi(const aoclsparse_int             nnz,
                                   aoclsparse_double_complex        alpha,
                                   const aoclsparse_double_complex *x,
                                   const aoclsparse_int            *indx,
                                   aoclsparse_double_complex       *y)
{
    std::complex<double> *alphap = reinterpret_cast<std::complex<double> *>(&alpha);
    return ref_axpyi<std::complex<double>>(nnz,
                                           *alphap,
                                           reinterpret_cast<const std::complex<double> *>(x),
                                           indx,
                                           reinterpret_cast<std::complex<double> *>(y));
}
/* Compute the Givens rotation of a dense vector y[] and sparse vector x[] defined by (x, nnz, indx)
 */
template <typename T>
inline aoclsparse_status ref_givens_rot(const aoclsparse_int  nnz,
                                        const T              *x_in,
                                        const aoclsparse_int *indx,
                                        const T              *y_in,
                                        T                    *x_out,
                                        T                    *y_out,
                                        const T               c,
                                        const T               s)
{
    if(x_in == nullptr || y_in == nullptr || x_out == nullptr || y_out == nullptr)
        return aoclsparse_status_invalid_pointer;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        if(indx[i] < 0)
            return aoclsparse_status_invalid_index_value;
        x_out[i]       = c * x_in[i] + s * y_in[indx[i]];
        y_out[indx[i]] = c * y_in[indx[i]] - s * x_in[i];
    }
    return aoclsparse_status_success;
}

/* Dot product of two dense vectors y[] and x[]. Used in DOTMV */
template <typename T>
aoclsparse_status ref_dense_dot(const aoclsparse_int size, const T *x, const T *y, T *d)
{
    if(x == nullptr || y == nullptr || d == nullptr)
        return aoclsparse_status_invalid_pointer;
    *d = aoclsparse_numeric::zero<T>();
    if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>)
    {
        if constexpr(std::is_same_v<T, std::complex<float>>)
        {
            std::complex<double> result = {0.0, 0.0};
            for(aoclsparse_int i = 0; i < size; i++)
            {
                result += static_cast<std::complex<double>>(std::conj(x[i]))
                          * static_cast<std::complex<double>>(y[i]);
            }
            *d = static_cast<T>(result);
        }
        else
        {

            for(aoclsparse_int i = 0; i < size; i++)
            {
                *d += std::conj(x[i]) * y[i];
            }
        }
    }
    else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            // to increase the precision of the reference dot, compute it in higher precision
            double result = 0.0;
            for(aoclsparse_int i = 0; i < size; i++)
            {
                result += static_cast<double>(x[i]) * static_cast<double>(y[i]);
            }
            *d = static_cast<T>(result);
        }
        else
        {
            for(aoclsparse_int i = 0; i < size; i++)
            {
                *d += x[i] * y[i];
            }
        }
    }
    else
    {
        return aoclsparse_status_wrong_type;
    }
    return aoclsparse_status_success;
}

template <>
inline aoclsparse_status ref_dense_dot(const aoclsparse_int             size,
                                       const aoclsparse_double_complex *x,
                                       const aoclsparse_double_complex *y,
                                       aoclsparse_double_complex       *d)
{
    const std::complex<double> *xp = reinterpret_cast<const std::complex<double> *>(x);
    const std::complex<double> *yp = reinterpret_cast<const std::complex<double> *>(y);
    std::complex<double>       *dp = reinterpret_cast<std::complex<double> *>(d);
    return ref_dense_dot(size, xp, yp, dp);
}

template <>
inline aoclsparse_status ref_dense_dot(const aoclsparse_int            size,
                                       const aoclsparse_float_complex *x,
                                       const aoclsparse_float_complex *y,
                                       aoclsparse_float_complex       *d)
{
    const std::complex<float> *xp = reinterpret_cast<const std::complex<float> *>(x);
    const std::complex<float> *yp = reinterpret_cast<const std::complex<float> *>(y);
    std::complex<float>       *dp = reinterpret_cast<std::complex<float> *>(d);
    return ref_dense_dot(size, xp, yp, dp);
}

/* Dot product of a dense vector y[] and sparse vector x[] defined by (x, nnz, indx)
 */
template <typename T>
inline aoclsparse_status ref_doti(
    const aoclsparse_int nnz, const T *x, const aoclsparse_int *indx, const T *y, T &result)
{
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        if(indx[i] < 0)
            return aoclsparse_status_invalid_index_value;
        result += x[i] * y[indx[i]];
    }
    return aoclsparse_status_success;
}
/* Complex dot product of a complex dense vector y[] and complex sparse vector x[] defined by (x, nnz, indx).
    The result is complex conjugated dot product if is_conjugated is true
 */
template <typename T>
inline aoclsparse_status ref_complex_dot(const aoclsparse_int  nnz,
                                         const T              *x,
                                         const aoclsparse_int *indx,
                                         const T              *y,
                                         T                    *complex_dot,
                                         bool                  is_conjugated)
{
    if(x == nullptr || y == nullptr || complex_dot == nullptr || indx == nullptr)
        return aoclsparse_status_invalid_pointer;

    *complex_dot = 0;
    if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>)
    {
        if(is_conjugated)
        {
            for(aoclsparse_int i = 0; i < nnz; i++)
            {
                if(indx[i] < 0)
                    return aoclsparse_status_invalid_index_value;
                *complex_dot += std::conj(x[i]) * y[indx[i]];
            }
        }
        else
        {
            for(aoclsparse_int i = 0; i < nnz; i++)
            {
                if(indx[i] < 0)
                    return aoclsparse_status_invalid_index_value;
                *complex_dot += x[i] * y[indx[i]];
            }
        }
    }
    else
    {
        // this function is to be used only with complex types that require complex dot product
        return aoclsparse_status_invalid_value;
    }
    return aoclsparse_status_success;
}
template <>
inline aoclsparse_status ref_complex_dot(const aoclsparse_int            nnz,
                                         const aoclsparse_float_complex *x,
                                         const aoclsparse_int           *indx,
                                         const aoclsparse_float_complex *y,
                                         aoclsparse_float_complex       *complex_dot,
                                         bool                            is_conjugated)
{
    return ref_complex_dot<std::complex<float>>(
        nnz,
        reinterpret_cast<const std::complex<float> *>(x),
        indx,
        reinterpret_cast<const std::complex<float> *>(y),
        reinterpret_cast<std::complex<float> *>(complex_dot),
        is_conjugated);
}
template <>
inline aoclsparse_status ref_complex_dot(const aoclsparse_int             nnz,
                                         const aoclsparse_double_complex *x,
                                         const aoclsparse_int            *indx,
                                         const aoclsparse_double_complex *y,
                                         aoclsparse_double_complex       *complex_dot,
                                         bool                             is_conjugated)
{
    return ref_complex_dot<std::complex<double>>(
        nnz,
        reinterpret_cast<const std::complex<double> *>(x),
        indx,
        reinterpret_cast<const std::complex<double> *>(y),
        reinterpret_cast<std::complex<double> *>(complex_dot),
        is_conjugated);
}
/* CSRMM reference implementation, supports float/double/complex datatypes.
Computes multiplication of a sparse matrix 'A' in CSR format and a dense matrix 'B',
resulting a dense vector 'C'. C = alpha * A * B + beta * C, where alpha and beta are scalars. */
template <typename T>
aoclsparse_status ref_csrmm_gen(T                     alpha,
                                aoclsparse_index_base base,
                                aoclsparse_order      order,
                                const T *__restrict__ csr_val,
                                const aoclsparse_int *__restrict__ csr_col_ind,
                                const aoclsparse_int *__restrict__ csr_row_ptr,
                                aoclsparse_int m,
                                const T       *B,
                                aoclsparse_int n,
                                aoclsparse_int ldb,
                                T              beta,
                                T             *C,
                                aoclsparse_int ldc)

{
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        for(aoclsparse_int j = 0; j < n; ++j)
        {
            aoclsparse_int idx_C;
            if(order == aoclsparse_order_column)
                idx_C = i + j * ldc;
            else
                idx_C = i * ldc + j;
            T sum = static_cast<T>(0);

            for(aoclsparse_int k = row_begin; k < row_end; ++k)
            {
                aoclsparse_int idx_B;
                if(order == aoclsparse_order_column)
                    idx_B = ((csr_col_ind[k] - base) + j * ldb);
                else
                    idx_B = (j + (csr_col_ind[k] - base) * ldb);
                sum += csr_val[k] * B[idx_B];
            }
            C[idx_C] = beta * C[idx_C] + (alpha * sum);
        }
    }
    return aoclsparse_status_success;
}
template <typename T>
aoclsparse_status ref_csrmm(aoclsparse_operation   op,
                            T                      alpha,
                            T                     *csr_val,
                            aoclsparse_int        *csr_col_ind,
                            aoclsparse_int        *csr_row_ptr,
                            aoclsparse_matrix_type mattype,
                            aoclsparse_index_base  base,
                            aoclsparse_order       order,
                            aoclsparse_int         m,
                            aoclsparse_int         k,
                            const T               *B,
                            aoclsparse_int         n,
                            aoclsparse_int         ldb,
                            T                      beta,
                            T                     *C,
                            aoclsparse_int         ldc)
{
    T zero = 0.0;
    T one  = 1.0;

    // Check for nullptr
    if(csr_val == nullptr || csr_col_ind == nullptr || csr_row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(B == nullptr || C == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Check for valid size
    if(m < 0 || n < 0 || k < 0)
        return aoclsparse_status_invalid_size;

    // Check for valid op
    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;
    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        if(op == aoclsparse_operation_conjugate_transpose)
            op = aoclsparse_operation_transpose;

    // Base check
    if(base != aoclsparse_index_base_zero && base != aoclsparse_index_base_one)
        return aoclsparse_status_invalid_value;

    // Matrix type check
    if(mattype != aoclsparse_matrix_type_general && mattype != aoclsparse_matrix_type_symmetric
       && mattype != aoclsparse_matrix_type_hermitian)
        return aoclsparse_status_not_implemented;
    if((mattype == aoclsparse_matrix_type_symmetric || mattype == aoclsparse_matrix_type_hermitian)
       && m != k)
        return aoclsparse_status_invalid_size;

    // Check order
    if(order != aoclsparse_order_row && order != aoclsparse_order_column)
        return aoclsparse_status_invalid_value;

    // quick return
    if(m == 0 || n == 0 || k == 0)
        return aoclsparse_status_success;
    if(alpha == zero && beta == one)
        return aoclsparse_status_success;

    // Check leading dimension of B and C
    aoclsparse_int check_ldb, check_ldc;
    check_ldb = (order == aoclsparse_order_column ? k : n);
    check_ldc = (order == aoclsparse_order_column ? m : n);

    if(ldb < (((aoclsparse_int)1) >= check_ldb ? (aoclsparse_int)1 : check_ldb))
        return aoclsparse_status_invalid_size;
    if(ldc < (((aoclsparse_int)1) >= check_ldc ? (aoclsparse_int)1 : check_ldc))
        return aoclsparse_status_invalid_size;

    // Transpose/conjugate transpose of general matrix type
    if(mattype == aoclsparse_matrix_type_general
       && (op == aoclsparse_operation_transpose || op == aoclsparse_operation_conjugate_transpose))
    {
        aoclsparse_int              nnz = csr_row_ptr[k] - base;
        std::vector<aoclsparse_int> csc_row_ind, csc_col_ptr;
        std::vector<T>              csc_val;

        csc_row_ind.resize(nnz);
        csc_col_ptr.resize(m + 1, 0);
        csc_val.resize(nnz);
        aoclsparse_status status = ref_csr2csc(base,
                                               k,
                                               m,
                                               csr_row_ptr,
                                               csr_col_ind,
                                               csr_val,
                                               csc_row_ind.data(),
                                               csc_col_ptr.data(),
                                               csc_val.data());
        if(status != aoclsparse_status_success)
            return status;

        if(op == aoclsparse_operation_conjugate_transpose)
        {
            for(aoclsparse_int idx = 0; idx < nnz; idx++)
                csc_val[idx] = aoclsparse::conj(csc_val[idx]);
        }

        return ref_csrmm_gen(alpha,
                             base,
                             order,
                             csc_val.data(),
                             csc_row_ind.data(),
                             csc_col_ptr.data(),
                             m,
                             B,
                             n,
                             ldb,
                             beta,
                             C,
                             ldc);
    }
    else
        return ref_csrmm_gen(
            alpha, base, order, csr_val, csr_col_ind, csr_row_ptr, m, B, n, ldb, beta, C, ldc);
    return aoclsparse_status_not_implemented;
}
template <>
inline aoclsparse_status ref_csrmm(aoclsparse_operation            op,
                                   aoclsparse_float_complex        alpha,
                                   aoclsparse_float_complex       *csr_val,
                                   aoclsparse_int                 *csr_col_ind,
                                   aoclsparse_int                 *csr_row_ptr,
                                   aoclsparse_matrix_type          mattype,
                                   aoclsparse_index_base           base,
                                   aoclsparse_order                order,
                                   aoclsparse_int                  m,
                                   [[maybe_unused]] aoclsparse_int k,
                                   const aoclsparse_float_complex *B,
                                   aoclsparse_int                  n,
                                   aoclsparse_int                  ldb,
                                   aoclsparse_float_complex        beta,
                                   aoclsparse_float_complex       *C,
                                   aoclsparse_int                  ldc)
{
    std::complex<float> *alphap = reinterpret_cast<std::complex<float> *>(&alpha);
    std::complex<float> *betap  = reinterpret_cast<std::complex<float> *>(&beta);
    return ref_csrmm<std::complex<float>>(op,
                                          *alphap,
                                          reinterpret_cast<std::complex<float> *>(csr_val),
                                          csr_col_ind,
                                          csr_row_ptr,
                                          mattype,
                                          base,
                                          order,
                                          m,
                                          k,
                                          reinterpret_cast<const std::complex<float> *>(B),
                                          n,
                                          ldb,
                                          *betap,
                                          reinterpret_cast<std::complex<float> *>(C),
                                          ldc);
}
template <>
inline aoclsparse_status ref_csrmm(aoclsparse_operation             op,
                                   aoclsparse_double_complex        alpha,
                                   aoclsparse_double_complex       *csr_val,
                                   aoclsparse_int                  *csr_col_ind,
                                   aoclsparse_int                  *csr_row_ptr,
                                   aoclsparse_matrix_type           mattype,
                                   aoclsparse_index_base            base,
                                   aoclsparse_order                 order,
                                   aoclsparse_int                   m,
                                   [[maybe_unused]] aoclsparse_int  k,
                                   const aoclsparse_double_complex *B,
                                   aoclsparse_int                   n,
                                   aoclsparse_int                   ldb,
                                   aoclsparse_double_complex        beta,
                                   aoclsparse_double_complex       *C,
                                   aoclsparse_int                   ldc)
{
    std::complex<double> *alphap = reinterpret_cast<std::complex<double> *>(&alpha);
    std::complex<double> *betap  = reinterpret_cast<std::complex<double> *>(&beta);
    return ref_csrmm<std::complex<double>>(op,
                                           *alphap,
                                           reinterpret_cast<std::complex<double> *>(csr_val),
                                           csr_col_ind,
                                           csr_row_ptr,
                                           mattype,
                                           base,
                                           order,
                                           m,
                                           k,
                                           reinterpret_cast<const std::complex<double> *>(B),
                                           n,
                                           ldb,
                                           *betap,
                                           reinterpret_cast<std::complex<double> *>(C),
                                           ldc);
}
#endif // AOCLSPARSE_REFERENCE_HPP
