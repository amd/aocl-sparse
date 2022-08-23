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
#ifndef AOCLSPARSE_CSRMV_HPP
#define AOCLSPARSE_CSRMV_HPP

#include <immintrin.h>
#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_context.h"

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

template <typename T>
aoclsparse_status aoclsparse_csrmv_vectorized(const T        alpha,
                                              aoclsparse_int m,
                                              aoclsparse_int n,
                                              aoclsparse_int nnz,
                                              const T* __restrict__ csr_val,
                                              const aoclsparse_int* __restrict__ csr_col_ind,
                                              const aoclsparse_int* __restrict__ csr_row_ptr,
                                              const T* __restrict__ x,
                                              const T beta,
                                              T* __restrict__ y,
                                              aoclsparse_context* context);

template <typename T>
aoclsparse_status aoclsparse_csrmv_vectorized_avx2(const T        alpha,
                                              aoclsparse_int m,
                                              aoclsparse_int n,
                                              aoclsparse_int nnz,
                                              const T* __restrict__ csr_val,
                                              const aoclsparse_int* __restrict__ csr_col_ind,
                                              const aoclsparse_int* __restrict__ csr_row_ptr,
                                              const T* __restrict__ x,
                                              const T beta,
                                              T* __restrict__ y,
                                              aoclsparse_context* context);

template <typename T>
aoclsparse_status aoclsparse_csrmv_vectorized_avx512(const T        alpha,
                                              aoclsparse_int m,
                                              aoclsparse_int n,
                                              aoclsparse_int nnz,
                                              const T* __restrict__ csr_val,
                                              const aoclsparse_int* __restrict__ csr_col_ind,
                                              const aoclsparse_int* __restrict__ csr_row_ptr,
                                              const T* __restrict__ x,
                                              const T beta,
                                              T* __restrict__ y,
                                              aoclsparse_context* context);

template <typename T>
aoclsparse_status aoclsparse_csrmv_general(const T        alpha,
                                           aoclsparse_int m,
                                           aoclsparse_int n,
                                           aoclsparse_int nnz,
                                           const T* __restrict__ csr_val,
                                           const aoclsparse_int* __restrict__ csr_col_ind,
                                           const aoclsparse_int* __restrict__ csr_row_ptr,
                                           const T* __restrict__ x,
                                           const T beta,
                                           T* __restrict__ y,
                                           aoclsparse_context* context)
{

#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads) schedule(dynamic, m / context->num_threads)
#endif
    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        T result = 0.0;

        for(aoclsparse_int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
        {
            result += csr_val[j] * x[csr_col_ind[j]];
        }

        // Perform alpha * A * x
        if(alpha != static_cast<double>(1))
        {
            result = alpha * result;
        }

        // Perform (beta * y) + (alpha * A * x)
        if(beta != static_cast<double>(0))
        {
            result += beta * y[i];
        }

        y[i] = result;
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csrmv_symm(const T        alpha,
                                        aoclsparse_int m,
                                        aoclsparse_int n,
                                        aoclsparse_int nnz,
                                        const T* __restrict__ csr_val,
                                        const aoclsparse_int* __restrict__ csr_col_ind,
                                        const aoclsparse_int* __restrict__ csr_row_ptr,
                                        const T* __restrict__ x,
                                        const T beta,
                                        T* __restrict__ y)
{
    // Perform (beta * y)
    if(beta != static_cast<double>(1))
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }
    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        // Diagonal element(if a non-zero) has to be multiplied once
        // in a symmetrc matrix with corresponding x vector element.
        // Last element of every row in a lower triangular symmetric
        // matrix is diagonal element. Diagonal element when equal to
        // zero , will not be present in the csr_val array . To
        // handle this corner case , last_ele_diag is initialised to
        // zero if diagonal element is zero, hence not multiplied.
        // last_ele_diag becomes one if diagonal element is non-zero
        // and hence multiplied once with corresponding x-vector element
        aoclsparse_int diag_idx      = csr_row_ptr[i + 1] - 1;
        aoclsparse_int last_ele_diag = !(csr_col_ind[diag_idx] ^ i);
        y[i] += last_ele_diag * alpha * csr_val[diag_idx] * x[i];
        aoclsparse_int end = csr_row_ptr[i + 1] - last_ele_diag;
        // Handle all the elements in a row other than the diagonal element
        // Each element has an equivelant occurence on other side of the
        // diagonal and hence need to multiply with two offsets of x-vector
        // and update 2 offsets of y-vector
        for(aoclsparse_int j = csr_row_ptr[i]; j < end; j++)
        {
            y[i] += alpha * csr_val[j] * x[csr_col_ind[j]];
            y[csr_col_ind[j]] += alpha * csr_val[j] * x[i];
        }
    }
    return aoclsparse_status_success;
}

/* this is adjusted aoclsparse_csrmv_symm() to work above opt_csr_mat for L & U triangle
 * computes if L:
 *   y = beta*y + alpha*(L+D+L')*x    if diag_type=non_unit  or
 *   y = beta*y + alpha*(L+I+L')*x    if diag_type=unit
 * and if U:
 *   y = beta*y + alpha*(U'+D+U)*x    if diag_type=non_unit  or
 *   y = beta*y + alpha*(U'+I+U)*x    if diag_type=unit
 * where L & D & U are strictly L triangle and diag and U of the matrix
 * I is identity, assumes diag is always present
 *
 * n & nnz kicked out of the interface as not needed
 */
template <typename T>
aoclsparse_status aoclsparse_csrmv_symm_internal(T                    alpha,
                                                 aoclsparse_int       m,
                                                 aoclsparse_diag_type diag_type,
                                                 aoclsparse_fill_mode fill_mode,
                                                 const T* __restrict__ csr_val,
                                                 const aoclsparse_int* __restrict__ csr_icol,
                                                 const aoclsparse_int* __restrict__ csr_icrow,
                                                 const aoclsparse_int* __restrict__ csr_idiag,
                                                 const aoclsparse_int* __restrict__ csr_iurow,
                                                 const T* __restrict__ x,
                                                 T beta,
                                                 T* __restrict__ y)
{
    // TODO test pointers & etc? Perhaps not needed, this will be called above optimized data
    // so probably just what came from the user, i.e., x & y?

    aoclsparse_int i, j, idx, idxend;
    T              val;

    // Perform (beta * y)
    if(beta == static_cast<T>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<T>(1))
    {
        for(i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    if(fill_mode == aoclsparse_fill_mode_lower)
    {
        for(i = 0; i < m; i++)
        {
            // strictly L elements in each row are icrow[i]..idiag[i]-1
            idxend = csr_idiag[i];
            // multiply with all strictry L triangle elements (and their transpose)
            for(idx = csr_icrow[i]; idx < idxend; idx++)
            {
                val = alpha * csr_val[idx];
                j   = csr_icol[idx];
                y[i] += val * x[j];
                y[j] += val * x[i];
            }
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idxend] * x[i];
            else // unit diagonal
                y[i] += alpha * x[i];
        }
    }
    else
    { // fill_mode==aoclsparse_fill_mode_upper
        for(i = 0; i < m; i++)
        {
            // diag is at csr_idiag[i]
            idx = csr_idiag[i];
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idx] * x[i];
            else // unit diagonal
                y[i] += alpha * x[i];
            // strictly U elements in each row are idiag[i]+1..icrow[i+1]-1
            idxend = csr_icrow[i + 1];
            // multiply with all strictry L triangle elements (and their transpose)
            for(idx = idx + 1; idx < idxend; idx++)
            {
                val = alpha * csr_val[idx];
                j   = csr_icol[idx];
                y[i] += val * x[j];
                y[j] += val * x[i];
            }
        }
    }
    return aoclsparse_status_success;
}

#endif // AOCLSPARSE_CSRMV_HPP
