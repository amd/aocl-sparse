/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_utils.hpp"

#include <immintrin.h>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

template <typename T>
aoclsparse_status aoclsparse_csrmv_vectorized(aoclsparse_index_base base,
                                              const T               alpha,
                                              aoclsparse_int        m,
                                              const T *__restrict__ csr_val,
                                              const aoclsparse_int *__restrict__ csr_col_ind,
                                              const aoclsparse_int *__restrict__ csr_row_ptr,
                                              const T *__restrict__ x,
                                              const T beta,
                                              T *__restrict__ y,
                                              aoclsparse_context *context);

template <typename T>
aoclsparse_status aoclsparse_csrmv_vectorized_avx2(aoclsparse_index_base base,
                                                   const T               alpha,
                                                   aoclsparse_int        m,
                                                   const T *__restrict__ csr_val,
                                                   const aoclsparse_int *__restrict__ csr_col_ind,
                                                   const aoclsparse_int *__restrict__ csr_row_ptr,
                                                   const T *__restrict__ x,
                                                   const T beta,
                                                   T *__restrict__ y,
                                                   aoclsparse_context *context);

template <typename T>
aoclsparse_status aoclsparse_csrmv_vectorized_avx2ptr(const aoclsparse_mat_descr descr,
                                                      const T                    alpha,
                                                      aoclsparse_int             m,
                                                      aoclsparse_int             n,
                                                      aoclsparse_int             nnz,
                                                      const T *__restrict__ aval,
                                                      const aoclsparse_int *__restrict__ icol,
                                                      const aoclsparse_int *__restrict__ crstart,
                                                      const aoclsparse_int *__restrict__ crend,
                                                      const T *__restrict__ x,
                                                      const T beta,
                                                      T *__restrict__ y,
                                                      aoclsparse_context *context);

template <typename T>
aoclsparse_status aoclsparse_csrmv_vectorized_avx512(aoclsparse_index_base base,
                                                     const T               alpha,
                                                     aoclsparse_int        m,
                                                     const T *__restrict__ csr_val,
                                                     const aoclsparse_int *__restrict__ csr_col_ind,
                                                     const aoclsparse_int *__restrict__ csr_row_ptr,
                                                     const T *__restrict__ x,
                                                     const T beta,
                                                     T *__restrict__ y);

template <typename T>
aoclsparse_status aoclsparse_csrmv_general(aoclsparse_index_base base,
                                           const T               alpha,
                                           aoclsparse_int        m,
                                           const T *__restrict__ csr_val,
                                           const aoclsparse_int *__restrict__ csr_col_ind,
                                           const aoclsparse_int *__restrict__ csr_row_ptr,
                                           const T *__restrict__ x,
                                           const T beta,
                                           T *__restrict__ y,
                                           [[maybe_unused]] aoclsparse_context *context)
{
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *x_fix           = x - base;
    /*
        to avoid base correction logic inside core time-sensitive loops, the base addresses of column index,
        csr values and x vector are corrected in advance as per base. Then the correction will not be needed
        inside the core loops.
        Notice, the j-loop which runs for row pointers, has no corrections wrt base.
        In case of one-based indexed array, the loop runs from 1 to nnz+1.
        So the accesses made by csr_col_ind, csr_val and x need to be carefully adjusted.
        The above hack takes care of this by indexing into correct values.

        eg:
        csr_col_ind[j] - base
        is same as

        *(csr_col_ind + j) - base
        which is equal to
        *(csr_col_ind - base) + j
        which is equal to
        csr_col_ind_fix[j] (assuming, csr_col_ind_fix = csr_col_ind - base and j >=1 if base = 1)
    */
#ifdef _OPENMP
    aoclsparse_int chunk = (m / context->num_threads) ? (m / context->num_threads) : 1;
#pragma omp parallel for num_threads(context->num_threads) schedule(dynamic, chunk)
#endif
    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        T result = 0.0;

        for(aoclsparse_int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
        {
            result += csr_val_fix[j] * x_fix[csr_col_ind_fix[j]];
        }

        // Perform alpha * A * x
        if(alpha != static_cast<T>(1))
        {
            result = alpha * result;
        }

        // Perform (beta * y) + (alpha * A * x)
        if(beta != static_cast<T>(0))
        {
            result += beta * y[i];
        }

        y[i] = result;
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csrmv_symm(aoclsparse_index_base base,
                                        const T               alpha,
                                        aoclsparse_int        m,
                                        const T *__restrict__ csr_val,
                                        const aoclsparse_int *__restrict__ csr_col_ind,
                                        const aoclsparse_int *__restrict__ csr_row_ptr,
                                        const T *__restrict__ x,
                                        const T beta,
                                        T *__restrict__ y)
{
    // Perform (beta * y)
    if(beta == static_cast<T>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<T>(1))
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
        aoclsparse_int diag_idx      = csr_row_ptr[i + 1] - base - 1;
        aoclsparse_int last_ele_diag = !((csr_col_ind[diag_idx] - base) ^ i);
        y[i] += last_ele_diag * alpha * csr_val[diag_idx] * x[i];
        aoclsparse_int end = csr_row_ptr[i + 1] - base - last_ele_diag;
        // Handle all the elements in a row other than the diagonal element
        // Each element has an equivelant occurence on other side of the
        // diagonal and hence need to multiply with two offsets of x-vector
        // and update 2 offsets of y-vector
        for(aoclsparse_int j = (csr_row_ptr[i] - base); j < end; j++)
        {
            y[i] += alpha * csr_val[j] * x[csr_col_ind[j] - base];
            y[csr_col_ind[j] - base] += alpha * csr_val[j] * x[i];
        }
    }
    return aoclsparse_status_success;
}

/* this is adjusted aoclsparse_csrmv_symm() to work above opt_csr_mat for L & U triangle
 * computes if L:
 *   y = beta*y + alpha*(L+D+L')*x    if diag_type=non_unit  or
 *   y = beta*y + alpha*(L+I+L')*x    if diag_type=unit or
 *   y = beta*y + alpha*(L+L')*x      if diag_type=zero
 * and if U:
 *   y = beta*y + alpha*(U'+D+U)*x    if diag_type=non_unit  or
 *   y = beta*y + alpha*(U'+I+U)*x    if diag_type=unit or
 *   y = beta*y + alpha*(U'+U)*x      if diag_type=zero
 * where L & D & U are strictly L triangle and diag and U of the matrix
 * I is identity, assumes diag is always present
 *
 * n & nnz kicked out of the interface as not needed
 */
template <typename T>
aoclsparse_status
    aoclsparse_csrmv_symm_internal(aoclsparse_index_base base,
                                   T                     alpha,
                                   aoclsparse_int        m,
                                   aoclsparse_diag_type  diag_type,
                                   aoclsparse_fill_mode  fill_mode,
                                   const T *__restrict__ csr_val,
                                   const aoclsparse_int *__restrict__ csr_icol,
                                   const aoclsparse_int *__restrict__ csr_icrow,
                                   const aoclsparse_int *__restrict__ csr_idiag,
                                   [[maybe_unused]] const aoclsparse_int *__restrict__ csr_iurow,
                                   const T *__restrict__ x,
                                   T beta,
                                   T *__restrict__ y)
{
    // TODO test pointers & etc? Perhaps not needed, this will be called above optimized data
    // so probably just what came from the user, i.e., x & y?

    aoclsparse_int i, j, idx, idxstart, idxend;
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
            idxstart = csr_icrow[i] - base;
            // strictly L elements in each row are icrow[i]..idiag[i]-1
            idxend = csr_idiag[i] - base;
            // multiply with all strictly L triangle elements (and their transpose)
            for(idx = idxstart; idx < idxend; idx++)
            {
                val = alpha * csr_val[idx];
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += val * x[i];
            }
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idxend] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
            //else zero diagonal
        }
    }
    else
    { // fill_mode==aoclsparse_fill_mode_upper
        for(i = 0; i < m; i++)
        {
            // diag is at csr_idiag[i]
            idx = csr_idiag[i] - base;
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idx] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
            //else zero diagonal

            // strictly U elements in each row are idiag[i]+1..icrow[i+1]-1
            idxend = csr_icrow[i + 1] - base;
            // multiply with all strictly U triangle elements (and their transpose)
            for(idx = idx + 1; idx < idxend; idx++)
            {
                val = alpha * csr_val[idx];
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += val * x[i];
            }
        }
    }
    return aoclsparse_status_success;
}

/* This is a modified version aoclsparse_csrmv_symm_internal() to work
 * with conjugate transpose
 */
template <typename T>
aoclsparse_status
    aoclsparse_csrmvh_symm_internal(aoclsparse_index_base base,
                                    T                     alpha,
                                    aoclsparse_int        m,
                                    aoclsparse_diag_type  diag_type,
                                    aoclsparse_fill_mode  fill_mode,
                                    const T *__restrict__ csr_val,
                                    const aoclsparse_int *__restrict__ csr_icol,
                                    const aoclsparse_int *__restrict__ csr_icrow,
                                    const aoclsparse_int *__restrict__ csr_idiag,
                                    [[maybe_unused]] const aoclsparse_int *__restrict__ csr_iurow,
                                    const T *__restrict__ x,
                                    T beta,
                                    T *__restrict__ y)
{
    // TODO test pointers & etc? Perhaps not needed, this will be called above optimized data
    // so probably just what came from the user, i.e., x & y?

    aoclsparse_int i, j, idx, idxstart, idxend;
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
            idxstart = csr_icrow[i] - base;
            // strictly L elements in each row are icrow[i]..idiag[i]-1
            idxend = csr_idiag[i] - base;
            // multiply with all strictry L triangle elements (and their transpose)
            for(idx = idxstart; idx < idxend; idx++)
            {
                val = alpha * aoclsparse::conj(csr_val[idx]);
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += val * x[i];
            }
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * aoclsparse::conj(csr_val[idxend]) * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
            //else zero diagonal
        }
    }
    else
    { // fill_mode==aoclsparse_fill_mode_upper
        for(i = 0; i < m; i++)
        {
            // diag is at csr_idiag[i]
            idx = csr_idiag[i] - base;
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
            //else zero diagonal

            // strictly U elements in each row are idiag[i]+1..icrow[i+1]-1
            idxend = csr_icrow[i + 1] - base;
            // multiply with all strictly U triangle elements (and their transpose)
            for(idx = idx + 1; idx < idxend; idx++)
            {
                val = alpha * aoclsparse::conj(csr_val[idx]);
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += val * x[i];
            }
        }
    }
    return aoclsparse_status_success;
}

/* The following function is a modified version of aoclsparse_csrmv_symm_internal() to work with hermitian matrices.
 * This function is called when the matrix type is hermitian and the operation is either aoclsparse_operation_none or
 * aoclsparse_operation_conjugate_transpose.
 */
template <typename T>
aoclsparse_status
    aoclsparse_csrmv_herm_internal(aoclsparse_index_base base,
                                   T                     alpha,
                                   aoclsparse_int        m,
                                   aoclsparse_diag_type  diag_type,
                                   aoclsparse_fill_mode  fill_mode,
                                   const T *__restrict__ csr_val,
                                   const aoclsparse_int *__restrict__ csr_icol,
                                   const aoclsparse_int *__restrict__ csr_icrow,
                                   const aoclsparse_int *__restrict__ csr_idiag,
                                   [[maybe_unused]] const aoclsparse_int *__restrict__ csr_iurow,
                                   const T *__restrict__ x,
                                   T beta,
                                   T *__restrict__ y)
{

    aoclsparse_int i, j, idx, idxstart, idxend;
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
            idxstart = csr_icrow[i] - base;
            // strictly L elements in each row are icrow[i]..idiag[i]-1
            idxend = csr_idiag[i] - base;
            // multiply with all strictly L triangle elements (and their transpose)
            for(idx = idxstart; idx < idxend; idx++)
            {
                val = alpha * csr_val[idx];
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
            }
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idxend] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
            //else zero diagonal
        }
    }
    else
    { // fill_mode==aoclsparse_fill_mode_upper
        for(i = 0; i < m; i++)
        {
            // diag is at csr_idiag[i]
            idx = csr_idiag[i] - base;
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idx] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
            //else zero diagonal

            // strictly U elements in each row are idiag[i]+1..icrow[i+1]-1
            idxend = csr_icrow[i + 1] - base;
            // multiply with all strictly U triangle elements (and their transpose)
            for(idx = idx + 1; idx < idxend; idx++)
            {
                val = alpha * csr_val[idx];
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += alpha * aoclsparse::conj(csr_val[idx]) * x[i];
            }
        }
    }
    return aoclsparse_status_success;
}

/* this is adjusted aoclsparse_csrmv_symm_internal() to work with hermitian matrices
 * for transpose op
 */
template <typename T>
aoclsparse_status
    aoclsparse_csrmv_hermt_internal(aoclsparse_index_base base,
                                    T                     alpha,
                                    aoclsparse_int        m,
                                    aoclsparse_diag_type  diag_type,
                                    aoclsparse_fill_mode  fill_mode,
                                    const T *__restrict__ csr_val,
                                    const aoclsparse_int *__restrict__ csr_icol,
                                    const aoclsparse_int *__restrict__ csr_icrow,
                                    const aoclsparse_int *__restrict__ csr_idiag,
                                    [[maybe_unused]] const aoclsparse_int *__restrict__ csr_iurow,
                                    const T *__restrict__ x,
                                    T beta,
                                    T *__restrict__ y)
{

    aoclsparse_int i, j, idx, idxstart, idxend;
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
            idxstart = csr_icrow[i] - base;
            // strictly L elements in each row are icrow[i]..idiag[i]-1
            idxend = csr_idiag[i] - base;
            // multiply with all strictly L triangle elements (and their transpose)
            for(idx = idxstart; idx < idxend; idx++)
            {
                val = alpha * aoclsparse::conj(csr_val[idx]);
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += (alpha * csr_val[idx] * x[i]);
            }
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idxend] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
            //else zero diagonal
        }
    }
    else
    { // fill_mode==aoclsparse_fill_mode_upper
        for(i = 0; i < m; i++)
        {
            // diag is at csr_idiag[i]
            idx = csr_idiag[i] - base;
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idx] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
            //else zero diagonal

            // strictly U elements in each row are idiag[i]+1..icrow[i+1]-1
            idxend = csr_icrow[i + 1] - base;
            // multiply with all strictly U triangle elements (and their transpose)
            for(idx = idx + 1; idx < idxend; idx++)
            {
                val = alpha * aoclsparse::conj(csr_val[idx]);
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += (alpha * csr_val[idx] * x[i]);
            }
        }
    }
    return aoclsparse_status_success;
}

/* Transposed SPMV
 * ============================
 * Performs SPMV operation on the transposed CSR sparse matrix and
 * x-vector.
 */
template <typename T>
aoclsparse_status aoclsparse_csrmvt(aoclsparse_index_base base,
                                    const T               alpha,
                                    aoclsparse_int        m,
                                    aoclsparse_int        n,
                                    const T *__restrict__ csr_val,
                                    const aoclsparse_int *__restrict__ csr_col_ind,
                                    const aoclsparse_int *__restrict__ csr_row_ptr,
                                    const T *__restrict__ x,
                                    const T beta,
                                    T *__restrict__ y)
{
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    T                    *y_fix           = y - base;
    if(beta == static_cast<T>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < n; i++)
        {
            y[i] = 0.0;
        }
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < n; i++)
        {
            y[i] = beta * y[i];
        }
    }
    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int row_start = csr_row_ptr[i];
        aoclsparse_int row_end   = csr_row_ptr[i + 1];
        T              axi       = alpha * x[i];
        for(aoclsparse_int j = row_start; j < row_end; j++)
        {
            aoclsparse_int col_idx = csr_col_ind_fix[j];
            y_fix[col_idx] += csr_val_fix[j] * axi;
        }
    }
    return aoclsparse_status_success;
}

/* Conjugate Transposed SPMV
 * ============================
 * Performs SPMV operation on the conjugate transposed CSR sparse matrix
 * and x-vector.
 */
template <typename T>
aoclsparse_status aoclsparse_csrmvh(aoclsparse_index_base base,
                                    const T               alpha,
                                    aoclsparse_int        m,
                                    aoclsparse_int        n,
                                    const T *__restrict__ csr_val,
                                    const aoclsparse_int *__restrict__ csr_col_ind,
                                    const aoclsparse_int *__restrict__ csr_row_ptr,
                                    const T *__restrict__ x,
                                    const T beta,
                                    T *__restrict__ y)
{
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    T                    *y_fix           = y - base;
    if(beta == static_cast<T>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < n; i++)
        {
            y[i] = 0.0;
        }
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < n; i++)
        {
            y[i] = beta * y[i];
        }
    }
    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int row_start = csr_row_ptr[i];
        aoclsparse_int row_end   = csr_row_ptr[i + 1];
        T              axi       = alpha * x[i];
        for(aoclsparse_int j = row_start; j < row_end; j++)
        {
            aoclsparse_int col_idx = csr_col_ind_fix[j];
            y_fix[col_idx] += aoclsparse::conj(csr_val_fix[j]) * axi;
        }
    }
    return aoclsparse_status_success;
}

/* Transposed SPMV
 * ============================
 * Performs SPMV operation on the transposed CSR sparse matrix and
 * x-vector, where rows of the matrix are given by the start and end
 * pointer (useful when a specific triangle part of the matrix is provided)
 */
template <typename T>
aoclsparse_status aoclsparse_csrmvt_ptr(const aoclsparse_mat_descr      descr,
                                        const T                         alpha,
                                        aoclsparse_int                  m,
                                        [[maybe_unused]] aoclsparse_int n,
                                        const T *__restrict__ csr_val,
                                        const aoclsparse_int *__restrict__ csr_col_ind,
                                        const aoclsparse_int *__restrict__ crstart,
                                        const aoclsparse_int *__restrict__ crend,
                                        const T *__restrict__ x,
                                        const T beta,
                                        T *__restrict__ y)
{
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    T                    *y_fix           = y - base;
    aoclsparse_int        start_offset = 0, end_offset = 0;

    // if the matrix is triangular without explicit diagonal,
    // compute corrections for start and end pointers
    if((descr->type != aoclsparse_matrix_type_general)
       && (descr->diag_type == aoclsparse_diag_type_unit
           || descr->diag_type == aoclsparse_diag_type_zero))
    {
        if(descr->fill_mode == aoclsparse_fill_mode_lower) /* L triangle */
            end_offset = -1;
        else /*U triangle*/
            start_offset = 1;
    }
    bool diag_first = start_offset && descr->diag_type == aoclsparse_diag_type_unit;
    bool diag_last  = end_offset && descr->diag_type == aoclsparse_diag_type_unit;

    if(beta == static_cast<T>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < n; i++)
        {
            y[i] = 0.0;
        }
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < n; i++)
        {
            y[i] = beta * y[i];
        }
    }

    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int rstart = crstart[i] + start_offset;
        aoclsparse_int rend   = crend[i] + end_offset;

        T axi = alpha * x[i];
        if(diag_first)
            y_fix[i + base] += axi;
        for(aoclsparse_int j = rstart; j < rend; j++)
        {
            aoclsparse_int col_idx = csr_col_ind_fix[j];
            y_fix[col_idx] += csr_val_fix[j] * axi;
        }

        if(diag_last)
            y_fix[i + base] += axi;
    }
    return aoclsparse_status_success;
}

/* Conjugate Transposed SPMV
 * ============================
 * Performs SPMV operation on the conjugated transposed CSR sparse matrix
 * and x-vector, where rows of the matrix are given by the start and end
 * pointer (useful when a specific triangle part of the matrix is provided)
 */
template <typename T>
aoclsparse_status aoclsparse_csrmvh_ptr(const aoclsparse_mat_descr descr,
                                        const T                    alpha,
                                        aoclsparse_int             m,
                                        aoclsparse_int             n,
                                        const T *__restrict__ csr_val,
                                        const aoclsparse_int *__restrict__ csr_col_ind,
                                        const aoclsparse_int *__restrict__ crstart,
                                        const aoclsparse_int *__restrict__ crend,
                                        const T *__restrict__ x,
                                        const T beta,
                                        T *__restrict__ y)
{
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    T                    *y_fix           = y - base;
    aoclsparse_int        start_offset = 0, end_offset = 0;

    // if the matrix is triangular without explicit diagonal,
    // compute corrections for start and end pointers
    if((descr->type != aoclsparse_matrix_type_general)
       && (descr->diag_type == aoclsparse_diag_type_unit
           || descr->diag_type == aoclsparse_diag_type_zero))
    {
        if(descr->fill_mode == aoclsparse_fill_mode_lower) /* L triangle */
            end_offset = -1;
        else /*U triangle*/
            start_offset = 1;
    }
    bool diag_first = start_offset && descr->diag_type == aoclsparse_diag_type_unit;
    bool diag_last  = end_offset && descr->diag_type == aoclsparse_diag_type_unit;

    if(beta == static_cast<T>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < n; i++)
        {
            y[i] = 0.0;
        }
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < n; i++)
        {
            y[i] = beta * y[i];
        }
    }

    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int rstart = crstart[i] + start_offset;
        aoclsparse_int rend   = crend[i] + end_offset;

        T axi = alpha * x[i];
        if(diag_first)
            y_fix[i + base] += axi;
        for(aoclsparse_int j = rstart; j < rend; j++)
        {
            aoclsparse_int col_idx = csr_col_ind_fix[j];
            y_fix[col_idx] += aoclsparse::conj(csr_val_fix[j]) * axi;
        }

        if(diag_last)
            y_fix[i + base] += axi;
    }
    return aoclsparse_status_success;
}

/* SPMV
 * ============================
 * Performs SPMV operation on the CSR sparse matrix and
 * x-vector, where rows of the matrix are given by the start and end
 * pointer (useful when a specific triangle part of the matrix is provided)
 */
template <typename T>
aoclsparse_status aoclsparse_csrmv_ptr(aoclsparse_mat_descr            descr,
                                       const T                         alpha,
                                       aoclsparse_int                  m,
                                       [[maybe_unused]] aoclsparse_int n,
                                       const T *__restrict__ csr_val,
                                       const aoclsparse_int *__restrict__ csr_col_ind,
                                       const aoclsparse_int *__restrict__ crstart,
                                       const aoclsparse_int *__restrict__ crend,
                                       const T *__restrict__ x,
                                       const T beta,
                                       T *__restrict__ y,
                                       [[maybe_unused]] aoclsparse_context *context)
{
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *x_fix           = x - base;
    T                     one             = 1;
    T                     zero            = 0;
    aoclsparse_int        start_offset = 0, end_offset = 0;

    // if the matrix is triangular without explicit diagonal,
    // compute corrections for start and end pointers
    if((descr->type != aoclsparse_matrix_type_general)
       && (descr->diag_type == aoclsparse_diag_type_unit
           || descr->diag_type == aoclsparse_diag_type_zero))
    {
        if(descr->fill_mode == aoclsparse_fill_mode_lower) /* L triangle */
            end_offset = -1;
        else /*U triangle*/
            start_offset = 1;
    }
    bool diag_first = start_offset && descr->diag_type == aoclsparse_diag_type_unit;
    bool diag_last  = end_offset && descr->diag_type == aoclsparse_diag_type_unit;

    if(beta == zero)
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < m; i++)
        {
            y[i] = 0.0;
        }
    }
    else if(beta != one)
    {
        for(aoclsparse_int i = 0; i < m; i++)
        {
            y[i] = beta * y[i];
        }
    }
    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int rstart = crstart[i] + start_offset;
        aoclsparse_int rend   = crend[i] + end_offset;

        T result = 0.0;
        if(diag_first)
            result += x_fix[i + base];

        for(aoclsparse_int j = rstart; j < rend; j++)
        {
            result += csr_val_fix[j] * x_fix[csr_col_ind_fix[j]];
        }
        if(diag_last)
            result += x_fix[i + base];

        y[i] += alpha * result;
    }
    return aoclsparse_status_success;
}

#endif // AOCLSPARSE_CSRMV_HPP
