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
 * ************************************************************************ */
#ifndef __AVX512F__
#error "File contains AVX512 kernels and needs to be compiled with AVX512 flags."
#else
#include "aoclsparse_context.h"
#include "aoclsparse_csrmv_avx512.hpp"

#include <immintrin.h>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

aoclsparse_status aoclsparse_csrmv_vectorized_avx512(aoclsparse_index_base base,
                                                     const double          alpha,
                                                     aoclsparse_int        m,
                                                     const double *__restrict__ csr_val,
                                                     const aoclsparse_int *__restrict__ csr_col_ind,
                                                     const aoclsparse_int *__restrict__ csr_row_ptr,
                                                     const double *__restrict__ x,
                                                     const double beta,
                                                     double *__restrict__ y)
{
    using namespace aoclsparse;
    __m256d               vec_y;
    __m512d               vec_vals_512, vec_x_512, vec_y_512;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const double         *csr_val_fix     = csr_val - base;
    const double         *x_fix           = x - base;

#ifdef _OPENMP
    aoclsparse_int        chunk           = (m / context::get_context()->get_num_threads())
                                                ? (m / context::get_context()->get_num_threads())
                                                : 1;
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk) private(vec_vals_512, vec_x_512, vec_y_512, vec_y)
#endif

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int j;
        double         result = 0.0;
        aoclsparse_int nnz    = csr_row_ptr[i + 1] - csr_row_ptr[i];
        aoclsparse_int k_iter = nnz / 8;
        aoclsparse_int k_rem  = nnz % 8;

        vec_y_512 = _mm512_setzero_pd();

        // Loop in multiples of 8 non-zeroes
        for(j = csr_row_ptr[i]; j < (csr_row_ptr[i + 1] - k_rem); j += 8)
        {
            //(csr_val[j] (csr_val[j+1] (csr_val[j+2] (csr_val[j+3]
            vec_vals_512 = _mm512_loadu_pd(&csr_val_fix[j]);

            // Gather the x vector elements from the column indices
            vec_x_512 = _mm512_set_pd(x_fix[csr_col_ind_fix[j + 7]],
                                      x_fix[csr_col_ind_fix[j + 6]],
                                      x_fix[csr_col_ind_fix[j + 5]],
                                      x_fix[csr_col_ind_fix[j + 4]],
                                      x_fix[csr_col_ind_fix[j + 3]],
                                      x_fix[csr_col_ind_fix[j + 2]],
                                      x_fix[csr_col_ind_fix[j + 1]],
                                      x_fix[csr_col_ind_fix[j]]);

            vec_y_512 = _mm512_fmadd_pd(vec_vals_512, vec_x_512, vec_y_512);
        }
        vec_y = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512, 0x0),
                              _mm512_extractf64x4_pd(vec_y_512, 0x1));

        if(k_iter)
        {

            // sum[0] += sum[1] ; sum[2] += sum[3]
            vec_y = _mm256_hadd_pd(vec_y, vec_y);
            // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
            __m128d sum_lo = _mm256_castpd256_pd128(vec_y);
            // Extract 128 bits to obtain sum[2] and sum[3]
            __m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);

            // Add remaining two sums
            __m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);

            // Store result
            /*
           __m128d in gcc is typedef as double
           but in Windows, this is defined as a struct
           */
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
            result = sse_sum.m128d_f64[0];
#else
            result = sse_sum[0];
#endif
        }
        // Remainder loop for nnz%8
        for(j = csr_row_ptr[i + 1] - k_rem; j < csr_row_ptr[i + 1]; j++)
        {
            result += csr_val_fix[j] * x_fix[csr_col_ind_fix[j]];
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
#endif