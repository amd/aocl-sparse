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

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_mv.hpp"

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

aoclsparse_status aoclsparse_dtcsrmv_avx2(const aoclsparse_index_base base,
                                          const double                alpha,
                                          aoclsparse_int              m,
                                          const double *__restrict__ val_L,
                                          const double *__restrict__ val_U,
                                          const aoclsparse_int *__restrict__ col_idx_L,
                                          const aoclsparse_int *__restrict__ col_idx_U,
                                          const aoclsparse_int *__restrict__ row_ptr_L,
                                          const aoclsparse_int *__restrict__ row_ptr_U,
                                          const double *__restrict__ x,
                                          const double beta,
                                          double *__restrict__ y)
{
    using namespace aoclsparse;

    __m256d               vec_vals, vec_x, vec_y;
    const aoclsparse_int *icol_fix_L = col_idx_L - base;
    const aoclsparse_int *icol_fix_U = col_idx_U - base;
    const double         *aval_fix_L = val_L - base;
    const double         *aval_fix_U = val_U - base;
    const double         *x_fix      = x - base;

#ifdef _OPENMP
    aoclsparse_int chunk = (m / context::get_context()->get_num_threads())
                               ? (m / context::get_context()->get_num_threads())
                               : 1;
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk) private(vec_vals, vec_x, vec_y)
#endif
    for(aoclsparse_int i = 0; i < m; i++)
    {
        // (L+D)x computation
        double result = 0.0;
        vec_y         = _mm256_setzero_pd();
        aoclsparse_int j;

        aoclsparse_int rstart  = row_ptr_L[i];
        aoclsparse_int rend    = row_ptr_L[i + 1];
        aoclsparse_int nnz     = rend - rstart;
        aoclsparse_int k_iterL = nnz / 4;
        aoclsparse_int k_rem   = nnz % 4;

        for(j = rstart; j < rend - k_rem; j += 4)
        {
            vec_vals = _mm256_loadu_pd(&aval_fix_L[j]);
            //Gather the x vector elements from the column indices
            vec_x = _mm256_set_pd(x_fix[icol_fix_L[j + 3]],
                                  x_fix[icol_fix_L[j + 2]],
                                  x_fix[icol_fix_L[j + 1]],
                                  x_fix[icol_fix_L[j]]);
            vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);
        }

        //Remainder loop for nnz%4
        for(j = rend - k_rem; j < rend; j++)
        {
            result += aval_fix_L[j] * x_fix[icol_fix_L[j]];
        }

        // (U)x
        rstart                 = row_ptr_U[i] + 1;
        rend                   = row_ptr_U[i + 1];
        nnz                    = rend - rstart;
        aoclsparse_int k_iterU = nnz / 4;
        k_rem                  = nnz % 4;

        for(j = rstart; j < rend - k_rem; j += 4)
        {
            vec_vals = _mm256_loadu_pd(&aval_fix_U[j]);
            vec_x    = _mm256_set_pd(x_fix[icol_fix_U[j + 3]],
                                  x_fix[icol_fix_U[j + 2]],
                                  x_fix[icol_fix_U[j + 1]],
                                  x_fix[icol_fix_U[j + 0]]);
            vec_y    = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);
        }

        // Horizontal addition
        if(k_iterL || k_iterU)
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
            result += sse_sum.m128d_f64[0];
#else
            result += sse_sum[0];
#endif
        }

        //Remainder loop for nnz%4
        for(j = rend - k_rem; j < rend; j++)
        {
            result += aval_fix_U[j] * x_fix[icol_fix_U[j]];
        }

        // Perform alpha * A * x
        result = alpha * result;
        // Perform (beta * y) + (alpha * A * x)
        if(beta != static_cast<double>(0))
        {
            result += beta * y[i];
        }
        y[i] = result;
    }
    return aoclsparse_status_success;
}