/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <immintrin.h>

template <typename T>
aoclsparse_status aoclsparse_csrmv(aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
				   const T*              alpha,
                                   const T*                  csr_val,
                                   const aoclsparse_int*      csr_row_ptr,
                                   const aoclsparse_int*      csr_col_ind,
                                   const T*             x,
                                   const T*             beta,
                                   T*                   y
)
{
#if 0
    for(int i = 0; i < m; i++)
    {
        for(int j = csr_row_ptr[i] ; j < csr_row_ptr[i+1] ; j++)
        {
            y[i] += csr_val[j] * x[csr_col_ind[j]];
        }
   }
#else
    __m256d vec_vals , vec_x , vec_y;
    __m128i vec_idx;
    for(int i = 0; i < m; i++)
    {
        int j;
        const int *colIndPtr = &csr_col_ind[csr_row_ptr[i]];
        const T *matValPtr = &csr_val[csr_row_ptr[i]];
        T result = 0.0;
        vec_y = _mm256_setzero_pd();
	int nnz = csr_row_ptr[i+1] - csr_row_ptr[i]; 
        int k_iter = nnz/4;
        int k_rem = nnz%4;

        for(j =  0 ; j < k_iter ; j++ )		
        {
            vec_vals = _mm256_loadu_pd((double const *)matValPtr);//(csr_val[j] (csr_val[j+1] (csr_val[j+2] (csr_val[j+3]

            vec_x  = _mm256_set_pd(x[*(colIndPtr+3)],
                                     x[*(colIndPtr+2)],
                                     x[*(colIndPtr+1)],
                                     x[*(colIndPtr)]);
            vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);
            matValPtr+=4;
            colIndPtr+=4;
        }
     
     	if(k_iter){// Horizontal addition
            // sum[0] += sum[1] ; sum[2] += sum[3]
            vec_y = _mm256_hadd_pd(vec_y, vec_y);
            // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
            __m128d sum_lo = _mm256_castpd256_pd128(vec_y);
            // Extract 128 bits to obtain sum[2] and sum[3]
            __m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);
            // Add remaining two sums
            __m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
            // Store result
            result = sse_sum[0];
        }
        for(j =  0 ; j < k_rem ; j++ )		
        {
            result += *matValPtr++ * x[*colIndPtr++];
        }
        y[i] += result ;
    } 
#endif
    return aoclsparse_status_success;
}
#endif // AOCLSPARSE_CSRMV_HPP

