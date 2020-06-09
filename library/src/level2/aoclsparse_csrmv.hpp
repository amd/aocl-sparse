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


aoclsparse_status aoclsparse_csrmv_general(aoclsparse_int             m,
        aoclsparse_int             n,
        aoclsparse_int             nnz,
        const double              alpha,
        const double* __restrict__ csr_val,
        const aoclsparse_int* __restrict__ csr_row_ptr,
        const aoclsparse_int* __restrict__ csr_col_ind,
        const double* __restrict__ x,
        const double             beta,
        double* __restrict__        y)
{
    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    const aoclsparse_int *colIndPtr;
    const double *matValPtr;
    matValPtr = &csr_val[csr_row_ptr[0]];
    colIndPtr = &csr_col_ind[csr_row_ptr[0]];

    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        double result = 0.0;
        aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];

        for(aoclsparse_int j =  0 ; j < nnz ; j++ )
        {
            result += *matValPtr++ * x[*colIndPtr++];
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

        y[i] = result ;
    } 

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csrmv_vectorized(aoclsparse_int             m,
        aoclsparse_int             n,
        aoclsparse_int             nnz,
        const double              alpha,
        const double* __restrict__ csr_val,
        const aoclsparse_int* __restrict__ csr_row_ptr,
        const aoclsparse_int* __restrict__ csr_col_ind,
        const double* __restrict__ x,
        const double             beta,
        double* __restrict__        y)
{
    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    __m256d vec_vals , vec_x , vec_y;
    const aoclsparse_int *colIndPtr;
    const double *matValPtr;
    matValPtr = &csr_val[csr_row_ptr[0]];
    colIndPtr = &csr_col_ind[csr_row_ptr[0]];

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int j;
        double result = 0.0;
        vec_y = _mm256_setzero_pd();
        aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i]; 
        aoclsparse_int k_iter = nnz/4;
        aoclsparse_int k_rem = nnz%4;

        //Loop in multiples of 4 non-zeroes
        for(j =  0 ; j < k_iter ; j++ )		
        {
            //(csr_val[j] (csr_val[j+1] (csr_val[j+2] (csr_val[j+3]
            vec_vals = _mm256_loadu_pd((double const *)matValPtr);

            //Gather the x vector elements from the column indices
            vec_x  = _mm256_set_pd(x[*(colIndPtr+3)],
                    x[*(colIndPtr+2)],
                    x[*(colIndPtr+1)],
                    x[*(colIndPtr)]);

            vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);

            matValPtr+=4;
            colIndPtr+=4;
        }

        // Horizontal addition
        if(k_iter){
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

        //Remainder loop for nnz%4
        for(j =  0 ; j < k_rem ; j++ )		
        {
            result += *matValPtr++ * x[*colIndPtr++];
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

        y[i] = result ;
    } 

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csrmv(aoclsparse_int           m,
        aoclsparse_int             n,
        aoclsparse_int             nnz,
        const float              alpha,
        const float* __restrict__ csr_val,
        const aoclsparse_int* __restrict__ csr_row_ptr,
        const aoclsparse_int* __restrict__ csr_col_ind,
        const float* __restrict__ x,
        const float             beta,
        float* __restrict__        y)
{
    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    __m256 vec_vals , vec_x ,vec_y;
    const aoclsparse_int *colIndPtr;
    const float *matValPtr;
    matValPtr = &csr_val[csr_row_ptr[0]];
    colIndPtr = &csr_col_ind[csr_row_ptr[0]];

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int j;
        float result = 0.0;
        vec_y = _mm256_setzero_ps();
        aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i]; 
        aoclsparse_int k_iter = nnz/8;
        aoclsparse_int k_rem = nnz%8;

        //Loop in multiples of 8
        for(j =  0 ; j < k_iter ; j++ )		
        {
            //(csr_val[j] csr_val[j+1] csr_val[j+2] csr_val[j+3] csr_val[j+4] csr_val[j+5] csr_val[j+6] csr_val[j+7]
            vec_vals = _mm256_loadu_ps(matValPtr);

            //Gather the xvector values from the column indices
            vec_x  = _mm256_set_ps(x[*(colIndPtr+7)],
                    x[*(colIndPtr+6)],
                    x[*(colIndPtr+5)],
                    x[*(colIndPtr+4)],
                    x[*(colIndPtr+3)],
                    x[*(colIndPtr+2)],
                    x[*(colIndPtr+1)],
                    x[*(colIndPtr)]);

            vec_y = _mm256_fmadd_ps(vec_vals, vec_x , vec_y); 

            matValPtr += 8;
            colIndPtr += 8;
        }

        // Horizontal addition of vec_y
        if(k_iter){
            // hiQuad = ( x7, x6, x5, x4 )            
            __m128 hiQuad = _mm256_extractf128_ps(vec_y, 1);
            // loQuad = ( x3, x2, x1, x0 )
            const __m128 loQuad = _mm256_castps256_ps128(vec_y);
            // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
            const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
            // loDual = ( -, -, x1 + x5, x0 + x4 )
            const __m128 loDual = sumQuad;
            // hiDual = ( -, -, x3 + x7, x2 + x6 )
            const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
            // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
            const __m128 sumDual = _mm_add_ps(loDual, hiDual);
            // lo = ( -, -, -, x0 + x2 + x4 + x6 )
            const __m128 lo = sumDual;
            // hi = ( -, -, -, x1 + x3 + x5 + x7 )
            const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
            // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
            const __m128 sum = _mm_add_ss(lo, hi);
            result = _mm_cvtss_f32(sum);
        }

        //Remainder loop
        for(j =  0 ; j < k_rem ; j++ )		
        {
            result += *matValPtr++ * x[*colIndPtr++];
        }

        // Perform alpha * A * x
        if(alpha != static_cast<float>(1))
        {
            result = alpha * result;
        }

        // Perform (beta * y) + (alpha * A * x)
        if(beta != static_cast<float>(0))
        {
            result += beta * y[i];
        }

        y[i] = result ;
    } 

    return aoclsparse_status_success;
}
#endif // AOCLSPARSE_CSRMV_HPP

