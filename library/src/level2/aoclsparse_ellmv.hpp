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
#ifndef AOCLSPARSE_ELLMV_HPP
#define AOCLSPARSE_ELLMV_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include <immintrin.h>

aoclsparse_status aoclsparse_ellmv(aoclsparse_operation      trans,
                                   const float                alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const float*              ell_val,
                                   const aoclsparse_int*      ell_col_ind,
                                   aoclsparse_int      ell_width,
                                   const aoclsparse_mat_descr descr,
                                   const float*             x,
                                   const float              beta,
                                   float*                   y )
{
    //TODO: Optimisation for float to be done
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        float result = 0.0;

        for(aoclsparse_int p = 0; p < ell_width; ++p)
        {
            aoclsparse_int idx = i * ell_width + p;
            aoclsparse_int col = ell_col_ind[idx] ;

            if(col >= 0)
            {
                result += (ell_val[idx] * x[col]);
            }
            else
            {
                break;
            }
        }

        if(alpha != static_cast<float>(1))
        {
            result = alpha * result;
        }

        if(beta != static_cast<float>(0))
        {
            result += beta * y[i];
        }

        y[i] = result ;
    }

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_ellmv(aoclsparse_operation      trans,
                                   const double               alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const double*              ell_val,
                                   const aoclsparse_int*      ell_col_ind,
                                   aoclsparse_int      ell_width,
                                   const aoclsparse_mat_descr descr,
                                   const double*             x,
                                   const double              beta,
                                   double*                   y )
{
    __m256d vec_vals , vec_x , vec_y;	
    aoclsparse_int k_iter = ell_width/4;
    aoclsparse_int k_rem = ell_width%4;

    for(aoclsparse_int i = 0; i < m; ++i)
    {
        double result = 0.0;
        aoclsparse_int idx = i * ell_width ;

        const aoclsparse_int *pell_col_ind;
        const double *pell_val;	    

        pell_col_ind = &ell_col_ind[idx];
        pell_val = &ell_val[idx];
        vec_y = _mm256_setzero_pd();
        
        //Loop over in multiple of 4
        for(aoclsparse_int p = 0; p < k_iter; ++p)
        {
            aoclsparse_int col = *pell_col_ind ;

            // Multiply only the valid non-zeroes, column index = -1 for padded zeroes 
            if(col >= 0)
            {
                //(ell_val[j] (ell_val[j+1] (ell_val[j+2] (ell_val[j+3]
                vec_vals = _mm256_loadu_pd((double const *)pell_val);

                vec_x  = _mm256_set_pd(x[*(pell_col_ind+3)],
                        x[*(pell_col_ind+2)],
                        x[*(pell_col_ind+1)],
                        x[*(pell_col_ind)]);

                vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);

                pell_val+=4;
                pell_col_ind+=4;
            }
            else
            {
                break;
            }
        }

        //Horizontal addition
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
            result = sse_sum[0];
        }

        //Remainder loop
        for(aoclsparse_int p = 0; p < k_rem; ++p)
        {
            aoclsparse_int col = *pell_col_ind ;

            if(col >= 0)
            {
                result += (*pell_val++ * x[*pell_col_ind++]);
            }
            else
            {
                break;
            }
        }

        if(alpha != static_cast<double>(1))
        {
            result = alpha * result;
        }

        if(beta != static_cast<double>(0))
        {
            result += beta * y[i];
        }

        y[i] = result ;
    }

    return aoclsparse_status_success;
}
#endif // AOCLSPARSE_ELLMV_HPP

