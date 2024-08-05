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
#include "aoclsparse_ellmv_avx512.hpp"

aoclsparse_status aoclsparse_dellmv_avx512(const double                    alpha,
                                           aoclsparse_int                  m,
                                           [[maybe_unused]] aoclsparse_int n,
                                           [[maybe_unused]] aoclsparse_int nnz,
                                           const double                   *ell_val,
                                           const aoclsparse_int           *ell_col_ind,
                                           aoclsparse_int                  ell_width,
                                           const aoclsparse_mat_descr      descr,
                                           const double                   *x,
                                           const double                    beta,
                                           double                         *y)
{
    __m256d vec_y;
    __m512d vec_vals_512, vec_x_512, vec_y_512;

    aoclsparse_index_base base   = descr->base;
    aoclsparse_int        k_iter = ell_width / 8;
    aoclsparse_int        k_rem  = ell_width % 8;

    for(aoclsparse_int i = 0; i < m; ++i)
    {
        double         result = 0.0;
        aoclsparse_int idx    = i * ell_width;
        k_rem                 = ell_width % 8;

        const aoclsparse_int *pell_col_ind;
        const double         *pell_val;

        pell_col_ind = &ell_col_ind[idx];
        pell_val     = &ell_val[idx];
        vec_y        = _mm256_setzero_pd();
        vec_y_512    = _mm512_setzero_pd();
        //Loop over in multiple of 4
        for(aoclsparse_int p = 0; p < k_iter; ++p)
        {
            aoclsparse_int col = *(pell_col_ind + 7) - base;
            // Multiply only the valid non-zeroes, column index = -1 for padded
            // zeroes
            if(col >= 0)
            {
                //(ell_val[j] (ell_val[j+1] (ell_val[j+2] (ell_val[j+3]
                vec_vals_512 = _mm512_loadu_pd(pell_val);
                vec_x_512    = _mm512_set_pd(x[*(pell_col_ind + 7) - base],
                                          x[*(pell_col_ind + 6) - base],
                                          x[*(pell_col_ind + 5) - base],
                                          x[*(pell_col_ind + 4) - base],
                                          x[*(pell_col_ind + 3) - base],
                                          x[*(pell_col_ind + 2) - base],
                                          x[*(pell_col_ind + 1) - base],
                                          x[*(pell_col_ind)-base]);

                vec_y_512 = _mm512_fmadd_pd(vec_vals_512, vec_x_512, vec_y_512);

                pell_val += 8;
                pell_col_ind += 8;
            }
            else
            {
                k_rem = 8;
                break;
            }
        }
        vec_y = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512, 0x0),
                              _mm512_extractf64x4_pd(vec_y_512, 0x1));

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

        //Remainder loop
        for(aoclsparse_int p = 0; p < k_rem; ++p)
        {
            aoclsparse_int col = *(pell_col_ind)-base;

            if(col >= 0)
            {
                result += (*pell_val++ * x[col]);
                pell_col_ind++;
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

        y[i] = result;
    }
    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_elltmv_avx512(const double                    alpha,
                                           aoclsparse_int                  m,
                                           [[maybe_unused]] aoclsparse_int n,
                                           [[maybe_unused]] aoclsparse_int nnz,
                                           const double                   *ell_val,
                                           const aoclsparse_int           *ell_col_ind,
                                           aoclsparse_int                  ell_width,
                                           const aoclsparse_mat_descr      descr,
                                           const double                   *x,
                                           const double                    beta,
                                           double                         *y)
{

    __m512d res, vvals, vx, vy, va, vb;

    va                         = _mm512_set1_pd(alpha);
    vb                         = _mm512_set1_pd(beta);
    res                        = _mm512_setzero_pd();
    aoclsparse_int        k    = ell_width;
    int                   blk  = 8;
    aoclsparse_index_base base = descr->base;
    for(aoclsparse_int j = 0; j < m / blk; j++)
    {
        res                 = _mm512_setzero_pd();
        aoclsparse_int joff = j * blk;
        for(aoclsparse_int i = 0; i < k; i++)
        {

            aoclsparse_int off = joff + i * m;

            vvals = _mm512_loadu_pd(ell_val + off);
            vx    = _mm512_set_pd(x[*(ell_col_ind + off + 7) - base],
                               x[*(ell_col_ind + off + 6) - base],
                               x[*(ell_col_ind + off + 5) - base],
                               x[*(ell_col_ind + off + 4) - base],
                               x[*(ell_col_ind + off + 3) - base],
                               x[*(ell_col_ind + off + 2) - base],
                               x[*(ell_col_ind + off + 1) - base],
                               x[*(ell_col_ind + off) - base]);
            res   = _mm512_fmadd_pd(vvals, vx, res);
        }

        if(alpha != static_cast<double>(1))
        {
            res = _mm512_mul_pd(va, res);
        }
        if(beta != static_cast<double>(0))
        {
            vy  = _mm512_loadu_pd(&y[j * blk]);
            res = _mm512_fmadd_pd(vb, vy, res);
        }
        _mm512_storeu_pd(&y[joff], res);
    }
    double rd;
    for(aoclsparse_int j = (m / blk) * blk; j < m; j++)
    {
        rd = 0.0;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            rd += *(ell_val + i * m + j) * (x[*(ell_col_ind + i * m + j) - base]);
        }

        if(alpha != static_cast<double>(1))
        {
            rd = alpha * rd;
        }

        if(beta != static_cast<double>(0))
        {
            rd += beta * y[j];
        }

        y[j] = rd;
    }

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_ellthybmv_avx512(const double                     alpha,
                                              aoclsparse_int                   m,
                                              [[maybe_unused]] aoclsparse_int  n,
                                              [[maybe_unused]] aoclsparse_int  nnz,
                                              const double                    *ell_val,
                                              const aoclsparse_int            *ell_col_ind,
                                              aoclsparse_int                   ell_width,
                                              aoclsparse_int                   ell_m,
                                              const double                    *csr_val,
                                              const aoclsparse_int            *csr_row_ind,
                                              const aoclsparse_int            *csr_col_ind,
                                              [[maybe_unused]] aoclsparse_int *row_idx_map,
                                              aoclsparse_int                  *csr_row_idx_map,
                                              const aoclsparse_mat_descr       descr,
                                              const double                    *x,
                                              const double                     beta,
                                              double                          *y)
{

    __m512d res, vvals, vx, vy, va, vb;
    va               = _mm512_set1_pd(alpha);
    vb               = _mm512_set1_pd(beta);
    res              = _mm512_setzero_pd();
    aoclsparse_int k = ell_width;
    if(ell_m == m)
    {
        return aoclsparse_elltmv_avx512(
            alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, beta, y);
    }

    // Create a temporary copy of the "y" elements corresponding to csr_row_idx_map.
    // This step is required when beta is non-zero
    double               *y_tmp = nullptr;
    aoclsparse_index_base base  = descr->base;
    if(beta != static_cast<double>(0))
    {
        try
        {
            y_tmp = new double[m - ell_m];
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        for(aoclsparse_int i = 0; i < m - ell_m; i++)
        {
            y_tmp[i] = y[csr_row_idx_map[i]];
        }
    }

    int blk = 8;
    for(aoclsparse_int j = 0; j < m / blk; j++)
    {
        res                 = _mm512_setzero_pd();
        aoclsparse_int joff = j * blk;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            aoclsparse_int off = joff + i * m;

            vvals = _mm512_loadu_pd(ell_val + off);
            vx    = _mm512_set_pd(x[*(ell_col_ind + off + 7) - base],
                               x[*(ell_col_ind + off + 6) - base],
                               x[*(ell_col_ind + off + 5) - base],
                               x[*(ell_col_ind + off + 4) - base],
                               x[*(ell_col_ind + off + 3) - base],
                               x[*(ell_col_ind + off + 2) - base],
                               x[*(ell_col_ind + off + 1) - base],
                               x[*(ell_col_ind + off) - base]);
            res   = _mm512_fmadd_pd(vvals, vx, res);
        }
        if(alpha != static_cast<double>(1))
        {
            res = _mm512_mul_pd(va, res);
        }

        if(beta != static_cast<double>(0))
        {
            vy  = _mm512_loadu_pd(&y[joff]);
            res = _mm512_fmadd_pd(vb, vy, res);
        }
        _mm512_storeu_pd(&y[joff], res);
    }

    double rd;
    for(aoclsparse_int j = (m / blk) * blk; j < m; j++)
    {
        rd = 0.0;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            rd += *(ell_val + i * m + j) * (x[*(ell_col_ind + i * m + j) - base]);
        }

        if(alpha != static_cast<double>(1))
        {
            rd = alpha * rd;
        }
        if(beta != static_cast<double>(0))
        {
            rd += beta * y[j];
        }

        y[j] = rd;
    }

    // reset some of the "y" elements corresponding to csr_row_idx_map.
    // This step is required when beta is non-zero
    if(beta != static_cast<double>(0))
    {
        for(aoclsparse_int i = 0; i < m - ell_m; i++)
        {
            y[csr_row_idx_map[i]] = y_tmp[i];
        }
        delete[] y_tmp;
    }

    // perform csr part if present
    __m512d               vec_vals, vec_x, vec_y_512;
    __m256d               vec_y;
    const aoclsparse_int *colIndPtr;
    const double         *matValPtr;

    for(aoclsparse_int i = 0; i < m - ell_m; ++i)
    {
        double         result    = 0.0;
        aoclsparse_int offset    = csr_row_idx_map[i];
        aoclsparse_int row_start = csr_row_ind[offset] - base;
        aoclsparse_int row_end   = csr_row_ind[offset + 1] - base;
        vec_y_512                = _mm512_setzero_pd();
        vec_y                    = _mm256_setzero_pd();
        matValPtr                = &csr_val[row_start];
        colIndPtr                = &csr_col_ind[row_start];
        aoclsparse_int nnz       = row_end - row_start;
        aoclsparse_int k_iter    = nnz / 8;
        aoclsparse_int k_rem     = nnz % 8;
        aoclsparse_int j;
        for(j = 0; j < k_iter; ++j)
        {
            vec_vals = _mm512_loadu_pd(matValPtr);

            // Gather the x vector elements from the column indices
            vec_x = _mm512_set_pd(x[*(colIndPtr + 7) - base],
                                  x[*(colIndPtr + 6) - base],
                                  x[*(colIndPtr + 5) - base],
                                  x[*(colIndPtr + 4) - base],
                                  x[*(colIndPtr + 3) - base],
                                  x[*(colIndPtr + 2) - base],
                                  x[*(colIndPtr + 1) - base],
                                  x[*(colIndPtr)-base]);

            vec_y_512 = _mm512_fmadd_pd(vec_vals, vec_x, vec_y_512);
            matValPtr += 8;
            colIndPtr += 8;
        }
        vec_y = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512, 0x0),
                              _mm512_extractf64x4_pd(vec_y_512, 0x1));

        // Horizontal addition
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
        for(j = 0; j < k_rem; j++)
        {
            result += *matValPtr++ * x[*(colIndPtr)-base];
            colIndPtr++;
        }

        // Perform alpha * A * x
        if(alpha != static_cast<double>(1))
        {
            result = alpha * result;
        }

        // Perform (beta * y) + (alpha * A * x)
        if(beta != static_cast<double>(0))
        {
            result += beta * y[csr_row_idx_map[i]];
        }

        y[csr_row_idx_map[i]] = result;
    }

    return aoclsparse_status_success;
}
#endif
