/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"

#include <immintrin.h>

aoclsparse_status aoclsparse_ellmv_template(const float           alpha,
                                            aoclsparse_int        m,
                                            aoclsparse_int        n,
                                            aoclsparse_int        nnz,
                                            const float          *ell_val,
                                            const aoclsparse_int *ell_col_ind,
                                            aoclsparse_int        ell_width,
                                            const float          *x,
                                            const float           beta,
                                            float                *y,
                                            aoclsparse_context   *context)

{
    //TODO: Optimisation for float to be done

#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads)
#endif
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        float result = 0.0;

        for(aoclsparse_int p = 0; p < ell_width; ++p)
        {
            aoclsparse_int idx = i * ell_width + p;
            aoclsparse_int col = ell_col_ind[idx];

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

        y[i] = result;
    }

    return aoclsparse_status_success;
}

#if USE_AVX512
aoclsparse_status aoclsparse_ellmv_template_avx512(const double          alpha,
                                                   aoclsparse_int        m,
                                                   aoclsparse_int        n,
                                                   aoclsparse_int        nnz,
                                                   const double         *ell_val,
                                                   const aoclsparse_int *ell_col_ind,
                                                   aoclsparse_int        ell_width,
                                                   const double         *x,
                                                   const double          beta,
                                                   double               *y,
                                                   aoclsparse_context   *context)
{

    __m256d vec_y;
    __m512d vec_vals_512, vec_x_512, vec_y_512;

    aoclsparse_int k_iter = ell_width / 8;
    aoclsparse_int k_rem  = ell_width % 8;

    for(aoclsparse_int i = 0; i < m; ++i)
    {
        double         result = 0.0;
        aoclsparse_int idx    = i * ell_width;

        const aoclsparse_int *pell_col_ind;
        const double         *pell_val;

        pell_col_ind = &ell_col_ind[idx];
        pell_val     = &ell_val[idx];
        vec_y        = _mm256_setzero_pd();
        vec_y_512    = _mm512_setzero_pd();
        //Loop over in multiple of 4
        for(aoclsparse_int p = 0; p < k_iter; ++p)
        {
            aoclsparse_int col = *pell_col_ind;
            // Multiply only the valid non-zeroes, column index = -1 for padded
            // zeroes
            if(col >= 0)
            {
                //(ell_val[j] (ell_val[j+1] (ell_val[j+2] (ell_val[j+3]
                vec_vals_512 = _mm512_loadu_pd((double const *)pell_val);
                vec_x_512    = _mm512_set_pd(x[*(pell_col_ind + 7)],
                                          x[*(pell_col_ind + 6)],
                                          x[*(pell_col_ind + 5)],
                                          x[*(pell_col_ind + 4)],
                                          x[*(pell_col_ind + 3)],
                                          x[*(pell_col_ind + 2)],
                                          x[*(pell_col_ind + 1)],
                                          x[*(pell_col_ind)]);

                vec_y_512 = _mm512_fmadd_pd(vec_vals_512, vec_x_512, vec_y_512);

                pell_val += 8;
                pell_col_ind += 8;
            }
            else
            {
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
            aoclsparse_int col = *pell_col_ind;

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

        y[i] = result;
    }
    return aoclsparse_status_success;
}
#endif

aoclsparse_status aoclsparse_ellmv_template_avx2(const double          alpha,
                                                 aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 aoclsparse_int        nnz,
                                                 const double         *ell_val,
                                                 const aoclsparse_int *ell_col_ind,
                                                 aoclsparse_int        ell_width,
                                                 const double         *x,
                                                 const double          beta,
                                                 double               *y,
                                                 aoclsparse_context   *context)
{

    __m256d vec_vals, vec_x, vec_y;

#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads) private(vec_vals, vec_x, vec_y)
#endif
    for(aoclsparse_int i = 0; i < m; ++i)
    {

        aoclsparse_int k_iter = ell_width / 4;
        aoclsparse_int k_rem  = ell_width % 4;
        double         result = 0.0;
        aoclsparse_int idx    = i * ell_width;

        const aoclsparse_int *pell_col_ind;
        const double         *pell_val;

        pell_col_ind = &ell_col_ind[idx];
        pell_val     = &ell_val[idx];
        vec_y        = _mm256_setzero_pd();

        // Loop over in multiple of 4
        for(aoclsparse_int p = 0; p < k_iter; ++p)
        {
            aoclsparse_int col = *pell_col_ind;
            // Multiply only the valid non-zeroes, column index = -1 for padded
            // zeroes
            if(col >= 0)
            {
                //(ell_val[j] (ell_val[j+1] (ell_val[j+2] (ell_val[j+3]
                vec_vals = _mm256_loadu_pd((double const *)pell_val);

                vec_x = _mm256_set_pd(x[*(pell_col_ind + 3)],
                                      x[*(pell_col_ind + 2)],
                                      x[*(pell_col_ind + 1)],
                                      x[*(pell_col_ind)]);

                vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);

                pell_val += 4;
                pell_col_ind += 4;
            }
            else
            {
                break;
            }
        }

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

        // Remainder loop
        for(aoclsparse_int p = 0; p < k_rem; ++p)
        {
            aoclsparse_int col = *pell_col_ind;

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

        y[i] = result;
    }

    return aoclsparse_status_success;
}

// ToDo: just an outline for now
aoclsparse_status aoclsparse_elltmv_template(const float           alpha,
                                             aoclsparse_int        m,
                                             aoclsparse_int        n,
                                             aoclsparse_int        nnz,
                                             const float          *ell_val,
                                             const aoclsparse_int *ell_col_ind,
                                             aoclsparse_int        ell_width,
                                             const float          *x,
                                             const float           beta,
                                             float                *y,
                                             aoclsparse_context   *context)
{
    aoclsparse_int k = ell_width;
    double         rd;

#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads) \
    schedule(dynamic, m / context->num_threads) private(rd)
#endif
    for(aoclsparse_int j = 0; j < m; j++)
    {
        rd = 0.0;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            rd += *(ell_val + i * m + j) * (x[*(ell_col_ind + i * m + j)]);
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

#if USE_AVX512
aoclsparse_status aoclsparse_elltmv_template_avx512(const double          alpha,
                                                    aoclsparse_int        m,
                                                    aoclsparse_int        n,
                                                    aoclsparse_int        nnz,
                                                    const double         *ell_val,
                                                    const aoclsparse_int *ell_col_ind,
                                                    aoclsparse_int        ell_width,
                                                    const double         *x,
                                                    const double          beta,
                                                    double               *y,
                                                    aoclsparse_context   *context)
{

    __m512d res, vvals, vx, vy, va, vb, vvals1, vx1, vy1;

    va                 = _mm512_set1_pd(alpha);
    vb                 = _mm512_set1_pd(beta);
    res                = _mm512_setzero_pd();
    aoclsparse_int k   = ell_width;
    int            blk = 8;
    for(aoclsparse_int j = 0; j < m / blk; j++)
    {
        res                 = _mm512_setzero_pd();
        aoclsparse_int joff = j * blk;
        for(aoclsparse_int i = 0; i < k; i++)
        {

            aoclsparse_int off = joff + i * m;

            vvals = _mm512_loadu_pd((double const *)(ell_val + off));
            vx    = _mm512_set_pd(x[*(ell_col_ind + off + 7)],
                               x[*(ell_col_ind + off + 6)],
                               x[*(ell_col_ind + off + 5)],
                               x[*(ell_col_ind + off + 4)],
                               x[*(ell_col_ind + off + 3)],
                               x[*(ell_col_ind + off + 2)],
                               x[*(ell_col_ind + off + 1)],
                               x[*(ell_col_ind + off)]);
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
            rd += *(ell_val + i * m + j) * (x[*(ell_col_ind + i * m + j)]);
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
#endif

aoclsparse_status aoclsparse_elltmv_template_avx2(const double          alpha,
                                                  aoclsparse_int        m,
                                                  aoclsparse_int        n,
                                                  aoclsparse_int        nnz,
                                                  const double         *ell_val,
                                                  const aoclsparse_int *ell_col_ind,
                                                  aoclsparse_int        ell_width,
                                                  const double         *x,
                                                  const double          beta,
                                                  double               *y,
                                                  aoclsparse_context   *context)
{
    __m256d res, vvals, vx, vy, va, vb, vvals1, vx1, vy1;

    va                        = _mm256_set1_pd(alpha);
    vb                        = _mm256_set1_pd(beta);
    res                       = _mm256_setzero_pd();
    aoclsparse_int k          = ell_width;
    aoclsparse_int blk        = 4;
    aoclsparse_int chunk_size = m / (blk * context->num_threads);
#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads) \
    schedule(dynamic, chunk_size) private(res, vvals, vx, vy, va, vb, vvals1, vx1, vy1)
#endif
    for(aoclsparse_int j = 0; j < m / blk; j++)
    {
        res                 = _mm256_setzero_pd();
        aoclsparse_int joff = j * blk;
        for(aoclsparse_int i = 0; i < k; i++)
        {

            aoclsparse_int off = joff + i * m;

            vvals = _mm256_loadu_pd((double const *)(ell_val + off));
            vx    = _mm256_set_pd(x[*(ell_col_ind + off + 3)],
                               x[*(ell_col_ind + off + 2)],
                               x[*(ell_col_ind + off + 1)],
                               x[*(ell_col_ind + off)]);
            res   = _mm256_fmadd_pd(vvals, vx, res);
        }

        if(alpha != static_cast<double>(1))
        {
            res = _mm256_mul_pd(va, res);
        }

        if(beta != static_cast<double>(0))
        {
            vy  = _mm256_loadu_pd(&y[j * blk]);
            res = _mm256_fmadd_pd(vb, vy, res);
        }
        _mm256_storeu_pd(&y[joff], res);
    }
    double rd;
#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads) private(rd)
#endif
    for(aoclsparse_int j = (m / blk) * blk; j < m; j++)
    {
        rd = 0.0;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            rd += *(ell_val + i * m + j) * (x[*(ell_col_ind + i * m + j)]);
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

// Hybrid ELL-CSR implementation
#if USE_AVX512
aoclsparse_status aoclsparse_ellthybmv_template_avx512(const double          alpha,
                                                       aoclsparse_int        m,
                                                       aoclsparse_int        n,
                                                       aoclsparse_int        nnz,
                                                       const double         *ell_val,
                                                       const aoclsparse_int *ell_col_ind,
                                                       aoclsparse_int        ell_width,
                                                       aoclsparse_int        ell_m,
                                                       const double         *csr_val,
                                                       const aoclsparse_int *csr_row_ind,
                                                       const aoclsparse_int *csr_col_ind,
                                                       aoclsparse_int       *row_idx_map,
                                                       aoclsparse_int       *csr_row_idx_map,
                                                       const double         *x,
                                                       const double          beta,
                                                       double               *y,
                                                       aoclsparse_context   *context)
{

    __m512d res, vvals, vx, vy, va, vb;
    va               = _mm512_set1_pd(alpha);
    vb               = _mm512_set1_pd(beta);
    res              = _mm512_setzero_pd();
    aoclsparse_int k = ell_width;
    if(ell_m == m)
    {
        return aoclsparse_elltmv_template_avx512(
            alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, x, beta, y, context);
    }

    int blk = 8;
    for(aoclsparse_int j = 0; j < m / blk; j++)
    {
        res                 = _mm512_setzero_pd();
        aoclsparse_int joff = j * blk;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            aoclsparse_int off = joff + i * m;

            vvals = _mm512_loadu_pd((double const *)(ell_val + off));
            vx    = _mm512_set_pd(x[*(ell_col_ind + off + 7)],
                               x[*(ell_col_ind + off + 6)],
                               x[*(ell_col_ind + off + 5)],
                               x[*(ell_col_ind + off + 4)],
                               x[*(ell_col_ind + off + 3)],
                               x[*(ell_col_ind + off + 2)],
                               x[*(ell_col_ind + off + 1)],
                               x[*(ell_col_ind + off)]);
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
            rd += *(ell_val + i * m + j) * (x[*(ell_col_ind + i * m + j)]);
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

    // perform csr part if present
    __m512d               vec_vals, vec_x, vec_y_512;
    __m256d               vec_y;
    const aoclsparse_int *colIndPtr;
    const double         *matValPtr;
    for(aoclsparse_int i = 0; i < m - ell_m; ++i)
    {
        double result      = 0.0;
        vec_y_512          = _mm512_setzero_pd();
        vec_y              = _mm256_setzero_pd();
        matValPtr          = &csr_val[csr_row_ind[csr_row_idx_map[i]]];
        colIndPtr          = &csr_col_ind[csr_row_ind[csr_row_idx_map[i]]];
        aoclsparse_int nnz = csr_row_ind[csr_row_idx_map[i] + 1] - csr_row_ind[csr_row_idx_map[i]];
        aoclsparse_int k_iter = nnz / 8;
        aoclsparse_int k_rem  = nnz % 8;
        aoclsparse_int j;
        for(j = 0; j < k_iter; ++j)
        {
            vec_vals = _mm512_loadu_pd((double const *)matValPtr);

            // Gather the x vector elements from the column indices
            vec_x = _mm512_set_pd(x[*(colIndPtr + 7)],
                                  x[*(colIndPtr + 6)],
                                  x[*(colIndPtr + 5)],
                                  x[*(colIndPtr + 4)],
                                  x[*(colIndPtr + 3)],
                                  x[*(colIndPtr + 2)],
                                  x[*(colIndPtr + 1)],
                                  x[*(colIndPtr)]);

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
            result += beta * y[csr_row_idx_map[i]];
        }

        y[csr_row_idx_map[i]] = result;
    }

    return aoclsparse_status_success;
}
#endif

aoclsparse_status aoclsparse_ellthybmv_template_avx2(const double          alpha,
                                                     aoclsparse_int        m,
                                                     aoclsparse_int        n,
                                                     aoclsparse_int        nnz,
                                                     const double         *ell_val,
                                                     const aoclsparse_int *ell_col_ind,
                                                     aoclsparse_int        ell_width,
                                                     aoclsparse_int        ell_m,
                                                     const double         *csr_val,
                                                     const aoclsparse_int *csr_row_ind,
                                                     const aoclsparse_int *csr_col_ind,
                                                     aoclsparse_int       *row_idx_map,
                                                     aoclsparse_int       *csr_row_idx_map,
                                                     const double         *x,
                                                     const double          beta,
                                                     double               *y,
                                                     aoclsparse_context   *context)
{
    __m256d res, vvals, vx, vy, va, vb;
    va               = _mm256_set1_pd(alpha);
    vb               = _mm256_set1_pd(beta);
    res              = _mm256_setzero_pd();
    aoclsparse_int k = ell_width;
    if(ell_m == m)
    {
        return aoclsparse_elltmv_template_avx2(
            alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, x, beta, y, context);
    }

    int            blk        = 4;
    aoclsparse_int chunk_size = m / (blk * context->num_threads);
#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads) \
    schedule(dynamic, chunk_size) private(res, vvals, vx, vy, va, vb)
#endif
    for(aoclsparse_int j = 0; j < m / blk; j++)
    {
        res                 = _mm256_setzero_pd();
        aoclsparse_int joff = j * blk;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            aoclsparse_int off = joff + i * m;

            vvals = _mm256_loadu_pd((double const *)(ell_val + off));
            vx    = _mm256_set_pd(x[*(ell_col_ind + off + 3)],
                               x[*(ell_col_ind + off + 2)],
                               x[*(ell_col_ind + off + 1)],
                               x[*(ell_col_ind + off)]);
            res   = _mm256_fmadd_pd(vvals, vx, res);
        }
        if(alpha != static_cast<double>(1))
        {
            res = _mm256_mul_pd(va, res);
        }

        if(beta != static_cast<double>(0))
        {
            vy  = _mm256_loadu_pd(&y[joff]);
            res = _mm256_fmadd_pd(vb, vy, res);
        }
        _mm256_storeu_pd(&y[joff], res);
    }

    double rd;
#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads) private(rd)
#endif
    for(aoclsparse_int j = (m / blk) * blk; j < m; j++)
    {
        rd = 0.0;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            rd += *(ell_val + i * m + j) * (x[*(ell_col_ind + i * m + j)]);
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

    // perform csr part if present
    __m256d               vec_vals, vec_x, vec_y;
    const aoclsparse_int *colIndPtr;
    const double         *matValPtr;
    chunk_size = 512;
#ifdef _OPENMP
#pragma omp parallel for num_threads(context->num_threads) \
    schedule(dynamic, chunk_size) private(vec_vals, vec_x, vec_y, colIndPtr, matValPtr)
#endif
    for(aoclsparse_int i = 0; i < m - ell_m; ++i)
    {
        double result      = 0.0;
        vec_y              = _mm256_setzero_pd();
        matValPtr          = &csr_val[csr_row_ind[csr_row_idx_map[i]]];
        colIndPtr          = &csr_col_ind[csr_row_ind[csr_row_idx_map[i]]];
        aoclsparse_int nnz = csr_row_ind[csr_row_idx_map[i] + 1] - csr_row_ind[csr_row_idx_map[i]];
        aoclsparse_int k_iter = nnz / 4;
        aoclsparse_int k_rem  = nnz % 4;
        aoclsparse_int j;
        for(j = 0; j < k_iter; ++j)
        {
            vec_vals = _mm256_loadu_pd((double const *)matValPtr);

            //Gather the x vector elements from the column indices
            vec_x = _mm256_set_pd(
                x[*(colIndPtr + 3)], x[*(colIndPtr + 2)], x[*(colIndPtr + 1)], x[*(colIndPtr)]);

            vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);

            matValPtr += 4;
            colIndPtr += 4;
        }

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

        //Remainder loop for nnz%4
        for(j = 0; j < k_rem; j++)
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
            result += beta * y[csr_row_idx_map[i]];
        }

        y[csr_row_idx_map[i]] = result;
    }

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_ellthybmv_template(const float           alpha,
                                                aoclsparse_int        m,
                                                aoclsparse_int        n,
                                                aoclsparse_int        nnz,
                                                const float          *ell_val,
                                                const aoclsparse_int *ell_col_ind,
                                                aoclsparse_int        ell_width,
                                                aoclsparse_int        ell_m,
                                                const float          *csr_val,
                                                const aoclsparse_int *csr_row_ind,
                                                const aoclsparse_int *csr_col_ind,
                                                aoclsparse_int       *row_idx_map,
                                                aoclsparse_int       *csr_row_idx_map,
                                                const float          *x,
                                                const float           beta,
                                                float                *y,
                                                aoclsparse_context   *context)

{

    // ToDo: Need to implement this functionality
    //
    return aoclsparse_status_not_implemented;
}

#endif // AOCLSPARSE_ELLMV_HPP
