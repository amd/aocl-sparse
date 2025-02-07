/* ************************************************************************
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_ellmv_avx512.hpp"

#include <immintrin.h>
//======================================
// ELLMV
//======================================
template <typename T>
aoclsparse_status aoclsparse_ellmv_ref(const T                    alpha,
                                       aoclsparse_int             m,
                                       const T                   *ell_val,
                                       const aoclsparse_int      *ell_col_ind,
                                       aoclsparse_int             ell_width,
                                       const aoclsparse_mat_descr descr,
                                       const T                   *x,
                                       const T                    beta,
                                       T                         *y)

{
    using namespace aoclsparse;

    //TODO: Optimisation for float to be done
    aoclsparse_index_base base = descr->base;
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        T result = 0.0;

        for(aoclsparse_int p = 0; p < ell_width; ++p)
        {
            aoclsparse_int idx = i * ell_width + p;
            aoclsparse_int col = ell_col_ind[idx] - base;
            // Multiply only the valid non-zeroes, column index = -1 for padded
            // zeroes
            if(col >= 0)
            {
                result += (ell_val[idx] * x[col]);
            }
            else
            {
                break;
            }
        }

        if(alpha != static_cast<T>(1))
        {
            result = alpha * result;
        }

        if(beta != static_cast<T>(0))
        {
            result += beta * y[i];
        }

        y[i] = result;
    }

    return aoclsparse_status_success;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, double>, aoclsparse_status>
    aoclsparse_dellmv_avx2(const double                    alpha,
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
    using namespace aoclsparse;
    __m256d               vec_vals, vec_x, vec_y;
    aoclsparse_index_base base = descr->base;
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) private( \
        vec_vals, vec_x, vec_y)
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
            aoclsparse_int col = *(pell_col_ind + 3) - base;
            // Multiply only the valid non-zeroes, column index = -1 for padded
            // zeroes
            if(col >= 0)
            {
                //(ell_val[j] (ell_val[j+1] (ell_val[j+2] (ell_val[j+3]
                vec_vals = _mm256_loadu_pd(pell_val);

                vec_x = _mm256_set_pd(x[*(pell_col_ind + 3) - base],
                                      x[*(pell_col_ind + 2) - base],
                                      x[*(pell_col_ind + 1) - base],
                                      x[*(pell_col_ind)-base]);

                vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);

                pell_val += 4;
                pell_col_ind += 4;
            }
            else
            {
                k_rem = 4;
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

template <typename T>
aoclsparse_status aoclsparse_ellmv_t(aoclsparse_operation            trans,
                                     const T                        *alpha,
                                     aoclsparse_int                  m,
                                     aoclsparse_int                  n,
                                     [[maybe_unused]] aoclsparse_int nnz,
                                     const T                        *ell_val,
                                     const aoclsparse_int           *ell_col_ind,
                                     aoclsparse_int                  ell_width,
                                     const aoclsparse_mat_descr      descr,
                                     const T                        *x,
                                     const T                        *beta,
                                     T                              *y)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    if(descr->type != aoclsparse_matrix_type_general)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    if(trans != aoclsparse_operation_none)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(ell_width < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Sanity check
    if((m == 0 || n == 0) && ell_width != 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
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

    if constexpr(std::is_same_v<T, float>)
    {
        return aoclsparse_ellmv_ref<T>(
            *alpha, m, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        using namespace aoclsparse;

#ifdef USE_AVX512
        if(context::get_context()->supports<context_isa_t::AVX512F>())
            return aoclsparse_dellmv_avx512(
                *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
        else
#endif
            return aoclsparse_dellmv_avx2<T>(
                *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
    }
}

//======================================
// ELLTMV
//======================================

// ToDo: just an outline for now
template <typename T>
aoclsparse_status aoclsparse_elltmv_ref(const T                         alpha,
                                        aoclsparse_int                  m,
                                        [[maybe_unused]] aoclsparse_int n,
                                        [[maybe_unused]] aoclsparse_int nnz,
                                        const T                        *ell_val,
                                        const aoclsparse_int           *ell_col_ind,
                                        aoclsparse_int                  ell_width,
                                        const aoclsparse_mat_descr      descr,
                                        const T                        *x,
                                        const T                         beta,
                                        T                              *y)
{
    using namespace aoclsparse;

    aoclsparse_int        k = ell_width;
    T                     rd;
    aoclsparse_index_base base = descr->base;

#ifdef _OPENMP
    aoclsparse_int chunk = (m / context::get_context()->get_num_threads())
                               ? (m / context::get_context()->get_num_threads())
                               : 1;
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk) private(rd)
#endif
    for(aoclsparse_int j = 0; j < m; j++)
    {
        rd = 0.0;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            rd += *(ell_val + i * m + j) * (x[*(ell_col_ind + i * m + j) - base]);
        }

        if(alpha != static_cast<T>(1))
        {
            rd = alpha * rd;
        }

        if(beta != static_cast<T>(0))
        {
            rd += beta * y[j];
        }
        y[j] = rd;
    }

    return aoclsparse_status_success;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, double>, aoclsparse_status>
    aoclsparse_elltmv_avx2(const double                    alpha,
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
    using namespace aoclsparse;

    __m256d res, vvals, vx, vy, va, vb;

    va                                   = _mm256_set1_pd(alpha);
    vb                                   = _mm256_set1_pd(beta);
    res                                  = _mm256_setzero_pd();
    aoclsparse_int                  k    = ell_width;
    aoclsparse_int                  blk  = 4;
    aoclsparse_index_base           base = descr->base;
    [[maybe_unused]] aoclsparse_int chunk_size
        = m / (blk * context::get_context()->get_num_threads());
#ifdef _OPENMP
    chunk_size = chunk_size ? chunk_size : 1;
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk_size) private(res, vvals, vx, vy)
#endif
    for(aoclsparse_int j = 0; j < m / blk; j++)
    {
        res                 = _mm256_setzero_pd();
        aoclsparse_int joff = j * blk;
        for(aoclsparse_int i = 0; i < k; i++)
        {

            aoclsparse_int off = joff + i * m;

            vvals = _mm256_loadu_pd(ell_val + off);
            vx    = _mm256_set_pd(x[*(ell_col_ind + off + 3) - base],
                               x[*(ell_col_ind + off + 2) - base],
                               x[*(ell_col_ind + off + 1) - base],
                               x[*(ell_col_ind + off) - base]);
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
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) private(rd)
#endif
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

template <typename T>
aoclsparse_status aoclsparse_elltmv_t(aoclsparse_operation       trans,
                                      const T                   *alpha,
                                      aoclsparse_int             m,
                                      aoclsparse_int             n,
                                      aoclsparse_int             nnz,
                                      const T                   *ell_val,
                                      const aoclsparse_int      *ell_col_ind,
                                      aoclsparse_int             ell_width,
                                      const aoclsparse_mat_descr descr,
                                      const T                   *x,
                                      const T                   *beta,
                                      T                         *y)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    if(descr->type != aoclsparse_matrix_type_general)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    if(trans != aoclsparse_operation_none)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(ell_width < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Sanity check
    if((m == 0 || n == 0) && ell_width != 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
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

    if constexpr(std::is_same_v<T, float>)
    {
        return aoclsparse_elltmv_ref<T>(
            *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
    }
    else
    {
        using namespace aoclsparse;

#ifdef USE_AVX512
        if(context::get_context()->supports<context_isa_t::AVX512F>())
            return aoclsparse_elltmv_avx512(
                *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
        else
#endif
            return aoclsparse_elltmv_avx2<T>(
                *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
    }
}

//======================================
// ELLTHYBMV
//======================================

// Hybrid ELL-CSR implementation

template <typename T>
std::enable_if_t<std::is_same_v<T, double>, aoclsparse_status>
    aoclsparse_ellthybmv_avx2(const double                     alpha,
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
    using namespace aoclsparse;

    __m256d res, vvals, vx, vy, va, vb;
    va  = _mm256_set1_pd(alpha);
    vb  = _mm256_set1_pd(beta);
    res = vvals = vx = vy = _mm256_setzero_pd();
    aoclsparse_int k      = ell_width;
    if(ell_m == m)
    {
        return aoclsparse_elltmv_avx2<T>(
            alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, beta, y);
    }

    // Create a temporary copy of the "y" elements corresponding to csr_row_idx_map.
    // This step is required when beta is non-zero
    double *y_tmp = nullptr;
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

    int                             blk = 4;
    [[maybe_unused]] aoclsparse_int chunk_size
        = m / (blk * context::get_context()->get_num_threads());
    aoclsparse_index_base base = descr->base;
#ifdef _OPENMP
    chunk_size = chunk_size ? chunk_size : 1;
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk_size) firstprivate(res, vvals, vx, vy)
#endif
    for(aoclsparse_int j = 0; j < m / blk; j++)
    {
        res                 = _mm256_setzero_pd();
        aoclsparse_int joff = j * blk;
        for(aoclsparse_int i = 0; i < k; i++)
        {
            aoclsparse_int off = joff + i * m;

            vvals = _mm256_loadu_pd(ell_val + off);
            vx    = _mm256_set_pd(x[*(ell_col_ind + off + 3) - base],
                               x[*(ell_col_ind + off + 2) - base],
                               x[*(ell_col_ind + off + 1) - base],
                               x[*(ell_col_ind + off) - base]);
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

    using namespace aoclsparse;

    double rd;
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) private(rd)
#endif
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

    // perform csr part if present
    __m256d               vec_vals, vec_x, vec_y;
    const aoclsparse_int *colIndPtr;
    const double         *matValPtr;
    chunk_size = 512;
    // reset some of the "y" elements corresponding to csr_row_idx_map.
    // this step is required when beta is non-zero
    if(beta != static_cast<double>(0))
    {
        for(aoclsparse_int i = 0; i < m - ell_m; i++)
        {
            y[csr_row_idx_map[i]] = y_tmp[i];
        }
        delete[] y_tmp;
    }

    base = descr->base;
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk_size) private(vec_vals, vec_x, vec_y, colIndPtr, matValPtr)
#endif
    for(aoclsparse_int i = 0; i < m - ell_m; ++i)
    {
        double         result    = 0.0;
        aoclsparse_int offset    = csr_row_idx_map[i];
        aoclsparse_int row_start = csr_row_ind[offset] - base;
        aoclsparse_int row_end   = csr_row_ind[offset + 1] - base;
        vec_y                    = _mm256_setzero_pd();
        matValPtr                = &csr_val[row_start];
        colIndPtr                = &csr_col_ind[row_start];
        aoclsparse_int nnz       = row_end - row_start;
        aoclsparse_int k_iter    = nnz / 4;
        aoclsparse_int k_rem     = nnz % 4;
        aoclsparse_int j;
        for(j = 0; j < k_iter; ++j)
        {
            vec_vals = _mm256_loadu_pd(matValPtr);

            //Gather the x vector elements from the column indices
            vec_x = _mm256_set_pd(x[*(colIndPtr + 3) - base],
                                  x[*(colIndPtr + 2) - base],
                                  x[*(colIndPtr + 1) - base],
                                  x[*(colIndPtr)-base]);

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

template <typename T>
aoclsparse_status aoclsparse_ellthybmv_t([[maybe_unused]] aoclsparse_operation trans,
                                         const T                              *alpha,
                                         aoclsparse_int                        m,
                                         aoclsparse_int                        n,
                                         aoclsparse_int                        nnz,
                                         const T                              *ell_val,
                                         const aoclsparse_int                 *ell_col_ind,
                                         aoclsparse_int                        ell_width,
                                         aoclsparse_int                        ell_m,
                                         const T                              *csr_val,
                                         const aoclsparse_int                 *csr_row_ind,
                                         const aoclsparse_int                 *csr_col_ind,
                                         aoclsparse_int                       *row_idx_map,
                                         aoclsparse_int                       *csr_row_idx_map,
                                         const aoclsparse_mat_descr            descr,
                                         const T                              *x,
                                         const T                              *beta,
                                         T                                    *y)

{
    // Only double is supported
    if constexpr(!std::is_same_v<T, double>)
    {
        // ToDo: Need to implement this functionality
        return aoclsparse_status_not_implemented;
    }
    else
    {
        using namespace aoclsparse;
#ifdef USE_AVX512
        if(context::get_context()->supports<context_isa_t::AVX512F>())
            return aoclsparse_ellthybmv_avx512(*alpha,
                                               m,
                                               n,
                                               nnz,
                                               ell_val,
                                               ell_col_ind,
                                               ell_width,
                                               ell_m,
                                               csr_val,
                                               csr_row_ind,
                                               csr_col_ind,
                                               row_idx_map,
                                               csr_row_idx_map,
                                               descr,
                                               x,
                                               *beta,
                                               y);
        else
#endif
            return aoclsparse_ellthybmv_avx2<T>(*alpha,
                                                m,
                                                n,
                                                nnz,
                                                ell_val,
                                                ell_col_ind,
                                                ell_width,
                                                ell_m,
                                                csr_val,
                                                csr_row_ind,
                                                csr_col_ind,
                                                row_idx_map,
                                                csr_row_idx_map,
                                                descr,
                                                x,
                                                *beta,
                                                y);
    }
}

#endif // AOCLSPARSE_ELLMV_HPP
