/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_SV_HPP
#define AOCLSPARSE_SV_HPP

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_csr_util.hpp"

#include <complex>
#include <immintrin.h>
#include <type_traits>

#define KT_ADDRESS_TYPE aoclsparse_int
#include "aoclsparse_kernel_templates.hpp"
#undef KT_ADDRESS_TYPE
#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

/* TRiangular SolVer dispatcher
 * ============================
 * TRSV dispatcher and various templated and vectorized triangular solve kernels
 * Solves A*x = alpha*b or A'*x = alpha*b with A lower (L) or upper (U) triangular.
 * Optimized version, requires A to have been previously "optimized". If A is not
 * optimized previously by user, it is optimized on the fly.
 */
template <typename T>
aoclsparse_status
    aoclsparse_trsv(const aoclsparse_operation transpose, /* matrix operation */
                    const T                    alpha, /* scalar for rescaling RHS */
                    aoclsparse_matrix          A, /* matrix data */
                    const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
                    const T                   *b, /* RHS */
                    T                         *x, /* solution */
                    const aoclsparse_int       kid /* Kernel ID request */);

/* Core computation of TRSV, assumed A is optimized
 * solves L*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_l_ref_core(const T               alpha,
                                                aoclsparse_int        m,
                                                aoclsparse_index_base base,
                                                const T *__restrict__ a,
                                                const aoclsparse_int *__restrict__ icol,
                                                const aoclsparse_int *__restrict__ ilrow,
                                                const aoclsparse_int *__restrict__ idiag,
                                                const T *__restrict__ b,
                                                T *__restrict__ x,
                                                const bool unit)
{
    aoclsparse_int        i, idx;
    T                     xi;
    const aoclsparse_int *icol_fix = icol - base;
    const T              *a_fix    = a - base;
    T                    *x_fix    = x - base;
    for(i = 0; i < m; i++)
    {
        xi = alpha * b[i];
        for(idx = ilrow[i]; idx < idiag[i]; idx++)
            xi -= a_fix[idx] * x_fix[icol_fix[idx]];
        if(!unit)
            xi /= a_fix[idiag[i]];
        x[i] = xi;
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves L'*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_lt_ref_core(const T               alpha,
                                                 aoclsparse_int        m,
                                                 aoclsparse_index_base base,
                                                 const T *__restrict__ a,
                                                 const aoclsparse_int *__restrict__ icol,
                                                 const aoclsparse_int *__restrict__ ilrow,
                                                 const aoclsparse_int *__restrict__ idiag,
                                                 const T *__restrict__ b,
                                                 T *__restrict__ x,
                                                 const bool unit)
{
    aoclsparse_int        i, idx;
    const aoclsparse_int *icol_fix = icol - base;
    const T              *a_fix    = a - base;
    T                    *x_fix    = x - base;

    for(i = 0; i < m; i++)
        x[i] = alpha * b[i];
    for(i = m - 1; i >= 0; i--)
    {
        if(!unit)
            x[i] /= a_fix[idiag[i]];
        // propagate value of x[i] through the column (used to be the row but now is transposed)
        for(idx = ilrow[i]; idx < idiag[i]; idx++)
            x_fix[icol_fix[idx]] -= a_fix[idx] * x[i];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves L^H*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_lh_ref_core(const T               alpha,
                                                 aoclsparse_int        m,
                                                 aoclsparse_index_base base,
                                                 const T *__restrict__ a,
                                                 const aoclsparse_int *__restrict__ icol,
                                                 const aoclsparse_int *__restrict__ ilrow,
                                                 const aoclsparse_int *__restrict__ idiag,
                                                 const T *__restrict__ b,
                                                 T *__restrict__ x,
                                                 const bool unit)
{
    aoclsparse_int        i, idx;
    const aoclsparse_int *icol_fix = icol - base;
    const T              *a_fix    = a - base;
    T                    *x_fix    = x - base;

    for(i = 0; i < m; i++)
        x[i] = alpha * b[i];
    for(i = m - 1; i >= 0; i--)
    {
        if(!unit)
            x[i] /= std::conj(a_fix[idiag[i]]);
        // propagate value of x[i] through the column (used to be the row but now is transposed)
        for(idx = ilrow[i]; idx < idiag[i]; idx++)
            x_fix[icol_fix[idx]] -= std::conj(a_fix[idx]) * x[i];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_u_ref_core(const T               alpha,
                                                aoclsparse_int        m,
                                                aoclsparse_index_base base,
                                                const T *__restrict__ a,
                                                const aoclsparse_int *__restrict__ icol,
                                                const aoclsparse_int *__restrict__ ilrow,
                                                const aoclsparse_int *__restrict__ iurow,
                                                const T *__restrict__ b,
                                                T *__restrict__ x,
                                                const bool unit)
{
    aoclsparse_int        i, idx, idxstart, idxend, idiag;
    T                     xi;
    const aoclsparse_int *icol_fix = icol - base;
    const T              *a_fix    = a - base;
    T                    *x_fix    = x - base;

    for(i = m - 1; i >= 0; i--)
    {
        idxstart = iurow[i];
        // ilrow[i+1]-1 always points to last element of U at row i
        idxend = ilrow[i + 1] - 1;
        xi     = alpha * b[i];
        for(idx = idxstart; idx <= idxend; idx++)
            xi -= a_fix[idx] * x_fix[icol_fix[idx]];
        x[i] = xi;
        if(!unit)
        {
            // urow[i]-1 always points to idiag[i]
            idiag = iurow[i] - 1;
            x[i] /= a_fix[idiag];
        }
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U'*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_ut_ref_core(const T               alpha,
                                                 aoclsparse_int        m,
                                                 aoclsparse_index_base base,
                                                 const T *__restrict__ a,
                                                 const aoclsparse_int *__restrict__ icol,
                                                 const aoclsparse_int *__restrict__ ilrow,
                                                 const aoclsparse_int *__restrict__ iurow,
                                                 const T *__restrict__ b,
                                                 T *__restrict__ x,
                                                 const bool unit)
{
    aoclsparse_int        i, idx, idxstart, idxend, idiag;
    const aoclsparse_int *icol_fix = icol - base;
    const T              *a_fix    = a - base;
    T                    *x_fix    = x - base;
    for(i = 0; i < m; i++)
        x[i] = alpha * b[i];
    for(i = 0; i < m; i++)
    {
        if(!unit)
        {
            // urow[i]-1 always points to idiag[i]
            idiag = iurow[i] - 1;
            x[i] /= a_fix[idiag];
        }
        idxstart = iurow[i];
        // ilrow[i+1]-1 always points to last element of U at row i
        idxend = ilrow[i + 1] - 1;
        for(idx = idxstart; idx <= idxend; idx++)
            x_fix[icol_fix[idx]] -= a_fix[idx] * x[i];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U^H*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_uh_ref_core(const T               alpha,
                                                 aoclsparse_int        m,
                                                 aoclsparse_index_base base,
                                                 const T *__restrict__ a,
                                                 const aoclsparse_int *__restrict__ icol,
                                                 const aoclsparse_int *__restrict__ ilrow,
                                                 const aoclsparse_int *__restrict__ iurow,
                                                 const T *__restrict__ b,
                                                 T *__restrict__ x,
                                                 const bool unit)
{
    aoclsparse_int        i, idx, idxstart, idxend, idiag;
    const aoclsparse_int *icol_fix = icol - base;
    const T              *a_fix    = a - base;
    T                    *x_fix    = x - base;
    for(i = 0; i < m; i++)
        x[i] = alpha * b[i];
    for(i = 0; i < m; i++)
    {
        if(!unit)
        {
            // urow[i]-1 always points to idiag[i]
            idiag = iurow[i] - 1;
            x[i] /= std::conj(a_fix[idiag]);
        }
        idxstart = iurow[i];
        // ilrow[i+1]-1 always points to last element of U at row i
        idxend = ilrow[i + 1] - 1;
        for(idx = idxstart; idx <= idxend; idx++)
            x_fix[icol_fix[idx]] -= std::conj(a_fix[idx]) * x[i];
    }
    return aoclsparse_status_success;
}

/*
 * Macro kernel templates for L, L', U and U' kernels (of type double only)
 *
 * Core computation of TRSV, assumed A is optimized
 * solves L*x = alpha*b
 */
template <typename T>
inline aoclsparse_status
    trsv_l_ref_core_avx([[maybe_unused]] const T               alpha,
                        [[maybe_unused]] aoclsparse_int        m,
                        [[maybe_unused]] aoclsparse_index_base base,
                        [[maybe_unused]] const T *__restrict__ a,
                        [[maybe_unused]] const aoclsparse_int *__restrict__ icol,
                        [[maybe_unused]] const aoclsparse_int *__restrict__ ilrow,
                        [[maybe_unused]] const aoclsparse_int *__restrict__ idiag,
                        [[maybe_unused]] const T *__restrict__ b,
                        [[maybe_unused]] T *__restrict__ x,
                        [[maybe_unused]] const bool unit)
{
    return aoclsparse_status_not_implemented;
}
template <> // DOUBLE
inline aoclsparse_status trsv_l_ref_core_avx<double>(const double          alpha,
                                                     aoclsparse_int        m,
                                                     aoclsparse_index_base base,
                                                     const double *__restrict__ a,
                                                     const aoclsparse_int *__restrict__ icol,
                                                     const aoclsparse_int *__restrict__ ilrow,
                                                     const aoclsparse_int *__restrict__ idiag,
                                                     const double *__restrict__ b,
                                                     double *__restrict__ x,
                                                     const bool unit)
{
    aoclsparse_int        i, idx, idxend;
    double                xi;
    __m256d               avec, xvec, xivec;
    __m128d               sum_lo, sum_hi, sse_sum;
    aoclsparse_int        idxcnt, tsz, idxrem;
    const aoclsparse_int *icol_fix = icol - base;
    const double         *a_fix    = a - base;
    double               *x_fix    = x - base;

    for(i = 0; i < m; i++)
    {
        idxend = idiag[i];
        xi     = alpha * b[i];
        // ---------------------------------------------
        // for (idx = icrow[i]; idx<idxend; idx++)
        //     yi = yi - a[idx] * y[icol[idx]];
        // ---------------------------------------------
        idxcnt = idxend - ilrow[i];
        tsz    = 4; // AVX2
        idxrem = idxcnt % tsz;
        for(idx = ilrow[i]; idx < idxend - idxrem; idx += tsz)
        {
            avec  = _mm256_loadu_pd(&a_fix[idx]);
            xvec  = _mm256_set_pd(x_fix[icol_fix[idx + 3]],
                                 x_fix[icol_fix[idx + 2]],
                                 x_fix[icol_fix[idx + 1]],
                                 x_fix[icol_fix[idx + 0]]);
            xivec = _mm256_mul_pd(avec, xvec);
            // horizontal sum:
            // sum[0] += sum[1] ; sum[2] += sum[3]
            xivec = _mm256_hadd_pd(xivec, xivec);
            // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
            sum_lo = _mm256_castpd256_pd128(xivec);
            // Extract 128 bits to obtain sum[2] and sum[3]
            sum_hi = _mm256_extractf128_pd(xivec, 1);
            // Add remaining two sums
            sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
            xi -= sse_sum.m128d_f64[0];
#else
            xi -= sse_sum[0];
#endif
        }
        // Remainder
        switch(idxrem)
        {
        case(3):
        {
            avec = _mm256_set_pd(0.0, a_fix[idx + 2], a_fix[idx + 1], a_fix[idx + 0]);
            xvec = _mm256_set_pd(
                0.0, x_fix[icol_fix[idx + 2]], x_fix[icol_fix[idx + 1]], x_fix[icol_fix[idx + 0]]);
            xivec = _mm256_mul_pd(avec, xvec);
            // horizontal sum:
            // sum[0] += sum[1] ; sum[2] += sum[3]
            xivec = _mm256_hadd_pd(xivec, xivec);
            // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
            sum_lo = _mm256_castpd256_pd128(xivec);
            // Extract 128 bits to obtain sum[2] and sum[3]
            sum_hi = _mm256_extractf128_pd(xivec, 1);
            // Add remaining two sums
            sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
            xi -= sse_sum.m128d_f64[0];
#else
            xi -= sse_sum[0];
#endif
            break;
        }
        default:
            for(idx = idxend - idxrem; idx < idxend; idx++)
                xi -= a_fix[idx] * x_fix[icol_fix[idx]];
        }
        x[i] = xi;
        if(!unit)
            x[i] /= a_fix[idxend];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves L'*x = alpha*b
 */
template <typename T>
inline aoclsparse_status
    trsv_lt_ref_core_avx([[maybe_unused]] const T               alpha,
                         [[maybe_unused]] aoclsparse_int        m,
                         [[maybe_unused]] aoclsparse_index_base base,
                         [[maybe_unused]] const T *__restrict__ a,
                         [[maybe_unused]] const aoclsparse_int *__restrict__ icol,
                         [[maybe_unused]] const aoclsparse_int *__restrict__ ilrow,
                         [[maybe_unused]] const aoclsparse_int *__restrict__ idiag,
                         [[maybe_unused]] const T *__restrict__ b,
                         [[maybe_unused]] T *__restrict__ x,
                         [[maybe_unused]] const bool unit)
{
    return aoclsparse_status_not_implemented;
}
template <> // DOUBLE
inline aoclsparse_status trsv_lt_ref_core_avx<double>(const double          alpha,
                                                      aoclsparse_int        m,
                                                      aoclsparse_index_base base,
                                                      const double *__restrict__ a,
                                                      const aoclsparse_int *__restrict__ icol,
                                                      const aoclsparse_int *__restrict__ ilrow,
                                                      const aoclsparse_int *__restrict__ idiag,
                                                      const double *__restrict__ b,
                                                      double *__restrict__ x,
                                                      const bool unit)
{
    aoclsparse_int        i, idx, idxend;
    aoclsparse_int        idxcnt, tsz, idxrem;
    double                xi, mxi;
    __m256d               avec, xvec, xivec;
    const aoclsparse_int *icol_fix = icol - base;
    const double         *a_fix    = a - base;
    double               *x_fix    = x - base;

    if(alpha != 0.0)
        for(i = 0; i < m; i++)
            x[i] = alpha * b[i];

    for(i = m - 1; i >= 0; i--)
    {
        idxend = idiag[i];

        if(!unit)
            x[i] /= a_fix[idiag[i]];
        xi  = x[i];
        mxi = -xi;

        idxcnt = idxend - ilrow[i];
        tsz    = 4; // AVX2
        idxrem = idxcnt % tsz;
        for(idx = ilrow[i]; idx < idxend - idxrem; idx += tsz)
        {
            xvec                     = _mm256_set_pd(x_fix[icol_fix[idx + 3]],
                                 x_fix[icol_fix[idx + 2]],
                                 x_fix[icol_fix[idx + 1]],
                                 x_fix[icol_fix[idx + 0]]);
            avec                     = _mm256_loadu_pd(&a_fix[idx]);
            xivec                    = _mm256_set1_pd(mxi);
            xvec                     = _mm256_fmadd_pd(avec, xivec, xvec);
            x_fix[icol_fix[idx + 3]] = xvec[3];
            x_fix[icol_fix[idx + 2]] = xvec[2];
            x_fix[icol_fix[idx + 1]] = xvec[1];
            x_fix[icol_fix[idx + 0]] = xvec[0];
        }
        // Remainder loop
        switch(idxrem)
        {
        case(3):
        {
            xvec = _mm256_set_pd(
                0.0, x_fix[icol_fix[idx + 2]], x_fix[icol_fix[idx + 1]], x_fix[icol_fix[idx + 0]]);
            avec  = _mm256_set_pd(0.0, a_fix[idx + 2], a_fix[idx + 1], a_fix[idx + 0]);
            xivec = _mm256_set1_pd(mxi);
            xvec  = _mm256_fmadd_pd(avec, xivec, xvec);
            x_fix[icol_fix[idx + 2]] = xvec[2];
            x_fix[icol_fix[idx + 1]] = xvec[1];
            x_fix[icol_fix[idx + 0]] = xvec[0];
            break;
        }
        default:
            for(idx = idxend - idxrem; idx < idxend; idx++)
                x_fix[icol_fix[idx]] -= a_fix[idx] * xi;
        }
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U*x = alpha*b
 */
template <typename T>
inline aoclsparse_status
    trsv_u_ref_core_avx([[maybe_unused]] const T               alpha,
                        [[maybe_unused]] aoclsparse_int        m,
                        [[maybe_unused]] aoclsparse_index_base base,
                        [[maybe_unused]] const T *__restrict__ a,
                        [[maybe_unused]] const aoclsparse_int *__restrict__ icol,
                        [[maybe_unused]] const aoclsparse_int *__restrict__ ilrow,
                        [[maybe_unused]] const aoclsparse_int *__restrict__ iurow,
                        [[maybe_unused]] const T *__restrict__ b,
                        [[maybe_unused]] T *__restrict__ x,
                        [[maybe_unused]] const bool unit)
{
    return aoclsparse_status_not_implemented;
}
template <> // DOUBLE
inline aoclsparse_status trsv_u_ref_core_avx<double>(const double          alpha,
                                                     aoclsparse_int        m,
                                                     aoclsparse_index_base base,
                                                     const double *__restrict__ a,
                                                     const aoclsparse_int *__restrict__ icol,
                                                     const aoclsparse_int *__restrict__ ilrow,
                                                     const aoclsparse_int *__restrict__ iurow,
                                                     const double *__restrict__ b,
                                                     double *__restrict__ x,
                                                     const bool unit)
{
    aoclsparse_int        i, idx, idxstart, idxend;
    double                xi;
    __m256d               avec, xvec, xivec;
    __m128d               sum_lo, sum_hi, sse_sum;
    aoclsparse_int        idxcnt, tsz, idxrem;
    aoclsparse_int        idiag;
    const aoclsparse_int *icol_fix = icol - base;
    const double         *a_fix    = a - base;
    double               *x_fix    = x - base;

    for(i = m - 1; i >= 0; i--)
    {
        idxstart = iurow[i];
        idxend   = ilrow[i + 1] - 1;
        xi       = alpha * b[i];
        idxcnt   = idxend - idxstart + 1;
        tsz      = 4; // AVX2
        idxrem   = idxcnt % tsz;
        for(idx = idxstart; idx <= idxend - idxrem; idx += tsz)
        {
            avec    = _mm256_loadu_pd(&a_fix[idx]);
            xvec    = _mm256_set_pd(x_fix[icol_fix[idx + 3]],
                                 x_fix[icol_fix[idx + 2]],
                                 x_fix[icol_fix[idx + 1]],
                                 x_fix[icol_fix[idx + 0]]);
            xivec   = _mm256_mul_pd(avec, xvec);
            xivec   = _mm256_hadd_pd(xivec, xivec);
            sum_lo  = _mm256_castpd256_pd128(xivec);
            sum_hi  = _mm256_extractf128_pd(xivec, 1);
            sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
            xi -= sse_sum.m128d_f64[0];
#else
            xi -= sse_sum[0];
#endif
        }
        // Remainder loop
        switch(idxrem)
        {
        case(3):
        {
            avec = _mm256_set_pd(0.0, a_fix[idx + 2], a_fix[idx + 1], a_fix[idx + 0]);
            xvec = _mm256_set_pd(
                0.0, x_fix[icol_fix[idx + 2]], x_fix[icol_fix[idx + 1]], x_fix[icol_fix[idx + 0]]);
            xivec   = _mm256_mul_pd(avec, xvec);
            xivec   = _mm256_hadd_pd(xivec, xivec);
            sum_lo  = _mm256_castpd256_pd128(xivec);
            sum_hi  = _mm256_extractf128_pd(xivec, 1);
            sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
            xi -= sse_sum.m128d_f64[0];
#else
            xi -= sse_sum[0];
#endif
            break;
        }
        default:
            for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                xi -= a_fix[idx] * x_fix[icol_fix[idx]];
        }
        x[i] = xi;
        if(!unit)
        {
            idiag = iurow[i] - 1;
            x[i] /= a_fix[idiag];
        }
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U'*x = alpha*b
 */
template <typename T>
inline aoclsparse_status
    trsv_ut_ref_core_avx([[maybe_unused]] const T               alpha,
                         [[maybe_unused]] aoclsparse_int        m,
                         [[maybe_unused]] aoclsparse_index_base base,
                         [[maybe_unused]] const T *__restrict__ a,
                         [[maybe_unused]] const aoclsparse_int *__restrict__ icol,
                         [[maybe_unused]] const aoclsparse_int *__restrict__ ilrow,
                         [[maybe_unused]] const aoclsparse_int *__restrict__ iurow,
                         [[maybe_unused]] const T *__restrict__ b,
                         [[maybe_unused]] T *__restrict__ x,
                         [[maybe_unused]] const bool unit)
{
    return aoclsparse_status_not_implemented;
}
template <> // DOUBLE
inline aoclsparse_status trsv_ut_ref_core_avx<double>(const double          alpha,
                                                      aoclsparse_int        m,
                                                      aoclsparse_index_base base,
                                                      const double *__restrict__ a,
                                                      const aoclsparse_int *__restrict__ icol,
                                                      const aoclsparse_int *__restrict__ ilrow,
                                                      const aoclsparse_int *__restrict__ iurow,
                                                      const double *__restrict__ b,
                                                      double *__restrict__ x,
                                                      const bool unit)
{
    aoclsparse_int        i, idx, idxstart, idxend, idiag;
    aoclsparse_int        idxcnt, tsz, idxrem;
    double                xi, mxi;
    __m256d               avec, xvec, xivec;
    const aoclsparse_int *icol_fix = icol - base;
    const double         *a_fix    = a - base;
    double               *x_fix    = x - base;

    if(alpha != 0.0)
        for(i = 0; i < m; i++)
            x[i] = alpha * b[i];

    for(i = 0; i < m; i++)
    {
        idxstart = iurow[i];
        idxend   = ilrow[i + 1] - 1;
        if(!unit)
        {
            idiag = iurow[i] - 1;
            x[i]  = x[i] / a_fix[idiag];
        }

        xi  = x[i];
        mxi = -xi;

        idxcnt = idxend - idxstart + 1;
        tsz    = 4; // AVX2
        idxrem = idxcnt % tsz;
        for(idx = idxstart; idx <= idxend - idxrem; idx += tsz)
        {
            xvec                     = _mm256_set_pd(x_fix[icol_fix[idx + 3]],
                                 x_fix[icol_fix[idx + 2]],
                                 x_fix[icol_fix[idx + 1]],
                                 x_fix[icol_fix[idx + 0]]);
            avec                     = _mm256_loadu_pd(&a_fix[idx]);
            xivec                    = _mm256_set1_pd(mxi);
            xvec                     = _mm256_fmadd_pd(avec, xivec, xvec);
            x_fix[icol_fix[idx + 3]] = xvec[3];
            x_fix[icol_fix[idx + 2]] = xvec[2];
            x_fix[icol_fix[idx + 1]] = xvec[1];
            x_fix[icol_fix[idx + 0]] = xvec[0];
        }
        // Remainder loop
        switch(idxrem)
        {
        case(3):
        {
            xvec = _mm256_set_pd(
                0.0, x_fix[icol_fix[idx + 2]], x_fix[icol_fix[idx + 1]], x_fix[icol_fix[idx + 0]]);
            avec  = _mm256_set_pd(0.0, a_fix[idx + 2], a_fix[idx + 1], a_fix[idx + 0]);
            xivec = _mm256_set1_pd(mxi);
            xvec  = _mm256_fmadd_pd(avec, xivec, xvec);
            x_fix[icol_fix[idx + 2]] = xvec[2];
            x_fix[icol_fix[idx + 1]] = xvec[1];
            x_fix[icol_fix[idx + 0]] = xvec[0];
            break;
        }
        default:
            for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                x_fix[icol_fix[idx]] -= a_fix[idx] * xi;
        }
    }
    return aoclsparse_status_success;
}

/*
 * Macro kernel templates for L, L^T, L^H, U, U^T and U^T kernels
 * ==============================================================
 */

using namespace kernel_templates;

/* TRSV linear operators over matrix */
enum trsv_op
{
    tran = 0, // alias for aoclsparse_operation_transpose
    herm = 1 // alias for aoclsparse_operation_conjugate_transpose
};

/* ## KERNEL TEMPLATE TRIANGULAR SOLVE
 * Template to define clean CSR TRSV for lower triangular matrices, unit or not diagonal
 * What: solves `alpha` `A` `x` = `b` with lower triangular matrix of A
 * With: `A` matrix, `b` dense arrays
 * Returns: dense array `x`
 *
 * ## User inputs
 * 
 * `aoclsparse_status kt_trsv_l<SZ,SUF,EXT>`:
 *
 * - `const T alpha,`
 * - `aoclsparse_int m,` (size of matrix `A`)
 * - `aoclsparse_index_base base,` (base for matrix `A`, either 0 or 1)
 * - `const T *a,` (matrix A values pointer)
 * - `const aoclsparse_int *icol,` (matrix csr column array pointer)
 * - `const aoclsparse_int *ilrow,` (matrix csr compressed row pointer)
 * - `const aoclsparse_int *idiag,` (csr pointer to diagonal elements)
 * - `const T *b,`
 * - `T *x,`
 * - `const bool unit` (`true` if to use unitary diagonal)
 *
 * ##  Template inputs
 *
 * - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
 * - `SUF` suffix of working type, i.e., `double` or `float`
 * - `EXP` AVX capability, kt_avxext e.g. `AVX` or `AVX512F`, etc...
 */

template <int SZ, typename SUF, kt_avxext EXT>
inline aoclsparse_status kt_trsv_l(const SUF             alpha,
                                   aoclsparse_int        m,
                                   aoclsparse_index_base base,
                                   const SUF *__restrict__ a,
                                   const aoclsparse_int *__restrict__ icol,
                                   const aoclsparse_int *__restrict__ ilrow,
                                   const aoclsparse_int *__restrict__ idiag,
                                   const SUF *__restrict__ b,
                                   SUF *__restrict__ x,
                                   const bool unit)
{
    aoclsparse_int       i, idx, idxend;
    SUF                  xi;
    avxvector_t<SZ, SUF> avec, xvec, pvec;
    aoclsparse_int       idxcnt, idxrem;
    // get the vector length (type size)
    const aoclsparse_int  tsz      = tsz_v<SZ, SUF>;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *a_fix    = a - base;
    SUF                  *x_fix    = x - base;
    for(i = 0; i < m; i++)
    {
        idxend = idiag[i];
        xi     = alpha * b[i];
        idxcnt = idxend - ilrow[i];
        idxrem = idxcnt % tsz;
        pvec   = kt_setzero_p<SZ, SUF>();
        for(idx = ilrow[i]; idx < idxend - idxrem; idx += tsz)
        {

            avec = kt_loadu_p<SZ, SUF>(&a_fix[idx]);
            xvec = kt_set_p<SZ, SUF>(x_fix, &icol_fix[idx]);
            pvec = kt_fmadd_p<SZ, SUF>(avec, xvec, pvec);
        }
        if(idxcnt - tsz >= 0)
        {
            xi -= kt_hsum_p<SZ, SUF>(pvec);
        }
        // process remainder
        // use type container size -1 with zero paddding -> intrinsic
        // rest -> use a loop
        // FIXME: Take care of corner case where tsz-1 = 1
        switch(idxrem)
        {
        case(tsz - 1):
            idx  = idxend - idxrem;
            avec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(a_fix, idx);
            xvec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(x_fix, &icol_fix[idx]);
            xi -= kt_dot_p<SZ, SUF>(avec, xvec);
            break;
        default:
            for(idx = idxend - idxrem; idx < idxend; idx++)
                xi -= a_fix[idx] * x_fix[icol_fix[idx]];
        }
        x[i] = xi;
        if(!unit)
            x[i] /= a_fix[idxend];
    }
    return aoclsparse_status_success;
}

/* ## KERNEL TEMPLATE TRIANGULAR SOLVE
 * Template to define clean CSR TRSV for lower triangular matrices, unit or not diagonal
 * What: solves `alpha` `A`^T `x` = `b` or `alpha` `A`^H `x` = `b`
 *       with lower triangular matrix of A
 * With: `A` matrix, `b` dense arrays
 * Returns: dense array `x`
 * Extra AVX vectors: YES
 *
 * ## User inputs
 * 
 * `aoclsparse_status kt_trsv_lt<SZ,SUF,EXT,OP>`:
 *
 * - `const T alpha,`
 * - `aoclsparse_int m,` (size of matrix `A`)
 * - `aoclsparse_index_base base,` (base for matrix `A`, either 0 or 1)
 * - `const T *a,` (matrix A values pointer)
 * - `const aoclsparse_int *icol,` (matrix csr column array pointer)
 * - `const aoclsparse_int *ilrow,` (matrix csr compressed row pointer)
 * - `const aoclsparse_int *idiag,` (csr pointer to diagonal elements)
 * - `const T *b,`
 * - `T *x,`
 * - `const bool unit` (`true` if to use unitary diagonal)
 *
 * ##  Template inputs
 *
 * - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
 * - `SUF` suffix of working type, i.e., `double` or `float`
 * - `EXP` AVX capability, kt_avxext e.g. `AVX` or `AVX512F`, etc...
 * - `OP` trsv_op enum for transposition operation type 
 *       trsv_op::tran Real-space transpose, and
 *       trsv_op::herm Complex-space conjugate transpose
 */
template <int SZ, typename SUF, kt_avxext EXT, trsv_op OP = trsv_op::tran>
inline aoclsparse_status kt_trsv_lt(const SUF             alpha,
                                    aoclsparse_int        m,
                                    aoclsparse_index_base base,
                                    const SUF *__restrict__ a,
                                    const aoclsparse_int *__restrict__ icol,
                                    const aoclsparse_int *__restrict__ ilrow,
                                    const aoclsparse_int *__restrict__ idiag,
                                    const SUF *__restrict__ b,
                                    SUF *__restrict__ x,
                                    const bool unit)
{
    aoclsparse_int       i, idx, idxend;
    aoclsparse_int       idxcnt, idxrem;
    SUF                  ad, xi, mxi;
    avxvector_t<SZ, SUF> avec, xvec, xivec;
    // get the vector length (considering the type size)
    const aoclsparse_int  tsz      = tsz_v<SZ, SUF>;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *a_fix    = a - base;
    SUF                  *x_fix    = x - base;
    const SUF            *xvptr    = reinterpret_cast<SUF *>(&xvec);

    if(alpha != (SUF)0)
        for(i = 0; i < m; i++)
            x[i] = alpha * b[i];
    for(i = m - 1; i >= 0; i--)
    {
        idxend = idiag[i];
        if(!unit)
        {

            ad = a_fix[idiag[i]];
            if constexpr((std::is_same_v<SUF, std::complex<float>>
                          || std::is_same_v<SUF, std::complex<double>>)&&(OP == trsv_op::herm))
                ad = std::conj(ad);
            x[i] /= ad;
        }
        xi     = x[i];
        mxi    = -xi;
        idxcnt = idxend - ilrow[i];
        idxrem = idxcnt % tsz;
        for(idx = ilrow[i]; idx < idxend - idxrem; idx += tsz)
        {
            xvec = kt_set_p<SZ, SUF>(x_fix, &icol_fix[idx]);
            avec = kt_loadu_p<SZ, SUF>(&a_fix[idx]);
            if constexpr((std::is_same_v<SUF, std::complex<float>>
                          || std::is_same_v<SUF, std::complex<double>>)&&(OP == trsv_op::herm))
                avec = kt_conj_p<SZ, SUF>(avec);
            xivec = kt_set1_p<SZ, SUF>(mxi);
            xvec  = kt_fmadd_p<SZ, SUF>(avec, xivec, xvec);
            kt_scatter_p<SZ, SUF>(xvec, x_fix, &icol_fix[idx]);
        }
        // process remainder
        // use packet-size -1 with zero paddding -> intrinsic
        // rest -> use a loop
        switch(idxrem)
        {
        case(tsz - 1):
            idx  = idxend - idxrem;
            xvec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(x_fix, &icol_fix[idx]);
            avec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(a_fix, idx);
            if constexpr((std::is_same_v<SUF, std::complex<float>>
                          || std::is_same_v<SUF, std::complex<double>>)&&(OP == trsv_op::herm))
                avec = kt_conj_p<SZ, SUF>(avec);
            xivec = kt_set1_p<SZ, SUF>(mxi);
            xvec  = kt_fmadd_p<SZ, SUF>(avec, xivec, xvec);
            for(aoclsparse_int k = 0; k < tsz - 1; k++)
                x_fix[icol_fix[idx + k]] = xvptr[k];
            break;
        default:
            if constexpr((std::is_same_v<SUF, std::complex<float>>
                          || std::is_same_v<SUF, std::complex<double>>)&&(OP == trsv_op::herm))
                for(idx = idxend - idxrem; idx < idxend; idx++)
                    x_fix[icol_fix[idx]] -= std::conj(a_fix[idx]) * xi;
            else
                for(idx = idxend - idxrem; idx < idxend; idx++)
                    x_fix[icol_fix[idx]] -= a_fix[idx] * xi;
        }
    }
    return aoclsparse_status_success;
}

/* ## KERNEL TEMPLATE TRIANGULAR SOLVE
 * Template to define clean CSR TRSV for upper triangular matrices, unit or not diagonal
 * What: solves `alpha` `A` `x` = `b` with upper triangular matrix of A
 * With: `A` matrix, `b` dense arrays
 * Returns: dense array `x`
 *
 * ## User inputs
 * 
 * `aoclsparse_status kt_trsv_u<SZ,SUF,EXT>`:
 *
 * - `const T alpha,`
 * - `aoclsparse_int m,` (size of matrix `A`)
 * - `aoclsparse_index_base base,` (base for matrix `A`, either 0 or 1)
 * - `const T *a,` (matrix A values pointer)
 * - `const aoclsparse_int *icol,` (matrix csr column array pointer)
 * - `const aoclsparse_int *ilrow,` (matrix csr compressed row pointer)
 * - `const aoclsparse_int *iurow,` (csr pointer to upper triangle elements)
 * - `const T *b,`
 * - `T *x,`
 * - `const bool unit` (`true` if to use unitary diagonal)
 *
 * ##  Template inputs
 *
 * - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
 * - `SUF` suffix of working type, i.e., `double` or `float`
 * - `EXP` AVX capability, kt_avxext e.g. `AVX` or `AVX512F`, etc...
 */
template <int SZ, typename SUF, kt_avxext EXT>
inline aoclsparse_status kt_trsv_u(const SUF             alpha,
                                   aoclsparse_int        m,
                                   aoclsparse_index_base base,
                                   const SUF *__restrict__ a,
                                   const aoclsparse_int *__restrict__ icol,
                                   const aoclsparse_int *__restrict__ ilrow,
                                   const aoclsparse_int *__restrict__ iurow,
                                   const SUF *__restrict__ b,
                                   SUF *__restrict__ x,
                                   const bool unit)
{
    aoclsparse_int       i, idiag, idx, idxstart, idxend;
    aoclsparse_int       idxcnt, idxrem;
    SUF                  xi;
    avxvector_t<SZ, SUF> avec, xvec, pvec;
    // get the vector length (type size)
    const aoclsparse_int  tsz      = tsz_v<SZ, SUF>;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *a_fix    = a - base;
    SUF                  *x_fix    = x - base;

    for(i = m - 1; i >= 0; i--)
    {
        idxstart = iurow[i];
        idxend   = ilrow[i + 1] - 1;
        xi       = alpha * b[i];
        idxcnt   = idxend - idxstart + 1;
        idxrem   = idxcnt % tsz;
        pvec     = kt_setzero_p<SZ, SUF>();
        for(idx = idxstart; idx <= idxend - idxrem; idx += tsz)
        {
            xvec = kt_set_p<SZ, SUF>(x_fix, &icol_fix[idx]);
            avec = kt_loadu_p<SZ, SUF>(&a_fix[idx]);
            pvec = kt_fmadd_p<SZ, SUF>(avec, xvec, pvec);
        }
        if(idxcnt - tsz >= 0)
        {
            xi -= kt_hsum_p<SZ, SUF>(pvec);
        }
        // process remainder
        // use packet-size -1 with zero paddding -> intrinsic
        // rest -> use a loop
        switch(idxrem)
        {
        case(tsz - 1):
            idx  = idxend - idxrem + 1;
            xvec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(x_fix, &icol_fix[idx]);
            avec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(a_fix, idx);
            xi -= kt_dot_p<SZ, SUF>(avec, xvec);
            break;
        default:
            for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                xi -= a_fix[idx] * x_fix[icol_fix[idx]];
        }
        x[i] = xi;
        if(!unit)
        {
            idiag = iurow[i] - 1;
            x[i] /= a_fix[idiag];
        }
    }
    return aoclsparse_status_success;
}

/* ## KERNEL TEMPLATE TRIANGULAR SOLVE
 * Template to define clean CSR TRSV for upper triangular matrices, unit or not diagonal
 * What: solves `alpha` `A`^T `x` = `b` or `alpha` `A`^H `x` = `b`
 *       with upper triangular matrix of A
 * With: `A` matrix, `b` dense arrays
 * Returns: dense array `x`
 *
 * ## User inputs
 * 
 * `aoclsparse_status kt_trsv_ut<SZ,SUF,EXT,OP>`:
 *
 * - `const T alpha,`
 * - `aoclsparse_int m,` (size of matrix `A`)
 * - `aoclsparse_index_base base,` (base for matrix `A`, either 0 or 1)
 * - `const T *a,` (matrix A values pointer)
 * - `const aoclsparse_int *icol,` (matrix csr column array pointer)
 * - `const aoclsparse_int *ilrow,` (matrix csr compressed row pointer)
 * - `const aoclsparse_int *iurow,` (csr pointer to upper triangular elements)
 * - `const T *b,`
 * - `T *x,`
 * - `const bool unit` (`true` if to use unitary diagonal)
 *
 * ##  Template inputs
 *
 * - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
 * - `SUF` suffix of working type, i.e., `double` or `float`
 * - `EXP` AVX capability, kt_avxext e.g. `AVX` or `AVX512F`, etc...
 * - `OP` trsv_op enum for transposition operation type 
 *       trsv_op::tran Real-space transpose, and
 *       trsv_op::herm Complex-spase conjugate transpose
 */
template <int SZ, typename SUF, kt_avxext EXT, trsv_op OP = trsv_op::tran>
inline aoclsparse_status kt_trsv_ut(const SUF             alpha,
                                    aoclsparse_int        m,
                                    aoclsparse_index_base base,
                                    const SUF *__restrict__ a,
                                    const aoclsparse_int *__restrict__ icol,
                                    const aoclsparse_int *__restrict__ ilrow,
                                    const aoclsparse_int *__restrict__ iurow,
                                    const SUF *__restrict__ b,
                                    SUF *__restrict__ x,
                                    const bool unit)
{
    aoclsparse_int       i, idx, idxstart, idxend, idiag;
    aoclsparse_int       idxcnt, idxrem;
    SUF                  xi, mxi, ad;
    avxvector_t<SZ, SUF> avec, xivec, xvec;
    // get the vector length (type size)
    const aoclsparse_int  tsz      = tsz_v<SZ, SUF>;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *a_fix    = a - base;
    SUF                  *x_fix    = x - base;
    const SUF            *xvptr    = reinterpret_cast<SUF *>(&xvec);

    if(alpha != (SUF)0.0)
        for(i = 0; i < m; i++)
            x[i] = alpha * b[i];
    for(i = 0; i < m; i++)
    {
        idxstart = iurow[i];
        idxend   = ilrow[i + 1] - 1;
        if(!unit)
        {
            idiag = iurow[i] - 1;
            ad    = a_fix[idiag];
            if constexpr((std::is_same_v<SUF, std::complex<float>>
                          || std::is_same_v<SUF, std::complex<double>>)&&(OP == trsv_op::herm))
                ad = std::conj(ad);
            x[i] = x[i] / ad;
        }
        xi     = x[i];
        mxi    = -xi;
        idxcnt = idxend - idxstart + 1;
        idxrem = idxcnt % tsz;
        for(idx = idxstart; idx <= idxend - idxrem; idx += tsz)
        {
            xvec = kt_set_p<SZ, SUF>(x_fix, &icol_fix[idx]);
            avec = kt_loadu_p<SZ, SUF>(&a_fix[idx]);
            if constexpr((std::is_same_v<SUF, std::complex<float>>
                          || std::is_same_v<SUF, std::complex<double>>)&&(OP == trsv_op::herm))
                avec = kt_conj_p<SZ, SUF>(avec);
            xivec = kt_set1_p<SZ, SUF>(mxi);
            xvec  = kt_fmadd_p<SZ, SUF>(avec, xivec, xvec);
            kt_scatter_p<SZ, SUF>(xvec, x_fix, &icol_fix[idx]);
        }
        // process remainder
        // use packet-size -1 with zero paddding -> intrinsic
        // rest -> use a loop
        switch(idxrem)
        {
        case(tsz - 1):
            idx  = idxend - idxrem + 1;
            xvec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(x_fix, &icol_fix[idx]);
            avec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(a_fix, idx);
            if constexpr((std::is_same_v<SUF, std::complex<float>>
                          || std::is_same_v<SUF, std::complex<double>>)&&(OP == trsv_op::herm))
                avec = kt_conj_p<SZ, SUF>(avec);
            xivec = kt_set1_p<SZ, SUF>(mxi);
            xvec  = kt_fmadd_p<SZ, SUF>(avec, xivec, xvec);
            for(aoclsparse_int k = 0; k < tsz - 1; k++)
                x_fix[icol_fix[idx + k]] = xvptr[k];
            break;
        default:
            if constexpr((std::is_same_v<SUF, std::complex<float>>
                          || std::is_same_v<SUF, std::complex<double>>)&&(OP == trsv_op::herm))
                for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                    x_fix[icol_fix[idx]] -= std::conj(a_fix[idx]) * xi;
            else
                for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                    x_fix[icol_fix[idx]] -= a_fix[idx] * xi;
        }
    }
    return aoclsparse_status_success;
}

/*
 * TRSV dispatcher
 * ===============
 */
template <typename T>
aoclsparse_status
    aoclsparse_trsv(const aoclsparse_operation transpose, /* matrix operation */
                    const T                    alpha, /* scalar for rescaling RHS */
                    aoclsparse_matrix          A, /* matrix data */
                    const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
                    const T                   *b, /* RHS */
                    T                         *x, /* solution */
                    const aoclsparse_int       kid /* user request of Kernel ID (kid) to use */)
{

    // Read the environment variables to update global variable
    // This function updates the num_threads only once.
    aoclsparse_init_once();
    // aoclsparse_context context;
    // context.num_threads = global_context.num_threads;
    // context.is_avx512   = global_context.is_avx512;

    // Quick initial checks
    if(!A || !x || !b || !descr)
        return aoclsparse_status_invalid_pointer;

    const aoclsparse_int nnz = A->nnz;
    const aoclsparse_int m   = A->m;

    if(m <= 0 || nnz <= 0)
        return aoclsparse_status_invalid_size;

    if(m != A->n) // Matrix not square
    {
        return aoclsparse_status_invalid_value;
    }

    // Check for base index incompatibility
    // There is an issue that zero-based indexing is defined in two separate places and
    // can lead to ambiguity, we check that both are consistent.
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    // Check if descriptor's index-base is valid (and A's index-base must be the same)
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    if(transpose != aoclsparse_operation_none && transpose != aoclsparse_operation_transpose
       && transpose != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_not_implemented;

    if(descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_triangular)
    {
        return aoclsparse_status_invalid_value;
    }

    if(descr->fill_mode != aoclsparse_fill_mode_lower
       && descr->fill_mode != aoclsparse_fill_mode_upper)
        return aoclsparse_status_not_implemented;

    // Unpack A and check
    if(!A->opt_csr_ready)
    {
        // user did not check the matrix, call optimize
        aoclsparse_status status = aoclsparse_csr_optimize<T>(A);
        if(status != aoclsparse_status_success)
            return status; // LCOV_EXCL_LINE
    }

    // From this point on A->opt_csr_ready is true

    // Make sure we have the right type before casting
    if(!((A->val_type == aoclsparse_dmat && std::is_same_v<T, double>)
         || (A->val_type == aoclsparse_smat && std::is_same_v<T, float>)
         || (A->val_type == aoclsparse_cmat && std::is_same_v<T, std::complex<float>>)
         || (A->val_type == aoclsparse_zmat && std::is_same_v<T, std::complex<double>>)))
        return aoclsparse_status_wrong_type;
    const T *a = (T *)((A->opt_csr_mat).csr_val);

    const bool unit = descr->diag_type == aoclsparse_diag_type_unit;

    if(!A->opt_csr_full_diag && !unit) // not of full rank, linear system cannot be solved
    {
        return aoclsparse_status_invalid_value;
    }

    const aoclsparse_int *icol = (A->opt_csr_mat).csr_col_ptr;
    // beggining of the row
    const aoclsparse_int *ilrow = (A->opt_csr_mat).csr_row_ptr;
    // position of the diagonal element (includes zeros) always has min(m,n) elements
    const aoclsparse_int *idiag = A->idiag;
    // ending of the row
    const aoclsparse_int *iurow = A->iurow;
    const bool            lower = descr->fill_mode == aoclsparse_fill_mode_lower;
    aoclsparse_index_base base  = A->internal_base_index;

    // CPU ID dispatcher sets recommended Kernel ID to use, this can be influenced by
    // the user-requested "kid" hint
    // TODO update when libcpuid is merged into aoclsparse
    aoclsparse_int usekid = 2; // Defaults to 2 (AVX2 256-bits)
    if(kid >= 0)
    {
        switch(kid)
        {
        case 1: // reference AVX2 256b implementation
            // Take care that kid=1 only works for real double
            if(!std::is_same_v<T, double>)
                return aoclsparse_status_not_implemented;
            usekid = kid;
            break;
        case 0: // reference implementation (no explicit vectorization)
        case 2: // AVX2 256b
            usekid = kid;
            break;
        case 3: // AVX-512F 512b
            if(sparse_global_context.is_avx512)
                usekid = kid;
            // Requested kid not available on host,
            // stay with kid suggested by CPU ID...
            break;
        default: // use kid suggested by CPU ID...
            break;
        }
    }

    /* Available kernel table
     * ======================
     * kernel                 | description                             | type support
     * -----------------------+----------------------------------------------------------------------
     * trsv_l_ref_core        | reference vanilla for Lx=b              | float/double/cfloat/cdouble
     * trsv_lt_ref_core       | reference vanilla for L^T x=b           | float/double/cfloat/cdouble
     * trsv_lh_ref_core       | reference vanilla for L^H x=b           |              cfloat/cdouble
     * trsv_u_ref_core        | reference vanilla for Ux=b              | float/double/cfloat/cdouble
     * trsv_ut_ref_core       | reference vanilla for U^T x=b           | float/double/cfloat/cdouble
     * trsv_uh_ref_core       | reference vanilla for U^H x=b           |              cfloat/cdouble
     * - - - - - - - - - - - -+- - - - - - - - - - - - - - - - - - - - -+- - - - - - - - - - - - - -
     * trsv_l_ref_core_avx    | hand-coded AVX2 kernel for Lx=b         | double only
     * trsv_lt_ref_core_avx   | hand-coded AVX2 kernel for L^Tx=b       | double only
     * trsv_u_ref_core_avx    | hand-coded AVX2 kernel for Ux=b         | double only
     * trsv_ut_ref_core_avx   | hand-coded AVX2 kernel for U^Tx=b       | double only
     * - - - - - - - - - - - -+- - - - - - - - - - - - - - - - - - - - -+- - - - - - - - - - - - - -
     * kt_trsv_l<256,*,AVX>   | L solver AVX extensions on 256-bit      | float/double/cfloat/cdouble
     *                        | wide register implementation            |
     * kt_trsv_lt<256,*,AVX>  | L^T/H solver AVX extensions on 256-bit  | float/double/cfloat/cdouble
     *                        | wide register implementation            |
     * kt_trsv_u<256,*,AVX>   | U solver AVX extensions on 256-bit      | float/double/cfloat/cdouble
     *                        | wide register implementation            |
     * kt_trsv_ut<256,*,AVX>  | U^T/H solver AVX extensions on 256-bit  | float/double/cfloat/cdouble
     *                        | wide register implementation            |
     * - - - - - - - - - - - -+- - - - - - - - - - - - - - - - - - - - -+- - - - - - - - - - - - - -
     * kt_trsv_l_<512,*,*>    | L solver AVX512F extensions on 512-bit  | float/double/cfloat/cdouble
     *                        | wide register implementation            |
     * kt_trsv_lt_<512,*,*>   | L^T/H solver AVX512F extensions on 512- | float/double/cfloat/cdouble
     *                        | bit wide register implementation        |
     * kt_trsv_u_<512,*,*>    | U solver AVX512F extensions on 512-bit  | float/double/cfloat/cdouble
     *                        | wide register implementation            |
     * kt_trsv_ut_<512,*,*>   | U^T/H solver AVX512F extensions on 512- | float/double/cfloat/cdouble
     *                        | bit wide register implementation        |
     * -----------------------+----------------------------------------------------------------------
     */
    if(lower)
    {
        switch(transpose)
        {
        case aoclsparse_operation_none:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                return kt_trsv_l<512, T, kt_avxext::ANY>(
                    alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
#endif
            case 2: // AVX2
                return kt_trsv_l<256, T, kt_avxext::AVX>(
                    alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            case 1: // Reference AVX implementation
                return trsv_l_ref_core_avx(alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            default: // Reference implementation
                return trsv_l_ref_core(alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            }
            break;
        case aoclsparse_operation_transpose:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                return kt_trsv_lt<512, T, kt_avxext::ANY>(
                    alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
#endif
            case 2: // AVX2
                return kt_trsv_lt<256, T, kt_avxext::AVX>(
                    alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            case 1: // Reference AVX implementation
                return trsv_lt_ref_core_avx(alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            default: // Reference implementation
                return trsv_lt_ref_core(alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            }
            break;
        case aoclsparse_operation_conjugate_transpose:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return kt_trsv_lt<512, T, kt_avxext::ANY, trsv_op::herm>(
                        alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                else
                    return kt_trsv_lt<512, T, kt_avxext::ANY>(
                        alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
#endif
            case 2: // AVX2
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return kt_trsv_lt<256, T, kt_avxext::AVX, trsv_op::herm>(
                        alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                else
                    return kt_trsv_lt<256, T, kt_avxext::AVX>(
                        alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            case 1: // Reference AVX implementation
                return trsv_lt_ref_core_avx(alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            default: // Reference implementation
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return trsv_lh_ref_core<T>(alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                else
                    return trsv_lt_ref_core(alpha, m, base, a, icol, ilrow, idiag, b, x, unit);
                break;
            }
            break;
        }
    }
    else // upper
    {
        switch(transpose)
        {
        case aoclsparse_operation_none:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                return kt_trsv_u<512, T, kt_avxext::ANY>(
                    alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
#endif
            case 2: // AVX2
                return kt_trsv_u<256, T, kt_avxext::AVX>(
                    alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            case 1: // Reference AVX implementation
                return trsv_u_ref_core_avx(alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            default: // Reference implementation
                return trsv_u_ref_core(alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            }
            break;
        case aoclsparse_operation_transpose:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                return kt_trsv_ut<512, T, kt_avxext::ANY>(
                    alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
#endif
            case 2: // AVX2
                return kt_trsv_ut<256, T, kt_avxext::AVX>(
                    alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            case 1: // Reference AVX implementation
                return trsv_ut_ref_core_avx(alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            default: // Reference implementation
                return trsv_ut_ref_core(alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            }
            break;
        case aoclsparse_operation_conjugate_transpose:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return kt_trsv_ut<512, T, kt_avxext::ANY, trsv_op::herm>(
                        alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                else
                    return kt_trsv_ut<512, T, kt_avxext::ANY>(
                        alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
#endif
            case 2: // AVX2
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return kt_trsv_ut<256, T, kt_avxext::AVX, trsv_op::herm>(
                        alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                else
                    return kt_trsv_ut<256, T, kt_avxext::AVX>(
                        alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            case 1: // Reference AVX implementation
                return trsv_ut_ref_core_avx(alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            default: // Reference implementation
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return trsv_uh_ref_core<T>(alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                else
                    return trsv_ut_ref_core(alpha, m, base, a, icol, ilrow, iurow, b, x, unit);
                break;
            }
            break;
        }
    }
    // It should never be reached...
    return aoclsparse_status_internal_error;
}
#endif // AOCLSPARSE_SV_HPP
