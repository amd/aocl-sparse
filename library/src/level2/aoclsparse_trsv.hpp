/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_descr.h"
#include "aoclsparse_csr_util.hpp"

#include <immintrin.h>

/*
 * TRiangular SolVer dispacher
 * TRSV dispacher and various templated and vectorized triangular solve kernels
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
                    const aoclsparse_int       avxversion /* optimization level */);

/* Core computation of TRSV, assumed A is optimized
 * solves L*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_l_ref_core(const T               alpha,
                                                aoclsparse_int        m,
                                                const T              *a,
                                                const aoclsparse_int *icol,
                                                const aoclsparse_int *ilrow,
                                                const aoclsparse_int *idiag,
                                                const T              *b,
                                                T                    *x,
                                                const bool            unit)
{
    aoclsparse_int i, idx;
    T              xi;
    for(i = 0; i < m; i++)
    {
        xi = alpha * b[i];
        // TODO #ifdef _OPENMP #pragma omp parallel for num_threads(thread->num_threads)  private(...)
        for(idx = ilrow[i]; idx < idiag[i]; idx++)
            xi -= a[idx] * x[icol[idx]];
        if(!unit)
            xi /= a[idiag[i]];
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
                                                 const T              *a,
                                                 const aoclsparse_int *icol,
                                                 const aoclsparse_int *ilrow,
                                                 const aoclsparse_int *idiag,
                                                 const T              *b,
                                                 T                    *x,
                                                 const bool            unit)
{
    aoclsparse_int i, idx;
    T              xi;

    for(i = 0; i < m; i++)
        x[i] = alpha * b[i];
    for(i = m - 1; i >= 0; i--)
    {
        if(!unit)
            x[i] /= a[idiag[i]];
        // propagate value of x[i] through the column (used to the row but now is transposed)
        // TODO OMP?
        for(idx = ilrow[i]; idx < idiag[i]; idx++)
            x[icol[idx]] -= a[idx] * x[i];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_u_ref_core(const T               alpha,
                                                aoclsparse_int        m,
                                                const T              *a,
                                                const aoclsparse_int *icol,
                                                const aoclsparse_int *ilrow,
                                                const aoclsparse_int *iurow,
                                                const T              *b,
                                                T                    *x,
                                                const bool            unit)
{
    aoclsparse_int i, idx, idxstart, idxend, idiag;
    T              xi;

    for(i = m - 1; i >= 0; i--)
    {
        idxstart = iurow[i];
        // ilrow[i+1]-1 always points to last element of U at row i
        idxend = ilrow[i + 1] - 1;
        xi     = alpha * b[i];
        // TODO #ifdef _OPENMP #pragma omp parallel for num_threads(thread->num_threads)  private(...)
        for(idx = idxstart; idx <= idxend; idx++)
            xi -= a[idx] * x[icol[idx]];
        x[i] = xi;
        if(!unit)
        {
            // urow[i]-1 always points to idiag[i]
            idiag = iurow[i] - 1;
            x[i] /= a[idiag];
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
                                                 const T              *a,
                                                 const aoclsparse_int *icol,
                                                 const aoclsparse_int *ilrow,
                                                 const aoclsparse_int *iurow,
                                                 const T              *b,
                                                 T                    *x,
                                                 const bool            unit)
{
    aoclsparse_int i, idx, idxstart, idxend, idiag;
    for(i = 0; i < m; i++)
        x[i] = alpha * b[i];
    for(i = 0; i < m; i++)
    {
        if(!unit)
        {
            // urow[i]-1 always points to idiag[i]
            idiag = iurow[i] - 1;
            x[i] /= a[idiag];
        }
        idxstart = iurow[i];
        // ilrow[i+1]-1 always points to last element of U at row i
        idxend = ilrow[i + 1] - 1;
        // TODO OMP?
        for(idx = idxstart; idx <= idxend; idx++)
            x[icol[idx]] -= a[idx] * x[i];
    }
    return aoclsparse_status_success;
}

/* 
 * AVX and AVX2 kernels
 */

/* Core computation of TRSV, assumed A is optimized
 * solves L*x = alpha*b
 */
template <typename T>
inline aoclsparse_status trsv_l_ref_core_avx(const T               alpha,
                                             aoclsparse_int        m,
                                             const T              *a,
                                             const aoclsparse_int *icol,
                                             const aoclsparse_int *ilrow,
                                             const aoclsparse_int *idiag,
                                             const T              *b,
                                             T                    *x,
                                             const bool            unit);
template <> // FLOAT
inline aoclsparse_status trsv_l_ref_core_avx<float>(const float           alpha,
                                                    aoclsparse_int        m,
                                                    const float          *a,
                                                    const aoclsparse_int *icol,
                                                    const aoclsparse_int *ilrow,
                                                    const aoclsparse_int *idiag,
                                                    const float          *b,
                                                    float                *x,
                                                    const bool            unit)
{
    return aoclsparse_status_not_implemented;
}
template <> // DOUBLE
inline aoclsparse_status trsv_l_ref_core_avx<double>(const double          alpha,
                                                     aoclsparse_int        m,
                                                     const double         *a,
                                                     const aoclsparse_int *icol,
                                                     const aoclsparse_int *ilrow,
                                                     const aoclsparse_int *idiag,
                                                     const double         *b,
                                                     double               *x,
                                                     const bool            unit)
{
    aoclsparse_int i, idx, idxend;
    double         xi;
    __m256d        avec, xvec, xivec;
    __m128d        sum_lo, sum_hi, sse_sum;
    aoclsparse_int idxcnt, idxk, idxrem;

    for(i = 0; i < m; i++)
    {
        idxend = idiag[i];
        xi     = alpha * b[i];
        // ---------------------------------------------
        // for (idx = icrow[i]; idx<idxend; idx++)
        //     yi = yi - a[idx] * y[icol[idx]];
        // ---------------------------------------------
        idxcnt = idxend - ilrow[i];
        idxk   = 4; // AVX2
        idxrem = idxcnt % idxk;
        // TODO #ifdef _OPENMP #pragma omp parallel for num_threads(thread->num_threads)  private(avec, xvec, xivec)
        for(idx = ilrow[i]; idx < idxend - idxrem; idx += idxk)
        {
            avec = _mm256_loadu_pd(&a[idx]);
            xvec = _mm256_set_pd(
                x[icol[idx + 3]], x[icol[idx + 2]], x[icol[idx + 1]], x[icol[idx + 0]]);
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
            avec  = _mm256_set_pd(0.0, a[idx + 2], a[idx + 1], a[idx + 0]);
            xvec  = _mm256_set_pd(0.0, x[icol[idx + 2]], x[icol[idx + 1]], x[icol[idx + 0]]);
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
                xi -= a[idx] * x[icol[idx]];
        }
        x[i] = xi;
        if(!unit)
            x[i] /= a[idxend];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves L'*x = alpha*b
 */
template <typename T>
inline aoclsparse_status trsv_lt_ref_core_avx(const T               alpha,
                                              aoclsparse_int        m,
                                              const T              *a,
                                              const aoclsparse_int *icol,
                                              const aoclsparse_int *ilrow,
                                              const aoclsparse_int *idiag,
                                              const T              *b,
                                              T                    *x,
                                              const bool            unit);
template <> // FLOAT
inline aoclsparse_status trsv_lt_ref_core_avx<float>(const float           alpha,
                                                     aoclsparse_int        m,
                                                     const float          *a,
                                                     const aoclsparse_int *icol,
                                                     const aoclsparse_int *ilrow,
                                                     const aoclsparse_int *idiag,
                                                     const float          *b,
                                                     float                *x,
                                                     const bool            unit)
{
    return aoclsparse_status_not_implemented;
}
template <> // DOUBLE
inline aoclsparse_status trsv_lt_ref_core_avx<double>(const double          alpha,
                                                      aoclsparse_int        m,
                                                      const double         *a,
                                                      const aoclsparse_int *icol,
                                                      const aoclsparse_int *ilrow,
                                                      const aoclsparse_int *idiag,
                                                      const double         *b,
                                                      double               *x,
                                                      const bool            unit)
{
    aoclsparse_int i, idx, idxend;
    aoclsparse_int idxcnt, idxk, idxrem;
    double         xi, mxi;
    __m128d        sum_lo, sum_hi, sse_sum;
    __m256d        avec, xvec, xivec;

    if(alpha != 0.0)
        for(i = 0; i < m; i++)
            x[i] = alpha * b[i];

    for(i = m - 1; i >= 0; i--)
    {
        idxend = idiag[i];

        if(!unit)
            x[i] /= a[idiag[i]];
        xi  = x[i];
        mxi = -xi;

        idxcnt = idxend - ilrow[i];
        idxk   = 4; // AVX2
        idxrem = idxcnt % idxk;
        // TODO OMP?
        for(idx = ilrow[i]; idx < idxend - idxrem; idx += idxk)
        {
            xvec = _mm256_set_pd(
                x[icol[idx + 3]], x[icol[idx + 2]], x[icol[idx + 1]], x[icol[idx + 0]]);
            avec             = _mm256_loadu_pd(&a[idx]);
            xivec            = _mm256_set1_pd(mxi);
            xvec             = _mm256_fmadd_pd(avec, xivec, xvec);
            x[icol[idx + 3]] = xvec[3];
            x[icol[idx + 2]] = xvec[2];
            x[icol[idx + 1]] = xvec[1];
            x[icol[idx + 0]] = xvec[0];
        }
        // Remainder loop
        switch(idxrem)
        {
        case(3):
        {
            xvec  = _mm256_set_pd(0.0, x[icol[idx + 2]], x[icol[idx + 1]], x[icol[idx + 0]]);
            avec  = _mm256_set_pd(0.0, a[idx + 2], a[idx + 1], a[idx + 0]);
            xivec = _mm256_set1_pd(mxi);
            xvec  = _mm256_fmadd_pd(avec, xivec, xvec);
            x[icol[idx + 2]] = xvec[2];
            x[icol[idx + 1]] = xvec[1];
            x[icol[idx + 0]] = xvec[0];
            break;
        }
        default:
            for(idx = idxend - idxrem; idx < idxend; idx++)
                x[icol[idx]] -= a[idx] * xi;
        }
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U*x = alpha*b
 */
template <typename T>
inline aoclsparse_status trsv_u_ref_core_avx(const T               alpha,
                                             aoclsparse_int        m,
                                             const T              *a,
                                             const aoclsparse_int *icol,
                                             const aoclsparse_int *ilrow,
                                             const aoclsparse_int *iurow,
                                             const T              *b,
                                             T                    *x,
                                             const bool            unit);
template <> // FLOAT
inline aoclsparse_status trsv_u_ref_core_avx<float>(const float           alpha,
                                                    aoclsparse_int        m,
                                                    const float          *a,
                                                    const aoclsparse_int *icol,
                                                    const aoclsparse_int *ilrow,
                                                    const aoclsparse_int *iurow,
                                                    const float          *b,
                                                    float                *x,
                                                    const bool            unit)
{
    return aoclsparse_status_not_implemented;
}
template <> // DOUBLE
inline aoclsparse_status trsv_u_ref_core_avx<double>(const double          alpha,
                                                     aoclsparse_int        m,
                                                     const double         *a,
                                                     const aoclsparse_int *icol,
                                                     const aoclsparse_int *ilrow,
                                                     const aoclsparse_int *iurow,
                                                     const double         *b,
                                                     double               *x,
                                                     const bool            unit)
{
    aoclsparse_int i, idx, idxstart, idxend;
    double         xi;
    __m256d        avec, xvec, xivec;
    __m128d        sum_lo, sum_hi, sse_sum;
    aoclsparse_int idxcnt, idxk, idxrem;
    aoclsparse_int idiag;

    for(i = m - 1; i >= 0; i--)
    {
        idxstart = iurow[i];
        idxend   = ilrow[i + 1] - 1;
        xi       = alpha * b[i];
        idxcnt   = idxend - idxstart + 1;
        idxk     = 4; // AVX2
        idxrem   = idxcnt % idxk;
        // TODO #ifdef _OPENMP #pragma omp parallel for num_threads(thread->num_threads)  private(avec, xvec, xivec)
        for(idx = idxstart; idx <= idxend - idxrem; idx += idxk)
        {
            avec = _mm256_loadu_pd(&a[idx]);
            xvec = _mm256_set_pd(
                x[icol[idx + 3]], x[icol[idx + 2]], x[icol[idx + 1]], x[icol[idx + 0]]);
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
            avec    = _mm256_set_pd(0.0, a[idx + 2], a[idx + 1], a[idx + 0]);
            xvec    = _mm256_set_pd(0.0, x[icol[idx + 2]], x[icol[idx + 1]], x[icol[idx + 0]]);
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
                xi -= a[idx] * x[icol[idx]];
        }
        x[i] = xi;
        if(!unit)
        {
            idiag = iurow[i] - 1;
            x[i] /= a[idiag];
        }
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U'*x = alpha*b
 */
template <typename T>
inline aoclsparse_status trsv_ut_ref_core_avx(const T               alpha,
                                              aoclsparse_int        m,
                                              const T              *a,
                                              const aoclsparse_int *icol,
                                              const aoclsparse_int *ilrow,
                                              const aoclsparse_int *iurow,
                                              const T              *b,
                                              T                    *x,
                                              const bool            unit);
template <> // FLOAT
inline aoclsparse_status trsv_ut_ref_core_avx<float>(const float           alpha,
                                                     aoclsparse_int        m,
                                                     const float          *a,
                                                     const aoclsparse_int *icol,
                                                     const aoclsparse_int *ilrow,
                                                     const aoclsparse_int *iurow,
                                                     const float          *b,
                                                     float                *x,
                                                     const bool            unit)
{
    return aoclsparse_status_not_implemented;
}
template <> // DOUBLE
inline aoclsparse_status trsv_ut_ref_core_avx<double>(const double          alpha,
                                                      aoclsparse_int        m,
                                                      const double         *a,
                                                      const aoclsparse_int *icol,
                                                      const aoclsparse_int *ilrow,
                                                      const aoclsparse_int *iurow,
                                                      const double         *b,
                                                      double               *x,
                                                      const bool            unit)
{
    aoclsparse_int i, idx, idxstart, idxend, idiag;
    aoclsparse_int idxcnt, idxk, idxrem;
    double         xi, mxi;
    __m128d        sum_lo, sum_hi, sse_sum;
    __m256d        avec, xvec, xivec;

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
            x[i]  = x[i] / a[idiag];
        }

        xi  = x[i];
        mxi = -xi;

        idxcnt = idxend - idxstart + 1;
        idxk   = 4; // AVX2
        idxrem = idxcnt % idxk;
        // TODO OMP?
        for(idx = idxstart; idx <= idxend - idxrem; idx += idxk)
        {
            xvec = _mm256_set_pd(
                x[icol[idx + 3]], x[icol[idx + 2]], x[icol[idx + 1]], x[icol[idx + 0]]);
            avec             = _mm256_loadu_pd(&a[idx]);
            xivec            = _mm256_set1_pd(mxi);
            xvec             = _mm256_fmadd_pd(avec, xivec, xvec);
            x[icol[idx + 3]] = xvec[3];
            x[icol[idx + 2]] = xvec[2];
            x[icol[idx + 1]] = xvec[1];
            x[icol[idx + 0]] = xvec[0];
        }
        // Remainder loop
        switch(idxrem)
        {
        case(3):
        {
            xvec  = _mm256_set_pd(0.0, x[icol[idx + 2]], x[icol[idx + 1]], x[icol[idx + 0]]);
            avec  = _mm256_set_pd(0.0, a[idx + 2], a[idx + 1], a[idx + 0]);
            xivec = _mm256_set1_pd(mxi);
            xvec  = _mm256_fmadd_pd(avec, xivec, xvec);
            x[icol[idx + 2]] = xvec[2];
            x[icol[idx + 1]] = xvec[1];
            x[icol[idx + 0]] = xvec[0];
            break;
        }
        default:
            for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                x[icol[idx]] -= a[idx] * xi;
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status
    aoclsparse_trsv(const aoclsparse_operation transpose, /* matrix operation */
                    const T                    alpha, /* scalar for rescaling RHS */
                    aoclsparse_matrix          A, /* matrix data */
                    const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
                    const T                   *b, /* RHS */
                    T                         *x, /* solution */
                    const aoclsparse_int       avxversion /* optimization level */)
{
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

    if(transpose != aoclsparse_operation_none && transpose != aoclsparse_operation_transpose)
        return aoclsparse_status_not_implemented;

    if(descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_triangular)
        return aoclsparse_status_invalid_value;

    if(descr->fill_mode != aoclsparse_fill_mode_lower
       && descr->fill_mode != aoclsparse_fill_mode_upper)
        return aoclsparse_status_not_implemented;

    // Unpack A and check
    if(!A->opt_csr_ready)
    {
        // user did not check the matrix, call optimize
        aoclsparse_status status = aoclsparse_csr_optimize<T>(A);
        if(status != aoclsparse_status_success)
            return status; // This should not happen...
    }
    // From this point on A->opt_csr_ready is true

    const bool unit = descr->diag_type == aoclsparse_diag_type_unit;

    if(!A->opt_csr_full_diag && !unit) // not of full rank, linear system cannot be solved
        return aoclsparse_status_invalid_value;

    const T              *a    = (T *)((A->opt_csr_mat).csr_val);
    const aoclsparse_int *icol = (A->opt_csr_mat).csr_col_ptr;
    // beggining of the row
    const aoclsparse_int *ilrow = (A->opt_csr_mat).csr_row_ptr;
    // position of the diagonal element (includes zeros) always has min(m,n) elements
    const aoclsparse_int *idiag = A->idiag;
    // ending of the row
    const aoclsparse_int *iurow = A->iurow;
    const bool            lower = descr->fill_mode == aoclsparse_fill_mode_lower;
    const bool            trans = transpose == aoclsparse_operation_transpose;

    if(lower)
    {
        if(!trans)
        {
            switch(avxversion)
            {
            case 2: // AVX512
                return aoclsparse_status_not_implemented;
                break;
            case 1: // AVX and AVX2
                return trsv_l_ref_core_avx(alpha, m, a, icol, ilrow, idiag, b, x, unit);
                break;
            default: // Reference implementation
                return trsv_l_ref_core(alpha, m, a, icol, ilrow, idiag, b, x, unit);
                break;
            }
        }
        else
        {
            switch(avxversion)
            {
            case 2: // AVX512
                return aoclsparse_status_not_implemented;
                break;
            case 1: // AVX and AVX2
                return trsv_lt_ref_core_avx(alpha, m, a, icol, ilrow, idiag, b, x, unit);
                break;
            default: // Reference implementation
                return trsv_lt_ref_core(alpha, m, a, icol, ilrow, idiag, b, x, unit);
                break;
            }
        }
    }
    else
    {
        if(!trans)
        {
            switch(avxversion)
            {
            case 2: // AVX512
                return aoclsparse_status_not_implemented;
                break;
            case 1: // AVX and AVX2
                return trsv_u_ref_core_avx(alpha, m, a, icol, ilrow, iurow, b, x, unit);
                break;
            default: // Reference implementation
                return trsv_u_ref_core(alpha, m, a, icol, ilrow, iurow, b, x, unit);
                break;
            }
        }
        else
        {
            switch(avxversion)
            {
            case 2: // AVX512
                return aoclsparse_status_not_implemented;
                break;
            case 1: // AVX and AVX2
                return trsv_ut_ref_core_avx(alpha, m, a, icol, ilrow, iurow, b, x, unit);
                break;
            default: // Reference implementation
                return trsv_ut_ref_core(alpha, m, a, icol, ilrow, iurow, b, x, unit);
                break;
            }
        }
    }
    return aoclsparse_status_success;
}
#endif // AOCLSPARSE_SV_HPP
