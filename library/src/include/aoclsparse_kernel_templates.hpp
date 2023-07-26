/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * Reusable vectorized kernel template library (KT)
 * ================================================
 *
 * SYNOPSYS
 * --------
 *
 * Micro kernels are classified on two levels:
 *  - Level 0 micro kernels, these expand directly to IMM intrinsic instructions, while
 *  - Level 1 micro kernels, make use of one or more level 0 micro kernels.
 *
 * For a simple implementation of BLAS Level 2 SPMV solver instantiated 8 times, see
 * ktlvl2_test.cpp
 * For BLAS Level 2 TRSV solver using this template see library/src/level2/aoclsparse_trsv.hpp
 */

// This file is not polished
// clang-format off

#ifndef AOCLSPARSE_KERNEL_TEMPLATES_T_HPP
#define AOCLSPARSE_KERNEL_TEMPLATES_T_HPP
#include <immintrin.h>
#include <type_traits>
#include <limits>
#include <cmath>
#include <complex>

#ifdef KT_ADDRESS_TYPE
#define kt_addr_t KT_ADDRESS_TYPE
#else
#define kt_addr_t size_t
#endif

#ifdef __OPTIMIZE__
    #define KT_FORCE_INLINE inline __attribute__((__always_inline__))
#else
    #define KT_FORCE_INLINE inline
#endif

namespace kernel_templates {

// AVX CPU instrinsic extensions to implement
// * ANY      All targets
// * AVX      All extensions up to AVX2: AVX, FMA, ...
// * AVX512F  AVX512 Foundations
// * AVX512VL Use zero-masked instrinsics, ...
// * AVX512DQ ...
// Each extension needs to be a superset of the previous
enum kt_avxext
{
    ANY      = 0xFFFF,
    AVX      = 1,
    AVX512F  = 1+2,
    AVX512VL = 1+2+4,
    AVX512DQ = 1+2+4+8
};

// Delta function (used for zero-padding)
template <typename T, int L> constexpr T pz(const T &x) noexcept
{
    if constexpr( L >= 0 )
        return x;
    else
        return (T) 0.0;
};

// compatibility macro for windows
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
    #define kt_sse_scl(kt_v) kt_v.m128d_f64[0];
#else
    #define kt_sse_scl(kt_v) kt_v[0];
#endif

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

template <int SZA, int SZB, typename SUFA, typename SUFB> struct kt_is_same  {
    constexpr operator bool() const noexcept { return SZA==SZB && std::is_same_v<SUFA,SUFB>; } };

// helper template type for avx full vectors
template <int, typename> struct avxvector { };
template <> struct avxvector<256, double>  { using type      = __m256d;             // Vector type
                                             using half_type = __m128d;             // Associated "half" length vector type
                                             static constexpr size_t value = 4;     // Vector length, packet size
                                             static constexpr size_t half = 4 >> 1; // Length of half vector
                                             static constexpr size_t count = 4;     // amount of elements of type `typename` that the vector holds
                                             constexpr operator size_t() const noexcept { return value; } }; // Convenience operator to return vector length
template <> struct avxvector<256, float>   { using type = __m256;
                                             using half_type = __m128;
                                             static constexpr size_t value = 8;
                                             static constexpr size_t half = 8 >> 1;
                                             static constexpr size_t count = 8;
                                             constexpr operator size_t() const noexcept { return value; } };
template <> struct avxvector<256, cdouble> { using type      = __m256d;
                                             using half_type = __m128d;
                                             static constexpr size_t value = 4;
                                             static constexpr size_t half = 4 >> 1;
                                             static constexpr size_t count = 4 >> 1;
                                             constexpr operator size_t() const noexcept { return value; } };
template <> struct avxvector<256, cfloat>  { using type = __m256;
                                             using half_type = __m128;
                                             static constexpr size_t value = 8;
                                             static constexpr size_t half = 8 >> 1;
                                             static constexpr size_t count = 8 >> 1;
                                             constexpr operator size_t() const noexcept { return value; } };
#ifdef USE_AVX512
template <> struct avxvector<512, double>  { using type = __m512d;
                                             using half_type = __m256d;
                                             static constexpr size_t value = 8;
                                             static constexpr size_t half = 8 >> 1;
                                             static constexpr size_t count = 8;
                                             constexpr operator size_t() const noexcept { return value; } };
template <> struct avxvector<512, float>   { using type = __m512;
                                             using half_type = __m256;
                                             static constexpr size_t value = 16;
                                             static constexpr size_t half = 16 >> 1;
                                             static constexpr size_t count = 16;
                                             constexpr operator size_t() const noexcept { return value; } };
template <> struct avxvector<512, cdouble> { using type = __m512d;
                                             using half_type = __m256d;
                                             static constexpr size_t value = 8;
                                             static constexpr size_t half = 8 >> 1;
                                             static constexpr size_t count = 8 >> 1;
                                             constexpr operator size_t() const noexcept { return value; } };
template <> struct avxvector<512, cfloat>  { using type = __m512;
                                             using half_type = __m256;
                                             static constexpr size_t value = 16;
                                             static constexpr size_t half = 16 >> 1;
                                             static constexpr size_t count = 16 >> 1;
                                             constexpr operator size_t() const noexcept { return value; } };
#endif

// helper template type for avx vectors
template <int SZ, typename SUF> using avxvector_t = typename avxvector<SZ, SUF>::type;
// helper template type for "half" avx vectors
template <int SZ, typename SUF> using avxvector_half_t = typename avxvector<SZ, SUF>::half_type;
// helper template value (length) for avx vectors (pack size)
template <int SZ, typename SUF> inline constexpr size_t avxvector_v = avxvector<SZ, SUF>::value;
// helper template value (length) for "half" avx vectors (half pack size)
template <int SZ, typename SUF> inline constexpr size_t avxvector_half_v = avxvector<SZ, SUF>::half;
template <int SZ, typename SUF> inline constexpr size_t hsz_v = avxvector<SZ, SUF>::half;
// helper template type-storage size (how many elements of a type fit in a vector)
// for real-valued types it matches with pack size, for complex it is half (real,imag)
template <int SZ, typename SUF> inline constexpr size_t tsz_v = avxvector<SZ, SUF>::count;

// Level 0 micro kernels
// kernels at this level expands to IMM intrinsic instructions
// =======================================================================

// Zero out an AVX register
// return an avxvector filled with zeroes.
//
// Example: `avxvector<256,float> v = kt_setzero_p<256,float>() is equivalent to `v = _mm256_setzero_ps()`
template<int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ,SUF> kt_setzero_p(void) noexcept;
template<> avxvector_t<256,float>   KT_FORCE_INLINE kt_setzero_p<256,float>   (void) noexcept { return _mm256_setzero_ps(); };
template<> avxvector_t<256,double>  KT_FORCE_INLINE kt_setzero_p<256,double>  (void) noexcept { return _mm256_setzero_pd(); };
template<> avxvector_t<256,cfloat>  KT_FORCE_INLINE kt_setzero_p<256,cfloat>  (void) noexcept { return _mm256_setzero_ps(); };
template<> avxvector_t<256,cdouble> KT_FORCE_INLINE kt_setzero_p<256,cdouble> (void) noexcept { return _mm256_setzero_pd(); };
#ifdef USE_AVX512
template<> avxvector_t<512,float>   KT_FORCE_INLINE kt_setzero_p<512,float>   (void) noexcept { return _mm512_setzero_ps(); };
template<> avxvector_t<512,double>  KT_FORCE_INLINE kt_setzero_p<512,double>  (void) noexcept { return _mm512_setzero_pd(); };
template<> avxvector_t<512,cfloat>  KT_FORCE_INLINE kt_setzero_p<512,cfloat>  (void) noexcept { return _mm512_setzero_ps(); };
template<> avxvector_t<512,cdouble> KT_FORCE_INLINE kt_setzero_p<512,cdouble> (void) noexcept { return _mm512_setzero_pd(); };
#endif

// Dense direct (un)aligned load to AVX register
// return an avxvector with the loaded content.
//
// Example: `avxvector_t<256,double> v = kt_loadu_p<256,double>(&a[7])` is equivalent to `v = _mm256_loadu_pd(&a[7])`
template<int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ,SUF> kt_loadu_p(const SUF *) noexcept;
template<> avxvector_t<256,float>   KT_FORCE_INLINE kt_loadu_p<256,float>   (const float *a)   noexcept { return _mm256_loadu_ps(a); };
template<> avxvector_t<256,double>  KT_FORCE_INLINE kt_loadu_p<256,double>  (const double *a)  noexcept { return _mm256_loadu_pd(a); };
template<> avxvector_t<256,cfloat>  KT_FORCE_INLINE kt_loadu_p<256,cfloat>  (const cfloat *a)  noexcept { return _mm256_loadu_ps(reinterpret_cast<const float*>(a));  };
template<> avxvector_t<256,cdouble> KT_FORCE_INLINE kt_loadu_p<256,cdouble> (const cdouble *a) noexcept { return _mm256_loadu_pd(reinterpret_cast<const double*>(a)); };
#ifdef USE_AVX512
template<> avxvector_t<512,float>   KT_FORCE_INLINE kt_loadu_p<512,float>   (const float *a)   noexcept { return _mm512_loadu_ps(a); };
template<> avxvector_t<512,double>  KT_FORCE_INLINE kt_loadu_p<512,double>  (const double *a)  noexcept { return _mm512_loadu_pd(a); };
template<> avxvector_t<512,cfloat>  KT_FORCE_INLINE kt_loadu_p<512,cfloat>  (const cfloat *a)  noexcept { return _mm512_loadu_ps(reinterpret_cast<const float*>(a));  };
template<> avxvector_t<512,cdouble> KT_FORCE_INLINE kt_loadu_p<512,cdouble> (const cdouble *a) noexcept { return _mm512_loadu_pd(reinterpret_cast<const double*>(a)); };
#endif
template<int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ,SUF> kt_load_p(const SUF *) noexcept;
template<> avxvector_t<256,float>   KT_FORCE_INLINE kt_load_p<256,float>    (const float *a)   noexcept { return _mm256_load_ps(a); };
template<> avxvector_t<256,double>  KT_FORCE_INLINE kt_load_p<256,double>   (const double *a)  noexcept { return _mm256_load_pd(a); };
template<> avxvector_t<256,cfloat>  KT_FORCE_INLINE kt_load_p<256,cfloat>   (const cfloat *a)  noexcept { return _mm256_load_ps(reinterpret_cast<const float*>(a));  };
template<> avxvector_t<256,cdouble> KT_FORCE_INLINE kt_load_p<256,cdouble>  (const cdouble *a) noexcept { return _mm256_load_pd(reinterpret_cast<const double*>(a)); };
#ifdef USE_AVX512
template<> avxvector_t<512,float>   KT_FORCE_INLINE kt_load_p<512,float>    (const float *a)   noexcept { return _mm512_load_ps(a); };
template<> avxvector_t<512,double>  KT_FORCE_INLINE kt_load_p<512,double>   (const double *a)  noexcept { return _mm512_load_pd(a); };
template<> avxvector_t<512,cfloat>  KT_FORCE_INLINE kt_load_p<512,cfloat>   (const cfloat *a)  noexcept { return _mm512_load_ps(reinterpret_cast<const float*>(a));  };
template<> avxvector_t<512,cdouble> KT_FORCE_INLINE kt_load_p<512,cdouble>  (const cdouble *a) noexcept { return _mm512_load_pd(reinterpret_cast<const double*>(a)); };
#endif

// Fill vector with a scalar value
// return an avxvector filled with the same scalar value.
//
// Example `avxvector_t<512, double> v = kt_set1_p<512, double>(x)` is equivalent to `v = _mm512_set1_pd(x)`
template<int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ,SUF> kt_set1_p(const SUF ) noexcept;
template<> avxvector_t<256,float>  KT_FORCE_INLINE kt_set1_p<256,float>  (const float x) noexcept  { return _mm256_set1_ps(x); };
template<> avxvector_t<256,double> KT_FORCE_INLINE kt_set1_p<256,double> (const double x) noexcept { return _mm256_set1_pd(x); };
// complex specialization requires intermediate representation
template<> avxvector_t<256,cfloat>  KT_FORCE_INLINE kt_set1_p<256,cfloat>  (const cfloat x) noexcept  {
    const float r = std::real(x);
    const float i = std::imag(x);
    // Note that loading is end -> start <=> [d c b a] <=> [i1, r1, i0, r0]
    return _mm256_set_ps(i,r,i,r,i,r,i,r);
};
template<> avxvector_t<256,cdouble> KT_FORCE_INLINE kt_set1_p<256,cdouble> (const cdouble x) noexcept {
    const double r = std::real(x);
    const double i = std::imag(x);
    return _mm256_set_pd(i,r,i,r);
};

#ifdef USE_AVX512
template<> avxvector_t<512,float>  KT_FORCE_INLINE kt_set1_p<512,float>  (const float x) noexcept  { return _mm512_set1_ps(x); };
template<> avxvector_t<512,double> KT_FORCE_INLINE kt_set1_p<512,double> (const double x) noexcept { return _mm512_set1_pd(x); };
template<> avxvector_t<512,cfloat>  KT_FORCE_INLINE kt_set1_p<512,cfloat>  (const cfloat x) noexcept  {
    const float r = std::real(x);
    const float i = std::imag(x);
    return _mm512_set_ps(i,r,i,r,i,r,i,r,i,r,i,r,i,r,i,r);
};
template<> avxvector_t<512,cdouble>  KT_FORCE_INLINE kt_set1_p<512,cdouble>  (const cdouble x) noexcept  {
    const double r = std::real(x);
    const double i = std::imag(x);
    return _mm512_set_pd(i,r,i,r,i,r,i,r);
};
#endif

// Unaligned set (load) to AVX register with indirect memory access
//  - `SZ` size (in bits) of AVX vector, 256 or 512
//  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
//  - `v` dense array for loading the data
//  - `b` map address within range of `v`
// return an avxvector with the loaded data.
//
// Example: `kt_set_p<256, double>(v, b)` expands to _mm256_set_pd(v[*(b+3)],v[*(b+2)],v[*(b+1)],v[*(b+0)])
template <int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ, SUF> kt_set_p(const SUF *v, const kt_addr_t * b) noexcept {
    if constexpr(kt_is_same<256, SZ, double, SUF>()) {
        return _mm256_set_pd(v[*(b+3U)], v[*(b+2U)], v[*(b+1U)], v[*(b+0U)]);
    }
    else if constexpr(kt_is_same<256, SZ, float, SUF>()) {
        return _mm256_set_ps(v[*(b+7U)], v[*(b+6U)], v[*(b+5U)], v[*(b+4U)],
                             v[*(b+3U)], v[*(b+2U)], v[*(b+1U)], v[*(b+0U)]);
    }
    else if constexpr(kt_is_same<256, SZ, cdouble, SUF>()) {
        const double * vv = reinterpret_cast<const double*>(v);
        return _mm256_set_pd(vv[2U*(*(b+1U))+1U], vv[2U*(*(b+1U))+0U],
                             vv[2U*(*(b+0U))+1U], vv[2U*(*(b+0U))+0U]);
    } else if constexpr(kt_is_same<256, SZ, cfloat, SUF>()) {
        const float * vv = reinterpret_cast<const float*>(v);
        return _mm256_set_ps(vv[2U*(*(b+3U))+1U], vv[2U*(*(b+3U))+0U],
                             vv[2U*(*(b+2U))+1U], vv[2U*(*(b+2U))+0U],
                             vv[2U*(*(b+1U))+1U], vv[2U*(*(b+1U))+0U],
                             vv[2U*(*(b+0U))+1U], vv[2U*(*(b+0U))+0U]);
    }
#ifdef USE_AVX512
    else if constexpr(kt_is_same<512, SZ, double, SUF>()) {
        return _mm512_set_pd(v[*(b+7U)], v[*(b+6U)], v[*(b+5U)], v[*(b+4U)],
                             v[*(b+3U)], v[*(b+2U)], v[*(b+1U)], v[*(b+0U)]);
    }
    else if constexpr(kt_is_same<512, SZ, float, SUF>()) {
        return _mm512_set_ps(v[*(b+15U)],v[*(b+14U)],v[*(b+13U)],v[*(b+12U)],
                             v[*(b+11U)],v[*(b+10U)],v[*(b+9U)], v[*(b+8U)],
                             v[*(b+7U)], v[*(b+6U)], v[*(b+5U)], v[*(b+4U)],
                             v[*(b+3U)], v[*(b+2U)], v[*(b+1U)], v[*(b+0U)]);
    }
    else if constexpr(kt_is_same<512, SZ, cdouble, SUF>()) {
        const double * vv = reinterpret_cast<const double*>(v);
        return _mm512_set_pd(vv[2U*(*(b+3U))+1U], vv[2U*(*(b+3U))+0U],
                             vv[2U*(*(b+2U))+1U], vv[2U*(*(b+2U))+0U],
                             vv[2U*(*(b+1U))+1U], vv[2U*(*(b+1U))+0U],
                             vv[2U*(*(b+0U))+1U], vv[2U*(*(b+0U))+0U]);
    }
    else if constexpr(kt_is_same<512, SZ, cfloat, SUF>()) {
        const float * vv = reinterpret_cast<const float*>(v);
        return _mm512_set_ps(vv[2U*(*(b+7U))+1U], vv[2U*(*(b+7U))+0U],
                             vv[2U*(*(b+6U))+1U], vv[2U*(*(b+6U))+0U],
                             vv[2U*(*(b+5U))+1U], vv[2U*(*(b+5U))+0U],
                             vv[2U*(*(b+4U))+1U], vv[2U*(*(b+4U))+0U],
                             vv[2U*(*(b+3U))+1U], vv[2U*(*(b+3U))+0U],
                             vv[2U*(*(b+2U))+1U], vv[2U*(*(b+2U))+0U],
                             vv[2U*(*(b+1U))+1U], vv[2U*(*(b+1U))+0U],
                             vv[2U*(*(b+0U))+1U], vv[2U*(*(b+0U))+0U]);
    }
#endif
// if no match then compiler will complain about not returning
};

// Unaligned load to AVX register with zero mask direct memory model.
// Copies `L` elements from `v` and pads with zero the rest of AVX vector
//  - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
//  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
//  - `EXT` type of kt_avxext to use, i.e., AVX, AVX512F, ...
//  - `L` number of elements from `v` to copy
//  - `v` dense array for loading the data
//  - `b` delta address within `v`
// return an avxvector with the loaded data.
//
// Example: `kt_maskz_set_p<256, float, AVX, 3>(v, b)` expands to `_mm256_set_ps(0f, 0f, 0f, 0f, 0f, v[b+2], v[b+1], v[b+0])`
// and      `kt_maskz_set_p<256, double, AVX512VL, 3>(v, b)` expands to _mm256_maskz_loadu_pd(7, &v[b])
template<int SZ, typename SUF, kt_avxext EXT, int L> KT_FORCE_INLINE avxvector_t<SZ, SUF>  kt_maskz_set_p(const SUF *v, const kt_addr_t b) noexcept {
    if constexpr(kt_is_same<256, SZ, double, SUF>()) {
#ifdef USE_AVX512
        if constexpr(EXT & AVX512VL)
            return _mm256_maskz_loadu_pd((1<<L)-1, &v[b]);
        else
#endif
            return _mm256_set_pd(pz<SUF,L-4>(v[b+3U]), pz<SUF,L-3>(v[b+2U]), pz<SUF,L-2>(v[b+1U]), pz<SUF,L-1>(v[b+0U]));
    }
    else if constexpr(kt_is_same<256, SZ, float, SUF>()) {
#ifdef USE_AVX512
        if constexpr(EXT & AVX512VL)
            return _mm256_maskz_loadu_ps((1<<L)-1, &v[b]);
        else
#endif
            return _mm256_set_ps(pz<SUF,L-8>(v[b+7U]), pz<SUF,L-7>(v[b+6U]), pz<SUF,L-6>(v[b+5U]), pz<SUF,L-5>(v[b+4U]),
                                 pz<SUF,L-4>(v[b+3U]), pz<SUF,L-3>(v[b+2U]), pz<SUF,L-2>(v[b+1U]), pz<SUF,L-1>(v[b+0U]));
    }
    else if constexpr(kt_is_same<256, SZ, cdouble, SUF>()) {
        const double * vv = reinterpret_cast<const double*>(v);
#ifdef USE_AVX512
        if constexpr(EXT & AVX512VL)
            return _mm256_maskz_loadu_pd((1<<(2*L))-1, &vv[2U*b]);
        else
#endif
            return _mm256_set_pd(pz<double,L-2>(vv[2U*b+3U]), pz<double,L-2>(vv[2U*b+2U]),
                                 pz<double,L-1>(vv[2U*b+1U]), pz<double,L-1>(vv[2U*b+0U]));
    }
    else if constexpr(kt_is_same<256, SZ, cfloat, SUF>()) {
        const float * vv = reinterpret_cast<const float*>(v);
#ifdef USE_AVX512
        if constexpr(EXT & AVX512VL)
            return _mm256_maskz_loadu_ps((1<<(2*L))-1, &vv[2U*b]);
        else
#endif
            return _mm256_set_ps( pz<float,L-4>(vv[2*b+7U]), pz<float,L-4>(vv[2*b+6U]),
                                  pz<float,L-3>(vv[2*b+5U]), pz<float,L-3>(vv[2*b+4U]),
                                  pz<float,L-2>(vv[2*b+3U]), pz<float,L-2>(vv[2*b+2U]),
                                  pz<float,L-1>(vv[2*b+1U]), pz<float,L-1>(vv[2*b+0U]));
    }
#ifdef USE_AVX512
    else if constexpr(kt_is_same<512, SZ, double, SUF>() && (EXT & AVX512F)) {
            return _mm512_maskz_loadu_pd((1<<L)-1, &v[b]);
    }
    else if constexpr(kt_is_same<512, SZ, float, SUF>() && (EXT & AVX512F)) {
            return _mm512_maskz_loadu_ps((1<<L)-1, &v[b]);
    }
    else if constexpr(kt_is_same<512, SZ, cdouble, SUF>() && (EXT & AVX512F)) {
            const double * vv = reinterpret_cast<const double*>(v);
            return _mm512_maskz_loadu_pd((1<<(2*L))-1, &vv[2U*b]);
    }
    else if constexpr(kt_is_same<512, SZ, cfloat, SUF>() && (EXT & AVX512F)) {
            const float * vv = reinterpret_cast<const float*>(v);
            return _mm512_maskz_loadu_ps((1<<(2*L))-1, &vv[2U*b]);
    }
#endif
// if no match compiler complains about not returning
};


// Unaligned load to AVX register with zero mask indirect memory model.
// Copies `L` elements from `v` and pads with zero the rest of AVX vector
//  - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
//  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
//  - `EXT` type of kt_avxext to use, i.e., AVX, AVX512F, ...
//  - `L` number of elements from `v` to copy
//  - `v` dense array for loading the data
//  - `b` map address within range of `v`
// return an avxvector with the loaded data.
//
// Example: `kt_maskz_set_p<256, double, AVX, 2>(v, b)` expands to `_mm256_set_pd(0.0, 0.0, v[*(b+1)], v[*(b+0)])`
template<int SZ, typename SUF, kt_avxext, int L> KT_FORCE_INLINE avxvector_t<SZ, SUF>  kt_maskz_set_p(const SUF *v, const kt_addr_t *b) noexcept {
    if constexpr(kt_is_same<256, SZ, double, SUF>()) {
        return _mm256_set_pd(pz<SUF,L-4>(v[*(b+3U)]), pz<SUF,L-3>(v[*(b+2U)]), pz<SUF,L-2>(v[*(b+1U)]), pz<SUF,L-1>(v[*(b+0U)]));
    }
    else if constexpr(kt_is_same<256, SZ, float, SUF>()) {
        return _mm256_set_ps(pz<SUF,L-8>(v[*(b+7U)]), pz<SUF,L-7>(v[*(b+6U)]), pz<SUF,L-6>(v[*(b+5U)]), pz<SUF,L-5>(v[*(b+4U)]),
                             pz<SUF,L-4>(v[*(b+3U)]), pz<SUF,L-3>(v[*(b+2U)]), pz<SUF,L-2>(v[*(b+1U)]), pz<SUF,L-1>(v[*(b+0U)]));
    }
    else if constexpr(kt_is_same<256, SZ, cdouble, SUF>()) {
        const double * vv = reinterpret_cast<const double*>(v);
        return _mm256_set_pd(pz<double,L-2>(vv[2U*(*(b+1U))+1U]), pz<double,L-2>(vv[2U*(*(b+1U))+0U]),
                             pz<double,L-1>(vv[2U*(*(b+0U))+1U]), pz<double,L-1>(vv[2U*(*(b+0U))+0U]));
    }
    else if constexpr(kt_is_same<256, SZ, cfloat, SUF>()) {
        const float * vv = reinterpret_cast<const float*>(v);
        return _mm256_set_ps(pz<float,L-4>(vv[2U*(*(b+3U))+1U]), pz<float,L-4>(vv[2U*(*(b+3U))+0U]),
                             pz<float,L-3>(vv[2U*(*(b+2U))+1U]), pz<float,L-3>(vv[2U*(*(b+2U))+0U]),
                             pz<float,L-2>(vv[2U*(*(b+1U))+1U]), pz<float,L-2>(vv[2U*(*(b+1U))+0U]),
                             pz<float,L-1>(vv[2U*(*(b+0U))+1U]), pz<float,L-1>(vv[2U*(*(b+0U))+0U]));
    }
#ifdef USE_AVX512
    else if constexpr(kt_is_same<512, SZ, double, SUF>()) {
        return _mm512_set_pd(pz<SUF,L-8>(v[*(b+7U)]), pz<SUF,L-7>(v[*(b+6U)]), pz<SUF,L-6>(v[*(b+5U)]), pz<SUF,L-5>(v[*(b+4U)]),
                             pz<SUF,L-4>(v[*(b+3U)]), pz<SUF,L-3>(v[*(b+2U)]), pz<SUF,L-2>(v[*(b+1U)]), pz<SUF,L-1>(v[*(b+0U)]));
    }
    else if constexpr(kt_is_same<512, SZ, float, SUF>()) {
        return _mm512_set_ps(pz<SUF,L-16>(v[*(b+15U)]), pz<SUF,L-15>(v[*(b+14U)]), pz<SUF,L-14>(v[*(b+13U)]), pz<SUF,L-13>(v[*(b+12U)]),
                             pz<SUF,L-12>(v[*(b+11U)]), pz<SUF,L-11>(v[*(b+10U)]), pz<SUF,L-10>(v[*(b+9U)]),  pz<SUF,L-9> (v[*(b+8U)]),
                             pz<SUF,L-8> (v[*(b+7U)]),  pz<SUF,L-7> (v[*(b+6U)]),  pz<SUF,L-6> (v[*(b+5U)]),  pz<SUF,L-5> (v[*(b+4U)]),
                             pz<SUF,L-4> (v[*(b+3U)]),  pz<SUF,L-3> (v[*(b+2U)]),  pz<SUF,L-2> (v[*(b+1U)]),  pz<SUF,L-1> (v[*(b+0U)]));
    }
    else if constexpr(kt_is_same<512, SZ, cdouble, SUF>()) {
        const double * vv = reinterpret_cast<const double*>(v);
        return _mm512_set_pd(pz<double,L-4>(vv[2U*(*(b+3U))+1U]), pz<double,L-4>(vv[2U*(*(b+3U))+0U]),
                             pz<double,L-3>(vv[2U*(*(b+2U))+1U]), pz<double,L-3>(vv[2U*(*(b+2U))+0U]),
                             pz<double,L-2>(vv[2U*(*(b+1U))+1U]), pz<double,L-2>(vv[2U*(*(b+1U))+0U]),
                             pz<double,L-1>(vv[2U*(*(b+0U))+1U]), pz<double,L-1>(vv[2U*(*(b+0U))+0U]));
    }
    else if constexpr(kt_is_same<512, SZ, cfloat, SUF>()) {
        const float * vv = reinterpret_cast<const float*>(v);
        return _mm512_set_ps(pz<float,L-8>(vv[2U*(*(b+7U))+1U]), pz<float,L-8>(vv[2U*(*(b+7U))+0U]),
                             pz<float,L-7>(vv[2U*(*(b+6U))+1U]), pz<float,L-7>(vv[2U*(*(b+6U))+0U]),
                             pz<float,L-6>(vv[2U*(*(b+5U))+1U]), pz<float,L-6>(vv[2U*(*(b+5U))+0U]),
                             pz<float,L-5>(vv[2U*(*(b+4U))+1U]), pz<float,L-5>(vv[2U*(*(b+4U))+0U]),
                             pz<float,L-4>(vv[2U*(*(b+3U))+1U]), pz<float,L-4>(vv[2U*(*(b+3U))+0U]),
                             pz<float,L-3>(vv[2U*(*(b+2U))+1U]), pz<float,L-3>(vv[2U*(*(b+2U))+0U]),
                             pz<float,L-2>(vv[2U*(*(b+1U))+1U]), pz<float,L-2>(vv[2U*(*(b+1U))+0U]),
                             pz<float,L-1>(vv[2U*(*(b+0U))+1U]), pz<float,L-1>(vv[2U*(*(b+0U))+0U]));
    }
#endif
// if no match compiler complains about not returning
};


// Scatter kernel
// TODO FIXME: Placeholder
template<int SZ, typename SUF> KT_FORCE_INLINE void kt_scatter_p(const avxvector_t<SZ,SUF> a, SUF *v, const kt_addr_t *b) noexcept {
    const SUF * acast = reinterpret_cast<const SUF*>(&a);
    for(size_t k = 0; k < tsz_v<SZ,SUF>; k++)
        v[b[k]] = acast[k];
}

// Vector addition of two AVX registers.
//  - `a` avxvector
//  - `b` avxvector
// return an avxvector with `a` + `b` elementwise.
//
// Example: `avxvector_t<256, double> c = kt_add_p(a, b)` is equivalent to `__256d c = _mm256_add_pd(a, b)`
KT_FORCE_INLINE avxvector_t<256,float>   kt_add_p(const avxvector_t<256,float>   a, const avxvector_t<256,float>   b) noexcept { return _mm256_add_ps(a, b); };
KT_FORCE_INLINE avxvector_t<256,double>  kt_add_p(const avxvector_t<256,double>  a, const avxvector_t<256,double>  b) noexcept { return _mm256_add_pd(a, b); };
// Note that these same ones work also for avxvector_t<256,cfloat> and cdouble

#ifdef USE_AVX512
KT_FORCE_INLINE avxvector_t<512,float>   kt_add_p(const avxvector_t<512,float>   a, const avxvector_t<512,float>   b) noexcept { return _mm512_add_ps(a, b); };
KT_FORCE_INLINE avxvector_t<512,double>  kt_add_p(const avxvector_t<512,double>  a, const avxvector_t<512,double>  b) noexcept { return _mm512_add_pd(a, b); };
#endif

// Vector product of two AVX registers.
//  - `a` avxvector
//  - `b` avxvector
// return an avxvector with `a` * `b` elementwise.
// Example: `avxvector_t<256, double> c = kt_mul_p(a, b)` is equivalent to `__256d c = _mm256_mul_pd(a, b)`
#ifdef KT_MUL_OVERLOAD
KT_FORCE_INLINE avxvector_t<256,float>  kt_mul_p(const avxvector_t<256,float>  a, const avxvector_t<256,float>  b) noexcept { return _mm256_mul_ps(a, b); };
KT_FORCE_INLINE avxvector_t<256,double> kt_mul_p(const avxvector_t<256,double> a, const avxvector_t<256,double> b) noexcept { return _mm256_mul_pd(a, b); };
#ifdef USE_AVX512
KT_FORCE_INLINE avxvector_t<512,float>  kt_mul_p(const avxvector_t<512,float>  a, const avxvector_t<512,float>  b) noexcept { return _mm512_mul_ps(a, b); };
KT_FORCE_INLINE avxvector_t<512,double> kt_mul_p(const avxvector_t<512,double> a, const avxvector_t<512,double> b) noexcept { return _mm512_mul_pd(a, b); };
#endif
#else
// Templated version of kt_mul_p to support complex arithmetics
template<int SZ, typename SUF> avxvector_t<SZ,SUF> KT_FORCE_INLINE kt_mul_p(const avxvector_t<SZ,SUF> a, const avxvector_t<SZ,SUF> b) noexcept;
template<> avxvector_t<256,float>   KT_FORCE_INLINE kt_mul_p<256,float>   (const avxvector_t<256,float>  a, const avxvector_t<256,float>  b) noexcept { return _mm256_mul_ps(a, b); };
template<> avxvector_t<256,double>  KT_FORCE_INLINE kt_mul_p<256,double>  (const avxvector_t<256,double> a, const avxvector_t<256,double> b) noexcept { return _mm256_mul_pd(a, b); };
// Complex arithmetic
template<> avxvector_t<256,cfloat>  KT_FORCE_INLINE kt_mul_p<256,cfloat>  (const avxvector_t<256,cfloat>  a, const avxvector_t<256,cfloat>  b) noexcept {
    // input vectors a = (x0+iy0, x1+iy1, ...) and b = (a0+ib0, a1+ib1, ...)
    // imaginary elements in the vector: half = (y0, y0, y1, y1, ...)
    __m256 half = _mm256_movehdup_ps(a);
    // tmp = (a0*iy0, ib0*iy0, a1*iy1, ib1*iy1, ...)
    __m256 tmp = _mm256_mul_ps(half, b);
    // real elements in the vector: half = (x0, x0, x1, x1, ...)
    half = _mm256_moveldup_ps(a);
    // c = (x0*a0-b0*y0, x0*ib0+a0*iy0, x1*a1-b1*y1, x1*ib1+a1*iy1, ...)
    __m256 c = _mm256_fmaddsub_ps(half, b, _mm256_permute_ps(tmp, 0xB1));
    return c;
};
template<> avxvector_t<256,cdouble> KT_FORCE_INLINE kt_mul_p<256,cdouble> (const avxvector_t<256,cdouble> a, const avxvector_t<256,cdouble> b) noexcept {
    // input vectors a = (x0+iy0, x1+iy1) and b = (a0+ib0, a1+ib1)
    // imaginary elements in the vector: half = (y0, y0, y1, y1)
    __m256d half = _mm256_movedup_pd(_mm256_permute_pd(a, 0x5));
    // tmp = (a0*iy0, ib0*iy0, a1*iy1, ib1*iy1)
    __m256d tmp = _mm256_mul_pd(half, b);
    // real elements in the vector: half = (x0, x0, x1, x1)
    half = _mm256_movedup_pd(a);
    // c = (x0*a0-b0*y0, x0*ib0+a0*iy0, x1*a1-b1*y1, x1*ib1+a1*iy1)
    __m256d c = _mm256_fmaddsub_pd(half, b, _mm256_permute_pd(tmp, 0x5));
    return c;
};

#ifdef USE_AVX512
template<> avxvector_t<512,float>   KT_FORCE_INLINE kt_mul_p<512,float>   (const avxvector_t<512,float>  a, const avxvector_t<512,float>  b) noexcept { return _mm512_mul_ps(a, b); };
template<> avxvector_t<512,double>  KT_FORCE_INLINE kt_mul_p<512,double>  (const avxvector_t<512,double> a, const avxvector_t<512,double> b) noexcept { return _mm512_mul_pd(a, b); };
template<> avxvector_t<512,cfloat> KT_FORCE_INLINE kt_mul_p<512,cfloat> (const avxvector_t<512,cfloat> a, const avxvector_t<512,cfloat> b) noexcept {
    // input vectors a = (x0+iy0, x1+iy1, ...) and b = (a0+ib0, a1+ib1, ...)
    // imaginary elements in the vector: half = (y0, y0, y1, y1, ...)
    __m512 half = _mm512_movehdup_ps(a);
    // tmp (a0*iy0, ib0*iy0, a1*iy1, ib1*iy1, ...)
    __m512 tmp = _mm512_mul_ps(half, b);
    // real elements in the vector: half = (x0, x0, x1, x1, ...)
    half = _mm512_moveldup_ps(a);
    // c = (x0*a0-b0*y0, x0*ib0+a0*iy0, x1*a1-b1*y1, x1*ib1+a1*iy1, ...)
    __m512 c = _mm512_fmaddsub_ps(half, b, _mm512_permute_ps(tmp, 0xB1));
    return c;
};
template<> avxvector_t<512,cdouble> KT_FORCE_INLINE kt_mul_p<512,cdouble> (const avxvector_t<512,cdouble> a, const avxvector_t<512,cdouble> b) noexcept {
    // input vectors a = (x0+iy0, x1+iy1, ...) and b = (a0+ib0, a1+ib1, ...)
    // imaginary elements in the vector: half = (y0, y0, y1, y1, ...)
    __m512d half = _mm512_movedup_pd(_mm512_permute_pd(a, 0x55));
    // tmp = (a0*iy0, ib0*iy0, a1*iy1, ib1*iy1, ...)
    __m512d tmp = _mm512_mul_pd(half, b);
    // real elements in the vector: half = (x0, x0, x1, x1, ...)
    half = _mm512_movedup_pd(a);
    // c = (x0*a0-b0*y0, x0*ib0+a0*iy0, x1*a1-b1*y1, x1*ib1+a1*iy1, ...)
    __m512d c = _mm512_fmaddsub_pd(half, b, _mm512_permute_pd(tmp, 0x55));
    return c;
};
#endif
#endif

// Vector fused multiply-add of three AVX registers.
//  - `a` avxvector
//  - `b` avxvector
//  - `c` avxvector
// return an avxvector with `a` * `b` + `c` elementwise.
// Example: `avxvector_t<256, double> d = kt_fmadd_p(a, b, c)` is equivalent to `__256d d = _mm256_mul_pd(a, b, c)`
#ifdef KT_MUL_OVERLOAD
KT_FORCE_INLINE avxvector_t<256,float>  kt_fmadd_p(const avxvector_t<256,float>  a, const avxvector_t<256,float>  b, const avxvector_t<256,float> c)  noexcept { return _mm256_fmadd_ps(a, b, c); };
KT_FORCE_INLINE avxvector_t<256,double> kt_fmadd_p(const avxvector_t<256,double> a, const avxvector_t<256,double> b, const avxvector_t<256,double> c) noexcept { return _mm256_fmadd_pd(a, b, c); };
#ifdef USE_AVX512
KT_FORCE_INLINE avxvector_t<512,float>  kt_fmadd_p(const avxvector_t<512,float>  a, const avxvector_t<512,float>  b, const avxvector_t<512,float> c)  noexcept { return _mm512_fmadd_ps(a, b, c); };
KT_FORCE_INLINE avxvector_t<512,double> kt_fmadd_p(const avxvector_t<512,double> a, const avxvector_t<512,double> b, const avxvector_t<512,double> c) noexcept { return _mm512_fmadd_pd(a, b, c); };
#endif
#else
template<int SZ, typename SUF> avxvector_t<SZ,SUF> KT_FORCE_INLINE kt_fmadd_p(const avxvector_t<SZ,SUF>      a, const avxvector_t<SZ,SUF>      b, const avxvector_t<SZ,SUF>      c) noexcept;
template<> avxvector_t<256,float>   KT_FORCE_INLINE kt_fmadd_p<256,float>    (const avxvector_t<256,float>   a, const avxvector_t<256,float>   b, const avxvector_t<256,float>   c) noexcept { return _mm256_fmadd_ps(a, b, c); };
template<> avxvector_t<256,double>  KT_FORCE_INLINE kt_fmadd_p<256,double>   (const avxvector_t<256,double>  a, const avxvector_t<256,double>  b, const avxvector_t<256,double>  c) noexcept { return _mm256_fmadd_pd(a, b, c); };
template<> avxvector_t<256,cfloat>  KT_FORCE_INLINE kt_fmadd_p<256,cfloat>   (const avxvector_t<256,cfloat>  a, const avxvector_t<256,cfloat>  b, const avxvector_t<256,cfloat>  c) noexcept { return kt_add_p(kt_mul_p<256,cfloat>(a,b), c); };
template<> avxvector_t<256,cdouble> KT_FORCE_INLINE kt_fmadd_p<256,cdouble>  (const avxvector_t<256,cdouble> a, const avxvector_t<256,cdouble> b, const avxvector_t<256,cdouble> c) noexcept { return kt_add_p(kt_mul_p<256,cdouble>(a,b), c); };

#ifdef USE_AVX512
template<> avxvector_t<512,float>   KT_FORCE_INLINE kt_fmadd_p<512,float>    (const avxvector_t<512,float>   a, const avxvector_t<512,float>   b, const avxvector_t<512,float>   c) noexcept { return _mm512_fmadd_ps(a, b, c); };
template<> avxvector_t<512,double>  KT_FORCE_INLINE kt_fmadd_p<512,double>   (const avxvector_t<512,double>  a, const avxvector_t<512,double>  b, const avxvector_t<512,double>  c) noexcept { return _mm512_fmadd_pd(a, b, c); };
template<> avxvector_t<512,cfloat>  KT_FORCE_INLINE kt_fmadd_p<512,cfloat>   (const avxvector_t<512,cfloat>  a, const avxvector_t<512,cfloat>  b, const avxvector_t<512,cfloat>  c) noexcept { return kt_add_p(kt_mul_p<512,cfloat>(a,b), c); };
template<> avxvector_t<512,cdouble> KT_FORCE_INLINE kt_fmadd_p<512,cdouble>  (const avxvector_t<512,cdouble> a, const avxvector_t<512,cdouble> b, const avxvector_t<512,cdouble> c) noexcept { return kt_add_p(kt_mul_p<512,cdouble>(a,b), c); };
#endif
#endif

// Horizontal sum (reduction) of an AVX register
//  - `v` avxvector
// return a scalar containing the horizontal sum of the elements of `v`, that is,
// `v[0] + v[1] + ... + v[N]` with `N` the appropiate vector size
template<int SZ, typename SUF> KT_FORCE_INLINE SUF  kt_hsum_p(const avxvector_t<SZ,SUF> v) noexcept;
template<> KT_FORCE_INLINE float kt_hsum_p<256,float>(avxvector_t<256,float> const v) noexcept {
    avxvector_half_t<256,float> l, h, s;
    avxvector_t<256,float> w = _mm256_hadd_ps(v, v);
    w = _mm256_hadd_ps(w, w); // only required for float
    l = _mm256_castps256_ps128(w);
    h = _mm256_extractf128_ps(w, 1);
    s  = _mm_add_ps(l, h);
    return kt_sse_scl(s);
};
template<> KT_FORCE_INLINE double kt_hsum_p<256,double>(avxvector_t<256,double> const v) noexcept {
    avxvector_half_t<256,double> l, h, s;
    avxvector_t<256,double> w = _mm256_hadd_pd(v, v);
    l = _mm256_castpd256_pd128(w);
    h = _mm256_extractf128_pd(w, 1);
    s  = _mm_add_pd(l, h);
    return kt_sse_scl(s);
};
template<> KT_FORCE_INLINE cfloat kt_hsum_p<256,cfloat>(avxvector_t<256,cfloat> const v) noexcept {
    // input vector v = (x0+iy0, x1+iy1, x2+iy2, x3+iy3)
    __m256 res, s, tmp;
    // only the indexes mentioned in the last two places are relevant
    __m256i idx = _mm256_set_epi32(6,7,2,3,0,1,5,4);
    // upper 128-bits of each lane are not used
    // tmp = (x1, y1, ..., x3, y3, ...)
    tmp = _mm256_permute_ps(v, 0b1110);
    // elements at indexes 0, 1, 4 and 5 hold the intermediate results
    res = _mm256_add_ps(v, tmp);
    // move elements in the 4th and 5th indexes to the 0th and 1st indexes
    tmp = _mm256_permutevar8x32_ps(res, idx);
    s = _mm256_add_ps(res, tmp);
    cfloat result = {s[0], s[1]};
    return result;
};
template<> KT_FORCE_INLINE cdouble kt_hsum_p<256,cdouble>(avxvector_t<256,cdouble> const v) noexcept {
    // input vector v = (x0+iy0, x1+iy1)
    __m256d s, tmp;
    // tmp = (x2, y2, x1, y1) upper half not relevant
    tmp = _mm256_permute4x64_pd(v, 0b1110);
    // s = (x1+x2, y1+y2) upper half not relevant
    s = _mm256_add_pd(v, tmp);
    cdouble result = {s[0], s[1]};
    return result;
};
#ifdef USE_AVX512
template<> KT_FORCE_INLINE float  kt_hsum_p<512,float> (avxvector_t<512,float> const v)  noexcept { return _mm512_reduce_add_ps(v); };
template<> KT_FORCE_INLINE double kt_hsum_p<512,double>(avxvector_t<512,double> const v) noexcept { return _mm512_reduce_add_pd(v); };
template<> KT_FORCE_INLINE cfloat kt_hsum_p<512,cfloat>(avxvector_t<512,cfloat> const v) noexcept {
    // input vector v = (x0+iy0, x1+iy1, ..., x7+iy7)
    __m512 res, s, tmp;
    __m512i idx = _mm512_set_epi32(15,14,11,10,9,8,13,12,7,6,3,2,1,0,5,4);
    // upper 128-bits of each 256-bit lane are not used
    // tmp = (x1, y1, ..., x3, y3, ..., x5, y5, ..., x7, y7, ...)
    tmp = _mm512_permute_ps(v, 0b11101110);
    // element at indexes 0, 1, 4, 5, 8, 9, 12 and 13 hold the intermediate results
    // res = (x0+x1, y0+y1, ..., x2+x3, y2+y3, ...)
    res = _mm512_add_ps(v, tmp);
    // only indexes mentioned in the last two places of each 256-bit lane are relevant
    tmp = _mm512_permutexvar_ps(idx, res);
    // s = (x0+x1+x2+x3, y0+y1+y2+y3, ..., x4+x5+x6+x7, y4+y5+y6+y7, ...)
    s = _mm512_add_ps(res, tmp);
    // only the indexes mentioned in the last two places are relevant
    idx = _mm512_set_epi32(15,14,13,12,11,10,7,6,5,4,3,2,1,0,9,8);
    tmp = _mm512_permutexvar_ps(idx, s);
    s = _mm512_add_ps(s, tmp);
    cfloat result = {s[0], s[1]};
    return result;
};
template<> KT_FORCE_INLINE cdouble kt_hsum_p<512,cdouble>(avxvector_t<512,cdouble> const v) noexcept {
    // input vector v = (x0+iy0, x1+iy1, x2+iy2, x3+iy3)
    // mem layout: v = (x0, y0, x1, y1, x2, y2, x3, y3)
    __m512d res, s, tmp;
    __m512i idx = _mm512_set_epi64(0,1,2,3,7,6,5,4);
    // tmp = (x1, y1, ..., x3, y3, ...) upper 128-bit of each 256-bit lane is not relevant
    tmp = _mm512_permutex_pd(v, 0b11101110);
    // res = (x0+x1, y0+y1, ..., x2+x3, y2+y3, ...)
    res = _mm512_add_pd(v, tmp);
    // tmp = (x2 + x3, y2 + y3, ...) rest is not relevant
    tmp = _mm512_permutexvar_pd(idx, res);
    // horizontal sum result is in the first 128-bits
    s = _mm512_add_pd(res, tmp);
    cdouble result = {s[0], s[1]};
    return result;
};
#endif

// Templated version of the conjugate operation
// Conjugate an AVX register
//  - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
//  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
//  - `a` avxvector
// returns `conjugate(a)` for complex types and returns `a`for real
template<int SZ, typename SUF> avxvector_t<SZ,SUF> KT_FORCE_INLINE kt_conj_p(const avxvector_t<SZ,SUF> a) noexcept {
    if constexpr (std::is_floating_point<SUF>::value) {
        return a;
    }
    else if constexpr ( kt_is_same<256, SZ, cfloat, SUF>() ) {
        __m256  mask = _mm256_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
        __m256 res = _mm256_xor_ps(mask, a);
        return res;
    }
    else if constexpr ( kt_is_same<256, SZ, cdouble, SUF>() ) {
        __m256d  mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
        __m256d res = _mm256_xor_pd(mask, a);
        return res;
    }
#ifdef USE_AVX512
    else if constexpr ( kt_is_same<512, SZ, cfloat, SUF>() ) {
        __m512  mask = _mm512_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
        __m512 res = _mm512_xor_ps(mask, a);
        return res;
    }
    else if constexpr ( kt_is_same<512, SZ, cdouble, SUF>() ) {
        __m512d  mask = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
        __m512d res = _mm512_xor_pd(mask, a);
        return res;
    }
#endif
}

// Level 1 micro kernels
// These micro kernels depend solely on LEVEL 0 micro kernels
// =======================================================================

// Dot-product of two AVX registers
//  - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
//  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
//  - `a` avxvector
//  - `b` avxvector
// returns a scalar containing the dot-product of a and b, <a,b>
template<int SZ, typename SUF> KT_FORCE_INLINE SUF kt_dot_p(const avxvector_t<SZ,SUF> a, const avxvector_t<SZ,SUF> b) noexcept {
    avxvector_t<SZ,SUF> c = kt_mul_p<SZ,SUF>(a, b);
    return kt_hsum_p<SZ, SUF>(c);
};

// Dot-product of two AVX registers (convenience callers)
//  - `a` avxvector of type FLOAT or DOUBLE only
//  - `b` avxvector of type FLOAT or DOUBLE only
// returns a scalar containing the dot-product of a and b, <a,b>
// Note: these wrappers should not be used with avxvectors of type cdouble or cfloat.
KT_FORCE_INLINE double kt_dot_p(avxvector_t<256,double> const a, avxvector_t<256,double> const b) noexcept {
    return kt_dot_p<256,double>(a, b);
}
KT_FORCE_INLINE float kt_dot_p(avxvector_t<256,float> const a, avxvector_t<256,float> const b) noexcept {
    return kt_dot_p<256,float>(a, b);
}
#ifdef USE_AVX512
KT_FORCE_INLINE double kt_dot_p(avxvector_t<512,double> const a, avxvector_t<512,double> const b) noexcept {
    return kt_dot_p<512,double>(a, b);
}
KT_FORCE_INLINE float kt_dot_p(avxvector_t<512,float> const a, avxvector_t<512,float> const b) noexcept {
    return kt_dot_p<512,float>(a, b);
}
#endif

// Templated version of the conjugate dot operation
// Conjugate dot-product of two AVX registers
//  - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
//  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
//  - `a` avxvector
//  - `b` avxvector
// if `a` and `b` are real then returns the dot-product of both, if complex then returns the dot-product of `a` and `conjugate(b)`
template<int SZ, typename SUF> SUF KT_FORCE_INLINE kt_cdot_p(const avxvector_t<SZ,SUF> a, const avxvector_t<SZ,SUF> b) noexcept {
    if constexpr (std::is_floating_point<SUF>::value){
        return kt_dot_p<SZ, SUF>(a, b);
    }
    else {
        return kt_dot_p<SZ,SUF>(a, kt_conj_p<SZ,SUF>(b));
    }
};

}

// Undefine...
#undef kt_addr_t

#endif // AOCLSPARSE_KERNEL_TEMPLATES_T_HPP
