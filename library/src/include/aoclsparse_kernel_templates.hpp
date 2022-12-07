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
 * For TRSV kernels using KT see library/src/level2/aoclsparse_trsv.hpp
 */

// This file is not polished
// clang-format off

#ifndef AOCLSPARSE_KERNEL_TEMPLATES_T_HPP
#define AOCLSPARSE_KERNEL_TEMPLATES_T_HPP
#include <immintrin.h>
#include <type_traits>
#include <limits>
#include <cmath>

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
// * **ANY** All targets
// * **AVX2** All extensions up to AVX2: AVX, FMA, ...
// * **AVX512F** AVX512 Foundations
// * **AVX512VL** Use zero-masked instrinsics, ...
// * **AVX512DQ** ...
// Each extension needs to be a superset of the previous, not ideal...
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

// helper template type for avx full vectors
template <int, typename> struct avxvector { };
template <> struct avxvector<256, double> { using type      = __m256d;             // Vector type
                                            using half_type = __m128d;             // Associated "half" length vector type
                                            static constexpr size_t value = 4;     // Vector length, packet size
                                            static constexpr size_t half = 4 >> 1; // Length of half vector
                                            constexpr operator size_t() const noexcept { return value; } }; // Convenience operator to return vector length
template <> struct avxvector<256, float>  { using type = __m256;
                                            using half_type = __m128;
                                            static constexpr size_t value = 8;
                                            static constexpr size_t half = 8 >> 1;
                                            constexpr operator size_t() const noexcept { return value; } };
#ifdef USE_AVX512
template <> struct avxvector<512, double> { using type = __m512d;
                                            using half_type = __m256d;
                                            static constexpr size_t value = 8;
                                            static constexpr size_t half = 8 >> 1;
                                            constexpr operator size_t() const noexcept { return value; } };
template <> struct avxvector<512, float>  { using type = __m512;
                                            using half_type = __m256;
                                            static constexpr size_t value = 16;
                                            static constexpr size_t half = 16 >> 1;
                                            constexpr operator size_t() const noexcept { return value; } };
#endif
// helper template type for avx vectors
template <int SZ, typename SUF> using avxvector_t = typename avxvector<SZ, SUF>::type;
// helper template type for "half" avx vectors
template <int SZ, typename SUF> using avxvector_half_t = typename avxvector<SZ, SUF>::half_type;
// helper template value (length) for avx vectors (pack size)
template <int SZ, typename SUF> inline constexpr size_t avxvector_v = avxvector<SZ, SUF>::value;
template <int SZ, typename SUF> inline constexpr size_t psz_v = avxvector<SZ, SUF>::value;
// helper template value (length) for "half" avx vectors (half pack size)
template <int SZ, typename SUF> inline constexpr size_t avxvector_half_v = avxvector<SZ, SUF>::half;
template <int SZ, typename SUF> inline constexpr size_t hsz_v = avxvector<SZ, SUF>::half;

// Level 0 micro kernels
// kernels at this level expands to IMM intrinsic instructions
// =======================================================================

// Zero out an AVX register
// return an avxvector filled with zeroes.
//
// Example: `avxvector<256,float> v = kt_setzero_p<256,float>() is equivalent to `v = _mm256_setzero_ps()`
template<int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ,SUF> kt_setzero_p(void)noexcept;
template<> avxvector_t<256,float>  KT_FORCE_INLINE kt_setzero_p<256,float>  (void) noexcept { return _mm256_setzero_ps(); };
template<> avxvector_t<256,double> KT_FORCE_INLINE kt_setzero_p<256,double> (void) noexcept { return _mm256_setzero_pd(); };
#ifdef USE_AVX512
template<> avxvector_t<512,float>  KT_FORCE_INLINE kt_setzero_p<512,float>  (void) noexcept { return _mm512_setzero_ps(); };
template<> avxvector_t<512,double> KT_FORCE_INLINE kt_setzero_p<512,double> (void) noexcept { return _mm512_setzero_pd(); };
#endif

// Dense direct (un)aligned load to AVX register
// return an avxvector with the loaded content.
//
// Example: `avxvector_t<256,double> v = kt_loadu_p<256,double>(&a[7])` is equivalent to `v = _mm256_loadu_pd(&a[7])`
template<int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ,SUF> kt_loadu_p(const SUF *) noexcept;
template<> avxvector_t<256,float>  KT_FORCE_INLINE kt_loadu_p<256,float>  (const float *a) noexcept  { return _mm256_loadu_ps(a); };
template<> avxvector_t<256,double> KT_FORCE_INLINE kt_loadu_p<256,double> (const double *a) noexcept { return _mm256_loadu_pd(a); };
#ifdef USE_AVX512
template<> avxvector_t<512,float>  KT_FORCE_INLINE kt_loadu_p<512,float>  (const float *a) noexcept  { return _mm512_loadu_ps(a); };
template<> avxvector_t<512,double> KT_FORCE_INLINE kt_loadu_p<512,double> (const double *a) noexcept { return _mm512_loadu_pd(a); };
#endif
template<int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ,SUF> kt_load_p(const SUF *) noexcept;
template<> avxvector_t<256,float>  KT_FORCE_INLINE kt_load_p<256,float>  (const float *a) noexcept  { return _mm256_loadu_ps(a); };
template<> avxvector_t<256,double> KT_FORCE_INLINE kt_load_p<256,double> (const double *a) noexcept { return _mm256_loadu_pd(a); };
#ifdef USE_AVX512
template<> avxvector_t<512,float>  KT_FORCE_INLINE kt_load_p<512,float>  (const float *a) noexcept  { return _mm512_loadu_ps(a); };
template<> avxvector_t<512,double> KT_FORCE_INLINE kt_load_p<512,double> (const double *a) noexcept { return _mm512_loadu_pd(a); };
#endif

// Fill vector with a scalar value
// return an avxvector filled with the same scalar value.
//
// Example `avxvector_t<512, double> v = kt_set1_p<512, double>(x)` is equivalent to `_mm512_set1_pd(x)`
template<int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ,SUF> kt_set1_p(const SUF ) noexcept;
template<> avxvector_t<256,float>  KT_FORCE_INLINE kt_set1_p<256,float>  (const float x) noexcept  { return _mm256_set1_ps(x); };
template<> avxvector_t<256,double> KT_FORCE_INLINE kt_set1_p<256,double> (const double x) noexcept { return _mm256_set1_pd(x); };
#ifdef USE_AVX512
template<> avxvector_t<512,float>  KT_FORCE_INLINE kt_set1_p<512,float>  (const float x) noexcept  { return _mm512_set1_ps(x); };
template<> avxvector_t<512,double> KT_FORCE_INLINE kt_set1_p<512,double> (const double x) noexcept { return _mm512_set1_pd(x); };
#endif

// Unaligned set (load) to AVX register with indirect memory access
//  - `SZ` size (in bits) of AVX vector, 256 or 512
//  - `SUF` suffix of working type, i.e., `double` or `float`
//  - `v` dense array for loading the data
//  - `b` map address within range of `v`
// return an avxvector with the loaded data.
//
// Example: `kt_set_p<256, double>(v, b)` expands to _mm256_set_pd(v[*(b+3)],v[*(b+2)],v[*(b+1)],v[*(b+0)])
template <int SZ, typename SUF> KT_FORCE_INLINE avxvector_t<SZ, SUF> kt_set_p(const SUF *v, const kt_addr_t * b) noexcept {
    if constexpr(SZ==256 && std::is_same_v<SUF,double>) {
        return _mm256_set_pd(v[*(b+3U)], v[*(b+2U)], v[*(b+1U)], v[*(b+0U)]);
    }
    else if constexpr(SZ==256 && std::is_same_v<SUF,float>) {
        return _mm256_set_ps(v[*(b+7U)], v[*(b+6U)], v[*(b+5U)], v[*(b+4U)],
                             v[*(b+3U)], v[*(b+2U)], v[*(b+1U)], v[*(b+0U)]);
    }
#ifdef USE_AVX512
    else if constexpr(SZ==512 && std::is_same_v<SUF,double>) {
        return _mm512_set_pd(v[*(b+7U)], v[*(b+6U)], v[*(b+5U)], v[*(b+4U)],
                             v[*(b+3U)], v[*(b+2U)], v[*(b+1U)], v[*(b+0U)]);
    }
    else if constexpr(SZ==512 && std::is_same_v<SUF,float>) {
        return _mm512_set_ps(v[*(b+15U)],v[*(b+14U)],v[*(b+13U)],v[*(b+12U)],
                             v[*(b+11U)],v[*(b+10U)],v[*(b+9U)], v[*(b+8U)],
                             v[*(b+7U)], v[*(b+6U)], v[*(b+5U)], v[*(b+4U)],
                             v[*(b+3U)], v[*(b+2U)], v[*(b+1U)], v[*(b+0U)]);
    }
#endif
// if no match then compiler will complain about not returning
};

// Unaligned load to AVX register with zero mask direct memory model.
// Copies `L` elements from `v` and pads with zero the rest of AVX vector
//  - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
//  - `SUF` suffix of working type, i.e., `double` or `float`
//  - `EXT` type of kt_avxext to use, i.e., AVX, AVX512F, ...
//  - `L` number of elements from `v` to copy
//  - `v` dense array for loading the data
//  - `b` delta address within `v`
// return an avxvector with the loaded data.
//
// Example: `kt_maskz_set_p<256, float, AVX, 3>(v, b)` expands to `_mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, v[b+2], v[b+1], v[b+0])`
// and      `kt_maskz_set_p<256, double, AVX512VL, 3>(v, b)` expands to _mm256_maskz_loadu_pd(7, &v[b])
template<int SZ, typename SUF, kt_avxext EXT, int L> KT_FORCE_INLINE avxvector_t<SZ, SUF>  kt_maskz_set_p(const SUF *v, const kt_addr_t b) noexcept {
    if constexpr(SZ==256 && std::is_same_v<SUF,double>) {
#ifdef USE_AVX512
        if constexpr(EXT & AVX512VL)
            return _mm256_maskz_loadu_pd((1<<L)-1, &v[b]);
        else
#endif
            return _mm256_set_pd(pz<SUF,L-4>(v[b+3U]), pz<SUF,L-3>(v[b+2U]), pz<SUF,L-2>(v[b+1U]), pz<SUF,L-1>(v[b+0U]));
    }
    else if constexpr(SZ==256 && std::is_same_v<SUF,float>) {
#ifdef USE_AVX512
        if constexpr(EXT & AVX512VL)
            return _mm256_maskz_loadu_ps((1<<L)-1, &v[b]);
        else
#endif
            return _mm256_set_ps(pz<SUF,L-8>(v[b+7U]), pz<SUF,L-7>(v[b+6U]), pz<SUF,L-6>(v[b+5U]), pz<SUF,L-5>(v[b+4U]),
                                 pz<SUF,L-4>(v[b+3U]), pz<SUF,L-3>(v[b+2U]), pz<SUF,L-2>(v[b+1U]), pz<SUF,L-1>(v[b+0U]));
    }
#ifdef USE_AVX512
    else if constexpr(SZ==512 && std::is_same_v<SUF,double> && (EXT & AVX512F)) {
            return _mm512_maskz_loadu_pd((1<<L)-1, &v[b]);
    }
    else if constexpr(SZ==512 && std::is_same_v<SUF,float> && (EXT & AVX512F)) {
            return _mm512_maskz_loadu_ps((1<<L)-1, &v[b]);
    }
#endif
// if no match then compiler will complain about not returning
};


// Unaligned load to AVX register with zero mask indirect memory model.
// Copies `L` elements from `v` and pads with zero the rest of AVX vector
//  - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
//  - `SUF` suffix of working type, i.e., `double` or `float`
//  - `EXT` type of kt_avxext to use, i.e., AVX, AVX512F, ...
//  - `L` number of elements from `v` to copy
//  - `v` dense array for loading the data
//  - `b` map address within range of `v`
// return an avxvector with the loaded data.
//
// Example: `kt_maskz_set_p<256, double, AVX, 2>(v, b)` expands to `_mm256_set_pd(0.0, 0.0, v[*(b+1)], v[*(b+0)])`
template<int SZ, typename SUF, kt_avxext, int L> KT_FORCE_INLINE avxvector_t<SZ, SUF>  kt_maskz_set_p(const SUF *v, const kt_addr_t *b) noexcept {
    if constexpr(SZ==256 && std::is_same_v<SUF,double>) {
        return _mm256_set_pd(pz<SUF,L-4>(v[*(b+3U)]), pz<SUF,L-3>(v[*(b+2U)]), pz<SUF,L-2>(v[*(b+1U)]), pz<SUF,L-1>(v[*(b+0U)]));
    }
    else if constexpr(SZ==256 && std::is_same_v<SUF,float>) {
        return _mm256_set_ps(pz<SUF,L-8>(v[*(b+7U)]), pz<SUF,L-7>(v[*(b+6U)]), pz<SUF,L-6>(v[*(b+5U)]), pz<SUF,L-5>(v[*(b+4U)]),
                             pz<SUF,L-4>(v[*(b+3U)]), pz<SUF,L-3>(v[*(b+2U)]), pz<SUF,L-2>(v[*(b+1U)]), pz<SUF,L-1>(v[*(b+0U)]));
    }
#ifdef USE_AVX512
    else if constexpr(SZ==512 && std::is_same_v<SUF,double>) {
        return _mm512_set_pd(pz<SUF,L-8>(v[*(b+7U)]), pz<SUF,L-7>(v[*(b+6U)]), pz<SUF,L-6>(v[*(b+5U)]), pz<SUF,L-5>(v[*(b+4U)]),
                             pz<SUF,L-4>(v[*(b+3U)]), pz<SUF,L-3>(v[*(b+2U)]), pz<SUF,L-2>(v[*(b+1U)]), pz<SUF,L-1>(v[*(b+0U)]));
    }
    else if constexpr(SZ==512 && std::is_same_v<SUF,float>) {
        return _mm512_set_ps(pz<SUF,L-16>(v[*(b+15U)]), pz<SUF,L-15>(v[*(b+14U)]), pz<SUF,L-14>(v[*(b+13U)]), pz<SUF,L-13>(v[*(b+12U)]),
            pz<SUF,L-12>(v[*(b+L-11U)]), pz<SUF,L-11>(v[*(b+L-10U)]), pz<SUF,L-10>(v[*(b+9U)]), pz<SUF,L-9>(v[*(b+8U)]),
            pz<SUF,L-8>(v[*(b+7U)]), pz<SUF,L-7>(v[*(b+6U)]), pz<SUF,L-6>(v[*(b+5U)]), pz<SUF,L-5>(v[*(b+4U)]),
            pz<SUF,L-4>(v[*(b+3U)]), pz<SUF,L-3>(v[*(b+2U)]), pz<SUF,L-2>(v[*(b+1U)]), pz<SUF,L-1>(v[*(b+0U)]));
    }
#endif
// if no match then compiler will complain about not returning
};

// Vector addition of two AVX registers.
//  - `a` avxvector
//  - `b` avxvector
// return an avxvector with `a` + `b` elementwise.
//
// Example: `avxvector_t<256, double> c = kt_add_p(a, b)` is equivalent to `__256d c = _mm256_add_pd(a, b)`
KT_FORCE_INLINE avxvector_t<256,float>  kt_add_p(const avxvector_t<256,float>  a, const avxvector_t<256,float>  b) noexcept { return _mm256_add_ps(a, b); };
KT_FORCE_INLINE avxvector_t<256,double> kt_add_p(const avxvector_t<256,double> a, const avxvector_t<256,double> b) noexcept { return _mm256_add_pd(a, b); };
#ifdef USE_AVX512
KT_FORCE_INLINE avxvector_t<512,float>  kt_add_p(const avxvector_t<512,float>  a, const avxvector_t<512,float>  b) noexcept { return _mm512_add_ps(a, b); };
KT_FORCE_INLINE avxvector_t<512,double> kt_add_p(const avxvector_t<512,double> a, const avxvector_t<512,double> b) noexcept { return _mm512_add_pd(a, b); };
#endif

// Vector product of two AVX registers.
//  - `a` avxvector
//  - `b` avxvector
// return an avxvector with `a` * `b` elementwise.
// Example: `avxvector_t<256, double> c = kt_mul_p(a, b)` is equivalent to `__256d c = _mm256_mul_pd(a, b)`
KT_FORCE_INLINE avxvector_t<256,float>  kt_mul_p(const avxvector_t<256,float>  a, const avxvector_t<256,float>  b) noexcept { return _mm256_mul_ps(a, b); };
KT_FORCE_INLINE avxvector_t<256,double> kt_mul_p(const avxvector_t<256,double> a, const avxvector_t<256,double> b) noexcept { return _mm256_mul_pd(a, b); };
#ifdef USE_AVX512
KT_FORCE_INLINE avxvector_t<512,float>  kt_mul_p(const avxvector_t<512,float>  a, const avxvector_t<512,float>  b) noexcept { return _mm512_mul_ps(a, b); };
KT_FORCE_INLINE avxvector_t<512,double> kt_mul_p(const avxvector_t<512,double> a, const avxvector_t<512,double> b) noexcept { return _mm512_mul_pd(a, b); };
#endif

// Vector fused multiply-add of three AVX registers.
//  - `a` avxvector
//  - `b` avxvector
//  - `c` avxvector
// return an avxvector with `a` * `b` + `c` elementwise.
// Example: `avxvector_t<256, double> d = kt_fmadd_p(a, b, c)` is equivalent to `__256d d = _mm256_mul_pd(a, b, c)`
KT_FORCE_INLINE avxvector_t<256,float>  kt_fmadd_p(const avxvector_t<256,float>  a, const avxvector_t<256,float>  b, const avxvector_t<256,float> c)  noexcept { return _mm256_fmadd_ps(a, b, c); };
KT_FORCE_INLINE avxvector_t<256,double> kt_fmadd_p(const avxvector_t<256,double> a, const avxvector_t<256,double> b, const avxvector_t<256,double> c) noexcept { return _mm256_fmadd_pd(a, b, c); };
#ifdef USE_AVX512
KT_FORCE_INLINE avxvector_t<512,float>  kt_fmadd_p(const avxvector_t<512,float>  a, const avxvector_t<512,float>  b, const avxvector_t<512,float> c)  noexcept { return _mm512_fmadd_ps(a, b, c); };
KT_FORCE_INLINE avxvector_t<512,double> kt_fmadd_p(const avxvector_t<512,double> a, const avxvector_t<512,double> b, const avxvector_t<512,double> c) noexcept { return _mm512_fmadd_pd(a, b, c); };
#endif

// Horizontal sum (reduction) of an AVX register
//  - `v` avxvector
// return a scalar containing the horizontal sum of the elements of `v`, that is,
// `v[0] + v[1] + ... + v[N]` with `N` the appropiate vector size
KT_FORCE_INLINE float kt_hsum_p(avxvector_t<256,float> const v) noexcept
{
    avxvector_half_t<256,float> kt_lv, kt_hv, kt_v;
    avxvector_t<256,float> w = _mm256_hadd_ps(v, v);
    w = _mm256_hadd_ps(w, w); // only required for float
    kt_lv = _mm256_castps256_ps128(w);
    kt_hv = _mm256_extractf128_ps(w, 1);
    kt_v  = _mm_add_ps(kt_lv, kt_hv);
    return kt_sse_scl(kt_v);
};
KT_FORCE_INLINE double kt_hsum_p(avxvector_t<256,double> const v) noexcept
{
    avxvector_half_t<256,double> kt_lv, kt_hv, kt_v;
    avxvector_t<256,double> w = _mm256_hadd_pd(v, v);
    kt_lv = _mm256_castpd256_pd128(w);
    kt_hv = _mm256_extractf128_pd(w, 1);
    kt_v  = _mm_add_pd(kt_lv, kt_hv);
    return kt_sse_scl(kt_v);
};
#ifdef USE_AVX512
KT_FORCE_INLINE float kt_hsum_p(avxvector_t<512,float> const v) noexcept { return _mm512_reduce_add_ps(v); };
KT_FORCE_INLINE double kt_hsum_p(avxvector_t<512,double> const v) noexcept { return _mm512_reduce_add_pd(v); };
#endif


// Level 1 micro kernels
// These micro kernels depend solely on LEVEL 0 micro kernels
// =======================================================================

// Dot-product of two AVX registers
//  - `SZ`  size (in bits) of AVX vector, i.e., 256 or 512
//  - `SUF` suffix of working type, i.e., `double` or `float`
//  - `a` avxvector
//  - `b` avxvector
// returns a scalar containing the dot-product of a and b, <a,b>
template<int SZ, typename SUF> KT_FORCE_INLINE SUF kt_dot_p(avxvector_t<SZ,SUF> const a, avxvector_t<SZ,SUF> const b) noexcept {
    avxvector_t<SZ,SUF> c = kt_mul_p(a, b);
    return kt_hsum_p(c);
};

// Dot-product of two AVX registers (convenience callers)
//  - `a` avxvector
//  - `b` avxvector
// returns a scalar containing the dot-product of a and b, <a,b>
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

}

// Undefine...
#undef kt_addr_t

#endif // AOCLSPARSE_KERNEL_TEMPLATES_T_HPP
