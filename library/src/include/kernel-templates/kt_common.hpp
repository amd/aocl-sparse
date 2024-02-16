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
 * ************************************************************************
 */
#ifndef _KT_COMMON_
#define _KT_COMMON_

#include <cmath>
#include <complex>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>

#ifdef __OPTIMIZE__
#define KT_FORCE_INLINE inline __attribute__((__always_inline__))
#else
#define KT_FORCE_INLINE inline
#endif

namespace kernel_templates
{
    /*
        This template is used to ensure that the aoclsparse_int type
        used by the AOCL-sparse is either int32_t or int64_t or uint32_t
        or uint64_t
    */
    template <typename T>
    using set_kt_int_type
        = std::enable_if_t<std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8), T>;

    using kt_int_t = set_kt_int_type<kt_int_t>;

    // AVX CPU instrinsic extensions to implement
    // * ANY      All targets
    // * AVX      All extensions up to AVX2: AVX, FMA, ...
    // * AVX512F  AVX512 Foundations
    // * AVX512DQ ...
    // * AVX512VL Use zero-masked instrinsics, ...
    // Each extension needs to be a superset of the previous
    enum kt_avxext : size_t
    {
        ANY      = ~0U,
        NONE     = 1,
        AVX      = 2,
        AVX2     = 2,
        AVX512F  = 2 + 4,
        AVX512DQ = 2 + 4 + 8,
        AVX512VL = 2 + 4 + 8 + 16
    };

    // Enum class that represents the vector lengths
    enum class bsz
    {
        b256,
        b512
    };

    // Delta function (used for zero-padding)
    template <typename T, int L>
    constexpr T pz(const T &x) noexcept
    {
        if constexpr(L >= 0)
            return x;
        else
            return (T)0.0;
    };

// compatibility macro for windows
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
#define kt_sse_scl(kt_v) kt_v.m128d_f64[0];
#else
#define kt_sse_scl(kt_v) kt_v[0];
#endif

    using cfloat  = std::complex<float>;
    using cdouble = std::complex<double>;

    template <bsz SZA, bsz SZB, typename SUFA, typename SUFB>
    struct kt_is_same
    {
        constexpr operator bool() const noexcept
        {
            return SZA == SZB && std::is_same_v<SUFA, SUFB>;
        }
    };

    // Checks if the base type is double - true only for double and cdouble
    template <typename T>
    struct kt_is_base_t_double
    {
        constexpr operator bool() const noexcept
        {
            return std::is_same<T, double>::value || std::is_same<T, cdouble>::value;
        }
    };

    // Checks if the base type is float - true only for float and cfloat
    template <typename T>
    struct kt_is_base_t_float
    {
        constexpr operator bool() const noexcept
        {
            return std::is_same<T, float>::value || std::is_same<T, cfloat>::value;
        }
    };

    // helper template type for avx full vectors
    template <bsz, typename>
    struct avxvector
    {
    };
    template <>
    struct avxvector<bsz::b256, double>
    {
        using type                    = __m256d; // Vector type
        using half_type               = __m128d; // Associated "half" length vector type
        static constexpr size_t value = 4; // Vector length, packet size
        static constexpr size_t half  = 4 >> 1; // Length of half vector
        // amount of elements of type `typename` that the vector holds
        static constexpr size_t count = 4;
        constexpr operator size_t() const noexcept
        {
            return value;
        }
    }; // Convenience operator to return vector length
    template <>
    struct avxvector<bsz::b256, float>
    {
        using type                    = __m256;
        using half_type               = __m128;
        static constexpr size_t value = 8;
        static constexpr size_t half  = 8 >> 1;
        static constexpr size_t count = 8;
        constexpr operator size_t() const noexcept
        {
            return value;
        }
    };
    template <>
    struct avxvector<bsz::b256, cdouble>
    {
        using type                    = __m256d;
        using half_type               = __m128d;
        static constexpr size_t value = 4;
        static constexpr size_t half  = 4 >> 1;
        static constexpr size_t count = 4 >> 1;
        constexpr operator size_t() const noexcept
        {
            return value;
        }
    };
    template <>
    struct avxvector<bsz::b256, cfloat>
    {
        using type                    = __m256;
        using half_type               = __m128;
        static constexpr size_t value = 8;
        static constexpr size_t half  = 8 >> 1;
        static constexpr size_t count = 8 >> 1;
        constexpr operator size_t() const noexcept
        {
            return value;
        }
    };
#ifdef __AVX512F__
    template <>
    struct avxvector<bsz::b512, double>
    {
        using type                    = __m512d;
        using half_type               = __m256d;
        static constexpr size_t value = 8;
        static constexpr size_t half  = 8 >> 1;
        static constexpr size_t count = 8;
        constexpr operator size_t() const noexcept
        {
            return value;
        }
    };
    template <>
    struct avxvector<bsz::b512, float>
    {
        using type                    = __m512;
        using half_type               = __m256;
        static constexpr size_t value = 16;
        static constexpr size_t half  = 16 >> 1;
        static constexpr size_t count = 16;
        constexpr operator size_t() const noexcept
        {
            return value;
        }
    };
    template <>
    struct avxvector<bsz::b512, cdouble>
    {
        using type                    = __m512d;
        using half_type               = __m256d;
        static constexpr size_t value = 8;
        static constexpr size_t half  = 8 >> 1;
        static constexpr size_t count = 8 >> 1;
        constexpr operator size_t() const noexcept
        {
            return value;
        }
    };
    template <>
    struct avxvector<bsz::b512, cfloat>
    {
        using type                    = __m512;
        using half_type               = __m256;
        static constexpr size_t value = 16;
        static constexpr size_t half  = 16 >> 1;
        static constexpr size_t count = 16 >> 1;
        constexpr operator size_t() const noexcept
        {
            return value;
        }
    };
#endif

    // helper template type for avx vectors
    template <bsz SZ, typename SUF>
    using avxvector_t = typename avxvector<SZ, SUF>::type;

    // helper template type for "half" avx vectors
    template <bsz SZ, typename SUF>
    using avxvector_half_t = typename avxvector<SZ, SUF>::half_type;

    // helper template value (length) for avx vectors (pack size)
    template <bsz SZ, typename SUF>
    inline constexpr size_t avxvector_v = avxvector<SZ, SUF>::value;

    // helper template value (length) for "half" avx vectors (half pack size)
    template <bsz SZ, typename SUF>
    inline constexpr size_t avxvector_half_v = avxvector<SZ, SUF>::half;

    template <bsz SZ, typename SUF>
    inline constexpr size_t hsz_v = avxvector<SZ, SUF>::half;

    // helper template type-storage size (how many elements of a type fit in a vector)
    // for real-valued types it matches with pack size, for complex it is half (real,imag)
    template <bsz SZ, typename SUF>
    inline constexpr size_t tsz_v = avxvector<SZ, SUF>::count;

    //-----------------
    // Level-0 kernels
    //-----------------

    // Zero out an AVX register
    // return an avxvector filled with zeroes.
    //
    // Example: `avxvector<bsz::b256,float> v = kt_setzero_p<bsz::b256,float>() is equivalent to `v = _mm256_setzero_ps()`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept;

    // -----------------------------------------------------------------------

    // Fill vector with a scalar value
    // return an avxvector filled with the same scalar value.
    //
    // Example `avxvector_t<bsz::b512, double> v = kt_set1_p<bsz::b512, double>(x)` is equivalent to `v = _mm512_set1_pd(x)`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept;

    // -----------------------------------------------------------------------

    // Unaligned set (load) to AVX register with indirect memory access
    //  - `SZ` size (in bits) of AVX vector, bsz::b256 or bsz::b512
    //  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
    //  - `v` dense array for loading the data
    //  - `b` map address within range of `v`
    // return an avxvector with the loaded data.
    //
    // Example: `kt_set_p<bsz::b256, double>(v, b)` expands to _mm256_set_pd(v[*(b+3)],v[*(b+2)],v[*(b+1)],v[*(b+0)])
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const kt_int_t *b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const kt_int_t *b) noexcept;

    // -----------------------------------------------------------------------

    // Unaligned load to AVX register with zero mask direct memory model.
    // Copies `L` elements from `v` and pads with zero the rest of AVX vector
    //  - `SZ`  size (in bits) of AVX vector, i.e., bsz::b256 or bsz::b512
    //  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
    //  - `EXT` type of kt_avxext to use, i.e., AVX, AVX512F, ...
    //  - `L` number of elements from `v` to copy
    //  - `v` dense array for loading the data
    //  - `b` delta address within `v`
    // return an avxvector with the loaded data.
    //
    // Example: `kt_maskz_set_p<256, float, AVX, 3>(v, b)` expands to `_mm256_set_ps(0f, 0f, 0f, 0f, 0f, v[b+2], v[b+1], v[b+0])`
    // and      `kt_maskz_set_p<256, double, AVX512VL, 3>(v, b)` expands to _mm256_maskz_loadu_pd(7, &v[b])
    template <bsz SZ, typename SUF, kt_avxext EXT, int L>
    KT_FORCE_INLINE
        std::enable_if_t<(SZ == bsz::b256 && EXT == kt_avxext::AVX2), avxvector_t<SZ, SUF>>
        kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept;

    template <bsz SZ, typename SUF, kt_avxext EXT, int L>
    KT_FORCE_INLINE
        std::enable_if_t<EXT == kt_avxext::AVX512VL || SZ == bsz::b512, avxvector_t<SZ, SUF>>
        kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept;

    // -----------------------------------------------------------------------

    // Unaligned load to AVX register with zero mask indirect memory model.
    // Copies `L` elements from `v` and pads with zero the rest of AVX vector
    //  - `SZ`  size (in bits) of AVX vector, i.e., bsz::b256 or bsz::b512
    //  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
    //  - `EXT` type of kt_avxext to use, i.e., AVX, AVX512F, ...
    //  - `L` number of elements from `v` to copy
    //  - `v` dense array for loading the data
    //  - `b` map address within range of `v`
    // return an avxvector with the loaded data.
    //
    // Example: `kt_maskz_set_p<256, double, AVX, 2>(v, b)` expands to `_mm256_set_pd(0.0, 0.0, v[*(b+1)], v[*(b+0)])`
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept;

    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept;

    // -----------------------------------------------------------------------

    // Dense direct aligned load to AVX register
    // return an avxvector with the loaded content.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept;

    // -----------------------------------------------------------------------

    // Dense direct (un)aligned load to AVX register
    // return an avxvector with the loaded content.
    //
    // Example: `avxvector_t<bsz::b256,double> v = kt_loadu_p<bsz::b256,double>(&a[7])` is equivalent to `v = _mm256_loadu_pd(&a[7])`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept;

    // -----------------------------------------------------------------------

    // Stores the values in an AVX register to a memory location (Memory does not have to be aligned)
    // returns void.
    //
    // Example:`kt_storeu_p<256,double>(&x, vec)` is equivalent to `_mm256_storeu_pd(&x, vec)`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, void>
                    kt_storeu_p(const SUF *, const avxvector_t<SZ, SUF>) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, void>
                    kt_storeu_p(const SUF *, const avxvector_t<SZ, SUF>) noexcept;

    // -----------------------------------------------------------------------

    // Vector addition of two AVX registers.
    //  - `a` avxvector
    //  - `b` avxvector
    // return an avxvector with `a` + `b` elementwise.
    //
    // Example: `avxvector_t<bsz::b256, double> c = kt_add_p(a, b)` is equivalent to `__256d c = _mm256_add_pd(a, b)`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    // -----------------------------------------------------------------------

    // Vector product of two AVX registers.
    //  - `a` avxvector
    //  - `b` avxvector
    // return an avxvector with `a` * `b` elementwise.
    // Example: `avxvector_t<bsz::b256, double> c = kt_mul_p(a, b)` is equivalent to `__256d c = _mm256_mul_pd(a, b)`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    // -----------------------------------------------------------------------

    // Vector fused multiply-add of three AVX registers.
    //  - `a` avxvector
    //  - `b` avxvector
    //  - `c` avxvector
    // return an avxvector with `a` * `b` + `c` elementwise.
    // Example: `avxvector_t<bsz::b256, double> d = kt_fmadd_p(a, b, c)` is equivalent to `__256d d = _mm256_mul_pd(a, b, c)`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_fmadd_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_fmadd_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept;

    // -----------------------------------------------------------------------

    // Horizontal sum (reduction) of an AVX register
    //  - `v` avxvector
    // return a scalar containing the horizontal sum of the elements of `v`, that is,
    // `v[0] + v[1] + ... + v[N]` with `N` the appropiate vector size
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept;

    // -----------------------------------------------------------------------

    // Templated version of the conjugate operation
    // Conjugate an AVX register
    //  - `SZ`  size (in bits) of AVX vector, i.e., bsz::b256 or bsz::b512
    //  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
    //  - `a` avxvector
    // returns `conjugate(a)` for complex types and returns `a`for real
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_conj_p(const avxvector_t<SZ, SUF> a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_conj_p(const avxvector_t<SZ, SUF> a) noexcept;
}
#endif
