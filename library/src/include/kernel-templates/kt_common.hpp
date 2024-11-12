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

#ifndef KERNEL_TEMPLATES_T_HPP
#error "Outside the scope of kernel templates sub-headers never use ``kt_common.hpp''directly;\
    include ``kernel_templates.hpp'' instead."
#endif

#ifndef _KT_COMMON_
#define _KT_COMMON_

#include <cmath>
#include <complex>
#include <cstdint>
#include <immintrin.h>
#include <tuple>
#include <type_traits>

#ifdef __OPTIMIZE__
#define KT_FORCE_INLINE inline __attribute__((__always_inline__))
#else
#define KT_FORCE_INLINE inline
#endif

/*
 * Guide to add a new datatype
 * ---------------------------
 *
 *  1. Increment the number of supported_base_t.
 *  2. Add a new type checker struct kt_is_base_t_(x) where x is
 *     the new datatype (Refer to 'kt_is_base_t_double')
 *  3. In get generator::type_idx, add a unique index ID  for the
 *     datatype (which is equal to the new supported_base_t - 1).
 *  4. In generator::get_vec_t, add the vector types of the new types
 *     to indices (supported_base_t - 1), (supported_base_t - 1) * 2 and
 *     (supported_base_t - 1) * 3.
 *  5. Add changes to generator::get_sz_v to calculate the
 *     psize (Packet size), hpsize (half-packet size) and tsz (type size (sizeof(x)))
 *     of the new datatype x
 *
 *
 *  Example
 *  ========
 *  To enable "bfloat16" datatype, where one bfloat16 element is 16 bits.
 *
 *  Step 1
 *  ------
 *  Increase by 1 the supported base type:
 *  constexpr int supported_base_t = 3;
 *
 *  Step 2
 *  -------
 *  Add type comparison operator:
 *  template <typename T>
 *  struct kt_is_base_t_bfloat16
 *  {
 *    constexpr operator bool() const noexcept
 *    {
 *        return std::is_same_v<T, bfloat16>;
 *    }
 *  };
 *
 *  Step 3
 *  ------
 *  Add oracle to get index into type database:
 *  template <typename T>
 *  constexpr int type_idx()
 *   {
 *       if constexpr(kt_is_base_t_float<T>())
 *            return 0;
 *       else if constexpr(kt_is_base_t_double<T>())
 *            return 1;
 *       else if constexpr(kt_is_base_t_bfloat16<T>())
 *            return 2; // Equal to new supported_base_t - 1
 *   }
 *
 *   Step 4
 *   ------
 *   Add oracle to get vector type:
 *   template <bsz SZ, typename SUF, v_type VT>
 *    using get_vec_t = type_switch<indx<SZ, SUF, VT>(), __m128, __m128d, __m256bh, __m256, __m256d, __m256bh
 * #ifdef __AVX512F__
 *                            ,__m512, __m512d __m512bh
 * #endif
 *
 *   Step 5
 *   ------
 *   Add oracle to get vector packet sizes:
 *
 *   template <typename T, typename SUF, bool isTSZ = false>
 *   constexpr int get_sz_v()
 *   {
 *       if constexpr(std::is_floating_point<SUF>::value || isTSZ == true || kt_is_base_t_bfloat16<T>())
 *            return sizeof(T) / sizeof(SUF);
 *       else
 *            return ((sizeof(T) / sizeof(SUF)) * 2);
 *   }
 *
 */

namespace kernel_templates
{
    using cfloat  = std::complex<float>;
    using cdouble = std::complex<double>;

    /*
     *   Number of supported "base" types: 2
     *
     * 1. float (and cfloat) maps to float intrinsics
     * 2. double (and cdouble) maps to double intrinsics
     * Add new types here
     * 3. ...
     */
    constexpr int supported_base_t = 2;

    // Enum class that represents the vector lengths
    enum class bsz
    {
        b256 = supported_base_t * 1,
        b512 = supported_base_t * 2
    };

    // For a given translation unit that includes this header,
    // this function returns bsz::b512 if the source is compiled
    // with AVX-512 flags else it returns bsz::b256.
    constexpr bsz get_bsz()
    {
#ifdef __AVX512F__
        return bsz::b512;
#else
        return bsz::b256;
#endif
    }

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

    // Checks if both the vector length and datatype are the same
    template <bsz SZA, bsz SZB, typename SUFA, typename SUFB>
    struct kt_is_same
    {
        constexpr operator bool() const noexcept
        {
            return SZA == SZB && std::is_same_v<SUFA, SUFB>;
        }
    };

    namespace generator
    {
        // AVXVECTOR GENERATOR
        // ===================
        // Here are the elements required to automate the process
        // of defining any of the available AVXVECTORS given SZ (intrinsic base type)
        // and SUF will populate all the members of the AVXVECTOR struct

        // Auxiliary template to build tuples (used as a function)
        // It will be used to create a list of types.
        // Purpose: Given an integer N, it returns the type in the Nth index
        template <std::size_t N, typename... T>
        using type_switch = typename std::tuple_element<N, std::tuple<T...>>::type;

        // Auxiliary function to get the base index of float, double, ...
        template <typename T>
        constexpr int index_t()
        {
            if constexpr(kt_is_base_t_float<T>())
                return 0;
            else if constexpr(kt_is_base_t_double<T>())
                return 1;
            // else if constexpr(...<T>)
            // return 2;
        }

        /*
        *  Function that calculates the index at which the required type is in the mm intinsic database
        *
        *  Example:  In case of DOUBLE, FULL, bsz::v512
        *  The function will return    (2*2) + 1 - 0 = 5. In index 5, the type '__m512d' is present.
        */
        template <bsz SZ, typename SUF, bool HALF>
        constexpr int index()
        {
            // clang-format off
            // (Index of the vector length) + (Index of the base type) - (Index adjustment for half/full type)
            return static_cast<int>(SZ) + index_t<SUF>() - (static_cast<int>(HALF) * supported_base_t);
            // clang-format on
        }

// Temporary suppression for GCC [up to at least 14.1] compiler on template parameters with attribute type
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
        // clang-format off
        // mm instrinsic database storing all mm types used by AVXVECTOR
        // Note on adding new types, refer to guide at top of file!
        template <bsz SZ, typename SUF, bool HALF>
        // index_t                                             float   double   float   double
        using get_vec_t = type_switch<index<SZ, SUF, HALF>(), __m128, __m128d, __m256, __m256d
        #ifdef __AVX512F__
        //                         float   double
                                , __m512, __m512d
        #endif
        >;
#pragma GCC diagnostic pop
        // clang-format on

        /*
        *  Function that returns the number of elements in a vector (both type size and pack size)
        *  based on the parameters.
        */
        template <typename T, typename SUF, bool isTSZ = false>
        constexpr int get_sz_v()
        {
            // For non-complex types: pack and type sizes always match
            if constexpr(std::is_floating_point<SUF>::value || isTSZ == true)
                return sizeof(T) / sizeof(SUF);
            else // For complex types: pack size is twice of type size (real, imag)
                return ((sizeof(T) / sizeof(SUF)) * 2);
        }
    }

    /*
     *   Ensure that the custom int type used is of size 4 or 8
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

    // Based on compilation returns the kt extension to
    // be used for a given translation unit
    constexpr kt_avxext get_kt_ext()
    {
#ifdef __AVX512F__
        return kt_avxext::AVX512F;
#else
        return kt_avxext::AVX2;
#endif
    }

    // --------------------------------------------------------------------------------------
    // Delta functions (used for zero-padding)
    // --------------------------------------------------------------------------------------

    // Specialization for REAL and COMPLEX space PZ with offset
    template <typename T, int L, bool get_real_part = false>
    constexpr auto pz(const T *x, const kt_int_t *idx, kt_int_t offset) noexcept
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            if constexpr(L >= 0)
                return x[*(idx + offset)];
            else
                return (T)0.0;
        }
        else if constexpr(std::is_same_v<T, std::complex<float>>
                          || std::is_same_v<T, std::complex<double>>)
        {
            if constexpr(L >= 0)
            {
                if constexpr(get_real_part == true)
                    return x[*(idx + offset)].real();
                else
                    return x[*(idx + offset)].imag();
            }
            else
            {
                using U = typename T::value_type;
                return U(0.0);
            }
        }
    };

    // Specialization for REAL and COMPLEX space PZ without offset
    template <typename T, int L, bool get_real_part = false>
    constexpr auto pz(const T *x, kt_int_t idx) noexcept
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            if constexpr(L >= 0)
                return x[idx];
            else
                return (T)0.0;
        }
        else if constexpr(std::is_same_v<T, std::complex<float>>
                          || std::is_same_v<T, std::complex<double>>)
        {
            if constexpr(L >= 0)
            {
                if constexpr(get_real_part == true)
                    return x[idx].real();
                else
                    return x[idx].imag();
            }
            else
            {
                using U = typename T::value_type;
                return U(0.0);
            }
        }
    };

    // --------------------------------------------------------------------------------------

// compatibility macro for windows
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
#define kt_sse_scl(kt_v) kt_v.m128d_f64[0]
#else
#define kt_sse_scl(kt_v) kt_v[0]
#endif

    /*
     *  avxvector struct
     */
    template <bsz SZ, typename SUF>
    struct avxvector
    {
        using type      = generator::get_vec_t<SZ, SUF, false>; // __m type for full vector
        using half_type = generator::get_vec_t<SZ, SUF, true>; // __m type for half vector
        static constexpr size_t p_size  = generator::get_sz_v<type, SUF>(); // pack size
        static constexpr size_t hp_size = generator::get_sz_v<half_type, SUF>(); // half pack size
        static constexpr size_t tsz     = generator::get_sz_v<type, SUF, true>(); // type size
        // Get the packet size:
        constexpr operator size_t() const noexcept
        {
            return p_size;
        }
    };

    // helper template type for avx vectors
    template <bsz SZ, typename SUF>
    using avxvector_t = typename avxvector<SZ, SUF>::type;

    // helper template type for "half" avx vectors
    template <bsz SZ, typename SUF>
    using avxvector_half_t = typename avxvector<SZ, SUF>::half_type;

    // helper template value for an avx vector' pack size
    template <bsz SZ, typename SUF>
    inline constexpr size_t avxvector_v = avxvector<SZ, SUF>::p_size;

    // helper template value for "half" avx vector' half pack size
    template <bsz SZ, typename SUF>
    inline constexpr size_t avxvector_half_v = avxvector<SZ, SUF>::hp_size;

    template <bsz SZ, typename SUF>
    inline constexpr size_t hsz_v = avxvector<SZ, SUF>::hp_size;

    // helper template type-storage size (how many elements of a type fit in a vector)
    // for real-valued types it matches with pack size, for complex it is half (real,imag)
    template <bsz SZ, typename SUF>
    inline constexpr size_t tsz_v = avxvector<SZ, SUF>::tsz;

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

    // Vector subtraction of two AVX registers.
    //  - `a` avxvector
    //  - `b` avxvector
    // return an avxvector with `a` - `b` elementwise.
    //
    // Example: `avxvector_t<bsz::b256, double> c = kt_sub_p(a, b)` is equivalent to `__256d c = _mm256_sub_pd(a, b)`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

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
    // Example: `avxvector_t<bsz::b256, double> d = kt_fmadd_p(a, b, c)` is equivalent to `__256d d = _mm256_fmadd_pd(a, b, c)`
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

    // Vector fused multiply-subtract of three AVX registers.
    //  - `a` avxvector
    //  - `b` avxvector
    //  - `c` avxvector
    // return an avxvector with `a` * `b` - `c` elementwise.
    // Example: `avxvector_t<bsz::b256, double> d = kt_fmsub_p(a, b, c)` is equivalent to `__256d d = _mm256_fmsub_pd(a, b, c)`
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_fmsub_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_fmsub_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept;

    // -----------------------------------------------------------------------

    //-----------------
    // Level-1 kernels
    //-----------------

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
