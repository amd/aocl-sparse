/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
        b128 = supported_base_t * 1,
        b256 = supported_base_t * 2,
        b512 = supported_base_t * 3
    };

    // For a given translation unit that includes this header,
    // this function returns bsz::b512 if the source is compiled
    // with AVX-512 flags else it returns bsz::b256.
    constexpr bsz get_bsz()
    {
#ifndef KT_AVX2_BUILD
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
#ifndef KT_AVX2_BUILD
        return kt_avxext::AVX512F;
#else
        return kt_avxext::AVX2;
#endif
    }
}
#endif
