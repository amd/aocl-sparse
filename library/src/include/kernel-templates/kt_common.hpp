/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *  constexpr int supported_base_t = 4;
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
 *      else if constexpr(kt_is_base_t_int<T>())
 *          return 2;
 *      else if constexpr(kt_is_base_t_bfloat16<T>())
 *          return 3; // Equal to new supported_base_t - 1
 *   }
 *
 *   Step 4
 *   ------
 *   Add oracle to get vector type:
 *   template <bsz SZ, typename SUF, v_type VT>
 *    using get_vec_t = type_switch<indx<SZ, SUF, VT>(),  __m64, __m64, __m64, __m64, __m128, __m128d, __m128i, __m128h, __m256, __m256d, __m256i, __m256h
 * #ifdef __AVX512F__
 *                            ,__m512i, __m512, __m512d, __m512h
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

    /**
     * @brief Half-precision floating-point type (fp16)
     *
     * When __AVX512FP16__ is available, fp16 is an alias for the compiler-native
     * _Float16 type, which provides IEEE 754 half-precision arithmetic, conversions,
     * and comparisons as a built-in type. On non-AVX512FP16 builds, fp16 is a
     * minimal tag type used only for SFINAE guards (it is not usable for computation).
     */
#ifdef __AVX512FP16__
    using fp16 = _Float16;
#else
    struct fp16
    {
    };
#endif

    /*
     *   Number of supported "base" types: 4
     *
     * 1. float (and cfloat) maps to float intrinsics
     * 2. double (and cdouble) maps to double intrinsics
     * 3. int (int32_t and int64_t) maps to integer intrinsics
     * 4. fp16 maps to half-precision intrinsics
     * Add new type here and update the supported_base_t accordingly.
     */
    constexpr int supported_base_t = 4;

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

    /**
     * @brief Type trait to check if base type is double-precision
     *
     * Returns true for both real double and complex double (cdouble) types.
     * Used for compile-time type dispatching in kernel templates.
     *
     * @tparam T Type to check
     */
    template <typename T>
    struct kt_is_base_t_double
    {
        /**
         * @brief Conversion operator for boolean evaluation
         * @return true if T is double or std::complex<double>, false otherwise
         */
        constexpr operator bool() const noexcept
        {
            return std::is_same<T, double>::value || std::is_same<T, cdouble>::value;
        }
    };

    /**
     * @brief Type trait to check if base type is single-precision
     *
     * Returns true for both real float and complex float (cfloat) types.
     * Used for compile-time type dispatching in kernel templates.
     *
     * @tparam T Type to check
     */
    template <typename T>
    struct kt_is_base_t_float
    {
        /**
         * @brief Conversion operator for boolean evaluation
         * @return true if T is float or std::complex<float>, false otherwise
         */
        constexpr operator bool() const noexcept
        {
            return std::is_same<T, float>::value || std::is_same<T, cfloat>::value;
        }
    };

    /**
     * @brief Type trait to check if base type is integer
     *
     * Returns true for both 32-bit and 64-bit integer types.
     * Used for compile-time type dispatching in kernel templates.
     *
     * @tparam T Type to check
     */
    template <typename T>
    struct kt_is_base_t_int
    {
        /**
         * @brief Conversion operator for boolean evaluation
         * @return true if T is int32_t or int64_t, false otherwise
         */
        constexpr operator bool() const noexcept
        {
            return std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value;
        }
    };

    /**
     * @brief Type trait to check if base type is half-precision (fp16)
     *
     * Returns true for fp16 type.
     * Used for compile-time type dispatching in kernel templates.
     *
     * @tparam T Type to check
     */
    template <typename T>
    struct kt_is_base_t_fp16
    {
        /**
         * @brief Conversion operator for boolean evaluation
         * @return true if T is fp16, false otherwise
         */
        constexpr operator bool() const noexcept
        {
            return std::is_same<T, fp16>::value;
        }
    };

    /**
     * @brief Type trait to check if type is real (non-complex)
     *
     * Returns true only for fp16, real float and double types, excluding complex variants.
     * Used to constrain template functions to real-only operations (e.g., kt_max_p).
     *
     * @tparam T Type to check
     */
    template <typename T>
    struct kt_type_is_real
    {
        /**
         * @brief Conversion operator for boolean evaluation
         * @return true if T is float, double, or fp16 (excluding complex types), false otherwise
         */
        constexpr operator bool() const noexcept
        {
            return std::is_same<T, double>::value || std::is_same<T, float>::value
                   || std::is_same<T, fp16>::value;
        }
    };

    /**
     * @brief Type trait to check if two vector configurations are identical
     *
     * Returns true if both vector size and scalar data type match.
     * Used for compile-time verification of matching vector types.
     *
     * @tparam SZA   First vector size type
     * @tparam SZB   Second vector size type
     * @tparam SUFA  First scalar data type
     * @tparam SUFB  Second scalar data type
     */
    template <bsz SZA, bsz SZB, typename SUFA, typename SUFB>
    struct kt_is_same
    {
        /**
         * @brief Conversion operator for boolean evaluation
         * @return true if both vector sizes and data types match, false otherwise
         */
        constexpr operator bool() const noexcept
        {
            return SZA == SZB && std::is_same_v<SUFA, SUFB>;
        }
    };

    /**
     * @brief Type trait to validate integer types for kernel template index parameters
     *
     * Checks if a type is a valid integer type for use as array indices in kernel templates.
     * Valid types must be integral and have a size of either 4 or 8 bytes.
     * This includes, for example, 32-bit and 64-bit signed and unsigned integer types.
     *
     * @tparam T Type to validate
     *
     * @par Example:
     * @code
     * static_assert(kt_is_valid_int<int32_t>());   // true  - 4-byte signed integer
     * static_assert(kt_is_valid_int<uint32_t>());  // true  - 4-byte unsigned integer
     * static_assert(kt_is_valid_int<int64_t>());   // true  - 8-byte signed integer
     * static_assert(kt_is_valid_int<uint64_t>());  // true  - 8-byte unsigned integer
     * static_assert(!kt_is_valid_int<int16_t>());  // false - wrong size
     * static_assert(!kt_is_valid_int<float>());    // false - not integral
     * @endcode
     */
    template <typename T>
    struct kt_is_valid_int
    {
        /**
         * @brief Conversion operator for boolean evaluation
         * @return true if T is a valid kernel template integer type, false otherwise
         */
        constexpr operator bool() const noexcept
        {
            return std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8);
        }
    };

    /**
     * @brief SFINAE helper type alias for valid kernel template integer types
     *
     * Used as a non-type template parameter (defaulted to 0) to enable function
     * templates only when the integer type IS is valid (int32_t or int64_t).
     * This provides a cleaner alternative to verbose std::enable_if_t constraints.
     *
     * @tparam T Integer type to validate (must satisfy kt_is_valid_int)
     *
     * @par Usage:
     * @code
     * // Instead of:
     * template <typename IS, typename = std::enable_if_t<kt_is_valid_int<IS>()>>
     * void foo(IS* indices);
     *
     * // Use:
     * template <typename IS, valid_kt_int<IS> = 0>
     * void foo(IS* indices);
     * @endcode
     *
     * @see kt_is_valid_int
     */
    template <typename T>
    using valid_kt_int = std::enable_if_t<kt_is_valid_int<T>{}, int>;

    // AVX CPU instrinsic extensions to implement
    // * ANY      All targets
    // * AVX      All extensions up to AVX2: AVX, FMA, ...
    // * AVX512F  AVX512 Foundations
    // * AVX512DQ ...
    // * AVX512VL Use zero-masked instrinsics, ...
    // Each extension needs to be a superset of the previous
    enum kt_avxext : size_t
    {
        ANY        = ~0U,
        NONE       = 1,
        AVX        = 2,
        AVX2       = 2,
        AVX512F    = 2 + 4,
        AVX512DQ   = 2 + 4 + 8,
        AVX512VL   = 2 + 4 + 8 + 16,
        AVX512FP16 = 2 + 4 + 8 + 16 + 32
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

    /**
     * @brief Data type trait to extract base and full types
     *
     * Primary template for real types where base_type and type are the same.
     * Specialized for complex types to extract the underlying real type.
     *
     * @tparam SUF Scalar data type (float, double, or complex variants)
     */
    template <typename SUF>
    struct kt_dt
    {
        /** @brief Underlying base type (same as SUF for real types) */
        using base_type = SUF;

        /** @brief Full type (same as SUF for real types) */
        using type = SUF;
    };

    /**
     * @brief Specialization for complex<float>
     *
     * Extracts float as the base type from complex<float>.
     */
    template <>
    struct kt_dt<std::complex<float>>
    {
        /** @brief Underlying base type (float) */
        using base_type = float;

        /** @brief Full complex type */
        using type = std::complex<float>;
    };
    /**
     * @brief Specialization for complex<double>
     *
     * Extracts double as the base type from complex<double>.
     */
    template <>
    struct kt_dt<std::complex<double>>
    {
        /** @brief Underlying base type (double) */
        using base_type = double;

        /** @brief Full complex type */
        using type = std::complex<double>;
    };

    // Enum class to represent the fused operation in scatter
    enum class fused_op
    {
        NONE, // No fused operation
        ADD, // Fused ADD
        SUB // Fused SUB
    };
}
#endif
