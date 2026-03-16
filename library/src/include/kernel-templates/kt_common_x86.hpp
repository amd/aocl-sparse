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
#error \
    "Outside the scope of kernel templates sub-headers never use ``kt_common_x86.hpp''directly;\
    include ``kernel_templates.hpp'' instead."
#endif
#ifndef _KT_COMMON_X86_
#define _KT_COMMON_X86_
#include "kt_common.hpp"

namespace kernel_templates
{
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
            else if constexpr(kt_is_base_t_int<T>())
                return 2;
            // else if constexpr(...<T>)
            // return 3;
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
        // Note on adding new types, refer to guide in kt_common.hpp!
        // __m64 is used for 64-bit vectors irrespective of the base type. Operation on __m64 is not
        // facilitated by the AVXVECTOR struct, but it is used for half vectors.
        template <bsz SZ, typename SUF, bool HALF>
        // index_t                                            float  double   int    float   double      int   float   double     int
        using get_vec_t = type_switch<index<SZ, SUF, HALF>(), __m64, __m64, __m64,  __m128, __m128d, __m128i, __m256, __m256d, __m256i
        #ifdef __AVX512F__
        //                                float   double      int
                                       , __m512, __m512d, __m512i
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
            if constexpr(std::is_floating_point<SUF>::value || isTSZ == true
                         || kt_is_base_t_int<SUF>())
                return sizeof(T) / sizeof(SUF);
            else // For complex types: pack size is twice of type size (real, imag)
                return ((sizeof(T) / sizeof(SUF)) * 2);
        }
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

    /**
     * @brief SIMD vector type traits and metadata
     *
     * Provides compile-time metadata and type information for SIMD vector types
     * including the full vector type, half vector type, and size information.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, int, complex types, etc.)
     */
    template <bsz SZ, typename SUF>
    struct avxvector
    {
        /** @brief Full SIMD vector type (e.g., __m256d, __m512ps) */
        using type = generator::get_vec_t<SZ, SUF, false>;

        /** @brief Half SIMD vector type (e.g., __m128d for 256-bit vectors) */
        using half_type = generator::get_vec_t<SZ, SUF, true>;

        /** @brief Number of elements that fit in the full vector */
        static constexpr size_t p_size = generator::get_sz_v<type, SUF>();

        /** @brief Number of elements that fit in the half vector */
        static constexpr size_t hp_size = generator::get_sz_v<half_type, SUF>();

        /** @brief Size in bytes of the scalar type */
        static constexpr size_t tsz = generator::get_sz_v<type, SUF, true>();

        /**
         * @brief Conversion operator to get the packet size
         * @return Number of elements in the full vector (p_size)
         */
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

    /**
     * @brief Zero out an AVX register
     *
     * Returns an AVX vector register filled with zeros. This function is specialized
     * for different vector sizes (128-bit, 256-bit, and 512-bit) and supports various data types
     * including float, double, integer types, and their complex counterparts.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, int, std::complex<float>, std::complex<double>, etc.)
     *
     * @return avxvector_t<SZ, SUF> An AVX vector register filled with zeros
     *
     * @note This function is force-inlined for optimal performance
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors
     *       - SZ == bsz::b256: AVX 256-bit vectors
     *       - SZ == bsz::b512: AVX-512 512-bit vectors
     * @note Supported data types: float, double, int, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Create a 256-bit vector of floats filled with zeros
     * avxvector_t<bsz::b256, float> v = kt_setzero_p<bsz::b256, float>();
     * // Equivalent to: v = _mm256_setzero_ps()
     *
     * // Create a 512-bit vector of doubles filled with zeros
     * avxvector_t<bsz::b512, double> w = kt_setzero_p<bsz::b512, double>();
     * // Equivalent to: w = _mm512_setzero_pd()
     *
     * // Create a 256-bit vector of complex doubles filled with zeros
     * avxvector_t<bsz::b256, std::complex<double>> z = kt_setzero_p<bsz::b256, std::complex<double>>();
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Fill vector with a scalar value
     *
     * Returns an AVX vector register with all elements set to the same scalar value.
     * This function is specialized for different vector sizes (128-bit, 256-bit, and 512-bit)
     * and supports various data types including float, double, integer types, and their complex counterparts.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, int, std::complex<float>, std::complex<double>, etc.)
     *
     * @param[in] x Scalar value to broadcast across all vector elements
     *
     * @return avxvector_t<SZ, SUF> An AVX vector register with all elements set to x
     *
     * @note This function is force-inlined for optimal performance
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors
     *       - SZ == bsz::b256: AVX 256-bit vectors
     *       - SZ == bsz::b512: AVX-512 512-bit vectors
     * @note Supported data types: float, double, int, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Create a 512-bit vector of doubles, all elements set to 3.14
     * avxvector_t<bsz::b512, double> v = kt_set1_p<bsz::b512, double>(3.14);
     * // Equivalent to: v = _mm512_set1_pd(3.14)
     *
     * // Create a 256-bit vector of floats, all elements set to 2.5f
     * avxvector_t<bsz::b256, float> w = kt_set1_p<bsz::b256, float>(2.5f);
     * // Equivalent to: w = _mm256_set1_ps(2.5f)
     *
     * // Create a 256-bit vector of complex doubles, all elements set to (1.0 + 2.0i)
     * std::complex<double> z(1.0, 2.0);
     * avxvector_t<bsz::b256, std::complex<double>> c = kt_set1_p<bsz::b256, std::complex<double>>(z);
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Unaligned set (load) to AVX register with indirect memory access
     *
     * Loads data from a dense array into an AVX vector register using indirect memory addressing.
     * The function uses an index array to gather elements from non-contiguous memory locations.
     * This function is specialized for different vector sizes (128-bit, 256-bit, and 512-bit)
     * and supports various data types including float, double, integer types, and their complex counterparts.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, int, std::complex<float>, std::complex<double>, etc.)
     *
     * @param[in] v Dense array containing the source data
     * @param[in] b Pointer to index array specifying which elements from v to load
     *
     * @return avxvector_t<SZ, SUF> An AVX vector register with the loaded data
     *
     * @note This function is force-inlined for optimal performance
     * @note The memory access is unaligned (no alignment requirements)
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors
     *       - SZ == bsz::b256: AVX 256-bit vectors
     *       - SZ == bsz::b512: AVX-512 512-bit vectors
     * @note Supported data types: float, double, int, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Load 4 doubles from v using indices in b
     * double v[10] = {0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
     * kt_int_t b[4] = {1, 3, 5, 7};
     * avxvector_t<bsz::b256, double> vec = kt_set_p<bsz::b256, double>(v, b);
     * // Equivalent to: vec = _mm256_set_pd(v[*(b+3)], v[*(b+2)], v[*(b+1)], v[*(b+0)])
     * //                    = _mm256_set_pd(v[7], v[5], v[3], v[1])
     * //                    = _mm256_set_pd(7.7, 5.5, 3.3, 1.1)
     *
     * // Load 4 floats using indirect addressing
     * float a[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
     * kt_int_t idx[4] = {0, 2, 4, 6};
     * avxvector_t<bsz::b128, float> w = kt_set_p<bsz::b128, float>(a, idx);
     * // w contains {a[0], a[2], a[4], a[6]} = {0.0f, 2.0f, 4.0f, 6.0f}
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const kt_int_t *b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const kt_int_t *b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const kt_int_t *b) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Unaligned load to AVX register with zero mask - direct memory model
     *
     * Loads L elements from consecutive memory locations starting at v[b] and fills the remaining
     * vector elements with zeros. The function uses the compile-time template parameter L determines which
     * positions should contain loaded data versus zeros.
     *
     * The implementation uses `_mm256_set_pd/ps` (or equivalent) with pz<SUF, L-N> which returns:
     * - v[b+N] if (L-N) >= 0 (load the element)
     * - 0 if (L-N) < 0 (pad with zero)
     *
     * For AVX512VL extension, uses efficient masked load instructions instead.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, int, std::complex<float>, std::complex<double>, etc.)
     * @tparam EXT AVX extension type (kt_avxext::AVX2, kt_avxext::AVX512VL, etc.)
     * @tparam L Number of elements to load from array (compile-time constant, must be ≤ vector capacity)
     *
     * @param[in] v Dense array containing the source data
     * @param[in] b Starting index within array v (loads from v[b], v[b+1], ..., v[b+L-1])
     *
     * @return avxvector_t<SZ, SUF> An AVX vector with L loaded elements and remaining elements zeroed
     *
     * @note This function is force-inlined for optimal performance
     * @note The memory access is unaligned (no alignment requirements)
     * @note For AVX512VL with bsz::b256, uses efficient masked load: _mm256_maskz_loadu_pd/ps((1<<L)-1, &v[b])
     * @note For AVX2, uses element-by-element set with zero padding via pz() helper
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors
     *       - SZ == bsz::b256 with EXT == AVX2: AVX 256-bit vectors (element-wise)
     *       - SZ == bsz::b256 with EXT == AVX512VL or SZ == bsz::b512: Masked load
     * @note Supported data types: float, double, int, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Load 3 doubles from v[2], v[3], v[4] and pad rest with zeros
     * double v[10] = {0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
     * avxvector_t<bsz::b256, double> vec = kt_maskz_set_p<bsz::b256, double, kt_avxext::AVX2, 3>(v, 2);
     * // With AVX2: _mm256_set_pd(pz<double,3-4>(v,2+3), pz<double,3-3>(v,2+2), pz<double,3-2>(v,2+1), pz<double,3-1>(v,2+0))
     * //          = _mm256_set_pd(pz<double,-1>(v,5), pz<double,0>(v,4), pz<double,1>(v,3), pz<double,2>(v,2))
     * //          = _mm256_set_pd(0.0, v[4], v[3], v[2])
     * //          = _mm256_set_pd(0.0, 4.4, 3.3, 2.2)
     *
     * // With AVX512VL, use efficient masked load
     * avxvector_t<bsz::b256, double> w = kt_maskz_set_p<bsz::b256, double, kt_avxext::AVX512VL, 3>(v, 2);
     * // Equivalent to: w = _mm256_maskz_loadu_pd(0b0111, &v[2])
     * //                  = {v[2], v[3], v[4], 0.0} = {2.2, 3.3, 4.4, 0.0}
     * @endcode
     */
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept;

    template <bsz SZ, typename SUF, kt_avxext EXT, int L>
    KT_FORCE_INLINE
        std::enable_if_t<(SZ == bsz::b256 && EXT == kt_avxext::AVX2), avxvector_t<SZ, SUF>>
        kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept;

    template <bsz SZ, typename SUF, kt_avxext EXT, int L>
    KT_FORCE_INLINE
        std::enable_if_t<EXT == kt_avxext::AVX512VL || SZ == bsz::b512, avxvector_t<SZ, SUF>>
        kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Unaligned load to AVX register with zero mask - indirect memory model
     *
     * Loads L elements from a dense array using indirect/gather addressing and fills the remaining
     * vector elements with zeros. The function uses an index array b to specify which elements to load.
     * The compile-time template parameter L determines which positions contain loaded data versus zeros.
     *
     * The implementation uses `_mm256_set_pd/ps` (or equivalent) with pz<SUF, L-N> which returns:
     * - v[*(b+N)] if (L-N) >= 0 (load element at indirect index)
     * - 0 if (L-N) < 0 (pad with zero)
     *
     * This allows for non-contiguous memory access patterns with zero-padding for partial vector loads.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, int, std::complex<float>, std::complex<double>, etc.)
     * @tparam EXT AVX extension type (kt_avxext::AVX2, kt_avxext::AVX512F, etc.)
     * @tparam L Number of elements to load from array (compile-time constant, must be ≤ vector capacity)
     *
     * @param[in] v Dense array containing the source data
     * @param[in] b Pointer to index array (loads v[b[0]], v[b[1]], ..., v[b[L-1]])
     *
     * @return avxvector_t<SZ, SUF> An AVX vector with L loaded elements and remaining elements zeroed
     *
     * @note This function is force-inlined for optimal performance
     * @note The memory access is unaligned (no alignment requirements)
     * @note Uses indirect addressing: loads v[*(b+0)], v[*(b+1)], ..., v[*(b+L-1)]
     * @note Remaining vector positions (L to vector_size-1) are filled with zeros via pz() helper
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors
     *       - SZ == bsz::b256: AVX 256-bit vectors
     *       - SZ == bsz::b512: AVX-512 512-bit vectors
     * @note Supported data types: float, double, int, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Load 2 doubles using indirect addressing, pad rest with zeros
     * double v[10] = {0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
     * kt_int_t indices[4] = {1, 5, 7, 9};
     * avxvector_t<bsz::b256, double> vec = kt_maskz_set_p<bsz::b256, double, kt_avxext::AVX2, 2>(v, indices);
     * // Expands to: _mm256_set_pd(pz<double,2-4>(v,indices,3), pz<double,2-3>(v,indices,2),
     * //                            pz<double,2-2>(v,indices,1), pz<double,2-1>(v,indices,0))
     * //           = _mm256_set_pd(pz<double,-2>(v,indices,3), pz<double,-1>(v,indices,2),
     * //                            pz<double,0>(v,indices,1), pz<double,1>(v,indices,0))
     * //           = _mm256_set_pd(0.0, 0.0, v[*(indices+1)], v[*(indices+0)])
     * //           = _mm256_set_pd(0.0, 0.0, v[5], v[1])
     * //           = _mm256_set_pd(0.0, 0.0, 5.5, 1.1)
     *
     * // Load 3 out of 8 floats with zero-padding
     * float a[12] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
     * kt_int_t idx[8] = {0, 3, 6, 9, 2, 5, 8, 11};
     * avxvector_t<bsz::b256, float> w = kt_maskz_set_p<bsz::b256, float, kt_avxext::AVX2, 3>(a, idx);
     * // w contains {a[idx[0]], a[idx[1]], a[idx[2]], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
     * //           = {a[0], a[3], a[6], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
     * //           = {0.0f, 3.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
     * @endcode
     */
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept;

    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept;

    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Dense direct aligned load to AVX register
     *
     * Loads a full vector from an aligned memory location into an AVX register. This function performs
     * an aligned memory load, which requires that the memory address is properly aligned to the vector
     * size boundary (16-byte for SSE, 32-byte for AVX2, 64-byte for AVX-512). The implementation uses
     * `_mm_load_*`, `_mm256_load_*`, or `_mm512_load_*` intrinsics depending on the vector size.
     *
     * For complex types (std::complex<float> and std::complex<double>), the function uses `reinterpret_cast`
     * to treat them as their base type (float or double respectively), as complex values are stored
     * contiguously in memory as [real, imag, real, imag, ...].
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, int, std::complex<float>, std::complex<double>, etc.)
     *
     * @param[in] a Pointer to aligned memory location to load from
     *
     * @return avxvector_t<SZ, SUF> An AVX vector register with the loaded data
     *
     * @warning The pointer `a` MUST be aligned to the vector size boundary:
     *          - 16-byte alignment for bsz::b128
     *          - 32-byte alignment for bsz::b256
     *          - 64-byte alignment for bsz::b512
     *          Unaligned access will cause undefined behavior or segmentation fault.
     *
     * @note This function is force-inlined for optimal performance
     * @note Aligned loads are faster than unaligned loads on most architectures
     * @note For unaligned memory access, use kt_loadu_p() instead
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors (_mm_load_*)
     *       - SZ == bsz::b256: AVX 256-bit vectors (_mm256_load_*)
     *       - SZ == bsz::b512: AVX-512 512-bit vectors (_mm512_load_*)
     * @note Supported data types: float, double, int32_t, int64_t, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Load 4 doubles from aligned memory
     * alignas(32) double a[4] = {1.1, 2.2, 3.3, 4.4};
     * avxvector_t<bsz::b256, double> v = kt_load_p<bsz::b256, double>(a);
     * // Equivalent to: v = _mm256_load_pd(a)
     *
     * // Load 8 floats from aligned memory
     * alignas(32) float b[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
     * avxvector_t<bsz::b256, float> w = kt_load_p<bsz::b256, float>(b);
     * // Equivalent to: w = _mm256_load_ps(b)
     *
     * // Load 2 complex doubles from aligned memory (stored as 4 doubles: r0, i0, r1, i1)
     * alignas(32) std::complex<double> c[2] = {{1.0, 2.0}, {3.0, 4.0}};
     * avxvector_t<bsz::b256, std::complex<double>> z = kt_load_p<bsz::b256, std::complex<double>>(c);
     * // Equivalent to: z = _mm256_load_pd(reinterpret_cast<const double*>(c))
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Dense direct unaligned load to AVX register
     *
     * Loads a full vector from an unaligned memory location into an AVX register. This function performs
     * an unaligned memory load, which does NOT require memory alignment. The implementation uses
     * `_mm_loadu_*`, `_mm256_loadu_*`, or `_mm512_loadu_*` intrinsics depending on the vector size.
     *
     * For complex types (std::complex<float> and std::complex<double>), the function uses `reinterpret_cast`
     * to treat them as their base type (float or double respectively), as complex values are stored
     * contiguously in memory as [real, imag, real, imag, ...].
     *
     * For SSE integer types, this function uses `_mm_lddqu_si128`, which is optimized for loads that
     * may cross cache-line boundaries.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, int, std::complex<float>, std::complex<double>, etc.)
     *
     * @param[in] a Pointer to memory location to load from (no alignment requirement)
     *
     * @return avxvector_t<SZ, SUF> An AVX vector register with the loaded data
     *
     * @note This function is force-inlined for optimal performance
     * @note NO alignment requirement - the pointer `a` can point to any memory address
     * @note Unaligned loads may be slightly slower than aligned loads on some architectures
     * @note For aligned memory access (better performance), use kt_load_p() instead
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors (_mm_loadu_*, _mm_lddqu_si128 for integers)
     *       - SZ == bsz::b256: AVX 256-bit vectors (_mm256_loadu_*)
     *       - SZ == bsz::b512: AVX-512 512-bit vectors (_mm512_loadu_*)
     * @note Supported data types: float, double, int32_t, int64_t, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Load 4 doubles from unaligned memory (offset by 1 element)
     * double a[8] = {0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
     * avxvector_t<bsz::b256, double> v = kt_loadu_p<bsz::b256, double>(&a[1]);
     * // Equivalent to: v = _mm256_loadu_pd(&a[1])
     * // v contains {1.1, 2.2, 3.3, 4.4}
     *
     * // Load 8 floats from unaligned memory
     * float b[12] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
     * avxvector_t<bsz::b256, float> w = kt_loadu_p<bsz::b256, float>(&b[3]);
     * // Equivalent to: w = _mm256_loadu_ps(&b[3])
     * // w contains {3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}
     *
     * // Load 2 complex doubles from unaligned memory
     * std::complex<double> c[4] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}};
     * avxvector_t<bsz::b256, std::complex<double>> z = kt_loadu_p<bsz::b256, std::complex<double>>(&c[1]);
     * // Equivalent to: z = _mm256_loadu_pd(reinterpret_cast<const double*>(&c[1]))
     * // z contains {c[1], c[2]} = {{3.0, 4.0}, {5.0, 6.0}}
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Store AVX register to unaligned memory location
     *
     * Stores the contents of an AVX vector register to an unaligned memory location. This function
     * performs an unaligned memory store, which does NOT require memory alignment. The implementation
     * uses `_mm_storeu_*`, `_mm256_storeu_*`, or `_mm512_storeu_*` intrinsics depending on the vector size.
     *
     * For complex types (std::complex<float> and std::complex<double>), the function uses `reinterpret_cast`
     * to treat them as their base type (float or double respectively), as complex values are stored
     * contiguously in memory as [real, imag, real, imag, ...].
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, int, std::complex<float>, std::complex<double>, etc.)
     *
     * @param[out] a Pointer to memory location to store to (no alignment requirement)
     * @param[in] v AVX vector register containing the data to store
     *
     * @return void
     *
     * @note This function is force-inlined for optimal performance
     * @note NO alignment requirement - the pointer `a` can point to any memory address
     * @note Unaligned stores may be slightly slower than aligned stores on some architectures
     * @note For aligned memory writes (better performance), use kt_store_p() if available
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors (_mm_storeu_*)
     *       - SZ == bsz::b256: AVX 256-bit vectors (_mm256_storeu_*)
     *       - SZ == bsz::b512: AVX-512 512-bit vectors (_mm512_storeu_*)
     * @note Supported data types: float, double, std::complex<float>, std::complex<double>
     * @note Integer types are supported for SSE/AVX2 (stored via float/double intrinsics)
     *
     * @par Example:
     * @code
     * // Store 4 doubles to unaligned memory (offset by 1 element)
     * double a[8];
     * avxvector_t<bsz::b256, double> v = kt_set1_p<bsz::b256, double>(3.14);
     * kt_storeu_p<bsz::b256, double>(&a[1], v);
     * // Equivalent to: _mm256_storeu_pd(&a[1], v)
     * // a[1], a[2], a[3], a[4] now contain 3.14
     *
     * // Store 8 floats to unaligned memory
     * float b[12];
     * avxvector_t<bsz::b256, float> w = kt_loadu_p<bsz::b256, float>(source);
     * kt_storeu_p<bsz::b256, float>(&b[3], w);
     * // Equivalent to: _mm256_storeu_ps(&b[3], w)
     *
     * // Store 2 complex doubles to unaligned memory
     * std::complex<double> c[4];
     * avxvector_t<bsz::b256, std::complex<double>> z = kt_load_p<bsz::b256, std::complex<double>>(source);
     * kt_storeu_p<bsz::b256, std::complex<double>>(&c[1], z);
     * // Equivalent to: _mm256_storeu_pd(reinterpret_cast<double*>(&c[1]), z)
     * // c[1] and c[2] now contain the values from z
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, void>
                    kt_storeu_p(SUF *a, const avxvector_t<SZ, SUF> v) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, void>
                    kt_storeu_p(const SUF *, const avxvector_t<SZ, SUF>) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, void>
                    kt_storeu_p(const SUF *, const avxvector_t<SZ, SUF>) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Vector addition of two AVX registers
     *
     * Performs element-wise addition of two AVX vector registers. This function is specialized for
     * different vector sizes (128-bit, 256-bit, and 512-bit) and supports various data types including
     * float, double, and their complex counterparts. The implementation uses `_mm_add_*`, `_mm256_add_*`,
     * or `_mm512_add_*` intrinsics depending on the vector size and data type.
     *
     * For complex types, the addition is performed on the underlying float or double representation,
     * which correctly implements complex addition since complex numbers are stored contiguously as
     * [real, imag, real, imag, ...] and addition is element-wise.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, std::complex<float>, std::complex<double>, etc.)
     *
     * @param[in] a First AVX vector register operand
     * @param[in] b Second AVX vector register operand
     *
     * @return avxvector_t<SZ, SUF> An AVX vector containing the element-wise sum a[i] + b[i]
     *
     * @note This function is force-inlined for optimal performance
     * @note For complex types, implements: (a_real + b_real) + i(a_imag + b_imag)
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors (_mm_add_ps/_mm_add_pd)
     *       - SZ == bsz::b256: AVX 256-bit vectors (_mm256_add_ps/_mm256_add_pd)
     *       - SZ == bsz::b512: AVX-512 512-bit vectors (_mm512_add_ps/_mm512_add_pd)
     * @note Supported data types: float, double, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Add two 256-bit vectors of doubles
     * avxvector_t<bsz::b256, double> a = kt_set1_p<bsz::b256, double>(1.0);
     * avxvector_t<bsz::b256, double> b = kt_set1_p<bsz::b256, double>(2.0);
     * avxvector_t<bsz::b256, double> c = kt_add_p<bsz::b256, double>(a, b);
     * // Equivalent to: c = _mm256_add_pd(a, b)
     * // c contains {3.0, 3.0, 3.0, 3.0}
     *
     * // Add two 256-bit vectors of floats
     * avxvector_t<bsz::b256, float> x = kt_loadu_p<bsz::b256, float>(arr1);
     * avxvector_t<bsz::b256, float> y = kt_loadu_p<bsz::b256, float>(arr2);
     * avxvector_t<bsz::b256, float> z = kt_add_p<bsz::b256, float>(x, y);
     * // Equivalent to: z = _mm256_add_ps(x, y)
     *
     * // Add two 256-bit vectors of complex doubles
     * avxvector_t<bsz::b256, std::complex<double>> ca = kt_load_p<bsz::b256, std::complex<double>>(comp_arr1);
     * avxvector_t<bsz::b256, std::complex<double>> cb = kt_load_p<bsz::b256, std::complex<double>>(comp_arr2);
     * avxvector_t<bsz::b256, std::complex<double>> cc = kt_add_p<bsz::b256, std::complex<double>>(ca, cb);
     * // Equivalent to: cc = _mm256_add_pd(ca, cb)
     * // Performs complex addition: (a.real + b.real) + i(a.imag + b.imag) for each element
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Vector subtraction of two AVX registers
     *
     * Performs element-wise subtraction of two AVX vector registers. This function is specialized for
     * different vector sizes (128-bit, 256-bit, and 512-bit) and supports various data types including
     * float, double, and their complex counterparts. The implementation uses `_mm_sub_*`, `_mm256_sub_*`,
     * or `_mm512_sub_*` intrinsics depending on the vector size and data type.
     *
     * For complex types, the subtraction is performed on the underlying float or double representation,
     * which correctly implements complex subtraction since complex numbers are stored contiguously as
     * [real, imag, real, imag, ...] and subtraction is element-wise.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, std::complex<float>, std::complex<double>, etc.)
     *
     * @param[in] a First AVX vector register operand (minuend)
     * @param[in] b Second AVX vector register operand (subtrahend)
     *
     * @return avxvector_t<SZ, SUF> An AVX vector containing the element-wise difference a[i] - b[i]
     *
     * @note This function is force-inlined for optimal performance
     * @note For complex types, implements: (a_real - b_real) + i(a_imag - b_imag)
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors (_mm_sub_ps/_mm_sub_pd)
     *       - SZ == bsz::b256: AVX 256-bit vectors (_mm256_sub_ps/_mm256_sub_pd)
     *       - SZ == bsz::b512: AVX-512 512-bit vectors (_mm512_sub_ps/_mm512_sub_pd)
     * @note Supported data types: float, double, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Subtract two 256-bit vectors of doubles
     * avxvector_t<bsz::b256, double> a = kt_set1_p<bsz::b256, double>(5.0);
     * avxvector_t<bsz::b256, double> b = kt_set1_p<bsz::b256, double>(3.0);
     * avxvector_t<bsz::b256, double> c = kt_sub_p<bsz::b256, double>(a, b);
     * // Equivalent to: c = _mm256_sub_pd(a, b)
     * // c contains {2.0, 2.0, 2.0, 2.0}
     *
     * // Subtract two 256-bit vectors of floats
     * avxvector_t<bsz::b256, float> x = kt_loadu_p<bsz::b256, float>(arr1);
     * avxvector_t<bsz::b256, float> y = kt_loadu_p<bsz::b256, float>(arr2);
     * avxvector_t<bsz::b256, float> z = kt_sub_p<bsz::b256, float>(x, y);
     * // Equivalent to: z = _mm256_sub_ps(x, y)
     *
     * // Subtract two 256-bit vectors of complex doubles
     * avxvector_t<bsz::b256, std::complex<double>> ca = kt_load_p<bsz::b256, std::complex<double>>(comp_arr1);
     * avxvector_t<bsz::b256, std::complex<double>> cb = kt_load_p<bsz::b256, std::complex<double>>(comp_arr2);
     * avxvector_t<bsz::b256, std::complex<double>> cc = kt_sub_p<bsz::b256, std::complex<double>>(ca, cb);
     * // Equivalent to: cc = _mm256_sub_pd(ca, cb)
     * // Performs complex subtraction: (a.real - b.real) + i(a.imag - b.imag) for each element
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Vector multiplication of two AVX registers
     *
     * Performs element-wise multiplication of two AVX vector registers. This function is specialized for
     * different vector sizes (128-bit, 256-bit, and 512-bit) and supports various data types including
     * float, double, and their complex counterparts. The implementation uses `_mm_mul_*`, `_mm256_mul_*`,
     * or `_mm512_mul_*` intrinsics depending on the vector size and data type.
     *
     * For real types (float, double), this performs simple element-wise multiplication.
     * For complex types (std::complex<float>, std::complex<double>), this implements proper complex
     * multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i, using optimized shuffle and
     * fused multiply-add/sub operations.
     *
     * @tparam SZ Vector size in bits (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF Suffix of working type (float, double, std::complex<float>, std::complex<double>, etc.)
     *
     * @param[in] a First AVX vector register operand
     * @param[in] b Second AVX vector register operand
     *
     * @return avxvector_t<SZ, SUF> An AVX vector containing the element-wise product a[i] * b[i]
     *
     * @note This function is force-inlined for optimal performance
     * @note For real types, performs: result[i] = a[i] * b[i]
     * @note For complex types, implements: (a_real + a_imag*i) * (b_real + b_imag*i) =
     *       (a_real*b_real - a_imag*b_imag) + (a_real*b_imag + a_imag*b_real)*i
     * @note Complex multiplication uses specialized shuffle operations and fused multiply-add/sub
     * @note Available specializations:
     *       - SZ == bsz::b128: SSE 128-bit vectors (_mm_mul_ps/_mm_mul_pd + complex handling)
     *       - SZ == bsz::b256: AVX 256-bit vectors (_mm256_mul_ps/_mm256_mul_pd + complex handling)
     *       - SZ == bsz::b512: AVX-512 512-bit vectors (_mm512_mul_ps/_mm512_mul_pd + complex handling)
     * @note Supported data types: float, double, std::complex<float>, std::complex<double>
     *
     * @par Example:
     * @code
     * // Multiply two 256-bit vectors of doubles
     * avxvector_t<bsz::b256, double> a = kt_set1_p<bsz::b256, double>(2.0);
     * avxvector_t<bsz::b256, double> b = kt_set1_p<bsz::b256, double>(3.0);
     * avxvector_t<bsz::b256, double> c = kt_mul_p<bsz::b256, double>(a, b);
     * // Equivalent to: c = _mm256_mul_pd(a, b)
     * // c contains {6.0, 6.0, 6.0, 6.0}
     *
     * // Multiply two 256-bit vectors of floats
     * avxvector_t<bsz::b256, float> x = kt_loadu_p<bsz::b256, float>(arr1);
     * avxvector_t<bsz::b256, float> y = kt_loadu_p<bsz::b256, float>(arr2);
     * avxvector_t<bsz::b256, float> z = kt_mul_p<bsz::b256, float>(x, y);
     * // Equivalent to: z = _mm256_mul_ps(x, y)
     *
     * // Multiply two 256-bit vectors of complex doubles
     * // Example: (2+3i) * (4+5i) = (8-15) + (10+12)i = -7 + 22i
     * avxvector_t<bsz::b256, std::complex<double>> ca = kt_load_p<bsz::b256, std::complex<double>>(comp_arr1);
     * avxvector_t<bsz::b256, std::complex<double>> cb = kt_load_p<bsz::b256, std::complex<double>>(comp_arr2);
     * avxvector_t<bsz::b256, std::complex<double>> cc = kt_mul_p<bsz::b256, std::complex<double>>(ca, cb);
     * // Uses optimized implementation with _mm256_movedup_pd, _mm256_permute_pd, _mm256_fmaddsub_pd
     * // Performs complex multiplication: (a.real*b.real - a.imag*b.imag) + (a.real*b.imag + a.imag*b.real)i
     * @endcode
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Performs fused multiply-add operation: a * b + c
     *
     * Computes the fused multiply-add of three SIMD vector registers. For real types,
     * uses hardware FMA instructions that compute a*b+c in a single operation with
     * higher precision (no intermediate rounding). For complex types, the operation
     * is computed as mul(a, b) + c using kt_mul_p and kt_add_p.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, std::complex<float>, or std::complex<double>)
     *
     * @param[in] a First input vector (multiplier)
     * @param[in] b Second input vector (multiplicand)
     * @param[in] c Third input vector (addend)
     *
     * @return Vector containing the element-wise result of a * b + c
     *
     * @note For real types (float, double):
     *       - Uses native FMA intrinsics (_mm*_fmadd_pd/_mm*_fmadd_ps)
     *       - Provides better accuracy than separate multiply and add
     *       - Single rounding instead of two separate rounding operations
     * @note For complex types (std::complex<float>, std::complex<double>):
     *       - Implemented as kt_add_p(kt_mul_p(a, b), c)
     *       - Uses complex multiplication algorithm followed by addition
     *       - Maintains complex arithmetic semantics: (a+bi)*(c+di)+(e+fi)
     *
     * @par Example:
     * @code
     * // For real numbers:
     * avxvector_t<bsz::b256, double> a = kt_set1_p<bsz::b256, double>(2.0);  // [2.0, 2.0, 2.0, 2.0]
     * avxvector_t<bsz::b256, double> b = kt_set1_p<bsz::b256, double>(3.0);  // [3.0, 3.0, 3.0, 3.0]
     * avxvector_t<bsz::b256, double> c = kt_set1_p<bsz::b256, double>(4.0);  // [4.0, 4.0, 4.0, 4.0]
     * avxvector_t<bsz::b256, double> result = kt_fmadd_p<bsz::b256, double>(a, b, c);
     * // result = [10.0, 10.0, 10.0, 10.0]  (2.0*3.0+4.0 = 10.0)
     *
     * // For complex numbers:
     * std::complex<double> ca(1.0, 2.0);  // 1+2i
     * std::complex<double> cb(3.0, 4.0);  // 3+4i
     * std::complex<double> cc(5.0, 6.0);  // 5+6i
     * avxvector_t<bsz::b256, cdouble> va = kt_set1_p<bsz::b256, cdouble>(ca);
     * avxvector_t<bsz::b256, cdouble> vb = kt_set1_p<bsz::b256, cdouble>(cb);
     * avxvector_t<bsz::b256, cdouble> vc = kt_set1_p<bsz::b256, cdouble>(cc);
     * avxvector_t<bsz::b256, cdouble> result = kt_fmadd_p<bsz::b256, cdouble>(va, vb, vc);
     * // result = [0+16i, 0+16i]  ((1+2i)*(3+4i)+(5+6i) = (-5+10i)+(5+6i) = 0+16i)
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - double: _mm_fmadd_pd
     *   - float: _mm_fmadd_ps
     *   - cdouble/cfloat: kt_add_p(kt_mul_p(a, b), c)
     * - AVX2 (bsz::b256):
     *   - double: _mm256_fmadd_pd
     *   - float: _mm256_fmadd_ps
     *   - cdouble/cfloat: kt_add_p(kt_mul_p(a, b), c)
     * - AVX-512 (bsz::b512):
     *   - double: _mm512_fmadd_pd
     *   - float: _mm512_fmadd_ps
     *   - cdouble/cfloat: kt_add_p(kt_mul_p(a, b), c)
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_fmadd_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept;

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

    /**
     * @brief Performs fused multiply-subtract operation: a * b - c
     *
     * Computes the fused multiply-subtract of three SIMD vector registers. For real types,
     * uses hardware FMS instructions that compute a*b-c in a single operation with
     * higher precision (no intermediate rounding). For complex types, the operation
     * is computed as mul(a, b) - c using kt_mul_p and kt_sub_p.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, std::complex<float>, or std::complex<double>)
     *
     * @param[in] a First input vector (multiplier)
     * @param[in] b Second input vector (multiplicand)
     * @param[in] c Third input vector (subtrahend)
     *
     * @return Vector containing the element-wise result of a * b - c
     *
     * @note For real types (float, double):
     *       - Uses native FMS intrinsics (_mm*_fmsub_pd/_mm*_fmsub_ps)
     *       - Provides better accuracy than separate multiply and subtract
     *       - Single rounding instead of two separate rounding operations
     * @note For complex types (std::complex<float>, std::complex<double>):
     *       - Implemented as kt_sub_p(kt_mul_p(a, b), c)
     *       - Uses complex multiplication algorithm followed by subtraction
     *       - Maintains complex arithmetic semantics: (a+bi)*(c+di)-(e+fi)
     *
     * @par Example:
     * @code
     * // For real numbers:
     * avxvector_t<bsz::b256, double> a = kt_set1_p<bsz::b256, double>(5.0);  // [5.0, 5.0, 5.0, 5.0]
     * avxvector_t<bsz::b256, double> b = kt_set1_p<bsz::b256, double>(3.0);  // [3.0, 3.0, 3.0, 3.0]
     * avxvector_t<bsz::b256, double> c = kt_set1_p<bsz::b256, double>(2.0);  // [2.0, 2.0, 2.0, 2.0]
     * avxvector_t<bsz::b256, double> result = kt_fmsub_p<bsz::b256, double>(a, b, c);
     * // result = [13.0, 13.0, 13.0, 13.0]  (5.0*3.0-2.0 = 13.0)
     *
     * // For complex numbers:
     * std::complex<double> ca(2.0, 3.0);  // 2+3i
     * std::complex<double> cb(4.0, 5.0);  // 4+5i
     * std::complex<double> cc(1.0, 1.0);  // 1+1i
     * avxvector_t<bsz::b256, cdouble> va = kt_set1_p<bsz::b256, cdouble>(ca);
     * avxvector_t<bsz::b256, cdouble> vb = kt_set1_p<bsz::b256, cdouble>(cb);
     * avxvector_t<bsz::b256, cdouble> vc = kt_set1_p<bsz::b256, cdouble>(cc);
     * avxvector_t<bsz::b256, cdouble> result = kt_fmsub_p<bsz::b256, cdouble>(va, vb, vc);
     * // result = [-8+21i, -8+21i]  ((2+3i)*(4+5i)-(1+1i) = -7+22i-1-1i = -8+21i)
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - double: _mm_fmsub_pd
     *   - float: _mm_fmsub_ps
     *   - cdouble/cfloat: kt_sub_p(kt_mul_p(a, b), c)
     * - AVX2 (bsz::b256):
     *   - double: _mm256_fmsub_pd
     *   - float: _mm256_fmsub_ps
     *   - cdouble/cfloat: kt_sub_p(kt_mul_p(a, b), c)
     * - AVX-512 (bsz::b512):
     *   - double: _mm512_fmsub_pd
     *   - float: _mm512_fmsub_ps
     *   - cdouble/cfloat: kt_sub_p(kt_mul_p(a, b), c)
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_fmsub_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept;

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

    /**
     * @brief Computes horizontal sum (reduction) of all elements in a SIMD vector
     *
     * Performs a reduction operation that sums all elements within a SIMD vector register,
     * returning a single scalar value. The implementation varies by architecture and data type,
     * using specialized intrinsics for optimal performance.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, std::complex<float>, or std::complex<double>)
     *
     * @param[in] v Input SIMD vector to be reduced
     *
     * @return Scalar value containing the sum of all vector elements
     *
     * @note For real types:
     *       - SSE (bsz::b128): Manual element access and addition (v[0]+v[1] for double, v[0]+v[1]+v[2]+v[3] for float)
     *       - AVX2 (bsz::b256): Uses _mm256_hadd_* followed by lane extraction and final addition
     *       - AVX-512 (bsz::b512): Uses _mm512_reduce_add_* for efficient single-instruction reduction
     * @note For complex types (std::complex<float>, std::complex<double>):
     *       - Returns complex result where both real and imaginary parts are independently summed
     *       - SSE (bsz::b128):
     *         - cdouble: No sum needed (single complex number), returns cdouble(v[0], v[1])
     *         - cfloat: Returns cfloat(v[0]+v[2], v[1]+v[3]) - 2 complex elements
     *       - AVX2/AVX-512: Uses permutation and addition operations to accumulate real/imaginary components separately
     *       - Implementation uses _mm*_permute* and _mm*_add_* intrinsics to maintain complex structure
     *
     * @par Example:
     * @code
     * // For real numbers (double):
     * double data[4] = {1.0, 2.0, 3.0, 4.0};
     * avxvector_t<bsz::b256, double> v = kt_loadu_p<bsz::b256, double>(data);
     * double sum = kt_hsum_p<bsz::b256, double>(v);
     * // sum = 10.0  (1.0 + 2.0 + 3.0 + 4.0)
     *
     * // For real numbers (float):
     * float data_f[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
     * avxvector_t<bsz::b256, float> vf = kt_loadu_p<bsz::b256, float>(data_f);
     * float sum_f = kt_hsum_p<bsz::b256, float>(vf);
     * // sum_f = 36.0f  (1.0 + 2.0 + ... + 8.0)
     *
     * // For complex numbers:
     * std::complex<double> cdata[2] = {{1.0, 2.0}, {3.0, 4.0}};  // (1+2i), (3+4i)
     * avxvector_t<bsz::b256, cdouble> vc = kt_loadu_p<bsz::b256, cdouble>(reinterpret_cast<cdouble*>(cdata));
     * std::complex<double> csum = kt_hsum_p<bsz::b256, cdouble>(vc);
     * // csum = (4+6i)  (1+2i + 3+4i)
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - double: Manual sum v[0] + v[1] (2 elements)
     *   - float: Manual sum v[0] + v[1] + v[2] + v[3] (4 elements)
     *   - cdouble: Direct cast cdouble(v[0], v[1]) (1 complex element)
     *   - cfloat: cfloat(v[0]+v[2], v[1]+v[3]) (2 complex elements)
     * - AVX2 (bsz::b256):
     *   - double: _mm256_hadd_pd + _mm256_castpd256_pd128 + _mm256_extractf128_pd + _mm_add_pd
     *   - float: _mm256_hadd_ps (twice) + _mm256_castps256_ps128 + _mm256_extractf128_ps + _mm_add_ps
     *   - cdouble: _mm256_permute4x64_pd + _mm256_add_pd (sums 2 complex elements)
     *   - cfloat: _mm256_permute_ps + _mm256_permutevar8x32_ps + _mm256_add_ps (sums 4 complex elements)
     * - AVX-512 (bsz::b512):
     *   - double: _mm512_reduce_add_pd (8 elements)
     *   - float: _mm512_reduce_add_ps (16 elements)
     *   - cdouble: _mm512_permutex_pd + _mm512_permutexvar_pd + _mm512_add_pd (sums 4 complex elements)
     *   - cfloat: _mm512_permute_ps + _mm512_permutexvar_ps + _mm512_add_ps (sums 8 complex elements)
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Computes the complex conjugate of vector elements
     *
     * For complex types, negates the imaginary part of each complex number in the vector,
     * computing the conjugate: conj(a+bi) = a-bi. For real types, returns the input unchanged
     * since real numbers are their own conjugates. The implementation uses XOR with a sign mask
     * to efficiently flip the sign bit of imaginary components.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, std::complex<float>, or std::complex<double>)
     *
     * @param[in] a Input SIMD vector
     *
     * @return For complex types: Vector with conjugated elements
     *         For real types: Input vector unchanged
     *
     * @note For real types (float, double):
     *       - Returns input unchanged (real numbers are their own conjugates)
     *       - No computational overhead
     * @note For complex types (std::complex<float>, std::complex<double>):
     *       - Uses XOR operation with sign mask to negate imaginary parts
     *       - Mask pattern: [0.0, -0.0, 0.0, -0.0, ...] to selectively flip sign bits
     *       - Efficient bit-level operation without arithmetic instructions
     *       - Memory layout: Complex numbers stored as [real0, imag0, real1, imag1, ...]
     *       - XOR with -0.0 flips the sign bit of imaginary components only
     *
     * @par Example:
     * @code
     * // For real numbers (unchanged):
     * avxvector_t<bsz::b256, double> vr = kt_set1_p<bsz::b256, double>(3.5);
     * avxvector_t<bsz::b256, double> result_r = kt_conj_p<bsz::b256, double>(vr);
     * // result_r = [3.5, 3.5, 3.5, 3.5]  (unchanged)
     *
     * // For complex numbers:
     * std::complex<double> c1(3.0, 4.0);  // 3+4i
     * std::complex<double> c2(1.0, -2.0); // 1-2i
     * std::complex<double> cdata[2] = {c1, c2};
     * avxvector_t<bsz::b256, cdouble> vc = kt_loadu_p<bsz::b256, cdouble>(reinterpret_cast<cdouble*>(cdata));
     * avxvector_t<bsz::b256, cdouble> result_c = kt_conj_p<bsz::b256, cdouble>(vc);
     * // result_c contains: [(3-4i), (1+2i)]
     * // Memory layout: [3.0, -4.0, 1.0, 2.0]
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - float/double: Returns input unchanged
     *   - cfloat: _mm_setr_ps(0.f, -0.f, 0.f, -0.f) mask + _mm_xor_ps
     *   - cdouble: _mm_setr_pd(0.0, -0.0) mask + _mm_xor_pd
     * - AVX2 (bsz::b256):
     *   - float/double: Returns input unchanged
     *   - cfloat: _mm256_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f) mask + _mm256_xor_ps
     *   - cdouble: _mm256_setr_pd(0.0, -0.0, 0.0, -0.0) mask + _mm256_xor_pd
     * - AVX-512 (bsz::b512):
     *   - float/double: Returns input unchanged
     *   - cfloat: _mm512_setr_ps(0.f, -0.f, ...) mask (16 values) + _mm512_xor_ps
     *   - cdouble: _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0) mask + _mm512_xor_pd
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_conj_p(const avxvector_t<SZ, SUF> a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_conj_p(const avxvector_t<SZ, SUF> a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_conj_p(const avxvector_t<SZ, SUF> a) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Performs blocked variant of fused multiply-add operation
     *
     * This is a specialized "blocked" variant of fused multiply-add that operates differently
     * depending on the data type. For real types, it behaves identically to kt_fmadd_p,
     * computing c = a*b + c. For complex types, it performs two separate operations to
     * support blocked complex arithmetic where real and imaginary parts are processed
     * separately for better vectorization.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, std::complex<float>, or std::complex<double>)
     *
     * @param[in]     a First input vector (multiplier)
     * @param[in]     b Second input vector (multiplicand)
     * @param[in,out] c Third input/output vector (accumulator for real types or real parts for complex)
     * @param[in,out] d Fourth input/output vector (used only for complex types to accumulate imaginary contributions)
     *
     * @return void (results stored in c and d parameters)
     *
     * @note For real types (float, double):
     *       - Equivalent to: c = kt_fmadd_p(a, b, c)
     *       - Parameter d is unused ([[maybe_unused]])
     *       - Single FMA operation: c = a*b + c
     * @note For complex types (std::complex<float>, std::complex<double>):
     *       - Two separate FMA operations for blocked complex arithmetic:
     *         1. c = _mm*_fmadd_*(a, b, c) - standard FMA
     *         2. d = _mm*_fmadd_*(a, permute(b), d) - FMA with permuted b
     *       - Permutation pattern swaps real/imaginary pairs within each complex number
     *       - cdouble: Uses _mm*_permute_pd(b, 0b0101) to swap pairs
     *       - cfloat: Uses _mm*_permute_ps(b, 0b10110001) to swap pairs
     *       - Supports efficient blocked complex matrix operations where real/imaginary
     *         parts are stored and processed separately
     * @note SSE (bsz::b128):
     *       - Currently simplified to use kt_fmadd_p for all types (complex blocked variant commented out)
     *       - Will be enabled when kt_hsum_B is fully implemented for bsz::b128
     *
     * @par Example:
     * @code
     * // For real numbers:
     * avxvector_t<bsz::b256, double> a = kt_set1_p<bsz::b256, double>(2.0);
     * avxvector_t<bsz::b256, double> b = kt_set1_p<bsz::b256, double>(3.0);
     * avxvector_t<bsz::b256, double> c = kt_set1_p<bsz::b256, double>(1.0);
     * avxvector_t<bsz::b256, double> d = kt_setzero_p<bsz::b256, double>();
     * kt_fmadd_B<bsz::b256, double>(a, b, c, d);
     * // c = [7.0, 7.0, 7.0, 7.0]  (2.0*3.0+1.0)
     * // d is unused
     *
     * // For complex numbers (blocked representation):
     * // Input: a contains real parts, b contains both real and imaginary
     * avxvector_t<bsz::b256, cdouble> a_complex = ...;  // complex vector
     * avxvector_t<bsz::b256, cdouble> b_complex = ...;  // complex vector
     * avxvector_t<bsz::b256, cdouble> c_acc = kt_setzero_p<bsz::b256, cdouble>();
     * avxvector_t<bsz::b256, cdouble> d_acc = kt_setzero_p<bsz::b256, cdouble>();
     * kt_fmadd_B<bsz::b256, cdouble>(a_complex, b_complex, c_acc, d_acc);
     * // c_acc and d_acc now contain accumulated results
     * // c_acc: a * b + c_acc
     * // d_acc: a * permute(b) + d_acc
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - All types: Currently uses kt_fmadd_p (simplified implementation)
     *   - Future complex implementation will use _mm_permute_pd/ps
     * - AVX2 (bsz::b256):
     *   - float/double: c = kt_fmadd_p(a, b, c), d unused
     *   - cdouble: c = _mm256_fmadd_pd(a, b, c); d = _mm256_fmadd_pd(a, _mm256_permute_pd(b, 0b0101), d)
     *   - cfloat: c = _mm256_fmadd_ps(a, b, c); d = _mm256_fmadd_ps(a, _mm256_permute_ps(b, 0b10110001), d)
     * - AVX-512 (bsz::b512):
     *   - float/double: c = kt_fmadd_p(a, b, c), d unused
     *   - cdouble: c = _mm512_fmadd_pd(a, b, c); d = _mm512_fmadd_pd(a, _mm512_permute_pd(b, 0b01010101), d)
     *   - cfloat: c = _mm512_fmadd_ps(a, b, c); d = _mm512_fmadd_ps(a, _mm512_permute_ps(b, 0b10110001), d)
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, void>
                    kt_fmadd_B(const avxvector_t<SZ, SUF>             a,
                               const avxvector_t<SZ, SUF>             b,
                               avxvector_t<SZ, SUF>                  &c,
                               [[maybe_unused]] avxvector_t<SZ, SUF> &d) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, void>
                    kt_fmadd_B(const avxvector_t<SZ, SUF>             a,
                               const avxvector_t<SZ, SUF>             b,
                               avxvector_t<SZ, SUF>                  &c,
                               [[maybe_unused]] avxvector_t<SZ, SUF> &d) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, void>
                    kt_fmadd_B(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               avxvector_t<SZ, SUF>      &c,
                               avxvector_t<SZ, SUF>      &d) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Computes horizontal sum (reduction) with blocked variant for complex types
     *
     * This is a specialized "blocked" variant of horizontal sum that operates differently
     * depending on the data type. For real types, it behaves identically to kt_hsum_p.
     * For complex types, it processes real and imaginary parts from separate input vectors
     * (a and b), which is useful for blocked storage formats where real and imaginary
     * components are stored separately for better vectorization.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, std::complex<float>, or std::complex<double>)
     *
     * @param[in] a First input vector (for real types: the data; for complex: real parts with alternating signs)
     * @param[in] b Second input vector (unused for real types; for complex: imaginary parts)
     *
     * @return Scalar value containing the horizontal sum
     *
     * @note For real types (float, double):
     *       - Equivalent to kt_hsum_p(a)
     *       - Parameter b is unused ([[maybe_unused]])
     * @note For complex types (std::complex<float>, std::complex<double>):
     *       - Processes real parts from vector a with alternating sign application
     *       - Processes imaginary parts from vector b
     *       - Real part computation: XOR with sign mask [0.0, -0.0, 0.0, -0.0, ...] then sum
     *       - Imaginary part computation: Direct horizontal sum of b
     *       - AVX2: Uses _mm256_hadd_* with lane extraction
     *       - AVX-512: Uses _mm512_reduce_add_* for efficient reduction
     *       - Returns complex number constructed from (real_sum, imag_sum)
     * @note SSE (bsz::b128):
     *       - Currently simplified to use kt_hsum_p for all types
     *       - Full blocked implementation to be added in future
     *
     * @par Example:
     * @code
     * // For real numbers:
     * double data[4] = {1.0, 2.0, 3.0, 4.0};
     * avxvector_t<bsz::b256, double> va = kt_loadu_p<bsz::b256, double>(data);
     * avxvector_t<bsz::b256, double> vb = kt_setzero_p<bsz::b256, double>();  // unused
     * double sum = kt_hsum_B<bsz::b256, double>(va, vb);
     * // sum = 10.0  (1.0 + 2.0 + 3.0 + 4.0)
     *
     * // For complex numbers (blocked storage):
     * // Assume blocked format where real and imaginary parts stored separately
     * // and processed through kt_fmadd_B operations
     * avxvector_t<bsz::b256, cdouble> a_real_parts = ...;  // Contains real contributions
     * avxvector_t<bsz::b256, cdouble> b_imag_parts = ...;  // Contains imaginary contributions
     * std::complex<double> csum = kt_hsum_B<bsz::b256, cdouble>(a_real_parts, b_imag_parts);
     * // csum = complex result with:
     * //   real part: sum of a_real_parts with alternating signs applied
     * //   imag part: sum of b_imag_parts
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - All types: Currently uses kt_hsum_p(a), parameter b unused
     * - AVX2 (bsz::b256):
     *   - float/double: kt_hsum_p(a), parameter b unused
     *   - cdouble:
     *     - Real: _mm256_xor_pd(signs, a) + _mm256_hadd_pd + lane extraction/addition
     *     - Imag: _mm256_hadd_pd(b) + lane extraction/addition
     *     - Returns cdouble(real_sum, imag_sum)
     *   - cfloat:
     *     - Real: _mm256_xor_ps(signs, a) + _mm256_hadd_ps (twice) + lane extraction/addition
     *     - Imag: _mm256_hadd_ps(b) (twice) + lane extraction/addition
     *     - Returns cfloat(real_sum, imag_sum)
     * - AVX-512 (bsz::b512):
     *   - float/double: kt_hsum_p(a), parameter b unused
     *   - cdouble:
     *     - Real: _mm512_xor_pd(signs, a) then _mm512_reduce_add_pd
     *     - Imag: _mm512_reduce_add_pd(b)
     *     - Returns cdouble(real_sum, imag_sum)
     *   - cfloat:
     *     - Real: _mm512_xor_ps(signs, a) then _mm512_reduce_add_ps
     *     - Imag: _mm512_reduce_add_ps(b)
     *     - Returns cfloat(real_sum, imag_sum)
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, SUF>
                    kt_hsum_B(const avxvector_t<SZ, SUF>                  a,
                              [[maybe_unused]] const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, SUF>
                    kt_hsum_B(const avxvector_t<SZ, SUF>                  a,
                              [[maybe_unused]] const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, SUF>
                    kt_hsum_B(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Computes element-wise maximum of two SIMD vectors
     *
     * Compares packed elements in two SIMD vectors and returns a vector containing
     * the maximum value from each corresponding element pair. This operation is only
     * available for real (non-complex) data types.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float or double only - real types)
     *
     * @param[in] a First input vector
     * @param[in] b Second input vector
     *
     * @return Vector containing element-wise maximum: result[i] = max(a[i], b[i])
     *
     * @note Only available for real types (float, double)
     * @note Not defined for complex types (std::complex<float>, std::complex<double>)
     * @note The function uses hardware-optimized comparison and selection
     * @note Return type is constrained by std::enable_if_t with kt_type_is_real<SUF>()
     * @note Handles special floating-point values according to IEEE 754 semantics:
     *       - If either operand is NaN, the result may be either operand
     *       - Distinguishes between +0.0 and -0.0 (max(+0.0, -0.0) returns +0.0)
     *
     * @par Example:
     * @code
     * // For double precision:
     * double data_a[4] = {1.0, 5.0, 3.0, 8.0};
     * double data_b[4] = {2.0, 4.0, 6.0, 7.0};
     * avxvector_t<bsz::b256, double> va = kt_loadu_p<bsz::b256, double>(data_a);
     * avxvector_t<bsz::b256, double> vb = kt_loadu_p<bsz::b256, double>(data_b);
     * avxvector_t<bsz::b256, double> vmax = kt_max_p<bsz::b256, double>(va, vb);
     * // vmax = [2.0, 5.0, 6.0, 8.0]
     * //         max(1.0,2.0), max(5.0,4.0), max(3.0,6.0), max(8.0,7.0)
     *
     * // For single precision:
     * float data_fa[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
     * float data_fb[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
     * avxvector_t<bsz::b256, float> vfa = kt_loadu_p<bsz::b256, float>(data_fa);
     * avxvector_t<bsz::b256, float> vfb = kt_loadu_p<bsz::b256, float>(data_fb);
     * avxvector_t<bsz::b256, float> vfmax = kt_max_p<bsz::b256, float>(vfa, vfb);
     * // vfmax = [8.0f, 7.0f, 6.0f, 5.0f, 5.0f, 6.0f, 7.0f, 8.0f]
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - double: _mm_max_pd (2 doubles)
     *   - float: _mm_max_ps (4 floats)
     * - AVX2 (bsz::b256):
     *   - double: _mm256_max_pd (4 doubles)
     *   - float: _mm256_max_ps (8 floats)
     * - AVX-512 (bsz::b512):
     *   - double: _mm512_max_pd (8 doubles)
     *   - float: _mm512_max_ps (16 floats)
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE
        std::enable_if_t<SZ == bsz::b128 && kt_type_is_real<SUF>(), avxvector_t<SZ, SUF>>
        kt_max_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE
        std::enable_if_t<SZ == bsz::b256 && kt_type_is_real<SUF>(), avxvector_t<SZ, SUF>>
        kt_max_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE
        std::enable_if_t<SZ == bsz::b512 && kt_type_is_real<SUF>(), avxvector_t<SZ, SUF>>
        kt_max_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Computes element-wise square (real types) or squared norm (complex types)
     *
     * For real types, computes element-wise square: a * a. For complex types, computes
     * the squared norm (squared magnitude) of each complex number: |a+bi|² = a²+b².
     * The complex result is stored with the squared norm duplicated for both real and
     * imaginary positions of each complex number.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, std::complex<float>, or std::complex<double>)
     *
     * @param[in] a Input SIMD vector
     *
     * @return For real types: Vector containing element-wise square
     *         For complex types: Vector with squared norms duplicated in real/imag positions
     *
     * @note For real types (float, double):
     *       - Simple element-wise multiplication: result[i] = a[i] * a[i]
     *       - Uses _mm*_mul_pd/_mm*_mul_ps(a, a)
     * @note For complex types (std::complex<float>, std::complex<double>):
     *       - Computes squared norm: |a+bi|² = a² + b²
     *       - Result duplicated in both real and imaginary positions
     *       - Implementation:
     *         1. Multiply as real values using kt_mul_p<SZ, base_t>(a, a) to get a², b²
     *         2. Shuffle/permute to swap real/imaginary pairs
     *         3. Add shuffled result to get a²+b² in both positions
     *       - Memory layout for result: [|z₀|², |z₀|², |z₁|², |z₁|², ...]
     *       - Useful for complex division (z/w = z*conj(w)/|w|²) and normalization (z/|z|)
     *
     * @par Example:
     * @code
     * // For real numbers:
     * double data[4] = {2.0, 3.0, 4.0, 5.0};
     * avxvector_t<bsz::b256, double> v = kt_loadu_p<bsz::b256, double>(data);
     * avxvector_t<bsz::b256, double> v_squared = kt_pow2_p<bsz::b256, double>(v);
     * // v_squared = [4.0, 9.0, 16.0, 25.0]
     *
     * // For complex numbers:
     * std::complex<double> c0(3.0, 4.0);  // 3+4i, |c0|² = 9+16 = 25
     * std::complex<double> c1(1.0, 1.0);  // 1+1i, |c1|² = 1+1 = 2
     * std::complex<double> cdata[2] = {c0, c1};
     * avxvector_t<bsz::b256, cdouble> vc = kt_loadu_p<bsz::b256, cdouble>(reinterpret_cast<cdouble*>(cdata));
     * avxvector_t<bsz::b256, cdouble> vc_norm2 = kt_pow2_p<bsz::b256, cdouble>(vc);
     * // vc_norm2 memory layout: [25.0, 25.0, 2.0, 2.0]
     * // Represents: [(25.0+25.0i), (2.0+2.0i)] where both components store the squared norm
     *
     * // Usage in complex division:
     * // To compute z/w, use: z * conj(w) / kt_pow2_p(w)
     * avxvector_t<bsz::b256, cdouble> w_norm2 = kt_pow2_p<bsz::b256, cdouble>(w);
     * avxvector_t<bsz::b256, cdouble> result = kt_div_p(kt_mul_p(z, kt_conj_p(w)), w_norm2);
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - double: _mm_mul_pd(a, a)
     *   - float: _mm_mul_ps(a, a)
     *   - cdouble: kt_mul_p<SZ, double>(a, a) + _mm_shuffle_pd(pow2, pow2, 0x1) + _mm_add_pd
     *   - cfloat: kt_mul_p<SZ, float>(a, a) + _mm_shuffle_ps(pow2, pow2, _MM_SHUFFLE(2,3,0,1)) + _mm_add_ps
     * - AVX2 (bsz::b256):
     *   - double: _mm256_mul_pd(a, a)
     *   - float: _mm256_mul_ps(a, a)
     *   - cdouble: kt_mul_p<SZ, double>(a, a) + _mm256_permute_pd(pow2, 0b0101) + _mm256_add_pd
     *   - cfloat: kt_mul_p<SZ, float>(a, a) + _mm256_permute_ps(pow2, 0b10110001) + _mm256_add_ps
     * - AVX-512 (bsz::b512):
     *   - double: _mm512_mul_pd(a, a)
     *   - float: _mm512_mul_ps(a, a)
     *   - cdouble: kt_mul_p<SZ, double>(a, a) + _mm512_permute_pd(pow2, 0x55) + _mm512_add_pd
     *   - cfloat: kt_mul_p<SZ, float>(a, a) + _mm512_permute_ps(pow2, 0xB1) + _mm512_add_ps
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_pow2_p(const avxvector_t<SZ, SUF> a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_pow2_p(const avxvector_t<SZ, SUF> a) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_pow2_p(const avxvector_t<SZ, SUF> a) noexcept;

    // -----------------------------------------------------------------------

    /**
     * @brief Performs element-wise division of two SIMD vectors
     *
     * Computes element-wise division a / b for SIMD vectors. For real types, uses
     * direct hardware division instructions. For complex types, implements complex
     * division using the formula: a/b = (a * conj(b)) / |b|², which avoids explicit
     * complex division by using multiplication and real division.
     *
     * @tparam SZ   Vector size type (bsz::b128, bsz::b256, or bsz::b512)
     * @tparam SUF  Scalar data type (float, double, std::complex<float>, or std::complex<double>)
     *
     * @param[in] a Numerator vector (dividend)
     * @param[in] b Denominator vector (divisor)
     *
     * @return Vector containing element-wise division: result[i] = a[i] / b[i]
     *
     * @note For real types (float, double):
     *       - Direct element-wise division using _mm*_div_pd/_mm*_div_ps
     *       - Standard floating-point division semantics
     *       - Division by zero produces infinity (±∞) or NaN according to IEEE 754
     * @note For complex types (std::complex<float>, std::complex<double>):
     *       - Uses mathematical formula: (a+bi)/(c+di) = (a+bi)*(c-di)/|c+di|²
     *       - Implementation steps:
     *         1. numerator = kt_mul_p(a, kt_conj_p(b)) - multiply a by conjugate of b
     *         2. denominator = kt_pow2_p(b) - compute squared norm |b|²
     *         3. result = _mm*_div_*(numerator, denominator) - divide as real values
     *       - The conjugate multiplication and squared norm transform complex division
     *         into a simpler real division operation
     *       - More numerically stable than computing real and imaginary parts separately
     *
     * @par Example:
     * @code
     * // For real numbers:
     * double data_a[4] = {10.0, 20.0, 30.0, 40.0};
     * double data_b[4] = {2.0, 4.0, 5.0, 8.0};
     * avxvector_t<bsz::b256, double> va = kt_loadu_p<bsz::b256, double>(data_a);
     * avxvector_t<bsz::b256, double> vb = kt_loadu_p<bsz::b256, double>(data_b);
     * avxvector_t<bsz::b256, double> result = kt_div_p<bsz::b256, double>(va, vb);
     * // result = [5.0, 5.0, 6.0, 5.0]  (10/2, 20/4, 30/5, 40/8)
     *
     * // For complex numbers:
     * std::complex<double> ca(10.0, 5.0);   // 10+5i
     * std::complex<double> cb(2.0, 1.0);    // 2+1i
     * // Manual calculation: (10+5i)/(2+1i) = (10+5i)*(2-1i)/(2²+1²) = (20-10i+10i-5i²)/5
     * //                                     = (20+5)/5 = (25+0i)/5 = 5+0i
     * avxvector_t<bsz::b256, cdouble> va_c = kt_set1_p<bsz::b256, cdouble>(ca);
     * avxvector_t<bsz::b256, cdouble> vb_c = kt_set1_p<bsz::b256, cdouble>(cb);
     * avxvector_t<bsz::b256, cdouble> result_c = kt_div_p<bsz::b256, cdouble>(va_c, vb_c);
     * // result_c = [(5+0i), (5+0i)]
     * @endcode
     *
     * <b>Intrinsic Equivalents:</b>
     * - SSE (bsz::b128):
     *   - double: _mm_div_pd
     *   - float: _mm_div_ps
     *   - cdouble: kt_mul_p(a, kt_conj_p(b)) then _mm_div_pd by kt_pow2_p(b)
     *   - cfloat: kt_mul_p(a, kt_conj_p(b)) then _mm_div_ps by kt_pow2_p(b)
     * - AVX2 (bsz::b256):
     *   - double: _mm256_div_pd
     *   - float: _mm256_div_ps
     *   - cdouble: kt_mul_p(a, kt_conj_p(b)) then _mm256_div_pd by kt_pow2_p(b)
     *   - cfloat: kt_mul_p(a, kt_conj_p(b)) then _mm256_div_ps by kt_pow2_p(b)
     * - AVX-512 (bsz::b512):
     *   - double: _mm512_div_pd
     *   - float: _mm512_div_ps
     *   - cdouble: kt_mul_p(a, kt_conj_p(b)) then _mm512_div_pd by kt_pow2_p(b)
     *   - cfloat: kt_mul_p(a, kt_conj_p(b)) then _mm512_div_ps by kt_pow2_p(b)
     */
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_div_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_div_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;

    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_div_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept;
}
#endif
