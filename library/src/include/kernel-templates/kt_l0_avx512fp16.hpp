/* ************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * AVX-512 FP16 L0 micro kernels
 * ==============================
 *
 * All fp16 SIMD operations are collected here (128-bit, 256-bit, and 512-bit
 * widths) because they uniformly require the __AVX512FP16__ ISA extension.
 * The corresponding non-fp16 overloads live in kt_l0_sse.hpp, kt_l0_avx2.hpp,
 * and kt_l0_avx512.hpp; those files exclude fp16 via SFINAE so there is no
 * ambiguity.
 */

#ifndef KERNEL_TEMPLATES_T_HPP
#error "Never use ``kt_l0_avx512fp16.hpp'' directly; include ``kernel_templates.hpp'' instead."
#endif

// clang-format off

#ifndef _KT_AVX512FP_
#define _KT_AVX512FP_
#include "kt_common_x86.hpp"

namespace kernel_templates
{
    // Zero out a register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_setzero_ph();
        else if constexpr(SZ == bsz::b256)
            return _mm256_setzero_ph();
        else if constexpr(SZ == bsz::b512)
            return _mm512_setzero_ph();
    }

    // Fill vector with a scalar value
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_set1_ph(x);
        else if constexpr(SZ == bsz::b256)
            return _mm256_set1_ph(x);
        else if constexpr(SZ == bsz::b512)
            return _mm512_set1_ph(x);
    }

    // Unaligned set (load) with indirect memory access
    template <bsz SZ, typename SUF, typename IS, valid_kt_int<IS>>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const IS *b) noexcept
    {
        if constexpr(SZ == bsz::b128)
        {
            return _mm_set_ph(v[*(b + 7U)], v[*(b + 6U)], v[*(b + 5U)], v[*(b + 4U)],
                              v[*(b + 3U)], v[*(b + 2U)], v[*(b + 1U)], v[*(b + 0U)]);
        }
        else if constexpr(SZ == bsz::b256)
        {
            return _mm256_set_ph(v[*(b + 15U)], v[*(b + 14U)], v[*(b + 13U)], v[*(b + 12U)],
                                 v[*(b + 11U)], v[*(b + 10U)], v[*(b + 9U)],  v[*(b + 8U)],
                                 v[*(b + 7U)],  v[*(b + 6U)],  v[*(b + 5U)],  v[*(b + 4U)],
                                 v[*(b + 3U)],  v[*(b + 2U)],  v[*(b + 1U)],  v[*(b + 0U)]);
        }
        else if constexpr(SZ == bsz::b512)
        {
            return _mm512_set_ph(v[*(b + 31U)], v[*(b + 30U)], v[*(b + 29U)], v[*(b + 28U)],
                                 v[*(b + 27U)], v[*(b + 26U)], v[*(b + 25U)], v[*(b + 24U)],
                                 v[*(b + 23U)], v[*(b + 22U)], v[*(b + 21U)], v[*(b + 20U)],
                                 v[*(b + 19U)], v[*(b + 18U)], v[*(b + 17U)], v[*(b + 16U)],
                                 v[*(b + 15U)], v[*(b + 14U)], v[*(b + 13U)], v[*(b + 12U)],
                                 v[*(b + 11U)], v[*(b + 10U)], v[*(b + 9U)],  v[*(b + 8U)],
                                 v[*(b + 7U)],  v[*(b + 6U)],  v[*(b + 5U)],  v[*(b + 4U)],
                                 v[*(b + 3U)],  v[*(b + 2U)],  v[*(b + 1U)],  v[*(b + 0U)]);
        }
    }

    // Unaligned load with zero mask direct memory model.
    template <bsz SZ, typename SUF, kt_avxext EXT, int L, typename IS, valid_kt_int<IS>>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const IS b) noexcept
    {
        if constexpr(SZ == bsz::b128)
        {
            return _mm_set_ph(pz<SUF, L - 8>(v, b + 7), pz<SUF, L - 7>(v, b + 6),
                              pz<SUF, L - 6>(v, b + 5), pz<SUF, L - 5>(v, b + 4),
                              pz<SUF, L - 4>(v, b + 3), pz<SUF, L - 3>(v, b + 2),
                              pz<SUF, L - 2>(v, b + 1), pz<SUF, L - 1>(v, b + 0));
        }
        else if constexpr(SZ == bsz::b256)
        {
            return _mm256_set_ph(pz<SUF, L - 16>(v, b + 15), pz<SUF, L - 15>(v, b + 14),
                                 pz<SUF, L - 14>(v, b + 13), pz<SUF, L - 13>(v, b + 12),
                                 pz<SUF, L - 12>(v, b + 11), pz<SUF, L - 11>(v, b + 10),
                                 pz<SUF, L - 10>(v, b + 9),  pz<SUF, L - 9> (v, b + 8),
                                 pz<SUF, L - 8> (v, b + 7),  pz<SUF, L - 7> (v, b + 6),
                                 pz<SUF, L - 6> (v, b + 5),  pz<SUF, L - 5> (v, b + 4),
                                 pz<SUF, L - 4> (v, b + 3),  pz<SUF, L - 3> (v, b + 2),
                                 pz<SUF, L - 2> (v, b + 1),  pz<SUF, L - 1> (v, b + 0));
        }
        else if constexpr(SZ == bsz::b512)
        {
            return _mm512_set_ph(pz<SUF, L - 32>(v, b + 31), pz<SUF, L - 31>(v, b + 30),
                                 pz<SUF, L - 30>(v, b + 29), pz<SUF, L - 29>(v, b + 28),
                                 pz<SUF, L - 28>(v, b + 27), pz<SUF, L - 27>(v, b + 26),
                                 pz<SUF, L - 26>(v, b + 25), pz<SUF, L - 25>(v, b + 24),
                                 pz<SUF, L - 24>(v, b + 23), pz<SUF, L - 23>(v, b + 22),
                                 pz<SUF, L - 22>(v, b + 21), pz<SUF, L - 21>(v, b + 20),
                                 pz<SUF, L - 20>(v, b + 19), pz<SUF, L - 19>(v, b + 18),
                                 pz<SUF, L - 18>(v, b + 17), pz<SUF, L - 17>(v, b + 16),
                                 pz<SUF, L - 16>(v, b + 15), pz<SUF, L - 15>(v, b + 14),
                                 pz<SUF, L - 14>(v, b + 13), pz<SUF, L - 13>(v, b + 12),
                                 pz<SUF, L - 12>(v, b + 11), pz<SUF, L - 11>(v, b + 10),
                                 pz<SUF, L - 10>(v, b + 9),  pz<SUF, L - 9> (v, b + 8),
                                 pz<SUF, L - 8> (v, b + 7),  pz<SUF, L - 7> (v, b + 6),
                                 pz<SUF, L - 6> (v, b + 5),  pz<SUF, L - 5> (v, b + 4),
                                 pz<SUF, L - 4> (v, b + 3),  pz<SUF, L - 3> (v, b + 2),
                                 pz<SUF, L - 2> (v, b + 1),  pz<SUF, L - 1> (v, b + 0));
        }
    }

    // Unaligned load with zero mask indirect memory model.
    template <bsz SZ, typename SUF, kt_avxext, int L, typename IS, valid_kt_int<IS>>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const IS *b) noexcept
    {
        if constexpr(SZ == bsz::b128)
        {
            return _mm_set_ph(pz<SUF, L - 8>(v, b, IS(7)), pz<SUF, L - 7>(v, b, IS(6)),
                              pz<SUF, L - 6>(v, b, IS(5)), pz<SUF, L - 5>(v, b, IS(4)),
                              pz<SUF, L - 4>(v, b, IS(3)), pz<SUF, L - 3>(v, b, IS(2)),
                              pz<SUF, L - 2>(v, b, IS(1)), pz<SUF, L - 1>(v, b, IS(0)));
        }
        else if constexpr(SZ == bsz::b256)
        {
            return _mm256_set_ph(pz<SUF, L - 16>(v, b, IS(15)), pz<SUF, L - 15>(v, b, IS(14)),
                                 pz<SUF, L - 14>(v, b, IS(13)), pz<SUF, L - 13>(v, b, IS(12)),
                                 pz<SUF, L - 12>(v, b, IS(11)), pz<SUF, L - 11>(v, b, IS(10)),
                                 pz<SUF, L - 10>(v, b, IS(9)),  pz<SUF, L - 9> (v, b, IS(8)),
                                 pz<SUF, L - 8> (v, b, IS(7)),  pz<SUF, L - 7> (v, b, IS(6)),
                                 pz<SUF, L - 6> (v, b, IS(5)),  pz<SUF, L - 5> (v, b, IS(4)),
                                 pz<SUF, L - 4> (v, b, IS(3)),  pz<SUF, L - 3> (v, b, IS(2)),
                                 pz<SUF, L - 2> (v, b, IS(1)),  pz<SUF, L - 1> (v, b, IS(0)));
        }
        else if constexpr(SZ == bsz::b512)
        {
            return _mm512_set_ph(pz<SUF, L - 32>(v, b, IS(31)), pz<SUF, L - 31>(v, b, IS(30)),
                                 pz<SUF, L - 30>(v, b, IS(29)), pz<SUF, L - 29>(v, b, IS(28)),
                                 pz<SUF, L - 28>(v, b, IS(27)), pz<SUF, L - 27>(v, b, IS(26)),
                                 pz<SUF, L - 26>(v, b, IS(25)), pz<SUF, L - 25>(v, b, IS(24)),
                                 pz<SUF, L - 24>(v, b, IS(23)), pz<SUF, L - 23>(v, b, IS(22)),
                                 pz<SUF, L - 22>(v, b, IS(21)), pz<SUF, L - 21>(v, b, IS(20)),
                                 pz<SUF, L - 20>(v, b, IS(19)), pz<SUF, L - 19>(v, b, IS(18)),
                                 pz<SUF, L - 18>(v, b, IS(17)), pz<SUF, L - 17>(v, b, IS(16)),
                                 pz<SUF, L - 16>(v, b, IS(15)), pz<SUF, L - 15>(v, b, IS(14)),
                                 pz<SUF, L - 14>(v, b, IS(13)), pz<SUF, L - 13>(v, b, IS(12)),
                                 pz<SUF, L - 12>(v, b, IS(11)), pz<SUF, L - 11>(v, b, IS(10)),
                                 pz<SUF, L - 10>(v, b, IS(9)),  pz<SUF, L - 9> (v, b, IS(8)),
                                 pz<SUF, L - 8> (v, b, IS(7)),  pz<SUF, L - 7> (v, b, IS(6)),
                                 pz<SUF, L - 6> (v, b, IS(5)),  pz<SUF, L - 5> (v, b, IS(4)),
                                 pz<SUF, L - 4> (v, b, IS(3)),  pz<SUF, L - 3> (v, b, IS(2)),
                                 pz<SUF, L - 2> (v, b, IS(1)),  pz<SUF, L - 1> (v, b, IS(0)));
        }
    }

    // Dense direct aligned load
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_load_ph(reinterpret_cast<void const*>(a));
        else if constexpr(SZ == bsz::b256)
            return _mm256_load_ph(reinterpret_cast<void const*>(a));
        else if constexpr(SZ == bsz::b512)
            return _mm512_load_ph(reinterpret_cast<void const*>(a));
    }

    // Dense direct (un)aligned load
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_loadu_ph(reinterpret_cast<void const*>(a));
        else if constexpr(SZ == bsz::b256)
            return _mm256_loadu_ph(reinterpret_cast<void const*>(a));
        else if constexpr(SZ == bsz::b512)
            return _mm512_loadu_ph(reinterpret_cast<void const*>(a));
    }

    // Stores values to a memory location (unaligned)
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, void>
                    kt_storeu_p(SUF *a, const avxvector_t<SZ, SUF> v) noexcept
    {
        if constexpr(SZ == bsz::b128)
            _mm_storeu_ph(reinterpret_cast<void *>(a), v);
        else if constexpr(SZ == bsz::b256)
            _mm256_storeu_ph(reinterpret_cast<void *>(a), v);
        else if constexpr(SZ == bsz::b512)
            _mm512_storeu_ph(reinterpret_cast<void *>(a), v);
    }

    // Vector addition
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_add_ph(a, b);
        else if constexpr(SZ == bsz::b256)
            return _mm256_add_ph(a, b);
        else if constexpr(SZ == bsz::b512)
            return _mm512_add_ph(a, b);
    }

    // Vector subtraction
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_sub_ph(a, b);
        else if constexpr(SZ == bsz::b256)
            return _mm256_sub_ph(a, b);
        else if constexpr(SZ == bsz::b512)
            return _mm512_sub_ph(a, b);
    }

    // Vector product
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_mul_ph(a, b);
        else if constexpr(SZ == bsz::b256)
            return _mm256_mul_ph(a, b);
        else if constexpr(SZ == bsz::b512)
            return _mm512_mul_ph(a, b);
    }

    // Vector fused multiply-add
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_fmadd_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_fmadd_ph(a, b, c);
        else if constexpr(SZ == bsz::b256)
            return _mm256_fmadd_ph(a, b, c);
        else if constexpr(SZ == bsz::b512)
            return _mm512_fmadd_ph(a, b, c);
    }

    // Vector fused multiply-subtract
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_fmsub_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_fmsub_ph(a, b, c);
        else if constexpr(SZ == bsz::b256)
            return _mm256_fmsub_ph(a, b, c);
        else if constexpr(SZ == bsz::b512)
            return _mm512_fmsub_ph(a, b, c);
    }

    // Horizontal sum (reduction)
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept
    {
        if constexpr(SZ == bsz::b128)
        {
            return _mm_reduce_add_ph(v);
        }
        else if constexpr(SZ == bsz::b256)
        {
            return _mm256_reduce_add_ph(v);
        }
        else if constexpr(SZ == bsz::b512)
        {
            return _mm512_reduce_add_ph(v);
        }
    }

    // Vector fused multiply-add - blocked variant (b256 and b512 only;
    // b128 is handled by the generic kt_fmadd_B in kt_l0_sse.hpp)
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>
                                      && (SZ == bsz::b256 || SZ == bsz::b512), void>
                    kt_fmadd_B(const avxvector_t<SZ, SUF>             a,
                               const avxvector_t<SZ, SUF>             b,
                               avxvector_t<SZ, SUF>                  &c,
                               [[maybe_unused]] avxvector_t<SZ, SUF> &d) noexcept
    {
        c = kt_fmadd_p<SZ, SUF>(a, b, c);
    }

    // Compare packed elements and return packed maximum values.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_max_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_max_ph(a, b);
        else if constexpr(SZ == bsz::b256)
            return _mm256_max_ph(a, b);
        else if constexpr(SZ == bsz::b512)
            return _mm512_max_ph(a, b);
    }

    // Vector element-wise pow2
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_pow2_p(const avxvector_t<SZ, SUF> a) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_mul_ph(a, a);
        else if constexpr(SZ == bsz::b256)
            return _mm256_mul_ph(a, a);
        else if constexpr(SZ == bsz::b512)
            return _mm512_mul_ph(a, a);
    }

    // Vector element-wise division
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                    kt_div_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(SZ == bsz::b128)
            return _mm_div_ph(a, b);
        else if constexpr(SZ == bsz::b256)
            return _mm256_div_ph(a, b);
        else if constexpr(SZ == bsz::b512)
            return _mm512_div_ph(a, b);
    }
}

#endif // _KT_AVX512FP_
