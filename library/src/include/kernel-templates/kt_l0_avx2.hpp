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
#error "Never use ``kt_l0_avx2.hpp'' directly; include ``kernel_templates.hpp'' instead."
#endif

// clang-format off

#ifndef _KT_L0_AVX2_
#define _KT_L0_AVX2_
#include "kt_common_x86.hpp"

namespace kernel_templates
{
    // Zero out an AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm256_setzero_ps();
        else if constexpr(kt_is_base_t_double<SUF>())
            return _mm256_setzero_pd();
        else if constexpr(kt_is_base_t_int<SUF>())
            return _mm256_setzero_si256();
    };

    // Fill vector with a scalar value
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm256_set1_pd(x);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm256_set1_ps(x);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {

            const double r = std::real(x);
            const double i = std::imag(x);
            return _mm256_set_pd(i, r, i, r);
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {

            const float r = std::real(x);
            const float i = std::imag(x);
            // Note that loading is end -> start <=> [d c b a] <=> [i1, r1, i0, r0]
            return _mm256_set_ps(i, r, i, r, i, r, i, r);
        }
        else if constexpr(std::is_same_v<SUF, int64_t>)
            return _mm256_set1_epi64x(x);
        else if constexpr(std::is_same_v<SUF, int32_t>)
            return _mm256_set1_epi32(x);
    };

    // Unaligned set (load) to AVX register with indirect memory access
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const kt_int_t *b) noexcept
    {
        if constexpr(std::is_same<SUF, double>::value)
        {
            return _mm256_set_pd(v[*(b + 3U)], v[*(b + 2U)], v[*(b + 1U)], v[*(b + 0U)]);
        }
        else if constexpr(std::is_same<SUF, float>::value)
        {
            return _mm256_set_ps(v[*(b + 7U)], v[*(b + 6U)], v[*(b + 5U)], v[*(b + 4U)],
                                 v[*(b + 3U)], v[*(b + 2U)], v[*(b + 1U)], v[*(b + 0U)]);
        }
        else if constexpr(std::is_same<SUF, cdouble>::value)
        {
            return _mm256_set_pd(v[(*(b + 1U))].imag(), v[(*(b + 1U))].real(),
                                 v[(*(b + 0U))].imag(), v[(*(b + 0U))].real());
        }
        else if constexpr(std::is_same<SUF, cfloat>::value)
        {
            return _mm256_set_ps(v[(*(b + 3U))].imag(), v[(*(b + 3U))].real(),
                                 v[(*(b + 2U))].imag(), v[(*(b + 2U))].real(),
                                 v[(*(b + 1U))].imag(), v[(*(b + 1U))].real(),
                                 v[(*(b + 0U))].imag(), v[(*(b + 0U))].real());
        }
        else if constexpr(std::is_same_v<SUF, int64_t>)
        {
            return _mm256_set_epi64x(v[*(b + 3U)], v[*(b + 2U)], v[*(b + 1U)], v[*(b + 0U)]);
        }
        else if constexpr(std::is_same_v<SUF, int32_t>)
        {
            return _mm256_set_epi32(v[*(b + 7U)], v[*(b + 6U)], v[*(b + 5U)], v[*(b + 4U)],
                                    v[*(b + 3U)], v[*(b + 2U)], v[*(b + 1U)], v[*(b + 0U)]);
        }
    };

    // Unaligned load to AVX register with zero mask direct memory model.
    template <bsz SZ, typename SUF, kt_avxext EXT, int L>
    KT_FORCE_INLINE
        std::enable_if_t<(SZ == bsz::b256 && EXT == kt_avxext::AVX2),avxvector_t<SZ, SUF>>
        kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept
    {
        if constexpr(kt_is_same<bsz::b256, SZ, double, SUF>())
        {
            return _mm256_set_pd(pz<SUF, L - 4>(v, b + 3), pz<SUF, L - 3>(v, b + 2),
                                 pz<SUF, L - 2>(v, b + 1), pz<SUF, L - 1>(v, b + 0));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, float, SUF>())
        {
            return _mm256_set_ps(pz<SUF, L - 8>(v, b + 7), pz<SUF, L - 7>(v, b + 6),
                                 pz<SUF, L - 6>(v, b + 5), pz<SUF, L - 5>(v, b + 4),
                                 pz<SUF, L - 4>(v, b + 3), pz<SUF, L - 3>(v, b + 2),
                                 pz<SUF, L - 2>(v, b + 1), pz<SUF, L - 1>(v, b + 0));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cdouble, SUF>())
        {
            return _mm256_set_pd(pz<SUF, L - 2, false>(v, b + 1), pz<SUF, L - 2, true> (v, b + 1),
                                 pz<SUF, L - 1, false>(v, b + 0), pz<SUF, L - 1, true> (v, b + 0));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cfloat, SUF>())
        {
            return _mm256_set_ps(pz<SUF, L - 4, false>(v, b + 3), pz<SUF, L - 4, true>(v, b + 3),
                                 pz<SUF, L - 3, false>(v, b + 2), pz<SUF, L - 3, true>(v, b + 2),
                                 pz<SUF, L - 2, false>(v, b + 1), pz<SUF, L - 2, true>(v, b + 1),
                                 pz<SUF, L - 1, false>(v, b + 0), pz<SUF, L - 1, true>(v, b + 0));

        }
    };

    // Unaligned load to AVX register with zero mask indirect memory model.
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept
    {
        if constexpr(kt_is_same<bsz::b256, SZ, double, SUF>())
        {
            return _mm256_set_pd(pz<SUF, L - 4>(v, b, 3), pz<SUF, L - 3>(v, b, 2),
                                 pz<SUF, L - 2>(v, b, 1), pz<SUF, L - 1>(v, b, 0));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, float, SUF>())
        {
            return _mm256_set_ps(pz<SUF, L - 8>(v, b, 7), pz<SUF, L - 7>(v, b, 6),
                                 pz<SUF, L - 6>(v, b, 5), pz<SUF, L - 5>(v, b, 4),
                                 pz<SUF, L - 4>(v, b, 3), pz<SUF, L - 3>(v, b, 2),
                                 pz<SUF, L - 2>(v, b, 1), pz<SUF, L - 1>(v, b, 0));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cdouble, SUF>())
        {
            return _mm256_set_pd(pz<std::complex<double>, L - 2, false>(v,b,1),
                                 pz<std::complex<double>, L - 2, true> (v,b,1),
                                 pz<std::complex<double>, L - 1, false>(v,b,0),
                                 pz<std::complex<double>, L - 1, true> (v,b,0));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cfloat, SUF>())
        {
            return _mm256_set_ps(pz<std::complex<float>, L - 4, false>(v, b, 3),
                                 pz<std::complex<float>, L - 4, true> (v, b, 3),
                                 pz<std::complex<float>, L - 3, false>(v, b, 2),
                                 pz<std::complex<float>, L - 3, true> (v, b, 2),
                                 pz<std::complex<float>, L - 2, false>(v, b, 1),
                                 pz<std::complex<float>, L - 2, true> (v, b, 1),
                                 pz<std::complex<float>, L - 1, false>(v, b, 0),
                                 pz<std::complex<float>, L - 1, true> (v, b, 0));
        }
    };

    // Dense direct aligned load to AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm256_load_ps(reinterpret_cast<const float *>(a));
        else if constexpr(kt_is_base_t_double<SUF>())
            return _mm256_load_pd(reinterpret_cast<const double *>(a));
        else if constexpr(kt_is_base_t_int<SUF>())
            return _mm256_load_si256(reinterpret_cast<__m256i const*>(a));

    };

    // Dense direct (un)aligned load to AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm256_loadu_ps(reinterpret_cast<const float *>(a));
        else if constexpr(kt_is_base_t_double<SUF>())
            return _mm256_loadu_pd(reinterpret_cast<const double *>(a));
        else if constexpr(kt_is_base_t_int<SUF>())
            return _mm256_loadu_si256(reinterpret_cast<__m256i const *>(a));
    };

    // Stores the values in an AVX register to a memory location (Memory does not have to be aligned)
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, void>
                    kt_storeu_p(SUF *a, const avxvector_t<SZ, SUF> v) noexcept
    {
        if constexpr(kt_is_base_t_double<SUF>())
            _mm256_storeu_pd(reinterpret_cast<double *>(a), v);
        else
            _mm256_storeu_ps(reinterpret_cast<float *>(a), v);
    };

    // Vector addition of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm256_add_ps(a, b);
        else
            return _mm256_add_pd(a, b);
    }

    // Vector subtraction of two AVX registers.
    // Note that sub_ps takes care of types float and complex float, same for double variant.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm256_sub_ps(a, b);
        else
            return _mm256_sub_pd(a, b);
    }

    // Vector product of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm256_mul_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm256_mul_ps(a, b);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            // input vectors a = (x0+iy0, x1+iy1) and b = (a0+ib0, a1+ib1)
            // imaginary elements in the vector: half = (y0, y0, y1, y1)
            avxvector_t<bsz::b256, cdouble> half = _mm256_movedup_pd(_mm256_permute_pd(a, 0x5));
            // tmp = (a0*iy0, ib0*iy0, a1*iy1, ib1*iy1)
            avxvector_t<bsz::b256, cdouble> tmp = _mm256_mul_pd(half, b);
            // real elements in the vector: half = (x0, x0, x1, x1)
            half = _mm256_movedup_pd(a);
            // c = (x0*a0-b0*y0, x0*ib0+a0*iy0, x1*a1-b1*y1, x1*ib1+a1*iy1)
            avxvector_t<bsz::b256, cdouble> c
                = _mm256_fmaddsub_pd(half, b, _mm256_permute_pd(tmp, 0x5));
            return c;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            // input vectors a = (x0+iy0, x1+iy1, ...) and b = (a0+ib0, a1+ib1, ...)
            // imaginary elements in the vector: half = (y0, y0, y1, y1, ...)
            avxvector_t<bsz::b256, cfloat> half = _mm256_movehdup_ps(a);
            // tmp = (a0*iy0, ib0*iy0, a1*iy1, ib1*iy1, ...)
            avxvector_t<bsz::b256, cfloat> tmp = _mm256_mul_ps(half, b);
            // real elements in the vector: half = (x0, x0, x1, x1, ...)
            half = _mm256_moveldup_ps(a);
            // c = (x0*a0-b0*y0, x0*ib0+a0*iy0, x1*a1-b1*y1, x1*ib1+a1*iy1, ...)
            avxvector_t<bsz::b256, cfloat> c
                = _mm256_fmaddsub_ps(half, b, _mm256_permute_ps(tmp, 0xB1));
            return c;
        }
    }

    // Vector fused multiply-add of three AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_fmadd_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
            return _mm256_fmadd_pd(a, b, c);
        else if constexpr(std::is_same_v<SUF, float>)
            return _mm256_fmadd_ps(a, b, c);
        else if constexpr(std::is_same_v<SUF, cdouble>)
            return kt_add_p<bsz::b256, cdouble>(kt_mul_p<bsz::b256, cdouble>(a, b), c);
        else if constexpr(std::is_same_v<SUF, cfloat>)
            return kt_add_p<bsz::b256, cfloat>(kt_mul_p<bsz::b256, cfloat>(a, b), c);
    }

    // Vector fused multiply-subtract of three AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_fmsub_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
            return _mm256_fmsub_pd(a, b, c);
        else if constexpr(std::is_same_v<SUF, float>)
            return _mm256_fmsub_ps(a, b, c);
        else if constexpr(std::is_same_v<SUF, cdouble>)
            return kt_sub_p<bsz::b256, cdouble>(kt_mul_p<bsz::b256, cdouble>(a, b), c);
        else if constexpr(std::is_same_v<SUF, cfloat>)
            return kt_sub_p<bsz::b256, cfloat>(kt_mul_p<bsz::b256, cfloat>(a, b), c);
    }

    // Horizontal sum (reduction) of an AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            avxvector_half_t<bsz::b256, double> l, h, s;
            avxvector_t<bsz::b256, double>      w = _mm256_hadd_pd(v, v);
            l                                     = _mm256_castpd256_pd128(w);
            h                                     = _mm256_extractf128_pd(w, 1);
            s                                     = _mm_add_pd(l, h);
            return kt_sse_scl(s);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            avxvector_half_t<bsz::b256, float> l, h, s;
            avxvector_t<bsz::b256, float>      w = _mm256_hadd_ps(v, v);
            w                                    = _mm256_hadd_ps(w, w); // only required for float
            l                                    = _mm256_castps256_ps128(w);
            h                                    = _mm256_extractf128_ps(w, 1);
            s                                    = _mm_add_ps(l, h);
            return kt_sse_scl(s);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            // input vector v = (x0+iy0, x1+iy1)
            avxvector_t<bsz::b256, cdouble> s, tmp;
            // tmp = (x2, y2, x1, y1) upper half not relevant
            tmp = _mm256_permute4x64_pd(v, 0b1110);
            // s = (x1+x2, y1+y2) upper half not relevant
            s              = _mm256_add_pd(v, tmp);
            cdouble result = {s[0], s[1]};
            return result;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            // input vector v = (x0+iy0, x1+iy1, x2+iy2, x3+iy3)
            avxvector_t<bsz::b256, cfloat> res, s, tmp;
            // only the indexes mentioned in the last two places are relevant
            __m256i idx = _mm256_set_epi32(6, 7, 2, 3, 0, 1, 5, 4);
            // upper 128-bits of each lane are not used
            // tmp = (x1, y1, ..., x3, y3, ...)
            tmp = _mm256_permute_ps(v, 0b1110);
            // elements at indexes 0, 1, 4 and 5 hold the intermediate results
            res = _mm256_add_ps(v, tmp);
            // move elements in the 4th and 5th indexes to the 0th and 1st indexes
            tmp           = _mm256_permutevar8x32_ps(res, idx);
            s             = _mm256_add_ps(res, tmp);
            cfloat result = {s[0], s[1]};
            return result;
        }
    }

    // Templated version of the conjugate operation
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_conj_p(const avxvector_t<SZ, SUF> a) noexcept
    {
        if constexpr(std::is_floating_point<SUF>::value)
        {
            return a;
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cfloat, SUF>())
        {
            avxvector_t<SZ, SUF> mask = _mm256_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
            avxvector_t<SZ, SUF> res  = _mm256_xor_ps(mask, a);
            return res;
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cdouble, SUF>())
        {
            avxvector_t<SZ, SUF> mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
            avxvector_t<SZ, SUF> res  = _mm256_xor_pd(mask, a);
            return res;
        }
    }
    // blocked kernels, these don't return algebraic objects

    // Vector fused multiply-add of three AVX registers - blocked variant
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, void>
                    kt_fmadd_B(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               avxvector_t<SZ, SUF> &c,
                               [[maybe_unused]] avxvector_t<SZ, SUF> &d
                               ) noexcept
    {
        if constexpr(std::is_same_v<SUF, double> || std::is_same_v<SUF, float>)
            c = kt_fmadd_p<SZ, SUF>(a, b, c);
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            c = _mm256_fmadd_pd(a, b, c);
            d = _mm256_fmadd_pd(a, _mm256_permute_pd(b, 0b0101), d);
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            c = _mm256_fmadd_ps(a, b, c);
            d = _mm256_fmadd_ps(a, _mm256_permute_ps(b, 0b10110001), d);
        }
    }

    // Horizontal sum (reduction) of an AVX register - blocked variant
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, SUF>
                    kt_hsum_B(const avxvector_t<SZ, SUF> a, [[maybe_unused]] const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double> || std::is_same_v<SUF, float>)
            return kt_hsum_p<SZ, SUF>(a);
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            avxvector_half_t<bsz::b256, double> l, h, sr, si;
            avxvector_t<bsz::b256, double> signs = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
            // Real part, alternate signs and sum up
            avxvector_t<bsz::b256, double> w = _mm256_xor_pd(signs, a);
            w                                = _mm256_hadd_pd(w, w);
            l                                = _mm256_castpd256_pd128(w);
            h                                = _mm256_extractf128_pd(w, 1);
            sr                               = _mm_add_pd(l, h);
            // Imaginary part sum up
            w  = _mm256_hadd_pd(b, b);
            l  = _mm256_castpd256_pd128(w);
            h  = _mm256_extractf128_pd(w, 1);
            si = _mm_add_pd(l, h);

            return SUF(kt_sse_scl(sr), kt_sse_scl(si));
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            avxvector_half_t<bsz::b256, float> l, h, sr, si;
            avxvector_t<bsz::b256, float> signs = _mm256_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
            // Real part, alternate signs and sum up
            avxvector_t<bsz::b256, float>      w = _mm256_xor_ps(signs, a);
            w = _mm256_hadd_ps(w, w);
            w                                    = _mm256_hadd_ps(w, w); // only required for float
            l                                    = _mm256_castps256_ps128(w);
            h                                    = _mm256_extractf128_ps(w, 1);
            sr                                   = _mm_add_ps(l, h);
            // Imaginary part sum up
            w  = _mm256_hadd_ps(b, b);
            w  = _mm256_hadd_ps(w, w); // only required for float
            l  = _mm256_castps256_ps128(w);
            h  = _mm256_extractf128_ps(w, 1);
            si = _mm_add_ps(l, h);

            return SUF(kt_sse_scl(sr), kt_sse_scl(si));
        }
    }

    // Compare packed SUF elements in a and b, and returns packed maximum values.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE
        std::enable_if_t<SZ == bsz::b256 && kt_type_is_real<SUF>(), avxvector_t<SZ, SUF>>
        kt_max_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm256_max_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm256_max_ps(a, b);
        }
    }

    // Vector element-wise pow2 of an AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_pow2_p(const avxvector_t<SZ, SUF> a) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm256_mul_pd(a, a);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm256_mul_ps(a, a);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            using base_t = typename kt_dt<SUF>::base_type;

            avxvector_t<SZ, SUF> pow2 = kt_mul_p<SZ, base_t>(a, a);
            avxvector_t<SZ, SUF> shuff = _mm256_permute_pd(pow2, 0b0101);
            pow2 = _mm256_add_pd(pow2, shuff);

            return pow2;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            using base_t = typename kt_dt<SUF>::base_type;

            avxvector_t<SZ, SUF> pow2 = kt_mul_p<SZ, base_t>(a, a);
            avxvector_t<SZ, SUF> shuff = _mm256_permute_ps(pow2, 0b10110001);
            pow2 = _mm256_add_ps(pow2, shuff);

            return pow2;
        }
    }

    // Vector element-wise division of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_div_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm256_div_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm256_div_ps(a, b);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            avxvector_t<SZ, SUF> nmrtr = kt_mul_p<SZ, SUF>(a, kt_conj_p<SZ, SUF>(b));
            avxvector_t<SZ, SUF> dentr = kt_pow2_p<SZ, SUF>(b);
            avxvector_t<SZ, SUF> result = _mm256_div_pd(nmrtr, dentr);

            return result;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            avxvector_t<SZ, SUF> nmrtr = kt_mul_p<SZ, SUF>(a, kt_conj_p<SZ, SUF>(b));
            avxvector_t<SZ, SUF> dentr = kt_pow2_p<SZ, SUF>(b);
            avxvector_t<SZ, SUF> result = _mm256_div_ps(nmrtr, dentr);

            return result;
        }
    }
}
#endif
