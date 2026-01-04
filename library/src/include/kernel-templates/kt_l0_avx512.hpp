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
#error "Never use ``kt_l0_avx512.hpp'' directly; include ``kernel_templates.hpp'' instead."
#endif

// clang-format off

#ifndef _KT_L0_AVX512_
#define _KT_L0_AVX512_
#include "kt_common_x86.hpp"

namespace kernel_templates
{
    // Zero out an AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm512_setzero_ps();
        else
            return _mm512_setzero_pd();
    };

    // Fill vector with a scalar value
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_set1_pd(x);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_set1_ps(x);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            const double r = std::real(x);
            const double i = std::imag(x);
            return _mm512_set_pd(i, r, i, r, i, r, i, r);
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            const float r = std::real(x);
            const float i = std::imag(x);
            return _mm512_set_ps(i, r, i, r, i, r, i, r, i, r, i, r, i, r, i, r);
        }
    };

    // Unaligned set (load) to AVX register with indirect memory access
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const kt_int_t *b) noexcept
    {

        if constexpr(std::is_same<SUF, double>::value)
        {
            return _mm512_set_pd(v[*(b + 7U)], v[*(b + 6U)], v[*(b + 5U)], v[*(b + 4U)],
                                 v[*(b + 3U)], v[*(b + 2U)], v[*(b + 1U)], v[*(b + 0U)]);
        }
        else if constexpr(std::is_same<SUF, float>::value)
        {
            return _mm512_set_ps(v[*(b + 15U)], v[*(b + 14U)], v[*(b + 13U)], v[*(b + 12U)],
                                 v[*(b + 11U)], v[*(b + 10U)], v[*(b + 9U)],  v[*(b + 8U)],
                                 v[*(b + 7U)],  v[*(b + 6U)],  v[*(b + 5U)],  v[*(b + 4U)],
                                 v[*(b + 3U)],  v[*(b + 2U)],  v[*(b + 1U)],  v[*(b + 0U)]);
        }
        else if constexpr(std::is_same<SUF, cdouble>::value)
        {
            return _mm512_set_pd(v[(*(b + 3U))].imag(), v[(*(b + 3U))].real(),
                                 v[(*(b + 2U))].imag(), v[(*(b + 2U))].real(),
                                 v[(*(b + 1U))].imag(), v[(*(b + 1U))].real(),
                                 v[(*(b + 0U))].imag(), v[(*(b + 0U))].real());
        }
        else if constexpr(std::is_same<SUF, cfloat>::value)
        {
            return _mm512_set_ps(v[(*(b + 7U))].imag(), v[(*(b + 7U))].real(),
                                 v[(*(b + 6U))].imag(), v[(*(b + 6U))].real(),
                                 v[(*(b + 5U))].imag(), v[(*(b + 5U))].real(),
                                 v[(*(b + 4U))].imag(), v[(*(b + 4U))].real(),
                                 v[(*(b + 3U))].imag(), v[(*(b + 3U))].real(),
                                 v[(*(b + 2U))].imag(), v[(*(b + 2U))].real(),
                                 v[(*(b + 1U))].imag(), v[(*(b + 1U))].real(),
                                 v[(*(b + 0U))].imag(), v[(*(b + 0U))].real());
        }
    };

    // Unaligned load to AVX register with zero mask direct memory model.
    template <bsz SZ, typename SUF, kt_avxext EXT, int L>
    KT_FORCE_INLINE
        std::enable_if_t<EXT == kt_avxext::AVX512VL || SZ == bsz::b512,
                         avxvector_t<SZ, SUF>>
        kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept
    {
        if constexpr(SZ == bsz::b256)
        {
            if constexpr(std::is_same_v<SUF, double>)
                return _mm256_maskz_loadu_pd((1 << L) - 1, &v[b]);
            else if constexpr(std::is_same_v<SUF, float>)
                return _mm256_maskz_loadu_ps((1 << L) - 1, &v[b]);
            else if constexpr(std::is_same_v<SUF, cdouble>)
            {
                return _mm256_maskz_loadu_pd((1 << (2 * L)) - 1, &v[b]);
            }
            else if constexpr(std::is_same_v<SUF, cfloat>)
            {
                return _mm256_maskz_loadu_ps((1 << (2 * L)) - 1, &v[b]);
            }
        }
        else if(SZ == bsz::b512 && (EXT & kt_avxext::AVX512F))
        {
            if constexpr(std::is_same_v<SUF, double>)
                return _mm512_maskz_loadu_pd((1 << L) - 1, &v[b]);
            else if constexpr(std::is_same_v<SUF, float>)
                return _mm512_maskz_loadu_ps((1 << L) - 1, &v[b]);
            else if constexpr(std::is_same_v<SUF, cdouble>)
            {
                return _mm512_maskz_loadu_pd((1 << (2 * L)) - 1, &v[b]);
            }
            else if constexpr(std::is_same_v<SUF, cfloat>)
            {
                return _mm512_maskz_loadu_ps((1 << (2 * L)) - 1, &v[b]);
            }
        }
    };

    // Unaligned load to AVX register with zero mask indirect memory model.
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept
    {

        if constexpr(kt_is_same<bsz::b512, SZ, double, SUF>())
        {
            return _mm512_set_pd(pz<SUF, L - 8>(v, b, 7), pz<SUF, L - 7>(v, b, 6U),
                                 pz<SUF, L - 6>(v, b, 5), pz<SUF, L - 5>(v, b, 4U),
                                 pz<SUF, L - 4>(v, b, 3), pz<SUF, L - 3>(v, b, 2U),
                                 pz<SUF, L - 2>(v, b, 1), pz<SUF, L - 1>(v, b, 0U));
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, float, SUF>())
        {
            return _mm512_set_ps(pz<SUF, L - 16>(v, b, 15), pz<SUF, L - 15>(v, b, 14),
                                 pz<SUF, L - 14>(v, b, 13), pz<SUF, L - 13>(v, b, 12),
                                 pz<SUF, L - 12>(v, b, 11), pz<SUF, L - 11>(v, b, 10),
                                 pz<SUF, L - 10>(v, b, 9),  pz<SUF, L - 9> (v, b, 8),
                                 pz<SUF, L - 8> (v, b, 7),  pz<SUF, L - 7> (v, b, 6),
                                 pz<SUF, L - 6> (v, b, 5),  pz<SUF, L - 5> (v, b, 4),
                                 pz<SUF, L - 4> (v, b, 3),  pz<SUF, L - 3> (v, b, 2),
                                 pz<SUF, L - 2> (v, b, 1),  pz<SUF, L - 1> (v, b, 0));
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, cdouble, SUF>())
        {
            return _mm512_set_pd(pz<SUF, L - 4, false>(v, b, 3), pz<SUF, L - 4, true> (v, b, 3),
                                 pz<SUF, L - 3, false>(v, b, 2), pz<SUF, L - 3, true> (v, b, 2),
                                 pz<SUF, L - 2, false>(v, b, 1), pz<SUF, L - 2, true> (v, b, 1),
                                 pz<SUF, L - 1, false>(v, b, 0), pz<SUF, L - 1, true> (v, b, 0));
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, cfloat, SUF>())
        {
            return _mm512_set_ps(pz<SUF, L - 8, false>(v, b, 7), pz<SUF, L - 8, true> (v, b, 7),
                                 pz<SUF, L - 7, false>(v, b, 6), pz<SUF, L - 7, true> (v, b, 6),
                                 pz<SUF, L - 6, false>(v, b, 5), pz<SUF, L - 6, true> (v, b, 5),
                                 pz<SUF, L - 5, false>(v, b, 4), pz<SUF, L - 5, true> (v, b, 4),
                                 pz<SUF, L - 4, false>(v, b, 3), pz<SUF, L - 4, true> (v, b, 3),
                                 pz<SUF, L - 3, false>(v, b, 2), pz<SUF, L - 3, true> (v, b, 2),
                                 pz<SUF, L - 2, false>(v, b, 1), pz<SUF, L - 2, true> (v, b, 1),
                                 pz<SUF, L - 1, false>(v, b, 0), pz<SUF, L - 1, true> (v, b, 0));
        }
    };

    // Dense direct aligned load to AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm512_load_ps(reinterpret_cast<const float *>(a));
        else
            return _mm512_load_pd(reinterpret_cast<const double *>(a));
    }

    // Dense direct (un)aligned load to AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm512_loadu_ps(reinterpret_cast<const float *>(a));
        else
            return _mm512_loadu_pd(reinterpret_cast<const double *>(a));
    };

    // Stores the values in an AVX register to a memory location (Memory does not have to be aligned)
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, void>
                    kt_storeu_p(SUF *a, const avxvector_t<SZ, SUF> v) noexcept
    {
        if constexpr(kt_is_base_t_double<SUF>())
            _mm512_storeu_pd(a, v);
        else
            _mm512_storeu_ps(a, v);
    }

    // Vector addition of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm512_add_ps(a, b);
        else
            return _mm512_add_pd(a, b);
    }

    // Vector subtraction of two AVX registers.
    // Note that sub_ps takes care of types float and complex float, same for double variant.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm512_sub_ps(a, b);
        else
            return _mm512_sub_pd(a, b);
    }

    // Vector product of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_mul_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_mul_ps(a, b);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            // input vectors a = (x0+iy0, x1+iy1, ...) and b = (a0+ib0, a1+ib1, ...)
            // imaginary elements in the vector: half = (y0, y0, y1, y1, ...)
            avxvector_t<bsz::b512, cdouble> half = _mm512_movedup_pd(_mm512_permute_pd(a, 0x55));
            // tmp = (a0*iy0, ib0*iy0, a1*iy1, ib1*iy1, ...)
            avxvector_t<bsz::b512, cdouble> tmp = _mm512_mul_pd(half, b);
            // real elements in the vector: half = (x0, x0, x1, x1, ...)
            half = _mm512_movedup_pd(a);
            // c = (x0*a0-b0*y0, x0*ib0+a0*iy0, x1*a1-b1*y1, x1*ib1+a1*iy1, ...)
            avxvector_t<bsz::b512, cdouble> c
                = _mm512_fmaddsub_pd(half, b, _mm512_permute_pd(tmp, 0x55));
            return c;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            // input vectors a = (x0+iy0, x1+iy1, ...) and b = (a0+ib0, a1+ib1, ...)
            // imaginary elements in the vector: half = (y0, y0, y1, y1, ...)
            avxvector_t<bsz::b512, cfloat> half = _mm512_movehdup_ps(a);
            // tmp (a0*iy0, ib0*iy0, a1*iy1, ib1*iy1, ...)
            avxvector_t<bsz::b512, cfloat> tmp = _mm512_mul_ps(half, b);
            // real elements in the vector: half = (x0, x0, x1, x1, ...)
            half = _mm512_moveldup_ps(a);
            // c = (x0*a0-b0*y0, x0*ib0+a0*iy0, x1*a1-b1*y1, x1*ib1+a1*iy1, ...)
            avxvector_t<bsz::b512, cfloat> c
                = _mm512_fmaddsub_ps(half, b, _mm512_permute_ps(tmp, 0xB1));
            return c;
        }
    }

    // Vector fused multiply-add of three AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_fmadd_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
            return _mm512_fmadd_pd(a, b, c);
        if constexpr(std::is_same_v<SUF, float>)
            return _mm512_fmadd_ps(a, b, c);
        else if constexpr(std::is_same_v<SUF, cdouble>)
            return kt_add_p<bsz::b512, cdouble>(kt_mul_p<bsz::b512, cdouble>(a, b), c);
        else if constexpr(std::is_same_v<SUF, cfloat>)
            return kt_add_p<bsz::b512, cfloat>(kt_mul_p<bsz::b512, cfloat>(a, b), c);
    }

    // Vector fused multiply-subtract of three AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_fmsub_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
            return _mm512_fmsub_pd(a, b, c);
        if constexpr(std::is_same_v<SUF, float>)
            return _mm512_fmsub_ps(a, b, c);
        else if constexpr(std::is_same_v<SUF, cdouble>)
            return kt_sub_p<bsz::b512, cdouble>(kt_mul_p<bsz::b512, cdouble>(a, b), c);
        else if constexpr(std::is_same_v<SUF, cfloat>)
            return kt_sub_p<bsz::b512, cfloat>(kt_mul_p<bsz::b512, cfloat>(a, b), c);
    }

    // Horizontal sum (reduction) of an AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_reduce_add_pd(v);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_reduce_add_ps(v);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            // input vector v = (x0+iy0, x1+iy1, x2+iy2, x3+iy3)
            // mem layout: v = (x0, y0, x1, y1, x2, y2, x3, y3)
            avxvector_t<bsz::b512, cdouble> res, s, tmp;
            __m512i                         idx = _mm512_set_epi64(0, 1, 2, 3, 7, 6, 5, 4);
            // tmp = (x1, y1, ..., x3, y3, ...) upper 128-bit of each 256-bit lane is not relevant
            tmp = _mm512_permutex_pd(v, 0b11101110);
            // res = (x0+x1, y0+y1, ..., x2+x3, y2+y3, ...)
            res = _mm512_add_pd(v, tmp);
            // tmp = (x2 + x3, y2 + y3, ...) rest is not relevant
            tmp = _mm512_permutexvar_pd(idx, res);
            // horizontal sum result is in the first 128-bits
            s              = _mm512_add_pd(res, tmp);
            cdouble result = {s[0], s[1]};
            return result;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            // input vector v = (x0+iy0, x1+iy1, ..., x7+iy7)
            avxvector_t<bsz::b512, cfloat> res, s, tmp;
            __m512i idx = _mm512_set_epi32(15, 14, 11, 10, 9, 8, 13, 12, 7, 6, 3, 2, 1, 0, 5, 4);
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
            idx           = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8);
            tmp           = _mm512_permutexvar_ps(idx, s);
            s             = _mm512_add_ps(s, tmp);
            cfloat result = {s[0], s[1]};
            return result;
        }
    }

    // Templated version of the conjugate operation
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_conj_p(const avxvector_t<SZ, SUF> a) noexcept
    {
        if constexpr(std::is_floating_point<SUF>::value)
        {
            return a;
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, cfloat, SUF>())
        {
            avxvector_t<SZ, SUF> mask = _mm512_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f,
                                                       0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
            avxvector_t<SZ, SUF> res = _mm512_xor_ps(mask, a);
            return res;
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, cdouble, SUF>())
        {
            avxvector_t<SZ, SUF> mask = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
            avxvector_t<SZ, SUF> res  = _mm512_xor_pd(mask, a);
            return res;
        }
    }

    // Vector fused multiply-add of three AVX registers - blocked variant
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, void>
                    kt_fmadd_B(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               avxvector_t<SZ, SUF> &c,
                               avxvector_t<SZ, SUF> &d
                               ) noexcept
    {
        if constexpr(std::is_same_v<SUF, double> || std::is_same_v<SUF, float>)
            c = kt_fmadd_p<SZ, SUF>(a, b, c);
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            c = _mm512_fmadd_pd(a, b, c);
            d = _mm512_fmadd_pd(a, _mm512_permute_pd(b, 0b01010101), d);
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            c = _mm512_fmadd_ps(a, b, c);
            d = _mm512_fmadd_ps(a, _mm512_permute_ps(b, 0b10110001), d);
        }
    }

    // Horizontal sum (reduction) of an AVX register - blocked variant
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, SUF>
                    kt_hsum_B(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double> || std::is_same_v<SUF, float>)
            return kt_hsum_p<SZ, SUF>(a);
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            avxvector_t<bsz::b512, double> t;
            avxvector_t<bsz::b512, double> signs = _mm512_setr_pd(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
            // Real part, alternate signs and sum up
            t = _mm512_xor_pd(signs, a);

            return SUF(_mm512_reduce_add_pd(t), _mm512_reduce_add_pd(b));
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            avxvector_t<bsz::b512, float> t;
            avxvector_t<bsz::b512, float> signs = _mm512_setr_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);
            // Real part, alternate signs and sum up
            t = _mm512_xor_ps(signs, a);

            return SUF(_mm512_reduce_add_ps(t), _mm512_reduce_add_ps(b));
        }
    }

    // Compare packed SUF elements in a and b, and returns packed maximum values.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE
        std::enable_if_t<SZ == bsz::b512 && kt_type_is_real<SUF>(), avxvector_t<SZ, SUF>>
        kt_max_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_max_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_max_ps(a, b);
        }
    }

    // Vector element-wise pow2 of an AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_pow2_p(const avxvector_t<SZ, SUF> a) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_mul_pd(a, a);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_mul_ps(a, a);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            using base_t = typename kt_dt<SUF>::base_type;
            avxvector_t<SZ, SUF> pow2 = kt_mul_p<SZ, base_t>(a, a);
            avxvector_t<SZ, SUF> shuff = _mm512_permute_pd(pow2, 0x55);
            pow2 = _mm512_add_pd(pow2, shuff);

            return pow2;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            using base_t = typename kt_dt<SUF>::base_type;
            avxvector_t<SZ, SUF> pow2 = kt_mul_p<SZ, base_t>(a, a);
            avxvector_t<SZ, SUF> shuff = _mm512_permute_ps(pow2, 0xB1);
            pow2 = _mm512_add_ps(pow2, shuff);

            return pow2;
        }
    }

    // Vector element-wise division of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_div_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_div_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_div_ps(a, b);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            avxvector_t<SZ, SUF> nmrtr = kt_mul_p<SZ, SUF>(a, kt_conj_p<SZ, SUF>(b));
            avxvector_t<SZ, SUF> dentr = kt_pow2_p<SZ, SUF>(b);
            avxvector_t<SZ, SUF> result = _mm512_div_pd(nmrtr, dentr);

            return result;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            avxvector_t<SZ, SUF> nmrtr = kt_mul_p<SZ, SUF>(a, kt_conj_p<SZ, SUF>(b));
            avxvector_t<SZ, SUF> dentr = kt_pow2_p<SZ, SUF>(b);
            avxvector_t<SZ, SUF> result = _mm512_div_ps(nmrtr, dentr);

            return result;
        }
    }
}
#endif
