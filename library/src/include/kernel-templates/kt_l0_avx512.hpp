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
#error "Never use "kt_l0_512.hpp" directly; include "kernel_templates.hpp" instead."
#endif

// clang-format off

#ifndef _KT_L0_AVX512_
#define _KT_L0_AVX512_
#include "kt_common.hpp"

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

    // Fill vector with a scalar value
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept
    {
        if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_set1_ps(x);
        }
        else if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_set1_pd(x);
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

    // Vector product of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_mul_ps(a, b);
        }
        else if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_mul_pd(a, b);
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
        if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_fmadd_ps(a, b, c);
        }
        else if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_fmadd_pd(a, b, c);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            return kt_add_p<bsz::b512, cdouble>(kt_mul_p<bsz::b512, cdouble>(a, b), c);
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            return kt_add_p<bsz::b512, cfloat>(kt_mul_p<bsz::b512, cfloat>(a, b), c);
        }
    }

    // Horizontal sum (reduction) of an AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept
    {
        if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm512_reduce_add_ps(v);
        }
        else if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm512_reduce_add_pd(v);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            // input vector v = (x0+iy0, x1+iy1, x2+iy2, x3+iy3)
            // mem layout: v = (x0, y0, x1, y1, x2, y2, x3, y3)
            avxvector_t<bsz::b512, cdouble> res, s, tmp;
            __m512i                         idx = _mm512_set_epi64(0, 1, 2, 3, 7, 6, 5, 4);
            // tmp = (x1, y1, ..., x3, y3, ...) upper 128-bit of each bsz::b256-bit lane is not relevant
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
            // upper 128-bits of each bsz::b256-bit lane are not used
            // tmp = (x1, y1, ..., x3, y3, ..., x5, y5, ..., x7, y7, ...)
            tmp = _mm512_permute_ps(v, 0b11101110);
            // element at indexes 0, 1, 4, 5, 8, 9, 12 and 13 hold the intermediate results
            // res = (x0+x1, y0+y1, ..., x2+x3, y2+y3, ...)
            res = _mm512_add_ps(v, tmp);
            // only indexes mentioned in the last two places of each bsz::b256-bit lane are relevant
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
            const double *vv = reinterpret_cast<const double *>(v);
            return _mm512_set_pd(vv[2U * (*(b + 3U)) + 1U], vv[2U * (*(b + 3U)) + 0U],
                                 vv[2U * (*(b + 2U)) + 1U], vv[2U * (*(b + 2U)) + 0U],
                                 vv[2U * (*(b + 1U)) + 1U], vv[2U * (*(b + 1U)) + 0U],
                                 vv[2U * (*(b + 0U)) + 1U], vv[2U * (*(b + 0U)) + 0U]);
        }
        else if constexpr(std::is_same<SUF, cfloat>::value)
        {
            const float *vv = reinterpret_cast<const float *>(v);
            return _mm512_set_ps(vv[2U * (*(b + 7U)) + 1U], vv[2U * (*(b + 7U)) + 0U],
                                 vv[2U * (*(b + 6U)) + 1U], vv[2U * (*(b + 6U)) + 0U],
                                 vv[2U * (*(b + 5U)) + 1U], vv[2U * (*(b + 5U)) + 0U],
                                 vv[2U * (*(b + 4U)) + 1U], vv[2U * (*(b + 4U)) + 0U],
                                 vv[2U * (*(b + 3U)) + 1U], vv[2U * (*(b + 3U)) + 0U],
                                 vv[2U * (*(b + 2U)) + 1U], vv[2U * (*(b + 2U)) + 0U],
                                 vv[2U * (*(b + 1U)) + 1U], vv[2U * (*(b + 1U)) + 0U],
                                 vv[2U * (*(b + 0U)) + 1U], vv[2U * (*(b + 0U)) + 0U]);
        }
    };

    // Unaligned load to AVX register with zero mask indirect memory model.
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept
    {

        if constexpr(kt_is_same<bsz::b512, SZ, double, SUF>())
        {
            return _mm512_set_pd(pz<SUF, L - 8>(v[*(b + 7U)]), pz<SUF, L - 7>(v[*(b + 6U)]),
                                 pz<SUF, L - 6>(v[*(b + 5U)]), pz<SUF, L - 5>(v[*(b + 4U)]),
                                 pz<SUF, L - 4>(v[*(b + 3U)]), pz<SUF, L - 3>(v[*(b + 2U)]),
                                 pz<SUF, L - 2>(v[*(b + 1U)]), pz<SUF, L - 1>(v[*(b + 0U)]));
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, float, SUF>())
        {
            return _mm512_set_ps(pz<SUF, L - 16>(v[*(b + 15U)]), pz<SUF, L - 15>(v[*(b + 14U)]),
                                 pz<SUF, L - 14>(v[*(b + 13U)]), pz<SUF, L - 13>(v[*(b + 12U)]),
                                 pz<SUF, L - 12>(v[*(b + 11U)]), pz<SUF, L - 11>(v[*(b + 10U)]),
                                 pz<SUF, L - 10>(v[*(b + 9U)]),  pz<SUF, L - 9>(v[*(b + 8U)]),
                                 pz<SUF, L - 8>(v[*(b + 7U)]),   pz<SUF, L - 7>(v[*(b + 6U)]),
                                 pz<SUF, L - 6>(v[*(b + 5U)]),   pz<SUF, L - 5>(v[*(b + 4U)]),
                                 pz<SUF, L - 4>(v[*(b + 3U)]),   pz<SUF, L - 3>(v[*(b + 2U)]),
                                 pz<SUF, L - 2>(v[*(b + 1U)]),   pz<SUF, L - 1>(v[*(b + 0U)]));
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, cdouble, SUF>())
        {
            const double *vv = reinterpret_cast<const double *>(v);
            return _mm512_set_pd(pz<double, L - 4>(vv[2U * (*(b + 3U)) + 1U]),
                                 pz<double, L - 4>(vv[2U * (*(b + 3U)) + 0U]),
                                 pz<double, L - 3>(vv[2U * (*(b + 2U)) + 1U]),
                                 pz<double, L - 3>(vv[2U * (*(b + 2U)) + 0U]),
                                 pz<double, L - 2>(vv[2U * (*(b + 1U)) + 1U]),
                                 pz<double, L - 2>(vv[2U * (*(b + 1U)) + 0U]),
                                 pz<double, L - 1>(vv[2U * (*(b + 0U)) + 1U]),
                                 pz<double, L - 1>(vv[2U * (*(b + 0U)) + 0U]));
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, cfloat, SUF>())
        {
            const float *vv = reinterpret_cast<const float *>(v);
            return _mm512_set_ps(pz<float, L - 8>(vv[2U * (*(b + 7U)) + 1U]),
                                 pz<float, L - 8>(vv[2U * (*(b + 7U)) + 0U]),
                                 pz<float, L - 7>(vv[2U * (*(b + 6U)) + 1U]),
                                 pz<float, L - 7>(vv[2U * (*(b + 6U)) + 0U]),
                                 pz<float, L - 6>(vv[2U * (*(b + 5U)) + 1U]),
                                 pz<float, L - 6>(vv[2U * (*(b + 5U)) + 0U]),
                                 pz<float, L - 5>(vv[2U * (*(b + 4U)) + 1U]),
                                 pz<float, L - 5>(vv[2U * (*(b + 4U)) + 0U]),
                                 pz<float, L - 4>(vv[2U * (*(b + 3U)) + 1U]),
                                 pz<float, L - 4>(vv[2U * (*(b + 3U)) + 0U]),
                                 pz<float, L - 3>(vv[2U * (*(b + 2U)) + 1U]),
                                 pz<float, L - 3>(vv[2U * (*(b + 2U)) + 0U]),
                                 pz<float, L - 2>(vv[2U * (*(b + 1U)) + 1U]),
                                 pz<float, L - 2>(vv[2U * (*(b + 1U)) + 0U]),
                                 pz<float, L - 1>(vv[2U * (*(b + 0U)) + 1U]),
                                 pz<float, L - 1>(vv[2U * (*(b + 0U)) + 0U]));
        }
    };

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

    // Unaligned load to AVX register with zero mask direct memory model.
    template <bsz SZ, typename SUF, kt_avxext EXT, int L>
    KT_FORCE_INLINE
        std::enable_if_t<((SZ == bsz::b256 && EXT == kt_avxext::AVX512VL) || SZ == bsz::b512),
                         avxvector_t<SZ, SUF>>
        kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept
    {
        if constexpr(kt_is_same<bsz::b256, SZ, double, SUF>())
        {
            if constexpr(EXT & AVX512VL)
                return _mm256_maskz_loadu_pd((1 << L) - 1, &v[b]);
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, float, SUF>())
        {
            if constexpr(EXT & AVX512VL)
                return _mm256_maskz_loadu_ps((1 << L) - 1, &v[b]);
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cdouble, SUF>())
        {
            const double *vv = reinterpret_cast<const double *>(v);
            if constexpr(EXT & AVX512VL)
                return _mm256_maskz_loadu_pd((1 << (2 * L)) - 1, &vv[2U * b]);
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cfloat, SUF>())
        {
            const float *vv = reinterpret_cast<const float *>(v);
            if constexpr(EXT & AVX512VL)
                return _mm256_maskz_loadu_ps((1 << (2 * L)) - 1, &vv[2U * b]);
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, double, SUF>() && (EXT & AVX512F))
        {
            return _mm512_maskz_loadu_pd((1 << L) - 1, &v[b]);
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, float, SUF>() && (EXT & AVX512F))
        {
            return _mm512_maskz_loadu_ps((1 << L) - 1, &v[b]);
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, cdouble, SUF>() && (EXT & AVX512F))
        {
            const double *vv = reinterpret_cast<const double *>(v);
            return _mm512_maskz_loadu_pd((1 << (2 * L)) - 1, &vv[2U * b]);
        }
        else if constexpr(kt_is_same<bsz::b512, SZ, cfloat, SUF>() && (EXT & AVX512F))
        {
            const float *vv = reinterpret_cast<const float *>(v);
            return _mm512_maskz_loadu_ps((1 << (2 * L)) - 1, &vv[2U * b]);
        }
    };
}
#endif
