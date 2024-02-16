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
#error "Never use "kt_l0_256.hpp" directly; include "kernel_templates.hpp" instead."
#endif

// clang-format off

#ifndef _KT_L0_AVX2_
#define _KT_L0_AVX2_
#include "kt_common.hpp"

namespace kernel_templates
{
    // Zero out an AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm256_setzero_ps();
        else
            return _mm256_setzero_pd();
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
            const double *vv = reinterpret_cast<const double *>(v);
            return _mm256_set_pd(vv[2U * (*(b + 1U)) + 1U], vv[2U * (*(b + 1U)) + 0U],
                                 vv[2U * (*(b + 0U)) + 1U], vv[2U * (*(b + 0U)) + 0U]);
        }
        else if constexpr(std::is_same<SUF, cfloat>::value)
        {
            const float *vv = reinterpret_cast<const float *>(v);
            return _mm256_set_ps(vv[2U * (*(b + 3U)) + 1U], vv[2U * (*(b + 3U)) + 0U],
                                 vv[2U * (*(b + 2U)) + 1U], vv[2U * (*(b + 2U)) + 0U],
                                 vv[2U * (*(b + 1U)) + 1U], vv[2U * (*(b + 1U)) + 0U],
                                 vv[2U * (*(b + 0U)) + 1U], vv[2U * (*(b + 0U)) + 0U]);
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
            return _mm256_set_pd(pz<SUF, L - 4>(v[b + 3U]), pz<SUF, L - 3>(v[b + 2U]),
                                 pz<SUF, L - 2>(v[b + 1U]), pz<SUF, L - 1>(v[b + 0U]));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, float, SUF>())
        {
            return _mm256_set_ps(pz<SUF, L - 8>(v[b + 7U]), pz<SUF, L - 7>(v[b + 6U]),
                                 pz<SUF, L - 6>(v[b + 5U]), pz<SUF, L - 5>(v[b + 4U]),
                                 pz<SUF, L - 4>(v[b + 3U]), pz<SUF, L - 3>(v[b + 2U]),
                                 pz<SUF, L - 2>(v[b + 1U]), pz<SUF, L - 1>(v[b + 0U]));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cdouble, SUF>())
        {
            const double *vv = reinterpret_cast<const double *>(v);
            return _mm256_set_pd(pz<double, L - 2>(vv[2U * b + 3U]),
                                 pz<double, L - 2>(vv[2U * b + 2U]),
                                 pz<double, L - 1>(vv[2U * b + 1U]),
                                 pz<double, L - 1>(vv[2U * b + 0U]));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cfloat, SUF>())
        {
            const float *vv = reinterpret_cast<const float *>(v);
            return _mm256_set_ps(pz<float, L - 4>(vv[2 * b + 7U]), pz<float, L - 4>(vv[2 * b + 6U]),
                                 pz<float, L - 3>(vv[2 * b + 5U]), pz<float, L - 3>(vv[2 * b + 4U]),
                                 pz<float, L - 2>(vv[2 * b + 3U]), pz<float, L - 2>(vv[2 * b + 2U]),
                                 pz<float, L - 1>(vv[2 * b + 1U]), pz<float, L - 1>(vv[2 * b + 0U]));
        }
    };

    // Unaligned load to AVX register with zero mask indirect memory model.
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept
    {
        if constexpr(kt_is_same<bsz::b256, SZ, double, SUF>())
        {
            return _mm256_set_pd(pz<SUF, L - 4>(v[*(b + 3U)]), pz<SUF, L - 3>(v[*(b + 2U)]),
                                 pz<SUF, L - 2>(v[*(b + 1U)]), pz<SUF, L - 1>(v[*(b + 0U)]));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, float, SUF>())
        {
            return _mm256_set_ps(pz<SUF, L - 8>(v[*(b + 7U)]), pz<SUF, L - 7>(v[*(b + 6U)]),
                                 pz<SUF, L - 6>(v[*(b + 5U)]), pz<SUF, L - 5>(v[*(b + 4U)]),
                                 pz<SUF, L - 4>(v[*(b + 3U)]), pz<SUF, L - 3>(v[*(b + 2U)]),
                                 pz<SUF, L - 2>(v[*(b + 1U)]), pz<SUF, L - 1>(v[*(b + 0U)]));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cdouble, SUF>())
        {
            const double *vv = reinterpret_cast<const double *>(v);
            return _mm256_set_pd(pz<double, L - 2>(vv[2U * (*(b + 1U)) + 1U]),
                                 pz<double, L - 2>(vv[2U * (*(b + 1U)) + 0U]),
                                 pz<double, L - 1>(vv[2U * (*(b + 0U)) + 1U]),
                                 pz<double, L - 1>(vv[2U * (*(b + 0U)) + 0U]));
        }
        else if constexpr(kt_is_same<bsz::b256, SZ, cfloat, SUF>())
        {
            const float *vv = reinterpret_cast<const float *>(v);
            return _mm256_set_ps(pz<float, L - 4>(vv[2U * (*(b + 3U)) + 1U]),
                                 pz<float, L - 4>(vv[2U * (*(b + 3U)) + 0U]),
                                 pz<float, L - 3>(vv[2U * (*(b + 2U)) + 1U]),
                                 pz<float, L - 3>(vv[2U * (*(b + 2U)) + 0U]),
                                 pz<float, L - 2>(vv[2U * (*(b + 1U)) + 1U]),
                                 pz<float, L - 2>(vv[2U * (*(b + 1U)) + 0U]),
                                 pz<float, L - 1>(vv[2U * (*(b + 0U)) + 1U]),
                                 pz<float, L - 1>(vv[2U * (*(b + 0U)) + 0U]));
        }
    };

    // Dense direct aligned load to AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm256_load_ps(reinterpret_cast<const float *>(a));
        else
            return _mm256_load_pd(reinterpret_cast<const double *>(a));
    };

    // Dense direct (un)aligned load to AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm256_loadu_ps(reinterpret_cast<const float *>(a));
        else
            return _mm256_loadu_pd(reinterpret_cast<const double *>(a));
    };

    // Stores the values in an AVX register to a memory location (Memory does not have to be aligned)
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, void>
                    kt_storeu_p(const SUF *a, const avxvector_t<SZ, SUF> v) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
            _mm256_storeu_pd(const_cast<double *>(a), v);
        else if constexpr(std::is_same_v<SUF, float>)
            _mm256_storeu_ps(const_cast<float *>(a), v);
        else if constexpr(std::is_same_v<SUF, cdouble>)
            _mm256_storeu_pd(reinterpret_cast<double *>(const_cast<cdouble *>(a)), v);
        else if constexpr(std::is_same_v<SUF, cfloat>)
            _mm256_storeu_ps(reinterpret_cast<float *>(const_cast<cfloat *>(a)), v);
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
}
#endif
