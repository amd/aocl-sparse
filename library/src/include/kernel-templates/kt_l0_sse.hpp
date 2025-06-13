/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#error "Never use ``kt_l0_sse.hpp'' directly; include ``kernel_templates.hpp'' instead."
#endif

#ifndef _KT_L0_SSE_
#define _KT_L0_SSE_
#include "kt_common_x86.hpp"

namespace kernel_templates
{
    // Zero out an AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_setzero_ps();
        else
            return _mm_setzero_pd();
    };

    // Fill vector with a scalar value
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_set1_p(const SUF x) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm_set1_pd(x);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm_set1_ps(x);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {

            const double r = std::real(x);
            const double i = std::imag(x);
            return _mm_set_pd(i, r);
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {

            const float r = std::real(x);
            const float i = std::imag(x);
            // Note that loading is end -> start <=> [d c b a] <=> [i1, r1, i0, r0]
            return _mm_set_ps(i, r, i, r);
        }
    };

    // Unaligned set (load) to AVX register with indirect memory access
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_set_p(const SUF *v, const kt_int_t *b) noexcept
    {
        if constexpr(std::is_same<SUF, double>::value)
        {
            return _mm_set_pd(v[*(b + 1U)], v[*(b + 0U)]);
        }
        else if constexpr(std::is_same<SUF, float>::value)
        {
            return _mm_set_ps(v[*(b + 3U)], v[*(b + 2U)], v[*(b + 1U)], v[*(b + 0U)]);
        }
        else if constexpr(std::is_same<SUF, cdouble>::value)
        {
            return _mm_set_pd(v[(*(b + 0U))].imag(), v[(*(b + 0U))].real());
        }
        else if constexpr(std::is_same<SUF, cfloat>::value)
        {
            return _mm_set_ps(v[(*(b + 1U))].imag(),
                              v[(*(b + 1U))].real(),
                              v[(*(b + 0U))].imag(),
                              v[(*(b + 0U))].real());
        }
    };

    // Dense direct aligned load to AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_load_ps(reinterpret_cast<const float *>(a));
        else
            return _mm_load_pd(reinterpret_cast<const double *>(a));
    };

    // Dense direct (un)aligned load to AVX register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_loadu_ps(reinterpret_cast<const float *>(a));
        else
            return _mm_loadu_pd(reinterpret_cast<const double *>(a));
    };

    // Stores the values in an AVX register to a memory location (Memory does not have to be aligned)
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, void>
                    kt_storeu_p(SUF *a, const avxvector_t<SZ, SUF> v) noexcept
    {
        if constexpr(kt_is_base_t_double<SUF>())
            _mm_storeu_pd(reinterpret_cast<double *>(a), v);
        else
            _mm_storeu_ps(reinterpret_cast<float *>(a), v);
    };

    // Vector addition of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_add_ps(a, b);
        else
            return _mm_add_pd(a, b);
    }

    // Vector subtraction of two AVX registers.
    // Note that sub_ps takes care of types float and complex float, same for double variant.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_sub_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_sub_ps(a, b);
        else
            return _mm_sub_pd(a, b);
    }

    // Vector product of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_mul_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm_mul_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm_mul_ps(a, b);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            // input vectors a = (x0+iy0) and b = (a0+ib0)

            // real elements in the vector: half = (x0 * a0, y0 * b0)
            avxvector_t<bsz::b128, cdouble> real = _mm_mul_pd(a, b);
            // imaginary elements in the vector: half = (y0 * a0, x0 * b0)
            avxvector_t<bsz::b128, cdouble> imag = _mm_mul_pd(a, _mm_shuffle_pd(b, b, 0x1));

            avxvector_t<bsz::b128, cdouble> t1 = _mm_shuffle_pd(real, imag, 0b11);
            avxvector_t<bsz::b128, cdouble> t2 = _mm_shuffle_pd(real, imag, 0b00);

            avxvector_t<bsz::b128, cdouble> result = _mm_addsub_pd(t2, t1);

            return result;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            // input vectors a = (x0+iy0, x1+iy1) and b = (a0+ib0, a1+ib1)
            // real elements in the vector
            avxvector_t<bsz::b128, cfloat> real = _mm_mul_ps(a, b);
            // imaginary elements in the vector
            avxvector_t<bsz::b128, cfloat> imag
                = _mm_mul_ps(a, _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1)));
            // Blend real and imaginary parts
            avxvector_t<bsz::b128, cfloat> t1 = _mm_blend_ps(real, imag, 0b1010);
            avxvector_t<bsz::b128, cfloat> t2 = _mm_blend_ps(imag, real, 0b1010);

            t2 = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(2, 3, 0, 1));

            avxvector_t<bsz::b128, cfloat> res = _mm_addsub_ps(t1, t2);

            return res;
        }
    }

    // Vector fused multiply-add of three AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_fmadd_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
            return _mm_fmadd_pd(a, b, c);
        else if constexpr(std::is_same_v<SUF, float>)
            return _mm_fmadd_ps(a, b, c);
        else if constexpr(std::is_same_v<SUF, cdouble>)
            return kt_add_p<bsz::b128, cdouble>(kt_mul_p<bsz::b128, cdouble>(a, b), c);
        else if constexpr(std::is_same_v<SUF, cfloat>)
            return kt_add_p<bsz::b128, cfloat>(kt_mul_p<bsz::b128, cfloat>(a, b), c);
    }

    // Vector fused multiply-subtract of three AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_fmsub_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b,
                               const avxvector_t<SZ, SUF> c) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
            return _mm_fmsub_pd(a, b, c);
        else if constexpr(std::is_same_v<SUF, float>)
            return _mm_fmsub_ps(a, b, c);
        else if constexpr(std::is_same_v<SUF, cdouble>)
            return kt_sub_p<bsz::b128, cdouble>(kt_mul_p<bsz::b128, cdouble>(a, b), c);
        else if constexpr(std::is_same_v<SUF, cfloat>)
            return kt_sub_p<bsz::b128, cfloat>(kt_mul_p<bsz::b128, cfloat>(a, b), c);
    }

    // Templated version of the conjugate operation
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_conj_p(const avxvector_t<SZ, SUF> a) noexcept
    {
        if constexpr(std::is_floating_point<SUF>::value)
        {
            return a;
        }
        else if constexpr(kt_is_same<bsz::b128, SZ, cfloat, SUF>())
        {
            avxvector_t<SZ, SUF> mask = _mm_setr_ps(0.f, -0.f, 0.f, -0.f);
            avxvector_t<SZ, SUF> res  = _mm_xor_ps(mask, a);
            return res;
        }
        else if constexpr(kt_is_same<bsz::b128, SZ, cdouble, SUF>())
        {
            avxvector_t<SZ, SUF> mask = _mm_setr_pd(0.0, -0.0);
            avxvector_t<SZ, SUF> res  = _mm_xor_pd(mask, a);
            return res;
        }
    }
}

#endif // _KT_L0_SSE_