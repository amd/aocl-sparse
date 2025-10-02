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
    // Zero out an SSE register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_setzero_p(void) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_setzero_ps();
        else if constexpr(kt_is_base_t_double<SUF>())
            return _mm_setzero_pd();
        else if constexpr(kt_is_base_t_int<SUF>())
            return _mm_setzero_si128();
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
        else if constexpr(std::is_same_v<SUF, int64_t>)
            return _mm_set1_epi64x(x);
        else if constexpr(std::is_same_v<SUF, int32_t>)
            return _mm_set1_epi32(x);
    };

    // Unaligned set (load) to SSE register with indirect memory access
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
        else if constexpr(std::is_same_v<SUF, int64_t>)
            return _mm_set_epi64x(v[*(b + 1U)], v[*(b + 0U)]);
        else if constexpr(std::is_same_v<SUF, int32_t>)
            return _mm_set_epi32(v[*(b + 3U)], v[*(b + 2U)], v[*(b + 1U)], v[*(b + 0U)]);
    };

    // Unaligned load to SSE register with zero mask direct memory model.
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t b) noexcept
    {
        if constexpr(kt_is_same<bsz::b128, SZ, double, SUF>())
        {
            return _mm_set_pd(pz<SUF, L - 2>(v, b + 1), pz<SUF, L - 1>(v, b + 0));
        }
        else if constexpr(kt_is_same<bsz::b128, SZ, float, SUF>())
        {
            return _mm_set_ps(pz<SUF, L - 4>(v, b + 3),
                              pz<SUF, L - 3>(v, b + 2),
                              pz<SUF, L - 2>(v, b + 1),
                              pz<SUF, L - 1>(v, b + 0));
        }
        else if constexpr(kt_is_same<bsz::b128, SZ, cdouble, SUF>())
        {
            return _mm_set_pd(pz<SUF, L - 1, false>(v, b + 0), pz<SUF, L - 1, true>(v, b + 0));
        }
        else if constexpr(kt_is_same<bsz::b128, SZ, cfloat, SUF>())
        {
            return _mm_set_ps(pz<SUF, L - 2, false>(v, b + 1),
                              pz<SUF, L - 2, true>(v, b + 1),
                              pz<SUF, L - 1, false>(v, b + 0),
                              pz<SUF, L - 1, true>(v, b + 0));
        }
    };

    // Unaligned load to SSE register with zero mask indirect memory model.
    template <bsz SZ, typename SUF, kt_avxext, int L>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_maskz_set_p(const SUF *v, const kt_int_t *b) noexcept
    {
        if constexpr(kt_is_same<bsz::b128, SZ, double, SUF>())
        {
            return _mm_set_pd(pz<SUF, L - 2>(v, b, 1), pz<SUF, L - 1>(v, b, 0));
        }
        else if constexpr(kt_is_same<bsz::b128, SZ, float, SUF>())
        {
            return _mm_set_ps(pz<SUF, L - 4>(v, b, 3),
                              pz<SUF, L - 3>(v, b, 2),
                              pz<SUF, L - 2>(v, b, 1),
                              pz<SUF, L - 1>(v, b, 0));
        }
        else if constexpr(kt_is_same<bsz::b128, SZ, cdouble, SUF>())
        {
            return _mm_set_pd(pz<std::complex<double>, L - 1, false>(v, b, 0),
                              pz<std::complex<double>, L - 1, true>(v, b, 0));
        }
        else if constexpr(kt_is_same<bsz::b128, SZ, cfloat, SUF>())
        {
            return _mm_set_ps(pz<std::complex<float>, L - 2, false>(v, b, 1),
                              pz<std::complex<float>, L - 2, true>(v, b, 1),
                              pz<std::complex<float>, L - 1, false>(v, b, 0),
                              pz<std::complex<float>, L - 1, true>(v, b, 0));
        }
    };

    // Dense direct aligned load to SSE register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_load_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_load_ps(reinterpret_cast<const float *>(a));
        else if constexpr(kt_is_base_t_double<SUF>())
            return _mm_load_pd(reinterpret_cast<const double *>(a));
        else if constexpr(kt_is_base_t_int<SUF>())
            return _mm_load_si128(reinterpret_cast<__m128i const *>(a));
    };

    // Dense direct (un)aligned load to SSE register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_loadu_p(const SUF *a) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_loadu_ps(reinterpret_cast<const float *>(a));
        else if constexpr(kt_is_base_t_double<SUF>())
            return _mm_loadu_pd(reinterpret_cast<const double *>(a));
        else if constexpr(kt_is_base_t_int<SUF>())
            return _mm_lddqu_si128(reinterpret_cast<__m128i const *>(a));
    };

    // Stores the values in an SSE register to a memory location (Memory does not have to be aligned)
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, void>
                    kt_storeu_p(SUF *a, const avxvector_t<SZ, SUF> v) noexcept
    {
        if constexpr(kt_is_base_t_double<SUF>())
            _mm_storeu_pd(reinterpret_cast<double *>(a), v);
        else
            _mm_storeu_ps(reinterpret_cast<float *>(a), v);
    };

    // Vector addition of two SSE registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_add_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(kt_is_base_t_float<SUF>())
            return _mm_add_ps(a, b);
        else
            return _mm_add_pd(a, b);
    }

    // Vector subtraction of two SSE registers.
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

    // Vector product of two SSE registers.
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

    // Vector fused multiply-add of three SSE registers.
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

    // Vector fused multiply-subtract of three SSE registers.
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

    // Horizontal sum (reduction) of an SSE register
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, SUF>
                    kt_hsum_p(avxvector_t<SZ, SUF> const v) noexcept
    {
        // To-Do: Implement horizontal sum for bsz::b128 and benchmark
        if constexpr(std::is_same_v<SUF, double>)
        {
            return v[0] + v[1]; // 2 elements
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return v[0] + v[1] + v[2] + v[3]; // 4 elements
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            return cdouble(v[0], v[1]); // no sum to process
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            return cfloat(v[0] + v[2], v[1] + v[3]); // 2 elements (cfloat)
        }
    }

    // Vector fused multiply-add of three SSE registers - blocked variant
    // In case of b128, kt_fmadd_B is the same as kt_fmadd_p.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, void>
                    kt_fmadd_B(const avxvector_t<SZ, SUF>             a,
                               const avxvector_t<SZ, SUF>             b,
                               avxvector_t<SZ, SUF>                  &c,
                               [[maybe_unused]] avxvector_t<SZ, SUF> &d) noexcept
    {
        c = kt_fmadd_p<SZ, SUF>(a, b, c);
        /*
            Note: The following code is commented out because it hsum_B
            is not implemented for complex types in bsz::b128. This will
            be enabled when kt_hsum_B is implemented for bsz::b128.

            if constexpr(std::is_same_v<SUF, double> || std::is_same_v<SUF, float>)
                c = kt_fmadd_p<SZ, SUF>(a, b, c);
            else if constexpr(std::is_same_v<SUF, cdouble>)
            {
                c = _mm_fmadd_pd(a, b, c);
                d = _mm_fmadd_pd(a, _mm_permute_pd(b, 0b01), d);
            }
            else if constexpr(std::is_same_v<SUF, cfloat>)
            {
                c = _mm_fmadd_ps(a, b, c);
                d = _mm_fmadd_ps(a, _mm_permute_ps(b, 0b10110001), d);
            }
        */
    }

    // Horizontal sum (reduction) of an SSE register - blocked variant
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, SUF>
                    kt_hsum_B(const avxvector_t<SZ, SUF>                  a,
                              [[maybe_unused]] const avxvector_t<SZ, SUF> b) noexcept
    {
        // To-Do: Implement blocked horizontal sum for bsz::b128 and benchmark
        return kt_hsum_p<SZ, SUF>(a);
    }

    // Compare packed SUF elements in a and b, and returns packed maximum values.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE
        std::enable_if_t<SZ == bsz::b128 && kt_type_is_real<SUF>(), avxvector_t<SZ, SUF>>
        kt_max_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm_max_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm_max_ps(a, b);
        }
    }

    // Vector element-wise pow2 of an AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_pow2_p(const avxvector_t<SZ, SUF> a) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm_mul_pd(a, a);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm_mul_ps(a, a);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            using base_t               = typename kt_dt<SUF>::base_type;
            avxvector_t<SZ, SUF> pow2  = kt_mul_p<SZ, base_t>(a, a);
            avxvector_t<SZ, SUF> shuff = _mm_shuffle_pd(pow2, pow2, 0x1);
            pow2                       = _mm_add_pd(pow2, shuff);

            return pow2;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            using base_t               = typename kt_dt<SUF>::base_type;
            avxvector_t<SZ, SUF> pow2  = kt_mul_p<SZ, base_t>(a, a);
            avxvector_t<SZ, SUF> shuff = _mm_shuffle_ps(pow2, pow2, _MM_SHUFFLE(2, 3, 0, 1));
            pow2                       = _mm_add_ps(pow2, shuff);

            return pow2;
        }
    }

    // Vector element-wise division of two AVX registers.
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                    kt_div_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_same_v<SUF, double>)
        {
            return _mm_div_pd(a, b);
        }
        else if constexpr(std::is_same_v<SUF, float>)
        {
            return _mm_div_ps(a, b);
        }
        else if constexpr(std::is_same_v<SUF, cdouble>)
        {
            avxvector_t<SZ, SUF> nmrtr  = kt_mul_p<SZ, SUF>(a, kt_conj_p<SZ, SUF>(b));
            avxvector_t<SZ, SUF> dentr  = kt_pow2_p<SZ, SUF>(b);
            avxvector_t<SZ, SUF> result = _mm_div_pd(nmrtr, dentr);

            return result;
        }
        else if constexpr(std::is_same_v<SUF, cfloat>)
        {
            avxvector_t<SZ, SUF> nmrtr  = kt_mul_p<SZ, SUF>(a, kt_conj_p<SZ, SUF>(b));
            avxvector_t<SZ, SUF> dentr  = kt_pow2_p<SZ, SUF>(b);
            avxvector_t<SZ, SUF> result = _mm_div_ps(nmrtr, dentr);

            return result;
        }
    }
}

#endif // _KT_L0_SSE_
