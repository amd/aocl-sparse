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
#error "Never use ``kt_l1.hpp'' directly; include ``kernel_templates.hpp'' instead."
#endif

#ifndef _KT_L1_
#define _KT_L1_
#include "kt_common_x86.hpp"

// Level 1 micro kernels

namespace kernel_templates
{
    // Dot-product of two AVX registers (convenience callers)
    //  - `a` avxvector
    //  - `b` avxvector
    // returns a scalar containing the dot-product of a and b, <a,b>
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE SUF kt_dot_p(const avxvector_t<SZ, SUF> a,
                                 const avxvector_t<SZ, SUF> b) noexcept
    {
        avxvector_t<SZ, SUF> c = kt_mul_p<SZ, SUF>(a, b);
        return kt_hsum_p<SZ, SUF>(c);
    };

    // Conjugate dot-product of two AVX registers
    //  - `SZ`  size (in bits) of AVX vector, i.e., bsz::b256 or bsz::b512
    //  - `SUF` suffix of working type, i.e., `double`, `float`, `cdouble`, or `cfloat`
    //  - `a` avxvector
    //  - `b` avxvector
    // if `a` and `b` are real then returns the dot-product of both, if complex
    // then returns the dot-product of `a` and `conjugate(b)`
    template <bsz SZ, typename SUF>
    SUF KT_FORCE_INLINE kt_cdot_p(const avxvector_t<SZ, SUF> a,
                                  const avxvector_t<SZ, SUF> b) noexcept
    {
        if constexpr(std::is_floating_point<SUF>::value)
        {
            return kt_dot_p<SZ, SUF>(a, b);
        }
        else
        {
            return kt_dot_p<SZ, SUF>(a, kt_conj_p<SZ, SUF>(b));
        }
    };
}
#endif
