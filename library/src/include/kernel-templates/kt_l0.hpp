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
#error "Never use ``kt_l0.hpp'' directly; include ``kernel_templates.hpp'' instead."
#endif

#ifndef _KT_L0_
#define _KT_L0_
#include "kt_common_x86.hpp"

// Add L0 micro kernels that are architecture independent here
namespace kernel_templates
{
    // Scatter kernel
    template <bsz SZ, typename SUF>
    KT_FORCE_INLINE void
        kt_scatter_p(const avxvector_t<SZ, SUF> a, SUF *v, const kt_int_t *b) noexcept
    {
        const SUF *acast = reinterpret_cast<const SUF *>(&a);
        for(size_t k = 0; k < tsz_v<SZ, SUF>; k++)
            v[b[k]] = acast[k];
    }
}
#endif