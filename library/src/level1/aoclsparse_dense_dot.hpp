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
 * ************************************************************************ */

#ifndef AOCLSPARSE_DENSE_DOT_HPP
#define AOCLSPARSE_DENSE_DOT_HPP

#include "aoclsparse.h"
#include "aoclsparse_dispatcher.hpp"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l1_kt.hpp"
#include "aoclsparse_utils.hpp"

namespace aoclsparse
{
    // The templated function performs dot product of two dense vectors, x nd y.
    // Precision types supported: complex (float and double), real (float and double).
    // For complex types, conjugated dot product is supported in addition to the dot product.
    template <typename T>
    aoclsparse_status dot_ref(const aoclsparse_int size,
                              const T *__restrict__ x,
                              const T *__restrict__ y,
                              T *__restrict__ d)
    {
        aoclsparse_int i;
        *d = aoclsparse_numeric::zero<T>();

        for(i = 0; i < size; i++)
        {
            *d += aoclsparse::conj(x[i]) * y[i];
        }
        return aoclsparse_status_success;
    }

    using namespace kernel_templates;

    template <typename T>
    aoclsparse_status dense_dot(const aoclsparse_int size,
                                const T *__restrict__ x,
                                const T *__restrict__ y,
                                T *__restrict__ d,
                                aoclsparse_int kid = -1)
    {
        using namespace kernel_templates;
        using namespace Dispatch;
        using namespace aoclsparse;

        // Creating pointer to the kernel
        using K = decltype(&dot_ref<T>);

        // clang-format off
        // Table of available kernels
        static constexpr Table<K> tbl[]{
        {dot_ref<T>,           context_isa_t::GENERIC, 0U | archs::ALL},
        {dot_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ZEN123},
    #ifdef USE_AVX512
        {dot_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ZEN4}
    #endif
        };
        // clang-format on

        // Inquire with the oracle
        K dot_kernel = Oracle<K, api::dense_dot>(tbl, kid);

        /* Dot product needs x and y of same size but
        * op = non-transpose, size of y=m, x=n
        * op = transpose, size of y=n, x=m
        * hence, taking minimum of m and n
        */
        return dot_kernel(size, x, y, d);
    }
}

#endif