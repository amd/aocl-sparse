/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_AXPYI_HPP
#define AOCLSPARSE_AXPYI_HPP
#include "aoclsparse.h"
#include "aoclsparse_dispatcher.hpp"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l1_kt.hpp"
#include "aoclsparse_utils.hpp"

/*
 * axpyi reference implementation
 * It is assumed that all pointers and data are valid.
 * Only check is made for indx to check that entries are not negative
 */
template <typename T>
inline aoclsparse_status axpyi_ref(aoclsparse_int nnz,
                                   T              a,
                                   const T *__restrict__ x,
                                   const aoclsparse_int *__restrict__ indx,
                                   T *__restrict__ y)
{
    aoclsparse_int i;
    for(i = 0; i < nnz; i++)
    {
        if(indx[i] < 0)
            return aoclsparse_status_invalid_index_value;

        y[indx[i]] = a * x[i] + y[indx[i]];
    }
    return aoclsparse_status_success;
}

/*
 * aoclsparse_axpyi_t dispatcher
 */
template <typename T>
aoclsparse_status aoclsparse_axpyi_t(aoclsparse_int nnz,
                                     T              a,
                                     const T *__restrict__ x,
                                     const aoclsparse_int *__restrict__ indx,
                                     T *__restrict__ y,
                                     aoclsparse_int kid = -1)
{
    using namespace aoclsparse;
    using namespace Dispatch;
    using namespace kernel_templates;

    // Defining kernel pointer type
    using K = decltype(&axpyi_ref<T>);

    // Check pointer arguments
    if((nullptr == x) || (nullptr == indx) || (nullptr == y))
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(nnz == 0)
    {
        return aoclsparse_status_success;
    }

    if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // clang-format off
    // Table of available kernels
    static constexpr Table<K> tbl[]{
    {axpyi_ref<T>,           context_isa_t::GENERIC, 0U | archs::ALL},
    {axpyi_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ZEN123},
#ifdef USE_AVX512
    {axpyi_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ZEN4}
#endif
    };
    // clang-format on

    // Inquire with the oracle
    auto kernel = Oracle<K, api::axpyi>(tbl, kid);

    if(!kernel)
        return aoclsparse_status_invalid_kid;

    return kernel(nnz, a, x, indx, y);
}
#endif
