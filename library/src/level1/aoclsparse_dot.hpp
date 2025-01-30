/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_DOT_HPP
#define AOCLSPARSE_DOT_HPP
#include "aoclsparse.h"
#include "aoclsparse_cntx_dispatcher.hpp"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l1_kt.hpp"
#include "aoclsparse_utils.hpp"

// The templated function performs dot product of a sparse vector (x) with a dense vector (y).
// Precision types supported: complex (float and double), real (float and double).
// For complex types, conjugated dot product is supported in addition to the dot product.
template <typename T>
inline aoclsparse_status dotp_ref(aoclsparse_int nnz,
                                  const T *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  const T *__restrict__ y,
                                  T *__restrict__ dot,
                                  bool conj)
{
    aoclsparse_int i;
    *dot = aoclsparse_numeric::zero<T>();

    if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>)
    {
        if(conj) // This if is only processed if T is complex<>
        {
            for(i = 0; i < nnz; i++)
            {
                *dot += std::conj(x[i]) * y[indx[i]];
            }
            return aoclsparse_status_success;
        }
    }

    for(i = 0; i < nnz; i++)
    {
        *dot += x[i] * y[indx[i]];
    }

    return aoclsparse_status_success;
}

// Wrapper to the dot-product function with the necessary validations.
template <typename T>
inline aoclsparse_status aoclsparse_dotp(aoclsparse_int nnz,
                                         const T *__restrict__ x,
                                         const aoclsparse_int *__restrict__ indx,
                                         const T *__restrict__ y,
                                         T *__restrict__ dot,
                                         bool           conj,
                                         aoclsparse_int kid = -1)
{
    // Validations
    if(nullptr == dot)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(nnz <= 0)
    {
        *dot = 0;
        return aoclsparse_status_invalid_size;
    }
    if((nullptr == x) || (nullptr == indx) || (nullptr == y))
    {
        return aoclsparse_status_invalid_pointer;
    }

    using namespace aoclsparse;
    using namespace Dispatch;
    using namespace kernel_templates;

    // Creating pointer to the kernel
    using K = decltype(&dotp_ref<T>);

    // clang-format off
    // Table of available kernels
    static constexpr Table<K> tbl[]{
       {dotp_ref<T>,           context_isa_t::GENERIC, 0U | archs::ALL},
       {dotp_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ALL},
       {dotp_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ALL}, // alias
ORL<K>({dotp_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ALL})
    };
    // clang-format on

    // Thread local kernel cache
    thread_local K kache  = nullptr;
    K              kernel = Oracle<K>(tbl, kache, kid);

    if(!kernel)
        return aoclsparse_status_invalid_kid;

    // Invoke the kernel
    return kernel(nnz, x, indx, y, dot, conj);
}
#endif
