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
#ifndef AOCLSPARSE_ROTI_HPP
#define AOCLSPARSE_ROTI_HPP

#include "aoclsparse.h"
#include "aoclsparse_dispatcher.hpp"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_utils.hpp"

/*
 * Reference implementation for Givens rotation
 * It is assumed that all pointers and data are valid.
 * The indx vector is checked for non-negative values.
 */
template <typename T>
inline aoclsparse_status roti_ref(aoclsparse_int nnz,
                                  T *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  T *__restrict__ y,
                                  T c,
                                  T s)
{
    aoclsparse_int i;
    T              temp;
    for(i = 0; i < nnz; i++)
    {
        if(indx[i] < 0)
            return aoclsparse_status_invalid_index_value;

        temp       = x[i];
        x[i]       = c * x[i] + s * y[indx[i]];
        y[indx[i]] = c * y[indx[i]] - s * temp;
    }
    return aoclsparse_status_success;
}

using namespace kernel_templates;

/*
 * KT implementation for Givens rotation
 * It is assumed that all pointers and data are valid.
 * The indx vector is NOT checked for non-negative values.
 */
template <bsz SZ, typename SUF>
inline aoclsparse_status roti_kt(aoclsparse_int nnz,
                                 SUF *__restrict__ x,
                                 const aoclsparse_int *__restrict__ indx,
                                 SUF *__restrict__ y,
                                 SUF c,
                                 SUF s)
{
    // Automatically determine the type of tsz
    constexpr aoclsparse_int tsz = tsz_v<SZ, SUF>;

    aoclsparse_int vc  = nnz / tsz;
    aoclsparse_int rem = nnz % tsz;

    avxvector_t<SZ, SUF> yv, xv, cv, sv;
    avxvector_t<SZ, SUF> tmp[2];

    // Broadcast the constants
    cv = kt_set1_p<SZ, SUF>(c);
    sv = kt_set1_p<SZ, SUF>(s);

    for(aoclsparse_int i = 0; i < vc; i++)
    {
        // Load the vectors
        xv = kt_loadu_p<SZ, SUF>(x + (i * tsz));
        yv = kt_set_p<SZ, SUF>(y, indx + (i * tsz));

        // tmp = x * c
        tmp[0] = kt_mul_p<SZ, SUF>(cv, xv);

        // tmp = x * s
        tmp[1] = kt_mul_p<SZ, SUF>(sv, xv);

        // tmp = y * s + x_tmp
        tmp[0] = kt_fmadd_p<SZ, SUF>(sv, yv, tmp[0]);

        // y_tmp = x * c - x_tmp
        tmp[1] = kt_fmsub_p<SZ, SUF>(cv, yv, tmp[1]);

        // Store the results
        kt_storeu_p<SZ, SUF>(x + (i * tsz), tmp[0]);
        kt_scatter_p<SZ, SUF>(tmp[1], y, indx + (i * tsz));
    }

    SUF temp;

    for(aoclsparse_int i = nnz - rem; i < nnz; i++)
    {
        temp       = x[i];
        x[i]       = c * x[i] + s * y[indx[i]];
        y[indx[i]] = c * y[indx[i]] - s * temp;
    }

    return aoclsparse_status_success;
}

/*
 * aoclsparse_roti dispatcher
 */
template <typename T>
inline aoclsparse_status aoclsparse_rot(aoclsparse_int nnz,
                                        T *__restrict__ x,
                                        const aoclsparse_int *__restrict__ indx,
                                        T *__restrict__ y,
                                        T              c,
                                        T              s,
                                        aoclsparse_int kid = -1)
{
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

    using namespace aoclsparse;
    using namespace Dispatch;

    // Creating pointer to the kernel
    using K = decltype(&roti_ref<T>);

    // clang-format off
    // Table of available kernels
    static constexpr Table<K> tbl[]{
    {roti_ref<T>,           context_isa_t::GENERIC, 0U | archs::ALL},
#ifdef __AVX2__
    {roti_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ZEN123},
#endif
#ifdef __AVX512F__
    {roti_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ZEN4}
#endif
    };
    // clang-format on

    // Inquire with the oracle
    auto kernel = Oracle<K, api::roti>(tbl, kid);

    if(!kernel)
        return aoclsparse_status_invalid_kid;

    return kernel(nnz, x, indx, y, c, s);
}
#endif
