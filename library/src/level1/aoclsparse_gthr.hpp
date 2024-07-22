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
 * ************************************************************************ */

#ifndef AOCLSPARSE_GTHR_HPP
#define AOCLSPARSE_GTHR_HPP

#include "aoclsparse.h"
#include "aoclsparse_dispatcher.hpp"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_utils.hpp"

/* Gather operation types and requirements for input/output vector y.
 * gather -> y be of type const T *
 * gatherz -> y be of type T *
 */
enum class gather_op
{
    gather,
    gatherz
};
template <typename T, gather_op OP>
using y_type = typename std::conditional<OP == gather_op::gather, const T *, T *>::type;

/* Gather and gather_zero reference implementations with stride or indexing
 * It is assumed that all pointers and data are valid.
 */
template <typename T, gather_op OP, Index::type I>
inline aoclsparse_status
    gthr_ref(const aoclsparse_int nnz, y_type<T, OP> y, T *x, Index::index_t<I> xi)
{
    y_type<T, OP> yp;

    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        if constexpr(I == Index::type::strided)
        {
            // treat "xi" as a stride distance
            yp = &y[xi * i];
        }
        else if constexpr(I == Index::type::indexed)
        {
            // treat "xi" as an indexing array
            if(xi[i] < 0)
            {
                return aoclsparse_status_invalid_index_value;
            }

            yp = &y[xi[i]];
        }

        x[i] = *yp; // copy out

        if constexpr(OP == gather_op::gatherz)
        {
            *yp = aoclsparse_numeric::zero<T>(); // zero out
        }
    }

    return aoclsparse_status_success;
}

using namespace kernel_templates;

template <bsz SZ, typename SUF, gather_op OP, kt_avxext EXT, Index::type I>
inline aoclsparse_status
    gthr_kt(aoclsparse_int nnz, y_type<SUF, OP> y, SUF *x, Index::index_t<I> xi)
{
    avxvector_t<SZ, SUF> yv;

    y_type<SUF, OP> yp;

    // Automatically determine the type of tsz
    constexpr auto tsz = tsz_v<SZ, SUF>;

    aoclsparse_int count = nnz / tsz;
    aoclsparse_int rem   = nnz % tsz;

    if constexpr(I == Index::type::strided)
    {
        aoclsparse_int xstride[tsz];

        for(auto i = 0U; i < tsz; ++i)
            xstride[i] = i * xi;

        for(aoclsparse_int i = 0U; i < count; ++i)
        {
            yv = kt_maskz_set_p<SZ, SUF, EXT, tsz>(y + ((i * xi) * tsz), xstride);

            kt_storeu_p<SZ, SUF>(x + (i * tsz), yv);
        }
    }
    else if(I == Index::type::indexed)
    {
        for(aoclsparse_int i = 0; i < count; ++i)
        {
            yv = kt_set_p<SZ, SUF>(y, xi + (i * tsz));

            kt_storeu_p<SZ, SUF>(x + (i * tsz), yv);

            if constexpr(OP == gather_op::gatherz)
            {
                kt_scatter_p<SZ, SUF>(kt_setzero_p<SZ, SUF>(), y, xi + (i * tsz));
            }
        }
    }

    for(auto i = nnz - rem; i < nnz; ++i)
    {
        if constexpr(I == Index::type::strided)
        {
            // treat "xi" as a stride distance
            yp = &y[xi * i];
        }
        else if constexpr(I == Index::type::indexed)
        {
            yp = &y[xi[i]];
        }

        x[i] = *yp; // copy out

        if constexpr(OP == gather_op::gatherz)
        {
            *yp = aoclsparse_numeric::zero<SUF>(); // zero out
        }
    }

    return aoclsparse_status_success;
}

template <typename T, gather_op OP, Index::type I>
constexpr Dispatch::api get_gthr_id()
{
    if constexpr(OP == gather_op::gather && I == Index::type::indexed)
        return Dispatch::api::gthr;
    else if constexpr(OP == gather_op::gatherz && I == Index::type::indexed)
        return Dispatch::api::gthrz;
    else if constexpr(OP == gather_op::gather && I == Index::type::strided)
        return Dispatch::api::gthrs;
}

/*
 * aoclsparse_gthrs dispatcher with strided or indexed array access
 * handles both cases gather and gatherz
 * Note that y is inout for gatherz and in otherwise.
 */
template <typename T, gather_op OP, Index::type I>
aoclsparse_status aoclsparse_gthr_t(
    aoclsparse_int nnz, y_type<T, OP> y, T *x, Index::index_t<I> xi, aoclsparse_int kid = -1)
{
    // Check size
    if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(nnz == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(y == nullptr || x == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if constexpr(I == Index::type::strided)
    {
        // xi is a stride distance, check distance
        if(xi < 0)
            return aoclsparse_status_invalid_size;
    }
    else
    {
        // xi is an index array, check pointer
        if(xi == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
    }

    using namespace aoclsparse;
    using namespace Dispatch;

    // Creating pointer to the kernel
    using K = decltype(&gthr_ref<T, OP, I>);

    // clang-format off
    // Table of available kernels
    static constexpr Table<K> tbl[]{
    {gthr_ref<T, OP, I>,           context_isa_t::GENERIC, 0U | archs::ALL},
#ifdef __AVX2__
    {gthr_kt<bsz::b256, T, OP, kt_avxext::AVX2, I>, context_isa_t::AVX2,    0U | archs::ZEN123},
#endif
#ifdef __AVX512F__
    {gthr_kt<bsz::b512, T, OP, kt_avxext::AVX2, I>, context_isa_t::AVX512F, 0U | archs::ZEN4}
#endif
    };
    // clang-format on

    // Inquire with the oracle
    auto kernel = Oracle<K, get_gthr_id<T, OP, I>()>(tbl, kid);

    if(!kernel)
        return aoclsparse_status_invalid_kid;

    // Invoke the kernel
    return kernel(nnz, y, x, xi);
}

#endif /* AOCLSPARSE_GTHR_HPP */
