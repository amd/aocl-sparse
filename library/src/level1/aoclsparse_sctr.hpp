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
#ifndef AOCLSPARSE_SCTR_HPP
#define AOCLSPARSE_SCTR_HPP
#endif

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_utils.hpp"

/*
 * Scatter reference implementation with stride or indexing
 * It is assumed that all pointers and data are valid.
 */
template <typename T, Index::type I>
inline aoclsparse_status
    sctr_ref(aoclsparse_int nnz, const T *__restrict__ x, Index::index_t<I> xi, T *__restrict__ y)
{
    aoclsparse_int i;

    for(i = 0; i < nnz; i++)
    {
        if constexpr(I == Index::type::strided)
        {
            // treat "xi" as a stride distance
            y[xi * i] = x[i];
        }
        else
        {
            // treat "xi" as an indexing array
            if(xi[i] < 0)
                return aoclsparse_status_invalid_index_value;

            y[xi[i]] = x[i];
        }
    }
    return aoclsparse_status_success;
}

using namespace kernel_templates;

/*
 * Scatter vector implementation with stride or indexing
 * It is assumed that all pointers and data are valid.
 * There is NO check for invalid indices
 */
template <bsz SZ, typename SUF, Index::type I>
inline aoclsparse_status sctr_kt(aoclsparse_int nnz,
                                 const SUF *__restrict__ x,
                                 Index::index_t<I> xi,
                                 SUF *__restrict__ y)
{
    avxvector_t<SZ, SUF> xv;

    // Automatically determine the type of tsz
    constexpr aoclsparse_int tsz = tsz_v<SZ, SUF>;

    aoclsparse_int count = nnz / tsz;
    aoclsparse_int rem   = nnz % tsz;

    if constexpr(I == Index::type::strided)
    {
        aoclsparse_int xstride[tsz];

        for(aoclsparse_int i = 0; i < tsz; ++i)
            xstride[i] = i * xi;

        for(aoclsparse_int i = 0; i < count; i++)
        {
            xv = kt_loadu_p<SZ, SUF>(x + (i * tsz));

            kt_scatter_p<SZ, SUF>(xv, y + ((i * xi) * tsz), xstride);
        }
    }
    else
    {
        for(aoclsparse_int i = 0; i < count; i++)
        {
            xv = kt_loadu_p<SZ, SUF>(x + (i * tsz));

            // treat "xi" as an indexing array
            kt_scatter_p<SZ, SUF>(xv, y, xi + (i * tsz));
        }
    }

    // Remainder
    for(aoclsparse_int i = nnz - rem; i < nnz; i++)
    {
        if constexpr(I == Index::type::strided)
        {
            // treat "xi" as a stride distance
            y[xi * i] = x[i];
        }
        else
        {
            // treat "xi" as an indexing array
            y[xi[i]] = x[i];
        }
    }
    return aoclsparse_status_success;
}

/*
 * aoclsparse_scatter dispatcher
 */
template <typename T, Index::type I>
inline aoclsparse_status aoclsparse_scatter(aoclsparse_int nnz,
                                            const T *__restrict__ x,
                                            Index::index_t<I> xi,
                                            T *__restrict__ y,
                                            aoclsparse_int kid)
{
    // Check pointer arguments
    if((nullptr == x) || (nullptr == y))
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(nnz == 0)
    {
        return aoclsparse_status_success;
    }

    //Check Size
    if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    if constexpr(I == Index::type::strided)
    {
        // xi is a stride distance, check distance
        if(xi <= 0)
            return aoclsparse_status_invalid_size;
    }
    else
    {
        // xi is an index array, check pointer
        if(xi == nullptr)
            return aoclsparse_status_invalid_pointer;
    }

    // Creating pointer to the kernel
    using K = decltype(&sctr_ref<T, I>);

    K kernel;

    // TODO: Replace with L1 dispatcher
    // Pick the kernel based on the KID passed
    switch(kid)
    {
    case 2:
#if defined(__AVX512F__)
        kernel = sctr_kt<bsz::b512, T, I>;
        break;
#endif
    case 1:
#if defined(__AVX2__)
        kernel = sctr_kt<bsz::b256, T, I>;
        break;
#endif
    default:
        kernel = sctr_ref<T, I>;
    }

    // Invoke the kernel
    return kernel(nnz, x, xi, y);
}