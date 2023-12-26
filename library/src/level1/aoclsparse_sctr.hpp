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
#include "aoclsparse_utils.hpp"

#include <complex>

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

/*
 * aoclsparse_scatter dispatcher
 */
template <typename T, Index::type I>
inline aoclsparse_status aoclsparse_scatter(aoclsparse_int nnz,
                                            const T *__restrict__ x,
                                            Index::index_t<I> xi,
                                            T *__restrict__ y,
                                            [[maybe_unused]] aoclsparse_int kid)
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

    return sctr_ref<T, I>(nnz, x, xi, y);
}