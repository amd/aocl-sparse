/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <complex>
#include <type_traits>

/*
 * Gather and gather_zero reference implementations
 * It is assumed that all pointers and data are valid.
 * Only check is made for indx to check that entries are not negative (remove if appropiate)
 */
template <typename T, bool L>
inline aoclsparse_status gthr_ref(const aoclsparse_int                  nnz,
                                  std::conditional_t<L, T *, const T *> y,
                                  T                                    *x,
                                  const aoclsparse_int                 *indx)
{
    if constexpr(L == true)
    {
        for(aoclsparse_int i = 0; i < nnz; ++i)
        {
            if(indx[i] < 0)
                return aoclsparse_status_invalid_index_value;
            x[i]       = y[indx[i]];
            y[indx[i]] = static_cast<T>(0);
        }
    }
    else
    {
        for(aoclsparse_int i = 0; i < nnz; ++i)
        {
            if(indx[i] < 0)
                return aoclsparse_status_invalid_index_value;
            x[i] = y[indx[i]];
        }
    }
    return aoclsparse_status_success;
}

/*
 * aoclsparse_gthr dispatcher
 * handles both cases gather (L:=false) and gatherz (L:=true)
 * Note that y is intent(inout) for gatherz and intent(in) otherwise.
 */
template <typename T, bool L>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int                  nnz,
                                  std::conditional_t<L, T *, const T *> y,
                                  T                                    *x,
                                  const aoclsparse_int                 *indx,
                                  const aoclsparse_int                  kid)
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
    if(y == nullptr || x == nullptr || indx == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    switch(kid)
    {
    default: // Reference implementation
        return gthr_ref<T, L>(nnz, y, x, indx);
        break;
    }

    return aoclsparse_status_success;
}

#endif /* AOCLSPARSE_GTHR_HPP */
