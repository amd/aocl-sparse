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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_AXPYI_HPP
#define AOCLSPARSE_AXPYI_HPP
#endif

#include "aoclsparse.h"
#include "aoclsparse_context.h"

#include <complex>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

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
                                   T *__restrict__ y,
                                   [[maybe_unused]] aoclsparse_int kid)
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
inline aoclsparse_status aoclsparse_axpyi_t(aoclsparse_int nnz,
                                            T              a,
                                            const T *__restrict__ x,
                                            const aoclsparse_int *__restrict__ indx,
                                            T *__restrict__ y,
                                            [[maybe_unused]] aoclsparse_int kid)
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
    else
    {
        return axpyi_ref<T>(nnz, a, x, indx, y, kid);
    }
}
