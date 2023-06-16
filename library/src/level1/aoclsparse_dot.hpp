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
#ifndef AOCLSPARSE_DOT_HPP
#define AOCLSPARSE_DOT_HPP
#endif

#include "aoclsparse.h"
#include "aoclsparse_context.h"

#include <complex>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

// The templated function performs dot product of a sparse vector (x) with a dense vector (y).
// Precision types supported: complex (float and double), real (float and double).
// For complex types, conjugated dot product is supported in addition to the dot product.
template <typename T>
inline aoclsparse_status dotp_ref(aoclsparse_int nnz,
                                  const T *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  const T *__restrict__ y,
                                  T *__restrict__ dot,
                                  bool                            conj,
                                  [[maybe_unused]] aoclsparse_int kid)
{
    aoclsparse_int i;
    *dot = 0;

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
                                         aoclsparse_int kid)
{
    // ToDo: switch based on kid.
    // At present calling the reference implementation

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
    return dotp_ref(nnz, x, indx, y, dot, conj, kid);
}
