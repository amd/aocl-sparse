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
#ifndef AOCLSPARSE_DOT_HPP
#define AOCLSPARSE_DOT_HPP
#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_kernel_templates.hpp"
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

using namespace kernel_templates;

// This KT function performs dot product of a sparse vector (x) with a dense vector (y).
// Precision types supported: complex (float and double), real (float and double).
// For complex types, conjugated dot product is supported in addition to the dot product.
template <bsz SZ, typename SUF>
inline aoclsparse_status dotp_kt(aoclsparse_int nnz,
                                 const SUF *__restrict__ x,
                                 const aoclsparse_int *__restrict__ indx,
                                 const SUF *__restrict__ y,
                                 SUF *__restrict__ dot,
                                 bool conj)
{
    // Automatically determine the type of tsz
    const auto tsz = tsz_v<SZ, SUF>;

    avxvector_t<SZ, SUF> xv, yv, acc;

    // Initialize the accumulation vector to zero
    acc = kt_setzero_p<SZ, SUF>();

    aoclsparse_int count = nnz / tsz;
    aoclsparse_int rem   = nnz % tsz;

    // Conjugate path for complex types
    if constexpr(std::is_same_v<SUF, std::complex<double>>
                 || std::is_same_v<SUF, std::complex<float>>)
    {
        if(conj)
        {
            for(aoclsparse_int i = 0; i < count; ++i)
            {
                auto itsz = i * tsz;

                // Load the 'x' vector
                xv = kt_loadu_p<SZ, SUF>(x + itsz);

                // Conjugate 'x'
                xv = kt_conj_p<SZ, SUF>(xv);

                // Indirect load of 'y' vector
                yv = kt_set_p<SZ, SUF>(y, indx + itsz);

                // tmp += 'xv' * 'yv'
                acc = kt_fmadd_p<SZ, SUF>(xv, yv, acc);
            }

            // Accumulate the intermediate results in the vector
            *dot = kt_hsum_p<SZ, SUF>(acc);

            // Remainder part
            for(auto i = nnz - rem; i < nnz; i++)
            {
                *dot += std::conj(x[i]) * y[indx[i]];
            }

            return aoclsparse_status_success;
        }
    }

    for(aoclsparse_int i = 0; i < count; ++i)
    {
        // Load the 'x' vector
        xv = kt_loadu_p<SZ, SUF>(x + (i * tsz));

        // Indirect load of 'y' vector
        yv = kt_set_p<SZ, SUF>(y, indx + (i * tsz));

        // tmp += 'xv' * 'yv'
        acc = kt_fmadd_p<SZ, SUF>(xv, yv, acc);
    }

    // Accumulate the intermediate results in the vector
    *dot = kt_hsum_p<SZ, SUF>(acc);

    for(auto i = nnz - rem; i < nnz; i++)
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

    // Creating pointer to the kernel
    using K = decltype(&dotp_ref<T>);

    K kernel;

    // TODO: Replace with L1 dispatcher
    // Pick the kernel based on the KID passed
    switch(kid)
    {
    case 2:
#if defined(__AVX512F__)
        kernel = dotp_kt<bsz::b512, T>;
        break;
#endif
    case 1:
#if defined(__AVX2__)
        kernel = dotp_kt<bsz::b256, T>;
        break;
#endif
    default:
        kernel = dotp_ref;
    }

    // Invoke the kernel
    return kernel(nnz, x, indx, y, dot, conj);
}
#endif
