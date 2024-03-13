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
#ifndef AOCLSPARSE_DOTMV_HPP
#define AOCLSPARSE_DOTMV_HPP

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_mv.hpp"

using namespace kernel_templates;
// The templated function performs dot product of two dense vectors, x nd y.
// Precision types supported: complex (float and double), real (float and double).
// For complex types, conjugated dot product is supported in addition to the dot product.
template <typename T>
aoclsparse_status aoclsparse_dot_ref(const aoclsparse_int size,
                                     const T *__restrict__ x,
                                     const T *__restrict__ y,
                                     T *__restrict__ d)
{
    aoclsparse_int i;
    *d = aoclsparse_numeric::zero<T>();

    for(i = 0; i < size; i++)
    {
        *d += aoclsparse::conj(x[i]) * y[i];
    }
    return aoclsparse_status_success;
}

template <bsz SZ, typename SUF>
inline aoclsparse_status aoclsparse_dot_kt(const aoclsparse_int size,
                                           const SUF *__restrict__ x,
                                           const SUF *__restrict__ y,
                                           SUF *__restrict__ d)
{
    // Number of elements to fit in vector
    const auto           tsz = tsz_v<SZ, SUF>;
    avxvector_t<SZ, SUF> xv, yv, tmp;

    // Initialize the accumulation vector to zero
    tmp = kt_setzero_p<SZ, SUF>();

    aoclsparse_int vc  = size / tsz;
    aoclsparse_int rem = size % tsz;

    for(auto i = 0U; i < vc; ++i)
    {
        // Load the 'x' vector
        xv = kt_loadu_p<SZ, SUF>(x + (i * tsz));
        // Conjugate 'x'
        xv = kt_conj_p<SZ, SUF>(xv);
        // Load the 'y' vector
        yv = kt_loadu_p<SZ, SUF>(y + (i * tsz));
        // tmp += 'xv' * 'yv'
        tmp = kt_fmadd_p<SZ, SUF>(xv, yv, tmp);
    }
    // Accumulate the intermediate results in the vector
    *d = kt_hsum_p<SZ, SUF>(tmp);
    // Remainder part that cannot be vectorized
    for(auto i = size - rem; i < size; i++)
    {
        *d += aoclsparse::conj(x[i]) * y[i];
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_dotmv_t(const aoclsparse_operation op,
                                     const T                    alpha,
                                     aoclsparse_matrix          A,
                                     const aoclsparse_mat_descr descr,
                                     const T                   *x,
                                     const T                    beta,
                                     T                         *y,
                                     T                         *d,
                                     aoclsparse_int             kid)
{
    aoclsparse_status status;

    if(d == nullptr)
    {
        // Remaining validations are done in mv and mv_t functions
        return aoclsparse_status_invalid_pointer;
    }

    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        status = aoclsparse_mv(op, alpha, A, descr, x, beta, y);
    else
        status = aoclsparse_mv_t(op, alpha, A, descr, x, beta, y);

    if(status)
    {
        return status;
    }

    // Creating pointer to the kernel
    using K = decltype(&aoclsparse_dot_ref<T>);
    K dot_kernel;

    /* Default AVX2.
     * TODO: Use the framework to read kid from context once it is available.
     */
    kid = 1;
    // Pick the kernel based on the KID passed
    switch(kid)
    {
    case 2:
#if defined(__AVX512F__)
        dot_kernel = aoclsparse_dot_kt<bsz::b512, T>;
        break;
#endif
    case 1:
#if defined(__AVX2__)
        dot_kernel = aoclsparse_dot_kt<bsz::b256, T>;
        break;
#endif
    default:
        dot_kernel = aoclsparse_dot_ref;
    }

    /* Dot product needs x and y of same size but
     * op = non-transpose, size of y=m, x=n
     * op = transpose, size of y=n, x=m
     * hence, taking minimum of m and n
     */
    return dot_kernel(std::min(A->m, A->n), x, y, d);
}

#endif // AOCLSPARSE_DOTMV_HPP
