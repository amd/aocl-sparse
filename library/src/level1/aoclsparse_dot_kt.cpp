/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_kernel_templates.hpp"

using namespace kernel_templates;

// This KT function performs dot product of a sparse vector (x) with a dense vector (y).
// Precision types supported: complex (float and double), real (float and double).
// For complex types, conjugated dot product is supported in addition to the dot product.
template <bsz SZ, typename SUF>
aoclsparse_status dotp_kt(aoclsparse_int nnz,
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

#define DOT_TEMPLATE_DECLARATION(BSZ, SUF)                                                \
    template aoclsparse_status dotp_kt<BSZ, SUF>(aoclsparse_int nnz,                      \
                                                 const SUF *__restrict__ x,               \
                                                 const aoclsparse_int *__restrict__ indx, \
                                                 const SUF *__restrict__ y,               \
                                                 SUF *__restrict__ dot,                   \
                                                 bool conj)

KT_INSTANTIATE(DOT_TEMPLATE_DECLARATION, get_bsz());
