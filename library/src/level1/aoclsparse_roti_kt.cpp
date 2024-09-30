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

#define ROTI_TEMPLATE_DECLARATION(BSZ, SUF)                                               \
    template aoclsparse_status roti_kt<BSZ, SUF>(aoclsparse_int nnz,                      \
                                                 SUF *__restrict__ x,                     \
                                                 const aoclsparse_int *__restrict__ indx, \
                                                 SUF *__restrict__ y,                     \
                                                 SUF c,                                   \
                                                 SUF s)

KT_INSTANTIATE(ROTI_TEMPLATE_DECLARATION, get_bsz());
