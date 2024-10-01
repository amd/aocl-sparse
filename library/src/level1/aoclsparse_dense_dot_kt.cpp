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
 * ************************************************************************ */

#include "aoclsparse.h"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l1_kt.hpp"
#include "aoclsparse_utils.hpp"

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status aoclsparse::dot_kt(const aoclsparse_int size,
                                     const SUF *__restrict__ x,
                                     const SUF *__restrict__ y,
                                     SUF *__restrict__ d)
{

    using namespace kernel_templates;

    // Number of elements to fit in vector
    const aoclsparse_int tsz = tsz_v<SZ, SUF>;
    avxvector_t<SZ, SUF> xv, yv, tmp;

    // Initialize the accumulation vector to zero
    tmp = kt_setzero_p<SZ, SUF>();

    aoclsparse_int vc  = size / tsz;
    aoclsparse_int rem = size % tsz;

    for(aoclsparse_int i = 0; i < vc; ++i)
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
    for(aoclsparse_int i = size - rem; i < size; i++)
    {
        *d += aoclsparse::conj(x[i]) * y[i];
    }
    return aoclsparse_status_success;
}

#define DENSE_DOT_TEMPLATE_DECLARATION(BSZ, SUF)                                       \
    template aoclsparse_status aoclsparse::dot_kt<BSZ, SUF>(const aoclsparse_int size, \
                                                            const SUF *__restrict__ x, \
                                                            const SUF *__restrict__ y, \
                                                            SUF *__restrict__ d);

KT_INSTANTIATE(DENSE_DOT_TEMPLATE_DECLARATION, kernel_templates::get_bsz())