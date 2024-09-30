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

template <bsz SZ, typename SUF>
aoclsparse_status axpyi_kt(aoclsparse_int nnz,
                           SUF            a,
                           const SUF *__restrict__ x,
                           const aoclsparse_int *__restrict__ indx,
                           SUF *__restrict__ y)
{
    const aoclsparse_int tsz = tsz_v<SZ, SUF>;
    aoclsparse_int       i   = 0;
    avxvector_t<SZ, SUF> xvec, yvec;
    avxvector_t<SZ, SUF> alpha = kt_set1_p<SZ, SUF>(a);

    for(; (i + (tsz - 1)) < nnz; i += tsz)
    {

        xvec = kt_loadu_p<SZ, SUF>(x + i);
        yvec = kt_set_p<SZ, SUF>(y, &indx[i]);

        yvec = kt_fmadd_p<SZ, SUF>(alpha, xvec, yvec);

        kt_scatter_p<SZ, SUF>(yvec, y, indx + i);
    }

    for(; i < nnz; i++)
    {
        y[indx[i]] = a * x[i] + y[indx[i]];
    }

    return aoclsparse_status_success;
}

#define AXPYI_TEMPLATE_DECLARATION(BSZ, DTYPE)                                               \
    template aoclsparse_status axpyi_kt<BSZ, DTYPE>(aoclsparse_int nnz,                      \
                                                    DTYPE          a,                        \
                                                    const DTYPE *__restrict__ x,             \
                                                    const aoclsparse_int *__restrict__ indx, \
                                                    DTYPE *__restrict__ y)

KT_INSTANTIATE(AXPYI_TEMPLATE_DECLARATION, get_bsz());
