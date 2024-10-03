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
#include "aoclsparse_utils.hpp"

using namespace kernel_templates;

/*
 * Scatter vector implementation with stride or indexing
 * It is assumed that all pointers and data are valid.
 * There is NO check for invalid indices
 */
template <bsz SZ, typename SUF, Index::type I>
aoclsparse_status sctr_kt(aoclsparse_int nnz,
                          const SUF *__restrict__ x,
                          Index::index_t<I> xi,
                          SUF *__restrict__ y)
{
    avxvector_t<SZ, SUF> xv;

    // Automatically determine the type of tsz
    constexpr aoclsparse_int tsz = tsz_v<SZ, SUF>;

    aoclsparse_int count = nnz / tsz;
    aoclsparse_int rem   = nnz % tsz;

    if constexpr(I == Index::type::strided)
    {
        aoclsparse_int xstride[tsz];

        for(aoclsparse_int i = 0; i < tsz; ++i)
            xstride[i] = i * xi;

        for(aoclsparse_int i = 0; i < count; i++)
        {
            xv = kt_loadu_p<SZ, SUF>(x + (i * tsz));

            kt_scatter_p<SZ, SUF>(xv, y + ((i * xi) * tsz), xstride);
        }
    }
    else
    {
        for(aoclsparse_int i = 0; i < count; i++)
        {
            xv = kt_loadu_p<SZ, SUF>(x + (i * tsz));

            // treat "xi" as an indexing array
            kt_scatter_p<SZ, SUF>(xv, y, xi + (i * tsz));
        }
    }

    // Remainder
    for(aoclsparse_int i = nnz - rem; i < nnz; i++)
    {
        if constexpr(I == Index::type::strided)
        {
            // treat "xi" as a stride distance
            y[xi * i] = x[i];
        }
        else
        {
            // treat "xi" as an indexing array
            y[xi[i]] = x[i];
        }
    }
    return aoclsparse_status_success;
}

#define SCTR_TEMPLATE_DECLARATION(BSZ, SUF, I)       \
    template aoclsparse_status sctr_kt<BSZ, SUF, I>( \
        aoclsparse_int nnz, const SUF *__restrict__ x, Index::index_t<I> xi, SUF *__restrict__ y);

#define SCTR_IDX_TEMPLATE_DECLARATION(BSZ, SUF) \
    SCTR_TEMPLATE_DECLARATION(BSZ, SUF, Index::type::indexed)

#define SCTR_STR_TEMPLATE_DECLARATION(BSZ, SUF) \
    SCTR_TEMPLATE_DECLARATION(BSZ, SUF, Index::type::strided)

// Generates instantiation
KT_INSTANTIATE(SCTR_IDX_TEMPLATE_DECLARATION, get_bsz());
KT_INSTANTIATE(SCTR_STR_TEMPLATE_DECLARATION, get_bsz());