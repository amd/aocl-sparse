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

template <bsz SZ, typename SUF, gather_op OP, kt_avxext EXT, Index::type I>
aoclsparse_status gthr_kt(aoclsparse_int nnz, y_type<SUF, OP> y, SUF *x, Index::index_t<I> xi)
{
    avxvector_t<SZ, SUF> yv;

    y_type<SUF, OP> yp;

    // Automatically determine the type of tsz
    constexpr auto tsz = tsz_v<SZ, SUF>;

    aoclsparse_int count = nnz / tsz;
    aoclsparse_int rem   = nnz % tsz;

    if constexpr(I == Index::type::strided)
    {
        aoclsparse_int xstride[tsz];

        for(auto i = 0U; i < tsz; ++i)
            xstride[i] = i * xi;

        for(aoclsparse_int i = 0U; i < count; ++i)
        {
            yv = kt_maskz_set_p<SZ, SUF, EXT, tsz>(y + ((i * xi) * tsz), xstride);

            kt_storeu_p<SZ, SUF>(x + (i * tsz), yv);
        }
    }
    else if(I == Index::type::indexed)
    {
        for(aoclsparse_int i = 0; i < count; ++i)
        {
            yv = kt_set_p<SZ, SUF>(y, xi + (i * tsz));

            kt_storeu_p<SZ, SUF>(x + (i * tsz), yv);

            if constexpr(OP == gather_op::gatherz)
            {
                kt_scatter_p<SZ, SUF>(kt_setzero_p<SZ, SUF>(), y, xi + (i * tsz));
            }
        }
    }

    for(auto i = nnz - rem; i < nnz; ++i)
    {
        if constexpr(I == Index::type::strided)
        {
            // treat "xi" as a stride distance
            yp = &y[xi * i];
        }
        else if constexpr(I == Index::type::indexed)
        {
            yp = &y[xi[i]];
        }

        x[i] = *yp; // copy out

        if constexpr(OP == gather_op::gatherz)
        {
            *yp = aoclsparse_numeric::zero<SUF>(); // zero out
        }
    }

    return aoclsparse_status_success;
}

// Template declaration macro used for instantiation
#define GTHR_IDX_TEMPLATE_DECLARATION(BSZ, SUF)                                   \
    template aoclsparse_status                                                    \
        gthr_kt<BSZ, SUF, gather_op::gather, get_kt_ext(), Index::type::indexed>( \
            aoclsparse_int                 nnz,                                   \
            y_type<SUF, gather_op::gather> y,                                     \
            SUF * x,                                                              \
            Index::index_t<Index::type::indexed> xi)

#define GTHRZ_IDX_TEMPLATE_DECLARATION(BSZ, SUF)                                   \
    template aoclsparse_status                                                     \
        gthr_kt<BSZ, SUF, gather_op::gatherz, get_kt_ext(), Index::type::indexed>( \
            aoclsparse_int                  nnz,                                   \
            y_type<SUF, gather_op::gatherz> y,                                     \
            SUF * x,                                                               \
            Index::index_t<Index::type::indexed> xi)

#define GTHR_STR_TEMPLATE_DECLARATION(BSZ, SUF)                                   \
    template aoclsparse_status                                                    \
        gthr_kt<BSZ, SUF, gather_op::gather, get_kt_ext(), Index::type::strided>( \
            aoclsparse_int                 nnz,                                   \
            y_type<SUF, gather_op::gather> y,                                     \
            SUF * x,                                                              \
            Index::index_t<Index::type::strided> xi)

// Generates instantiation
KT_INSTANTIATE(GTHR_IDX_TEMPLATE_DECLARATION, get_bsz());
KT_INSTANTIATE(GTHRZ_IDX_TEMPLATE_DECLARATION, get_bsz());
KT_INSTANTIATE(GTHR_STR_TEMPLATE_DECLARATION, get_bsz());
