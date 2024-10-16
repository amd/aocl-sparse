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
#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l2_kt.hpp"

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status aoclsparse::csrmv_kt(aoclsparse_index_base base,
                                       const SUF             alpha,
                                       const aoclsparse_int  m,
                                       const SUF *__restrict__ aval,
                                       const aoclsparse_int *__restrict__ icol,
                                       const aoclsparse_int *__restrict__ row,
                                       const SUF *__restrict__ x,
                                       const SUF beta,
                                       SUF *__restrict__ y)
{

    using namespace kernel_templates;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *aval_fix = aval - base;
    const SUF            *x_fix    = x - base;
    avxvector_t<SZ, SUF>  va, vx, vb;
    const size_t          k = tsz_v<SZ, SUF>;
    // Perform (beta * y)
    if(beta == static_cast<SUF>(0))
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<SUF>(1))
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }
#ifdef _OPENMP
    aoclsparse_int chunk = (m / context::get_context()->get_num_threads())
                               ? (m / context::get_context()->get_num_threads())
                               : 1;
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk) private(va, vx, vb)
#endif
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int j;
        SUF            result  = 0.0;
        vb                     = kt_setzero_p<SZ, SUF>();
        aoclsparse_int crend   = row[i + 1];
        aoclsparse_int crstart = row[i];
        aoclsparse_int nnz     = crend - crstart;
        aoclsparse_int k_iter  = nnz / k;
        aoclsparse_int k_rem   = nnz % k;

        //Loop in multiples of K non-zeroes
        for(j = crstart; j < crend - k_rem; j += k)
        {
            va = kt_loadu_p<SZ, SUF>(&aval_fix[j]);
            vx = kt_set_p<SZ, SUF>(x_fix, &icol_fix[j]);
            vb = kt_fmadd_p<SZ, SUF>(va, vx, vb);
        }
        if(k_iter)
        {
            // Horizontal addition
            result = kt_hsum_p<SZ, SUF>(vb);
        }
        //Remainder loop for nnz%k
        for(j = crend - k_rem; j < crend; j++)
        {
            result += aval_fix[j] * x_fix[icol_fix[j]];
        }

        // Perform alpha * A * x
        result *= alpha;
        y[i] += result;
    }
    return aoclsparse_status_success;
}

#define CSRMV_TEMPLATE_DECLARATION(BSZ, SUF)                   \
    template aoclsparse_status aoclsparse::csrmv_kt<BSZ, SUF>( \
        aoclsparse_index_base base,                            \
        const SUF             alpha,                           \
        const aoclsparse_int  m,                               \
        const SUF *__restrict__ aval,                          \
        const aoclsparse_int *__restrict__ icol,               \
        const aoclsparse_int *__restrict__ row,                \
        const SUF *__restrict__ x,                             \
        const SUF beta,                                        \
        SUF *__restrict__ y);

KT_INSTANTIATE(CSRMV_TEMPLATE_DECLARATION, kernel_templates::get_bsz());