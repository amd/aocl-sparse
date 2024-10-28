/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include "common_data_utils.h"
using kt_int_t = size_t;

#include "kernel-templates/kernel_templates.hpp"

using namespace kernel_templates;

namespace TestsKT
{
    template <bsz SZ, typename SUF>
    void kt_spmv(const size_t                  m,
                 [[maybe_unused]] const size_t n,
                 [[maybe_unused]] const size_t nnz,
                 const SUF *__restrict__ aval,
                 const size_t *__restrict__ icol,
                 const size_t *__restrict__ crstart,
                 const size_t *__restrict__ crend,
                 const SUF *__restrict__ x,
                 SUF *__restrict__ y)
    {

        avxvector_t<SZ, SUF> va, vx, vb;
        size_t               j;
        const size_t         k = tsz_v<SZ, SUF>;

        for(size_t i = 0; i < m; i++)
        {
            SUF result    = 0.0;
            vb            = kt_setzero_p<SZ, SUF>();
            size_t nnz    = crend[i] - crstart[i];
            size_t k_iter = nnz / k;
            size_t k_rem  = nnz % k;

            //Loop in multiples of K non-zeroes
            for(j = crstart[i]; j < crend[i] - k_rem; j += k)
            {
                va = kt_loadu_p<SZ, SUF>(&aval[j]);
                vx = kt_set_p<SZ, SUF>(x, &icol[j]);
                vb = kt_fmadd_p<SZ, SUF>(va, vx, vb);
            }
            if(k_iter)
            {
                // Horizontal addition
                result = kt_hsum_p<SZ, SUF>(vb);
            }
            //Remainder loop for nnz%k
            for(j = crend[i] - k_rem; j < crend[i]; j++)
            {
                result += aval[j] * x[icol[j]];
            }

            // Perform alpha * A * x
            // result *= alpha;
            // result += beta * y[i];
            y[i] = result;
        }
    }

}
#define KT_SPMV_TEST(BSZ, SUF)                                                   \
    template void TestsKT::kt_spmv<BSZ, SUF>(const size_t                  m,    \
                                             [[maybe_unused]] const size_t n,    \
                                             [[maybe_unused]] const size_t nnz,  \
                                             const SUF *__restrict__ aval,       \
                                             const size_t *__restrict__ icol,    \
                                             const size_t *__restrict__ crstart, \
                                             const size_t *__restrict__ crend,   \
                                             const SUF *__restrict__ x,          \
                                             SUF *__restrict__ y);

KT_SPMV_TEST(get_bsz(), double);
KT_SPMV_TEST(get_bsz(), float);
KT_SPMV_TEST(get_bsz(), std::complex<double>);
KT_SPMV_TEST(get_bsz(), std::complex<float>);
