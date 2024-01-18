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

#include "aoclsparse.h"
#include "aoclsparse_sp2md.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Computes the product of two sparse matrices stored in compressed sparse row (CSR) format
 * and stores the result in a dense format. Supports s/d/c/z data types.
 */
extern "C" aoclsparse_status aoclsparse_ssp2md(const aoclsparse_operation opA,
                                               const aoclsparse_mat_descr descrA,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_operation opB,
                                               const aoclsparse_mat_descr descrB,
                                               const aoclsparse_matrix    B,
                                               const float                alpha,
                                               const float                beta,
                                               float                     *C,
                                               const aoclsparse_order     layout,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    return aoclsparse_sp2md_t(opA, descrA, A, opB, descrB, B, alpha, beta, C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_dsp2md(const aoclsparse_operation opA,
                                               const aoclsparse_mat_descr descrA,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_operation opB,
                                               const aoclsparse_mat_descr descrB,
                                               const aoclsparse_matrix    B,
                                               const double               alpha,
                                               const double               beta,
                                               double                    *C,
                                               const aoclsparse_order     layout,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    return aoclsparse_sp2md_t(opA, descrA, A, opB, descrB, B, alpha, beta, C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_csp2md(const aoclsparse_operation     opA,
                                               const aoclsparse_mat_descr     descrA,
                                               const aoclsparse_matrix        A,
                                               const aoclsparse_operation     opB,
                                               const aoclsparse_mat_descr     descrB,
                                               const aoclsparse_matrix        B,
                                               const aoclsparse_float_complex alpha,
                                               const aoclsparse_float_complex beta,
                                               aoclsparse_float_complex      *C,
                                               const aoclsparse_order         layout,
                                               const aoclsparse_int           ldc)

{
    const aoclsparse_int      kid    = -1; /* auto */
    const std::complex<float> calpha = *reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> cbeta  = *reinterpret_cast<const std::complex<float> *>(&beta);
    return aoclsparse_sp2md_t(
        opA, descrA, A, opB, descrB, B, calpha, cbeta, (std::complex<float> *)C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_zsp2md(const aoclsparse_operation      opA,
                                               const aoclsparse_mat_descr      descrA,
                                               const aoclsparse_matrix         A,
                                               const aoclsparse_operation      opB,
                                               const aoclsparse_mat_descr      descrB,
                                               const aoclsparse_matrix         B,
                                               const aoclsparse_double_complex alpha,
                                               const aoclsparse_double_complex beta,
                                               aoclsparse_double_complex      *C,
                                               const aoclsparse_order          layout,
                                               const aoclsparse_int            ldc)

{
    const aoclsparse_int       kid    = -1; /* auto */
    const std::complex<double> calpha = *reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> cbeta  = *reinterpret_cast<const std::complex<double> *>(&beta);
    return aoclsparse_sp2md_t(
        opA, descrA, A, opB, descrB, B, calpha, cbeta, (std::complex<double> *)C, layout, ldc, kid);
}