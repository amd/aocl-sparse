/* ************************************************************************
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_csrmm.hpp"

#include <algorithm>
#include <cmath>

extern "C" aoclsparse_status aoclsparse_scsrmm(aoclsparse_operation       op,
                                               const float                alpha,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_mat_descr descr,
                                               aoclsparse_order           order,
                                               const float               *B,
                                               aoclsparse_int             n,
                                               aoclsparse_int             ldb,
                                               const float                beta,
                                               float                     *C,
                                               aoclsparse_int             ldc)
{
    const aoclsparse_int kid = -1;

    return aoclsparse_csrmm<float>(op, alpha, A, descr, order, B, n, ldb, beta, C, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_dcsrmm(aoclsparse_operation       op,
                                               const double               alpha,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_mat_descr descr,
                                               aoclsparse_order           order,
                                               const double              *B,
                                               aoclsparse_int             n,
                                               aoclsparse_int             ldb,
                                               const double               beta,
                                               double                    *C,
                                               aoclsparse_int             ldc)
{
    const aoclsparse_int kid = -1;
    return aoclsparse_csrmm<double>(op, alpha, A, descr, order, B, n, ldb, beta, C, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_ccsrmm(aoclsparse_operation            op,
                                               const aoclsparse_float_complex  alpha,
                                               const aoclsparse_matrix         A,
                                               const aoclsparse_mat_descr      descr,
                                               aoclsparse_order                order,
                                               const aoclsparse_float_complex *B,
                                               aoclsparse_int                  n,
                                               aoclsparse_int                  ldb,
                                               const aoclsparse_float_complex  beta,
                                               aoclsparse_float_complex       *C,
                                               aoclsparse_int                  ldc)
{
    const aoclsparse_int       kid    = -1;
    const std::complex<float>  alphap = *reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float>  betap  = *reinterpret_cast<const std::complex<float> *>(&beta);
    const std::complex<float> *b      = reinterpret_cast<const std::complex<float> *>(B);
    std::complex<float>       *c      = reinterpret_cast<std::complex<float> *>(C);

    return aoclsparse_csrmm<std::complex<float>>(
        op, alphap, A, descr, order, b, n, ldb, betap, c, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_zcsrmm(aoclsparse_operation             op,
                                               const aoclsparse_double_complex  alpha,
                                               const aoclsparse_matrix          A,
                                               const aoclsparse_mat_descr       descr,
                                               aoclsparse_order                 order,
                                               const aoclsparse_double_complex *B,
                                               aoclsparse_int                   n,
                                               aoclsparse_int                   ldb,
                                               const aoclsparse_double_complex  beta,
                                               aoclsparse_double_complex       *C,
                                               aoclsparse_int                   ldc)
{
    const aoclsparse_int        kid    = -1;
    const std::complex<double>  alphap = *reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double>  betap  = *reinterpret_cast<const std::complex<double> *>(&beta);
    const std::complex<double> *b      = reinterpret_cast<const std::complex<double> *>(B);
    std::complex<double>       *c      = reinterpret_cast<std::complex<double> *>(C);

    return aoclsparse_csrmm<std::complex<double>>(
        op, alphap, A, descr, order, b, n, ldb, betap, c, ldc, kid);
}
