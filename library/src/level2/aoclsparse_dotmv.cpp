/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include "aoclsparse_dotmv.hpp"

#include <complex>

extern "C" aoclsparse_status aoclsparse_sdotmv(const aoclsparse_operation op,
                                               const float                alpha,
                                               aoclsparse_matrix          A,
                                               const aoclsparse_mat_descr descr,
                                               const float               *x,
                                               const float                beta,
                                               float                     *y,
                                               float                     *d)
{
    aoclsparse_int kid = -1;
    return aoclsparse_dotmv_t(op, alpha, A, descr, x, beta, y, d, kid);
}

extern "C" aoclsparse_status aoclsparse_ddotmv(const aoclsparse_operation op,
                                               const double               alpha,
                                               aoclsparse_matrix          A,
                                               const aoclsparse_mat_descr descr,
                                               const double              *x,
                                               const double               beta,
                                               double                    *y,
                                               double                    *d)
{
    aoclsparse_int kid = -1;
    return aoclsparse_dotmv_t(op, alpha, A, descr, x, beta, y, d, kid);
}

extern "C" aoclsparse_status aoclsparse_cdotmv(const aoclsparse_operation      op,
                                               const aoclsparse_float_complex  alpha,
                                               aoclsparse_matrix               A,
                                               const aoclsparse_mat_descr      descr,
                                               const aoclsparse_float_complex *x,
                                               const aoclsparse_float_complex  beta,
                                               aoclsparse_float_complex       *y,
                                               aoclsparse_float_complex       *d)
{
    aoclsparse_int kid = -1;
    return aoclsparse_dotmv_t(op,
                              *((std::complex<float> *)&alpha),
                              A,
                              descr,
                              (std::complex<float> *)x,
                              *((std::complex<float> *)&beta),
                              (std::complex<float> *)y,
                              (std::complex<float> *)d,
                              kid);
}

extern "C" aoclsparse_status aoclsparse_zdotmv(const aoclsparse_operation       op,
                                               const aoclsparse_double_complex  alpha,
                                               aoclsparse_matrix                A,
                                               const aoclsparse_mat_descr       descr,
                                               const aoclsparse_double_complex *x,
                                               const aoclsparse_double_complex  beta,
                                               aoclsparse_double_complex       *y,
                                               aoclsparse_double_complex       *d)
{
    aoclsparse_int kid = -1;
    return aoclsparse_dotmv_t(op,
                              *((std::complex<double> *)&alpha),
                              A,
                              descr,
                              (std::complex<double> *)x,
                              *((std::complex<double> *)&beta),
                              (std::complex<double> *)y,
                              (std::complex<double> *)d,
                              kid);
}
