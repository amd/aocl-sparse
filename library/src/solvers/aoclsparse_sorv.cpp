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
#include "aoclsparse_sorv.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
extern "C" aoclsparse_status aoclsparse_ssorv(aoclsparse_sor_type        sor_type,
                                              const aoclsparse_mat_descr descr,
                                              const aoclsparse_matrix    A,
                                              float                      omega,
                                              float                      alpha,
                                              float                     *x,
                                              const float               *b)
{
    aoclsparse_int kid = -1;
    return aoclsparse_sorv_t(sor_type, descr, A, omega, alpha, x, b, kid);
}
extern "C" aoclsparse_status aoclsparse_dsorv(aoclsparse_sor_type        sor_type,
                                              const aoclsparse_mat_descr descr,
                                              const aoclsparse_matrix    A,
                                              double                     omega,
                                              double                     alpha,
                                              double                    *x,
                                              const double              *b)
{
    aoclsparse_int kid = -1;
    return aoclsparse_sorv_t(sor_type, descr, A, omega, alpha, x, b, kid);
}
extern "C" aoclsparse_status aoclsparse_csorv(aoclsparse_sor_type             sor_type,
                                              const aoclsparse_mat_descr      descr,
                                              const aoclsparse_matrix         A,
                                              aoclsparse_float_complex        omega,
                                              aoclsparse_float_complex        alpha,
                                              aoclsparse_float_complex       *x,
                                              const aoclsparse_float_complex *b)
{
    aoclsparse_int             kid    = -1;
    std::complex<float>       *omegap = reinterpret_cast<std::complex<float> *>(&omega);
    std::complex<float>       *alphap = reinterpret_cast<std::complex<float> *>(&alpha);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    return aoclsparse_sorv_t(sor_type, descr, A, *omegap, *alphap, xp, bp, kid);
}
extern "C" aoclsparse_status aoclsparse_zsorv(aoclsparse_sor_type              sor_type,
                                              const aoclsparse_mat_descr       descr,
                                              const aoclsparse_matrix          A,
                                              aoclsparse_double_complex        omega,
                                              aoclsparse_double_complex        alpha,
                                              aoclsparse_double_complex       *x,
                                              const aoclsparse_double_complex *b)
{
    aoclsparse_int              kid    = -1;
    std::complex<double>       *omegap = reinterpret_cast<std::complex<double> *>(&omega);
    std::complex<double>       *alphap = reinterpret_cast<std::complex<double> *>(&alpha);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    return aoclsparse_sorv_t(sor_type, descr, A, *omegap, *alphap, xp, bp, kid);
}