/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_trsv.hpp"

#include <complex>

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

extern "C" aoclsparse_status aoclsparse_strsv(aoclsparse_operation       trans,
                                              const float                alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              const float               *b,
                                              float                     *x)
{
    const aoclsparse_int kid  = -1; /* auto */
    const aoclsparse_int incb = 1, incx = 1;

    return aoclsparse_trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsv(aoclsparse_operation       trans,
                                              const double               alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              const double              *b,
                                              double                    *x)
{
    const aoclsparse_int kid  = -1; /* auto */
    const aoclsparse_int incb = 1, incx = 1;

    return aoclsparse_trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ctrsv(aoclsparse_operation            trans,
                                              const aoclsparse_float_complex  alpha,
                                              aoclsparse_matrix               A,
                                              const aoclsparse_mat_descr      descr,
                                              const aoclsparse_float_complex *b,
                                              aoclsparse_float_complex       *x)
{
    const aoclsparse_int       kid    = -1; /* auto */
    const std::complex<float> *alphap = reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    const aoclsparse_int       incb = 1, incx = 1;

    return aoclsparse_trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ztrsv(aoclsparse_operation             trans,
                                              const aoclsparse_double_complex  alpha,
                                              aoclsparse_matrix                A,
                                              const aoclsparse_mat_descr       descr,
                                              const aoclsparse_double_complex *b,
                                              aoclsparse_double_complex       *x)
{
    const aoclsparse_int        kid    = -1; /* auto */
    const std::complex<double> *alphap = reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    const aoclsparse_int        incb = 1, incx = 1;

    return aoclsparse_trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_strsv_strided(aoclsparse_operation       trans,
                                                      const float                alpha,
                                                      aoclsparse_matrix          A,
                                                      const aoclsparse_mat_descr descr,
                                                      const float               *b,
                                                      const aoclsparse_int       incb,
                                                      float                     *x,
                                                      const aoclsparse_int       incx)
{
    const aoclsparse_int kid = -1; /* auto */

    return aoclsparse_trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsv_strided(aoclsparse_operation       trans,
                                                      const double               alpha,
                                                      aoclsparse_matrix          A,
                                                      const aoclsparse_mat_descr descr,
                                                      const double              *b,
                                                      const aoclsparse_int       incb,
                                                      double                    *x,
                                                      const aoclsparse_int       incx)
{
    const aoclsparse_int kid = -1; /* auto */

    return aoclsparse_trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ctrsv_strided(aoclsparse_operation            trans,
                                                      const aoclsparse_float_complex  alpha,
                                                      aoclsparse_matrix               A,
                                                      const aoclsparse_mat_descr      descr,
                                                      const aoclsparse_float_complex *b,
                                                      const aoclsparse_int            incb,
                                                      aoclsparse_float_complex       *x,
                                                      const aoclsparse_int            incx)
{
    const aoclsparse_int       kid    = -1; /* auto */
    const std::complex<float> *alphap = reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);

    return aoclsparse_trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ztrsv_strided(aoclsparse_operation             trans,
                                                      const aoclsparse_double_complex  alpha,
                                                      aoclsparse_matrix                A,
                                                      const aoclsparse_mat_descr       descr,
                                                      const aoclsparse_double_complex *b,
                                                      const aoclsparse_int             incb,
                                                      aoclsparse_double_complex       *x,
                                                      const aoclsparse_int             incx)
{
    const aoclsparse_int        kid    = -1; /* auto */
    const std::complex<double> *alphap = reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);

    return aoclsparse_trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_strsv_kid(aoclsparse_operation       trans,
                                                  const float                alpha,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  const float               *b,
                                                  float                     *x,
                                                  const aoclsparse_int       kid)
{
    const aoclsparse_int incb = 1, incx = 1;

    return aoclsparse_trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsv_kid(aoclsparse_operation       trans,
                                                  const double               alpha,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  const double              *b,
                                                  double                    *x,
                                                  const aoclsparse_int       kid)
{
    const aoclsparse_int incb = 1, incx = 1;

    return aoclsparse_trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ctrsv_kid(aoclsparse_operation            trans,
                                                  const aoclsparse_float_complex  alpha,
                                                  aoclsparse_matrix               A,
                                                  const aoclsparse_mat_descr      descr,
                                                  const aoclsparse_float_complex *b,
                                                  aoclsparse_float_complex       *x,
                                                  const aoclsparse_int            kid)
{
    const std::complex<float> *alphap = reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    const aoclsparse_int       incb = 1, incx = 1;

    return aoclsparse_trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ztrsv_kid(aoclsparse_operation             trans,
                                                  const aoclsparse_double_complex  alpha,
                                                  aoclsparse_matrix                A,
                                                  const aoclsparse_mat_descr       descr,
                                                  const aoclsparse_double_complex *b,
                                                  aoclsparse_double_complex       *x,
                                                  const aoclsparse_int             kid)
{
    const std::complex<double> *alphap = reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    const aoclsparse_int        incb = 1, incx = 1;

    return aoclsparse_trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}
