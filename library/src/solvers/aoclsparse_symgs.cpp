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
#include "aoclsparse_symgs.hpp"

#include <complex>
/*
 *===========================================================================
 *   C wrapper for SYMGS
 * ===========================================================================
 */

extern "C" aoclsparse_status aoclsparse_ssymgs(aoclsparse_operation       trans,
                                               aoclsparse_matrix          A,
                                               const aoclsparse_mat_descr descr,
                                               const float                alpha,
                                               const float               *b,
                                               float                     *x)
{
    const aoclsparse_int kid = -1; /* auto */
    return aoclsparse_symgs(trans, A, descr, alpha, b, x, (float *)nullptr, kid, false);
}

extern "C" aoclsparse_status aoclsparse_dsymgs(aoclsparse_operation       trans,
                                               aoclsparse_matrix          A,
                                               const aoclsparse_mat_descr descr,
                                               const double               alpha,
                                               const double              *b,
                                               double                    *x)
{
    const aoclsparse_int kid = -1; /* auto */
    return aoclsparse_symgs(trans, A, descr, alpha, b, x, (double *)nullptr, kid, false);
}

extern "C" aoclsparse_status aoclsparse_csymgs(aoclsparse_operation            trans,
                                               aoclsparse_matrix               A,
                                               const aoclsparse_mat_descr      descr,
                                               const aoclsparse_float_complex  alpha,
                                               const aoclsparse_float_complex *b,
                                               aoclsparse_float_complex       *x)
{
    const aoclsparse_int       kid    = -1; /* auto */
    const std::complex<float>  alphap = {alpha.real, alpha.imag};
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    return aoclsparse_symgs(
        trans, A, descr, alphap, bp, xp, (std::complex<float> *)nullptr, kid, false);
}
extern "C" aoclsparse_status aoclsparse_zsymgs(aoclsparse_operation             trans,
                                               aoclsparse_matrix                A,
                                               const aoclsparse_mat_descr       descr,
                                               const aoclsparse_double_complex  alpha,
                                               const aoclsparse_double_complex *b,
                                               aoclsparse_double_complex       *x)
{
    const aoclsparse_int        kid    = -1; /* auto */
    const std::complex<double>  alphap = {alpha.real, alpha.imag};
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    return aoclsparse_symgs(
        trans, A, descr, alphap, bp, xp, (std::complex<double> *)nullptr, kid, false);
}

extern "C" aoclsparse_status aoclsparse_ssymgs_kid(aoclsparse_operation       trans,
                                                   aoclsparse_matrix          A,
                                                   const aoclsparse_mat_descr descr,
                                                   const float                alpha,
                                                   const float               *b,
                                                   float                     *x,
                                                   const aoclsparse_int       kid)
{
    return aoclsparse_symgs(trans, A, descr, alpha, b, x, (float *)nullptr, kid, false);
}

extern "C" aoclsparse_status aoclsparse_dsymgs_kid(aoclsparse_operation       trans,
                                                   aoclsparse_matrix          A,
                                                   const aoclsparse_mat_descr descr,
                                                   const double               alpha,
                                                   const double              *b,
                                                   double                    *x,
                                                   const aoclsparse_int       kid)
{
    return aoclsparse_symgs(trans, A, descr, alpha, b, x, (double *)nullptr, kid, false);
}

extern "C" aoclsparse_status aoclsparse_csymgs_kid(aoclsparse_operation            trans,
                                                   aoclsparse_matrix               A,
                                                   const aoclsparse_mat_descr      descr,
                                                   const aoclsparse_float_complex  alpha,
                                                   const aoclsparse_float_complex *b,
                                                   aoclsparse_float_complex       *x,
                                                   const aoclsparse_int            kid)
{
    const std::complex<float>  alphap = {alpha.real, alpha.imag};
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    return aoclsparse_symgs(
        trans, A, descr, alphap, bp, xp, (std::complex<float> *)nullptr, kid, false);
}
extern "C" aoclsparse_status aoclsparse_zsymgs_kid(aoclsparse_operation             trans,
                                                   aoclsparse_matrix                A,
                                                   const aoclsparse_mat_descr       descr,
                                                   const aoclsparse_double_complex  alpha,
                                                   const aoclsparse_double_complex *b,
                                                   aoclsparse_double_complex       *x,
                                                   const aoclsparse_int             kid)
{
    const std::complex<double>  alphap = {alpha.real, alpha.imag};
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    return aoclsparse_symgs(
        trans, A, descr, alphap, bp, xp, (std::complex<double> *)nullptr, kid, false);
}

/*
 *===========================================================================
 *   C wrapper for SYMGS-MV
 * ===========================================================================
 */

/*
 * Performs Symmetric Gauss Seidel Preconditioned iterations to solve Ax = b followed
 * by a sparse product of input matrix A and the solution vector 'x'
 */
extern "C" aoclsparse_status aoclsparse_ssymgs_mv(aoclsparse_operation       trans,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  const float                alpha,
                                                  const float               *b,
                                                  float                     *x,
                                                  float                     *y)
{
    const aoclsparse_int kid = -1; /* auto */
    return aoclsparse_symgs(trans, A, descr, alpha, b, x, y, kid, true);
}

extern "C" aoclsparse_status aoclsparse_dsymgs_mv(aoclsparse_operation       trans,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  const double               alpha,
                                                  const double              *b,
                                                  double                    *x,
                                                  double                    *y)
{
    const aoclsparse_int kid = -1; /* auto */
    return aoclsparse_symgs(trans, A, descr, alpha, b, x, y, kid, true);
}

extern "C" aoclsparse_status aoclsparse_csymgs_mv(aoclsparse_operation            trans,
                                                  aoclsparse_matrix               A,
                                                  const aoclsparse_mat_descr      descr,
                                                  const aoclsparse_float_complex  alpha,
                                                  const aoclsparse_float_complex *b,
                                                  aoclsparse_float_complex       *x,
                                                  aoclsparse_float_complex       *y)
{
    const aoclsparse_int       kid    = -1; /* auto */
    const std::complex<float>  alphap = {alpha.real, alpha.imag};
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    std::complex<float>       *yp     = reinterpret_cast<std::complex<float> *>(y);
    return aoclsparse_symgs(trans, A, descr, alphap, bp, xp, yp, kid, true);
}

extern "C" aoclsparse_status aoclsparse_zsymgs_mv(aoclsparse_operation             trans,
                                                  aoclsparse_matrix                A,
                                                  const aoclsparse_mat_descr       descr,
                                                  const aoclsparse_double_complex  alpha,
                                                  const aoclsparse_double_complex *b,
                                                  aoclsparse_double_complex       *x,
                                                  aoclsparse_double_complex       *y)
{
    const aoclsparse_int        kid    = -1; /* auto */
    const std::complex<double>  alphap = {alpha.real, alpha.imag};
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    std::complex<double>       *yp     = reinterpret_cast<std::complex<double> *>(y);
    return aoclsparse_symgs(trans, A, descr, alphap, bp, xp, yp, kid, true);
}
extern "C" aoclsparse_status aoclsparse_ssymgs_mv_kid(aoclsparse_operation       trans,
                                                      aoclsparse_matrix          A,
                                                      const aoclsparse_mat_descr descr,
                                                      const float                alpha,
                                                      const float               *b,
                                                      float                     *x,
                                                      float                     *y,
                                                      const aoclsparse_int       kid)
{
    return aoclsparse_symgs(trans, A, descr, alpha, b, x, y, kid, true);
}

extern "C" aoclsparse_status aoclsparse_dsymgs_mv_kid(aoclsparse_operation       trans,
                                                      aoclsparse_matrix          A,
                                                      const aoclsparse_mat_descr descr,
                                                      const double               alpha,
                                                      const double              *b,
                                                      double                    *x,
                                                      double                    *y,
                                                      const aoclsparse_int       kid)
{
    return aoclsparse_symgs(trans, A, descr, alpha, b, x, y, kid, true);
}

extern "C" aoclsparse_status aoclsparse_csymgs_mv_kid(aoclsparse_operation            trans,
                                                      aoclsparse_matrix               A,
                                                      const aoclsparse_mat_descr      descr,
                                                      const aoclsparse_float_complex  alpha,
                                                      const aoclsparse_float_complex *b,
                                                      aoclsparse_float_complex       *x,
                                                      aoclsparse_float_complex       *y,
                                                      const aoclsparse_int            kid)
{
    const std::complex<float>  alphap = {alpha.real, alpha.imag};
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    std::complex<float>       *yp     = reinterpret_cast<std::complex<float> *>(y);
    return aoclsparse_symgs(trans, A, descr, alphap, bp, xp, yp, kid, true);
}

extern "C" aoclsparse_status aoclsparse_zsymgs_mv_kid(aoclsparse_operation             trans,
                                                      aoclsparse_matrix                A,
                                                      const aoclsparse_mat_descr       descr,
                                                      const aoclsparse_double_complex  alpha,
                                                      const aoclsparse_double_complex *b,
                                                      aoclsparse_double_complex       *x,
                                                      aoclsparse_double_complex       *y,
                                                      const aoclsparse_int             kid)
{
    const std::complex<double>  alphap = {alpha.real, alpha.imag};
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    std::complex<double>       *yp     = reinterpret_cast<std::complex<double> *>(y);
    return aoclsparse_symgs(trans, A, descr, alphap, bp, xp, yp, kid, true);
}