/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_trsm.hpp"

#include <complex>
/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

extern "C" aoclsparse_status aoclsparse_strsm(const aoclsparse_operation trans,
                                              const float                alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              aoclsparse_order           order,
                                              const float               *B,
                                              aoclsparse_int             n,
                                              aoclsparse_int             ldb,
                                              float                     *X,
                                              aoclsparse_int             ldx)
{
    const aoclsparse_int kid = -1; /* auto */

    return aoclsparse_trsm(trans, alpha, A, descr, order, B, n, ldb, X, ldx, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsm(const aoclsparse_operation trans,
                                              const double               alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              aoclsparse_order           order,
                                              const double              *B,
                                              aoclsparse_int             n,
                                              aoclsparse_int             ldb,
                                              double                    *X,
                                              aoclsparse_int             ldx)
{
    const aoclsparse_int kid = -1; /* auto */

    return aoclsparse_trsm(trans, alpha, A, descr, order, B, n, ldb, X, ldx, kid);
}

extern "C" aoclsparse_status aoclsparse_ctrsm(const aoclsparse_operation      trans,
                                              const aoclsparse_float_complex  alpha,
                                              aoclsparse_matrix               A,
                                              const aoclsparse_mat_descr      descr,
                                              aoclsparse_order                order,
                                              const aoclsparse_float_complex *B,
                                              aoclsparse_int                  n,
                                              aoclsparse_int                  ldb,
                                              aoclsparse_float_complex       *X,
                                              aoclsparse_int                  ldx)
{
    const aoclsparse_int       kid    = -1; /* auto */
    const std::complex<float> *alphap = reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> *Bp     = reinterpret_cast<const std::complex<float> *>(B);
    std::complex<float>       *Xp     = reinterpret_cast<std::complex<float> *>(X);
    return aoclsparse_trsm(trans, *alphap, A, descr, order, Bp, n, ldb, Xp, ldx, kid);
}
extern "C" aoclsparse_status aoclsparse_ztrsm(const aoclsparse_operation       trans,
                                              const aoclsparse_double_complex  alpha,
                                              aoclsparse_matrix                A,
                                              const aoclsparse_mat_descr       descr,
                                              aoclsparse_order                 order,
                                              const aoclsparse_double_complex *B,
                                              aoclsparse_int                   n,
                                              aoclsparse_int                   ldb,
                                              aoclsparse_double_complex       *X,
                                              aoclsparse_int                   ldx)
{
    const aoclsparse_int        kid    = -1; /* auto */
    const std::complex<double> *alphap = reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> *Bp     = reinterpret_cast<const std::complex<double> *>(B);
    std::complex<double>       *Xp     = reinterpret_cast<std::complex<double> *>(X);
    return aoclsparse_trsm(trans, *alphap, A, descr, order, Bp, n, ldb, Xp, ldx, kid);
}

extern "C" aoclsparse_status aoclsparse_strsm_kid(const aoclsparse_operation trans,
                                                  const float                alpha,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  aoclsparse_order           order,
                                                  const float               *B,
                                                  aoclsparse_int             n,
                                                  aoclsparse_int             ldb,
                                                  float                     *X,
                                                  aoclsparse_int             ldx,
                                                  const aoclsparse_int       kid)
{
    return aoclsparse_trsm(trans, alpha, A, descr, order, B, n, ldb, X, ldx, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsm_kid(const aoclsparse_operation trans,
                                                  const double               alpha,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  aoclsparse_order           order,
                                                  const double              *B,
                                                  aoclsparse_int             n,
                                                  aoclsparse_int             ldb,
                                                  double                    *X,
                                                  aoclsparse_int             ldx,
                                                  const aoclsparse_int       kid)
{
    return aoclsparse_trsm(trans, alpha, A, descr, order, B, n, ldb, X, ldx, kid);
}

extern "C" aoclsparse_status aoclsparse_ctrsm_kid(const aoclsparse_operation      trans,
                                                  const aoclsparse_float_complex  alpha,
                                                  aoclsparse_matrix               A,
                                                  const aoclsparse_mat_descr      descr,
                                                  aoclsparse_order                order,
                                                  const aoclsparse_float_complex *B,
                                                  aoclsparse_int                  n,
                                                  aoclsparse_int                  ldb,
                                                  aoclsparse_float_complex       *X,
                                                  aoclsparse_int                  ldx,
                                                  const aoclsparse_int            kid)
{
    const std::complex<float> *alphap = reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> *Bp     = reinterpret_cast<const std::complex<float> *>(B);
    std::complex<float>       *Xp     = reinterpret_cast<std::complex<float> *>(X);
    return aoclsparse_trsm(trans, *alphap, A, descr, order, Bp, n, ldb, Xp, ldx, kid);
}
extern "C" aoclsparse_status aoclsparse_ztrsm_kid(const aoclsparse_operation       trans,
                                                  const aoclsparse_double_complex  alpha,
                                                  aoclsparse_matrix                A,
                                                  const aoclsparse_mat_descr       descr,
                                                  aoclsparse_order                 order,
                                                  const aoclsparse_double_complex *B,
                                                  aoclsparse_int                   n,
                                                  aoclsparse_int                   ldb,
                                                  aoclsparse_double_complex       *X,
                                                  aoclsparse_int                   ldx,
                                                  const aoclsparse_int             kid)
{
    const std::complex<double> *alphap = reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> *Bp     = reinterpret_cast<const std::complex<double> *>(B);
    std::complex<double>       *Xp     = reinterpret_cast<std::complex<double> *>(X);
    return aoclsparse_trsm(trans, *alphap, A, descr, order, Bp, n, ldb, Xp, ldx, kid);
}
