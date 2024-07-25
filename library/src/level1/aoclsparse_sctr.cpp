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
#include "aoclsparse_sctr.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Performs indexed scatter operation of a complex (single precision) compressed sparse vector (x)
 * to a complex full storage vector (y).
 */
extern "C" aoclsparse_status
    aoclsparse_csctr(const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, void *y)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_scatter<std::complex<float>, Index::type::indexed>(
        nnz, (std::complex<float> *)x, indx, (std::complex<float> *)y, kid);
    return status;
}

/*
 * Performs indexed scatter operation of a complex (double precision) compressed sparse vector (x)
 * to a complex full storage vector (y).
 */
extern "C" aoclsparse_status
    aoclsparse_zsctr(const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, void *y)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_scatter<std::complex<double>, Index::type::indexed>(
        nnz, (std::complex<double> *)x, indx, (std::complex<double> *)y, kid);
    return status;
}

/*
 * Performs indexed scatter operation of a real (single precision) compressed sparse vector (x)
 * to a real full storage vector (y).
 */
extern "C" aoclsparse_status
    aoclsparse_ssctr(const aoclsparse_int nnz, const float *x, const aoclsparse_int *indx, float *y)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_scatter<float, Index::type::indexed>(nnz, x, indx, y, kid);
    return status;
}

/*
 * Performs indexed scatter operation of a real (double precision) compressed sparse vector (x)
 * to a real full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_dsctr(const aoclsparse_int  nnz,
                                              const double         *x,
                                              const aoclsparse_int *indx,
                                              double               *y)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_scatter<double, Index::type::indexed>(nnz, x, indx, y, kid);
    return status;
}

/*
 * Performs strided scatter operation of a complex (single precision) compressed sparse vector (x)
 * to a complex full storage vector (y).
 */
extern "C" aoclsparse_status
    aoclsparse_csctrs(const aoclsparse_int nnz, const void *x, aoclsparse_int stride, void *y)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_scatter<std::complex<float>, Index::type::strided>(
        nnz, (std::complex<float> *)x, stride, (std::complex<float> *)y, kid);
    return status;
}

/*
 * Performs strided scatter operation of a complex (double precision) compressed sparse vector (x)
 * to a complex full storage vector (y).
 */
extern "C" aoclsparse_status
    aoclsparse_zsctrs(const aoclsparse_int nnz, const void *x, aoclsparse_int stride, void *y)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_scatter<std::complex<double>, Index::type::strided>(
        nnz, (std::complex<double> *)x, stride, (std::complex<double> *)y, kid);
    return status;
}

/*
 * Performs strided scatter operation of a real (single precision) compressed sparse vector (x)
 * to a real full storage vector (y).
 */
extern "C" aoclsparse_status
    aoclsparse_ssctrs(const aoclsparse_int nnz, const float *x, aoclsparse_int stride, float *y)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_scatter<float, Index::type::strided>(nnz, x, stride, y, kid);
    return status;
}

/*
 * Performs strided scatter operation of a real (double precision) compressed sparse vector (x)
 * to a real full storage vector (y).
 */
extern "C" aoclsparse_status
    aoclsparse_dsctrs(const aoclsparse_int nnz, const double *x, aoclsparse_int stride, double *y)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_scatter<double, Index::type::strided>(nnz, x, stride, y, kid);
    return status;
}

/*
 *===========================================================================
 *   C wrapper with KID
 * ===========================================================================
 */

/*
 * Performs indexed scatter operation of a complex (single precision) compressed sparse vector (x)
 * to a complex full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_csctr_kid(const aoclsparse_int  nnz,
                                                  const void           *x,
                                                  const aoclsparse_int *indx,
                                                  void                 *y,
                                                  aoclsparse_int        kid)
{
    aoclsparse_status status;
    status = aoclsparse_scatter<std::complex<float>, Index::type::indexed>(
        nnz, (std::complex<float> *)x, indx, (std::complex<float> *)y, kid);
    return status;
}

/*
 * Performs indexed scatter operation of a complex (double precision) compressed sparse vector (x)
 * to a complex full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_zsctr_kid(const aoclsparse_int  nnz,
                                                  const void           *x,
                                                  const aoclsparse_int *indx,
                                                  void                 *y,
                                                  aoclsparse_int        kid)
{
    aoclsparse_status status;
    status = aoclsparse_scatter<std::complex<double>, Index::type::indexed>(
        nnz, (std::complex<double> *)x, indx, (std::complex<double> *)y, kid);
    return status;
}

/*
 * Performs indexed scatter operation of a real (single precision) compressed sparse vector (x)
 * to a real full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_ssctr_kid(const aoclsparse_int  nnz,
                                                  const float          *x,
                                                  const aoclsparse_int *indx,
                                                  float                *y,
                                                  aoclsparse_int        kid)
{
    aoclsparse_status status;
    status = aoclsparse_scatter<float, Index::type::indexed>(nnz, x, indx, y, kid);
    return status;
}

/*
 * Performs indexed scatter operation of a real (double precision) compressed sparse vector (x)
 * to a real full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_dsctr_kid(const aoclsparse_int  nnz,
                                                  const double         *x,
                                                  const aoclsparse_int *indx,
                                                  double               *y,
                                                  aoclsparse_int        kid)
{
    aoclsparse_status status;
    status = aoclsparse_scatter<double, Index::type::indexed>(nnz, x, indx, y, kid);
    return status;
}

/*
 * Performs strided scatter operation of a complex (single precision) compressed sparse vector (x)
 * to a complex full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_csctrs_kid(
    const aoclsparse_int nnz, const void *x, aoclsparse_int stride, void *y, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_scatter<std::complex<float>, Index::type::strided>(
        nnz, (std::complex<float> *)x, stride, (std::complex<float> *)y, kid);
    return status;
}

/*
 * Performs strided scatter operation of a complex (double precision) compressed sparse vector (x)
 * to a complex full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_zsctrs_kid(
    const aoclsparse_int nnz, const void *x, aoclsparse_int stride, void *y, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_scatter<std::complex<double>, Index::type::strided>(
        nnz, (std::complex<double> *)x, stride, (std::complex<double> *)y, kid);
    return status;
}

/*
 * Performs strided scatter operation of a real (single precision) compressed sparse vector (x)
 * to a real full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_ssctrs_kid(
    const aoclsparse_int nnz, const float *x, aoclsparse_int stride, float *y, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_scatter<float, Index::type::strided>(nnz, x, stride, y, kid);
    return status;
}

/*
 * Performs strided scatter operation of a real (double precision) compressed sparse vector (x)
 * to a real full storage vector (y).
 */
extern "C" aoclsparse_status aoclsparse_dsctrs_kid(
    const aoclsparse_int nnz, const double *x, aoclsparse_int stride, double *y, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_scatter<double, Index::type::strided>(nnz, x, stride, y, kid);
    return status;
}
