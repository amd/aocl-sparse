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
 * ************************************************************************ */

#include "aoclsparse.h"
#include "aoclsparse_gthr.hpp"

/*
 *===========================================================================
 * C wrapper
 * ===========================================================================
 */

// gather with index
extern "C" aoclsparse_status
    aoclsparse_sgthr(const aoclsparse_int nnz, const float *y, float *x, const aoclsparse_int *indx)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<float, gather_op::gather, Index::type::indexed>(nnz, y, x, indx, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_dgthr(const aoclsparse_int  nnz,
                                              const double         *y,
                                              double               *x,
                                              const aoclsparse_int *indx)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<double, gather_op::gather, Index::type::indexed>(nnz, y, x, indx, kid);
    return status;
}

extern "C" aoclsparse_status
    aoclsparse_cgthr(const aoclsparse_int nnz, const void *y, void *x, const aoclsparse_int *indx)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<float>, gather_op::gather, Index::type::indexed>(
        nnz, (const std::complex<float> *)y, (std::complex<float> *)x, indx, kid);
    return status;
}

extern "C" aoclsparse_status
    aoclsparse_zgthr(const aoclsparse_int nnz, const void *y, void *x, const aoclsparse_int *indx)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<double>, gather_op::gather, Index::type::indexed>(
        nnz, (const std::complex<double> *)y, (std::complex<double> *)x, indx, kid);
    return status;
}

// gather_zero with index
extern "C" aoclsparse_status
    aoclsparse_sgthrz(const aoclsparse_int nnz, float *y, float *x, const aoclsparse_int *indx)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<float, gather_op::gatherz, Index::type::indexed>(nnz, y, x, indx, kid);
    return status;
}

extern "C" aoclsparse_status
    aoclsparse_dgthrz(const aoclsparse_int nnz, double *y, double *x, const aoclsparse_int *indx)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<double, gather_op::gatherz, Index::type::indexed>(nnz, y, x, indx, kid);
    return status;
}

extern "C" aoclsparse_status
    aoclsparse_cgthrz(const aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<float>, gather_op::gatherz, Index::type::indexed>(
        nnz, (std::complex<float> *)y, (std::complex<float> *)x, indx, kid);
    return status;
}

extern "C" aoclsparse_status
    aoclsparse_zgthrz(const aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<double>, gather_op::gatherz, Index::type::indexed>(
        nnz, (std::complex<double> *)y, (std::complex<double> *)x, indx, kid);
    return status;
}

// gather with stride
extern "C" aoclsparse_status
    aoclsparse_sgthrs(const aoclsparse_int nnz, const float *y, float *x, aoclsparse_int stride)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<float, gather_op::gather, Index::type::strided>(nnz, y, x, stride, kid);
    return status;
}

extern "C" aoclsparse_status
    aoclsparse_dgthrs(const aoclsparse_int nnz, const double *y, double *x, aoclsparse_int stride)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_gthr_t<double, gather_op::gather, Index::type::strided>(
        nnz, y, x, stride, kid);
    return status;
}

extern "C" aoclsparse_status
    aoclsparse_cgthrs(const aoclsparse_int nnz, const void *y, void *x, aoclsparse_int stride)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<float>, gather_op::gather, Index::type::strided>(
        nnz, (const std::complex<float> *)y, (std::complex<float> *)x, stride, kid);
    return status;
}

extern "C" aoclsparse_status
    aoclsparse_zgthrs(const aoclsparse_int nnz, const void *y, void *x, aoclsparse_int stride)
{
    aoclsparse_int    kid = -1;
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<double>, gather_op::gather, Index::type::strided>(
        nnz, (const std::complex<double> *)y, (std::complex<double> *)x, stride, kid);
    return status;
}

/*
 *===========================================================================
 * C wrapper with kid
 * ===========================================================================
 */

// gather with index
extern "C" aoclsparse_status aoclsparse_sgthr_kid(const aoclsparse_int  nnz,
                                                  const float          *y,
                                                  float                *x,
                                                  const aoclsparse_int *indx,
                                                  aoclsparse_int        kid)
{
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<float, gather_op::gather, Index::type::indexed>(nnz, y, x, indx, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_dgthr_kid(const aoclsparse_int  nnz,
                                                  const double         *y,
                                                  double               *x,
                                                  const aoclsparse_int *indx,
                                                  aoclsparse_int        kid)
{
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<double, gather_op::gather, Index::type::indexed>(nnz, y, x, indx, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_cgthr_kid(const aoclsparse_int  nnz,
                                                  const void           *y,
                                                  void                 *x,
                                                  const aoclsparse_int *indx,
                                                  aoclsparse_int        kid)
{
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<float>, gather_op::gather, Index::type::indexed>(
        nnz, (const std::complex<float> *)y, (std::complex<float> *)x, indx, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_zgthr_kid(const aoclsparse_int  nnz,
                                                  const void           *y,
                                                  void                 *x,
                                                  const aoclsparse_int *indx,
                                                  aoclsparse_int        kid)
{
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<double>, gather_op::gather, Index::type::indexed>(
        nnz, (const std::complex<double> *)y, (std::complex<double> *)x, indx, kid);
    return status;
}

// gather_zero with index
extern "C" aoclsparse_status aoclsparse_sgthrz_kid(
    const aoclsparse_int nnz, float *y, float *x, const aoclsparse_int *indx, aoclsparse_int kid)
{
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<float, gather_op::gatherz, Index::type::indexed>(nnz, y, x, indx, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_dgthrz_kid(
    const aoclsparse_int nnz, double *y, double *x, const aoclsparse_int *indx, aoclsparse_int kid)
{
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<double, gather_op::gatherz, Index::type::indexed>(nnz, y, x, indx, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_cgthrz_kid(
    const aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<float>, gather_op::gatherz, Index::type::indexed>(
        nnz, (std::complex<float> *)y, (std::complex<float> *)x, indx, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_zgthrz_kid(
    const aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<double>, gather_op::gatherz, Index::type::indexed>(
        nnz, (std::complex<double> *)y, (std::complex<double> *)x, indx, kid);
    return status;
}

// gather with stride
extern "C" aoclsparse_status aoclsparse_sgthrs_kid(
    const aoclsparse_int nnz, const float *y, float *x, aoclsparse_int stride, aoclsparse_int kid)
{
    aoclsparse_status status;
    status
        = aoclsparse_gthr_t<float, gather_op::gather, Index::type::strided>(nnz, y, x, stride, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_dgthrs_kid(
    const aoclsparse_int nnz, const double *y, double *x, aoclsparse_int stride, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_gthr_t<double, gather_op::gather, Index::type::strided>(
        nnz, y, x, stride, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_cgthrs_kid(
    const aoclsparse_int nnz, const void *y, void *x, aoclsparse_int stride, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<float>, gather_op::gather, Index::type::strided>(
        nnz, (const std::complex<float> *)y, (std::complex<float> *)x, stride, kid);
    return status;
}

extern "C" aoclsparse_status aoclsparse_zgthrs_kid(
    const aoclsparse_int nnz, const void *y, void *x, aoclsparse_int stride, aoclsparse_int kid)
{
    aoclsparse_status status;
    status = aoclsparse_gthr_t<std::complex<double>, gather_op::gather, Index::type::strided>(
        nnz, (const std::complex<double> *)y, (std::complex<double> *)x, stride, kid);
    return status;
}
