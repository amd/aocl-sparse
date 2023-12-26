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
#include "aoclsparse_dot.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Computes the dot product of a conjugated complex (single precision) vector stored
 * in a compressed format (x) and a complex dense vector (y).
 */
extern "C" aoclsparse_status aoclsparse_cdotci(
    const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, const void *y, void *dot)
{
    const aoclsparse_int kid    = -1; /* auto */
    const bool           conj   = true;
    aoclsparse_status    status = aoclsparse_status_success;

    status = aoclsparse_dotp<std::complex<float>>(nnz,
                                                  (std::complex<float> *)x,
                                                  indx,
                                                  (std::complex<float> *)y,
                                                  (std::complex<float> *)dot,
                                                  conj,
                                                  kid);
    return status;
}

/*
 * Computes the dot product of a conjugated complex (double precision) vector stored
 * in a compressed format (x) and a complex dense vector (y).
 */
extern "C" aoclsparse_status aoclsparse_zdotci(
    const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, const void *y, void *dot)
{
    const aoclsparse_int kid    = -1; /* auto */
    const bool           conj   = true;
    aoclsparse_status    status = aoclsparse_status_success;
    status                      = aoclsparse_dotp<std::complex<double>>(nnz,
                                                   (std::complex<double> *)x,
                                                   indx,
                                                   (std::complex<double> *)y,
                                                   (std::complex<double> *)dot,
                                                   conj,
                                                   kid);
    return status;
}

/*
 * Computes the dot product of a complex (single precision) vector stored
 * in a compressed format (x) and a complex dense vector (y).
 */
extern "C" aoclsparse_status aoclsparse_cdotui(
    const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, const void *y, void *dot)
{
    const aoclsparse_int kid    = -1; /* auto */
    const bool           conj   = false;
    aoclsparse_status    status = aoclsparse_status_success;
    status                      = aoclsparse_dotp<std::complex<float>>(nnz,
                                                  (std::complex<float> *)x,
                                                  indx,
                                                  (std::complex<float> *)y,
                                                  (std::complex<float> *)dot,
                                                  conj,
                                                  kid);
    return status;
}

/*
 * Computes the dot product of a complex (double precision) vector stored
 * in a compressed format (x) and a complex dense vector (y).
 */
extern "C" aoclsparse_status aoclsparse_zdotui(
    const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, const void *y, void *dot)
{
    const aoclsparse_int kid    = -1; /* auto */
    const bool           conj   = false;
    aoclsparse_status    status = aoclsparse_status_success;
    status                      = aoclsparse_dotp<std::complex<double>>(nnz,
                                                   (std::complex<double> *)x,
                                                   indx,
                                                   (std::complex<double> *)y,
                                                   (std::complex<double> *)dot,
                                                   conj,
                                                   kid);
    return status;
}

/*
 * Computes the dot product of a real (single precision) vector stored
 * in a compressed format (x) and a real dense vector (y).
 */
extern "C" float aoclsparse_sdoti(const aoclsparse_int  nnz,
                                  const float          *x,
                                  const aoclsparse_int *indx,
                                  const float          *y)
{
    const aoclsparse_int kid  = -1; /* auto */
    const bool           conj = false;
    float                dot;
    aoclsparse_dotp<float>(nnz, x, indx, y, &dot, conj, kid);
    return dot;
}

/*
 * Computes the dot product of a real (double precision) vector stored
 * in a compressed format (x) and a real dense vector (y).
 */
extern "C" double aoclsparse_ddoti(const aoclsparse_int  nnz,
                                   const double         *x,
                                   const aoclsparse_int *indx,
                                   const double         *y)
{
    const aoclsparse_int kid  = -1; /* auto */
    const bool           conj = false;
    double               dot;
    aoclsparse_dotp<double>(nnz, x, indx, y, &dot, conj, kid);
    return dot;
}
