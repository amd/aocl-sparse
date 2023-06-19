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
#include "aoclsparse_gthr.hpp"

/*
 *===========================================================================
 * C wrapper
 * ===========================================================================
 */

extern "C" aoclsparse_status
    aoclsparse_sgthr(const aoclsparse_int nnz, const float *y, float *x, const aoclsparse_int *indx)
{
    aoclsparse_int kid = 0;
    return aoclsparse_gthr<float, false>(nnz, y, x, indx, kid);
}

extern "C" aoclsparse_status aoclsparse_dgthr(const aoclsparse_int  nnz,
                                              const double         *y,
                                              double               *x,
                                              const aoclsparse_int *indx)
{
    aoclsparse_int kid = 0;
    return aoclsparse_gthr<double, false>(nnz, y, x, indx, kid);
}

extern "C" aoclsparse_status
    aoclsparse_cgthr(const aoclsparse_int nnz, const void *y, void *x, const aoclsparse_int *indx)
{
    aoclsparse_int kid = 0;
    return aoclsparse_gthr<std::complex<float>, false>(
        nnz, (const std::complex<float> *)y, (std::complex<float> *)x, indx, kid);
}

extern "C" aoclsparse_status
    aoclsparse_zgthr(const aoclsparse_int nnz, const void *y, void *x, const aoclsparse_int *indx)
{
    aoclsparse_int kid = 0;
    return aoclsparse_gthr<std::complex<double>, false>(
        nnz, (const std::complex<double> *)y, (std::complex<double> *)x, indx, kid);
}

extern "C" aoclsparse_status
    aoclsparse_sgthrz(const aoclsparse_int nnz, float *y, float *x, const aoclsparse_int *indx)
{
    aoclsparse_int kid = 0;
    return aoclsparse_gthr<float, true>(nnz, y, x, indx, kid);
}

extern "C" aoclsparse_status
    aoclsparse_dgthrz(const aoclsparse_int nnz, double *y, double *x, const aoclsparse_int *indx)
{
    aoclsparse_int kid = 0;
    return aoclsparse_gthr<double, true>(nnz, y, x, indx, kid);
}

extern "C" aoclsparse_status
    aoclsparse_cgthrz(const aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx)
{
    aoclsparse_int kid = 0;
    return aoclsparse_gthr<std::complex<float>, true>(
        nnz, (std::complex<float> *)y, (std::complex<float> *)x, indx, kid);
}

extern "C" aoclsparse_status
    aoclsparse_zgthrz(const aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx)
{
    aoclsparse_int kid = 0;
    return aoclsparse_gthr<std::complex<double>, true>(
        nnz, (std::complex<double> *)y, (std::complex<double> *)x, indx, kid);
}

#undef C_IMPL
