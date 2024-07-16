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
#include "aoclsparse_axpyi.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Performs vector-vector addition operation of the form y = ax + y, where
 * a: scalar value
 * x: compressed sparse vector
 * y: dense vector
 */
extern "C" aoclsparse_status aoclsparse_saxpyi(
    const aoclsparse_int nnz, const float a, const float *x, const aoclsparse_int *indx, float *y)
{
    aoclsparse_int kid = -1;
    return aoclsparse_axpyi_t(nnz, a, x, indx, y, kid);
}

extern "C" aoclsparse_status aoclsparse_daxpyi(const aoclsparse_int  nnz,
                                               const double          a,
                                               const double         *x,
                                               const aoclsparse_int *indx,
                                               double               *y)
{
    aoclsparse_int kid = -1;
    return aoclsparse_axpyi_t(nnz, a, x, indx, y, kid);
}

extern "C" aoclsparse_status aoclsparse_caxpyi(
    const aoclsparse_int nnz, const void *a, const void *x, const aoclsparse_int *indx, void *y)
{
    aoclsparse_int kid = -1;
    if(a == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_axpyi_t(nnz,
                              *((std::complex<float> *)a),
                              (std::complex<float> *)x,
                              indx,
                              (std::complex<float> *)y,
                              kid);
}

extern "C" aoclsparse_status aoclsparse_zaxpyi(
    const aoclsparse_int nnz, const void *a, const void *x, const aoclsparse_int *indx, void *y)
{
    aoclsparse_int kid = -1;
    if(a == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_axpyi_t(nnz,
                              *((std::complex<double> *)a),
                              (std::complex<double> *)x,
                              indx,
                              (std::complex<double> *)y,
                              kid);
}

extern "C" aoclsparse_status aoclsparse_saxpyi_kid(const aoclsparse_int  nnz,
                                                   const float           a,
                                                   const float          *x,
                                                   const aoclsparse_int *indx,
                                                   float                *y,
                                                   aoclsparse_int        kid)
{
    return aoclsparse_axpyi_t(nnz, a, x, indx, y, kid);
}

extern "C" aoclsparse_status aoclsparse_daxpyi_kid(const aoclsparse_int  nnz,
                                                   const double          a,
                                                   const double         *x,
                                                   const aoclsparse_int *indx,
                                                   double               *y,
                                                   aoclsparse_int        kid)
{
    return aoclsparse_axpyi_t(nnz, a, x, indx, y, kid);
}

extern "C" aoclsparse_status aoclsparse_caxpyi_kid(const aoclsparse_int  nnz,
                                                   const void           *a,
                                                   const void           *x,
                                                   const aoclsparse_int *indx,
                                                   void                 *y,
                                                   aoclsparse_int        kid)
{
    if(a == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_axpyi_t(nnz,
                              *((std::complex<float> *)a),
                              (std::complex<float> *)x,
                              indx,
                              (std::complex<float> *)y,
                              kid);
}

extern "C" aoclsparse_status aoclsparse_zaxpyi_kid(const aoclsparse_int  nnz,
                                                   const void           *a,
                                                   const void           *x,
                                                   const aoclsparse_int *indx,
                                                   void                 *y,
                                                   aoclsparse_int        kid)
{
    if(a == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_axpyi_t(nnz,
                              *((std::complex<double> *)a),
                              (std::complex<double> *)x,
                              indx,
                              (std::complex<double> *)y,
                              kid);
}