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
#include "aoclsparse_roti.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Performs Givens rotation of two single precision vectors
 * x (compressed form) and y (full storage form)
 */
extern "C" aoclsparse_status aoclsparse_sroti(const aoclsparse_int  nnz,
                                              float                *x,
                                              const aoclsparse_int *indx,
                                              float                *y,
                                              const float           c,
                                              const float           s)
{
    const aoclsparse_int kid = -1;
    return aoclsparse_rot<float>(nnz, x, indx, y, c, s, kid);
}

/*
 * Performs Givens rotation of two single precision vectors
 * x (compressed form) and y (full storage form)
 */
extern "C" aoclsparse_status aoclsparse_droti(const aoclsparse_int  nnz,
                                              double               *x,
                                              const aoclsparse_int *indx,
                                              double               *y,
                                              const double          c,
                                              const double          s)
{
    const aoclsparse_int kid = -1;
    return aoclsparse_rot<double>(nnz, x, indx, y, c, s, kid);
}

/*
 *===========================================================================
 *   C wrapper with KID
 * ===========================================================================
 */

/*
 * Performs Givens rotation of two single precision vectors
 * x (compressed form) and y (full storage form)
 */
extern "C" aoclsparse_status aoclsparse_sroti_kid(const aoclsparse_int  nnz,
                                                  float                *x,
                                                  const aoclsparse_int *indx,
                                                  float                *y,
                                                  const float           c,
                                                  const float           s,
                                                  aoclsparse_int        kid)
{
    return aoclsparse_rot<float>(nnz, x, indx, y, c, s, kid);
}

/*
 * Performs Givens rotation of two single precision vectors
 * x (compressed form) and y (full storage form)
 */
extern "C" aoclsparse_status aoclsparse_droti_kid(const aoclsparse_int  nnz,
                                                  double               *x,
                                                  const aoclsparse_int *indx,
                                                  double               *y,
                                                  const double          c,
                                                  const double          s,
                                                  aoclsparse_int        kid)
{
    return aoclsparse_rot<double>(nnz, x, indx, y, c, s, kid);
}
