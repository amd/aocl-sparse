/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#ifndef AOCLSPARSE_AUXILIARY_HPP
#define AOCLSPARSE_AUXILIARY_HPP

#include "aoclsparse_mat_structures.h"

#include <cmath>
#include <limits>

void aoclsparse_init_csrmat(aoclsparse_matrix A);

/* 
    subtraction of 2 vectors
*/
template <typename T>
aoclsparse_status aoclsparse_wxmy(const aoclsparse_int n, const T *xv, const T *yv, T *wv)
{
    aoclsparse_status exit_status = aoclsparse_status_success;
    if(xv == NULL || yv == NULL || wv == NULL)
    {
        return aoclsparse_status_invalid_pointer;
    }
    for(aoclsparse_int i = 0; i < n; i++)
    {
        wv[i] = xv[i] - yv[i];
    }

    return exit_status;
}

/* 
    Computes dot product of 2 vectors
*/
template <typename T>
aoclsparse_status aoclsparse_ddot(const aoclsparse_int n, const T *xv, const T *yv, T &result)
{
    aoclsparse_status exit_status  = aoclsparse_status_success;
    T                 local_result = 0.0;
    if(xv == NULL || yv == NULL)
    {
        return aoclsparse_status_invalid_pointer;
    }
    for(aoclsparse_int i = 0; i < n; i++)
    {
        local_result += xv[i] * yv[i];
    }
    result = local_result;
    return exit_status;
}

/* 
    Computes norm-2 of 2 vectors
*/
template <typename T>
aoclsparse_status aoclsparse_dnorm2(const aoclsparse_int n, const T *xv, T &result)
{
    aoclsparse_status exit_status = aoclsparse_status_success;
    if(xv == NULL)
    {
        return aoclsparse_status_invalid_pointer;
    }
    T sum = 0.0;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        sum += xv[i] * xv[i];
    }
    result = sqrt(sum);
    return exit_status;
}

/* 
    scales a vector by a specific value provided
*/
template <typename T>
aoclsparse_status aoclsparse_scale(const aoclsparse_int n, T *xv, T sfactor)
{
    aoclsparse_status exit_status = aoclsparse_status_success;
    if(xv == NULL)
    {
        return aoclsparse_status_invalid_pointer;
    }
    for(aoclsparse_int i = 0; i < n; i++)
    {
        xv[i] = xv[i] * sfactor;
    }
    return exit_status;
}
/* 
    Perform a comparison test to determine if the value is near zero
*/
template <typename T>
bool aoclsparse_zerocheck(const T &value)
{
    bool is_value_zero = false;
    is_value_zero      = std::fabs(value) <= std::numeric_limits<T>::epsilon();
    return is_value_zero;
}
#endif