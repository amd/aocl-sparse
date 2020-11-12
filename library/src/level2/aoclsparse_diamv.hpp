/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sdia
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
#ifndef AOCLSPARSE_DIAMV_HPP
#define AOCLSPARSE_DIAMV_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include <algorithm>

template <typename T>
aoclsparse_status aoclsparse_diamv(const T                alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const T*              dia_val,
                                   const aoclsparse_int*      dia_offset,
                                   aoclsparse_int      dia_num_diag,
                                   const T*             x,
                                   const T              beta,
                                   T*                   y )
{
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        y[i] = beta * y[i];
    }
    for(aoclsparse_int i = 0; i < dia_num_diag; ++i)
    {
        aoclsparse_int offset   = dia_offset[i];
        aoclsparse_int istart   = std::max((aoclsparse_int)0 , -offset);
        aoclsparse_int jstart   = std::max((aoclsparse_int)0 , offset);
        aoclsparse_int num_values = std ::min(m-istart , n-jstart);

        for(aoclsparse_int j = 0; j < num_values; ++j)
        {
            y[istart + j] += alpha * dia_val[istart + i * m + j]
                               * x[j + jstart];
        }
    }

    return aoclsparse_status_success;
}

#endif // AOCLSPARSE_DIAMV_HPP

