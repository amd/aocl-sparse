/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.All rights reserved.
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
aoclsparse_status diamv_ref(const T               alpha,
                            aoclsparse_int        m,
                            aoclsparse_int        n,
                            const T              *dia_val,
                            const aoclsparse_int *dia_offset,
                            aoclsparse_int        dia_num_diag,
                            const T              *x,
                            const T               beta,
                            T                    *y)
{
    // Perform (beta * y)
    if(beta == static_cast<T>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    for(aoclsparse_int i = 0; i < dia_num_diag; ++i)
    {
        aoclsparse_int offset     = dia_offset[i];
        aoclsparse_int istart     = (std::max)((aoclsparse_int)0, -offset);
        aoclsparse_int jstart     = (std::max)((aoclsparse_int)0, offset);
        aoclsparse_int num_values = (std::min)(m - istart, n - jstart);

        for(aoclsparse_int j = 0; j < num_values; ++j)
        {
            y[istart + j] += alpha * dia_val[istart + i * m + j] * x[j + jstart];
        }
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_diamv_t(aoclsparse_operation            trans,
                                     const T                        *alpha,
                                     aoclsparse_int                  m,
                                     aoclsparse_int                  n,
                                     [[maybe_unused]] aoclsparse_int nnz,
                                     const T                        *dia_val,
                                     const aoclsparse_int           *dia_offset,
                                     aoclsparse_int                  dia_num_diag,
                                     const aoclsparse_mat_descr      descr,
                                     const T                        *x,
                                     const T                        *beta,
                                     T                              *y)
{
    if((alpha == nullptr) || (beta == nullptr) || (dia_val == nullptr) || (dia_offset == nullptr)
       || (x == nullptr) || (y == nullptr) || (descr == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    if(descr->type != aoclsparse_matrix_type_general)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    if(trans != aoclsparse_operation_none)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || dia_num_diag < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || dia_num_diag == 0)
    {
        return aoclsparse_status_success;
    }

    return diamv_ref(*alpha, m, n, dia_val, dia_offset, dia_num_diag, x, *beta, y);
}

#endif // AOCLSPARSE_DIAMV_HPP
