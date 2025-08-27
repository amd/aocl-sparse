/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_OPTMV_HELPERS_HPP
#define AOCLSPARSE_OPTMV_HELPERS_HPP

#include "aoclsparse.h"

// Kernel to scale vectors
// To-do: Move to L1 section like dot operation
template <typename T>
aoclsparse_status vscale(T *v, T c, aoclsparse_int sz)
{
    T zero = 0;

    if(!v)
        return aoclsparse_status_invalid_pointer;

    if(c != zero)
    {
        for(aoclsparse_int i = 0; i < sz; i++)
            v[i] = c * v[i];
    }
    else
    {
        for(aoclsparse_int i = 0; i < sz; i++)
            v[i] = zero;
    }

    return aoclsparse_status_success;
}

template <typename T>
bool is_mtx_frmt_supported_mv(aoclsparse_matrix_format_type mtx_t)
{
    if constexpr(aoclsparse::is_dt_complex<T>())
    {
        // Only CSR and BSR are supported for complex types
        if(mtx_t != aoclsparse_csr_mat && mtx_t != aoclsparse_bsr_mat)
            return false;
    }
    else
    {
        // Only CSR, TCSR and BSR input format supported
        if(mtx_t != aoclsparse_csr_mat && mtx_t != aoclsparse_tcsr_mat
           && mtx_t != aoclsparse_bsr_mat)
            return false;
    }

    return true;
}
#endif