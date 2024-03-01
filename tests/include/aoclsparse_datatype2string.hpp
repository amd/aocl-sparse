/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef AOCLSPARSE_DATATYPE2STRING_HPP
#define AOCLSPARSE_DATATYPE2STRING_HPP

#include "aoclsparse.h"

typedef enum aoclsparse_datatype_
{
    aoclsparse_datatype_f32_r = 151, /**< 32 bit floating point, real */
    aoclsparse_datatype_f64_r = 152, /**< 64 bit floating point, real */
    aoclsparse_datatype_f32_c = 154, /**< 32 bit floating point, complex real */
    aoclsparse_datatype_f64_c = 155 /**< 64 bit floating point, complex real */
} aoclsparse_datatype;

typedef enum aoclsparse_matrix_init_
{
    /**< Random-sorted initialization */
    aoclsparse_matrix_random = 0,
    /**< Read from .mtx (matrix market) file */
    aoclsparse_matrix_file_mtx = 1,
    /**< Read from .csr (csr binary) file */
    aoclsparse_matrix_file_bin = 2,
    /**< random matrix with full-diagonal which is diagonally dominant */
    aoclsparse_matrix_random_diag_dom = 3
} aoclsparse_matrix_init;

constexpr auto aoclsparse_matrix2string(aoclsparse_matrix_init matrix)
{
    switch(matrix)
    {
    case aoclsparse_matrix_random:
        return "rand";
    case aoclsparse_matrix_file_mtx:
        return "mtx";
    case aoclsparse_matrix_file_bin:
        return "csr";
    case aoclsparse_matrix_random_diag_dom:
        return "rand_diagonal_dominant";
    default:
        return "invalid";
    }
}

constexpr auto aoclsparse_datatype2string(aoclsparse_datatype type)
{
    switch(type)
    {
    case aoclsparse_datatype_f32_r:
        return "f32_r";
    case aoclsparse_datatype_f64_r:
        return "f64_r";
    case aoclsparse_datatype_f32_c:
        return "f32_c";
    case aoclsparse_datatype_f64_c:
        return "f64_c";
    default:
        return "invalid";
    }
}

#endif // AOCLSPARSE_DATATYPE2STRING_HPP
