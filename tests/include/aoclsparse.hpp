/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

/*! \file
 *  \brief aoclsparse.hpp exposes C++ templated Sparse Linear Algebra interface
 *  with only the precision templated.
 */

#pragma once
#ifndef AOCLSPARSE_HPP
#define AOCLSPARSE_HPP

#include <aoclsparse.h>

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                 const T*             alpha,
                                 aoclsparse_int             m,
                                 aoclsparse_int             n,
                                 aoclsparse_int             nnz,
                                 const T*             csr_val,
                                 const aoclsparse_int*      csr_col_ind,
                                 const aoclsparse_int*      csr_row_ptr,
                                 const aoclsparse_mat_descr descr,
                                 const T*             x,
                                 const T*             beta,
                                 T*                   y);

template <typename T>
aoclsparse_status aoclsparse_ellmv(aoclsparse_operation       trans,
                                 const T*             alpha,
                                 aoclsparse_int             m,
                                 aoclsparse_int             n,
                                 aoclsparse_int             nnz,
                                 const T*             ell_val,
                                 const aoclsparse_int*      ell_col_ind,
                                 const aoclsparse_int      ell_width,
                                 const aoclsparse_mat_descr descr,
                                 const T*             x,
                                 const T*             beta,
                                 T*                   y);


#endif // AOCLSPARSE_HPP
