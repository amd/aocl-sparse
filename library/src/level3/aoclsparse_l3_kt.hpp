/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_L3_KT_HPP
#define AOCLSPARSE_L3_KT_HPP
#include "aoclsparse.h"
#include "aoclsparse_kernel_templates.hpp"

namespace aoclsparse
{
    template <kernel_templates::bsz SZ, typename SUF>
    aoclsparse_status csrmm_col_kt(const SUF                  alpha,
                                   const aoclsparse_mat_descr descr,
                                   const SUF *__restrict__ csr_val,
                                   const aoclsparse_int *__restrict__ csr_col_ind,
                                   const aoclsparse_int *__restrict__ csr_row_ptr,
                                   aoclsparse_int m,
                                   const SUF     *B,
                                   aoclsparse_int n,
                                   aoclsparse_int ldb,
                                   SUF            beta,
                                   SUF           *C,
                                   aoclsparse_int ldc);

    template <kernel_templates::bsz SZ, typename SUF>
    aoclsparse_status csrmm_row_kt(const SUF                  alpha,
                                   const aoclsparse_mat_descr descr,
                                   const SUF *__restrict__ csr_val,
                                   const aoclsparse_int *__restrict__ csr_col_ind,
                                   const aoclsparse_int *__restrict__ csr_row_ptr,
                                   aoclsparse_int m,
                                   const SUF     *B,
                                   aoclsparse_int n,
                                   aoclsparse_int ldb,
                                   SUF            beta,
                                   SUF           *C,
                                   aoclsparse_int ldc);
}

#endif
