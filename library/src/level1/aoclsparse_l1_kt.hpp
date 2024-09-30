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
#ifndef AOCLSPARSE_L1_KT_HPP
#define AOCLSPARSE_L1_KT_HPP
#include "aoclsparse_kernel_templates.hpp"

// Extern declaration of KT kernels
template <kernel_templates::bsz SZ, typename T>
aoclsparse_status axpyi_kt(aoclsparse_int nnz,
                           T              a,
                           const T *__restrict__ x,
                           const aoclsparse_int *__restrict__ indx,
                           T *__restrict__ y);

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status dotp_kt(aoclsparse_int nnz,
                          const SUF *__restrict__ x,
                          const aoclsparse_int *__restrict__ indx,
                          const SUF *__restrict__ y,
                          SUF *__restrict__ dot,
                          bool conj);

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status roti_kt(aoclsparse_int nnz,
                          SUF *__restrict__ x,
                          const aoclsparse_int *__restrict__ indx,
                          SUF *__restrict__ y,
                          SUF c,
                          SUF s);

#endif // of AOCLSPARSE_L1_KT_HPP
