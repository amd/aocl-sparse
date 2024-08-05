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
 * ************************************************************************ */
#ifndef AOCLSPARSE_CSRMV_K_HPP
#define AOCLSPARSE_CSRMV_K_HPP
#include "aoclsparse.h"

#include <complex>

// AVX512 kernels
// =================
// Only builds with AVX512 will have this functions defined

aoclsparse_status aoclsparse_csrmv_vectorized_avx512(aoclsparse_index_base base,
                                                     const double          alpha,
                                                     aoclsparse_int        m,
                                                     const double *__restrict__ csr_val,
                                                     const aoclsparse_int *__restrict__ csr_col_ind,
                                                     const aoclsparse_int *__restrict__ csr_row_ptr,
                                                     const double *__restrict__ x,
                                                     const double beta,
                                                     double *__restrict__ y);
aoclsparse_status aoclsparse_zcsrmv_avx512(aoclsparse_index_base      base,
                                           const std::complex<double> alpha,
                                           const aoclsparse_int       m,
                                           const std::complex<double> *__restrict__ aval,
                                           const aoclsparse_int *__restrict__ icol,
                                           const aoclsparse_int *__restrict__ row,
                                           const std::complex<double> *__restrict__ x,
                                           const std::complex<double> beta,
                                           std::complex<double> *__restrict__ y);
#endif
