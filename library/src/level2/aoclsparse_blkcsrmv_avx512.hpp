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
#ifndef AOCLSPARSE_BLKCSRMV_K_HPP
#define AOCLSPARSE_BLKCSRMV_K_HPP
#include "aoclsparse.h"

#include <immintrin.h>

#ifdef _MSC_VER_
#include <intrin.h>
#endif

/*
*    NOTE: These declarations are enabled only when compiled with AVX512F
*    and AVX512VL flags. Without those flags, the definition cannot be enabled.
*/
// Function definitions.
// -----------------------
#if defined __AVX512F__ && defined __AVX512VL__
aoclsparse_status
    aoclsparse_blkcsrmv_1x8_vectorized_avx512(aoclsparse_index_base base,
                                              const double          alpha,
                                              aoclsparse_int        m,
                                              const uint8_t *__restrict__ masks,
                                              const double *__restrict__ blk_csr_val,
                                              const aoclsparse_int *__restrict__ blk_col_ind,
                                              const aoclsparse_int *__restrict__ blk_row_ptr,
                                              const double *__restrict__ x,
                                              const double beta,
                                              double *__restrict__ y);

aoclsparse_status
    aoclsparse_blkcsrmv_2x8_vectorized_avx512(aoclsparse_index_base base,
                                              const double          alpha,
                                              aoclsparse_int        m,
                                              const uint8_t *__restrict__ masks,
                                              const double *__restrict__ blk_csr_val,
                                              const aoclsparse_int *__restrict__ blk_col_ind,
                                              const aoclsparse_int *__restrict__ blk_row_ptr,
                                              const double *__restrict__ x,
                                              const double beta,
                                              double *__restrict__ y);

aoclsparse_status
    aoclsparse_blkcsrmv_4x8_vectorized_avx512(aoclsparse_index_base base,
                                              const double          alpha,
                                              aoclsparse_int        m,
                                              const uint8_t *__restrict__ masks,
                                              const double *__restrict__ blk_csr_val,
                                              const aoclsparse_int *__restrict__ blk_col_ind,
                                              const aoclsparse_int *__restrict__ blk_row_ptr,
                                              const double *__restrict__ x,
                                              const double beta,
                                              double *__restrict__ y);
#endif // For AVX512F

#endif