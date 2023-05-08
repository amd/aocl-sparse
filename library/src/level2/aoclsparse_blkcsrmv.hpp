/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_BLKCSRMV_HPP
#define AOCLSPARSE_BLKCSRMV_HPP
#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

template <typename T>
aoclsparse_status
    aoclsparse_blkcsrmv_1x8_vectorized_avx512(const T        alpha,
                                            aoclsparse_int m,
                                            const uint8_t *__restrict__ masks,
                                            const T *__restrict__ blk_csr_val,
                                            const aoclsparse_int *__restrict__ blk_col_ind,
                                            const aoclsparse_int *__restrict__ blk_row_ptr,
                                            const T *__restrict__ x,
                                            const T beta,
                                            T *__restrict__ y);

template <typename T>
aoclsparse_status
    aoclsparse_blkcsrmv_2x8_vectorized_avx512(const T        alpha,
                                            aoclsparse_int m,
                                            const uint8_t *__restrict__ masks,
                                            const T *__restrict__ blk_csr_val,
                                            const aoclsparse_int *__restrict__ blk_col_ind,
                                            const aoclsparse_int *__restrict__ blk_row_ptr,
                                            const T *__restrict__ x,
                                            const T beta,
                                            T *__restrict__ y);

template <typename T>
aoclsparse_status
    aoclsparse_blkcsrmv_4x8_vectorized_avx512(const T        alpha,
                                            aoclsparse_int m,
                                            const uint8_t *__restrict__ masks,
                                            const T *__restrict__ blk_csr_val,
                                            const aoclsparse_int *__restrict__ blk_col_ind,
                                            const aoclsparse_int *__restrict__ blk_row_ptr,
                                            const T *__restrict__ x,
                                            const T beta,
                                            T *__restrict__ y);

#endif // AOCLSPARSE_BLKCSRMV_HPP
