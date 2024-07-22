/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse.h"
#include "aoclsparse_blkcsrmv.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 * mini dispatcher and a C wrapper, to direct block csrmv calls on AVX512 and
 * non-AVX512 machines
 *
 */
extern "C" aoclsparse_status aoclsparse_dblkcsrmv(aoclsparse_operation       trans,
                                                  const double              *alpha,
                                                  aoclsparse_int             m,
                                                  aoclsparse_int             n,
                                                  aoclsparse_int             nnz,
                                                  const uint8_t             *masks,
                                                  const double              *blk_csr_val,
                                                  const aoclsparse_int      *blk_col_ind,
                                                  const aoclsparse_int      *blk_row_ptr,
                                                  const aoclsparse_mat_descr descr,
                                                  const double              *x,
                                                  const double              *beta,
                                                  double                    *y,
                                                  aoclsparse_int             nRowsblk)
{
    return aoclsparse_blkcsrmv_t<double>(trans,
                                         alpha,
                                         m,
                                         n,
                                         nnz,
                                         masks,
                                         blk_csr_val,
                                         blk_col_ind,
                                         blk_row_ptr,
                                         descr,
                                         x,
                                         beta,
                                         y,
                                         nRowsblk);
}
