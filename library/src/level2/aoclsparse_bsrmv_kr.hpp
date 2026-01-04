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
#ifndef AOCLSPARSE_BSRMV_KR_HPP
#define AOCLSPARSE_BSRMV_KR_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_bsrmv_bldr.hpp"
#include "aoclsparse_utils.hpp"

namespace aoclsparse
{
    template <typename T, bool alpha_s, bool beta_s>
    aoclsparse_status bsrmv_gn(T                               alpha,
                               aoclsparse_int                  mb,
                               [[maybe_unused]] aoclsparse_int nb,
                               aoclsparse_int                  bsr_dim,
                               const T *__restrict__ bsr_val,
                               const aoclsparse_int *__restrict__ bsr_col_ind,
                               const aoclsparse_int *__restrict__ bsr_row_ptr,
                               const aoclsparse_mat_descr descr,
                               const T *__restrict__ x,
                               T beta,
                               T *__restrict__ y)
    {
        aoclsparse_index_base base = descr->base;
        // Loop over the block rows
        for(aoclsparse_int ai = 0; ai < mb; ++ai)
        {
            // BSR row entry and exit point
            aoclsparse_int row_begin = bsr_row_ptr[ai] - base;
            aoclsparse_int row_end   = bsr_row_ptr[ai + 1] - base;
            // Loop over the individual rows within the BSR block
            for(aoclsparse_int bi = 0; bi < bsr_dim; ++bi)
            {
                // BSR block row accumulator
                T sum = static_cast<T>(0);
                // Loop over all BSR blocks in the current row
                for(aoclsparse_int aj = row_begin; aj < row_end; ++aj)
                {
                    // Column index into x vector
                    aoclsparse_int col = bsr_col_ind[aj] - base;
                    const T       *bsr_val_ptr;
                    const T       *x_ptr;
                    bsr_val_ptr = bsr_val + (bsr_dim * bsr_dim * aj) + bi;
                    x_ptr       = x + (bsr_dim * col);
                    // Loop over the columns of the BSR block
                    for(aoclsparse_int bj = 0; bj < bsr_dim; ++bj)
                    {
                        // sum of all entries over all BSR blocks in the current row
                        sum += (*(bsr_val_ptr + (bsr_dim * bj)) * *(x_ptr + bj));
                    }
                }
                // Perform alpha * A * x
                if constexpr(alpha_s)
                {
                    bsrmv_builder::alpha_scal<T, 1>(&sum, alpha);
                }

                // Perform (beta * y) + (alpha * A * x)
                if constexpr(beta_s)
                {
                    bsrmv_builder::beta_scal<T, 1>(&sum, beta, &y[ai * bsr_dim + bi]);
                }
                // BSR block row sum
                y[ai * bsr_dim + bi] = sum;
            }
        }
        return aoclsparse_status_success;
    }

    template <typename T, size_t KERNEL_DIM, bool alpha_s, bool beta_s>
    aoclsparse_status bsrmv_nxn(T                               alpha,
                                aoclsparse_int                  mb,
                                [[maybe_unused]] aoclsparse_int nb,
                                const T *__restrict__ bsr_val,
                                const aoclsparse_int *__restrict__ bsr_col_ind,
                                const aoclsparse_int *__restrict__ bsr_row_ptr,
                                const aoclsparse_mat_descr descr,
                                const T *__restrict__ x,
                                T beta,
                                T *__restrict__ y)
    {
        aoclsparse_index_base base = descr->base;

        aoclsparse_int row_begin, row_end, col;

        const T *bsr_val_ptr;
        const T *x_ptr;

        constexpr size_t KER_SQ = KERNEL_DIM * KERNEL_DIM;

        // Loop over the block rows
        for(aoclsparse_int ai = 0; ai < mb; ++ai)
        {
            T *y_ptr = y + ai * KERNEL_DIM;

            // BSR row entry and exit point
            row_begin = bsr_row_ptr[ai] - base;
            row_end   = bsr_row_ptr[ai + 1] - base;

            // BSR block row accumulator
            T sum[KERNEL_DIM] = {};

            // Loop over all BSR blocks in the current row
            for(aoclsparse_int aj = row_begin; aj < row_end; ++aj)
            {
                // Column index into x vector
                col = bsr_col_ind[aj] - base;

                bsr_val_ptr = bsr_val + (KER_SQ * aj);
                x_ptr       = x + (KERNEL_DIM * col);

                // Compute the sum of the 'KERNEL_DIM' rows within the BSR blocks of the current
                // BSR row
                bsrmv_builder::compute<T, KERNEL_DIM, KERNEL_DIM>(sum, bsr_val_ptr, x_ptr);
            }
            // Perform alpha * A * x
            if constexpr(alpha_s)
            {
                bsrmv_builder::alpha_scal<T, KERNEL_DIM>(sum, alpha);
            }

            // Perform (beta * y) + (alpha * A * x)
            if constexpr(beta_s)
            {
                bsrmv_builder::beta_scal<T, KERNEL_DIM>(sum, beta, y_ptr);
            }

            // BSR block row sum
            bsrmv_builder::store_res<T, KERNEL_DIM>(sum, y_ptr);
        }
        return aoclsparse_status_success;
    }
}
#endif