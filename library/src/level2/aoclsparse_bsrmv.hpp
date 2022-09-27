/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_BSRMV_HPP
#define AOCLSPARSE_BSRMV_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

template <typename T>
aoclsparse_status aoclsparse_bsrmv_general(T              alpha,
                                           aoclsparse_int mb,
                                           aoclsparse_int nb,
                                           aoclsparse_int bsr_dim,
                                           const T *__restrict__ bsr_val,
                                           const aoclsparse_int *__restrict__ bsr_col_ind,
                                           const aoclsparse_int *__restrict__ bsr_row_ptr,
                                           const T *__restrict__ x,
                                           T beta,
                                           T *__restrict__ y)
{
    // Loop over the block rows
    for(aoclsparse_int ai = 0; ai < mb; ++ai)
    {
        // BSR row entry and exit point
        aoclsparse_int row_begin = bsr_row_ptr[ai];
        aoclsparse_int row_end   = bsr_row_ptr[ai + 1];
        // Loop over the individual rows within the BSR block
        for(aoclsparse_int bi = 0; bi < bsr_dim; ++bi)
        {
            // BSR block row accumulator
            T sum = static_cast<T>(0);
            // Loop over all BSR blocks in the current row
            for(aoclsparse_int aj = row_begin; aj < row_end; ++aj)
            {
                // Column index into x vector
                aoclsparse_int col = bsr_col_ind[aj];
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
            if(alpha != static_cast<float>(1))
            {
                sum = alpha * sum;
            }

            // Perform (beta * y) + (alpha * A * x)
            if(beta != static_cast<float>(0))
            {
                sum += beta * y[ai * bsr_dim + bi];
            }
            // BSR block row sum
            y[ai * bsr_dim + bi] = sum;
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_bsrmv_2x2(T              alpha,
                                       aoclsparse_int mb,
                                       aoclsparse_int nb,
                                       const T *__restrict__ bsr_val,
                                       const aoclsparse_int *__restrict__ bsr_col_ind,
                                       const aoclsparse_int *__restrict__ bsr_row_ptr,
                                       const T *__restrict__ x,
                                       T beta,
                                       T *__restrict__ y)
{
    // BSR block dimension
    static constexpr int bsr_dim = 2;
    // Loop over the block rows
    for(aoclsparse_int ai = 0; ai < mb; ++ai)
    {
        // BSR row entry and exit point
        aoclsparse_int row_begin = bsr_row_ptr[ai];
        aoclsparse_int row_end   = bsr_row_ptr[ai + 1];

        // BSR block row accumulator
        T sum0 = static_cast<T>(0);
        T sum1 = static_cast<T>(0);

        // Loop over all BSR blocks in the current row
        for(aoclsparse_int aj = row_begin; aj < row_end; ++aj)
        {
            // Column index into x vector
            aoclsparse_int col = bsr_col_ind[aj];
            const T       *bsr_val_ptr;
            const T       *x_ptr;
            bsr_val_ptr = bsr_val + (bsr_dim * bsr_dim * aj);
            x_ptr       = x + (bsr_dim * col);
            // Compute the sum of the two rows within the BSR blocks of the current
            // BSR row
            sum0 += *bsr_val_ptr * *x_ptr;
            sum1 += *(bsr_val_ptr + 1) * *x_ptr;
            sum0 += *(bsr_val_ptr + 2) * *(x_ptr + 1);
            sum1 += *(bsr_val_ptr + 3) * *(x_ptr + 1);
        }
        // Perform alpha * A * x
        if(alpha != static_cast<float>(1))
        {
            sum0 = alpha * sum0;
            sum1 = alpha * sum1;
        }

        // Perform (beta * y) + (alpha * A * x)
        if(beta != static_cast<float>(0))
        {
            sum0 += beta * y[ai * bsr_dim + 0];
            sum1 += beta * y[ai * bsr_dim + 1];
        }
        // BSR block row sum
        y[ai * bsr_dim + 0] = sum0;
        y[ai * bsr_dim + 1] = sum1;
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_bsrmv_3x3(T              alpha,
                                       aoclsparse_int mb,
                                       aoclsparse_int nb,
                                       const T *__restrict__ bsr_val,
                                       const aoclsparse_int *__restrict__ bsr_col_ind,
                                       const aoclsparse_int *__restrict__ bsr_row_ptr,
                                       const T *__restrict__ x,
                                       T beta,
                                       T *__restrict__ y)
{
    // BSR block dimension
    static constexpr int bsr_dim = 3;
    // Loop over the block rows
    for(aoclsparse_int ai = 0; ai < mb; ++ai)
    {
        // BSR row entry and exit point
        aoclsparse_int row_begin = bsr_row_ptr[ai];
        aoclsparse_int row_end   = bsr_row_ptr[ai + 1];

        // BSR block row accumulator
        T sum0 = static_cast<T>(0);
        T sum1 = static_cast<T>(0);
        T sum2 = static_cast<T>(0);

        // Loop over all BSR blocks in the current row
        for(aoclsparse_int aj = row_begin; aj < row_end; ++aj)
        {
            // Column index into x vector
            aoclsparse_int col = bsr_col_ind[aj];
            const T       *bsr_val_ptr;
            const T       *x_ptr;
            bsr_val_ptr = bsr_val + (bsr_dim * bsr_dim * aj);
            x_ptr       = x + (bsr_dim * col);

            // Compute the sum of the three rows within the BSR blocks of the current
            // BSR row
            sum0 += *bsr_val_ptr * *x_ptr;
            sum1 += *(bsr_val_ptr + 1) * *x_ptr;
            sum2 += *(bsr_val_ptr + 2) * *x_ptr;
            sum0 += *(bsr_val_ptr + 3) * *(x_ptr + 1);
            sum1 += *(bsr_val_ptr + 4) * *(x_ptr + 1);
            sum2 += *(bsr_val_ptr + 5) * *(x_ptr + 1);
            sum0 += *(bsr_val_ptr + 6) * *(x_ptr + 2);
            sum1 += *(bsr_val_ptr + 7) * *(x_ptr + 2);
            sum2 += *(bsr_val_ptr + 8) * *(x_ptr + 2);
        }
        // Perform alpha * A * x
        if(alpha != static_cast<float>(1))
        {
            sum0 = alpha * sum0;
            sum1 = alpha * sum1;
            sum2 = alpha * sum2;
        }

        // Perform (beta * y) + (alpha * A * x)
        if(beta != static_cast<float>(0))
        {
            sum0 += beta * y[ai * bsr_dim + 0];
            sum1 += beta * y[ai * bsr_dim + 1];
            sum2 += beta * y[ai * bsr_dim + 2];
        }

        // BSR block row sum
        y[ai * bsr_dim + 0] = sum0;
        y[ai * bsr_dim + 1] = sum1;
        y[ai * bsr_dim + 2] = sum2;
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_bsrmv_4x4(T              alpha,
                                       aoclsparse_int mb,
                                       aoclsparse_int nb,
                                       const T *__restrict__ bsr_val,
                                       const aoclsparse_int *__restrict__ bsr_col_ind,
                                       const aoclsparse_int *__restrict__ bsr_row_ptr,
                                       const T *__restrict__ x,
                                       T beta,
                                       T *__restrict__ y)
{
    // BSR block dimension
    static constexpr int bsr_dim = 4;
    // Loop over the block rows
    for(aoclsparse_int ai = 0; ai < mb; ++ai)
    {
        // BSR row entry and exit point
        aoclsparse_int row_begin = bsr_row_ptr[ai];
        aoclsparse_int row_end   = bsr_row_ptr[ai + 1];

        // BSR block row accumulator
        T sum0 = static_cast<T>(0);
        T sum1 = static_cast<T>(0);
        T sum2 = static_cast<T>(0);
        T sum3 = static_cast<T>(0);

        // Loop over all BSR blocks in the current row
        for(aoclsparse_int aj = row_begin; aj < row_end; ++aj)
        {
            // Column index into x vector
            aoclsparse_int col = bsr_col_ind[aj];
            const T       *bsr_val_ptr;
            const T       *x_ptr;
            bsr_val_ptr = bsr_val + (bsr_dim * bsr_dim * aj);
            x_ptr       = x + (bsr_dim * col);

            // Compute the sum of the four rows within the BSR blocks of the current
            // BSR row
            sum0 += *bsr_val_ptr * *x_ptr;
            sum1 += *(bsr_val_ptr + 1) * *x_ptr;
            sum2 += *(bsr_val_ptr + 2) * *x_ptr;
            sum3 += *(bsr_val_ptr + 3) * *x_ptr;
            sum0 += *(bsr_val_ptr + 4) * *(x_ptr + 1);
            sum1 += *(bsr_val_ptr + 5) * *(x_ptr + 1);
            sum2 += *(bsr_val_ptr + 6) * *(x_ptr + 1);
            sum3 += *(bsr_val_ptr + 7) * *(x_ptr + 1);
            sum0 += *(bsr_val_ptr + 8) * *(x_ptr + 2);
            sum1 += *(bsr_val_ptr + 9) * *(x_ptr + 2);
            sum2 += *(bsr_val_ptr + 10) * *(x_ptr + 2);
            sum3 += *(bsr_val_ptr + 11) * *(x_ptr + 2);
            sum0 += *(bsr_val_ptr + 12) * *(x_ptr + 3);
            sum1 += *(bsr_val_ptr + 13) * *(x_ptr + 3);
            sum2 += *(bsr_val_ptr + 14) * *(x_ptr + 3);
            sum3 += *(bsr_val_ptr + 15) * *(x_ptr + 3);
        }
        // Perform alpha * A * x
        if(alpha != static_cast<float>(1))
        {
            sum0 = alpha * sum0;
            sum1 = alpha * sum1;
            sum2 = alpha * sum2;
            sum3 = alpha * sum3;
        }

        // Perform (beta * y) + (alpha * A * x)
        if(beta != static_cast<float>(0))
        {
            sum0 += beta * y[ai * bsr_dim + 0];
            sum1 += beta * y[ai * bsr_dim + 1];
            sum2 += beta * y[ai * bsr_dim + 2];
            sum3 += beta * y[ai * bsr_dim + 3];
        }

        // BSR block row sum
        y[ai * bsr_dim + 0] = sum0;
        y[ai * bsr_dim + 1] = sum1;
        y[ai * bsr_dim + 2] = sum2;
        y[ai * bsr_dim + 3] = sum3;
    }
    return aoclsparse_status_success;
}

#endif // AOCLSPARSE_BSRMV_HPP
