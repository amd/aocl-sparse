
/* ************************************************************************
 * Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_CSRMM_HPP
#define AOCLSPARSE_CSRMM_HPP
#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_dispatcher.hpp"
#include "aoclsparse_l3_kt.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <immintrin.h>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif
typedef union
{
    __m256d v;
    double  d[4] __attribute__((aligned(64)));
} v4df_t;
typedef union
{
    __m128d v;
    double  d[2] __attribute__((aligned(64)));
} v2df_t;
template <typename T>
aoclsparse_status aoclsparse_csrmm_col_major_ref(T                          alpha,
                                                 const aoclsparse_mat_descr descr,
                                                 const T *__restrict__ csr_val,
                                                 const aoclsparse_int *__restrict__ csr_col_ind,
                                                 const aoclsparse_int *__restrict__ csr_row_ptr,
                                                 aoclsparse_int m,
                                                 const T       *B,
                                                 aoclsparse_int n,
                                                 aoclsparse_int ldb,
                                                 T              beta,
                                                 T             *C,
                                                 aoclsparse_int ldc)
{
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *B_fix           = B - base;
    for(aoclsparse_int j = 0; j < n; ++j)
    {
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            aoclsparse_int idx_C     = i + j * ldc;
            T              sum       = 0.0;
            for(aoclsparse_int k = row_begin; k < row_end; ++k)
            {
                aoclsparse_int idx_B = (csr_col_ind_fix[k] + j * ldb);
                sum                  = csr_val_fix[k] * B_fix[idx_B] + sum;
            }
            C[idx_C] = (beta * C[idx_C]) + (alpha * sum);
        }
    }
    return aoclsparse_status_success;
}
template <typename T>
aoclsparse_status aoclsparse_csrmm_row_major_ref(T                          alpha,
                                                 const aoclsparse_mat_descr descr,
                                                 const T *__restrict__ csr_val,
                                                 const aoclsparse_int *__restrict__ csr_col_ind,
                                                 const aoclsparse_int *__restrict__ csr_row_ptr,
                                                 aoclsparse_int m,
                                                 const T       *B,
                                                 aoclsparse_int n,
                                                 aoclsparse_int ldb,
                                                 T              beta,
                                                 T             *C,
                                                 aoclsparse_int ldc)
{
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *B_fix           = B - (base * ldb);
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_begin = csr_row_ptr[i];
        aoclsparse_int row_end   = csr_row_ptr[i + 1];
        aoclsparse_int idx_C     = i * ldc;
        for(aoclsparse_int k = 0; k < n; ++k)
        {
            C[idx_C + k] = C[idx_C + k] * beta;
        }
        for(aoclsparse_int j = row_begin; j < row_end; ++j)
        {
            aoclsparse_int idx_B = csr_col_ind_fix[j] * ldb;
            for(aoclsparse_int k = 0; k < n; ++k)
            {
                C[idx_C + k] += csr_val_fix[j] * B_fix[idx_B + k] * alpha;
            }
        }
    }
    return aoclsparse_status_success;
}
template <typename T>
aoclsparse_status aoclsparse_csrmm_sym_row_ref(T                          alpha,
                                               const aoclsparse_mat_descr descr,
                                               const T *__restrict__ csr_val,
                                               const aoclsparse_int *__restrict__ csr_col_ind,
                                               const aoclsparse_int *__restrict__ csr_row_ptr,
                                               aoclsparse_int m,
                                               const T       *B,
                                               aoclsparse_int n,
                                               aoclsparse_int ldb,
                                               T             *C,
                                               aoclsparse_int ldc)
{
    T                     one  = 1.0;
    aoclsparse_index_base base = descr->base;
    // Variables to identify the type of the matrix
    const aoclsparse_fill_mode fill = descr->fill_mode;
    const aoclsparse_diag_type diag = descr->diag_type;
    for(int i = 0; i < m; i++)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        if(diag == aoclsparse_diag_type_unit)
        {
            for(int j = 0; j < n; j++)
            {
                aoclsparse_int idx_c = i * ldc + j;
                aoclsparse_int idx_b = i * ldb + j;
                C[idx_c] += one * B[idx_b] * alpha;
            }
        }
        for(int k = row_begin; k < row_end; k++)
        {
            bool is_diag = (i == (csr_col_ind[k] - base));
            if(is_diag && (diag == aoclsparse_diag_type_non_unit))
            {
                for(int j = 0; j < n; j++)
                {
                    aoclsparse_int idx_c = i * ldc + j;
                    aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                    C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                }
            }
            else
            {
                // this conditional can be hoisted outside the above loops, but would have replicate the code
                // Todo: evaluate the performance and make the changes
                if(fill == aoclsparse_fill_mode_lower)
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i * ldc + j;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                        // Access only lower triangle, update the idx_b and idx_c to process upper triangle of the matrix.
                        // Having a conditional is not efficient, but required if the the matrix A is not sorted.
                        // ToDo: sort matrix A by column indices to get rid of the conditional
                        if(i > (csr_col_ind[k] - base))
                        {
                            C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                            idx_b = i * ldb + j;
                            idx_c = (csr_col_ind[k] - base) * ldc + j;
                            C[idx_c] += aoclsparse::conj(csr_val[k]) * (B[idx_b]) * alpha;
                        }
                    }
                }
                else // fill == aoclsparse_fill_mode_upper
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i * ldc + j;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                        // Access only upper triangle
                        // Having a conditional is not efficient, but required if the the matrix A is not sorted.
                        // ToDo: sort matrix A by column indices to get rid of the conditional
                        if(i < (csr_col_ind[k] - base))
                        {
                            C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                            idx_b = i * ldb + j;
                            idx_c = (csr_col_ind[k] - base) * ldc + j;
                            C[idx_c] += aoclsparse::conj(csr_val[k]) * (B[idx_b]) * alpha;
                        }
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}
template <typename T>
aoclsparse_status aoclsparse_csrmm_sym_col_ref(T                          alpha,
                                               const aoclsparse_mat_descr descr,
                                               const T *__restrict__ csr_val,
                                               const aoclsparse_int *__restrict__ csr_col_ind,
                                               const aoclsparse_int *__restrict__ csr_row_ptr,
                                               aoclsparse_int m,
                                               const T       *B,
                                               aoclsparse_int n,
                                               aoclsparse_int ldb,
                                               T             *C,
                                               aoclsparse_int ldc)
{
    T                     one  = 1.0;
    aoclsparse_index_base base = descr->base;
    // Variables to identify the type of the matrix
    const aoclsparse_fill_mode fill = descr->fill_mode;
    const aoclsparse_diag_type diag = descr->diag_type;
    for(int i = 0; i < m; i++)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        if(diag == aoclsparse_diag_type_unit)
        {
            for(int j = 0; j < n; j++)
            {
                aoclsparse_int idx_c = i + j * ldc;
                aoclsparse_int idx_b = i + j * ldb;
                C[idx_c] += one * B[idx_b] * alpha;
            }
        }
        for(int k = row_begin; k < row_end; k++)
        {
            bool is_diag = (i == (csr_col_ind[k] - base));
            if(is_diag && (diag == aoclsparse_diag_type_non_unit))
            {
                for(int j = 0; j < n; j++)
                {
                    aoclsparse_int idx_c = i + j * ldc;
                    aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                    C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                }
            }
            else
            {
                // this conditional can be hoisted outside the above loops, but would have replicate the code
                // Todo: evaluate the performance and make the changes
                if(fill == aoclsparse_fill_mode_lower)
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i + j * ldc;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                        // Access only lower triangle, update the idx_b and idx_c to process upper triangle of the matrix.
                        // Having a conditional is not efficient, but required if the the matrix A is not sorted.
                        // ToDo: sort matrix A by column indices to get rid of the conditional
                        if(i > (csr_col_ind[k] - base))
                        {
                            C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                            idx_b = i + j * ldb;
                            idx_c = (csr_col_ind[k] - base) + j * ldc;
                            C[idx_c] += aoclsparse::conj(csr_val[k]) * (B[idx_b]) * alpha;
                        }
                    }
                }
                else // fill == aoclsparse_fill_mode_upper
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i + j * ldc;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                        // Access only upper triangle
                        // Having a conditional is not efficient, but required if the the matrix A is not sorted.
                        // ToDo: sort matrix A by column indices to get rid of the conditional
                        if(i < (csr_col_ind[k] - base))
                        {
                            C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                            idx_b = i + j * ldb;
                            idx_c = (csr_col_ind[k] - base) + j * ldc;
                            C[idx_c] += aoclsparse::conj(csr_val[k]) * (B[idx_b]) * alpha;
                        }
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csrmm_row_major(const float                alpha,
                                             const aoclsparse_mat_descr descr,
                                             const float *__restrict__ csr_val,
                                             const aoclsparse_int *__restrict__ csr_col_ind,
                                             const aoclsparse_int *__restrict__ csr_row_ptr,
                                             aoclsparse_int m,
                                             const float   *B,
                                             aoclsparse_int n,
                                             aoclsparse_int ldb,
                                             const float    beta,
                                             float         *C,
                                             aoclsparse_int ldc)
{
    // Number of sub-blocks of 4 columns in B matrix
    aoclsparse_int j_iter = n / 4;
    // Remainder numbers of column of B after multiple of 4
    aoclsparse_int j_rem = n % 4;
    // Indices of elements of input dense matrix B
    aoclsparse_int idx_B   = 0;
    aoclsparse_int idx_B_1 = 0;
    aoclsparse_int idx_B_2 = 0;
    aoclsparse_int idx_B_3 = 0;
    // Indices of output dense matrix C in row major format
    // Four elements of one row of C matrix gets updated in
    // one iteration.
    aoclsparse_int        idx_C           = 0;
    aoclsparse_int        idx_C_1         = 0;
    aoclsparse_int        idx_C_2         = 0;
    aoclsparse_int        idx_C_3         = 0;
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const float          *csr_val_fix     = csr_val - base;
    const float          *B_fix           = B - (base * ldb);
    // Iterate along sub-blocks of 4 columns of B matrix
    for(aoclsparse_int j = 0; j < j_iter * 4; j += 4)
    {
        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;
            // Indices of output dense matrix C in row major format
            // Four elements of one row of C matrix gets updated in
            // one iteration.
            idx_C   = i * ldc + j;
            idx_C_1 = i * ldc + j + 1;
            idx_C_2 = i * ldc + j + 2;
            idx_C_3 = i * ldc + j + 3;
            // Accumulator for 4 elements of C matrix
            float          sum[4] = {static_cast<float>(0)};
            aoclsparse_int k_iter = (row_end - row_begin) / 4;
            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j);
                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val_fix[k] * B_fix[idx_B];
                sum[0] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[0] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[0] += csr_val_fix[k + 3] * B_fix[idx_B_3];
                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j + 1);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j + 1);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j + 1);
                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val_fix[k] * B_fix[idx_B];
                sum[1] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[1] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[1] += csr_val_fix[k + 3] * B_fix[idx_B_3];
                // Indices of elements of (j+2)nd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j + 2);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j + 2);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j + 2);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j + 2);
                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+2)nd column of B matrix  and
                // add to accumulator
                sum[2] += csr_val_fix[k] * B_fix[idx_B];
                sum[2] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[2] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[2] += csr_val_fix[k + 3] * B_fix[idx_B_3];
                // Indices of elements of (j+3)rd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j + 3);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j + 3);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j + 3);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j + 3);
                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+3)rd column of B matrix  and
                // add to accumulator
                sum[3] += csr_val_fix[k] * B_fix[idx_B];
                sum[3] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[3] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[3] += csr_val_fix[k + 3] * B_fix[idx_B_3];
            }
            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st,(j+2)nd,(j+3)rd
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j);
                idx_B_1 = ((csr_col_ind_fix[k]) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind_fix[k]) * ldb + j + 2);
                idx_B_3 = ((csr_col_ind_fix[k]) * ldb + j + 3);
                // Multiply csr_val[k] by corresponding elements of
                // 4 columns of B matrix  and add to accumulator
                sum[0] += csr_val_fix[k] * B_fix[idx_B];
                sum[1] += csr_val_fix[k] * B_fix[idx_B_1];
                sum[2] += csr_val_fix[k] * B_fix[idx_B_2];
                sum[3] += csr_val_fix[k] * B_fix[idx_B_3];
            }
            // if beta = 0 , C= alpha*A*B
            if(beta == static_cast<float>(0))
            {
                C[idx_C]   = alpha * sum[0];
                C[idx_C_1] = alpha * sum[1];
                C[idx_C_2] = alpha * sum[2];
                C[idx_C_3] = alpha * sum[3];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(beta, C[idx_C], alpha * sum[0]);
                C[idx_C_1] = std::fma(beta, C[idx_C_1], alpha * sum[1]);
                C[idx_C_2] = std::fma(beta, C[idx_C_2], alpha * sum[2]);
                C[idx_C_3] = std::fma(beta, C[idx_C_3], alpha * sum[3]);
            }
        }
    }
    // if 3 == Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 3)
    {
        aoclsparse_int j = j_iter * 4;
        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;
            // Indices of output dense matrix C in row major format
            // Three elements of one row of C matrix gets updated in
            // one iteration.
            idx_C   = i * ldc + j;
            idx_C_1 = i * ldc + j + 1;
            idx_C_2 = i * ldc + j + 2;
            // Accumulator for 3 elements of C matrix
            float          sum[3] = {static_cast<float>(0)};
            aoclsparse_int k_iter = (row_end - row_begin) / 4;
            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j);
                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val_fix[k] * B_fix[idx_B];
                sum[0] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[0] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[0] += csr_val_fix[k + 3] * B_fix[idx_B_3];
                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j + 1);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j + 1);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j + 1);
                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val_fix[k] * B_fix[idx_B];
                sum[1] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[1] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[1] += csr_val_fix[k + 3] * B_fix[idx_B_3];
                // Indices of elements of (j+2)nd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j + 2);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j + 2);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j + 2);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j + 2);
                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+2)nd column of B matrix  and
                // add to accumulator
                sum[2] += csr_val_fix[k] * B_fix[idx_B];
                sum[2] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[2] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[2] += csr_val_fix[k + 3] * B_fix[idx_B_3];
            }
            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st,(j+2)nd
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j);
                idx_B_1 = ((csr_col_ind_fix[k]) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind_fix[k]) * ldb + j + 2);
                // Multiply csr_val[k] by corresponding elements of
                // 3 columns of B matrix  and add to accumulator
                sum[0] += csr_val_fix[k] * B_fix[idx_B];
                sum[1] += csr_val_fix[k] * B_fix[idx_B_1];
                sum[2] += csr_val_fix[k] * B_fix[idx_B_2];
            }
            // if beta = 0 , C= alpha*A*B
            if(beta == static_cast<float>(0))
            {
                C[idx_C]   = alpha * sum[0];
                C[idx_C_1] = alpha * sum[1];
                C[idx_C_2] = alpha * sum[2];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(beta, C[idx_C], alpha * sum[0]);
                C[idx_C_1] = std::fma(beta, C[idx_C_1], alpha * sum[1]);
                C[idx_C_2] = std::fma(beta, C[idx_C_2], alpha * sum[2]);
            }
        }
    }
    // if 2 == Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 2)
    {
        aoclsparse_int j = j_iter * 4;
        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;
            // Indices of output dense matrix C in row major format
            // Two elements of one row of C matrix gets updated in
            // one iteration.
            idx_C   = i * ldc + j;
            idx_C_1 = i * ldc + j + 1;
            // Accumulator for 2 elements of C matrix
            float          sum[2] = {static_cast<float>(0)};
            aoclsparse_int k_iter = (row_end - row_begin) / 4;
            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j);
                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val_fix[k] * B_fix[idx_B];
                sum[0] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[0] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[0] += csr_val_fix[k + 3] * B_fix[idx_B_3];
                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j + 1);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j + 1);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j + 1);
                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val_fix[k] * B_fix[idx_B];
                sum[1] += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum[1] += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum[1] += csr_val_fix[k + 3] * B_fix[idx_B_3];
            }
            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j);
                idx_B_1 = ((csr_col_ind_fix[k]) * ldb + j + 1);
                // Multiply csr_val[k] by corresponding elements of
                // 2 columns of B matrix  and add to accumulator
                sum[0] += csr_val_fix[k] * B_fix[idx_B];
                sum[1] += csr_val_fix[k] * B_fix[idx_B_1];
            }
            // if beta = 0 , C= alpha*A*B
            if(beta == static_cast<float>(0))
            {
                C[idx_C]   = alpha * sum[0];
                C[idx_C_1] = alpha * sum[1];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(beta, C[idx_C], alpha * sum[0]);
                C[idx_C_1] = std::fma(beta, C[idx_C_1], alpha * sum[1]);
            }
        }
    }
    // if 1 == Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 1)
    {
        aoclsparse_int j = j_iter * 4;
        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;
            // Index of output dense matrix C in row major format
            // ONe element of one row of C matrix gets updated in
            // one iteration.
            idx_C = i * ldc + j;
            // Accumulator for 1 element of C matrix
            float          sum    = static_cast<float>(0);
            aoclsparse_int k_iter = (row_end - row_begin) / 4;
            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind_fix[k]) * ldb + j);
                idx_B_1 = ((csr_col_ind_fix[k + 1]) * ldb + j);
                idx_B_2 = ((csr_col_ind_fix[k + 2]) * ldb + j);
                idx_B_3 = ((csr_col_ind_fix[k + 3]) * ldb + j);
                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum += csr_val_fix[k] * B_fix[idx_B];
                sum += csr_val_fix[k + 1] * B_fix[idx_B_1];
                sum += csr_val_fix[k + 2] * B_fix[idx_B_2];
                sum += csr_val_fix[k + 3] * B_fix[idx_B_3];
            }
            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of element of jth column of B matrix
                // to be multiplied against csr_val[k]
                idx_B = ((csr_col_ind_fix[k]) * ldb + j);
                // Multiply csr_val[k] by corresponding elements of
                // 1 columns of B matrix  and add to accumulator
                sum += csr_val_fix[k] * B_fix[idx_B];
            }
            // if beta = 0 , C= alpha*A*B
            if(beta == static_cast<float>(0))
            {
                C[idx_C] = alpha * sum;
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C] = std::fma(beta, C[idx_C], alpha * sum);
            }
        }
    }
    return aoclsparse_status_success;
}
// This function performs scaling for a dense matrix, 'mtrx', by a value 'beta'
template <typename T>
aoclsparse_status scale_dense_matrix(
    aoclsparse_order order, T *mtrx, aoclsparse_int m, aoclsparse_int n, aoclsparse_int ld, T beta)
{
    if(beta == aoclsparse_numeric::zero<T>())
    {
        if(order == aoclsparse_order_column)
        {
            for(aoclsparse_int j = 0; j < n; ++j)
            {
                for(aoclsparse_int i = 0; i < m; ++i)
                {
                    mtrx[i + j * ld] = 0;
                }
            }
        }
        else // order == aoclsparse_order_row
        {
            for(aoclsparse_int i = 0; i < m; ++i)
            {
                for(aoclsparse_int j = 0; j < n; ++j)
                {
                    mtrx[i * ld + j] = 0;
                }
            }
        }
    }
    else
    {
        if(order == aoclsparse_order_column)
        {
            for(aoclsparse_int j = 0; j < n; ++j)
            {
                for(aoclsparse_int i = 0; i < m; ++i)
                {
                    aoclsparse_int idx_C = i + j * ld;
                    mtrx[idx_C]          = beta * mtrx[idx_C];
                }
            }
        }
        else // order == aoclsparse_order_row
        {
            for(aoclsparse_int i = 0; i < m; ++i)
            {
                for(aoclsparse_int j = 0; j < n; ++j)
                {
                    aoclsparse_int idx_C = i * ld + j;
                    mtrx[idx_C]          = beta * mtrx[idx_C];
                }
            }
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csrmm_t(aoclsparse_operation       op,
                                     const T                    alpha,
                                     const aoclsparse_matrix    A,
                                     const aoclsparse_mat_descr descr,
                                     aoclsparse_order           order,
                                     const T                   *B,
                                     aoclsparse_int             n,
                                     aoclsparse_int             ldb,
                                     const T                    beta,
                                     T                         *C,
                                     aoclsparse_int             ldc,
                                     aoclsparse_int             kid)
{
    using namespace aoclsparse;
    using namespace Dispatch;
    using namespace kernel_templates;

    // Check for valid matrix, descriptor
    if(A == nullptr || B == nullptr || C == nullptr || descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Only CSR input format supported
    if(A->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_not_implemented;
    }
    // check if op is valid
    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;
    // check if the matrix type is implemented
    if(descr->type != aoclsparse_matrix_type_general
       && descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_hermitian)
        return aoclsparse_status_not_implemented;
    // check if the matrix is square for symmetric/hermitial matrices
    if((descr->type == aoclsparse_matrix_type_symmetric
        || descr->type == aoclsparse_matrix_type_hermitian)
       && A->m != A->n)
    {
        return aoclsparse_status_invalid_size;
    }
    // check if the layout is valid
    if(order != aoclsparse_order_row && order != aoclsparse_order_column)
        return aoclsparse_status_invalid_value;

    T zero{0.0};
    T one{1.0};

    aoclsparse_int m = A->m;
    aoclsparse_int k = A->n;
    aoclsparse_int m_c{0}, n_c{0};

    const aoclsparse_int *csr_col_ind = A->csr_mat.csr_col_ptr;
    const aoclsparse_int *csr_row_ptr = A->csr_mat.csr_row_ptr;
    const T              *csr_val     = static_cast<T *>(A->csr_mat.csr_val);

    // Variables to identify the type of the matrix
    const aoclsparse_matrix_type mat_type = descr->type;

    // Verify the matrix types and T are consistent
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;
    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    // Check for base index incompatibility
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    // Check sizes
    if(m < 0 || n < 0 || k < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        return aoclsparse_status_success;
    }
    if(alpha == zero && beta == one)
    {
        return aoclsparse_status_success;
    }
    // Check the rest of pointer arguments
    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check leading dimension of B
    aoclsparse_int check_ldb;
    if(op == aoclsparse_operation_none)
        check_ldb = (order == aoclsparse_order_column ? k : n);
    else
        check_ldb = (order == aoclsparse_order_column ? m : n);
    if(ldb < (((aoclsparse_int)1) >= check_ldb ? (aoclsparse_int)1 : check_ldb))
    {
        return aoclsparse_status_invalid_size;
    }
    // Check leading dimension of C
    aoclsparse_int check_ldc;
    if(op == aoclsparse_operation_none)
        check_ldc = (order == aoclsparse_order_column ? m : n);
    else
        check_ldc = (order == aoclsparse_order_column ? k : n);
    if(ldc < (((aoclsparse_int)1) >= check_ldc ? (aoclsparse_int)1 : check_ldc))
    {
        return aoclsparse_status_invalid_size;
    }
    // a few kernels are already fused with beta, so not updating C for those kernels
    if(op == aoclsparse_operation_none)
    {
        m_c = m;
    }
    else
    {
        m_c = k;
    }
    n_c = n;
    // In case of general type matrices, beta scaling is done inside the kernel
    // To support early return cases for alpha == zero scenario
    if(alpha == zero || mat_type != aoclsparse_matrix_type_general) [[maybe_unused]]
        aoclsparse_status st = scale_dense_matrix(order, C, m_c, n_c, ldc, beta);
    if(alpha == zero)
    {
        return aoclsparse_status_success; // Early return
    }
    // Invokes kernels for symmetric and hermitian matrices
    if(mat_type != aoclsparse_matrix_type_general)
    {
        [[maybe_unused]] std::vector<T> csr_val_A;
        T                              *val_A = const_cast<T *>(csr_val);
        if(op == aoclsparse_operation_conjugate_transpose || op == aoclsparse_operation_transpose)
        {
            csr_val_A.resize(A->nnz);
            // For symmetric and hermitian matrices, we only use:
            // 1. Orginal row pointers and column indices,
            // 2. Apply conjugate on original value array(non-transposed csr_val), because of the following reasons:
            //    - Symmetric matrices are equal to its transpose.
            //    - Hertmitian matrices are equal to its conjugate transpose.
            // This enables kernel to process only the required triangle of the matrix (either upper or lower triangle)
            // using the orignal row pointers and column indices. This is useful only in Hermitian matrices.
            // Apply conjugate on transposed value array.
            for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
                csr_val_A[idx] = aoclsparse::conj(csr_val[idx]);
            val_A = csr_val_A.data();
        }
        using K = decltype(&aoclsparse_csrmm_sym_col_ref<T>);
        K kernel;
        if(order == aoclsparse_order_column)
            kernel = aoclsparse_csrmm_sym_col_ref;
        else // order == aoclsparse_order_row
            kernel = aoclsparse_csrmm_sym_row_ref;
        return kernel(alpha, descr, val_A, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
    }
    else // mat_type == aoclsparse_matrix_type_general
    {
        std::vector<aoclsparse_int> csr_row_ptr_A;
        std::vector<aoclsparse_int> csr_col_ind_A;
        std::vector<T>              csr_val_A;
        aoclsparse_int             *row_ptr_A = const_cast<aoclsparse_int *>(csr_row_ptr);
        aoclsparse_int             *col_ind_A = const_cast<aoclsparse_int *>(csr_col_ind);
        T                          *val_A     = const_cast<T *>(csr_val);
        aoclsparse_int              mb        = m;
        if(op == aoclsparse_operation_conjugate_transpose || op == aoclsparse_operation_transpose)
        {
            csr_col_ind_A.resize(A->nnz);
            csr_row_ptr_A.resize(A->n + 1);
            csr_val_A.resize(A->nnz);
            aoclsparse_status status = aoclsparse_csr2csc_template(A->m,
                                                                   A->n,
                                                                   A->nnz,
                                                                   descr->base,
                                                                   descr->base,
                                                                   csr_row_ptr,
                                                                   csr_col_ind,
                                                                   csr_val,
                                                                   csr_col_ind_A.data(),
                                                                   csr_row_ptr_A.data(),
                                                                   csr_val_A.data());
            if(status != aoclsparse_status_success)
                return aoclsparse_status_internal_error;
            // Apply conjugate on transposed value array.
            if(op == aoclsparse_operation_conjugate_transpose)
            {
                for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
                    csr_val_A[idx] = aoclsparse::conj(csr_val_A[idx]);
            }
            row_ptr_A = csr_row_ptr_A.data();
            col_ind_A = csr_col_ind_A.data();
            val_A     = csr_val_A.data();
            mb        = k;
        }
        if(order == aoclsparse_order_column)
        {
            // Column order
            using K = decltype(&aoclsparse_csrmm_col_major_ref<T>);

            // clang-format off
            // Table of available kernels
            static constexpr Table<K> tbl[]{
            {aoclsparse_csrmm_col_major_ref<T>, context_isa_t::GENERIC, 0U | archs::ALL},
            {csrmm_col_kt<bsz::b256, T>,        context_isa_t::AVX2,    0U | archs::ALL},
     ORL<K>({csrmm_col_kt<bsz::b256, T>,        context_isa_t::AVX512VL,0U | archs::ALL}),
     ORL<K>({csrmm_col_kt<bsz::b512, T>,        context_isa_t::AVX512F, 0U | archs::ALL})
            };
            // clang-format on

            // Inquire with the oracle
            auto kernel = Oracle<K, Dispatch::api::csrmm>(tbl, kid);

            if(!kernel)
                return aoclsparse_status_invalid_kid;

            // Invoke the kernel
            return kernel(alpha, descr, val_A, col_ind_A, row_ptr_A, mb, B, n, ldb, beta, C, ldc);
        }
        else
        {
            // Row order
            using K = decltype(&aoclsparse_csrmm_row_major_ref<T>);
            K kernel;
            switch(kid)
            {
            case 1:
                if constexpr(std::is_same_v<T, float>)
                {
                    // Special case hand-written single precision
                    kernel = aoclsparse_csrmm_row_major;
                    break;
                }
                [[fallthrough]];
            case 3:
            case 2:
            case 0:
            case -1:
                kernel = aoclsparse_csrmm_row_major_ref;
                break;
            default:
                return aoclsparse_status_invalid_kid;
            }
            return kernel(alpha, descr, val_A, col_ind_A, row_ptr_A, mb, B, n, ldb, beta, C, ldc);
        }
    }
    return aoclsparse_status_not_implemented;
}
#endif /* AOCLSPARSE_CSRMM_HPP*/
