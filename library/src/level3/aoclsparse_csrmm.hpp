/* ************************************************************************
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <immintrin.h>

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

aoclsparse_status aoclsparse_csrmm_col_major(const double              *alpha,
                                             const aoclsparse_mat_descr descr,
                                             const double *__restrict__ csr_val,
                                             const aoclsparse_int *__restrict__ csr_col_ind,
                                             const aoclsparse_int *__restrict__ csr_row_ptr,
                                             aoclsparse_int                  m,
                                             [[maybe_unused]] aoclsparse_int k,
                                             const double                   *B,
                                             aoclsparse_int                  n,
                                             aoclsparse_int                  ldb,
                                             const double                   *beta,
                                             double                         *C,
                                             aoclsparse_int                  ldc)
{
    // Number of sub-blocks of 4 columns in B matrix
    aoclsparse_int j_iter = n / 4;

    // Remainder numbers of column of B after multiple of 4
    aoclsparse_int j_rem = n % 4;

    // Offsets to each of four columns j,j+1,j+2,j+3
    // of B dense matrix in column major format
    aoclsparse_int j_offset   = 0;
    aoclsparse_int j_offset_1 = 0;
    aoclsparse_int j_offset_2 = 0;
    aoclsparse_int j_offset_3 = 0;

    // Indices of elements of input dense matrix B
    aoclsparse_int idx_B   = 0;
    aoclsparse_int idx_B_1 = 0;
    aoclsparse_int idx_B_2 = 0;
    aoclsparse_int idx_B_3 = 0;
    aoclsparse_int idx_B_4 = 0;
    aoclsparse_int idx_B_5 = 0;
    aoclsparse_int idx_B_6 = 0;
    aoclsparse_int idx_B_7 = 0;

    // Indices of output dense matrix C in column major format
    // Four elements of one row of C matrix gets updated in
    // one iteration.
    aoclsparse_int idx_C   = 0;
    aoclsparse_int idx_C_1 = 0;
    aoclsparse_int idx_C_2 = 0;
    aoclsparse_int idx_C_3 = 0;

    // xmm registers for A , B ,C matrix elements
    v2df_t                vec_A, vec_B;
    v2df_t                vec_B_1;
    v2df_t                vec_B_2;
    v2df_t                vec_B_3;
    v2df_t                vec_C;
    v2df_t                vec_C_1;
    v2df_t                vec_C_2;
    v2df_t                vec_C_3;
    aoclsparse_index_base base = descr->base;

    // Iterate along sub-blocks of 4 columns of B matrix
    for(aoclsparse_int j = 0; j < j_iter * 4; j += 4)
    {
        // Offsets to each of four columns j,j+1,j+2,j+3
        // of B dense matrix in column major format
        j_offset   = j * ldb;
        j_offset_1 = (j + 1) * ldb;
        j_offset_2 = (j + 2) * ldb;
        j_offset_3 = (j + 3) * ldb;

        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;
            idx_B_4 = 0;
            idx_B_5 = 0;
            idx_B_6 = 0;
            idx_B_7 = 0;

            // Indices of output dense matrix C in column major format
            // Four elements of one row of C matrix gets updated in
            // one iteration.
            idx_C   = i + j * ldc;
            idx_C_1 = i + ((j + 1) * ldc);
            idx_C_2 = i + ((j + 2) * ldc);
            idx_C_3 = i + ((j + 3) * ldc);

            // Accumulator for 4 elements of C matrix
            double sum[4];

            // Set Accumulators to zero
            vec_C.v   = _mm_setzero_pd();
            vec_C_1.v = _mm_setzero_pd();
            vec_C_2.v = _mm_setzero_pd();
            vec_C_3.v = _mm_setzero_pd();

            const double *csr_val_ptr = &csr_val[row_begin];

            //Iterate over non-zeroes of ith row of A in multiples of 2
            for(aoclsparse_int k = row_begin; k < row_end - 1; k += 2)
            {
                //Load csr_val[k] csr_val[k+1] into xmm register
                vec_A.v = _mm_loadu_pd(csr_val_ptr);
                csr_val_ptr += 2;

                // Column indices of csr_val[k] csr_val[k+1]
                aoclsparse_int csr_col_ind_k   = csr_col_ind[k] - base;
                aoclsparse_int csr_col_ind_k_1 = csr_col_ind[k + 1] - base;

                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B      = (csr_col_ind_k + j_offset);
                idx_B_1    = (csr_col_ind_k_1 + j_offset);
                vec_B.d[0] = B[idx_B];
                vec_B.d[1] = B[idx_B_1];

                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B_2      = (csr_col_ind_k + j_offset_1);
                idx_B_3      = (csr_col_ind_k_1 + j_offset_1);
                vec_B_1.d[0] = B[idx_B_2];
                vec_B_1.d[1] = B[idx_B_3];

                // Indices of elements of (j+2)nd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B_4      = (csr_col_ind_k + j_offset_2);
                idx_B_5      = (csr_col_ind_k_1 + j_offset_2);
                vec_B_2.d[0] = B[idx_B_4];
                vec_B_2.d[1] = B[idx_B_5];

                // Indices of elements of (j+3)rd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B_6      = (csr_col_ind_k + j_offset_3);
                idx_B_7      = (csr_col_ind_k_1 + j_offset_3);
                vec_B_3.d[0] = B[idx_B_6];
                vec_B_3.d[1] = B[idx_B_7];

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of jth column of B matrix  and add to
                // accumulator
                vec_B.v = _mm_mul_pd(vec_A.v, vec_B.v);
                vec_C.v = _mm_add_pd(vec_C.v, vec_B.v);

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of (j+1)st column of B matrix  and add to
                // accumulator
                vec_B_1.v = _mm_mul_pd(vec_A.v, vec_B_1.v);
                vec_C_1.v = _mm_add_pd(vec_C_1.v, vec_B_1.v);

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of (j+2)nd column of B matrix  and add to
                // accumulator
                vec_B_2.v = _mm_mul_pd(vec_A.v, vec_B_2.v);
                vec_C_2.v = _mm_add_pd(vec_C_2.v, vec_B_2.v);

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of (j+3)rd column of B matrix  and add to
                // accumulator
                vec_B_3.v = _mm_mul_pd(vec_A.v, vec_B_3.v);
                vec_C_3.v = _mm_add_pd(vec_C_3.v, vec_B_3.v);
            }

            //Remainder one non-zero of ith row of A, if nnz in the row is odd
            if(((row_end - row_begin) % 2) == 1)
            {
                //Load last single non-zero of ith row into xmm register
                vec_A.d[0] = csr_val[row_end - 1];

                // Indices of elements of jth,(j+1)st,(j+2)nd,(j+3)rd
                // columns of B matrix to be multiplied against
                // csr_val[row_end - 1]
                // Load the specific B elements into xmm registers
                idx_B        = (csr_col_ind[row_end - 1] - base + j_offset);
                idx_B_1      = (csr_col_ind[row_end - 1] - base + j_offset_1);
                idx_B_2      = (csr_col_ind[row_end - 1] - base + j_offset_2);
                idx_B_3      = (csr_col_ind[row_end - 1] - base + j_offset_3);
                vec_B.d[0]   = B[idx_B];
                vec_B_1.d[0] = B[idx_B_1];
                vec_B_2.d[0] = B[idx_B_2];
                vec_B_3.d[0] = B[idx_B_3];

                // Multiply last single non-zero by corresponding
                // elements of 4 columns of B matrix  and add to
                // accumulator
                vec_B.v   = _mm_mul_sd(vec_A.v, vec_B.v);
                vec_C.v   = _mm_add_sd(vec_C.v, vec_B.v);
                vec_B_1.v = _mm_mul_sd(vec_A.v, vec_B_1.v);
                vec_C_1.v = _mm_add_sd(vec_C_1.v, vec_B_1.v);
                vec_B_2.v = _mm_mul_sd(vec_A.v, vec_B_2.v);
                vec_C_2.v = _mm_add_sd(vec_C_2.v, vec_B_2.v);
                vec_B_3.v = _mm_mul_sd(vec_A.v, vec_B_3.v);
                vec_C_3.v = _mm_add_sd(vec_C_3.v, vec_B_3.v);
            }
            // Horizontal addition of lower and higher double
            // values in accumulator
            sum[0] = vec_C.d[0] + vec_C.d[1];
            sum[1] = vec_C_1.d[0] + vec_C_1.d[1];
            sum[2] = vec_C_2.d[0] + vec_C_2.d[1];
            sum[3] = vec_C_3.d[0] + vec_C_3.d[1];

            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<double>(0))
            {
                // if beta = 0 & alpha = 1, C= A*B
                if(*alpha == static_cast<double>(1))
                {
                    C[idx_C]   = sum[0];
                    C[idx_C_1] = sum[1];
                    C[idx_C_2] = sum[2];
                    C[idx_C_3] = sum[3];
                }
                // if beta = 0 & alpha != 1, C= alpha*A*B
                else
                {
                    C[idx_C]   = *alpha * sum[0];
                    C[idx_C_1] = *alpha * sum[1];
                    C[idx_C_2] = *alpha * sum[2];
                    C[idx_C_3] = *alpha * sum[3];
                }
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
                C[idx_C_2] = std::fma(*beta, C[idx_C_2], *alpha * sum[2]);
                C[idx_C_3] = std::fma(*beta, C[idx_C_3], *alpha * sum[3]);
            }
        }
    }

    // if 3 == Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 3)
    {
        aoclsparse_int j = j_iter * 4;

        // Offsets to each of last three columns
        // of B dense matrix in column major format
        j_offset   = j * ldb;
        j_offset_1 = (j + 1) * ldb;
        j_offset_2 = (j + 2) * ldb;

        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;
            idx_B_4 = 0;
            idx_B_5 = 0;

            // Indices of output dense matrix C in column major format
            // Three elements of one row of C matrix gets updated in
            // one iteration.
            idx_C   = i + j * ldc;
            idx_C_1 = i + ((j + 1) * ldc);
            idx_C_2 = i + ((j + 2) * ldc);

            // Accumulator for 4 elements of C matrix
            double sum[3] = {static_cast<double>(0)};

            // Set Accumulators to zero
            vec_C.v   = _mm_setzero_pd();
            vec_C_1.v = _mm_setzero_pd();
            vec_C_2.v = _mm_setzero_pd();

            const double *csr_val_ptr = &csr_val[row_begin];

            //Iterate over non-zeroes of ith row of A in multiples of 2
            for(aoclsparse_int k = row_begin; k < row_end - 1; k += 2)
            {
                //Load csr_val[k] csr_val[k+1] into xmm register
                vec_A.v = _mm_loadu_pd(csr_val_ptr);
                csr_val_ptr += 2;

                // Column indices of csr_val[k] csr_val[k+1]
                aoclsparse_int csr_col_ind_k   = csr_col_ind[k] - base;
                aoclsparse_int csr_col_ind_k_1 = csr_col_ind[k + 1] - base;

                // Indices of elements of third last column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B      = (csr_col_ind_k + j_offset);
                idx_B_1    = (csr_col_ind_k_1 + j_offset);
                vec_B.d[0] = B[idx_B];
                vec_B.d[1] = B[idx_B_1];

                // Indices of elements of 2nd last column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B_2      = (csr_col_ind_k + j_offset_1);
                idx_B_3      = (csr_col_ind_k_1 + j_offset_1);
                vec_B_1.d[0] = B[idx_B_2];
                vec_B_1.d[1] = B[idx_B_3];

                // Indices of elements of last column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B_4      = (csr_col_ind_k + j_offset_2);
                idx_B_5      = (csr_col_ind_k_1 + j_offset_2);
                vec_B_2.d[0] = B[idx_B_4];
                vec_B_2.d[1] = B[idx_B_5];

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of third last column of B matrix  and add to
                // accumulator
                vec_B.v = _mm_mul_pd(vec_A.v, vec_B.v);
                vec_C.v = _mm_add_pd(vec_C.v, vec_B.v);

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of second last column of B matrix  and add to
                // accumulator
                vec_B_1.v = _mm_mul_pd(vec_A.v, vec_B_1.v);
                vec_C_1.v = _mm_add_pd(vec_C_1.v, vec_B_1.v);

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of last column of B matrix  and add to
                // accumulator
                vec_B_2.v = _mm_mul_pd(vec_A.v, vec_B_2.v);
                vec_C_2.v = _mm_add_pd(vec_C_2.v, vec_B_2.v);
            }
            //Remainder one non-zero of ith row of A, if nnz in the row is odd
            if(((row_end - row_begin) % 2) == 1)
            {
                //Load last single non-zero of ith row into xmm register
                vec_A.d[0] = csr_val[row_end - 1];

                // Indices of elements of last three columns of
                // B matrix to be multiplied against csr_val[row_end - 1]
                // Load the specific B elements into xmm registers
                idx_B        = (csr_col_ind[row_end - 1] - base + j_offset);
                idx_B_1      = (csr_col_ind[row_end - 1] - base + j_offset_1);
                idx_B_2      = (csr_col_ind[row_end - 1] - base + j_offset_2);
                vec_B.d[0]   = B[idx_B];
                vec_B_1.d[0] = B[idx_B_1];
                vec_B_2.d[0] = B[idx_B_2];

                // Multiply last single non-zero by corresponding
                // elements of 3 columns of B matrix  and add to
                // accumulator
                vec_B.v   = _mm_mul_sd(vec_A.v, vec_B.v);
                vec_C.v   = _mm_add_sd(vec_C.v, vec_B.v);
                vec_B_1.v = _mm_mul_sd(vec_A.v, vec_B_1.v);
                vec_C_1.v = _mm_add_sd(vec_C_1.v, vec_B_1.v);
                vec_B_2.v = _mm_mul_sd(vec_A.v, vec_B_2.v);
                vec_C_2.v = _mm_add_sd(vec_C_2.v, vec_B_2.v);
            }
            // Horizontal addition of lower and higher double
            // values in accumulator
            sum[0] = vec_C.d[0] + vec_C.d[1];
            sum[1] = vec_C_1.d[0] + vec_C_1.d[1];
            sum[2] = vec_C_2.d[0] + vec_C_2.d[1];

            // if beta == 0 ,C= alpha*A*B
            if(*beta == static_cast<double>(0))
            {
                // if beta == 0 & alpha == 1, C= A*B
                if(*alpha == static_cast<double>(1))
                {
                    C[idx_C]   = sum[0];
                    C[idx_C_1] = sum[1];
                    C[idx_C_2] = sum[2];
                }
                // if beta == 0 & alpha != 1, C= alpha*A*B
                else
                {
                    C[idx_C]   = *alpha * sum[0];
                    C[idx_C_1] = *alpha * sum[1];
                    C[idx_C_2] = *alpha * sum[2];
                }
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
                C[idx_C_2] = std::fma(*beta, C[idx_C_2], *alpha * sum[2]);
            }
        }
    }
    // if 2== Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 2)
    {
        aoclsparse_int j = j_iter * 4;
        // Offsets to each of last three columns
        // of B dense matrix in column major format
        j_offset   = j * ldb;
        j_offset_1 = (j + 1) * ldb;

        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;
            idx_C   = i + j * ldc;
            idx_C_1 = i + ((j + 1) * ldc);

            // Accumulator for 4 elements of C matrix
            double sum[2] = {static_cast<double>(0)};

            // Set Accumulators to zero
            vec_C.v   = _mm_setzero_pd();
            vec_C_1.v = _mm_setzero_pd();

            const double *csr_val_ptr = &csr_val[row_begin];

            //Iterate over non-zeroes of ith row of A in multiples of 2
            for(aoclsparse_int k = row_begin; k < row_end - 1; k += 2)
            {
                //Load csr_val[k] csr_val[k+1] into xmm register
                vec_A.v = _mm_loadu_pd(csr_val_ptr);
                csr_val_ptr += 2;

                // Column indices of csr_val[k] csr_val[k+1]
                aoclsparse_int csr_col_ind_k   = csr_col_ind[k] - base;
                aoclsparse_int csr_col_ind_k_1 = csr_col_ind[k + 1] - base;

                // Indices of elements of 2nd last column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B      = (csr_col_ind_k + j_offset);
                idx_B_1    = (csr_col_ind_k_1 + j_offset);
                vec_B.d[0] = B[idx_B];
                vec_B.d[1] = B[idx_B_1];

                // Indices of elements of last column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B_2      = (csr_col_ind_k + j_offset_1);
                idx_B_3      = (csr_col_ind_k_1 + j_offset_1);
                vec_B_1.d[0] = B[idx_B_2];
                vec_B_1.d[1] = B[idx_B_3];

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of second last column of B matrix  and add to
                // accumulator
                vec_B.v = _mm_mul_pd(vec_A.v, vec_B.v);
                vec_C.v = _mm_add_pd(vec_C.v, vec_B.v);

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of last column of B matrix  and add to
                // accumulator
                vec_B_1.v = _mm_mul_pd(vec_A.v, vec_B_1.v);
                vec_C_1.v = _mm_add_pd(vec_C_1.v, vec_B_1.v);
            }
            //Remainder one non-zero of ith row of A, if nnz in the row is odd
            if(((row_end - row_begin) % 2) == 1)
            {
                //Load last single non-zero of ith row into xmm register
                vec_A.d[0] = csr_val[row_end - 1];

                // Indices of elements of last two columns of
                // B matrix to be multiplied against csr_val[row_end - 1]
                // Load the specific B elements into xmm registers
                idx_B        = (csr_col_ind[row_end - 1] - base + j_offset);
                idx_B_1      = (csr_col_ind[row_end - 1] - base + j_offset_1);
                vec_B.d[0]   = B[idx_B];
                vec_B_1.d[0] = B[idx_B_1];

                // Multiply last single non-zero by corresponding
                // elements of 2 columns of B matrix  and add to
                // accumulator
                vec_B.v   = _mm_mul_sd(vec_A.v, vec_B.v);
                vec_C.v   = _mm_add_sd(vec_C.v, vec_B.v);
                vec_B_1.v = _mm_mul_sd(vec_A.v, vec_B_1.v);
                vec_C_1.v = _mm_add_sd(vec_C_1.v, vec_B_1.v);
            }
            // Horizontal addition of lower and higher double
            // values in accumulator
            sum[0] = vec_C.d[0] + vec_C.d[1];
            sum[1] = vec_C_1.d[0] + vec_C_1.d[1];

            // if beta == 0 ,C= alpha*A*B
            if(*beta == static_cast<double>(0))
            {
                // if beta == 0 & alpha == 1, C= A*B
                if(*alpha == static_cast<double>(1))
                {
                    C[idx_C]   = sum[0];
                    C[idx_C_1] = sum[1];
                }
                // if beta == 0 & alpha != 1, C= alpha*A*B
                else
                {
                    C[idx_C]   = *alpha * sum[0];
                    C[idx_C_1] = *alpha * sum[1];
                }
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
            }
        }
    }
    // if 1 == Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 1)
    {
        aoclsparse_int j = j_iter * 4;
        // Offsets to each of last three columns
        // of B dense matrix in column major format
        j_offset = j * ldb;

        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_C   = i + j * ldc;

            // Accumulator for 4 elements of C matrix
            double sum = static_cast<double>(0);

            // Set Accumulators to zero
            vec_C.v = _mm_setzero_pd();

            const double *csr_val_ptr = &csr_val[row_begin];

            //Iterate over non-zeroes of ith row of A in multiples of 2
            for(aoclsparse_int k = row_begin; k < row_end - 1; k += 2)
            {
                //Load csr_val[k] csr_val[k+1] into xmm register
                vec_A.v = _mm_loadu_pd(csr_val_ptr);
                csr_val_ptr += 2;

                // Column indices of csr_val[k] csr_val[k+1]
                aoclsparse_int csr_col_ind_k   = csr_col_ind[k] - base;
                aoclsparse_int csr_col_ind_k_1 = csr_col_ind[k + 1] - base;

                // Indices of elements of last column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // Load the specific B elements into xmm register
                idx_B      = (csr_col_ind_k + j_offset);
                idx_B_1    = (csr_col_ind_k_1 + j_offset);
                vec_B.d[0] = B[idx_B];
                vec_B.d[1] = B[idx_B_1];

                // Multiply csr_val[k] csr_val[k+1] by corresponding
                // elements of last column of B matrix  and add to
                // accumulator
                vec_B.v = _mm_mul_pd(vec_A.v, vec_B.v);
                vec_C.v = _mm_add_pd(vec_C.v, vec_B.v);
            }
            //Remainder one non-zero of ith row of A, if nnz in the row is odd
            if(((row_end - row_begin) % 2) == 1)
            {
                //Load last single non-zero of ith row into xmm register
                vec_A.d[0] = csr_val[row_end - 1];

                // Index of elements of last columns of
                // B matrix to be multiplied against csr_val[row_end - 1]
                // Load the specific B elements into xmm registers
                idx_B      = (csr_col_ind[row_end - 1] - base + j_offset);
                vec_B.d[0] = B[idx_B];

                // Multiply last single non-zero by corresponding
                // element of last columns of B matrix  and add to
                // accumulator
                vec_B.v = _mm_mul_sd(vec_A.v, vec_B.v);
                vec_C.v = _mm_add_sd(vec_C.v, vec_B.v);
            }
            // Horizontal addition of lower and higher double
            // values in accumulator
            sum = vec_C.d[0] + vec_C.d[1];

            // if beta == 0 ,C= alpha*A*B
            if(*beta == static_cast<double>(0))
            {
                // if beta == 0 & alpha == 1, C= A*B
                if(*alpha == static_cast<double>(1))
                {
                    C[idx_C] = sum;
                }
                // if beta == 0 & alpha != 1, C= alpha*A*B
                else
                {
                    C[idx_C] = *alpha * sum;
                }
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C] = std::fma(*beta, C[idx_C], *alpha * sum);
            }
        }
    }
    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csrmm_col_major(const float               *alpha,
                                             const aoclsparse_mat_descr descr,
                                             const float *__restrict__ csr_val,
                                             const aoclsparse_int *__restrict__ csr_col_ind,
                                             const aoclsparse_int *__restrict__ csr_row_ptr,
                                             aoclsparse_int                  m,
                                             [[maybe_unused]] aoclsparse_int k,
                                             const float                    *B,
                                             aoclsparse_int                  n,
                                             aoclsparse_int                  ldb,
                                             const float                    *beta,
                                             float                          *C,
                                             aoclsparse_int                  ldc)
{
    // Number of sub-blocks of 4 columns in B matrix
    aoclsparse_int j_iter = n / 4;

    // Remainder numbers of column of B after multiple of 4
    aoclsparse_int j_rem = n % 4;

    // Offsets to each of four columns j,j+1,j+2,j+3
    // of B dense matrix in column major format
    aoclsparse_int j_offset   = 0;
    aoclsparse_int j_offset_1 = 0;
    aoclsparse_int j_offset_2 = 0;
    aoclsparse_int j_offset_3 = 0;

    // Indices of elements of input dense matrix B
    aoclsparse_int idx_B   = 0;
    aoclsparse_int idx_B_1 = 0;
    aoclsparse_int idx_B_2 = 0;
    aoclsparse_int idx_B_3 = 0;

    // Indices of output dense matrix C in column major format
    // Four elements of one row of C matrix gets updated in
    // one iteration.
    aoclsparse_int idx_C   = 0;
    aoclsparse_int idx_C_1 = 0;
    aoclsparse_int idx_C_2 = 0;
    aoclsparse_int idx_C_3 = 0;

    aoclsparse_index_base base = descr->base;
    // Iterate along sub-blocks of 4 columns of B matrix
    for(aoclsparse_int j = 0; j < j_iter * 4; j += 4)
    {
        // Offsets to each of four columns j,j+1,j+2,j+3
        // of B dense matrix in column major format
        j_offset   = j * ldb;
        j_offset_1 = (j + 1) * ldb;
        j_offset_2 = (j + 2) * ldb;
        j_offset_3 = (j + 3) * ldb;

        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;

            // Indices of output dense matrix C in column major format
            // Four elements of one row of C matrix gets updated in
            // one iteration.
            idx_C   = i + j * ldc;
            idx_C_1 = i + ((j + 1) * ldc);
            idx_C_2 = i + ((j + 2) * ldc);
            idx_C_3 = i + ((j + 3) * ldc);

            // Accumulator for 4 elements of C matrix
            float sum[4] = {static_cast<float>(0)};

            aoclsparse_int k_iter = (row_end - row_begin) / 4;

            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[0] += csr_val[k + 1] * B[idx_B_1];
                sum[0] += csr_val[k + 2] * B[idx_B_2];
                sum[0] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset_1);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset_1);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset_1);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset_1);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k + 1] * B[idx_B_1];
                sum[1] += csr_val[k + 2] * B[idx_B_2];
                sum[1] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+2)nd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset_2);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset_2);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset_2);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset_2);

                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+2)nd column of B matrix  and
                // add to accumulator
                sum[2] += csr_val[k] * B[idx_B];
                sum[2] += csr_val[k + 1] * B[idx_B_1];
                sum[2] += csr_val[k + 2] * B[idx_B_2];
                sum[2] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+3)rd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset_3);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset_3);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset_3);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset_3);

                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+3)rd column of B matrix  and
                // add to accumulator
                sum[3] += csr_val[k] * B[idx_B];
                sum[3] += csr_val[k + 1] * B[idx_B_1];
                sum[3] += csr_val[k + 2] * B[idx_B_2];
                sum[3] += csr_val[k + 3] * B[idx_B_3];
            }

            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st,(j+2)nd,(j+3)rd
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind[k] - base) + j_offset);
                idx_B_1 = ((csr_col_ind[k] - base) + j_offset_1);
                idx_B_2 = ((csr_col_ind[k] - base) + j_offset_2);
                idx_B_3 = ((csr_col_ind[k] - base) + j_offset_3);

                // Multiply csr_val[k] by corresponding elements of
                // 4 columns of B matrix  and add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k] * B[idx_B_1];
                sum[2] += csr_val[k] * B[idx_B_2];
                sum[3] += csr_val[k] * B[idx_B_3];
            }
            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<float>(0))
            {
                C[idx_C]   = *alpha * sum[0];
                C[idx_C_1] = *alpha * sum[1];
                C[idx_C_2] = *alpha * sum[2];
                C[idx_C_3] = *alpha * sum[3];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
                C[idx_C_2] = std::fma(*beta, C[idx_C_2], *alpha * sum[2]);
                C[idx_C_3] = std::fma(*beta, C[idx_C_3], *alpha * sum[3]);
            }
        }
    }

    // if 3 == Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 3)
    {
        aoclsparse_int j = j_iter * 4;

        // Offsets to each of last three columns
        // of B dense matrix in column major format
        j_offset   = j * ldb;
        j_offset_1 = (j + 1) * ldb;
        j_offset_2 = (j + 2) * ldb;

        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;

            // Indices of output dense matrix C in column major format
            // Three elements of one row of C matrix gets updated in
            // one iteration.
            idx_C   = i + j * ldc;
            idx_C_1 = i + ((j + 1) * ldc);
            idx_C_2 = i + ((j + 2) * ldc);

            // Accumulator for 3 elements of C matrix
            float sum[3] = {static_cast<float>(0)};

            aoclsparse_int k_iter = (row_end - row_begin) / 4;

            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[0] += csr_val[k + 1] * B[idx_B_1];
                sum[0] += csr_val[k + 2] * B[idx_B_2];
                sum[0] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset_1);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset_1);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset_1);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset_1);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k + 1] * B[idx_B_1];
                sum[1] += csr_val[k + 2] * B[idx_B_2];
                sum[1] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+2)nd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset_2);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset_2);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset_2);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset_2);

                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+2)nd column of B matrix  and
                // add to accumulator
                sum[2] += csr_val[k] * B[idx_B];
                sum[2] += csr_val[k + 1] * B[idx_B_1];
                sum[2] += csr_val[k + 2] * B[idx_B_2];
                sum[2] += csr_val[k + 3] * B[idx_B_3];
            }

            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st,(j+2)nd
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind[k] - base) + j_offset);
                idx_B_1 = ((csr_col_ind[k] - base) + j_offset_1);
                idx_B_2 = ((csr_col_ind[k] - base) + j_offset_2);

                // Multiply csr_val[k] by corresponding elements of
                // 3 columns of B matrix  and add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k] * B[idx_B_1];
                sum[2] += csr_val[k] * B[idx_B_2];
            }
            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<float>(0))
            {
                C[idx_C]   = *alpha * sum[0];
                C[idx_C_1] = *alpha * sum[1];
                C[idx_C_2] = *alpha * sum[2];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
                C[idx_C_2] = std::fma(*beta, C[idx_C_2], *alpha * sum[2]);
            }
        }
    }
    // if 2 == Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 2)
    {
        aoclsparse_int j = j_iter * 4;
        // Offsets to each of last two columns
        // of B dense matrix in column major format
        j_offset   = j * ldb;
        j_offset_1 = (j + 1) * ldb;

        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;

            // Indices of output dense matrix C in column major format
            // Two elements of one row of C matrix gets updated in
            // one iteration.
            idx_C   = i + j * ldc;
            idx_C_1 = i + ((j + 1) * ldc);

            // Accumulator for 2 elements of C matrix
            float sum[2] = {static_cast<float>(0)};

            aoclsparse_int k_iter = (row_end - row_begin) / 4;

            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[0] += csr_val[k + 1] * B[idx_B_1];
                sum[0] += csr_val[k + 2] * B[idx_B_2];
                sum[0] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset_1);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset_1);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset_1);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset_1);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k + 1] * B[idx_B_1];
                sum[1] += csr_val[k + 2] * B[idx_B_2];
                sum[1] += csr_val[k + 3] * B[idx_B_3];
            }

            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind[k] - base) + j_offset);
                idx_B_1 = ((csr_col_ind[k] - base) + j_offset_1);

                // Multiply csr_val[k] by corresponding elements of
                // 2 columns of B matrix  and add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k] * B[idx_B_1];
            }
            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<float>(0))
            {
                C[idx_C]   = *alpha * sum[0];
                C[idx_C_1] = *alpha * sum[1];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
            }
        }
    }
    // if 1 == Remainder columns of B after subblocks of multiple of 4
    if(j_rem == 1)
    {
        aoclsparse_int j = j_iter * 4;
        // Offsets to each of last column
        // of B dense matrix in column major format
        j_offset = j * ldb;

        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

            // Indices of elements of input dense matrix B
            idx_B   = 0;
            idx_B_1 = 0;
            idx_B_2 = 0;
            idx_B_3 = 0;

            // Index of output dense matrix C in column major format
            // ONe element of one row of C matrix gets updated in
            // one iteration.
            idx_C = i + j * ldc;

            // Accumulator for 1 element of C matrix
            float sum = static_cast<float>(0);

            aoclsparse_int k_iter = (row_end - row_begin) / 4;

            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) + j_offset);
                idx_B_1 = ((csr_col_ind[k + 1] - base) + j_offset);
                idx_B_2 = ((csr_col_ind[k + 2] - base) + j_offset);
                idx_B_3 = ((csr_col_ind[k + 3] - base) + j_offset);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum += csr_val[k] * B[idx_B];
                sum += csr_val[k + 1] * B[idx_B_1];
                sum += csr_val[k + 2] * B[idx_B_2];
                sum += csr_val[k + 3] * B[idx_B_3];
            }

            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of element of jth column of B matrix
                // to be multiplied against csr_val[k]
                idx_B = ((csr_col_ind[k] - base) + j_offset);

                // Multiply csr_val[k] by corresponding elements of
                // 1 columns of B matrix  and add to accumulator
                sum += csr_val[k] * B[idx_B];
            }
            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<float>(0))
            {
                C[idx_C] = *alpha * sum;
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C] = std::fma(*beta, C[idx_C], *alpha * sum);
            }
        }
    }
    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csrmm_row_major(const double              *alpha,
                                             const aoclsparse_mat_descr descr,
                                             const double *__restrict__ csr_val,
                                             const aoclsparse_int *__restrict__ csr_col_ind,
                                             const aoclsparse_int *__restrict__ csr_row_ptr,
                                             aoclsparse_int                  m,
                                             [[maybe_unused]] aoclsparse_int k,
                                             const double                   *B,
                                             aoclsparse_int                  n,
                                             aoclsparse_int                  ldb,
                                             const double                   *beta,
                                             double                         *C,
                                             aoclsparse_int                  ldc)
{
    aoclsparse_index_base base = descr->base;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        for(aoclsparse_int j = 0; j < n; ++j)
        {
            double         row_begin = csr_row_ptr[i] - base;
            double         row_end   = csr_row_ptr[i + 1] - base;
            aoclsparse_int idx_C     = i * ldc + j;

            double sum = static_cast<double>(0);

            for(aoclsparse_int k = row_begin; k < row_end; ++k)
            {
                aoclsparse_int idx_B = 0;
                idx_B                = (j + (csr_col_ind[k] - base) * ldb);

                sum = std::fma(csr_val[k], B[idx_B], sum);
            }

            if(*beta == static_cast<double>(0))
            {
                C[idx_C] = *alpha * sum;
            }
            else
            {
                C[idx_C] = std::fma(*beta, C[idx_C], *alpha * sum);
            }
        }
    }
    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csrmm_row_major(const float               *alpha,
                                             const aoclsparse_mat_descr descr,
                                             const float *__restrict__ csr_val,
                                             const aoclsparse_int *__restrict__ csr_col_ind,
                                             const aoclsparse_int *__restrict__ csr_row_ptr,
                                             aoclsparse_int                  m,
                                             [[maybe_unused]] aoclsparse_int k,
                                             const float                    *B,
                                             aoclsparse_int                  n,
                                             aoclsparse_int                  ldb,
                                             const float                    *beta,
                                             float                          *C,
                                             aoclsparse_int                  ldc)
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
    aoclsparse_int idx_C   = 0;
    aoclsparse_int idx_C_1 = 0;
    aoclsparse_int idx_C_2 = 0;
    aoclsparse_int idx_C_3 = 0;

    aoclsparse_index_base base = descr->base;

    // Iterate along sub-blocks of 4 columns of B matrix
    for(aoclsparse_int j = 0; j < j_iter * 4; j += 4)
    {
        // Iterate along rows of sparse matrix A
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            // Pointer to the first and last nonzero
            // of ith row
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

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
            float sum[4] = {static_cast<float>(0)};

            aoclsparse_int k_iter = (row_end - row_begin) / 4;

            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[0] += csr_val[k + 1] * B[idx_B_1];
                sum[0] += csr_val[k + 2] * B[idx_B_2];
                sum[0] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j + 1);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j + 1);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j + 1);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k + 1] * B[idx_B_1];
                sum[1] += csr_val[k + 2] * B[idx_B_2];
                sum[1] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+2)nd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j + 2);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j + 2);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j + 2);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j + 2);

                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+2)nd column of B matrix  and
                // add to accumulator
                sum[2] += csr_val[k] * B[idx_B];
                sum[2] += csr_val[k + 1] * B[idx_B_1];
                sum[2] += csr_val[k + 2] * B[idx_B_2];
                sum[2] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+3)rd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j + 3);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j + 3);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j + 3);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j + 3);

                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+3)rd column of B matrix  and
                // add to accumulator
                sum[3] += csr_val[k] * B[idx_B];
                sum[3] += csr_val[k + 1] * B[idx_B_1];
                sum[3] += csr_val[k + 2] * B[idx_B_2];
                sum[3] += csr_val[k + 3] * B[idx_B_3];
            }

            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st,(j+2)nd,(j+3)rd
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j);
                idx_B_1 = ((csr_col_ind[k] - base) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind[k] - base) * ldb + j + 2);
                idx_B_3 = ((csr_col_ind[k] - base) * ldb + j + 3);

                // Multiply csr_val[k] by corresponding elements of
                // 4 columns of B matrix  and add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k] * B[idx_B_1];
                sum[2] += csr_val[k] * B[idx_B_2];
                sum[3] += csr_val[k] * B[idx_B_3];
            }
            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<float>(0))
            {
                C[idx_C]   = *alpha * sum[0];
                C[idx_C_1] = *alpha * sum[1];
                C[idx_C_2] = *alpha * sum[2];
                C[idx_C_3] = *alpha * sum[3];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
                C[idx_C_2] = std::fma(*beta, C[idx_C_2], *alpha * sum[2]);
                C[idx_C_3] = std::fma(*beta, C[idx_C_3], *alpha * sum[3]);
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
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

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
            float sum[3] = {static_cast<float>(0)};

            aoclsparse_int k_iter = (row_end - row_begin) / 4;

            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[0] += csr_val[k + 1] * B[idx_B_1];
                sum[0] += csr_val[k + 2] * B[idx_B_2];
                sum[0] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j + 1);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j + 1);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j + 1);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k + 1] * B[idx_B_1];
                sum[1] += csr_val[k + 2] * B[idx_B_2];
                sum[1] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+2)nd column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j + 2);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j + 2);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j + 2);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j + 2);

                // Multiply csr_val[k] csr_val[k+1]icsr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+2)nd column of B matrix  and
                // add to accumulator
                sum[2] += csr_val[k] * B[idx_B];
                sum[2] += csr_val[k + 1] * B[idx_B_1];
                sum[2] += csr_val[k + 2] * B[idx_B_2];
                sum[2] += csr_val[k + 3] * B[idx_B_3];
            }

            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st,(j+2)nd
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j);
                idx_B_1 = ((csr_col_ind[k] - base) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind[k] - base) * ldb + j + 2);

                // Multiply csr_val[k] by corresponding elements of
                // 3 columns of B matrix  and add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k] * B[idx_B_1];
                sum[2] += csr_val[k] * B[idx_B_2];
            }
            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<float>(0))
            {
                C[idx_C]   = *alpha * sum[0];
                C[idx_C_1] = *alpha * sum[1];
                C[idx_C_2] = *alpha * sum[2];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
                C[idx_C_2] = std::fma(*beta, C[idx_C_2], *alpha * sum[2]);
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
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

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
            float sum[2] = {static_cast<float>(0)};

            aoclsparse_int k_iter = (row_end - row_begin) / 4;

            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[0] += csr_val[k + 1] * B[idx_B_1];
                sum[0] += csr_val[k + 2] * B[idx_B_2];
                sum[0] += csr_val[k + 3] * B[idx_B_3];

                // Indices of elements of (j+1)st column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j + 1);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j + 1);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j + 1);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j + 1);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of (j+1)st column of B matrix  and
                // add to accumulator
                sum[1] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k + 1] * B[idx_B_1];
                sum[1] += csr_val[k + 2] * B[idx_B_2];
                sum[1] += csr_val[k + 3] * B[idx_B_3];
            }

            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of elements of jth,(j+1)st
                // columns of B matrix to be multiplied against
                // csr_val[k]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j);
                idx_B_1 = ((csr_col_ind[k] - base) * ldb + j + 1);

                // Multiply csr_val[k] by corresponding elements of
                // 2 columns of B matrix  and add to accumulator
                sum[0] += csr_val[k] * B[idx_B];
                sum[1] += csr_val[k] * B[idx_B_1];
            }
            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<float>(0))
            {
                C[idx_C]   = *alpha * sum[0];
                C[idx_C_1] = *alpha * sum[1];
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C]   = std::fma(*beta, C[idx_C], *alpha * sum[0]);
                C[idx_C_1] = std::fma(*beta, C[idx_C_1], *alpha * sum[1]);
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
            aoclsparse_int row_begin = csr_row_ptr[i] - base;
            aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

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
            float sum = static_cast<float>(0);

            aoclsparse_int k_iter = (row_end - row_begin) / 4;

            //Iterate over non-zeroes of ith row of A in multiples of 4
            for(aoclsparse_int k = row_begin; k < (row_begin + k_iter * 4); k += 4)
            {
                // Indices of elements of jth column of B matrix
                // to be multiplied against csr_val[k] csr_val[k+1]
                // csr_val[k+2] csr_val[k+3]
                idx_B   = ((csr_col_ind[k] - base) * ldb + j);
                idx_B_1 = ((csr_col_ind[k + 1] - base) * ldb + j);
                idx_B_2 = ((csr_col_ind[k + 2] - base) * ldb + j);
                idx_B_3 = ((csr_col_ind[k + 3] - base) * ldb + j);

                // Multiply csr_val[k] csr_val[k+1] csr_val[k+2] csr_val[k+3]
                // by corresponding elements of jth column of B matrix  and
                // add to accumulator
                sum += csr_val[k] * B[idx_B];
                sum += csr_val[k + 1] * B[idx_B_1];
                sum += csr_val[k + 2] * B[idx_B_2];
                sum += csr_val[k + 3] * B[idx_B_3];
            }

            // Remainder (3/2/1) non-zero of ith row of A,
            // if nnz in the row is not a multiple of 4
            for(aoclsparse_int k = (row_begin + k_iter * 4); k < row_end; k++)
            {
                // Indices of element of jth column of B matrix
                // to be multiplied against csr_val[k]
                idx_B = ((csr_col_ind[k] - base) * ldb + j);

                // Multiply csr_val[k] by corresponding elements of
                // 1 columns of B matrix  and add to accumulator
                sum += csr_val[k] * B[idx_B];
            }
            // if beta = 0 , C= alpha*A*B
            if(*beta == static_cast<float>(0))
            {
                C[idx_C] = *alpha * sum;
            }
            // if beta != 0 & alpha != 1, C= beta*C + alpha*A*B
            else
            {
                C[idx_C] = std::fma(*beta, C[idx_C], *alpha * sum);
            }
        }
    }
    return aoclsparse_status_success;
}
#endif /* AOCLSPARSE_CSRMM_HPP*/
