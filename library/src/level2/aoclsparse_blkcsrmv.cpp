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

#include "aoclsparse.h"
#include "aoclsparse_blkcsrmv.hpp"
#include <immintrin.h>

#ifdef _MSC_VER_
#include <intrin.h>
#endif

#if USE_AVX512
int bits_set(uint8_t x)
{
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
    return __popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}

template <>
aoclsparse_status
    aoclsparse_blkcsrmv_1x8_vectorized_avx512(const double   alpha,
                                            aoclsparse_int m,
                                            aoclsparse_int n,
                                            aoclsparse_int nnz,
                                            const uint8_t *__restrict__ masks,
                                            const double *__restrict__ blk_csr_val,
                                            const aoclsparse_int *__restrict__ blk_col_ind,
                                            const aoclsparse_int *__restrict__ blk_row_ptr,
                                            const double *__restrict__ x,
                                            const double beta,
                                            double *__restrict__ y,
                                            aoclsparse_context *context)
{
    const int nRowsblk = 1;
    __m512d vec_vals_512, vec_x_512, vec_y_512, zero_msk;
    __m256d vec_y;
    zero_msk = _mm512_setzero_pd();
    int iVal = 0;

    for(int iRow = 0; iRow < m; iRow += nRowsblk)
    {
        double sum = 0;
        vec_y_512  = _mm512_setzero_pd();

        for(int iBlk = blk_row_ptr[iRow]; iBlk < blk_row_ptr[iRow + 1]; iBlk += 1)
        {
            //Load the column vector only once for the entire block
            const int iCol = blk_col_ind[iBlk];
            vec_x_512      = _mm512_loadu_pd((double const *)&x[iCol]);

            //Read mask value and perform mask_expandload for each row in the block
            //Increment iVal to number of nnz loaded, to move pointer in value array for next row
            const uint8_t msk = masks[iBlk];
            vec_vals_512
                = _mm512_mask_expandloadu_pd(zero_msk, msk, (double const *)&blk_csr_val[iVal]);
            iVal += bits_set(msk);

            //Perform fma on loaded vectors
            vec_y_512 = _mm512_fmadd_pd(vec_vals_512, vec_x_512, vec_y_512);
        }

        //Perform horizontal addition
        vec_y = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512, 0x0),
                              _mm512_extractf64x4_pd(vec_y_512, 0x1));
        vec_y = _mm256_hadd_pd(vec_y, vec_y);
        // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
        __m128d sum_lo = _mm256_castpd256_pd128(vec_y);
        // Extract 128 bits to obtain sum[2] and sum[3]
        __m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);

        // Add remaining two sums
        __m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
        sum += sse_sum.m128d_f64[0];
#else
        sum += sse_sum[0];
#endif
        y[iRow] = sum;
    }

    return aoclsparse_status_success;
}

template <>
aoclsparse_status
    aoclsparse_blkcsrmv_2x8_vectorized_avx512(const double   alpha,
                                            aoclsparse_int m,
                                            aoclsparse_int n,
                                            aoclsparse_int nnz,
                                            const uint8_t *__restrict__ masks,
                                            const double *__restrict__ blk_csr_val,
                                            const aoclsparse_int *__restrict__ blk_col_ind,
                                            const aoclsparse_int *__restrict__ blk_row_ptr,
                                            const double *__restrict__ x,
                                            const double beta,
                                            double *__restrict__ y,
                                            aoclsparse_context *context)
{
    const int nRowsblk = 2;
    __m512d vec_vals_512[nRowsblk], vec_y_512[nRowsblk], zero_msk, vec_x_512;
    __m256d vec_y;
    zero_msk = _mm512_setzero_pd();
    int iVal = 0;

    for(int iRow = 0; iRow < m; iRow += nRowsblk)
    {

        double sum[nRowsblk] = {0, 0};

        vec_y_512[0] = _mm512_setzero_pd();
        vec_y_512[1] = _mm512_setzero_pd();

        for(int iBlk = blk_row_ptr[iRow]; iBlk < blk_row_ptr[iRow + 1]; iBlk += 1)
        {
            //Load the column vector only once for the entire block
            const int iCol = blk_col_ind[iBlk];
            vec_x_512      = _mm512_loadu_pd((double const *)&x[iCol]);

            //Read mask value and perform mask_expandload for each row in the block
            //Increment iVal to number of nnz loaded, to move pointer in value array for next row
            uint8_t msk = masks[iBlk * nRowsblk];
            vec_vals_512[0]
                = _mm512_mask_expandloadu_pd(zero_msk, msk, (double const *)&blk_csr_val[iVal]);
            iVal += bits_set(msk);

            msk = masks[iBlk * nRowsblk + 1];
            vec_vals_512[1]
                = _mm512_mask_expandloadu_pd(zero_msk, msk, (double const *)&blk_csr_val[iVal]);
            iVal += bits_set(msk);

            //Perform fma on loaded vectors
            vec_y_512[0] = _mm512_fmadd_pd(vec_vals_512[0], vec_x_512, vec_y_512[0]);
            vec_y_512[1] = _mm512_fmadd_pd(vec_vals_512[1], vec_x_512, vec_y_512[1]);
        }

        //Doing horizontal sum for 1st row
        vec_y = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512[0], 0x0),
                              _mm512_extractf64x4_pd(vec_y_512[0], 0x1));
        vec_y = _mm256_hadd_pd(vec_y, vec_y);
        // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
        __m128d sum_lo = _mm256_castpd256_pd128(vec_y);
        // Extract 128 bits to obtain sum[2] and sum[3]
        __m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);

        // Add remaining two sums
        __m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
        sum[0] += sse_sum.m128d_f64[0];
#else
        sum[0] += sse_sum[0];
#endif
        y[iRow] = sum[0];

        //Doing horizontal sum for 2nd row (if present)
        if(iRow + 1 == m)
            break;

        vec_y   = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512[1], 0x0),
                              _mm512_extractf64x4_pd(vec_y_512[1], 0x1));
        vec_y   = _mm256_hadd_pd(vec_y, vec_y);
        sum_lo  = _mm256_castpd256_pd128(vec_y);
        sum_hi  = _mm256_extractf128_pd(vec_y, 1);
        sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
        sum[1] += sse_sum.m128d_f64[0];
#else
        sum[1] += sse_sum[0];
#endif
        y[iRow + 1] = sum[1];
    }

    return aoclsparse_status_success;
}

template <>
aoclsparse_status
    aoclsparse_blkcsrmv_4x8_vectorized_avx512(const double   alpha,
                                            aoclsparse_int m,
                                            aoclsparse_int n,
                                            aoclsparse_int nnz,
                                            const uint8_t *__restrict__ masks,
                                            const double *__restrict__ blk_csr_val,
                                            const aoclsparse_int *__restrict__ blk_col_ind,
                                            const aoclsparse_int *__restrict__ blk_row_ptr,
                                            const double *__restrict__ x,
                                            const double beta,
                                            double *__restrict__ y,
                                            aoclsparse_context *context)
{
    const int nRowsblk = 4;
    __m512d vec_vals_512[nRowsblk], vec_y_512[nRowsblk], zero_msk, vec_x_512;
    __m256d vec_y;
    zero_msk = _mm512_setzero_pd();
    int iVal = 0;

    for(int iRow = 0; iRow < m; iRow += nRowsblk)
    {
        double sum[nRowsblk] = {0, 0, 0, 0};
        vec_y_512[0]         = _mm512_setzero_pd();
        vec_y_512[1]         = _mm512_setzero_pd();
        vec_y_512[2]         = _mm512_setzero_pd();
        vec_y_512[3]         = _mm512_setzero_pd();

        for(int iBlk = blk_row_ptr[iRow]; iBlk < blk_row_ptr[iRow + 1]; iBlk += 1)
        {
            //Load the column vector only once for the entire block
            const int iCol = blk_col_ind[iBlk];
            vec_x_512      = _mm512_loadu_pd((double const *)&x[iCol]);

            //Read mask value and perform mask_expandload for each row in the block
            //Increment iVal to number of nnz loaded, to move pointer in value array for next row
            uint8_t msk = masks[iBlk * nRowsblk];
            vec_vals_512[0]
                = _mm512_mask_expandloadu_pd(zero_msk, msk, (double const *)&blk_csr_val[iVal]);
            iVal += bits_set(msk);

            msk = masks[iBlk * nRowsblk + 1];
            vec_vals_512[1]
                = _mm512_mask_expandloadu_pd(zero_msk, msk, (double const *)&blk_csr_val[iVal]);
            iVal += bits_set(msk);

            msk = masks[iBlk * nRowsblk + 2];
            vec_vals_512[2]
                = _mm512_mask_expandloadu_pd(zero_msk, msk, (double const *)&blk_csr_val[iVal]);
            iVal += bits_set(msk);

            msk = masks[iBlk * nRowsblk + 3];
            vec_vals_512[3]
                = _mm512_mask_expandloadu_pd(zero_msk, msk, (double const *)&blk_csr_val[iVal]);
            iVal += bits_set(msk);

            //Perform fma on loaded vectors
            vec_y_512[0] = _mm512_fmadd_pd(vec_vals_512[0], vec_x_512, vec_y_512[0]);
            vec_y_512[1] = _mm512_fmadd_pd(vec_vals_512[1], vec_x_512, vec_y_512[1]);
            vec_y_512[2] = _mm512_fmadd_pd(vec_vals_512[2], vec_x_512, vec_y_512[2]);
            vec_y_512[3] = _mm512_fmadd_pd(vec_vals_512[3], vec_x_512, vec_y_512[3]);
        }

        //Doing horizontal sum for 1st row
        vec_y = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512[0], 0x0),
                              _mm512_extractf64x4_pd(vec_y_512[0], 0x1));
        vec_y = _mm256_hadd_pd(vec_y, vec_y);
        // Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
        __m128d sum_lo = _mm256_castpd256_pd128(vec_y);
        // Extract 128 bits to obtain sum[2] and sum[3]
        __m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);

        // Add remaining two sums
        __m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
        sum[0] += sse_sum.m128d_f64[0];
#else
        sum[0] += sse_sum[0];
#endif
        y[iRow] = sum[0];

        //Doing horizontal sum for 2nd row (if present)
        if(iRow + 1 == m)
            break;

        vec_y   = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512[1], 0x0),
                              _mm512_extractf64x4_pd(vec_y_512[1], 0x1));
        vec_y   = _mm256_hadd_pd(vec_y, vec_y);
        sum_lo  = _mm256_castpd256_pd128(vec_y);
        sum_hi  = _mm256_extractf128_pd(vec_y, 1);
        sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
        sum[1] += sse_sum.m128d_f64[0];
#else
        sum[1] += sse_sum[0];
#endif
        y[iRow + 1] = sum[1];

        //Doing horizontal sum for 3rd row (if present)
        if(iRow + 2 == m)
            break;

        vec_y   = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512[2], 0x0),
                              _mm512_extractf64x4_pd(vec_y_512[2], 0x1));
        vec_y   = _mm256_hadd_pd(vec_y, vec_y);
        sum_lo  = _mm256_castpd256_pd128(vec_y);
        sum_hi  = _mm256_extractf128_pd(vec_y, 1);
        sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
        sum[2] += sse_sum.m128d_f64[0];
#else
        sum[2] += sse_sum[0];
#endif
        y[iRow + 2] = sum[2];

        //Doing horizontal sum for 4th row (if present)
        if(iRow + 3 == m)
            break;

        vec_y   = _mm256_add_pd(_mm512_extractf64x4_pd(vec_y_512[3], 0x0),
                              _mm512_extractf64x4_pd(vec_y_512[3], 0x1));
        vec_y   = _mm256_hadd_pd(vec_y, vec_y);
        sum_lo  = _mm256_castpd256_pd128(vec_y);
        sum_hi  = _mm256_extractf128_pd(vec_y, 1);
        sse_sum = _mm_add_pd(sum_lo, sum_hi);
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
        sum[3] += sse_sum.m128d_f64[0];
#else
        sum[3] += sse_sum[0];
#endif
        y[iRow + 3] = sum[3];
    }

    return aoclsparse_status_success;
}


//This routine performs sparse-matrix multiplication on matrices stored in blocked CSR format.
//Supports blocking factors of size 1x8, 2x8 and 4x8. Blocking size is chosen depending on the matrix characteristics.
//We currently support blocked SpMV for only single threaded usecases.
extern "C" aoclsparse_status aoclsparse_dblkcsrmv(aoclsparse_operation       trans,
                                                const double *             alpha,
                                                aoclsparse_int             m,
                                                aoclsparse_int             n,
                                                aoclsparse_int             nnz,
                                                const uint8_t *            masks,
                                                const double *             blk_csr_val,
                                                const aoclsparse_int *     blk_col_ind,
                                                const aoclsparse_int *     blk_row_ptr,
                                                const aoclsparse_mat_descr descr,
                                                const double *             x,
                                                const double *             beta,
                                                double *                   y,
                                                aoclsparse_int             nRowsblk)
{

    // Read the environment variables to update global variable
    // This function updates the num_threads only once.
    aoclsparse_init_once();
    aoclsparse_context context;
    context.num_threads = global_context.num_threads;
    context.is_avx512   = global_context.is_avx512;
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    // Support General and symmetric matrices.
    // Return for any other matrix type
    if((descr->type != aoclsparse_matrix_type_general)
       && (descr->type != aoclsparse_matrix_type_symmetric))
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    if(trans != aoclsparse_operation_none)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(blk_csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(blk_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(blk_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(x == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(y == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(context.is_avx512)
    {
        if(nRowsblk == 1)
            return aoclsparse_blkcsrmv_1x8_vectorized_avx512(
                *alpha, m, n, nnz, masks, blk_csr_val, blk_col_ind, blk_row_ptr, x, *beta, y, &context);
        if(nRowsblk == 2)
            return aoclsparse_blkcsrmv_2x8_vectorized_avx512(
                *alpha, m, n, nnz, masks, blk_csr_val, blk_col_ind, blk_row_ptr, x, *beta, y, &context);
        if(nRowsblk == 4)
            return aoclsparse_blkcsrmv_4x8_vectorized_avx512(
                *alpha, m, n, nnz, masks, blk_csr_val, blk_col_ind, blk_row_ptr, x, *beta, y, &context);
	else return aoclsparse_status_invalid_size;
    }
}
#endif
