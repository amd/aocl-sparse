/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <iostream>

aoclsparse_status aoclsparse_csrmm_col_major_vec_mnk(const float*            alpha,
	const float* __restrict__          csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int                     m,
	aoclsparse_int                     k,
	const float*                       B,
	aoclsparse_int                     n,
	aoclsparse_int                     ldb,
	const float*                       beta,
	float*                             C,
	aoclsparse_int                     ldc)
{
    const aoclsparse_int *colIndPtr;
    const float *matValPtr;
    aoclsparse_int idx_B = 0, idx_B_1 = 0, idx_B_2 = 0, idx_B_3 = 0;
    aoclsparse_int idx_B_4 = 0, idx_B_5 = 0, idx_B_6 = 0, idx_B_7 = 0;

    for(aoclsparse_int i = 0; i < m; ++i)
    {
	for(aoclsparse_int j = 0; j < n; ++j)
	{
	    __m256 vec_vals , vec_x , vec_y;
	    matValPtr = &csr_val[csr_row_ptr[i]];
	    colIndPtr = &csr_col_ind[csr_row_ptr[i]];
	    aoclsparse_int idx_C     =  i + j * ldc;

	    float sum = 0.0;
	    vec_y = _mm256_setzero_ps();
	    aoclsparse_int l;
	    aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];
	    aoclsparse_int k_iter = nnz/8;
	    aoclsparse_int k_rem = nnz%8;

	    //Loop in multiples of 4 non-zeroes
	    for(l =  0 ; l < k_iter ; l++ )
	    {
		//(csr_val[j] csr_val[j+1] csr_val[j+2] csr_val[j+3] csr_val[j+4] csr_val[j+5] csr_val[j+6] csr_val[j+7]
		vec_vals = _mm256_loadu_ps(matValPtr);

		idx_B = (colIndPtr[0] + j * ldb);
		idx_B_1 = (colIndPtr[1] + j * ldb);
		idx_B_2 = (colIndPtr[2] + j * ldb);
		idx_B_3 = (colIndPtr[3] + j * ldb);
		idx_B_4 = (colIndPtr[4] + j * ldb);
		idx_B_5 = (colIndPtr[5] + j * ldb);
		idx_B_6 = (colIndPtr[6] + j * ldb);
		idx_B_7 = (colIndPtr[7] + j * ldb);
		//Gather the x vector elements from the column indices
		vec_x  = _mm256_set_ps(B[idx_B_7],
			B[idx_B_6],
			B[idx_B_5],
			B[idx_B_4],
			B[idx_B_3],
			B[idx_B_2],
			B[idx_B_1],
			B[idx_B]);

		vec_y = _mm256_fmadd_ps(vec_vals, vec_x , vec_y);

		matValPtr+=8;
		colIndPtr+=8;
	    }

	    // Horizontal addition
	    if(k_iter){
		// hiQuad = ( x7, x6, x5, x4 )
		__m128 hiQuad = _mm256_extractf128_ps(vec_y, 1);
		// loQuad = ( x3, x2, x1, x0 )
		const __m128 loQuad = _mm256_castps256_ps128(vec_y);
		// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
		const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x1 + x5, x0 + x4 )
		const __m128 loDual = sumQuad;
		// hiDual = ( -, -, x3 + x7, x2 + x6 )
		const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
		const __m128 sumDual = _mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 + x4 + x6 )
		const __m128 lo = sumDual;
		// hi = ( -, -, -, x1 + x3 + x5 + x7 )
		const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
		const __m128 sse_sum = _mm_add_ss(lo, hi);
		sum = _mm_cvtss_f32(sse_sum);

	    }
	    //Remainder loop for nnz%4
	    for(l = 0 ; l < k_rem ; l++ )
	    {
		idx_B = (colIndPtr[l] + j * ldb);
		sum += *matValPtr++ * B[idx_B];
	    }


	    if(*beta == static_cast<float>(0))
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

aoclsparse_status aoclsparse_csrmm_col_major_vec_nmk(const float*            alpha,
	const float* __restrict__          csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int                     m,
	aoclsparse_int                     k,
	const float*                       B,
	aoclsparse_int                     n,
	aoclsparse_int                     ldb,
	const float*                       beta,
	float*                             C,
	aoclsparse_int                     ldc)
{
    const aoclsparse_int *colIndPtr;
    const float *matValPtr;
    aoclsparse_int idx_B = 0, idx_B_1 = 0, idx_B_2 = 0, idx_B_3 = 0;
    aoclsparse_int idx_B_4 = 0, idx_B_5 = 0, idx_B_6 = 0, idx_B_7 = 0;

    for(aoclsparse_int j = 0; j < n; ++j)
    {
	for(aoclsparse_int i = 0; i < m; ++i)
	{
	    __m256 vec_vals , vec_x , vec_y;
	    matValPtr = &csr_val[csr_row_ptr[i]];
	    colIndPtr = &csr_col_ind[csr_row_ptr[i]];
	    aoclsparse_int idx_C     =  i + j * ldc;

	    float sum = 0.0;
	    vec_y = _mm256_setzero_ps();
	    aoclsparse_int l;
	    aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];
	    aoclsparse_int k_iter = nnz/8;
	    aoclsparse_int k_rem = nnz%8;

	    //Loop in multiples of 4 non-zeroes
	    for(l =  0 ; l < k_iter ; l++ )
	    {
		//(csr_val[j] csr_val[j+1] csr_val[j+2] csr_val[j+3] csr_val[j+4] csr_val[j+5] csr_val[j+6] csr_val[j+7]
		vec_vals = _mm256_loadu_ps(matValPtr);

		idx_B = (colIndPtr[0] + j * ldb);
		idx_B_1 = (colIndPtr[1] + j * ldb);
		idx_B_2 = (colIndPtr[2] + j * ldb);
		idx_B_3 = (colIndPtr[3] + j * ldb);
		idx_B_4 = (colIndPtr[4] + j * ldb);
		idx_B_5 = (colIndPtr[5] + j * ldb);
		idx_B_6 = (colIndPtr[6] + j * ldb);
		idx_B_7 = (colIndPtr[7] + j * ldb);
		//Gather the x vector elements from the column indices
		vec_x  = _mm256_set_ps(B[idx_B_7],
			B[idx_B_6],
			B[idx_B_5],
			B[idx_B_4],
			B[idx_B_3],
			B[idx_B_2],
			B[idx_B_1],
			B[idx_B]);

		vec_y = _mm256_fmadd_ps(vec_vals, vec_x , vec_y);

		matValPtr+=8;
		colIndPtr+=8;
	    }

	    // Horizontal addition
	    if(k_iter){
		// hiQuad = ( x7, x6, x5, x4 )
		__m128 hiQuad = _mm256_extractf128_ps(vec_y, 1);
		// loQuad = ( x3, x2, x1, x0 )
		const __m128 loQuad = _mm256_castps256_ps128(vec_y);
		// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
		const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x1 + x5, x0 + x4 )
		const __m128 loDual = sumQuad;
		// hiDual = ( -, -, x3 + x7, x2 + x6 )
		const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
		const __m128 sumDual = _mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 + x4 + x6 )
		const __m128 lo = sumDual;
		// hi = ( -, -, -, x1 + x3 + x5 + x7 )
		const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
		const __m128 sse_sum = _mm_add_ss(lo, hi);
		sum = _mm_cvtss_f32(sse_sum);

	    }
	    //Remainder loop for nnz%4
	    for(l = 0 ; l < k_rem ; l++ )
	    {
		idx_B = (colIndPtr[l] + j * ldb);
		sum += *matValPtr++ * B[idx_B];
	    }


	    if(*beta == static_cast<float>(0))
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
aoclsparse_status aoclsparse_csrmm_col_major_vec_mnk(const double*            alpha,
	const double* __restrict__         csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int                     m,
	aoclsparse_int                     k,
	const double*                      B,
	aoclsparse_int                     n,
	aoclsparse_int                     ldb,
	const double*                      beta,
	double*                            C,
	aoclsparse_int                     ldc)
{
    const aoclsparse_int *colIndPtr;
    const double *matValPtr;
    aoclsparse_int idx_B = 0, idx_B_1 = 0, idx_B_2 = 0, idx_B_3 = 0;

    for(aoclsparse_int i = 0; i < m; ++i)
    {
	for(aoclsparse_int j = 0; j < n; ++j)
	{
	    __m256d vec_vals , vec_x , vec_y;
	    matValPtr = &csr_val[csr_row_ptr[i]];
	    colIndPtr = &csr_col_ind[csr_row_ptr[i]];
	    aoclsparse_int idx_C     =  i + j * ldc;

	    double sum = 0.0;
	    vec_y = _mm256_setzero_pd();
	    aoclsparse_int l;
	    aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];
	    aoclsparse_int k_iter = nnz/4;
	    aoclsparse_int k_rem = nnz%4;

	    //Loop in multiples of 4 non-zeroes
	    for(l =  0 ; l < k_iter ; l++ )
	    {
		//(csr_val[j] (csr_val[j+1] (csr_val[j+2] (csr_val[j+3]
		vec_vals = _mm256_loadu_pd((double const *)matValPtr);

		idx_B = (colIndPtr[0] + j * ldb);
		idx_B_1 = (colIndPtr[1] + j * ldb);
		idx_B_2 = (colIndPtr[2] + j * ldb);
		idx_B_3 = (colIndPtr[3] + j * ldb);
		//Gather the x vector elements from the column indices
		vec_x  = _mm256_set_pd(B[idx_B_3],
			B[idx_B_2],
			B[idx_B_1],
			B[idx_B]);

		vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);
		matValPtr+=4;
		colIndPtr+=4;
	    }

	    // Horizontal addition
	    if(k_iter){
		// sum[0] += sum[1] ; sum[2] += sum[3]
		vec_y = _mm256_hadd_pd(vec_y, vec_y);
		// Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
		__m128d sum_lo = _mm256_castpd256_pd128(vec_y);
		// Extract 128 bits to obtain sum[2] and sum[3]
		__m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);
		// Add remaining two sums
		__m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
		// Store result
		sum = sse_sum[0];
	    }
	    //Remainder loop for nnz%4
	    for(l = 0 ; l < k_rem ; l++ )
	    {
		idx_B = (colIndPtr[l] + j * ldb);
		sum += *matValPtr++ * B[idx_B];
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

aoclsparse_status aoclsparse_csrmm_col_major_vec_nmk(const double*            alpha,
	const double* __restrict__         csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int                     m,
	aoclsparse_int                     k,
	const double*                      B,
	aoclsparse_int                     n,
	aoclsparse_int                     ldb,
	const double*                      beta,
	double*                            C,
	aoclsparse_int                     ldc)
{
    const aoclsparse_int *colIndPtr;
    const double *matValPtr;
    aoclsparse_int idx_B = 0, idx_B_1 = 0, idx_B_2 = 0, idx_B_3 = 0;

    for(aoclsparse_int j = 0; j < n; ++j)
    {
	for(aoclsparse_int i = 0; i < m; ++i)
	{
	    __m256d vec_vals , vec_x , vec_y;
	    matValPtr = &csr_val[csr_row_ptr[i]];
	    colIndPtr = &csr_col_ind[csr_row_ptr[i]];
	    aoclsparse_int idx_C     =  i + j * ldc;

	    double sum = 0.0;
	    vec_y = _mm256_setzero_pd();
	    aoclsparse_int l;
	    aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];
	    aoclsparse_int k_iter = nnz/4;
	    aoclsparse_int k_rem = nnz%4;

	    //Loop in multiples of 4 non-zeroes
	    for(l =  0 ; l < k_iter ; l++ )
	    {
		//(csr_val[j] (csr_val[j+1] (csr_val[j+2] (csr_val[j+3]
		vec_vals = _mm256_loadu_pd((double const *)matValPtr);

		idx_B = (colIndPtr[0] + j * ldb);
		idx_B_1 = (colIndPtr[1] + j * ldb);
		idx_B_2 = (colIndPtr[2] + j * ldb);
		idx_B_3 = (colIndPtr[3] + j * ldb);
		//Gather the x vector elements from the column indices
		vec_x  = _mm256_set_pd(B[idx_B_3],
			B[idx_B_2],
			B[idx_B_1],
			B[idx_B]);

		vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);
		matValPtr+=4;
		colIndPtr+=4;
	    }

	    // Horizontal addition
	    if(k_iter){
		// sum[0] += sum[1] ; sum[2] += sum[3]
		vec_y = _mm256_hadd_pd(vec_y, vec_y);
		// Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
		__m128d sum_lo = _mm256_castpd256_pd128(vec_y);
		// Extract 128 bits to obtain sum[2] and sum[3]
		__m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);
		// Add remaining two sums
		__m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
		// Store result
		sum = sse_sum[0];
	    }
	    //Remainder loop for nnz%4
	    for(l = 0 ; l < k_rem ; l++ )
	    {
		idx_B = (colIndPtr[l] + j * ldb);
		sum += *matValPtr++ * B[idx_B];
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

aoclsparse_status aoclsparse_csrmm_row_major_vec(const float*            alpha,
	const float* __restrict__          csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int                     m,
	aoclsparse_int                     k,
	const float*                       B,
	aoclsparse_int                     n,
	aoclsparse_int                     ldb,
	const float*                       beta,
	float*                             C,
	aoclsparse_int                     ldc)
{
    const aoclsparse_int *colIndPtr;
    const float *matValPtr;
    aoclsparse_int idx_B = 0, idx_B_1 = 0, idx_B_2 = 0, idx_B_3 = 0;
    aoclsparse_int idx_B_4 = 0, idx_B_5 = 0, idx_B_6 = 0, idx_B_7 = 0;

    for(aoclsparse_int j = 0; j < n; ++j)
    {
	for(aoclsparse_int i = 0; i < m; ++i)
	{
	    __m256 vec_vals , vec_x , vec_y;
	    matValPtr = &csr_val[csr_row_ptr[i]];
	    colIndPtr = &csr_col_ind[csr_row_ptr[i]];
	    aoclsparse_int idx_C     =  i * ldc + j;

	    float sum = 0.0;
	    vec_y = _mm256_setzero_ps();
	    aoclsparse_int l;
	    aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];
	    aoclsparse_int k_iter = nnz/4;
	    aoclsparse_int k_rem = nnz%4;

	    //Loop in multiples of 4 non-zeroes
	    for(l =  0 ; l < k_iter ; l++ )
	    {
		//(csr_val[j] csr_val[j+1] csr_val[j+2] csr_val[j+3] csr_val[j+4] csr_val[j+5] csr_val[j+6] csr_val[j+7]
		vec_vals = _mm256_loadu_ps((float const *)matValPtr);

		idx_B = (j + colIndPtr[0] * ldb);
		idx_B_1 = (j + colIndPtr[1] * ldb);
		idx_B_2 = (j + colIndPtr[2] * ldb);
		idx_B_3 = (j + colIndPtr[3] * ldb);
		idx_B_4 = (j + colIndPtr[4] * ldb);
		idx_B_5 = (j + colIndPtr[5] * ldb);
		idx_B_6 = (j + colIndPtr[6] * ldb);
		idx_B_7 = (j + colIndPtr[7] * ldb);
		//Gather the x vector elements from the column indices
		vec_x  = _mm256_set_ps(B[idx_B_7],
			B[idx_B_6],
			B[idx_B_5],
			B[idx_B_4],
			B[idx_B_3],
			B[idx_B_2],
			B[idx_B_1],
			B[idx_B]);

		vec_y = _mm256_fmadd_ps(vec_vals, vec_x, vec_y);
		matValPtr+=8;
		colIndPtr+=8;
	    }

	    // Horizontal addition
	    if(k_iter){
		// hiQuad = ( x7, x6, x5, x4 )
		__m128 hiQuad = _mm256_extractf128_ps(vec_y, 1);
		// loQuad = ( x3, x2, x1, x0 )
		const __m128 loQuad = _mm256_castps256_ps128(vec_y);
		// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
		const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x1 + x5, x0 + x4 )
		const __m128 loDual = sumQuad;
		// hiDual = ( -, -, x3 + x7, x2 + x6 )
		const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
		const __m128 sumDual = _mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 + x4 + x6 )
		const __m128 lo = sumDual;
		// hi = ( -, -, -, x1 + x3 + x5 + x7 )
		const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
		const __m128 sse_sum = _mm_add_ss(lo, hi);
		sum = _mm_cvtss_f32(sse_sum);
	    }
	    //Remainder loop for nnz%4
	    for(l = 0 ; l < k_rem ; l++ )
	    {
		idx_B = (j + colIndPtr[l] * ldb) ;
		sum += *matValPtr++ * B[idx_B];
	    }

	    if(*beta == static_cast<float>(0))
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

aoclsparse_status aoclsparse_csrmm_row_major_vec(const double*            alpha,
	const double* __restrict__         csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int                     m,
	aoclsparse_int                     k,
	const double*                      B,
	aoclsparse_int                     n,
	aoclsparse_int                     ldb,
	const double*                      beta,
	double*                            C,
	aoclsparse_int                     ldc)
{
    const aoclsparse_int *colIndPtr;
    const double *matValPtr;
    aoclsparse_int idx_B = 0, idx_B_1 = 0, idx_B_2 = 0, idx_B_3 = 0;

    for(aoclsparse_int j = 0; j < n; ++j)
    {
	for(aoclsparse_int i = 0; i < m; ++i)
	{
	    __m256d vec_vals , vec_x , vec_y;
	    matValPtr = &csr_val[csr_row_ptr[i]];
	    colIndPtr = &csr_col_ind[csr_row_ptr[i]];
	    aoclsparse_int idx_C     =  i * ldc + j;

	    double sum = 0.0;
	    vec_y = _mm256_setzero_pd();
	    aoclsparse_int l;
	    aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];
	    aoclsparse_int k_iter = nnz/4;
	    aoclsparse_int k_rem = nnz%4;

	    //Loop in multiples of 4 non-zeroes
	    for(l =  0 ; l < k_iter ; l++ )
	    {
		//(csr_val[j] (csr_val[j+1] (csr_val[j+2] (csr_val[j+3]
		vec_vals = _mm256_loadu_pd((double const *)matValPtr);

		idx_B = (j + colIndPtr[0] * ldb);
		idx_B_1 = (j + colIndPtr[1] * ldb);
		idx_B_2 = (j + colIndPtr[2] * ldb);
		idx_B_3 = (j + colIndPtr[3] * ldb);
		//Gather the x vector elements from the column indices
		vec_x  = _mm256_set_pd(B[idx_B_3],
			B[idx_B_2],
			B[idx_B_1],
			B[idx_B]);

		vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);
		matValPtr+=4;
		colIndPtr+=4;
	    }

	    // Horizontal addition
	    if(k_iter){
		// sum[0] += sum[1] ; sum[2] += sum[3]
		vec_y = _mm256_hadd_pd(vec_y, vec_y);
		// Cast avx_sum to 128 bit to obtain sum[0] and sum[1]
		__m128d sum_lo = _mm256_castpd256_pd128(vec_y);
		// Extract 128 bits to obtain sum[2] and sum[3]
		__m128d sum_hi = _mm256_extractf128_pd(vec_y, 1);
		// Add remaining two sums
		__m128d sse_sum = _mm_add_pd(sum_lo, sum_hi);
		// Store result
		sum = sse_sum[0];
	    }
	    //Remainder loop for nnz%4
	    for(l = 0 ; l < k_rem ; l++ )
	    {
		idx_B = (j + colIndPtr[l] * ldb) ;
		sum += *matValPtr++ * B[idx_B];
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

    template <typename T>
aoclsparse_status aoclsparse_csrmm_col_major(const T*            alpha,
	const T* __restrict__              csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int                     m,
	aoclsparse_int                     k,
	const T*                           B,
	aoclsparse_int                     n,
	aoclsparse_int                     ldb,
	const T*                           beta,
	T*                                 C,
	aoclsparse_int                     ldc)
{
    for(aoclsparse_int j = 0; j < n; ++j)
    {
	for(aoclsparse_int i = 0; i < m; ++i)
	{
	    T row_begin = csr_row_ptr[i] ;
	    T row_end   = csr_row_ptr[i + 1] ;
	    aoclsparse_int idx_C =  i + j * ldc;

	    T sum = static_cast<T>(0);

	    for(aoclsparse_int k = row_begin; k < row_end; ++k)
	    {
		aoclsparse_int idx_B = 0;
		idx_B = (csr_col_ind[k] + j * ldb);

		sum = std::fma(csr_val[k], B[idx_B], sum);
	    }
	    if(*beta == static_cast<T>(0))
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

    template <typename T>
aoclsparse_status aoclsparse_csrmm_row_major(const T*            alpha,
	const T* __restrict__              csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int                     m,
	aoclsparse_int                     k,
	const T*                           B,
	aoclsparse_int                     n,
	aoclsparse_int                     ldb,
	const T*                           beta,
	T*                                 C,
	aoclsparse_int                     ldc)
{
    for(aoclsparse_int i = 0; i < m; ++i)
    {
	for(aoclsparse_int j = 0; j < n; ++j)
	{
	    T row_begin = csr_row_ptr[i] ;
	    T row_end   = csr_row_ptr[i + 1] ;
	    aoclsparse_int idx_C =  i * ldc + j;

	    T sum = static_cast<T>(0);

	    for(aoclsparse_int k = row_begin; k < row_end; ++k)
	    {
		aoclsparse_int idx_B = 0;
		idx_B = (j + (csr_col_ind[k] ) * ldb);

		sum = std::fma(csr_val[k], B[idx_B], sum);
	    }

	    if(*beta == static_cast<T>(0))
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
#endif /* AOCLSPARSE_CSRMM_HPP*/
