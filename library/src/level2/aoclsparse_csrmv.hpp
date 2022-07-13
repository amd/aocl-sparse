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
#ifndef AOCLSPARSE_CSRMV_HPP
#define AOCLSPARSE_CSRMV_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_pthread.h"
#include <immintrin.h>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif


aoclsparse_status aoclsparse_csrmv_vectorized(
	const float               alpha,
	aoclsparse_int                         m,
	aoclsparse_int                         n,
	aoclsparse_int                         nnz,
	const float* __restrict__              csr_val,
	const aoclsparse_int* __restrict__     csr_col_ind,
	const aoclsparse_int* __restrict__     csr_row_ptr,
	const float* __restrict__              x,
	const float                            beta,
	float* __restrict__                    y,
	aoclsparse_thread                      *thread)
{
    __m256 vec_vals , vec_x ,vec_y;

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread->num_threads)  private(vec_vals , vec_x ,vec_y)
#endif
    for(aoclsparse_int i = 0; i < m; i++)
    {
	aoclsparse_int j;
	float result = 0.0;
	vec_y = _mm256_setzero_ps();
	aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];
	aoclsparse_int k_iter = nnz/8;
	aoclsparse_int k_rem = nnz%8;

	//Loop in multiples of 8
	for(j =  csr_row_ptr[i] ; j < csr_row_ptr[i + 1] - k_rem ; j+=8 )
	{
	    //(csr_val[j] csr_val[j+1] csr_val[j+2] csr_val[j+3] csr_val[j+4] csr_val[j+5] csr_val[j+6] csr_val[j+7]
	    vec_vals = _mm256_loadu_ps(&csr_val[j]);

	    //Gather the xvector values from the column indices
	    vec_x  = _mm256_set_ps(x[csr_col_ind[j+7]],
		    x[csr_col_ind[j+6]],
		    x[csr_col_ind[j+5]],
		    x[csr_col_ind[j+4]],
		    x[csr_col_ind[j+3]],
		    x[csr_col_ind[j+2]],
		    x[csr_col_ind[j+1]],
		    x[csr_col_ind[j]]);

	    vec_y = _mm256_fmadd_ps(vec_vals, vec_x , vec_y);

	}

	// Horizontal addition of vec_y
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
	    const __m128 sum = _mm_add_ss(lo, hi);
	    result = _mm_cvtss_f32(sum);
	}

	//Remainder loop
	for(j =  csr_row_ptr[i + 1] - k_rem ; j < csr_row_ptr[i + 1] ; j++ )
	{
	    result += csr_val[j] * x[csr_col_ind[j]];
	}

	// Perform alpha * A * x
	if(alpha != static_cast<float>(1))
	{
	    result = alpha * result;
	}

	// Perform (beta * y) + (alpha * A * x)
	if(beta != static_cast<float>(0))
	{
	    result += beta * y[i];
	}

	y[i] = result ;
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csrmv_general(const T               alpha,
	aoclsparse_int                         m,
	aoclsparse_int                         n,
	aoclsparse_int                         nnz,
	const T* __restrict__                  csr_val,
	const aoclsparse_int* __restrict__     csr_col_ind,
	const aoclsparse_int* __restrict__     csr_row_ptr,
	const T* __restrict__                  x,
	const T                                beta,
	T* __restrict__                        y,
	aoclsparse_thread                      *thread)
{

#ifdef _OPENMP
#pragma omp parallel for num_threads(thread->num_threads) schedule(dynamic,m/thread->num_threads)
#endif
    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
	T result = 0.0;

	for(aoclsparse_int j =  csr_row_ptr[i] ; j < csr_row_ptr[i + 1] ; j++ )
	{
	    result += csr_val[j] * x[csr_col_ind[j]];
	}

	// Perform alpha * A * x
	if(alpha != static_cast<double>(1))
	{
	    result = alpha * result;
	}

	// Perform (beta * y) + (alpha * A * x)
	if(beta != static_cast<double>(0))
	{
	    result += beta * y[i];
	}

	y[i] = result ;
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csrmv_symm(const T               alpha,
	aoclsparse_int                         m,
	aoclsparse_int                         n,
	aoclsparse_int                         nnz,
	const T* __restrict__                  csr_val,
	const aoclsparse_int* __restrict__     csr_col_ind,
	const aoclsparse_int* __restrict__     csr_row_ptr,
	const T* __restrict__                  x,
	const T                                beta,
	T* __restrict__                        y)
{
    // Perform (beta * y)
    if(beta != static_cast<double>(1))
    {
	for(aoclsparse_int i = 0; i < m; i++)
	    y[i] = beta * y[i];
    }
    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
    for(aoclsparse_int i = 0; i < m; i++)
    {
	// Diagonal element(if a non-zero) has to be multiplied once
	// in a symmetrc matrix with corresponding x vector element.
	// Last element of every row in a lower triangular symmetric
	// matrix is diagonal element. Diagonal element when equal to
	// zero , will not be present in the csr_val array . To
	// handle this corner case , last_ele_diag is initialised to
	// zero if diagonal element is zero, hence not multiplied.
	// last_ele_diag becomes one if diagonal element is non-zero
	// and hence multiplied once with corresponding x-vector element
	aoclsparse_int diag_idx = csr_row_ptr[i+1] - 1 ;
	aoclsparse_int last_ele_diag= !(csr_col_ind[diag_idx] ^ i);
	y[i]  +=  last_ele_diag * alpha * csr_val[diag_idx] * x[i];
	aoclsparse_int end = csr_row_ptr[i+1] - last_ele_diag ;
	// Handle all the elements in a row other than the diagonal element
	// Each element has an equivelant occurence on other side of the
	// diagonal and hence need to multiply with two offsets of x-vector
	// and update 2 offsets of y-vector
	for(aoclsparse_int j =  csr_row_ptr[i] ; j < end; j++ )
	{
	    y[i] +=  alpha * csr_val[j] * x[csr_col_ind[j]];
	    y[csr_col_ind[j]] += alpha * csr_val[j] * x[i];
	}
    }
    return aoclsparse_status_success;
}


aoclsparse_status aoclsparse_csrmv_vectorized(const double               alpha,
	aoclsparse_int                         m,
	aoclsparse_int                         n,
	aoclsparse_int                         nnz,
	const double* __restrict__             csr_val,
	const aoclsparse_int* __restrict__     csr_col_ind,
	const aoclsparse_int* __restrict__     csr_row_ptr,
	const double* __restrict__             x,
	const double                           beta,
	double* __restrict__                   y,
	aoclsparse_thread                      *thread)
{
    __m256d vec_vals , vec_x , vec_y;
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread->num_threads) schedule(dynamic,m/thread->num_threads) private(vec_vals , vec_x ,vec_y)
#endif
    for(aoclsparse_int i = 0; i < m; i++)
    {
	aoclsparse_int j;
	double result = 0.0;
	vec_y = _mm256_setzero_pd();
	aoclsparse_int nnz = csr_row_ptr[i+1] - csr_row_ptr[i];
	aoclsparse_int k_iter = nnz/4;
	aoclsparse_int k_rem = nnz%4;

	//Loop in multiples of 4 non-zeroes
	for(j =  csr_row_ptr[i] ; j < csr_row_ptr[i + 1] - k_rem ; j+=4 )
	{
	    //(csr_val[j] (csr_val[j+1] (csr_val[j+2] (csr_val[j+3]
	    vec_vals = _mm256_loadu_pd((double const *)&csr_val[j]);

	    //Gather the x vector elements from the column indices
	    vec_x  = _mm256_set_pd(x[csr_col_ind[j + 3]],
		    x[csr_col_ind[j + 2]],
		    x[csr_col_ind[j + 1]],
		    x[csr_col_ind[j]]);

	    vec_y = _mm256_fmadd_pd(vec_vals, vec_x, vec_y);

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
	    /*
	       __m128d in gcc is typedef as double
	       but in Windows, this is defined as a struct
	       */
#if !defined(__clang__) && (defined(_WIN32) || defined(_WIN64))
	    result = sse_sum.m128d_f64[0];
#else
	    result = sse_sum[0];
#endif
	}

	//Remainder loop for nnz%4
	for(j =  csr_row_ptr[i + 1] - k_rem ; j < csr_row_ptr[i + 1] ; j++ )
	{
	    result += csr_val[j] * x[csr_col_ind[j]];
	}

	// Perform alpha * A * x
	if(alpha != static_cast<double>(1))
	{
	    result = alpha * result;
	}

	// Perform (beta * y) + (alpha * A * x)
	if(beta != static_cast<double>(0))
	{
	    result += beta * y[i];
	}

	y[i] = result ;
    }

    return aoclsparse_status_success;
}

#endif // AOCLSPARSE_CSRMV_HPP

