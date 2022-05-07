/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#ifndef TESTING_CSRMM_HPP
#define TESTING_CSRMM_HPP

#include "aoclsparse.h"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_utility.hpp"
#include "aoclsparse_random.hpp"

template <typename T>
aoclsparse_status aoclsparse_csrmm_col_major_ref(const T*            alpha,
	const T* __restrict__         csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int             m,
	aoclsparse_int             k,
	const T*              B,
	aoclsparse_int             n,
	aoclsparse_int             ldb,
	const T*              beta,
	T*                    C,
	aoclsparse_int             ldc)
{
    for(aoclsparse_int i = 0; i < m; ++i)
    {
	for(aoclsparse_int j = 0; j < n; ++j)
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
aoclsparse_status aoclsparse_csrmm_row_major_ref(const T*            alpha,
	const T* __restrict__         csr_val,
	const aoclsparse_int* __restrict__ csr_col_ind,
	const aoclsparse_int* __restrict__ csr_row_ptr,
	aoclsparse_int             m,
	aoclsparse_int             k,
	const T*              B,
	aoclsparse_int             n,
	aoclsparse_int             ldb,
	const T*              beta,
	T*                    C,
	aoclsparse_int             ldc)
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

template <typename T>
void testing_csrmm(const Arguments& arg)
{
    aoclsparse_int         M         = arg.M;
    aoclsparse_int         N         = arg.N;
    aoclsparse_int         K         = arg.K;
    aoclsparse_int         nnz       = arg.nnz;
    aoclsparse_operation   transA    = arg.transA;
    aoclsparse_index_base  base      = arg.baseA;
    aoclsparse_matrix_init mat       = arg.matrix;
    std::string           filename = arg.filename;
    aoclsparse_order      order  = arg.order;
    bool issymm = false;
    double cpu_gbyte;
    double cpu_gflops;
    double cpu_time_used,cpu_time_start;
    int number_hot_calls ;
    T alpha = static_cast<T>(arg.alpha);
    T beta  = static_cast<T>(arg.beta);

    // Create matrix descriptor
    aoclsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));

    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<T>             csr_val;

    aoclsparse_seedrand();
#if 0
    // Print aoclsparse version
    std::cout << aoclsparse_get_version() << std::endl;
#endif
    // Sample matrix
    aoclsparse_init_csr_matrix(csr_row_ptr,
	    csr_col_ind,
	    csr_val,
	    (transA == aoclsparse_operation_none ? M : K),
	    (transA == aoclsparse_operation_none ? K : M),
	    nnz,
	    base,
	    mat,
	    filename.c_str(),
	    issymm,
	    true);
    if(mat == aoclsparse_matrix_file_mtx)
	N = M;

    // Some matrix properties
    aoclsparse_int A_m = (transA == aoclsparse_operation_none ? M : K);
    aoclsparse_int A_n = (transA == aoclsparse_operation_none ? K : M);
    aoclsparse_int B_m = K ;
    aoclsparse_int B_n = N ;
    aoclsparse_int C_m = M;
    aoclsparse_int C_n = N;
    aoclsparse_int ldb = order == aoclsparse_order_column ?  K :  N ;
    aoclsparse_int ldc = order == aoclsparse_order_column ?  M :  N;
    aoclsparse_int nrowB = order == aoclsparse_order_column ? ldb : B_m;
    aoclsparse_int ncolB = order == aoclsparse_order_column ? B_n : ldb;
    aoclsparse_int nrowC = order == aoclsparse_order_column ? ldc : C_m;
    aoclsparse_int ncolC = order == aoclsparse_order_column ? C_n : ldc;
    aoclsparse_matrix csr;
    aoclsparse_create_csr(csr, base, A_m, A_n, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data());
    // Allocate memory for matrix
    std::vector<T> B(nrowB * ncolB);
    std::vector<T> C(nrowC * ncolC);
    std::vector<T> C_gold(nrowC * ncolC);

    // Initialize data
    aoclsparse_init<T>(B, nrowB, ncolB, nrowB);
    aoclsparse_init<T>(C, nrowC, ncolC, nrowC);
    C_gold = C;

    if(arg.unit_check)
    {
	CHECK_AOCLSPARSE_ERROR(aoclsparse_csrmm(transA,
		    &alpha,
		    csr,
		    descr,
		    order,
		    B.data(),
		    B_n,
		    ldb,
		    &beta,
		    C.data(),
		    ldc));
	// Reference SPMM CSR implementation
	if(order == aoclsparse_order_column)
	    aoclsparse_csrmm_col_major_ref<T>(&alpha,
		    csr_val.data(),
		    csr_col_ind.data(),
		    csr_row_ptr.data(),
		    A_m,
		    A_n,
		    B.data(),
		    B_n,
		    ldb,
		    &beta,
		    C_gold.data(),
		    ldc);
	else
	    aoclsparse_csrmm_row_major_ref<T>(&alpha,
		    csr_val.data(),
		    csr_col_ind.data(),
		    csr_row_ptr.data(),
		    A_m,
		    A_n,
		    B.data(),
		    B_n,
		    ldb,
		    &beta,
		    C_gold.data(),
		    ldc);

	near_check_general<T>(nrowC, ncolC, ldc, C_gold.data(), C.data());
    }
    number_hot_calls  = arg.iters;

    cpu_time_used = DBL_MAX;

    // Performance run
    for(int iter = 0; iter < number_hot_calls; ++iter)
    {
	cpu_time_start = aoclsparse_clock();
	CHECK_AOCLSPARSE_ERROR(aoclsparse_csrmm(transA,
		    &alpha,
		    csr,
		    descr,
		    order,
		    B.data(),
		    B_n,
		    ldb,
		    &beta,
		    C.data(),
		    ldc));
	cpu_time_used = aoclsparse_clock_min_diff( cpu_time_used, cpu_time_start );
    }

    cpu_gflops
	= csrmm_gflop_count<T>(N, nnz, C_m * C_n, beta != static_cast<T>(0)) / cpu_time_used ;
    cpu_gbyte
	= csrmm_gbyte_count<T>(A_m, nnz,B_m * B_n, C_m * C_n, beta != static_cast<T>(0)) / cpu_time_used ;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "K"
	<< std::setw(12) << "nnz_A" << std::setw(12) << "nnz_B" << std::setw(12) << "nnz_C"
	<< std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
	<< "GFlop/s" << std::setw(12) << "GB/s"
	<< std::setw(12) << "msec" << std::setw(12) << "iter" << std::setw(12)
	<< "verified" << std::endl;

    std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << K
	<< std::setw(12) << nnz << std::setw(12) << B_m * B_n << std::setw(12) << C_m * C_n
	<< std::setw(12) << alpha << std::setw(12) << beta << std::setw(12)
	<< cpu_gflops << std::setw(12) << cpu_gbyte
	<< std::setw(12) << std::scientific << cpu_time_used * 1e3
	<< std::setw(12) << number_hot_calls << std::setw(12)
	<< (arg.unit_check ? "yes" : "no") << std::endl;

}

#endif // TESTING_CSRMM_HPP
