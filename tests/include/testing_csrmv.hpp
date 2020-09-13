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

#pragma once
#ifndef TESTING_CSRMV_HPP
#define TESTING_CSRMV_HPP

#include "aoclsparse.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_utility.hpp"
#include "aoclsparse_random.hpp"

template <typename T>
void testing_csrmv(const Arguments& arg)
{
    aoclsparse_int         M         = arg.M;
    aoclsparse_int         N         = arg.N;
    aoclsparse_int         nnz       = arg.nnz;
    aoclsparse_operation   trans     = arg.transA;
    aoclsparse_index_base  base      = arg.baseA;
    aoclsparse_matrix_init mat       = arg.matrix;
    std::string           filename = arg.filename;
    bool issymm;
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
    int  ver;

    aoclsparse_get_version(&ver);

    std::cout << "aocl-sparse version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
        << ver % 100 << std::endl;
#endif
    // Sample matrix
    aoclsparse_init_csr_matrix(csr_row_ptr,
            csr_col_ind,
            csr_val,
            M,
            N,
            nnz,
            base,
            mat,
            filename.c_str(),
            issymm,
            true);

    // Allocate memory for vectors
    std::vector<T> x(N);
    std::vector<T> y(M);
    std::vector<T> y_gold(M);

    // Initialize data
    aoclsparse_init<T>(x, 1, N, 1);
    aoclsparse_init<T>(y, 1, M, 1);
    y_gold = y;
    if(arg.unit_check)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_csrmv(trans,
                    &alpha,
                    M,
                    N,
                    nnz,
                    csr_val.data(),
                    csr_col_ind.data(),
                    csr_row_ptr.data(),
                    descr,
                    x.data(),
                    &beta,
                    y.data()));
        // Reference SPMV CSR implementation
        for(int i = 0; i < M; i++)
        {
            T result = 0.0;
            for(int j = csr_row_ptr[i]-base ; j < csr_row_ptr[i+1]-base ; j++)
            {
                result += alpha * csr_val[j] * x[csr_col_ind[j] - base];
            }
            y_gold[i] = (beta * y_gold[i]) + result;
        }
        near_check_general<T>(1, M, 1, y_gold.data(), y.data());
    }
    int number_hot_calls  = arg.iters;

    double cpu_time_used = DBL_MAX;

    // Performance run
    for(int iter = 0; iter < number_hot_calls; ++iter)
    {
        double cpu_time_start = aoclsparse_clock();
        CHECK_AOCLSPARSE_ERROR(aoclsparse_csrmv(trans,
                    &alpha,
                    M,
                    N,
                    nnz,
                    csr_val.data(),
                    csr_col_ind.data(),
                    csr_row_ptr.data(),
                    descr,
                    x.data(),
                    &beta,
                    y.data()));
        cpu_time_used = aoclsparse_clock_min_diff( cpu_time_used, cpu_time_start );
    }

    double cpu_gflops
        = spmv_gflop_count<T>(M, nnz, beta != static_cast<T>(0)) / cpu_time_used ;
    double cpu_gbyte
        = csrmv_gbyte_count<T>(M, N, nnz, beta != static_cast<T>(0)) / cpu_time_used ;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
        << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
        << "GFlop/s" << std::setw(12) << "GB/s"
        << std::setw(12) << "msec" << std::setw(12) << "iter" << std::setw(12)
        << "verified" << std::endl;

    std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << nnz
        << std::setw(12) << alpha << std::setw(12) << beta << std::setw(12)
        << cpu_gflops
        << std::setw(12) << cpu_gbyte << std::setw(12) << cpu_time_used * 1e3
        << std::setw(12) << number_hot_calls << std::setw(12)
        << (arg.unit_check ? "yes" : "no") << std::endl;
}

#endif // TESTING_CSRMV_HPP
