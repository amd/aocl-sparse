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

#pragma once
#ifndef TESTING_SP2MD_HPP
#define TESTING_SP2MD_HPP

#include "aoclsparse.h"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

template <typename T>
void testing_sp2md(const Arguments &arg)
{
    aoclsparse_int         M        = arg.M;
    aoclsparse_int         N        = arg.N;
    aoclsparse_int         K        = arg.K;
    aoclsparse_int         nnz_A    = arg.nnz;
    aoclsparse_int         nnz_B    = arg.nnz;
    aoclsparse_int         nnz_C    = 0;
    aoclsparse_operation   transA   = arg.transA;
    aoclsparse_operation   transB   = arg.transB;
    aoclsparse_index_base  base     = arg.baseA;
    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    bool                   issymm   = false;
    double                 cpu_gbyte;
    double                 cpu_gflops;
    double                 cpu_time_used, cpu_time_start;
    int                    number_hot_calls;

    std::vector<aoclsparse_int> csr_row_ptr_A;
    std::vector<aoclsparse_int> csr_col_ind_A;
    std::vector<T>              csr_val_A;
    std::vector<aoclsparse_int> csr_row_ptr_B;
    std::vector<aoclsparse_int> csr_col_ind_B;
    std::vector<T>              csr_val_B;

    aoclsparse_seedrand();
#if 0
    // Print aoclsparse version
    std::cout << aoclsparse_get_version() << std::endl;
#endif

    aoclsparse_local_mat_descr descrA;
    // Set matrix index base
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descrA, base));

    if(mat == aoclsparse_matrix_random)
    {
        // random generate matrix A
        aoclsparse_init_csr_matrix(csr_row_ptr_A,
                                   csr_col_ind_A,
                                   csr_val_A,
                                   M,
                                   K,
                                   nnz_A,
                                   base,
                                   mat,
                                   filename.c_str(),
                                   issymm,
                                   true);

        // random generate matrix B
        aoclsparse_init_csr_matrix(csr_row_ptr_B,
                                   csr_col_ind_B,
                                   csr_val_B,
                                   K,
                                   N,
                                   nnz_B,
                                   base,
                                   mat,
                                   filename.c_str(),
                                   issymm,
                                   true);
    }
    else if(mat == aoclsparse_matrix_file_mtx)
    {
        // Read from mtx file into matrix A
        aoclsparse_init_csr_matrix(csr_row_ptr_A,
                                   csr_col_ind_A,
                                   csr_val_A,
                                   M,
                                   K,
                                   nnz_A,
                                   base,
                                   mat,
                                   filename.c_str(),
                                   issymm,
                                   true);

        nnz_B = nnz_A;
        N     = M;
        if(M != K) //Not square matrix
        {
            csr_col_ind_B.resize(nnz_A);
            csr_row_ptr_B.resize(K + 1, 0);
            csr_val_B.resize(nnz_A);

            /*B matrix = transpose of A ie. csrtocsc(A)*/
            CHECK_AOCLSPARSE_ERROR(aoclsparse_csr2csc(M,
                                                      K,
                                                      nnz_A,
                                                      descrA,
                                                      base,
                                                      csr_row_ptr_A.data(),
                                                      csr_col_ind_A.data(),
                                                      csr_val_A.data(),
                                                      csr_col_ind_B.data(),
                                                      csr_row_ptr_B.data(),
                                                      csr_val_B.data()));
        }
        else //Square Matrix A, then copy same into matrix B
        {
            csr_row_ptr_B = csr_row_ptr_A;
            csr_col_ind_B = csr_col_ind_A;
            csr_val_B     = csr_val_A;
        }
    }
    aoclsparse_matrix A;
    aoclsparse_create_csr(
        &A, base, M, K, nnz_A, csr_row_ptr_A.data(), csr_col_ind_A.data(), csr_val_A.data());
    aoclsparse_matrix B;
    aoclsparse_create_csr(
        &B, base, K, N, nnz_B, csr_row_ptr_B.data(), csr_col_ind_B.data(), csr_val_B.data());

    std::cout << "STILL IN DEVELOPMENT!!!\n";
    return;
    std::cout << "M: " << M << ", K: " << K << ", NNZA: " << nnz_A << "\n";
    std::cout << "K: " << K << ", N: " << N << ", NNZB: " << nnz_B << "\n";

    number_hot_calls = arg.iters;

    cpu_time_used = DBL_MAX;

    // Performance run
    for(int iter = 0; iter < number_hot_calls; ++iter)
    {
        T *C           = new T[M * N];
        cpu_time_start = aoclsparse_clock();
        CHECK_AOCLSPARSE_ERROR( //opA, descrA, A, opB, descrB, B, alpha, beta, C, layout, ldc
            aoclsparse_sp2md<T>(
                transA, descrA, A, transB, descrA, B, 1, 2, C, aoclsparse_order_row, N, -1));
        cpu_time_used = aoclsparse_clock_min_diff(cpu_time_used, cpu_time_start);
        delete[] C;
    }

    cpu_gflops = 0.0;

    cpu_gbyte = 0.;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "K"
              << std::setw(12) << "nnz_A" << std::setw(12) << "nnz_B" << std::setw(12) << "nnz_C"
              << std::setw(12) << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(12) << "msec"
              << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

    std::cout << std::setw(12) << M << std::setw(12) << N << std::setw(12) << K << std::setw(12)
              << nnz_A << std::setw(12) << nnz_B << std::setw(12) << nnz_C << std::setw(12)
              << cpu_gflops << std::setw(12) << cpu_gbyte << std::setw(12) << cpu_time_used * 1e3
              << std::setw(12) << number_hot_calls << std::setw(12)
              << (arg.unit_check ? "yes" : "no") << std::endl;

    aoclsparse_destroy(&A);
    aoclsparse_destroy(&B);
}

#endif // TESTING_SPMMD_HPP
