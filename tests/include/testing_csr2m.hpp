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

#pragma once
#ifndef TESTING_CSR2M_HPP
#define TESTING_CSR2M_HPP

#include "aoclsparse.h"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"
//boost ublas
// Ignore compiler warning from Boost
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
// Restore
#pragma GCC diagnostic pop

namespace uBLAS = boost::numeric::ublas;

template <typename T>
void aoclsparse_order_column_index(aoclsparse_int  m,
                                   aoclsparse_int  nnz,
                                   aoclsparse_int *csr_row_ptr,
                                   aoclsparse_int *csr_col_ind,
                                   T              *csr_val)
{
    std::vector<aoclsparse_int> col(nnz);
    std::vector<T>              val(nnz);

    memcpy(col.data(), csr_col_ind, sizeof(aoclsparse_int) * nnz);
    memcpy(val.data(), csr_val, sizeof(T) * nnz);
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int              row_begin = csr_row_ptr[i];
        aoclsparse_int              row_end   = csr_row_ptr[i + 1];
        aoclsparse_int              row_nnz   = row_end - row_begin;
        std::vector<aoclsparse_int> perm(row_nnz);
        for(aoclsparse_int j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        aoclsparse_int *col_entry = &col[row_begin];
        T              *val_entry = &val[row_begin];

        std::sort(perm.begin(), perm.end(), [&](const aoclsparse_int &a, const aoclsparse_int &b) {
            return col_entry[a] <= col_entry[b];
        });

        for(aoclsparse_int j = 0; j < row_nnz; ++j)
        {
            csr_col_ind[row_begin + j] = col_entry[perm[j]];
            csr_val[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

template <typename T>
void testing_csr2m(const Arguments &arg)
{
    aoclsparse_int         M     = arg.M;
    aoclsparse_int         N     = arg.N;
    aoclsparse_int         K     = arg.K;
    aoclsparse_int         nnz_A = arg.nnz;
    aoclsparse_int         nnz_B = arg.nnz;
    aoclsparse_int         nnz_C;
    aoclsparse_operation   transA   = arg.transA;
    aoclsparse_operation   transB   = arg.transB;
    aoclsparse_index_base  base     = arg.baseA, baseCSC;
    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    bool                   issymm   = false;
    double                 cpu_gbyte;
    double                 cpu_gflops;
    double                 cpu_time_used, cpu_time_start;
    int                    number_hot_calls;
    aoclsparse_index_base  baseC;
    // Create matrix descriptor
    aoclsparse_local_mat_descr descrA;
    aoclsparse_local_mat_descr descrB;

    // Set matrix index base
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descrA, base));
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descrB, base));

    std::vector<aoclsparse_int> csr_row_ptr_A;
    std::vector<aoclsparse_int> csr_col_ind_A;
    std::vector<T>              csr_val_A;
    std::vector<aoclsparse_int> csr_row_ptr_B;
    std::vector<aoclsparse_int> csr_col_ind_B;
    std::vector<T>              csr_val_B;

    aoclsparse_request request = aoclsparse_stage_full_computation;
    aoclsparse_seedrand();
#if 0
    // Print aoclsparse version
    std::cout << aoclsparse_get_version() << std::endl;
#endif
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
                                   true,
                                   arg.sort);

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
                                   true,
                                   arg.sort);
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
                                   true,
                                   arg.sort);
        nnz_B = nnz_A;
        N     = M;
        if(M != K) //Not square matrix
        {
            csr_col_ind_B.resize(nnz_A);
            csr_row_ptr_B.resize(K + 1, 0);
            csr_val_B.resize(nnz_A);
            //Output-base index of csc (i.e., csr B matrix) buffer
            baseCSC = aoclsparse_index_base_zero;

            /*B matrix = transpose of A ie. csrtocsc(A)*/
            CHECK_AOCLSPARSE_ERROR(aoclsparse_csr2csc(M,
                                                      K,
                                                      nnz_A,
                                                      descrA,
                                                      baseCSC,
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
    aoclsparse_matrix csrA = NULL;
    aoclsparse_create_csr(
        &csrA, base, M, K, nnz_A, csr_row_ptr_A.data(), csr_col_ind_A.data(), csr_val_A.data());
    aoclsparse_matrix csrB = NULL;
    aoclsparse_create_csr(
        &csrB, base, K, N, nnz_B, csr_row_ptr_B.data(), csr_col_ind_B.data(), csr_val_B.data());

    aoclsparse_matrix csrC          = NULL;
    aoclsparse_int   *csr_row_ptr_C = NULL;
    aoclsparse_int   *csr_col_ind_C = NULL;
    T                *csr_val_C     = NULL;
    aoclsparse_int    C_M, C_N;

    using dMatrixType
        = uBLAS::compressed_matrix<T, uBLAS::row_major, 0, uBLAS::unbounded_array<aoclsparse_int>>;

    static dMatrixType ublasCsrA;
    static dMatrixType ublasCsrB;
    static dMatrixType ublasCsrC;

    ublasCsrA = dMatrixType(M, K, nnz_A);
    ublasCsrB = dMatrixType(K, N, nnz_B);

    ublasCsrA.complete_index1_data();
    ublasCsrB.complete_index1_data();

    memcpy(ublasCsrA.value_data().begin(), csr_val_A.data(), nnz_A * sizeof(T));
    aoclsparse_int *ublasA_csr_row_ptr = ublasCsrA.index1_data().begin();
    for(aoclsparse_int i = 0; i < M + 1; i++)
    {
        ublasA_csr_row_ptr[i] = csr_row_ptr_A[i] - base;
    }
    aoclsparse_int *ublasA_csr_col_idx = ublasCsrA.index2_data().begin();
    for(aoclsparse_int i = 0; i < nnz_A; i++)
    {
        ublasA_csr_col_idx[i] = csr_col_ind_A[i] - base;
    }

    memcpy(ublasCsrB.value_data().begin(), csr_val_B.data(), nnz_B * sizeof(T));
    aoclsparse_int *ublasB_csr_row_ptr = ublasCsrB.index1_data().begin();
    for(aoclsparse_int i = 0; i < K + 1; i++)
    {
        ublasB_csr_row_ptr[i] = csr_row_ptr_B[i] - base;
    }
    aoclsparse_int *ublasB_csr_col_idx = ublasCsrB.index2_data().begin();
    for(aoclsparse_int i = 0; i < nnz_B; i++)
    {
        ublasB_csr_col_idx[i] = csr_col_ind_B[i] - base;
    }
    if(arg.unit_check)
    {
        if(arg.stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            CHECK_AOCLSPARSE_ERROR(
                aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC));
        }
        else if(arg.stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            CHECK_AOCLSPARSE_ERROR(
                aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC));

            request = aoclsparse_stage_finalize;
            CHECK_AOCLSPARSE_ERROR(
                aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC));
        }
        aoclsparse_export_csr<T>(
            csrC, &baseC, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C);

        aoclsparse_order_column_index(C_M, nnz_C, csr_row_ptr_C, csr_col_ind_C, csr_val_C);

        /*uBlas reference library sparse prod*/
        ublasCsrC = dMatrixType(C_M, C_N, nnz_C);
        ublasCsrC.complete_index1_data();
        ublasCsrC = uBLAS::sparse_prod(ublasCsrA, ublasCsrB, ublasCsrC);
        ublasCsrC.complete_index1_data();

        unit_check_general(1, C_M + 1, 1, ublasCsrC.index1_data().begin(), csr_row_ptr_C);
        if(near_check_general<T>(1, nnz_C, 1, ublasCsrC.value_data().begin(), csr_val_C))
        {
            aoclsparse_destroy(&csrA);
            aoclsparse_destroy(&csrB);
            aoclsparse_destroy(&csrC);
            return;
        }
        unit_check_general(1, nnz_C, 1, ublasCsrC.index2_data().begin(), csr_col_ind_C);
    }
    number_hot_calls = arg.iters;

    cpu_time_used = DBL_MAX;

    // Performance run
    for(int iter = 0; iter < number_hot_calls; ++iter)
    {
        aoclsparse_destroy(&csrC);
        request        = aoclsparse_stage_full_computation;
        cpu_time_start = aoclsparse_clock();
        CHECK_AOCLSPARSE_ERROR(
            aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC));
        cpu_time_used = aoclsparse_clock_min_diff(cpu_time_used, cpu_time_start);
    }

    aoclsparse_export_csr<T>(
        csrC, &baseC, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C);
    cpu_gflops = csr2m_gflop_count(
                     M, base, csr_row_ptr_A.data(), csr_col_ind_A.data(), csr_row_ptr_B.data())
                 / cpu_time_used;
    cpu_gbyte = csr2m_gbyte_count<T>(M, N, K, nnz_A, nnz_B, nnz_C) / cpu_time_used;

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
    aoclsparse_destroy(&csrA);
    aoclsparse_destroy(&csrB);
    aoclsparse_destroy(&csrC);
}

#endif // TESTING_CSR2M_HPP
