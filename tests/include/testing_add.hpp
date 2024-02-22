/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef TESTING_ADD_HPP
#define TESTING_ADD_HPP
#include "aoclsparse.h"
#include "aoclsparse.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_reference.hpp"
#include "aoclsparse_stats.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

#include <iostream>

#ifdef EXT_BENCHMARKING
#include "ext_benchmarking.hpp"
#else
#include "aoclsparse_no_ext_benchmarking.hpp"
#endif

template <typename T>
int testing_csradd_aocl(const Arguments &arg, testdata<T> &td, double timings[])
{
    int                  status = 0;
    aoclsparse_operation op     = arg.transA;
    T                    alpha  = td.alpha;

    aoclsparse_matrix A = NULL, B = NULL, C = NULL;

    try
    {
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_csr(&A,
                                                         arg.baseA,
                                                         td.m,
                                                         td.n,
                                                         td.nnzA,
                                                         td.csr_row_ptrA.data(),
                                                         td.csr_col_indA.data(),
                                                         td.csr_valA.data()));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_csr(&B,
                                                         arg.baseB,
                                                         td.mB,
                                                         td.nB,
                                                         td.nnzB,
                                                         td.csr_row_ptrB.data(),
                                                         td.csr_col_indB.data(),
                                                         td.csr_valB.data()));

        int number_hot_calls = arg.iters;
        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_destroy(&C));
            double cpu_time_start = aoclsparse_clock();
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_add(op, A, alpha, B, &C));
            timings[iter] = aoclsparse_clock_diff(cpu_time_start);
        }

        NEW_CHECK_AOCLSPARSE_ERROR(aocl_csr_sorted_export(
            C, td.baseC, td.mC, td.nC, td.nnzC, td.csr_row_ptrC, td.csr_col_indC, td.csr_valC));
    }
    catch(BenchmarkException &e)
    {
        status = 1;
    }
    aoclsparse_destroy(&A);
    aoclsparse_destroy(&B);
    aoclsparse_destroy(&C);

    return status;
}

/*
 *  CSR ADD Benchmarking Function
 *
 *  used for benchmarking csr add API comparing it with other similar function from diff library using aoclsparse_stats
 *
 *  It performs Benchmarking for C := alpha*op(A) + B Solver
 */
template <typename T>
int testing_add(const Arguments &arg)
{
    int                    status = 0;
    aoclsparse_operation   op     = arg.transA;
    aoclsparse_index_base  baseA = arg.baseA, baseB = arg.baseB;
    aoclsparse_matrix_init mat = arg.matrix, matB = arg.matrixB;
    std::string            filename = arg.filename, filenameB = arg.filenameB;
    aoclsparse_matrix_sort sort = arg.sort;
    bool                   issymm;

    testdata<T> td;

    // the queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;
    testqueue.push_back({"aocl", &testing_csradd_aocl<T>});
    register_tests_csradd(testqueue);

    // data for comparing results
    vector<double>          timings(arg.iters, 0.0);
    std::vector<data_stats> tstats(testqueue.size());

    // At present alpha and beta have the same real / imaginary parts.
    // TODO: support distinct real and imaginary parts
    if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                 || std::is_same_v<T, std::complex<float>>)
    {
        td.alpha = {static_cast<float>(arg.alpha), static_cast<float>(arg.alpha)};
    }
    else if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                      || std::is_same_v<T, std::complex<double>>)
    {
        td.alpha = {static_cast<double>(arg.alpha), static_cast<double>(arg.alpha)};
    }
    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        td.alpha = static_cast<T>(arg.alpha);
    }

    aoclsparse_seedrand();

    td.m    = arg.M;
    td.n    = arg.N;
    td.nnzA = arg.nnz;

    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_init_csr_matrix(td.csr_row_ptrA,
                                                          td.csr_col_indA,
                                                          td.csr_valA,
                                                          td.m,
                                                          td.n,
                                                          td.nnzA,
                                                          baseA,
                                                          mat,
                                                          filename.c_str(),
                                                          issymm,
                                                          true,
                                                          sort));

    // derive size of B matrix based on A (unless it is given as a file)
    if(op == aoclsparse_operation_none)
    {
        td.mB = td.m;
        td.nB = td.n;
    }
    else
    {
        td.mB = td.n;
        td.nB = td.m;
    }
    td.nnzB = arg.nnzB;

    // random generate matrix B
    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_init_csr_matrix(td.csr_row_ptrB,
                                                          td.csr_col_indB,
                                                          td.csr_valB,
                                                          td.mB,
                                                          td.nB,
                                                          td.nnzB,
                                                          baseB,
                                                          matB,
                                                          filenameB.c_str(),
                                                          issymm,
                                                          true,
                                                          sort));

    aoclsparse_int              nnz_C = td.nnzA + td.nnzB;
    std::vector<aoclsparse_int> row_ptr_C_ref, col_ind_C_ref;
    std::vector<T>              val_C_ref;

    if(arg.unit_check)
    {
        row_ptr_C_ref.resize(td.mB + 1);
        col_ind_C_ref.resize(nnz_C);
        val_C_ref.resize(nnz_C);
        NEW_CHECK_AOCLSPARSE_ERROR(ref_add(op,
                                           baseA,
                                           td.m,
                                           td.n,
                                           td.csr_row_ptrA.data(),
                                           td.csr_col_indA.data(),
                                           td.csr_valA.data(),
                                           td.alpha,
                                           baseB,
                                           td.mB,
                                           td.nB,
                                           td.csr_row_ptrB.data(),
                                           td.csr_col_indB.data(),
                                           td.csr_valB.data(),
                                           nnz_C,
                                           row_ptr_C_ref.data(),
                                           col_ind_C_ref.data(),
                                           val_C_ref.data()));
    }

    std::string prob_name = gen_problem_name(arg, td);
    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        // analyze the results - at the moment just take the minimum
        int verify = 0; // assume not tested

        if(arg.unit_check)
        {
            verify = 1;
            if(csrmat_check(td.mB,
                            td.nB,
                            nnz_C,
                            baseA,
                            row_ptr_C_ref,
                            col_ind_C_ref,
                            val_C_ref,
                            td.mC,
                            td.nC,
                            td.nnzC,
                            td.baseC,
                            td.csr_row_ptrC,
                            td.csr_col_indC,
                            td.csr_valC))
            {
                status++;
                verify = 2;
            }
        }

        compute_stats(timings.data(), timings.size(), tstats[itest]);
        twosample_test_result cmp, *pcmp = NULL;

        // compare the run against the first run (AOCL)
        if(itest > 0)
        {
            cmp  = twosample_test(tstats[itest], tstats[0]);
            pcmp = &cmp;
        }
        print_results(
            testqueue[itest].name, prob_name.c_str(), verify, tstats[itest], pcmp, itest == 0);
    }

    return status;
}

#endif // TESTING_ADD_HPP
