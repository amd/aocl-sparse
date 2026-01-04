/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef TESTING_SP2M_HPP
#define TESTING_SP2M_HPP

#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_reference.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

#ifdef EXT_BENCHMARKING
#include "ext_benchmarking.hpp" // defines register_tests_*()
#else
#include "aoclsparse_no_ext_benchmarking.hpp" // defines them as empty/do nothing
#endif

template <typename T>
int testing_sp2m_aocl(const Arguments &arg, testdata<T> &td, double timings[], int hint)
{
    int                  status = 0;
    aoclsparse_operation transA = arg.transA;
    aoclsparse_operation transB = arg.transB;

    // Create matrix descriptor & set it as requested by command line arguments
    aoclsparse_mat_descr descrA = NULL;
    aoclsparse_mat_descr descrB = NULL;
    aoclsparse_matrix    A      = NULL;
    aoclsparse_matrix    B      = NULL;
    aoclsparse_matrix    C      = NULL;
    try
    {
        // sp2m supports only general matrices right now, don't bother filling uplo & diag
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_mat_descr(&descrA));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_type(descrA, arg.mattypeA));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descrA, arg.baseA));

        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_mat_descr(&descrB));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_type(descrB, arg.mattypeB));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descrB, arg.baseB));

        int number_hot_calls = arg.iters;

        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_csr<T>(&A,
                                                            arg.baseA,
                                                            td.m,
                                                            td.n,
                                                            td.nnzA,
                                                            td.csr_row_ptrA.data(),
                                                            td.csr_col_indA.data(),
                                                            td.csr_valA.data()));
        if(hint > 0)
        {
            //to identify hint id(which routine is to be executed, destroyed later)
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_2m_hint(A, transA, descrA, hint));

            // Optimize the matrix, "A"
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_optimize(A));
        }

        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_csr<T>(&B,
                                                            arg.baseB,
                                                            td.mB,
                                                            td.nB,
                                                            td.nnzB,
                                                            td.csr_row_ptrB.data(),
                                                            td.csr_col_indB.data(),
                                                            td.csr_valB.data()));

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            aoclsparse_destroy(&C);
            double cpu_time_start = aoclsparse_clock();
            if(arg.stage == 0)
            {
                NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_sp2m(
                    transA, descrA, A, transB, descrB, B, aoclsparse_stage_full_computation, &C));
            }
            else if(arg.stage == 1)
            {
                NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_sp2m(
                    transA, descrA, A, transB, descrB, B, aoclsparse_stage_nnz_count, &C));
                NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_sp2m(
                    transA, descrA, A, transB, descrB, B, aoclsparse_stage_finalize, &C));
            }
            timings[iter] = aoclsparse_clock_diff(cpu_time_start);
        }

        // export last result matrix into td.*C with sorted row elements
        NEW_CHECK_AOCLSPARSE_ERROR(aocl_csr_sorted_export(
            C, td.baseC, td.mC, td.nC, td.nnzC, td.csr_row_ptrC, td.csr_col_indC, td.csr_valC));
    }
    catch(BenchmarkException &e)
    {
        status = 1;
    }

    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy_mat_descr(descrB);
    aoclsparse_destroy(&A);
    aoclsparse_destroy(&B);
    aoclsparse_destroy(&C);

    return status;
}

template <typename T>
int testing_sp2m(const Arguments &arg)
{
    int                  status = 0;
    aoclsparse_operation transA = arg.transA;
    aoclsparse_operation transB = arg.transB;
    bool                 issymm;

    std::vector<testsetting<T>> testqueue;
    testqueue.push_back({"aocl", [](const Arguments &arg, testdata<T> &td, double timings[]) {
                             return testing_sp2m_aocl<T>(arg, td, timings, /*hint=*/1000);
                         }});
    register_tests_sp2m(testqueue);

    // space for results and their statistics
    std::vector<double>     timings(arg.iters);
    std::vector<data_stats> tstats(testqueue.size());

    aoclsparse_seedrand();

    // create relevant test data for this API
    testdata<T> td;
    td.m    = arg.M;
    td.n    = arg.K;
    td.nnzA = arg.nnz;

    // Sample matrix
    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_init_csr_matrix(td.csr_row_ptrA,
                                                          td.csr_col_indA,
                                                          td.csr_valA,
                                                          td.m,
                                                          td.n,
                                                          td.nnzA,
                                                          arg.baseA,
                                                          arg.matrix,
                                                          arg.filename.c_str(),
                                                          issymm,
                                                          true,
                                                          arg.sort));

    // derive size of B matrix based on A (unless it is given as a file)
    aoclsparse_int inner_dim = (transA == aoclsparse_operation_none) ? td.n : td.m;
    if(transB == aoclsparse_operation_none)
    {
        td.mB = inner_dim;
        td.nB = arg.N;
    }
    else
    {
        td.mB = arg.N;
        td.nB = inner_dim;
    }
    td.nnzB = arg.nnzB;
    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_init_csr_matrix(td.csr_row_ptrB,
                                                          td.csr_col_indB,
                                                          td.csr_valB,
                                                          td.mB,
                                                          td.nB,
                                                          td.nnzB,
                                                          arg.baseB,
                                                          arg.matrixB,
                                                          arg.filenameB.c_str(),
                                                          issymm,
                                                          true,
                                                          arg.sort));

    // Vectors for storing gold data
    aoclsparse_int              m_gold, n_gold, nnz_gold;
    aoclsparse_index_base       base_gold;
    std::vector<aoclsparse_int> csr_row_ptrC_gold;
    std::vector<aoclsparse_int> csr_col_indC_gold;
    std::vector<T>              csr_valC_gold;

    std::string prob_name = gen_problem_name(arg, td);

    for(size_t itest = 0; itest < testqueue.size(); ++itest)
    {
        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        // Check the results against the reference result
        // Take copy of the result of itest = 0 as gold
        int verify = 0; // assume not tested
        if(itest == 0)
        {
            m_gold            = td.mC;
            n_gold            = td.nC;
            nnz_gold          = td.nnzC;
            base_gold         = td.baseC;
            csr_row_ptrC_gold = td.csr_row_ptrC;
            csr_col_indC_gold = td.csr_col_indC;
            csr_valC_gold     = td.csr_valC;
        }
        // Check the results of itest > 0 against the gold (itest = 0)
        else if(arg.unit_check)
        {
            verify = 1; // assume pass
            if(csrmat_check(m_gold,
                            n_gold,
                            nnz_gold,
                            base_gold,
                            csr_row_ptrC_gold,
                            csr_col_indC_gold,
                            csr_valC_gold,
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

#endif // TESTING_SP2M_HPP
