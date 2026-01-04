/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
#ifndef TESTING_CSR2CSC_HPP
#define TESTING_CSR2CSC_HPP

#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"
#include "aoclsparse_stats.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

template <typename T>
int testing_csr2csc_aocl(const Arguments &arg, testdata<T> &td, double timings[])
{
    int                   status = 0;
    aoclsparse_int        m      = td.m;
    aoclsparse_int        n      = td.n;
    aoclsparse_int        nnz    = td.nnzA;
    aoclsparse_index_base baseA  = arg.baseA;

    // Create matrix descriptor
    aoclsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, baseA));

    try
    {
        td.csr_row_ptrB.resize(nnz), td.csr_col_indB.resize(n + 1), td.csr_valB.resize(nnz);
        int number_hot_calls = arg.iters;

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            double cpu_time_start = aoclsparse_clock();
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_csr2csc(m,
                                                          n,
                                                          nnz,
                                                          descr,
                                                          baseA,
                                                          td.csr_row_ptrA.data(),
                                                          td.csr_col_indA.data(),
                                                          td.csr_valA.data(),
                                                          td.csr_row_ptrB.data(),
                                                          td.csr_col_indB.data(),
                                                          td.csr_valB.data()));
            timings[iter] = aoclsparse_clock_diff(cpu_time_start);
        }
    }
    catch(BenchmarkException &e)
    {
        status = 1;
    }

    return status;
}

template <typename T>
int testing_csr2csc(const Arguments &arg)
{
    int                   status = 0;
    aoclsparse_int        M      = arg.M;
    aoclsparse_int        N      = arg.N;
    aoclsparse_int        nnz    = arg.nnz;
    aoclsparse_index_base base   = arg.baseA;

    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    bool                   issymm;

    int number_hot_calls = arg.iters;

    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<T>              csr_val;
    std::vector<aoclsparse_int> csc_row_ind;
    std::vector<aoclsparse_int> csc_col_ptr;
    std::vector<T>              csc_val;

    testdata<T>                 td;
    std::vector<testsetting<T>> testqueue;
    testqueue.push_back({"aocl", &testing_csr2csc_aocl<T>});
    register_tests_csr2csc(testqueue);

    // data for comparing results
    std::vector<double>     timings(number_hot_calls, 0.0);
    std::vector<data_stats> tstats(testqueue.size());

    td.m    = M;
    td.n    = N;
    td.nnzA = nnz;

    aoclsparse_seedrand();
#if 0
    // Print aoclsparse version
    std::cout << aoclsparse_get_version() << std::endl;
#endif

    // Generating Matrix
    aoclsparse_init_csr_matrix(td.csr_row_ptrA,
                               td.csr_col_indA,
                               td.csr_valA,
                               td.m,
                               td.n,
                               td.nnzA,
                               base,
                               mat,
                               filename.c_str(),
                               issymm,
                               false);

    M   = td.m;
    N   = td.n;
    nnz = td.nnzA;

    if(arg.unit_check)
    {
        csc_row_ind.resize(nnz);
        csc_col_ptr.resize(N + 1, 0);
        csc_val.resize(nnz);

        //Output-base index of csc buffer
        CHECK_AOCLSPARSE_ERROR(ref_csr2csc(base,
                                           M,
                                           N,
                                           td.csr_row_ptrA.data(),
                                           td.csr_col_indA.data(),
                                           td.csr_valA.data(),
                                           csc_row_ind.data(),
                                           csc_col_ptr.data(),
                                           csc_val.data()));
    }

    std::string bench_name = gen_problem_name(arg, td);

    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        int verify = 0; // assume not tested

        if(arg.unit_check)
        {
            verify = 1; // assume pass
            if(unit_check_general(
                   csc_row_ind.size(), 1, 0, csc_row_ind.data(), td.csr_row_ptrB.data())
               || unit_check_general(
                   csc_col_ptr.size(), 1, 0, csc_col_ptr.data(), td.csr_col_indB.data())
               || near_check_general(csc_val.size(), 1, 0, csc_val.data(), td.csr_valB.data()))
            {
                status++;
                verify = 2;
            }
        }

        td.csr_row_ptrB.clear();
        td.csr_col_indB.clear();
        td.csr_valB.clear();

        compute_stats(timings.data(), timings.size(), tstats[itest]);
        twosample_test_result cmp, *pcmp = NULL;

        // compare the run against the first run (AOCL)
        if(itest > 0)
        {
            cmp  = twosample_test(tstats[itest], tstats[0]);
            pcmp = &cmp;
        }
        print_results(
            testqueue[itest].name, bench_name.c_str(), verify, tstats[itest], pcmp, itest == 0);
    }
    return status;
}

#endif // TESTING_CSR2CSC_HPP
