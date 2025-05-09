/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef TESTING_ILU_HPP
#define TESTING_ILU_HPP

#include "aoclsparse_analysis.h"
#include "aoclsparse_convert.h"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

#include <fstream>
#include <string>

#ifdef EXT_BENCHMARKING
#include "ext_benchmarking.hpp" // defines register_tests_*()
#else
#include "aoclsparse_no_ext_benchmarking.hpp" // defines them as empty/do nothing
#endif

template <typename T>
int testing_ilu_aocl(const Arguments &arg, testdata<T> &td, double timings[])
{
    int                    status  = 0;
    aoclsparse_int         m       = td.m;
    aoclsparse_int         nnz     = td.nnzA;
    aoclsparse_operation   trans   = arg.transA;
    aoclsparse_matrix_type mattype = arg.mattypeA;
    aoclsparse_fill_mode   fill    = arg.uplo;
    aoclsparse_diag_type   diag    = arg.diag;
    aoclsparse_index_base  base    = arg.baseA;
    aoclsparse_matrix      A;
    T                     *precond_csr_val = NULL, *approx_inv_diag = NULL;
    // Create matrix descriptor & set it as requested by command line arguments
    aoclsparse_mat_descr descr = nullptr;
    try
    {
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_mat_descr(&descr));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_type(descr, mattype));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_fill_mode(descr, fill));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_diag_type(descr, diag));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));

        int number_hot_calls = arg.iters;
        int hint             = 1000;

        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_csr<T>(&A,
                                                            base,
                                                            m,
                                                            m,
                                                            nnz,
                                                            td.csr_row_ptrA.data(),
                                                            td.csr_col_indA.data(),
                                                            td.csr_valA.data()));
        //to identify hint id(which routine is to be executed, destroyed later)
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_lu_smoother_hint(A, trans, descr, hint));
        // Optimize the matrix, "A"
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_optimize(A));

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            std::fill(td.y.begin(), td.y.end(), aoclsparse_numeric::zero<T>());
            double cpu_time_start = aoclsparse_clock();
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_ilu_smoother(
                trans, A, descr, &precond_csr_val, approx_inv_diag, td.x.data(), td.b.data()));
            std::copy(precond_csr_val, precond_csr_val + td.nnzA, td.y.begin());
            timings[iter] = aoclsparse_clock_diff(cpu_time_start);
        }
    }
    catch(BenchmarkException &e)
    {
        status = 1;
    }
    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(&A);
    return status;
}

template <typename T>
int testing_ilu(const Arguments &arg)
{
    int                    status   = 0;
    aoclsparse_index_base  base     = arg.baseA;
    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    aoclsparse_matrix_sort sort     = arg.sort;
    bool                   issymm;
    // the queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;
    testqueue.push_back({"aocl_ilu_hint", &testing_ilu_aocl<T>});
    register_tests_ilu(testqueue);

    // create relevant test data for this API
    testdata<T> td;
    td.m    = arg.M;
    td.n    = arg.N;
    td.nnzA = arg.nnz;

    // space for results
    std::vector<double> timings(arg.iters);
    // and their statistics
    std::vector<data_stats> tstats(testqueue.size());

    aoclsparse_seedrand();

    // Sample matrix
    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_init_csr_matrix(td.csr_row_ptrA,
                                                          td.csr_col_indA,
                                                          td.csr_valA,
                                                          td.m,
                                                          td.n,
                                                          td.nnzA,
                                                          base,
                                                          mat,
                                                          filename.c_str(),
                                                          issymm,
                                                          true,
                                                          sort));

    //exit since ILU expects a square matrix
    if(td.m != td.n)
    {
        std::cerr << "ILU Preconditioner requires a square matrix that can be either symmetric or "
                     "triangular"
                  << std::endl;
        return -1;
    }

    std::vector<T> csr_val_gold;

    td.x.resize(td.n);
    td.b.resize(td.n);
    td.y.resize(
        td.nnzA); //used to collect ILU factorization data from the library kernel. And then this is used for validation.
    csr_val_gold.resize(td.nnzA);

    if(arg.unit_check)
    {
        csr_val_gold = td.csr_valA;
        try
        {
            //compute ILU0 using reference c code and store the LU factors as part of csr_val_gold
            NEW_CHECK_AOCLSPARSE_ERROR(
                ref_csrilu0(td.n, base, td.csr_row_ptrA, td.csr_col_indA, csr_val_gold.data()));
        }
        catch(BenchmarkException &e)
        {
            std::cerr << "Error computing reference ILU0 results" << std::endl;
            return 2;
        }
    }

    std::string prob_name = gen_problem_name(arg, td);

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        status = 0;
        std::cout << "-----" << testqueue[itest].name << "-----" << std::endl;

        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        // Check the results against the reference result
        int verify = 0; // assume not tested
        if(arg.unit_check)
        {
            verify = 1; // assume pass

            if(!status)
            {
                if(near_check_general<T>(1, td.nnzA, 1, csr_val_gold.data(), td.y.data()))
                {
                    std::cerr << "Near check failed" << std::endl;
                    status += 1;
                    verify = 2;
                }
            }
            else
            {
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

#endif // TESTING_ILU_HPP
