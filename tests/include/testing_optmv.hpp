/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef TESTING_OPTMV_HPP
#define TESTING_OPTMV_HPP

#include "aoclsparse_analysis.h"
#include "aoclsparse_convert.h"
#include "aoclsparse.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_reference.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

#ifdef EXT_BENCHMARKING
#include "ext_benchmarking.hpp"
#else
#include "aoclsparse_no_ext_benchmarking.hpp"
#endif

template <typename T>
int testing_optmv_aocl(const Arguments &arg, testdata<T> &td, double timings[], int hint)
{
    int                    status  = 0;
    aoclsparse_int         m       = td.m;
    aoclsparse_int         n       = td.n;
    aoclsparse_int         nnz     = td.nnzA;
    aoclsparse_operation   trans   = arg.transA;
    aoclsparse_matrix_type mattype = arg.mattypeA;
    aoclsparse_fill_mode   fill    = arg.uplo;
    aoclsparse_diag_type   diag    = arg.diag;
    aoclsparse_index_base  base    = arg.baseA;

    // Create matrix descriptor & set it as requested by command line arguments
    aoclsparse_mat_descr descr = NULL;
    aoclsparse_matrix    A     = NULL;
    try
    {
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_mat_descr(&descr));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_type(descr, mattype));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_fill_mode(descr, fill));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_diag_type(descr, diag));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));

        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_csr<T>(&A,
                                                            base,
                                                            m,
                                                            n,
                                                            nnz,
                                                            td.csr_row_ptrA.data(),
                                                            td.csr_col_indA.data(),
                                                            td.csr_valA.data()));

        if(hint > 0)
        {
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mv_hint(A, trans, descr, hint));
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_optimize(A));
        }

        // Performance run
        int number_hot_calls = arg.iters;
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            td.y                  = td.y_in;
            double cpu_time_start = aoclsparse_clock();
            NEW_CHECK_AOCLSPARSE_ERROR(
                aoclsparse_mv(trans, &td.alpha, A, descr, td.x.data(), &td.beta, td.y.data()));
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
int testing_optmv(const Arguments &arg)
{
    int                    status   = 0;
    aoclsparse_operation   trans    = arg.transA;
    aoclsparse_matrix_type mattype  = arg.mattypeA;
    aoclsparse_fill_mode   fill     = arg.uplo;
    aoclsparse_diag_type   diag     = arg.diag;
    aoclsparse_index_base  base     = arg.baseA;
    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    aoclsparse_matrix_sort sort     = arg.sort;
    bool                   issymm;

    // the queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;
    testqueue.push_back(
        {"aocl_optmv_hint", [](const Arguments &arg, testdata<T> &td, double timings[]) {
             return testing_optmv_aocl<T>(arg, td, timings, /*hint=*/1000);
         }});
    register_tests_csrmv(testqueue);

    // create relevant test data for this API
    testdata<T> td;
    td.m    = arg.M;
    td.n    = arg.N;
    td.nnzA = arg.nnz;

    // space for the API time measurements
    std::vector<double> timings(arg.iters);
    // space for statistics
    std::vector<data_stats> tstats(testqueue.size());

    // At present alpha and beta have the same real / imaginary parts.
    // TODO: support distinct real and imaginary parts
    if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                 || std::is_same_v<T, std::complex<float>>)
    {
        td.alpha = {static_cast<float>(arg.alpha), static_cast<float>(arg.alpha)};
        td.beta  = {static_cast<float>(arg.beta), static_cast<float>(arg.beta)};
    }
    else if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                      || std::is_same_v<T, std::complex<double>>)
    {
        td.alpha = {static_cast<double>(arg.alpha), static_cast<double>(arg.alpha)};
        td.beta  = {static_cast<double>(arg.beta), static_cast<double>(arg.beta)};
    }
    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        td.alpha = static_cast<T>(arg.alpha);
        td.beta  = static_cast<T>(arg.beta);
    }

    aoclsparse_seedrand();

    // Sample matrix
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
                               true,
                               sort);

    // Allocate memory for vectors
    aoclsparse_int xdim, ydim;
    if(trans == aoclsparse_operation_none)
    {
        xdim = td.n;
        ydim = td.m;
    }
    else
    {
        xdim = td.m;
        ydim = td.n;
    }

    td.x.resize(xdim);
    td.y.resize(ydim);
    td.y_in.resize(ydim);
    std::vector<T> y_gold(ydim); // reference result

    // Initialize data
    aoclsparse_init<T>(td.x, 1, xdim, 1);
    aoclsparse_init<T>(td.y_in, 1, ydim, 1);
    y_gold = td.y_in;
    td.y   = td.y_in;

    if(arg.unit_check)
    {
        CHECK_AOCLSPARSE_ERROR(ref_csrmv(trans,
                                         td.alpha,
                                         td.m,
                                         td.n,
                                         td.csr_valA.data(),
                                         td.csr_col_indA.data(),
                                         td.csr_row_ptrA.data(),
                                         mattype,
                                         fill,
                                         diag,
                                         base,
                                         td.x.data(),
                                         td.beta,
                                         y_gold.data()));
    }

    std::string prob_name = gen_problem_name(arg, td);

    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        // Check the results against the reference result
        int verify = 0; // assume not tested
        if(arg.unit_check)
        {
            verify = 1; // assume pass
            if(near_check_general<T>(1, ydim, 1, y_gold.data(), td.y.data()))
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

#endif // TESTING_OPTMV_HPP
