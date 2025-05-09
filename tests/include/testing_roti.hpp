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
#ifndef TESTING_ROTI_HPP
#define TESTING_ROTI_HPP

#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_reference.hpp"
#include "aoclsparse_stats.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

#ifdef EXT_BENCHMARKING
#include "ext_benchmarking.hpp"
#else
#include "aoclsparse_no_ext_benchmarking.hpp"
#endif

template <typename T>
int testing_roti_aocl(const Arguments &arg, testdata<T> &td, double timings[], aoclsparse_int kid)
{
    int            status = 0;
    aoclsparse_int nnz    = td.nnzA; //no of non-zero values in the output vector, n = nnz

    int number_hot_calls = arg.iters;
    try
    {
        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            td.y                  = td.y_in;
            td.x                  = td.x_in;
            double cpu_time_start = aoclsparse_clock();

            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_roti(
                nnz, td.x.data(), td.indx.data(), td.y.data(), td.alpha, td.beta, kid))

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
int testing_roti(const Arguments &arg)
{
    int status = 0;
    // the queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;

    using FN = decltype(&testing_roti_aocl<T>);

    populate_queue_kid<T, FN>(testqueue, arg, testing_roti_aocl<T>);

    register_tests_roti(testqueue);

    // create relevant test data for this API
    testdata<T> td;
    td.n    = arg.N;
    td.nnzA = aoclsparse_init_spvec_size(arg.nnz, arg.N);

    // space for the API time measurements
    std::vector<double> timings(arg.iters);
    // and their statistics
    std::vector<data_stats> tstats(testqueue.size());

    aoclsparse_seedrand();

    // Allocate memory for vectors
    aoclsparse_int xdim, ydim;
    ydim = td.n;
    xdim = td.nnzA;

    td.y.resize(ydim);
    td.y_in.resize(ydim);
    td.x.resize(xdim);
    td.x_in.resize(xdim);
    td.indx.resize(xdim);

    std::vector<T> y_gold(ydim, aoclsparse_numeric::zero<T>());
    std::vector<T> x_gold(xdim, aoclsparse_numeric::zero<T>());

    // Initialize data
    aoclsparse_init<T>(td.x_in, 1, xdim, 1);
    aoclsparse_init<T>(td.y, 1, ydim, 1);
    aoclsparse_init_index(td.indx, xdim, 0, ydim);

    if(arg.unit_check)
    {
        try
        {
            NEW_CHECK_AOCLSPARSE_ERROR(ref_givens_rot(xdim,
                                                      td.x_in.data(),
                                                      td.indx.data(),
                                                      td.y_in.data(),
                                                      x_gold.data(),
                                                      y_gold.data(),
                                                      td.alpha,
                                                      td.beta));
        }
        catch(BenchmarkException &e)
        {
            std::cerr << "Error computing reference roti results" << std::endl;
            return 2;
        }
    }

    std::string prob_name = gen_problem_name(arg, td);

    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        int verify = 0; // assume not tested
        // Check the results against the reference result
        if(arg.unit_check)
        {
            verify = 1; // assume pass
            if(near_check_general<T>(1, ydim, 1, y_gold.data(), td.y.data()))
            {
                status++;
                verify = 2;
            }
            if(near_check_general<T>(1, xdim, 1, x_gold.data(), td.x.data()))
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

#endif // TESTING_ROTI_HPP
