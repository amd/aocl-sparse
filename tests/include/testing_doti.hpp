/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef TESTING_DOTI_HPP
#define TESTING_DOTI_HPP

#include "aoclsparse.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_dot.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
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

template <typename T, bool CALL_INTERNAL>
int testing_doti_aocl(const Arguments &arg, testdata<T> &td, double timings[])
{
    aoclsparse_int nnz              = td.nnzA; //no of non-zero values in the output vector, n = nnz
    int            number_hot_calls = arg.iters;
    // Performance run
    for(int iter = 0; iter < number_hot_calls; ++iter)
    {
        td.s                  = aoclsparse_numeric::zero<T>();
        double cpu_time_start = aoclsparse_clock();
        if constexpr(CALL_INTERNAL)
        {
            NEW_CHECK_AOCLSPARSE_ERROR((aoclsparse_dotp<T>(
                nnz, td.x.data(), td.indx.data(), td.y.data(), &(td.s), false, arg.kid)));
        }
        else
        {
            td.s = aoclsparse_dot<T, T>(
                nnz, td.x.data(), td.indx.data(), td.y.data(), &(td.s), false, -1);
        }

        timings[iter] = aoclsparse_clock_diff(cpu_time_start);
    }

    // dot product of reals returns float/double, so returning success
    return 0;
}

template <typename T>
int testing_doti(const Arguments &arg)
{
    int status = 0;
    // the queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;

    // When kernel ID is -1 invoke the public interface. Else invoke the dispatcher.
    if(arg.kid == -1)
        testqueue.push_back({"aocl", &testing_doti_aocl<T, false>});
    else
        testqueue.push_back({"aocl", &testing_doti_aocl<T, true>});

    register_tests_doti(testqueue);

    // create relevant test data for this API
    testdata<T> td;
    td.n    = arg.N;
    td.nnzA = arg.nnz;

    // space for the API time measurements
    std::vector<double> timings(arg.iters);
    // and their statistics
    std::vector<data_stats> tstats(testqueue.size());

    aoclsparse_seedrand();

    // Allocate memory for vectors
    aoclsparse_int xdim, ydim;
    T              doti_gold;
    ydim = td.n;
    xdim = td.nnzA;

    td.y.resize(ydim);
    td.x.resize(xdim);
    td.indx.resize(xdim);

    // Initialize data
    aoclsparse_init<T>(td.x, 1, xdim, 1);
    aoclsparse_init<T>(td.y, 1, ydim, 1);
    aoclsparse_init_index(td.indx, xdim, 0, ydim);

    if(arg.unit_check)
    {
        try
        {
            NEW_CHECK_AOCLSPARSE_ERROR(
                ref_doti(xdim, td.x.data(), td.indx.data(), td.y.data(), doti_gold));
        }
        catch(BenchmarkException &e)
        {
            std::cerr << "Error computing reference doti results" << std::endl;
            return 2;
        }
    }

    int         number_hot_calls = arg.iters;
    std::string prob_name        = gen_problem_name(arg, td);

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        // Reset output
        td.s = aoclsparse_numeric::quiet_NaN<T>();

        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        int verify = 0; // assume not tested
        // Check the results against the reference result
        if(arg.unit_check)
        {
            verify = 1; // assume pass
            if(near_check_general<T>(1, 1, 1, &doti_gold, &td.s))
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

#endif // TESTING_DOTI_HPP
