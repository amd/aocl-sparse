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
#ifndef TESTING_AXPYI_HPP
#define TESTING_AXPYI_HPP

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
int testing_axpyi_aocl(const Arguments &arg, testdata<T> &td, double timings[])
{
    int            status = 0;
    aoclsparse_int nnz    = td.nnzA; //no of non-zero values in the output vector, n = nnz
    aoclsparse_int kid    = -1;

    int number_hot_calls = arg.iters;
    try
    {
        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            td.y                  = td.y_in;
            double cpu_time_start = aoclsparse_clock();
            NEW_CHECK_AOCLSPARSE_ERROR(
                aoclsparse_axpyi(nnz, td.alpha, td.x.data(), td.indx.data(), td.y.data(), kid));
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
int testing_axpyi(const Arguments &arg)
{
    int status = 0;
    // the queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;
    testqueue.push_back({"aocl_axpyi", &testing_axpyi_aocl<T>});
    register_tests_axpyi(testqueue);

    // create relevant test data for this API
    testdata<T> td;
    td.n    = arg.N;
    td.nnzA = arg.nnz;

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

    // space for the API time measurements
    std::vector<double> timings(arg.iters);

    aoclsparse_seedrand();

    // Allocate memory for vectors
    aoclsparse_int xdim, ydim;
    ydim = td.n;
    xdim = td.nnzA;

    td.y_in.resize(ydim);
    td.y.resize(ydim);
    td.x.resize(xdim);
    td.indx.resize(xdim);

    std::vector<T> y_gold(ydim);

    // Initialize data
    aoclsparse_init<T>(td.x, 1, xdim, 1);
    aoclsparse_init<T>(td.y_in, 1, ydim, 1);
    aoclsparse_init_index(td.indx, xdim, 0, ydim);

    if(arg.unit_check)
    {
        try
        {
            y_gold = td.y_in;
            NEW_CHECK_AOCLSPARSE_ERROR(
                ref_axpyi(xdim, td.alpha, td.x.data(), td.indx.data(), y_gold.data()));
        }
        catch(BenchmarkException &e)
        {
            std::cerr << "Error computing reference axpyi results" << std::endl;
            return 2;
        }
    }

    int number_hot_calls = arg.iters;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);
    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        std::cout << "-----" << testqueue[itest].name << "-----" << std::endl;

        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        // Check the results against the reference result
        if(arg.unit_check)
        {
            if(near_check_general<T>(1, ydim, 1, y_gold.data(), td.y.data()))
            {
                return 2;
            }
        }

        // analyze the results - at the moment just take the minimum
        double cpu_time_used = DBL_MAX;
        for(int iter = 0; iter < number_hot_calls; ++iter)
            cpu_time_used = (std::min)(cpu_time_used, timings[iter]);

        // store/print results
        std::cout << std::setw(12) << "N" << std::setw(12) << "nnz" << std::setw(12) << "msec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << td.n << std::setw(12) << td.nnzA << std::setw(12)
                  << std::scientific << cpu_time_used * 1e3 << std::setw(12) << number_hot_calls
                  << std::setw(12) << (arg.unit_check ? "yes" : "no") << std::endl;
    }
    return status;
}

#endif // TESTING_AXPYI_HPP
