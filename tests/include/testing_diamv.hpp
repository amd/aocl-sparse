/* ************************************************************************
 * Copyright (c) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sdia
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
#ifndef TESTING_DIAMV_HPP
#define TESTING_DIAMV_HPP

#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_reference.hpp"
#include "aoclsparse_stats.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

#include <limits>
#include <string>
#include <type_traits>

#ifdef EXT_BENCHMARKING
#include "ext_benchmarking.hpp"
#else
#include "aoclsparse_no_ext_benchmarking.hpp"
#endif

template <typename T>
int testing_diamv_aocl(const Arguments &arg,
                       testdata<T>     &td,
                       double           timings[],
                       aoclsparse_int   diamv_mode = -1,
                       aoclsparse_int   diamv_kid  = -1)
{
    int                   status = 0;
    aoclsparse_int        m      = td.m;
    aoclsparse_int        n      = td.n;
    aoclsparse_operation  trans  = arg.transA;
    aoclsparse_index_base base   = arg.baseA;

    // Create matrix descriptor
    aoclsparse_local_mat_descr descr;
    try
    {
        // Set matrix index base
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));

        // DIA format arrays
        std::vector<aoclsparse_int> dia_offset;
        std::vector<T>              dia_val;
        aoclsparse_int              dia_num_diag;

        // Convert CSR matrix to DIA
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_csr2dia_ndiag(
            m, n, descr, td.nnzA, td.csr_row_ptrA.data(), td.csr_col_indA.data(), &dia_num_diag));
        aoclsparse_int size = (m > n) ? m : n;
        if(size < 0 || dia_num_diag < 0)
        {
            return 1;
        }

        size_t nnz_dia = static_cast<size_t>(size);
        if(static_cast<size_t>(dia_num_diag) > 0
           && nnz_dia > (std::numeric_limits<size_t>::max() / static_cast<size_t>(dia_num_diag)))
        {
            return 1;
        }
        nnz_dia *= static_cast<size_t>(dia_num_diag);
        // Allocate DIA matrix
        dia_offset.resize(dia_num_diag);
        dia_val.resize(nnz_dia);

        // Convert CSR matrix to DIA
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_csr2dia(m,
                                                      n,
                                                      descr,
                                                      td.csr_row_ptrA.data(),
                                                      td.csr_col_indA.data(),
                                                      td.csr_valA.data(),
                                                      dia_num_diag,
                                                      dia_offset.data(),
                                                      dia_val.data()));

        // Performance run
        int number_hot_calls = arg.iters;
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            td.y                  = td.y_in;
            double cpu_time_start = aoclsparse_clock();
            if(diamv_mode < 0 && diamv_kid < 0)
            {
                NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_diamv(trans,
                                                            &(td.alpha),
                                                            m,
                                                            n,
                                                            td.nnzA,
                                                            dia_val.data(),
                                                            dia_offset.data(),
                                                            dia_num_diag,
                                                            descr,
                                                            td.x.data(),
                                                            &(td.beta),
                                                            td.y.data()));
            }
            else if constexpr(std::is_same_v<T, float>)
            {
                NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_sdiamv_kid(trans,
                                                                 &(td.alpha),
                                                                 m,
                                                                 n,
                                                                 td.nnzA,
                                                                 dia_val.data(),
                                                                 dia_offset.data(),
                                                                 dia_num_diag,
                                                                 descr,
                                                                 td.x.data(),
                                                                 &(td.beta),
                                                                 td.y.data(),
                                                                 diamv_mode,
                                                                 diamv_kid));
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_ddiamv_kid(trans,
                                                                 &(td.alpha),
                                                                 m,
                                                                 n,
                                                                 td.nnzA,
                                                                 dia_val.data(),
                                                                 dia_offset.data(),
                                                                 dia_num_diag,
                                                                 descr,
                                                                 td.x.data(),
                                                                 &(td.beta),
                                                                 td.y.data(),
                                                                 diamv_mode,
                                                                 diamv_kid));
            }
            else
            {
                NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_status_not_implemented);
            }
            timings[iter] = aoclsparse_clock_diff(cpu_time_start);
        }
    }
    catch(BenchmarkException &)
    {
        status = 1;
    }
    catch(std::bad_alloc &)
    {
        status = 1;
    }
    return status;
}

template <typename T>
int testing_diamv(const Arguments &arg)
{
    int                    status   = 0;
    aoclsparse_operation   trans    = arg.transA;
    aoclsparse_index_base  base     = arg.baseA;
    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    aoclsparse_matrix_sort sort     = arg.sort;
    bool                   issymm;

    // The queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;

    std::vector<aoclsparse_int> mode_list;
    if(arg.diamv_mode >= 0)
        mode_list.push_back(arg.diamv_mode);
    else
        mode_list.push_back(-1);

    const std::vector<aoclsparse_int> &kid_list = arg.kid_list;
    for(auto mode : mode_list)
    {
        for(auto kid : kid_list)
        {
            std::string name;
            name = "aocl";
            if(mode >= 0)
                name += "/diamv-mode=" + std::to_string(mode);
            if(kid >= 0)
                name += "/kid=" + std::to_string(kid);

            testqueue.push_back(
                {name, [mode, kid](const Arguments &arg, testdata<T> &td, double timings[]) -> int {
                     return testing_diamv_aocl<T>(arg, td, timings, mode, kid);
                 }});
        }
    }

    register_tests_diamv(testqueue);

    // create relevant test data for this API
    testdata<T> td;
    td.m    = arg.M;
    td.n    = arg.N;
    td.nnzA = arg.nnz;

    // space for the API time measurements
    std::vector<double> timings(arg.iters);
    // space for statistics
    std::vector<data_stats> tstats(testqueue.size());

    td.alpha = static_cast<T>(arg.alpha);
    td.beta  = static_cast<T>(arg.beta);

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
                                         arg.mattypeA,
                                         arg.uplo,
                                         arg.diag,
                                         base,
                                         td.x.data(),
                                         td.beta,
                                         y_gold.data()));
    }

    std::string prob_name = gen_problem_name(arg, td);

    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        // Run the test loop
        int run_status = testqueue[itest].tf(arg, td, timings.data());
        status += run_status;

        // Check the results against the reference result
        int verify = 0; // assume not tested
        if(arg.unit_check && run_status == 0)
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

#endif // TESTING_DIAMV_HPP
