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
#ifndef TESTING_TRSM_HPP
#define TESTING_TRSM_HPP

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
#include "aoclsparse_utils.hpp"

#ifdef EXT_BENCHMARKING
#include "ext_benchmarking.hpp" // defines register_tests_*()
#else
#include "aoclsparse_no_ext_benchmarking.hpp" // defines them as empty/do nothing
#endif

template <typename T>
int testing_trsm_aocl(const Arguments &arg, testdata<T> &td, double timings[])
{
    int                    status  = 0;
    aoclsparse_int         m       = td.m;
    aoclsparse_int         nnz     = td.nnzA;
    aoclsparse_operation   trans   = arg.transA;
    aoclsparse_matrix_type mattype = arg.mattypeA;
    aoclsparse_fill_mode   fill    = arg.uplo;
    aoclsparse_diag_type   diag    = arg.diag;
    aoclsparse_index_base  base    = arg.baseA;
    aoclsparse_order       order   = arg.order;
    aoclsparse_matrix      A;

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
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_sm_hint(A, trans, descr, order, hint));
        // Optimize the matrix, "A"
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_optimize(A));

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            //reset y[] before trsm computation
            std::fill(td.y.begin(), td.y.end(), aoclsparse_numeric::zero<T>());
            double cpu_time_start = aoclsparse_clock();
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_trsm_kid(trans,
                                                           td.alpha,
                                                           A,
                                                           descr,
                                                           order,
                                                           td.x.data(),
                                                           td.k,
                                                           td.ldx,
                                                           td.y.data(),
                                                           td.ldy,
                                                           arg.kid));
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
/* TRSM solver testing driver
  * Solves A * Y = alpha X, for matrix Y using matrices A and X.
  * Matrix X is RHS and is generated from matrix y_gold using MV, X = (1/alpha)*A*y_gold.
  */
template <typename T>
int testing_trsm(const Arguments &arg)
{
    int                    status   = 0;
    aoclsparse_operation   trans    = arg.transA;
    aoclsparse_fill_mode   fill     = arg.uplo;
    aoclsparse_diag_type   diag     = arg.diag;
    aoclsparse_index_base  base     = arg.baseA;
    aoclsparse_order       order    = arg.order;
    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    aoclsparse_matrix_sort sort     = arg.sort;
    bool                   issymm;
    T                      invalpha;
    // the queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;
    testqueue.push_back({"aocl_trsm_hint", &testing_trsm_aocl<T>});
    register_tests_trsm(testqueue);

    // create relevant test data for this API
    testdata<T> td;
    td.m    = arg.M;
    td.n    = arg.N;
    td.k    = arg.K;
    td.nnzA = arg.nnz;

    // space for results
    std::vector<double> timings(arg.iters);

    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        td.alpha = arg.alpha;
        invalpha = 1. / td.alpha;
    }
    else
    {
        //assume complex data type
        tolerance_t<T> v;
        td.alpha = {(tolerance_t<T>)arg.alpha, (tolerance_t<T>)arg.alpha};
        // Given the special form, 1/td.alpha = (1-i)/2alpha
        v        = 1. / (2. * arg.alpha);
        invalpha = {v, -v};
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

    //exit since TRSM expects a square matrix
    if(td.m != td.n)
    {
        std::cerr << "TRSM requires a square matrix that can be either symmetric or triangular"
                  << std::endl;
        return -1;
    }
    if(td.nnzA < td.m)
    {
        std::cerr << "TRSM requires a square matrix that is with full diagonal and therefore nnz "
                     "entries to atleast be size m"
                  << std::endl;
        return -1;
    }

    aoclsparse_int mm, nn;
    // Allocate memory for matrix
    std::vector<T> y_gold;

    if(order == aoclsparse_order_column)
    {
        //test for ldy and ldx greater than matrix dimension
        td.ldy = td.m + 2;
        td.ldx = td.m + 2;
        mm     = td.k;
        nn     = td.m;
    }
    else if(order == aoclsparse_order_row)
    {
        //test for ldy and ldx greater than matrix dimension
        td.ldy = td.k + td.m + 2;
        td.ldx = td.k + td.m + 2;
        mm     = td.m;
        nn     = td.k;
    }
    td.x.resize(mm * td.ldx);
    td.y.resize(mm * td.ldy);
    y_gold.resize(mm * td.ldy);

    if(order == aoclsparse_order_column)
    {
        //k,m
        aoclsparse_init<T>(y_gold, mm, td.ldy, mm);
    }
    else if(order == aoclsparse_order_row)
    {
        //m, k
        aoclsparse_init<T>(y_gold, td.ldy, mm, td.ldy);
    }

    // Verify
    if(arg.unit_check)
    {
        T zero = aoclsparse_numeric::zero<T>();
        if(order == aoclsparse_order_column) //col major order
        {
            for(int col = 0; col < td.k; col++)
            {
                try
                {
                    //Generate rhs (X) using known vector 'y_gold'
                    NEW_CHECK_AOCLSPARSE_ERROR(ref_csrmv(trans,
                                                         invalpha,
                                                         td.n,
                                                         td.n,
                                                         td.csr_valA.data(),
                                                         td.csr_col_indA.data(),
                                                         td.csr_row_ptrA.data(),
                                                         aoclsparse_matrix_type_triangular,
                                                         fill,
                                                         diag,
                                                         base,
                                                         &y_gold[(col * td.ldy)],
                                                         zero,
                                                         &td.x[(col * td.ldx)]));
                }
                catch(BenchmarkException &e)
                {
                    std::cerr << "Error computing reference TRSM results" << std::endl;
                    return 2;
                }
            }
        }
        else if(order == aoclsparse_order_row) //row major order
        {
            std::vector<T> wcolygold, wcolx;
            wcolygold.resize(td.m);
            wcolx.resize(td.m);

            for(int col = 0; col < td.k; col++)
            {
                try
                {
                    NEW_CHECK_AOCLSPARSE_ERROR(
                        ref_gathers(td.n, &y_gold[col], &wcolygold[0], td.ldy));
                    //Generate rhs (X) using known vector 'y_gold'
                    NEW_CHECK_AOCLSPARSE_ERROR(ref_csrmv(trans,
                                                         invalpha,
                                                         td.n,
                                                         td.n,
                                                         td.csr_valA.data(),
                                                         td.csr_col_indA.data(),
                                                         td.csr_row_ptrA.data(),
                                                         aoclsparse_matrix_type_triangular,
                                                         fill,
                                                         diag,
                                                         base,
                                                         wcolygold.data(),
                                                         zero,
                                                         wcolx.data()));
                    NEW_CHECK_AOCLSPARSE_ERROR(ref_scatters(td.n, &wcolx[0], td.ldx, &td.x[col]));
                }
                catch(BenchmarkException &e)
                {
                    std::cerr << "Error computing reference TRSM results" << std::endl;
                    return 2;
                }
            }
        }
    }

    int    number_hot_calls = arg.iters;
    double gflop            = td.k * csrsv_gflop_count<T>(td.m, td.nnzA, diag);
    double gbyte            = td.k * csrsv_gbyte_count<T>(td.m, td.nnzA);

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
        if(arg.unit_check)
        {
            if(near_check_general<T>(nn, mm, td.ldy, y_gold.data(), td.y.data()))
            {
                std::cerr << "Near check failed" << std::endl;
                status += 1;
            }
        }

        // analyzethe results - at the moment just take minimum as before
        double cpu_time_used = DBL_MAX;
        for(int iter = 0; iter < number_hot_calls; ++iter)
            cpu_time_used = (std::min)(cpu_time_used, timings[iter]);

        // count flops
        double cpu_gflops = gflop / cpu_time_used;
        double cpu_gbyte  = gbyte / cpu_time_used;

        // store/print results
        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "K"
                  << std::setw(12) << "ldx" << std::setw(12) << "ldy" << std::setw(12) << "nnz"
                  << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(12) << "msec"
                  << std::setw(12) << "iter" << std::setw(12) << "layout" << std::setw(12)
                  << "verified" << std::endl;

        std::cout << std::setw(12) << td.m << std::setw(12) << td.n << std::setw(12) << td.k
                  << std::setw(12) << td.ldx << std::setw(12) << td.ldy << std::setw(12) << td.nnzA
                  << std::setw(12) << arg.alpha << std::setw(12) << arg.beta << std::setw(12)
                  << cpu_gflops << std::setw(12) << cpu_gbyte << std::setw(12) << std::scientific
                  << cpu_time_used * 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.order ? "col-major" : "row-major") << std::setw(12)
                  << ((arg.unit_check && !status) ? "yes" : "no") << std::endl;
    }
    return status;
}

#endif // TESTING_TRSM_HPP