/* ************************************************************************
    * Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef TESTING_CSRMM_HPP
#define TESTING_CSRMM_HPP

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

#ifdef EXT_BENCHMARKING
#include "ext_benchmarking.hpp"
#else
#include "aoclsparse_no_ext_benchmarking.hpp"
#endif

template <typename T>
int testing_csrmm_aocl(const Arguments &arg, testdata<T> &td, double timings[], aoclsparse_int kid)
{
    int                    status  = 0;
    aoclsparse_int         m       = td.m;
    aoclsparse_int         n       = td.n;
    aoclsparse_int         k       = td.k;
    aoclsparse_int         nnz     = td.nnzA;
    aoclsparse_operation   trans   = arg.transA;
    aoclsparse_matrix_type mattype = arg.mattypeA;
    aoclsparse_fill_mode   fill    = arg.uplo;
    aoclsparse_diag_type   diag    = arg.diag;
    aoclsparse_index_base  base    = arg.baseA;
    aoclsparse_order       order   = arg.order;

    aoclsparse_mat_descr descr = nullptr;
    aoclsparse_matrix    A;
    try
    {
        // Create descriptor
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_mat_descr(&descr));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_type(descr, mattype));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_fill_mode(descr, fill));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_diag_type(descr, diag));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));

        // Create sparse matrix 'A'
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_create_csr<T>(&A,
                                                            base,
                                                            m,
                                                            k,
                                                            nnz,
                                                            td.csr_row_ptrA.data(),
                                                            td.csr_col_indA.data(),
                                                            td.csr_valA.data()));

        int number_hot_calls = arg.iters;
        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            td.y                  = td.y_in;
            double cpu_time_start = aoclsparse_clock();
            NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_csrmm(trans,
                                                        td.alpha,
                                                        A,
                                                        descr,
                                                        order,
                                                        td.x.data(),
                                                        n,
                                                        td.ldx,
                                                        td.beta,
                                                        td.y.data(),
                                                        td.ldy,
                                                        kid));
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
/* CSRMM testing driver
* Solves y := alpha*op(A)*x + beta*y
* where A(mxk) is an aoclsparse matrix in CSR format
* x(mxn) and y(mxn) are dense matrices
* alpha and beta are scalars
*/
template <typename T>
int testing_csrmm(const Arguments &arg)
{
    int                    status   = 0;
    aoclsparse_operation   trans    = arg.transA;
    aoclsparse_index_base  base     = arg.baseA;
    aoclsparse_matrix_type mattype  = arg.mattypeA;
    aoclsparse_order       order    = arg.order;
    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    aoclsparse_matrix_sort sort     = arg.sort;
    bool                   issymm;

    // the queue of test functions to run
    std::vector<testsetting<T>> testqueue;

    using FN = decltype(&testing_csrmm_aocl<T>);

    populate_queue_kid<T, FN>(testqueue, arg, testing_csrmm_aocl<T>);

    register_tests_csrmm(testqueue);

    // create relevant test data for this API
    testdata<T> td;
    td.m    = arg.M;
    td.n    = arg.N;
    td.k    = arg.K;
    td.nnzA = arg.nnz;

    // space for results
    std::vector<double> timings(arg.iters);
    // and their statistics
    std::vector<data_stats> tstats(testqueue.size());

    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        td.alpha = arg.alpha;
        td.beta  = arg.beta;
    }
    else
    {
        //else part assumes complex data type
        td.alpha = {(tolerance_t<T>)arg.alpha, (tolerance_t<T>)arg.alpha};
        td.beta  = {(tolerance_t<T>)arg.beta, (tolerance_t<T>)arg.beta};
    }
    aoclsparse_seedrand();

    // Sample matrix
    aoclsparse_init_csr_matrix(td.csr_row_ptrA,
                               td.csr_col_indA,
                               td.csr_valA,
                               td.m,
                               td.k,
                               td.nnzA,
                               base,
                               mat,
                               filename.c_str(),
                               issymm,
                               true,
                               sort);

    // Some matrix properties
    aoclsparse_int xdim, ydim;
    xdim = (trans == aoclsparse_operation_none ? td.m : td.k);
    ydim = (trans == aoclsparse_operation_none ? td.k : td.m);

    if(order == aoclsparse_order_column)
    {
        td.ldx = ydim + 1; // >= ydim
        td.ldy = xdim + 1; // >= xdim
        td.x.resize(td.ldx * td.n);
        td.y.resize(td.ldy * td.n);
        aoclsparse_init<T>(td.x, td.ldx, td.n, td.ldx);
        aoclsparse_init<T>(td.y, td.ldy, td.n, td.ldy);
    }

    else if(order == aoclsparse_order_row)
    {
        td.ldx = td.n + 1; // >= td.n
        td.ldy = td.n + 1; // >= td.n
        td.x.resize(ydim * td.ldx);
        td.y.resize(xdim * td.ldy);
        aoclsparse_init<T>(td.x, ydim, td.ldx, ydim);
        aoclsparse_init<T>(td.y, xdim, td.ldy, xdim);
    }

    td.y_in               = td.y;
    std::vector<T> y_gold = td.y;

    if(arg.unit_check)
    {
        try
        {
            NEW_CHECK_AOCLSPARSE_ERROR(ref_csrmm(trans,
                                                 td.alpha,
                                                 td.csr_valA.data(),
                                                 td.csr_col_indA.data(),
                                                 td.csr_row_ptrA.data(),
                                                 mattype,
                                                 base,
                                                 order,
                                                 xdim,
                                                 ydim,
                                                 td.x.data(),
                                                 td.n,
                                                 td.ldx,
                                                 td.beta,
                                                 y_gold.data(),
                                                 td.ldy));
        }
        catch(BenchmarkException &e)
        {
            std::cerr << "Error computing reference CSRMM results" << std::endl;
            return 2;
        }
    }

    std::string prob_name = gen_problem_name(arg, td);

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    for(unsigned itest = 0; itest < testqueue.size(); ++itest)
    {
        std::cout << "-----" << testqueue[itest].name << "-----" << std::endl;

        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        // Check the results against the reference result
        int verify = 0;
        if(arg.unit_check)
        {
            verify = 1;
            if((order == aoclsparse_order_column
                && (near_check_general<T>(td.ldy, td.n, td.ldy, y_gold.data(), td.y.data())))
               || ((order == aoclsparse_order_row)
                   && (near_check_general<T>(xdim, td.ldy, xdim, y_gold.data(), td.y.data()))))
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
            // twosample_print(cmp, testqueue[itest].name, tstats[itest],
            // testqueue[0].name, tstats[0]); //, bool debug=false, bool onehdr=true)
        }
        print_results(
            testqueue[itest].name, prob_name.c_str(), verify, tstats[itest], pcmp, itest == 0);
    }
    return status;
}

#endif // TESTING_CSRMM_HPP
