/* ************************************************************************
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef TESTING_BSRMV_HPP
#define TESTING_BSRMV_HPP

#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_stats.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

template <typename T>
int testing_bsrmv_aocl(const Arguments &arg, testdata<T> &td, double timings[])
{
    int                  status = 0;
    aoclsparse_int       m      = td.m;
    aoclsparse_int       n      = td.n;
    aoclsparse_operation trans  = arg.transA;

    try
    {
        // Create the BSR matrix
        aoclsparse_matrix mtx;

        aoclsparse_create_bsr<T>(&mtx,
                                 arg.baseA,
                                 arg.order,
                                 m,
                                 n,
                                 td.bsr_dim,
                                 td.csr_row_ptrA.data(),
                                 td.csr_col_indA.data(),
                                 td.csr_valA.data(),
                                 false);

        int number_hot_calls = arg.iters;
        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            td.y                  = td.y_in;
            double cpu_time_start = aoclsparse_clock();
            CHECK_AOCLSPARSE_ERROR(aoclsparse_mv<T>(
                trans, &td.alpha, mtx, td.descr, td.x.data(), &td.beta, td.y.data()));
            timings[iter] = aoclsparse_clock_diff(cpu_time_start);
        }

        // Destroy the matrix
        aoclsparse_destroy(&mtx);
    }
    catch(BenchmarkException &e)
    {
        status = 1;
    }
    return status;
}

template <typename T>
int testing_bsrmv(const Arguments &arg)
{
    int                    status   = 0;
    aoclsparse_int         M        = arg.M;
    aoclsparse_int         N        = arg.N;
    aoclsparse_int         nnz      = arg.nnz;
    aoclsparse_operation   trans    = arg.transA;
    aoclsparse_order       order    = arg.order;
    aoclsparse_index_base  base     = arg.baseA;
    aoclsparse_matrix_init mat      = arg.matrix;
    std::string            filename = arg.filename;
    aoclsparse_int         bsr_dim  = arg.block_dim;
    bool                   issymm;
    T                      alpha;
    T                      beta;

    testdata<T> td;
    td.m    = arg.M;
    td.n    = arg.N;
    td.nnzA = arg.nnz;

    //At present alpha and beta have the same real / imaginary parts.
    //TODO: support distinct real and imaginary parts
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
    else if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        td.alpha = static_cast<T>(arg.alpha);
        td.beta  = static_cast<T>(arg.beta);
    }

    alpha = td.alpha;
    beta  = td.beta;

    // The queue of test functions to run, normally it would be just one API
    // unless more tests are registered via EXT_BENCHMARKING
    std::vector<testsetting<T>> testqueue;
    testqueue.push_back({"aocl", &testing_bsrmv_aocl<T>});

    // Create matrix descriptor
    aoclsparse_local_mat_descr descr;

    // Set matrix index base
    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));
    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_type(descr, arg.mattypeA));
    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_fill_mode(descr, arg.uplo));
    NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_diag_type(descr, arg.diag));

    // Allocate memory for matrix
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<T>              csr_val;

    std::vector<double> timings(arg.iters);
    // and their statistics
    std::vector<data_stats> tstats(testqueue.size());

    aoclsparse_seedrand();

    // Sample matrix
    aoclsparse_init_csr_matrix(
        csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base, mat, filename.c_str(), issymm, false);

    // Update BSR block dimensions from generated matrix
    aoclsparse_int mb = (M + bsr_dim - 1) / bsr_dim;
    aoclsparse_int nb = (N + bsr_dim - 1) / bsr_dim;

    // Allocate memory for vectors
    std::vector<T> x(nb * bsr_dim);
    std::vector<T> y(mb * bsr_dim);
    std::vector<T> y_gold(mb * bsr_dim);

    // Initialize data
    aoclsparse_init<T>(x, 1, nb * bsr_dim, 1);
    aoclsparse_init<T>(y, 1, mb * bsr_dim, 1);
    y_gold = y;

    // Convert CSR to BSR
    aoclsparse_int              nnzb;
    std::vector<aoclsparse_int> bsr_row_ptr(mb + 1);

    CHECK_AOCLSPARSE_ERROR(aoclsparse_csr2bsr_nnz(
        M, N, descr, csr_row_ptr.data(), csr_col_ind.data(), bsr_dim, bsr_row_ptr.data(), &nnzb));

    std::vector<aoclsparse_int> bsr_col_ind(nnzb);
    std::vector<T>              bsr_val(nnzb * bsr_dim * bsr_dim);
    CHECK_AOCLSPARSE_ERROR(aoclsparse_csr2bsr<T>(M,
                                                 N,
                                                 descr,
                                                 order,
                                                 csr_val.data(),
                                                 csr_row_ptr.data(),
                                                 csr_col_ind.data(),
                                                 bsr_dim,
                                                 bsr_val.data(),
                                                 bsr_row_ptr.data(),
                                                 bsr_col_ind.data()));

    td.m            = mb;
    td.n            = nb;
    td.bsr_dim      = bsr_dim;
    td.csr_col_indA = bsr_col_ind;
    td.csr_row_ptrA = bsr_row_ptr;
    td.csr_valA     = bsr_val;
    td.x            = x;
    td.y            = y;
    td.alpha        = alpha;
    td.beta         = beta;
    td.descr        = descr;

    aoclsparse_matrix mtx;

    aoclsparse_create_bsr<T>(&mtx,
                             base,
                             order,
                             mb,
                             nb,
                             bsr_dim,
                             bsr_row_ptr.data(),
                             bsr_col_ind.data(),
                             bsr_val.data(),
                             false);

    if(arg.unit_check)
    {
        CHECK_AOCLSPARSE_ERROR(
            aoclsparse_mv<T>(trans, &alpha, mtx, descr, x.data(), &beta, y.data()));

        using compute_type = typename internal_t_map<T>::type;

        compute_type *beta_t      = reinterpret_cast<compute_type *>(&beta);
        compute_type *alpha_t     = reinterpret_cast<compute_type *>(&alpha);
        compute_type *y_gold_temp = reinterpret_cast<compute_type *>(y_gold.data());
        compute_type *val         = reinterpret_cast<compute_type *>(csr_val.data());

        // Reference SPMV CSR implementation
        for(int i = 0; i < M; i++)
        {
            compute_type result{};

            for(int j = csr_row_ptr[i] - base; j < csr_row_ptr[i + 1] - base; j++)
            {
                compute_type *x_temp = reinterpret_cast<compute_type *>(&x[csr_col_ind[j] - base]);

                result += (*alpha_t) * val[j] * (*x_temp);
            }

            y_gold_temp[i] = (*beta_t * y_gold_temp[i]) + result;
        }
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
            if(near_check_general<T>(1, M, 1, y_gold.data(), td.y.data()))
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

    aoclsparse_destroy(&mtx);

    return status;
}

#endif // TESTING_CSRMV_HPP
