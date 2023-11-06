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
#ifndef TESTING_SYMGS_HPP
#define TESTING_SYMGS_HPP

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
#include "ext_benchmarking.hpp" // defines register_tests_*()
#else
#include "aoclsparse_no_ext_benchmarking.hpp" // defines them as empty/do nothing
#endif

template <typename T>
int testing_symgs_aocl(const Arguments &arg, testdata<T> &td, double timings[])
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
    aoclsparse_matrix    A;
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
                                                            n,
                                                            nnz,
                                                            td.csr_row_ptrA.data(),
                                                            td.csr_col_indA.data(),
                                                            td.csr_valA.data()));
        //to identify hint id(which routine is to be executed, destroyed later)
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_symgs_hint(A, trans, descr, hint));
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_set_memory_hint(A, aoclsparse_memory_usage_minimal));

        // Optimize the matrix, "A"
        NEW_CHECK_AOCLSPARSE_ERROR(aoclsparse_optimize(A));

        //run for predecided iterations and then check for residual and convergence
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            double cpu_time_start = aoclsparse_clock();
            NEW_CHECK_AOCLSPARSE_ERROR(
                aoclsparse_symgs(trans, A, descr, td.alpha, td.b.data(), td.x.data()));
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
int testing_symgs(const Arguments &arg)
{
    int                    status  = 0;
    aoclsparse_int         m       = arg.M;
    aoclsparse_int         n       = arg.N;
    aoclsparse_int         nnz     = arg.nnz;
    aoclsparse_operation   trans   = arg.transA;
    aoclsparse_matrix_type mattype = arg.mattypeA;
    aoclsparse_fill_mode   fill    = arg.uplo;
    aoclsparse_diag_type   diag    = arg.diag;
    aoclsparse_index_base  base    = arg.baseA;
    aoclsparse_matrix_init mat     = arg.matrix;
    bool                   issymm;
    std::string            filename = arg.filename;
    std::vector<T>         x_gold; // reference result

    //std::vector<std::pair<std::string,testfunc>> testqueue;  // or std::map?
    std::vector<testsetting<T>> testqueue;
    testqueue.push_back({"aocl_symgs_hint", &testing_symgs_aocl<T>});
    register_tests_symgs(testqueue);

    // create relevant test data for this API
    testdata<T> td;

    // space for results
    std::vector<double> timings(arg.iters);

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

    if(mat == aoclsparse_matrix_random)
    {
        std::cerr
            << "SYMGS requires diagonally dominant and symmetric positive definite matrix for "
               "ensuring convergence."
            << std::endl
            << "Current implementation of random matrix generation does not support this property."
            << std::endl;
        return 2;
    }

    aoclsparse_init_csr_matrix(td.csr_row_ptrA,
                               td.csr_col_indA,
                               td.csr_valA,
                               m,
                               n,
                               nnz,
                               base,
                               mat,
                               filename.c_str(),
                               issymm,
                               true);

    td.m    = m;
    td.n    = n;
    td.nnzA = nnz;

    //exit since SYMGS expects a square matrix
    if(td.m != td.n)
    {
        std::cerr << "SYMGS requires a square matrix" << std::endl;
        return -1;
    }

    // Allocate memory for vectors
    td.x.resize(n);
    td.b.resize(m);
    x_gold.resize(n);

    //initialize x
    aoclsparse_init<T>(x_gold, 1, m, 1);

    //Generate RHS using a known "x_gold"
    NEW_CHECK_AOCLSPARSE_ERROR(ref_csrmv(trans,
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
                                         x_gold.data(),
                                         td.beta,
                                         td.b.data()));

    int number_hot_calls = arg.iters;
    //flops and bw is computed over total no of iterations the kernel ran.
    double total_iterations = 0, fnops_precond = 0;
    double fbytes_precond = 0;
    double cpu_gflops = 0.0, cpu_gbyte = 0.0;
    total_iterations = 2; //2 spmv calls and 2 trsv calls

    if(td.beta == aoclsparse_numeric::zero<T>())
    {
        // number of spmv flops
        fnops_precond += total_iterations * (2.0 * nnz);
        //spmv read/writes
        fbytes_precond += ((m + 1 + nnz) * sizeof(aoclsparse_int) + (m + n + nnz) * sizeof(T));
    }
    else
    {
        // number of spmv flops
        fnops_precond += total_iterations * (2.0 * nnz + m /*non-zero beta computations*/);
        //spmv read/writes
        fbytes_precond += ((m + 1 + nnz) * sizeof(aoclsparse_int)
                           + (m + n + nnz + m /*non-zero beta computations*/) * sizeof(T));
    }
    // number of trsv flops
    fnops_precond
        += total_iterations * (2.0 * nnz + m + (diag == aoclsparse_diag_type_non_unit ? m : 0));
    // trsv read/writes
    fbytes_precond += ((m + 1 + nnz) * sizeof(aoclsparse_int) + (m + n + nnz) * sizeof(T));

    fnops_precond  = fnops_precond / 1.0E9;
    fbytes_precond = fbytes_precond / 1.0E9;

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    for(size_t itest = 0; itest < testqueue.size(); ++itest)
    {
        std::cout << "-----" << testqueue[itest].name << "-----" << std::endl;

        // Run the test loop
        status += testqueue[itest].tf(arg, td, timings.data());

        // SYMGS is a precondtioner which fits into a larger Sparse Solver and
        // on it own, requires more iterations to converge depending upon the matrix properties.
        // Validation with a reference "x_gold" (as in A.x_gold = alpha.b) for a single iteration of SYMGS is not
        // an exact match, provided "x_gold" is the final convergence solution.

        // analyze the results - at the moment just take minimum as before
        double cpu_time_used = DBL_MAX;
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            cpu_time_used = (std::min)(cpu_time_used, timings[iter]);
        }

        // count flops
        cpu_gflops = fnops_precond / cpu_time_used;
        cpu_gbyte  = fbytes_precond / cpu_time_used;

        // store/print results
        std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
                  << std::setw(12) << "alpha" << std::setw(12) << "beta" << std::setw(12)
                  << "GFlop/s" << std::setw(12) << "GB/s" << std::setw(12) << "msec"
                  << std::setw(12) << "iter" << std::setw(12) << "verified" << std::endl;

        std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz
                  << std::setw(12) << arg.alpha << std::setw(12) << arg.beta << std::setw(12)
                  << cpu_gflops << std::setw(12) << cpu_gbyte << std::setw(12) << std::scientific
                  << cpu_time_used * 1e3 << std::setw(12) << number_hot_calls << std::setw(12)
                  << (arg.unit_check ? "yes" : "no") << std::endl;
    }
    return status;
}

#endif // TESTING_SYMGS_HPP
