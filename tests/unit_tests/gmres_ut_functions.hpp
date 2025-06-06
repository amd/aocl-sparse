/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

#include "aoclsparse.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"

#include <cfloat>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>

#define VERBOSE 1

typedef struct
{
    std::string option;
    std::string value;
} itsol_opts;

// precond template ('type') alias
template <typename T>
using PrecondType
    = aoclsparse_int (*)(aoclsparse_int flag, aoclsparse_int n, const T *u, T *v, void *udata);

// precond template ('type') alias
template <typename T>
using MonitType
    = aoclsparse_int (*)(aoclsparse_int n, const T *x, const T *r, T rinfo[100], void *udata);

// Identity preconditioner
template <typename T>
aoclsparse_int precond_identity([[maybe_unused]] aoclsparse_int flag,
                                aoclsparse_int                  n,
                                const T                        *u,
                                T                              *v,
                                [[maybe_unused]] void          *udata)
{
    for(aoclsparse_int i = 0; i < n; i++)
        v[i] = u[i];

    return 0;
}

// Dummy preconditioner - only to be used as an argument but it shouldn't be called
template <typename T>
aoclsparse_int precond_dummy([[maybe_unused]] aoclsparse_int flag,
                             [[maybe_unused]] aoclsparse_int n,
                             [[maybe_unused]] const T       *u,
                             [[maybe_unused]] T             *v,
                             [[maybe_unused]] void          *udata)
{
    // request stop if called as it shouldn't be called
    return 1;
}

// Dummy monitoring function doing nothing
template <typename T>
aoclsparse_int monit_dummy([[maybe_unused]] aoclsparse_int n,
                           [[maybe_unused]] const T       *x,
                           [[maybe_unused]] const T       *r,
                           [[maybe_unused]] T              rinfo[100],
                           [[maybe_unused]] void          *udata)
{
    return 0;
}

// Monitoring function printing progress
template <typename T>
aoclsparse_int monit_print([[maybe_unused]] aoclsparse_int n,
                           [[maybe_unused]] const T       *x,
                           [[maybe_unused]] const T       *r,
                           tolerance_t<T>                  rinfo[100],
                           [[maybe_unused]] void          *udata)
{
    std::cout << "Iteration " << (aoclsparse_int)rinfo[30] << ": rel. tolerance "
              << rinfo[0] / rinfo[1] << ", abs. tolerance " << rinfo[0] << std::endl;

    return 0;
}

// Monitoring function which requires stop at 2nd iteration
template <typename T>
aoclsparse_int monit_stopit2([[maybe_unused]] aoclsparse_int n,
                             [[maybe_unused]] const T       *x,
                             [[maybe_unused]] const T       *r,
                             T                               rinfo[100],
                             [[maybe_unused]] void          *udata)
{
    if(rinfo[30] > 1)
        // Test user stop
        return 1;

    return 0;
}
// Monitoring function requiring stop at predefined tolerance
template <typename T>
aoclsparse_int monit_tolstop([[maybe_unused]] aoclsparse_int n,
                             [[maybe_unused]] const T       *x,
                             [[maybe_unused]] const T       *r,
                             T                               rinfo[100],
                             [[maybe_unused]] void          *udata)
{
    T tol = expected_precision<T>(100);

    if(rinfo[30] > 1 && rinfo[0] < tol) // check for (premature) stop
        return 1; // Request user stop

    return 0;
}

template <typename T>
bool check_for_residual_tolerance(T &residual, T &rhs_relative_tol, T &abs_tolerance)
{
    bool result;
    //printf("residual = %e, abs_tolerance = %e, rel_tolerance = %e, (resid > tol) = %d\n",
    //        residual, abs_tolerance, rhs_relative_tol,
    //        (residual > abs_tolerance));
    //fflush(stdout);
    if(residual < abs_tolerance) // check for premature stop abs_tolerance
    {
        result = true;
    }
    else
    {
        result = false;
        //check if relative tolerance is reached, instead of absolute tolerance
        if(residual < rhs_relative_tol)
        {
            result = true;
        }
        else
        {
            result = false;
        }
    }
    return result;
}

template <typename T>
void test_gmres(matrix_id               mid,
                std::vector<itsol_opts> opts       = {},
                PrecondType<T>          precond    = nullptr,
                MonitType<T>            monit      = nullptr,
                aoclsparse_status       status_exp = aoclsparse_status_success)
{
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<T>              csr_val;
    aoclsparse_int              m, n, nnz;
    // Create aocl sparse matrix
    aoclsparse_matrix    A;
    aoclsparse_mat_descr descr;
    void                *udata = nullptr;
    T                    tol;
    // Initialize initial point x0 and right hand side b
    T                      *expected_sol = NULL;
    T                      *x            = NULL;
    T                      *b            = NULL;
    T                       init_x = 1.0, ref_x = 0.5;
    T                       norm = 0.0;
    T                       rinfo[100];
    T                       alpha = 1.0, beta = 0.;
    aoclsparse_itsol_handle handle = NULL;

    tol = expected_precision<T>(1.0);

    // create GMRES handle
    itsol_init<T>(&handle);

    ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
    ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
              aoclsparse_status_success);
    //test for small unsymmetric matrix with no preconditioner
    ASSERT_EQ(create_matrix(mid, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, A, descr, VERBOSE),
              aoclsparse_status_success);

    expected_sol = new T[n];

    b = new T[n];

    x = new T[n];

    for(int i = 0; i < n; i++)
    {
        expected_sol[i] = ref_x;
        x[i]            = init_x;
    }
    // add options
    for(auto op : opts)
    {
        EXPECT_EQ(aoclsparse_itsol_option_set(handle, op.option.c_str(), op.value.c_str()),
                  aoclsparse_status_success)
            << "Options " << op.option << "could not be set to " << op.value << std::endl;
    }

    EXPECT_EQ(aoclsparse_mv(aoclsparse_operation_none, &alpha, A, descr, expected_sol, &beta, b),
              aoclsparse_status_success);

    // Call GMRES solver
    EXPECT_EQ(itsol_solve(handle, n, A, descr, b, x, rinfo, precond, monit, &udata), status_exp);

    if(status_exp == aoclsparse_status_success)
    {
        // test expected solution
        EXPECT_TRUE(check_for_residual_tolerance<T>(rinfo[0], rinfo[1], tol));
    }

    //residual = xSol - x
    //xSol obtained by initial spmv
    //x obtained by ILU-GMRES iterations
    for(aoclsparse_int i = 0; i < n; i++)
    {
        T a = expected_sol[i] - x[i];
        norm += a * a;
    }
    norm = sqrt(norm);

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
              << std::setw(16) << "expected_tol" << std::setw(12) << "residual" << std::setw(18)
              << "iters to converge" << std::setw(16) << "L2Norm on x" << std::endl;

    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz << std::setw(16)
              << std::scientific << tol << std::setw(12) << std::scientific << rinfo[0]
              << std::setw(18) << (int)rinfo[30] << std::setw(16) << std::scientific << norm
              << std::endl;

    delete[] expected_sol;
    delete[] b;
    delete[] x;

    aoclsparse_itsol_destroy(&handle);
    EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    printf("\n");
    fflush(stdout);

    return;
}

//complex GMRES tests
template <typename T>
void test_dcomplex_gmres(aoclsparse_int tolerance_scale)
{
    aoclsparse_status           status_exp = aoclsparse_status_success;
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<T>              csr_val;
    aoclsparse_int              m, n, nnz;
    // Create aocl sparse matrix
    aoclsparse_matrix    A;
    aoclsparse_mat_descr descr;
    void                *udata = nullptr;
    tolerance_t<T>       tol;
    // Initialize initial point x0 and right hand side b
    std::vector<T> expected_sol;
    std::vector<T> x;
    std::vector<T> b;
    T              alpha, beta;

    tolerance_t<T> rinfo[100];
    tol = expected_precision<tolerance_t<T>>((tolerance_t<T>)tolerance_scale);

    //complex
    alpha = {1.0, 0.0};
    beta  = {0.0, 0.0};

    n           = 9;
    m           = 9;
    nnz         = 49;
    csr_row_ptr = {0, 5, 10, 15, 20, 26, 32, 38, 44, 49};
    csr_col_ind = {0, 1, 3, 4, 7, 0, 1, 2, 4, 5, 1, 2, 3, 5, 6, 0, 2, 3, 6, 7, 0, 1, 4, 5, 6,
                   8, 1, 2, 4, 5, 7, 8, 2, 3, 4, 6, 7, 8, 0, 3, 5, 6, 7, 8, 4, 5, 6, 7, 8};
    csr_val
        = {{0.75, 0.075}, {0.14, 0.014}, {0.11, 0.011}, {0.14, 0.014}, {0.11, 0.011}, {0.08, 0.008},
           {0.69, 0.069}, {0.11, 0.011}, {0.08, 0.008}, {0.11, 0.011}, {0.09, 0.009}, {0.67, 0.067},
           {0.08, 0.008}, {0.09, 0.009}, {0.08, 0.008}, {0.09, 0.009}, {0.14, 0.014}, {0.73, 0.073},
           {0.14, 0.014}, {0.09, 0.009}, {0.04, 0.004}, {0.04, 0.004}, {0.54, 0.054}, {0.14, 0.014},
           {0.11, 0.011}, {0.25, 0.025}, {0.05, 0.005}, {0.05, 0.005}, {0.08, 0.008}, {0.45, 0.045},
           {0.08, 0.008}, {0.15, 0.015}, {0.04, 0.004}, {0.04, 0.004}, {0.09, 0.009}, {0.47, 0.047},
           {0.09, 0.009}, {0.18, 0.018}, {0.05, 0.005}, {0.05, 0.005}, {0.14, 0.014}, {0.11, 0.011},
           {0.55, 0.055}, {0.25, 0.025}, {0.08, 0.008}, {0.08, 0.008}, {0.09, 0.009}, {0.08, 0.008},
           {0.17, 0.017}};

    expected_sol.resize(n, {0.5, 0.05});
    x.resize(n, {1.0, 0.1});
    b.resize(n, {0.0, 0.0});

    // create GMRES handle
    aoclsparse_itsol_handle handle = NULL;
    if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
    {
        aoclsparse_itsol_z_init(&handle);
    }
    else if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
    {
        aoclsparse_itsol_c_init(&handle);
    }

    ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
    ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
              aoclsparse_status_success);
    ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general),
              aoclsparse_status_success);

    //test for small unsymmetric matrix with no preconditioner
    ASSERT_EQ(aoclsparse_create_csr<T>(
                  &A, descr->base, m, n, nnz, &csr_row_ptr[0], &csr_col_ind[0], &csr_val[0]),
              aoclsparse_status_success);

    // add options
    EXPECT_EQ(aoclsparse_itsol_option_set(handle, "iterative method", "GMRES"),
              aoclsparse_status_success);
    EXPECT_EQ(aoclsparse_itsol_option_set(handle, "gmres preconditioner", "ILU0"),
              aoclsparse_status_success);
    EXPECT_EQ(aoclsparse_itsol_option_set(handle, "gmres iteration limit", "50"),
              aoclsparse_status_success);
    EXPECT_EQ(aoclsparse_itsol_option_set(handle, "gmres restart iterations", "6"),
              aoclsparse_status_success);
    char abs_tolerance_string[16];
    sprintf(abs_tolerance_string, "%e", tol);
    EXPECT_EQ(aoclsparse_itsol_option_set(handle, "gmres abs tolerance", abs_tolerance_string),
              aoclsparse_status_success);

    ASSERT_EQ(ref_csrmv(aoclsparse_operation_none,
                        alpha,
                        m,
                        n,
                        &csr_val[0],
                        &csr_col_ind[0],
                        &csr_row_ptr[0],
                        aoclsparse_matrix_type_general,
                        aoclsparse_fill_mode_lower,
                        aoclsparse_diag_type_non_unit,
                        aoclsparse_index_base_zero,
                        &expected_sol[0],
                        beta, //beta
                        &b[0]),
              aoclsparse_status_success);

    if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
    {
        EXPECT_EQ(aoclsparse_itsol_z_solve(
                      handle, n, A, descr, &b[0], &x[0], rinfo, NULL, monit_print, udata),
                  status_exp);
    }
    else if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
    {
        EXPECT_EQ(aoclsparse_itsol_c_solve(
                      handle, n, A, descr, &b[0], &x[0], rinfo, NULL, monit_print, udata),
                  status_exp);
    }

    if(status_exp == aoclsparse_status_success)
    {
        // test expected solution
        EXPECT_TRUE(check_for_residual_tolerance<tolerance_t<T>>(rinfo[0], rinfo[1], tol));
    }

    tolerance_t<T> norm = 0.0f, norm_diff = 0.0f;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        aoclsparse_double_complex diff;
        diff.real = expected_sol[i].real - x[i].real;
        diff.imag = expected_sol[i].imag - x[i].imag;
        norm      = sqrt(pow(diff.real, 2) + pow(diff.imag, 2));
        norm_diff += pow(norm, 2);
    }
    norm = sqrt(norm);

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    std::cout << std::setw(12) << "M" << std::setw(12) << "N" << std::setw(12) << "nnz"
              << std::setw(16) << "expected_tol" << std::setw(12) << "residual" << std::setw(18)
              << "iters to converge" << std::setw(16) << "L2Norm on x" << std::endl;

    std::cout << std::setw(12) << m << std::setw(12) << n << std::setw(12) << nnz << std::setw(16)
              << std::scientific << tol << std::setw(12) << std::scientific << rinfo[0]
              << std::setw(18) << (int)rinfo[30] << std::setw(16) << std::scientific << norm
              << std::endl;

    aoclsparse_itsol_destroy(&handle);
    EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);

    return;
}
