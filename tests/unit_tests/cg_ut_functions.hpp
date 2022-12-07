/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "gtest/gtest.h"
#include "solver_data_utils.h"
#include "aoclsparse.hpp"
#include "aoclsparse_itsol_functions.hpp"

#include <iostream>

#define VERBOSE 1

#define EXPECT_ARR_NEAR(n, x, y, abs_error) \
    for(int j = 0; j < (n); j++)            \
    EXPECT_NEAR((x[j]), (y[j]), abs_error)  \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

// Helper to define precision to which we expect the results match
template <typename T>
T expected_precision(T scale = (T)1.0);
template <>
double expected_precision<double>(double scale)
{
    return scale * 1e-5;
}

template <>
float expected_precision<float>(float scale)
{
    return scale * 5e-1f; // TODO FIXME
}

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
aoclsparse_int
    precond_identity(aoclsparse_int flag, aoclsparse_int n, const T *u, T *v, void *udata)
{
    for(aoclsparse_int i = 0; i < n; i++)
        v[i] = u[i];

    return 0;
}

// Dummy preconditioner - only to be used as an argument but it shouldn't be called
template <typename T>
aoclsparse_int precond_dummy(aoclsparse_int flag, aoclsparse_int n, const T *u, T *v, void *udata)
{
    // request stop if called as it shouldn't be called
    return 1;
}

// Dummy monitoring function doing nothing
template <typename T>
aoclsparse_int monit_dummy(aoclsparse_int n, const T *x, const T *r, T rinfo[100], void *udata)
{
    return 0;
}

// Monitoring function printing progress
template <typename T>
aoclsparse_int monit_print(aoclsparse_int n, const T *x, const T *r, T rinfo[100], void *udata)
{
    std::cout << "Iteration " << (aoclsparse_int)rinfo[30] << ": rel. tolerance "
              << rinfo[0] / rinfo[1] << ", abs. tolerance " << rinfo[0] << std::endl;

    return 0;
}

// Monitoring function which requires stop at 2nd iteration
template <typename T>
aoclsparse_int monit_stopit2(aoclsparse_int n, const T *x, const T *r, T rinfo[100], void *udata)
{
    if(rinfo[30] > 1)
        // Test user stop
        return 1;

    return 0;
}

// Monitoring function requiring stop at predefined tolerance
template <typename T>
aoclsparse_int monit_tolstop(aoclsparse_int n, const T *x, const T *r, T rinfo[100], void *udata)
{
    T tol = expected_precision<T>(0.1);

    if(rinfo[30] > 1 && rinfo[0] < tol) // check for (premature) stop
        return 1; // Request user stop

    return 0;
}

template <typename T>
void test_cg_error(aoclsparse_status       status_exp,
                   matrix_id               mid,
                   std::vector<itsol_opts> opts    = {},
                   PrecondType<T>          precond = nullptr,
                   MonitType<T>            monit   = nullptr)
{
    aoclsparse_matrix           A;
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<T>              aval, b, x;
    aoclsparse_int              n, m, nnz;
    aoclsparse_mat_descr        descr;
    T                           rinfo[100];
    void                       *udata = nullptr;

    ASSERT_EQ(create_matrix(mid, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE),
              aoclsparse_status_success);

    // Create the iteartive solver handle
    aoclsparse_itsol_handle handle = nullptr;
    itsol_init<T>(&handle);

    // Initialize rhs and initial point
    b.resize(n);
    x.resize(n);
    for(aoclsparse_int i = 0; i < n; i++)
    {
        x[i] = b[i] = 1.0;
    }

    // add options
    for(auto op : opts)
    {
        EXPECT_EQ(aoclsparse_itsol_option_set(handle, op.option.c_str(), op.value.c_str()),
                  aoclsparse_status_success)
            << "Options " << op.option << "could not be set to " << op.value << std::endl;
    }

    // Call the CG solver - expect the given error
    EXPECT_EQ(itsol_solve(handle, n, A, descr, &b[0], &x[0], rinfo, precond, monit, &udata),
              status_exp)
        << "return condition not as expected";

    aoclsparse_destroy(A);
    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr);
}

template <typename T>
void test_cg_double_call()
{
    aoclsparse_matrix           A;
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<T>              aval, b, x, x_exp;
    aoclsparse_int              n, m, nnz;
    aoclsparse_mat_descr        descr;
    T                           rinfo[100];
    void                       *udata   = nullptr;
    PrecondType<T>              precond = nullptr;
    MonitType<T>                monit   = nullptr;

    matrix_id mid = sample_cg_mat;
    ASSERT_EQ(create_matrix(mid, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE),
              aoclsparse_status_success);
    b = {1, 1, 1, 1, 1, 1, 1, 1};
    x = {1, 1, 1, 1, 1, 1, 1, 1};

    // Create the iteartive solver handle
    aoclsparse_itsol_handle handle = nullptr;
    itsol_init<T>(&handle);

    // Call the CG solver for the first time
    EXPECT_EQ(itsol_solve(handle, n, A, descr, &b[0], &x[0], rinfo, precond, monit, udata),
              aoclsparse_status_success);

    // Initialize rhs and initial point for the second call
    T alpha = 1.0, beta = 0.0;
    b.resize(n);
    x.resize(n);
    x_exp.resize(n);
    for(aoclsparse_int i = 0; i < n; i++)
    {
        x_exp[i] = i;
        x[i]     = 1.0;
    }
    EXPECT_EQ(aoclsparse_mv(aoclsparse_operation_none, &alpha, A, descr, &x_exp[0], &beta, &b[0]),
              aoclsparse_status_success);

    // Second call of the CG solver
    EXPECT_EQ(itsol_solve(handle, n, A, descr, &b[0], &x[0], rinfo, precond, monit, udata),
              aoclsparse_status_success);

    // test expected solution
    EXPECT_ARR_NEAR(n, x_exp, x, expected_precision<T>());

    aoclsparse_destroy(A);
    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr);
}

// Test of successful solve via CGM
// given an input matrix, options, preconditioner and monitoring,
// compute system's RHS to have solution vector [0,1,...,n-1],
// test for absolute difference element-wise of the computed solution
template <typename T>
void test_cg_positive(matrix_id               mid,
                      std::vector<itsol_opts> opts       = {},
                      PrecondType<T>          precond    = nullptr,
                      MonitType<T>            monit      = nullptr,
                      aoclsparse_status       status_exp = aoclsparse_status_success)
{
    aoclsparse_matrix           A;
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<T>              aval, b, x, x_exp;
    aoclsparse_int              n, m, nnz;
    aoclsparse_mat_descr        descr;
    T                           rinfo[100];
    void                       *udata = nullptr;

    // Create the iteartive solver handle
    aoclsparse_itsol_handle handle = nullptr;
    itsol_init<T>(&handle);

    ASSERT_EQ(create_matrix(mid, m, n, nnz, icrow, icol, aval, A, descr, VERBOSE),
              aoclsparse_status_success);

    // Initialize rhs and initial point
    T alpha = 1.0, beta = 0.0;
    b.resize(n);
    x.resize(n);
    x_exp.resize(n);
    for(aoclsparse_int i = 0; i < n; i++)
    {
        x_exp[i] = i;
        x[i]     = 1.0;
    }
    EXPECT_EQ(aoclsparse_mv(aoclsparse_operation_none, &alpha, A, descr, &x_exp[0], &beta, &b[0]),
              aoclsparse_status_success);

    // add options
    for(auto op : opts)
    {
        EXPECT_EQ(aoclsparse_itsol_option_set(handle, op.option.c_str(), op.value.c_str()),
                  aoclsparse_status_success)
            << "Options " << op.option << "could not be set to " << op.value << std::endl;
    }

    // Call the CG solver
    EXPECT_EQ(itsol_solve(handle, n, A, descr, &b[0], &x[0], rinfo, precond, monit, udata),
              status_exp);

    // test expected solution
    EXPECT_ARR_NEAR(n, x_exp, x, expected_precision<T>());

    aoclsparse_destroy(A);
    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr);
}