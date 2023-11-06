/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main(void)
{
    std::cout << "------------------------------------------------" << std::endl
              << " Symmetric Gauss Seidel Preconditioner" << std::endl
              << " sample program for complex data type" << std::endl
              << "------------------------------------------------" << std::endl
              << std::endl;

    /*
     * This example illustrates how to use Symmetric Gauss Seidel API for complex data
        with an iterative solver, here we have simulated that effect using a loop.
        The complex matrix used is
        A = [
                   10+i  -1-0.1i    2+0.2i        0;
                -1-0.1i  11+1.1i   -1-0.1i   3+0.3i;
                 2+0.2i  -1-0.1i      10+i  -1-0.1i;
                      0   3+0.3i   -1-0.1i   8+0.8i;
            ]
     */

    // Create a tri-diagonal matrix in CSR format
    const aoclsparse_int                   n = 4, m = 4, nnz = 14;
    aoclsparse_int                         icrow[5] = {0, 3, 7, 11, 14};
    aoclsparse_int                         icol[14] = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3};
    std::vector<aoclsparse_double_complex> aval;
    // clang-format off
    aval.assign({{10,1}, {-1,-0.1}, {2,0.2}, {-1,-0.1},
                 {11,1.1}, {-1,-0.1}, {3,0.3}, {2,0.2},
                 {-1,-0.1}, {10,1}, {-1,-0.1}, {3, 0.3},
                 {-1,-0.1}, {8, 0.8}});
    // clang-format on
    aoclsparse_status status;
    bool              oki, ok;

    const double macheps      = std::numeric_limits<double>::epsilon();
    const double safe_macheps = (double)2.0 * macheps;
    double       exp_tol      = 20.0 * sqrt(safe_macheps);

    // Create aoclsparse matrix and its descriptor
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans;
    status = aoclsparse_create_zcsr(&A, base, m, n, nnz, icrow, icol, &aval[0]);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_create_dcsr, status = " << status << "."
                  << std::endl;
        return 3;
    }
    aoclsparse_create_mat_descr(&descr_a);

    aoclsparse_double_complex alpha = {1.0, 0.};

    std::vector<aoclsparse_double_complex> x, b, xref;
    x.assign({{1, 0.1}, {1, 0.1}, {1, 0.1}, {1, 0.1}});
    b.assign({{13.859999999999999, 2.7999999999999998},
              {29.699999999999999, 6},
              {25.739999999999998, 5.2000000000000002},
              {34.649999999999999, 7}});
    xref.assign({{0.99999999992700483, 0.099999999992700442},
                 {2.0000000217847944, 0.20000000217847949},
                 {3.0000000112573724, 0.30000000112573733},
                 {3.9999999237056727, 0.39999999237056733}});

    // Indicate to access the lower part of matrix A
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_symmetric);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    trans = aoclsparse_operation_none;

    status = aoclsparse_set_symgs_hint(A, trans, descr_a, 1);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_set_symgs_hint, status = " << status << "."
                  << std::endl;
        return 3;
    }
    status = aoclsparse_optimize(A);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_optimize, status = " << status << "."
                  << std::endl;
        return 3;
    }
    // Solve till convergence
    std::cout << std::setw(12) << " iter";
    for(int i = 0; i < n; i++)
        std::cout << std::setw(33) << "x[" << i << "]";
    std::cout << std::endl;
    for(int i = 0; i < 8; i++)
    {
        status = aoclsparse_zsymgs(trans, A, descr_a, alpha, &b[0], &x[0]);
        if(status != aoclsparse_status_success)
        {
            std::cerr << "Error returned from aoclsparse_zsymgs, status = " << status << "."
                      << std::endl;
            return 3;
        }
        std::cout << std::setw(12) << (int)i << " ";
        for(int j = 0; j < m; ++j)
        {
            std::cout << std::setw(8) << std::scientific << std::setprecision(6) << "(" << x[j].real
                      << ", " << x[j].imag << ")";
        }
        std::cout << std::endl;
    }
    // Print and check the result
    std::cout << "Solving A x = alpha b" << std::endl;
    std::cout << "  where x and b are dense vectors" << std::endl;
    std::cout << "  Solution found after 8 iterations, x = " << std::endl;
    std::cout << std::fixed;
    std::cout.precision(6);
    ok = true;
    for(int i = 0; i < m; ++i)
    {
        oki = std::abs(x[i].real - xref[i].real) <= exp_tol
              && std::abs(x[i].imag - xref[i].imag) <= exp_tol;
        std::cout << "(" << x[i].real << ", " << x[i].imag << "i) " << (oki ? "  " : "! ");
        ok &= oki;
    }
    std::cout << std::endl;

    // Destroy the aoclsparse memory
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(&A);

    return ok ? 0 : 5;
}
