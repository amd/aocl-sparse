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
    std::cout
        << "------------------------------------------------" << std::endl
        << " Symmetric Gauss Seidel Preconditioner followed by sparse matrix-vector multiplication"
        << std::endl
        << " sample program for complex data type" << std::endl
        << "------------------------------------------------" << std::endl
        << std::endl;

    /*
     * This example illustrates how to use Symmetric Gauss Seidel API to solve
     * linear system of equations. The complex matrix used is
     *     | 111+11.1i     2+0.2i          0          0 |
     * A = |    2+0.2i  111+11.1i     2+0.2i          0 |
     *     |         0     2+0.2i  111+11.1i     2+0.2i |
     *     |         0          0     2+0.2i  111+11.1i |
     */

    // Create a tri-diagonal matrix in CSR format
    const aoclsparse_int                   n = 4, m = 4, nnz = 10;
    aoclsparse_int                         icrow[5] = {0, 2, 5, 8, 10};
    aoclsparse_int                         icol[10] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<aoclsparse_double_complex> aval;
    // clang-format off
    aval.assign({{111,11.1}, {2,0.2}, {2,0.2}, {111,11.1},
                 {2,0.2}, {2,0.2}, {111,11.1}, {2,0.2},
                 {2,0.2}, {111,11.1}});
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

    std::vector<aoclsparse_double_complex> x, b, y, xref, yref;
    x.assign({{1, 0.1}, {1, 0.1}, {1, 0.1}, {1, 0.1}});
    b.assign({{113.850000, 23.000000},
              {227.700000, 46.000000},
              {341.550000, 69.000000},
              {445.500000, 90.000000}});
    y.assign({{0, 0.0}, {0, 0.0}, {0, 0.0}, {0, 0.0}});
    xref.assign({{1.0000056, 0.1000006},
                 {1.9996866, 0.1999687},
                 {2.9993739, 0.2999374},
                 {3.9990376, 0.3999038}});
    yref.assign({{113.8500000, 23.0000000},
                 {227.6643355, 45.9927951},
                 {341.4786710, 68.9855901},
                 {445.3930073, 89.9783853}});

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
    // Solve
    status = aoclsparse_zsymgs_mv(trans, A, descr_a, alpha, &b[0], &x[0], &y[0]);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_zsymgs, status = " << status << "."
                  << std::endl;
        return 3;
    }
    // Print and check the result
    std::cout << "Solving A x = alpha b" << std::endl;
    std::cout << "  where x and b are dense vectors" << std::endl;
    std::cout << "  Solution vector x = " << std::endl;
    std::cout << std::fixed;
    std::cout.precision(1);
    ok = true;
    for(int i = 0; i < m; ++i)
    {
        oki = std::abs(x[i].real - xref[i].real) <= exp_tol
              && std::abs(x[i].imag - xref[i].imag) <= exp_tol;
        std::cout << "(" << x[i].real << ", " << x[i].imag << "i) " << (oki ? "  " : "! ");
        ok &= oki;
    }
    std::cout << std::endl;
    std::cout << "  Product vector x = " << std::endl;
    for(int i = 0; i < m; ++i)
    {
        oki = std::abs(y[i].real - yref[i].real) <= exp_tol
              && std::abs(y[i].imag - yref[i].imag) <= exp_tol;
        std::cout << "(" << y[i].real << ", " << y[i].imag << "i) " << (oki ? "  " : "! ");
        ok &= oki;
    }
    std::cout << std::endl;
    // Destroy the aoclsparse memory
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(&A);

    return ok ? 0 : 5;
}
