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
        << " sample program for real data type" << std::endl
        << "------------------------------------------------" << std::endl
        << std::endl;

    /*
     * This example illustrates how to use Symmetric Gauss Seidel API to solve
     * linear system of equations. The real matrix used is
     *     | 111  2  0  0 |
     * A = | 2  111  2  0 |
     *     | 0  2  111  2 |
     *     | 0  0  2  111 |
     */

    // Create a tri-diagonal matrix in CSR format
    const aoclsparse_int n = 4, m = 4, nnz = 10;
    aoclsparse_int       icrow[5] = {0, 2, 5, 8, 10};
    aoclsparse_int       icol[10] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    double               aval[10] = {111, 2, 2, 111, 2, 2, 111, 2, 2, 111};
    aoclsparse_status    status;
    bool                 oki, ok;

    const double macheps      = std::numeric_limits<double>::epsilon();
    const double safe_macheps = (double)2.0 * macheps;
    double       exp_tol      = 20.0 * sqrt(safe_macheps);

    // Create aoclsparse matrix and its descriptor
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans;
    status = aoclsparse_create_dcsr(&A, base, m, n, nnz, icrow, icol, aval);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_create_dcsr, status = " << status << "."
                  << std::endl;
        return 3;
    }
    aoclsparse_create_mat_descr(&descr_a);

    double alpha = 1.0;
    double x[m]  = {1, 1, 1, 1};
    double y[m]  = {0, 0, 0, 0};

    double              b[m] = {115, 230, 345, 450};
    std::vector<double> xref;
    xref.assign({1.000006, 1.999687, 2.999374, 3.999038});
    std::vector<double> yref;
    yref.assign({115.0000000, 229.9639753, 344.9279505, 449.8919266});

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
    status = aoclsparse_dsymgs_mv(trans, A, descr_a, alpha, b, x, y);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dsymgs_mv, status = " << status << "."
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
        oki = std::abs(x[i] - xref[i]) <= exp_tol;
        std::cout << x[i] << (oki ? "  " : "! ");
        ok &= oki;
    }
    std::cout << std::endl;
    std::cout << "  and Product vector y = " << std::endl;
    for(int i = 0; i < m; ++i)
    {
        oki = std::abs(y[i] - yref[i]) <= exp_tol;
        std::cout << y[i] << (oki ? "  " : "! ");
        ok &= oki;
    }
    std::cout << std::endl;

    // Destroy the aoclsparse memory
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(&A);

    return ok ? 0 : 5;
}
