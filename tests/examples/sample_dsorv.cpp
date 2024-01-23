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
    std::cout << "------------------------------------------------" << std::endl
              << " Successive over-relaxation Preconditioner" << std::endl
              << " sample program for real data type" << std::endl
              << "------------------------------------------------" << std::endl
              << std::endl;

    /*
     * This example illustrates how to use Successive over-relaxation API to solve
     * linear system of equations. The real matrix used is
     *     |  111.1   2.345    0.0      0.0   |
     * A = |   3.12   9.87    -56.2     0.0   |
     *     |   0.0     0.0   -39.678    0.0   |
     *     | 32.0987  76.9   -903.40  -25.106 |
     */

    // Create a tri-diagonal matrix in CSR format
    const aoclsparse_int n = 4, m = 4, nnz = 10;
    aoclsparse_int       icrow[5] = {0, 2, 5, 6, 10};
    aoclsparse_int       icol[10] = {0, 1, 0, 1, 2, 2, 1, 3, 2, 0};
    double aval[10] = {111.1, 2.345, 3.12, 9.87, -56.2, -39.678, 76.9, -25.106, -903.40, 32.0987};
    aoclsparse_status   status;
    bool                oki, ok;
    double              omega = 0.5;
    double              alpha = 1.0;
    double              x[m]  = {1, -41, 5.7, 0.341};
    double              b[m]  = {157.5045, -489.033, -321.3918, -7433.72955};
    std::vector<double> xref
        = {1.6415369036903691, -29.305197322163828, 6.9000000000000004, -19.757185084519875};
    const double macheps      = std::numeric_limits<double>::epsilon();
    const double safe_macheps = (double)2.0 * macheps;
    double       exp_tol      = 20.0 * sqrt(safe_macheps);

    // Create aoclsparse matrix and its descriptor
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    status = aoclsparse_create_dcsr(&A, base, m, n, nnz, icrow, icol, aval);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_create_dcsr, status = " << status << "."
                  << std::endl;
        return 3;
    }
    aoclsparse_create_mat_descr(&descr_a);
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_general);

    status = aoclsparse_set_sorv_hint(A, descr_a, aoclsparse_sor_forward, 1);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_set_sorv_hint, status = " << status << "."
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
    status = aoclsparse_dsorv(aoclsparse_sor_forward, descr_a, A, omega, alpha, x, b);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dsorv, status = " << status << "."
                  << std::endl;
        return 3;
    }
    // Print and check the result
    std::cout << "Solving A x = b where x and b are dense vectors" << std::endl;
    std::cout << "Solution vector x = " << std::endl;
    std::cout << std::fixed;
    std::cout.precision(12);
    ok = true;
    for(int i = 0; i < m; ++i)
    {
        oki = std::abs(x[i] - xref[i]) <= exp_tol;
        std::cout << x[i] << (oki ? "  " : "! ");
        ok &= oki;
    }
    std::cout << std::endl;

    // Destroy the aoclsparse memory
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(&A);

    return ok ? 0 : 5;
}