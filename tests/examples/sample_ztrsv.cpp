/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <assert.h>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main()
{
    std::cout << "-------------------------------" << std::endl
              << " Triangle solve sample program" << std::endl
              << "-------------------------------" << std::endl
              << std::endl;

    /* Solve two complex linear systems of equations 
     * Lx = alpha b, and U^Tx = alpha b,
     * where L is the lower triangular part of A and
     * U is the upper triangular part of A. U^H is
     * the conjugate transpose of U (a lower triangular matrix).
     * The complex matrix A is the tri-diagonal matrix in CSR format
     * 
     * | 1+3i  2+5i     0     0 |
     * | 3     1+2i  2-2i     0 |
     * |    0     3     1  2+3i |
     * |    0     0  3+1i  1+1i |
     */
    const aoclsparse_int                   n = 4, m = 4, nnz = 10;
    aoclsparse_int                         icrow[5] = {0, 2, 5, 8, 10};
    aoclsparse_int                         icol[18] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<aoclsparse_double_complex> aval     = {{1., 3.},
                                                       {2., 5.},
                                                       {3., 0.},
                                                       {1., 2.},
                                                       {2., -2.},
                                                       {3., 0.},
                                                       {1., 0.},
                                                       {2., 3.},
                                                       {3., 1.},
                                                       {1., 1.}};
    aoclsparse_status                      status;
    std::vector<aoclsparse_double_complex> ref;

    // create aoclsparse matrix and its descriptor
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    status = aoclsparse_create_zcsr(A, base, m, n, nnz, icrow, icol, aval.data());
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error creating the matrix, status = " << status << std::endl;
        return 1;
    }
    aoclsparse_create_mat_descr(&descr_a);

    /* Solve the lower triangular system Lx = b, 
     * here alpha=1 and b = [1+i, 4+2i, 4+i, 4].
     */
    std::vector<aoclsparse_double_complex> b = {{1., 1.}, {4., 2.}, {4., 1}, {4., 0}};
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_triangular);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    status = aoclsparse_set_sv_hint(A, trans, descr_a, 1);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error setting SV hint, status = " << status << std::endl;
        return 1;
    }
    status = aoclsparse_optimize(A);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_optimize, status = " << status << "."
                  << std::endl;
        return 2;
    }

    // Call the triangle solver
    aoclsparse_double_complex alpha = {1.0, 0.};
    aoclsparse_double_complex x[n];
    status = aoclsparse_ztrsv(trans, alpha, A, descr_a, b.data(), x);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_ztrsv, status = " << status << "."
                  << std::endl;
        return 3;
    }

    // Print and check the result
    double tol = 20 * std::numeric_limits<double>::epsilon();
    ref.assign({{0.4, -0.2}, {1.6, -0.6}, {-0.8, 2.8}, {0.8, -8.4}});
    std::cout << "Solving Lx = alpha b: " << std::endl << "  x = ";
    std::cout << std::fixed;
    std::cout.precision(1);
    bool oki, ok = true;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        oki = std::abs(x[i].real - ref[i].real) <= tol && std::abs(x[i].imag - ref[i].imag) <= tol;
        std::cout << "(" << x[i].real << ", " << x[i].imag << "i) " << (oki ? " " : "!  ");
        ok &= oki;
    }
    std::cout << std::endl;

    /* Solve the lower triangular system U^Hx = b, 
     * here alpha=1-i and b is unchanged.
     */
    alpha = {1, -1};
    // Indicate to use only the conjugate transpose of the upper part of A
    trans = aoclsparse_operation_conjugate_transpose;
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_upper);
    status = aoclsparse_ztrsv(trans, alpha, A, descr_a, b.data(), x);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_ztrsv, status = " << status << "."
                  << std::endl;
        return 3;
    }

    // Print and check the result
    ref.assign({{0.2, 0.6}, {1.4, 0.6}, {3.4, -7.0}, {-1.0, 19.2}});
    std::cout << std::endl;
    std::cout << "Solving U^Hx = alpha*b: " << std::endl << "  x = ";
    for(aoclsparse_int i = 0; i < n; i++)
    {
        oki = std::abs(x[i].real - ref[i].real) <= tol && std::abs(x[i].imag - ref[i].imag) <= tol;
        std::cout << "(" << x[i].real << ", " << x[i].imag << "i) " << (oki ? " " : "!  ");
        ok &= oki;
    }
    std::cout << std::endl;

    // Destroy the aoclsparse memory
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(A);

    return ok ? 0 : 4;
}
