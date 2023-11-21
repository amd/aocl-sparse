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

#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main(void)
{
    std::cout << "------------------------------------------------" << std::endl
              << " Triangle Solve with Multiple Right Hand Sides" << std::endl
              << " sample program for complex data type" << std::endl
              << "------------------------------------------------" << std::endl
              << std::endl;

    /*
     * This example illustrates how to solve two triangular
     * linear system of equations. The complex matrix used is
     *     | 1+3i  2+5i     0     0 |
     * A = | 3     1+2i  2-2i     0 |
     *     |    0     3     1  2+3i |
     *     |    0     0  3+1i  1+1i |
     *
     * The first linear system is L X = alpha * B with L = tril(A), X and B
     * are two dense matrices of size 4x2 stored in column-major format.
     *
     * The second linear system is U^H X = alpha * B with U = triu(A), X and B
     * are two dense matrices of size 4x2 stored in row-major format.
     *
     */

    // Create a tri-diagonal matrix in CSR format
    const aoclsparse_int                   n = 4, m = 4, nnz = 10;
    aoclsparse_int                         icrow[5] = {0, 2, 5, 8, 10};
    aoclsparse_int                         icol[10] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<aoclsparse_double_complex> aval;
    // clang-format off
    aval.assign({{1, 3}, {2, 5}, {3, 0}, {1, 2}, {2, -2}, {3, 0},
                 {1, 0}, {2, 3}, {3, 1}, {1, 1}});
    // clang-format on
    aoclsparse_status status;
    aoclsparse_int    ldb, ldx;
    bool              oki, ok;
    double            tol = 20 * std::numeric_limits<double>::epsilon();

    aoclsparse_matrix     A;
    const aoclsparse_int  k     = 2; // number of columns of X and B
    aoclsparse_index_base base  = aoclsparse_index_base_zero;
    aoclsparse_order      order = aoclsparse_order_column;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    status = aoclsparse_create_zcsr(&A, base, m, n, nnz, icrow, icol, aval.data());
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_create_zcsr, status = " << status << "."
                  << std::endl;
        return 3;
    }
    aoclsparse_create_mat_descr(&descr_a);

    /* Case 1: Solving the lower triangular system L X = B.
     *     |-2+4i,   -3+i|
     * B = |3+13i,  4+11i| stored in column-major layout
     *     |16+7i,  16+2i| k = 2 and ldb = 4
     *     |15+11i,10+16i|
     *
     * Linear system L X = B, with solution matrix
     *     |1+i,     i|
     * X = |4+2i,    4| stored in column-major layout
     *     |4+i,  4+2i| k = 2 and ldx = 4
     *     |4,    3+3i|
     */

    aoclsparse_double_complex alpha = {1.0, 0.};
    aoclsparse_double_complex X[m * k];

    std::vector<aoclsparse_double_complex> B;
    B.assign({{-2, 4}, {3, 13}, {16, 7}, {15, 11}, {-3, 1}, {4, 11}, {16, 2}, {10, 16}});
    std::vector<aoclsparse_double_complex> XRef;
    XRef.assign({{1, 1}, {4, 2}, {4, 1}, {4, 0}, {0, 1}, {4, 0}, {4, 2}, {3, 3}});

    // indicate to only access the lower triangular part of matrix A
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_triangular);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);

    // Prepare and call solver
    order  = aoclsparse_order_column;
    ldb    = m;
    ldx    = m;
    status = aoclsparse_set_sm_hint(A, trans, descr_a, order, k, 1);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_set_sm_hint, status = " << status << "."
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
    status = aoclsparse_ztrsm(trans, alpha, A, descr_a, order, B.data(), k, ldb, &X[0], ldx);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error from aoclsparse_ztrsm, status = " << status << std::endl;
        return 3;
    }

    // Print and check the result
    std::cout << "Solving L X = alpha B, where  L=tril(A), and" << std::endl;
    std::cout << "  X and B are dense rectangular martices (column-major layout)" << std::endl;
    std::cout << "  Solution matrix X = " << std::endl;
    std::cout << std::fixed;
    std::cout.precision(1);
    ok = true;
    size_t idx;
    for(int row = 0; row < m; ++row)
    {
        for(int col = 0; col < k; ++col)
        {
            idx = row + col * ldx;
            oki = std::abs(X[idx].real - XRef[idx].real) <= tol
                  && std::abs(X[idx].imag - XRef[idx].imag) <= tol;
            std::cout << "(" << X[idx].real << ", " << X[idx].imag << "i) " << (oki ? "  " : "! ");
            ok &= oki;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    /* Case 2: Solving the lower triangular system U^H X = alpha*B.
     *     |2+4i,    -1+3i|
     * B = |9+15i,    6+9i| stored in row-major layout
     *     |-13+8i,-10+12i| k = 2 and ldx = 2
     *     |14+15i,  8+20i|
     *
     * Linear system L X = B, with solution matrix
     *     |1+i,     i|
     * X = |4+2i,    4| stored in row-major layout
     *     |4+i,  4+2i| k = 2 and ldx = 2
     *     |4,    3+3i|
     */

    B.assign({{2, 4}, {-1, 3}, {9, 15}, {6, 9}, {-13, 8}, {-10, 12}, {14, 15}, {8, 20}});
    // Store same XRef in row-major format
    XRef.assign({{1, 1}, {0, 1}, {4, 2}, {4, 0}, {4, 1}, {4, 2}, {4, 0}, {3, 3}});
    alpha = {0, -1};
    ldb   = k;
    ldx   = k;
    // Indicate to use only the conjugate transpose of the upper part of A
    trans = aoclsparse_operation_conjugate_transpose;
    order = aoclsparse_order_row;
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_upper);
    status = aoclsparse_set_sm_hint(A, trans, descr_a, order, k, 1);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_set_sm_hint, status = " << status << "."
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
    status = aoclsparse_ztrsm(trans, alpha, A, descr_a, order, B.data(), k, ldb, &X[0], ldx);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error from aoclsparse_ztrsm, status = " << status << std::endl;
        return 3;
    }
    // Print and check the result
    std::cout << "Solving U^H X = alpha B, where  U=triu(A), and" << std::endl;
    std::cout << "  X and B are dense rectangular martices (row-major layout)" << std::endl;
    std::cout << "  Solution matrix X = " << std::endl;
    for(int row = 0; row < m; ++row)
    {
        for(int col = 0; col < k; col++)
        {
            idx = row * ldx + col;
            oki = std::abs(X[idx].real - XRef[idx].real) <= tol
                  && std::abs(X[idx].imag - XRef[idx].imag) <= tol;
            std::cout << "(" << X[idx].real << ", " << X[idx].imag << "i) " << (oki ? "  " : "! ");
            ok &= oki;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    // Destroy the aoclsparse memory
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(&A);

    return ok ? 0 : 5;
}
