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

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main(void)
{
    std::cout << "------------------------------------------------" << std::endl
              << " Triangle Solve with Multiple Right Hand Sides" << std::endl
              << " sample program for real data type" << std::endl
              << "------------------------------------------------" << std::endl
              << std::endl;

    /*
     * This example illustrates how to solve two triangular
     * linear system of equations. The real matrix used is
     *     | 1  2  0  0 |
     * A = | 3  1  2  0 |
     *     | 0  3  1  2 |
     *     | 0  0  3  1 |
     *
     * The first linear system is L X = alpha * B with L = tril(A), X and B
     * are two dense matrices of size 4x2 stored in column-major format.
     *
     * The second linear system is U X = alpha * B with U = triu(A), X and B
     * are two dense matrices of size 4x2 stored in row-major format.
     *
     */

    // Create a tri-diagonal matrix in CSR format
    const aoclsparse_int n = 4, m = 4, nnz = 10;
    aoclsparse_int       icrow[5] = {0, 2, 5, 8, 10};
    aoclsparse_int       icol[10] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    double               aval[10] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
    aoclsparse_status    status;
    const aoclsparse_int k = 2;
    aoclsparse_int       ldb, ldx;
    bool                 oki, ok;
    double               tol = 20.0 * std::numeric_limits<double>::epsilon();

    // Create aoclsparse matrix and its descriptor
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_order      order;
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

    /* Case 1: Solving the lower triangular system L^T X = alpha*B.
     *     |1, -1|
     * B = |2, -2| stored in column-major layout
     *     |3, -6| k = 2 and ldb = 4
     *     |4, -8|
     *
     * Linear system L X = B, with solution matrix
     *     | 86,-167|
     * X = |-29,  56| stored in column-major layout
     *     |  9, -18| k = 2 and ldx = 4
     *     | -4,   8|
     */

    double alpha = -1.0;
    double X[m * k];

    double              B[m * k] = {1, 2, 3, 4, -1, -2, -6, -8};
    std::vector<double> XRef;
    XRef.assign({86, -29, 9, -4, -167, 56, -18, 8});

    // Indicate to only access the lower triangular part of matrix A
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_triangular);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    trans = aoclsparse_operation_transpose;

    order  = aoclsparse_order_column;
    ldb    = m;
    ldx    = m;
    status = aoclsparse_set_sm_hint(A, trans, descr_a, order, n, 1);
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
    status = aoclsparse_dtrsm(trans, alpha, A, descr_a, order, &B[0], k, ldb, &X[0], ldx);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dtrsm, status = " << status << "."
                  << std::endl;
        return 3;
    }

    // Print and check the result
    std::cout << "Solving L^T X = alpha B, where  L=tril(A), and" << std::endl;
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
            oki = std::abs(X[idx] - XRef[idx]) <= tol;
            std::cout << X[idx] << (oki ? "  " : "! ");
            ok &= oki;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    /* Case 2: Solving the lower triangular system U X = alpha*B.
     * In this case we "reinterpret" B as a col-major layout, having the
     * form
     *     | 1,  2|
     * B = | 3,  4| stored in row-major layout
     *     |-1, -2| k = 2 and ldb = 2
     *     |-6, -8|
     *
     * Linear system L X = B, with solution matrix
     *     | 39,  50|
     * X = |-19, -24| stored in row-major layout
     *     | 11,  14| k = 2 and ldx = 2
     *     | -6,  -8|
     */

    // Store same XRef in row-major format
    XRef.assign({39, 50, -19, -24, 11, 14, -6, -8});
    alpha = 1.0;
    ldb   = k;
    ldx   = k;
    // Indicate to use only the conjugate transpose of the upper part of A
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_upper);
    trans  = aoclsparse_operation_none;
    order  = aoclsparse_order_row;
    status = aoclsparse_set_sm_hint(A, trans, descr_a, order, n, 1);
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
    status = aoclsparse_dtrsm(trans, alpha, A, descr_a, order, &B[0], k, ldb, &X[0], ldx);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dtrsm, status = " << status << "."
                  << std::endl;
        return 3;
    }
    // Print and check the result
    std::cout << "Solving U X = alpha B, where  U=triu(A), and" << std::endl;
    std::cout << "  X and B are dense rectangular martices (row-major layout)" << std::endl;
    std::cout << "  Solution matrix X = " << std::endl;
    for(int row = 0; row < m; ++row)
    {
        for(int col = 0; col < k; col++)
        {
            idx = row * ldx + col;
            oki = std::abs(X[idx] - XRef[idx]) <= tol;
            std::cout << X[idx] << (oki ? " " : "!  ");
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
