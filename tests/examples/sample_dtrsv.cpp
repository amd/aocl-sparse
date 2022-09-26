/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

int main()
{
    std::cout << "-------------------------------" << std::endl
              << " Triangle solve sample program" << std::endl
              << "-------------------------------" << std::endl
              << std::endl;

    // Create a tri-diagonal matrix in CSR format
    // | 1  2  0  0 |
    // | 3  1  2  0 |
    // | 0  3  1  2 |
    // | 0  0  3  1 |
    aoclsparse_int n = 4, m = 4, nnz = 10;
    aoclsparse_int icrow[5] = {0, 2, 5, 8, 10};
    aoclsparse_int icol[18] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    double aval[18] = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
    aoclsparse_status status;

    // create aoclsparse matrix and its descriptor
    aoclsparse_matrix A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr descr_a;
    aoclsparse_operation trans = aoclsparse_operation_none;
    aoclsparse_create_dcsr(A, base, m, n, nnz, icrow, icol, aval);
    aoclsparse_create_mat_descr(&descr_a);

    // A = L + D + U where:
    // - L and U are the strict lower and upper triangle of A
    // - D is the diagonal
    // Use the matrix descriptor to solve (L+D)x = b
    // b = [1, 4, 4, 4]^t
    double b[4] = {1, 4, 4, 4};
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_triangular);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    status = aoclsparse_set_sv_hint(A, trans, descr_a, 1);
    if (status != aoclsparse_status_success)
    {
        std::cerr << "Error setting SV hint, status = " << status << std::endl;
        return 1;
    }
    status = aoclsparse_optimize(A);
    if (status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_optimize, status = " << status << "." << std::endl;
        return 2;
    }
    // Call the triangle solver
    double alpha = 1.0;
    double x[n];
    status = aoclsparse_dtrsv(trans, alpha, A, descr_a, b, x);
    if (status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dtrsv, status = " << status << "." << std::endl;
        return 3;
    }

    // Print the result
    std::cout << "Solving (L+D)x = b: " << std::endl
              << "  x = ";
    std::cout << std::fixed;
    std::cout.precision(1);
    for (aoclsparse_int i = 0; i < n; i++)
        std::cout << x[i] << "  ";
    std::cout << std::endl;

    // The same method can be used to solve (U+D)x = b
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_upper);
    status = aoclsparse_dtrsv(trans, alpha, A, descr_a, b, x);
    if (status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dtrsv, status = " << status << "." << std::endl;
        return 3;
    }

    // Print the result
    std::cout << std::endl;
    std::cout << "Solving (U+D)x = b: " << std::endl
              << "  x = ";
    for (aoclsparse_int i = 0; i < n; i++)
        std::cout << x[i] << "  ";
    std::cout << std::endl;

    // destroy the aoclsparse memory
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(A);

    return 0;
}
