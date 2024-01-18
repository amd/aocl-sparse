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

#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

/* Example that illustrates matrix-matrix product C=A*B where matrices A and B

 * are sparse and C is dense. Matrices are
 *
 *     [ 1+2i  i    2-i ]         [ 1-i   0 ]                          [ 13+i    5-3i   ]
 * A = [ 0     2i    0  ] and B = [  0    i ] with expected result C = [ 0        -2    ]
 *     [ 0    0.1+4i 3  ]         [ 4+2i  3 ]                          [ 12+6i   5+0.1i ]
 *
 * dense matrix C is stored by rows.
 */
int main(void)
{

    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_matrix     A;
    aoclsparse_matrix     B;
    aoclsparse_status     status;

    // Matrix sizes
    aoclsparse_int            m = 3, n = 2, k = 3;
    aoclsparse_int            nnz_A = 6, nnz_B = 4;
    aoclsparse_int            row_ptr_A[] = {0, 3, 4, 6};
    aoclsparse_int            col_ind_A[] = {0, 1, 2, 1, 1, 2};
    aoclsparse_double_complex val_A[]     = {{1, 2}, {0, 1}, {2, -1}, {0, 2}, {0.1, 4}, {3, 0}};
    status = aoclsparse_create_zcsr(&A, base, m, k, nnz_A, row_ptr_A, col_ind_A, val_A);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    aoclsparse_int            row_ptr_B[] = {0, 1, 2, 4};
    aoclsparse_int            col_ind_B[] = {0, 1, 0, 1};
    aoclsparse_double_complex val_B[]     = {{1, -1}, {0, 1}, {4, 2}, {3, 0}};
    status = aoclsparse_create_zcsr(&B, base, k, n, nnz_B, row_ptr_B, col_ind_B, val_B);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    // expected output matrices
    aoclsparse_double_complex C_exp[] = {{13, 1}, {5, -3}, {0, 0}, {-2, 0}, {12, 6}, {5, 0.1}};

    // output matrices;
    aoclsparse_double_complex C[6];
    aoclsparse_operation      op = aoclsparse_operation_none;

    memset(C, 0, sizeof(aoclsparse_double_complex) * 6);
    status = aoclsparse_zspmmd(op, A, B, aoclsparse_order_row, C, n);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    std::cout << std::fixed;
    std::cout.precision(2);
    bool okij, ok_C = true;
    //Initializing precision tolerance range for double
    const double tol = (std::numeric_limits<double>::epsilon());

    aoclsparse_int i, j;
    std::cout << "The output C matrix\n";
    for(i = 0; i < m; i++)
    {
        for(j = 0; j < n; j++)
        {
            okij = std::abs(C[i * n + j].real - C_exp[i * n + j].real) <= tol
                   && std::abs(C[i * n + j].imag - C_exp[i * n + j].imag) <= tol;
            ok_C &= okij;
            std::cout << std::setw(10) << "(" << C[i * n + j].real << ", " << C[i * n + j].imag
                      << ")" << std::setw(3) << (okij ? "" : " !");
        }
        std::cout << "\n";
    }
    aoclsparse_destroy(&A);
    aoclsparse_destroy(&B);

    return ((ok_C ? 0 : 4));
}
