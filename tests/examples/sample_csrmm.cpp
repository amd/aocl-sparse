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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// Example to illustrate the usage of aoclsparse_dcsrmm API
int main(void)
{

    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_matrix     A;
    aoclsparse_status     status;
    aoclsparse_order      order = aoclsparse_order_row;
    aoclsparse_mat_descr  descr;

    // By default aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
    status = aoclsparse_create_mat_descr(&descr);
    if(status != aoclsparse_status_success)
        return status;

    // Set aoclsparse_index_base to aoclsparse_index_base_zero.
    status = aoclsparse_set_mat_index_base(descr, base);
    if(status != aoclsparse_status_success)
        return status;

    // Matrix sizes
    aoclsparse_int m = 3, n = 3, k = 3;
    aoclsparse_int nnz = 4;
    // Matrix A
    // [ 0.  42.   0.2]
    // [ 4.6  0.   0. ]
    // [ 0.   0.  -8. ]
    aoclsparse_int row_ptr[] = {0, 2, 3, 4};
    aoclsparse_int col_ind[] = {1, 2, 0, 2};
    double         csr_val[] = {42., 0.2, 4.6, -8};

    //Dense Matrix B
    double B[]   = {-1.0, -2.7, 3.0, 4.5, 5.8, -6.0, 1.0, -2.0, 3.0};
    double alpha = 1, beta = 0;
    status = aoclsparse_create_dcsr(A, base, m, k, nnz, row_ptr, col_ind, csr_val);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    // expected output matrices
    double C_exp[] = {189.2, 243.2, -251.4, -4.6, -12.42, 13.8, -8, 16, -24};

    // output matrices;
    double               C[9] = {0};
    aoclsparse_operation op   = aoclsparse_operation_none;

    status = aoclsparse_dcsrmm(op, alpha, A, descr, order, B, n, k, beta, C, n);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    std::cout << std::fixed;
    std::cout.precision(2);
    bool okij, ok_C = true;
    //Initializing precision tolerance range for double
    const double tol = std::sqrt((std::numeric_limits<double>::epsilon()));

    aoclsparse_int i, j;
    std::cout << "The output C matrix\n";
    for(i = 0; i < m; i++)
    {
        for(j = 0; j < n; j++)
        {
            okij = std::abs(C[i * n + j] - C_exp[i * n + j]) <= tol;
            ok_C &= okij;
            std::cout << std::setw(10) << C[i * n + j] << std::setw(3) << (okij ? "" : " !");
        }
        std::cout << "\n";
    }
    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(A);

    return ((ok_C ? 0 : 4));
}
