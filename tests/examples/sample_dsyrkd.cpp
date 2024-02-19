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

#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// An example to illustrate the usage of aoclsparse_syrk
int main(void)
{
    std::cout << "-------------------------------" << std::endl
              << "----- syrkd sample program -----" << std::endl
              << "-------------------------------" << std::endl
              << std::endl;

    aoclsparse_matrix     A;
    aoclsparse_status     status;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr;
    double                alpha  = 2.1;
    double                beta   = 1.3;
    aoclsparse_int        ldc    = 10;
    aoclsparse_operation  op     = aoclsparse_operation_none;
    aoclsparse_order      layout = aoclsparse_order_row;
    aoclsparse_int        m_C;
    double               *C     = nullptr;
    double               *C_exp = nullptr;

    // By default aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
    status = aoclsparse_create_mat_descr(&descr);
    if(status != aoclsparse_status_success)
        return status;

    // Matrix sizes
    aoclsparse_int m = 4, k = 3;
    aoclsparse_int nnz = 7;
    // Matrix A
    // [ 0.  -1.2   2.3]
    // [ 4.6  0.    0. ]
    // [ 0.   3.0  -8.1 ]
    // [ 0.3   0.  -5.1 ]
    aoclsparse_int row_ptr[] = {0, 2, 3, 5, 7};
    aoclsparse_int col_ind[] = {1, 2, 0, 1, 2, 0, 2};
    double         csr_val[] = {-1.2, 2.3, 4.6, 3.0, -8.1, 0.3, -5.1};

    status = aoclsparse_create_dcsr(&A, base, m, k, nnz, row_ptr, col_ind, csr_val);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    // output matrix dimension
    m_C = 4;

    C     = (double *)malloc(sizeof(double) * ldc * m_C);
    C_exp = (double *)malloc(sizeof(double) * ldc * m_C);

    // set the large C matrix to -1, only a window of size 4x4 (upper triangular part)
    // is updated with the syrkd call
    for(aoclsparse_int i = 0; i < ldc * m_C; i++)
    {
        C[i]     = -1;
        C_exp[i] = -1;
    }

    // Expected value of C which is C_exp
    // [ 12.8330    -1.3000   -47.9830   -25.9330,  -1, -1, ....]
    // [ -1         43.1360    -1.3000     1.5980,  -1, -1, ....]
    // [ -1         -1         155.3810    85.4510, -1, -1, ....]
    // [ -1         -1         -1          53.5100, -1, -1, ....]
    // [ -1         -1         -1          -1,      -1, -1, ....]
    // ...
    C_exp[0]  = 12.8330;
    C_exp[1]  = -1.3000;
    C_exp[2]  = -47.9830;
    C_exp[3]  = -25.9330;
    C_exp[11] = 43.1360;
    C_exp[12] = -1.3000;
    C_exp[13] = 1.5980;
    C_exp[22] = 155.3810;
    C_exp[23] = 85.4510;
    C_exp[33] = 53.5100;

    status = aoclsparse_dsyrkd(op, A, alpha, beta, C, layout, ldc);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    bool oki, ok = true;
    //Initializing precision tolerance range for double
    const double tol = 1e-03;
    std::cout << "Expected output\n"
              << "-------------------------------\n";
    for(aoclsparse_int i = 0; i < m_C; i++)
    {
        for(aoclsparse_int j = 0; j < ldc; j++)
        {
            std::cout << std::setw(10) << C_exp[i * ldc + j] << std::setw(3) << "";
        }
        std::cout << std::endl;
    }

    std::cout << "Actual output\n"
              << "-------------------------------\n";
    for(aoclsparse_int i = 0; i < m_C; i++)
    {
        for(aoclsparse_int j = 0; j < ldc; j++)
        {
            oki = ((std::abs(C[i * ldc + j] - C_exp[i * ldc + j]) <= tol));
            ok &= oki;
            std::cout << std::setw(10) << C[i * ldc + j] << std::setw(3) << (oki ? "" : " !");
        }
        std::cout << std::endl;
    }
    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(&A);
    free(C);
    free(C_exp);

    return (ok ? 0 : 6);
}
