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

/* Example that illustrates matrix-matrix product C=A*B^T where matrices A and B

 * are sparse and C is dense. Matrices are
 *
 *     [   0  1.1 ]         [ -7.4 0 ]                          [    0, 0,  5.5 ]
 * A = [ 3.5    0 ] and B = [    0 0 ] with expected result C = [-25.9, 0,    0 ]
 *     [   0  5.1 ]         [    0 5 ]                          [    0, 0, 25.5 ]
 *     [   0   -6 ]                                             [    0, 0,  -30 ]
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
    aoclsparse_int m_a = 4, n_a = 2, m_b = 3, n_b = 2;
    aoclsparse_int nnz_A = 4, nnz_B = 2;
    aoclsparse_int row_ptr_A[] = {0, 1, 2, 3, 4};
    aoclsparse_int col_ind_A[] = {1, 0, 1, 1};
    double         val_A[]     = {1.1, 3.5, 5.1, -6.0};
    status = aoclsparse_create_dcsr(&A, base, m_a, n_a, nnz_A, row_ptr_A, col_ind_A, val_A);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    aoclsparse_int row_ptr_B[] = {0, 1, 1, 2};
    aoclsparse_int col_ind_B[] = {0, 1};
    double         val_B[]     = {-7.4, 5.0};
    status = aoclsparse_create_dcsr(&B, base, m_b, n_b, nnz_B, row_ptr_B, col_ind_B, val_B);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    // expected output matrix
    double C_exp[] = {0, 0, 5.5, -25.9, 0, 0, 0, 0, 25.5, 0, 0, -30.0}; // when beta = 0;
    //double C_exp[] = {1, 1, 6.5, -24.9, 1, 1, 1, 1, 26.5, 1, 1, -29.0}; // when beta = 1;

    // output matrices;
    double C[12];
    std::fill_n(C, 12, 1);
    aoclsparse_operation opA = aoclsparse_operation_none;
    aoclsparse_operation opB = aoclsparse_operation_transpose;

    aoclsparse_mat_descr descrA;
    aoclsparse_create_mat_descr(&descrA);
    aoclsparse_set_mat_index_base(descrA, base);
    aoclsparse_mat_descr descrB;
    aoclsparse_create_mat_descr(&descrB);
    aoclsparse_set_mat_index_base(descrB, base);

    double alpha, beta;
    alpha              = 1.0;
    beta               = 0.0;
    aoclsparse_int ldc = m_b;

    status = aoclsparse_dsp2md(
        opA, descrA, A, opB, descrB, B, alpha, beta, C, aoclsparse_order_row, ldc);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    std::cout << std::fixed;
    std::cout.precision(2);
    bool okij, ok_C = true;
    //Initializing precision tolerance range for double
    const double tol = (std::numeric_limits<double>::epsilon()) * 1000;

    aoclsparse_int i, j;
    std::cout << "The output C matrix\n";
    for(i = 0; i < m_a; i++)
    {
        for(j = 0; j < ldc; j++)
        {
            okij = std::abs(C[i * ldc + j] - C_exp[i * ldc + j]) <= tol;
            ok_C &= okij;
            std::cout << std::setw(10) << C[i * ldc + j] << std::setw(3) << (okij ? "" : " !");
        }
        std::cout << "\n";
    }

    aoclsparse_destroy_mat_descr(descrB);
    aoclsparse_destroy(&B);
    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy(&A);

    return ((ok_C ? 0 : 4));
}
