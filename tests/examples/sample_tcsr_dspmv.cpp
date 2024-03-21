/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <limits>

#define M 4
#define N 4
#define NNZ 9

int main(void)
{
    std::cout << "---------------------------------" << std::endl
              << " TCSR matrix, SPMV sample program " << std::endl
              << "---------------------------------" << std::endl
              << std::endl;

    aoclsparse_operation trans = aoclsparse_operation_none;

    double alpha = 1.0;
    double beta  = 0.0;

    // Create matrix descriptor
    aoclsparse_mat_descr descr;
    // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
    // and aoclsparse_index_base to aoclsparse_index_base_zero.
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_index_base base = aoclsparse_index_base_zero;

    // Initialize matrix
    // 1 0 2 3
    // 0 4 0 0
    // 5 0 6 0
    // 7 8 0 9
    aoclsparse_int    row_ptr_L[M + 1]   = {0, 1, 2, 4, 7};
    aoclsparse_int    row_ptr_U[M + 1]   = {0, 3, 4, 5, 6};
    aoclsparse_int    col_idx_L[7]       = {0, 1, 0, 2, 0, 1, 3};
    aoclsparse_int    col_idx_U[6]       = {0, 2, 3, 1, 2, 3};
    double            val_L[7]           = {1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double            val_U[6]           = {1.0, 2.0, 3.0, 4.0, 6.0, 9.0};
    double            y_gen_exp[M]       = {19.0, 8.0, 23.0, 59.0};
    double            y_tri_lower_exp[M] = {1.0, 8.0, 23.0, 59.0};
    double            y_tri_upper_exp[M] = {19.0, 8.0, 18.0, 36.0};
    aoclsparse_status status;
    aoclsparse_matrix A;
    bool              oki, ok;
    double            tol = 20 * std::numeric_limits<double>::epsilon();

    // create TCSR matrix
    status = aoclsparse_create_dtcsr(
        &A, base, M, N, NNZ, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);

    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_create_dtcsr, status = " << status << "."
                  << std::endl;
        return 3;
    }

    // Initialise vectors
    double x[N] = {1.0, 2.0, 3.0, 4.0};
    double y_gen[M], y_tri_lower[M], y_tri_upper[M];

    // Invoke SPMV API (double precision) for general matrix type
    status = aoclsparse_dmv(trans, &alpha, A, descr, x, &beta, y_gen);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dmv, status = " << status << "." << std::endl;
        return 3;
    }

    // Print and check results
    std::cout << "Computing y = Ax : " << std::endl << " y = ";
    ok = true;
    for(aoclsparse_int i = 0; i < M; i++)
    {
        oki = std::abs(y_gen[i] - y_gen_exp[i]) <= tol;
        std::cout << y_gen[i] << (oki ? " " : "!  ");
        ok &= oki;
    }
    std::cout << std::endl;

    // Invoke SPMV API for triangular matrix type, fill mode: lower
    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
    status = aoclsparse_dmv(trans, &alpha, A, descr, x, &beta, y_tri_lower);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dmv, status = " << status << "." << std::endl;
        return 4;
    }

    // Print and check results
    std::cout << std::endl;
    std::cout << "Computing y = (L+D)x : " << std::endl << " y = ";
    for(aoclsparse_int i = 0; i < M; i++)
    {
        oki = std::abs(y_tri_lower[i] - y_tri_lower_exp[i]) <= tol;
        std::cout << y_tri_lower[i] << (oki ? " " : "!  ");
        ok &= oki;
    }
    std::cout << std::endl;

    // Invoke SPMV API for triangular matrix type, fill mode: upper
    aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
    status = aoclsparse_dmv(trans, &alpha, A, descr, x, &beta, y_tri_upper);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_dmv, status = " << status << "." << std::endl;
        return 4;
    }

    // Print and check results
    std::cout << std::endl;
    std::cout << "Computing y = (D+U)x : " << std::endl << " y = ";
    for(aoclsparse_int i = 0; i < M; i++)
    {
        oki = std::abs(y_tri_upper[i] - y_tri_upper_exp[i]) <= tol;
        std::cout << y_tri_upper[i] << (oki ? " " : "!  ");
        ok &= oki;
    }
    std::cout << std::endl;

    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(&A);
    return ok ? 0 : 5;
}
