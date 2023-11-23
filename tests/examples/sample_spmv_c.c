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

#include <math.h>
#include <stdio.h>

#define M 5
#define N 5
#define NNZ 8

int main(void)
{
    aoclsparse_status     status;
    aoclsparse_matrix     A     = NULL;
    aoclsparse_mat_descr  descr = NULL;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_index_base base  = aoclsparse_index_base_zero;

    double alpha = 1.0;
    double beta  = 0.0;

    // Input matrix
    //  1  0  0  2  0
    //  0  3  0  0  0
    //  0  0  4  0  0
    //  0  5  0  6  7
    //  0  0  0  0  8
    aoclsparse_int csr_row_ptr[M + 1] = {0, 2, 3, 4, 7, 8};
    aoclsparse_int csr_col_ind[NNZ]   = {0, 3, 1, 2, 1, 3, 4, 4};
    double         csr_val[NNZ]       = {1, 2, 3, 4, 5, 6, 7, 8};

    // Input vectors
    double x[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[M];

    // Expected reference result & tolerance
    double         y_exp[M] = {9.0, 6.0, 12.0, 69.0, 40.0};
    double         tol      = 1e-12;
    aoclsparse_int oki, ok = 1;

    // Print aoclsparse version
    printf("%s\n", aoclsparse_get_version());

    // Create matrix descriptor
    // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
    // and aoclsparse_index_base to aoclsparse_index_base_zero.
    aoclsparse_create_mat_descr(&descr);

    // Initialise sparse matrix
    status = aoclsparse_create_dcsr(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);
    if(status != aoclsparse_status_success)
    {
        printf("Error while creating a sparse matrix, status = %i\n", status);
        return 1;
    }

    // Hint the system what operation to expect // to identify hint id(which routine is to be executed, destroyed later)
    status = aoclsparse_set_mv_hint(A, trans, descr, 1);
    if(status != aoclsparse_status_success)
    {
        printf("Error while hinting operation, status = %i\n", status);
        return 1;
    }

    // Optimize the matrix
    status = aoclsparse_optimize(A);
    if(status != aoclsparse_status_success)
    {
        printf("Error while optimizing the matrix, status = %i\n", status);
        return 1;
    }

    // Invoke SPMV API (double precision)
    printf("Invoking aoclsparse_dmv...\n");
    status = aoclsparse_dmv(trans, &alpha, A, descr, x, &beta, y);
    if(status != aoclsparse_status_success)
    {
        printf("Error while computing SPMV, status = %i\n", status);
        return 1;
    }

    // Print and check the results
    printf("Output vector:\n");
    for(aoclsparse_int i = 0; i < M; i++)
    {
        oki = fabs(y[i] - y_exp[i]) <= tol;
        printf("  %lf %c\n", y[i], oki ? ' ' : '!');
        ok = ok && oki;
    }

    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(&A);
    return ok ? 0 : 2;
}
