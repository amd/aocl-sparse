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

#include <iostream>
#include <limits>

#define M 6
#define N 5
#define NNZ 8

int main(void)
{
    aoclsparse_operation op = aoclsparse_operation_none;

    double alpha = 1.0;
    double beta  = 0.0;

    // Print aoclsparse version
    std::cout << aoclsparse_get_version() << std::endl;

    // Create matrix descriptor
    aoclsparse_mat_descr descr;
    // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
    // and aoclsparse_index_base to aoclsparse_index_base_zero.
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_index_base base = aoclsparse_index_base_zero;

    // Initialise matrix
    //  1  0  0  2  0
    //  0  3  0  0  0
    //  0  0  4  0  0
    //  0  5  0  6  7
    //  0  0  0  0  8
    //  0  0  0  0  0
    aoclsparse_int    csr_row_ptr[M + 1] = {0, 2, 3, 4, 7, 8, 8};
    aoclsparse_int    csr_col_ind[NNZ]   = {0, 3, 1, 2, 1, 3, 4, 4};
    double            csr_val[NNZ]       = {1, 2, 3, 4, 5, 6, 7, 8};
    aoclsparse_matrix A;
    aoclsparse_create_dcsr(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

    // Initialise vectors
    double x[N]     = {1.0, 1.0, 1.0, 1.0, 1.0};
    double y[M]     = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double d        = 0;
    double y_exp[M] = {3.0, 3.0, 4.0, 18.0, 8.0};
    double d_exp    = 36;

    //to identify hint id(which routine is to be executed, destroyed later)
    //aoclsparse_set_dotmv_hint(A, op, descr, 1);

    // Optimize the matrix, "A"
    aoclsparse_optimize(A);

    std::cout << "Invoking aoclsparse_ddotmv..";
    //Invoke DOTMV API (double precision)
    aoclsparse_ddotmv(op, alpha, A, descr, x, beta, y, &d);
    std::cout << "Done." << std::endl;

    // Print and check the result
    std::cout << std::endl << "y = ";
    bool   oki, ok = true;
    double tol = 20 * std::numeric_limits<double>::epsilon();
    for(aoclsparse_int i = 0; i < M; i++)
    {
        oki = std::abs(y[i] - y_exp[i]) <= tol;
        std::cout << y[i] << (oki ? " " : "!  ");
        ok &= oki;
    }
    // std::cout << std::endl;
    // std::cout << "Output Vector:" << std::endl;
    // for(aoclsparse_int i = 0; i < M; i++)
    //     std::cout << "y[" << i << "] = "y[i] << std::endl;

    oki = std::abs(d - d_exp) <= tol;
    std::cout << std::endl << "Output dot product: " << d << (oki ? " " : "!  ") << std::endl;

    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(&A);
    return 0;
}
