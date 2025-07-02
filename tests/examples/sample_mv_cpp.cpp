/* ************************************************************************
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc.
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
/*
    Description: In this example, we create a sparse matrix in CSR format
                 and perform a matrix-vector multiplication (SpMV) using the
                 C++ interface of aoclsparse.
*/
#include "aoclsparse.hpp"

#include <iostream>
#include <vector>

int main(void)
{
    aoclsparse_operation trans = aoclsparse_operation_none;

    double alpha = 1.0;
    double beta  = 0.0;

    const aoclsparse_int m   = 5; // Number of rows
    const aoclsparse_int n   = 5; // Number of columns
    const aoclsparse_int nnz = 8; // Number of non-zero elements

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
    std::vector<aoclsparse_int> csr_row     = {0, 2, 3, 4, 7, 8};
    std::vector<aoclsparse_int> csr_col_ind = {0, 3, 1, 2, 1, 3, 4, 4};
    std::vector<double>         csr_val     = {1, 2, 3, 4, 5, 6, 7, 8};
    aoclsparse_matrix           A;
    auto                        status = aoclsparse::create_csr(
        &A, base, m, n, nnz, csr_row.data(), csr_col_ind.data(), csr_val.data());

    if(status != aoclsparse_status_success)
    {
        std::cout << "Error creating the matrix, status = " << status << std::endl;
        return 1;
    }

    // Initialise vectors
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y(m, 0.0); // Output vector initialized to zero

    //to identify hint id(which routine is to be executed, destroyed later)
    aoclsparse_set_mv_hint(A, trans, descr, 1);

    // Optimize the matrix, "A"
    aoclsparse_optimize(A);

    std::cout << "Invoking aoclsparse_dmv..";
    //Invoke SPMV API (double precision)
    aoclsparse::mv(trans, &alpha, A, descr, x.data(), &beta, y.data());

    std::cout << "Done." << std::endl;

    std::cout << "Output Vector:" << std::endl;
    for(aoclsparse_int i = 0; i < m; i++)
        std::cout << y[i] << std::endl;

    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(&A);
    return 0;
}
