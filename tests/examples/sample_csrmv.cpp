/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#define M 5
#define N 5
#define NNZ 8

int main(int argc, char* argv[])
{
    aoclsparse_operation   trans     = aoclsparse_operation_none;

    double alpha = 1.0;
    double beta  = 0.0;

    // Print aoclsparse version
    aoclsparse_int ver;
    aoclsparse_get_version(&ver);
    std::cout << "aocl-sparse version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
              << ver % 100 << std::endl;

    // Create matrix descriptor
    aoclsparse_mat_descr descr;
    // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
    // and aoclsparse_index_base to aoclsparse_index_base_zero.
    aoclsparse_create_mat_descr(&descr);

    // Initialise matrix
    aoclsparse_int csr_row_ptr[M+1] = {0, 2, 3, 4, 7, 8};
    aoclsparse_int csr_col_ind[NNZ]= {0, 3, 1, 2, 1, 3, 4, 4};
    double         csr_val[NNZ] = {1 , 6 , 1.050e+01, 1.500e-02, 2.505e+02, -2.800e+02 , 3.332e+01 , 1.200e+01};
    // Initialise vectors
    double x[N] = { 1.0, 2.0, 3.0, 4.0, 5.0};
    double y[M];

    std::cout << "Invoking aoclsparse_dcsrmv..";
    //Invoke SPMV API for CSR storage format(double precision)
    aoclsparse_dcsrmv(trans,
                      &alpha,
                      M,
                      N,
                      NNZ,
                      csr_val,
                      csr_col_ind,
                      csr_row_ptr,
                      descr,
                      x,
                      &beta,
                      y);
    std::cout << "Done." << std::endl;
    std::cout << "Output Vector:" << std::endl;
    for(aoclsparse_int i=0;i < M; i++)
        std::cout << y[i] << std::endl;

    aoclsparse_destroy_mat_descr(descr);
    return 0;
}
