/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

/* Computes multiplication of 2 sparse matrices A and B in CSR format using C++ interface. */

// Comment this out for single stage computation
#define TWO_STAGE_COMPUTATION

#define PRINT_OUTPUT

void check_error(aoclsparse_status status, const char *api_name)
{
    if(status != aoclsparse_status_success)
    {
        std::cerr << "ERROR in " << api_name << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    aoclsparse_status     status;
    aoclsparse_int        nnz_C;
    aoclsparse_request    request;
    aoclsparse_index_base base   = aoclsparse_index_base_zero;
    aoclsparse_operation  transA = aoclsparse_operation_none;
    aoclsparse_operation  transB = aoclsparse_operation_none;

    // Print aoclsparse version
    std::cout << aoclsparse_get_version() << std::endl;

    // Initialise matrix descriptor and csr matrix structure of inputs A and B
    aoclsparse_mat_descr descrA;
    aoclsparse_mat_descr descrB;
    aoclsparse_matrix    csrA;
    aoclsparse_matrix    csrB;

    // Create matrix descriptor of input matrices
    // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
    // and aoclsparse_index_base to aoclsparse_index_base_zero.
    status = aoclsparse_create_mat_descr(&descrA);

    check_error(status, "aoclsparse_create_mat_descr for A");

    status = aoclsparse_create_mat_descr(&descrB);

    check_error(status, "aoclsparse_create_mat_descr for B");

    // Matrix sizes
    aoclsparse_int m = 3, n = 3, k = 3;
    aoclsparse_int nnz_A = 6, nnz_B = 4;
    // Matrix A
    // 	1  0  2
    // 	0  0  3
    // 	4  5  6
    aoclsparse_int row_ptr_A[] = {0, 2, 3, 6};
    aoclsparse_int col_ind_A[] = {0, 2, 2, 0, 1, 2};
    float          val_A[]     = {1, 2, 3, 4, 5, 6};

    // Use C++ template interface to create CSR matrix
    status = aoclsparse::create_csr<float>(&csrA, base, m, k, nnz_A, row_ptr_A, col_ind_A, val_A);

    check_error(status, "aoclsparse::create_csr<float> for A");

    // Matrix B
    // 	1  2  0
    // 	0  0  3
    // 	0  4  0
    aoclsparse_int row_ptr_B[] = {0, 2, 3, 4};
    aoclsparse_int col_ind_B[] = {0, 1, 2, 1};
    float          val_B[]     = {1, 2, 3, 4};

    // Use C++ template interface to create CSR matrix
    status = aoclsparse::create_csr<float>(&csrB, base, k, n, nnz_B, row_ptr_B, col_ind_B, val_B);
    check_error(status, "aoclsparse::create_csr<float> for B");

    aoclsparse_matrix csrC          = nullptr;
    aoclsparse_int   *csr_row_ptr_C = nullptr;
    aoclsparse_int   *csr_col_ind_C = nullptr;
    float            *csr_val_C     = nullptr;
    aoclsparse_int    C_M, C_N;

#ifdef TWO_STAGE_COMPUTATION
    std::cout << "Invoking aoclsparse::sp2m (C++ interface) with aoclsparse_stage_nnz_count..";
    // aoclsparse_stage_nnz_count : Only rowIndex array of the CSR matrix
    // is computed internally. The output sparse CSR matrix can be
    // extracted to measure the memory required for full operation.
    request = aoclsparse_stage_nnz_count;
    status  = aoclsparse::sp2m<float>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC);

    check_error(status, "aoclsparse::sp2m (C++ interface) with aoclsparse_stage_nnz_count");

    status = aoclsparse_export_scsr(
        csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C);

    check_error(status, "aoclsparse_export_scsr");

#ifdef PRINT_OUTPUT
    std::cout << "C_M"
              << "\t"
              << "C_N"
              << "\t"
              << "nnz_C" << std::endl;
    std::cout << C_M << "\t" << C_N << "\t" << nnz_C << std::endl;

    std::cout << "csr_row_ptr_C: " << std::endl;
    for(aoclsparse_int i = 0; i < C_M + 1; i++)
        std::cout << csr_row_ptr_C[i] << "\t";
    std::cout << std::endl;
#endif

    std::cout << "Invoking aoclsparse::sp2m (C++ interface) with aoclsparse_stage_finalize..";
    // aoclsparse_stage_finalize : Finalize computation of remaining
    // output arrays ( column indices and values of output matrix entries) .
    // Has to be called only after aoclsparse::sp2m call with
    // aoclsparse_stage_nnz_count parameter.
    request = aoclsparse_stage_finalize;
    status  = aoclsparse::sp2m<float>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC);

    check_error(status, "aoclsparse::sp2m (C++ interface) with aoclsparse_stage_finalize");

    status = aoclsparse_export_scsr(
        csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C);

    check_error(status, "aoclsparse_export_scsr after finalize");

#ifdef PRINT_OUTPUT
    std::cout << "C_M"
              << "\t"
              << "C_N"
              << "\t"
              << "nnz_C" << std::endl;
    std::cout << C_M << "\t" << C_N << "\t" << nnz_C << std::endl;

    std::cout << "csr_row_ptr_C: " << std::endl;
    for(aoclsparse_int i = 0; i < C_M + 1; i++)
        std::cout << csr_row_ptr_C[i] << "\t";
    std::cout << std::endl;

    std::cout << "csr_col_ind_C: " << std::endl;
    for(aoclsparse_int i = 0; i < nnz_C; i++)
        std::cout << csr_col_ind_C[i] << "\t";
    std::cout << std::endl;

    std::cout << "csr_val_C: " << std::endl;
    for(aoclsparse_int i = 0; i < nnz_C; i++)
        std::cout << csr_val_C[i] << "\t";
    std::cout << std::endl;
#endif

#else // SINGLE STAGE
    std::cout
        << "Invoking aoclsparse::sp2m (C++ interface) with aoclsparse_stage_full_computation..";
    // aoclsparse_stage_full_computation :  Whole computation is performed in
    // single step.
    request = aoclsparse_stage_full_computation;
    status  = aoclsparse::sp2m<float>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC);
    check_error(status, "aoclsparse::sp2m (C++ interface) with aoclsparse_stage_full_computation");

    status = aoclsparse_export_scsr(
        csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C);

    check_error(status, "aoclsparse_export_scsr after full computation");

#ifdef PRINT_OUTPUT
    std::cout << "C_M"
              << "\t"
              << "C_N"
              << "\t"
              << "nnz_C" << std::endl;
    std::cout << C_M << "\t" << C_N << "\t" << nnz_C << std::endl;

    std::cout << "csr_row_ptr_C: " << std::endl;
    for(aoclsparse_int i = 0; i < C_M + 1; i++)
        std::cout << csr_row_ptr_C[i] << "\t";
    std::cout << std::endl;

    std::cout << "csr_col_ind_C: " << std::endl;
    for(aoclsparse_int i = 0; i < nnz_C; i++)
        std::cout << csr_col_ind_C[i] << "\t";
    std::cout << std::endl;

    std::cout << "csr_val_C: " << std::endl;
    for(aoclsparse_int i = 0; i < nnz_C; i++)
        std::cout << csr_val_C[i] << "\t";
    std::cout << std::endl;
#endif

#endif

    // Note: Errors from destroy functions are ignored.
    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy_mat_descr(descrB);
    aoclsparse_destroy(&csrA);
    aoclsparse_destroy(&csrB);
    aoclsparse_destroy(&csrC);
    return 0;
}
