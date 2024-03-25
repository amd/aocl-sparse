/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iomanip>
#include <iostream>

/* Computes triple symmetric product of complex sparse matrices
 * C:=opA(A)*B*A, where opA = conjugate transpose
*/

// Comment this out for single stage computation
//#define TWO_STAGE_COMPUTATION

int main(void)
{
    std::cout << "-------------------------------" << std::endl
              << "----- SYPR sample program -----" << std::endl
              << "-------------------------------" << std::endl
              << std::endl;

    aoclsparse_status     status;
    aoclsparse_int        nnz_C;
    aoclsparse_request    request;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_operation  opA  = aoclsparse_operation_conjugate_transpose;

    // Print aoclsparse version
    std::cout << aoclsparse_get_version() << std::endl;

    // Initialise matrix descriptor and csr matrix structure of inputs A and B
    aoclsparse_mat_descr descrB;
    aoclsparse_matrix    csrA;
    aoclsparse_matrix    csrB;

    // Create matrix descriptor of input matrices
    status = aoclsparse_create_mat_descr(&descrB);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_create_mat_descr, status = " << status << "."
                  << std::endl;
        return 1;
    }

    aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_hermitian);
    aoclsparse_set_mat_fill_mode(descrB, aoclsparse_fill_mode_lower);
    aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_zero);

    // Matrix sizes
    aoclsparse_int m_a = 5, n_a = 5, m_b = 5, n_b = 5;
    aoclsparse_int nnz_A = 10, nnz_B = 9;
    // Matrix A
    aoclsparse_int            row_ptr_A[] = {0, 1, 2, 5, 9, 10};
    aoclsparse_int            col_ind_A[] = {0, 0, 1, 2, 4, 0, 1, 2, 3, 4};
    aoclsparse_double_complex val_A[]     = {{-0.86, 0.45},
                                             {-2.62, -0.44},
                                             {-0.87, 0.13},
                                             {-0.66, -1.09},
                                             {0.05, -2.37},
                                             {-1.48, -0.42},
                                             {-0.58, -0.70},
                                             {0.31, -0.96},
                                             {-0.88, -2.37},
                                             {-1.23, 0.21}};
    status = aoclsparse_create_zcsr(&csrA, base, m_a, n_a, nnz_A, row_ptr_A, col_ind_A, val_A);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_create_zcsr, status = " << status << "."
                  << std::endl;
        return 2;
    }

    // Hermitian Matrix B
    aoclsparse_int            row_ptr_B[] = {0, 1, 2, 4, 6, 9};
    aoclsparse_int            col_ind_B[] = {0, 1, 0, 2, 1, 3, 1, 2, 4};
    aoclsparse_double_complex val_B[]     = {{-1.59, 0},
                                             {0.46, 0},
                                             {0.07, -0.51},
                                             {-1.52, 0},
                                             {0.21, -1.33},
                                             {-1.37, 0},
                                             {1.42, -2.08},
                                             {-2.26, -1.00},
                                             {-1.81, 0}};

    status = aoclsparse_create_zcsr(&csrB, base, m_b, n_b, nnz_B, row_ptr_B, col_ind_B, val_B);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_create_zcsr, status = " << status << "."
                  << std::endl;
        return 3;
    }

    aoclsparse_matrix          csrC          = NULL;
    aoclsparse_int            *csr_row_ptr_C = NULL;
    aoclsparse_int            *csr_col_ind_C = NULL;
    aoclsparse_double_complex *csr_val_C     = NULL;
    aoclsparse_int             C_M, C_N;

    // expected output matrix
    aoclsparse_int            C_M_exp = 5, C_N_exp = 5, nnz_C_exp = 14;
    aoclsparse_int            csr_row_ptr_C_exp[] = {0, 5, 9, 12, 13, 14};
    aoclsparse_int            csr_col_ind_C_exp[] = {0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 4, 3, 3, 4};
    aoclsparse_double_complex csr_val_C_exp[]     = {{-0.982439, 0.000000},
                                                     {-3.380974, 2.107664},
                                                     {-4.156461, -1.960739},
                                                     {-10.188348, 1.376974},
                                                     {5.609324, 4.536282},
                                                     {-2.308344, 0.000000},
                                                     {-1.331714, -2.631938},
                                                     {-2.972078, -1.039282},
                                                     {-1.922892, -1.975280},
                                                     {-3.862273, 0.000000},
                                                     {-3.714510, 1.465694},
                                                     {-2.743288, 2.163915},
                                                     {-8.756081, -0.000000},
                                                     {-3.022874, 0.000000}};

#ifdef TWO_STAGE_COMPUTATION
    std::cout << "Invoking aoclsparse_sypr with aoclsparse_stage_nnz_count...\n";
    // aoclsparse_stage_nnz_count: Only rowIndex array of the CSR matrix
    // is computed internally.
    request = aoclsparse_stage_nnz_count;
    status  = aoclsparse_sypr(opA, csrA, csrB, descrB, &csrC, request);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_sypr, status = " << status << "." << std::endl;
        return 4;
    }

    std::cout << "Invoking aoclsparse_sypr with aoclsparse_stage_finalize...\n";
    // aoclsparse_stage_finalize: Finalize computation of remaining
    // output arrays (column indices and values of output matrix entries).
    // Has to be called only after aoclsparse_sypr call with
    // aoclsparse_stage_nnz_count parameter.
    request = aoclsparse_stage_finalize;
    status  = aoclsparse_sypr(opA, csrA, csrB, descrB, &csrC, request);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_sypr, status = " << status << "." << std::endl;
        return 5;
    }

#else // SINGLE STAGE
    std::cout << "Invoking aoclsparse_sypr with aoclsparse_stage_full_computation...\n";
    // aoclsparse_stage_full_computation: Whole computation is performed in
    // a single step.
    request = aoclsparse_stage_full_computation;
    status  = aoclsparse_sypr(opA, csrA, csrB, descrB, &csrC, request);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_sypr, status = " << status << "." << std::endl;
        return 6;
    }

#endif

    aoclsparse_export_zcsr(
        csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C);
    // Check and print the result
    std::cout << std::fixed;
    std::cout.precision(1);
    bool oka, okb, okc, oki, okj, okk, ok = true;
    std::cout << std::endl
              << "Output Matrix C: " << std::endl
              << std::setw(11) << "C_M" << std::setw(3) << "" << std::setw(11) << "expected"
              << std::setw(2) << "" << std::setw(11) << "C_N" << std::setw(3) << "" << std::setw(11)
              << "expected" << std::setw(2) << "" << std::setw(11) << "nnz_C" << std::setw(3) << ""
              << std::setw(11) << "expected" << std::endl;
    oka = C_M == C_M_exp;
    ok &= oka;
    std::cout << std::setw(11) << C_M << std::setw(3) << "" << std::setw(11) << C_M_exp
              << std::setw(2) << (oka ? "" : " !");
    okb = C_N == C_N_exp;
    ok &= okb;
    std::cout << std::setw(11) << C_N << std::setw(3) << "" << std::setw(11) << C_N_exp
              << std::setw(2) << (okb ? "" : " !");
    okc = nnz_C == nnz_C_exp;
    ok &= okc;
    std::cout << std::setw(11) << nnz_C << std::setw(3) << "" << std::setw(11) << nnz_C_exp
              << std::setw(2) << (okc ? "" : " !");
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(14) << "csr_val_C" << std::setw(16) << "expected" << std::setw(14)
              << "csr_col_ind_C" << std::setw(14) << "expected" << std::setw(14) << "csr_row_ptr_C"
              << std::setw(14) << "expected" << std::endl;
    // Initializing precision tolerance range for double
    const double tol = 1e-06;
    for(aoclsparse_int i = 0; i < nnz_C; i++)
    {
        oki = ((std::abs(csr_val_C[i].real - csr_val_C_exp[i].real) <= tol)
               && (std::abs(csr_val_C[i].imag - csr_val_C_exp[i].imag) <= tol));
        ok &= oki;
        std::cout << "(" << std::setw(5) << csr_val_C[i].real << "," << std::setw(5)
                  << csr_val_C[i].imag << "i) "
                  << " (" << std::setw(5) << csr_val_C_exp[i].real << "," << std::setw(5)
                  << csr_val_C_exp[i].imag << "i) " << std::setw(2) << (oki ? "" : " !");
        okj = csr_col_ind_C[i] == csr_col_ind_C_exp[i];
        ok &= okj;
        std::cout << std::setw(11) << csr_col_ind_C[i] << std::setw(3) << "" << std::setw(11)
                  << csr_col_ind_C_exp[i] << std::setw(2) << (okj ? "" : " !");
        if(i <= C_M)
        {
            okk = csr_row_ptr_C[i] == csr_row_ptr_C_exp[i];
            ok &= okk;
            std::cout << " " << std::setw(11) << csr_row_ptr_C[i] << std::setw(3) << ""
                      << std::setw(11) << csr_row_ptr_C_exp[i] << std::setw(2) << (okk ? "" : " !");
        }
        std::cout << std::endl;
    }

    aoclsparse_destroy_mat_descr(descrB);
    aoclsparse_destroy(&csrA);
    aoclsparse_destroy(&csrB);
    aoclsparse_destroy(&csrC);
    return (ok ? 0 : 7);
}
