/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/* Computes multiplication of 2 sparse matrices A and B in CSR format. */

// Comment this out for single stage computation
//#define TWO_STAGE_COMPUTATION

int main(void)
{
    std::cout << "-------------------------------" << std::endl
              << "----- SP2M sample program -----" << std::endl
              << "-------------------------------" << std::endl
              << std::endl;

    aoclsparse_status     status;
    aoclsparse_int        nnz_C;
    aoclsparse_request    request;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_operation  opA  = aoclsparse_operation_transpose;
    aoclsparse_operation  opB  = aoclsparse_operation_none;

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
    aoclsparse_create_mat_descr(&descrA);
    aoclsparse_create_mat_descr(&descrB);

    // Matrix sizes
    aoclsparse_int m = 5, n = 5, k = 5;
    aoclsparse_int nnz_A = 10, nnz_B = 10;
    // Matrix A
    aoclsparse_int            row_ptr_A[] = {0, 1, 2, 5, 9, 10};
    aoclsparse_int            col_ind_A[] = {0, 0, 1, 2, 4, 0, 1, 2, 3, 4};
    aoclsparse_double_complex val_A[]     = {{-0.86238, 0.454626},
                                             {-2.62138, -0.442597},
                                             {-0.875679, 0.137933},
                                             {-0.661939, -1.09106},
                                             {0.0501717, -2.37527},
                                             {-1.48812, -0.420546},
                                             {-0.588085, -0.708977},
                                             {0.310933, -0.96569},
                                             {-0.88964, -2.37881},
                                             {-1.23201, 0.213152}};
    aoclsparse_create_zcsr(
        csrA, base, k, m, nnz_A, row_ptr_A, col_ind_A, (aoclsparse_double_complex *)val_A);

    // Matrix B
    aoclsparse_int            row_ptr_B[] = {0, 4, 4, 7, 8, 10};
    aoclsparse_int            col_ind_B[] = {0, 1, 2, 4, 0, 1, 2, 2, 2, 3};
    aoclsparse_double_complex val_B[]     = {{-1.59204, -0.259325},
                                             {0.467532, -0.980612},
                                             {0.078412, -0.513591},
                                             {-1.52364, 0.403911},
                                             {0.211966, -1.33485},
                                             {-1.37901, -1.44562},
                                             {1.42472, -2.08662},
                                             {-2.26549, -1.0073},
                                             {-1.75098, 0.207783},
                                             {-1.8152, 0.482205}};

    aoclsparse_create_zcsr(
        csrB, base, k, n, nnz_B, row_ptr_B, col_ind_B, (aoclsparse_double_complex *)val_B);

    aoclsparse_matrix          csrC          = NULL;
    aoclsparse_int            *csr_row_ptr_C = NULL;
    aoclsparse_int            *csr_col_ind_C = NULL;
    aoclsparse_double_complex *csr_val_C     = NULL;
    aoclsparse_int             C_M, C_N;

    // expected output matrices
    aoclsparse_int            C_M_exp = 5, C_N_exp = 5, nnz_C_exp = 15;
    aoclsparse_int            csr_row_ptr_C_exp[] = {0, 4, 7, 10, 11, 15};
    aoclsparse_int            csr_col_ind_C_exp[] = {0, 1, 2, 4, 0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 3};
    aoclsparse_double_complex csr_val_C_exp[]     = {{1.49084, -0.500145},
                                                     {0.0426217, 1.05821},
                                                     {3.11358, 2.93029},
                                                     {1.13033, -1.04101},
                                                     {-0.0014949, 1.19813},
                                                     {1.40697, 1.07569},
                                                     {-0.34163, 4.22229},
                                                     {-1.59671, 0.652319},
                                                     {-0.664445, 2.4615},
                                                     {-4.89686, 1.70132},
                                                     {-0.380707, 6.28532},
                                                     {-3.15998, -0.570448},
                                                     {-3.50293, 3.20298},
                                                     {-2.77187, -4.11799},
                                                     {2.13356, -0.980994}};

#ifdef TWO_STAGE_COMPUTATION
    std::cout << "Invoking aoclsparse_sp2m with aoclsparse_stage_nnz_count..\n";
    // aoclsparse_stage_nnz_count : Only rowIndex array of the CSR matrix
    // is computed internally.
    request = aoclsparse_stage_nnz_count;
    status  = aoclsparse_sp2m(opA, descrA, csrA, opB, descrB, csrB, request, &csrC);
    if(status != aoclsparse_status_success)
        return 1;

    std::cout << "Invoking aoclsparse_sp2m with aoclsparse_stage_finalize..\n";
    // aoclsparse_stage_finalize : Finalize computation of remaining
    // output arrays ( column indices and values of output matrix entries) .
    // Has to be called only after aoclsparse_sp2m call with
    // aoclsparse_stage_nnz_count parameter.
    request = aoclsparse_stage_finalize;
    status  = aoclsparse_sp2m(opA, descrA, csrA, opB, descrB, csrB, request, &csrC);
    if(status != aoclsparse_status_success)
        return 2;

#else // SINGLE STAGE
    std::cout << "Invoking aoclsparse_sp2m with aoclsparse_stage_full_computation..\n";
    // aoclsparse_stage_full_computation :  Whole computation is performed in
    // single step.
    request = aoclsparse_stage_full_computation;
    status  = aoclsparse_sp2m(opA, descrA, csrA, opB, descrB, csrB, request, &csrC);
    if(status != aoclsparse_status_success)
        return 3;

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
              << std::setw(3) << "" << std::setw(11) << "C_N" << std::setw(3) << "" << std::setw(11)
              << "expected" << std::setw(3) << "" << std::setw(11) << "nnz_C" << std::setw(3) << ""
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
    std::cout << std::setw(11) << "csr_val_C" << std::setw(3) << "" << std::setw(11) << "expected"
              << std::setw(3) << "" << std::setw(11) << "csr_col_ind_C" << std::setw(3) << ""
              << std::setw(11) << "expected" << std::setw(3) << "" << std::setw(11)
              << "csr_row_ptr_C" << std::setw(3) << "" << std::setw(11) << "expected" << std::endl;
    //Initializing precision tolerance range for double
    const double tol = 1e-03;
    for(aoclsparse_int i = 0; i < nnz_C; i++)
    {
        oki = ((std::abs(csr_val_C[i].real - csr_val_C_exp[i].real) <= tol)
               && (std::abs(csr_val_C[i].imag - csr_val_C_exp[i].imag) <= tol));
        ok &= oki;
        std::cout << std::setw(11) << "(" << csr_val_C[i].real << ", " << csr_val_C[i].imag << "i) "
                  << std::setw(3) << "" << std::setw(11) << "(" << csr_val_C_exp[i].real << ", "
                  << csr_val_C_exp[i].imag << "i) " << std::setw(2) << (oki ? "" : " !");
        okj = csr_col_ind_C[i] == csr_col_ind_C_exp[i];
        ok &= okj;
        std::cout << std::setw(11) << csr_col_ind_C[i] << std::setw(3) << "" << std::setw(11)
                  << csr_col_ind_C_exp[i] << std::setw(2) << (okj ? "" : " !");
        if(i < C_M)
        {
            okk = csr_row_ptr_C[i] == csr_row_ptr_C_exp[i];
            ok &= okk;
            std::cout << " " << std::setw(11) << csr_row_ptr_C[i] << std::setw(3) << ""
                      << std::setw(11) << csr_row_ptr_C_exp[i] << std::setw(2) << (okk ? "" : " !");
        }
        std::cout << std::endl;
    }

    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy_mat_descr(descrB);
    aoclsparse_destroy(csrA);
    aoclsparse_destroy(csrB);
    aoclsparse_destroy(csrC);
    return (ok ? 0 : 6);
}
