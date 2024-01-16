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
              << "----- syrk sample program -----" << std::endl
              << "-------------------------------" << std::endl
              << std::endl;

    aoclsparse_matrix     A;
    aoclsparse_matrix     C;
    aoclsparse_status     status;
    aoclsparse_index_base base          = aoclsparse_index_base_zero;
    aoclsparse_int       *csr_row_ptr_C = NULL;
    aoclsparse_int       *csr_col_ind_C = NULL;
    double               *csr_val_C     = NULL;
    aoclsparse_int        C_m, C_n, C_nnz;

    // Matrix sizes
    aoclsparse_int m = 4, k = 3;
    aoclsparse_int nnz = 7;
    // Matrix A
    // [ 0.  -1.2   2.3]
    // [ 4.6  0.    0. ]
    // [ 0.   3.0  -8.1 ]
    // [ 0.3   0.  -5.1 ]
    aoclsparse_int       row_ptr[] = {0, 2, 3, 5, 7};
    aoclsparse_int       col_ind[] = {1, 2, 0, 1, 2, 0, 2};
    double               csr_val[] = {-1.2, 2.3, 4.6, 3.0, -8.1, 0.3, -5.1};
    aoclsparse_operation op        = aoclsparse_operation_none;

    status = aoclsparse_create_dcsr(&A, base, m, k, nnz, row_ptr, col_ind, csr_val);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    // expected output matrices
    aoclsparse_int C_m_exp = 4, C_n_exp = 4, C_nnz_exp = 8;
    aoclsparse_int csr_row_ptr_C_exp[] = {0, 3, 5, 7, 8};
    aoclsparse_int csr_col_ind_C_exp[] = {0, 2, 3, 1, 3, 2, 3, 3};
    double         csr_val_C_exp[]
        = {6.73000, -22.23000, -11.73000, 21.16000, 1.38000, 74.61000, 41.31000, 26.10000};

    // output matrices;

    status = aoclsparse_syrk(op, A, &C);
    if(status != aoclsparse_status_success)
    {
        return status;
    }
    aoclsparse_export_dcsr(
        C, &base, &C_m, &C_n, &C_nnz, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C);

    bool oka, okb, okc, oki, okj, okk, ok = true;
    std::cout << std::endl
              << "Output Matrix C: " << std::endl
              << std::setw(11) << "C_m" << std::setw(3) << "" << std::setw(11) << "expected"
              << std::setw(3) << "" << std::setw(11) << "C_n" << std::setw(3) << "" << std::setw(11)
              << "expected" << std::setw(3) << "" << std::setw(11) << "C_nnz" << std::setw(3) << ""
              << std::setw(11) << "expected" << std::endl;
    oka = C_m == C_m_exp;
    ok &= oka;
    std::cout << std::setw(11) << C_m << std::setw(3) << "" << std::setw(11) << C_m_exp
              << std::setw(2) << (oka ? "" : " !");
    okb = C_n == C_n_exp;
    ok &= okb;
    std::cout << std::setw(11) << C_n << std::setw(3) << "" << std::setw(11) << C_n_exp
              << std::setw(2) << (okb ? "" : " !");
    okc = C_nnz == C_nnz_exp;
    ok &= okc;
    std::cout << std::setw(11) << C_nnz << std::setw(3) << "" << std::setw(11) << C_nnz_exp
              << std::setw(2) << (okc ? "" : " !");
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(11) << "csr_val_C" << std::setw(3) << "" << std::setw(11) << "expected"
              << std::setw(3) << "" << std::setw(11) << "csr_col_ind_C" << std::setw(3) << ""
              << std::setw(11) << "expected" << std::setw(3) << "" << std::setw(11)
              << "csr_row_ptr_C" << std::setw(3) << "" << std::setw(11) << "expected" << std::endl;
    //Initializing precision tolerance range for double
    const double tol = 1e-03;
    for(aoclsparse_int i = 0; i < C_nnz; i++)
    {
        oki = ((std::abs(csr_val_C[i] - csr_val_C_exp[i]) <= tol));
        ok &= oki;
        std::cout << std::setw(11) << csr_val_C[i] << "" << std::setw(11) << csr_val_C_exp[i]
                  << (oki ? "" : " !");
        okj = csr_col_ind_C[i] == csr_col_ind_C_exp[i];
        ok &= okj;
        std::cout << std::setw(11) << csr_col_ind_C[i] << std::setw(3) << "" << std::setw(11)
                  << csr_col_ind_C_exp[i] << std::setw(2) << (okj ? "" : " !");
        if(i <= C_m)
        {
            okk = csr_row_ptr_C[i] == csr_row_ptr_C_exp[i];
            ok &= okk;
            std::cout << " " << std::setw(11) << csr_row_ptr_C[i] << std::setw(3) << ""
                      << std::setw(11) << csr_row_ptr_C_exp[i] << std::setw(2) << (okk ? "" : " !");
        }
        std::cout << std::endl;
    }
    aoclsparse_destroy(&A);
    aoclsparse_destroy(&C);

    return (ok ? 0 : 6);
}
