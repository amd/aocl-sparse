/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <iomanip>
#include <iostream>
#include <math.h>

// Define custom log printer
void printer(float rinfo[100], bool header)
{
    std::ios_base::fmtflags fmt = std::cout.flags();
    fmt |= std::cout.scientific | std::cout.right;
    if(header)
        std::cout << std::setw(5) << std::right << " iter"
                  << " " << std::setw(16) << std::right << "optim"
                  << "  " << std::setw(3) << std::endl;
    std::cout << std::setw(5) << std::right << (int)rinfo[30] << " " << std::setw(16) << std::right
              << std::scientific << rinfo[0] << "  " << std::endl;
    std::resetiosflags(fmt);
}

int main()
{
    // CSR symmetric matrix. Only the lower triangle is stored
    aoclsparse_int icrow[9] = {0, 1, 2, 5, 6, 8, 11, 15, 18};
    aoclsparse_int icol[18] = {0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
    float          a[18]    = {19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9};
    aoclsparse_int n = 8, nnz = 18;

    // Create aocl sparse matrix
    // csrmv cannot be used with aoclsparse_matrix at thi time TODO
    /* aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_create_scsr(A, base, n, n, nnz, icrow, icol, a);
    aoclsparse_optimize(A); */

    // Create matrix descriptor
    aoclsparse_mat_descr descr_a;
    aoclsparse_create_mat_descr(&descr_a);
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_symmetric);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);

    // create CG handle
    aoclsparse_itsol_handle handle = nullptr;
    aoclsparse_itsol_s_init(&handle);

    // Change options (update to use )
    if(aoclsparse_itsol_option_set(handle, "CG Abs Tolerance", "5.0e-06")
       != aoclsparse_status_success)
        printf("Warning an option could not be set\n");

    // Initialize initial point x0 and right hand side b
    float                x[n]            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float                b[n]            = {0.0};
    float                expected_sol[n] = {1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0};
    float                alpha = 1.0, beta = 0.;
    float                y[n];
    aoclsparse_operation trans = aoclsparse_operation_none;
    aoclsparse_scsrmv(trans, &alpha, n, n, nnz, a, icol, icrow, descr_a, expected_sol, &beta, b);

    // initialize size and rhs inside the handle
    aoclsparse_itsol_s_rci_input(handle, n, b);

    // Call CG solver
    aoclsparse_itsol_rci_job ircomm = aoclsparse_rci_start;
    aoclsparse_status        status;
    float*                   u = nullptr;
    float*                   v = nullptr;
    float                    rinfo[100];
    float                    tol = 1.0e-10;
    bool                     hdr;
    std::cout << std::endl;
    while(ircomm != aoclsparse_rci_stop)
    {
        status = aoclsparse_itsol_s_rci_solve(handle, &ircomm, &u, &v, x, rinfo);
        if(status != aoclsparse_status_success)
            break;
        switch(ircomm)
        {
        case aoclsparse_rci_mv:
            // Compute v = Au
            beta  = 0.0;
            alpha = 1.0;
            aoclsparse_scsrmv(trans, &alpha, n, n, nnz, a, icol, icrow, descr_a, u, &beta, v);
            break;

        case aoclsparse_rci_precond:
            //symgs_ref_avx(alpha, n, a, icol, icrow, u, v, y);
            for(int i = 0; i < n; i++)
                v[i] = u[i];
            break;

        case aoclsparse_rci_stopping_criterion:
            // print iteration log
            hdr = ((int)rinfo[30] % 100) == 0;
            printer(rinfo, hdr);
            if(rinfo[0] < tol)
            {
                std::cout << "User stop. Final residual: " << rinfo[0] << std::endl;
                ircomm = aoclsparse_rci_interrupt;
            }
            break;

        default:
            break;
        }
    }
    switch(status)
    {
    case aoclsparse_status_success:
        std::cout.precision(2);
        std::cout << std::scientific;
        aoclsparse_itsol_handle_prn_options(handle);
        std::cout << std::endl
                  << "Solution found: (residual = " << rinfo[0] << " in " << (int)rinfo[30]
                  << " iterations)" << std::endl
                  << "   Final X* = ";
        for(int i = 0; i < n; i++)
            std::cout << std::setw(9) << x[i] << " ";
        std::cout << std::endl;
        std::cout << "Expected X* = ";
        for(int i = 0; i < n; i++)
            std::cout << std::setw(9) << expected_sol[i] << " ";
        std::cout << std::endl;
        break;

    case aoclsparse_status_maxit:
        std::cout << "solve stopped after " << (int)rinfo[30] << " iterations" << std::endl
                  << "residual = " << rinfo[0] << std::endl;

    default:
        std::cout << "Something unexpected happened!" << std::endl;
    }
    aoclsparse_itsol_destroy(&handle);

    return 0;
}
