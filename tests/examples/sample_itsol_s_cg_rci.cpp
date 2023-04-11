/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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
#include <vector>

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
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<float>          aval;
    aoclsparse_int              n = 8, nnz = 18;
    icrow.assign({0, 1, 2, 5, 6, 8, 11, 15, 18});
    icol.assign({0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7});
    aval.assign({19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9});

    // Create aocl sparse matrix and its descriptor
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_create_scsr(A, base, n, n, nnz, icrow.data(), icol.data(), aval.data());
    aoclsparse_create_mat_descr(&descr_a);
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_symmetric);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    aoclsparse_set_sv_hint(A, trans, descr_a, 100);
    aoclsparse_optimize(A);

    // create CG handle
    aoclsparse_itsol_handle handle = nullptr;
    aoclsparse_itsol_s_init(&handle);

    // Change options (update to use )
    if(aoclsparse_itsol_option_set(handle, "CG Rel Tolerance", "1.0e-06")
           != aoclsparse_status_success
       || aoclsparse_itsol_option_set(handle, "CG preconditioner", "user")
              != aoclsparse_status_success)
        std::cout << "Warning an option could not be set" << std::endl;

    // Initialize initial point x0 and right hand side b
    std::vector<float> x, b, expected_sol, y;
    x.assign({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    b.assign({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    expected_sol.assign({1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0});
    float alpha = 1.0, beta = 0.;
    y.resize(n);
    aoclsparse_smv(trans, &alpha, A, descr_a, expected_sol.data(), &beta, b.data());

    // initialize size and rhs inside the handle
    aoclsparse_itsol_s_rci_input(handle, n, b.data());

    // Call CG solver
    aoclsparse_itsol_rci_job ircomm = aoclsparse_rci_start;
    aoclsparse_status        status;
    float                   *u = nullptr;
    float                   *v = nullptr;
    float                    rinfo[100];
    float                    tol = 1.0e-5;
    bool                     hdr;
    std::cout << std::endl;
    while(ircomm != aoclsparse_rci_stop)
    {
        status = aoclsparse_itsol_s_rci_solve(handle, &ircomm, &u, &v, x.data(), rinfo);
        if(status != aoclsparse_status_success)
            break;
        switch(ircomm)
        {
        case aoclsparse_rci_mv:
            // Compute v = Au
            beta   = 0.0;
            alpha  = 1.0;
            status = aoclsparse_smv(trans, &alpha, A, descr_a, u, &beta, v);
            if(status != aoclsparse_status_success)
                ircomm = aoclsparse_rci_stop;
            break;

        case aoclsparse_rci_precond:
            // apply Symmetric Gauss-Seidel preconditioner step
            status = aoclsparse_strsv(aoclsparse_operation_none, alpha, A, descr_a, u, y.data());
            if(status != aoclsparse_status_success)
            {
                ircomm = aoclsparse_rci_stop;
                break;
            }
            for(aoclsparse_int i = 0; i < n; i++)
                y[i] *= aval[icrow[i + 1] - 1];
            status
                = aoclsparse_strsv(aoclsparse_operation_transpose, alpha, A, descr_a, y.data(), v);
            if(status != aoclsparse_status_success)
                ircomm = aoclsparse_rci_stop;
            break;

        case aoclsparse_rci_stopping_criterion:
            // No operations required, can be used to monitor the progress of the solve
            // or defining a custom stopping criterion
            // print iteration log
            hdr = ((int)rinfo[30] % 100) == 0;
            printer(rinfo, hdr);
            // request solver to stop if custom criterion is met
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
    // Print the final results if the internal stopping criterion or the user defined one were met
    switch(status)
    {
    case aoclsparse_status_user_stop:
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
        break;

    default:
        std::cout << "Something unexpected happened!" << std::endl;
    }
    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(A);

    return 0;
}
