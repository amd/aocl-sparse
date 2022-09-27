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
#include <vector>

aoclsparse_int monit(const double *x, const double *r, double *rinfo, void *udata)
{
    int                     it  = (int)rinfo[30];
    int                     n   = *(int *)(udata);
    std::ios_base::fmtflags fmt = std::cout.flags();
    fmt |= std::ios_base::scientific | std::ios_base::right | std::ios_base::showpos;
    if(!(it % 10))
    {
        std::cout << std::setw(5) << std::right << " iter"
                  << " " << std::setw(16) << std::right << "optim";
        for(int i = 0; i < n; i++)
            std::cout << std::setw(8) << std::right << "x[" << i << "]";
        std::cout << std::endl;
    }
    std::cout << std::setw(5) << std::right << (int)rinfo[30] << " " << std::setw(16) << std::right
              << std::scientific << std::setprecision(8) << rinfo[0];
    std::cout << std::setprecision(2) << std::showpos;
    for(int i = 0; i < n; i++)
        std::cout << " " << x[i];
    std::cout << std::endl;
    std::cout << std::resetiosflags(fmt);
    if(rinfo[0] < 1.0e-12) // check for premature stop
        return 1; // request to interrupt
    return 0;
}

int main()
{
    // CSR symmetric matrix. Only the lower triangle is stored
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<double>         aval;
    aoclsparse_int              n = 8, nnz = 18;
    icrow.assign({0, 1, 2, 5, 6, 8, 11, 15, 18});
    icol.assign({0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7});
    aval.assign({19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9});

    // Create aocl sparse matrix
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_create_dcsr(A, base, n, n, nnz, icrow.data(), icol.data(), aval.data());
    aoclsparse_create_mat_descr(&descr_a);
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_symmetric);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    aoclsparse_set_sv_hint(A, trans, descr_a, 100);
    aoclsparse_optimize(A);

    // Initialize initial point x0 and right hand side b
    std::vector<double> x, b, expected_sol, y;
    x.assign({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    b.assign({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    expected_sol.assign({1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0, 1.E0, 0.E0});
    double alpha = 1.0, beta = 0.;
    y.resize(n);
    aoclsparse_dmv(trans, &alpha, A, descr_a, expected_sol.data(), &beta, b.data());

    // create CG handle
    aoclsparse_itsol_handle handle = nullptr;
    aoclsparse_itsol_d_init(&handle);

    if(aoclsparse_itsol_option_set(handle, "CG Abs Tolerance", "5.0e-6")
           != aoclsparse_status_success
       || aoclsparse_itsol_option_set(handle, "CG Preconditioner", "SGS")
              != aoclsparse_status_success)
        std::cout << "Warning an option could not be set" << std::endl;

    // Call CG solver
    aoclsparse_status status;
    double            rinfo[100];
    status = aoclsparse_itsol_d_solve(
        handle, n, A, descr_a, b.data(), x.data(), rinfo, nullptr, monit, &n);
    if(status == aoclsparse_status_success)
    {
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
    }
    else if(status == aoclsparse_status_user_stop)
    {
        std::cout << "User requested to terminate at interation " << (aoclsparse_int)rinfo[30]
                  << std::endl;
    }
    else
        std::cout << "Something unexpected happened. Status = " << status << std::endl;

    aoclsparse_itsol_destroy(&handle);

    return 0;
}
