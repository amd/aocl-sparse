/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#include <cfloat>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>

#define PREMATURE_STOP_TOLERANCE 1e-4
const char gmres_sol[] = "gmres_direct_z";

double calculate_l2Norm_solvers(const aoclsparse_double_complex *xSol,
                                const aoclsparse_double_complex *x,
                                aoclsparse_int                   n)
{
    double norm = 0.0f, norm_diff = 0.0f;
    //xSol obtained by initial spmv
    //x obtained by ILU-GMRES iterations
    for(aoclsparse_int i = 0; i < n; i++)
    {
        aoclsparse_double_complex diff;
        diff.real = xSol[i].real - x[i].real;
        diff.imag = xSol[i].imag - x[i].imag;

        norm = sqrt(pow(diff.real, 2) + pow(diff.imag, 2));
        norm_diff += pow(norm, 2);
    }
    return sqrt(norm_diff);
}

aoclsparse_int monit(aoclsparse_int                   n,
                     const aoclsparse_double_complex *x,
                     const aoclsparse_double_complex *r __attribute__((unused)),
                     double                          *rinfo,
                     void                            *udata __attribute__((unused)))
{
    int    it  = (int)rinfo[30];
    double tol = PREMATURE_STOP_TOLERANCE;

    std::ios oldState(nullptr);
    oldState.copyfmt(std::cout);

    std::ios_base::fmtflags fmt = std::cout.flags();
    fmt |= std::ios_base::scientific | std::ios_base::right | std::ios_base::showpos;

    if(!(it % 5))
    {
        std::cout << std::setw(5) << std::right << " iter"
                  << " " << std::setw(16) << std::right << "residual";
        for(int i = 0; i < n; i++)
            std::cout << std::setw(16) << std::right << "x[" << i << "]";
        std::cout << std::endl;
    }
    std::cout << std::setw(5) << std::right << (int)rinfo[30] << " " << std::setw(16) << std::right
              << std::scientific << std::setprecision(8) << rinfo[0];
    std::cout << std::setprecision(2) << std::showpos;
    for(int i = 0; i < n; i++)
        std::cout << " {" << x[i].real << ", " << x[i].imag << "}";
    std::cout << std::endl;
    std::cout << std::resetiosflags(fmt);

    //reset std::cout state
    std::cout.copyfmt(oldState);
    if(rinfo[0] < tol) // check for premature stop
    {
        return 1; // request to interrupt
    }
    return 0;
}

int main()
{
    std::vector<aoclsparse_int>            csr_row_ptr;
    std::vector<aoclsparse_int>            csr_col_ind;
    std::vector<aoclsparse_double_complex> csr_val;

    int               n, nnz;
    aoclsparse_status exit_status = aoclsparse_status_success;

    std::string filename = "cage4.mtx";
    n                    = 9;
    nnz                  = 49;
    csr_row_ptr.assign({0, 5, 10, 15, 20, 26, 32, 38, 44, 49});
    csr_col_ind.assign({0, 1, 3, 4, 7, 0, 1, 2, 4, 5, 1, 2, 3, 5, 6, 0, 2, 3, 6, 7, 0, 1, 4, 5, 6,
                        8, 1, 2, 4, 5, 7, 8, 2, 3, 4, 6, 7, 8, 0, 3, 5, 6, 7, 8, 4, 5, 6, 7, 8});
    csr_val.assign({{0.75, 0.075}, {0.14, 0.014}, {0.11, 0.011}, {0.14, 0.014}, {0.11, 0.011},
                    {0.08, 0.008}, {0.69, 0.069}, {0.11, 0.011}, {0.08, 0.008}, {0.11, 0.011},
                    {0.09, 0.009}, {0.67, 0.067}, {0.08, 0.008}, {0.09, 0.009}, {0.08, 0.008},
                    {0.09, 0.009}, {0.14, 0.014}, {0.73, 0.073}, {0.14, 0.014}, {0.09, 0.009},
                    {0.04, 0.004}, {0.04, 0.004}, {0.54, 0.054}, {0.14, 0.014}, {0.11, 0.011},
                    {0.25, 0.025}, {0.05, 0.005}, {0.05, 0.005}, {0.08, 0.008}, {0.45, 0.045},
                    {0.08, 0.008}, {0.15, 0.015}, {0.04, 0.004}, {0.04, 0.004}, {0.09, 0.009},
                    {0.47, 0.047}, {0.09, 0.009}, {0.18, 0.018}, {0.05, 0.005}, {0.05, 0.005},
                    {0.14, 0.014}, {0.11, 0.011}, {0.55, 0.055}, {0.25, 0.025}, {0.08, 0.008},
                    {0.08, 0.008}, {0.09, 0.009}, {0.08, 0.008}, {0.17, 0.017}});

    // Create aocl sparse matrix
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_create_zcsr(&A,
                           base,
                           (aoclsparse_int)n,
                           (aoclsparse_int)n,
                           (aoclsparse_int)nnz,
                           csr_row_ptr.data(),
                           csr_col_ind.data(),
                           csr_val.data());
    aoclsparse_create_mat_descr(&descr_a);
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_symmetric);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    aoclsparse_set_mat_index_base(descr_a, base);
    aoclsparse_set_sv_hint(A, trans, descr_a, 100);
    aoclsparse_optimize(A);

    double                    norm       = 0.0;
    double                    rinfo[100] = {0.0};
    aoclsparse_double_complex alpha = {1.0, 0.}, beta = {.0, 0.};
    int                       rs_iters = 5;
    char                      rs_iters_string[16];
    // Initialize initial point x0 and right hand side b
    std::vector<aoclsparse_double_complex> expected_sol(n, {1.0, 0.1});
    std::vector<aoclsparse_double_complex> x(n, {3.0, 0.3});
    std::vector<aoclsparse_double_complex> b(n, {0.0, 0.0});

    aoclsparse_zmv(trans, &alpha, A, descr_a, expected_sol.data(), &beta, b.data());
    aoclsparse_itsol_handle handle = NULL;
    // create GMRES handle
    aoclsparse_itsol_z_init(&handle);

    exit_status = aoclsparse_itsol_option_set(handle, "iterative method", "GMRES");
    if(exit_status != aoclsparse_status_success)
        printf("Warning an iterative method option could not be set, exit status = %d\n",
               exit_status);

    exit_status = aoclsparse_itsol_option_set(handle, "gmres preconditioner", "ILU0");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres preconditioner option could not be set, exit status = %d\n",
               exit_status);

    exit_status = aoclsparse_itsol_option_set(handle, "gmres iteration limit", "50");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres iteration limit option could not be set, exit status = %d\n",
               exit_status);

    sprintf(rs_iters_string, "%d", (int)rs_iters);
    exit_status = aoclsparse_itsol_option_set(handle, "gmres restart iterations", rs_iters_string);
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres restart iterations option could not be set, exit status = %d\n",
               exit_status);

    exit_status = aoclsparse_itsol_option_set(handle, "gmres abs tolerance", "1.0e-04");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres rel tolerance option could not be set, exit status = %d\n",
               exit_status);

    aoclsparse_itsol_handle_prn_options(handle);

    // Call GMRES solver
    aoclsparse_status status;
    status = aoclsparse_itsol_z_solve(
        handle, n, A, descr_a, b.data(), x.data(), rinfo, NULL, monit, &n);
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
            std::cout << std::setw(9) << "{" << x[i].real << ", " << x[i].imag << "}"
                      << " ";
        std::cout << std::endl;
        std::cout << "Expected X* = ";
        for(int i = 0; i < n; i++)
            std::cout << std::setw(9) << "{" << expected_sol[i].real << ", " << expected_sol[i].imag
                      << "}"
                      << " ";
        std::cout << std::endl;
    }
    else if(status == aoclsparse_status_user_stop)
    {
        std::cout << "User stop. [" << gmres_sol << "] Final residual: " << rinfo[0]
                  << ", iterations to converge = " << (int)rinfo[30] << "\n";
    }
    else if(status == aoclsparse_status_maxit)
    {
        std::cout << "Maximum iterations reached. Terminated at iteration  [" << gmres_sol
                  << "] Final residual: " << rinfo[0]
                  << ", iterations to converge = " << (int)rinfo[30] << "\n";
    }
    else
    {
        std::cout << "Something unexpected happened!, Status = " << status << std::endl;
    }
    if(aoclsparse_status_success == status || aoclsparse_status_user_stop == status
       || aoclsparse_status_maxit == status)
    {
        norm = calculate_l2Norm_solvers(&expected_sol[0], &x[0], n);
        printf("L2-Norm between solution and expected complex vectors = %e\n", norm);
    }

    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(&A);
    return 0;
}
