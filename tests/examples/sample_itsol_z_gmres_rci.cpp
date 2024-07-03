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
const char gmres_sol[] = "gmres_rci_z";

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

    // create matrix descriptor
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
    aoclsparse_set_sv_hint(A, trans, descr_a, 100);
    aoclsparse_set_mat_index_base(descr_a, base);
    aoclsparse_set_lu_smoother_hint(A, trans, descr_a, 100);
    aoclsparse_optimize(A);

    aoclsparse_double_complex alpha = {1.0, 0.}, beta = {.0, 0.};
    double                    norm     = 0.0;
    int                       rs_iters = 5;
    char                      rs_iters_string[16];

    // Initialize initial point x0 and right hand side b
    std::vector<aoclsparse_double_complex> expected_sol(n, {0.5, 0.05});
    std::vector<aoclsparse_double_complex> x(n, {1.0, 0.1});
    std::vector<aoclsparse_double_complex> b(n, {0.0, 0.0});

    aoclsparse_zmv(trans, &alpha, A, descr_a, &expected_sol[0], &beta, &b[0]);
    aoclsparse_itsol_handle handle = nullptr;
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

    sprintf(rs_iters_string, "%d", rs_iters);
    exit_status = aoclsparse_itsol_option_set(handle, "gmres restart iterations", rs_iters_string);
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres restart iterations option could not be set, exit status = %d\n",
               exit_status);

    aoclsparse_itsol_handle_prn_options(handle);

    // initialize size and rhs inside the handle
    aoclsparse_itsol_z_rci_input(handle, n, &b[0]);
    // Call GMRES solver
    aoclsparse_itsol_rci_job   ircomm = aoclsparse_rci_start;
    aoclsparse_status          status;
    aoclsparse_double_complex *io1        = nullptr;
    aoclsparse_double_complex *io2        = nullptr;
    double                     rinfo[100] = {0.0};
    double                     tol        = PREMATURE_STOP_TOLERANCE;
    aoclsparse_double_complex *pcsr_val   = nullptr;

    while(ircomm != aoclsparse_rci_stop)
    {
        status = aoclsparse_itsol_z_rci_solve(handle, &ircomm, &io1, &io2, &x[0], rinfo);
        if(status != aoclsparse_status_success)
            break;
        switch(ircomm)
        {
        case aoclsparse_rci_mv:
            //User can instead use their custom implementation of SpMV routine
            // Compute v = Au
            aoclsparse_zmv(trans, &alpha, A, descr_a, io1, &beta, io2);
            break;

        case aoclsparse_rci_precond:
            //User can instead use their custom implementation of ILU Preconditioner
            //Run ILU Preconditioner only once in the beginning
            //Run Triangular Solve using ILU0 factorization
            aoclsparse_zilu_smoother(trans,
                                     A,
                                     descr_a,
                                     &pcsr_val,
                                     NULL,
                                     io2, //x = ?, io1 = z+j*n,
                                     (const aoclsparse_double_complex *)io1); //rhs, io2 = v+j*n
            break;

        case aoclsparse_rci_stopping_criterion:
            if(rinfo[0] < tol)
            {
                std::cout << "User stop. [" << gmres_sol << "] Final residual: " << rinfo[0]
                          << ", iterations to converge = " << (int)rinfo[30] << "\n";
                ircomm = aoclsparse_rci_interrupt;
            }
            break;

        default:
            break;
        }
    }
    if(aoclsparse_status_success == status || aoclsparse_status_user_stop == status
       || aoclsparse_status_maxit == status)
    {
        norm = calculate_l2Norm_solvers(&expected_sol[0], &x[0], n);
        printf("L2-Norm between solution and expected complex vectors = %e\n", norm);
    }
    else
    {
        std::cout << "Something unexpected happened!, Status = " << status << std::endl;
    }

    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(&A);
    return 0;
}
