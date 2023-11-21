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

#include <cfloat>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>

#define PREMATURE_STOP_TOLERANCE 1e-4
const char gmres_sol[] = "gmres_direct_d";

double calculate_l2Norm_solvers(const double *xSol, const double *x, aoclsparse_int n)
{
    double norm = 0.0f;
    //residual = xSol - x
    //xSol obtained by initial spmv
    //x obtained by ILU-GMRES iterations
    for(aoclsparse_int i = 0; i < n; i++)
    {
        double a = xSol[i] - x[i];
        norm += a * a;
    }
    norm = sqrt(norm);
    return norm;
}

aoclsparse_int monit(aoclsparse_int n,
                     const double  *x,
                     const double  *r __attribute__((unused)),
                     double        *rinfo,
                     void          *udata __attribute__((unused)))
{
    int    it  = (int)rinfo[30];
    double tol = PREMATURE_STOP_TOLERANCE;

    std::ios oldState(nullptr);
    oldState.copyfmt(std::cout);

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
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<double>         csr_val;

    int               m, n, nnz;
    aoclsparse_status exit_status = aoclsparse_status_success;

    std::string filename = "cage4.mtx";
    //symmetry = 0; //real unsymmetric, symmetry = 0%, spd = no
    //https://www.cise.ufl.edu/research/sparse/MM/vanHeukelum/cage4.tar.gz
    n   = 9;
    m   = 9;
    nnz = 49;
    csr_row_ptr.assign({0, 5, 10, 15, 20, 26, 32, 38, 44, 49});
    csr_col_ind.assign({0, 1, 3, 4, 7, 0, 1, 2, 4, 5, 1, 2, 3, 5, 6, 0, 2, 3, 6, 7, 0, 1, 4, 5, 6,
                        8, 1, 2, 4, 5, 7, 8, 2, 3, 4, 6, 7, 8, 0, 3, 5, 6, 7, 8, 4, 5, 6, 7, 8});
    csr_val.assign({0.75, 0.14, 0.11, 0.14, 0.11, 0.08, 0.69, 0.11, 0.08, 0.11, 0.09, 0.67, 0.08,
                    0.09, 0.08, 0.09, 0.14, 0.73, 0.14, 0.09, 0.04, 0.04, 0.54, 0.14, 0.11, 0.25,
                    0.05, 0.05, 0.08, 0.45, 0.08, 0.15, 0.04, 0.04, 0.09, 0.47, 0.09, 0.18, 0.05,
                    0.05, 0.14, 0.11, 0.55, 0.25, 0.08, 0.08, 0.09, 0.08, 0.17});

    // Create aocl sparse matrix
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_create_dcsr(&A,
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

    // Initialize initial point x0 and right hand side b
    double *expected_sol = NULL;
    double *x            = NULL;
    double *b            = NULL;
    double  norm         = 0.0;
    double  rinfo[100];
    double  alpha = 1.0, beta = 0.;
    int     rs_iters = 7;
    char    rs_iters_string[16];

    expected_sol = new double[n];
    x            = new double[n];
    b            = new double[n];

    double init_x = 1.0, ref_x = 0.5;
    for(int i = 0; i < n; i++)
    {
        expected_sol[i] = ref_x;
        x[i]            = init_x;
    }

    aoclsparse_dmv(trans, &alpha, A, descr_a, expected_sol, &beta, b);

    double                  tol    = PREMATURE_STOP_TOLERANCE;
    aoclsparse_itsol_handle handle = NULL;
    // create GMRES handle
    aoclsparse_itsol_d_init(&handle);

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

    aoclsparse_itsol_handle_prn_options(handle);

    // Call GMRES solver
    aoclsparse_status status;

    status = aoclsparse_itsol_d_solve(handle, n, A, descr_a, b, x, rinfo, NULL, monit, &n);
    if(status == aoclsparse_status_success || status == aoclsparse_status_user_stop
       || aoclsparse_status_maxit == status)
    {
        norm = calculate_l2Norm_solvers(expected_sol, x, n);
        printf("input = %s\n", filename.c_str());
        printf("solver = %s\n", gmres_sol);
        printf("no of rows = %d\n", (int)m);
        printf("no of cols = %d\n", (int)n);
        printf("no of nnz = %d\n", (int)nnz);
        printf("monitoring tolerance = %e\n", tol);
        printf("restart iterations = %d\n", (int)rs_iters);
        printf("residual achieved = %e\n", rinfo[0]);
        printf("total iterations = %d\n", (int)rinfo[30]);
        printf("l2 Norm = %e\n", norm);
    }
    else
    {
        std::cout << "Something unexpected happened!, Status = " << status << std::endl;
    }

    delete[] expected_sol;
    delete[] x;
    delete[] b;
    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy_mat_descr(descr_a);
    aoclsparse_destroy(&A);
    printf("\n");
    fflush(stdout);

    return 0;
}
