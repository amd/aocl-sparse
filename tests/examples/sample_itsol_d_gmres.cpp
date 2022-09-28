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

#include <cfloat>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>

#define PARAM_LOG
#define DOUBLE_PRECISION_TOLERANCE 1e-6
#define RUN_MODE 0
int        run_mode;
const char gmres_sol[] = "gmres_direct_d";

template <typename T>
void clear_local_resource(T *x)
{
    if(x != NULL)
    {
        free(x);
        x = NULL;
    }
}

double calculate_l2Norm_solvers(const double *xvRef, const double *xvTest, aoclsparse_int n)
{
    double norm = 0.0f;
    //residual = xvRef - xvTest
    //xvRef obtained by initial spmv
    //xvTest obtained by ILU iterations
    for(aoclsparse_int i = 0; i < n; i++)
    {
        double a = xvRef[i] - xvTest[i];
        norm += a * a;
    }
    norm = sqrt(norm);
    return norm;
}

aoclsparse_int monit(aoclsparse_int n, const double *x, const double *r, double *rinfo, void *udata)
{
    int    it  = (int)rinfo[30];
    double tol = DOUBLE_PRECISION_TOLERANCE;

    if(rinfo[0] < tol || it >= 50) // check for premature stop
    {
        return 1; // request to interrupt
    }
    return 0;
}

int main()
{
    // CSR symmetric matrix. Only the lower triangle is stored
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<double>         csr_val;

    aoclsparse_int    m, n, nnz;
    aoclsparse_status exit_status = aoclsparse_status_success;

    std::string filename = "Trefethen_20.mtx";
    //symmetry = 1
    n   = 20;
    m   = 20;
    nnz = 158;
    csr_row_ptr.assign(
        {0, 6, 13, 21, 29, 37, 45, 53, 61, 70, 79, 88, 97, 105, 113, 121, 129, 137, 145, 152, 158});
    csr_col_ind.assign(
        {0,  1,  2,  4,  8,  16, 0,  1,  2,  3,  5,  9,  17, 0,  1,  2,  3,  4,  6,  10, 18, 1,  2,
         3,  4,  5,  7,  11, 19, 0,  2,  3,  4,  5,  6,  8,  12, 1,  3,  4,  5,  6,  7,  9,  13, 2,
         4,  5,  6,  7,  8,  10, 14, 3,  5,  6,  7,  8,  9,  11, 15, 0,  4,  6,  7,  8,  9,  10, 12,
         16, 1,  5,  7,  8,  9,  10, 11, 13, 17, 2,  6,  8,  9,  10, 11, 12, 14, 18, 3,  7,  9,  10,
         11, 12, 13, 15, 19, 4,  8,  10, 11, 12, 13, 14, 16, 5,  9,  11, 12, 13, 14, 15, 17, 6,  10,
         12, 13, 14, 15, 16, 18, 7,  11, 13, 14, 15, 16, 17, 19, 0,  8,  12, 14, 15, 16, 17, 18, 1,
         9,  13, 15, 16, 17, 18, 19, 2,  10, 14, 16, 17, 18, 19, 3,  11, 15, 17, 18, 19});
    csr_val.assign(
        {2.00,  1.00,  1.00, 1.00, 1.00,  1.00,  1.00,  3.00,  1.00,  1.00,  1.00,  1.00, 1.00,
         1.00,  1.00,  5.00, 1.00, 1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  7.00,  1.00, 1.00,
         1.00,  1.00,  1.00, 1.00, 1.00,  1.00,  11.00, 1.00,  1.00,  1.00,  1.00,  1.00, 1.00,
         1.00,  13.00, 1.00, 1.00, 1.00,  1.00,  1.00,  1.00,  1.00,  17.00, 1.00,  1.00, 1.00,
         1.00,  1.00,  1.00, 1.00, 19.00, 1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00, 1.00,
         23.00, 1.00,  1.00, 1.00, 1.00,  1.00,  1.00,  1.00,  1.00,  29.00, 1.00,  1.00, 1.00,
         1.00,  1.00,  1.00, 1.00, 1.00,  31.00, 1.00,  1.00,  1.00,  1.00,  1.00,  1.00, 1.00,
         1.00,  37.00, 1.00, 1.00, 1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  41.00, 1.00, 1.00,
         1.00,  1.00,  1.00, 1.00, 1.00,  43.00, 1.00,  1.00,  1.00,  1.00,  1.00,  1.00, 1.00,
         47.00, 1.00,  1.00, 1.00, 1.00,  1.00,  1.00,  1.00,  53.00, 1.00,  1.00,  1.00, 1.00,
         1.00,  1.00,  1.00, 1.00, 59.00, 1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00, 61.00,
         1.00,  1.00,  1.00, 1.00, 1.00,  1.00,  1.00,  67.00, 1.00,  1.00,  1.00,  1.00, 1.00,
         1.00,  71.00});

    // Create aocl sparse matrix
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_create_dcsr(
        A, base, n, n, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data());
    aoclsparse_create_mat_descr(&descr_a);
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_symmetric);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    aoclsparse_set_sv_hint(A, trans, descr_a, 100);
    aoclsparse_optimize(A);

    // Initialize initial point x0 and right hand side b
    double *expected_sol = NULL;
    double *x            = NULL;
    double *b            = NULL;
    double  norm         = 0.0;
    run_mode             = RUN_MODE;
    double rinfo[100];
    double alpha = 1.0, beta = 0.;

    expected_sol = (double *)malloc(sizeof(double) * n);
    if(NULL == expected_sol)
    {
        return -1;
    }

    x = (double *)malloc(sizeof(double) * n);
    if(NULL == x)
    {
        return -1;
    }

    b = (double *)malloc(sizeof(double) * n);
    if(NULL == b)
    {
        return -1;
    }
    double init_x = 1.0, ref_x = 0.5;
    for(int i = 0; i < n; i++)
    {
        expected_sol[i] = ref_x;
        x[i]            = init_x;
    }

    aoclsparse_dmv(trans, &alpha, A, descr_a, expected_sol, &beta, b);
    // create GMRES handle
    double tol = DOUBLE_PRECISION_TOLERANCE;
    bool   hdr;
    //TODO: create buffer copy inside aocl-sparse to store the original csr_val before ILU preocnditioning
    aoclsparse_itsol_handle handle = nullptr;
    aoclsparse_itsol_d_init(&handle);

    // Change options (update to use )
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

    exit_status = aoclsparse_itsol_option_set(handle, "gmres abs tolerance", "1e-6");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres abs tolerance option could not be set, exit status = %d\n",
               exit_status);

    exit_status = aoclsparse_itsol_option_set(handle, "gmres rel tolerance", "1e-12");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres rel tolerance option could not be set, exit status = %d\n",
               exit_status);

    aoclsparse_int rs_iters = (int)20;
    char           rs_iters_string[16];
    sprintf(rs_iters_string, "%d", rs_iters);
    exit_status = aoclsparse_itsol_option_set(handle, "gmres restart iterations", rs_iters_string);
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres restart iterations option could not be set, exit status = %d\n",
               exit_status);

#ifdef PARAM_LOG
    if(run_mode == 0) //detailed prints
    {
        aoclsparse_itsol_handle_prn_options(handle);
    }
#endif

    // Call GMRES solver
    aoclsparse_status status;
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();

    status = aoclsparse_itsol_d_solve(handle, n, A, descr_a, b, x, rinfo, nullptr, monit, &n);
    auto                         t2             = high_resolution_clock::now();
    duration<double, std::milli> cpu_time_gmres = t2 - t1;
    if(status == aoclsparse_status_success || status == aoclsparse_status_user_stop
       || aoclsparse_status_maxit == status)
    {
        norm = calculate_l2Norm_solvers(expected_sol, x, n);

#ifdef PARAM_LOG
        if(run_mode == 1) //batch run mode
        {
            hdr = false;
            if(hdr)
            {
                printf("input,soln,m,n,nnz,expect_tol, restart_iters, resid, "
                       "conv_iters,time_gmres,norm\n");
            }
            //input,m,n,nnz
            printf("%s, ", filename.c_str());
            //printf("8x8_hardcoded, ");
            fflush(stdout);
            //soln
            printf("%s, ", gmres_sol);
            fflush(stdout);
            printf("%d, %d, %d, ", m, n, nnz);
            fflush(stdout);
            //expect_tol, restart_iters, resid, conv_iters,
            printf("%e, %d, %e, %d, ", tol, rs_iters, rinfo[0], (int)rinfo[30]);
            fflush(stdout);
            //time_gmres
            printf("%e, ", cpu_time_gmres);
            fflush(stdout);
            //l2norm
            printf("%e, ", norm);
            fflush(stdout);
        }
        else
        {
            printf("input = %s\n", filename.c_str());
            //printf("8x8_hardcoded\n");
            fflush(stdout);
            printf("Solution = %s\n", gmres_sol);
            fflush(stdout);
            printf("no of rows = %d\n", m);
            fflush(stdout);
            printf("no of cols = %d\n", n);
            fflush(stdout);
            printf("no of nnz = %d\n", nnz);
            fflush(stdout);
            printf("expected tolerance = %e\n", tol);
            fflush(stdout);
            printf("restart iterations = %d\n", rs_iters);
            fflush(stdout);
            printf("residual achieved = %e\n", rinfo[0]);
            fflush(stdout);
            printf("Total iterations = %d\n", (int)rinfo[30]);
            fflush(stdout);
            printf("time GMRES = %e\n", cpu_time_gmres);
            fflush(stdout);
            printf("L2 Norm = %e\n", norm);
            fflush(stdout);
            //printf("exit triggered = %d\n", status);
            //fflush(stdout);
        }
        printf("\n");
#endif
    }
    else
    {
        std::cout << "Something unexpected happened!, Status = " << status << std::endl;
    }

    clear_local_resource(expected_sol);
    clear_local_resource(x);
    clear_local_resource(b);
    aoclsparse_itsol_destroy(&handle);
    aoclsparse_destroy(A);

    return 0;
}
