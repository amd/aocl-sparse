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
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>

#define PREMATURE_STOP_TOLERANCE 1e-2
const char gmres_sol[] = "gmres_rci_s";

template <typename T>
void clear_local_resource(T *x)
{
    if(x != NULL)
    {
        free(x);
        x = NULL;
    }
}
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

float calculate_l2Norm_solvers(const float *xSol, const float *x, aoclsparse_int n)
{
    float norm = 0.0f;
    //residual = xSol - x
    //xSol obtained by initial spmv
    //x obtained by ILU-GMRES iterations
    for(aoclsparse_int i = 0; i < n; i++)
    {
        float a = xSol[i] - x[i];
        norm += a * a;
    }
    norm = sqrt(norm);
    return norm;
}

int main()
{
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<float>          csr_val;

    aoclsparse_int    m, n, nnz;
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

    // create matrix descriptor
    aoclsparse_matrix     mat;
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_create_scsr(
        A, base, n, n, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data());
    aoclsparse_create_mat_descr(&descr_a);
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_symmetric);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    aoclsparse_set_sv_hint(A, trans, descr_a, 100);
    aoclsparse_set_lu_smoother_hint(A, trans, descr_a, 100);
    aoclsparse_optimize(A);

    // Initialize initial point x0 and right hand side b
    float         *expected_sol = NULL;
    float         *x            = NULL;
    float         *b            = NULL;
    float          alpha = 1.0, beta = 0.;
    float          norm     = 0.0;
    aoclsparse_int rs_iters = (int)20;
    char           rs_iters_string[16];

    expected_sol = (float *)malloc(sizeof(float) * n);
    if(NULL == expected_sol)
    {
        return -1;
    }

    x = (float *)malloc(sizeof(float) * n);
    if(NULL == x)
    {
        return -1;
    }

    b = (float *)malloc(sizeof(float) * n);
    if(NULL == b)
    {
        return -1;
    }
    float init_x = 1.0, ref_x = 0.5;
    for(int i = 0; i < n; i++)
    {
        expected_sol[i] = ref_x;
        x[i]            = init_x;
    }

    aoclsparse_smv(trans, &alpha, A, descr_a, expected_sol, &beta, b);

    aoclsparse_itsol_handle handle = nullptr;
    // create GMRES handle
    aoclsparse_itsol_s_init(&handle);

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
    aoclsparse_itsol_s_rci_input(handle, n, b);

    // Call GMRES solver
    aoclsparse_itsol_rci_job    ircomm = aoclsparse_rci_start;
    aoclsparse_status           status;
    float                      *io1 = nullptr;
    float                      *io2 = nullptr;
    float                       rinfo[100];
    float                       tol               = PREMATURE_STOP_TOLERANCE;
    bool                        precond_done_flag = false;
    std::vector<aoclsparse_int> diag_offset(n);
    std::vector<aoclsparse_int> nnz_entries(n, 0);
    std::vector<float>          precond_csr_val;
    precond_csr_val = csr_val;
    float *pcsr_val = precond_csr_val.data();

    while(ircomm != aoclsparse_rci_stop)
    {
        status = aoclsparse_itsol_s_rci_solve(handle, &ircomm, &io1, &io2, x, rinfo);
        if(status != aoclsparse_status_success)
            break;
        switch(ircomm)
        {
        case aoclsparse_rci_mv:
            //User can instead use their custom implementation of SpMV routine

            // Compute v = Au
            beta  = 0.0;
            alpha = 1.0;
            aoclsparse_smv(trans, &alpha, A, descr_a, io1, &beta, io2);
            break;

        case aoclsparse_rci_precond:
            //User can instead use their custom implementation of ILU Preconditioner

            //Run ILU Preconditioner only once in the beginning
            //Run Triangular Solve using ILU0 factorization
            aoclsparse_silu_smoother(trans,
                                     A,
                                     descr_a,
                                     &pcsr_val,
                                     NULL,
                                     io1, //x = ?, io1 = z+j*n,
                                     io2); //rhs, io2 = v+j*n
            break;

        case aoclsparse_rci_stopping_criterion:
            if(rinfo[0] < tol)
            {
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
        norm = calculate_l2Norm_solvers(expected_sol, x, n);
        printf("input = %s\n", filename.c_str());
        printf("solver = %s\n", gmres_sol);
        printf("no of rows = %d\n", m);
        printf("no of cols = %d\n", n);
        printf("no of nnz = %d\n", nnz);
        printf("monitoring tolerance = %e\n", tol);
        printf("restart iterations = %d\n", rs_iters);
        printf("residual achieved = %e\n", rinfo[0]);
        printf("total iterations = %d\n", (int)rinfo[30]);
        printf("l2 Norm = %e\n", norm);
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
    printf("\n");
    fflush(stdout);

    return 0;
}
