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
#include <cfloat>
#include <chrono>

#define PARAM_LOG
#define DOUBLE_PRECISION_TOLERANCE 1e-6
#define RUN_MODE 0
int run_mode;
const char gmres_sol[] = "gmres_rci_d";

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
void printer(double rinfo[100], bool header)
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

double calculate_l2Norm_solvers(const double* xvRef, const double* xvTest, aoclsparse_int n)
{
    double norm = 0.0f;
    //residual = xvRef - xvTest
    //xvRef obtained by initial spmv
    //xvTest obtained by ILU iterations
    for (aoclsparse_int i = 0; i < n; i++) 
    {
        double a = xvRef[i] - xvTest[i];
        norm += a*a;
    }
    norm = sqrt(norm);
    return norm;
}
template <typename T>
aoclsparse_int spmv(aoclsparse_int nrows, T* pl_values, aoclsparse_int *pl_col_idx, 
                    aoclsparse_int *pl_row_offsets, T *xv, T *bv)
{
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (aoclsparse_int i = 0; i < nrows; i++) 
    {             
        T sum = 0.0;                               
        //column loop, right            
        for(aoclsparse_int j = pl_row_offsets[i] ; j < pl_row_offsets[i+1] ; j++)
        {                
            aoclsparse_int curCol = pl_col_idx[j];
            sum += pl_values[j] * xv[curCol];                 
        }
        bv[i] = sum;          
    }     
  
    return 0; 
}
template <typename T>
inline void ref_csrilu0(aoclsparse_int                     M,
                         const std::vector<aoclsparse_int>& csr_row_ptr,
                         const std::vector<aoclsparse_int>& csr_col_ind,
                         std::vector<T>&                    csr_val,
                         std::vector<aoclsparse_int>&       diag_offset,
                         std::vector<aoclsparse_int>&       nnz_entries,
                         aoclsparse_int*                    struct_pivot,
                         aoclsparse_int*                    numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = -1;
    *numeric_pivot = -1;
    aoclsparse_int base = 0;

    // pointer of upper part of each row
    //std::vector<aoclsparse_int> diag_offset(M);
    //std::vector<aoclsparse_int> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(aoclsparse_int ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        aoclsparse_int row_begin = csr_row_ptr[ai] - base;
        aoclsparse_int row_end   = csr_row_ptr[ai + 1] - base;
        aoclsparse_int j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = j;
        }

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if(csr_col_ind[j] - base < ai)
            {

                aoclsparse_int col_j  = csr_col_ind[j] - base;
                aoclsparse_int diag_j = diag_offset[col_j];

                if(csr_val[diag_j] != static_cast<T>(0))
                {
                    // multiplication factor
                    csr_val[j] = csr_val[j] / csr_val[diag_j];

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(aoclsparse_int k = diag_j + 1; k < csr_row_ptr[col_j + 1] - base; ++k)
                    {
                        // if nnz at this position do linear combination
                        if(nnz_entries[csr_col_ind[k] - base] != 0)
                        {
                            aoclsparse_int idx = nnz_entries[csr_col_ind[k] - base];
                            csr_val[idx]      = std::fma(-csr_val[j], csr_val[k], csr_val[idx]);
                        }
                    }
                }
                else
                {
                    // Numerical zero diagonal
                    *numeric_pivot = col_j + base;
                    return;
                }
            }
            else if(csr_col_ind[j] - base == ai)
            {
                has_diag = true;
                break;
            }
            else
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            *struct_pivot  = ai + base;
            *numeric_pivot = ai + base;
            return;
        }

        // set diagonal pointer to diagonal element
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = 0;
        }
    }
}

template <typename T>
aoclsparse_status ilu_solve(aoclsparse_int                       m,
                            aoclsparse_int                       n,	
                            aoclsparse_int* __restrict__         lu_diag_ptr,						   
                            T* __restrict__            	         csr_val,								   
                            const aoclsparse_int* __restrict__   row_offsets,
                            const aoclsparse_int* __restrict__   column_indices,
                            T* __restrict__                      xv,
                            const T* __restrict__                bv)
{
	aoclsparse_status ret = aoclsparse_status_success;
    aoclsparse_int i, k;    

   //Forward Solve
   //Solve L . y = b
   for(i = 0; i < m; i++)
   {
       T sum = bv[i];
       for(k = row_offsets[i]; k < lu_diag_ptr[i]; k++)
       {
           aoclsparse_int col_idx = column_indices[k];
           T temp = 0.0;
           temp = csr_val[k] * xv[col_idx];           
           sum = sum - temp;
       }
       xv[i] = sum;
   } 

   //Backward Solve
   // Solve: U . x = y
   for(i = m - 1; i >= 0; i--)
   {              
       aoclsparse_int diag_idx = lu_diag_ptr[i];
       T diag_elem;
       for(k = lu_diag_ptr[i] + 1; k < row_offsets[i+1]; k++)
       {
           aoclsparse_int col_idx = column_indices[k];
           T temp = 0.0;
           temp = csr_val[k] * xv[col_idx];                       
           xv[i] = xv[i] - temp;
       }    
       diag_elem = csr_val[diag_idx];     
       if(diag_elem != 0.0)
       {
            xv[i] = xv[i]/diag_elem;  
       }   
   }     
	return ret;
}
template <typename T>
void ilu_preconditioner(aoclsparse_int                       M,
                        aoclsparse_int                       N,
                        const std::vector<aoclsparse_int>&   csr_row_ptr,
                        const std::vector<aoclsparse_int>&   csr_col_ind,
                        std::vector<T>&                      csr_val,
                        std::vector<aoclsparse_int>&         diag_offset,
                        std::vector<aoclsparse_int>&         nnz_entries,                        
                        bool                                 &precond_done_flag,
                        T                                    *x,        //unknown in Ax = b
                        T                                    *b)        //rhs(known) in Ax = b
{
    aoclsparse_int h_analysis_pivot_gold; //unused for now
    aoclsparse_int h_solve_pivot_gold;  //unused for now
    // pointer of upper part of each row
  
    if(precond_done_flag == false)
    {
        ref_csrilu0<T>(M, 
                       csr_row_ptr, 
                       csr_col_ind, 
                       csr_val, 
                       diag_offset, 
                       nnz_entries, 
                       &h_analysis_pivot_gold, 
                       &h_solve_pivot_gold);
        precond_done_flag = true;           
    }    
   
    ilu_solve(M, 
              N,
              diag_offset.data(),
              csr_val.data(),
              csr_row_ptr.data(),
              csr_col_ind.data(),
              x,
              b);
}
int main()
{
    // CSR symmetric matrix. Only the lower triangle is stored  
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<double>             csr_val;

    aoclsparse_int m, n, nnz;
    aoclsparse_status        exit_status = aoclsparse_status_success;

    std::string filename="Trefethen_20.mtx";
    //symmetry = 1
    n = 20;
    m = 20;
    nnz = 158;
    csr_row_ptr.assign({0, 6, 13, 21, 29, 37, 45, 53, 61, 70, 79, 88, 97, 105, 113, 121, 129, 137, 145, 152, 158});
    csr_col_ind.assign({0, 1, 2, 4, 8, 16, 0, 1, 2, 3, 5, 9, 17, 0, 1, 2, 3, 4, 6, 10, 18, 1, 2, 3, 4, 5, 7, 11, 19, 0, 2, 3, 4, 5, 6, 8, 12, 1, 3, 4, 5, 6, 7, 9, 13, 2, 4, 5, 6, 7, 8, 10, 14, 3, 5, 6, 7, 8, 9, 11, 15, 0, 4, 6, 7, 8, 9, 10, 12, 16, 1, 5, 7, 8, 9, 10, 11, 13, 17, 2, 6, 8, 9, 10, 11, 12, 14, 18, 3, 7, 9, 10, 11, 12, 13, 15, 19, 4, 8, 10, 11, 12, 13, 14, 16, 5, 9, 11, 12, 13, 14, 15, 17, 6, 10, 12, 13, 14, 15, 16, 18, 7, 11, 13, 14, 15, 16, 17, 19, 0, 8, 12, 14, 15, 16, 17, 18, 1, 9, 13, 15, 16, 17, 18, 19, 2, 10, 14, 16, 17, 18, 19, 3, 11, 15, 17, 18, 19});
    csr_val.assign({2.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 3.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 5.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 7.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 11.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 13.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 17.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 19.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 23.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 29.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 31.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 37.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 41.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 43.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 47.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 53.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 59.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 61.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 67.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 71.00});

    // create matrix descriptor
    aoclsparse_matrix     mat;
    aoclsparse_matrix     A;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_mat_descr  descr_a;
    aoclsparse_operation  trans = aoclsparse_operation_none;
    aoclsparse_create_dcsr(A, base, n, n, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data());
    aoclsparse_create_mat_descr(&descr_a);
    aoclsparse_set_mat_type(descr_a, aoclsparse_matrix_type_symmetric);
    aoclsparse_set_mat_fill_mode(descr_a, aoclsparse_fill_mode_lower);
    aoclsparse_set_sv_hint(A, trans, descr_a, 100);
    aoclsparse_set_lu_smoother_hint(A, trans, descr_a, 100);
    aoclsparse_optimize(A);

    // Initialize initial point x0 and right hand side b
    double *expected_sol=NULL;
    double *x=NULL;
    double *b=NULL; 

    double               alpha = 1.0, beta = 0.;
    double               y[n];
    double norm=0.0;
    run_mode = RUN_MODE;

    expected_sol = (double *) malloc(sizeof(double)* n);
    if(NULL == expected_sol)
    {
        return -1;
    }  

    x = (double *) malloc(sizeof(double)* n);
    if(NULL == x)
    {
        return -1;
    }  

    b = (double *) malloc(sizeof(double)* n);
    if(NULL == b)
    {
        return -1;
    }
    double init_x=1.0, ref_x = 0.5;
    for (int i = 0; i < n; i ++)
    {                  
        expected_sol[i] =ref_x;
        x[i] = init_x;
    }     

    aoclsparse_dmv(trans, &alpha, A, descr_a, expected_sol, &beta, b);

    // create CG handle
    aoclsparse_itsol_handle handle = nullptr;
    aoclsparse_itsol_d_init(&handle);

    // Change options (update to use )
    exit_status = aoclsparse_itsol_option_set(handle, "iterative method", "GMRES");
    if(exit_status!= aoclsparse_status_success)
        printf("Warning an iterative method option could not be set, exit status = %d\n", exit_status);    
    
    exit_status = aoclsparse_itsol_option_set(handle, "gmres preconditioner", "ILU0");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres preconditioner option could not be set, exit status = %d\n", exit_status);        
       
    exit_status = aoclsparse_itsol_option_set(handle, "gmres iteration limit", "50");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres iteration limit option could not be set, exit status = %d\n", exit_status);    
      
    exit_status = aoclsparse_itsol_option_set(handle, "gmres abs tolerance", "1e-6");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres abs tolerance option could not be set, exit status = %d\n", exit_status);         
  
    exit_status = aoclsparse_itsol_option_set(handle, "gmres rel tolerance", "1e-12");
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres rel tolerance option could not be set, exit status = %d\n", exit_status);         
      
    aoclsparse_int rs_iters;
    char rs_iters_string[16];    
    rs_iters = (int) 20;
    sprintf(rs_iters_string, "%d", rs_iters);
    exit_status = aoclsparse_itsol_option_set(handle, "gmres restart iterations", rs_iters_string);
    if(exit_status != aoclsparse_status_success)
        printf("Warning gmres restart iterations option could not be set, exit status = %d\n", exit_status);         

    #ifdef PARAM_LOG
    if(run_mode == 0)   //detailed prints
    {    
        aoclsparse_itsol_handle_prn_options(handle);
    }
    #endif    

    // initialize size and rhs inside the handle
    aoclsparse_itsol_d_rci_input(handle, n, b);

    // Call CG solver
    aoclsparse_itsol_rci_job ircomm = aoclsparse_rci_start;
    aoclsparse_status        status;
    double*                  io1 = nullptr;
    double*                  io2 = nullptr;
    double                   rinfo[100];
    double                   tol = DOUBLE_PRECISION_TOLERANCE;
    bool                     hdr;
    bool                     precond_done_flag = false;
    std::vector<aoclsparse_int> diag_offset(n);
    std::vector<aoclsparse_int> nnz_entries(n, 0);    
    std::vector<double>             precond_csr_val;
    precond_csr_val = csr_val;          

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();
    while(ircomm != aoclsparse_rci_stop)
    {
        status = aoclsparse_itsol_d_rci_solve(handle, &ircomm, &io1, &io2, x, rinfo);
        if(status != aoclsparse_status_success)
            break;
        switch(ircomm)
        {
        case aoclsparse_rci_mv:
            // Compute v = Au
            //beta  = 0.0;
            //alpha = 1.0;
            //aoclsparse_dmv(trans, &alpha, A, descr_a, io1, &beta, io2);
            spmv(n, csr_val.data(), csr_col_ind.data(), csr_row_ptr.data(), io1, io2);
            break;

        case aoclsparse_rci_precond:        
            //for(int i = 0; i < n; i++)
            //    io1[i] = io2[i];     
            //Run ILU Preconditioner only once in the beginning
            //Run Triangular Solve using ILU0 factorization 
            ilu_preconditioner(m,
                               n, 
                               csr_row_ptr, 
                               csr_col_ind, 
                               precond_csr_val,
                               diag_offset,
                               nnz_entries,
                               precond_done_flag,
                               io1,            //x = ?, io1 = z+j*n, 
                               io2);            //rhs, io2 = v+j*n                                    
            break;

        case aoclsparse_rci_stopping_criterion:
            hdr = ((int)rinfo[30] % 10) == 0;
            //printer(rinfo, hdr);
            if(rinfo[0] < tol || rinfo[30] >= 50)
            {
                //std::cout << "User stop. Final residual: " << rinfo[0] << std::endl;
                ircomm = aoclsparse_rci_interrupt;
            }
            break;

        default:
            break;
        }
    }   
    auto t2 = high_resolution_clock::now();         
    duration<double, std::milli> cpu_time_gmres = t2 - t1;
    if(aoclsparse_status_success == status || aoclsparse_status_user_stop == status 
        || aoclsparse_status_maxit == status)
    {
        norm = calculate_l2Norm_solvers(expected_sol, x, n);
        #ifdef PARAM_LOG
            if(run_mode == 1)   //batch run mode
            {
                hdr = false;
                if(hdr)
                {
                    printf("input,soln,m,n,nnz,expect_tol, restart_iters, resid, conv_iters,time_gmres,norm\n");
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
