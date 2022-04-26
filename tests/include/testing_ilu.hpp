/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#ifndef TESTING_ILU_HPP
#define TESTING_ILU_HPP

#include "aoclsparse.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_utility.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_convert.h"

#include <string>
#include <fstream>

using namespace std;

#define MAX_ILU_CONVERGENCE_ITERATIONS 100


aoclsparse_int g_max_iters = 0;

//#undef PARAM_LOG
#define PARAM_LOG

//#define DEBUG_LOG

#ifdef DEBUG_LOG
template <typename T>
void log_vector_info(T * xvRef, aoclsparse_int N)
{
    for (aoclsparse_int i = 0; i < N; i++) 
    {
        printf("[i = %3d] [xRef = %.2f]\n", i, xvRef[i]);
    }
    return;
}
template <typename T>
void Display_in_matrix_form(aoclsparse_int num_rows, aoclsparse_int num_cols, aoclsparse_int num_nonzeros, aoclsparse_int * row_offsets, aoclsparse_int * column_indices, T * val_array)
{       
    printf("Input Matrix (%d x %d, nonzeros = %d):\n", (aoclsparse_int) num_rows, (aoclsparse_int) num_cols, (aoclsparse_int) num_nonzeros);
    for (aoclsparse_int row = 0; row < num_rows; row++)
    {            
        T value_to_print;
        aoclsparse_int col_offset = row_offsets[row];
        for (aoclsparse_int col = 0; col < num_cols; col++)
        {                                       
            if(col_offset < row_offsets[row + 1] && col == column_indices[col_offset])
            {
                value_to_print = val_array[col_offset];                                        
                col_offset++;
            }
            else
            {
                value_to_print = 0.0;
            }
            printf("%4.2f ", value_to_print);              
        }
        printf("\n");
    }
    fflush(stdout);        
}
template <typename T>
void Display_csrformat(aoclsparse_int num_rows, aoclsparse_int num_cols, aoclsparse_int num_nonzeros, aoclsparse_int * row_offsets, aoclsparse_int * column_indices, T * values)
{
    printf("Input Matrix (%d vertices, %d nonzeros):\n", (int) num_rows, (int) num_nonzeros);
    for (int row = 0; row < num_rows; row++)
    {
        printf("%d [@%d, #%d]: ", row, row_offsets[row], row_offsets[row + 1] - row_offsets[row]);
        for (int col_offset = row_offsets[row]; col_offset < row_offsets[row + 1]; col_offset++)
        {
            printf("%d (%f), ", column_indices[col_offset], values[col_offset]);
        }
        printf("\n");
    }
    fflush(stdout);
}
#endif
template <typename T>
void Dump_mtx_File_csc(const string&        market_filename,
                        aoclsparse_int       num_rows,
                        aoclsparse_int       num_cols,
                        aoclsparse_int       nnz,
                        aoclsparse_int       *csc_row_ind,
                        aoclsparse_int       *csc_col_ptr,
                        T                    *csc_val,
                        int                  idx)
{
    int nnz_count = 0;
    T *values_ptr = NULL;
    ofstream fA;
    string test_matrix, DateString;
    time_t t;
    t = time(NULL);
    struct tm tm = *localtime(&t);
    int day=tm.tm_mday;
    int mon = tm.tm_mon+1;
    DateString = "% Generated ";
    if(mon < 10)
    {
        DateString += "0";
    }
    DateString += std::to_string(day);
    DateString += "-";
    switch(mon)
    {
        case 1:
            DateString += "Jan";
            break;
        case 2:
            DateString += "Jan";
            break;
        case 3:
            DateString += "Mar";
            break;
        case 4:
            DateString += "Apr";
            break;
        case 5:
            DateString += "May";
            break;
        case 6:
            DateString += "June";
            break;
        case 7:
            DateString += "July";
            break;
        case 8:
            DateString += "Aug";
            break;
        case 9:
            DateString += "Sep";
            break;
        case 10:
            DateString += "Oct";
            break;
        case 11:
            DateString += "Nov";
            break;
        case 12:
            DateString += "Dec";
            break;
    }
    DateString += "-";
    int yr = tm.tm_year+1900;
    DateString += std::to_string(yr);   
    switch(idx)
    {
        case 1:
            test_matrix = "L_";
            for (int col=0; col < num_cols; col++)
            {
                for (int row_offset = csc_col_ptr[col]; row_offset < csc_col_ptr[col + 1]; row_offset++)
                {
                    int row_idx = csc_row_ind[row_offset];
                    if(row_idx >= col)
                    {
                        nnz_count++;
                    }
                }
            }
            values_ptr = csc_val;
            break;
        case 2:
            test_matrix = "U_";
            for (int col=0; col < num_cols; col++)
            {
                for (int row_offset = csc_col_ptr[col]; row_offset < csc_col_ptr[col + 1]; row_offset++)
                {
                    int row_idx = csc_row_ind[row_offset];
                    if(row_idx <= col)
                    {
                        nnz_count++;
                    }
                }
            }
            values_ptr = csc_val;
            break;
        default:
            printf("Invalid input id for Matrix Dump\n");
            break;
    }
    test_matrix += market_filename;
    #if 0
        printf("input matrix name= %s\n", market_filename.c_str());  
        fflush(stdout);     
        printf("test matrix name= %s\n", test_matrix.c_str());  
        fflush(stdout);        
    #endif     
    fA.open(test_matrix);
    fA << "%%MatrixMarket matrix coordinate real general\n";
    fA << DateString;
    fA << std::endl;
    fA << num_rows;
    fA << " ";
    fA << num_rows;
    fA << " ";
    fA << nnz_count;
    fA << std::endl;

    switch(idx)
    {
        case 1:
            //L only and 1's in the diagonal
            for (int col=0; col < num_cols; col++)
            {
                for (int row_offset = csc_col_ptr[col]; row_offset < csc_col_ptr[col + 1]; row_offset++)
                {
                    double nnz_value;
                    int row_idx = csc_row_ind[row_offset];
                    if(row_idx > col)
                    {
                        nnz_value = values_ptr[row_offset];
                        fA << (int)(row_idx+1);
                        fA << " ";
                        fA << (col+1);
                        fA << " ";
                        fA << std::setprecision(2) << std::fixed;
                        fA << nnz_value;
                        fA << std::endl;
                    }
                    else if(row_idx == col)
                    {
                        nnz_value = 1;
                        fA << (int)(row_idx+1);
                        fA << " ";
                        fA << (col+1);
                        fA << " ";
                        fA << std::setprecision(2) << std::fixed;
                        fA << nnz_value;
                        fA << std::endl;
                    }
                }
            }
            break;
        case 2:
            //U only
            for (int col=0; col < num_cols; col++)
            {
                for (int row_offset = csc_col_ptr[col]; row_offset < csc_col_ptr[col + 1]; row_offset++)
                {
                    double nnz_value;
                    int row_idx = csc_row_ind[row_offset];
                    if(row_idx <= col)
                    {
                        nnz_value = values_ptr[row_offset];
                        fA << (int)(row_idx+1);
                        fA << " ";
                        fA << (col+1);
                        fA << " ";
                        fA << std::setprecision(2) << std::fixed;
                        fA << nnz_value;
                        fA << std::endl;
                    }
                }
            }
            break;
        default:
            printf("Invalid input id for Matrix Dump\n");
            break;
    }
    fA.close();
}

template <typename T>
double calculate_l2Norm(const T* xvRef, const T* xvTest, aoclsparse_int n)
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
double abs_relative_approc_error(T* xvNew, T* xvOld, aoclsparse_int n)
{  
    double max_arae_percent = 0;
    for(aoclsparse_int i =0; i < n; i++)
    {
        double temp = xvNew[i] - xvOld[i];
        temp = temp/xvNew[i];

        temp = temp * 100;
        max_arae_percent = (std::max)( max_arae_percent, temp);
    }
    return max_arae_percent;
}
template <typename T>
void ilu_factorization_solution(aoclsparse_operation      	trans,
                                aoclsparse_matrix          A, 
                                aoclsparse_int              N,
                                const aoclsparse_mat_descr 	descr,
                                const T*               diag,
                                const T*               approx_inv_diag,                                
                                T*                     x_old, 
                                T*                     x,
                                const T*               b, 
                                aoclsparse_int &            iter, 
                                double &                    min_arae_percent,
                                double &                cpu_time_start,
                                double &                cpu_time_fact)
{
	double temp_arae_percent=0.0; 

    cpu_time_start = aoclsparse_clock();
    /*
        First call to ilu_smoother API calculates L and U factor
        and does a initial LU solve
    */
    CHECK_AOCLSPARSE_ERROR(aoclsparse_ilu_smoother(trans,
                                                    A,
                                                    descr,
                                                    diag,
                                                    approx_inv_diag,
                                                    x,
                                                    b));  
    cpu_time_fact = aoclsparse_clock_min_diff(cpu_time_fact , cpu_time_start );  

    min_arae_percent = DBL_MAX;

    temp_arae_percent = abs_relative_approc_error(x, x_old, N);

    min_arae_percent = (std::min)( min_arae_percent, temp_arae_percent);
  

    iter = 0;    
    while(iter < g_max_iters)
    {          
        aoclsparse_copy_vector(x, x_old, N);  
		
        CHECK_AOCLSPARSE_ERROR(aoclsparse_ilu_smoother(trans,
                                                        A,
                                                        descr,
                                                        diag,
                                                        approx_inv_diag,
                                                        x,
                                                        b));        

        temp_arae_percent = abs_relative_approc_error(x, x_old, N);

        min_arae_percent = (std::min)( min_arae_percent, temp_arae_percent);        
         
        iter++;
    } 
}

template <typename T>
inline void ref_csrilu0(aoclsparse_int                     M,
                         const std::vector<aoclsparse_int>& csr_row_ptr,
                         const std::vector<aoclsparse_int>& csr_col_ind,
                         std::vector<T>&                    csr_val,
                         aoclsparse_int*                    struct_pivot,
                         aoclsparse_int*                    numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = -1;
    *numeric_pivot = -1;
    aoclsparse_int base = 0;

    // pointer of upper part of each row
    std::vector<aoclsparse_int> diag_offset(M);
    std::vector<aoclsparse_int> nnz_entries(M, 0);

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
void testing_ilu(const Arguments& arg)
{
    aoclsparse_int         M         = arg.M;
    aoclsparse_int         N         = arg.N;
    aoclsparse_int         nnz       = arg.nnz;
    aoclsparse_matrix_init mat       = arg.matrix;
    aoclsparse_operation   trans     = arg.transA;
    aoclsparse_index_base  base      = arg.baseA;
    bool issymm;
    std::string           filename = arg.filename;
    aoclsparse_int iter;
    double min_arae_percent;
    aoclsparse_int run_mode=1;
    T *diag=NULL, *approx_inv_diag=NULL;
    T alpha = static_cast<T>(arg.alpha);
    T beta  = static_cast<T>(arg.beta);

    // Create matrix descriptor
    aoclsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));

    // Allocate memory for matrix
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<T>             csr_val;
    aoclsparse_seedrand();
#if 0
    // Print aoclsparse version
    std::cout << aoclsparse_get_version() << std::endl;
#endif

    // Sample matrix
    aoclsparse_init_csr_matrix(csr_row_ptr,
	    csr_col_ind,
	    csr_val,
	    M,
	    N,
	    nnz,
	    base,
	    mat,
	    filename.c_str(),
	    issymm,
	    true);


    // Manu -- new
	aoclsparse_matrix A; // = new _aoclsparse_matrix;
    CHECK_AOCLSPARSE_ERROR(aoclsparse_create_dcsr(A, base, M, N, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()));


    //Display_csrformat<T>(M, N, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data());

#if 0//def DEBUG_LOG
        printf("symmetry: %d\n", issymm);
        //printf("A before creating sparse structure:\n");
        //Display_csrformat<T>(M, N, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data());
        printf("A from sparse structure:\n");
        Display_in_matrix_form(A);
#endif  
  
    // Allocate memory for vectors
    //std::vector<T> x_ref(N);
    //std::vector<T> b(M);
    //std::vector<T> x_old(N);
    //std::vector<T> x(N);

    T *x_ref=NULL;
    T *x_old=NULL;
    T *x=NULL;
    T *b=NULL;

    x_ref = (T *) malloc(sizeof(T)* N);
    if(NULL == x_ref)
    {
        return;
    }

    x_old = (T *) malloc(sizeof(T)* N);
    if(NULL == x_old)
    {
        return;
    }   

    x = (T *) malloc(sizeof(T)* N);
    if(NULL == x)
    {
        return;
    }  

    b = (T *) malloc(sizeof(T)* M);
    if(NULL == b)
    {
        return;
    }             

    // Initialize data
    double val=1.0;
    //aoclsparse_init<T>(x_ref, 1, N, 1);
    aoclsparse_random_vector(x_ref, N);
    aoclsparse_set_vector<T>(x_old, val, N);
    aoclsparse_copy_vector<T>(x_old, x, N);

    aoclsparse_int h_analysis_pivot_gold; //unused for now
    aoclsparse_int h_solve_pivot_gold;  //unused for now
    std::vector<T>              csr_val_gold(nnz);
    std::vector<T>             ref_copy_csr_val(nnz);
    //save the csr values for in-place operation of ilu0
    aoclsparse_copy_vector<T>(csr_val.data(), ref_copy_csr_val.data(), nnz);

    if(arg.unit_check)
    {    
        aoclsparse_copy_vector<T>(csr_val.data(), csr_val_gold.data(), nnz);                  
		//compute ILU0 using reference c code and store the LU factors as part of csr_val_gold
        ref_csrilu0<T>(M, csr_row_ptr, csr_col_ind, csr_val_gold, &h_analysis_pivot_gold, &h_solve_pivot_gold);
    }    
	// Reference SPMV CSR implementation
	for(int i = 0; i < M; i++)
	{
	    T result = 0.0;
	    for(int j = csr_row_ptr[i] - base; j < csr_row_ptr[i+1] - base; j++)
	    {
		result += alpha * csr_val[j] * x_ref[csr_col_ind[j] - base];
	    }
	    b[i] = (beta * b[i]) + result;

	}

    g_max_iters  = MAX_ILU_CONVERGENCE_ITERATIONS;
    int number_hot_calls = arg.iters;

    double cpu_time_analysis = DBL_MAX;
    double cpu_time_start = aoclsparse_clock();
    //Basic routine type checks
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_lu_smoother_hint(A, trans, descr, 0));

    // Optimize the matrix, "A"
    CHECK_AOCLSPARSE_ERROR(aoclsparse_optimize(A));
	
	cpu_time_analysis = aoclsparse_clock_min_diff(cpu_time_analysis , cpu_time_start );

    

    double cpu_time_fact = DBL_MAX;    
    // Warm up and functionality run
    ilu_factorization_solution<T>(trans, A, N, descr, diag, approx_inv_diag, x_old,x, b, iter, min_arae_percent, cpu_time_start, cpu_time_fact);
    

    if(arg.unit_check)
    {    
        // Check solution vector if no pivot has been found
        if(h_analysis_pivot_gold == -1 && h_solve_pivot_gold == -1)
        {
            near_check_general<T>(1, nnz, 1, csr_val_gold.data(), csr_val.data());
#ifdef DEBUG_LOG        
        for(int i = 0; i < nnz; i++)
        {
            printf("[%d] csr_val[%d] = %f, csr_val_gold[%d] = %f\n", i, i, csr_val[i], i, csr_val_gold[i]);
            fflush(stdout);
        }        
#endif              
        }
    }
    double norm = calculate_l2Norm<T>(x_ref, x, N);
	double cpu_time_solution = DBL_MAX;

    // Performance run
    for(int iter = 0; iter < number_hot_calls; ++iter)
    {
        cpu_time_start = aoclsparse_clock();
        CHECK_AOCLSPARSE_ERROR(aoclsparse_ilu_smoother(trans,
                                                        A,
                                                        descr,
                                                        diag,
                                                        approx_inv_diag,
                                                        x,
                                                        b));
        cpu_time_solution = aoclsparse_clock_min_diff(cpu_time_solution , cpu_time_start );
    }

	double cpu_gbyte = csrilu0_gbyte_count<T>(M, nnz) / cpu_time_solution * 1e6;

#ifdef PARAM_LOG
    if(run_mode == 1)   //batch run mode
    {
        //input,m,nnz
        printf("%s, ", filename.c_str());
        fflush(stdout);
        printf("%d, %d, ", M, nnz);               
        fflush(stdout);
        //conv_iters,arae_%,l2norm
        printf("%d, %.10lf, %.10lf, ", iter, min_arae_percent, norm);
        fflush(stdout);
        //bytes
        printf("%.10lf, ", cpu_gbyte);
        fflush(stdout);
        //analysis_t
        printf("%.10lf, ", cpu_time_analysis);
        fflush(stdout);
        //fact_t
        printf("%.10lf, ", cpu_time_fact);
        fflush(stdout);        
        //sol_t                     
        printf("%.10lf, ", cpu_time_solution);                              
        fflush(stdout);
        //is_verified
        printf("%s, ", (arg.unit_check ? "yes" : "no"));                              
        fflush(stdout);
        printf("\n");
        fflush(stdout);
    }        
    else
    {
        printf("%s, ", filename.c_str());
        fflush(stdout);
        printf("no of rows = %d\n", M);
        fflush(stdout);
        printf("no of nnz = %d\n", nnz);        
        fflush(stdout);
        printf("Total iterations to converge = %d\n", iter);
        fflush(stdout);
        printf("Minimum Absolute Relative Aprrox Error = %.10lf\n", min_arae_percent);  
        fflush(stdout);
        printf("L2 Norm = %.10lf\n", norm);                 
        fflush(stdout);
        printf("GB/s = %.10lf\n", cpu_gbyte);
        fflush(stdout);
        printf("Analysis msec = %.10lf\n", cpu_time_analysis);
        fflush(stdout);
        printf("Fact msec = %.10lf\n", cpu_time_fact);
        fflush(stdout);        
        printf("Sol msec = %.10lf\n", cpu_time_solution);
        fflush(stdout);
        printf("is verified = %s\n", (arg.unit_check ? "yes" : "no"));  
        fflush(stdout);        
    }   
#endif


    //dump L and U factors in MTX CSC format for validation against octave outputs
    #if 0
        aoclsparse_int       *csc_row_ind=NULL;
        aoclsparse_int       *csc_col_ptr=NULL;
        T    *csc_val=NULL; 

        csc_val = (T *) malloc(sizeof(T)* nnz);
        if(NULL == csc_val)
        {
            return;
        }
        csc_col_ptr = (aoclsparse_int *) malloc(sizeof(aoclsparse_int)* (N+1));
        if(NULL == csc_col_ptr)
        {
            return;
        }
        csc_row_ind = (aoclsparse_int *) malloc(sizeof(aoclsparse_int)* (nnz));
        if(NULL == csc_row_ind)
        {
            return;
        }             

        for(aoclsparse_int ni = 0; ni < nnz; ni++)
        {
            double cval=0.0;
            csc_val[ni] = (T) cval;
            csc_row_ind[ni] = 0;
        }
        for(aoclsparse_int ni = 0; ni < (N+1); ni++)
        {
            csc_col_ptr[ni] = 0;
        }        

        CHECK_AOCLSPARSE_ERROR(aoclsparse_csr2csc(M, N, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(),
            csc_row_ind, csc_col_ptr, csc_val));

        Dump_mtx_File_csc<T>(filename, M, N, nnz, csc_row_ind, csc_col_ptr, csc_val, 1);                //L
        Dump_mtx_File_csc<T>(filename, M, N, nnz, csc_row_ind, csc_col_ptr, csc_val, 2);                //U

        if (csc_row_ind != NULL)
        {
            free(csc_row_ind);
            csc_row_ind = NULL;
        }
        if (csc_col_ptr != NULL)
        {
            free(csc_col_ptr);
            csc_col_ptr = NULL;            
        }
        if (csc_val != NULL)         
        {
            free(csc_val);
            csc_val = NULL;            
        }        
    #endif    


cleanup:
    if(x_ref != NULL)
    {
        free(x_ref);
        x_ref = NULL;
    }
    if(x_old != NULL)
    {
        free(x_old);
        x_old = NULL;
    }
    if(x != NULL)
    {
        free(x);
        x = NULL;
    }
    if(b != NULL)
    {
        free(b);
        b = NULL;
    } 
    aoclsparse_destroy(A);
}

#endif // TESTING_ILU_HPP
