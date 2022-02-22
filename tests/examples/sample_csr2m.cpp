/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstring>

/*Computes multiplication of 2 sparse matrices A and B
 * A is a M x K sparse matrix with nnz_A number of non-zeroes
 * B is a K x N sparse matrix with nnz_B number of non-zeroes
 * */
#define M 3
#define N 3
#define K 3
#define NNZ_A 6
#define NNZ_B 4

//Comment this out for single stage computation
#define TWO_STAGE_COMPUTATION

#define PRINT_OUTPUT

void aoclsparse_create_csr(aoclsparse_mat_csr &csrA,
	aoclsparse_mat_csr &csrB,
	aoclsparse_mat_descr *descrA,
	aoclsparse_mat_descr *descrB)
{
    // Create matrix descriptor of input A
    // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
    // and aoclsparse_index_base to aoclsparse_index_base_zero.
    aoclsparse_create_mat_descr(descrA);

    // Create matrix descriptor of input B
    // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
    // and aoclsparse_index_base to aoclsparse_index_base_zero.
    aoclsparse_create_mat_descr(descrB);

    aoclsparse_index_base base = aoclsparse_index_base_zero;
    // Initialise matrix A
    // 	1  0  2
    // 	0  0  3
    // 	4  5  6
    aoclsparse_int row_ptr_A[] = {0, 2, 3, 6};
    aoclsparse_int col_ind_A[] = {0, 2, 2, 0, 1, 2};
    float             val_A[] = {1 , 2 , 3, 4, 5, 6};;
    aoclsparse_int *csr_row_ptr_A = (aoclsparse_int *)malloc( (M + 1) * sizeof(aoclsparse_int) );
    aoclsparse_int *csr_col_ind_A = (aoclsparse_int *)malloc( NNZ_A * sizeof(aoclsparse_int) );
    float             *csr_val_A = (float *)malloc( (NNZ_A) * sizeof(float) );
    memcpy( csr_row_ptr_A, row_ptr_A, (M + 1) * sizeof(aoclsparse_int));
    memcpy( csr_col_ind_A, col_ind_A,  NNZ_A  * sizeof(aoclsparse_int));
    memcpy( csr_val_A, val_A,  NNZ_A  * sizeof(float));
    aoclsparse_create_mat_csr(csrA, base, M, K, NNZ_A, csr_row_ptr_A, csr_col_ind_A, csr_val_A);

    // Initialise matrix B
    // 	1  2  0
    // 	0  0  3
    // 	0  4  0
    aoclsparse_int row_ptr_B[K + 1] = {0, 2, 3, 4};
    aoclsparse_int col_ind_B[NNZ_B]= {0, 1, 2, 1};
    float         val_B[NNZ_B] = {1 , 2 , 3, 4};
    aoclsparse_int *csr_row_ptr_B = (aoclsparse_int *)malloc( (K + 1) * sizeof(aoclsparse_int) );
    aoclsparse_int *csr_col_ind_B = (aoclsparse_int *)malloc( NNZ_B * sizeof(aoclsparse_int) );
    float             *csr_val_B = (float *)malloc( (NNZ_B) * sizeof(float) );
    memcpy( csr_row_ptr_B, row_ptr_B, (K + 1) * sizeof(aoclsparse_int));
    memcpy( csr_col_ind_B, col_ind_B,  NNZ_B  * sizeof(aoclsparse_int));
    memcpy( csr_val_B, val_B,  NNZ_B  * sizeof(float));
    aoclsparse_create_mat_csr(csrB, base, K, N, NNZ_B, csr_row_ptr_B, csr_col_ind_B, csr_val_B);
    return ;
}

int main(int argc, char* argv[])
{
    aoclsparse_status status;
    aoclsparse_int         nnz_C      ;
    aoclsparse_request request;
    aoclsparse_index_base base ;
    aoclsparse_operation transA = aoclsparse_operation_none;
    aoclsparse_operation transB = aoclsparse_operation_none;

    // Print aoclsparse version
    aoclsparse_int ver;
    aoclsparse_get_version(&ver);
    std::cout << "aocl-sparse version: " << ver / 100000 << "." << ver / 100 % 1000 << "."
	<< ver % 100 << std::endl;


    // Initialise matrix descriptor and csr matrix structure of inputs A and B
    aoclsparse_mat_descr descrA;
    aoclsparse_mat_descr descrB;
    aoclsparse_mat_csr csrA ;
    aoclsparse_mat_csr csrB ;
    aoclsparse_create_csr(csrA, csrB, &descrA, &descrB);

    aoclsparse_mat_csr csrC = NULL;
    aoclsparse_int *csr_row_ptr_C = NULL;
    aoclsparse_int *csr_col_ind_C = NULL;
    float         *csr_val_C = NULL;
    aoclsparse_int C_M, C_N;

#ifdef TWO_STAGE_COMPUTATION
    std::cout << "Invoking aoclsparse_scsr2m with aoclsparse_stage_nnz_count..";
    // aoclsparse_stage_nnz_count : Only rowIndex array of the CSR matrix
    // is computed internally. The output sparse CSR matrix can be
    // extracted to measure the memory required for full operation.
    request =  aoclsparse_stage_nnz_count;
    status = aoclsparse_scsr2m(transA,
	    descrA,
	    csrA,
	    transB,
	    descrB,
	    csrB,
	    request,
	    &csrC);
    if (status == aoclsparse_status_success)
	std::cout << "DONE\n";
    else
    {
	std::cout << "ERROR in aoclsparse_scsr2m\n";
	exit(EXIT_FAILURE);
    }

    aoclsparse_export_mat_csr(csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, (void **)&csr_val_C);

#ifdef PRINT_OUTPUT
    std::cout << "C_M" << "\t" << "C_N" << "\t" << "nnz_C" << std::endl;
    std::cout << C_M << "\t" << C_N << "\t" << nnz_C << std::endl;

    std::cout << "csr_row_ptr_C: " << std::endl;
    for(aoclsparse_int i = 0 ; i < C_M + 1 ; i++)
	std::cout << csr_row_ptr_C[i] << "\t";
    std::cout <<std::endl;
#endif

    std::cout << "Invoking aoclsparse_scsr2m with aoclsparse_stage_finalize..";
    // aoclsparse_stage_finalize : Finalize computation of remaining
    // output arrays ( column indices and values of output matrix entries) .
    // Has to be called only after aoclsparse_scsr2m call with
    // aoclsparse_stage_nnz_count parameter.
    request =  aoclsparse_stage_finalize;
    status = aoclsparse_scsr2m(transA,
	    descrA,
	    csrA,
	    transB,
	    descrB,
	    csrB,
	    request,
	    &csrC);
    if (status == aoclsparse_status_success)
	std::cout << "DONE\n";
    else
    {
	std::cout << "ERROR in aoclsparse_scsr2m\n";
	exit(EXIT_FAILURE);
    }
    aoclsparse_export_mat_csr(csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, (void **)&csr_val_C);

#ifdef PRINT_OUTPUT
    std::cout << "csr_col_ind_C: " << std::endl;
    for(aoclsparse_int i = 0 ; i < nnz_C ; i++)
	std::cout << csr_col_ind_C[i] << "\t";
    std::cout <<std::endl;

    std::cout << "csr_val_C: " << std::endl;
    for(aoclsparse_int i = 0 ; i < nnz_C ; i++)
	std::cout << csr_val_C[i] << "\t";
    std::cout <<std::endl;
#endif

#else // SINGLE STAGE
    std::cout << "Invoking aoclsparse_scsr2m with aoclsparse_stage_full_computation..";
    // aoclsparse_stage_full_computation :  Whole computation is performed in
    // single step.
    request =  aoclsparse_stage_full_computation;
    status = aoclsparse_scsr2m(transA,
	    descrA,
	    csrA,
	    transB,
	    descrB,
	    csrB,
	    request,
	    &csrC);
    if (status == aoclsparse_status_success)
	std::cout << "DONE\n";
    else
    {
	std::cout << "ERROR in aoclsparse_scsr2m\n";
	exit(EXIT_FAILURE);
    }

    aoclsparse_export_mat_csr(csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, (void **)&csr_val_C);

#ifdef PRINT_OUTPUT
    std::cout << "C_M" << "\t" << "C_N" << "\t" << "nnz_C" << std::endl;
    std::cout << C_M << "\t" << C_N << "\t" << nnz_C << std::endl;

    std::cout << "csr_row_ptr_C: " << std::endl;
    for(aoclsparse_int i = 0 ; i < C_M + 1 ; i++)
	std::cout << csr_row_ptr_C[i] << "\t";
    std::cout <<std::endl;

    std::cout << "csr_col_ind_C: " << std::endl;
    for(aoclsparse_int i = 0 ; i < nnz_C ; i++)
	std::cout << csr_col_ind_C[i] << "\t";
    std::cout <<std::endl;

    std::cout << "csr_val_C: " << std::endl;
    for(aoclsparse_int i = 0 ; i < nnz_C ; i++)
	std::cout << csr_val_C[i] << "\t";
    std::cout <<std::endl;
#endif

#endif
    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy_mat_descr(descrB);
    aoclsparse_destroy_mat_csr(csrA);
    aoclsparse_destroy_mat_csr(csrB);
    aoclsparse_destroy_mat_csr(csrC);
    return 0;

}

