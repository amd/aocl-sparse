/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_csr.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

aoclsparse_status aoclsparse_dcsr2m_nnz_count(
	aoclsparse_int             m,
	aoclsparse_int             n,
	aoclsparse_int*            nnz_C,
	const aoclsparse_int*      csr_row_ptr_A,
	const aoclsparse_int*      csr_col_ind_A,
	const aoclsparse_int*      csr_row_ptr_B,
	const aoclsparse_int*      csr_col_ind_B,
	aoclsparse_int*            csr_row_ptr_C)
{

    csr_row_ptr_C[0]=0;

    std::vector<aoclsparse_int> nnz(n, -1);

    // Loop over rows of A
    for(aoclsparse_int i = 0; i < m; i++)
    {
	aoclsparse_int num_nonzeros = 0;

	// Loop over columns of A
	for(aoclsparse_int j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
	{
	    // Current column of A
	    aoclsparse_int col_A = csr_col_ind_A[j];

	    // Loop over columns of B in row j
	    for(aoclsparse_int k = csr_row_ptr_B[col_A]; k < csr_row_ptr_B[col_A + 1]; k++)
	    {
		// Current column of B
		aoclsparse_int col_B = csr_col_ind_B[k];

		// Check if a new nnz is generated
		if(nnz[col_B] != i)
		{
		    nnz[col_B] = i;
		    num_nonzeros++;
		}
	    }
	}
	csr_row_ptr_C[i + 1] = num_nonzeros;
    }

    // Scan to obtain row offsets
    for(aoclsparse_int i = 1; i <= m; i++)
	csr_row_ptr_C[i] += csr_row_ptr_C[i - 1];

    *nnz_C = csr_row_ptr_C[m];
    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_dcsr2m_finalize(
	aoclsparse_int             m,
	aoclsparse_int             n,
	const aoclsparse_int*      csr_row_ptr_A,
	const aoclsparse_int*      csr_col_ind_A,
	const double*              csr_val_A,
	const aoclsparse_int*      csr_row_ptr_B,
	const aoclsparse_int*      csr_col_ind_B,
	const double*              csr_val_B,
	aoclsparse_int*            csr_row_ptr_C,
	aoclsparse_int*            csr_col_ind_C,
	double*                    csr_val_C)
{
    std::vector<aoclsparse_int> nnz(n, -1);
    std::vector<double> sum(n, 0.0);

    // Loop over rows of A
    for (aoclsparse_int i = 0; i < m; i++)
    {
	aoclsparse_int head   = -100;
	aoclsparse_int length = 0;

	aoclsparse_int row_begin_A = csr_row_ptr_A[i];
	aoclsparse_int row_end_A   = csr_row_ptr_A[i + 1];

	// Loop over columns of A
	for (aoclsparse_int j = row_begin_A; j < row_end_A; j++)
	{
	    // Current column of A
	    aoclsparse_int col_A = csr_col_ind_A[j];
	    // Current value of A
	    double val_A = csr_val_A[j];

	    aoclsparse_int row_begin_B = csr_row_ptr_B[col_A];
	    aoclsparse_int row_end_B   = csr_row_ptr_B[col_A + 1];

	    // Loop over columns of B in row col_A
	    for (aoclsparse_int k = row_begin_B; k < row_end_B; k++)
	    {
		// Current column of B
		aoclsparse_int col_B = csr_col_ind_B[k];
		// Current value of B
		double val_B = csr_val_B[k];

		sum[col_B] = sum[col_B] + val_A * val_B ;

		if (nnz[col_B] == -1)
		{
		    nnz[col_B] = head;
		    head = col_B;
		    length++;
		}
	    }
	}
	aoclsparse_int offset = csr_row_ptr_C[i+1] -1;

	for (aoclsparse_int j = 0; j < length; j++)
	{
	    csr_col_ind_C[offset] = head;
	    csr_val_C[offset] = sum[head];
	    offset--;

	    aoclsparse_int temp = head;
	    head = nnz[head];

	    // clear arrays
	    nnz[temp] = -1;
	    sum[temp] = 0.0;
	}
    }
    return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_dcsr2m(
	aoclsparse_operation       transA,
	const aoclsparse_mat_descr descrA,
	const aoclsparse_mat_csr   csrA,
	aoclsparse_operation       transB,
	const aoclsparse_mat_descr descrB,
	const aoclsparse_mat_csr   csrB,
	aoclsparse_request         request,
	aoclsparse_mat_csr         *csrC)
{
    // Check for valid handle and matrix descriptor
    if((descrA == nullptr) ||(descrB == nullptr))
    {
	return aoclsparse_status_invalid_pointer;
    }

    if((csrA == nullptr) || (csrB == nullptr) || (csrC == nullptr))
    {
	return aoclsparse_status_invalid_pointer;
    }
    if(transA != aoclsparse_operation_none)
    {
	// TODO
	return aoclsparse_status_not_implemented;
    }
    if(transB != aoclsparse_operation_none)
    {
	// TODO
	return aoclsparse_status_not_implemented;
    }

    // Check index base
    if(descrA->base != aoclsparse_index_base_zero)
    {
	// TODO
	return aoclsparse_status_not_implemented;
    }

    if(descrA->type != aoclsparse_matrix_type_general)
    {
	// TODO
	return aoclsparse_status_not_implemented;
    }

    if(csrA->n != csrB->m)
	return aoclsparse_status_invalid_value;

    switch (request) {

	case aoclsparse_stage_nnz_count:
	    {
		aoclsparse_int m = csrA->m;
		aoclsparse_int n = csrB->n;
		aoclsparse_int nnz_C = 0;
		aoclsparse_int *csr_row_ptr_C = (aoclsparse_int *)malloc( (m + 1) * sizeof(aoclsparse_int) );

		aoclsparse_dcsr2m_nnz_count( m,
			n,
			&nnz_C,
			csrA->csr_row_ptr,
			csrA->csr_col_ind,
			csrB->csr_row_ptr,
			csrB->csr_col_ind,
			csr_row_ptr_C);
		*csrC = new _aoclsparse_mat_csr;
		(*csrC)->m = m;
		(*csrC)->n = n;
		(*csrC)->csr_nnz = nnz_C;
		(*csrC)->csr_row_ptr = csr_row_ptr_C;
		break;
	    }
	case aoclsparse_stage_finalize:
	    {
		if(((*csrC)->m == 0) || ((*csrC)->n == 0) || ((*csrC)->csr_nnz == 0))
		    return aoclsparse_status_invalid_value;
		if((*csrC)->csr_row_ptr == nullptr)
		    return aoclsparse_status_invalid_pointer;

		aoclsparse_int m = (*csrC)->m;
		aoclsparse_int n = (*csrC)->n;
		aoclsparse_int nnz_C = (*csrC)->csr_nnz;

		aoclsparse_int *csr_row_ptr_C = (*csrC)->csr_row_ptr;
		aoclsparse_int *csr_col_ind_C = (aoclsparse_int *)malloc(nnz_C * sizeof(aoclsparse_int));
		double *csr_val_C = (double *)malloc(nnz_C * sizeof(double));

		aoclsparse_dcsr2m_finalize( m,
			n,
			csrA->csr_row_ptr,
			csrA->csr_col_ind,
			(const double*)csrA->csr_val,
			csrB->csr_row_ptr,
			csrB->csr_col_ind,
			(const double*)csrB->csr_val,
			csr_row_ptr_C,
			csr_col_ind_C,
			csr_val_C);

		(*csrC)->csr_col_ind = csr_col_ind_C;
		(*csrC)->csr_val = csr_val_C;
		break;
	    }
	case aoclsparse_stage_full_computation:
	    {
		aoclsparse_int m = csrA->m;
		aoclsparse_int n = csrB->n;
		aoclsparse_int nnz_C = 0;
		aoclsparse_int *csr_row_ptr_C = (aoclsparse_int *)malloc( (m + 1) * sizeof(aoclsparse_int) );

		aoclsparse_dcsr2m_nnz_count( m,
			n,
			&nnz_C,
			csrA->csr_row_ptr,
			csrA->csr_col_ind,
			csrB->csr_row_ptr,
			csrB->csr_col_ind,
			csr_row_ptr_C);

		aoclsparse_int *csr_col_ind_C = (aoclsparse_int *)malloc(nnz_C * sizeof(aoclsparse_int));
		double *csr_val_C = (double *)malloc(nnz_C * sizeof(double));

		aoclsparse_dcsr2m_finalize( m,
			n,
			csrA->csr_row_ptr,
			csrA->csr_col_ind,
			(const double*)csrA->csr_val,
			csrB->csr_row_ptr,
			csrB->csr_col_ind,
			(const double*)csrB->csr_val,
			csr_row_ptr_C,
			csr_col_ind_C,
			csr_val_C);
		*csrC = new _aoclsparse_mat_csr;
		(*csrC)->m = m;
		(*csrC)->n = n;
		(*csrC)->csr_nnz = nnz_C;
		(*csrC)->csr_row_ptr = csr_row_ptr_C;
		(*csrC)->csr_col_ind = csr_col_ind_C;
		(*csrC)->csr_val = csr_val_C;
		break;
	    }
	default:
	    return aoclsparse_status_invalid_value;

    }
	    return aoclsparse_status_success;
}

