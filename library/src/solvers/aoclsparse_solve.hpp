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
#ifndef AOCLSPARSE_ILU0_HPP
#define AOCLSPARSE_ILU0_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

template <typename T>
aoclsparse_status aoclsparse_ilu_solve(aoclsparse_int          m,
                                   aoclsparse_int                       n,	
                                   aoclsparse_int* __restrict__          lu_diag_ptr,						   
								   T* __restrict__            	csr_val,								   
								   const aoclsparse_int* __restrict__   row_offsets,
                                   const aoclsparse_int* __restrict__   column_indices,
                                   T* __restrict__                       xv,
                                   const T* __restrict__                 bv)
{
	aoclsparse_status ret = aoclsparse_status_success;
    aoclsparse_int i, k;    

   //Forward Solve
   //Solve L . y = b
   for(i = 0; i < m; i++)
   {
       double sum = bv[i];
       for(k = row_offsets[i]; k < lu_diag_ptr[i]; k++)
       {
           aoclsparse_int col_idx = column_indices[k];
           double temp = 0.0;
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
           double temp = 0.0;
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
aoclsparse_status aoclsparse_ilu_solve_template(aoclsparse_int               m,
                                   aoclsparse_int                       n,
                                   aoclsparse_int* __restrict__         lu_diag_ptr,
								   T* __restrict__            	        csr_val,
                                   const aoclsparse_int* __restrict__   csr_row_ptr,
								   const aoclsparse_int* __restrict__   csr_col_ind,                              								   
                                   T* __restrict__                  	x,
                                   const T* __restrict__            	b )
{
	aoclsparse_status ret = aoclsparse_status_success;
    ret = aoclsparse_ilu_solve(m,
                        n,
                        lu_diag_ptr,
                        csr_val,
                        csr_row_ptr,
                        csr_col_ind,                        
                        x,
                        b);	
	return ret;
}
#endif // AOCLSPARSE_ILU0_HPP

