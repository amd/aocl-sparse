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
#ifndef AOCLSPARSE_ILU_HPP
#define AOCLSPARSE_ILU_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_types.h"
#include "aoclsparse_ilu0.hpp"
#include "aoclsparse_analysis.hpp"
#include <immintrin.h>

#include<iostream>

template <typename T>
aoclsparse_status aoclsparse_ilu_template(aoclsparse_operation          op,
                                    aoclsparse_matrix                   A,
                                    const aoclsparse_mat_descr          descr,
                                    const T*                   	diag,
                                    const T*                   	approx_inv_diag,                                    
                                    T*                             x,
                                    const T*                       b )
{
    aoclsparse_status ret = aoclsparse_status_success;

    if(A->ilu_info.ilu_ready == false)
    {
        //optimize or allocate working buffers for ILU if not already done
        aoclsparse_optimize_ilu(A);
    }

    switch(A->ilu_info.ilu_fact_type)
    {
        case 0:
            //Invoke ILU0 API for CSR storage format
                aoclsparse_ilu0_template<T>(A->m,
                                        A->n,
                                        A->nnz,
                                        A->ilu_info.lu_diag_ptr,
                                        A->ilu_info.col_idx_mapper,
                                        &(A->ilu_info.ilu_factorized),
                                        &(A->ilu_info.ilu_fact_type),
                                        (T *)A->ilu_info.precond_csr_val,                
                                        A->csr_mat.csr_row_ptr,
                                        A->csr_mat.csr_col_ptr,
                                        diag,
                                        approx_inv_diag,             
                                        x,
                                        b);                       
            break;
        default:
            ret = aoclsparse_status_invalid_value;
            break;
    }
    
    return ret;
}

#endif // AOCLSPARSE_ILU_HPP

