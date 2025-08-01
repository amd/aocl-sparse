/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_types.h"
#include "aoclsparse_analysis.hpp"
#include "aoclsparse_ilu0.hpp"
#include "aoclsparse_mat_structures.hpp"

#include <immintrin.h>
#include <iostream>

template <typename T>
aoclsparse_status aoclsparse_ilu_template(aoclsparse_operation       op,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          T                        **precond_csr_val,
                                          T                         *x,
                                          const T                   *b)
{
    aoclsparse_status ret = aoclsparse_status_success;

    if(descr == nullptr || A == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(x == nullptr || b == nullptr || precond_csr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(A->mats.empty() || !A->mats[0])
        return aoclsparse_status_invalid_pointer;

    if(op != aoclsparse_operation_none)
    {
        return aoclsparse_status_not_implemented;
    }
    // Only CSR input format supported
    if(A->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_not_implemented;
    }
    if(descr->type != aoclsparse_matrix_type_general)
    {
        return aoclsparse_status_not_implemented;
    }
    // Matrix need to be at least partially sorted with all diagonal elements
    if(A->sort != aoclsparse_fully_sorted && A->sort != aoclsparse_partially_sorted)
        return aoclsparse_status_unsorted_input;
    if(!A->fulldiag)
        return aoclsparse_status_numerical_error;

    // Check consistency
    if(A->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }
    if((A->mats[0]->base != aoclsparse_index_base_zero)
       && (A->mats[0]->base != aoclsparse_index_base_one))
    {
        return aoclsparse_status_invalid_value;
    }
    if(A->mats[0]->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }

    // Check sizes (only square matrices are supported)
    if((A->m < 0) || (A->n < 0) || (A->m != A->n))
    {
        return aoclsparse_status_invalid_size;
    }
    // Quick return if possible
    if(A->m == 0 || A->n == 0)
    {
        return aoclsparse_status_success;
    }

    if(A->ilu_info.ilu_ready == false)
    {
        /*
            currently optimize API allocates working buffers needed for ILU
            functionality. ILU Optimize functionality to be extended in future
        */
        ret = aoclsparse_optimize_ilu(A);
        if(ret)
            return ret;
    }

    switch(A->ilu_info.ilu_fact_type)
    {
    case aoclsparse_ilu0:
    {
        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
        if(!csr_mat)
            return aoclsparse_status_not_implemented;
        //Invoke ILU0 API for CSR storage format
        ret = aoclsparse_ilu0_template<T>(A->n,
                                          A->ilu_info.lu_diag_ptr,
                                          &(A->ilu_info.ilu_factorized),
                                          &(A->ilu_info.ilu_fact_type),
                                          (T *)A->ilu_info.precond_csr_val,
                                          csr_mat->ptr,
                                          csr_mat->ind,
                                          csr_mat->base,
                                          precond_csr_val,
                                          x,
                                          b);
        if(ret)
            return ret;
        break;
    }
    default:
        ret = aoclsparse_status_internal_error;
        break;
    }

    return ret;
}

#endif // AOCLSPARSE_ILU_HPP
