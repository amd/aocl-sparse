/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_SORV_HPP
#define AOCLSPARSE_SORV_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_types.h"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_utils.hpp"

#include <complex>

template <typename T>
aoclsparse_status aoclsparse_csr_check_full_diag(aoclsparse_int        size,
                                                 aoclsparse_index_base base,
                                                 aoclsparse::csr      *csr_mat)
{
    bool            full_diag = false;
    aoclsparse_int  i, j, idxstart, idxend, col;
    aoclsparse_int *row_ptr = csr_mat->ptr;
    aoclsparse_int *col_idx = csr_mat->ind - base;
    T              *csr_val = (T *)csr_mat->val - base;
    T               zero    = 0;

    // check for non-zero full diagonal
    for(i = 0; i < size; i++)
    {
        full_diag = false;
        idxstart  = row_ptr[i];
        idxend    = row_ptr[i + 1];
        for(j = idxstart; j < idxend; j++)
        {
            col = col_idx[j] - base;
            if(i == col)
            {
                if(full_diag)
                {
                    // found duplicate (can be zero or non-zero) diagonal after a non-zero diagonal
                    full_diag = false;
                    break;
                }
                if(csr_val[j] != zero)
                {
                    full_diag = true;
                }
                else
                {
                    // found zero diagonal
                    break;
                }
            }
        }
        if(!full_diag)
            break;
    }
    return (full_diag) ? aoclsparse_status_success : aoclsparse_status_invalid_value;
}

template <typename T>
aoclsparse_status aoclsparse_sor_forward_sol(
    const aoclsparse_matrix A, const aoclsparse_mat_descr descr, T omega, T *x, const T *b)
{
    aoclsparse_int   i, j, idxstart, idxend;
    aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
    if(!csr_mat)
        return aoclsparse_status_not_implemented;
    aoclsparse_int *row_ptr = csr_mat->ptr;
    aoclsparse_int *col_idx = csr_mat->ind - descr->base;
    T              *csr_val = (T *)csr_mat->val - descr->base;
    T               axi;
    T               diag_ele;
    aoclsparse_int  col;

    for(i = 0; i < A->n; i++)
    {
        idxstart = row_ptr[i];
        idxend   = row_ptr[i + 1];
        axi      = 0;
        diag_ele = 1;
        for(j = idxstart; j < idxend; j++)
        {
            col = col_idx[j] - descr->base;
            if(i != col)
            {
                // exclude diagonal from this computation
                axi += csr_val[j] * x[col];
            }
            else
            {
                diag_ele = csr_val[j];
            }
        }
        x[i] += omega * ((b[i] - axi) / diag_ele - x[i]);
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_sorv_t(aoclsparse_sor_type        sor_type,
                                    const aoclsparse_mat_descr descr,
                                    const aoclsparse_matrix    A,
                                    T                          omega,
                                    T                          alpha,
                                    T                         *x,
                                    const T                   *b,
                                    aoclsparse_int             kid)
{
    if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>)
    {
        return aoclsparse_status_not_implemented;
    }
    if((A == nullptr) || (descr == nullptr) || (x == nullptr) || (b == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(A->m != A->n) // Matrix not square
    {
        return aoclsparse_status_invalid_size;
    }
    if(A->m == 0)
    {
        /*
          If m=0 then size of x will also be 0. No need to do anything.
          If m is non-zero but nnz=0 that means no diagonals are present. We will fail from
          aoclsparse_csr_check_full_diag()
        */
        return aoclsparse_status_success;
    }
    // Check sizes
    if((A->m < 0) || (A->n < 0) || (A->nnz < 0))
    {
        return aoclsparse_status_invalid_value;
    }
    // Only CSR input format supported
    if(A->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_not_implemented;
    }
    // Only implemented for general matrix
    if(descr->type != aoclsparse_matrix_type_general)
    {
        return aoclsparse_status_not_implemented;
    }
    if(A->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }
    // Invalid value validations
    if(sor_type != aoclsparse_sor_forward)
    {
        if((sor_type == aoclsparse_sor_backward) || (sor_type == aoclsparse_sor_symmetric))
            return aoclsparse_status_not_implemented;
        else
            return aoclsparse_status_invalid_value;
    }
    if((descr->base != aoclsparse_index_base_zero) && (descr->base != aoclsparse_index_base_one))
    {
        return aoclsparse_status_invalid_value;
    }
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    // fill_mode and diag_type are not applicable for general matrix.
    // if((descr->fill_mode != aoclsparse_fill_mode_lower)
    //    && (descr->fill_mode != aoclsparse_fill_mode_upper))
    // {
    //     return aoclsparse_status_invalid_value;
    // }
    // if((descr->diag_type != aoclsparse_diag_type_non_unit)
    //    && (descr->diag_type != aoclsparse_diag_type_unit))
    // {
    //     /* We need diagonal elements to solve for x with current algorithm.
    //      * Hence aoclsparse_diag_type_zero is an invalid value.
    //      */
    //     return aoclsparse_status_invalid_value;
    // }

    // Check if all diagonal elements present and are non-zero
    if(!A->opt_csr_full_diag)
    {
        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
        if(!csr_mat)
            return aoclsparse_status_not_implemented;
        aoclsparse_status status = aoclsparse_csr_check_full_diag<T>(A->m, descr->base, csr_mat);
        if(status != aoclsparse_status_success)
        {
            return status;
        }
        // this will not interfere with other APIs as they call aoclsparse_csr_optimize() based on
        // opt_csr_mat.is_optimized before checking opt_csr_full_diag
        A->opt_csr_full_diag = true;
    }
    // normalize or set x with alpha
    T zero = 0;
    if(alpha != zero)
    {
        for(aoclsparse_int i = 0; i < A->n; i++)
            x[i] = alpha * x[i];
    }
    else
    {
        for(aoclsparse_int i = 0; i < A->n; i++)
            x[i] = alpha;
    }

    (void)kid;
    // only forward sor is supported now.
    return aoclsparse_sor_forward_sol(A, descr, omega, x, b);
}
#endif // AOCLSPARSE_SORV_HPP
