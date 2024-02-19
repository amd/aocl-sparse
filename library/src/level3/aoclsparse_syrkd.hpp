/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_SYRKD_HPP
#define AOCLSPARSE_SYRKD_HPP
#endif

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_sypr.hpp"
#include "aoclsparse_utils.hpp"

#include <complex>
#include <iostream>
#include <vector>

template <typename T>
struct syrkd_params
{
    T                alpha_p, beta_p;
    aoclsparse_order layout_p;
    aoclsparse_int   ldc_p;
};

template <typename T, aoclsparse_order layout>
void inline compute_output_row(aoclsparse_int         i,
                               T                      val_A,
                               aoclsparse_int         iwstart,
                               aoclsparse_int         iwend,
                               const aoclsparse_int  *icolW,
                               const T               *valW,
                               aoclsparse_index_base  baseW,
                               struct syrkd_params<T> params,
                               T                     *C)
{
    aoclsparse_int ldc = params.ldc_p;

    for(aoclsparse_int idxW = iwstart; idxW < iwend; ++idxW)
    {
        // mark all the nonzeroes in the flag array
        aoclsparse_int j = icolW[idxW] - baseW;

        if(j < i) // L triangle element, skip
            continue;

        if constexpr(layout == aoclsparse_order_row)
        {
            C[i * ldc + j] += val_A * valW[idxW];
        }
        else
        {
            C[i + j * ldc] += val_A * valW[idxW];
        }
    }
}

/* Computes C = A^T*A (or A^H*A for complex types) where A is sorted CSR m x k,
 * and the result C is a dense matrix of dimension k x k.
 */
template <typename T>
aoclsparse_status aoclsparse_syrkd_online_atb(aoclsparse_int         m,
                                              aoclsparse_int         k,
                                              aoclsparse_index_base  baseA,
                                              const aoclsparse_int  *icrowA,
                                              const aoclsparse_int  *icolA,
                                              const T               *valA,
                                              struct syrkd_params<T> params,
                                              T                     *C)
{

    if(icrowA == nullptr || icolA == nullptr || valA == nullptr || C == nullptr)
        return aoclsparse_status_invalid_pointer;

    aoclsparse_int    idxa, row;
    aoclsparse_status status;
    aoclsparse_order  layout = params.layout_p;

    // On Fly Transpose
    oftrans oft;
    status = oft.init(m, k, icrowA, baseA, icrowA + 1, baseA, icolA, baseA);
    if(status != aoclsparse_status_success)
        return status;

    // Build i-th row of C, thus pass i-th column of A
    if(layout == aoclsparse_order_row)
    {
        for(aoclsparse_int i = 0; i < k; i++)
        {
            row = oft.rfirst(i);
            while(row >= 0)
            {
                idxa    = oft.ridx(row);
                T val_A = params.alpha_p * aoclsparse::conj(valA[idxa]);

                compute_output_row<T, aoclsparse_order_row>(i,
                                                            val_A,
                                                            icrowA[row] - baseA,
                                                            icrowA[row + 1] - baseA,
                                                            icolA,
                                                            valA,
                                                            baseA,
                                                            params,
                                                            C);
                row = oft.rnext(row);
            }
        }
    }
    else
    {
        for(aoclsparse_int i = 0; i < k; i++)
        {
            row = oft.rfirst(i);
            while(row >= 0)
            {
                idxa    = oft.ridx(row);
                T val_A = params.alpha_p * aoclsparse::conj(valA[idxa]);

                compute_output_row<T, aoclsparse_order_column>(i,
                                                               val_A,
                                                               icrowA[row] - baseA,
                                                               icrowA[row + 1] - baseA,
                                                               icolA,
                                                               valA,
                                                               baseA,
                                                               params,
                                                               C);
                row = oft.rnext(row);
            }
        }
    }
    return aoclsparse_status_success;
}

// syrkd main entry point
// Validates input and dispatches to appropriate kernel.
template <typename T>
inline aoclsparse_status aoclsparse_syrkd_t(const aoclsparse_operation      op,
                                            const aoclsparse_matrix         A,
                                            T                               alpha,
                                            T                               beta,
                                            T                              *C,
                                            const aoclsparse_order          layout,
                                            aoclsparse_int                  ldc,
                                            [[maybe_unused]] aoclsparse_int kid)
{
    if((A == nullptr) || (C == nullptr))
        return aoclsparse_status_invalid_pointer;

    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;

    if(layout != aoclsparse_order_row && layout != aoclsparse_order_column)
        return aoclsparse_status_invalid_value;

    if(A->input_format != aoclsparse_csr_mat)
        return aoclsparse_status_not_implemented;

    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    // Verify op and matrix types match
    if(((A->val_type == aoclsparse_cmat) || (A->val_type == aoclsparse_zmat))
       && (op == aoclsparse_operation_transpose))
        return aoclsparse_status_not_implemented;

    // we need fully sorted rows if we apply on-fly transposition
    if(A->sort != aoclsparse_fully_sorted && op != aoclsparse_operation_none)
        return aoclsparse_status_unsorted_input;

    aoclsparse_int    m = A->m, n = A->n;
    aoclsparse_status status;

    aoclsparse_int        *csr_row_ptr_A = A->csr_mat.csr_row_ptr;
    aoclsparse_int        *csr_col_ind_A = A->csr_mat.csr_col_ptr;
    T                     *csr_val_A     = (T *)A->csr_mat.csr_val;
    T                      zero          = aoclsparse_numeric::zero<T>();
    struct syrkd_params<T> params;
    params.alpha_p  = alpha;
    params.beta_p   = beta;
    params.layout_p = layout;
    params.ldc_p    = ldc;

    aoclsparse_int m_C = op == aoclsparse_operation_none ? m : n;
    if(ldc < m_C)
        return aoclsparse_status_invalid_value;

    if(beta != zero)
    {
        // ideally we can skip this if beta == 1 as we accumulate into C later
        if(layout == aoclsparse_order_row)
        {

            for(aoclsparse_int i = 0; i < m_C; i++)
            {
                for(aoclsparse_int j = i; j < m_C; j++)
                {
                    C[i * ldc + j] = beta * C[i * ldc + j];
                }
            }
        }
        else // layout is aoclsparse_order_column
        {
            for(aoclsparse_int i = 0; i < m_C; i++)
            {
                for(aoclsparse_int j = i; j < m_C; j++)
                {
                    C[i + j * ldc] = beta * C[i + j * ldc];
                }
            }
        }
    }
    else
    {
        if(layout == aoclsparse_order_row)
        {

            for(aoclsparse_int i = 0; i < m_C; i++)
            {
                for(aoclsparse_int j = i; j < m_C; j++)
                {
                    C[i * ldc + j] = 0;
                }
            }
        }
        else // layout is aoclsparse_order_column
        {
            for(aoclsparse_int i = 0; i < m_C; i++)
            {
                for(aoclsparse_int j = i; j < m_C; j++)
                {
                    C[i + j * ldc] = 0;
                }
            }
        }
    }
    // Quick return for a 0-sized matrix
    if((A->m == 0) || (A->n == 0) || (A->nnz == 0))
    {
        // No need to do anything as C is already updated
        return aoclsparse_status_success;
    }

    if(op == aoclsparse_operation_none)
    {

        // For this algorithm, first need to convert A to CSC and then pass that to ref1
        std::vector<aoclsparse_int> csc_row_ind_A;
        std::vector<aoclsparse_int> csc_col_ptr_A;
        std::vector<T>              csc_val_A;
        try
        {
            csc_row_ind_A.resize(A->nnz);
            csc_col_ptr_A.resize(n + 1, 0);
            csc_val_A.resize(A->nnz);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        status = aoclsparse_csr2csc_template(m,
                                             n,
                                             A->nnz,
                                             A->base,
                                             A->base,
                                             csr_row_ptr_A,
                                             csr_col_ind_A,
                                             csr_val_A,
                                             csc_row_ind_A.data(),
                                             csc_col_ptr_A.data(),
                                             csc_val_A.data());
        if(status != aoclsparse_status_success)
        {
            return status;
        }

        // we need Hermitian, so far we transposed so now conjugate
        for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
            csc_val_A[idx] = aoclsparse::conj(csc_val_A[idx]);

        status = aoclsparse_syrkd_online_atb(
            n, m, A->base, csc_col_ptr_A.data(), csc_row_ind_A.data(), csc_val_A.data(), params, C);
    }
    else
    {
        // As the matrix with op is already in CSC, we can call ref directtly by interchaning m and n
        status = aoclsparse_syrkd_online_atb(
            m, n, A->base, csr_row_ptr_A, csr_col_ind_A, csr_val_A, params, C);
    }
    return status;
}
