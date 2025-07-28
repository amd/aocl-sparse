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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_SYRK_HPP
#define AOCLSPARSE_SYRK_HPP
#endif

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_sypr.hpp"

#include <complex>
#include <iostream>
#include <vector>

// Overestimates the number of nonzeros in the upper triangle of AA' or A'A
aoclsparse_status estimate_nnz(const aoclsparse_operation op,
                               aoclsparse_index_base      base,
                               aoclsparse_int             m,
                               aoclsparse_int             n,
                               const aoclsparse_int      *csr_row_ptr_A,
                               const aoclsparse_int      *csr_col_ind_A,
                               aoclsparse_int            &nnz);

// Compute C=AA' for A mxn, by multiplying a dense row of A by A' for each row.
// Only upper triangle of C is stored as CSR, C needs to be allocated big enough
// prior the computation. Exact number of nnz in C is returned.
template <typename T>
aoclsparse_status aoclsparse_aat_dense_row(aoclsparse_int        m,
                                           aoclsparse_int        n,
                                           aoclsparse_index_base base,
                                           const aoclsparse_int *csr_row_ptr_A,
                                           const aoclsparse_int *csr_col_ind_A,
                                           const T              *csr_val_A,
                                           aoclsparse_int       &nnz_C,
                                           aoclsparse_matrix     C)
{
    if(csr_row_ptr_A == nullptr || csr_col_ind_A == nullptr || csr_val_A == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(!C || C->mats.empty())
        return aoclsparse_status_internal_error;

    aoclsparse::csr *C_csr = dynamic_cast<aoclsparse::csr *>(C->mats[0]);
    if(!C_csr)
        return aoclsparse_status_not_implemented;

    if(!C_csr->ptr || !C_csr->ind || !C_csr->val)
        return aoclsparse_status_invalid_operation;

    T zero        = aoclsparse_numeric::zero<T>();
    csr_col_ind_A = csr_col_ind_A - base;
    csr_val_A     = csr_val_A - base;

    aoclsparse_int *csr_row_ptr_C = C_csr->ptr;
    aoclsparse_int *csr_col_ind_C = C_csr->ind;
    T              *csr_val_C     = (T *)C_csr->val;
    aoclsparse_int  i, j, k;

    std::vector<T> trow; // dense representation of row A_i*
    try
    {
        trow.resize(n, 0);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    nnz_C            = 0;
    csr_row_ptr_C[0] = base;
    for(i = 0; i < m; i++)
    {
        for(j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            trow[csr_col_ind_A[j] - base] = csr_val_A[j];
        for(j = i; j < m; j++) // build U triangle only
        {
            T c_ij = 0;
            for(k = csr_row_ptr_A[j]; k < csr_row_ptr_A[j + 1]; ++k)
            {
                c_ij += trow[csr_col_ind_A[k] - base] * aoclsparse::conj(csr_val_A[k]);
            }
            if(c_ij != zero)
            {
                csr_col_ind_C[nnz_C] = j + base;
                csr_val_C[nnz_C]     = c_ij;
                nnz_C += 1;
            }
        }
        csr_row_ptr_C[i + 1] = base + nnz_C;
        for(j = csr_row_ptr_A[i]; j < csr_row_ptr_A[i + 1]; j++)
            trow[csr_col_ind_A[j] - base] = 0;
    }
    return aoclsparse_status_success;
}

// syrk main entry point, returns upper triangle of AA' (op=none) or A'A
// with the same base as A.
// Validates input and dispatches to appropriate kernel.
template <typename T>
aoclsparse_status aoclsparse_syrk_t(const aoclsparse_operation      op,
                                    const aoclsparse_matrix         A,
                                    aoclsparse_matrix              *C,
                                    [[maybe_unused]] aoclsparse_int kid)
{
    if((A == nullptr) || A->mats.empty() || (C == nullptr))
        return aoclsparse_status_invalid_pointer;

    *C = NULL;

    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
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

    aoclsparse::csr *A_csr = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
    if(!A_csr)
        return aoclsparse_status_not_implemented;

    aoclsparse_int *icrowA = A_csr->ptr;
    aoclsparse_int *icolA  = A_csr->ind;
    T              *valA   = (T *)A_csr->val;

    // overestimate size of the output and allocate the memory
    aoclsparse_int nnz_C;
    status = estimate_nnz(op, A->base, m, n, icrowA, icolA, nnz_C);
    if(status != aoclsparse_status_success)
        return status;
    aoclsparse_int   m_C   = op == aoclsparse_operation_none ? m : n;
    aoclsparse::csr *C_csr = nullptr;
    try
    {
        *C = new _aoclsparse_matrix;
        C_csr
            = new aoclsparse::csr(m_C, m_C, nnz_C, aoclsparse_csr_mat, A->base, get_data_type<T>());
        (*C)->mats.push_back(C_csr);
    }
    catch(std::bad_alloc &)
    {
        if(C_csr)
        {
            delete C_csr;
        }
        aoclsparse_destroy(C);
        return aoclsparse_status_memory_error;
    }
    // Quick return for a 0-sized matrix
    if((A->m == 0) || (A->n == 0) || (A->nnz == 0))
    {
        // preserve base of A in C
        for(aoclsparse_int i = 0; i <= m_C; ++i)
            C_csr->ptr[i] = A->base;
        aoclsparse_init_mat(*C, A->base, m_C, m_C, 0, aoclsparse_csr_mat);
        (*C)->val_type = get_data_type<T>();
        return aoclsparse_status_success;
    }

    if(op == aoclsparse_operation_none) // A*A'
    {
        // These conditions are based on very basic benchmarking, need to be updated later
        if((m < 3000) && (m < n) && (A->nnz <= m * 10))
        {
            status = aoclsparse_aat_dense_row(m, n, A->base, icrowA, icolA, valA, nnz_C, *C);
        }
        else
        {
            // For this algorithm, first need to convert A to CSC and then pass that to ref1
            std::vector<aoclsparse_int> icolAt;
            std::vector<aoclsparse_int> icrowAt;
            std::vector<T>              valAt;
            try
            {
                icolAt.resize(A->nnz);
                icrowAt.resize(n + 1, 0);
                valAt.resize(A->nnz);
            }
            catch(std::bad_alloc &)
            {
                aoclsparse_destroy(C);
                return aoclsparse_status_memory_error;
            }
            status = aoclsparse_csr2csc_template(m,
                                                 n,
                                                 A->nnz,
                                                 A->base,
                                                 A->base,
                                                 icrowA,
                                                 icolA,
                                                 valA,
                                                 icolAt.data(),
                                                 icrowAt.data(),
                                                 valAt.data());
            if(status != aoclsparse_status_success)
            {
                aoclsparse_destroy(C);
                return status;
            }

            // we need Hermitian, so far we transposed so now conjugate
            for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
                valAt[idx] = aoclsparse::conj(valAt[idx]);

            status = aoclsparse_sp2m_online_atb<T, aoclsparse_stage_full_computation, true>(
                n,
                m,
                m,
                A->base,
                icrowAt.data(),
                icolAt.data(),
                valAt.data(),
                A->base,
                icrowAt.data(),
                icolAt.data(),
                valAt.data(),
                A->base,
                C_csr->ptr,
                C_csr->ind,
                (T *)C_csr->val,
                &nnz_C);
        }
    }
    else // A'*A
    {
        status = aoclsparse_sp2m_online_atb<T, aoclsparse_stage_full_computation, true>(
            m,
            n,
            n,
            A->base,
            icrowA,
            icolA,
            valA,
            A->base,
            icrowA,
            icolA,
            valA,
            A->base,
            C_csr->ptr,
            C_csr->ind,
            (T *)C_csr->val,
            &nnz_C);
    }
    if(status != aoclsparse_status_success)
    {
        aoclsparse_destroy(C); // C is incomplete, so destroy it
        return status;
    }
    // finalize C
    aoclsparse_init_mat(*C, A->base, m_C, m_C, nnz_C, aoclsparse_csr_mat);
    (*C)->val_type = get_data_type<T>();
    return aoclsparse_status_success;
}
