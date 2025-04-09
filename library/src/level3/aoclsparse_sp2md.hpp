/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_SP2MD_HPP
#define AOCLSPARSE_SP2MD_HPP
#endif

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_utils.hpp"

#include <complex>
#include <iostream>
#include <vector>

// Computes the product of two sparse matrices and stores the output
// as a dense matrix in the column major fomat
// Note: this is a duplicate of *ref_row with minor changes to the way C is stored.
// Incorporating this logic into *ref_row will have performance implications due to
// conditionals within the innermost loop.
// Assumption: all inputs are valid.
template <typename T>
inline aoclsparse_status
    aoclsparse_sp2md_ref_col(const aoclsparse_operation                  opA,
                             [[maybe_unused]] const aoclsparse_mat_descr descrA,
                             const aoclsparse_matrix                     A,
                             [[maybe_unused]] const aoclsparse_mat_descr descrB,
                             const aoclsparse_matrix                     B,
                             T                                           alpha,
                             T                                          *C,
                             aoclsparse_int                              ldc,
                             [[maybe_unused]] aoclsparse_int             kid)
{
    aoclsparse_int        m_a;
    const aoclsparse_int *rowp_a, *colidx_a, *rowp_b, *colidx_b;
    aoclsparse_int        base_a = A->base;
    aoclsparse_int        base_b = B->base;
    const T              *val_a, *val_b;
    m_a      = A->m;
    rowp_a   = A->csr_mat.ptr;
    colidx_a = A->csr_mat.ind - base_a;
    rowp_b   = B->csr_mat.ptr;
    colidx_b = B->csr_mat.ind - base_b;
    val_a    = (T *)A->csr_mat.val - base_a;
    val_b    = (T *)B->csr_mat.val - base_b;

    aoclsparse_int i, j, k;
    aoclsparse_int ci;
    T              val;
    if(opA == aoclsparse_operation_none)
    {
        // Correction required for 1-based index as colidx_a is used for indirection
        rowp_b = rowp_b - base_a;
        for(i = 0; i < m_a; i++)
        {
            // compute the values of ith row of C (Ci)
            for(j = rowp_a[i]; j < rowp_a[i + 1]; j++)
            {
                val = alpha * val_a[j];
                // updates all relevant values of Ci
                for(k = rowp_b[colidx_a[j]]; k < rowp_b[colidx_a[j] + 1]; k++)
                {
                    C[i + (colidx_b[k] - base_b) * ldc] += val * val_b[k];
                }
            }
        }
    }
    else if(opA == aoclsparse_operation_transpose)
    {
        for(i = 0; i < m_a; i++)
        {
            for(j = rowp_a[i]; j < rowp_a[i + 1]; j++)
            {
                ci  = (colidx_a[j] - base_a);
                val = alpha * val_a[j];
                for(k = rowp_b[i]; k < rowp_b[i + 1]; k++)
                {
                    C[ci + (colidx_b[k] - base_b) * ldc] += val * val_b[k];
                }
            }
        }
    }
    else if(opA == aoclsparse_operation_conjugate_transpose)
    {
        T cnj_val;
        for(i = 0; i < m_a; i++)
        {
            for(j = rowp_a[i]; j < rowp_a[i + 1]; j++)
            {
                ci      = (colidx_a[j] - base_a);
                cnj_val = alpha * aoclsparse::conj(val_a[j]);
                for(k = rowp_b[i]; k < rowp_b[i + 1]; k++)
                {
                    C[ci + (colidx_b[k] - base_b) * ldc] += cnj_val * val_b[k];
                }
            }
        }
    }
    return aoclsparse_status_success;
}

//ToDo: Handle opB != aoclsparse_operation_none cases
template <typename T>
inline aoclsparse_status
    aoclsparse_sp2md_ref_row(const aoclsparse_operation                  opA,
                             [[maybe_unused]] const aoclsparse_mat_descr descrA,
                             const aoclsparse_matrix                     A,
                             [[maybe_unused]] const aoclsparse_mat_descr descrB,
                             const aoclsparse_matrix                     B,
                             T                                           alpha,
                             T                                          *C,
                             aoclsparse_int                              ldc,
                             [[maybe_unused]] aoclsparse_int             kid)
{
    aoclsparse_int        m_a;
    const aoclsparse_int *rowp_a, *colidx_a, *rowp_b, *colidx_b;
    aoclsparse_int        base_a = A->base;
    aoclsparse_int        base_b = B->base;
    const T              *val_a, *val_b;
    m_a      = A->m;
    rowp_a   = A->csr_mat.ptr;
    colidx_a = A->csr_mat.ind - base_a;
    rowp_b   = B->csr_mat.ptr;
    colidx_b = B->csr_mat.ind - base_b;
    val_a    = (T *)A->csr_mat.val - base_a;
    val_b    = (T *)B->csr_mat.val - base_b;

    aoclsparse_int i, j, k;
    aoclsparse_int ci;
    T              val;
    if(opA == aoclsparse_operation_none)
    {
        // correction required for 1-based index as colidx_a is used for indirection
        rowp_b = rowp_b - base_a;
        for(i = 0; i < m_a; i++)
        {
            ci = i * ldc;
            for(j = rowp_a[i]; j < rowp_a[i + 1]; j++)
            {
                val = alpha * val_a[j];
                for(k = rowp_b[colidx_a[j]]; k < rowp_b[colidx_a[j] + 1]; k++)
                {
                    C[ci + colidx_b[k] - base_b] += val * val_b[k];
                }
            }
        }
    }
    else if(opA == aoclsparse_operation_transpose)
    {
        for(i = 0; i < m_a; i++)
        {
            for(j = rowp_a[i]; j < rowp_a[i + 1]; j++)
            {
                ci  = (colidx_a[j] - base_a) * ldc;
                val = alpha * val_a[j];
                for(k = rowp_b[i]; k < rowp_b[i + 1]; k++)
                {
                    C[ci + colidx_b[k] - base_b] += val * val_b[k];
                }
            }
        }
    }
    else if(opA == aoclsparse_operation_conjugate_transpose)
    {
        T cnj_val;
        for(i = 0; i < m_a; i++)
        {
            for(j = rowp_a[i]; j < rowp_a[i + 1]; j++)
            {
                ci      = (colidx_a[j] - base_a) * ldc;
                cnj_val = alpha * aoclsparse::conj(val_a[j]);
                for(k = rowp_b[i]; k < rowp_b[i + 1]; k++)
                {
                    C[(colidx_a[j] - base_a) * ldc + colidx_b[k] - base_b] += cnj_val * val_b[k];
                }
            }
        }
    }
    return aoclsparse_status_success;
}

aoclsparse_int static inline get_combined_op_type(aoclsparse_operation op1,
                                                  aoclsparse_operation op2)
{
    aoclsparse_int ret;
    ret = (op1 != aoclsparse_operation_none) | (op2 != aoclsparse_operation_none) << 1;
    return ret;
}

// sp2md main entry point
// Validates input and dispatches to appropriate kernel.
template <typename T>
inline aoclsparse_status aoclsparse_sp2md_t(const aoclsparse_operation      opA,
                                            const aoclsparse_mat_descr      descrA,
                                            const aoclsparse_matrix         A,
                                            const aoclsparse_operation      opB,
                                            const aoclsparse_mat_descr      descrB,
                                            const aoclsparse_matrix         B,
                                            T                               alpha,
                                            T                               beta,
                                            T                              *C,
                                            aoclsparse_order                layout,
                                            aoclsparse_int                  ldc,
                                            [[maybe_unused]] aoclsparse_int kid)
{
    aoclsparse_int m_c, n_c;
    bool           dim_check;
    T              one = 1, zero = 0;
    // Descriptors are ignored at present as we only support "aoclsparse_matrix_type_general"
    if(!(descrA->type == aoclsparse_matrix_type_general
         && descrB->type == aoclsparse_matrix_type_general))
    {
        return aoclsparse_status_not_implemented;
    }

    if((layout != aoclsparse_order_row) && (layout != aoclsparse_order_column))
    {
        return aoclsparse_status_invalid_value;
    }

    // ToDo: switch based on kid.
    // At present calling the reference implementation

    // All validations
    // Input validations
    if((nullptr == A) || (nullptr == B) || (nullptr == C))
    {
        return aoclsparse_status_invalid_pointer;
    }
    if((A->mat_type != aoclsparse_csr_mat) || (B->mat_type != aoclsparse_csr_mat))
    {
        return aoclsparse_status_not_implemented;
    }

    // Verify the matrix types and T are consistent
    if(!((A->val_type == aoclsparse_smat && std::is_same_v<T, float>)
         || (A->val_type == aoclsparse_dmat && std::is_same_v<T, double>)
         || (A->val_type == aoclsparse_cmat && std::is_same_v<T, std::complex<float>>)
         || (A->val_type == aoclsparse_zmat && std::is_same_v<T, std::complex<double>>)))
        return aoclsparse_status_wrong_type;
    if(!((B->val_type == aoclsparse_smat && std::is_same_v<T, float>)
         || (B->val_type == aoclsparse_dmat && std::is_same_v<T, double>)
         || (B->val_type == aoclsparse_cmat && std::is_same_v<T, std::complex<float>>)
         || (B->val_type == aoclsparse_zmat && std::is_same_v<T, std::complex<double>>)))
        return aoclsparse_status_wrong_type;

    aoclsparse_int combined_op_type = get_combined_op_type(opA, opB);
    switch(combined_op_type)
    {
    case 0: // opA = opB = aoclsparse_operation_none
        dim_check = (A->n == B->m);
        m_c       = A->m;
        n_c       = B->n;
        break;
    case 1: // opA !=aoclsparse_operation_none, opB = aoclsparse_operation_none
        dim_check = (A->m == B->m);
        m_c       = A->n;
        n_c       = B->n;
        break;

    case 2: // opA = aoclsparse_operation_none, opB ! aoclsparse_operation_none
        dim_check = (A->n == B->n);
        m_c       = A->m;
        n_c       = B->m;
        break;
    case 3: // opA !=aoclsparse_operation_none, opB != aoclsparse_operation_none
        dim_check = (A->m == B->n);
        m_c       = A->n;
        n_c       = B->m;
        break;
    }

    if(!dim_check)
    {
        return aoclsparse_status_invalid_size;
    }

    // Validate ldc
    aoclsparse_int ldc_min = (layout == aoclsparse_order_row) ? n_c : m_c;
    if(ldc < ldc_min)
    {
        return aoclsparse_status_invalid_size;
    }

    // Check if base index is consistent
    if((A->base != descrA->base) || (B->base != descrB->base))
    {
        return aoclsparse_status_invalid_value;
    }

    // Holds data related to opB(B)
    aoclsparse_matrix           B_op;
    std::vector<aoclsparse_int> csr_row_ptr_B_op;
    std::vector<aoclsparse_int> csr_col_ind_B_op;
    std::vector<T>              csr_val_B_op;
    aoclsparse_status           status;

    // When opB is either transpose or conjugate transpose,
    // create a new matrix in CSC format and treat it as a
    // transposed (conjugate transpose) CSR matrix
    // ToDo: an efficient algorithm without explicit transpose
    if(opB != aoclsparse_operation_none)
    {
        try
        {
            csr_val_B_op.resize(B->nnz);
            csr_col_ind_B_op.resize(B->nnz);
            csr_row_ptr_B_op.resize(B->n + 1);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        status = aoclsparse_csr2csc_template(B->m,
                                             B->n,
                                             B->nnz,
                                             descrB->base,
                                             descrB->base,
                                             B->csr_mat.ptr,
                                             B->csr_mat.ind,
                                             static_cast<T *>(B->csr_mat.val),
                                             csr_col_ind_B_op.data(),
                                             csr_row_ptr_B_op.data(),
                                             csr_val_B_op.data());
        if(status != aoclsparse_status_success)
        {
            return status;
        }

        // ToDo: This operation can be done during csr2csc, for now performing it separately
        if(opB == aoclsparse_operation_conjugate_transpose)
        {
            for(aoclsparse_int i = 0; i < B->nnz; i++)
            {
                csr_val_B_op[i] = aoclsparse::conj(csr_val_B_op[i]);
            }
        }
        status = aoclsparse_create_csr_t(&B_op,
                                         descrB->base,
                                         B->n,
                                         B->m,
                                         B->nnz,
                                         csr_row_ptr_B_op.data(),
                                         csr_col_ind_B_op.data(),
                                         (T *)csr_val_B_op.data());
        if(status != aoclsparse_status_success)
        {
            return status;
        }
    }

    // Update the elements of the output matrix with beta
    // beta is zero for calls from spmmd
    if(layout == aoclsparse_order_row)
    {
        if(beta != one)
        {
            for(aoclsparse_int i = 0; i < m_c; i++)
            {
                for(aoclsparse_int j = 0; j < n_c; j++)
                {
                    C[i * ldc + j] = beta * C[i * ldc + j];
                }
            }
        }
        if(alpha == zero)
            return aoclsparse_status_success;
        if(opB == aoclsparse_operation_none)
        {
            return aoclsparse_sp2md_ref_row(opA, descrA, A, descrB, B, alpha, C, ldc, kid);
        }
        else
        {
            status = aoclsparse_sp2md_ref_row(opA, descrA, A, descrB, B_op, alpha, C, ldc, kid);
            aoclsparse_destroy(&B_op);
            return status;
        }
    }
    else
    {
        if(beta != one)
        {
            for(aoclsparse_int i = 0; i < m_c; i++)
            {
                for(aoclsparse_int j = 0; j < n_c; j++)
                {
                    C[i + j * ldc] = beta * C[i + j * ldc];
                }
            }
        }
        if(alpha == zero)
            return aoclsparse_status_success;

        if(opB == aoclsparse_operation_none)
        {
            return aoclsparse_sp2md_ref_col(opA, descrA, A, descrB, B, alpha, C, ldc, kid);
        }
        else
        {
            status = aoclsparse_sp2md_ref_col(opA, descrA, A, descrB, B_op, alpha, C, ldc, kid);
            aoclsparse_destroy(&B_op);
            return status;
        }
    }
}
