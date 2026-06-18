/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse.hpp"
#include "aoclsparse_auxiliary.hpp"

#include <algorithm>
#include <complex>
#include <vector>

// Computes the product of two sparse matrices and stores the output
// as a dense matrix in the column major fomat
// Note: this is a duplicate of *ref_row with minor changes to the way C is stored.
// Incorporating this logic into *ref_row will have performance implications due to
// conditionals within the innermost loop.
// Assumption: all inputs are valid.
template <typename T, bool CONJ_A = false>
inline aoclsparse_status
    aoclsparse_sp2md_ref_col(const aoclsparse_operation                  opA,
                             [[maybe_unused]] const aoclsparse_mat_descr descrA,
                             const aoclsparse::csr                      *A_csr,
                             [[maybe_unused]] const aoclsparse_mat_descr descrB,
                             const aoclsparse::csr                      *B_csr,
                             T                                           alpha,
                             T                                          *C,
                             aoclsparse_int                              ldc,
                             [[maybe_unused]] aoclsparse_int             kid)
{
    if(!A_csr || !B_csr || !C)
        return aoclsparse_status_invalid_pointer;

    aoclsparse_int        m_a;
    const aoclsparse_int *rowp_a, *colidx_a, *rowp_b, *colidx_b;
    aoclsparse_int        base_a = A_csr->base;
    aoclsparse_int        base_b = B_csr->base;
    const T              *val_a, *val_b;

    m_a      = A_csr->m;
    rowp_a   = A_csr->ptr;
    colidx_a = A_csr->ind - base_a;
    rowp_b   = B_csr->ptr;
    colidx_b = B_csr->ind - base_b;
    val_a    = (T *)A_csr->val - base_a;
    val_b    = (T *)B_csr->val - base_b;

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
                if constexpr(CONJ_A)
                    val = alpha * aoclsparse::conj(val_a[j]);
                else
                    val = alpha * val_a[j];
                // updates all relevant values of Ci
                for(k = rowp_b[colidx_a[j]]; k < rowp_b[colidx_a[j] + 1]; k++)
                {
                    C[i + (colidx_b[k] - base_b) * ldc] += val * val_b[k];
                }
            }
        }
    }
    else // opA == aoclsparse_operation_transpose (conjugate_transpose handled via CONJ_A=true)
    {
        for(i = 0; i < m_a; i++)
        {
            for(j = rowp_a[i]; j < rowp_a[i + 1]; j++)
            {
                ci = (colidx_a[j] - base_a);
                if constexpr(CONJ_A)
                    val = alpha * aoclsparse::conj(val_a[j]);
                else
                    val = alpha * val_a[j];
                for(k = rowp_b[i]; k < rowp_b[i + 1]; k++)
                {
                    C[ci + (colidx_b[k] - base_b) * ldc] += val * val_b[k];
                }
            }
        }
    }
    return aoclsparse_status_success;
}

//ToDo: Handle opB != aoclsparse_operation_none cases
template <typename T, bool CONJ_A = false>
inline aoclsparse_status
    aoclsparse_sp2md_ref_row(const aoclsparse_operation                  opA,
                             [[maybe_unused]] const aoclsparse_mat_descr descrA,
                             const aoclsparse::csr                      *A_csr,
                             [[maybe_unused]] const aoclsparse_mat_descr descrB,
                             const aoclsparse::csr                      *B_csr,
                             T                                           alpha,
                             T                                          *C,
                             aoclsparse_int                              ldc,
                             [[maybe_unused]] aoclsparse_int             kid)
{
    if(!A_csr || !B_csr || !C)
        return aoclsparse_status_invalid_pointer;

    aoclsparse_int        m_a;
    const aoclsparse_int *rowp_a, *colidx_a, *rowp_b, *colidx_b;
    aoclsparse_int        base_a = A_csr->base;
    aoclsparse_int        base_b = B_csr->base;
    const T              *val_a, *val_b;

    m_a      = A_csr->m;
    rowp_a   = A_csr->ptr;
    colidx_a = A_csr->ind - base_a;
    rowp_b   = B_csr->ptr;
    colidx_b = B_csr->ind - base_b;
    val_a    = (T *)A_csr->val - base_a;
    val_b    = (T *)B_csr->val - base_b;

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
                if constexpr(CONJ_A)
                    val = alpha * aoclsparse::conj(val_a[j]);
                else
                    val = alpha * val_a[j];
                for(k = rowp_b[colidx_a[j]]; k < rowp_b[colidx_a[j] + 1]; k++)
                {
                    C[ci + colidx_b[k] - base_b] += val * val_b[k];
                }
            }
        }
    }
    else // opA == aoclsparse_operation_transpose (conjugate_transpose handled via CONJ_A=true)
    {
        for(i = 0; i < m_a; i++)
        {
            for(j = rowp_a[i]; j < rowp_a[i + 1]; j++)
            {
                ci = (colidx_a[j] - base_a) * ldc;
                if constexpr(CONJ_A)
                    val = alpha * aoclsparse::conj(val_a[j]);
                else
                    val = alpha * val_a[j];
                for(k = rowp_b[i]; k < rowp_b[i + 1]; k++)
                {
                    C[ci + colidx_b[k] - base_b] += val * val_b[k];
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
    if(!A || !B || !C)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Find the raw CSR/CSC object for A and B (stable post-optimize).
    aoclsparse::csr *raw_A = A->get_first_mtx_if_valid<aoclsparse::csr>();
    aoclsparse::csr *raw_B = B->get_first_mtx_if_valid<aoclsparse::csr>();
    if(!raw_A || !raw_B)
        return aoclsparse_status_not_implemented;
    if((raw_A->doid != aoclsparse::doid::gn && raw_A->doid != aoclsparse::doid::gt)
       || (raw_B->doid != aoclsparse::doid::gn && raw_B->doid != aoclsparse::doid::gt))
        return aoclsparse_status_not_implemented;

    bool             owns_mat_B = false;
    aoclsparse::csr *B_op       = nullptr;
    // Verify the matrix types and T are consistent
    if(A->val_type != get_data_type<T>() || B->val_type != get_data_type<T>())
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

    // Overflow check for dense matrix C offset computations in LP64 mode
    // Kernels compute: C[row * ldc + col] (row-major) or C[row + col * ldc] (col-major)
    // Maximum row index = m_c - 1, maximum col index = n_c - 1
    {
        aoclsparse_int c_dim = (layout == aoclsparse_order_row) ? m_c : n_c;
        // Validate full dense address range by checking c_dim * ldc
        if(aoclsparse_numeric::aoclsparse_int_product_overflow(c_dim, ldc))
        {
            return aoclsparse_status_invalid_size;
        }
    }

    if((raw_A->base != descrA->base) || (raw_B->base != descrB->base))
        return aoclsparse_status_invalid_value;

    aoclsparse_status status;

    // A-side: zero allocations; bit-decode eff_A (bit1=transpose, bit0=conjugate).
    aoclsparse::csr       *mat_A_eff = raw_A;
    const aoclsparse::doid eff_A
        = aoclsparse::get_effective_doid(raw_A->doid, aoclsparse::get_doid<T>(descrA, opA));
    const bool           eff_trans_A = (static_cast<int>(eff_A) & 2) != 0;
    const bool           eff_conj_A  = (static_cast<int>(eff_A) & 1) != 0;
    aoclsparse_operation opA_eff
        = eff_trans_A ? aoclsparse_operation_transpose : aoclsparse_operation_none;

    // B-side: effective DOID bit-decode; bit 1 = transpose, bit 0 = conjugate.
    const aoclsparse::doid eff_B
        = aoclsparse::get_effective_doid(raw_B->doid, aoclsparse::get_doid<T>(descrB, opB));
    const bool need_trans_B = (static_cast<int>(eff_B) & 2) != 0;
    const bool need_conj_B  = (static_cast<int>(eff_B) & 1) != 0;

    if(!need_trans_B && !need_conj_B)
    {
        B_op = raw_B;
    }
    else if(need_trans_B)
    {
        // csr2csc input is (raw_B->m × raw_B->n); output rows/cols are swapped.
        try
        {
            B_op = new aoclsparse::csr(
                raw_B->n, raw_B->m, B->nnz, aoclsparse_csr_mat, raw_B->base, raw_B->val_type);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        aoclsparse_status st_b = aoclsparse_csr2csc_template(raw_B->m,
                                                             raw_B->n,
                                                             B->nnz,
                                                             raw_B->base,
                                                             raw_B->base,
                                                             raw_B->ptr,
                                                             raw_B->ind,
                                                             static_cast<const T *>(raw_B->val),
                                                             B_op->ind,
                                                             B_op->ptr,
                                                             static_cast<T *>(B_op->val));
        if(st_b != aoclsparse_status_success)
        {
            delete B_op;
            return st_b;
        }
        if(need_conj_B)
        {
            T *v = static_cast<T *>(B_op->val);
            for(aoclsparse_int i = 0; i < B->nnz; i++)
                v[i] = aoclsparse::conj(v[i]);
        }
        owns_mat_B = true;
    }
    else
    {
        // Conjugate-only: same physical shape as raw storage; values conjugated.
        try
        {
            B_op = new aoclsparse::csr(
                raw_B->m, raw_B->n, B->nnz, aoclsparse_csr_mat, raw_B->base, raw_B->val_type);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        std::copy(raw_B->ptr, raw_B->ptr + raw_B->m + 1, B_op->ptr);
        std::copy(raw_B->ind, raw_B->ind + B->nnz, B_op->ind);
        const T *src_v = static_cast<const T *>(raw_B->val);
        T       *dst_v = static_cast<T *>(B_op->val);
        for(aoclsparse_int i = 0; i < B->nnz; i++)
            dst_v[i] = aoclsparse::conj(src_v[i]);
        owns_mat_B = true;
    }

    // Update the elements of the output matrix with beta
    // beta is zero for calls from spmmd
    if(layout == aoclsparse_order_row)
    {
        if(beta == zero)
        {
            for(aoclsparse_int i = 0; i < m_c; i++)
                for(aoclsparse_int j = 0; j < n_c; j++)
                    C[i * ldc + j] = zero;
        }
        else if(beta != one)
        {
            for(aoclsparse_int i = 0; i < m_c; i++)
                for(aoclsparse_int j = 0; j < n_c; j++)
                    C[i * ldc + j] = beta * C[i * ldc + j];
        }
        if(alpha == zero)
            status = aoclsparse_status_success;
        else if(eff_conj_A)
            status = aoclsparse_sp2md_ref_row<T, true>(
                opA_eff, descrA, mat_A_eff, descrB, B_op, alpha, C, ldc, kid);
        else
            status = aoclsparse_sp2md_ref_row<T, false>(
                opA_eff, descrA, mat_A_eff, descrB, B_op, alpha, C, ldc, kid);
    }
    else
    {
        if(beta == zero)
        {
            for(aoclsparse_int j = 0; j < n_c; j++)
                for(aoclsparse_int i = 0; i < m_c; i++)
                    C[i + j * ldc] = zero;
        }
        else if(beta != one)
        {
            for(aoclsparse_int j = 0; j < n_c; j++)
                for(aoclsparse_int i = 0; i < m_c; i++)
                    C[i + j * ldc] = beta * C[i + j * ldc];
        }
        if(alpha == zero)
            status = aoclsparse_status_success;
        else if(eff_conj_A)
            status = aoclsparse_sp2md_ref_col<T, true>(
                opA_eff, descrA, mat_A_eff, descrB, B_op, alpha, C, ldc, kid);
        else
            status = aoclsparse_sp2md_ref_col<T, false>(
                opA_eff, descrA, mat_A_eff, descrB, B_op, alpha, C, ldc, kid);
    }
    if(owns_mat_B)
        delete B_op;
    return status;
}

#endif // AOCLSPARSE_SP2MD_HPP
