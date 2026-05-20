/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_sypr.hpp"
#include "aoclsparse_utils.hpp"

#include <complex>
#include <vector>

template <typename T>
struct syrkd_params
{
    T                alpha_p, beta_p;
    aoclsparse_order layout_p;
    aoclsparse_int   ldc_p;
};

template <typename T, aoclsparse_order layout, bool CONJLEFT>
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
        aoclsparse_int j = icolW[idxW] - baseW;

        if(j < i) // L triangle element, skip
            continue;

        if constexpr(layout == aoclsparse_order_row)
        {
            if constexpr(CONJLEFT)
                C[i * ldc + j] += val_A * valW[idxW];
            else
                C[i * ldc + j] += val_A * aoclsparse::conj(valW[idxW]);
        }
        else
        {
            if constexpr(CONJLEFT)
                C[i + j * ldc] += val_A * valW[idxW];
            else
                C[i + j * ldc] += val_A * aoclsparse::conj(valW[idxW]);
        }
    }
}

/* Computes C += alpha * op(M) * op(M)^H into the upper triangle of dense C.
 * M is a sorted CSR matrix of dimension m x k; C output is k x k.
 *
 * CONJLEFT=true  (default): C += alpha * M^H * M   (left factor conjugated)
 * CONJLEFT=false          : C += alpha * M^T * conj(M) (right factor conjugated)
 * For real types both are equivalent since conj() is a no-op.
 */
template <typename T, bool CONJLEFT = true>
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
                idxa = oft.ridx(row);
                T val_A;
                if constexpr(CONJLEFT)
                    val_A = params.alpha_p * aoclsparse::conj(valA[idxa]);
                else
                    val_A = params.alpha_p * valA[idxa];

                compute_output_row<T, aoclsparse_order_row, CONJLEFT>(i,
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
                idxa = oft.ridx(row);
                T val_A;
                if constexpr(CONJLEFT)
                    val_A = params.alpha_p * aoclsparse::conj(valA[idxa]);
                else
                    val_A = params.alpha_p * valA[idxa];

                compute_output_row<T, aoclsparse_order_column, CONJLEFT>(i,
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

    // Complex + op_transpose unsupported regardless of format (CSR/CSC);
    // uses original op, not eff_op.
    if(((A->val_type == aoclsparse_cmat) || (A->val_type == aoclsparse_zmat))
       && (op == aoclsparse_operation_transpose))
        return aoclsparse_status_not_implemented;

    aoclsparse_status status;

    aoclsparse::csr *csr_mat = A->get_first_mtx_if_valid<aoclsparse::csr>();
    if(!csr_mat)
        return aoclsparse_status_not_implemented;

    // For CSR (doid::gn): csr_mat->m == A->m, csr_mat->n == A->n.
    // For CSC (doid::gt): csr_mat->m == A->n (n_user), csr_mat->n == A->m (m_user).
    // Reading from csr_mat is correct for both — it gives the dimensions of the
    // matrix as actually stored in memory (A^T for CSC).
    aoclsparse_int m = csr_mat->m, n = csr_mat->n;

    /* Dispatch table — CSR (doid::gn) and CSC (doid::gt, stores A^T internally):
     *   CSR | op_none      : β·C + α·A·A^H   (real: A·A^T)
     *   CSR | op_t / op_h  : β·C + α·A^H·A   (real: A^T·A)
     *   CSC | op_none      : β·C + α·A^T·conj(A)  (real: A^T·A);  eff_op=op_h,    conj_flip=true
     *   CSC | op_t / op_h  : β·C + α·conj(A)·A^T  (real: A·A^T);  eff_op=op_none, conj_flip=true
     */
    // Accept CSR (gn) and CSC-stored-as-transposed-CSR (gt);
    // reject all other doid values
    bool is_doid_gt = (csr_mat->doid == aoclsparse::doid::gt);
    if(!is_doid_gt && csr_mat->doid != aoclsparse::doid::gn)
        return aoclsparse_status_not_implemented;

    // For CSC (doid::gt), internal storage is A^T. Flip op so the unchanged CSR
    // compute paths produce correct results. conj_flip=true signals that conjugation
    // shifts to the right factor in the else-branch atb call, and that the
    // pre-conjugation loop in the eff_op==op_none path must be skipped.
    aoclsparse_operation eff_op    = op;
    bool                 conj_flip = false;
    if(is_doid_gt)
    {
        eff_op    = (op == aoclsparse_operation_none) ? aoclsparse_operation_conjugate_transpose
                                                      : aoclsparse_operation_none;
        conj_flip = true;
    }

    // we need fully sorted rows if we apply on-fly transposition
    if(A->sort != aoclsparse_fully_sorted && eff_op != aoclsparse_operation_none)
        return aoclsparse_status_unsorted_input;

    aoclsparse_int        *csr_row_ptr_A = csr_mat->ptr;
    aoclsparse_int        *csr_col_ind_A = csr_mat->ind;
    T                     *csr_val_A     = (T *)csr_mat->val;
    T                      zero          = aoclsparse_numeric::zero<T>();
    struct syrkd_params<T> params;
    params.alpha_p  = alpha;
    params.beta_p   = beta;
    params.layout_p = layout;
    params.ldc_p    = ldc;

    aoclsparse_int m_C = eff_op == aoclsparse_operation_none ? m : n;
    if(ldc < m_C)
        return aoclsparse_status_invalid_value;

    // Overflow check for dense matrix C offset computations in LP64 mode
    // SYRKD computes a symmetric m_C x m_C output matrix (upper triangle)
    // Kernels compute: C[i * ldc + j] (row-major) or C[i + j * ldc] (col-major)
    // With ldc >= m_C, the maximum dense index is bounded by m_C * ldc.
    if(aoclsparse_lp64_product_overflow(m_C, ldc))
    {
        return aoclsparse_status_invalid_size;
    }

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

    if(eff_op == aoclsparse_operation_none)
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
                                             csr_mat->base,
                                             csr_mat->base,
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

        if(conj_flip)
            // CSC op_h: β·C + α·A^H·A  (real: β·C + α·A^T·A)
            // CSC op_h    (conj_flip=true):  CONJLEFT=true  → csr2csc(A^T)=A,  atb computes A^H·A
            status = aoclsparse_syrkd_online_atb<T, true>(n,
                                                          m,
                                                          csr_mat->base,
                                                          csc_col_ptr_A.data(),
                                                          csc_row_ind_A.data(),
                                                          csc_val_A.data(),
                                                          params,
                                                          C);
        else
            // CSR op_none (conj_flip=false): CONJLEFT=false → csr2csc(A)=A^T, atb computes A·A^H
            // CSR op_none: β·C + α·A·A^H  (real: β·C + α·A·A^T)
            status = aoclsparse_syrkd_online_atb<T, false>(n,
                                                           m,
                                                           csr_mat->base,
                                                           csc_col_ptr_A.data(),
                                                           csc_row_ind_A.data(),
                                                           csc_val_A.data(),
                                                           params,
                                                           C);
    }
    else
    {
        if(conj_flip)
            // CSC op_none: β·C + α·A·A^H  (real: β·C + α·A·A^T)
            status = aoclsparse_syrkd_online_atb<
                T,
                false>( // CONJLEFT=false: α·M^T·conj(M) on internal A^T
                m,
                n,
                csr_mat->base,
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                params,
                C);
        else
            // CSR op_h/op_t: β·C + α·A^H·A  (real: β·C + α·A^T·A)
            status = aoclsparse_syrkd_online_atb<T, true>( // CONJLEFT=true: α·M^H·M (default)
                m,
                n,
                csr_mat->base,
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                params,
                C);
    }
    return status;
}
#endif // AOCLSPARSE_SYRKD_HPP
