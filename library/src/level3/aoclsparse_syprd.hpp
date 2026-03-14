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
 * ************************************************************************ */

#ifndef AOCLSPARSE_SYPRD_HPP
#define AOCLSPARSE_SYPRD_HPP
#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <immintrin.h>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

// CONJLEFT=false (default): right factor conjugated — computes M*B*M^H.
// CONJLEFT=true:            left factor conjugated  — computes conj(M)*B*M^T,
//           to be used when M=A^T to compute A^H*B*A
// for B and C stored as row-major.
template <typename T, bool CONJLEFT = false>
aoclsparse_status aoclsparse_syprd_row_ref(const T *__restrict__ val,
                                           const aoclsparse_int *__restrict__ col_ind,
                                           const aoclsparse_int *__restrict__ row_ptr,
                                           aoclsparse_index_base base,
                                           aoclsparse_int        m,
                                           aoclsparse_int        n,
                                           const T              *B,
                                           aoclsparse_int        ldb,
                                           T                     alpha,
                                           T                     beta,
                                           T                    *C,
                                           aoclsparse_int        ldc)
{
    T                     zero = aoclsparse_numeric::zero<T>();
    const aoclsparse_int *csr_col_ind, *csr_row_ptr;
    const T              *csr_val;
    csr_val     = val - base;
    csr_col_ind = col_ind - base;
    csr_row_ptr = row_ptr;

    if(beta != zero)
    {
        //Apply beta * C for the upper triangular matrix
        for(aoclsparse_int i = 0; i < m; i++)
        {
            aoclsparse_int idx_c;
            idx_c = i * ldc;
            for(aoclsparse_int j = i; j < m; ++j)
            {
                C[idx_c + j] = beta * C[idx_c + j];
            }
        }
    }
    else
    {
        for(aoclsparse_int i = 0; i < m; i++)
        {
            aoclsparse_int idx_c;
            idx_c = i * ldc;
            for(aoclsparse_int j = i; j < m; ++j)
            {
                C[idx_c + j] = zero;
            }
        }
    }

    if(alpha == zero)
    {
        return aoclsparse_status_success;
    }

    // Perform matrix multiplication.
    // Store the intermediate result in temp and multiply with the sparse matrix in a transposed way.
    // This logic can be applied to the syprd_col_ref as well.
    std::vector<T> temp;
    try
    {
        temp.resize(n);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    for(aoclsparse_int i = 0; i < m; i++)
    {
        // temp = i-th row of alpha*M*hermitized(B)       for CONJLEFT=false (default)
        // or                 alpha*conj(M)*hermitized(B) for CONJLEFT=true
        aoclsparse_int row_begin = csr_row_ptr[i];
        aoclsparse_int row_end   = csr_row_ptr[i + 1];
        for(aoclsparse_int j = 0; j < n; j++)
            temp[j] = zero;
        for(aoclsparse_int k = row_begin; k < row_end; k++)
        {
            aoclsparse_int colM = csr_col_ind[k] - base;
            T              valM;
            if constexpr(CONJLEFT)
                valM = alpha * aoclsparse::conj(csr_val[k]);
            else
                valM = alpha * csr_val[k];
            for(aoclsparse_int j = 0; j < colM; j++)
            {
                temp[j] += aoclsparse::conj(B[j * ldb + colM]) * valM;
            }
            for(aoclsparse_int j = colM; j < n; j++)
            {
                temp[j] += B[colM * ldb + j] * valM;
            }
        }

        // compute i-th row of upper triangle of C as
        //    temp * M^H  for CONJLEFT=false  or
        //    temp * M^T  for CONJLEFT=true
        for(aoclsparse_int j = i; j < m; j++)
        {
            row_begin            = csr_row_ptr[j];
            row_end              = csr_row_ptr[j + 1];
            aoclsparse_int idx_c = i * ldc + j;
            for(aoclsparse_int k = row_begin; k < row_end; k++)
            {
                aoclsparse_int idx_temp = csr_col_ind[k] - base;
                if constexpr(CONJLEFT)
                    C[idx_c] += temp[idx_temp] * csr_val[k];
                else
                    C[idx_c] += temp[idx_temp] * aoclsparse::conj(csr_val[k]);
            }
        }
    }
    return aoclsparse_status_success;
}

// CONJLEFT=false (default): right factor conjugated — computes M*B*M^H.
// CONJLEFT=true:            left factor conjugated  — use when M=A^T to compute A^H*B*A.
template <typename T, bool CONJLEFT = false>
aoclsparse_status aoclsparse_syprd_col_ref(const T *__restrict__ val,
                                           const aoclsparse_int *__restrict__ col_ind,
                                           const aoclsparse_int *__restrict__ row_ptr,
                                           aoclsparse_index_base base,
                                           aoclsparse_int        m,
                                           aoclsparse_int        n,
                                           const T              *B,
                                           aoclsparse_int        ldb,
                                           T                     alpha,
                                           T                     beta,
                                           T                    *C,
                                           aoclsparse_int        ldc)
{
    T                     zero = aoclsparse_numeric::zero<T>();
    const aoclsparse_int *csr_col_ind, *csr_row_ptr;
    const T              *csr_val;
    csr_val     = val - base;
    csr_col_ind = col_ind - base;
    csr_row_ptr = row_ptr;

    if(beta != zero)
    {
        //Apply beta * C for the upper triangular matrix
        for(aoclsparse_int i = 0; i < m; i++)
        {
            for(aoclsparse_int j = i; j < m; ++j)
            {
                aoclsparse_int idx_c = i + j * ldc;
                C[idx_c]             = beta * C[idx_c];
            }
        }
    }
    else
    {
        for(aoclsparse_int i = 0; i < m; i++)
        {
            for(aoclsparse_int j = i; j < m; ++j)
            {
                aoclsparse_int idx_c = i + j * ldc;
                C[idx_c]             = zero;
            }
        }
    }
    if(alpha == zero)
    {
        return aoclsparse_status_success;
    }

    // Perform matrix multiplication.
    // Store the intermediate result in temp and multiply with the sparse matrix in a transposed way.
    std::vector<T> temp;
    try
    {
        temp.resize(n);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    for(aoclsparse_int i = 0; i < m; i++)
    {
        // temp = i-th row of alpha*M*hermitized(B)       for CONJLEFT=false (default)
        // or                 alpha*conj(M)*hermitized(B) for CONJLEFT=true
        aoclsparse_int row_begin = csr_row_ptr[i];
        aoclsparse_int row_end   = csr_row_ptr[i + 1];
        for(aoclsparse_int j = 0; j < n; j++)
        {
            temp[j] = zero;
            for(aoclsparse_int k = row_begin; k < row_end; k++)
            {
                aoclsparse_int colM = csr_col_ind[k] - base;
                T B_val = (j < colM) ? aoclsparse::conj(B[colM * ldb + j]) : B[j * ldb + colM];
                T valM;
                if constexpr(CONJLEFT)
                    valM = aoclsparse::conj(csr_val[k]);
                else
                    valM = csr_val[k];
                temp[j] += valM * B_val;
            }
            temp[j] *= alpha;
        }

        // compute i-th row of upper triangle of C as
        //    temp * M^H  for CONJLEFT=false  or
        //    temp * M^T  for CONJLEFT=true
        for(aoclsparse_int j = i; j < m; j++)
        {
            row_begin            = csr_row_ptr[j];
            row_end              = csr_row_ptr[j + 1];
            aoclsparse_int idx_c = i + j * ldc;
            for(aoclsparse_int k = row_begin; k < row_end; k++)
            {
                aoclsparse_int idx_temp = csr_col_ind[k] - base;
                if constexpr(CONJLEFT)
                    C[idx_c] += temp[idx_temp] * csr_val[k];
                else
                    C[idx_c] += temp[idx_temp] * aoclsparse::conj(csr_val[k]);
            }
        }
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_syprd(aoclsparse_operation            op,
                                   const aoclsparse_matrix         A,
                                   const T                        *B,
                                   aoclsparse_order                orderB,
                                   aoclsparse_int                  ldb,
                                   T                               alpha,
                                   T                               beta,
                                   T                              *C,
                                   aoclsparse_order                orderC,
                                   aoclsparse_int                  ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    if(A == nullptr || A->mats.empty() || B == nullptr || C == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;

    if(orderB != aoclsparse_order_row && orderB != aoclsparse_order_column)
        return aoclsparse_status_invalid_value;

    if(orderC != aoclsparse_order_row && orderC != aoclsparse_order_column)
        return aoclsparse_status_invalid_value;

    if(orderB != orderC)
    {
        return aoclsparse_status_invalid_operation;
    }

    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    aoclsparse_int   m     = A->m;
    aoclsparse_int   k     = A->n;
    aoclsparse::csr *A_csr = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
    if(!A_csr)
        return aoclsparse_status_not_implemented;

    /* Dispatch table — CSR (doid::gn) and CSC (doid::gt, stores A^T internally):
     *   CSR | op_none      : β·C + α·A·B·A^H    (real: A·B·A^T)       conjleft=false
     *   CSR | op_t / op_h  : β·C + α·A^H·B·A    (real: A^T·B·A)       conjleft=true
     *   CSC | op_none      : β·C + α·A^T·B·conj(A)  (real: A^T·B·A);  eff_op=op_t,    conjleft=false
     *   CSC | op_t / op_h  : β·C + α·conj(A)·B·A^T  (real: A·B·A^T);  eff_op=op_none, conjleft=true
     */
    // Accept CSC (doid::gt) and CSR (doid::gn); reject all other formats
    const bool is_doid_gt = (A_csr->doid == aoclsparse::doid::gt);
    if(!is_doid_gt && A_csr->doid != aoclsparse::doid::gn)
        return aoclsparse_status_not_implemented;

    // op_transpose on complex types produces a non-Hermitian result — mathematically invalid.
    // Block for both CSR and CSC; uses original op (format-independent invalidity).
    if(((A->val_type == aoclsparse_cmat) || (A->val_type == aoclsparse_zmat))
       && (op == aoclsparse_operation_transpose))
        return aoclsparse_status_not_implemented;

    aoclsparse_index_base base        = A_csr->base;
    const aoclsparse_int *csr_col_ind = A_csr->ind;
    const aoclsparse_int *csr_row_ptr = A_csr->ptr;
    const T              *csr_val     = static_cast<T *>(A_csr->val);

    T zero = aoclsparse_numeric::zero<T>();
    T one  = aoclsparse_numeric::one<T>();

    // csr_m/csr_n: actual dimensions of the data in A_csr.
    // For CSR (gn): csr_m = m, csr_n = k (user dimensions unchanged).
    // For CSC (gt): A_csr stores A^T, so csr_m = k (user cols), csr_n = m (user rows).
    const aoclsparse_int csr_m = A_csr->m;
    const aoclsparse_int csr_n = A_csr->n;

    // Op-flip for CSC: CSC stores A^T internally.
    // op_none  (user wants A·B·A^H)  → eff_op = op_t  (kernel receives A^T directly, no alloc)
    // op_t/op_h (user wants A^T·B·A) → eff_op = op_none (csr2csc(A^T) = A)
    // CSR:  eff_op = op unchanged.
    aoclsparse_operation eff_op = op;
    if(is_doid_gt)
    {
        if(op == aoclsparse_operation_none)
            eff_op = aoclsparse_operation_transpose;
        else // op_t (real only, already guarded above) or op_h
            eff_op = aoclsparse_operation_none;
    }

    // Check sizes
    if(m < 0 || k < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0)
    {
        return aoclsparse_status_success;
    }

    // Check the rest of pointer arguments
    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(alpha == zero && beta == one)
    {
        return aoclsparse_status_success;
    }

    // ldb/ldc use the original op (user intent on the logical matrix A, not stored A_csr).
    // For CSC op_none: B is m_a × m_a (user A is m×k, result C is m×m), so check_ldb = k, check_ldc = m.
    // This is equivalent to using eff_op = op_t with csr_m/csr_n (csr_m=k for CSC).
    // Using original op preserves consistent user-facing dimension validation.

    // Check leading dimension of B
    aoclsparse_int check_ldb;
    check_ldb = (op == aoclsparse_operation_none ? k : m);
    if(ldb < (((aoclsparse_int)1) >= check_ldb ? (aoclsparse_int)1 : check_ldb))
    {
        return aoclsparse_status_invalid_size;
    }

    // Check leading dimension of C
    aoclsparse_int check_ldc;
    check_ldc = (op == aoclsparse_operation_none ? m : k);
    if(ldc < (((aoclsparse_int)1) >= check_ldc ? (aoclsparse_int)1 : check_ldc))
    {
        return aoclsparse_status_invalid_size;
    }

    // CONJLEFT=true when complex type AND user op is op_conj_trans.
    // This rule is format-independent: CSR op_h and CSC op_h both set conjleft=true.
    // conjleft drives the kernel template instantiation — no heap allocation needed.
    const bool conjleft
        = (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)&&(
            op == aoclsparse_operation_conjugate_transpose);

    // Direct path — A_csr data used as-is (no csr2csc).
    // CSR (gn) op_none: β·C + α·A·B·A^T (real)  /  β·C + α·A·B·A^H (cx)
    // CSC (gt) op_t:    β·C + α·A^T·B·A  (real only; cx op_t blocked above)
    // CSC (gt) op_h:    β·C + α·conj(A)·B·A^T
    if(eff_op == aoclsparse_operation_none)
    {
        if(orderB == aoclsparse_order_column)
        {
            if(conjleft) // β·C + α·conj(A)·B·A^T  [CSC (gt) op_h]
                return aoclsparse_syprd_col_ref<T, true>(csr_val,
                                                         csr_col_ind,
                                                         csr_row_ptr,
                                                         base,
                                                         csr_m,
                                                         csr_n,
                                                         B,
                                                         ldb,
                                                         alpha,
                                                         beta,
                                                         C,
                                                         ldc);
            else // β·C + α·A·B·A^T (real), A·B·A^H (cx)  [CSR (gn) op_none]
                // β·C + α·A^T·B·A                         [CSC (gt) op_t]
                return aoclsparse_syprd_col_ref<T, false>(csr_val,
                                                          csr_col_ind,
                                                          csr_row_ptr,
                                                          base,
                                                          csr_m,
                                                          csr_n,
                                                          B,
                                                          ldb,
                                                          alpha,
                                                          beta,
                                                          C,
                                                          ldc);
        }
        else
        {
            if(conjleft) // β·C + α·conj(A)·B·A^T  [CSC (gt) op_h]
                return aoclsparse_syprd_row_ref<T, true>(csr_val,
                                                         csr_col_ind,
                                                         csr_row_ptr,
                                                         base,
                                                         csr_m,
                                                         csr_n,
                                                         B,
                                                         ldb,
                                                         alpha,
                                                         beta,
                                                         C,
                                                         ldc);
            else // β·C + α·A·B·A^T (real), A·B·A^H (cx)  [CSR (gn) op_none]
                // β·C + α·A^T·B·A                         [CSC (gt) op_t]
                return aoclsparse_syprd_row_ref<T, false>(csr_val,
                                                          csr_col_ind,
                                                          csr_row_ptr,
                                                          base,
                                                          csr_m,
                                                          csr_n,
                                                          B,
                                                          ldb,
                                                          alpha,
                                                          beta,
                                                          C,
                                                          ldc);
        }
    }
    else // eff_op is transpose or conjugate
    {
        std::vector<aoclsparse_int> csr_row_ptr_A;
        std::vector<aoclsparse_int> csr_col_ind_A;
        std::vector<T>              csr_val_A;

        try
        {
            csr_val_A.resize(A->nnz);
            csr_col_ind_A.resize(A->nnz);
            csr_row_ptr_A.resize(csr_n + 1);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        aoclsparse_status status = aoclsparse_csr2csc_template(csr_m,
                                                               csr_n,
                                                               A->nnz,
                                                               base,
                                                               base,
                                                               csr_row_ptr,
                                                               csr_col_ind,
                                                               csr_val,
                                                               csr_col_ind_A.data(),
                                                               csr_row_ptr_A.data(),
                                                               csr_val_A.data());
        if(status != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        // csr2csc path — transposed data used.
        // CSR (gn) op_h: β·C + α·A^H·B·A
        // CSR (gn) op_t: β·C + α·A^T·B·A    (real only; cx op_t blocked above)
        // CSC (gt) op_none: β·C + α·A·B·A^T (real)  /  β·C + α·A^T·B·conj(A) (cx)

        if(orderB == aoclsparse_order_column)
        {
            if(conjleft) // β·C + α·A^H·B·A  [CSR (gn) op_h]
                return aoclsparse_syprd_col_ref<T, true>(csr_val_A.data(),
                                                         csr_col_ind_A.data(),
                                                         csr_row_ptr_A.data(),
                                                         base,
                                                         csr_n,
                                                         csr_m,
                                                         B,
                                                         ldb,
                                                         alpha,
                                                         beta,
                                                         C,
                                                         ldc);
            else // β·C + α·A^T·B·A          [CSR (gn) op_t]
                // β·C + α·A·B·A^T (real),
                // β·C + α·A^T·B·conj(A) (cx)  [CSC (gt) op_none]
                return aoclsparse_syprd_col_ref<T, false>(csr_val_A.data(),
                                                          csr_col_ind_A.data(),
                                                          csr_row_ptr_A.data(),
                                                          base,
                                                          csr_n,
                                                          csr_m,
                                                          B,
                                                          ldb,
                                                          alpha,
                                                          beta,
                                                          C,
                                                          ldc);
        }
        else
        {
            if(conjleft) // β·C + α·A^H·B·A  [CSR (gn) op_h]
                return aoclsparse_syprd_row_ref<T, true>(csr_val_A.data(),
                                                         csr_col_ind_A.data(),
                                                         csr_row_ptr_A.data(),
                                                         base,
                                                         csr_n,
                                                         csr_m,
                                                         B,
                                                         ldb,
                                                         alpha,
                                                         beta,
                                                         C,
                                                         ldc);
            else // β·C + α·A^T·B·A          [CSR (gn) op_t]
                // β·C + α·A·B·A^T (real),
                // β·C + α·A^T·B·conj(A) (cx)  [CSC (gt) op_none]
                return aoclsparse_syprd_row_ref<T, false>(csr_val_A.data(),
                                                          csr_col_ind_A.data(),
                                                          csr_row_ptr_A.data(),
                                                          base,
                                                          csr_n,
                                                          csr_m,
                                                          B,
                                                          ldb,
                                                          alpha,
                                                          beta,
                                                          C,
                                                          ldc);
        }
    }
    return aoclsparse_status_not_implemented;
}

#endif /* AOCLSPARSE_SYPRD_HPP*/
