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

#ifndef AOCLSPARSE_SYPRD_HPP
#define AOCLSPARSE_SYPRD_HPP
#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <immintrin.h>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

template <typename T>
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
    T                     temp[n];
    T                     zero = 0.0;
    const aoclsparse_int *csr_col_ind, *csr_row_ptr;
    const T              *csr_val;
    csr_val     = val - base;
    csr_col_ind = col_ind - base;
    csr_row_ptr = row_ptr;

    //Apply beta * C for the upper triangular matrix and return success
    if(alpha == zero)
    {
        for(aoclsparse_int i = 0; i < m; ++i)
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

        // Perform matrix multiplication.
        // Store the intermediate result in temp and multiply with the sparse matrix in a transposed way.
        // This logic can be applied to the syprd_col_ref as well.
        for(aoclsparse_int i = 0; i < m; i++)
        {
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            for(aoclsparse_int j = 0; j < n; j++)
                temp[j] = zero;
            for(aoclsparse_int k = row_begin; k < row_end; k++)
            {
                for(aoclsparse_int j = 0; j < csr_col_ind[k] - base; j++)
                {
                    temp[j] += alpha * aoclsparse::conj(B[j * ldb + csr_col_ind[k] - base])
                               * csr_val[k];
                }
                for(aoclsparse_int j = csr_col_ind[k] - base; j < n; j++)
                {
                    temp[j] += alpha * B[(csr_col_ind[k] - base) * ldb + j] * csr_val[k];
                }
            }

            for(aoclsparse_int j = i; j < m; j++)
            {
                row_begin            = csr_row_ptr[j];
                row_end              = csr_row_ptr[j + 1];
                aoclsparse_int idx_c = i * ldc + j;
                for(aoclsparse_int k = row_begin; k < row_end; k++)
                {
                    aoclsparse_int idx_temp = csr_col_ind[k] - base;
                    C[idx_c] += temp[idx_temp] * aoclsparse::conj(csr_val[k]);
                }
            }
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
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
    T                     temp[n];
    T                     zero = 0.0;
    const aoclsparse_int *csr_col_ind, *csr_row_ptr;
    const T              *csr_val;
    csr_val     = val - base;
    csr_col_ind = col_ind - base;
    csr_row_ptr = row_ptr;

    //Apply beta * C for the upper triangular matrix and return success
    if(alpha == zero)
    {
        for(aoclsparse_int i = 0; i < m; ++i)
        {
            for(aoclsparse_int j = i; j < m; ++j)
            {
                aoclsparse_int idx_c;
                idx_c    = i + j * ldc;
                C[idx_c] = beta * C[idx_c];
            }
        }
    }
    else
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

        // Perform matrix multiplication.
        // Store the intermediate result in temp and multiply with the sparse matrix in a transposed way.
        for(aoclsparse_int i = 0; i < m; i++)
        {
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            for(aoclsparse_int j = 0; j < n; j++)
            {
                temp[j] = zero;
                for(aoclsparse_int k = row_begin; k < row_end; k++)
                {
                    T B_val = (j < (csr_col_ind[k] - base))
                                  ? aoclsparse::conj(B[(csr_col_ind[k] - base) * ldb + j])
                                  : B[j * ldb + csr_col_ind[k] - base];
                    temp[j] += (alpha * (csr_val[k]) * B_val);
                }
            }
            for(aoclsparse_int j = 0; j < m; j++)
            {
                row_begin            = csr_row_ptr[j];
                row_end              = csr_row_ptr[j + 1];
                aoclsparse_int idx_c = i + j * ldc;
                for(aoclsparse_int k = row_begin; k < row_end && (i <= j); k++)
                {
                    aoclsparse_int idx_temp = csr_col_ind[k] - base;
                    C[idx_c] += temp[idx_temp] * aoclsparse::conj(csr_val[k]);
                }
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
    aoclsparse_int   m     = A->m;
    aoclsparse_int   k     = A->n;
    aoclsparse::csr *A_csr = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
    if(!A_csr)
        return aoclsparse_status_not_implemented;
    aoclsparse_index_base base        = A_csr->base;
    const aoclsparse_int *csr_col_ind = A_csr->ind;
    const aoclsparse_int *csr_row_ptr = A_csr->ptr;
    const T              *csr_val     = static_cast<T *>(A_csr->val);

    T zero = 0.0;
    T one  = 1.0;

    // The operation is invalid if matrix B and C has different layout ordering
    if(orderB != orderC)
    {
        return aoclsparse_status_invalid_operation;
    }

    // Verify the matrix types and T are consistent
    if(!((A->val_type == aoclsparse_smat && std::is_same_v<T, float>)
         || (A->val_type == aoclsparse_dmat && std::is_same_v<T, double>)
         || (A->val_type == aoclsparse_cmat && std::is_same_v<T, std::complex<float>>)
         || (A->val_type == aoclsparse_zmat && std::is_same_v<T, std::complex<double>>)))
        return aoclsparse_status_wrong_type;

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

    // Bool variable to simplify complex if statements
    const bool cmplx_type
        = (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>);
    const bool op_trans_or_conj
        = (op == aoclsparse_operation_conjugate_transpose || op == aoclsparse_operation_transpose);

    if(op == aoclsparse_operation_none)
    {
        if(orderB == aoclsparse_order_column)
        {
            return aoclsparse_syprd_col_ref(
                csr_val, csr_col_ind, csr_row_ptr, base, m, k, B, ldb, alpha, beta, C, ldc);
        }
        else
        {
            return aoclsparse_syprd_row_ref(
                csr_val, csr_col_ind, csr_row_ptr, base, m, k, B, ldb, alpha, beta, C, ldc);
        }
    }
    else if(op_trans_or_conj)
    {
        std::vector<aoclsparse_int> csr_row_ptr_A;
        std::vector<aoclsparse_int> csr_col_ind_A;
        std::vector<T>              csr_val_A;

        csr_val_A.resize(A->nnz);
        csr_col_ind_A.resize(A->nnz);
        csr_row_ptr_A.resize(A->n + 1);
        aoclsparse_status status = aoclsparse_csr2csc_template(A->m,
                                                               A->n,
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

        if((op == aoclsparse_operation_conjugate_transpose) && cmplx_type)
        {
            for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
                csr_val_A[idx] = aoclsparse::conj(csr_val_A[idx]);
        }

        if(orderB == aoclsparse_order_column)
        {
            return aoclsparse_syprd_col_ref(csr_val_A.data(),
                                            csr_col_ind_A.data(),
                                            csr_row_ptr_A.data(),
                                            base,
                                            k,
                                            m,
                                            B,
                                            ldb,
                                            alpha,
                                            beta,
                                            C,
                                            ldc);
        }
        else
        {
            return aoclsparse_syprd_row_ref(csr_val_A.data(),
                                            csr_col_ind_A.data(),
                                            csr_row_ptr_A.data(),
                                            base,
                                            k,
                                            m,
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
