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
 * ************************************************************************ */
#ifndef AOCLSPARSE_CSRADD_HPP
#define AOCLSPARSE_CSRADD_HPP

#include "aoclsparse_auxiliary.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

aoclsparse_status aoclsparse_add_csr_count_nnz(const aoclsparse_int        M,
                                               const aoclsparse_int        N,
                                               const aoclsparse_index_base base_A,
                                               const aoclsparse_index_base base_B,
                                               aoclsparse_int             &C_nnz,
                                               const aoclsparse_int       *A_row_ptr,
                                               const aoclsparse_int       *A_col_ptr,
                                               const aoclsparse_int       *B_row_ptr,
                                               const aoclsparse_int       *B_col_ptr,
                                               aoclsparse_int             *C_row_ptr)
{
    using namespace aoclsparse;
    aoclsparse_int status = aoclsparse_status_success;

    C_row_ptr[0] = base_A;

#ifdef _OPENMP
#pragma omp parallel num_threads(context::get_context()->get_num_threads()) reduction(max : status)
#endif
    {
#ifdef _OPENMP
        aoclsparse_int num_threads = omp_get_num_threads();
        aoclsparse_int thread_num  = omp_get_thread_num();
        aoclsparse_int lstart      = M * thread_num / num_threads;
        aoclsparse_int lend        = M * (thread_num + 1) / num_threads;
        status                     = aoclsparse_status_success;
#else
        aoclsparse_int lstart = 0;
        aoclsparse_int lend   = M;
#endif
        std::vector<aoclsparse_int> nnz;
        aoclsparse_int              non_zero_count = 0;

        try
        {
            nnz.resize(N + 1, -1);
        }
        catch(std::bad_alloc &)
        {
            status = aoclsparse_status_memory_error;
        }
        if(status == aoclsparse_status_success)
        {
            for(aoclsparse_int i = lstart; i < lend; i++)
            {
                non_zero_count       = 0;
                aoclsparse_int start = A_row_ptr[i] - base_A;
                aoclsparse_int end   = A_row_ptr[i + 1] - base_A;
                for(aoclsparse_int j = start; j < end; j++)
                {
                    aoclsparse_int col_A = A_col_ptr[j];
                    non_zero_count++;
                    nnz[col_A] = i;
                }
                start = B_row_ptr[i] - base_B;
                end   = B_row_ptr[i + 1] - base_B;
                for(aoclsparse_int j = start; j < end; j++)
                {
                    aoclsparse_int col_B = B_col_ptr[j] - base_B + base_A;
                    if(nnz[col_B] != i)
                    {
                        nnz[col_B] = i;
                        non_zero_count++;
                    }
                }
                C_row_ptr[i + 1] = non_zero_count;
            }
        }
    }
    if(status == aoclsparse_status_success)
    {
        int64_t running_sum = base_A;
        for(aoclsparse_int i = 1; i < M + 1; i++)
        {
            running_sum += C_row_ptr[i];
            C_row_ptr[i] = static_cast<aoclsparse_int>(running_sum);
        }

        // Check for overflow: if running_sum exceeds aoclsparse_int range,
        // the prefix sum has overflowed. This check happens after safe 64-bit
        // computation (no Undefined Behavior), and truncated values
        // in C_row_ptr are discarded.
        if(running_sum > aoclsparse_numeric::int_max)
        {
            status = aoclsparse_status_invalid_size;
        }
        else
        {
            C_nnz = C_row_ptr[M] - base_A;
        }
    }
    return (aoclsparse_status)status;
}

template <typename T, bool CONJ_A = false>
aoclsparse_status aoclsparse_add_csr_ref(const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_index_base base_A,
                                         const aoclsparse_index_base base_B,
                                         const aoclsparse_int        A_nnz,
                                         const aoclsparse_int        B_nnz,
                                         aoclsparse_int             &C_nnz,
                                         const aoclsparse_int       *A_row_ptr,
                                         const aoclsparse_int       *A_col_ptr,
                                         const T                    *A_val,
                                         const T                     alpha,
                                         const aoclsparse_int       *B_row_ptr,
                                         const aoclsparse_int       *B_col_ptr,
                                         const T                    *B_val,
                                         aoclsparse::csr           **C)
{
    using namespace aoclsparse;
    aoclsparse_int status = aoclsparse_status_success;

    if(A_row_ptr == nullptr || (A_nnz != 0 && (A_col_ptr == nullptr || A_val == nullptr)))
        return aoclsparse_status_invalid_pointer;

    if(B_row_ptr == nullptr || (B_nnz != 0 && (B_col_ptr == nullptr || B_val == nullptr)))
        return aoclsparse_status_invalid_pointer;

    if(C == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Set pointers to NULL initially
    *C = nullptr;

    // Handle empty matrix case - allocate 0-nnz matrix via constructor
    if(M == 0 || N == 0 || (A_nnz + B_nnz) == 0)
    {
        try
        {
            // Constructor handles row pointer initialization for empty matrix
            *C = new aoclsparse::csr(M, N, 0, aoclsparse_csr_mat, base_A, get_data_type<T>());
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        C_nnz = 0;
        return aoclsparse_status_success;
    }

    aoclsparse_int num_of_threads = context::get_context()->get_num_threads();

    // Compute exact NNZ (and C_row_ptr) in the first stage when:
    // - running multi-threaded (each thread needs exact row boundaries
    //   where to start, i.e., C_row_ptr[i]), or
    // - rough estimate of C_nnz as A_nnz + B_nnz would overflow aoclsparse_int
    // In other cases, nnz can be overestimated and, the exact nnz and
    // C_row_ptr[] is built in the main computation loop.
    bool cptr_computed = (num_of_threads != 1)
                         || (static_cast<uint64_t>(A_nnz) + static_cast<uint64_t>(B_nnz)
                             > static_cast<uint64_t>(aoclsparse_numeric::int_max));

    if(cptr_computed)
    {
        // First allocate C with just the row pointer array
        try
        {
            *C = new aoclsparse::csr(M, N, -1, aoclsparse_csr_mat, base_A, get_data_type<T>());
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }

        // Count the exact nnz in first stage before computation
        if(aoclsparse_add_csr_count_nnz(
               M, N, base_A, base_B, C_nnz, A_row_ptr, A_col_ptr, B_row_ptr, B_col_ptr, (*C)->ptr)
           != aoclsparse_status_success)
        {
            delete *C;
            *C = nullptr;
            return aoclsparse_status_internal_error;
        }

        // Now allocate the column indices and values arrays
        try
        {
            (*C)->ind = new aoclsparse_int[C_nnz];
            (*C)->val = ::operator new(C_nnz * sizeof(T));
            (*C)->nnz = C_nnz;
        }
        catch(std::bad_alloc &)
        {
            delete *C;
            *C = nullptr;
            return aoclsparse_status_memory_error;
        }
    }
    else
    {
        // Single thread, no overflow risk: overestimate nnz and allocate C matrix directly
        C_nnz = A_nnz + B_nnz;
        try
        {
            *C = new aoclsparse::csr(M, N, C_nnz, aoclsparse_csr_mat, base_A, get_data_type<T>());
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        (*C)->ptr[0] = base_A;
    }

    // Get pointers for easier access
    aoclsparse_int *C_row_ptr = (*C)->ptr;
    aoclsparse_int *C_col_ptr = (*C)->ind;
    T              *C_val     = reinterpret_cast<T *>((*C)->val);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_of_threads) reduction(max : status)
#endif
    {
#ifdef _OPENMP
        aoclsparse_int thread_num = omp_get_thread_num();
        aoclsparse_int lstart     = M * thread_num / num_of_threads;
        aoclsparse_int lend       = M * (thread_num + 1) / num_of_threads;
        status                    = aoclsparse_status_success;
#else
        aoclsparse_int lstart = 0;
        aoclsparse_int lend   = M;
#endif
        std::vector<aoclsparse_int> nnz;
        std::vector<aoclsparse_int> col_rec;
        aoclsparse_int              C_idx = 0;

        try
        {
            nnz.resize(N + 1, -1);
            col_rec.resize(N + 1, -1);
        }
        catch(std::bad_alloc &)
        {
            status = aoclsparse_status_memory_error;
        }
        if(status == aoclsparse_status_success)
        {
            for(aoclsparse_int i = lstart; i < lend; i++)
            {
                aoclsparse_int start = A_row_ptr[i] - base_A;
                aoclsparse_int end   = A_row_ptr[i + 1] - base_A;
                if(cptr_computed)
                    C_idx = C_row_ptr[i] - base_A;

                for(aoclsparse_int j = start; j < end; j++)
                {
                    aoclsparse_int col_A = A_col_ptr[j];
                    nnz[col_A]           = i;
                    col_rec[col_A]       = C_idx;
                    C_col_ptr[C_idx]     = col_A;
                    if constexpr(CONJ_A)
                        C_val[C_idx++] = alpha * aoclsparse::conj(A_val[j]);
                    else
                        C_val[C_idx++] = alpha * A_val[j];
                }
                start = B_row_ptr[i] - base_B;
                end   = B_row_ptr[i + 1] - base_B;
                for(aoclsparse_int j = start; j < end; j++)
                {
                    aoclsparse_int col_B = B_col_ptr[j] - base_B + base_A;
                    if(nnz[col_B] != i)
                    {
                        C_col_ptr[C_idx] = col_B;
                        C_val[C_idx++]   = B_val[j];
                        nnz[col_B]       = i;
                    }
                    else
                    {
                        C_val[col_rec[col_B]] += B_val[j];
                    }
                }
                if(!cptr_computed)
                {
                    C_row_ptr[i + 1] = C_idx + base_A;
                }
            }
        }
    }
    if(status == aoclsparse_status_success)
    {
        if(!cptr_computed)
        {
            C_nnz     = C_row_ptr[M] - base_A;
            (*C)->nnz = C_nnz;
        }
    }
    else
    {
        // If anything goes wrong, just destruct C
        delete *C;
        *C = nullptr;
    }

    return (aoclsparse_status)status;
}

template <typename T>
aoclsparse_status aoclsparse_add_t(const aoclsparse_operation op,
                                   const aoclsparse_matrix    A,
                                   const T                    alpha,
                                   const aoclsparse_matrix    B,
                                   aoclsparse_matrix         *C)
{

    if(A == nullptr || B == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(A->input_format != aoclsparse_csr_mat || B->input_format != aoclsparse_csr_mat)
        return aoclsparse_status_not_implemented;

    if(A->val_type != get_data_type<T>() || B->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    if(op == aoclsparse_operation_none)
    {
        if(A->m != B->m || A->n != B->n)
            return aoclsparse_status_invalid_size;
    }
    else
    {
        if(A->m != B->n || A->n != B->m)
            return aoclsparse_status_invalid_size;
    }

    aoclsparse_int C_nnz = 0;

    aoclsparse::csr *raw_A = A->get_first_mtx_if_valid<aoclsparse::csr>();
    aoclsparse::csr *raw_B = B->get_first_mtx_if_valid<aoclsparse::csr>();

    if(!raw_A || !raw_B)
        return aoclsparse_status_not_implemented;
    // Accept gn (CSR) and gt (CSC) for A and B; reject all other formats
    if(raw_A->doid != aoclsparse::doid::gn && raw_A->doid != aoclsparse::doid::gt)
        return aoclsparse_status_not_implemented;
    if(raw_B->doid != aoclsparse::doid::gn && raw_B->doid != aoclsparse::doid::gt)
        return aoclsparse_status_not_implemented;

    const bool is_doid_gt_A = (raw_A->doid == aoclsparse::doid::gt);
    const bool is_doid_gt_B = (raw_B->doid == aoclsparse::doid::gt);

    T *A_val = reinterpret_cast<T *>(raw_A->val);
    T *B_val = reinterpret_cast<T *>(raw_B->val);

    aoclsparse_status status = aoclsparse_status_success;
    aoclsparse::csr  *C_csr  = nullptr;

    // needs_transpose_A: CSC+op_none (undo stored A^T) or CSR+op_trans/conj_trans (compute A^T).
    // needs_transpose_B: B has no op; if stored as CSC (B^T internally), transpose to recover B.
    const bool needs_transpose   = (is_doid_gt_A == (op == aoclsparse_operation_none));
    const bool needs_transpose_B = is_doid_gt_B;

    // Effective output dimensions after optional transpose
    const aoclsparse_int M_eff = needs_transpose ? raw_A->n : raw_A->m;
    const aoclsparse_int N_eff = needs_transpose ? raw_A->m : raw_A->n;

    std::vector<aoclsparse_int> temp_row_ptr, temp_col_ptr;
    std::vector<T>              temp_val;
    const aoclsparse_int       *A_row_ptr = raw_A->ptr;
    const aoclsparse_int       *A_col_ptr = raw_A->ind;
    const T                    *A_val_ptr = A_val;

    if(needs_transpose)
    {
        try
        {
            temp_row_ptr.resize(raw_A->n + 1);
            temp_col_ptr.resize(raw_A->nnz);
            temp_val.resize(raw_A->nnz);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        status = aoclsparse_csr2csc_template(raw_A->m,
                                             raw_A->n,
                                             raw_A->nnz,
                                             raw_A->base,
                                             raw_A->base,
                                             raw_A->ptr,
                                             raw_A->ind,
                                             A_val,
                                             temp_col_ptr.data(),
                                             temp_row_ptr.data(),
                                             temp_val.data());
        if(status != aoclsparse_status_success)
            return status;
        A_row_ptr = temp_row_ptr.data();
        A_col_ptr = temp_col_ptr.data();
        A_val_ptr = temp_val.data();
    }

    std::vector<aoclsparse_int> temp_row_ptr_B, temp_col_ptr_B;
    std::vector<T>              temp_val_B;
    const aoclsparse_int       *B_row_ptr = raw_B->ptr;
    const aoclsparse_int       *B_col_ptr = raw_B->ind;
    const T                    *B_val_ptr = B_val;

    if(needs_transpose_B)
    {
        try
        {
            temp_row_ptr_B.resize(raw_B->n + 1);
            temp_col_ptr_B.resize(raw_B->nnz);
            temp_val_B.resize(raw_B->nnz);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        status = aoclsparse_csr2csc_template(raw_B->m,
                                             raw_B->n,
                                             raw_B->nnz,
                                             raw_B->base,
                                             raw_B->base,
                                             raw_B->ptr,
                                             raw_B->ind,
                                             B_val,
                                             temp_col_ptr_B.data(),
                                             temp_row_ptr_B.data(),
                                             temp_val_B.data());
        if(status != aoclsparse_status_success)
            return status;
        B_row_ptr = temp_row_ptr_B.data();
        B_col_ptr = temp_col_ptr_B.data();
        B_val_ptr = temp_val_B.data();
    }

    // Inline conjugation via CONJ_A template param; aoclsparse::conj is no-op for real types.
    const bool do_conj = (op == aoclsparse_operation_conjugate_transpose);
    if(do_conj)
        status = aoclsparse_add_csr_ref<T, true>(M_eff,
                                                 N_eff,
                                                 raw_A->base,
                                                 raw_B->base,
                                                 A->nnz,
                                                 B->nnz,
                                                 C_nnz,
                                                 A_row_ptr,
                                                 A_col_ptr,
                                                 A_val_ptr,
                                                 alpha,
                                                 B_row_ptr,
                                                 B_col_ptr,
                                                 B_val_ptr,
                                                 &C_csr);
    else
        status = aoclsparse_add_csr_ref<T, false>(M_eff,
                                                  N_eff,
                                                  raw_A->base,
                                                  raw_B->base,
                                                  A->nnz,
                                                  B->nnz,
                                                  C_nnz,
                                                  A_row_ptr,
                                                  A_col_ptr,
                                                  A_val_ptr,
                                                  alpha,
                                                  B_row_ptr,
                                                  B_col_ptr,
                                                  B_val_ptr,
                                                  &C_csr);

    // Only allocate the main matrix C at the end to avoid returning partially filled matrix
    if(status == aoclsparse_status_success)
    {
        try
        {
            *C = new _aoclsparse_matrix;
            (*C)->mats.push_back(C_csr);
        }
        catch(std::bad_alloc &)
        {
            if(C_csr)
            {
                delete C_csr;
            }
            if(*C)
            {
                delete *C;
                *C = nullptr;
            }
            return aoclsparse_status_memory_error;
        }
        aoclsparse_init_mat(*C, B->m, B->n, C_nnz, aoclsparse_csr_mat);
        (*C)->val_type = get_data_type<T>();
        return aoclsparse_status_success;
    }
    else
    {
        // C_csr is already cleaned up in aoclsparse_add_csr_ref on failure
        return status;
    }
}

#endif // AOCLSPARSE_CSRADD_HPP
