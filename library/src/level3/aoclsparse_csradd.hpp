/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
#include <aoclsparse_mat_structures.h>
#include <cstring>
#include <vector>

template <typename T>
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
                                         aoclsparse_int            *&C_row_ptr,
                                         aoclsparse_int            *&C_col_ptr,
                                         T                         *&C_val)
{
    if(A_row_ptr == nullptr || (A_nnz != 0 && (A_col_ptr == nullptr || A_val == nullptr)))
        return aoclsparse_status_invalid_pointer;

    if(B_row_ptr == nullptr || (B_nnz != 0 && (B_col_ptr == nullptr || B_val == nullptr)))
        return aoclsparse_status_invalid_pointer;

    C_nnz = 0;

    try
    {
        C_row_ptr = new aoclsparse_int[M + 1];
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    if(M == 0 || N == 0 || (A_nnz + B_nnz) == 0)
    {
        for(int i = 0; i < M + 1; i++)
        {
            C_row_ptr[i] = base_A;
        }
        try
        {
            C_col_ptr = new aoclsparse_int[0];
            C_val     = static_cast<T *>(::operator new(sizeof(T)));
        }
        catch(std::bad_alloc &)
        {

            delete[] C_row_ptr;
            delete[] C_col_ptr;
            ::operator delete(C_val);
            return aoclsparse_status_memory_error;
        }
        return aoclsparse_status_success;
    }
    std::vector<aoclsparse_int> nnz;
    std::vector<aoclsparse_int> col_rec;

    try
    {
        nnz.resize(N + 1, -1);
        col_rec.resize(N + 1, -1);
        C_col_ptr = new aoclsparse_int[A_nnz + B_nnz];
        C_val     = static_cast<T *>(::operator new((A_nnz + B_nnz) * sizeof(T)));
    }
    catch(std::bad_alloc &)
    {
        delete[] C_col_ptr;
        delete[] C_row_ptr;
        ::operator delete(C_val);
        return aoclsparse_status_memory_error;
    }
    C_row_ptr[0] = base_A;
    for(aoclsparse_int i = 0; i < M; i++)
    {
        aoclsparse_int start = A_row_ptr[i] - base_A;
        aoclsparse_int end   = A_row_ptr[i + 1] - base_A;
        for(aoclsparse_int j = start; j < end; j++)
        {
            aoclsparse_int col_A = A_col_ptr[j];
            nnz[col_A]           = i;
            col_rec[col_A]       = C_nnz;
            C_col_ptr[C_nnz]     = col_A;
            C_val[C_nnz++]       = alpha * A_val[j];
        }
        start = B_row_ptr[i] - base_B;
        end   = B_row_ptr[i + 1] - base_B;
        for(aoclsparse_int j = start; j < end; j++)
        {
            aoclsparse_int col_B = B_col_ptr[j] - base_B + base_A;
            if(nnz[col_B] != i)
            {
                C_col_ptr[C_nnz] = col_B;
                C_val[C_nnz++]   = B_val[j];
                nnz[col_B]       = i;
            }
            else
            {
                C_val[col_rec[col_B]] += B_val[j];
            }
        }
        C_row_ptr[i + 1] = C_nnz + base_A;
    }
    return aoclsparse_status_success;
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

    aoclsparse_int   *C_row_ptr = nullptr, *C_col_ptr = nullptr, C_nnz = 0;
    T                *C_val  = nullptr;
    T                *A_val  = reinterpret_cast<T *>(A->csr_mat.csr_val);
    T                *B_val  = reinterpret_cast<T *>(B->csr_mat.csr_val);
    aoclsparse_status status = aoclsparse_status_success;

    if(op == aoclsparse_operation_none)
    {
        if((status = aoclsparse_add_csr_ref(A->m,
                                            A->n,
                                            A->base,
                                            B->base,
                                            A->nnz,
                                            B->nnz,
                                            C_nnz,
                                            A->csr_mat.csr_row_ptr,
                                            A->csr_mat.csr_col_ptr,
                                            A_val,
                                            alpha,
                                            B->csr_mat.csr_row_ptr,
                                            B->csr_mat.csr_col_ptr,
                                            B_val,
                                            C_row_ptr,
                                            C_col_ptr,
                                            C_val))
           != aoclsparse_status_success)
            return status;
    }
    else
    {
        std::vector<aoclsparse_int> temp_row_ptr, temp_col_ptr;
        std::vector<T>              temp_val;
        try
        {
            temp_row_ptr.resize(A->n + 1);
            temp_col_ptr.resize(A->nnz);
            temp_val.resize(A->nnz);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        if((status = aoclsparse_csr2csc_template(A->m,
                                                 A->n,
                                                 A->nnz,
                                                 A->base,
                                                 A->base,
                                                 A->csr_mat.csr_row_ptr,
                                                 A->csr_mat.csr_col_ptr,
                                                 A_val,
                                                 temp_col_ptr.data(),
                                                 temp_row_ptr.data(),
                                                 temp_val.data()))
           != aoclsparse_status_success)
            return status;
        if constexpr(std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, std::complex<double>>)
        {
            if(op == aoclsparse_operation_conjugate_transpose)
            {
                // transpose is done, now conjugate
                for(aoclsparse_int i = 0; i < A->nnz; i++)
                    temp_val[i] = std::conj(temp_val[i]);
            }
        }
        if((status = aoclsparse_add_csr_ref(A->n,
                                            A->m,
                                            A->base,
                                            B->base,
                                            A->nnz,
                                            B->nnz,
                                            C_nnz,
                                            temp_row_ptr.data(),
                                            temp_col_ptr.data(),
                                            temp_val.data(),
                                            alpha,
                                            B->csr_mat.csr_row_ptr,
                                            B->csr_mat.csr_col_ptr,
                                            B_val,
                                            C_row_ptr,
                                            C_col_ptr,
                                            C_val))
           != aoclsparse_status_success)
            return status;
    }
    try
    {
        *C = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    aoclsparse_init_mat(*C, A->base, B->m, B->n, C_nnz, aoclsparse_csr_mat);
    (*C)->val_type            = get_data_type<T>();
    (*C)->csr_mat.csr_row_ptr = C_row_ptr;
    (*C)->csr_mat.csr_col_ptr = C_col_ptr;
    (*C)->csr_mat.csr_val     = C_val;
    (*C)->csr_mat_is_users    = false;
    return aoclsparse_status_success;
}
