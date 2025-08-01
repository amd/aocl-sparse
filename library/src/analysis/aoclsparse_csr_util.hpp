/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_INPUT_CHECK_HPP
#define AOCLSPARSE_INPUT_CHECK_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_types.h"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

/* mark if the matrix has a specific shape, e.g., only the lower triangle values are stored
 * Note: this is different than matrix type + fill mode since in the inspector/ executor mode,
 * the descriptor can be used to signify the function to only work with part of the matrices
 */
enum aoclsparse_shape
{
    shape_general = 0,
    shape_lower_triangle,
    shape_upper_triangle,
};

aoclsparse_status aoclsparse_mat_check_internal(aoclsparse_int          maj_dim,
                                                aoclsparse_int          min_dim,
                                                aoclsparse_int          nnz,
                                                const aoclsparse_int   *idx_ptr,
                                                const aoclsparse_int   *indices,
                                                const void             *val,
                                                aoclsparse_shape        shape,
                                                aoclsparse_index_base   base,
                                                aoclsparse_matrix_sort &mat_sort,
                                                bool                   &mat_fulldiag,
                                                void (*error_handler)(aoclsparse_status status,
                                                                      std::string       message),
                                                bool fast_chck = false);

aoclsparse_status aoclsparse_csr_csc_check_sort_diag(aoclsparse_int        m,
                                                     aoclsparse_int        n,
                                                     aoclsparse_index_base base,
                                                     const aoclsparse_int *idx_ptr,
                                                     const aoclsparse_int *indices,
                                                     bool                 &sorted,
                                                     bool                 &fulldiag);
aoclsparse_status aoclsparse_csr_csc_indices(aoclsparse_int        m,
                                             aoclsparse_index_base base,
                                             const aoclsparse_int *icrow,
                                             const aoclsparse_int *icol,
                                             aoclsparse_int      **idiag,
                                             aoclsparse_int      **iurow);

/* Function to sort CSR or CSC matrix.
 * Input parameters :-
 * maj_dim            : major dimension - row(m) for CSR, col(n) for CSC
 * min_dim            : minor dimension - col(n) for CSR, row(m) for CSR
 * nnz                : non-zero count
 * src_base/dest_base : 0-base or 1-base
 * src_idx_ptr        : csr_row_ptr from aoclsparse::csr or col_ptr from aoclsparse::csc
 * src_idx/dest_idx   : csr_col_ptr from aoclsparse::csr or row_ind from aoclsparse::csc
 * src_val/dest/val   : csr_val from aoclsparse::csr or val from aoclsparse::csc
 *
 * Possible exit: memory alloc, invalid pointer
 *
 * Possible cases:
 * 1) src_base:0, dest_base:0 => No additional handling
 * 2) src_base:0, dest_base:1 => not possible
 * 3) src_base:1, dest_base:0 => optimize_csr does base correction in dest copy in copy_csr.
 *      Handle perserving destination base while copying src_idx to dest_idx in sorting logic.
 * 4) src_base:1, dest_base:1 => No additional handling
 */
template <typename T>
aoclsparse_status aoclsparse_sort_idx_val(aoclsparse_int                  maj_dim,
                                          [[maybe_unused]] aoclsparse_int min_dim,
                                          aoclsparse_int                  nnz,
                                          aoclsparse_index_base           src_base,
                                          aoclsparse_int                 *src_idx_ptr,
                                          aoclsparse_int                 *src_idx,
                                          T                              *src_val,
                                          aoclsparse_index_base           dest_base,
                                          aoclsparse_int                 *dest_idx,
                                          T                              *dest_val)
{
    aoclsparse_int i, j, idx, nnzrow;

    if(maj_dim == 0 || nnz == 0)
        return aoclsparse_status_success;

    if((src_idx_ptr == nullptr) || (src_idx == nullptr) || (src_val == nullptr)
       || (dest_idx == nullptr) || (dest_val == nullptr))
        return aoclsparse_status_invalid_pointer;

    std::vector<aoclsparse_int> perm;
    try
    {
        perm.resize(nnz);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    for(j = 0; j < nnz; j++)
        perm[j] = j;

    for(i = 0; i < maj_dim; i++)
    {
        // sort each row according to its column indices
        //TODO: In case input base is one-base, base-index for matrix "A" should be A->base
        //      and inner loop lines which overwrite csr_col_ptr/aval_s (of matrix "As")
        //      should have base correction since RHS is input matrix "A" that is of one-base.
        //      Add unit test to catch this error, assuming the following input conditions:
        //      input base index of "A" - one-base, and input to this function "aoclsparse_sort_csr"
        //      is A->internal_base_index (which is zero-base after copy_csr correction). So logically,
        //      we should observe a failure with these conditions since As's csr_col_ptr/aval_s would not
        //      be base-corrected since A->internal_base_index is zero-base.
        idx    = src_idx_ptr[i] - src_base;
        nnzrow = src_idx_ptr[i + 1] - src_base - idx;
        std::sort(std::begin(perm) + idx,
                  std::begin(perm) + idx + nnzrow,
                  [&](const aoclsparse_int &a, const aoclsparse_int &b) {
                      return src_idx[a] <= src_idx[b];
                  });
        for(j = idx; j < idx + nnzrow; j++)
        {
            dest_idx[j] = src_idx[perm[j]] - static_cast<aoclsparse_int>(src_base != dest_base);
            dest_val[j] = src_val[perm[j]];
        }
    }

    return aoclsparse_status_success;
}

/* create some artificial fill-ins with zeros on the diagonal if some elements
 * are missing, in such a case the arrays are reallocated.
 * Assumes the rows are sorted */
template <typename T>
aoclsparse_status aoclsparse_csr_csc_fill_diag(aoclsparse_int        m,
                                               aoclsparse_int        n,
                                               aoclsparse_int        nnz,
                                               aoclsparse_index_base base,
                                               aoclsparse_int      **dst_idx_ptr,
                                               aoclsparse_int      **dst_indices,
                                               T                   **dst_val)
{
    aoclsparse_int i, j, count, idx, idxend;

    //validate the input double pointers first before dereferencing them later in try block
    if(!dst_indices || !dst_idx_ptr || !dst_val)
        return aoclsparse_status_invalid_pointer;

    try
    {
        std::vector<aoclsparse_int> missing_diag(m, -1);

        if(!(*dst_indices) || !(*dst_idx_ptr) || !(*dst_val))
            return aoclsparse_status_invalid_pointer;

        // Check each row for missing element on the diagonal
        count = 0;
        bool diag_found;
        for(i = 0; i < m; i++)
        {
            diag_found = false;
            idxend     = (*dst_idx_ptr)[i + 1] - base;
            for(idx = ((*dst_idx_ptr)[i] - base); idx < idxend; idx++)
            {
                j = (*dst_indices)[idx] - base;
                if(i == j)
                {
                    diag_found = true;
                    break;
                }
                if(j > i)
                    break;
            }
            if(!diag_found && i < n)
            {
                missing_diag[i] = idx + count;
                count++;
            }
        }

        if(count <= 0)
        {
            return aoclsparse_status_success;
        }
        aoclsparse_int *icol = nullptr, *icrow = nullptr;
        void           *aval    = nullptr;
        T              *csr_val = nullptr;
        aoclsparse_int  nnz_new = nnz + count;
        csr_val                 = static_cast<T *>(*dst_val);

        try
        {
            icrow = new aoclsparse_int[m + 1];
            icol  = new aoclsparse_int[nnz_new];
            aval  = ::operator new(sizeof(T) * nnz_new);
        }
        catch(std::bad_alloc &)
        {
            delete[] icrow;
            delete[] icol;
            ::operator delete(aval);
            return aoclsparse_status_memory_error;
        }
        aoclsparse_int n_added = 0, nnz_curr = 0;
        for(i = 0; i < m; i++)
        {
            idxend   = (*dst_idx_ptr)[i + 1] - base;
            icrow[i] = (*dst_idx_ptr)[i] - base + n_added;
            // Copy the row into the new matrix
            for(idx = ((*dst_idx_ptr)[i] - base); idx < idxend; idx++)
            {
                if(nnz_curr == missing_diag[i])
                {
                    // add the missing diagonal at the correct place
                    n_added++;
                    icol[nnz_curr]                   = i;
                    static_cast<T *>(aval)[nnz_curr] = aoclsparse_numeric::zero<T>();
                    nnz_curr++;
                }
                static_cast<T *>(aval)[nnz_curr] = csr_val[idx];
                icol[nnz_curr]                   = (*dst_indices)[idx] - base;
                nnz_curr++;
            }
            if(nnz_curr == missing_diag[i])
            {
                // In empty rows cases, need to add the diagonal
                n_added++;
                icol[nnz_curr]                   = i;
                static_cast<T *>(aval)[nnz_curr] = aoclsparse_numeric::zero<T>();
                nnz_curr++;
            }
        }
        icrow[m] = nnz_new;

        // replace A vectors by the new filled ones
        delete[](*dst_indices);
        delete[](*dst_idx_ptr);
        ::operator delete((*dst_val));
        (*dst_indices) = icol;
        (*dst_idx_ptr) = icrow;
        (*dst_val)     = static_cast<T *>(aval);
    }
    catch(std::bad_alloc &)
    {
        // missing_diag allocation failure
        return aoclsparse_status_memory_error;
    }

    return aoclsparse_status_success;
}

// Creates matrix copies of CSR
template <typename T>
aoclsparse_status aoclsparse_matrix_transform(aoclsparse_matrix A)
{
    if(!A || A->mats.empty())
        return aoclsparse_status_invalid_pointer;

    aoclsparse_status status = aoclsparse_status_success;

    // If the input matrix is in CSR format
    if(A->input_format == aoclsparse_csr_mat)
    {
        aoclsparse_optimize_data *optd    = A->optim_data;
        aoclsparse::csr          *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
        if(!csr_mat)
            return aoclsparse_status_not_implemented;
        if(csr_mat->mat_type != A->input_format || !csr_mat->ptr || !csr_mat->ind || !csr_mat->val)
            return aoclsparse_status_invalid_pointer;
        while(optd)
        {
            //if(optd->act == aoclsparse_action_mv)
            {
                aoclsparse::doid doid = optd->doid;

                // Check if the matrix copy already exists
                bool found_mat = false;
                for(size_t i = 1; i < A->mats.size(); i++)
                {
                    if(A->mats[i] && A->mats[i]->doid == doid)
                    {
                        found_mat = true;
                        break;
                    }
                }

                // Generate matrix copy
                if(!found_mat)
                {
                    switch(doid)
                    {
                    case aoclsparse::doid::gt:
                    {
                        // TODO: Create a CSC mat copy when CSC support for mv is implemented
                        aoclsparse::csr *mat_copy = nullptr;
                        try
                        {
                            // Create a matrix copy
                            // Interchanged m, n dimensions, for the csc conversion
                            mat_copy = new aoclsparse::csr(A->n,
                                                           A->m,
                                                           A->nnz,
                                                           aoclsparse_csr_mat,
                                                           aoclsparse_index_base_zero,
                                                           csr_mat->val_type);
                        }
                        catch(std::bad_alloc &)
                        {
                            return aoclsparse_status_memory_error;
                        }
                        // convert to 0-base
                        status = aoclsparse_csr2csc_template(A->m,
                                                             A->n,
                                                             A->nnz,
                                                             A->base,
                                                             aoclsparse_index_base_zero,
                                                             csr_mat->ptr,
                                                             csr_mat->ind,
                                                             (T *)csr_mat->val,
                                                             mat_copy->ind,
                                                             mat_copy->ptr,
                                                             (T *)mat_copy->val);

                        if(status != aoclsparse_status_success)
                        {
                            delete mat_copy;
                            return status;
                        }
                        try
                        {
                            A->mats.push_back(mat_copy);
                        }
                        catch(std::bad_alloc &)
                        {
                            delete mat_copy;
                            return aoclsparse_status_memory_error;
                        }

                        mat_copy->doid = doid;
                        break;
                    }
                    default:
                        break;
                    }
                }
            }
            optd = optd->next;
        }
    }
    return status;
}

/* Given input matrix in CSR/CSC format, check it and create the matching
 * clean version opt_csr_mat/opt_csc_mat, respectively */
template <typename T>
aoclsparse_status aoclsparse_csr_csc_optimize(aoclsparse_matrix A, aoclsparse::csr **opt_csr_mat)
{
    aoclsparse_status status;

    if(!A || A->mats.empty() || !opt_csr_mat)
        return aoclsparse_status_invalid_pointer;

    // Make sure we have the right type before proceeding
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    //Make sure base-index is the correct value
    if(A->base != aoclsparse_index_base_zero && A->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    // Stores optimized csr ptr
    *opt_csr_mat = nullptr;

    // Get first matrix from A->mats
    aoclsparse::csr *src_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
    if(!src_mat)
        return aoclsparse_status_not_implemented;
    if(!src_mat->ptr || !src_mat->ind || !src_mat->val)
        return aoclsparse_status_invalid_pointer;
    // Quick exit if the matrix is already optimized
    if(src_mat->is_optimized)
    {
        *opt_csr_mat = src_mat;
        return aoclsparse_status_success;
    }

    // Check if the optimized matrix is already in A->mats
    for(size_t i = 1; i < A->mats.size(); i++)
    {
        aoclsparse::csr *temp_opt_mat = dynamic_cast<aoclsparse::csr *>(A->mats[i]);
        if(temp_opt_mat && temp_opt_mat->is_optimized)
        {
            *opt_csr_mat = temp_opt_mat;
            return aoclsparse_status_success;
        }
    }

    T             *src_val = static_cast<T *>(src_mat->val);
    aoclsparse_int m_mat   = (A->input_format == aoclsparse_csr_mat) ? A->m : A->n;
    aoclsparse_int n_mat   = (A->input_format == aoclsparse_csr_mat) ? A->n : A->m;

    // Check the user's matrix format
    // First check the matrix is a valid matrix
    status = aoclsparse_mat_check_internal(m_mat,
                                           n_mat,
                                           A->nnz,
                                           src_mat->ptr,
                                           src_mat->ind,
                                           src_val,
                                           shape_general,
                                           A->base,
                                           A->sort,
                                           A->fulldiag,
                                           nullptr);
    if(status != aoclsparse_status_success)
    {
        return status;
    }

    // Check if the matrix is sorted with full diag
    bool sorted, fulldiag;
    status = aoclsparse_csr_csc_check_sort_diag(
        m_mat, n_mat, A->base, src_mat->ptr, src_mat->ind, sorted, fulldiag);
    if(status != aoclsparse_status_success)
    {
        // Shouldn't happen, pointers have already been checked
        return aoclsparse_status_internal_error;
    }

    if(sorted && fulldiag)
    {
        // The matrix is already in the correct format, use directly user's memory
        // Set idiag and iurow pointers
        status = aoclsparse_csr_csc_indices(m_mat,
                                            A->internal_base_index,
                                            src_mat->ptr,
                                            src_mat->ind,
                                            &(src_mat->idiag),
                                            &(src_mat->iurow));
        if(status != aoclsparse_status_success)
        {
            return status;
        }
        src_mat->is_optimized = true;
        *opt_csr_mat          = src_mat;
    }
    else
    {
        // Optimized matrix
        aoclsparse::csr *opt_mat = nullptr;
        try
        {
            opt_mat = new aoclsparse::csr(m_mat,
                                          n_mat,
                                          src_mat->nnz,
                                          src_mat->mat_type,
                                          aoclsparse_index_base_zero,
                                          src_mat->val_type);
        }
        catch(const std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }

        T *opt_val = static_cast<T *>(opt_mat->val);
        // copy the matrix
        for(aoclsparse_int i = 0; i < m_mat + 1; i++)
        {
            opt_mat->ptr[i] = src_mat->ptr[i] - src_mat->base;
        }
        for(aoclsparse_int i = 0; i < src_mat->nnz; i++)
        {
            opt_mat->ind[i] = src_mat->ind[i] - src_mat->base;
            opt_val[i]      = src_val[i];
        }

        //since the correction is already performed during above copy, the execution kernel and the
        //subsequent calls to sort, diagonal fill and idiag/iurow compute can
        //treat storage buffers in opt_csr_mat as zero-based indexing and need not perform
        //double correction
        A->internal_base_index = aoclsparse_index_base_zero;

        if(!sorted)
        {
            aoclsparse_sort_idx_val<T>(m_mat,
                                       n_mat,
                                       A->nnz,
                                       A->base,
                                       src_mat->ptr,
                                       src_mat->ind,
                                       src_val,
                                       A->internal_base_index,
                                       opt_mat->ind,
                                       opt_val);
            // check again for full diagonal
            status = aoclsparse_csr_csc_check_sort_diag(
                m_mat, n_mat, A->internal_base_index, opt_mat->ptr, opt_mat->ind, sorted, fulldiag);
            if(status != aoclsparse_status_success)
            {
                delete opt_mat;
                return status;
            }
        }
        if(!fulldiag)
        {
            status = aoclsparse_csr_csc_fill_diag<T>(m_mat,
                                                     n_mat,
                                                     A->nnz,
                                                     A->internal_base_index,
                                                     &(opt_mat->ptr),
                                                     &(opt_mat->ind),
                                                     reinterpret_cast<T **>(&(opt_mat->val)));
            if(status != aoclsparse_status_success)
            {
                delete opt_mat;
                return status;
            }
        }

        // mat_idiag and mat_iurow ptrs are set
        status = aoclsparse_csr_csc_indices(m_mat,
                                            A->internal_base_index,
                                            opt_mat->ptr,
                                            opt_mat->ind,
                                            &(opt_mat->idiag),
                                            &(opt_mat->iurow));
        if(status != aoclsparse_status_success)
        {
            delete opt_mat;
            return status;
        }
        // TODO: The dimensions of opt_mat should remain unchanged.
        opt_mat->m            = A->m;
        opt_mat->n            = A->n;
        opt_mat->is_optimized = true;
        opt_mat->nnz          = opt_mat->ptr[m_mat];

        try
        {
            A->mats.push_back(opt_mat);
        }
        catch(const std::bad_alloc &)
        {
            delete opt_mat;
            return aoclsparse_status_memory_error;
        }
        *opt_csr_mat = opt_mat;
    }
    //being full-diagonal is property of the original matrix. So need to
    A->opt_csr_full_diag = fulldiag;
    A->optimized         = true;

    return aoclsparse_status_success;
}

// Check TCSR matrix inputs and create idiag for the lower and iurow for
// the upper triangular matrix
template <typename T>
aoclsparse_status aoclsparse_tcsr_optimize(aoclsparse_matrix A, aoclsparse::tcsr **opt_tcsr_mat)
{
    if(!A || A->mats.empty() || !opt_tcsr_mat)
        return aoclsparse_status_invalid_pointer;
    // Check if the user input matrix is in TCSR format
    aoclsparse::tcsr *tcsr_mat = dynamic_cast<aoclsparse::tcsr *>(A->mats[0]);
    if(!tcsr_mat)
    {
        return aoclsparse_status_not_implemented;
    }
    if(!tcsr_mat->row_ptr_L || !tcsr_mat->row_ptr_U)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(tcsr_mat->is_optimized)
    {
        *opt_tcsr_mat = tcsr_mat;
        return aoclsparse_status_success;
    }
    // Make sure we have the right type before proceeding
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    // Create idiag and iurow
    try
    {
        tcsr_mat->idiag = new aoclsparse_int[A->m];
        tcsr_mat->iurow = new aoclsparse_int[A->m];
    }
    catch(std::bad_alloc &)
    {
        delete[] tcsr_mat->idiag;
        tcsr_mat->idiag = nullptr;
        delete[] tcsr_mat->iurow;
        tcsr_mat->iurow = nullptr;
        return aoclsparse_status_memory_error;
    }

    for(aoclsparse_int i = 0; i < A->m; i++)
    {
        // Diagonal is at the end of the each row in the lower triangular part
        tcsr_mat->idiag[i] = tcsr_mat->row_ptr_L[i + 1] - 1;
        // Diagonal is at the beginning of each row in the upper triangular part
        // Increment row_ptr_U to get the position of upper triangle element
        tcsr_mat->iurow[i] = tcsr_mat->row_ptr_U[i] + 1;
    }
    *opt_tcsr_mat          = tcsr_mat;
    tcsr_mat->is_optimized = true;
    A->opt_csr_full_diag   = A->fulldiag;
    return aoclsparse_status_success;
}

#endif // AOCLSPARSE_INPUT_CHECK_HPP
