/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_types.h"
#include "aoclsparse_auxiliary.hpp"

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

aoclsparse_status aoclsparse_mat_check_internal(aoclsparse_int        maj_dim,
                                                aoclsparse_int        min_dim,
                                                aoclsparse_int        nnz,
                                                const aoclsparse_int *idx_ptr,
                                                const aoclsparse_int *indices,
                                                const void           *val,
                                                aoclsparse_shape      shape,
                                                aoclsparse_index_base base,
                                                void (*error_handler)(aoclsparse_status status,
                                                                      std::string       message));
aoclsparse_status aoclsparse_csr_check_sort_diag(aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 aoclsparse_index_base base,
                                                 const aoclsparse_csr  csr_mat,
                                                 bool                 &sorted,
                                                 bool                 &fulldiag);

aoclsparse_status aoclsparse_csr_indices(aoclsparse_int        m,
                                         aoclsparse_index_base base,
                                         const aoclsparse_int *icrow,
                                         const aoclsparse_int *icol,
                                         aoclsparse_int      **idiag,
                                         aoclsparse_int      **iurow);

/* Copy a csr matrix. If the input is 1-base (i.e., base = 1), then the output
 * arrays of As would be base corrected and be 0-base. The function can also
 * preserve the input base in output arrays of As, provided the input base argument
 * is zero-base.
 * Possible exit: invalid size, invalid pointer, memory alloc
 */
template <typename T>
aoclsparse_status aoclsparse_copy_csr(aoclsparse_int                  m,
                                      [[maybe_unused]] aoclsparse_int n,
                                      aoclsparse_int                  nnz,
                                      aoclsparse_index_base           base,
                                      const aoclsparse_csr            A,
                                      aoclsparse_csr                  As)
{
    aoclsparse_int i;
    T             *aval, *aval_s;

    if((m < 0) || (nnz < 0))
        return aoclsparse_status_invalid_size;
    if((A == nullptr) || (As == nullptr))
        return aoclsparse_status_invalid_pointer;
    if(A->csr_col_ptr == nullptr || A->csr_row_ptr == nullptr || A->csr_val == nullptr)
        return aoclsparse_status_invalid_pointer;

    try
    {
        As->csr_row_ptr = new aoclsparse_int[m + 1];
        As->csr_col_ptr = new aoclsparse_int[nnz];
        As->csr_val     = ::operator new(nnz * sizeof(T));
    }
    catch(std::bad_alloc &)
    {
        delete[] As->csr_row_ptr;
        delete[] As->csr_col_ptr;
        ::operator delete(As->csr_val);
        return aoclsparse_status_memory_error;
    }

    aval   = static_cast<T *>(A->csr_val);
    aval_s = static_cast<T *>(As->csr_val);

    // copy the matrix
    for(i = 0; i < m + 1; i++)
        As->csr_row_ptr[i] = A->csr_row_ptr[i] - base;
    for(i = 0; i < nnz; i++)
    {
        As->csr_col_ptr[i] = A->csr_col_ptr[i] - base;
        aval_s[i]          = aval[i];
    }

    return aoclsparse_status_success;
}

/* Function to sort CSR or CSC matrix.
 * Input parameters :-
 * maj_dim            : major dimension - row(m) for CSR, col(n) for CSC
 * min_dim            : minor dimension - col(n) for CSR, row(m) for CSR
 * nnz                : non-zero count
 * src_base/dest_base : 0-base or 1-base
 * src_idx_ptr        : csr_row_ptr from _aoclsparse_csr or col_ptr from _aoclsparse_csc
 * src_idx/dest_idx   : csr_col_ptr from _aoclsparse_csr or row_ind from _aoclsparse_csc
 * src_val/dest/val   : csr_val from _aoclsparse_csr or val from _aoclsparse_csc
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

/* create some artificial fill-ins with zeros on the diagonal if some elements are missing
 * Assumes the rows are sorted */
template <typename T>
aoclsparse_status aoclsparse_csr_fill_diag(aoclsparse_int        m,
                                           aoclsparse_int        n,
                                           aoclsparse_int        nnz,
                                           aoclsparse_index_base base,
                                           aoclsparse_csr        A)
{
    aoclsparse_int i, j, count, idx, idxend;
    try
    {
        std::vector<aoclsparse_int> missing_diag(m, -1);

        if(!A->csr_col_ptr || !A->csr_row_ptr || !A->csr_val)
            return aoclsparse_status_invalid_pointer;

        // Check each row for missing element on the diagonal
        count = 0;
        bool diag_found;
        for(i = 0; i < m; i++)
        {
            diag_found = false;
            idxend     = A->csr_row_ptr[i + 1] - base;
            for(idx = (A->csr_row_ptr[i] - base); idx < idxend; idx++)
            {
                j = A->csr_col_ptr[idx] - base;
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
        csr_val                 = static_cast<T *>(A->csr_val);

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
            idxend   = A->csr_row_ptr[i + 1] - base;
            icrow[i] = A->csr_row_ptr[i] - base + n_added;
            // Copy the row into the new matrix
            for(idx = (A->csr_row_ptr[i] - base); idx < idxend; idx++)
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
                icol[nnz_curr]                   = A->csr_col_ptr[idx] - base;
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
        delete[] A->csr_col_ptr;
        delete[] A->csr_row_ptr;
        ::operator delete(A->csr_val);
        A->csr_col_ptr = icol;
        A->csr_row_ptr = icrow;
        A->csr_val     = aval;
    }
    catch(std::bad_alloc &)
    {
        // missing_diag allocation failure
        return aoclsparse_status_memory_error;
    }

    return aoclsparse_status_success;
}

/* Check a CSR matrix inputs and create a clean version */
template <typename T>
aoclsparse_status aoclsparse_csr_optimize(aoclsparse_matrix A)
{
    aoclsparse_status status;

    if(!A)
        return aoclsparse_status_invalid_pointer;

    // Make sure we have the right type before proceeding
    if(!((A->val_type == aoclsparse_dmat && std::is_same_v<T, double>)
         || (A->val_type == aoclsparse_smat && std::is_same_v<T, float>)
         || (A->val_type == aoclsparse_zmat && std::is_same_v<T, std::complex<double>>)
         || (A->val_type == aoclsparse_cmat && std::is_same_v<T, std::complex<float>>)
         || (A->val_type == aoclsparse_zmat && std::is_same_v<T, aoclsparse_double_complex>)
         || (A->val_type == aoclsparse_cmat && std::is_same_v<T, aoclsparse_float_complex>)))
        return aoclsparse_status_wrong_type;

    //Make sure base-index is the correct value
    if(A->base != aoclsparse_index_base_zero && A->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    // Check the user's matrix format
    // First check the matrix is a valid matrix
    status = aoclsparse_mat_check_internal(A->m,
                                           A->n,
                                           A->nnz,
                                           A->csr_mat.csr_row_ptr,
                                           A->csr_mat.csr_col_ptr,
                                           A->csr_mat.csr_val,
                                           shape_general,
                                           A->base,
                                           nullptr);
    if(status != aoclsparse_status_success)
        // The matrix has invalid data, abort optimize and return error
        return status;

    // Check if the matrix is sorted with full diag
    bool sorted, fulldiag;
    status = aoclsparse_csr_check_sort_diag(A->m, A->n, A->base, &A->csr_mat, sorted, fulldiag);
    if(status != aoclsparse_status_success)
        // Shouldn't happen, pointers have already been checked
        return aoclsparse_status_internal_error;

    // build the clean CSR matrix
    if(sorted && fulldiag)
    {
        // The matrix is already in the correct format, use directly user's memory
        A->opt_csr_mat.csr_row_ptr = A->csr_mat.csr_row_ptr;
        A->opt_csr_mat.csr_col_ptr = A->csr_mat.csr_col_ptr;
        A->opt_csr_mat.csr_val     = A->csr_mat.csr_val;
        A->opt_csr_is_users        = true;
        //since user's csr buffers are used for execution kernel, the base-index correction
        //will happen during execution
        A->internal_base_index = A->base;
    }
    else
    {
        // Create a copy of the user's data to be able to manipulate it
        A->opt_csr_is_users = false;
        status = aoclsparse_copy_csr<T>(A->m, A->n, A->nnz, A->base, &A->csr_mat, &A->opt_csr_mat);
        if(status != aoclsparse_status_success)
            return status;
        //since the correction is already performed during above copy, the execution kernel and the
        //subsequent calls to sort, diagonal fill and idiag/iurow compute can
        //treat storage buffers in opt_csr_mat as zero-based indexing and need not perform
        //double correction
        A->internal_base_index = aoclsparse_index_base_zero;
    }
    if(!sorted)
    {
        aoclsparse_sort_idx_val<T>(A->m,
                                   A->n,
                                   A->nnz,
                                   A->base,
                                   A->csr_mat.csr_row_ptr,
                                   A->csr_mat.csr_col_ptr,
                                   static_cast<T *>(A->csr_mat.csr_val),
                                   A->internal_base_index,
                                   A->opt_csr_mat.csr_col_ptr,
                                   static_cast<T *>(A->opt_csr_mat.csr_val));
        // check again for full diagonal
        status = aoclsparse_csr_check_sort_diag(
            A->m, A->n, A->internal_base_index, &A->opt_csr_mat, sorted, fulldiag);
        if(status != aoclsparse_status_success)
            return status;
    }
    if(!fulldiag)
    {
        status = aoclsparse_csr_fill_diag<T>(
            A->m, A->n, A->nnz, A->internal_base_index, &A->opt_csr_mat);
        if(status != aoclsparse_status_success)
            return status;
    }
    status = aoclsparse_csr_indices(A->m,
                                    A->internal_base_index,
                                    A->opt_csr_mat.csr_row_ptr,
                                    A->opt_csr_mat.csr_col_ptr,
                                    &A->idiag,
                                    &A->iurow);
    if(status != aoclsparse_status_success)
        return status;
    A->opt_csr_ready     = true;
    A->opt_csr_full_diag = fulldiag;
    A->optimized         = true;

    return aoclsparse_status_success;
}

#endif // AOCLSPARSE_INPUT_CHECK_HPP
