/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse_descr.h"
#include "aoclsparse_types.h"
#include "aoclsparse.hpp"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_mat_structures.hpp"

/********************************************************************************
 * \brief aoclsparse::create_csr sets the sparse matrix in the CSR format
 * for any data type
********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse::create_csr(aoclsparse_matrix    *mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *row_ptr,
                                         aoclsparse_int       *col_idx,
                                         T                    *val,
                                         bool                  fast_chck)
{
    aoclsparse_status status;
    if(!mat)
        return aoclsparse_status_invalid_pointer;
    *mat = nullptr;
    // Validate the input parameters
    aoclsparse_matrix_sort mat_sort;
    bool                   mat_fulldiag;

    // Check the matrix parameters for correctness
    status = aoclsparse_mat_check_internal(M,
                                           N,
                                           nnz,
                                           row_ptr,
                                           col_idx,
                                           val,
                                           shape_general,
                                           base,
                                           mat_sort,
                                           mat_fulldiag,
                                           nullptr,
                                           fast_chck);

    if(status != aoclsparse_status_success)
    {
        return status;
    }

    aoclsparse::csr *csr_mat = nullptr;
    try
    {
        *mat    = new _aoclsparse_matrix;
        csr_mat = new aoclsparse::csr(
            M, N, nnz, aoclsparse_csr_mat, base, get_data_type<T>(), row_ptr, col_idx, val);
        (*mat)->mats.push_back(csr_mat);
    }
    catch(std::bad_alloc &)
    {
        if(csr_mat)
            delete csr_mat;
        if(*mat)
        {
            delete *mat;
            *mat = nullptr;
        }
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(*mat, M, N, nnz, aoclsparse_csr_mat);
    (*mat)->val_type = get_data_type<T>();
    (*mat)->sort     = mat_sort;
    (*mat)->fulldiag = mat_fulldiag;
    (*mat)->mat_type = aoclsparse_csr_mat;
    return aoclsparse_status_success;
}

#define CRTE_CSR(SUF)                                                                           \
    template DLL_PUBLIC aoclsparse_status aoclsparse::create_csr(aoclsparse_matrix    *mat,     \
                                                                 aoclsparse_index_base base,    \
                                                                 aoclsparse_int        M,       \
                                                                 aoclsparse_int        N,       \
                                                                 aoclsparse_int        nnz,     \
                                                                 aoclsparse_int       *row_ptr, \
                                                                 aoclsparse_int       *col_idx, \
                                                                 SUF                  *val,     \
                                                                 bool check_matrix);

INSTANTIATE_FOR_ALL_TYPES(CRTE_CSR);

/********************************************************************************
 * \brief aoclsparse::create_bsr sets the sparse matrix in the BSR format
 * for any data type
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse::create_bsr(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_order      order,
                                         const aoclsparse_int        bM,
                                         const aoclsparse_int        bN,
                                         const aoclsparse_int        block_dim,
                                         aoclsparse_int             *row_ptr,
                                         aoclsparse_int             *col_idx,
                                         T                          *val,
                                         bool                        fast_chck)
{
    aoclsparse_status status;
    if(!mat || !row_ptr)
        return aoclsparse_status_invalid_pointer;
    *mat = nullptr;
    if(block_dim <= 0)
        return aoclsparse_status_invalid_value;
    if(bM < 0)
        return aoclsparse_status_invalid_size;

    // Temporary pointer to an instance of the BSR matrix class.
    aoclsparse::bsr       *bsr_mat = nullptr;
    aoclsparse_matrix_sort mat_sort;
    bool                   mat_fulldiag;
    aoclsparse_int         bnnz = row_ptr[bM] - base;
    // Validate other input parameters
    if((status = aoclsparse_mat_check_internal(bM,
                                               bN,
                                               bnnz,
                                               row_ptr,
                                               col_idx,
                                               val,
                                               shape_general,
                                               base,
                                               mat_sort,
                                               mat_fulldiag,
                                               nullptr,
                                               fast_chck))
       != aoclsparse_status_success)
    {
        return status;
    }

    try
    {
        *mat    = new _aoclsparse_matrix;
        bsr_mat = new aoclsparse::bsr(bM,
                                      bN,
                                      bnnz,
                                      aoclsparse_bsr_mat,
                                      base,
                                      get_data_type<T>(),
                                      block_dim,
                                      order,
                                      row_ptr,
                                      col_idx,
                                      val);
        (*mat)->mats.push_back(bsr_mat);
    }
    catch(std::bad_alloc &)
    {
        delete bsr_mat;
        delete *mat;
        *mat = nullptr;
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(
        *mat, bM * block_dim, bN * block_dim, bnnz * block_dim * block_dim, aoclsparse_bsr_mat);
    (*mat)->val_type = get_data_type<T>();
    (*mat)->mat_type = aoclsparse_bsr_mat;
    (*mat)->sort     = mat_sort;
    (*mat)->fulldiag = false;
    return aoclsparse_status_success;
}

#define CRTE_BSR(SUF)                                                                               \
    template DLL_PUBLIC aoclsparse_status aoclsparse::create_bsr(aoclsparse_matrix          *mat,   \
                                                                 const aoclsparse_index_base base,  \
                                                                 const aoclsparse_order      order, \
                                                                 const aoclsparse_int        bM,    \
                                                                 const aoclsparse_int        bN,    \
                                                                 const aoclsparse_int block_dim,    \
                                                                 aoclsparse_int      *row_ptr,      \
                                                                 aoclsparse_int      *col_idx,      \
                                                                 SUF                 *val,          \
                                                                 bool                 check_matrix);
INSTANTIATE_FOR_ALL_TYPES(CRTE_BSR);