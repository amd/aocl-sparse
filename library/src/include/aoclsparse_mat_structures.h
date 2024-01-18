/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef AOCLSPARSE_MAT_STRUCTS_H
#define AOCLSPARSE_MAT_STRUCTS_H

#include "aoclsparse.h"
#include "aoclsparse_optimize_data.hpp"

/********************************************************************************
 * \brief aoclsparse_csr is a structure holding the aoclsparse matrix
 * in csr format. It must be initialized using aoclsparse_create_(s/d)csr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
 *******************************************************************************/
struct _aoclsparse_csr
{
    // CSR matrix part
    aoclsparse_int *csr_row_ptr = nullptr;
    aoclsparse_int *csr_col_ptr = nullptr;
    void           *csr_val     = nullptr;
    aoclsparse_int *blk_row_ptr = nullptr;
    aoclsparse_int *blk_col_ptr = nullptr;
    void           *blk_val     = nullptr;
    uint8_t        *masks       = nullptr;
    aoclsparse_int  nRowsblk    = 0;
};

/********************************************************************************
 * \brief aoclsparse_ell is a structure holding the aoclsparse matrix
 * in ELL format. It is used internally during the optimization process.
 * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
 *******************************************************************************/
struct _aoclsparse_ell
{
    // ELL matrix part
    aoclsparse_int  ell_width   = 0;
    aoclsparse_int *ell_col_ind = nullptr;
    void           *ell_val     = nullptr;
};

/********************************************************************************
 * \brief aoclsparse_ell_csr_hyb is a structure holding the aoclsparse matrix
 * in ELL-CSR hybrid format. It is used internally during the optimization process.
 * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
 *******************************************************************************/
struct _aoclsparse_ell_csr_hyb
{
    // ELL matrix part
    aoclsparse_int  ell_width   = 0;
    aoclsparse_int  ell_m       = 0;
    aoclsparse_int *ell_col_ind = nullptr;
    void           *ell_val     = nullptr;

    // CSR part
    aoclsparse_int *csr_row_id_map = nullptr;
    //   aoclsparse_int* row_id_map =  nullptr;
    aoclsparse_int *csr_col_ptr = nullptr; // points to the corresponding CSR pointer
    void           *csr_val     = nullptr; // points to the corresponding CSR pointer
};

/********************************************************************************
 * \brief aoclsparse_coo is a structure holding the aoclsparse matrix
 * in COO format. It is used internally during the optimization process.
 * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
 *******************************************************************************/
struct _aoclsparse_coo
{
    // COO matrix part
    aoclsparse_int *row_ind = nullptr;
    aoclsparse_int *col_ind = nullptr;
    void           *val     = nullptr;
};

/********************************************************************************
 * \brief _aoclsparse_ilu is a structure holding data members for ILU operation.
 * It is used internally during the optimization process which includes ILU factorization.
 * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
 *******************************************************************************/
struct _aoclsparse_ilu
{
    aoclsparse_int     *lu_diag_ptr     = NULL; //pointer to diagonal elements in csr values array
    aoclsparse_int     *col_idx_mapper  = NULL; //working array
    void               *precond_csr_val = NULL; //copy of buffer for ilu precondioned factors
    bool                ilu_factorized  = false; //flag to indicate if ILU factorization is done
    aoclsparse_ilu_type ilu_fact_type; // indicator of ILU factorization type
    // true: ILU Optimization/Working-Buffer-Allocation already done, else needs to be performed^M
    bool ilu_ready = false;
};

/********************************************************************************
 * \brief aoclsparse_csc is a structure holding the aoclsparse matrix
 * in csc format. It must be initialized using aoclsparse_create_(s/d/c/z)csc()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
 *******************************************************************************/
struct _aoclsparse_csc
{
    // CSC matrix part
    aoclsparse_int *col_ptr = nullptr;
    aoclsparse_int *row_idx = nullptr;
    void           *val     = nullptr;
};

/********************************************************************************
 * \brief _aoclsparse_matrix is a structure holding generic aoclsparse matrices.
 * It should be used by all the sparse routines to initialize the sparse matrices.
 * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
 *******************************************************************************/
struct _aoclsparse_matrix
{
    // generic sparse matrix properties
    aoclsparse_int m;
    aoclsparse_int n;
    aoclsparse_int nnz;
    bool           optimized = false;
    //index-base provided by user, read-only!
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    /*
        internal base-index after correction, for consumption in execution kernels
        1. the internal base index applies to the clean csr structure opt_csr_mat, in which
            all the pointers (row_ptr, col_ind, idiag, iurow) are either in
                    1.  base-0: if user provides a csr matrix which is unsorted or without
                        full-diag, then csr cleanup and base correction is performed
                    2.  base-1: if user provides a clean sorted csr matrix with full-diag in 1-base,
                         then opt_csr_mat just points to this input matrix without copy or
                         base-correction.
        2. All the conversion routines and internal spmv storage formats such as ell, ellt, ellt-hyb,
            dia, bsr, blkcsr and br4 preserve the base-index as was provided by user in his input
            csr matrix. Therefore, the final execution kernels also will use the original input
            base-index.
    */
    aoclsparse_index_base       internal_base_index;
    aoclsparse_matrix_data_type val_type;

    // indicates internal matrix representation
    aoclsparse_matrix_format_type mat_type;
    // indicates actual matrix passed
    aoclsparse_matrix_format_type input_format;
    // Optimization hints linked list
    aoclsparse_optimize_data *optim_data = nullptr;

    // csr matrix
    bool                   csr_mat_is_users = false;
    struct _aoclsparse_csr csr_mat;

    // csr matrix for avx2
    struct _aoclsparse_csr csr_mat_br4;

    // ellt matrix
    struct _aoclsparse_ell ell_mat;

    // ell-csr-hyb matrix
    struct _aoclsparse_ell_csr_hyb ell_csr_hyb_mat;

    // coo matrix
    bool                   coo_mat_is_users = false;
    struct _aoclsparse_coo coo_mat;

    //ilu members
    struct _aoclsparse_ilu ilu_info;

    // optimized csr matrix
    // It is checked, sorted in rows, has a diagonal element in each row,
    // however, some diagonal elements might have been added as zeros
    struct _aoclsparse_csr opt_csr_mat;
    // the matrix has been 'optimized', it can be used
    bool opt_csr_ready = false;
    // if true, user's csr_mat was fine to use so opt_csr_mat points
    // to the same memory. Deallocate only if !opt_csr_is_users
    bool opt_csr_is_users = false;
    // the original matrix had full (nonzero) diagonal, so the matrix
    // is safe for TRSVs
    bool opt_csr_full_diag;
    // store if the matrix has already been optimized for this blocked SpMV
    bool blk_optimized = false;
    // position where the diagonal is located in every row
    aoclsparse_int *idiag = nullptr;
    // position where the first strictly upper triangle element is/would be located in every row
    aoclsparse_int *iurow = nullptr;

    // csc matrix
    bool                   csc_mat_is_users = false;
    struct _aoclsparse_csc csc_mat;

    // used to indicate if any additional memory required further performance optimization purposes
    aoclsparse_memory_usage mem_policy = aoclsparse_memory_usage_unrestricted;
};

#endif // AOCLSPARSE_MAT_STRUCTS_H
