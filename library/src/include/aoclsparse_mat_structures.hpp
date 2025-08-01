/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
#include "aoclsparse_mtx_dispatcher.hpp"

#include <vector>

enum aoclsparse_hinted_action
{
    aoclsparse_action_none = 0,
    aoclsparse_action_mv,
    aoclsparse_action_sv,
    aoclsparse_action_mm,
    aoclsparse_action_2m,
    aoclsparse_action_ilu0,
    aoclsparse_action_sm_row,
    aoclsparse_action_sm_col,
    aoclsparse_action_dotmv,
    aoclsparse_action_symgs,
    aoclsparse_action_sorv_forward,
    aoclsparse_action_sorv_backward,
    aoclsparse_action_sorv_symm,
    aoclsparse_action_max, // add any new action before this
};

/* Linked list of all the hint information that was passed through the
 * functions aoclsparse_*_hint()
 */
struct aoclsparse_optimize_data
{
    aoclsparse_hinted_action act   = aoclsparse_action_none;
    aoclsparse_operation     trans = aoclsparse_operation_none;
    aoclsparse_matrix_type   type;
    aoclsparse_fill_mode     fill_mode;
    aoclsparse_int           kid  = -1; // optimal kid for the optimized data
    aoclsparse::doid         doid = aoclsparse::doid::len; // Invalid doid

    // number of operations estimated
    aoclsparse_int nop = 0;
    // store if the matrix has already been optimized for this specific operation
    bool action_optimized = false;
    // next hint requested
    aoclsparse_optimize_data *next = nullptr;
};

/* Add a new optimize hint to the list */
aoclsparse_status aoclsparse_add_hint(aoclsparse_optimize_data  *&list,
                                      aoclsparse_hinted_action    op,
                                      aoclsparse_mat_descr        desc,
                                      aoclsparse_operation        trans,
                                      aoclsparse_matrix_data_type dt,
                                      aoclsparse_int              nop,
                                      aoclsparse_int              kid);

/* Deallocate the aoclsparse_optimize_data linked list*/
void aoclsparse_optimize_destroy(aoclsparse_optimize_data *&opt);

namespace aoclsparse
{
    /********************************************************************************
     * \brief base_mtx is a base class for all sparse matrix formats.
     * This class serves as a base class for various sparse matrix formats such as CSR,
     * ELL, ELL-CSR hybrid, COO, etc. It provides a common interface and can be extended
     * to include specific data members and methods for different sparse matrix formats.
     *******************************************************************************/
    class base_mtx
    {
    protected:
        // flag to indicate if the matrix is internally created, not user provided. This is used to
        // determine if the matrix should be destroyed at the end of the optimization process. This flag is set
        // to true for all internally created matrices.
        bool is_internal = true;

    public:
        aoclsparse_int m;
        aoclsparse_int n;
        aoclsparse_int nnz;
        // Holds the descriptor and operation values of the matrix
        aoclsparse::doid              doid = doid::len; // by default invalid
        aoclsparse_matrix_format_type mat_type;
        aoclsparse_index_base         base;
        aoclsparse_matrix_data_type   val_type;

        // default constructor
        base_mtx() = default;

        // constructor to initialize the base matrix with given parameters
        base_mtx(aoclsparse_int                m,
                 aoclsparse_int                n,
                 aoclsparse_int                nnz,
                 aoclsparse_matrix_format_type mat_type,
                 aoclsparse_index_base         base,
                 aoclsparse_matrix_data_type   val_type,
                 bool                          is_internal)
            : is_internal(is_internal) // Initialize member variables with constructor parameters
            , m(m)
            , n(n)
            , nnz(nnz)
            , mat_type(mat_type)
            , base(base)
            , val_type(val_type)
        {
            if(m < 0 || n < 0)
                throw std::bad_array_new_length();
        }

        // default destructor
        virtual ~base_mtx() = default;
    };

    /********************************************************************************
     * \brief csr is a class for storing AOCL Sparse matrices in CSR or CSC format.
    * CSR matrices should be initialized using aoclsparse_create_(s/d/c/z)csr(),
    * and CSC matrices using aoclsparse_create_(s/d/c/z)csc().
    * The returned handle must be used in all subsequent library function calls
    * involving the matrix. Destroy the matrix at the end using
    * aoclsparse_destroy_mat_structs().
     *******************************************************************************/
    class csr : public base_mtx
    {
        // CSR/CSC matrix part
    public:
        // For CSR matrix format, 'ptr' points to csr_row_ptr
        // For CSC matrix format, 'ptr' points to csc_col_ptr
        aoclsparse_int *ptr = nullptr;
        // For CSR format, 'ind' stores column indices (csr_col_ind)
        // For CSC format, 'ind' stores row indices (csc_row_ind)
        aoclsparse_int *ind = nullptr;
        // 'val' stores the values of CSR/CSC matrix
        void *val = nullptr;
        // position where the diagonal is located in every row
        aoclsparse_int *idiag = nullptr;
        // position where the first strictly upper triangle element is/would be located in every row
        aoclsparse_int *iurow = nullptr;
        // if optimized, set true
        bool is_optimized = false;

        // Default constructor
        csr() = default;
        /* Parameterized constructor:
        Initializes the matrix with the specified parameters and allocates memory for the CSR data arrays.
        If nnz<0, only csr_row_ptr is allocated to support the first stage of nnz computation.
        This constructor is used when the matrix is created internally.
        */
        csr(aoclsparse_int                m,
            aoclsparse_int                n,
            aoclsparse_int                nnz,
            aoclsparse_matrix_format_type mat_type,
            aoclsparse_index_base         base,
            aoclsparse_matrix_data_type   val_type)
            : base_mtx(m, n, nnz, mat_type, base, val_type, true)
        {
            try
            {
                // row/column pointer array can be allocated regardless of nnz value
                // This enables array creation when matrix dimensions are known but nnz is not yet set
                // For CSC format, allocate n+1 (columns + 1) for csc_col_ptr
                // For CSR format, allocate m+1 (rows + 1) for csr_row_ptr
                aoclsparse_int len = (mat_type == aoclsparse_csc_mat) ? (n + 1) : (m + 1);
                ptr                = new aoclsparse_int[len];
                if(nnz == 0)
                {
                    // For empty matrices, set all row/column pointers to the base index
                    std::fill(ptr, ptr + len, base);
                }
                // Only allocate memory if nnz is valid (non-negative)
                if(nnz >= 0)
                {
                    // Allocate index array:
                    // For CSR, stores column indices; for CSC, stores row indices
                    ind = new aoclsparse_int[nnz];
                    // Allocate value array for matrix entries
                    val = ::operator new(data_size[val_type] * nnz);
                }
            }
            catch(...)
            {
                // Clean up any allocated memory on failure
                delete[] ptr;
                delete[] ind;
                ::operator delete(val);
                throw;
            }
        }
        // Parameterized constructor that sets up the matrix using user-provided, pre-allocated data arrays.
        csr(aoclsparse_int                m,
            aoclsparse_int                n,
            aoclsparse_int                nnz,
            aoclsparse_matrix_format_type mat_type,
            aoclsparse_index_base         base,
            aoclsparse_matrix_data_type   val_type,
            aoclsparse_int               *ptr,
            aoclsparse_int               *ind,
            void                         *val,
            aoclsparse_int               *diag = nullptr,
            aoclsparse_int               *urow = nullptr)
            : base_mtx(m, n, nnz, mat_type, base, val_type, false)
            , ptr(ptr)
            , ind(ind)
            , val(val)
            , idiag(diag)
            , iurow(urow)
        {
        }
        // destructor
        virtual ~csr()
        {
            // Free the memory allocated if the matrix was internally allocated (is_internal = true)
            if(is_internal)
            {
                delete[] ptr;
                delete[] ind;
                ::operator delete(val);
            }
            // idiag and iurow are always deleted, as they may be allocated for optimized matrices
            delete[] idiag;
            delete[] iurow;
        }
    };

    /********************************************************************************
     * \brief blk_csr is a class holding the aoclsparse matrix
     * in Block CSR (Compressed Sparse Row) format. It is used internally during the
     * optimization process. It should be destroyed at the end using
     * aoclsparse_destroy_mat_structs().
     *******************************************************************************/
    class blk_csr : public base_mtx
    {
    public:
        aoclsparse_int *blk_row_ptr = nullptr;
        aoclsparse_int *blk_col_ptr = nullptr;
        void           *blk_val     = nullptr;
        uint8_t        *masks       = nullptr;
        aoclsparse_int  nRowsblk    = 0;

        /* Parameterized constructor:
        Initializes the matrix with the specified parameters and allocates memory for the Block CSR data arrays.
        This constructor is used when the matrix is created internally.
        */
        blk_csr(aoclsparse_int              m,
                aoclsparse_int              n,
                aoclsparse_int              nnz,
                aoclsparse_int              nRowsblk,
                aoclsparse_index_base       base,
                aoclsparse_matrix_data_type val_type,
                aoclsparse_int              blk_width,
                aoclsparse_int              total_blks)
            : base_mtx(m, n, nnz, aoclsparse_blkcsr_mat, base, val_type, true)
            , nRowsblk(nRowsblk)
        {
            // Validate dimensions before allocation
            if(nnz < 0 || nRowsblk < 0 || blk_width < 0 || total_blks < 0)
                throw std::bad_array_new_length();
            try
            {
                blk_row_ptr = new aoclsparse_int[m + 1];
                masks       = new uint8_t[total_blks * nRowsblk];
                blk_col_ptr = new aoclsparse_int[nnz];
                blk_val     = ::operator new((nnz + nRowsblk * blk_width) * sizeof(double));
            }
            catch(...)
            {
                delete[] blk_row_ptr;
                delete[] blk_col_ptr;
                ::operator delete(blk_val);
                delete[] masks;
                throw;
            }
        }

        // Parameterized constructor that sets up the matrix using user-provided, pre-allocated data arrays.
        blk_csr(aoclsparse_int              m,
                aoclsparse_int              n,
                aoclsparse_int              nnz,
                aoclsparse_int              nRowsblk,
                aoclsparse_index_base       base,
                aoclsparse_matrix_data_type val_type,
                aoclsparse_int             *blk_row_ptr,
                aoclsparse_int             *blk_col_ptr,
                void                       *blk_val,
                uint8_t                    *masks)
            : base_mtx(m, n, nnz, aoclsparse_blkcsr_mat, base, val_type, false)
            , blk_row_ptr(blk_row_ptr)
            , blk_col_ptr(blk_col_ptr)
            , blk_val(blk_val)
            , masks(masks)
            , nRowsblk(nRowsblk)
        {
        }
        // Destructor
        virtual ~blk_csr()
        {
            // Free the memory allocated if the matrix was internally allocated (is_internal = true)
            if(is_internal)
            {
                delete[] blk_row_ptr;
                delete[] blk_col_ptr;
                ::operator delete(blk_val);
                delete[] masks;
            }
        }
    };

    /********************************************************************************
     * \brief tcsr is a class holding a sparse matrix in TCSR
     * (Triangular Storage) format.
     * Both triangles (L+D and D+U) are stored in two separate arrays, they are stored like
     * CSR with partial sorting (L+D and D+U order is followed, but the indices within L or U
     * group may not be sorted)
     *  - One array with L elements potentially unsorted, followed by D elements in the L+D part
     *    of the matrix.
     *  - Another array with D elements, followed by U elements potentially unsorted in the D+U part
     *    of the matrix.
     *  - Currently TCSR storage matrix supports only the matrices with full(non-zero) diagonals.
     *
     * Both the lower and upper triangular parts are stored and work as a normal CSR:
     * The lower triangular part:
     *   - row pointers: row_ptr_L[0] ... row_ptr_L[m],
     *   - column indices: col_idx_L[0] ... col_idx_L[row_ptr_L[m]-1-A->base]
     *   - values: with same indices for val_L as for col_idx_L
     * The upper triangular part:
     *   - row pointers: row_ptr_U[0] ... row_ptr_U[m],
     *   - column indices: col_idx_U[0], ... col_idx_U[row_ptr_U[m]-1-A->base]
     *   - values: with same indices for val_U as for col_idx_U
     *
     * It must be initialized using aoclsparse_create_(s/d/c/z)tcsr()
     * and the returned handle must be passed to all subsequent library function
     * calls that involve the matrix.
     * It should be destroyed at the end using aoclsparse_destroy().
     *******************************************************************************/
    class tcsr : public base_mtx
    {
    public:
        // size(row_ptr_L) = m (number of rows of the aoclsparse_matrix) + 1
        aoclsparse_int *row_ptr_L = nullptr; // points to every row of the lower part of the matrix
        // size(row_ptr_U) = m + 1
        aoclsparse_int *row_ptr_U = nullptr; // points to every row of the upper part of the matrix
        // size(col_idx_L) = no.of strictly lower triangular elements + m
        aoclsparse_int *col_idx_L
            = nullptr; // contains col idx of the lower part of the tcsr matrix
        // size(col_idx_U) = no.of strictly upper triangular elements + m
        aoclsparse_int *col_idx_U
            = nullptr; // contains col idx of the upper part of the tcsr matrix
        //size(val_L) = no.of strictly lower triangular elements + m
        void *val_L = nullptr; // contains values of the lower part of the tcsr matrix
        //size(val_U) = no.of strictly upper triangular elements + m
        void *val_U = nullptr; // contains values of the upper part of the tcsr matrix
        // For TCSR matrix, idiag points to the position of diagonals in the lower triangular part of the matrix.
        aoclsparse_int *idiag = nullptr;
        // For TCSR matrix, iurow points to the position of upper triangle element in the upper triangular part of the matrix.
        aoclsparse_int *iurow = nullptr;
        // if optimized, set true
        bool is_optimized = false;

        /* Parameterized constructor:
        Initializes the matrix with the specified parameters and allocates memory for the Block CSR data arrays.
        This constructor is used when the matrix is created internally.
        */
        tcsr(aoclsparse_int              m,
             aoclsparse_int              n,
             aoclsparse_int              nnz,
             aoclsparse_index_base       base,
             aoclsparse_matrix_data_type val_type)
            : base_mtx(m, n, nnz, aoclsparse_tcsr_mat, base, val_type, true)
        {
            if(nnz < 0)
                throw std::bad_array_new_length();
            try
            {
                row_ptr_L = new aoclsparse_int[m + 1];
                row_ptr_U = new aoclsparse_int[m + 1];
                col_idx_L = new aoclsparse_int[nnz];
                col_idx_U = new aoclsparse_int[nnz];
                val_L     = ::operator new[](nnz *data_size[val_type]);
                val_U     = ::operator new[](nnz *data_size[val_type]);
            }
            catch(...)
            {
                // Clean up any allocated memory on failure
                delete[] row_ptr_L;
                delete[] row_ptr_U;
                delete[] col_idx_L;
                delete[] col_idx_U;
                ::operator delete(val_L);
                ::operator delete(val_U);
                throw;
            }
        }

        // Parameterized constructor
        tcsr(aoclsparse_int              m,
             aoclsparse_int              n,
             aoclsparse_int              nnz,
             aoclsparse_index_base       base,
             aoclsparse_matrix_data_type val_type,
             aoclsparse_int             *row_ptr_L,
             aoclsparse_int             *row_ptr_U,
             aoclsparse_int             *col_idx_L,
             aoclsparse_int             *col_idx_U,
             void                       *val_L,
             void                       *val_U)
            : base_mtx(m, n, nnz, aoclsparse_tcsr_mat, base, val_type, false)
            , row_ptr_L(row_ptr_L)
            , row_ptr_U(row_ptr_U)
            , col_idx_L(col_idx_L)
            , col_idx_U(col_idx_U)
            , val_L(val_L)
            , val_U(val_U)
        {
        }

        // Destructor
        ~tcsr()
        {
            // Free the memory allocated if the matrix was internally allocated (is_internal = true)
            if(is_internal)
            {
                delete[] row_ptr_L;
                delete[] row_ptr_U;
                delete[] col_idx_L;
                delete[] col_idx_U;
                ::operator delete(val_L);
                ::operator delete(val_U);
            }
            delete[] idiag;
            delete[] iurow;
        }
    };

    /********************************************************************************
     * \brief ell is a class holding the aoclsparse matrix
     * in ELL format. It is used internally during the optimization process.
     * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
     *******************************************************************************/
    class ell : public base_mtx
    {
        // ELL matrix part
    public:
        aoclsparse_int  ell_width   = 0;
        aoclsparse_int *ell_col_ind = nullptr;
        void           *ell_val     = nullptr;

        // Parameterized constructor that sets up the matrix using user-provided, pre-allocated data arrays.
        ell(aoclsparse_int              m,
            aoclsparse_int              n,
            aoclsparse_int              nnz,
            aoclsparse_index_base       base,
            aoclsparse_matrix_data_type val_type,
            aoclsparse_int              ell_width,
            aoclsparse_int             *ell_col_ind,
            void                       *ell_val)
            : base_mtx(m, n, nnz, aoclsparse_ell_mat, base, val_type, false)
            , ell_width(ell_width)
            , ell_col_ind(ell_col_ind)
            , ell_val(ell_val)
        {
        }

        // Destructor
        ~ell()
        {
            if(is_internal)
            {
                delete[] ell_col_ind;
                ::operator delete(ell_val);
            }
        }
    };

    /********************************************************************************
     * \brief ell_csr_hyb is a class holding the aoclsparse matrix
     * in ELL-CSR hybrid format. It is used internally during the optimization process.
     * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
     *******************************************************************************/
    class ell_csr_hyb : public base_mtx
    {
        // ELL matrix part
    public:
        aoclsparse_int  ell_width   = 0;
        aoclsparse_int  ell_m       = 0;
        aoclsparse_int *ell_col_ind = nullptr;
        void           *ell_val     = nullptr;

        // CSR part
        aoclsparse_int *csr_row_id_map = nullptr;
        //   aoclsparse_int* row_id_map =  nullptr;
        aoclsparse_int *csr_col_ptr = nullptr; // points to the corresponding CSR pointer
        void           *csr_val     = nullptr; // points to the corresponding CSR pointer

        // Parameterized constructor
        ell_csr_hyb(aoclsparse_int              m,
                    aoclsparse_int              n,
                    aoclsparse_int              nnz,
                    aoclsparse_index_base       base,
                    aoclsparse_matrix_data_type val_type,
                    aoclsparse_int              ell_width,
                    aoclsparse_int              ell_m)
            : base_mtx(m, n, nnz, aoclsparse_ellt_csr_hyb_mat, base, val_type, true)
            , ell_width(ell_width)
            , ell_m(ell_m)
        {
            // Validate dimensions before allocation
            if(m < ell_m || ell_width < 0)
                throw std::bad_array_new_length();
            try
            {
                ell_col_ind    = new aoclsparse_int[ell_width * m];
                csr_row_id_map = new aoclsparse_int[m - ell_m];
                ell_val        = ::operator new(data_size[val_type] * ell_width * m);
            }
            catch(...)
            {
                delete[] ell_col_ind;
                delete[] csr_row_id_map;
                ::operator delete(ell_val);
                throw;
            }
        }

        // Parameterized constructor that sets up the matrix using user-provided, pre-allocated data arrays.
        ell_csr_hyb(aoclsparse_int              m,
                    aoclsparse_int              n,
                    aoclsparse_int              nnz,
                    aoclsparse_index_base       base,
                    aoclsparse_matrix_data_type val_type,
                    aoclsparse_int              ell_width,
                    aoclsparse_int              ell_m,
                    aoclsparse_int             *ell_col_ind,
                    void                       *ell_val,
                    aoclsparse_int             *csr_row_id_map,
                    aoclsparse_int             *csr_col_ptr,
                    void                       *csr_val)
            : base_mtx(m, n, nnz, aoclsparse_ellt_csr_hyb_mat, base, val_type, false)
            , ell_width(ell_width)
            , ell_m(ell_m)
            , ell_col_ind(ell_col_ind)
            , ell_val(ell_val)
            , csr_row_id_map(csr_row_id_map)
            , csr_col_ptr(csr_col_ptr)
            , csr_val(csr_val)
        {
        }
        // Destructor
        ~ell_csr_hyb()
        {
            // Free the memory allocated if the matrix was internally allocated (is_internal = true)
            if(is_internal)
            {
                delete[] ell_col_ind;
                delete[] csr_row_id_map;
                ::operator delete(ell_val);
            }
        }
    };

    /********************************************************************************
     * \brief coo is a class holding the aoclsparse matrix
     * in COO format. It is used internally during the optimization process.
     * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
     *******************************************************************************/
    class coo : public base_mtx
    {
        // COO matrix part
    public:
        aoclsparse_int *row_ind = nullptr;
        aoclsparse_int *col_ind = nullptr;
        void           *val     = nullptr;

        // Parameterized constructor
        coo(aoclsparse_int              m,
            aoclsparse_int              n,
            aoclsparse_int              nnz,
            aoclsparse_index_base       base,
            aoclsparse_matrix_data_type val_type)
            : base_mtx(m, n, nnz, aoclsparse_coo_mat, base, val_type, true)
        {
            // Validate dimensions before allocation
            if(nnz < 0)
                throw std::bad_array_new_length();
            try
            {
                row_ind = new aoclsparse_int[nnz];
                col_ind = new aoclsparse_int[nnz];
                val     = ::operator new(data_size[val_type] * nnz);
            }
            catch(...)
            {
                delete[] row_ind;
                delete[] col_ind;
                ::operator delete(val);
                throw;
            }
        }

        // Parameterized constructor that sets up the matrix using user-provided, pre-allocated data arrays.
        coo(aoclsparse_int              m,
            aoclsparse_int              n,
            aoclsparse_int              nnz,
            aoclsparse_index_base       base,
            aoclsparse_matrix_data_type val_type,
            aoclsparse_int             *row_ind,
            aoclsparse_int             *col_ind,
            void                       *val)
            : base_mtx(m, n, nnz, aoclsparse_coo_mat, base, val_type, false)
            , row_ind(row_ind)
            , col_ind(col_ind)
            , val(val)
        {
        }

        // Destructor
        ~coo()
        {
            if(is_internal)
            {
                delete[] row_ind;
                delete[] col_ind;
                ::operator delete(val);
            }
        }
    };
}

/********************************************************************************
 * \brief _aoclsparse_ilu is a structure holding data members for ILU operation.
 * It is used internally during the optimization process which includes ILU factorization.
 * It should be destroyed at the end using aoclsparse_destroy_mat_structs().
 *******************************************************************************/
struct _aoclsparse_ilu
{
    aoclsparse_int     *lu_diag_ptr     = NULL; //pointer to diagonal elements in csr values array
    void               *precond_csr_val = NULL; //copy of buffer for ilu precondioned factors
    bool                ilu_factorized  = false; //flag to indicate if ILU factorization is done
    aoclsparse_ilu_type ilu_fact_type; // indicator of ILU factorization type
    // true: ILU Optimization/Working-Buffer-Allocation already done, else needs to be performed^M
    bool ilu_ready = false;
};

/********************************************************************************
 * \brief _aoclsparse_symgs is a structure holding data members for SYMGS operation.
 * It holds working buffers such as residual buffer for Gauss Seidel operatoin
 * It should be destroyed at the end using aoclsparse_destroy_symgs().
 *******************************************************************************/
struct _aoclsparse_symgs
{
    void *r = NULL; //residual
    void *q = NULL; //temporary buffer
    // true: SGS Optimization/Working-Buffer-Allocation already done, else needs to be performed
    bool sgs_ready = false;
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

    // Holds all matrices, regardless of type, including internally created matrices and copies
    std::vector<aoclsparse::base_mtx *> mats;

    //ilu members
    struct _aoclsparse_ilu ilu_info;

    //symgs
    struct _aoclsparse_symgs symgs_info;

    // the original matrix had full (nonzero) diagonal, so the matrix
    // is safe for TRSVs
    bool opt_csr_full_diag = false;
    // store if the matrix has already been optimized for this blocked SpMV
    bool blk_optimized = false;

    // used to indicate if any additional memory required further performance optimization purposes
    aoclsparse_memory_usage mem_policy = aoclsparse_memory_usage_unrestricted;

    // check if the matrix has full(nonzero) diagonal
    // if the matrix is rectangular, only the square submatrix is considered
    bool fulldiag = false;
    // the sorting pattern of the matrix
    aoclsparse_matrix_sort sort = aoclsparse_unknown_sort;
};

#endif // AOCLSPARSE_MAT_STRUCTS_H
