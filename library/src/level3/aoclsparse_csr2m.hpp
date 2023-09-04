/* ************************************************************************
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLSPARSE_CSR2M_HPP
#define AOCLSPARSE_CSR2M_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>
/*
 * This function performs the first stage of matrix-matrix multiplication,
 * which computes the row pointer values and number of non-zeroes in resultant
 * sparse matrix. This function also allocates memory for CSR arrays of output matrix.
 * Input Parameter opflag denotes operations on A and B matrices, set as
 * 0 if A*B , 1 if At*B , 2 if A*Bt, 3 if At*Bt from the caller function.
 * */
template <typename T>
aoclsparse_status aoclsparse_csr2m_nnz_count(aoclsparse_int             m,
                                             aoclsparse_int             n,
                                             const aoclsparse_mat_descr descrA,
                                             const aoclsparse_int      *csr_row_ptr_A,
                                             const aoclsparse_int      *csr_col_ind_A,
                                             const aoclsparse_mat_descr descrB,
                                             const aoclsparse_int      *csr_row_ptr_B,
                                             const aoclsparse_int      *csr_col_ind_B,
                                             aoclsparse_matrix         *C,
                                             aoclsparse_int             opflag)

{
    // Check for valid matrix descriptors
    if((descrA == nullptr) || (descrB == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    if((csr_row_ptr_A == nullptr) || (csr_col_ind_A == nullptr) || (csr_row_ptr_B == nullptr)
       || (csr_col_ind_B == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_index_base       baseA = descrA->base;
    aoclsparse_index_base       baseB = descrB->base;
    std::vector<aoclsparse_int> nnz;
    try
    {
        nnz.resize(n, -1);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    aoclsparse_int  nnz_C         = 0;
    aoclsparse_int *csr_row_ptr_C = nullptr;
    try
    {
        csr_row_ptr_C = new aoclsparse_int[m + 1];
    }
    catch(std::bad_alloc &)
    {
        /* Memory  allocation fail*/
        return aoclsparse_status_memory_error;
    }

    csr_row_ptr_C[0] = 0;

    aoclsparse_int num_nonzeros = 0;
    // Loop over rows of A
    for(aoclsparse_int i = 0; i < m; i++)
    {

        // Loop over columns of A
        for(aoclsparse_int j = (csr_row_ptr_A[i] - baseA); j < (csr_row_ptr_A[i + 1] - baseA); j++)
        {
            // Current column of A
            aoclsparse_int col_A   = csr_col_ind_A[j] - baseA;
            aoclsparse_int nnz_row = csr_row_ptr_B[col_A + 1] - csr_row_ptr_B[col_A];
            aoclsparse_int k_iter  = nnz_row / 4;
            aoclsparse_int k_rem   = nnz_row % 4;
            aoclsparse_int row_B   = csr_row_ptr_B[col_A] - baseB;

            // Loop over columns of B in row j in groups of 4
            for(aoclsparse_int k = 0; k < k_iter * 4; k += 4)
            {
                // Current column of B
                aoclsparse_int col_B = csr_col_ind_B[row_B + k] - baseB;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    num_nonzeros++;
                }

                // Current column of B
                col_B = csr_col_ind_B[row_B + k + 1] - baseB;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    num_nonzeros++;
                }

                // Current column of B
                col_B = csr_col_ind_B[row_B + k + 2] - baseB;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    num_nonzeros++;
                }

                // Current column of B
                col_B = csr_col_ind_B[row_B + k + 3] - baseB;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    num_nonzeros++;
                }
            }
            // Loop over remaining columns of B in row j
            for(aoclsparse_int k = 0; k < k_rem; k++)
            {
                // Current column of B
                aoclsparse_int col_B = csr_col_ind_B[row_B + (k_iter * 4) + k] - baseB;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    num_nonzeros++;
                }
            }
        }
        csr_row_ptr_C[i + 1] = num_nonzeros;
    }

    // Number of non-zeroes of resultant matrix C
    nnz_C = csr_row_ptr_C[m];

    // Creates a new resultant matrix C
    // And allocates memory for column index and value
    // arrays of resultant matrix C
    aoclsparse_int *csr_col_ind_C = nullptr;
    void           *csr_val_C     = nullptr;
    try
    {
        *C            = new _aoclsparse_matrix;
        csr_col_ind_C = new aoclsparse_int[nnz_C];
        csr_val_C     = ::operator new(nnz_C * sizeof(T));
    }
    catch(std::bad_alloc &)
    {
        // Insufficient memory for output allocation
        delete *C;
        delete[] csr_col_ind_C;
        delete[] csr_row_ptr_C;
        ::operator delete(csr_val_C);
        *C = nullptr;
        return aoclsparse_status_memory_error;
    }

    // For At * Bt = (B * A)t, Resultant matrix will be represented internally as CSC,
    // It should be transposed back to CSR representation after finalize stage.
    if(opflag == 3)
    {
        aoclsparse_init_mat(*C, aoclsparse_index_base_zero, n, m, nnz_C, aoclsparse_csc_mat);
        (*C)->val_type = get_data_type<T>();
        // Assign the resultant C matrix arrays to CSC format
        (*C)->csc_mat.col_ptr = csr_row_ptr_C;
        (*C)->csc_mat.row_idx = csr_col_ind_C;
        (*C)->csc_mat.val     = csr_val_C;
        // Allocate memory for CSR arrays here
        try
        {
            (*C)->csr_mat.csr_row_ptr = new aoclsparse_int[n + 1];
            (*C)->csr_mat.csr_col_ptr = new aoclsparse_int[nnz_C];
            (*C)->csr_mat.csr_val     = ::operator new(nnz_C * sizeof(T));
        }
        catch(std::bad_alloc &)
        {
            aoclsparse_destroy(*C);
            return aoclsparse_status_memory_error;
        }
    }
    // For A*B, At*B and A*Bt, Resultant matrix is represented as CSR,
    // Assign the resultant C matrix arrays to CSR format
    else
    {
        aoclsparse_init_mat(*C, aoclsparse_index_base_zero, m, n, nnz_C, aoclsparse_csr_mat);
        (*C)->input_format        = aoclsparse_csr_mat;
        (*C)->val_type            = get_data_type<T>();
        (*C)->csr_mat.csr_row_ptr = csr_row_ptr_C;
        (*C)->csr_mat.csr_col_ptr = csr_col_ind_C;
        (*C)->csr_mat.csr_val     = csr_val_C;
    }
    return aoclsparse_status_success;
}

// This function finalize computation. Can also be used when the matrix
// structure remains unchanged and only values of the resulting matrix C
// need to be recomputed.
template <typename T>
aoclsparse_status aoclsparse_csr2m_finalize(aoclsparse_int             m_a,
                                            aoclsparse_int             n_b,
                                            aoclsparse_operation       opA,
                                            const aoclsparse_mat_descr descrA,
                                            const aoclsparse_int      *csr_row_ptr_A,
                                            const aoclsparse_int      *csr_col_ind_A,
                                            const T                   *csr_val_A,
                                            aoclsparse_operation       opB,
                                            const aoclsparse_mat_descr descrB,
                                            const aoclsparse_int      *csr_row_ptr_B,
                                            const aoclsparse_int      *csr_col_ind_B,
                                            const T                   *csr_val_B,
                                            aoclsparse_matrix         *C,
                                            aoclsparse_int             opflag)
{
    // Check for valid pointers
    if((descrA == nullptr) || (descrB == nullptr) || (*C == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    if((csr_row_ptr_A == nullptr) || (csr_col_ind_A == nullptr) || (csr_val_A == nullptr)
       || (csr_row_ptr_B == nullptr) || (csr_col_ind_B == nullptr) || (csr_val_B == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(((*C)->csr_mat.csr_row_ptr == nullptr) || ((*C)->csr_mat.csr_col_ptr == nullptr)
       || ((*C)->csr_mat.csr_val == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Retrieve the C matrix values and array addresses generated
    // in nnz_count stage.
    aoclsparse_int  m;
    aoclsparse_int  n;
    aoclsparse_int  nnz_C         = (*C)->nnz;
    aoclsparse_int *csr_row_ptr_C = NULL;
    aoclsparse_int *csr_col_ind_C = NULL;
    T              *csr_val_C     = NULL;

    // For At * Bt = (B * A)t, Check for valid CSR and CSC arrays pointers
    // Retrieve CSC matrix arrays from C for finalizing multiplication.
    if(opflag == 3)
    {
        m = (*C)->n;
        n = (*C)->m;
        if(((*C)->csc_mat.col_ptr == nullptr) || ((*C)->csc_mat.row_idx == nullptr)
           || ((*C)->csc_mat.val == nullptr))
            return aoclsparse_status_invalid_pointer;
        csr_row_ptr_C = (*C)->csc_mat.col_ptr;
        csr_col_ind_C = (*C)->csc_mat.row_idx;
        csr_val_C     = (T *)(*C)->csc_mat.val;
    }
    // For A*B, At*B and A*Bt, Check for valid CSR arrays pointers
    // Retrieve CSR matrix arrays from C
    else
    {
        m             = (*C)->m;
        n             = (*C)->n;
        csr_row_ptr_C = (*C)->csr_mat.csr_row_ptr;
        csr_col_ind_C = (*C)->csr_mat.csr_col_ptr;
        csr_val_C     = (T *)(*C)->csr_mat.csr_val;
    }

    // Check if C matrix sizes retrieved from matrix structure is valid
    if((m != m_a) || (n != n_b))
        return aoclsparse_status_invalid_size;

    // Check for valid matrix descriptors
    if((descrA == nullptr) || (descrB == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_index_base       baseA = descrA->base;
    aoclsparse_index_base       baseB = descrB->base;
    std::vector<aoclsparse_int> nnz;
    std::vector<T>              sum;
    try
    {
        nnz.resize(n, -1);
        sum.resize(n, 0.0);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    // Loop over rows of A
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int row_begin_A = csr_row_ptr_A[i] - baseA;
        aoclsparse_int row_end_A   = csr_row_ptr_A[i + 1] - baseA;

        aoclsparse_int idxC = csr_row_ptr_C[i]; // where to write first element in this row in C
        // Loop over columns of A
        for(aoclsparse_int j = row_begin_A; j < row_end_A; j++)
        {
            // Current column of A
            aoclsparse_int col_A = csr_col_ind_A[j] - baseA;
            // Current value of A
            T val_A;
            if(opA == aoclsparse_operation_conjugate_transpose)
                val_A = aoclsparse::conj(csr_val_A[j]);
            else
                val_A = csr_val_A[j];

            aoclsparse_int row_begin_B = csr_row_ptr_B[col_A] - baseB;
            aoclsparse_int row_end_B   = csr_row_ptr_B[col_A + 1] - baseB;

            // Loop over columns of B in row col_A
            for(aoclsparse_int k = row_begin_B; k < row_end_B; k++)
            {
                // Current column of B
                aoclsparse_int col_B = csr_col_ind_B[k] - baseB;
                // Current value of B
                T val_B;
                if(opB == aoclsparse_operation_conjugate_transpose)
                    val_B = aoclsparse::conj(csr_val_B[k]);
                else
                    val_B = csr_val_B[k];
                if(nnz[col_B] != i)
                {
                    // create new element in C of index col_B
                    nnz[col_B]            = i;
                    csr_col_ind_C[idxC++] = col_B;
                    sum[col_B]            = val_A * val_B;
                }
                else // the element already exist, just added in sum
                    sum[col_B] = sum[col_B] + val_A * val_B;
            }
        }

        //  Check if the computed nonzeroes matches what we expect in the row
        if(idxC != csr_row_ptr_C[i + 1])
            return aoclsparse_status_internal_error;

        // copy values from sum to csr_val_C based on csr_col_ind_C
        for(idxC = csr_row_ptr_C[i]; idxC < csr_row_ptr_C[i + 1]; idxC++)
            csr_val_C[idxC] = sum[csr_col_ind_C[idxC]];
    }
    if(opflag == 3)
    {
        /* Transpose the results A^T*B^T = (B*A)^T */
        _aoclsparse_mat_descr descrC;
        descrC.base              = (*C)->base;
        aoclsparse_status status = aoclsparse_csr2csc_template(m,
                                                               n,
                                                               nnz_C,
                                                               &descrC,
                                                               descrC.base,
                                                               csr_row_ptr_C,
                                                               csr_col_ind_C,
                                                               csr_val_C,
                                                               (*C)->csr_mat.csr_col_ptr,
                                                               (*C)->csr_mat.csr_row_ptr,
                                                               (T *)(*C)->csr_mat.csr_val);
        if(status != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        (*C)->input_format = aoclsparse_csr_mat;
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csr2m_t(aoclsparse_operation       opA,
                                     const aoclsparse_mat_descr descrA,
                                     const aoclsparse_matrix    A,
                                     aoclsparse_operation       opB,
                                     const aoclsparse_mat_descr descrB,
                                     const aoclsparse_matrix    B,
                                     aoclsparse_request         request,
                                     aoclsparse_matrix         *C)
{
    aoclsparse_status status = aoclsparse_status_success;
    // Check for valid handle and matrix descriptor
    if((descrA == nullptr) || (descrB == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }

    if((A == nullptr) || (B == nullptr) || (C == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Initialise *C to nullptr for full_computation & first stage
    if(request != aoclsparse_stage_finalize)
    {
        *C = nullptr;
    }
    if((A->input_format != aoclsparse_csr_mat) || (B->input_format != aoclsparse_csr_mat))
    {
        return aoclsparse_status_invalid_value;
    }

    if(A->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }

    if(B->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }

    // Check index base
    if(descrA->base != aoclsparse_index_base_zero && descrA->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    if(descrB->base != aoclsparse_index_base_zero && descrB->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    if(A->base != descrA->base)
        return aoclsparse_status_invalid_value;

    if(B->base != descrB->base)
        return aoclsparse_status_invalid_value;

    if((descrA->type != aoclsparse_matrix_type_general)
       || (descrB->type != aoclsparse_matrix_type_general))
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }
    // For double and float , conjugate transpose is same as transpose
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        if(opA == aoclsparse_operation_conjugate_transpose)
        {
            opA = aoclsparse_operation_transpose;
        }
        if(opB == aoclsparse_operation_conjugate_transpose)
        {
            opB = aoclsparse_operation_transpose;
        }
    }

    aoclsparse_int m_a, n_a, m_b, n_b;
    aoclsparse_int opflag = 0;
    // A * B
    if((opA == aoclsparse_operation_none) && (opB == aoclsparse_operation_none))
    {
        m_a    = A->m;
        n_a    = A->n;
        m_b    = B->m;
        n_b    = B->n;
        opflag = 0;
    }
    // At * B
    else if(((opA == aoclsparse_operation_transpose)
             || (opA == aoclsparse_operation_conjugate_transpose))
            && (opB == aoclsparse_operation_none))
    {
        m_a    = A->n;
        n_a    = A->m;
        m_b    = B->m;
        n_b    = B->n;
        opflag = 1;
    }
    // A * Bt
    else if((opA == aoclsparse_operation_none)
            && ((opB == aoclsparse_operation_transpose)
                || (opB == aoclsparse_operation_conjugate_transpose)))
    {
        m_a    = A->m;
        n_a    = A->n;
        m_b    = B->n;
        n_b    = B->m;
        opflag = 2;
    }
    // At * Bt
    else if(((opA == aoclsparse_operation_transpose)
             || (opA == aoclsparse_operation_conjugate_transpose))
            && ((opB == aoclsparse_operation_transpose)
                || (opB == aoclsparse_operation_conjugate_transpose)))
    {
        m_a    = A->n;
        n_a    = A->m;
        m_b    = B->n;
        n_b    = B->m;
        opflag = 3;
    }
    // Invalid operation value
    else
        return aoclsparse_status_invalid_value;

    // Invalid size for matrix multiplication
    if(n_a != m_b)
        return aoclsparse_status_invalid_size;

    // Quick return for size 0 matrices, Do nothing
    // Return Valid Non-NULL pointers of C array.
    if((m_a == 0) || (n_a == 0) || (n_b == 0) || (A->nnz == 0) || (B->nnz == 0))
    {
        if(*C == nullptr)
        {
            try
            {
                *C                        = new _aoclsparse_matrix;
                (*C)->csr_mat.csr_row_ptr = new aoclsparse_int[m_a + 1]();
                (*C)->csr_mat.csr_col_ptr = new aoclsparse_int[0];
                (*C)->csr_mat.csr_val     = ::operator new(0);
            }
            catch(std::bad_alloc &)
            {
                /*Insufficient memory for output allocation */
                aoclsparse_destroy(*C);
                return aoclsparse_status_memory_error;
            }
            aoclsparse_init_mat(*C, aoclsparse_index_base_zero, m_a, n_b, 0, aoclsparse_csr_mat);
            (*C)->val_type = get_data_type<T>();
        }
        return aoclsparse_status_success;
    }
    _aoclsparse_mat_descr descrA_t;
    aoclsparse_copy_mat_descr(&descrA_t, descrA);
    _aoclsparse_mat_descr descrB_t;
    aoclsparse_copy_mat_descr(&descrB_t, descrB);
    aoclsparse_int *csr_row_ptr_A = NULL;
    aoclsparse_int *csr_col_ind_A = NULL;
    T              *csr_val_A     = NULL;
    aoclsparse_int *csr_row_ptr_B = NULL;
    aoclsparse_int *csr_col_ind_B = NULL;
    T              *csr_val_B     = NULL;
    // A * B , Retrieve A and B CSR arrays
    if(opflag == 0)
    {
        csr_row_ptr_A = A->csr_mat.csr_row_ptr;
        csr_col_ind_A = A->csr_mat.csr_col_ptr;
        csr_val_A     = (T *)A->csr_mat.csr_val;
        csr_row_ptr_B = B->csr_mat.csr_row_ptr;
        csr_col_ind_B = B->csr_mat.csr_col_ptr;
        csr_val_B     = (T *)B->csr_mat.csr_val;
    }
    // At * B , Transpose A CSR arrays
    // Retrieve B CSR arrays
    else if(opflag == 1)
    {
        try
        {
            csr_row_ptr_A = new aoclsparse_int[A->n + 1];
            csr_col_ind_A = new aoclsparse_int[A->nnz];
            csr_val_A     = new T[A->nnz];
        }
        catch(std::bad_alloc &)
        {
            /* Memory  allocation fail*/
            delete[] csr_row_ptr_A;
            delete[] csr_col_ind_A;
            delete[] csr_val_A;
            return aoclsparse_status_memory_error;
        }

        aoclsparse_status status = aoclsparse_csr2csc_template(A->m,
                                                               A->n,
                                                               A->nnz,
                                                               &descrA_t,
                                                               descrA_t.base,
                                                               A->csr_mat.csr_row_ptr,
                                                               A->csr_mat.csr_col_ptr,
                                                               (const T *)A->csr_mat.csr_val,
                                                               csr_col_ind_A,
                                                               csr_row_ptr_A,
                                                               csr_val_A);
        if(status != aoclsparse_status_success)
        {
            delete[] csr_row_ptr_A;
            delete[] csr_col_ind_A;
            delete[] csr_val_A;
            return aoclsparse_status_memory_error;
        }
        csr_row_ptr_B = B->csr_mat.csr_row_ptr;
        csr_col_ind_B = B->csr_mat.csr_col_ptr;
        csr_val_B     = (T *)B->csr_mat.csr_val;
    }
    // A * Bt , Transpose B CSR arrays
    // Retrieve A CSR arrays
    else if(opflag == 2)
    {
        csr_row_ptr_A = A->csr_mat.csr_row_ptr;
        csr_col_ind_A = A->csr_mat.csr_col_ptr;
        csr_val_A     = (T *)A->csr_mat.csr_val;
        try
        {
            csr_row_ptr_B = new aoclsparse_int[B->n + 1];
            csr_col_ind_B = new aoclsparse_int[B->nnz];
            csr_val_B     = new T[B->nnz];
        }
        catch(std::bad_alloc &)
        {
            /* Memory  allocation fail*/
            delete[] csr_row_ptr_B;
            delete[] csr_col_ind_B;
            delete[] csr_val_B;
            return aoclsparse_status_memory_error;
        }

        aoclsparse_status status = aoclsparse_csr2csc_template(B->m,
                                                               B->n,
                                                               B->nnz,
                                                               &descrB_t,
                                                               descrB_t.base,
                                                               B->csr_mat.csr_row_ptr,
                                                               B->csr_mat.csr_col_ptr,
                                                               (const T *)B->csr_mat.csr_val,
                                                               csr_col_ind_B,
                                                               csr_row_ptr_B,
                                                               csr_val_B);
        if(status != aoclsparse_status_success)
        {
            delete[] csr_row_ptr_B;
            delete[] csr_col_ind_B;
            delete[] csr_val_B;
            return aoclsparse_status_memory_error;
        }
    }
    // At * Bt, Swap A and B matrix arrays as At*Bt = (B*A)t
    // Swap descriptors and operations of A and B
    else if(opflag == 3)
    {
        csr_row_ptr_A = B->csr_mat.csr_row_ptr;
        csr_col_ind_A = B->csr_mat.csr_col_ptr;
        csr_val_A     = (T *)B->csr_mat.csr_val;
        csr_row_ptr_B = A->csr_mat.csr_row_ptr;
        csr_col_ind_B = A->csr_mat.csr_col_ptr;
        csr_val_B     = (T *)A->csr_mat.csr_val;
        _aoclsparse_mat_descr tmp;
        tmp                       = descrA_t;
        descrA_t                  = descrB_t;
        descrB_t                  = tmp;
        aoclsparse_operation temp = opA;
        opA                       = opB;
        opB                       = temp;
        aoclsparse_int t          = m_a;
        m_a                       = n_b;
        n_b                       = t;
    }

    switch(request)
    {

    case aoclsparse_stage_nnz_count:
    {
        status = aoclsparse_csr2m_nnz_count<T>(m_a,
                                               n_b,
                                               &descrA_t,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               &descrB_t,
                                               csr_row_ptr_B,
                                               csr_col_ind_B,
                                               C,
                                               opflag);
        break;
    }
    case aoclsparse_stage_finalize:
    {

        status = aoclsparse_csr2m_finalize(m_a,
                                           n_b,
                                           opA,
                                           &descrA_t,
                                           csr_row_ptr_A,
                                           csr_col_ind_A,
                                           csr_val_A,
                                           opB,
                                           &descrB_t,
                                           csr_row_ptr_B,
                                           csr_col_ind_B,
                                           csr_val_B,
                                           C,
                                           opflag);

        break;
    }
    case aoclsparse_stage_full_computation:
    {
        status = aoclsparse_csr2m_nnz_count<T>(m_a,
                                               n_b,
                                               &descrA_t,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               &descrB_t,
                                               csr_row_ptr_B,
                                               csr_col_ind_B,
                                               C,
                                               opflag);

        if(status == aoclsparse_status_success)
        {
            status = aoclsparse_csr2m_finalize(m_a,
                                               n_b,
                                               opA,
                                               &descrA_t,
                                               csr_row_ptr_A,
                                               csr_col_ind_A,
                                               csr_val_A,
                                               opB,
                                               &descrB_t,
                                               csr_row_ptr_B,
                                               csr_col_ind_B,
                                               csr_val_B,
                                               C,
                                               opflag);
        }
        break;
    }
    default:
        status = aoclsparse_status_invalid_value;
    }
    // Memory free of local arrays
    if(opflag == 1)
    {
        delete[] csr_row_ptr_A;
        csr_row_ptr_A = NULL;
        delete[] csr_col_ind_A;
        csr_col_ind_A = NULL;
        delete[] csr_val_A;
        csr_val_A = NULL;
    }
    else if(opflag == 2)
    {
        delete[] csr_row_ptr_B;
        csr_row_ptr_B = NULL;
        delete[] csr_col_ind_B;
        csr_col_ind_B = NULL;
        delete[] csr_val_B;
        csr_val_B = NULL;
    }
    return status;
}

#endif /* AOCLSPARSE_CSR2M_HPP*/
