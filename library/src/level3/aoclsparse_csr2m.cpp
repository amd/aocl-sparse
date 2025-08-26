/* ************************************************************************
 * Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse.hpp"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
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
    using namespace aoclsparse;
    aoclsparse_int status = aoclsparse_status_success;
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
    aoclsparse_index_base baseA = descrA->base;
    aoclsparse_index_base baseB = descrB->base;

    aoclsparse_int   nnz_C = 0;
    aoclsparse::csr *csr_C = nullptr;
    try
    {
        // Set base to zero for internal consistency - all computations use zero-based indexing
        csr_C = new aoclsparse::csr(
            m, n, -1, aoclsparse_csr_mat, aoclsparse_index_base_zero, get_data_type<T>());
    }
    catch(std::bad_alloc &)
    {
        /* Memory  allocation fail*/
        return aoclsparse_status_memory_error;
    }

    csr_C->ptr[0] = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(context::get_context()->get_num_threads()) reduction(max : status)
#endif
    {
#ifdef _OPENMP
        aoclsparse_int num_threads = omp_get_num_threads();
        aoclsparse_int thread_num  = omp_get_thread_num();
        aoclsparse_int start       = m * thread_num / num_threads;
        aoclsparse_int end         = m * (thread_num + 1) / num_threads;
        status                     = aoclsparse_status_success;
#else
        aoclsparse_int start = 0;
        aoclsparse_int end   = m;
#endif
        std::vector<aoclsparse_int> nnz;
        aoclsparse_int              num_nonzeros = 0;
        try
        {
            nnz.resize(n, -1);
        }
        catch(std::bad_alloc &)
        {
            status = aoclsparse_status_memory_error;
        }
        if(status == aoclsparse_status_success)
        {
            // Loop over rows of A
            for(aoclsparse_int i = start; i < end; i++)
            {
                num_nonzeros = 0;

                // Loop over columns of A
                for(aoclsparse_int j = (csr_row_ptr_A[i] - baseA);
                    j < (csr_row_ptr_A[i + 1] - baseA);
                    j++)
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
                csr_C->ptr[i + 1] = num_nonzeros;
            }
        }
    }
    if(status == aoclsparse_status_success)
    {
        for(aoclsparse_int i = 1; i < m + 1; i++)
        {
            csr_C->ptr[i] += csr_C->ptr[i - 1];
        }

        // Number of non-zeroes of resultant matrix C
        nnz_C      = csr_C->ptr[m];
        csr_C->nnz = nnz_C;

        // Creates a new resultant matrix C
        // And allocates memory for column index and value
        // arrays of resultant matrix C
        try
        {
            *C         = new _aoclsparse_matrix;
            csr_C->ind = new aoclsparse_int[nnz_C];
            csr_C->val = ::operator new(nnz_C * sizeof(T));
            (*C)->mats.push_back(csr_C);
        }
        catch(std::bad_alloc &)
        {
            // Insufficient memory for output allocation
            delete csr_C;
            aoclsparse_destroy(C);
            return aoclsparse_status_memory_error;
        }

        // For At * Bt = (B * A)t, Resultant matrix will be represented internally as CSC,
        // It should be transposed back to CSR representation after finalize stage.
        if(opflag == 3)
        {
            aoclsparse_init_mat(*C, n, m, nnz_C, aoclsparse_csr_mat);
            (*C)->val_type = get_data_type<T>();
            // Assign the resultant C matrix arrays to CSC format
            csr_C->doid = aoclsparse::doid::gt;
            // Allocate memory for CSR arrays here
            aoclsparse::csr *csr_mat = nullptr;
            try
            {
                csr_mat = new aoclsparse::csr(n,
                                              m,
                                              nnz_C,
                                              aoclsparse_csr_mat,
                                              aoclsparse_index_base_zero,
                                              get_data_type<T>());
                (*C)->mats.push_back(csr_mat);
            }
            catch(std::bad_alloc &)
            {
                if(csr_mat)
                    delete csr_mat;
                aoclsparse_destroy(C);
                return aoclsparse_status_memory_error;
            }
        }
        // For A*B, At*B and A*Bt, Resultant matrix is represented as CSR,
        // Assign the resultant C matrix arrays to CSR format
        else
        {
            aoclsparse_init_mat(*C, m, n, nnz_C, aoclsparse_csr_mat);
            (*C)->input_format = aoclsparse_csr_mat;
            (*C)->val_type     = get_data_type<T>();
        }
    }
    else
    {
        delete csr_C;
    }
    return (aoclsparse_status)status;
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
    using namespace aoclsparse;
    aoclsparse_int status = aoclsparse_status_success;

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

    if((*C)->mats.empty())
        return aoclsparse_status_invalid_pointer;

    aoclsparse::csr *csr_mat = nullptr, *csc_mat = nullptr;

    // Find CSR and CSC matrices in (*C)->mats
    for(auto *mat : (*C)->mats)
    {
        if(auto *temp_mat = dynamic_cast<aoclsparse::csr *>(mat))
        {
            bool is_csc = (temp_mat->doid == aoclsparse::doid::gt);
            if(!is_csc && csr_mat == nullptr)
                csr_mat = temp_mat;
            else if(is_csc && csc_mat == nullptr)
                csc_mat = temp_mat;
            // Early exit if both found
            if(csr_mat && csc_mat)
                break;
        }
    }

    if((csr_mat == nullptr) || (csr_mat->ptr == nullptr) || (csr_mat->ind == nullptr)
       || (csr_mat->val == nullptr))
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
        if((csc_mat->ptr == nullptr) || (csc_mat->ind == nullptr) || (csc_mat->val == nullptr))
            return aoclsparse_status_invalid_pointer;
        csr_row_ptr_C = csc_mat->ptr;
        csr_col_ind_C = csc_mat->ind;
        csr_val_C     = (T *)csc_mat->val;
    }
    // For A*B, At*B and A*Bt, Check for valid CSR arrays pointers
    // Retrieve CSR matrix arrays from C
    else
    {
        m             = (*C)->m;
        n             = (*C)->n;
        csr_row_ptr_C = csr_mat->ptr;
        csr_col_ind_C = csr_mat->ind;
        csr_val_C     = (T *)csr_mat->val;
    }

    // Check if C matrix sizes retrieved from matrix structure is valid
    if((m != m_a) || (n != n_b))
        return aoclsparse_status_invalid_size;

    // Check for valid matrix descriptors
    if((descrA == nullptr) || (descrB == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_index_base baseA = descrA->base;
    aoclsparse_index_base baseB = descrB->base;
#ifdef _OPENMP
#pragma omp parallel num_threads(context::get_context()->get_num_threads()) reduction(max : status)
#endif
    {
#ifdef _OPENMP
        aoclsparse_int num_threads = omp_get_num_threads();
        aoclsparse_int thread_num  = omp_get_thread_num();
        aoclsparse_int start       = m * thread_num / num_threads;
        aoclsparse_int end         = m * (thread_num + 1) / num_threads;
        status                     = aoclsparse_status_success;
#else
        aoclsparse_int start = 0;
        aoclsparse_int end   = m;
#endif
        std::vector<aoclsparse_int> nnz;
        std::vector<T>              sum;
        try
        {
            nnz.resize(n, -1);
            sum.resize(n, 0.0);
        }
        catch(std::bad_alloc &)
        {
            status = aoclsparse_status_memory_error;
        }
        if(status == aoclsparse_status_success)
        {
            // Loop over rows of A
            for(aoclsparse_int i = start; i < end; i++)
            {
                aoclsparse_int row_begin_A = csr_row_ptr_A[i] - baseA;
                aoclsparse_int row_end_A   = csr_row_ptr_A[i + 1] - baseA;

                aoclsparse_int idxC
                    = csr_row_ptr_C[i]; // where to write first element in this row in C
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
                    status = aoclsparse_status_internal_error;
                else
                {
                    // copy values from sum to csr_val_C based on csr_col_ind_C
                    for(idxC = csr_row_ptr_C[i]; idxC < csr_row_ptr_C[i + 1]; idxC++)
                        csr_val_C[idxC] = sum[csr_col_ind_C[idxC]];
                }

                // copy values from sum to csr_val_C based on csr_col_ind_C
                for(idxC = csr_row_ptr_C[i]; idxC < csr_row_ptr_C[i + 1]; idxC++)
                    csr_val_C[idxC] = sum[csr_col_ind_C[idxC]];
            }
        }
    }
    if(status == aoclsparse_status_success)
    {
        if(opflag == 3)
        {
            /* Transpose the results A^T*B^T = (B*A)^T */
            status = aoclsparse_csr2csc_template(m,
                                                 n,
                                                 nnz_C,
                                                 csc_mat->base,
                                                 csc_mat->base,
                                                 csr_row_ptr_C,
                                                 csr_col_ind_C,
                                                 csr_val_C,
                                                 csr_mat->ind,
                                                 csr_mat->ptr,
                                                 (T *)csr_mat->val);
            if(status != aoclsparse_status_success)
                status = aoclsparse_status_internal_error;
            else
                (*C)->input_format = aoclsparse_csr_mat;
        }
    }
    return (aoclsparse_status)status;
}

template <typename T>
aoclsparse_status aoclsparse::sp2m(aoclsparse_operation       opA,
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
    if(A->mats.empty() || !A->mats[0] || B->mats.empty() || !B->mats[0])
        return aoclsparse_status_invalid_pointer;
    // Initialise *C to nullptr for full_computation & first stage
    if(request != aoclsparse_stage_finalize)
    {
        *C = nullptr;
    }
    if((A->input_format != aoclsparse_csr_mat) || (B->input_format != aoclsparse_csr_mat))
    {
        return aoclsparse_status_not_implemented;
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

    if(A->mats[0]->base != descrA->base)
        return aoclsparse_status_invalid_value;

    if(B->mats[0]->base != descrB->base)
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
            aoclsparse::csr *csr_mat = nullptr;
            try
            {
                *C      = new _aoclsparse_matrix;
                csr_mat = new aoclsparse::csr(m_a,
                                              n_b,
                                              0,
                                              aoclsparse_csr_mat,
                                              aoclsparse_index_base_zero,
                                              get_data_type<T>());
                (*C)->mats.push_back(csr_mat);
            }
            catch(std::bad_alloc &)
            {
                /*Insufficient memory for output allocation */
                if(csr_mat)
                {
                    delete csr_mat;
                }
                aoclsparse_destroy(C);
                return aoclsparse_status_memory_error;
            }
            aoclsparse_init_mat(*C, m_a, n_b, 0, aoclsparse_csr_mat);
            (*C)->val_type = get_data_type<T>();
        }
        return aoclsparse_status_success;
    }
    _aoclsparse_mat_descr descrA_t;
    aoclsparse_copy_mat_descr(&descrA_t, descrA);
    _aoclsparse_mat_descr descrB_t;
    aoclsparse_copy_mat_descr(&descrB_t, descrB);
    // Locate the first CSR matrix object in the mats vector of A and B
    aoclsparse::csr *csr_src_A = nullptr, *csr_src_B = nullptr;
    for(auto *mat : A->mats)
    {
        if(auto *temp = dynamic_cast<aoclsparse::csr *>(mat))
        {
            if(temp->mat_type == aoclsparse_csr_mat)
            {
                csr_src_A = temp;
                break;
            }
        }
    }
    for(auto *mat : B->mats)
    {
        if(auto *temp = dynamic_cast<aoclsparse::csr *>(mat))
        {
            if(temp->mat_type == aoclsparse_csr_mat)
            {
                csr_src_B = temp;
                break;
            }
        }
    }
    aoclsparse::csr *csr_A = nullptr, *csr_B = nullptr;

    // A * B , Retrieve A and B CSR arrays
    if(opflag == 0)
    {
        csr_A = csr_src_A;
        csr_B = csr_src_B;
    }
    // At * B , Transpose A CSR arrays
    // Retrieve B CSR arrays
    else if(opflag == 1)
    {
        try
        {
            csr_A = new aoclsparse::csr(A->n,
                                        A->m,
                                        A->nnz,
                                        aoclsparse_csr_mat,
                                        aoclsparse_index_base_zero,
                                        get_data_type<T>());
        }
        catch(std::bad_alloc &)
        {
            /* Memory  allocation fail*/
            return aoclsparse_status_memory_error;
        }

        aoclsparse_status status = aoclsparse_csr2csc_template(A->m,
                                                               A->n,
                                                               A->nnz,
                                                               csr_src_A->base,
                                                               csr_src_A->base,
                                                               csr_src_A->ptr,
                                                               csr_src_A->ind,
                                                               (const T *)csr_src_A->val,
                                                               csr_A->ind,
                                                               csr_A->ptr,
                                                               (T *)csr_A->val);
        if(status != aoclsparse_status_success)
        {
            delete csr_A;
            return aoclsparse_status_memory_error;
        }
        csr_B = csr_src_B;
    }
    // A * Bt , Transpose B CSR arrays
    // Retrieve A CSR arrays
    else if(opflag == 2)
    {
        csr_A = csr_src_A;
        try
        {
            csr_B = new aoclsparse::csr(B->n,
                                        B->m,
                                        B->nnz,
                                        aoclsparse_csr_mat,
                                        aoclsparse_index_base_zero,
                                        get_data_type<T>());
        }
        catch(std::bad_alloc &)
        {
            /* Memory  allocation fail*/
            return aoclsparse_status_memory_error;
        }

        aoclsparse_status status = aoclsparse_csr2csc_template(B->m,
                                                               B->n,
                                                               B->nnz,
                                                               csr_src_B->base,
                                                               csr_src_B->base,
                                                               csr_src_B->ptr,
                                                               csr_src_B->ind,
                                                               (const T *)csr_src_B->val,
                                                               csr_B->ind,
                                                               csr_B->ptr,
                                                               (T *)csr_B->val);
        if(status != aoclsparse_status_success)
        {
            delete csr_B;
            return aoclsparse_status_memory_error;
        }
    }
    // At * Bt, Swap A and B matrix arrays as At*Bt = (B*A)t
    // Swap descriptors and operations of A and B
    else if(opflag == 3)
    {
        csr_A = csr_src_B;
        csr_B = csr_src_A;
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
                                               csr_A->ptr,
                                               csr_A->ind,
                                               &descrB_t,
                                               csr_B->ptr,
                                               csr_B->ind,
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
                                           csr_A->ptr,
                                           csr_A->ind,
                                           (T *)csr_A->val,
                                           opB,
                                           &descrB_t,
                                           csr_B->ptr,
                                           csr_B->ind,
                                           (T *)csr_B->val,
                                           C,
                                           opflag);

        break;
    }
    case aoclsparse_stage_full_computation:
    {
        status = aoclsparse_csr2m_nnz_count<T>(m_a,
                                               n_b,
                                               &descrA_t,
                                               csr_A->ptr,
                                               csr_A->ind,
                                               &descrB_t,
                                               csr_B->ptr,
                                               csr_B->ind,
                                               C,
                                               opflag);

        if(status == aoclsparse_status_success)
        {
            status = aoclsparse_csr2m_finalize(m_a,
                                               n_b,
                                               opA,
                                               &descrA_t,
                                               csr_A->ptr,
                                               csr_A->ind,
                                               (T *)csr_A->val,
                                               opB,
                                               &descrB_t,
                                               csr_B->ptr,
                                               csr_B->ind,
                                               (T *)csr_B->val,
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
        delete csr_A;
    }
    else if(opflag == 2)
    {
        delete csr_B;
    }
    return status;
}

#define SP2M_DISPATCHER(SUF)                                                                        \
    template DLL_PUBLIC aoclsparse_status aoclsparse::sp2m<SUF>(aoclsparse_operation       opA,     \
                                                                const aoclsparse_mat_descr descrA,  \
                                                                const aoclsparse_matrix    A,       \
                                                                aoclsparse_operation       opB,     \
                                                                const aoclsparse_mat_descr descrB,  \
                                                                const aoclsparse_matrix    B,       \
                                                                aoclsparse_request         request, \
                                                                aoclsparse_matrix         *C);

INSTANTIATE_FOR_ALL_TYPES(SP2M_DISPATCHER);
