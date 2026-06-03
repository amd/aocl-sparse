/* ************************************************************************
 * Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse.hpp"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_context.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_mtx_dispatcher.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
#include <cstring>
#include <shared_mutex>
#include <vector>
/*
 * This function performs the first stage of matrix-matrix multiplication,
 * which computes the row pointer values and number of non-zeroes in resultant
 * sparse matrix. This function also allocates memory for CSR arrays of output matrix.
 * Input Parameter opflag denotes operations on A and B matrices, set as
 * 0 if A*B , 1 if At*B , 2 if A*Bt, 3 if At*Bt from the caller function.
 * The template parameters BASEA and BASEB are used to check if the input
 * matrices are in 0 or 1 based index format. The default value is set to
 * 0 based index format.
 * */
template <typename T, bool BASEA = false, bool BASEB = false>
inline aoclsparse_status aoclsparse_csr2m_nnz_count(aoclsparse_int             m,
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

                aoclsparse_int jstart = csr_row_ptr_A[i];
                aoclsparse_int jend   = csr_row_ptr_A[i + 1];
                // If baseA is 1
                if constexpr(BASEA)
                {
                    jstart -= baseA;
                    jend -= baseA;
                }

                // Loop over columns of A
                for(aoclsparse_int j = jstart; j < jend; j++)
                {
                    // Current column of A
                    aoclsparse_int col_A = csr_col_ind_A[j];
                    if constexpr(BASEA)
                    {
                        col_A -= baseA;
                    }
                    aoclsparse_int nnz_row = csr_row_ptr_B[col_A + 1] - csr_row_ptr_B[col_A];
                    aoclsparse_int k_iter  = nnz_row / 4;
                    aoclsparse_int k_rem   = nnz_row % 4;
                    aoclsparse_int row_B   = csr_row_ptr_B[col_A];
                    if constexpr(BASEB)
                    {
                        row_B -= baseB;
                    }

                    // Loop over columns of B in row j in groups of 4
                    for(aoclsparse_int k = 0; k < k_iter * 4; k += 4)
                    {
                        // Current column of B
                        aoclsparse_int col_B = csr_col_ind_B[row_B + k];
                        if constexpr(BASEB)
                        {
                            col_B -= baseB;
                        }

                        // Check if a new nnz is generated
                        if(nnz[col_B] != i)
                        {
                            nnz[col_B] = i;
                            num_nonzeros++;
                        }

                        // Current column of B
                        col_B = csr_col_ind_B[row_B + k + 1];
                        if constexpr(BASEB)
                        {
                            col_B -= baseB;
                        }

                        // Check if a new nnz is generated
                        if(nnz[col_B] != i)
                        {
                            nnz[col_B] = i;
                            num_nonzeros++;
                        }

                        // Current column of B
                        col_B = csr_col_ind_B[row_B + k + 2];
                        if constexpr(BASEB)
                        {
                            col_B -= baseB;
                        }

                        // Check if a new nnz is generated
                        if(nnz[col_B] != i)
                        {
                            nnz[col_B] = i;
                            num_nonzeros++;
                        }

                        // Current column of B
                        col_B = csr_col_ind_B[row_B + k + 3];
                        if constexpr(BASEB)
                        {
                            col_B -= baseB;
                        }

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
                        aoclsparse_int col_B = csr_col_ind_B[row_B + (k_iter * 4) + k];
                        if constexpr(BASEB)
                        {
                            col_B -= baseB;
                        }
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
        // Single-pass prefix sum with 64-bit accumulation to avoid UB on overflow
        // Truncation assignment is defined behavior; overflow check happens after loop
        int64_t running_sum = 0; // note that C is always 0-base, csr_C->ptr[0]=0 already
        for(aoclsparse_int i = 1; i < m + 1; i++)
        {
            running_sum += csr_C->ptr[i];
            csr_C->ptr[i] = static_cast<aoclsparse_int>(running_sum);
        }

        // Check for overflow AFTER loop (no UB occurred since all arithmetic was 64-bit)
        if(running_sum > aoclsparse_numeric::int_max)
        {
            delete csr_C;
            return aoclsparse_status_invalid_size;
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
// The template parameters BASEA and BASEB are used to check if the input
// matrices are in 0 or 1 based index format. The default value is set to
// 0 based index format.
template <typename T, bool BASEA = false, bool BASEB = false>
inline aoclsparse_status aoclsparse_csr2m_finalize(aoclsparse_int             m_a,
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

    if(!(*C)->get_first_mtx_if_valid<aoclsparse::base_mtx>())
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
        if((csc_mat == nullptr) || (csc_mat->ptr == nullptr) || (csc_mat->ind == nullptr)
           || (csc_mat->val == nullptr))
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
        // col_acc is a temporary structure used to track nnz creation and to accumulate partial products
        struct col_acc
        {
            aoclsparse_int idx;
            T              sum;
        };
        col_acc *acc = NULL;
        try
        {
            acc = new col_acc[n + baseB];
#pragma omp simd
            for(aoclsparse_int i = 0; i < n + baseB; i++)
            {
                acc[i].idx = -1;
                acc[i].sum = 0;
            }
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
                aoclsparse_int row_begin_A = csr_row_ptr_A[i];
                aoclsparse_int row_end_A   = csr_row_ptr_A[i + 1];
                if constexpr(BASEA)
                {
                    row_begin_A -= baseA;
                    row_end_A -= baseA;
                }

                aoclsparse_int idxC
                    = csr_row_ptr_C[i]; // where to write first element in this row in C
                // Loop over columns of A
                for(aoclsparse_int j = row_begin_A; j < row_end_A; j++)
                {
                    // Current column of A
                    aoclsparse_int col_A = csr_col_ind_A[j];
                    if constexpr(BASEA)
                    {
                        col_A -= baseA;
                    }
                    // Current value of A
                    T val_A = csr_val_A[j];
                    if(opA == aoclsparse_operation_conjugate_transpose)
                        val_A = aoclsparse::conj(csr_val_A[j]);

                    aoclsparse_int row_begin_B = csr_row_ptr_B[col_A];
                    aoclsparse_int row_end_B   = csr_row_ptr_B[col_A + 1];
                    if constexpr(BASEB)
                    {
                        row_begin_B -= baseB;
                        row_end_B -= baseB;
                    }

                    // Loop over columns of B in row col_A
                    for(aoclsparse_int k = row_begin_B; k < row_end_B; k++)
                    {
                        // Current column of B
                        aoclsparse_int col_B = csr_col_ind_B[k];
                        // Current value of B
                        T val_B = csr_val_B[k];
                        if(opB == aoclsparse_operation_conjugate_transpose)
                            val_B = aoclsparse::conj(csr_val_B[k]);

                        aoclsparse_int nnz_col_b = acc[col_B].idx;
                        T              val       = val_A * val_B;
                        if(nnz_col_b != i)
                        {
                            // create new element in C of index col_B
                            csr_col_ind_C[idxC] = col_B;
                            acc[col_B].idx      = i;
                            acc[col_B].sum      = val;
                            idxC++;
                        }
                        else // the element already exist, just added in sum
                            acc[col_B].sum = acc[col_B].sum + val;
                    }
                }

                //  Check if the computed nonzeroes matches what we expect in the row
                if(idxC != csr_row_ptr_C[i + 1])
                    status = aoclsparse_status_internal_error;
                else
                {
                    // copy values from sum to csr_val_C based on csr_col_ind_C
                    for(idxC = csr_row_ptr_C[i]; idxC < csr_row_ptr_C[i + 1]; idxC++)
                    {
                        csr_val_C[idxC] = acc[csr_col_ind_C[idxC]].sum;
                        if constexpr(BASEB)
                        {
                            csr_col_ind_C[idxC] -= baseB;
                        }
                    }
                }
            }
        }
        delete[] acc;
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

    aoclsparse::base_mtx *A_first = A->get_first_mtx_if_valid<aoclsparse::base_mtx>();
    aoclsparse::base_mtx *B_first = B->get_first_mtx_if_valid<aoclsparse::base_mtx>();
    if(!A_first || !B_first)
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

    if(!A->is_descr_matching(descrA))
        return aoclsparse_status_invalid_value;

    if(!B->is_descr_matching(descrB))
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
        m_a = A->m;
        n_a = A->n;
        m_b = B->m;
        n_b = B->n;
    }
    // At * B
    else if(((opA == aoclsparse_operation_transpose)
             || (opA == aoclsparse_operation_conjugate_transpose))
            && (opB == aoclsparse_operation_none))
    {
        m_a = A->n;
        n_a = A->m;
        m_b = B->m;
        n_b = B->n;
    }
    // A * Bt
    else if((opA == aoclsparse_operation_none)
            && ((opB == aoclsparse_operation_transpose)
                || (opB == aoclsparse_operation_conjugate_transpose)))
    {
        m_a = A->m;
        n_a = A->n;
        m_b = B->n;
        n_b = B->m;
    }
    // At * Bt
    else if(((opA == aoclsparse_operation_transpose)
             || (opA == aoclsparse_operation_conjugate_transpose))
            && ((opB == aoclsparse_operation_transpose)
                || (opB == aoclsparse_operation_conjugate_transpose)))
    {
        m_a = A->n;
        n_a = A->m;
        m_b = B->n;
        n_b = B->m;
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
    // The mats vector may contain multiple matrix representations (e.g. CSR, CSC, ELL)
    // created during optimisation. We need the first csr-derived object — either a plain
    // CSR (doid::gn) or a CSC stored as A^T (doid::gt) — for the SpGEMM computation.
    // Each search acquires its own shared lock on mats_guard so that a concurrent
    // push_back (from an implicit optimisation on another thread) does not cause a data
    // race on the vector. The lock is released as soon as the pointer is extracted;
    // the pointed-to object itself remains valid and stable after the lock is released.
    aoclsparse::csr *raw_A = nullptr, *raw_B = nullptr;
    {
        std::shared_lock<std::shared_mutex> rlock_A(A->mats_guard);
        for(auto *mat : A->mats)
        {
            if(auto *temp = dynamic_cast<aoclsparse::csr *>(mat))
            {
                raw_A = temp;
                break;
            }
        }
    }
    {
        std::shared_lock<std::shared_mutex> rlock_B(B->mats_guard);
        for(auto *mat : B->mats)
        {
            if(auto *temp = dynamic_cast<aoclsparse::csr *>(mat))
            {
                raw_B = temp;
                break;
            }
        }
    }

    // Validate that CSR matrices were found in both A and B
    if(raw_A == nullptr || raw_B == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Validate base consistency using the discovered raw matrices
    if(raw_A->base != descrA->base)
        return aoclsparse_status_invalid_value;
    if(raw_B->base != descrB->base)
        return aoclsparse_status_invalid_value;

    // Permit gn (plain CSR) and gt (CSC stored as A^T); reject all other doids
    if((raw_A->doid != aoclsparse::doid::gn && raw_A->doid != aoclsparse::doid::gt)
       || (raw_B->doid != aoclsparse::doid::gn && raw_B->doid != aoclsparse::doid::gt))
        return aoclsparse_status_not_implemented;

    // Map user op to request doid for general matrices
    auto op_to_req_doid = [](aoclsparse_operation op) -> aoclsparse::doid {
        static constexpr aoclsparse_int bits[] = {0, 2, 3}; // gn, gt, gh
        return static_cast<aoclsparse::doid>(bits[op - 111]);
    };

    const aoclsparse::doid eff_doid_A
        = aoclsparse::get_effective_doid(raw_A->doid, op_to_req_doid(opA));
    const aoclsparse::doid eff_doid_B
        = aoclsparse::get_effective_doid(raw_B->doid, op_to_req_doid(opB));

    // Extract structural-transpose and conjugation flags from effective doids
    const bool eff_trans_A = (static_cast<int>(eff_doid_A) & 2) != 0;
    const bool eff_trans_B = (static_cast<int>(eff_doid_B) & 2) != 0;
    bool       conj_A      = (static_cast<int>(eff_doid_A) & 1) != 0;
    bool       conj_B      = (static_cast<int>(eff_doid_B) & 1) != 0;

    // Derive opflag from effective structural bits (accounts for CSC automatically)
    opflag = (eff_trans_A ? 1 : 0) | (eff_trans_B ? 2 : 0);

    bool owns_mat_A = false, owns_mat_B = false;

    // Transposes src (rows↔cols swapped); handles both CSR→A^T and CSC→A.
    aoclsparse::csr *transposed      = nullptr;
    auto             make_transposed = [&](aoclsparse::csr *src) -> aoclsparse_status {
        try
        {
            transposed = new aoclsparse::csr(
                src->n, src->m, src->nnz, aoclsparse_csr_mat, src->base, get_data_type<T>());
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        aoclsparse_status st = aoclsparse_csr2csc_template(src->m,
                                                           src->n,
                                                           src->nnz,
                                                           src->base,
                                                           src->base,
                                                           src->ptr,
                                                           src->ind,
                                                           (const T *)src->val,
                                                           transposed->ind,
                                                           transposed->ptr,
                                                           (T *)transposed->val);
        if(st != aoclsparse_status_success)
        {
            delete transposed;
            transposed = nullptr;
            return st;
        }
        return aoclsparse_status_success;
    };

    aoclsparse::csr *mat_A = nullptr, *mat_B = nullptr;

    if(opflag == 3)
    {
        // A_eff^T × B_eff^T = (B_eff · A_eff)^T — swap roles, no transpose needed
        mat_A = raw_B;
        mat_B = raw_A;
        {
            _aoclsparse_mat_descr tmp = descrA_t;
            descrA_t                  = descrB_t;
            descrB_t                  = tmp;
        }
        {
            aoclsparse_operation temp = opA;
            opA                       = opB;
            opB                       = temp;
        }
        {
            aoclsparse_int t = m_a;
            m_a              = n_b;
            n_b              = t;
        }
        std::swap(conj_A, conj_B);
    }
    else
    {
        mat_A = raw_A;
        mat_B = raw_B;
        if(eff_trans_A)
        {
            if(make_transposed(raw_A) != aoclsparse_status_success)
                return aoclsparse_status_memory_error;
            mat_A      = transposed;
            owns_mat_A = true;
        }
        if(eff_trans_B)
        {
            if(make_transposed(raw_B) != aoclsparse_status_success)
            {
                if(owns_mat_A)
                {
                    delete mat_A;
                    mat_A = nullptr;
                }
                return aoclsparse_status_memory_error;
            }
            mat_B      = transposed;
            owns_mat_B = true;
        }
    }

    // Set opA/opB to reflect effective conjugation for the finalize kernel
    opA = conj_A ? aoclsparse_operation_conjugate_transpose : aoclsparse_operation_none;
    opB = conj_B ? aoclsparse_operation_conjugate_transpose : aoclsparse_operation_none;

    aoclsparse_index_base baseA = descrA_t.base;
    aoclsparse_index_base baseB = descrB_t.base;

    switch(request)
    {

    case aoclsparse_stage_nnz_count:
    {
        if(baseA == aoclsparse_index_base_zero && baseB == aoclsparse_index_base_zero)
        {
            status = aoclsparse_csr2m_nnz_count<T, false, false>(m_a,
                                                                 n_b,
                                                                 &descrA_t,
                                                                 mat_A->ptr,
                                                                 mat_A->ind,
                                                                 &descrB_t,
                                                                 mat_B->ptr,
                                                                 mat_B->ind,
                                                                 C,
                                                                 opflag);
        }
        else if(baseA == aoclsparse_index_base_one && baseB == aoclsparse_index_base_zero)
        {
            status = aoclsparse_csr2m_nnz_count<T, true, false>(m_a,
                                                                n_b,
                                                                &descrA_t,
                                                                mat_A->ptr,
                                                                mat_A->ind,
                                                                &descrB_t,
                                                                mat_B->ptr,
                                                                mat_B->ind,
                                                                C,
                                                                opflag);
        }
        else if(baseA == aoclsparse_index_base_zero && baseB == aoclsparse_index_base_one)
        {
            status = aoclsparse_csr2m_nnz_count<T, false, true>(m_a,
                                                                n_b,
                                                                &descrA_t,
                                                                mat_A->ptr,
                                                                mat_A->ind,
                                                                &descrB_t,
                                                                mat_B->ptr,
                                                                mat_B->ind,
                                                                C,
                                                                opflag);
        }
        else if(baseA == aoclsparse_index_base_one && baseB == aoclsparse_index_base_one)
        {
            status = aoclsparse_csr2m_nnz_count<T, true, true>(m_a,
                                                               n_b,
                                                               &descrA_t,
                                                               mat_A->ptr,
                                                               mat_A->ind,
                                                               &descrB_t,
                                                               mat_B->ptr,
                                                               mat_B->ind,
                                                               C,
                                                               opflag);
        }
        break;
    }
    case aoclsparse_stage_finalize:
    {

        if(baseA == aoclsparse_index_base_zero && baseB == aoclsparse_index_base_zero)
        {
            status = aoclsparse_csr2m_finalize<T, false, false>(m_a,
                                                                n_b,
                                                                opA,
                                                                &descrA_t,
                                                                mat_A->ptr,
                                                                mat_A->ind,
                                                                (T *)mat_A->val,
                                                                opB,
                                                                &descrB_t,
                                                                mat_B->ptr,
                                                                mat_B->ind,
                                                                (T *)mat_B->val,
                                                                C,
                                                                opflag);
        }
        else if(baseA == aoclsparse_index_base_one && baseB == aoclsparse_index_base_zero)
        {
            status = aoclsparse_csr2m_finalize<T, true, false>(m_a,
                                                               n_b,
                                                               opA,
                                                               &descrA_t,
                                                               mat_A->ptr,
                                                               mat_A->ind,
                                                               (T *)mat_A->val,
                                                               opB,
                                                               &descrB_t,
                                                               mat_B->ptr,
                                                               mat_B->ind,
                                                               (T *)mat_B->val,
                                                               C,
                                                               opflag);
        }
        else if(baseA == aoclsparse_index_base_zero && baseB == aoclsparse_index_base_one)
        {
            status = aoclsparse_csr2m_finalize<T, false, true>(m_a,
                                                               n_b,
                                                               opA,
                                                               &descrA_t,
                                                               mat_A->ptr,
                                                               mat_A->ind,
                                                               (T *)mat_A->val,
                                                               opB,
                                                               &descrB_t,
                                                               mat_B->ptr,
                                                               mat_B->ind,
                                                               (T *)mat_B->val,
                                                               C,
                                                               opflag);
        }
        else if(baseA == aoclsparse_index_base_one && baseB == aoclsparse_index_base_one)
        {
            status = aoclsparse_csr2m_finalize<T, true, true>(m_a,
                                                              n_b,
                                                              opA,
                                                              &descrA_t,
                                                              mat_A->ptr,
                                                              mat_A->ind,
                                                              (T *)mat_A->val,
                                                              opB,
                                                              &descrB_t,
                                                              mat_B->ptr,
                                                              mat_B->ind,
                                                              (T *)mat_B->val,
                                                              C,
                                                              opflag);
        }
        break;
    }
    case aoclsparse_stage_full_computation:
    {
        if(baseA == aoclsparse_index_base_zero && baseB == aoclsparse_index_base_zero)
        {
            status = aoclsparse_csr2m_nnz_count<T, false, false>(m_a,
                                                                 n_b,
                                                                 &descrA_t,
                                                                 mat_A->ptr,
                                                                 mat_A->ind,
                                                                 &descrB_t,
                                                                 mat_B->ptr,
                                                                 mat_B->ind,
                                                                 C,
                                                                 opflag);

            if(status == aoclsparse_status_success)
            {
                status = aoclsparse_csr2m_finalize<T, false, false>(m_a,
                                                                    n_b,
                                                                    opA,
                                                                    &descrA_t,
                                                                    mat_A->ptr,
                                                                    mat_A->ind,
                                                                    (T *)mat_A->val,
                                                                    opB,
                                                                    &descrB_t,
                                                                    mat_B->ptr,
                                                                    mat_B->ind,
                                                                    (T *)mat_B->val,
                                                                    C,
                                                                    opflag);
            }
        }
        else if(baseA == aoclsparse_index_base_zero && baseB == aoclsparse_index_base_one)
        {
            status = aoclsparse_csr2m_nnz_count<T, false, true>(m_a,
                                                                n_b,
                                                                &descrA_t,
                                                                mat_A->ptr,
                                                                mat_A->ind,
                                                                &descrB_t,
                                                                mat_B->ptr,
                                                                mat_B->ind,
                                                                C,
                                                                opflag);

            if(status == aoclsparse_status_success)
            {
                status = aoclsparse_csr2m_finalize<T, false, true>(m_a,
                                                                   n_b,
                                                                   opA,
                                                                   &descrA_t,
                                                                   mat_A->ptr,
                                                                   mat_A->ind,
                                                                   (T *)mat_A->val,
                                                                   opB,
                                                                   &descrB_t,
                                                                   mat_B->ptr,
                                                                   mat_B->ind,
                                                                   (T *)mat_B->val,
                                                                   C,
                                                                   opflag);
            }
        }
        else if(baseA == aoclsparse_index_base_one && baseB == aoclsparse_index_base_zero)
        {
            status = aoclsparse_csr2m_nnz_count<T, true, false>(m_a,
                                                                n_b,
                                                                &descrA_t,
                                                                mat_A->ptr,
                                                                mat_A->ind,
                                                                &descrB_t,
                                                                mat_B->ptr,
                                                                mat_B->ind,
                                                                C,
                                                                opflag);

            if(status == aoclsparse_status_success)
            {
                status = aoclsparse_csr2m_finalize<T, true, false>(m_a,
                                                                   n_b,
                                                                   opA,
                                                                   &descrA_t,
                                                                   mat_A->ptr,
                                                                   mat_A->ind,
                                                                   (T *)mat_A->val,
                                                                   opB,
                                                                   &descrB_t,
                                                                   mat_B->ptr,
                                                                   mat_B->ind,
                                                                   (T *)mat_B->val,
                                                                   C,
                                                                   opflag);
            }
        }
        else if(baseA == aoclsparse_index_base_one && baseB == aoclsparse_index_base_one)
        {
            status = aoclsparse_csr2m_nnz_count<T, true, true>(m_a,
                                                               n_b,
                                                               &descrA_t,
                                                               mat_A->ptr,
                                                               mat_A->ind,
                                                               &descrB_t,
                                                               mat_B->ptr,
                                                               mat_B->ind,
                                                               C,
                                                               opflag);

            if(status == aoclsparse_status_success)
            {
                status = aoclsparse_csr2m_finalize<T, true, true>(m_a,
                                                                  n_b,
                                                                  opA,
                                                                  &descrA_t,
                                                                  mat_A->ptr,
                                                                  mat_A->ind,
                                                                  (T *)mat_A->val,
                                                                  opB,
                                                                  &descrB_t,
                                                                  mat_B->ptr,
                                                                  mat_B->ind,
                                                                  (T *)mat_B->val,
                                                                  C,
                                                                  opflag);
            }
        }
        break;
    }
    default:
        status = aoclsparse_status_invalid_value;
    }
    if(owns_mat_A)
        delete mat_A;
    if(owns_mat_B)
        delete mat_B;
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
