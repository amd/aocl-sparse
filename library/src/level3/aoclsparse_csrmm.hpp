
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
#ifndef AOCLSPARSE_CSRMM_HPP
#define AOCLSPARSE_CSRMM_HPP
#include "aoclsparse_cntx_dispatcher.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_l3_kt.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <shared_mutex>
#include <vector>

template <typename T, bool CONJ_VAL = false>
aoclsparse_status aoclsparse_csrmm_col_major_ref(T                          alpha,
                                                 const aoclsparse_mat_descr descr,
                                                 const T *__restrict__ csr_val,
                                                 const aoclsparse_int *__restrict__ csr_col_ind,
                                                 const aoclsparse_int *__restrict__ csr_row_ptr,
                                                 aoclsparse_int m,
                                                 const T       *B,
                                                 aoclsparse_int n,
                                                 aoclsparse_int ldb,
                                                 T              beta,
                                                 T             *C,
                                                 aoclsparse_int ldc)
{
    using namespace aoclsparse;
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *B_fix           = B - base;
#ifdef _OPENMP
#pragma omp parallel num_threads(context::get_context()->get_num_threads())
#endif
    {
#ifdef _OPENMP
        aoclsparse_int num_threads = omp_get_num_threads();
        aoclsparse_int thread_num  = omp_get_thread_num();
        aoclsparse_int start       = n * thread_num / num_threads;
        aoclsparse_int end         = n * (thread_num + 1) / num_threads;
#else
        aoclsparse_int start = 0;
        aoclsparse_int end   = n;
#endif
        for(aoclsparse_int j = start; j < end; ++j)
        {
            for(aoclsparse_int i = 0; i < m; ++i)
            {
                aoclsparse_int row_begin = csr_row_ptr[i];
                aoclsparse_int row_end   = csr_row_ptr[i + 1];
                aoclsparse_int idx_C     = i + j * ldc;
                T              sum       = 0.0;
                for(aoclsparse_int k = row_begin; k < row_end; ++k)
                {
                    aoclsparse_int idx_B = (csr_col_ind_fix[k] + j * ldb);
                    // Zero-overhead: false branch eliminated at compile time
                    const T aval = CONJ_VAL ? aoclsparse::conj(csr_val_fix[k]) : csr_val_fix[k];
                    sum          = aval * B_fix[idx_B] + sum;
                }
                C[idx_C] = (beta * C[idx_C]) + (alpha * sum);
            }
        }
    }
    return aoclsparse_status_success;
}
template <typename T, bool CONJ_VAL = false>
aoclsparse_status aoclsparse_csrmm_row_major_ref(T                          alpha,
                                                 const aoclsparse_mat_descr descr,
                                                 const T *__restrict__ csr_val,
                                                 const aoclsparse_int *__restrict__ csr_col_ind,
                                                 const aoclsparse_int *__restrict__ csr_row_ptr,
                                                 aoclsparse_int m,
                                                 const T       *B,
                                                 aoclsparse_int n,
                                                 aoclsparse_int ldb,
                                                 T              beta,
                                                 T             *C,
                                                 aoclsparse_int ldc)
{
    using namespace aoclsparse;
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *B_fix           = B - (base * ldb);

#ifdef _OPENMP
#pragma omp parallel num_threads(context::get_context()->get_num_threads())
#endif
    {
#ifdef _OPENMP
        aoclsparse_int num_threads = omp_get_num_threads();
        aoclsparse_int thread_num  = omp_get_thread_num();
        aoclsparse_int start       = m * thread_num / num_threads;
        aoclsparse_int end         = m * (thread_num + 1) / num_threads;
#else
        aoclsparse_int start = 0;
        aoclsparse_int end   = m;
#endif
        for(aoclsparse_int i = start; i < end; ++i)
        {
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            aoclsparse_int idx_C     = i * ldc;
            for(aoclsparse_int k = 0; k < n; ++k)
            {
                C[idx_C + k] = C[idx_C + k] * beta;
            }
            for(aoclsparse_int j = row_begin; j < row_end; ++j)
            {
                aoclsparse_int idx_B = csr_col_ind_fix[j] * ldb;
                // Zero-overhead: false branch eliminated at compile time
                const T aval = CONJ_VAL ? aoclsparse::conj(csr_val_fix[j]) : csr_val_fix[j];
                for(aoclsparse_int k = 0; k < n; ++k)
                {
                    C[idx_C + k] += aval * B_fix[idx_B + k] * alpha;
                }
            }
        }
    }
    return aoclsparse_status_success;
}

// The parameter HERM specifies if the input csr matrix described by
// <descr, csr_val, csr_col_ind and csr_row_ptr> is hermitian.
template <typename T, bool HERM = false, bool CONJ_VAL = false>
aoclsparse_status aoclsparse_csrmm_sym_row_ref(T                          alpha,
                                               const aoclsparse_mat_descr descr,
                                               const T *__restrict__ csr_val,
                                               const aoclsparse_int *__restrict__ csr_col_ind,
                                               const aoclsparse_int *__restrict__ csr_row_ptr,
                                               aoclsparse_int m,
                                               const T       *B,
                                               aoclsparse_int n,
                                               aoclsparse_int ldb,
                                               T             *C,
                                               aoclsparse_int ldc)
{
    T                     one  = 1.0;
    aoclsparse_index_base base = descr->base;
    // Variables to identify the type of the matrix
    const aoclsparse_fill_mode fill = descr->fill_mode;
    const aoclsparse_diag_type diag = descr->diag_type;
    for(int i = 0; i < m; i++)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        if(diag == aoclsparse_diag_type_unit)
        {
            for(int j = 0; j < n; j++)
            {
                aoclsparse_int idx_c = i * ldc + j;
                aoclsparse_int idx_b = i * ldb + j;
                C[idx_c] += one * B[idx_b] * alpha;
            }
        }
        for(int k = row_begin; k < row_end; k++)
        {
            // Zero-overhead: false branch eliminated at compile time
            const T v       = CONJ_VAL ? aoclsparse::conj(csr_val[k]) : csr_val[k];
            bool    is_diag = (i == (csr_col_ind[k] - base));
            if(is_diag && (diag == aoclsparse_diag_type_non_unit))
            {
                for(int j = 0; j < n; j++)
                {
                    aoclsparse_int idx_c = i * ldc + j;
                    aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                    C[idx_c] += v * B[idx_b] * alpha;
                }
            }
            else
            {
                // this conditional can be hoisted outside the above loops, but would have replicate the code
                // Todo: evaluate the performance and make the changes
                if(fill == aoclsparse_fill_mode_lower)
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i * ldc + j;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                        // Access only lower triangle, update the idx_b and idx_c to process upper triangle of the matrix.
                        // Having a conditional is not efficient, but required if the the matrix A is not sorted.
                        // ToDo: sort matrix A by column indices to get rid of the conditional
                        if(i > (csr_col_ind[k] - base))
                        {
                            C[idx_c] += v * B[idx_b] * alpha;
                            idx_b = i * ldb + j;
                            idx_c = (csr_col_ind[k] - base) * ldc + j;
                            if constexpr(HERM)
                            {
                                C[idx_c] += aoclsparse::conj(v) * (B[idx_b]) * alpha;
                            }
                            else
                            {
                                C[idx_c] += v * (B[idx_b]) * alpha;
                            }
                        }
                    }
                }
                else // fill == aoclsparse_fill_mode_upper
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i * ldc + j;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                        // Access only upper triangle
                        // Having a conditional is not efficient, but required if the the matrix A is not sorted.
                        // ToDo: sort matrix A by column indices to get rid of the conditional
                        if(i < (csr_col_ind[k] - base))
                        {
                            C[idx_c] += v * B[idx_b] * alpha;
                            idx_b = i * ldb + j;
                            idx_c = (csr_col_ind[k] - base) * ldc + j;
                            if constexpr(HERM)
                            {
                                C[idx_c] += aoclsparse::conj(v) * (B[idx_b]) * alpha;
                            }
                            else
                            {
                                C[idx_c] += v * (B[idx_b]) * alpha;
                            }
                        }
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}

// The parameter HERM specifies if the input csr matrix described by
// <descr, csr_val, csr_col_ind and csr_row_ptr> is hermitian.
template <typename T, bool HERM = false, bool CONJ_VAL = false>
aoclsparse_status aoclsparse_csrmm_sym_col_ref(T                          alpha,
                                               const aoclsparse_mat_descr descr,
                                               const T *__restrict__ csr_val,
                                               const aoclsparse_int *__restrict__ csr_col_ind,
                                               const aoclsparse_int *__restrict__ csr_row_ptr,
                                               aoclsparse_int m,
                                               const T       *B,
                                               aoclsparse_int n,
                                               aoclsparse_int ldb,
                                               T             *C,
                                               aoclsparse_int ldc)
{
    T                     one  = 1.0;
    aoclsparse_index_base base = descr->base;
    // Variables to identify the type of the matrix
    const aoclsparse_fill_mode fill = descr->fill_mode;
    const aoclsparse_diag_type diag = descr->diag_type;
    for(int i = 0; i < m; i++)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        if(diag == aoclsparse_diag_type_unit)
        {
            for(int j = 0; j < n; j++)
            {
                aoclsparse_int idx_c = i + j * ldc;
                aoclsparse_int idx_b = i + j * ldb;
                C[idx_c] += one * B[idx_b] * alpha;
            }
        }
        for(int k = row_begin; k < row_end; k++)
        {
            // Zero-overhead: false branch eliminated at compile time
            const T v       = CONJ_VAL ? aoclsparse::conj(csr_val[k]) : csr_val[k];
            bool    is_diag = (i == (csr_col_ind[k] - base));
            if(is_diag && (diag == aoclsparse_diag_type_non_unit))
            {
                for(int j = 0; j < n; j++)
                {
                    aoclsparse_int idx_c = i + j * ldc;
                    aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                    C[idx_c] += v * B[idx_b] * alpha;
                }
            }
            else
            {
                // this conditional can be hoisted outside the above loops, but would have replicate the code
                // Todo: evaluate the performance and make the changes
                if(fill == aoclsparse_fill_mode_lower)
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i + j * ldc;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                        // Access only lower triangle, update the idx_b and idx_c to process upper triangle of the matrix.
                        // Having a conditional is not efficient, but required if the the matrix A is not sorted.
                        // ToDo: sort matrix A by column indices to get rid of the conditional
                        if(i > (csr_col_ind[k] - base))
                        {
                            C[idx_c] += v * B[idx_b] * alpha;
                            idx_b = i + j * ldb;
                            idx_c = (csr_col_ind[k] - base) + j * ldc;
                            if constexpr(HERM)
                            {
                                C[idx_c] += aoclsparse::conj(v) * (B[idx_b]) * alpha;
                            }
                            else
                            {
                                C[idx_c] += v * (B[idx_b]) * alpha;
                            }
                        }
                    }
                }
                else // fill == aoclsparse_fill_mode_upper
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i + j * ldc;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                        // Access only upper triangle
                        // Having a conditional is not efficient, but required if the the matrix A is not sorted.
                        // ToDo: sort matrix A by column indices to get rid of the conditional
                        if(i < (csr_col_ind[k] - base))
                        {
                            C[idx_c] += v * B[idx_b] * alpha;
                            idx_b = i + j * ldb;
                            idx_c = (csr_col_ind[k] - base) + j * ldc;
                            if constexpr(HERM)
                            {
                                C[idx_c] += aoclsparse::conj(v) * (B[idx_b]) * alpha;
                            }
                            else
                            {
                                C[idx_c] += v * (B[idx_b]) * alpha;
                            }
                        }
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}

// This function performs scaling for a dense matrix, 'mtrx', by a value 'beta'
template <typename T>
aoclsparse_status scale_dense_matrix(
    aoclsparse_order order, T *mtrx, aoclsparse_int m, aoclsparse_int n, aoclsparse_int ld, T beta)
{
    using namespace aoclsparse;
    if(beta == aoclsparse_numeric::zero<T>())
    {
        if(order == aoclsparse_order_column)
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
            for(aoclsparse_int j = 0; j < n; ++j)
            {
                for(aoclsparse_int i = 0; i < m; ++i)
                {
                    mtrx[i + j * ld] = 0;
                }
            }
        }
        else // order == aoclsparse_order_row
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
            for(aoclsparse_int i = 0; i < m; ++i)
            {
                for(aoclsparse_int j = 0; j < n; ++j)
                {
                    mtrx[i * ld + j] = 0;
                }
            }
        }
    }
    else
    {
        if(order == aoclsparse_order_column)
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
            for(aoclsparse_int j = 0; j < n; ++j)
            {
                for(aoclsparse_int i = 0; i < m; ++i)
                {
                    aoclsparse_int idx_C = i + j * ld;
                    mtrx[idx_C]          = beta * mtrx[idx_C];
                }
            }
        }
        else // order == aoclsparse_order_row
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
            for(aoclsparse_int i = 0; i < m; ++i)
            {
                for(aoclsparse_int j = 0; j < n; ++j)
                {
                    aoclsparse_int idx_C = i * ld + j;
                    mtrx[idx_C]          = beta * mtrx[idx_C];
                }
            }
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_csrmm_t(aoclsparse_operation       op,
                                     const T                    alpha,
                                     const aoclsparse_matrix    A,
                                     const aoclsparse_mat_descr descr,
                                     aoclsparse_order           order,
                                     const T                   *B,
                                     aoclsparse_int             n,
                                     aoclsparse_int             ldb,
                                     const T                    beta,
                                     T                         *C,
                                     aoclsparse_int             ldc,
                                     aoclsparse_int             kid)
{
    using namespace aoclsparse;
    using namespace Dispatch;
    using namespace kernel_templates;

    // Check for valid matrix, descriptor
    if(A == nullptr || B == nullptr || C == nullptr || descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Only CSR input format supported
    if(A->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_not_implemented;
    }
    // check if op is valid
    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;
    // check if the matrix type is implemented
    if(descr->type != aoclsparse_matrix_type_general
       && descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_hermitian)
        return aoclsparse_status_not_implemented;
    // check if the matrix is square for symmetric/hermitial matrices
    if((descr->type == aoclsparse_matrix_type_symmetric
        || descr->type == aoclsparse_matrix_type_hermitian)
       && A->m != A->n)
    {
        return aoclsparse_status_invalid_size;
    }
    // check if the layout is valid
    if(order != aoclsparse_order_row && order != aoclsparse_order_column)
        return aoclsparse_status_invalid_value;

    // Verify the matrix types and T are consistent
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    if(!A->is_descr_matching(descr))
        return aoclsparse_status_invalid_value;

    T zero{0.0};
    T one{1.0};

    aoclsparse_int m = A->m;
    aoclsparse_int k = A->n;
    aoclsparse_int m_c{0}, n_c{0};

    aoclsparse::csr *csr_mat = A->get_first_mtx_if_valid<aoclsparse::csr>();
    if(!csr_mat)
        return aoclsparse_status_not_implemented;

    // CSR and CSC are supported via the internal doid convention:
    //   - doid::gn : CSR
    //   - doid::gt : CSC stored internally in CSR layout (raw->m = A->n, raw->n = A->m)
    // Other doid values are not supported here.
    // Note: A->input_format alone is not sufficient to distinguish CSR vs CSC — doid is authoritative.
    bool is_doid_gt = (csr_mat->doid == aoclsparse::doid::gt);
    if(!is_doid_gt && csr_mat->doid != aoclsparse::doid::gn)

        return aoclsparse_status_not_implemented;
    const aoclsparse_int *csr_col_ind = csr_mat->ind;
    const aoclsparse_int *csr_row_ptr = csr_mat->ptr;
    const T              *csr_val     = static_cast<T *>(csr_mat->val);

    // Variables to identify the type of the matrix
    const aoclsparse_matrix_type mat_type = descr->type;
    // Check sizes
    if(m < 0 || n < 0 || k < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        return aoclsparse_status_success;
    }
    if(alpha == zero && beta == one)
    {
        return aoclsparse_status_success;
    }
    // Check the rest of pointer arguments
    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    T                          *val_A;
    aoclsparse_int             *col_ind_A;
    aoclsparse_int             *row_ptr_A;
    std::vector<aoclsparse_int> csr_row_ptr_A;
    std::vector<aoclsparse_int> csr_col_ind_A;
    std::vector<T>              csr_val_A;
    aoclsparse_int              mb;
    aoclsparse_status           status;
    // If mat_found is set, pointers already reference the optimized matrix in A->mats.
    bool                  mat_found = false;
    _aoclsparse_mat_descr descr_t;
    aoclsparse_copy_mat_descr(&descr_t, descr);
    // req_doid: doid of the logical operation the user requested on logical matrix A.
    // eff_doid: doid of the in-memory layout; trans_doid() maps gn↔gt, gh↔gc,
    //           sl↔su, slc↔suc, hl↔huc, hu↔hlc — encodes all CSR↔CSC and fill flips.
    // conj_flip: true when the kernel must conjugate values (general gc/gh path only).
    aoclsparse::doid req_doid = aoclsparse::get_doid<T>(descr, op);
    aoclsparse::doid eff_doid = is_doid_gt ? aoclsparse::trans_doid(req_doid) : req_doid;
    bool conj_flip        = (eff_doid == aoclsparse::doid::gc || eff_doid == aoclsparse::doid::gh);
    aoclsparse::doid d_id = eff_doid;
    // For sym/herm CSC: A^T flips lower↔upper — remap descr_t.fill_mode to match stored layout.
    if(is_doid_gt && descr->type != aoclsparse_matrix_type_general)
        descr_t.fill_mode = (descr->fill_mode == aoclsparse_fill_mode_lower)
                                ? aoclsparse_fill_mode_upper
                                : aoclsparse_fill_mode_lower;
    mb = m; //Number of rows in matrix A

    // Check leading dimension of B
    aoclsparse_int check_ldb;
    if(op == aoclsparse_operation_none)
        check_ldb = (order == aoclsparse_order_column ? k : n);
    else
        check_ldb = (order == aoclsparse_order_column ? m : n);
    if(ldb < (((aoclsparse_int)1) >= check_ldb ? (aoclsparse_int)1 : check_ldb))
    {
        return aoclsparse_status_invalid_size;
    }
    // Check leading dimension of C
    aoclsparse_int check_ldc;
    if(op == aoclsparse_operation_none)
        check_ldc = (order == aoclsparse_order_column ? m : n);
    else
        check_ldc = (order == aoclsparse_order_column ? k : n);
    if(ldc < (((aoclsparse_int)1) >= check_ldc ? (aoclsparse_int)1 : check_ldc))
    {
        return aoclsparse_status_invalid_size;
    }
    // a few kernels are already fused with beta, so not updating C for those kernels
    if(op == aoclsparse_operation_none)
    {
        m_c = m;
    }
    else
    {
        m_c = k;
    }
    n_c = n;

    // Overflow check for dense matrix B and C offset computations in LP64 mode
    // Kernels compute indices like: i + j * ld (col-major) or i * ld + j (row-major).
    // The maximum index is strictly less than dim * ld, so we validate dim * ld.
    {
        aoclsparse_int c_dim, b_dim;
        aoclsparse_int b_rows = (op == aoclsparse_operation_none) ? k : m;

        if(order == aoclsparse_order_column)
        {
            c_dim = n;
            b_dim = n;
        }
        else // row major
        {
            c_dim = m_c;
            b_dim = b_rows;
        }
        if(aoclsparse_lp64_product_overflow(c_dim, ldc)
           || aoclsparse_lp64_product_overflow(b_dim, ldb))
        {
            return aoclsparse_status_invalid_size;
        }
    }

    // To support early return cases for alpha == zero scenario
    if(alpha == zero)
    {
        status = scale_dense_matrix(order, C, m_c, n_c, ldc, beta);
        return status; // Early return
    }
    /*
         * This loop iterates over the list of optimized matrices in A->mats and selects the one that matches
         * the required operation (doid). If found, it sets the pointers to the optimized matrix data and marks
         * mat_found as true for direct kernel invocation.
         */
    {
        std::shared_lock<std::shared_mutex> rlock(A->mats_guard);
        for(auto mat : A->mats)
        {
            aoclsparse::csr *csr_m = dynamic_cast<aoclsparse::csr *>(mat);
            if(csr_m != nullptr && mat->doid == req_doid)
            {
                // Extract the matrix
                val_A     = (T *)csr_m->val;
                col_ind_A = csr_m->ind;
                row_ptr_A = csr_m->ptr;
                mb        = csr_m->m;

                // diag adjustment only meaningful for sym/herm; general type has no diag_val
                if(descr_t.diag_type != mat->mtx_diag
                   && descr_t.type != aoclsparse_matrix_type_general)
                {
                    status = aoclsparse_set_mat_diag<T>(A->m, descr_t, csr_m);
                    if(status != aoclsparse_status_success)
                        return status;
                }
                // reset op & descr
                op                = aoclsparse_operation_none;
                descr_t.type      = aoclsparse_matrix_type_general;
                descr_t.fill_mode = aoclsparse_fill_mode_lower;
                descr_t.base      = csr_m->base;
                mat_found         = true;
                // reset doid and conjugation: optimized matrix is already the correct form
                d_id      = doid::gn;
                conj_flip = false;
                break;
            }
        }
    }

    // Conjugated doids capture all cases where values must be
    // conjugated (sym conj-trans, herm CSR, herm CSC — all encoded by trans_doid()).
    // mat_found path resets d_id to gn and op to none — conj_sym is false there (general).
    const bool conj_sym
        = (d_id == doid::slc || d_id == doid::suc || d_id == doid::hlc || d_id == doid::huc);

    switch(d_id)
    {
    case doid::sl:
    case doid::su:
    case doid::slc:
    case doid::suc:
        status = scale_dense_matrix(order, C, m_c, n_c, ldc, beta);
        if(status != aoclsparse_status_success)
            return status;

        if(order == aoclsparse_order_column)
        {
            if(conj_sym)
                return aoclsparse_csrmm_sym_col_ref<T, false, true>(
                    alpha, &descr_t, csr_val, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
            else
                return aoclsparse_csrmm_sym_col_ref<T, false, false>(
                    alpha, &descr_t, csr_val, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
        }
        else
        {
            if(conj_sym)
                return aoclsparse_csrmm_sym_row_ref<T, false, true>(
                    alpha, &descr_t, csr_val, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
            else
                return aoclsparse_csrmm_sym_row_ref<T, false, false>(
                    alpha, &descr_t, csr_val, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
        }
    case doid::hl:
    case doid::hu:
    case doid::hlc:
    case doid::huc:
        status = scale_dense_matrix(order, C, m_c, n_c, ldc, beta);
        if(status != aoclsparse_status_success)
            return status;

        if(order == aoclsparse_order_column)
        {
            if(conj_sym)
                return aoclsparse_csrmm_sym_col_ref<T, true, true>(
                    alpha, &descr_t, csr_val, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
            else
                return aoclsparse_csrmm_sym_col_ref<T, true, false>(
                    alpha, &descr_t, csr_val, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
        }
        else
        {
            if(conj_sym)
                return aoclsparse_csrmm_sym_row_ref<T, true, true>(
                    alpha, &descr_t, csr_val, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
            else
                return aoclsparse_csrmm_sym_row_ref<T, true, false>(
                    alpha, &descr_t, csr_val, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
        }
    case doid::gc: // conj path (gc/gh): conj_flip=true, CONJ_VAL=true kernel handles conjugation
    case doid::gn:
    case doid::gt:
    case doid::gh:
        /*
             * If mat_found is set to true, it indicates that an optimized matrix matching
             * the required operation has already been found in A->mats. In this case,
             * the pointers val_A, col_ind_A, and row_ptr_A are already set to the optimized
             * matrix data. Therefore, we can skip further processing and directly invoke
             * the appropriate kernel using these pointers.
             */
        if(mat_found)
            break;
        row_ptr_A = const_cast<aoclsparse_int *>(csr_row_ptr);
        col_ind_A = const_cast<aoclsparse_int *>(csr_col_ind);
        val_A     = const_cast<T *>(csr_val);
        mb        = csr_mat->m;

        if(d_id == doid::gt || d_id == doid::gh)
        {
            try
            {
                csr_col_ind_A.resize(A->nnz);
                // csr2csc(csr_mat->m, csr_mat->n) writes csr_mat->n + 1 entries to row_ptr output
                csr_row_ptr_A.resize(csr_mat->n + 1);
                csr_val_A.resize(A->nnz);
            }
            catch(std::bad_alloc &)
            {
                return aoclsparse_status_memory_error;
            }

            // csr_mat holds the raw input; csr_mat->m/n are its stored dimensions
            aoclsparse_status status = aoclsparse_csr2csc_template(csr_mat->m,
                                                                   csr_mat->n,
                                                                   A->nnz,
                                                                   descr->base,
                                                                   descr->base,
                                                                   csr_row_ptr,
                                                                   csr_col_ind,
                                                                   csr_val,
                                                                   csr_col_ind_A.data(),
                                                                   csr_row_ptr_A.data(),
                                                                   csr_val_A.data());
            if(status != aoclsparse_status_success)
                return aoclsparse_status_internal_error;
            if(d_id == doid::gh)
                conj_flip = true; // delegate conjugation to CONJ_VAL=true kernel
            row_ptr_A = csr_row_ptr_A.data();
            col_ind_A = csr_col_ind_A.data();
            val_A     = csr_val_A.data();
            mb        = csr_mat->n;
        }
        break;
    default:
        return aoclsparse_status_not_implemented;
    }
    if(order == aoclsparse_order_column)
    {
        // Column order
        using K = decltype(&aoclsparse_csrmm_col_major_ref<T, false>);

        // clang-format off
        // Plain (CONJ_VAL=false) and conjugated (CONJ_VAL=true) are two separate 4-entry tables.
        static constexpr Table<K> tbl_plain[]{
            {aoclsparse_csrmm_col_major_ref<T, false>,         context_isa_t::GENERIC,   0U | archs::ALL},
            {csrmm_col_kt<bsz::b256, T, false>,                context_isa_t::AVX2,      0U | archs::ALL},
     ORL<K>({csrmm_col_kt<bsz::b256, T, false>,               context_isa_t::AVX512VL,  0U | archs::ALL}),
     ORL<K>({csrmm_col_kt<bsz::b512, T, false>,               context_isa_t::AVX512F,   0U | archs::ALL}),
        };
        static constexpr Table<K> tbl_conj[]{
            {aoclsparse_csrmm_col_major_ref<T, true>,          context_isa_t::GENERIC,   0U | archs::ALL},
            {csrmm_col_kt<bsz::b256, T, true>,                 context_isa_t::AVX2,      0U | archs::ALL},
     ORL<K>({csrmm_col_kt<bsz::b256, T, true>,                context_isa_t::AVX512VL,  0U | archs::ALL}),
     ORL<K>({csrmm_col_kt<bsz::b512, T, true>,                context_isa_t::AVX512F,   0U | archs::ALL}),
        };
        // clang-format on

        thread_local K kache_plain = nullptr, kache_conj = nullptr;
        K             &kache  = conj_flip ? kache_conj : kache_plain;
        K              kernel = Oracle<K>(conj_flip ? tbl_conj : tbl_plain, kache, kid, 0, 4);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        // Invoke the kernel
        return kernel(alpha, &descr_t, val_A, col_ind_A, row_ptr_A, mb, B, n, ldb, beta, C, ldc);
    }
    else
    {
        // Row order
        using K = decltype(&aoclsparse_csrmm_row_major_ref<T, false>);

        // clang-format off
        // Plain (CONJ_VAL=false) and conjugated (CONJ_VAL=true) are two separate 4-entry tables.
        static constexpr Table<K> tbl_plain[]{
            {aoclsparse_csrmm_row_major_ref<T, false>,         context_isa_t::GENERIC,   0U | archs::ALL},
            {csrmm_row_kt<bsz::b256, T, false>,                context_isa_t::AVX2,      0U | archs::ALL},
     ORL<K>({csrmm_row_kt<bsz::b256, T, false>,               context_isa_t::AVX512VL,  0U | archs::ALL}),
     ORL<K>({csrmm_row_kt<bsz::b512, T, false>,               context_isa_t::AVX512F,   0U | archs::ALL}),
        };
        static constexpr Table<K> tbl_conj[]{
            {aoclsparse_csrmm_row_major_ref<T, true>,          context_isa_t::GENERIC,   0U | archs::ALL},
            {csrmm_row_kt<bsz::b256, T, true>,                 context_isa_t::AVX2,      0U | archs::ALL},
     ORL<K>({csrmm_row_kt<bsz::b256, T, true>,                context_isa_t::AVX512VL,  0U | archs::ALL}),
     ORL<K>({csrmm_row_kt<bsz::b512, T, true>,                context_isa_t::AVX512F,   0U | archs::ALL}),
        };
        // clang-format on

        thread_local K kache_plain = nullptr, kache_conj = nullptr;
        K             &kache  = conj_flip ? kache_conj : kache_plain;
        K              kernel = Oracle<K>(conj_flip ? tbl_conj : tbl_plain, kache, kid, 0, 4);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        // Invoke the kernel
        return kernel(alpha, &descr_t, val_A, col_ind_A, row_ptr_A, mb, B, n, ldb, beta, C, ldc);
    }
}
#endif /* AOCLSPARSE_CSRMM_HPP*/
