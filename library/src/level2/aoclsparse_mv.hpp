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
#ifndef AOCLSPARSE_OPTMV_HPP
#define AOCLSPARSE_OPTMV_HPP

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_csrmv.hpp"

#include <complex>
#include <immintrin.h>

template <typename T>
aoclsparse_status aoclsparse_dcsr_mat_br4(aoclsparse_operation       op,
                                          const T                    alpha,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          const T                   *x,
                                          const T                    beta,
                                          T                         *y);

template <typename T>
aoclsparse_status aoclsparse_mv_general(aoclsparse_operation       op,
                                        const T                    alpha,
                                        aoclsparse_matrix          A,
                                        const aoclsparse_mat_descr descr,
                                        const T                   *x,
                                        const T                    beta,
                                        T                         *y);

/* templated SpMV for complex types - can be extended for floats and doubles*/
template <typename T>
aoclsparse_status aoclsparse_mv_t(aoclsparse_operation op,
                                  T                    alpha, /* ToDo: this is not a pointer */
                                  aoclsparse_matrix    A,
                                  const aoclsparse_mat_descr descr,
                                  const T                   *x,
                                  T                          beta,
                                  T                         *y)
{
    aoclsparse_status     status;
    _aoclsparse_mat_descr descr_cpy;
    //aoclsparse_mat_descr descr_cpy = descr;
    aoclsparse_copy_mat_descr(&descr_cpy, descr);
    if(A == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_not_implemented;
    }

    if(descr == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Check pointer arguments
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Validate descriptor's index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    // Make sure the base index of descriptor and aoclsparse matrix are the same
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    // Check transpose
    if((op != aoclsparse_operation_none) && (op != aoclsparse_operation_transpose)
       && (op != aoclsparse_operation_conjugate_transpose))
    {
        return aoclsparse_status_invalid_value;
    }

    // Make sure we have the right type before casting
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    if(descr->type != aoclsparse_matrix_type_general
       && descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_triangular
       && descr->type != aoclsparse_matrix_type_hermitian)
        return aoclsparse_status_invalid_value;

    if((descr->type == aoclsparse_matrix_type_symmetric
        || descr->type == aoclsparse_matrix_type_hermitian)
       && A->m != A->n)
        return aoclsparse_status_invalid_value;

    // Quick return if possible
    T zero = 0;
    if(A->m == 0 || A->n == 0)
    {
        aoclsparse_int dim = op == aoclsparse_operation_none ? A->m : A->n;
        if(beta != zero)
        {
            for(aoclsparse_int i = 0; i < dim; i++)
                y[i] = beta * y[i];
        }
        else
        {
            for(aoclsparse_int i = 0; i < dim; i++)
                y[i] = zero;
        }
        return aoclsparse_status_success;
    }

    if((descr->type == aoclsparse_matrix_type_triangular
        || descr->type == aoclsparse_matrix_type_symmetric
        || descr->type == aoclsparse_matrix_type_hermitian)
       && !A->opt_csr_ready)
    {
        status = aoclsparse_csr_optimize<T>(A);
        if(status)
            return status;
        descr_cpy.base = A->internal_base_index;
    }

    // Read the environment variables to update global variable
    // This function updates the num_threads only once.
    aoclsparse_init_once();

    aoclsparse_int    *crstart = nullptr;
    aoclsparse_int    *crend   = nullptr;
    aoclsparse_context context;
    context.num_threads = sparse_global_context.num_threads;

    aoclsparse_int m = A->m, n = A->n;

    // Dispatcher
    switch(op)
    {
    case aoclsparse_operation_none:
        if(descr->type == aoclsparse_matrix_type_symmetric)
        {
            return aoclsparse_csrmv_symm_internal(descr_cpy.base,
                                                  alpha,
                                                  A->m,
                                                  descr_cpy.diag_type,
                                                  descr_cpy.fill_mode,
                                                  (T *)A->opt_csr_mat.csr_val,
                                                  A->opt_csr_mat.csr_col_ptr,
                                                  A->opt_csr_mat.csr_row_ptr,
                                                  A->idiag,
                                                  A->iurow,
                                                  x,
                                                  beta,
                                                  y);
        }
        else if(descr->type == aoclsparse_matrix_type_general)
        {
            return aoclsparse_csrmv_general(descr->base,
                                            alpha,
                                            m,
                                            (T *)A->csr_mat.csr_val,
                                            A->csr_mat.csr_col_ptr,
                                            A->csr_mat.csr_row_ptr,
                                            x,
                                            beta,
                                            y,
                                            &context);
        }
        else if(descr->type == aoclsparse_matrix_type_triangular)
        {
            if(descr->fill_mode == aoclsparse_fill_mode_lower)
            {
                // y = alpha L * x + beta y
                crstart = A->opt_csr_mat.csr_row_ptr;
                crend   = A->iurow;
            }
            else if(descr->fill_mode == aoclsparse_fill_mode_upper)
            {
                // y = alpha U * x + beta y
                crstart = A->idiag;
                crend   = &A->opt_csr_mat.csr_row_ptr[1];
            }
            return aoclsparse_csrmv_ptr(&descr_cpy,
                                        alpha,
                                        A->m,
                                        A->n,
                                        (T *)A->opt_csr_mat.csr_val,
                                        A->opt_csr_mat.csr_col_ptr,
                                        crstart,
                                        crend,
                                        x,
                                        beta,
                                        y,
                                        &context);
        }
        else if(descr->type == aoclsparse_matrix_type_hermitian)
        {
            return aoclsparse_csrmv_herm_internal(descr_cpy.base,
                                                  alpha,
                                                  A->m,
                                                  descr_cpy.diag_type,
                                                  descr_cpy.fill_mode,
                                                  (T *)A->opt_csr_mat.csr_val,
                                                  A->opt_csr_mat.csr_col_ptr,
                                                  A->opt_csr_mat.csr_row_ptr,
                                                  A->idiag,
                                                  A->iurow,
                                                  x,
                                                  beta,
                                                  y);
        }
        break;

    case aoclsparse_operation_transpose:
        if(descr->type == aoclsparse_matrix_type_symmetric)
        {
            return aoclsparse_csrmv_symm_internal(descr_cpy.base,
                                                  alpha,
                                                  A->m,
                                                  descr_cpy.diag_type,
                                                  descr_cpy.fill_mode,
                                                  (T *)A->opt_csr_mat.csr_val,
                                                  A->opt_csr_mat.csr_col_ptr,
                                                  A->opt_csr_mat.csr_row_ptr,
                                                  A->idiag,
                                                  A->iurow,
                                                  x,
                                                  beta,
                                                  y);
        }
        else if(descr->type == aoclsparse_matrix_type_general)
        {
            return aoclsparse_csrmvt(descr->base,
                                     alpha,
                                     m,
                                     n,
                                     (T *)A->csr_mat.csr_val,
                                     A->csr_mat.csr_col_ptr,
                                     A->csr_mat.csr_row_ptr,
                                     x,
                                     beta,
                                     y);
        }
        else if(descr->type == aoclsparse_matrix_type_triangular)
        {
            if(descr->fill_mode == aoclsparse_fill_mode_lower)
            {
                // y = alpha L * x + beta y
                crstart = A->opt_csr_mat.csr_row_ptr;
                crend   = A->iurow;
            }
            else if(descr->fill_mode == aoclsparse_fill_mode_upper)
            {
                // y = alpha U * x + beta y
                crstart = A->idiag;
                crend   = &A->opt_csr_mat.csr_row_ptr[1];
            }
            return aoclsparse_csrmvt_ptr(&descr_cpy,
                                         alpha,
                                         A->m,
                                         A->n,
                                         (T *)A->opt_csr_mat.csr_val,
                                         A->opt_csr_mat.csr_col_ptr,
                                         crstart,
                                         crend,
                                         x,
                                         beta,
                                         y);
        }
        else if(descr->type == aoclsparse_matrix_type_hermitian)
        {
            return aoclsparse_status_not_implemented;
        }
        break;

    case aoclsparse_operation_conjugate_transpose:
        if(descr->type == aoclsparse_matrix_type_symmetric)
        {
            return aoclsparse_csrmvh_symm_internal(descr_cpy.base,
                                                   alpha,
                                                   A->m,
                                                   descr_cpy.diag_type,
                                                   descr_cpy.fill_mode,
                                                   (T *)A->opt_csr_mat.csr_val,
                                                   A->opt_csr_mat.csr_col_ptr,
                                                   A->opt_csr_mat.csr_row_ptr,
                                                   A->idiag,
                                                   A->iurow,
                                                   x,
                                                   beta,
                                                   y);
        }
        else if(descr->type == aoclsparse_matrix_type_general)
        {
            return aoclsparse_csrmvh(descr->base,
                                     alpha,
                                     m,
                                     n,
                                     (T *)A->csr_mat.csr_val,
                                     A->csr_mat.csr_col_ptr,
                                     A->csr_mat.csr_row_ptr,
                                     x,
                                     beta,
                                     y);
        }
        else if(descr->type == aoclsparse_matrix_type_triangular)
        {
            if(descr->fill_mode == aoclsparse_fill_mode_lower)
            {
                // y = alpha L * x + beta y
                crstart = A->opt_csr_mat.csr_row_ptr;
                crend   = A->iurow;
            }
            else if(descr->fill_mode == aoclsparse_fill_mode_upper)
            {
                // y = alpha U * x + beta y
                crstart = A->idiag;
                crend   = &A->opt_csr_mat.csr_row_ptr[1];
            }
            return aoclsparse_csrmvh_ptr(&descr_cpy,
                                         alpha,
                                         A->m,
                                         A->n,
                                         (T *)A->opt_csr_mat.csr_val,
                                         A->opt_csr_mat.csr_col_ptr,
                                         crstart,
                                         crend,
                                         x,
                                         beta,
                                         y);
        }
        else if(descr->type == aoclsparse_matrix_type_hermitian)
        {
            return aoclsparse_csrmv_herm_internal(descr_cpy.base,
                                                  alpha,
                                                  A->m,
                                                  descr_cpy.diag_type,
                                                  descr_cpy.fill_mode,
                                                  (T *)A->opt_csr_mat.csr_val,
                                                  A->opt_csr_mat.csr_col_ptr,
                                                  A->opt_csr_mat.csr_row_ptr,
                                                  A->idiag,
                                                  A->iurow,
                                                  x,
                                                  beta,
                                                  y);
        }
        break;

    default:
        return aoclsparse_status_invalid_value;
        break;
    }

    return aoclsparse_status_not_implemented;
}

/* templated version to dispatch optimized SPMV
 * Note, the assumption is that x&y are compatible with the size of A
 * Compute y:= beta*y + alpha*A*x    or   + alpha*A'*x
 */
template <typename T>
aoclsparse_status aoclsparse_mv(aoclsparse_operation       op,
                                T                          alpha,
                                aoclsparse_matrix          A,
                                const aoclsparse_mat_descr descr,
                                const T                   *x,
                                T                          beta,
                                T                         *y)
{
    aoclsparse_status status;

    _aoclsparse_mat_descr descr_cpy;
    //aoclsparse_mat_descr descr_cpy = descr;
    aoclsparse_copy_mat_descr(&descr_cpy, descr);

    // still check A, in case the template is called directly
    // now A->mat_type should match T
    if(A == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_not_implemented;
    }

    if(descr == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Check pointer arguments
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Validate descriptor's index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    // Validate aoclsparse matrix's index base
    if(A->base != aoclsparse_index_base_zero && A->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    // Make sure the base index of descriptor and aoclsparse matrix are the same
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    // Check transpose
    if((op != aoclsparse_operation_none) && (op != aoclsparse_operation_transpose))
    {
        if(op == aoclsparse_operation_conjugate_transpose)
        {
            op = aoclsparse_operation_transpose;
        }
        else
        {
            return aoclsparse_status_invalid_value;
        }
    }

    // Make sure we have the right type before casting
    if(!((A->val_type == aoclsparse_dmat && std::is_same_v<T, double>)
         || (A->val_type == aoclsparse_smat && std::is_same_v<T, float>)))
        return aoclsparse_status_wrong_type;

    if(descr->type != aoclsparse_matrix_type_general
       && descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_triangular)
        return aoclsparse_status_not_implemented;

    if(descr->type == aoclsparse_matrix_type_symmetric && A->m != A->n)
        return aoclsparse_status_invalid_value;

    // Quick return if possible
    T zero = 0;
    if(A->m == 0 || A->n == 0)
    {
        aoclsparse_int dim = op == aoclsparse_operation_none ? A->m : A->n;
        if(beta != zero)
        {
            for(aoclsparse_int i = 0; i < dim; i++)
                y[i] = beta * y[i];
        }
        else
        {
            for(aoclsparse_int i = 0; i < dim; i++)
                y[i] = zero;
        }
        return aoclsparse_status_success;
    }

    // In UK/HPCG branch this would be triggered every time
    // Let's do it only for triangular/symmetric matrices
    if((descr->type == aoclsparse_matrix_type_triangular
        || descr->type == aoclsparse_matrix_type_symmetric)
       && !A->opt_csr_ready)
    {
        status = aoclsparse_csr_optimize<T>(A);
        if(status)
            return status;
        descr_cpy.base = A->internal_base_index;
    }

    // Read the environment variables to update global variable
    // This function updates the num_threads only once.
    aoclsparse_init_once();

    aoclsparse_int    *crstart = nullptr;
    aoclsparse_int    *crend   = nullptr;
    aoclsparse_context context;
    context.num_threads = sparse_global_context.num_threads;

    // Dispatcher
    if(descr->type == aoclsparse_matrix_type_general)
    {
        // TODO trigger appropriate optimization?
        return aoclsparse_mv_general(op, alpha, A, descr, x, beta, y);
        // In UK/HPCG branch this would go only to AVX2 CSR
        // y = alpha A * x + beta y
        /*return aoclsparse_csrmv_vectorized_avx2ptr(descr,
                                                   alpha,
                                                   A->m,
                                                   A->n,
                                                   A->nnz,
                                                   (T *)A->opt_csr_mat.csr_val,
                                                   A->opt_csr_mat.csr_col_ptr,
                                                   A->opt_csr_mat.csr_row_ptr,
                                                   &A->opt_csr_mat.csr_row_ptr[1],
                                                   x,
                                                   beta,
                                                   y,
                                                   &context);*/
    }
    else if(descr->type == aoclsparse_matrix_type_symmetric)
    {
        // can dispatch our data directly
        // transposed and non-transposed operation
        // y = alpha A * x + beta y
        return aoclsparse_csrmv_symm_internal(descr_cpy.base,
                                              alpha,
                                              A->m,
                                              descr_cpy.diag_type,
                                              descr_cpy.fill_mode,
                                              (T *)A->opt_csr_mat.csr_val,
                                              A->opt_csr_mat.csr_col_ptr,
                                              A->opt_csr_mat.csr_row_ptr,
                                              A->idiag,
                                              A->iurow,
                                              x,
                                              beta,
                                              y);
    }
    else
    {
        //Triangular SPMV
        if(descr->type == aoclsparse_matrix_type_triangular)
        {
            if(descr->fill_mode == aoclsparse_fill_mode_lower)
            {
                // y = alpha L * x + beta y
                crstart = A->opt_csr_mat.csr_row_ptr;
                crend   = A->iurow;
            }
            else if(descr->fill_mode == aoclsparse_fill_mode_upper)
            {
                // y = alpha U * x + beta y
                crstart = A->idiag;
                crend   = &A->opt_csr_mat.csr_row_ptr[1];
            }
        }
        //kernels as per transpose operation
        if(op == aoclsparse_operation_none)
        {
            return aoclsparse_csrmv_vectorized_avx2ptr(&descr_cpy,
                                                       alpha,
                                                       A->m,
                                                       A->n,
                                                       A->nnz,
                                                       (T *)A->opt_csr_mat.csr_val,
                                                       A->opt_csr_mat.csr_col_ptr,
                                                       crstart,
                                                       crend,
                                                       x,
                                                       beta,
                                                       y,
                                                       &context);
        }
        else if(op == aoclsparse_operation_transpose)
        {
            return aoclsparse_csrmvt_ptr(&descr_cpy,
                                         alpha,
                                         A->m,
                                         A->n,
                                         (T *)A->opt_csr_mat.csr_val,
                                         A->opt_csr_mat.csr_col_ptr,
                                         crstart,
                                         crend,
                                         x,
                                         beta,
                                         y);
        }
    }
    return aoclsparse_status_not_implemented;
}

#endif // AOCLSPARSE_OPTMV_HPP
