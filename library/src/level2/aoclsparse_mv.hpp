/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_blkcsrmv.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_csrmv.hpp"
#include "aoclsparse_ellmv.hpp"

#include <complex>
#include <immintrin.h>

// Kernels
// -----------------------------------
template <typename T>
std::enable_if_t<std::is_same_v<T, double>, aoclsparse_status>
    aoclsparse_dcsr_mat_br4([[maybe_unused]] aoclsparse_operation       op,
                            const T                                     alpha,
                            aoclsparse_matrix                           A,
                            [[maybe_unused]] const aoclsparse_mat_descr descr,
                            const T                                    *x,
                            const T                                     beta,
                            T                                          *y)
{
    using namespace aoclsparse;

    __m256d               res, vvals, vx, vy, va, vb;
    aoclsparse_index_base base = A->base;

    va  = _mm256_set1_pd(alpha);
    vb  = _mm256_set1_pd(beta);
    res = _mm256_setzero_pd();

    aoclsparse_int                 *tcptr = A->csr_mat_br4.csr_col_ptr;
    aoclsparse_int                 *rptr  = A->csr_mat_br4.csr_row_ptr;
    aoclsparse_int                 *cptr;
    double                         *tvptr = (double *)A->csr_mat_br4.csr_val;
    const double                   *vptr;
    aoclsparse_int                  blk = 4;
    [[maybe_unused]] aoclsparse_int chunk_size
        = (A->m) / (blk * context::get_context()->get_num_threads());

#ifdef _OPENMP
    chunk_size = chunk_size ? chunk_size : 1;
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk_size) private(res, vvals, vx, vy, vptr, cptr)
#endif
    for(aoclsparse_int i = 0; i < (A->m) / blk; i++)
    {

        aoclsparse_int r = rptr[i * blk];
        vptr             = tvptr + r - base;
        cptr             = tcptr + r - base;

        res = _mm256_setzero_pd();
        // aoclsparse_int nnz = rptr[i*blk];
        aoclsparse_int nnz = rptr[i * blk + 1] - r;
        for(aoclsparse_int j = 0; j < nnz; ++j)
        {
            aoclsparse_int off = j * blk;
            vvals              = _mm256_loadu_pd((double const *)(vptr + off));

            vx = _mm256_set_pd(x[*(cptr + off + 3) - base],
                               x[*(cptr + off + 2) - base],
                               x[*(cptr + off + 1) - base],
                               x[*(cptr + off) - base]);

            res = _mm256_fmadd_pd(vvals, vx, res);
        }
        /*
	   tc += blk*nnz;
	   vptr += blk*nnz;
	   cptr += blk*nnz;
	   */

        if(alpha != static_cast<double>(1))
        {
            res = _mm256_mul_pd(va, res);
        }

        if(beta != static_cast<double>(0))
        {
            vy  = _mm256_loadu_pd(&y[i * blk]);
            res = _mm256_fmadd_pd(vb, vy, res);
        }
        _mm256_storeu_pd(&y[i * blk], res);
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(aoclsparse::context::get_context()->get_num_threads())
#endif
    for(aoclsparse_int k = ((A->m) / blk) * blk; k < A->m; ++k)
    {
        double result = 0;
        /*
	   aoclsparse_int nnz = A->csr_mat_br4.csr_row_ptr[k];
	   for(j = 0; j < nnz; ++j)
	   {
	   result += ((double *)A->csr_mat_br4.csr_val)[tc] * x[A->csr_mat_br4.csr_col_ptr[tc]];
	   tc++;;
	   }
	   */
        for(aoclsparse_int j = (A->csr_mat_br4.csr_row_ptr[k] - base);
            j < (A->csr_mat_br4.csr_row_ptr[k + 1] - base);
            ++j)
        {
            result
                += ((double *)A->csr_mat_br4.csr_val)[j] * x[A->csr_mat_br4.csr_col_ptr[j] - base];
        }

        if(alpha != static_cast<double>(1))
        {
            result = alpha * result;
        }

        if(beta != static_cast<double>(0))
        {
            result += beta * y[k];
        }
        y[k] = result;
    }

    return aoclsparse_status_success;
}

/* templated SpMV for complex types - can be extended for floats and doubles*/
template <typename T>
std::enable_if_t<std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>,
                 aoclsparse_status>
    aoclsparse_mv_t(aoclsparse_operation       op,
                    const T                   *alpha,
                    aoclsparse_matrix          A,
                    const aoclsparse_mat_descr descr,
                    const T                   *x,
                    const T                   *beta,
                    T                         *y)
{
    if(alpha == nullptr || beta == nullptr)
        return aoclsparse_status_invalid_pointer;

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
        return aoclsparse_status_invalid_size;

    // Quick return if possible
    T zero = 0;
    /* Diag_type is applicable for symm/herm/tri matrices. Internal functions
     * will handle diag_type=unit and nnz=0 case for such matrices. General
     * matrix with nnz=0 should update y (aoclsparse_dcsrmv() has quick return
     * which doesn't update y. Hence adding this nnz=0 & general matrix check here.)
     */
    if(A->m == 0 || A->n == 0 || (A->nnz == 0 && descr->type == aoclsparse_matrix_type_general))
    {
        aoclsparse_int dim = op == aoclsparse_operation_none ? A->m : A->n;
        if(*beta != zero)
        {
            for(aoclsparse_int i = 0; i < dim; i++)
                y[i] = (*beta) * y[i];
        }
        else
        {
            for(aoclsparse_int i = 0; i < dim; i++)
                y[i] = zero;
        }
        return aoclsparse_status_success;
    }

    aoclsparse_status     status;
    _aoclsparse_mat_descr descr_cpy;

    aoclsparse_copy_mat_descr(&descr_cpy, descr);

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

    aoclsparse_int *crstart = nullptr;
    aoclsparse_int *crend   = nullptr;

    aoclsparse_int m = A->m, n = A->n;

    // Dispatcher
    switch(op)
    {
    case aoclsparse_operation_none:
        if(descr->type == aoclsparse_matrix_type_symmetric)
        {
            return aoclsparse_csrmv_symm_internal(descr_cpy.base,
                                                  *alpha,
                                                  A->m,
                                                  descr_cpy.diag_type,
                                                  descr_cpy.fill_mode,
                                                  (T *)A->opt_csr_mat.csr_val,
                                                  A->opt_csr_mat.csr_col_ptr,
                                                  A->opt_csr_mat.csr_row_ptr,
                                                  A->idiag,
                                                  A->iurow,
                                                  x,
                                                  *beta,
                                                  y);
        }
        else if(descr->type == aoclsparse_matrix_type_general)
        {
            return aoclsparse_csrmv_general(descr->base,
                                            *alpha,
                                            m,
                                            (T *)A->csr_mat.csr_val,
                                            A->csr_mat.csr_col_ptr,
                                            A->csr_mat.csr_row_ptr,
                                            x,
                                            *beta,
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
            return aoclsparse_csrmv_ref(&descr_cpy,
                                        *alpha,
                                        A->m,
                                        A->n,
                                        (T *)A->opt_csr_mat.csr_val,
                                        A->opt_csr_mat.csr_col_ptr,
                                        crstart,
                                        crend,
                                        x,
                                        *beta,
                                        y);
        }
        else if(descr->type == aoclsparse_matrix_type_hermitian)
        {
            return aoclsparse_csrmv_herm_internal(descr_cpy.base,
                                                  *alpha,
                                                  A->m,
                                                  descr_cpy.diag_type,
                                                  descr_cpy.fill_mode,
                                                  (T *)A->opt_csr_mat.csr_val,
                                                  A->opt_csr_mat.csr_col_ptr,
                                                  A->opt_csr_mat.csr_row_ptr,
                                                  A->idiag,
                                                  A->iurow,
                                                  x,
                                                  *beta,
                                                  y);
        }
        break;

    case aoclsparse_operation_transpose:
        if(descr->type == aoclsparse_matrix_type_symmetric)
        {
            return aoclsparse_csrmv_symm_internal(descr_cpy.base,
                                                  *alpha,
                                                  A->m,
                                                  descr_cpy.diag_type,
                                                  descr_cpy.fill_mode,
                                                  (T *)A->opt_csr_mat.csr_val,
                                                  A->opt_csr_mat.csr_col_ptr,
                                                  A->opt_csr_mat.csr_row_ptr,
                                                  A->idiag,
                                                  A->iurow,
                                                  x,
                                                  *beta,
                                                  y);
        }
        else if(descr->type == aoclsparse_matrix_type_general)
        {
            return aoclsparse_csrmvt(descr->base,
                                     *alpha,
                                     m,
                                     n,
                                     (T *)A->csr_mat.csr_val,
                                     A->csr_mat.csr_col_ptr,
                                     A->csr_mat.csr_row_ptr,
                                     x,
                                     *beta,
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
                                         *alpha,
                                         A->m,
                                         A->n,
                                         (T *)A->opt_csr_mat.csr_val,
                                         A->opt_csr_mat.csr_col_ptr,
                                         crstart,
                                         crend,
                                         x,
                                         *beta,
                                         y);
        }
        else if(descr->type == aoclsparse_matrix_type_hermitian)
        {
            return aoclsparse_csrmv_hermt_internal(descr_cpy.base,
                                                   *alpha,
                                                   A->m,
                                                   descr_cpy.diag_type,
                                                   descr_cpy.fill_mode,
                                                   (T *)A->opt_csr_mat.csr_val,
                                                   A->opt_csr_mat.csr_col_ptr,
                                                   A->opt_csr_mat.csr_row_ptr,
                                                   A->idiag,
                                                   A->iurow,
                                                   x,
                                                   *beta,
                                                   y);
        }
        break;

    case aoclsparse_operation_conjugate_transpose:
        if(descr->type == aoclsparse_matrix_type_symmetric)
        {
            return aoclsparse_csrmvh_symm_internal(descr_cpy.base,
                                                   *alpha,
                                                   A->m,
                                                   descr_cpy.diag_type,
                                                   descr_cpy.fill_mode,
                                                   (T *)A->opt_csr_mat.csr_val,
                                                   A->opt_csr_mat.csr_col_ptr,
                                                   A->opt_csr_mat.csr_row_ptr,
                                                   A->idiag,
                                                   A->iurow,
                                                   x,
                                                   *beta,
                                                   y);
        }
        else if(descr->type == aoclsparse_matrix_type_general)
        {
            return aoclsparse_csrmvh(descr->base,
                                     *alpha,
                                     m,
                                     n,
                                     (T *)A->csr_mat.csr_val,
                                     A->csr_mat.csr_col_ptr,
                                     A->csr_mat.csr_row_ptr,
                                     x,
                                     *beta,
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
                                         *alpha,
                                         A->m,
                                         A->n,
                                         (T *)A->opt_csr_mat.csr_val,
                                         A->opt_csr_mat.csr_col_ptr,
                                         crstart,
                                         crend,
                                         x,
                                         *beta,
                                         y);
        }
        else if(descr->type == aoclsparse_matrix_type_hermitian)
        {
            return aoclsparse_csrmv_herm_internal(descr_cpy.base,
                                                  *alpha,
                                                  A->m,
                                                  descr_cpy.diag_type,
                                                  descr_cpy.fill_mode,
                                                  (T *)A->opt_csr_mat.csr_val,
                                                  A->opt_csr_mat.csr_col_ptr,
                                                  A->opt_csr_mat.csr_row_ptr,
                                                  A->idiag,
                                                  A->iurow,
                                                  x,
                                                  *beta,
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
std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>, aoclsparse_status>
    aoclsparse_mv_t(aoclsparse_operation       op,
                    const T                   *alpha,
                    aoclsparse_matrix          A,
                    const aoclsparse_mat_descr descr,
                    const T                   *x,
                    const T                   *beta,
                    T                         *y)
{
    if(alpha == nullptr || beta == nullptr)
        return aoclsparse_status_invalid_pointer;

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
        return aoclsparse_status_invalid_size;

    // Quick return if possible
    T zero = 0;
    /* Diag_type is applicable for symm/herm/tri matrices. Internal functions
     * will handle diag_type=unit and nnz=0 case for such matrices. General
     * matrix with nnz=0 should update y (aoclsparse_dcsrmv() has quick return
     * which doesn't update y. Hence adding this nnz=0 & general matrix check here.)
     */
    if(A->m == 0 || A->n == 0 || (A->nnz == 0 && descr->type == aoclsparse_matrix_type_general))
    {
        aoclsparse_int dim = op == aoclsparse_operation_none ? A->m : A->n;
        if(*beta != zero)
        {
            for(aoclsparse_int i = 0; i < dim; i++)
                y[i] = (*beta) * y[i];
        }
        else
        {
            for(aoclsparse_int i = 0; i < dim; i++)
                y[i] = zero;
        }
        return aoclsparse_status_success;
    }

    aoclsparse_status     status;
    _aoclsparse_mat_descr descr_cpy;

    aoclsparse_copy_mat_descr(&descr_cpy, descr);

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

    // Dispatcher
    if(descr->type == aoclsparse_matrix_type_general)
    {
        _aoclsparse_mat_descr descr_cpy;

        switch(A->mat_type)
        {
        case aoclsparse_csr_mat:
            if constexpr(std::is_same_v<T, double>)
            {
                if(A->blk_optimized)
                    return aoclsparse_blkcsrmv_t<T>(op,
                                                    alpha,
                                                    A->m,
                                                    A->n,
                                                    A->nnz,
                                                    A->csr_mat.masks,
                                                    (T *)A->csr_mat.blk_val,
                                                    A->csr_mat.blk_col_ptr,
                                                    A->csr_mat.blk_row_ptr,
                                                    descr,
                                                    x,
                                                    beta,
                                                    y,
                                                    A->csr_mat.nRowsblk);
            }

            aoclsparse_copy_mat_descr(&descr_cpy, descr);

            return aoclsparse_csrmv_t<T>(op,
                                         alpha,
                                         A->m,
                                         A->n,
                                         A->nnz,
                                         (T *)A->csr_mat.csr_val,
                                         A->csr_mat.csr_col_ptr,
                                         A->csr_mat.csr_row_ptr,
                                         &descr_cpy,
                                         x,
                                         beta,
                                         y);

        case aoclsparse_ellt_mat:
        case aoclsparse_ellt_csr_hyb_mat:
            return (aoclsparse_ellthybmv_t<T>(op,
                                              alpha,
                                              A->m,
                                              A->n,
                                              A->nnz,
                                              (T *)A->ell_csr_hyb_mat.ell_val,
                                              A->ell_csr_hyb_mat.ell_col_ind,
                                              A->ell_csr_hyb_mat.ell_width,
                                              A->ell_csr_hyb_mat.ell_m,
                                              (T *)A->ell_csr_hyb_mat.csr_val,
                                              A->csr_mat.csr_row_ptr,
                                              A->csr_mat.csr_col_ptr,
                                              nullptr,
                                              A->ell_csr_hyb_mat.csr_row_id_map,
                                              descr,
                                              x,
                                              beta,
                                              y));
        case aoclsparse_csr_mat_br4:
            if constexpr(std::is_same_v<T, double>)
            {
                return (aoclsparse_dcsr_mat_br4(op, *alpha, A, descr, x, *beta, y));
            }
            else
            {
                return aoclsparse_status_not_implemented;
            }
        case aoclsparse_ell_csr_hyb_mat:
        case aoclsparse_dia_mat:
        case aoclsparse_ell_mat:
        case aoclsparse_csc_mat:
        case aoclsparse_coo_mat:
        default:
            return aoclsparse_status_invalid_value;
        }

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
                                                   y);*/
    }
    else if(descr->type == aoclsparse_matrix_type_symmetric)
    {
        // can dispatch our data directly
        // transposed and non-transposed operation
        // y = alpha A * x + beta y
        return aoclsparse_csrmv_symm_internal(descr_cpy.base,
                                              *alpha,
                                              A->m,
                                              descr_cpy.diag_type,
                                              descr_cpy.fill_mode,
                                              (T *)A->opt_csr_mat.csr_val,
                                              A->opt_csr_mat.csr_col_ptr,
                                              A->opt_csr_mat.csr_row_ptr,
                                              A->idiag,
                                              A->iurow,
                                              x,
                                              *beta,
                                              y);
    }
    else if(descr->type == aoclsparse_matrix_type_triangular)
    {
        aoclsparse_int *csr_start = nullptr, *csr_end = nullptr;
        //Triangular SPMV
        if(descr->fill_mode == aoclsparse_fill_mode_lower)
        {
            // y = alpha L * x + beta y
            csr_start = A->opt_csr_mat.csr_row_ptr;
            csr_end   = A->iurow;
        }
        else if(descr->fill_mode == aoclsparse_fill_mode_upper)
        {
            // y = alpha U * x + beta y
            csr_start = A->idiag;
            csr_end   = &A->opt_csr_mat.csr_row_ptr[1];
        }

        //kernels as per transpose operation
        if(op == aoclsparse_operation_none)
        {
            // Only double kernel is vectorized for this path
            if constexpr(std::is_same_v<T, double>)
            {
                return aoclsparse_csrmv_vectorized_avx2ptr(&descr_cpy,
                                                           *alpha,
                                                           A->m,
                                                           A->n,
                                                           A->nnz,
                                                           (T *)A->opt_csr_mat.csr_val,
                                                           A->opt_csr_mat.csr_col_ptr,
                                                           csr_start,
                                                           csr_end,
                                                           x,
                                                           *beta,
                                                           y);
            }
            else
            {
                return aoclsparse_csrmv_ref(&descr_cpy,
                                            *alpha,
                                            A->m,
                                            A->n,
                                            (T *)A->opt_csr_mat.csr_val,
                                            A->opt_csr_mat.csr_col_ptr,
                                            csr_start,
                                            csr_end,
                                            x,
                                            *beta,
                                            y);
            }
        }
        else if(op == aoclsparse_operation_transpose)
        {
            return aoclsparse_csrmvt_ptr(&descr_cpy,
                                         *alpha,
                                         A->m,
                                         A->n,
                                         (T *)A->opt_csr_mat.csr_val,
                                         A->opt_csr_mat.csr_col_ptr,
                                         csr_start,
                                         csr_end,
                                         x,
                                         *beta,
                                         y);
        }
    }
    return aoclsparse_status_not_implemented;
}

#endif // AOCLSPARSE_OPTMV_HPP
