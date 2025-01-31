/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_blkcsrmv.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_csrmv.hpp"
#include "aoclsparse_ellmv.hpp"
#include "aoclsparse_error_check.hpp"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_mat_structures.hpp"

#include <complex>
#include <immintrin.h>

// Kernels
// -----------------------------------

aoclsparse_status aoclsparse_dtcsrmv_avx2(const aoclsparse_index_base base,
                                          const double                alpha,
                                          aoclsparse_int              m,
                                          const double *__restrict__ val_L,
                                          const double *__restrict__ val_U,
                                          const aoclsparse_int *__restrict__ col_idx_L,
                                          const aoclsparse_int *__restrict__ col_idx_U,
                                          const aoclsparse_int *__restrict__ row_ptr_L,
                                          const aoclsparse_int *__restrict__ row_ptr_U,
                                          const double *__restrict__ x,
                                          const double beta,
                                          double *__restrict__ y);
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

// Kernel to scale vectors
template <typename T>
aoclsparse_status vscale(T *v, T c, aoclsparse_int sz)
{
    T zero = 0;

    if(!v)
        return aoclsparse_status_invalid_pointer;

    if(c != zero)
    {
        for(aoclsparse_int i = 0; i < sz; i++)
            v[i] = c * v[i];
    }
    else
    {
        for(aoclsparse_int i = 0; i < sz; i++)
            v[i] = zero;
    }

    return aoclsparse_status_success;
}

template <typename T>
bool is_mtx_frmt_supported_mv(aoclsparse_matrix_format_type mtx_t)
{
    if constexpr(aoclsparse::is_dt_complex<T>())
    {
        // Only CSR is supported for complex types
        if(mtx_t != aoclsparse_csr_mat)
            return false;
    }
    else
    {
        // Only CSR and TCSR input format supported
        if(mtx_t != aoclsparse_csr_mat && mtx_t != aoclsparse_tcsr_mat)
            return false;
    }

    return true;
}

/* templated SpMV for complex types - can be extended for floats and doubles*/
template <typename T>
aoclsparse_status aoclsparse_mv_t(aoclsparse_operation       op,
                                  const T                   *alpha,
                                  aoclsparse_matrix          A,
                                  const aoclsparse_mat_descr descr,
                                  const T                   *x,
                                  const T                   *beta,
                                  T                         *y)
{
    using namespace aoclsparse;

    // Error handling ----------------------------------------------------------
    //--------------------------------------------------------------------------

    if(alpha == nullptr || beta == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(descr == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Check pointer arguments
    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    // Validate descriptor's and matrix' index base
    if(!is_valid_base(descr->base) || !is_valid_base(A->base))
    {
        return aoclsparse_status_invalid_value;
    }

    // Make sure the base index of descriptor and aoclsparse matrix are the same
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    // Check transpose
    if(!is_valid_op(op))
    {
        return aoclsparse_status_invalid_value;
    }

    // Make sure we have the right type before casting
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    if(!is_valid_mtx_t(descr->type))
        return aoclsparse_status_invalid_value;

    if((descr->type == aoclsparse_matrix_type_symmetric
        || descr->type == aoclsparse_matrix_type_hermitian)
       && A->m != A->n)
        return aoclsparse_status_invalid_size;

    // Datatype specific support check -----------------------------------------

    //--------------------------------------------------------------------------

    if(!is_mtx_frmt_supported_mv<T>(A->input_format))
        return aoclsparse_status_not_implemented;

    if constexpr(!is_dt_complex<T>())
    {
        // For real types, conjugate transpose is equal to transpose
        if(op == aoclsparse_operation_conjugate_transpose)
            op = aoclsparse_operation_transpose;

        if(descr->type == aoclsparse_matrix_type_hermitian)
            return aoclsparse_status_not_implemented;
    }

    //--------------------------------------------------------------------------

    /* Diag_type is applicable for symm/herm/tri matrices. Internal functions
     * will handle diag_type=unit and nnz=0 case for such matrices. General
     * matrix with nnz=0 should update y (aoclsparse_dcsrmv() has quick return
     * which doesn't update y. Hence adding this nnz=0 & general matrix check here.)
     */
    if(A->m == 0 || A->n == 0 || (A->nnz == 0 && descr->type == aoclsparse_matrix_type_general))
    {
        aoclsparse_int dim = op == aoclsparse_operation_none ? A->m : A->n;

        return vscale(y, *beta, dim);
    }

    aoclsparse_status     status;
    _aoclsparse_mat_descr descr_cpy;
    aoclsparse_int        kid  = -1;
    aoclsparse::doid      d_id = aoclsparse::get_doid<T>(descr, op);

    aoclsparse_optimize_data *ptr = A->optim_data;

    while(ptr != nullptr)
    {
        // The hint and doid should match
        if(ptr->act == aoclsparse_action_mv && d_id == ptr->doid)
        {
            kid = A->optim_data->kid;
            break;
        }

        ptr = ptr->next;
    }

    aoclsparse_copy_mat_descr(&descr_cpy, descr);

    /*
     *  Optimize is called only for triangular,
     *  symmetric and Hermitian matrices.
     */
    if(descr->type != aoclsparse_matrix_type_general)
    {
        if(!A->opt_csr_ready)
        {
            if constexpr(!aoclsparse::is_dt_complex<T>())
            {
                if(A->input_format == aoclsparse_tcsr_mat)
                    status = aoclsparse_tcsr_optimize<T>(A);
                else
                    status = aoclsparse_csr_csc_optimize<T>(A);
            }
            else
            {
                status = aoclsparse_csr_csc_optimize<T>(A);
            }
            if(status)
                return status;
        }
        descr_cpy.base = A->internal_base_index;
    }

    if constexpr(is_dt_complex<T>())
    {
        /*
        *  Optimize is called only for triangular, symmetric and Hermitian matrices.
        *  So, the pointers and descriptor passed as parameters to CSRMV interface needs
        *  to be modified accordingly.
        */
        T              *val;
        aoclsparse_int *col;
        aoclsparse_int *row;
        aoclsparse_int *idiag;
        aoclsparse_int *irow;

        _aoclsparse_mat_descr descr_t;

        if(d_id == doid::gn || d_id == doid::gt || d_id == doid::gh || d_id == doid::gc)
        {
            val   = (T *)A->csr_mat.csr_val;
            col   = A->csr_mat.csr_col_ptr;
            row   = A->csr_mat.csr_row_ptr;
            idiag = nullptr;
            irow  = nullptr;
            aoclsparse_copy_mat_descr(&descr_t, descr);
        }
        else
        {
            // everything else uses optimized matrix
            val   = (T *)A->opt_csr_mat.csr_val;
            col   = A->opt_csr_mat.csr_col_ptr;
            row   = A->opt_csr_mat.csr_row_ptr;
            idiag = A->idiag;
            irow  = A->iurow;
            aoclsparse_copy_mat_descr(&descr_t, &descr_cpy);
        }

        // Invoke CSRMV interface with do_check set to false
        return aoclsparse_csrmv_t<T, false>(op,
                                            alpha,
                                            A->m,
                                            A->n,
                                            A->nnz,
                                            val,
                                            col,
                                            row,
                                            &descr_t,
                                            x,
                                            beta,
                                            y,
                                            idiag,
                                            irow,
                                            d_id,
                                            kid);
    }
    else // Real datatypes
    {
        switch(d_id)
        {
        case doid::gn:
        case doid::gt:
        case doid::gh:
        case doid::gc:
        {
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

                return aoclsparse_csrmv_t<T>(op,
                                             alpha,
                                             A->m,
                                             A->n,
                                             A->nnz,
                                             (T *)A->csr_mat.csr_val,
                                             A->csr_mat.csr_col_ptr,
                                             A->csr_mat.csr_row_ptr,
                                             descr,
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
            case aoclsparse_tcsr_mat:
                if constexpr(std::is_same_v<T, double>)
                {
                    if(op == aoclsparse_operation_none)
                    {
                        return (aoclsparse_dtcsrmv_avx2(descr->base,
                                                        *alpha,
                                                        A->m,
                                                        (double *)A->tcsr_mat.val_L,
                                                        (double *)A->tcsr_mat.val_U,
                                                        A->tcsr_mat.col_idx_L,
                                                        A->tcsr_mat.col_idx_U,
                                                        A->tcsr_mat.row_ptr_L,
                                                        A->tcsr_mat.row_ptr_U,
                                                        x,
                                                        *beta,
                                                        y));
                    }
                }
                return aoclsparse_status_not_implemented;
            default:
                return aoclsparse_status_invalid_value;
            }
        }
        case doid::hl: // Hermitian maps to symmetric in case of real datatypes
        case doid::hu:
        case doid::hlc:
        case doid::huc: // but we return early above because tests are written like that
        case doid::sl: // sl, su, slc and suc map to the same path
        case doid::su:
        case doid::slc:
        case doid::suc:
        {
            // can dispatch our data directly
            // transposed and non-transposed operation
            // y = alpha A * x + beta y

            T              *csr_val = nullptr;
            aoclsparse_int *csr_col = nullptr, *csr_crow = nullptr, *csr_diag = nullptr,
                           *csr_urow = nullptr;
            if(A->mat_type != aoclsparse_tcsr_mat) // CSR Matrix
            {
                csr_val  = (T *)A->opt_csr_mat.csr_val;
                csr_col  = A->opt_csr_mat.csr_col_ptr;
                csr_crow = A->opt_csr_mat.csr_row_ptr;
                csr_diag = A->idiag;
                csr_urow = A->iurow;
            }
            else // TCSR Matrix
            {
                if(descr->fill_mode == aoclsparse_fill_mode_lower)
                {
                    csr_val  = (T *)A->tcsr_mat.val_L;
                    csr_col  = A->tcsr_mat.col_idx_L;
                    csr_crow = A->tcsr_mat.row_ptr_L;
                    csr_diag = A->idiag;
                    csr_urow = A->tcsr_mat.row_ptr_L + 1;
                }
                // fill mode: upper
                else
                {
                    csr_val  = (T *)A->tcsr_mat.val_U;
                    csr_col  = A->tcsr_mat.col_idx_U;
                    csr_crow = A->tcsr_mat.row_ptr_U;
                    csr_diag = A->tcsr_mat.row_ptr_U;
                    csr_urow = A->iurow;
                }
            }
            {
                using K
                    = decltype(&aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b256, T, true>);
                K kernel = aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b256, T, true>;
#ifdef USE_AVX512
                if(context::get_context()->supports<context_isa_t::AVX512F>())
                    kernel = aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b512, T, true>;
#endif
                return kernel(descr_cpy.base,
                              *alpha,
                              A->m,
                              descr_cpy.diag_type,
                              descr_cpy.fill_mode,
                              csr_val,
                              csr_col,
                              csr_crow,
                              csr_diag,
                              csr_urow,
                              x,
                              *beta,
                              y);
            }
        }
        case doid::tln:
        case doid::tlt:
        case doid::tlh:
        case doid::tlc:
        case doid::tun:
        case doid::tut:
        case doid::tuh:
        case doid::tuc:
        {
            T              *csr_val     = nullptr;
            aoclsparse_int *csr_col_ind = nullptr, *csr_start = nullptr, *csr_end = nullptr;

            if(A->mat_type != aoclsparse_tcsr_mat) // CSR matrix
            {
                csr_val     = (T *)A->opt_csr_mat.csr_val;
                csr_col_ind = A->opt_csr_mat.csr_col_ptr;
                if(descr->fill_mode == aoclsparse_fill_mode_lower)
                {
                    csr_start = A->opt_csr_mat.csr_row_ptr;
                    csr_end   = A->iurow;
                }
                else
                {
                    csr_start = A->idiag;
                    csr_end   = &A->opt_csr_mat.csr_row_ptr[1];
                }
            }
            else // TCSR matrix
            {
                if(descr->fill_mode == aoclsparse_fill_mode_lower)
                {
                    csr_val     = (T *)A->tcsr_mat.val_L;
                    csr_col_ind = A->tcsr_mat.col_idx_L;
                    csr_start   = A->tcsr_mat.row_ptr_L;
                    csr_end     = A->tcsr_mat.row_ptr_L + 1;
                }
                else
                {
                    csr_val     = (T *)A->tcsr_mat.val_U;
                    csr_col_ind = A->tcsr_mat.col_idx_U;
                    csr_start   = A->tcsr_mat.row_ptr_U;
                    csr_end     = A->tcsr_mat.row_ptr_U + 1;
                }
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
                                                               csr_val,
                                                               csr_col_ind,
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
                                                csr_val,
                                                csr_col_ind,
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
                                             csr_val,
                                             csr_col_ind,
                                             csr_start,
                                             csr_end,
                                             x,
                                             *beta,
                                             y);
            }
        }
        break;
        default:
            return aoclsparse_status_internal_error;
        }

        return aoclsparse_status_not_implemented;
    }
}

#endif // AOCLSPARSE_OPTMV_HPP
