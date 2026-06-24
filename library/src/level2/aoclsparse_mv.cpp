/* ************************************************************************
 * Copyright (c) 2022-2026 Advanced Micro Devices, Inc.
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
#include "aoclsparse_blkcsrmv.hpp"
#include "aoclsparse_bsrmv.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_csrmv.hpp"
#include "aoclsparse_ellmv.hpp"
#include "aoclsparse_error_check.hpp"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_magic_box.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_mv_helpers.hpp"
#include "aoclsparse_tcsr.hpp"

#include <shared_mutex>

/* templated SpMV for complex types - can be extended for floats and doubles*/
template <typename T>
aoclsparse_status aoclsparse::mv(aoclsparse_operation       op,
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

    if(!A->get_first_mtx_if_valid<aoclsparse::base_mtx>())
        return aoclsparse_status_invalid_pointer;

    if(!A->is_descr_matching(descr))
        return aoclsparse_status_invalid_value;

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

    aoclsparse_status status;
    aoclsparse::doid  d_id = aoclsparse::get_doid<T>(descr, op);
    aoclsparse_int    kid  = aoclsparse::get_kid(A->optim_data, d_id, aoclsparse_action_mv);

    // By default we will use our input format
    aoclsparse_matrix_format_type mtx_t = aoclsparse_uninitialized_mat;

    aoclsparse::base_mtx *best_mtx = aoclsparse::get_best_matrix<T>(A, d_id, mtx_t);

    // Get a runnable format: use best matrix, or CSR/CSC optimize for symm/herm/tri.
    // CSR/CSC: symm/herm/tri and best matrix is CSR but unoptimized and wrong doid.
    if(descr->type != aoclsparse_matrix_type_general && mtx_t != aoclsparse_tcsr_mat)
    {
        aoclsparse::csr *best_csr = dynamic_cast<aoclsparse::csr *>(best_mtx);
        const bool       best_needs_csr_optimize
            = (best_csr != nullptr && !best_csr->is_optimized && best_csr->doid != d_id);

        if(best_needs_csr_optimize)
        {
            aoclsparse::csr *opt_csr = nullptr;
            status                   = aoclsparse_csr_csc_optimize<T>(A, opt_csr);
            if(status != aoclsparse_status_success)
                return status;
            best_mtx = opt_csr;
            mtx_t    = aoclsparse_csr_mat;
        }
    }

    // Common DOID and descriptor adjustment for all matrix formats.
    // This block is placed before the format-specific switch/case so that every
    // format (CSR, BLKCSR, BSR, TCSR, etc.) receives consistently adjusted
    // descriptor and DOID values. Previously this logic lived only in the CSR
    // case, leading to duplication and inconsistency for other formats.
    //
    // We work on a local copy (descr_t) to avoid modifying the caller's descriptor.
    _aoclsparse_mat_descr descr_t;
    aoclsparse_copy_mat_descr(&descr_t, descr);
    bool exact_match = false;

    if(best_mtx)
    {
        // Align base indexing with the matrix's stored base (0-based or 1-based)
        descr_t.base = best_mtx->base;

        if(best_mtx->doid == d_id)
        {
            // Exact match: the matrix was optimized/stored for exactly the requested
            // operation+type. The kernel can treat it as a plain general (gn) matrix
            // with no special symmetry/triangular handling, since those properties
            // are already baked into the stored data. Reset all descriptor fields to
            // neutral values to reflect this.
            exact_match       = true;
            op                = aoclsparse_operation_none;
            descr_t.type      = aoclsparse_matrix_type_general;
            descr_t.fill_mode = aoclsparse_fill_mode_lower;
            descr_t.diag_type = aoclsparse_diag_type_non_unit;
            d_id              = doid::gn;
        }
        else
        {
            // Non-exact match: compute the kernel dispatch DOID from the stored
            // matrix's DOID and the requested DOID.
            d_id = aoclsparse::get_effective_doid(best_mtx->doid, d_id);

            // When the matrix is stored in transposed form (CSC = gt, or conj-transposed
            // = gh), the physical row/column layout is swapped relative to the logical
            // layout. This means upper triangle data is physically in the lower triangle
            // and vice versa, so we flip fill_mode to match the physical layout that
            // the kernel will traverse.
            if(best_mtx->doid == doid::gt || best_mtx->doid == doid::gh)
            {
                if(descr_t.fill_mode == aoclsparse_fill_mode_upper)
                    descr_t.fill_mode = aoclsparse_fill_mode_lower;
                else if(descr_t.fill_mode == aoclsparse_fill_mode_lower)
                    descr_t.fill_mode = aoclsparse_fill_mode_upper;
            }
        }
    }

    switch(mtx_t)
    {
    case aoclsparse_csr_mat:
    {
        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(best_mtx);

        if(!csr_mat)
        {
            return aoclsparse_status_not_implemented;
        }

        // CSR-specific diagonal adjustment for exact-match matrices.
        // When the matrix was stored for the exact requested DOID (e.g., symmetric-lower
        // with unit diagonal), the diagonal values in the CSR data may need to be
        // materialized or zeroed to match the descriptor's diag_type (unit vs zero vs
        // non-unit). This is only needed when the stored diag type (csr_mat->mtx_diag)
        // differs from what the user requested (descr->diag_type).
        // N.B. We compare against the *original* descr->diag_type (not descr_t) because
        // descr_t.diag_type was already reset to non_unit in the exact-match block above.
        if(exact_match && descr->diag_type != csr_mat->mtx_diag)
        {
            status = aoclsparse_set_mat_diag<T>(A->m, *descr, csr_mat);
            if(status != aoclsparse_status_success)
                return status;
        }

        return aoclsparse_csrmv_t<T, false>(op,
                                            alpha,
                                            csr_mat->m,
                                            csr_mat->n,
                                            csr_mat->nnz,
                                            (T *)csr_mat->val,
                                            csr_mat->ind,
                                            csr_mat->ptr,
                                            &descr_t,
                                            x,
                                            beta,
                                            y,
                                            csr_mat->idiag,
                                            csr_mat->iurow,
                                            d_id,
                                            kid);
    }
    case aoclsparse_blkcsr_mat:
        if constexpr(std::is_same_v<T, double>)
        {
            if(auto *blk_csr_mat = dynamic_cast<aoclsparse::blk_csr *>(best_mtx))
            {
                return aoclsparse_blkcsrmv_t<T>(op,
                                                alpha,
                                                A->m,
                                                A->n,
                                                A->nnz,
                                                blk_csr_mat->masks,
                                                (T *)blk_csr_mat->blk_val,
                                                blk_csr_mat->blk_col_ptr,
                                                blk_csr_mat->blk_row_ptr,
                                                &descr_t,
                                                x,
                                                beta,
                                                y,
                                                blk_csr_mat->nRowsblk);
            }
        }
        return aoclsparse_status_not_implemented;
    case aoclsparse_ellt_mat:
    case aoclsparse_ellt_csr_hyb_mat:
    {
        aoclsparse::csr *csr_mat = A->get_first_mtx_if_valid<aoclsparse::csr>();
        if(!csr_mat)
            return aoclsparse_status_not_implemented;
        std::shared_lock<std::shared_mutex> rlock(A->mats_guard);
        for(auto *mat : A->mats)
        {
            if(auto *ell_csr_hyb_mat = dynamic_cast<aoclsparse::ell_csr_hyb *>(mat))
            {
                return (aoclsparse_ellthybmv_t<T>(op,
                                                  alpha,
                                                  A->m,
                                                  A->n,
                                                  A->nnz,
                                                  (T *)ell_csr_hyb_mat->ell_val,
                                                  ell_csr_hyb_mat->ell_col_ind,
                                                  ell_csr_hyb_mat->ell_width,
                                                  ell_csr_hyb_mat->ell_m,
                                                  (T *)ell_csr_hyb_mat->csr_val,
                                                  csr_mat->ptr,
                                                  csr_mat->ind,
                                                  nullptr,
                                                  ell_csr_hyb_mat->csr_row_id_map,
                                                  &descr_t,
                                                  x,
                                                  beta,
                                                  y));
            }
        }
        return aoclsparse_status_invalid_pointer;
    }
    case aoclsparse_csr_mat_br4:
        if constexpr(std::is_same_v<T, double>)
        {
            return (aoclsparse_dcsr_mat_br4(op, *alpha, A, &descr_t, x, *beta, y));
        }
        else
        {
            return aoclsparse_status_not_implemented;
        }
    case aoclsparse_tcsr_mat:
    {
        return aoclsparse::tcsrmv(&descr_t, alpha, A, x, beta, y, d_id, kid);
    }
    case aoclsparse_bsr_mat:
    {
        // Use the best matrix found
        aoclsparse::bsr *bsr_m = dynamic_cast<aoclsparse::bsr *>(best_mtx);

        // nullptr check
        if(!bsr_m)
        {
            // If no matching matrix was found, return not implemented
            return aoclsparse_status_not_implemented;
        }

        if(bsr_m->order != aoclsparse_order_column)
        {
            // Currently only column major is supported
            return aoclsparse_status_not_implemented;
        }

        return aoclsparse::bsrmv<T>(op,
                                    alpha,
                                    bsr_m->bm,
                                    bsr_m->bn,
                                    bsr_m->block_dim,
                                    static_cast<const T *>(bsr_m->val),
                                    bsr_m->ind,
                                    bsr_m->ptr,
                                    &descr_t,
                                    x,
                                    beta,
                                    y);
    }
    default:
        return aoclsparse_status_not_implemented;
    }

    return aoclsparse_status_not_implemented;
}

#define MV_DISPATCHER(SUF)                                                                      \
    template DLL_PUBLIC aoclsparse_status aoclsparse::mv<SUF>(aoclsparse_operation       op,    \
                                                              const SUF                 *alpha, \
                                                              aoclsparse_matrix          A,     \
                                                              const aoclsparse_mat_descr descr, \
                                                              const SUF                 *x,     \
                                                              const SUF                 *beta,  \
                                                              SUF                       *y);

INSTANTIATE_FOR_ALL_TYPES(MV_DISPATCHER);

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
extern "C" aoclsparse_status aoclsparse_smv(aoclsparse_operation       op,
                                            const float               *alpha,
                                            aoclsparse_matrix          A,
                                            const aoclsparse_mat_descr descr,
                                            const float               *x,
                                            const float               *beta,
                                            float                     *y)
{
    return aoclsparse::mv<float>(op, alpha, A, descr, x, beta, y);
}

extern "C" aoclsparse_status aoclsparse_dmv(aoclsparse_operation       op,
                                            const double              *alpha,
                                            aoclsparse_matrix          A,
                                            const aoclsparse_mat_descr descr,
                                            const double              *x,
                                            const double              *beta,
                                            double                    *y)
{
    return aoclsparse::mv<double>(op, alpha, A, descr, x, beta, y);
}

extern "C" aoclsparse_status aoclsparse_cmv(aoclsparse_operation            op,
                                            const aoclsparse_float_complex *alpha,
                                            aoclsparse_matrix               A,
                                            const aoclsparse_mat_descr      descr,
                                            const aoclsparse_float_complex *x,
                                            const aoclsparse_float_complex *beta,
                                            aoclsparse_float_complex       *y)
{
    return aoclsparse::mv<std::complex<float>>(op,
                                               ((const std::complex<float> *)alpha),
                                               A,
                                               descr,
                                               (std::complex<float> *)x,
                                               ((const std::complex<float> *)beta),
                                               (std::complex<float> *)y);
}

extern "C" aoclsparse_status aoclsparse_zmv(aoclsparse_operation             op,
                                            const aoclsparse_double_complex *alpha,
                                            aoclsparse_matrix                A,
                                            const aoclsparse_mat_descr       descr,
                                            const aoclsparse_double_complex *x,
                                            const aoclsparse_double_complex *beta,
                                            aoclsparse_double_complex       *y)
{
    return aoclsparse::mv<std::complex<double>>(op,
                                                ((const std::complex<double> *)alpha),
                                                A,
                                                descr,
                                                (std::complex<double> *)x,
                                                ((const std::complex<double> *)beta),
                                                (std::complex<double> *)y);
}
