/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR mtx PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef AOCLSPARSE_TCSR_HPP
#define AOCLSPARSE_TCSR_HPP
#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_mtx_dispatcher.hpp"

// Kernels
// ----------------------------------------------------------------------------
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

namespace aoclsparse
{
    template <typename T>
    aoclsparse_status tcsrmv(const aoclsparse_mat_descr descr,
                             const T                   *alpha,
                             const aoclsparse_matrix    mtx,
                             const T *__restrict__ x,
                             const T *beta,
                             T *__restrict__ y,
                             aoclsparse::doid                doid,
                             [[maybe_unused]] aoclsparse_int kid = -1)
    {
        T              *val = nullptr;
        aoclsparse_int *col = nullptr, *crow = nullptr, *diag = nullptr, *urow = nullptr,
                       *rstart = nullptr, *rend = nullptr;

        // The user creates the tcsr matrix, which should be located at mtx->mats[0]
        aoclsparse::tcsr *tcsr_mat = dynamic_cast<aoclsparse::tcsr *>(mtx->mats[0]);
        if(!tcsr_mat)
        {
            return aoclsparse_status_not_implemented;
        }

        if(doid == doid::tln || doid == doid::tlt || doid == doid::tlh || doid == doid::tlc)
        {
            val    = (T *)tcsr_mat->val_L;
            col    = tcsr_mat->col_idx_L;
            rstart = tcsr_mat->row_ptr_L;
            rend   = tcsr_mat->row_ptr_L + 1;
        }
        else if(doid == doid::tun || doid == doid::tut || doid == doid::tuh || doid == doid::tuc)
        {
            val    = (T *)tcsr_mat->val_U;
            col    = tcsr_mat->col_idx_U;
            rstart = tcsr_mat->row_ptr_U;
            rend   = tcsr_mat->row_ptr_U + 1;
        }
        else if(doid == doid::sl || doid == doid::slc || doid == doid::hl || doid == doid::hlc)
        {
            val  = (T *)tcsr_mat->val_L;
            col  = tcsr_mat->col_idx_L;
            crow = tcsr_mat->row_ptr_L;
            diag = tcsr_mat->idiag;
            urow = tcsr_mat->row_ptr_L + 1;
        }
        else if(doid == doid::su || doid == doid::suc || doid == doid::hu || doid == doid::huc)
        {
            val  = (T *)tcsr_mat->val_U;
            col  = tcsr_mat->col_idx_U;
            crow = tcsr_mat->row_ptr_U;
            diag = tcsr_mat->row_ptr_U;
            urow = tcsr_mat->iurow;
        }

        // TCSR dispatcher
        switch(doid)
        {
        case doid::gn:
            if constexpr(std::is_same_v<double, T>)
                return aoclsparse_dtcsrmv_avx2(descr->base,
                                               *alpha,
                                               mtx->m,
                                               (double *)tcsr_mat->val_L,
                                               (double *)tcsr_mat->val_U,
                                               tcsr_mat->col_idx_L,
                                               tcsr_mat->col_idx_U,
                                               tcsr_mat->row_ptr_L,
                                               tcsr_mat->row_ptr_U,
                                               x,
                                               *beta,
                                               y);
            else
                return aoclsparse_status_not_implemented;
        case doid::gt:
        case doid::gh:
        case doid::gc:
            return aoclsparse_status_not_implemented;
        case doid::hl: // Hermitian maps to symmetric in case of real datatypes
        case doid::hu:
        case doid::hlc:
        case doid::huc: // but we return early above because tests are written like that
        case doid::sl: // sl, su, slc and suc map to the same path
        case doid::su:
        case doid::slc:
        case doid::suc:
        {
            using K  = decltype(&aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b256, T, true>);
            K kernel = aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b256, T, true>;
#ifdef USE_AVX512
            if(context::get_context()->supports<context_isa_t::AVX512F>())
                kernel = aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b512, T, true>;
#endif
            return kernel(descr->base,
                          *alpha,
                          mtx->m,
                          descr->diag_type,
                          descr->fill_mode,
                          val,
                          col,
                          crow,
                          diag,
                          urow,
                          x,
                          *beta,
                          y);
        }
            /*return aoclsparse_csrmv_symm_internal(descr->base,
                                                  *alpha,
                                                  mtx->m,
                                                  descr->diag_type,
                                                  descr->fill_mode,
                                                  val,
                                                  col,
                                                  crow,
                                                  diag,
                                                  urow,
                                                  x,
                                                  *beta,
                                                  y);*/
        case doid::tln:
        case doid::tun:
            // Only double kernel is vectorized for this path
            if constexpr(std::is_same_v<T, double>)
            {
                return aoclsparse_csrmv_vectorized_avx2ptr(
                    descr, *alpha, mtx->m, mtx->n, mtx->nnz, val, col, rstart, rend, x, *beta, y);
            }
            else
            {
                return ref_csrmv_tri(
                    descr, *alpha, mtx->m, mtx->n, val, col, rstart, rend, x, *beta, y);
            }
        case doid::tlt:
        case doid::tut:
            return ref_csrmv_tri_th(
                descr, *alpha, mtx->m, mtx->n, val, col, rstart, rend, x, *beta, y);
        case doid::tlh:
        case doid::tlc:
            return aoclsparse_status_not_implemented;
        case doid::tuh:
        case doid::tuc:
            return aoclsparse_status_not_implemented;
        default:
            return aoclsparse_status_internal_error;
        }

        // Should never reach
        // Kept in-place to catch bugs
        return aoclsparse_status_not_implemented;
    }

}

#endif
