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

#include "aoclsparse_descr.h"
#include "aoclsparse.hpp"
#include "aoclsparse_blkcsrmv.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_csrmv.hpp"
#include "aoclsparse_ellmv.hpp"
#include "aoclsparse_error_check.hpp"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_mv_helpers.hpp"
#include "aoclsparse_tcsr.hpp"

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

    if(A == nullptr || A->mats.empty())
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

    aoclsparse_status status;
    aoclsparse_int    kid  = -1;
    aoclsparse::doid  d_id = aoclsparse::get_doid<T>(descr, op);

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

    /*
     *  Optimize is called only for triangular,
     *  symmetric and Hermitian matrices.
     */
    bool is_optimized = false;
    for(auto mat : A->mats)
    {
        if(auto csr = dynamic_cast<aoclsparse::csr *>(mat); csr && csr->is_optimized)
        {
            is_optimized = true;
            break;
        }
        else if(auto tcsr = dynamic_cast<aoclsparse::tcsr *>(mat); tcsr && tcsr->is_optimized)
        {
            is_optimized = true;
            break;
        }
    }
    if(descr->type != aoclsparse_matrix_type_general)
    {
        if(!is_optimized)
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
    }

    // By default we will use our input format
    // but double and general SPMV might be optimized to a different format
    aoclsparse_matrix_format_type mtx_t = A->input_format;
    if constexpr(std::is_same_v<T, double>)
    {
        if(d_id == doid::gn)
        {
            if(A->mat_type == aoclsparse_csr_mat && A->blk_optimized)
                mtx_t = aoclsparse_blkcsr_mat;
            else
                mtx_t = A->mat_type;
        }
    }

    switch(mtx_t)
    {
    case aoclsparse_csr_mat:
    {
        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
        if(!csr_mat)
            return aoclsparse_status_not_implemented;

        aoclsparse::csr *copy_csr = nullptr;
        aoclsparse::csr *opt_csr  = nullptr;

        // Search for a matrix copy matching our descriptor/operation (DOID)
        for(size_t i = 1; i < A->mats.size(); i++)
        {
            aoclsparse::csr *csr_m = dynamic_cast<aoclsparse::csr *>(A->mats[i]);
            if(csr_m && csr_m->mat_type == mtx_t)
            {
                if(csr_m->doid == d_id)
                {
                    copy_csr = csr_m;
                    break;
                }
                else if(csr_m->is_optimized)
                {
                    opt_csr = csr_m;
                }
            }
        }

        _aoclsparse_mat_descr descr_t;
        aoclsparse_copy_mat_descr(&descr_t, descr);
        // If a matching matrix was found, update parameters accordingly
        if(copy_csr)
        {
            csr_mat           = copy_csr;
            op                = aoclsparse_operation_none;
            descr_t.type      = aoclsparse_matrix_type_general;
            descr_t.fill_mode = aoclsparse_fill_mode_lower;
            descr_t.diag_type = aoclsparse_diag_type_non_unit;
            descr_t.base      = copy_csr->base;
            d_id              = doid::gn;
        }
        // Otherwise, if not a general/triangular/hermitian/conjugate, use optimized if available
        else if(!(d_id == doid::gn || d_id == doid::gt || d_id == doid::gh || d_id == doid::gc)
                && opt_csr)
        {
            csr_mat      = opt_csr;
            descr_t.base = opt_csr->base;
        }

        // Invoke CSRMV interface with do_check set to false
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
            if(A->blk_optimized)
            {
                for(auto *mat : A->mats)
                {
                    if(auto *blk_csr_mat = dynamic_cast<aoclsparse::blk_csr *>(mat))
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
                                                        descr,
                                                        x,
                                                        beta,
                                                        y,
                                                        blk_csr_mat->nRowsblk);
                    }
                }
                return aoclsparse_status_invalid_pointer;
            }
        }
        return aoclsparse_status_not_implemented;
    case aoclsparse_ellt_mat:
    case aoclsparse_ellt_csr_hyb_mat:
    {
        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
        if(!csr_mat)
            return aoclsparse_status_not_implemented;
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
                                                  descr,
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
            return (aoclsparse_dcsr_mat_br4(op, *alpha, A, descr, x, *beta, y));
        }
        else
        {
            return aoclsparse_status_not_implemented;
        }
    case aoclsparse_tcsr_mat:
        return aoclsparse::tcsrmv(descr, alpha, A, x, beta, y, d_id, kid);
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
