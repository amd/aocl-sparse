/* ************************************************************************
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_CSRMV_HPP
#define AOCLSPARSE_CSRMV_HPP

#include "aoclsparse_cntx_dispatcher.hpp"
#include "aoclsparse_csrmv_avx512.hpp"
#include "aoclsparse_csrmv_kr.hpp"
#include "aoclsparse_error_check.hpp"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_mtx_dispatcher.hpp"

template <typename T, bool do_check = true>
aoclsparse_status aoclsparse_csrmv_t(aoclsparse_operation       trans,
                                     const T                   *alpha,
                                     aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     aoclsparse_int             nnz,
                                     const T                   *val,
                                     const aoclsparse_int      *col,
                                     const aoclsparse_int      *row,
                                     const aoclsparse_mat_descr descr,
                                     const T                   *x,
                                     const T                   *beta,
                                     T                         *y,
                                     const aoclsparse_int      *idiag = nullptr,
                                     const aoclsparse_int      *iurow = nullptr,
                                     aoclsparse::doid           d_id  = aoclsparse::doid::len,
                                     aoclsparse_int             kid   = -1)
{
    using namespace aoclsparse;
    using namespace Dispatch;
    using namespace kernel_templates;

    doid lcl_doid;

    if constexpr(!do_check)
    {
        // When the call is from the optmv interface
        // condition checking is skipped and the doid
        // from the optmv interface is used
        lcl_doid = d_id;
    }
    else
    {
        if(alpha == nullptr || beta == nullptr)
            return aoclsparse_status_invalid_pointer;

        if(descr == nullptr)
            return aoclsparse_status_invalid_pointer;

        // Check index base
        if(!is_valid_base(descr->base))
            return aoclsparse_status_invalid_value;

        if(!is_valid_mtx_t(descr->type))
            return aoclsparse_status_invalid_value;

        if(!is_valid_op(trans))
            return aoclsparse_status_invalid_value;

        // Support General and symmetric matrices.
        // Return for any other matrix type
        if((descr->type != aoclsparse_matrix_type_general)
           && (descr->type != aoclsparse_matrix_type_symmetric))
        {
            // TODO
            return aoclsparse_status_not_implemented;
        }

        if((descr->type == aoclsparse_matrix_type_symmetric
            || descr->type == aoclsparse_matrix_type_hermitian)
           && m != n)
            return aoclsparse_status_invalid_size;

        // Check sizes
        if(m < 0 || n < 0 || nnz < 0)
            return aoclsparse_status_invalid_size;

        // Check pointer arguments
        if(val == nullptr || row == nullptr || col == nullptr || x == nullptr || y == nullptr)
            return aoclsparse_status_invalid_pointer;

        // TODO extend for symmetric when using the new kernels
        if(descr->type == aoclsparse_matrix_type_triangular && (!idiag || !iurow))
            return aoclsparse_status_invalid_pointer;

        // When the call is from the public interface of csrmv build your own doid
        lcl_doid = get_doid<T>(descr, trans);
    }

    // pointers to start/end of the approriate triangle
    const aoclsparse_int *rstart = nullptr, *rend = nullptr;
    if(lcl_doid == doid::tln || lcl_doid == doid::tlt || lcl_doid == doid::tlh
       || lcl_doid == doid::tlc)
    {
        rstart = row;
        rend   = iurow;
    }
    else if(lcl_doid == doid::tun || lcl_doid == doid::tut || lcl_doid == doid::tuh
            || lcl_doid == doid::tuc)
    {
        rstart = idiag;
        rend   = &row[1];
    }

    if constexpr(is_dt_complex<T>())
    {
        switch(lcl_doid)
        {
        case doid::gn:
        {
            using K = decltype(&aoclsparse::csrmv_kt<kernel_templates::bsz::b256, T>);

            // Move to magic box
            // If kid is not set and size range matches, change the kernel
            if((nnz <= (4 * m)) && (kid == -1))
                kid = 0;
            else
                kid = (kid == -1) ? 1 : kid; // AVX2 default

            // clang-format off
            // Table of available kernels
            static constexpr Table<K> tbl[]{
                {ref_csrmv_gn<T>,           context_isa_t::GENERIC, 0U | archs::ALL},
                {aoclsparse::csrmv_kt<bsz::b256, T>,    context_isa_t::AVX2,    0U | archs::ALL},
                {aoclsparse::csrmv_kt<bsz::b256, T>,    context_isa_t::AVX2,    0U | archs::ALL}, // alias
            ORL<K>({aoclsparse::csrmv_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ALL})
            };
            //clang-format on

            // Thread local kernel cache
            thread_local K kache  = nullptr;
            K              kernel = Oracle<K>(tbl, kache, kid);

            if(!kernel)
                return aoclsparse_status_invalid_kid;

            return kernel(descr->base, *alpha, m, val, col, row, x, *beta, y);
        }
        case doid::gt:
        {
            using K = decltype(&aoclsparse::csrmvt_kt<kernel_templates::bsz::b256, T>);

            // AVX2 default
            kid = (kid == -1) ? 1 : kid;

            // clang-format off
            // Table of available kernels
            static constexpr Table<K> tbl[]{
                {aoclsparse::csrmvt_kt<bsz::b256, T>,    context_isa_t::GENERIC, 0U | archs::ALL}, // Replace with reference
                {aoclsparse::csrmvt_kt<bsz::b256, T>,    context_isa_t::AVX2,    0U | archs::ALL},
                {aoclsparse::csrmvt_kt<bsz::b256, T>,    context_isa_t::AVX2,    0U | archs::ALL}, // alias
            ORL<K>({aoclsparse::csrmvt_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ALL})
            };
            //clang-format on

            // Thread local kernel cache
            thread_local K kache  = nullptr;
            K              kernel = Oracle<K>(tbl, kache, kid);

            if(!kernel)
                return aoclsparse_status_invalid_kid;

            return kernel(descr->base, *alpha, m, n, val, col, row, x, *beta, y);
        }
        case doid::gh:
            return ref_csrmv_th<T, true>(descr->base, *alpha, m, n, val, col, row, x, *beta, y);
        case doid::gc:
            break;
        case doid::sl: // sl and su map to the same kernel
        case doid::su:
        {
            using K  = decltype(&aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b256, T, false>);

            // clang-format off
            // Table of available kernels
            static constexpr Table<K> tbl[]{
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, false>,    context_isa_t::GENERIC, 0U | archs::ALL}, // Replace with reference
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, false>,    context_isa_t::AVX2,    0U | archs::ALL},
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, false>,    context_isa_t::AVX2,    0U | archs::ALL}, // alias
            ORL<K>({aoclsparse::csrmv_symm_kt<bsz::b512, T, false>, context_isa_t::AVX512F, 0U | archs::ALL})
            };
            //clang-format on

            // Thread local kernel cache
            thread_local K kache  = nullptr;
            K              kernel = Oracle<K>(tbl, kache, kid);

            if(!kernel)
                return aoclsparse_status_invalid_kid;

            return kernel(descr->base,
                          *alpha,
                          m,
                          descr->diag_type,
                          descr->fill_mode,
                          val,
                          col,
                          row,
                          idiag,
                          iurow,
                          x,
                          *beta,
                          y);
        }
        case doid::slc: // slc and suc map to the same kernel
        case doid::suc:
            return aoclsparse_csrmvh_symm_internal(descr->base,
                                                   *alpha,
                                                   m,
                                                   descr->diag_type,
                                                   descr->fill_mode,
                                                   val,
                                                   col,
                                                   row,
                                                   idiag,
                                                   iurow,
                                                   x,
                                                   *beta,
                                                   y);
        case doid::hl: // hl and hu map to the same kernel
        case doid::hu:
        {
            using K  = decltype(&aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b256, T, true>);

            // clang-format off
            // Table of available kernels
            static constexpr Table<K> tbl[]{
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, true>,    context_isa_t::GENERIC, 0U | archs::ALL}, // Replace with reference
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, true>,    context_isa_t::AVX2,    0U | archs::ALL},
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, true>,    context_isa_t::AVX2,    0U | archs::ALL}, // alias
            ORL<K>({aoclsparse::csrmv_symm_kt<bsz::b512, T, true>, context_isa_t::AVX512F, 0U | archs::ALL})
            };
            //clang-format on

            // Thread local kernel cache
            thread_local K kache  = nullptr;
            K              kernel = Oracle<K>(tbl, kache, kid);

            if(!kernel)
                return aoclsparse_status_invalid_kid;

            return kernel(descr->base,
                        *alpha,
                        m,
                        descr->diag_type,
                        descr->fill_mode,
                        val,
                        col,
                        row,
                        idiag,
                        iurow,
                        x,
                        *beta,
                        y);
        }
        case doid::hlc: // hlc and huc map to the same kernel
        case doid::huc:
            return aoclsparse_csrmv_hermt_internal(descr->base,
                                                   *alpha,
                                                   m,
                                                   descr->diag_type,
                                                   descr->fill_mode,
                                                   val,
                                                   col,
                                                   row,
                                                   idiag,
                                                   iurow,
                                                   x,
                                                   *beta,
                                                   y);
        case doid::tln:
        case doid::tun:
            return ref_csrmv_tri(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlt:
        case doid::tut:
            return ref_csrmv_tri_th(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlh:
        case doid::tuh:
            return ref_csrmv_tri_th<T, true>(
                descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlc:
        case doid::tuc:
            break;
        default:
            return aoclsparse_status_internal_error;
        }

        return aoclsparse_status_not_implemented;
    }
    else // For real datatypes
    {

        switch(lcl_doid)
        {
        case doid::gn:
            if constexpr(std::is_same_v<T, float>)
            {
                return aoclsparse_csrmv_vectorized(
                    descr->base, *alpha, m, val, col, row, x, *beta, y);
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                using K = decltype(&aoclsparse::ref_csrmv_gn<T>);

                // Sparse matrices with Mean nnz = nnz/m <10 have very few non-zeroes in most of the rows
                // and few unevenly long rows . Loop unrolling and vectorization doesnt optimise performance
                // for this category of matrices . Hence , we invoke the generic dcsrmv kernel without
                // vectorization and innerloop unrolling . For the other category of sparse matrices
                // (Mean nnz > 10) , we continue to invoke the vectorised version of csrmv , since
                // it improves performance.
                if(nnz <= (10 * m))
                    kid = 0;


                // clang-format off
                // Table of available kernels
                static constexpr Table<K> tbl[]{
                    {ref_csrmv_gn,             context_isa_t::GENERIC, 0U | archs::ALL},
                    {aoclsparse_csrmv_vectorized_avx2,     context_isa_t::AVX2,    0U | archs::ALL},
                    {aoclsparse_csrmv_vectorized_avx2,     context_isa_t::AVX2,    0U | archs::ALL}, // alias
                ORL<K>({aoclsparse_csrmv_vectorized_avx512, context_isa_t::AVX512F, 0U | archs::ALL})
                };
                //clang-format on

                // Thread local kernel cache
                thread_local K kache  = nullptr;
                K              kernel = Oracle<K>(tbl, kache, kid);

                if(!kernel)
                    return aoclsparse_status_invalid_kid;

                // Invoke the kernel
                return kernel(descr->base, *alpha, m, val, col, row, x, *beta, y);
            }
        case doid::gt:
        {
            using K = decltype(&aoclsparse::csrmvt_kt<kernel_templates::bsz::b256, T>);

            // clang-format off
            // Table of available kernels
            static constexpr Table<K> tbl[]{
                {aoclsparse::csrmvt_kt<bsz::b256, T>,    context_isa_t::GENERIC, 0U | archs::ALL}, // Replace with reference
                {aoclsparse::csrmvt_kt<bsz::b256, T>,    context_isa_t::AVX2,    0U | archs::ALL},
                {aoclsparse::csrmvt_kt<bsz::b256, T>,    context_isa_t::AVX2,    0U | archs::ALL}, // alias
            ORL<K>({aoclsparse::csrmvt_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ALL})
            };
            //clang-format on

            // Thread local kernel cache
            thread_local K kache  = nullptr;
            K              kernel = Oracle<K>(tbl, kache, kid);

            if(!kernel)
                return aoclsparse_status_invalid_kid;

            return kernel(descr->base, *alpha, m, n, val, col, row, x, *beta, y);
        }
        case doid::hl: // Hermitian maps to symmetric in case of real datatypes
        case doid::hu:
        case doid::hlc:
        case doid::huc: // but we return early above because tests are written like that
        case doid::sl: // sl, su, slct and suct map to the same path
        case doid::su:
        case doid::slc:
        case doid::suc:
            // If the call is not from the MV interface, then the matrix will not be optimized
            if constexpr(do_check)
            {
                return aoclsparse_csrmv_symm(descr->base, *alpha, m, val, col, row, x, *beta, y);
            }
            else
            {
                using K  = decltype(&aoclsparse::csrmv_symm_kt<kernel_templates::bsz::b256, T, true>);

            // clang-format off
            // Table of available kernels
            static constexpr Table<K> tbl[]{
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, true>,    context_isa_t::GENERIC, 0U | archs::ALL}, // Replace with reference
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, true>,    context_isa_t::AVX2,    0U | archs::ALL},
                {aoclsparse::csrmv_symm_kt<bsz::b256, T, true>,    context_isa_t::AVX2,    0U | archs::ALL}, // alias
            ORL<K>({aoclsparse::csrmv_symm_kt<bsz::b512, T, true>, context_isa_t::AVX512F, 0U | archs::ALL})
            };
            //clang-format on

            // Thread local kernel cache
            thread_local K kache  = nullptr;
            K              kernel = Oracle<K>(tbl, kache, kid);

            if(!kernel)
                return aoclsparse_status_invalid_kid;

            return kernel(descr->base,
                            *alpha,
                            m,
                            descr->diag_type,
                            descr->fill_mode,
                            val,
                            col,
                            row,
                            idiag,
                            iurow,
                            x,
                            *beta,
                            y);
            }
        case doid::tln:
        case doid::tun:
            if constexpr(std::is_same_v<T, double>)
            {
                return aoclsparse_csrmv_vectorized_avx2ptr(
                    descr, *alpha, m, n, nnz, val, col, rstart, rend, x, *beta, y);
            }
            else
            {
                return ref_csrmv_tri(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
            }
        case doid::tlt:
        case doid::tut:
            return ref_csrmv_tri_th(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlh:
        case doid::tlc:
        case doid::tuh:
        case doid::tuc:

        default:
            return aoclsparse_status_not_implemented;
        }
    }
}

#endif // AOCLSPARSE_CSRMV_HPP
