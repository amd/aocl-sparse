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

#include "aoclsparse.h"
#include "aoclsparse_bsrmv.hpp"
#include "aoclsparse_bsrmv_kr.hpp"
#include "aoclsparse_cntx_dispatcher.hpp"
#include "aoclsparse_l2_kt.hpp"

namespace aoclsparse
{
    template <typename T, size_t dim, bool alpha_s, bool beta_s>
    auto get_kernel()
    {
        using namespace kernel_templates;
        using namespace Dispatch;

        // Kernel type declaration for dispatcher ORL
        using K = decltype(&aoclsparse::bsrmv_nxn<T, 8, alpha_s, beta_s>);
#ifdef USE_AVX512
        constexpr bsz max_isa = bsz::b512;
#else
        constexpr bsz max_isa = bsz::b256;
#endif

        if constexpr(std::is_same_v<T, double>)
        {
            if constexpr(dim == 4)
                return aoclsparse::bsrmv_nxn_v<bsz::b256, T, 4, alpha_s, beta_s>;
            else if constexpr(dim == 8)
                return ORL<K>(aoclsparse::bsrmv_nxn_v<max_isa, T, 8, alpha_s, beta_s>,
                              aoclsparse::bsrmv_nxn_v<bsz::b256, T, 8, alpha_s, beta_s>);
            else if constexpr(dim == 16)
                return aoclsparse::bsrmv_nxn_v<max_isa, T, 16, alpha_s, beta_s>;
        }
        else if constexpr(std::is_same_v<T, float>)
        {
            if constexpr(dim == 8)
                return aoclsparse::bsrmv_nxn_v<bsz::b256, T, 8, alpha_s, beta_s>;
            else if constexpr(dim == 16)
                return ORL<K>(aoclsparse::bsrmv_nxn_v<max_isa, T, 16, alpha_s, beta_s>,
                              aoclsparse::bsrmv_nxn_v<bsz::b256, T, 16, alpha_s, beta_s>);
        }

        return aoclsparse::bsrmv_nxn<T, dim, alpha_s, beta_s>;
    }

    // To-do instantiate dispatcher and remove static
    template <typename T, bool alpha_s, bool beta_s>
    static aoclsparse_status bsrmv(aoclsparse_operation       trans,
                                   const T                   *alpha,
                                   aoclsparse_int             mb,
                                   aoclsparse_int             nb,
                                   aoclsparse_int             bsr_dim,
                                   const T                   *bsr_val,
                                   const aoclsparse_int      *bsr_col_ind,
                                   const aoclsparse_int      *bsr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const T                   *x,
                                   const T                   *beta,
                                   T                         *y)
    {
        using namespace kernel_templates;
        using namespace Dispatch;

        if(descr == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }

        if((descr->base != aoclsparse_index_base_zero)
           && (descr->base != aoclsparse_index_base_one))
        {
            return aoclsparse_status_invalid_value;
        }

        if(descr->type != aoclsparse_matrix_type_general)
        {
            // TODO
            return aoclsparse_status_not_implemented;
        }

        if(trans != aoclsparse_operation_none)
        {
            // TODO
            return aoclsparse_status_not_implemented;
        }

        // Check sizes
        if(mb < 0 || nb < 0 || bsr_dim <= 0)
        {
            return aoclsparse_status_invalid_size;
        }

        // Quick return if possible
        if(mb == 0 || nb == 0)
        {
            return aoclsparse_status_success;
        }

        // Check pointer arguments
        if(bsr_val == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(bsr_row_ptr == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(bsr_col_ind == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(x == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(y == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }

        using K = decltype(&aoclsparse::bsrmv_nxn<T, 2, alpha_s, beta_s>);

        K kernel;

        static K tbl[] = {get_kernel<T, 2, alpha_s, beta_s>(),
                          get_kernel<T, 3, alpha_s, beta_s>(),
                          get_kernel<T, 4, alpha_s, beta_s>(),
                          get_kernel<T, 5, alpha_s, beta_s>(),
                          get_kernel<T, 6, alpha_s, beta_s>(),
                          get_kernel<T, 7, alpha_s, beta_s>(),
                          get_kernel<T, 8, alpha_s, beta_s>(),
                          get_kernel<T, 16, alpha_s, beta_s>()};

        if((bsr_dim > 1 && bsr_dim <= 8) || bsr_dim == 16)
        {
            // For block size 16, kernel is 7th index.
            // For other block sizes, kernel is in block size - 2 index.
            kernel = (bsr_dim != 16) ? tbl[bsr_dim - 2] : tbl[7];

            return kernel(*alpha, mb, nb, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, *beta, y);
        }
        else
        {
            return aoclsparse::bsrmv_gn<T, alpha_s, beta_s>(
                *alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, *beta, y);
        }
    }
}

/*
* This is the main interface for BSRMV. With the dispatcher optimized for
* alpha and beta scalar' values.
* Note: Use this interface when adding to '_mv' interfaces.
*/
template <typename T>
aoclsparse_status aoclsparse::bsrmv(aoclsparse_operation       trans,
                                    const T                   *alpha,
                                    aoclsparse_int             mb,
                                    aoclsparse_int             nb,
                                    aoclsparse_int             bsr_dim,
                                    const T                   *bsr_val,
                                    const aoclsparse_int      *bsr_col_ind,
                                    const aoclsparse_int      *bsr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const T                   *x,
                                    const T                   *beta,
                                    T                         *y)
{
    using namespace aoclsparse;

    if(!alpha || !beta)
    {
        return aoclsparse_status_invalid_pointer;
    }

    T zero = aoclsparse_numeric::zero<T>();
    T one  = aoclsparse_numeric::one<T>();

    aoclsparse_status status;
    if(*alpha != one && *beta != zero)
        status = bsrmv<T, true, true>(
            trans, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, beta, y);
    else if(*alpha == one && *beta != zero)
        status = bsrmv<T, false, true>(
            trans, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, beta, y);
    else if(*alpha != one && *beta == zero)
        status = bsrmv<T, true, false>(
            trans, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, beta, y);
    else if(*alpha == one && *beta == zero)
        status = bsrmv<T, false, false>(
            trans, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, beta, y);

    return status;
}

#define BSRMV_DISP(SUF)                                                                       \
    template aoclsparse_status aoclsparse::bsrmv<SUF>(aoclsparse_operation       trans,       \
                                                      const SUF                 *alpha,       \
                                                      aoclsparse_int             mb,          \
                                                      aoclsparse_int             nb,          \
                                                      aoclsparse_int             bsr_dim,     \
                                                      const SUF                 *bsr_val,     \
                                                      const aoclsparse_int      *bsr_col_ind, \
                                                      const aoclsparse_int      *bsr_row_ptr, \
                                                      const aoclsparse_mat_descr descr,       \
                                                      const SUF                 *x,           \
                                                      const SUF                 *beta,        \
                                                      SUF                       *y);

// Instantiate for all supported types
INSTANTIATE_FOR_ALL_TYPES(BSRMV_DISP);

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

extern "C" aoclsparse_status aoclsparse_sbsrmv(aoclsparse_operation       trans,
                                               const float               *alpha,
                                               aoclsparse_int             mb,
                                               aoclsparse_int             nb,
                                               aoclsparse_int             bsr_dim,
                                               const float               *bsr_val,
                                               const aoclsparse_int      *bsr_col_ind,
                                               const aoclsparse_int      *bsr_row_ptr,
                                               const aoclsparse_mat_descr descr,
                                               const float               *x,
                                               const float               *beta,
                                               float                     *y)
{
    return aoclsparse::bsrmv<float>(
        trans, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, beta, y);
}

extern "C" aoclsparse_status aoclsparse_dbsrmv(aoclsparse_operation       trans,
                                               const double              *alpha,
                                               aoclsparse_int             mb,
                                               aoclsparse_int             nb,
                                               aoclsparse_int             bsr_dim,
                                               const double              *bsr_val,
                                               const aoclsparse_int      *bsr_col_ind,
                                               const aoclsparse_int      *bsr_row_ptr,
                                               const aoclsparse_mat_descr descr,
                                               const double              *x,
                                               const double              *beta,
                                               double                    *y)
{
    return aoclsparse::bsrmv<double>(
        trans, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, beta, y);
}
