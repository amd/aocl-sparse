/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.
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
#include "aoclsparse_context.h"
#include "aoclsparse_ellmv.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
extern "C" aoclsparse_status aoclsparse_sellmv(aoclsparse_operation            trans,
                                               const float                    *alpha,
                                               aoclsparse_int                  m,
                                               aoclsparse_int                  n,
                                               [[maybe_unused]] aoclsparse_int nnz,
                                               const float                    *ell_val,
                                               const aoclsparse_int           *ell_col_ind,
                                               aoclsparse_int                  ell_width,
                                               const aoclsparse_mat_descr      descr,
                                               const float                    *x,
                                               const float                    *beta,
                                               float                          *y)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
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
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(ell_width < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Sanity check
    if((m == 0 || n == 0) && ell_width != 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
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

    return aoclsparse_ellmv_template(
        *alpha, m, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
}

extern "C" aoclsparse_status aoclsparse_dellmv(aoclsparse_operation       trans,
                                               const double              *alpha,
                                               aoclsparse_int             m,
                                               aoclsparse_int             n,
                                               aoclsparse_int             nnz,
                                               const double              *ell_val,
                                               const aoclsparse_int      *ell_col_ind,
                                               aoclsparse_int             ell_width,
                                               const aoclsparse_mat_descr descr,
                                               const double              *x,
                                               const double              *beta,
                                               double                    *y)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
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
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(ell_width < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Sanity check
    if((m == 0 || n == 0) && ell_width != 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
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

    using namespace aoclsparse;

#if USE_AVX512
    if(context::get_context()->supports<context_isa_t::AVX512F>())
        return aoclsparse_ellmv_template_avx512(
            *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
    else
        return aoclsparse_ellmv_template_avx2(
            *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
#else
    return aoclsparse_ellmv_template_avx2(
        *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
#endif
}

extern "C" aoclsparse_status aoclsparse_selltmv(aoclsparse_operation       trans,
                                                const float               *alpha,
                                                aoclsparse_int             m,
                                                aoclsparse_int             n,
                                                aoclsparse_int             nnz,
                                                const float               *ell_val,
                                                const aoclsparse_int      *ell_col_ind,
                                                aoclsparse_int             ell_width,
                                                const aoclsparse_mat_descr descr,
                                                const float               *x,
                                                const float               *beta,
                                                float                     *y)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
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
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(ell_width < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Sanity check
    if((m == 0 || n == 0) && ell_width != 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
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

    return aoclsparse_elltmv_template(
        *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
}

extern "C" aoclsparse_status aoclsparse_delltmv(aoclsparse_operation       trans,
                                                const double              *alpha,
                                                aoclsparse_int             m,
                                                aoclsparse_int             n,
                                                aoclsparse_int             nnz,
                                                const double              *ell_val,
                                                const aoclsparse_int      *ell_col_ind,
                                                aoclsparse_int             ell_width,
                                                const aoclsparse_mat_descr descr,
                                                const double              *x,
                                                const double              *beta,
                                                double                    *y)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
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
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(n < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(ell_width < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Sanity check
    if((m == 0 || n == 0) && ell_width != 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || ell_width == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(ell_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(ell_col_ind == nullptr)
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

    using namespace aoclsparse;

#if USE_AVX512
    if(context::get_context()->supports<context_isa_t::AVX512F>())
        return aoclsparse_elltmv_template_avx512(
            *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
    else
        return aoclsparse_elltmv_template_avx2(
            *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
#else
    return aoclsparse_elltmv_template_avx2(
        *alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, *beta, y);
#endif
}

extern "C" aoclsparse_status aoclsparse_sellthybmv([[maybe_unused]] aoclsparse_operation trans,
                                                   const float                          *alpha,
                                                   aoclsparse_int                        m,
                                                   aoclsparse_int                        n,
                                                   aoclsparse_int                        nnz,
                                                   const float                          *ell_val,
                                                   const aoclsparse_int      *ell_col_ind,
                                                   aoclsparse_int             ell_width,
                                                   aoclsparse_int             ell_m,
                                                   const float               *csr_val,
                                                   const aoclsparse_int      *csr_row_ind,
                                                   const aoclsparse_int      *csr_col_ind,
                                                   aoclsparse_int            *row_idx_map,
                                                   aoclsparse_int            *csr_row_idx_map,
                                                   const aoclsparse_mat_descr descr,
                                                   const float               *x,
                                                   const float               *beta,
                                                   float                     *y)
{
    return aoclsparse_ellthybmv_template(*alpha,
                                         m,
                                         n,
                                         nnz,
                                         ell_val,
                                         ell_col_ind,
                                         ell_width,
                                         ell_m,
                                         csr_val,
                                         csr_row_ind,
                                         csr_col_ind,
                                         row_idx_map,
                                         csr_row_idx_map,
                                         descr,
                                         x,
                                         *beta,
                                         y);
}

extern "C" aoclsparse_status aoclsparse_dellthybmv([[maybe_unused]] aoclsparse_operation trans,
                                                   const double                         *alpha,
                                                   aoclsparse_int                        m,
                                                   aoclsparse_int                        n,
                                                   aoclsparse_int                        nnz,
                                                   const double                         *ell_val,
                                                   const aoclsparse_int      *ell_col_ind,
                                                   aoclsparse_int             ell_width,
                                                   aoclsparse_int             ell_m,
                                                   const double              *csr_val,
                                                   const aoclsparse_int      *csr_row_ind,
                                                   const aoclsparse_int      *csr_col_ind,
                                                   aoclsparse_int            *row_idx_map,
                                                   aoclsparse_int            *csr_row_idx_map,
                                                   const aoclsparse_mat_descr descr,
                                                   const double              *x,
                                                   const double              *beta,
                                                   double                    *y)
{
    using namespace aoclsparse;
#if USE_AVX512
    if(context::get_context()->supports<context_isa_t::AVX512F>())
        return aoclsparse_ellthybmv_template_avx512(*alpha,
                                                    m,
                                                    n,
                                                    nnz,
                                                    ell_val,
                                                    ell_col_ind,
                                                    ell_width,
                                                    ell_m,
                                                    csr_val,
                                                    csr_row_ind,
                                                    csr_col_ind,
                                                    row_idx_map,
                                                    csr_row_idx_map,
                                                    descr,
                                                    x,
                                                    *beta,
                                                    y);
    else
        return aoclsparse_ellthybmv_template_avx2(*alpha,
                                                  m,
                                                  n,
                                                  nnz,
                                                  ell_val,
                                                  ell_col_ind,
                                                  ell_width,
                                                  ell_m,
                                                  csr_val,
                                                  csr_row_ind,
                                                  csr_col_ind,
                                                  row_idx_map,
                                                  csr_row_idx_map,
                                                  descr,
                                                  x,
                                                  *beta,
                                                  y);
#else
    return aoclsparse_ellthybmv_template_avx2(*alpha,
                                              m,
                                              n,
                                              nnz,
                                              ell_val,
                                              ell_col_ind,
                                              ell_width,
                                              ell_m,
                                              csr_val,
                                              csr_row_ind,
                                              csr_col_ind,
                                              row_idx_map,
                                              csr_row_idx_map,
                                              descr,
                                              x,
                                              *beta,
                                              y);
#endif
}
