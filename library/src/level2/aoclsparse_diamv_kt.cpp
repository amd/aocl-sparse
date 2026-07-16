/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_context.hpp"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace
{
    template <kernel_templates::bsz SZ, typename SUF>
    aoclsparse_status diamv_rowmaj_kt_impl(const SUF      alpha,
                                           aoclsparse_int m,
                                           aoclsparse_int n,
                                           const SUF *__restrict__ dia_val,
                                           const aoclsparse_int *__restrict__ dia_offset,
                                           aoclsparse_int dia_num_diag,
                                           aoclsparse_int row_begin,
                                           aoclsparse_int row_end,
                                           const SUF *__restrict__ x,
                                           const SUF beta,
                                           SUF *__restrict__ y)
    {
        using namespace kernel_templates;
        const size_t tsz = tsz_v<SZ, SUF>;

        // Same beta flags as csrmv_kt (fused y update per row); scale by alpha like csrmv_kt (result *= alpha).
        const bool is_beta_zero = (beta == static_cast<SUF>(0));
        const bool is_beta_one  = (beta == static_cast<SUF>(1));

        const avxvector_t<SZ, SUF> beta_v  = kt_set1_p<SZ, SUF>(beta);
        const avxvector_t<SZ, SUF> alpha_v = kt_set1_p<SZ, SUF>(alpha);

        // DIA storage: entry for diagonal j at row r is at dia_val[j*m + r]. This kernel walks
        // by row and sums over j, so we precompute diag_base[j] = dia_val + j*m once per
        // diagonal and use diag_base[j]+row (or +i for SIMD) in the hot loops instead of
        // forming dia_val + j*m + row every time.
        std::vector<const SUF *> diag_base;
        try
        {
            diag_base.resize(static_cast<size_t>(dia_num_diag));
            for(aoclsparse_int j = 0; j < dia_num_diag; ++j)
            {
                diag_base[static_cast<size_t>(j)]
                    = dia_val + static_cast<size_t>(j) * static_cast<size_t>(m);
            }
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }

        auto scalar_row = [&](aoclsparse_int row) {
            SUF sum = static_cast<SUF>(0);
            for(aoclsparse_int j = 0; j < dia_num_diag; ++j)
            {
                const aoclsparse_int col = row + dia_offset[j];
                if((col >= 0) && (col < n))
                {
                    sum += diag_base[static_cast<size_t>(j)][row] * x[col];
                }
            }
            const SUF result = alpha * sum;
            const SUF yi     = y[row];
            y[row] = is_beta_zero ? result : (is_beta_one ? (yi + result) : ((yi * beta) + result));
        };

        for(aoclsparse_int row = 0; row < (std::min)(row_begin, m); ++row)
            scalar_row(row);

        aoclsparse_int i = row_begin;
        while(i + static_cast<aoclsparse_int>(tsz) <= row_end)
        {
            avxvector_t<SZ, SUF> sum_v = kt_setzero_p<SZ, SUF>();

            for(aoclsparse_int j = 0; j < dia_num_diag; ++j)
            {
                const aoclsparse_int offset = dia_offset[j];
                const SUF *const val = diag_base[static_cast<size_t>(j)] + static_cast<size_t>(i);
                const SUF *const xv  = x + (i + offset);

                const avxvector_t<SZ, SUF> vals_v = kt_loadu_p<SZ, SUF>(val);
                const avxvector_t<SZ, SUF> x_v    = kt_loadu_p<SZ, SUF>(xv);
                sum_v                             = kt_fmadd_p<SZ, SUF>(vals_v, x_v, sum_v);
            }

            const avxvector_t<SZ, SUF> result_v = kt_mul_p<SZ, SUF>(sum_v, alpha_v);
            const avxvector_t<SZ, SUF> y_v      = kt_loadu_p<SZ, SUF>(&y[i]);
            const avxvector_t<SZ, SUF> out_v
                = is_beta_zero ? result_v
                               : (is_beta_one ? kt_add_p<SZ, SUF>(y_v, result_v)
                                              : kt_fmadd_p<SZ, SUF>(y_v, beta_v, result_v));
            kt_storeu_p<SZ, SUF>(&y[i], out_v);

            i += static_cast<aoclsparse_int>(tsz);
        }

        for(; i < m; ++i)
            scalar_row(i);

        return aoclsparse_status_success;
    }

    template <kernel_templates::bsz SZ, typename SUF>
    aoclsparse_status diamv_diagmaj_kt_impl(const SUF      alpha,
                                            aoclsparse_int m,
                                            aoclsparse_int n,
                                            const SUF *__restrict__ dia_val,
                                            const aoclsparse_int *__restrict__ dia_offset,
                                            aoclsparse_int                  dia_num_diag,
                                            [[maybe_unused]] aoclsparse_int row_begin,
                                            [[maybe_unused]] aoclsparse_int row_end,
                                            const SUF *__restrict__ x,
                                            const SUF beta,
                                            SUF *__restrict__ y)
    {
        using namespace kernel_templates;
        const size_t tsz = tsz_v<SZ, SUF>;

        // Perform (beta * y) — same pattern as csrmvt_kt before the transpose-style accumulation.
        if(beta == static_cast<SUF>(0))
        {
            aoclsparse_int             ii     = 0;
            const avxvector_t<SZ, SUF> zero_v = kt_setzero_p<SZ, SUF>();
            while(ii + static_cast<aoclsparse_int>(tsz) <= m)
            {
                kt_storeu_p<SZ, SUF>(&y[ii], zero_v);
                ii += static_cast<aoclsparse_int>(tsz);
            }
            for(; ii < m; ++ii)
                y[ii] = static_cast<SUF>(0);
        }
        else if(beta != static_cast<SUF>(1))
        {
            const avxvector_t<SZ, SUF> beta_v = kt_set1_p<SZ, SUF>(beta);
            aoclsparse_int             ii     = 0;
            while(ii + static_cast<aoclsparse_int>(tsz) <= m)
            {
                avxvector_t<SZ, SUF> yv = kt_loadu_p<SZ, SUF>(&y[ii]);
                kt_storeu_p<SZ, SUF>(&y[ii], kt_mul_p<SZ, SUF>(yv, beta_v));
                ii += static_cast<aoclsparse_int>(tsz);
            }
            for(; ii < m; ++ii)
                y[ii] *= beta;
        }

        // aoclsparse_diamv_t returns early when dia_num_diag == 0; the loop below is then a no-op.
        const avxvector_t<SZ, SUF> alpha_v = kt_set1_p<SZ, SUF>(alpha);

        for(aoclsparse_int d = 0; d < dia_num_diag; ++d)
        {
            const aoclsparse_int offset     = dia_offset[d];
            const aoclsparse_int istart     = (std::max)((aoclsparse_int)0, -offset);
            const aoclsparse_int jstart     = (std::max)((aoclsparse_int)0, offset);
            const aoclsparse_int num_values = (std::min)(m - istart, n - jstart);

            // Skip diagonals that do not intersect the matrix.
            if(num_values <= 0)
                continue;

            const SUF *__restrict__ vp = dia_val + istart + static_cast<size_t>(d) * m;
            const SUF *__restrict__ xp = x + jstart;
            SUF *__restrict__ yp       = y + istart;

            aoclsparse_int j = 0;
            while(j + static_cast<aoclsparse_int>(tsz) <= num_values)
            {
                const avxvector_t<SZ, SUF> vv       = kt_loadu_p<SZ, SUF>(vp + j);
                const avxvector_t<SZ, SUF> vx       = kt_loadu_p<SZ, SUF>(xp + j);
                avxvector_t<SZ, SUF>       vy       = kt_loadu_p<SZ, SUF>(yp + j);
                const avxvector_t<SZ, SUF> alpha_vv = kt_mul_p<SZ, SUF>(vv, alpha_v);
                vy                                  = kt_fmadd_p<SZ, SUF>(alpha_vv, vx, vy);
                kt_storeu_p<SZ, SUF>(yp + j, vy);
                j += static_cast<aoclsparse_int>(tsz);
            }
            for(; j < num_values; ++j)
                yp[j] += alpha * vp[j] * xp[j];
        }

        return aoclsparse_status_success;
    }
} // namespace

// Thin aoclsparse::diamv_*_kt wrappers forward to anonymous-namespace implementations.
// Same layering as csrmv_kt.cpp: explicit instantiations live here; wrappers are the
// stable symbols referenced by kernel tables.

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status aoclsparse::diamv_rowmaj_kt(const SUF      alpha,
                                              aoclsparse_int m,
                                              aoclsparse_int n,
                                              const SUF *__restrict__ dia_val,
                                              const aoclsparse_int *__restrict__ dia_offset,
                                              aoclsparse_int dia_num_diag,
                                              aoclsparse_int row_begin,
                                              aoclsparse_int row_end,
                                              const SUF *__restrict__ x,
                                              const SUF beta,
                                              SUF *__restrict__ y)
{
    return diamv_rowmaj_kt_impl<SZ, SUF>(
        alpha, m, n, dia_val, dia_offset, dia_num_diag, row_begin, row_end, x, beta, y);
}

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status aoclsparse::diamv_diagmaj_kt(const SUF      alpha,
                                               aoclsparse_int m,
                                               aoclsparse_int n,
                                               const SUF *__restrict__ dia_val,
                                               const aoclsparse_int *__restrict__ dia_offset,
                                               aoclsparse_int dia_num_diag,
                                               aoclsparse_int row_begin,
                                               aoclsparse_int row_end,
                                               const SUF *__restrict__ x,
                                               const SUF beta,
                                               SUF *__restrict__ y)
{
    return diamv_diagmaj_kt_impl<SZ, SUF>(
        alpha, m, n, dia_val, dia_offset, dia_num_diag, row_begin, row_end, x, beta, y);
}

#define DIAMV_TEMPLATE_DECLARATION(BSZ, SUF)                          \
    template aoclsparse_status aoclsparse::diamv_rowmaj_kt<BSZ, SUF>( \
        const SUF      alpha,                                         \
        aoclsparse_int m,                                             \
        aoclsparse_int n,                                             \
        const SUF *__restrict__ dia_val,                              \
        const aoclsparse_int *__restrict__ dia_offset,                \
        aoclsparse_int dia_num_diag,                                  \
        aoclsparse_int row_begin,                                     \
        aoclsparse_int row_end,                                       \
        const SUF *__restrict__ x,                                    \
        const SUF beta,                                               \
        SUF *__restrict__ y);

#define DIAMV_DIAGMAJ_TEMPLATE_DECLARATION(BSZ, SUF)                   \
    template aoclsparse_status aoclsparse::diamv_diagmaj_kt<BSZ, SUF>( \
        const SUF      alpha,                                          \
        aoclsparse_int m,                                              \
        aoclsparse_int n,                                              \
        const SUF *__restrict__ dia_val,                               \
        const aoclsparse_int *__restrict__ dia_offset,                 \
        aoclsparse_int dia_num_diag,                                   \
        aoclsparse_int row_begin,                                      \
        aoclsparse_int row_end,                                        \
        const SUF *__restrict__ x,                                     \
        const SUF beta,                                                \
        SUF *__restrict__ y);

DIAMV_TEMPLATE_DECLARATION(kernel_templates::get_bsz(), float);
DIAMV_TEMPLATE_DECLARATION(kernel_templates::get_bsz(), double);
DIAMV_DIAGMAJ_TEMPLATE_DECLARATION(kernel_templates::get_bsz(), float);
DIAMV_DIAGMAJ_TEMPLATE_DECLARATION(kernel_templates::get_bsz(), double);
