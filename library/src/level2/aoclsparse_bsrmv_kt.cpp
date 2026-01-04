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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "aoclsparse_descr.h"
#include "aoclsparse_bsrmv_bldr.hpp"
#include "aoclsparse_l2_kt.hpp"

using namespace kernel_templates;

template <kernel_templates::bsz SZ, typename SUF, size_t KERNEL_DIM, bool alpha_s, bool beta_s>
aoclsparse_status aoclsparse::bsrmv_nxn_v(SUF                             alpha,
                                          aoclsparse_int                  mb,
                                          [[maybe_unused]] aoclsparse_int nb,
                                          const SUF *__restrict__ bsr_val,
                                          const aoclsparse_int *__restrict__ bsr_col_ind,
                                          const aoclsparse_int *__restrict__ bsr_row_ptr,
                                          const aoclsparse_mat_descr descr,
                                          const SUF *__restrict__ x,
                                          SUF beta,
                                          SUF *__restrict__ y)
{
    constexpr auto tsz = tsz_v<SZ, SUF>;

    static_assert(KERNEL_DIM % tsz == 0,
                  "The kernel dimension should be divisible by the vector type size");

    aoclsparse_index_base base = descr->base;

    aoclsparse_int row_begin, row_end, col;

    const SUF *bsr_val_ptr;
    const SUF *x_ptr;

    constexpr aoclsparse_int KER_SQ = KERNEL_DIM * KERNEL_DIM;

    // When alpha_s and beta_s are false, these vectors may go unused.
    [[maybe_unused]] avxvector_t<SZ, SUF> alpha_v, beta_v;

    // BSR block row accumulator
    avxvector_t<SZ, SUF> sum_v[KERNEL_DIM / tsz];

    if constexpr(alpha_s)
    {
        alpha_v = kt_set1_p<SZ, SUF>(alpha);
    }

    if constexpr(beta_s)
    {
        beta_v = kt_set1_p<SZ, SUF>(beta);
    }

    // Loop over the block rows
    for(aoclsparse_int ai = 0; ai < mb; ++ai)
    {
        SUF *y_ptr = y + ai * KERNEL_DIM;

        // BSR row entry and exit point
        row_begin = bsr_row_ptr[ai] - base;
        row_end   = bsr_row_ptr[ai + 1] - base;

        bsrmv_builder::setzero_v<SZ, SUF, KERNEL_DIM>(sum_v);

        // Loop over all BSR blocks in the current row
        for(aoclsparse_int aj = row_begin; aj < row_end; ++aj)
        {
            // Column index into x vector
            col = bsr_col_ind[aj] - base;

            bsr_val_ptr = bsr_val + (KER_SQ * aj);
            x_ptr       = x + (KERNEL_DIM * col);

            // Compute the sum of the 'n' columns within the BSR blocks of the current
            // BSR row
            bsrmv_builder::compute_v<SZ, SUF, KERNEL_DIM>(sum_v, bsr_val_ptr, x_ptr);
        }
        // Perform alpha * A * x
        if constexpr(alpha_s)
        {
            bsrmv_builder::alpha_scal_v<SZ, SUF, KERNEL_DIM>(sum_v, alpha_v);
        }

        // Perform (beta * y) + (alpha * A * x)
        if constexpr(beta_s)
        {
            bsrmv_builder::beta_scal_v<SZ, SUF, KERNEL_DIM>(sum_v, beta_v, y_ptr);
        }

        // BSR block row sum
        bsrmv_builder::store_res_v<SZ, SUF, KERNEL_DIM>(sum_v, y_ptr);
    }

    return aoclsparse_status_success;
}
#define BSRMV_GN_DECL(SZ, SUF, DIM, ALPHA, BETA)                                   \
    template aoclsparse_status aoclsparse::bsrmv_nxn_v<SZ, SUF, DIM, ALPHA, BETA>( \
        SUF                             alpha,                                     \
        aoclsparse_int                  mb,                                        \
        [[maybe_unused]] aoclsparse_int nb,                                        \
        const SUF *__restrict__ bsr_val,                                           \
        const aoclsparse_int *__restrict__ bsr_col_ind,                            \
        const aoclsparse_int *__restrict__ bsr_row_ptr,                            \
        const aoclsparse_mat_descr descr,                                          \
        const SUF *__restrict__ x,                                                 \
        SUF beta,                                                                  \
        SUF *__restrict__ y);

#define BSRMV_GN_DECL_CONST(SZ, SUF, DIM)    \
    BSRMV_GN_DECL(SZ, SUF, DIM, true, true)  \
    BSRMV_GN_DECL(SZ, SUF, DIM, true, false) \
    BSRMV_GN_DECL(SZ, SUF, DIM, false, true) \
    BSRMV_GN_DECL(SZ, SUF, DIM, false, false)

#define BSRMV_GN_DECL_DIM8(SZ, SUF) BSRMV_GN_DECL_CONST(get_bsz(), SUF, 8)
#define BSRMV_GN_DECL_DIM16(SZ, SUF) BSRMV_GN_DECL_CONST(get_bsz(), SUF, 16)

#define BSRMV_GN_DECL_DIM4(SZ, SUF) BSRMV_GN_DECL_CONST(get_bsz(), SUF, 4)

#ifdef KT_AVX2_BUILD
BSRMV_GN_DECL_DIM4(bsz::b256, double)
BSRMV_GN_DECL_DIM8(bsz::b256, double)
BSRMV_GN_DECL_DIM16(bsz::b256, double)
BSRMV_GN_DECL_DIM8(bsz::b256, float)
BSRMV_GN_DECL_DIM16(bsz::b512, float)
#else
BSRMV_GN_DECL_DIM8(bsz::b512, double)
BSRMV_GN_DECL_DIM16(bsz::b512, double)
BSRMV_GN_DECL_DIM16(bsz::b512, float)
#endif