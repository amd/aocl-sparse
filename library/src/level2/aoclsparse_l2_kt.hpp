/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_L2_KT_HPP
#define AOCLSPARSE_L2_KT_HPP
#include "aoclsparse.h"
#include "aoclsparse_kernel_templates.hpp"

namespace aoclsparse
{
    template <kernel_templates::bsz SZ, typename SUF>
    aoclsparse_status csrmv_kt(aoclsparse_index_base base,
                               const SUF             alpha,
                               aoclsparse_int        m,
                               const SUF *__restrict__ aval,
                               const aoclsparse_int *__restrict__ icol,
                               const aoclsparse_int *__restrict__ row,
                               const SUF *__restrict__ x,
                               const SUF beta,
                               SUF *__restrict__ y);

    template <kernel_templates::bsz SZ, typename SUF>
    aoclsparse_status csrmvt_kt(aoclsparse_index_base base,
                                const SUF             alpha,
                                aoclsparse_int        m,
                                aoclsparse_int        n,
                                const SUF *__restrict__ aval,
                                const aoclsparse_int *__restrict__ icol,
                                const aoclsparse_int *__restrict__ row,
                                const SUF *__restrict__ x,
                                const SUF beta,
                                SUF *__restrict__ y);

    template <kernel_templates::bsz SZ, typename SUF, bool HERM>
    aoclsparse_status csrmv_symm_kt(aoclsparse_index_base base,
                                    const SUF             alpha,
                                    aoclsparse_int        m,
                                    aoclsparse_diag_type  diag_type,
                                    aoclsparse_fill_mode  fill_mode,
                                    const SUF *__restrict__ aval,
                                    const aoclsparse_int *__restrict__ icol,
                                    const aoclsparse_int *__restrict__ icrow,
                                    const aoclsparse_int *__restrict__ idiag,
                                    const aoclsparse_int *__restrict__ iurow,
                                    const SUF *__restrict__ x,
                                    const SUF beta,
                                    SUF *__restrict__ y);

    template <kernel_templates::bsz SZ, typename SUF, size_t KERNEL_DIM, bool alpha_s, bool beta_s>
    aoclsparse_status bsrmv_nxn_v(SUF                             alpha,
                                  aoclsparse_int                  mb,
                                  [[maybe_unused]] aoclsparse_int nb,
                                  const SUF *__restrict__ bsr_val,
                                  const aoclsparse_int *__restrict__ bsr_col_ind,
                                  const aoclsparse_int *__restrict__ bsr_row_ptr,
                                  const aoclsparse_mat_descr descr,
                                  const SUF *__restrict__ x,
                                  SUF beta,
                                  SUF *__restrict__ y);
}

template <kernel_templates::bsz SZ,
          typename SUF,
          kernel_templates::kt_avxext EXT,
          bool                        CONJ = false>
aoclsparse_status kt_trsv_l(const SUF             alpha,
                            aoclsparse_int        m,
                            aoclsparse_index_base base,
                            const SUF *__restrict__ a,
                            const aoclsparse_int *__restrict__ icol,
                            const aoclsparse_int *__restrict__ ilrow,
                            const aoclsparse_int *__restrict__ idiag,
                            const SUF *__restrict__ b,
                            aoclsparse_int incb,
                            SUF *__restrict__ x,
                            aoclsparse_int incx,
                            const bool     unit);

template <kernel_templates::bsz SZ,
          typename SUF,
          kernel_templates::kt_avxext EXT,
          bool                        CONJ = false>
aoclsparse_status kt_trsv_lt(const SUF             alpha,
                             aoclsparse_int        m,
                             aoclsparse_index_base base,
                             const SUF *__restrict__ a,
                             const aoclsparse_int *__restrict__ icol,
                             const aoclsparse_int *__restrict__ ilrow,
                             const aoclsparse_int *__restrict__ idiag,
                             const SUF *__restrict__ b,
                             aoclsparse_int incb,
                             SUF *__restrict__ x,
                             aoclsparse_int incx,
                             const bool     unit);

template <kernel_templates::bsz SZ,
          typename SUF,
          kernel_templates::kt_avxext EXT,
          bool                        CONJ = false>
aoclsparse_status kt_trsv_u(const SUF             alpha,
                            aoclsparse_int        m,
                            aoclsparse_index_base base,
                            const SUF *__restrict__ a,
                            const aoclsparse_int *__restrict__ icol,
                            const aoclsparse_int *__restrict__ ilrow,
                            const aoclsparse_int *__restrict__ iurow,
                            const SUF *__restrict__ b,
                            aoclsparse_int incb,
                            SUF *__restrict__ x,
                            aoclsparse_int incx,
                            const bool     unit);

template <kernel_templates::bsz SZ,
          typename SUF,
          kernel_templates::kt_avxext EXT,
          bool                        CONJ = false>
aoclsparse_status kt_trsv_ut(const SUF             alpha,
                             aoclsparse_int        m,
                             aoclsparse_index_base base,
                             const SUF *__restrict__ a,
                             const aoclsparse_int *__restrict__ icol,
                             const aoclsparse_int *__restrict__ ilrow,
                             const aoclsparse_int *__restrict__ iurow,
                             const SUF *__restrict__ b,
                             aoclsparse_int incb,
                             SUF *__restrict__ x,
                             aoclsparse_int incx,
                             const bool     unit);

#endif