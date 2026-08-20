/* ************************************************************************
 * Copyright (c) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sdia
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
#ifndef AOCLSPARSE_DIAMV_HPP
#define AOCLSPARSE_DIAMV_HPP

// aoclsparse.h and kernel_templates come transitively via aoclsparse_cntx_dispatcher.hpp
// and aoclsparse_l2_kt.hpp.
#include "aoclsparse_descr.h"
#include "aoclsparse_cntx_dispatcher.hpp"
#include "aoclsparse_context.hpp"
#include "aoclsparse_l2_kt.hpp"

#include <algorithm>

template <typename T>
aoclsparse_status diamv_ref(const T        alpha,
                            aoclsparse_int m,
                            aoclsparse_int n,
                            const T *__restrict__ dia_val,
                            const aoclsparse_int *__restrict__ dia_offset,
                            aoclsparse_int                  dia_num_diag,
                            [[maybe_unused]] aoclsparse_int row_begin,
                            [[maybe_unused]] aoclsparse_int row_end,
                            const T *__restrict__ x,
                            const T beta,
                            T *__restrict__ y)
{
    // Perform (beta * y)
    if(beta == static_cast<T>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    for(aoclsparse_int i = 0; i < dia_num_diag; ++i)
    {
        aoclsparse_int offset     = dia_offset[i];
        aoclsparse_int istart     = (std::max)((aoclsparse_int)0, -offset);
        aoclsparse_int jstart     = (std::max)((aoclsparse_int)0, offset);
        aoclsparse_int num_values = (std::min)(m - istart, n - jstart);

        for(aoclsparse_int j = 0; j < num_values; ++j)
        {
            y[istart + j] += alpha * dia_val[istart + i * m + j] * x[j + jstart];
        }
    }

    return aoclsparse_status_success;
}

// -----------------------------------------------------------------
// Kernel selection heuristic (Turin-only tuning):
// Choose row-major KT (diamv_rowmaj_kt) vs diagonal-major KT (diamv_diagmaj_kt)
// via diamv_compute_kernel_heuristic.
// Thresholds and boundary_pct were tuned on AMD Turin; other CPUs may benefit
// from different values.
//
//   use_diagmaj =
//       (boundary_pct > 8)
//    OR (dia_num_diag > 25)
//    OR (m < Msmall)
//    OR (n > Nlarge)
//
//   boundary_pct = 100 * (rows outside interior band) / m
//   interior band: [row_begin, row_end) with
//     row_begin = max(0, -min(dia_offset)),
//     row_end   = min(m, max(0, n - max(dia_offset))).
//
// Constants: Msmall = 10000, Nlarge = 300000.
// -----------------------------------------------------------------
struct diamv_kernel_heuristic
{
    bool           use_diagmaj;
    aoclsparse_int row_begin;
    aoclsparse_int row_end;
};

// Single scan of dia_offset: row bounds for the row-major SIMD band and heuristic choice.
inline diamv_kernel_heuristic diamv_compute_kernel_heuristic(aoclsparse_int        m,
                                                             aoclsparse_int        n,
                                                             const aoclsparse_int *dia_offset,
                                                             aoclsparse_int        dia_num_diag)
{
    constexpr aoclsparse_int Msmall                 = 10000;
    constexpr aoclsparse_int Nlarge                 = 300000;
    constexpr aoclsparse_int boundary_pct_threshold = 8;

    if(dia_num_diag == 0)
    {
        // No diagonals: avoid reading dia_offset[0]; row bounds cover all rows for row-major path.
        return diamv_kernel_heuristic{false, 0, m};
    }

    aoclsparse_int min_offset = dia_offset[0];
    aoclsparse_int max_offset = dia_offset[0];
    for(aoclsparse_int j = 1; j < dia_num_diag; ++j)
    {
        min_offset = (std::min)(min_offset, dia_offset[j]);
        max_offset = (std::max)(max_offset, dia_offset[j]);
    }
    const aoclsparse_int row_begin = (std::max)((aoclsparse_int)0, -min_offset);
    const aoclsparse_int row_end   = (std::min)(m, (std::max)((aoclsparse_int)0, n - max_offset));

    const aoclsparse_int interior_rows
        = (row_end > row_begin) ? (row_end - row_begin) : static_cast<aoclsparse_int>(0);
    const aoclsparse_int boundary_rows = m - interior_rows;
    const aoclsparse_int boundary_pct
        = (m > 0) ? static_cast<aoclsparse_int>((static_cast<long long>(100) * boundary_rows) / m)
                  : static_cast<aoclsparse_int>(0);

    const bool use_diagmaj = (boundary_pct > boundary_pct_threshold) || (dia_num_diag > 25)
                             || (m < Msmall) || (n > Nlarge);
    return diamv_kernel_heuristic{use_diagmaj, row_begin, row_end};
}

template <typename T>
aoclsparse_status aoclsparse_diamv_t(aoclsparse_operation            trans,
                                     const T                        *alpha,
                                     aoclsparse_int                  m,
                                     aoclsparse_int                  n,
                                     [[maybe_unused]] aoclsparse_int nnz,
                                     const T                        *dia_val,
                                     const aoclsparse_int           *dia_offset,
                                     aoclsparse_int                  dia_num_diag,
                                     const aoclsparse_mat_descr      descr,
                                     const T                        *x,
                                     const T                        *beta,
                                     T                              *y,
                                     aoclsparse_int                  mode = -1,
                                     aoclsparse_int                  kid  = -1)
{
    if((alpha == nullptr) || (beta == nullptr) || (dia_val == nullptr) || (dia_offset == nullptr)
       || (x == nullptr) || (y == nullptr) || (descr == nullptr))
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
    if(m < 0 || n < 0 || dia_num_diag < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible (dia_num_diag==0 is handled below so y gets beta scaling like csrmv)
    if(m == 0 || n == 0)
    {
        return aoclsparse_status_success;
    }

    using namespace aoclsparse;
    using namespace Dispatch;
    using namespace kernel_templates;

    using K = decltype(&aoclsparse::diamv_rowmaj_kt<bsz::b256, T>);

    aoclsparse_int oracle_kid = kid;

    const diamv_kernel_heuristic heuristic
        = diamv_compute_kernel_heuristic(m, n, dia_offset, dia_num_diag);
    bool use_diagmaj = heuristic.use_diagmaj;

    // Test-only internal mode selection:
    //   -1 = auto (heuristic-selected family)
    //    0 = reference
    //    1 = row-major kernel family
    //    2 = diagonal-major kernel family
    // kid is an independent Oracle selector within the chosen family.
    switch(mode)
    {
    case 0:
        // Reference mode takes precedence over kid.
        oracle_kid = 0;
        break;
    case 1:
        use_diagmaj = false;
        break;
    case 2:
        use_diagmaj = true;
        break;
    default:
        // Set mode to -1 (auto) by default
        mode = -1;
        // Keep heuristic-selected family and Oracle auto ISA in auto mode.
        break;
    }

    // clang-format off
    // row-major path fuses beta per row; diagonal-major path scales y once then accumulates
    // Kernel Attribute Tables: one per kernel family
    static constexpr Table<K> tbl_rowmaj[]{
        {diamv_ref<T>,                                     context_isa_t::GENERIC, 0U | archs::ALL},
        {aoclsparse::diamv_rowmaj_kt<bsz::b256, T>,       context_isa_t::AVX2,    0U | archs::ALL},
        {aoclsparse::diamv_rowmaj_kt<bsz::b256, T>,       context_isa_t::AVX2,    0U | archs::ALL}, // alias
    ORL<K>({aoclsparse::diamv_rowmaj_kt<bsz::b512, T>,    context_isa_t::AVX512F, 0U | archs::ALL})
    };

    static constexpr Table<K> tbl_diagmaj[]{
        {diamv_ref<T>,                                     context_isa_t::GENERIC, 0U | archs::ALL},
        {aoclsparse::diamv_diagmaj_kt<bsz::b256, T>,      context_isa_t::AVX2,    0U | archs::ALL},
        {aoclsparse::diamv_diagmaj_kt<bsz::b256, T>,      context_isa_t::AVX2,    0U | archs::ALL}, // alias
    ORL<K>({aoclsparse::diamv_diagmaj_kt<bsz::b512, T>,   context_isa_t::AVX512F, 0U | archs::ALL})
    };
    // clang-format on

    // Thread-local kernel caches: one per kernel family
    thread_local K kache_rowmaj  = nullptr;
    thread_local K kache_diagmaj = nullptr;

    K kernel = use_diagmaj ? Oracle<K>(tbl_diagmaj, kache_diagmaj, oracle_kid)
                           : Oracle<K>(tbl_rowmaj, kache_rowmaj, oracle_kid);

    if(!kernel)
        return aoclsparse_status_invalid_kid;

    return kernel(*alpha,
                  m,
                  n,
                  dia_val,
                  dia_offset,
                  dia_num_diag,
                  heuristic.row_begin,
                  heuristic.row_end,
                  x,
                  *beta,
                  y);
}

#endif // AOCLSPARSE_DIAMV_HPP
