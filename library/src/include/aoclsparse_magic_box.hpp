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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_MGBOX_HPP
#define AOCLSPARSE_MGBOX_HPP

#include "aoclsparse.hpp"
#include "aoclsparse_cntx_dispatcher.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_mtx_dispatcher.hpp"

namespace aoclsparse
{
    inline aoclsparse_int get_kid(aoclsparse_optimize_data *opt_d,
                                  aoclsparse::doid          d_id,
                                  aoclsparse_hinted_action  act)
    {
        aoclsparse_int kid = -1;

        while(opt_d != nullptr)
        {
            // The hint and doid should match
            if(opt_d->act == act && d_id == opt_d->doid)
            {
                kid = opt_d->kid;
                break;
            }

            opt_d = opt_d->next;
        }

        return kid;
    }

    /* Query architecture (ISA) support for a (format, eff_doid) combination.
       This is a key part of matrix selection: it checks whether the current hardware
       can execute a kernel for the given matrix format and effective DOID.

       The effective DOID determines which kernel path the dispatcher will take:
         - Same-group exact match produces gn, which dispatches the general normal
           kernel (mv.cpp resets the descriptor to general for exact matches).
         - Same-group non-exact produces gc/gt/gh, which dispatches the corresponding
           general-group kernel (the dispatcher's switch has case gn/gt/gh).
         - Cross-group produces a full target DOID (e.g., sl, tut), dispatching
           the family-specific kernel.

       For CSR, the isDOID* predicates map directly to the dispatcher's case labels.
       For BLKCSR/BSR/TCSR, only specific DOIDs are supported (e.g., gn only for
       general), so isDOIDgenNT is too broad (it includes gt which these formats
       don't support). Those use explicit eff_doid == doid::gn checks instead.

       The ISA lists are ordered from best (fastest) to worst. Dispatch::get_supported()
       returns the best available ISA on this machine, or UNSET if none is available.
       A return value of 0 (UNSET) means no kernel exists for this format+doid on this
       hardware, and the matrix will be excluded from selection in get_best_matrix(). */
    template <typename T>
    aoclsparse_int get_arch_score(aoclsparse_matrix_format_type mtx_t, aoclsparse::doid eff_doid)
    {
        context_isa_t score = context_isa_t::UNSET;

        switch(mtx_t)
        {
        case aoclsparse_csr_mat:
            if(isDOIDgenNT(eff_doid))
                score = Dispatch::get_supported(
                    {context_isa_t::AVX512F, context_isa_t::AVX2, context_isa_t::GENERIC});
            else if(isDOIDgenH(eff_doid) && aoclsparse::is_dt_complex<T>())
                score = Dispatch::get_supported({context_isa_t::GENERIC});
            else if(isDOIDsymm(eff_doid) || isDOIDherm(eff_doid))
                score = Dispatch::get_supported(
                    {context_isa_t::AVX512F, context_isa_t::AVX2, context_isa_t::GENERIC});
            else if(isDOIDtriN(eff_doid))
                score = Dispatch::get_supported({context_isa_t::AVX2, context_isa_t::GENERIC});
            else if(isDOIDtriT(eff_doid))
                score = Dispatch::get_supported({context_isa_t::GENERIC});
            else if(isDOIDtriH(eff_doid) && aoclsparse::is_dt_complex<T>())
                score = Dispatch::get_supported({context_isa_t::GENERIC});
            break;
        case aoclsparse_blkcsr_mat:
            if(eff_doid == doid::gn && std::is_same_v<T, double>)
                score = Dispatch::get_supported({context_isa_t::AVX512F});
            break;
        case aoclsparse_bsr_mat:
            if(eff_doid == doid::gn)
                score = Dispatch::get_supported(
                    {context_isa_t::AVX512F, context_isa_t::AVX2, context_isa_t::GENERIC});
            break;
        case aoclsparse_tcsr_mat:
            if(!aoclsparse::is_dt_complex<T>())
            {
                if(eff_doid == doid::gn && std::is_same_v<T, double>)
                    score = Dispatch::get_supported({context_isa_t::AVX2});
                else if(isDOIDsymm(eff_doid) || isDOIDherm(eff_doid))
                    score = Dispatch::get_supported(
                        {context_isa_t::AVX512F, context_isa_t::AVX2, context_isa_t::GENERIC});
                else if(isDOIDtriN(eff_doid) && std::is_same_v<T, double>)
                    score = Dispatch::get_supported({context_isa_t::AVX2, context_isa_t::GENERIC});
                else if(isDOIDtriN(eff_doid))
                    score = Dispatch::get_supported({context_isa_t::GENERIC});
                else if(isDOIDtriT(eff_doid))
                    score = Dispatch::get_supported({context_isa_t::GENERIC});
            }
            break;
        case aoclsparse_ellt_mat:
        case aoclsparse_ellt_csr_hyb_mat:
            if(eff_doid == doid::gn && std::is_same_v<T, double>)
                score = Dispatch::get_supported({context_isa_t::AVX512F, context_isa_t::AVX2});
            break;
        case aoclsparse_csr_mat_br4:
            if(eff_doid == doid::gn && std::is_same_v<T, double>)
                score = Dispatch::get_supported({context_isa_t::AVX2});
            break;
        case aoclsparse_ell_mat:
        case aoclsparse_ell_csr_hyb_mat:
        case aoclsparse_dia_mat:
        case aoclsparse_coo_mat:
        default:
            score = context_isa_t::UNSET;
        }

        return static_cast<aoclsparse_int>(score);
    }

    /* Score an effective DOID. The effective DOID (from get_effective_doid)
       fully encodes the work the kernel must do, so the score depends on it alone.

       Score mapping by effective DOID group:
         General (gn/gc/gt/gh) — same-group result:
           gn=100 (exact), gc=80 (conjugate), gt=70 (transpose), gh=60 (conj-transpose)
         Symmetric/hermitian — cross-group result:
           40 if no conjugation needed, 35 if conjugation needed.
           Symmetric: conj = op bit 0.  Hermitian: conj = bit0 XOR bit1.
         Triangular — cross-group result:
           {40, 35, 32, 30}[op], where op encodes normal/conj/trans/conj-trans.
         doid::len — incompatible pair:
           0.

       Used by get_best_matrix() alongside get_arch_score() and get_matrix_score()
       to rank candidate matrices. */
    inline aoclsparse_int get_doid_score(aoclsparse::doid eff_doid)
    {
        int eff_v = static_cast<int>(eff_doid);
        if(eff_v >= static_cast<int>(doid::len))
            return 0;

        // eff_grp: family of the effective DOID
        //   0 = general (same-group), 1 = symmetric, 2 = hermitian,
        //   3 = triangular-lower, 4 = triangular-upper
        int eff_grp = eff_v >> 2;

        // eff_op: operation bits
        //   0 = normal, 1 = conjugate, 2 = transpose, 3 = conjugate-transpose
        int eff_op = eff_v & 3;

        // General: same-group operation scores
        if(eff_grp == 0)
        {
            // clang-format off
            static constexpr aoclsparse_int scores[] = {100, 80, 70, 60};
            // clang-format on
            return scores[eff_op];
        }

        // Triangular cross-group: all 4 operation costs apply
        if(eff_grp >= 3)
        {
            // clang-format off
            static constexpr aoclsparse_int scores[] = {40, 35, 32, 30};
            // clang-format on
            return scores[eff_op];
        }

        // Symmetric/hermitian cross-group: cost depends only on conjugation
        // Symmetric (group 1): conjugation = bit 0
        // Hermitian (group 2): conjugation = bit0 XOR bit1 (because hu has op=11
        //   but is non-conjugate: hl(00)→0, hlc(01)→1, huc(10)→1, hu(11)→0)
        int conj = (eff_grp == 1) ? (eff_op & 1) : ((eff_op & 1) ^ (eff_op >> 1));
        return conj ? 35 : 40;
    }

    /* Score bonus based on matrix-specific properties (e.g., optimization state).
       This contributes to the total score in get_best_matrix() alongside
       the DOID conversion score and the architecture score. Currently only
       CSR matrices with completed optimization receive a bonus and
       non-generic formats (blkcsr, ...) as they were created in optimize
       phase so should get preference over CSR or CSR optimized. */
    template <typename T>
    aoclsparse_int get_matrix_score(const base_mtx *mat)
    {
        aoclsparse_int score = 0;
        if(mat->mat_type == aoclsparse_csr_mat)
        {
            auto csr_mtx = dynamic_cast<const csr *>(mat);
            if(csr_mtx && csr_mtx->is_optimized)
            {
                score += 10;
            }
        }
        else if(mat->mat_type != aoclsparse_coo_mat)
        {
            // CSR and COO are the baseline formats, so we give a bonus
            // to all other formats to prefer them when available.
            score += 11;
        }

        return score;
    }

    /* Select the best matrix from A->mats for the requested DOID.
       Each candidate matrix is scored on three independent criteria that are
       summed to produce a total score:
         1. DOID conversion score (get_doid_score): how cheap is it to convert from
            the matrix's stored DOID to the requested DOID. A score of 0 means
            incompatible, and the matrix is skipped.
         2. Matrix property score (get_matrix_score): bonuses for desirable matrix
            properties such as completed optimization.
         3. Architecture score (get_arch_score): checks whether a kernel exists for
            this matrix format and effective DOID on the
            current hardware. A score of 0 means no kernel is available, and the
            matrix is skipped. The effective DOID is derived by get_effective_doid().
       The matrix with the highest total score wins.
       Returns the index of the best matrix in A->mats, or -1 if none is compatible.
       Outputs mtx_t = format type of the best matrix (e.g. aoclsparse_csr_mat). */
    template <typename T>
    aoclsparse_int get_best_matrix(aoclsparse_matrix              A,
                                   aoclsparse::doid               d_id,
                                   aoclsparse_matrix_format_type &mtx_t)
    {
        aoclsparse_int max_score = 0;
        aoclsparse_int best_mtx  = -1;

        for(size_t itr = 0; itr < A->mats.size(); ++itr)
        {
            // Step 1: Get effective DOID and derive compatibility score
            aoclsparse::doid eff_doid   = get_effective_doid(A->mats[itr]->doid, d_id);
            aoclsparse_int   doid_score = get_doid_score(eff_doid);
            if(doid_score == 0)
                continue;

            aoclsparse_int score = doid_score;

            // Step 2: Add matrix property bonuses
            score += get_matrix_score<T>(A->mats[itr]);

            // Step 3: Check architecture support for this format + effective DOID
            int arch_score = get_arch_score<T>(A->mats[itr]->mat_type, eff_doid);
            if(arch_score == 0)
                continue;

            score += arch_score;

            // To-do: Matrix format based multiplier/modification of score to be added here

            if(score > max_score)
            {
                max_score = score;
                best_mtx  = itr;
                mtx_t     = A->mats[itr]->mat_type;
            }
        }

        return best_mtx;
    }

}
#endif
