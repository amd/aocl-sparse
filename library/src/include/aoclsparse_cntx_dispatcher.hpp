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
#ifndef AOCLSPARSE_DISPATCH_HPP
#define AOCLSPARSE_DISPATCH_HPP

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_utils.hpp"

namespace Dispatch
{
    /***************************************************************************
     * Dispatcher: Kernel Attribute Table (KAT)
     * |--------------------+----------------------+-------------------------|
     * |       Kernel       |        minimal       |                         |
     * |                    | CPU flag requirement | Supported architectures |
     * |--------------------+----------------------+-------------------------|
     * | pointer to <kernel>|  one of cpu_flags::  |  one or more of archs:: |
     * | pointer to <kernel>|  one of cpu_flags::  |  one or more of archs:: |
     * |        ...         |         ...          |           ...           |
     * | pointer to <kernel>|  one of cpu_flags::  |  one or more of archs:: |
     * | endtable<>         |                      |  logic-ored with zero   |
     * |--------------------+----------------------+-------------------------|
     ***************************************************************************/
    template <typename K>
    struct Table
    {
        K                         kernel; // kernel function pointer
        aoclsparse::context_isa_t flag; // minimum CPU flag requirement for kernel
        unsigned int              arch; // suggested architecture(s) for kernel
    };

    template <typename K>
    constexpr Table<K> ORL([[maybe_unused]] Table<K> T)
    {
#ifdef USE_AVX512
        return T;
#else
        return {nullptr, aoclsparse::context_isa_t::UNSET, 0U | aoclsparse::archs::UNKNOWN};
#endif
    }

    inline bool in_range(aoclsparse_int n, aoclsparse_int lower, aoclsparse_int upper)
    {
        // returns true if lower <= n <= upper
        return (lower <= n && n <= upper);
    }

    /***************************************************************************
     * Oracle - choose the best kernel based on a "Kernel Attribute Table"
     *          and user preferences.
     *
     * Input Parameters
     * ----------------
     *
     * tbl - Kernel Attribute Table (KAT)
     * best_kernel - Kernel cached in the dispatcher (must be thread_local or static)
     * kid - kernel ID (default value = -1 = auto)
     * begin - Starting index for table search (default value = 0)
     * end   - Ending index for table search (default value = N)
     *
     * Usage of begin, end parameters
     * ------------------------------
     *
     * Example -
     * begin=3, end=5
     * Upper bound for valid kids is 2 (5 - 3), so there are only 2 valid KIDs which start counting
     * from zero .i.e. kid = 0, kid = 1 are the ONLY valid values.
     *
     *
     * Template Parameters
     * -------------------
     *
     * 1) "typename K" is the type of the kernel's pointer.
     * 2) "aoclsparse_int N" is the number of kernels (rows) in the
     *    "Kernel Attribute Table".
     *
     * Decides the best kernel based on a "Kernel Attribute Table".
     * The param precedence is as follows: (0 highest -> 3 lowest)
     * 0) kid:        User requested KID to use (kid < 0 means AUTO).
     *                If KID is in the valid range 0, ..., (N - 1) and the
     *                requested kernel is supported by the machine then the
     *                kernel is returned. Otherwise nullptr is returned.
     * 1) isa:        Kernel cpu flag requirement (hw must have support for
     *                this ISA)
     * 2) AOCL_INSTRUCTION_ENABLE={UNSET|GENERIC|AVX2|AVX512}: (environment var)
     *                ISA path the user wants to exercise, there may be
     *                multiple candidates and the one with highest score
     *                is chosen.
     *                UNSET indicates that there is no preference.
     * 3) arch:       Suggested hardware platform (provides a scoring scheme)
     *                e.g. On a Zen3 machine, a kernel designed for Zen3 scores
     *                higher that a kernel designed for Zen3 and Zen4, and so forth.
     *
     * Scoring theme: Is only relevant when kid=AUTO (kid<0).
     *                The higher the score for a kernel the more suitable,
     *                score is based on how "tailored" the kernel is to the local
     *                machine/setup/isa preference
     * score = 1      Indicates the kernel CAN run but it is not a good match.
     * score = 2..31  Indicates that kernel CAN run on the machine but it is not
     *                the most recommended (the kernel table indicates that the
     *                local machine is one of at least two supported architectures)
     * score = 32(=L) Indicates that the kernel was designed for the local
     *                machine/setup.
     * score > 100    Indicates that the kernel's ISA matches with the one requested
     *                using the environmental variable AOCL_ENABLE_INSTRUCTIONS.
     *                Note that "score-100" gives information on how "tailored" the
     *                kernel is.
     *
     * Breaking ties: if two or more kernels have the highest score then the
     * last one (top to bottom) in the table is the winner (highest KID)
     *
     * Reducing search space
     * ---------------------
     *
     * When the user wants to use only a portion of the "Kernel Attribute Table"
     * then 'begin' and 'end' indicate the row search span.
     *
     * Example
     *
     * This KAT table holds the kernels for 2 different operations: OP1, and OP2.
     *
     *     {foo_ref<T, OP1>,           context_isa_t::GENERIC, 0U | archs::ALL},
     *     {foo_kt<bsz::b256, T, OP1>, context_isa_t::AVX2,    0U | archs::ALL},
     * ORL({foo_kt<bsz::b512, T, OP1>, context_isa_t::AVX512F, 0U | archs::ALL})
     *     {foo_ref<T, OP2>,           context_isa_t::GENERIC, 0U | archs::ALL},
     *     {foo_kt<bsz::b256, T, OP2>, context_isa_t::AVX2,    0U | archs::ALL},
     * ORL({foo_kt<bsz::b512, T, OP2>, context_isa_t::AVX512F, 0U | archs::ALL})
     *
     * The same oracle can be used to look up the best kernel.
     *
     * For OP1, call the oracle with begin=0 and end=3
     *
     * kernel = Oracle<K>(tbl, kcache_op1, kid, 0, 3)
     *
     * For OP2, call with the rest of the row-span
     *
     * kernel = Oracle<K>(tbl, kcache_op2, kid, 3, 6)
     *
     * Oracle assumptions
     * ------------------
     *
     * 1. The oracle will always return a valid kernel given the table is
     *    coherent, not empty, and if KID is "valid". Valid KID meaning:
     *    KID<0 (auto); or 0<=KID<N with KID identifying a kernel that can be
     *    executed on the current architecture, see next point.
     * 2. The user might request a kernel via KID parameter (KID>=0) that is a
     *    legit value (KID<N) but cannot be executed on the system, in which
     *    case it returns a null pointer.
     * 3. The oracle expects that ALL inputs are valid, i.e., no checks are
     *    done except for KID.
     * 4. Begin and end parameters are used to reduce the search space in the
     *    table, i.e., the oracle will only search for kernels in the range. Other
     *    conventions apply:
     *    a) The range is valid if 0 <= begin < end <= N, where N is the number of
     *       kernels in the table.
     *    b) valid kid must be in the range [0, end - begin).
     *
     * Conventions
     * -----------
     *
     * Some general guides follow,
     *
     * KID = 0 a reference implementation (avx2)
     * KID = 1 refers always to a 256bit wide AVX register kernel
     *         using intrinsics from AVX2 flag
     * KID = 2 refers always to a 256bit wide AVX register kernel
     *         that may use  intrinsics from AVX-512* flags
     * KID = 3 refers always to a 512bit wide AVX register kernel
     *         using intrinsics from AVX-512* flags
     * KID > 3 are "exotic" kernels that may use IP
     *
     *
     * Kernel Attribute Table Templates
     * --------------------------------
     *
     *  Two templates are propoped here that cover most uses-cases.
     *
     * General case
     * ------------
     *
     * 80% of the times a vanilla KAT table will suffice, that is, it only
     * defines a "good" default dispatching that caters to the best possible fit,
     * where there are not "exotic" algorithms.
     *
     * static constexpr Table<K> tbl[]{
     *     {foo_ref<T>,           context_isa_t::GENERIC, 0U | archs::ALL},
     *     {foo_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ALL},
     *     {foo_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ALL}, // alias
     * ORL({foo_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ALL})
     * };
     *
     * This table follows the conventions and tries to accomodate based on
     * CPU flags.
     *
     * ORL<K>() meta-function takes care of enabling the kernel on AVX-512 builds, so
     * all kernels requiring avx512* flags need to be ring-fenced with it.
     *
     * General case with kernels using specific CPU extensions (VANILLA KAT)
     * ---------------------------------------------------------------------
     *
     * 20% of cases use KT "microkernels" that require to specify what AVX extension
     * to exploit, e.g. kt_maskz_set_p<..., EXT>. For these kernel, the KAT will
     * have an extra entry to allow for this. Generally, the extensions used are VL or DQ,
     * for "kt_mask_set_p" the acceleration extension is AVX512VL
     * static constexpr Table<K> tbl[]{
     *     {baz_ref<T>,                                context_isa_t::GENERIC, 0U | archs::ALL},
     *     {baz_kt<bsz::b256, T, kt_avxext::AVX2>,     context_isa_t::AVX2,    0U | archs::ALL},
     * ORL({baz_kt<bsz::b256, T, kt_avxext::AVX512VL>, context_isa_t::AVX512VL,0U | archs::ALL}),
     * ORL({baz_kt<bsz::b512, T, kt_avxext::AVX512F>,  context_isa_t::AVX512F, 0U | archs::ALL})
     * };
     *
     * Exotic cases
     * ------------
     *
     * This would represent KAT where IP is taken into account to fireup
     * tuned variants and "extends" the vanilla KAT
     *
     * static constexpr Table<K> tbl[]{
     * // vanilla KAT
     *     {foo_ref<T>,           context_isa_t::GENERIC, 0U | archs::ALL},
     *     {foo_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ALL},
     * ORL({foo_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ALL},
     * // Exotic kernels
     *     {foo_kt<bsz::b256, T>, context_isa_t::AVX2,    0U | archs::ZEN4},
     * ORL({foo_kt<bsz::b512, T>, context_isa_t::AVX512F, 0U | archs::ZEN5}
     * };
     *
     * This KAT adds two kernels that are tuned to Zen 4 and Zen 5 machines
     * respectively, and as expected, are dispatched whenever the hardware
     * architecture matches.
     *
     **************************************************************************/
    template <typename K, aoclsparse_int N>
    K Oracle(const Table<K> (&tbl)[N],
             K                    best_kernel,
             const aoclsparse_int kid   = -1,
             const aoclsparse_int begin = 0,
             const aoclsparse_int end   = N)
    {
        using namespace aoclsparse;

        // Check if user requested kid is NOT auto
        if(kid >= 0)
        {
            // Invalid kid requested
            if(kid >= (end - begin) || begin >= end || !in_range(begin, 0, N - 1)
               || !in_range(end, 0, N))
            {
                /*
                    Conditions for invalid kid:
                    1. kid >= end - begin: requested KID is out of range .i.e. KID range [begin, end)
                    2. begin >= end: invalid range
                    3. begin is not in the range [0, N): range start is out of bounds
                    4. end is not in the range [0, N]: range end is out of bounds
                */

                return nullptr; // Needs to be handled by the dispatcher
            }

            // The kid requested is valid and supported by the machine
            if(context::get_context()->supports(tbl[begin + kid].flag))
                return tbl[begin + kid].kernel; // Return the user requested kernel
            else
                return nullptr; // Needs to be handled by the dispatcher
        }

        size_t maxscore{0};

        /* Scan Table<> only if
         *
         * 1. best_kernel is unknown, or
         * 2. (best_kernel is known) but the cpu flag override changed
         */
        if(best_kernel == nullptr || tl_isa_hint.is_isa_updated())
        {
            // Get local architecture.
            archs localarch = context::get_context()->get_archs();

            // Get CPU flag override
            context_isa_t isa_flag = tl_isa_hint.get_isa_hint();

            // Scoring
            const unsigned int L = sizeof(unsigned int) * 8U;

            size_t score{0}, multi{0};

            for(aoclsparse_int kcnt = begin; kcnt < end; ++kcnt)
            {
                // If the kernel is supported, score the kernel
                if(context::get_context()->supports(tbl[kcnt].flag))
                {
                    // arch_bits: at most one bit matches: localarch is unique and single bit, tbl[*].arch
                    // can be multiple bits, unless it is set to archs::UNKNOWN which has zero bits set.
                    // This special zero bits case (archs::UNKNOWN) means the kernel in this row is NULL
                    // is taken into account when computing for the first time the score.
                    unsigned int arch_bits = localarch & tbl[kcnt].arch; // at most one bit set (*)
                    // (*) if not archs::UNKNOWN
                    bool arch = arch_bits != 0; // 0 or 1

                    // tbl[*].arch as more than one bit set if not UNKNOWN
                    multi = (~arch_bits & tbl[kcnt].arch) != 0;

                    // score multi based on how many (extra) bits are set and penalise accordingly
                    // 0 or the count of bits set excluding the localarch bit
                    multi = L - multi * (POPCOUNT(tbl[kcnt].arch) - 1);

                    // score = 1 (no match) or 2+ (multi match) or L (max score = exact match): higher better
                    // take care of archs::UNKNOWN here
                    score = (!arch * 1 + arch * multi) * (arch_bits != 0);

                    // check to see if kernel flags match with requested ISA path
                    score += 100
                             * (isa_flag
                                == tbl[kcnt].flag); // score 0 or 1:32 + 100 if ISA path matches

                    if(maxscore <= score) // update with last hit of highscore
                    {
                        maxscore    = score;
                        best_kernel = tbl[kcnt].kernel;
                    }
                }
            }
        }

        return best_kernel;
    }
}
#endif
