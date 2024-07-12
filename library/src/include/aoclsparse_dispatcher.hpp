/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
    // Identifier used to uniquely differentiate the kernel dispachers
    // when actual kernel signature is undistinguishable
    enum class api
    {
        axpyi,
        dotci,
        dotui,
        doti,
        sctr,
        sctrs,
        roti,
        gthr,
        gthrz,
        gthrs,
        reserved0,
        reserved1,
        reserved2,
        reserved3,
        reserved4,
        reserved5
    };

    /***************************************************************************
     * Dispatcher: Kernel Attribute Table
     * This table contains
     * |       Kernel       | CPU flag requirement | Supported architectures |
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
        K                         kernel; // kernel function
        aoclsparse::context_isa_t flag; // minimum CPU flag requirement for kernel
        unsigned int              arch; // suggested architecture(s) for kernel
    };

    /***************************************************************************
     * Oracle
     *
     * Decides the best kernel based on a "Kernel Attribute Table".
     * The precedence is as follows: (0 highest -> 3 lowest)
     * 0) kid:        User requested KID to use (kid < 0 means auto).
     *                If KID is in the valid range 0, ..., (N - 1) and the
     *                requested kernel is supported by the machine then the
     *                kernel is returned. Otherwise nullptr is returned.
     * 1) isa:        Kernel cpu flag requirement (must at least have it
     *                supported)
     * 2) AOCL_INSTRUCTION_ENABLE={UNSET|GENERIC|AVX2|AVX512}:
     *                ISA path the user wants to exercise, there may be
     *                multiple candidates and the one with highest score
     *                is chosen.
     *                UNSET indicates that there is no preference.
     * 3) arch:       Suggested hardware platform (provides a scoring scheme)
     *                a kernel designed for Zen3 scores higher that a kernel
     *                designed for Zen3 and Zen4, and so forth.
     *
     * Scoring theme: Is only relevant when kid=AUTO or kid is invalid.
     *                The higher the score for a kernel the more suitable,
     *                score is based on how "tailored" the kernel is to the local
     *                machine/setup/isa preference
     * score = 1      Indicates the kernel is can run but it is not a good match
     * score = 2..31  Indicates that kernel can run on the machine but it is not
     *                the most recommended (the kernel table indicates that the
     *                local machine is one of at least two supported architectures)
     * score = 32(=L) Indicates that the kernel was designed for the local
     *                machine/setup.
     * score > 100    Indicates that the kernel's ISA matches with the one requested
     *                using the environmental variable AOCL_ENABLE_INSTRUCTIONS.
     *                Note that score - 100 gives information on how "tailored" the
     *                kernel is.
     *
     * Breaking ties: if two or more kernels have the highest score then the
     * 1st one (top to bottom) in the table is the winner (lowest KID)
     *
     * Notes
     *
     * 1. The oracle will always return a valid kernel given the table is not empty
     *    and KID is valid.
     * 2. The user might request a kernel that cannot be run on the system, in which
     *    case it returns a null pointer.
     * 3. The oracle expects that ALL input and components, except for KID, are VALID.
     **************************************************************************/
    template <typename K, api A, aoclsparse_int N>
    K Oracle(const Table<K> (&tbl)[N], aoclsparse_int kid = -1)
    {
        using namespace aoclsparse;

        // Check if user requested kid is NOT auto
        if(kid >= 0)
        {
            // The kid requested is valid and supported by the machine
            if(kid < N && (context::get_context()->supports(tbl[kid].flag)))
                return tbl[kid].kernel; // Return the user requested kernel
            else
                return nullptr; // Needs to be handled by the dispatcher
        }

        // The incumbent kernel
        thread_local K best_kernel{nullptr};

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

            for(aoclsparse_int kcnt = 0; kcnt < N; ++kcnt)
            {
                // If the kernel is supported, score the kernel
                if(context::get_context()->supports(tbl[kcnt].flag))
                {
                    // at most one bit matches: localarch is unique and single bit, tbl.arch can be multiple bits
                    unsigned int arch_bits = localarch & tbl[kcnt].arch; // at most one bit set
                    bool         arch      = arch_bits != 0; // 0 or 1

                    multi = (~arch_bits & tbl[kcnt].arch) != 0; // tbl.arch as more than one bit set

                    // score multi based on how many (extra) bits are set and penalise accordingly
                    // 0 or the count of bits set excluding the localarch bit
                    multi = L - multi * (POPCOUNT(tbl[kcnt].arch) - 1);

                    // score = 1 (no match) or 2+ (multi match) or L (max score = exact match): higher better
                    score = !arch * 1 + arch * multi;

                    // check to see if kernel flags match with requested ISA path
                    score += 100
                             * (isa_flag
                                == tbl[kcnt].flag); // score 0 or 1:32 + 100 if ISA path matches

                    if(maxscore < score)
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