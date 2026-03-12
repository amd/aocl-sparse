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
#include "gtest/gtest.h"

/* DOID integer constants matching the current [group:3][op:2] bit encoding.
   These mirror the aoclsparse::doid enum which is not exposed in the public API. */
// clang-format off
constexpr aoclsparse_int GN  = 0;
constexpr aoclsparse_int GC  = 1;
constexpr aoclsparse_int GT  = 2;
constexpr aoclsparse_int GH  = 3;
constexpr aoclsparse_int SL  = 4;
constexpr aoclsparse_int SLC = 5;
constexpr aoclsparse_int SU  = 6;
constexpr aoclsparse_int SUC = 7;
constexpr aoclsparse_int HL  = 8;
constexpr aoclsparse_int HLC = 9;
constexpr aoclsparse_int HUC = 10;
constexpr aoclsparse_int HU  = 11;
constexpr aoclsparse_int TLN = 12;
constexpr aoclsparse_int TLC = 13;
constexpr aoclsparse_int TLT = 14;
constexpr aoclsparse_int TLH = 15;
constexpr aoclsparse_int TUN = 16;
constexpr aoclsparse_int TUC = 17;
constexpr aoclsparse_int TUT = 18;
constexpr aoclsparse_int TUH = 19;
constexpr aoclsparse_int LEN = 20;
// clang-format on

struct doid_score_entry
{
    aoclsparse_int req;
    aoclsparse_int mat;
    aoclsparse_int expected_score;
    aoclsparse_int expected_eff;
};

/* Reference table: every valid (req, mat) pair with expected score and effective DOID.
   The effective DOID is what get_effective_doid() returns -- the operation or target
   DOID the kernel needs:
     - Same-group (all families): pure operation = mat_op ^ req_op (gn/gc/gt/gh).
       An exact match yields gn (no operation needed).
     - Cross-group (general mat → non-general req): full target DOID (e.g. sl, tut, huc).
   Scores are unchanged from the original hash map. */
// clang-format off
static const doid_score_entry reference_table[] = {
    // ===== Exact matches (20 entries) =====
    // Exact match → no operation needed → eff = gn for ALL groups
    {GN,  GN,  100, GN},
    {GC,  GC,  100, GN},
    {GT,  GT,  100, GN},
    {GH,  GH,  100, GN},
    {SL,  SL,  100, GN},
    {SLC, SLC, 100, GN},
    {SU,  SU,  100, GN},
    {SUC, SUC, 100, GN},
    {HL,  HL,  100, GN},
    {HLC, HLC, 100, GN},
    {HU,  HU,  100, GN},
    {HUC, HUC, 100, GN},
    {TLN, TLN, 100, GN},
    {TLC, TLC, 100, GN},
    {TLT, TLT, 100, GN},
    {TLH, TLH, 100, GN},
    {TUN, TUN, 100, GN},
    {TUC, TUC, 100, GN},
    {TUT, TUT, 100, GN},
    {TUH, TUH, 100, GN},

    // ===== General intra-group (12 non-exact entries) =====
    {GN, GT, 70, GT},
    {GN, GC, 80, GC},
    {GN, GH, 60, GH},
    {GT, GN, 70, GT},
    {GT, GH, 80, GC},
    {GT, GC, 60, GH},
    {GH, GC, 70, GT},
    {GH, GN, 60, GH},
    {GH, GT, 80, GC},
    {GC, GH, 70, GT},
    {GC, GT, 60, GH},
    {GC, GN, 80, GC},

    // ===== Symmetric Lower intra-group + cross-group (10 entries) =====
    // Intra-group: eff = mat_op ^ req_op (always gn or gc for symmetric)
    {SL,  SLC, 80, GC },
    // Cross-group: score = 40 (no conj needed) or 35 (conj needed)
    {SL,  GN,  40, SL },
    {SL,  GT,  40, SU },
    {SL,  GC,  35, SLC},
    {SL,  GH,  35, SUC},
    {SLC, SL,  80, GC },
    {SLC, GC,  40, SL },
    {SLC, GN,  35, SLC},
    {SLC, GT,  35, SUC},
    {SLC, GH,  40, SU },

    // ===== Symmetric Upper intra-group + cross-group (10 entries) =====
    {SU,  SUC, 80, GC },
    {SU,  GN,  40, SU },
    {SU,  GT,  40, SL },
    {SU,  GC,  35, SUC},
    {SU,  GH,  35, SLC},
    {SUC, SU,  80, GC },
    {SUC, GC,  40, SU },
    {SUC, GN,  35, SUC},
    {SUC, GT,  35, SLC},
    {SUC, GH,  40, SL },

    // ===== Hermitian Lower intra-group + cross-group (10 entries) =====
    {HL,  HLC, 80, GC },
    {HL,  GN,  40, HL },
    {HL,  GT,  35, HUC},
    {HL,  GC,  35, HLC},
    {HL,  GH,  40, HU },
    {HLC, HL,  80, GC },
    {HLC, GC,  40, HL },
    {HLC, GN,  35, HLC},
    {HLC, GT,  40, HU },
    {HLC, GH,  35, HUC},

    // ===== Hermitian Upper intra-group + cross-group (10 entries) =====
    {HU,  HUC, 80, GC },
    {HU,  GN,  40, HU },
    {HU,  GT,  35, HLC},
    {HU,  GC,  35, HUC},
    {HU,  GH,  40, HL },
    {HUC, HU,  80, GC },
    {HUC, GC,  40, HU },
    {HUC, GN,  35, HUC},
    {HUC, GT,  40, HL },
    {HUC, GH,  35, HLC},

    // ===== Triangular Lower intra-group + cross-group (28 entries) =====
    // Intra-group: eff = mat_op ^ req_op (gn/gc/gt/gh)
    {TLN, TLT, 70, GT },
    {TLN, TLC, 80, GC },
    {TLN, TLH, 60, GH },
    // Cross-group: eff = full target DOID
    {TLN, GN,  40, TLN},
    {TLN, GT,  32, TUT},
    {TLN, GC,  35, TLC},
    {TLN, GH,  30, TUH},
    {TLT, TLN, 70, GT },
    {TLT, TLH, 80, GC },
    {TLT, TLC, 60, GH },
    {TLT, GT,  40, TUN},
    {TLT, GN,  32, TLT},
    {TLT, GH,  35, TUC},
    {TLT, GC,  30, TLH},
    {TLH, TLC, 70, GT },
    {TLH, TLN, 60, GH },
    {TLH, TLT, 80, GC },
    {TLH, GH,  40, TUN},
    {TLH, GC,  32, TLT},
    {TLH, GN,  30, TLH},
    {TLH, GT,  35, TUC},
    {TLC, TLH, 70, GT },
    {TLC, TLT, 60, GH },
    {TLC, TLN, 80, GC },
    {TLC, GC,  40, TLN},
    {TLC, GH,  32, TUT},
    {TLC, GN,  35, TLC},
    {TLC, GT,  30, TUH},

    // ===== Triangular Upper intra-group + cross-group (28 entries) =====
    {TUN, TUT, 70, GT },
    {TUN, TUC, 80, GC },
    {TUN, TUH, 60, GH },
    {TUN, GN,  40, TUN},
    {TUN, GT,  32, TLT},
    {TUN, GC,  35, TUC},
    {TUN, GH,  30, TLH},
    {TUT, TUN, 70, GT },
    {TUT, TUH, 80, GC },
    {TUT, TUC, 60, GH },
    {TUT, GT,  40, TLN},
    {TUT, GN,  32, TUT},
    {TUT, GH,  35, TLC},
    {TUT, GC,  30, TUH},
    {TUH, TUC, 70, GT },
    {TUH, TUN, 60, GH },
    {TUH, TUT, 80, GC },
    {TUH, GH,  40, TLN},
    {TUH, GC,  32, TUT},
    {TUH, GN,  30, TUH},
    {TUH, GT,  35, TLC},
    {TUC, TUH, 70, GT },
    {TUC, TUT, 60, GH },
    {TUC, TUN, 80, GC },
    {TUC, GC,  40, TUN},
    {TUC, GH,  32, TLT},
    {TUC, GN,  35, TUC},
    {TUC, GT,  30, TLH},
};
// clang-format on

static const char *doid_name(aoclsparse_int d)
{
    // clang-format off
    static const char *names[] = {
        "gn", "gc", "gt", "gh",
        "sl", "slc", "su", "suc",
        "hl", "hlc", "huc", "hu",
        "tln", "tlc", "tlt", "tlh",
        "tun", "tuc", "tut", "tuh",
    };
    // clang-format on
    if(d >= 0 && d < LEN)
        return names[d];
    return "INVALID";
}

TEST(DoidScoreTest, ReferenceTableMatch)
{
    for(const auto &entry : reference_table)
    {
        aoclsparse_int eff_doid = -1;
        aoclsparse_int score    = aoclsparse_debug_doid_score(entry.mat, entry.req, &eff_doid);

        EXPECT_EQ(score, entry.expected_score)
            << "Score mismatch for req=" << doid_name(entry.req) << " mat=" << doid_name(entry.mat)
            << ": got " << score << ", expected " << entry.expected_score;

        EXPECT_EQ(eff_doid, entry.expected_eff)
            << "Eff_doid mismatch for req=" << doid_name(entry.req)
            << " mat=" << doid_name(entry.mat) << ": got " << doid_name(eff_doid) << "(" << eff_doid
            << ")"
            << ", expected " << doid_name(entry.expected_eff) << "(" << entry.expected_eff << ")";
    }
}

TEST(DoidScoreTest, IncompatiblePairsReturnZero)
{
    for(aoclsparse_int req = 0; req < LEN; ++req)
    {
        for(aoclsparse_int mat = 0; mat < LEN; ++mat)
        {
            bool found = false;
            for(const auto &entry : reference_table)
            {
                if(entry.req == req && entry.mat == mat)
                {
                    found = true;
                    break;
                }
            }

            if(!found)
            {
                aoclsparse_int eff_doid = -1;
                aoclsparse_int score    = aoclsparse_debug_doid_score(mat, req, &eff_doid);

                EXPECT_EQ(score, 0) << "Expected incompatible (score=0) for req=" << doid_name(req)
                                    << " mat=" << doid_name(mat) << ", but got score=" << score;

                EXPECT_EQ(eff_doid, LEN)
                    << "Expected eff_doid=LEN for incompatible req=" << doid_name(req)
                    << " mat=" << doid_name(mat) << ", but got " << doid_name(eff_doid) << "("
                    << eff_doid << ")";
            }
        }
    }
}
