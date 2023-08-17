/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "gtest/gtest.h"

#include <cstdlib>
#include <iostream>

#include "alci/cxx/alci.hh"
#include "alci/cxx/cache.hh"
#include "alci/cxx/cpu.hh"

// Taken from AOCL-Utils ISA example
// Run query command and expect not to break

namespace
{
    using namespace std;
    using namespace alci;

    TEST(Utils, IsaCache)
    {
        alci::CacheInfo cachedata{};

        typedef struct Query
        {
            CacheLevel level;
            CacheType  type;
        } query_t;

        vector<query_t> query{{CacheLevel::e_Level_1, CacheType::eData},
                              {CacheLevel::e_Level_1, CacheType::eInstruction},
                              {CacheLevel::e_Level_2, CacheType::eUnified},
                              {CacheLevel::e_Level_3, CacheType::eUnified}};

        for(auto q : query)
        {
            cout << "Cache info:\n-----------";
            cout << "\n  " << q.level << "-" << q.type << "\n";
            cout << "  ------------------\n";

            cout << "  Cache size (KB)    : " << (cachedata.getSize(q.level, q.type) / 1024)
                 << "\n";
            cout << "  No. of ways (Bytes): " << cachedata.getWay(q.level, q.type) << "\n";
            cout << "  Line size (Bytes)  : " << cachedata.getLane(q.level, q.type) << "\n";
        }
    }

    TEST(Utils, IsaCPU)
    {
        alci::Cpu core{0};

        cout << "----- Platform details -----\n";
        cout << "Core info:\n----------\n";

        cout << "Is AMD           : " << (core.isAmd() ? "YES" : "NO") << "\n";
        cout << "Vendor           : " << core.getVendor() << "\n";
        cout << "Architecture     : " << core.getUarch() << "\n";

        cout << "isUarchZen       : " << (core.isUarch(Uarch::eZen) ? "YES" : "NO") << "\n";
        cout << "isUarchZen2      : " << (core.isUarch(Uarch::eZen2) ? "YES" : "NO") << "\n";
        cout << "isUarchZen3      : " << (core.isUarch(Uarch::eZen3) ? "YES" : "NO") << "\n";
        cout << "isUarchZen4      : " << (core.isUarch(Uarch::eZen4) ? "YES" : "NO") << "\n";

        cout << "Features supported:\n-------------------\n";
        cout << "AVX support   : " << (core.isAvailable(ALC_E_FLAG_AVX) ? "YES" : "NO") << "\n";
        cout << "AVX2 support  : " << (core.isAvailable(ALC_E_FLAG_AVX2) ? "YES" : "NO") << "\n";
        cout << "AVX512 support: " << (core.isAvailable(ALC_E_FLAG_AVX512F) ? "YES" : "NO") << "\n";

        if(core.isAvailable(ALC_E_FLAG_AVX512F))
        {
            cout << "  AVX512DQ         : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512DQ) ? "YES" : "NO") << "\n";
            cout << "  AVX512PF         : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512PF) ? "YES" : "NO") << "\n";
            cout << "  AVX512ER         : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512ER) ? "YES" : "NO") << "\n";
            cout << "  AVX512CD         : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512CD) ? "YES" : "NO") << "\n";
            cout << "  AVX512BW         : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512BW) ? "YES" : "NO") << "\n";
            cout << "  AVX512VL         : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512VL) ? "YES" : "NO") << "\n";
            cout << "  AVX512_IFMA      : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512_IFMA) ? "YES" : "NO") << "\n";
            cout << "  AVX512_VNNI      : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512_VNNI) ? "YES" : "NO") << "\n";
            cout << "  AVX512_BITALG    : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512_BITALG) ? "YES" : "NO") << "\n";
            cout << "  AVX512_VBMI      : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512_VBMI) ? "YES" : "NO") << "\n";
            cout << "  AVX512_VBMI2     : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512_VBMI2) ? "YES" : "NO") << "\n";
            cout << "  AVX512_VPOPCNTDQ : "
                 << (core.isAvailable(ALC_E_FLAG_AVX512_VPOPCNTDQ) ? "YES" : "NO") << "\n";
        }
    }
}
