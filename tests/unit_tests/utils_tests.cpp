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
#include "aoclsparse.h"
#include "gtest/gtest.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_utils.hpp"

#include <complex>
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

// AOCLSPARSE_NUMERIC unit-tests
// Tests for object zero<T>

template <typename T>
bool eq(T x, bool ltor = true)
{
    if(ltor)
        return x == aoclsparse_numeric::zero<T>();
    else
        return aoclsparse_numeric::zero<T>() == x;
}

template <typename T>
bool neq(T x, bool ltor = true)
{
    if(ltor)
        return x != aoclsparse_numeric::zero<T>();
    else
        return aoclsparse_numeric::zero<T>() != x;
}

namespace
{
    using namespace aoclsparse_numeric;
    TEST(Utils, Zero)
    {
        // test accessor
        float z_s = zero<float>();
        EXPECT_EQ(z_s, 0.0f);

        double z_d = zero<double>();
        EXPECT_EQ(z_d, 0.0);

        aoclsparse_float_complex z_c = zero<aoclsparse_float_complex>();
        EXPECT_EQ(z_c.imag, 0.0f);
        EXPECT_EQ(z_c.real, 0.0f);

        aoclsparse_double_complex z_z = zero<aoclsparse_double_complex>();
        EXPECT_EQ(z_z.imag, 0.0);
        EXPECT_EQ(z_z.real, 0.0);

        std::complex<float> z_cc = zero<std::complex<float>>();
        EXPECT_EQ(z_cc, 0.0f + 0.0if);

        std::complex<double> z_zz = zero<std::complex<double>>();
        EXPECT_EQ(z_zz, 0.0 + 0.0i);

        // test ::value
        z_s = zero<float>::value;
        EXPECT_EQ(z_s, 0.0f);

        z_s = zero<double>::value;
        EXPECT_EQ(z_d, 0.0);

        z_c = zero<aoclsparse_float_complex>::value;
        EXPECT_EQ(z_c.imag, 0.0f);
        EXPECT_EQ(z_c.real, 0.0f);

        z_z = zero<aoclsparse_double_complex>::value;
        EXPECT_EQ(z_z.imag, 0.0);
        EXPECT_EQ(z_z.real, 0.0);

        z_cc = zero<std::complex<float>>::value;
        EXPECT_EQ(z_cc, 0.0f + 0.0if);

        z_zz = zero<std::complex<double>>::value;
        EXPECT_EQ(z_zz, 0.0 + 0.0i);
    }
    TEST(Utils, ZeroOptr)
    {
        float                     zero_s{0}, one_s{1};
        double                    zero_d{0}, one_d{1};
        aoclsparse_float_complex  zero_c{0, 0}, one_c{1, 0}, one_ic{0, 1}, one_bc{1, 1};
        aoclsparse_double_complex zero_z{0, 0}, one_z{1, 0}, one_iz{0, 1}, one_bz{1, 1};
        std::complex<float>       zero_cc{0}, one_cc{1, 0}, one_icc{0, 1}, one_bcc{1, 1};
        std::complex<double>      zero_zz{0}, one_zz{1, 0}, one_izz{0, 1}, one_bzz{1, 1};
        // x==0, x!=0
        EXPECT_EQ(eq(zero_s), true);
        EXPECT_EQ(eq(zero_d), true);
        EXPECT_EQ(eq(zero_c), true);
        EXPECT_EQ(eq(zero_z), true);
        EXPECT_EQ(eq(zero_cc), true);
        EXPECT_EQ(eq(zero_zz), true);

        EXPECT_EQ(eq(one_s), false);
        EXPECT_EQ(eq(one_d), false);
        EXPECT_EQ(eq(one_c), false);
        EXPECT_EQ(eq(one_z), false);
        EXPECT_EQ(eq(one_cc), false);
        EXPECT_EQ(eq(one_zz), false);
        EXPECT_EQ(eq(one_ic), false);
        EXPECT_EQ(eq(one_iz), false);
        EXPECT_EQ(eq(one_icc), false);
        EXPECT_EQ(eq(one_izz), false);
        EXPECT_EQ(eq(one_bc), false);
        EXPECT_EQ(eq(one_bz), false);
        EXPECT_EQ(eq(one_bcc), false);
        EXPECT_EQ(eq(one_bzz), false);

        EXPECT_EQ(neq(zero_s), false);
        EXPECT_EQ(neq(zero_d), false);
        EXPECT_EQ(neq(zero_c), false);
        EXPECT_EQ(neq(zero_z), false);
        EXPECT_EQ(neq(zero_cc), false);
        EXPECT_EQ(neq(zero_zz), false);

        EXPECT_EQ(neq(one_s), true);
        EXPECT_EQ(neq(one_d), true);
        EXPECT_EQ(neq(one_c), true);
        EXPECT_EQ(neq(one_z), true);
        EXPECT_EQ(neq(one_cc), true);
        EXPECT_EQ(neq(one_zz), true);
        EXPECT_EQ(neq(one_ic), true);
        EXPECT_EQ(neq(one_iz), true);
        EXPECT_EQ(neq(one_icc), true);
        EXPECT_EQ(neq(one_izz), true);
        EXPECT_EQ(neq(one_bc), true);
        EXPECT_EQ(neq(one_bz), true);
        EXPECT_EQ(neq(one_bcc), true);
        EXPECT_EQ(neq(one_bzz), true);

        // 0==x, 0!=x
        EXPECT_EQ((eq(zero_s, false)), true);
        EXPECT_EQ((eq(zero_d, false)), true);
        EXPECT_EQ((eq(zero_c, false)), true);
        EXPECT_EQ((eq(zero_z, false)), true);
        EXPECT_EQ((eq(zero_cc, false)), true);
        EXPECT_EQ((eq(zero_zz, false)), true);

        EXPECT_EQ((eq(one_s, false)), false);
        EXPECT_EQ((eq(one_d, false)), false);
        EXPECT_EQ((eq(one_c, false)), false);
        EXPECT_EQ((eq(one_z, false)), false);
        EXPECT_EQ((eq(one_cc, false)), false);
        EXPECT_EQ((eq(one_zz, false)), false);
        EXPECT_EQ((eq(one_ic, false)), false);
        EXPECT_EQ((eq(one_iz, false)), false);
        EXPECT_EQ((eq(one_icc, false)), false);
        EXPECT_EQ((eq(one_izz, false)), false);
        EXPECT_EQ((eq(one_bc, false)), false);
        EXPECT_EQ((eq(one_bz, false)), false);
        EXPECT_EQ((eq(one_bcc, false)), false);
        EXPECT_EQ((eq(one_bzz, false)), false);

        EXPECT_EQ((neq(zero_s, false)), false);
        EXPECT_EQ((neq(zero_d, false)), false);
        EXPECT_EQ((neq(zero_c, false)), false);
        EXPECT_EQ((neq(zero_z, false)), false);
        EXPECT_EQ((neq(zero_cc, false)), false);
        EXPECT_EQ((neq(zero_zz, false)), false);

        EXPECT_EQ((neq(one_s, false)), true);
        EXPECT_EQ((neq(one_d, false)), true);
        EXPECT_EQ((neq(one_c, false)), true);
        EXPECT_EQ((neq(one_z, false)), true);
        EXPECT_EQ((neq(one_cc, false)), true);
        EXPECT_EQ((neq(one_zz, false)), true);
        EXPECT_EQ((neq(one_ic, false)), true);
        EXPECT_EQ((neq(one_iz, false)), true);
        EXPECT_EQ((neq(one_icc, false)), true);
        EXPECT_EQ((neq(one_izz, false)), true);
        EXPECT_EQ((neq(one_bc, false)), true);
        EXPECT_EQ((neq(one_bz, false)), true);
        EXPECT_EQ((neq(one_bcc, false)), true);
        EXPECT_EQ((neq(one_bzz, false)), true);
    }
}
