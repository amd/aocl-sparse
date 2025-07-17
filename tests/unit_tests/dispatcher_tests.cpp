/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"

#include <complex>
#include <map>
#include <stdlib.h>
#include <string>
#include <thread>
#include <typeinfo>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dispatch_Test
{
    // -------------------------------------------------------------------------
    // UNIT TESTS
    // -------------------------------------------------------------------------

    /*
        Tokenizes the following strings to unique integers:
        1. ZEN
        2. ZEN1
        3. ZEN2
        4. ZEN3
        5. ZEN4
        ...
        n-1. ZEN123
        n. ZENS
    */
    constexpr uint32_t tokenize_isa(const char *isa, size_t size = 0) noexcept
    {
        uint32_t u_id = 1086237398;

        for(const char *ch = isa; ch < isa + size; ++ch)
            u_id = ((u_id << 20) + u_id) + (unsigned char)*ch;

        return u_id;
    }

    TEST(Oracle, DispatchL1)
    {
        debug_info  info;
        std::string dispatcher = "dispatch_l1";

        [[maybe_unused]] aoclsparse_status st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated, info.arch);

        std::string arch{info.arch};

        uint32_t arch_id = tokenize_isa(arch.c_str(), arch.length());

        EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);

        switch(arch_id)
        {
        case tokenize_isa("ZEN5", 4):
        case tokenize_isa("ZEN4", 4):
            // Auto kernel
            if(can_exec_avx512_tests())
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          3 + 10);
            else
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          2 + 10);

            // Request specific kernel 3. On AVX512 build on Zen 4 machines, this kernel can be launched.
            if(can_exec_avx512_tests())
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 3),
                          3 + 10);
            else // On AVX2 build on Zen 4 machines, this kernel is invalid.
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 3),
                          aoclsparse_status_invalid_kid);

            // Force generic path
            EXPECT_EQ(aoclsparse_enable_instructions("GENERIC"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 0 + 10);

            // make sure kid is taken into account
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 1), 1 + 10);

            // Force AVX2 path on Zen4 is kid=2 not kid=1
            EXPECT_EQ(aoclsparse_enable_instructions("avx2"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 2 + 10);

            // Force AVX512 path on Zen4 is kid=3
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);

            // Request AVX512 path. On AVX512 build on Zen 4 machines, the kernel can be launched.
            if(can_exec_avx512_tests())
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          3 + 10);
            else // On AVX2 build on Zen 4 machines, the  AVX2 kernel is dispatched.
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          2 + 10);

            break;
        case tokenize_isa("ZEN", 3):
        case tokenize_isa("ZEN2", 4):
        case tokenize_isa("ZEN3", 4):
        case tokenize_isa("ZEN123", 6):
            // Auto kernel
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 1 + 10);

            // Request specific kernel 3. Not supported by the architecture
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 3),
                      aoclsparse_status_invalid_kid);

            // Force generic path
            EXPECT_EQ(aoclsparse_enable_instructions("Generic"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 0 + 10);

            // make sure kid is taken into account
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 1), 1 + 10);

            // Force AVX2 path on Zen3 is kid=1
            EXPECT_EQ(aoclsparse_enable_instructions("avx2"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 1 + 10);

            // Invalid AVX512 path on Zen3... revert to auto best
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 1 + 10);

            // kid = auto
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 1 + 10);
            break;

        case tokenize_isa("GENERIC", 7):
            // Auto kernel
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 0 + 10);

            // Request specific kernel 3
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 3), 0 + 10);

            // Force generic path
            EXPECT_EQ(aoclsparse_enable_instructions("Generic"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 0 + 10);

            // make sure kid is not taken into account
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 1), 0 + 10);

            // Force AVX512 path on Zen4 is kid=3
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 0 + 10);
            EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);
            break;
        }
    }

    TEST(Oracle, DispatchUniq)
    {
        int         kid;
        std::string dispatcher = "dispatch_only_ref";

        kid = aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1);
        EXPECT_EQ(kid, 0 + 0);
        kid = aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_zmat, -1);
        EXPECT_EQ(kid, 0 + 0);
    }

    TEST(Oracle, DispatchMulti)
    {
        int         kid;
        std::string dispatcher = "dispatch_multi";

        kid = aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 2);
        EXPECT_EQ(kid, 2 + 0);

        kid = aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_dmat, -1);
        EXPECT_EQ(kid, 7 + 0);
    }

    /* No exact kernel in table */
    TEST(Oracle, DispatchNoExact)
    {
        debug_info                         info;
        [[maybe_unused]] aoclsparse_status st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated, info.arch);

        std::string dispatcher = "dispatch_noexact";

        std::string arch{info.arch};

        uint32_t arch_id = tokenize_isa(arch.c_str(), arch.length());
        EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);
        switch(arch_id)
        {
        case tokenize_isa("ZEN5", 4):
        case tokenize_isa("ZEN4", 4):
            // Auto
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 2);
            // AVX512 noise
            EXPECT_EQ(aoclsparse_enable_instructions("avx2"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 2);
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 2);
            break;
        case tokenize_isa("ZEN", 3):
        case tokenize_isa("ZEN2", 4):
        case tokenize_isa("ZEN3", 4):
        case tokenize_isa("ZEN123", 6):
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 1);
            // noise
            EXPECT_EQ(aoclsparse_enable_instructions("Generic"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 1);
            break;
        case tokenize_isa("GENERIC", 6):
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1), 0);
            break;
        }
    }

    /* Test scores for multi-match */
    TEST(Oracle, DispatchBig)
    {
        // This test differentiates Zen 1/1+/2 from 3
        // simple id is not usable here

        debug_info info;

        [[maybe_unused]] aoclsparse_status st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated, info.arch);

        std::string dispatcher = "dispatch";

        std::string arch{info.arch};

        uint32_t arch_id = tokenize_isa(arch.c_str(), arch.length());

        EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);
        switch(arch_id)
        {
        case tokenize_isa("ZEN5", 4):
        case tokenize_isa("ZEN4", 4):
            // kid auto contend ZEN123 vs ZENS vs ZEN3
            if(can_exec_avx512_tests())
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          6 + 1000); // AVX512 path priority by table
            else
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          4 + 1000);
            // request AVX512 path contend ZEN4 vs ZENS
            if(can_exec_avx512_tests())
            {
                // This cannot be executed in AVX2 builds
                EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          6 + 1000); // AVX512 path priority by table
            }
            break;
        case tokenize_isa("ZEN3", 4):
            // kid auto contend ZEN123 vs ZENS vs ZEN3
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                      3 + 1000);
            break;
        case tokenize_isa("GENERIC", 6):
            // test already covered
            break;
        case tokenize_isa("ZEN", 3):
        case tokenize_isa("ZEN2", 4):
            // Zen 1/1+/2:
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                      1 + 1000);
            break;
        default:
            // For unknown architectures
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                      0 + 1000);
            break;
        }
    }

    // Dispatch two threads, set the hint and check for correct kernel
    TEST(Oracle, DispatchThreads)
    {
        using namespace std::string_literals;
        using namespace aoclsparse;

        auto lt_driver = [](int *ok, int kid, const std::string isa) {
            // set isa hint (on thread_local copy)
            aoclsparse_status status = aoclsparse_enable_instructions(isa.c_str());
            EXPECT_EQ(status, aoclsparse_status_success);

            std::string dispatcher = "dispatch_isa";

            // Check if expected kid is launched
            *ok = kid == aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1);
        };

        int         ok1, ok2;
        std::thread t1(lt_driver, &ok1, 0, "generic"s);
        std::thread t2(lt_driver, &ok2, 1, "avx2"s);
        t1.join();
        t2.join();
        EXPECT_TRUE(ok1);
        EXPECT_TRUE(ok2);
    }

    TEST(Oracle, InvalidKID)
    {
        std::string dispatcher = "dispatch_l1";
        // Non-existent kernel kid
        EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, 2024),
                  aoclsparse_status_invalid_kid);
    }

#ifdef _OPENMP
    // Dispatch some threads, set the hint and check for correct kernel
    TEST(Oracle, DispatchOMP)
    {
        using namespace std::string_literals;
        using namespace aoclsparse;

        auto lt_driver = [](int *ok, int kid, const std::string &isa) {
            // set isa hint (on thread copy)
            aoclsparse_status status = aoclsparse_enable_instructions(isa.c_str());
            EXPECT_EQ(status, aoclsparse_status_success);

            std::string dispatcher = "dispatch_isa";

            // Check if expected kid is launched
            *ok = kid == aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1);
        };

        const std::string hints[2] = {"generic"s, "avx2"s};
        int               tid[2];
        int               ok{0};
        int               tcnt{1};

        std::string p_hints;
        int         p_ok;

#pragma omp parallel shared(hints, ok, tid, tcnt, lt_driver) private(p_hints, \
                                                                         p_ok) // num_threads(2)
        {
#pragma omp master
            tcnt = omp_get_num_threads();

#pragma omp for
            for(int k = 0; k < 2; ++k)
            {
                tid[k] = omp_get_thread_num();

                p_hints = hints[k];
                lt_driver(&p_ok, k, p_hints);

#pragma omp atomic
                ok += p_ok;
            }
        }

        // Make sure there was one thread per each lt_driver call...
        if(tcnt > 1)
        {
            EXPECT_NE(tid[0], tid[1]) << "Actual number of threads = " << tcnt;
        }

        EXPECT_EQ(ok, 2);
    }
#endif

    /* Test scores for 256b vectors on AVX512VL machines */
    TEST(Oracle, DispatchAVX512VL)
    {
        debug_info info;

        [[maybe_unused]] aoclsparse_status st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated, info.arch);

        std::string dispatcher = "dispatch_AVX512VL";
        std::string arch{info.arch};
        uint32_t    arch_id = tokenize_isa(arch.c_str(), arch.length());

        EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);
        switch(arch_id)
        {
        case tokenize_isa("ZEN5", 4):
        case tokenize_isa("ZEN4", 4):
        case tokenize_isa("UNKNOWN", 7):
            if(can_exec_avx512_tests())
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_dmat, -1),
                          1 + 4000); // AVX512VL 256bit path priority by table
            else
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          0 + 4000);
            EXPECT_EQ(aoclsparse_enable_instructions("AVX2"), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_cmat, -1),
                      0 + 4000);
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            if(can_exec_avx512_tests())
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_zmat, -1),
                          1 + 4000); // AVX512VL 256bit path priority by table
            else
                EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_smat, -1),
                          0 + 4000);
            break;
        case tokenize_isa("ZEN3", 4):
        case tokenize_isa("ZEN2", 4):
        case tokenize_isa("ZEN1", 4):
            EXPECT_EQ(aoclsparse_debug_dispatcher(dispatcher.c_str(), aoclsparse_dmat, -1),
                      0 + 4000);
            break;
        }
    }

    TEST(Oracle, Invalid_kid_Range)
    {
        // This test checks the range dispatcher with invalid kid values.
        // The range dispatcher is expected to return a specific error code
        // when the kid is out of the valid range or when begin and end are not valid.

        std::string dispatcher = "dispatch_range";

        // Valid range kid is only 2, 3, 4 even though the table has 7 entries.
        // Kernel ID in range
        EXPECT_EQ(aoclsparse::test::dispatcher<float>(dispatcher, 1, 0, 2), 1001);

        EXPECT_EQ(aoclsparse::test::dispatcher<float>(dispatcher, 0, 0, 1), 1000);

        // Kernel ID in range with non-default lower-bound
        EXPECT_EQ(aoclsparse::test::dispatcher<float>(dispatcher, 1, 2, 5), 1003);

        // Edge case
        EXPECT_EQ(aoclsparse::test::dispatcher<double>(dispatcher, 2, 0, 2),
                  aoclsparse_status_invalid_kid);

        // No search range
        EXPECT_EQ(aoclsparse::test::dispatcher<std::complex<double>>(dispatcher, 0, 0, 0),
                  aoclsparse_status_invalid_kid);

        // Kernel ID out of range
        EXPECT_EQ(aoclsparse::test::dispatcher<std::complex<double>>(dispatcher, 3, 0, 2),
                  aoclsparse_status_invalid_kid);

        // Begin and end are not valid values
        EXPECT_EQ(aoclsparse::test::dispatcher<std::complex<float>>(dispatcher, 1, 2, 0),
                  aoclsparse_status_invalid_kid);

        // Begin is not in range
        EXPECT_EQ(aoclsparse::test::dispatcher<std::complex<float>>(dispatcher, 1, -1, 2),
                  aoclsparse_status_invalid_kid);

        // End is not in range
        EXPECT_EQ(aoclsparse::test::dispatcher<std::complex<float>>(dispatcher, 1, 0, 8),
                  aoclsparse_status_invalid_kid);
    }
}
