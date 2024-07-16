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
// Test disabled temporarily
// Re-enable with appropriate interfaces
#if 0
#include "aoclsparse.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"
#include "aoclsparse_dispatcher.hpp"

#include <complex>
#include <map>
#include <stdlib.h>
#include <string>
#include <thread>
#include <typeinfo>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dispatchTest
{
    enum mock
    {
        CASE_1,
        CASE_2,
        CASE_3
    };

    // -------------------------------------------------------------------------
    // KERNELS
    // -------------------------------------------------------------------------

    // boilerplate reference kernel
    template <int K, typename T>
    int kernel_ref()
    {
        T           t{0};
        std::string id{typeid(t).name()};
        std::cout << "K=" + std::to_string(K) + " " + std::string(__func__) + "<" + id + ">"
                  << std::endl;
        return K;
    }

    // boilerplate KT kernel
    template <int K, int SZ, typename SUF, mock M>
    int kernel_kt()
    {
        SUF         t{0};
        std::string id{typeid(t).name()};
        std::cout << "K=" + std::to_string(K) + " " + std::string(__func__) + "<" + id + ","
                         + std::to_string(SZ) + "," + std::to_string(M) + ">"
                  << std::endl;
        return K;
    }

    // -------------------------------------------------------------------------
    // DISPATCHERS
    // -------------------------------------------------------------------------

    // dispatcher to single kernel
    template <typename T>
    int dispatch_only_ref(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name                 cpu flag requirements    suggested architecture
        {kernel_ref<0, T>,      context_isa_t::GENERIC, 0|archs::ALL}
        };
        // clang-format on

        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved0>(tbl, kid);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();

        return 0 + okid;
    }

    /* Boilerplate dispatcher common for all Level 1 Sparse BLAS API entry points */
    template <typename T>
    int dispatch_l1(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name                              cpu flag requirements   suggested architecture
        {kernel_ref<0, T>,                   context_isa_t::GENERIC, 0|archs::ALL },
        {kernel_kt<1, 256, T, mock::CASE_1>, context_isa_t::AVX2,    0|archs::ZENS}, // Zen 1+ AVX2
        {kernel_kt<2, 256, T, mock::CASE_1>, context_isa_t::AVX2,    0|archs::ZEN4}, // Zen 4+ AVX2
#ifdef USE_AVX512
        {kernel_kt<3, 512, T, mock::CASE_2>, context_isa_t::AVX512F, 0|archs::ZEN4} // Zen 4+ AVX512F
#endif
        };
        // clang-format on
        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved1>(tbl, kid);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();

        return 10 + okid;
    }

    template <typename T>
    int dispatch_multi(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        if constexpr(std::is_same_v<T, float>)
        {
            static constexpr Dispatch::Table<K> tbl[]{
                {kernel_ref<0, T>, context_isa_t::GENERIC, 0 | archs::ALL},
                {kernel_ref<1, T>, context_isa_t::GENERIC, 0 | archs::ALL},
                {kernel_ref<2, T>, context_isa_t::GENERIC, 0 | archs::ALL}};
            auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved2>(tbl, kid);

            if(kernel == nullptr)
                return aoclsparse_status_invalid_kid;

            return kernel();
        }
        else
        {
            static constexpr Dispatch::Table<K> tbl[]{
                {kernel_ref<7, T>, context_isa_t::GENERIC, 0 | archs::ALL}};
            auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved3>(tbl, kid);

            if(kernel == nullptr)
                return aoclsparse_status_invalid_kid;

            return kernel();
        }
    }

    // Table does not provide exact kernel match
    template <typename T>
    int dispatch_noexact(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name            cpu flag requirements   suggested architecture
        {kernel_ref<0, T>, context_isa_t::GENERIC, 0|archs::ALL},
        {kernel_ref<1, T>, context_isa_t::GENERIC, 0|archs::ZEN123},
        {kernel_ref<2, T>, context_isa_t::GENERIC, 0|archs::ZENS}
        };
        // clang-format on
        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved4>(tbl, kid);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();
        return 0 + okid;
    }

    // high complexity dispatcher to test corner cases
    template <typename T>
    int dispatch(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);
        using namespace aoclsparse;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name                              cpu flag requirements    suggested architecture
        {kernel_ref<0, T>,                   context_isa_t::GENERIC,  0|archs::ALL},    // 0 All machines
        {kernel_ref<1, T>,                   context_isa_t::AVX2,     0|archs::ZEN123}, // 1 Naples/Rome/Milan: Zen AVX2
        {kernel_kt<2, 256, T, mock::CASE_1>, context_isa_t::AVX2,     0|archs::ZENS},   // 2 AMD platform AVX2
        {kernel_kt<3, 256, T, mock::CASE_1>, context_isa_t::AVX2,     0|archs::ZEN3},   // 3 Milan: Zen3 AVX2
        {kernel_kt<4, 256, T, mock::CASE_1>, context_isa_t::AVX2,     0|archs::ZEN4},   // 4 Bergamo/Genoa/Sienna: Zen4 AVX2
#ifdef USE_AVX512
        {kernel_kt<5, 512, T, mock::CASE_3>, context_isa_t::AVX512DQ, 0|archs::ZENS},   // 5 AMD platform AVX512F
        {kernel_kt<6, 512, T, mock::CASE_3>, context_isa_t::AVX512VL, 0|archs::ZEN4}   // 6 Bergamo/Genoa/Sienna: Zen4 AVX512F
#endif
        };
        // clang-format on

        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved5>(tbl, kid);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();

        return 1000 + okid;
    }

    // Table does not provide exactly one kernel per ISA_HINT
    template <typename T>
    int dispatch_isa(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name            cpu flag requirements    suggested architecture
        {kernel_ref<0, T>, context_isa_t::GENERIC, 0|archs::ALL},
        {kernel_ref<1, T>, context_isa_t::AVX2, 0|archs::ALL}
        };
        // clang-format on
        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved4>(tbl, kid);

        if(kernel == nullptr)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();

        return 0 + okid;
    }

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
        6. ZEN123
        7. ZENS

        Note: Test the logic again when you want to add tokeenization for
        new strings
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
        debug_info info;

        [[maybe_unused]] aoclsparse_status st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated, info.arch);

        std::string arch{info.arch};

        uint32_t arch_id = tokenize_isa(arch.c_str(), arch.length());

        EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);

        switch(arch_id)
        {
        case tokenize_isa("ZEN4", 4):
            // Auto kernel
            EXPECT_EQ(dispatch_l1<float>(), 2 + 10);

            // Request specific kernel 3
            EXPECT_EQ(dispatch_l1<float>(3), 3 + 10);

            // Force generic path
            EXPECT_EQ(aoclsparse_enable_instructions("GENERIC"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_l1<float>(), 0 + 10);

            // make sure kid is taken into account
            EXPECT_EQ(dispatch_l1<float>(1), 1 + 10);

            // Force AVX2 path on Zen4 is kid=2 not kid=1
            EXPECT_EQ(aoclsparse_enable_instructions("avx2"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_l1<float>(-1), 2 + 10);

            // Force AVX512 path on Zen4 is kid=3
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_l1<float>(-1), 3 + 10);

            break;
        case tokenize_isa("ZEN", 3):
        case tokenize_isa("ZEN2", 4):
        case tokenize_isa("ZEN3", 4):
        case tokenize_isa("ZEN123", 6):
            // Auto kernel
            EXPECT_EQ(dispatch_l1<float>(), 1 + 10);

            // Request specific kernel 3. Not supported by the architecture
            EXPECT_EQ(dispatch_l1<float>(3), aoclsparse_status_invalid_kid);

            // Force generic path
            EXPECT_EQ(aoclsparse_enable_instructions("Generic"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_l1<float>(), 0 + 10);

            // make sure kid is taken into account
            EXPECT_EQ(dispatch_l1<float>(1), 1 + 10);

            // Force AVX2 path on Zen3 is kid=1
            EXPECT_EQ(aoclsparse_enable_instructions("avx2"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_l1<float>(-1), 1 + 10);

            // Invalid AVX512 path on Zen3... revert to auto best
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_l1<float>(-1), 1 + 10);

            // kid = auto
            EXPECT_EQ(dispatch_l1<float>(), 1 + 10);
            break;

        case tokenize_isa("GENERIC", 7):
            // Auto kernel
            EXPECT_EQ(dispatch_l1<float>(), 0 + 10);

            // Request specific kernel 3
            EXPECT_EQ(dispatch_l1<float>(3), 0 + 10);

            // Force generic path
            EXPECT_EQ(aoclsparse_enable_instructions("Generic"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_l1<float>(), 0 + 10);

            // make sure kid is not taken into account
            EXPECT_EQ(dispatch_l1<float>(1), 0 + 10);

            // Force AVX512 path on Zen4 is kid=3
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_l1<float>(), 0 + 10);
            EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);
            break;
        }
    }

    TEST(Oracle, DispatchUniq)
    {
        int kid;
        kid = dispatch_only_ref<float>();
        EXPECT_EQ(kid, 0 + 0);
        kid = dispatch_only_ref<std::complex<double>>();
        EXPECT_EQ(kid, 0 + 0);
    }

    TEST(Oracle, DispatchMulti)
    {
        int kid;
        kid = dispatch_multi<float>(2);
        EXPECT_EQ(kid, 2 + 0);

        kid = dispatch_multi<double>();
        EXPECT_EQ(kid, 7 + 0);
    }

    /* No exact kernel in table */
    TEST(Oracle, DispatchNoExact)
    {
        debug_info                         info;
        [[maybe_unused]] aoclsparse_status st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated, info.arch);

        std::string arch{info.arch};

        uint32_t arch_id = tokenize_isa(arch.c_str(), arch.length());
        EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);
        switch(arch_id)
        {
        case tokenize_isa("ZEN4", 4):
            // Auto
            EXPECT_EQ(dispatch_noexact<float>(), 2);
            // AVX512 noise
            EXPECT_EQ(aoclsparse_enable_instructions("avx2"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_noexact<float>(), 2);
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_noexact<float>(), 2);
            break;
        case tokenize_isa("ZEN", 3):
        case tokenize_isa("ZEN2", 4):
        case tokenize_isa("ZEN3", 4):
        case tokenize_isa("ZEN123", 6):
            EXPECT_EQ(dispatch_noexact<float>(), 1);
            // noise
            EXPECT_EQ(aoclsparse_enable_instructions("Generic"), aoclsparse_status_success);
            EXPECT_EQ(dispatch_noexact<float>(-1), 1);
            break;
        case tokenize_isa("GENERIC", 6):
            EXPECT_EQ(dispatch_noexact<float>(), 0);
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

        std::string arch{info.arch};

        uint32_t arch_id = tokenize_isa(arch.c_str(), arch.length());

        EXPECT_EQ(aoclsparse_enable_instructions(""), aoclsparse_status_success);
        switch(arch_id)
        {
        case tokenize_isa("ZEN4", 4):
            // kid auto contend ZEN123 vs ZENS vs ZEN3
            EXPECT_EQ(dispatch<float>(), 4 + 1000); // AVX2 path priority by table
            // request AVX512 path contend ZEN4 vs ZENS
            EXPECT_EQ(aoclsparse_enable_instructions("AVX512"), aoclsparse_status_success);
            EXPECT_EQ(dispatch<float>(), 4 + 1000); // AVX2 path priority by table
            break;
        case tokenize_isa("ZEN3", 4):
            // kid auto contend ZEN123 vs ZENS vs ZEN3
            EXPECT_EQ(dispatch<float>(), 3 + 1000);
            break;
        case tokenize_isa("GENERIC", 6):
            // test already covered
            break;
        default:
            // Zen 1/1+/2:
            EXPECT_EQ(dispatch<float>(), 1 + 1000);
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

            // Check if expected kid is launched
            *ok = kid == dispatch_isa<float>();
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
        // Non-existent kernel kid
        EXPECT_EQ(dispatch_l1<float>(2024), aoclsparse_status_invalid_kid);
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

            // Check if expected kid is launched
            *ok = kid == dispatch_isa<float>();
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
#pragma omp          master
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
}
#endif