/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#include "aoclsparse.h"
#include "aoclsparse_auxiliary.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"

#include <omp.h>
#include <thread>
#include <type_traits>

#include "Au/Cpuid/X86Cpu.hh"

namespace contextTest
{
    class debug_info
    {
    public:
        char           *global_isa;
        char           *tl_isa;
        aoclsparse_int *sparse_nt;
        bool           *is_isa_updated;

        debug_info()
        {
            global_isa     = new char[20];
            tl_isa         = new char[20];
            sparse_nt      = new aoclsparse_int;
            is_isa_updated = new bool;
        }

        ~debug_info()
        {
            delete[] global_isa;
            delete[] tl_isa;
            delete sparse_nt;
            delete is_isa_updated;
        }
    };

    // Test if number of threads returned by get_num_threads() can be launched
    TEST(context, threadingTest)
    {

#ifdef _OPENMP
        size_t n = 10;
        // Get the number of threads from the sparse global object
        debug_info info;

        aoclsparse_debug_get(info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated);

#pragma omp parallel num_threads(*(info.sparse_nt))
        for(size_t i = 0; i < n; ++i)
        {
            // Get the number of threads in the parallel region
            size_t nt = omp_get_num_threads();

            // Check if that is equal to sparse NT
            EXPECT_EQ(nt, *(info.sparse_nt));
        }
#endif
    }

    void change_tl_isa(char tl_isa_hint[])
    {
        debug_info info;

        [[maybe_unused]] auto st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated);

        // Check if the global isa hint is same as that of the tl isa during init
        EXPECT_TRUE(!strcmp(info.global_isa, info.tl_isa));

        // Check if the new and old ISA preference are the same at init
        EXPECT_TRUE(*(info.is_isa_updated));

        // Enable a different instruction
        [[maybe_unused]] auto s = aoclsparse_enable_instructions(tl_isa_hint);

        st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated);

        // Test if the expected value is set
        // ToDo: This test can fail if the target hardware doesn't support "tl_isa_hint" - need to revisit
        EXPECT_TRUE(!strcmp(tl_isa_hint, info.tl_isa));

        // Check if the global isa hint is different from that of the child thread
        // ToDo: This test can still fail if the target hardware doesn't support AVX2 and AVX512 - need to revisit
        EXPECT_FALSE(!strcmp(info.global_isa, info.tl_isa));

        // Check if the new and old ISA preference are NOT the same at init
        EXPECT_FALSE(*(info.is_isa_updated));
    }

    // Test if the isa hint is initialized to the global context' isa
    TEST(context, isaInit)
    {
        debug_info info;

        [[maybe_unused]] auto s = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated);

        EXPECT_TRUE(!strcmp(info.global_isa, info.tl_isa));

        // Check if the new and old ISA preference are the same at init
        EXPECT_TRUE(info.is_isa_updated);
    }

    // Test if the isa hint change is thread local
    TEST(context, tl_isa_change)
    {
        char       isa[8] = "GENERIC";
        Au::X86Cpu Cpu    = {0};

        // Enable a different instruction for the calling thread
        [[maybe_unused]] auto s = aoclsparse_enable_instructions("AVX512");

        std::thread tr(change_tl_isa, isa);
        tr.join();

        debug_info info;

        aoclsparse_debug_get(info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated);

        // If AVX512 is supported by the core, then the tl isa will be AVX512
        if(Cpu.hasFlag(Au::ECpuidFlag::avx512f))
            EXPECT_TRUE(!strcmp("AVX512", info.tl_isa));
        else if(Cpu.hasFlag(Au::ECpuidFlag::avx2))
            EXPECT_TRUE(!strcmp("AVX2", info.tl_isa));
        else
            EXPECT_TRUE(!strcmp("GENERIC", info.tl_isa));
    }

    /*
        This function modifies the thread local isa hint to the mentioned
        value and saves the modified value to a ledger.
    */
    void modify_isa(char new_value[], char ledger[])
    {
        [[maybe_unused]] auto s = aoclsparse_enable_instructions(new_value);

        debug_info info;

        [[maybe_unused]] auto st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated);

        strcpy(ledger, info.tl_isa);
    }

    // Spawn CPP threads and test if they set the expected values
    void cpp_thread_instance()
    {
        constexpr size_t thread_count = 3;
        Au::X86Cpu       Cpu          = {0};

        char init_v_1[10] = "AVX2";
        char init_v_2[10] = "AVX512";
        char init_v_3[10] = "GENERIC";
        char ledger_1[10];
        char ledger_2[10];
        char ledger_3[10];

        std::thread t[thread_count];

        t[0] = std::thread(modify_isa, init_v_1, ledger_1);
        t[1] = std::thread(modify_isa, init_v_2, ledger_2);
        t[2] = std::thread(modify_isa, init_v_3, ledger_3);

        for(size_t i = 0; i < 2; ++i)
            t[i].join();

        if(Cpu.hasFlag(Au::ECpuidFlag::avx2))
            EXPECT_TRUE(!strcmp(init_v_1, ledger_1));
        else
            EXPECT_TRUE(!strcmp("GENERIC", ledger_1));

        // If AVX512 is supported by the core, then the tl isa will be AVX512
        if(Cpu.hasFlag(Au::ECpuidFlag::avx512f))
            EXPECT_TRUE(!strcmp(init_v_2, ledger_2));
        else if(Cpu.hasFlag(Au::ECpuidFlag::avx2))
            EXPECT_TRUE(!strcmp("AVX2", ledger_2));
        else
            EXPECT_TRUE(!strcmp("GENERIC", ledger_2));

        t[2].join();

        EXPECT_TRUE(!strcmp(init_v_3, ledger_3));
    }

    // Driver for CPP thread multi-instance
    TEST(context, multiInstanceCppThread)
    {
        cpp_thread_instance();
    }

    void omp_thread_instance_test()
    {
#ifdef _OPENMP
        Au::X86Cpu Cpu = {0};

        char init_ledger[3][10] = {"AVX2", "AVX512", "GENERIC"};
        char res_ledger[3][10];

#pragma omp parallel for
        for(size_t i = 0; i < 3; ++i)
        {
            modify_isa(init_ledger[i], res_ledger[i]);
        }

        if(Cpu.hasFlag(Au::ECpuidFlag::avx2))
            EXPECT_TRUE(!strcmp(init_ledger[0], res_ledger[0]));
        else
            EXPECT_TRUE(!strcmp("GENERIC", res_ledger[0]));

        // If AVX512 is supported by the core, then the tl isa will be AVX512
        if(Cpu.hasFlag(Au::ECpuidFlag::avx512f))
            EXPECT_TRUE(!strcmp(init_ledger[1], res_ledger[1]));
        else if(Cpu.hasFlag(Au::ECpuidFlag::avx2))
            EXPECT_TRUE(!strcmp("AVX2", res_ledger[1]));
        else
            EXPECT_TRUE(!strcmp("GENERIC", res_ledger[1]));

        EXPECT_TRUE(!strcmp(init_ledger[2], res_ledger[2]));
#endif
    }

    // Test multi-instance created by OMP
    TEST(context, multiInstanceOMP)
    {
#ifdef _OPENMP
        // Single region
        omp_thread_instance_test();

        // Nested regions
#pragma omp parallel for
        for(size_t i = 0; i < 3; ++i)
        {
            omp_thread_instance_test();
        }
#endif
    }

    // Test multi-instance created by OMP
    TEST(context, CppPlusOMPthreads)
    {
#ifdef _OPENMP

        // Nested regions
#pragma omp parallel for
        for(size_t i = 0; i < 3; ++i)
        {
            cpp_thread_instance();
        }
#endif
    }

    // Spawn CPP threads and test for invalid ISA
    void cpp_thread_invalid()
    {
        constexpr size_t thread_count = 3;
        Au::X86Cpu       Cpu          = {0};

        char init_v_1[10] = "AVX2";
        char init_v_2[10] = "AVX512";
        char init_v_3[10] = "RACECAR";
        char ledger_1[10];
        char ledger_2[10];
        char ledger_3[10];
        char ledger_base[10];

        debug_info info;

        // Get the debug information
        [[maybe_unused]] auto st = aoclsparse_debug_get(
            info.global_isa, info.sparse_nt, info.tl_isa, info.is_isa_updated);

        // Copy the global isa
        strcpy(ledger_base, info.global_isa);

        std::thread t[thread_count];

        t[0] = std::thread(modify_isa, init_v_1, ledger_1);
        t[1] = std::thread(modify_isa, init_v_2, ledger_2);
        t[2] = std::thread(modify_isa, init_v_3, ledger_3);

        for(size_t i = 0; i < 2; ++i)
            t[i].join();

        if(Cpu.hasFlag(Au::ECpuidFlag::avx2))
            EXPECT_TRUE(!strcmp(init_v_1, ledger_1));
        else
            EXPECT_TRUE(!strcmp("GENERIC", ledger_1));

        // If AVX512 is supported by the core, then the tl isa will be AVX512
        if(Cpu.hasFlag(Au::ECpuidFlag::avx512f))
            EXPECT_TRUE(!strcmp(init_v_2, ledger_2));
        else if(Cpu.hasFlag(Au::ECpuidFlag::avx2))
            EXPECT_TRUE(!strcmp("AVX2", ledger_2));
        else
            EXPECT_TRUE(!strcmp("GENERIC", ledger_2));

        t[2].join();

        // The ISA returned should not match the invalid ISA
        EXPECT_FALSE(!strcmp(init_v_3, ledger_3));
        // The ISA returned should match the global default ISA
        EXPECT_TRUE(!strcmp(ledger_base, ledger_3));
    }

    // Test multi-instance created by OMP for invalid ISA usage
    TEST(context, InvalidISA)
    {
#ifdef _OPENMP

        // Nested regions
#pragma omp parallel for
        for(size_t i = 0; i < 3; ++i)
        {
            cpp_thread_invalid();
        }
#endif
    }
}