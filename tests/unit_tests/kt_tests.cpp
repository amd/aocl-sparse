/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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
#include "common_data_utils.h"
#include "gtest/gtest.h"

#include <string>

// -----------------------------------------------------------------------------
// Redefine the Declaration / Instantiation macros of the drivers to:
// * Forward declare the drivers, and
// * Define the GTESTs
#define KT_TEST_DO3(FUNC, SZ, SUF)                                                       \
    void FUNC##_##SZ##_##SUF();                                                          \
    TEST(KT_TEST, FUNC##_##SZ##_##SUF)                                                   \
    {                                                                                    \
        bool ok{true};                                                                   \
        if(#SUF##s == "_Float16"s || #SUF##s == "fp16"s)                                 \
        {                                                                                \
            ok = can_exec_avx512fp16_tests();                                            \
        }                                                                                \
        else if(#SZ##s == "b512"s)                                                       \
        {                                                                                \
            ok = can_exec_avx512_tests();                                                \
        }                                                                                \
        if(ok)                                                                           \
        {                                                                                \
            FUNC##_##SZ##_##SUF();                                                       \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            GTEST_SKIP() << "No runtime support for " << #SUF << " with " << #SZ << "."; \
        }                                                                                \
    }

#define KT_TEST_DO4(FUNC, SZ, SUF, IDX)                                                     \
    void FUNC##_##SZ##_##SUF##_##IDX();                                                     \
    TEST(KT_TEST, FUNC##_##SZ##_##SUF##_##IDX)                                              \
    {                                                                                       \
        bool ok{true};                                                                      \
        if(#SUF##s == "_Float16"s || #SUF##s == "fp16"s)                                    \
        {                                                                                   \
            ok = can_exec_avx512fp16_tests();                                               \
        }                                                                                   \
        else if(#SZ##s == "b512"s)                                                          \
        {                                                                                   \
            ok = can_exec_avx512_tests();                                                   \
        }                                                                                   \
        if(ok)                                                                              \
        {                                                                                   \
            FUNC##_##SZ##_##SUF##_##IDX();                                                  \
        }                                                                                   \
        else                                                                                \
        {                                                                                   \
            GTEST_SKIP() << "No runtime support for " << #SUF << " with " << #SZ << " and " \
                         << #IDX << ".";                                                    \
        }                                                                                   \
    }
// -----------------------------------------------------------------------------

namespace TestsKT
{
    using namespace std::literals::string_literals;

    // Drivers for type tests forward declarations
    // -------------------------------------------
    void kt_base_t_check();
    void kt_base_t_check_fp16();

    void kt_is_same_test();
    void kt_is_same_test_fp16();

    void kt_types_128();
    void kt_types_128_fp16();

    void kt_ctypes_128();
    void kt_ctypes_128_fp16();

    void kt_types_256();
    void kt_types_256_fp16();

    void kt_ctypes_256();
    void kt_ctypes_256_fp16();

    void kt_types_512();

    void kt_ctypes_512();

    TEST(KT_TYPE, KT_BASE_T_CHECK)
    {
        kt_base_t_check();
    }

    TEST(KT_TYPE, KT_BASE_T_CHECK_FP16)
    {
        if(can_exec_avx512fp16_tests())
            kt_base_t_check_fp16();
        else
            GTEST_SKIP() << "No runtime support for AVX512FP16.";
    }

    TEST(KT_TYPE, KT_IS_SAME)
    {
        kt_is_same_test();
    }

    TEST(KT_TYPE, KT_IS_SAME_FP16)
    {
        if(can_exec_avx512fp16_tests())
            kt_is_same_test_fp16();
        else
            GTEST_SKIP() << "No runtime support for AVX512FP16.";
    }

    TEST(KT_TYPE, KT_TYPES_128)
    {
        kt_types_128();
    }

    TEST(KT_TYPE, KT_TYPES_128_FP16)
    {
        if(can_exec_avx512fp16_tests())
            kt_types_128_fp16();
        else
            GTEST_SKIP() << "No runtime support for AVX512FP16.";
    }

    TEST(KT_TYPE, KT_CTYPES_128)
    {
        kt_ctypes_128();
    }

    TEST(KT_TYPE, KT_CTYPES_128_FP16)
    {
        if(can_exec_avx512fp16_tests())
            kt_ctypes_128_fp16();
        else
            GTEST_SKIP() << "No runtime support for AVX512FP16.";
    }

    TEST(KT_TYPE, KT_TYPES_256)
    {
        kt_types_256();
    }

    TEST(KT_TYPE, KT_TYPES_256_FP16)
    {
        if(can_exec_avx512fp16_tests())
            kt_types_256_fp16();
        else
            GTEST_SKIP() << "No runtime support for AVX512FP16.";
    }

    TEST(KT_TYPE, KT_CTYPES_256)
    {
        kt_ctypes_256();
    }

    TEST(KT_TYPE, KT_CTYPES_256_FP16)
    {
        if(can_exec_avx512fp16_tests())
            kt_ctypes_256_fp16();
        else
            GTEST_SKIP() << "No runtime support for AVX512FP16.";
    }

    TEST(KT_TYPE, KT_TYPES_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_types_512();
        }
    }

    TEST(KT_TYPE, KT_CTYPES_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_ctypes_512();
        }
    }

// Add the KT_TEST tests
#define KT_TEST_ADD_ONLY_AVX2
// Add the AVX2 tests
#include "kt_kernels.hpp"
#undef KT_TEST_ADD_ONLY_AVX2
// Add the AVX512 tests
#include "kt_kernels.hpp"

}
