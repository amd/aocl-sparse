/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

using kt_int_t = size_t;

#include "kernel-templates/kernel_templates.hpp"

using namespace kernel_templates;

namespace TestsKT
{
    // Test function declaration
    // -------------------------
    void kt_base_t_check();

    void kt_is_same_test();

    void kt_types_128();

    void kt_ctypes_128();

    void kt_types_256();

    void kt_ctypes_256();

    void kt_types_512();

    void kt_ctypes_512();

    template <bsz SZ>
    void kt_loadu_p_test();

    template <bsz SZ>
    void kt_setzero_p_test();

    template <bsz SZ>
    void kt_set1_p_test();

    template <bsz SZ>
    void kt_add_p_test();

    template <bsz SZ>
    void kt_sub_p_test();

    template <bsz SZ>
    void kt_mul_p_test();

    template <bsz SZ>
    void kt_fmadd_p_test();

    template <bsz SZ>
    void kt_fmsub_p_test();

    template <bsz SZ>
    void kt_set_p_test();

    void kt_maskz_set_p_256_avx();

    void kt_maskz_set_p_256_AVX512vl();

    void kt_maskz_set_p_512_AVX512f();

    template <bsz SZ>
    void kt_hsum_p_test();

    template <bsz SZ>
    void kt_conj_p_test();

    template <bsz SZ>
    void kt_dot_p_test();

    template <bsz SZ>
    void kt_cdot_p_test();

    template <bsz SZ>
    void kt_storeu_p_test();

    // -------------------------
    template <bsz SZ>
    void kt_fmadd_B_test();

    template <bsz SZ>
    void kt_hsum_B_test();
    // -------------------------

    TEST(KT_L0, KT_BASE_T_CHECK)
    {
        kt_base_t_check();
    }

    TEST(KT_L0, KT_IS_SAME)
    {
        kt_is_same_test();
    }

    TEST(KT_L0, KT_TYPES_128)
    {
        kt_types_128();
    }

    TEST(KT_L0, KT_CTYPES_128)
    {
        kt_ctypes_128();
    }

    TEST(KT_L0, KT_TYPES_256)
    {
        kt_types_256();
    }

    TEST(KT_L0, KT_CTYPES_256)
    {
        kt_ctypes_256();
    }

    TEST(KT_L0, KT_TYPES_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_types_512();
        }
    }

    TEST(KT_L0, KT_CTYPES_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_ctypes_512();
        }
    }

    /*
     * Test loadu intrinsic to load 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_loadu_p_128)
    {
        kt_loadu_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_loadu_p_256)
    {
        kt_loadu_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_loadu_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_loadu_p_test<bsz::b512>();
        }
    }

    /*
     * Test setzero intrinsic to zero-out 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_setzero_p_128)
    {
        kt_setzero_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_setzero_p_256)
    {
        kt_setzero_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_setzero_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_setzero_p_test<bsz::b512>();
        }
    }

    /*
     * Test set1 intrinsic to load a scalat into 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_set1_p_128)
    {
        kt_set1_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_set1_p_256)
    {
        kt_set1_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_set1_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_set1_p_test<bsz::b512>();
        }
    }

    /*
     * Test add intrinsic to sum two: 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_add_p_128)
    {
        kt_add_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_add_p_256)
    {
        kt_add_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_add_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_add_p_test<bsz::b512>();
        }
    }

    /*
     * Test intrinsic to subtract two: 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_sub_p_128)
    {
        kt_sub_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_sub_p_256)
    {
        kt_sub_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_sub_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_sub_p_test<bsz::b512>();
        }
    }

    /*
     * Test mul intrinsic to multiply two 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_mul_p_128)
    {
        kt_mul_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_mul_p_256)
    {
        kt_mul_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_mul_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_mul_p_test<bsz::b512>();
        }
    }

    /*
     * Test fmadd intrinsic to fuse-multiply-add three
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_fmadd_p_128)
    {
        kt_fmadd_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_fmadd_p_256)
    {
        kt_fmadd_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_fmadd_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_fmadd_p_test<bsz::b512>();
        }
    }

    /*
     * Test fmsub intrinsic to fused-multiply-subtract three
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_fmsub_p_128)
    {
        kt_fmsub_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_fmsub_p_256)
    {
        kt_fmsub_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_fmsub_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_fmsub_p_test<bsz::b512>();
        }
    }

    /*
     * Test "set" intrinsic to indirectly load using a "map"
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_set_p_128)
    {
        kt_set_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_set_p_256)
    {
        kt_set_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_set_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_set_p_test<bsz::b512>();
        }
    }

    TEST(KT_L0, kt_maskz_set_p_256_AVX)
    {
        kt_maskz_set_p_256_avx();
    }

    TEST(KT_L0, kt_maskz_set_p_256_AVX512VL)
    {
        if(can_exec_avx512_tests())
        {
            kt_maskz_set_p_256_AVX512vl();
        }
    }

    TEST(KT_L0, kt_maskz_set_p_512_AVX512F)
    {
        if(can_exec_avx512_tests())
        {
            kt_maskz_set_p_512_AVX512f();
        }
    }

    /*
     * Test "hsum" intrinsic to horizontally-reduce via summation
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_hsum_p_256)
    {
        kt_hsum_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_hsum_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_hsum_p_test<bsz::b512>();
        }
    }

    /*
     * Test "dot-product" intrinsic on
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_conj_p_128)
    {
        kt_conj_p_test<bsz::b128>();
    }

    TEST(KT_L0, kt_conj_p_256)
    {
        kt_conj_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_conj_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_conj_p_test<bsz::b512>();
        }
    }

    /*
     * Test "dot-product" intrinsic on two
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L1, kt_dot_p_256)
    {
        kt_dot_p_test<bsz::b256>();
    }

    TEST(KT_L1, kt_dot_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_dot_p_test<bsz::b512>();
        }
    }

    /*
     * Test "complex dot-product" intrinsic on two
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L1, kt_cdot_p_256)
    {
        kt_cdot_p_test<bsz::b256>();
    }

    TEST(KT_L1, kt_cdot_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_cdot_p_test<bsz::b512>();
        }
    }

    /*
        Test "store" KT to store elements to memory
    */
    TEST(KT_L0, kt_storeu_p_256)
    {
        kt_storeu_p_test<bsz::b256>();
    }

    TEST(KT_L0, kt_storeu_p_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_storeu_p_test<bsz::b512>();
        }
    }

    /*
     * Test fmadd BLOCK VARIANT for fuse-multiply-add three
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_Block_L0, kt_fmadd_B_256)
    {
        kt_fmadd_B_test<bsz::b256>();
    }

    TEST(KT_Block_L0, kt_fmadd_B_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_fmadd_B_test<bsz::b512>();
        }
    }

    TEST(KT_Block_L0, kt_hsum_B_256)
    {
        kt_hsum_B_test<bsz::b256>();
    }

    TEST(KT_Block_L0, kt_hsum_B_512)
    {
        if(can_exec_avx512_tests())
        {
            kt_hsum_B_test<bsz::b512>();
        }
    }
}
