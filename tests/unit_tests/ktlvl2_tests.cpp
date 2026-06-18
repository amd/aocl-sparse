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
 * ************************************************************************ */
#include "aoclsparse.h"
#include "common_data_utils.h"

#include <algorithm>
#include <complex>
#include <limits>
#include <type_traits>

namespace TestsKT
{
    // forward declaration of driver
    void driver_spmv_b1_fdcz(void);
    void driver_spmv_b2_fdcz(void);
    void driver_spmv_b5_fdcz(void);
    void driver_spmv_b125_h(void);

    TEST(KT_L2, kt_spmv_all_fp16)
    {
        if(can_exec_avx512fp16_tests())
        {
            driver_spmv_b125_h();
        }
        else
        {
            GTEST_SKIP() << "No AVX512FP16 on local machine.\n";
        }
    }

    TEST(KT_L2, kt_spmv_b128_all)
    {
        driver_spmv_b1_fdcz();
    }

    TEST(KT_L2, kt_spmv_b256_all)
    {
        driver_spmv_b2_fdcz();
    }

    TEST(KT_L2, kt_spmv_b512_all)
    {
        if(can_exec_avx512_tests())
        {
            driver_spmv_b5_fdcz();
        }
        else
        {
            GTEST_SKIP() << "No AVX512 on local machine.\n";
        }
    }
}
