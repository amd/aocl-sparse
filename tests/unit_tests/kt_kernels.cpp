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

#include <complex>
#include <iostream>
#include <memory>
#include <typeinfo>

using namespace kernel_templates;

namespace TestsKT
{
    class KTTCommonData
    {
    public:
        // clang-format off
        size_t   map[32]{ 3,  1,  0,  2,  4,  7,  5,  6,  8, 15, 13,  9, 11, 10, 12, 14,
                         29, 19, 18, 31, 20, 23, 22, 25, 24, 16, 21, 26, 27, 30, 28, 17};
        float    vs[32]{1.5f,    2.75f,  2.0f,   3.875f,  3.5f,   2.5f,    4.0f,    1.25f,
                        2.5f,    1.25f,  3.125f, 0.125f, -3.5f,  -1.25f,   8.125f, 10.25f,
                        3.5f,    6.25f,  7.125f, 3.0f,    6.5f,   1.226f, -2.5f,    5.0f,
                        9.125f, -1.125f, 2.5f,   5.125f,  4.125f, 3.25f,   3.5f,    5.5f};
        double   vd[16]{1.5,     2.25,   3.5,    0.5,     8.25,  -3.25,    6.5,    -1.25,
                        9.125,   5.5,   -2.25,   2.5,     7.25,  -6.25,    9.125,  -0.25};
        std::complex<float>  vc[16]{ {2.5f, -1.5f},  {4.0f, -2.0f},  {4.0f, -3.0f}, {8.25f, -4.0f},
                                     {1.5f, -5.0f},   {3.5f, -6.25f},  {7.75f, -7.0f},  {9.5f, -8.0f},
                                     {-2.5f, -1.5f},  {-3.25f, -2.0f}, {-5.5f, -3.0f},  {-7.25f, -4.0f},
                                     {-9.75f, -5.0f}, {-2.2f, -6.0f},  {-4.75f, -7.5f}, {-6.0f,  -8.125f}};
        std::complex<double> vz[8]{  {1.5, -12},     {0.5, -21.0},    {0.125, -13.0},  {3.5,   -4.5},
                                     {5.25, -8.125},  {8.5, -6.75},    {9.5, -7.25},    {2.125, -3.0}};

        int32_t  bi32[32]{3,  1,  0,  2,  4,  7,  5,  6,  8, 15, 13,  9, 11, 10, 12, 14,
                         90, 19, 18, 31, 20, 23, 22, 25, 24, 16, 21, 26, 27, 30, 28, 17};

        int64_t  bi64[16]{13,  12,  90,  2,  4,  17,  25,  6, 80, 19, 18, 31, 20, 23, 22, 25};

        template <typename T>
        constexpr const T* get_data() const noexcept
        {
            if constexpr(std::is_same_v<T, float>)
                return vs;
            else if constexpr(std::is_same_v<T, double>)
                return vd;
            else if constexpr(std::is_same_v<T, cfloat>)
                return reinterpret_cast<const T*>(vc);
            else if constexpr(std::is_same_v<T, cdouble>)
                return reinterpret_cast<const T*>(vz);
            else if constexpr(std::is_same_v<T, int32_t>)
                return bi32;
            else if constexpr(std::is_same_v<T, int64_t>)
                return bi64;
            else
                return nullptr; // Unsupported type
        }
        // clang-format on
    };

    template <typename T>
    std::string get_typename()
    {
        if constexpr(std::is_same_v<T, float>)
            return "float";
        else if constexpr(std::is_same_v<T, double>)
            return "double";
        else if constexpr(std::is_same_v<T, cfloat>)
            return "std::complex<float>";
        else if constexpr(std::is_same_v<T, cdouble>)
            return "std::complex<double>";
        else if constexpr(std::is_same_v<T, int32_t>)
            return "int32_t";
        else if constexpr(std::is_same_v<T, int64_t>)
            return "int64_t";
    }

    const KTTCommonData D;

    // These functions only need to be defined in AVX2 builds
#ifdef KT_AVX2_BUILD
    void kt_base_t_check()
    {
        EXPECT_TRUE(kt_is_base_t_float<float>());
        EXPECT_TRUE(kt_is_base_t_float<cfloat>());
        EXPECT_FALSE(kt_is_base_t_float<double>());
        EXPECT_FALSE(kt_is_base_t_float<cdouble>());

        EXPECT_TRUE(kt_is_base_t_double<double>());
        EXPECT_TRUE(kt_is_base_t_double<cdouble>());
        EXPECT_FALSE(kt_is_base_t_double<float>());
        EXPECT_FALSE(kt_is_base_t_double<cfloat>());

        EXPECT_TRUE(kt_is_base_t_int<int32_t>());
        EXPECT_TRUE(kt_is_base_t_int<int64_t>());
        EXPECT_FALSE(kt_is_base_t_int<float>());
        EXPECT_FALSE(kt_is_base_t_int<double>());
    }

    void kt_is_same_test()
    {
        EXPECT_TRUE((kt_is_same<bsz::b256, bsz::b256, double, double>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b256, double, cdouble>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b512, double, double>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b512, float, cfloat>()));
    }

    void kt_types_128()
    {
        /*
         * These bsz::b128 bit tests check that the correct data type
         * and packed sizes are "returned".
         */
        // bsz::b128 float
        EXPECT_EQ(typeid(avxvector<bsz::b128, float>::type), typeid(__m128));
        EXPECT_EQ(typeid(avxvector<bsz::b128, float>::half_type), typeid(__m64));
        EXPECT_EQ((avxvector<bsz::b128, float>::p_size), 4U);
        EXPECT_EQ((avxvector<bsz::b128, float>()), 4U);
        EXPECT_EQ((avxvector<bsz::b128, float>::hp_size), 2U);
        EXPECT_EQ((avxvector<bsz::b128, float>::tsz), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b128, float>), typeid(__m128));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b128, float>), typeid(__m64));
        EXPECT_EQ((avxvector_v<bsz::b128, float>), 4U);
        EXPECT_EQ((avxvector<bsz::b128, float>()), 4U);
        EXPECT_EQ((avxvector_half_v<bsz::b128, float>), 2U);
        EXPECT_EQ((hsz_v<bsz::b128, float>), 2U);
        EXPECT_EQ((tsz_v<bsz::b128, float>), 4U);

        // bsz::b128 double
        EXPECT_EQ(typeid(avxvector<bsz::b128, double>::type), typeid(__m128d));
        EXPECT_EQ(typeid(avxvector<bsz::b128, double>::half_type), typeid(__m64));
        EXPECT_EQ((avxvector<bsz::b128, double>::p_size), 2U);
        EXPECT_EQ((avxvector<bsz::b128, double>()), 2U);
        EXPECT_EQ((avxvector<bsz::b128, double>::hp_size), 1U);
        EXPECT_EQ((avxvector<bsz::b128, double>::tsz), 2U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b128, double>), typeid(__m128d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b128, double>), typeid(__m64));
        EXPECT_EQ((avxvector_v<bsz::b128, double>), 2U);
        EXPECT_EQ((avxvector<bsz::b128, double>()), 2U);
        EXPECT_EQ((avxvector_half_v<bsz::b128, double>), 1U);
        EXPECT_EQ((hsz_v<bsz::b128, double>), 1U);
        EXPECT_EQ((tsz_v<bsz::b128, double>), 2U);

        // Integer types

        // int32_t
        EXPECT_EQ(typeid(avxvector<bsz::b128, int32_t>::type), typeid(__m128i));
        EXPECT_EQ(typeid(avxvector<bsz::b128, int32_t>::half_type), typeid(__m64));
        EXPECT_EQ((avxvector<bsz::b128, int32_t>::p_size), 4U);
        EXPECT_EQ((avxvector<bsz::b128, int32_t>()), 4U);
        EXPECT_EQ((avxvector<bsz::b128, int32_t>::hp_size), 2U);
        EXPECT_EQ((avxvector<bsz::b128, int32_t>::tsz), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b128, int32_t>), typeid(__m128i));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b128, int32_t>), typeid(__m64));
        EXPECT_EQ((avxvector_v<bsz::b128, int32_t>), 4U);
        EXPECT_EQ((avxvector<bsz::b128, int32_t>()), 4U);
        EXPECT_EQ((avxvector_half_v<bsz::b128, int32_t>), 2U);
        EXPECT_EQ((hsz_v<bsz::b128, int32_t>), 2U);
        EXPECT_EQ((tsz_v<bsz::b128, int32_t>), 4U);

        // int64_t
        EXPECT_EQ(typeid(avxvector<bsz::b128, int64_t>::type), typeid(__m128i));
        EXPECT_EQ(typeid(avxvector<bsz::b128, int64_t>::half_type), typeid(__m64));
        EXPECT_EQ((avxvector<bsz::b128, int64_t>::p_size), 2U);
        EXPECT_EQ((avxvector<bsz::b128, int64_t>()), 2U);
        EXPECT_EQ((avxvector<bsz::b128, int64_t>::hp_size), 1U);
        EXPECT_EQ((avxvector<bsz::b128, int64_t>::tsz), 2U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b128, int64_t>), typeid(__m128i));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b128, int64_t>), typeid(__m64));
        EXPECT_EQ((avxvector_v<bsz::b128, int64_t>), 2U);
        EXPECT_EQ((avxvector<bsz::b128, int64_t>()), 2U);
        EXPECT_EQ((avxvector_half_v<bsz::b128, int64_t>), 1U);
        EXPECT_EQ((hsz_v<bsz::b128, int64_t>), 1U);
        EXPECT_EQ((tsz_v<bsz::b128, int64_t>), 2U);
    }

    void kt_ctypes_128()
    {
        /*
         * These bsz::b128 bit tests check that the correct complex data type
         * and packed sizes are "returned".
         */
        // bsz::b128 cfloat
        EXPECT_EQ(typeid(avxvector<bsz::b128, cfloat>::type), typeid(__m128));
        EXPECT_EQ(typeid(avxvector<bsz::b128, cfloat>::half_type), typeid(__m64));
        EXPECT_EQ((avxvector<bsz::b128, cfloat>::p_size), 4U);
        EXPECT_EQ((avxvector<bsz::b128, cfloat>()), 4U);
        EXPECT_EQ((avxvector<bsz::b128, cfloat>::hp_size), 2U);
        EXPECT_EQ((avxvector<bsz::b128, cfloat>::tsz), 2U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b128, cfloat>), typeid(__m128));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b128, cfloat>), typeid(__m64));
        EXPECT_EQ((avxvector_v<bsz::b128, cfloat>), 4U);
        EXPECT_EQ((avxvector<bsz::b128, cfloat>()), 4U);
        EXPECT_EQ((avxvector_half_v<bsz::b128, cfloat>), 2U);
        EXPECT_EQ((hsz_v<bsz::b128, cfloat>), 2U);
        EXPECT_EQ((tsz_v<bsz::b128, cfloat>), 2U);

        // bsz::b128 cdouble
        EXPECT_EQ(typeid(avxvector<bsz::b128, cdouble>::type), typeid(__m128d));
        EXPECT_EQ(typeid(avxvector<bsz::b128, cdouble>::half_type), typeid(__m64));
        EXPECT_EQ((avxvector<bsz::b128, cdouble>::p_size), 2U);
        EXPECT_EQ((avxvector<bsz::b128, cdouble>()), 2U);
        EXPECT_EQ((avxvector<bsz::b128, cdouble>::hp_size), 0U);
        EXPECT_EQ((avxvector<bsz::b128, cdouble>::tsz), 1U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b128, cdouble>), typeid(__m128d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b128, cdouble>), typeid(__m64));
        EXPECT_EQ((avxvector_v<bsz::b128, cdouble>), 2U);
        EXPECT_EQ((avxvector<bsz::b128, cdouble>()), 2U);
        EXPECT_EQ((avxvector_half_v<bsz::b128, cdouble>), 0U);
        EXPECT_EQ((hsz_v<bsz::b128, cdouble>), 0U);
        EXPECT_EQ((tsz_v<bsz::b128, cdouble>), 1U);
    }

    void kt_types_256()
    {
        /*
         * These bsz::b256 bit tests check that the correct data type
         * and packed sizes are "returned".
         */
        // bsz::b256 float
        EXPECT_EQ(typeid(avxvector<bsz::b256, float>::type), typeid(__m256));
        EXPECT_EQ(typeid(avxvector<bsz::b256, float>::half_type), typeid(__m128));
        EXPECT_EQ((avxvector<bsz::b256, float>::p_size), 8U);
        EXPECT_EQ((avxvector<bsz::b256, float>()), 8U);
        EXPECT_EQ((avxvector<bsz::b256, float>::hp_size), 4U);
        EXPECT_EQ((avxvector<bsz::b256, float>::tsz), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b256, float>), typeid(__m256));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b256, float>), typeid(__m128));
        EXPECT_EQ((avxvector_v<bsz::b256, float>), 8U);
        EXPECT_EQ((avxvector<bsz::b256, float>()), 8U);
        EXPECT_EQ((avxvector_half_v<bsz::b256, float>), 4U);
        EXPECT_EQ((hsz_v<bsz::b256, float>), 4U);
        EXPECT_EQ((tsz_v<bsz::b256, float>), 8U);

        // bsz::b256 double
        EXPECT_EQ(typeid(avxvector<bsz::b256, double>::type), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector<bsz::b256, double>::half_type), typeid(__m128d));
        EXPECT_EQ((avxvector<bsz::b256, double>::p_size), 4U);
        EXPECT_EQ((avxvector<bsz::b256, double>()), 4U);
        EXPECT_EQ((avxvector<bsz::b256, double>::hp_size), 2U);
        EXPECT_EQ((avxvector<bsz::b256, double>::tsz), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b256, double>), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b256, double>), typeid(__m128d));
        EXPECT_EQ((avxvector_v<bsz::b256, double>), 4U);
        EXPECT_EQ((avxvector<bsz::b256, double>()), 4U);
        EXPECT_EQ((avxvector_half_v<bsz::b256, double>), 2U);
        EXPECT_EQ((hsz_v<bsz::b256, double>), 2U);
        EXPECT_EQ((tsz_v<bsz::b256, double>), 4U);

        // Integer types test
        // int32_t
        EXPECT_EQ(typeid(avxvector<bsz::b256, int32_t>::type), typeid(__m256i));
        EXPECT_EQ(typeid(avxvector<bsz::b256, int32_t>::half_type), typeid(__m128i));
        EXPECT_EQ((avxvector<bsz::b256, int32_t>::p_size), 8U);
        EXPECT_EQ((avxvector<bsz::b256, int32_t>()), 8U);
        EXPECT_EQ((avxvector<bsz::b256, int32_t>::hp_size), 4U);
        EXPECT_EQ((avxvector<bsz::b256, int32_t>::tsz), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b256, int32_t>), typeid(__m256i));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b256, int32_t>), typeid(__m128i));
        EXPECT_EQ((avxvector_v<bsz::b256, int32_t>), 8U);
        EXPECT_EQ((avxvector<bsz::b256, int32_t>()), 8U);
        EXPECT_EQ((avxvector_half_v<bsz::b256, int32_t>), 4U);
        EXPECT_EQ((hsz_v<bsz::b256, int32_t>), 4U);
        EXPECT_EQ((tsz_v<bsz::b256, int32_t>), 8U);

        // int64_t
        EXPECT_EQ(typeid(avxvector<bsz::b256, int64_t>::type), typeid(__m256i));
        EXPECT_EQ(typeid(avxvector<bsz::b256, int64_t>::half_type), typeid(__m128i));
        EXPECT_EQ((avxvector<bsz::b256, int64_t>::p_size), 4U);
        EXPECT_EQ((avxvector<bsz::b256, int64_t>()), 4U);
        EXPECT_EQ((avxvector<bsz::b256, int64_t>::hp_size), 2U);
        EXPECT_EQ((avxvector<bsz::b256, int64_t>::tsz), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b256, int64_t>), typeid(__m256i));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b256, int64_t>), typeid(__m128i));
        EXPECT_EQ((avxvector_v<bsz::b256, int64_t>), 4U);
        EXPECT_EQ((avxvector<bsz::b256, int64_t>()), 4U);
        EXPECT_EQ((avxvector_half_v<bsz::b256, int64_t>), 2U);
        EXPECT_EQ((hsz_v<bsz::b256, int64_t>), 2U);
        EXPECT_EQ((tsz_v<bsz::b256, int64_t>), 4U);
    }

    void kt_ctypes_256()
    {
        /*
         * These bsz::b256 bit tests check that the correct complex data type
         * and packed sizes are "returned".
         */
        // bsz::b256 cfloat
        EXPECT_EQ(typeid(avxvector<bsz::b256, cfloat>::type), typeid(__m256));
        EXPECT_EQ(typeid(avxvector<bsz::b256, cfloat>::half_type), typeid(__m128));
        EXPECT_EQ((avxvector<bsz::b256, cfloat>::p_size), 8U);
        EXPECT_EQ((avxvector<bsz::b256, cfloat>()), 8U);
        EXPECT_EQ((avxvector<bsz::b256, cfloat>::hp_size), 4U);
        EXPECT_EQ((avxvector<bsz::b256, cfloat>::tsz), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b256, cfloat>), typeid(__m256));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b256, cfloat>), typeid(__m128));
        EXPECT_EQ((avxvector_v<bsz::b256, cfloat>), 8U);
        EXPECT_EQ((avxvector<bsz::b256, cfloat>()), 8U);
        EXPECT_EQ((avxvector_half_v<bsz::b256, cfloat>), 4U);
        EXPECT_EQ((hsz_v<bsz::b256, cfloat>), 4U);
        EXPECT_EQ((tsz_v<bsz::b256, cfloat>), 4U);

        // bsz::b256 cdouble
        EXPECT_EQ(typeid(avxvector<bsz::b256, cdouble>::type), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector<bsz::b256, cdouble>::half_type), typeid(__m128d));
        EXPECT_EQ((avxvector<bsz::b256, cdouble>::p_size), 4U);
        EXPECT_EQ((avxvector<bsz::b256, cdouble>()), 4U);
        EXPECT_EQ((avxvector<bsz::b256, cdouble>::hp_size), 2U);
        EXPECT_EQ((avxvector<bsz::b256, cdouble>::tsz), 2U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b256, cdouble>), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b256, cdouble>), typeid(__m128d));
        EXPECT_EQ((avxvector_v<bsz::b256, cdouble>), 4U);
        EXPECT_EQ((avxvector<bsz::b256, cdouble>()), 4U);
        EXPECT_EQ((avxvector_half_v<bsz::b256, cdouble>), 2U);
        EXPECT_EQ((hsz_v<bsz::b256, cdouble>), 2U);
        EXPECT_EQ((tsz_v<bsz::b256, cdouble>), 2U);
    }
#else
    void kt_types_512()
    {
        /*
         * These bsz::b512 bit tests check that the correct data type
         * and packed sizes are "returned".
         */
        // bsz::b512 float
        EXPECT_EQ(typeid(avxvector<bsz::b512, float>::type), typeid(__m512));
        EXPECT_EQ(typeid(avxvector<bsz::b512, float>::half_type), typeid(__m256));
        EXPECT_EQ((avxvector<bsz::b512, float>::p_size), 16U);
        EXPECT_EQ((avxvector<bsz::b512, float>()), 16U);
        EXPECT_EQ((avxvector<bsz::b512, float>::hp_size), 8U);
        EXPECT_EQ((avxvector<bsz::b512, float>::tsz), 16U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b512, float>), typeid(__m512));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b512, float>), typeid(__m256));
        EXPECT_EQ((avxvector_v<bsz::b512, float>), 16U);
        EXPECT_EQ((avxvector<bsz::b512, float>()), 16U);
        EXPECT_EQ((avxvector_half_v<bsz::b512, float>), 8U);
        EXPECT_EQ((hsz_v<bsz::b512, float>), 8U);
        EXPECT_EQ((tsz_v<bsz::b512, float>), 16U);

        // bsz::b512 double
        EXPECT_EQ(typeid(avxvector<bsz::b512, double>::type), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector<bsz::b512, double>::half_type), typeid(__m256d));
        EXPECT_EQ((avxvector<bsz::b512, double>::p_size), 8U);
        EXPECT_EQ((avxvector<bsz::b512, double>()), 8U);
        EXPECT_EQ((avxvector<bsz::b512, double>::hp_size), 4U);
        EXPECT_EQ((avxvector<bsz::b512, double>::tsz), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b512, double>), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b512, double>), typeid(__m256d));
        EXPECT_EQ((avxvector_v<bsz::b512, double>), 8U);
        EXPECT_EQ((avxvector<bsz::b512, double>()), 8U);
        EXPECT_EQ((avxvector_half_v<bsz::b512, double>), 4U);
        EXPECT_EQ((hsz_v<bsz::b512, double>), 4U);
        EXPECT_EQ((tsz_v<bsz::b512, double>), 8U);

        // Integer types test
        // int32_t
        EXPECT_EQ(typeid(avxvector<bsz::b512, int32_t>::type), typeid(__m512i));
        EXPECT_EQ(typeid(avxvector<bsz::b512, int32_t>::half_type), typeid(__m256i));
        EXPECT_EQ((avxvector<bsz::b512, int32_t>::p_size), 16U);
        EXPECT_EQ((avxvector<bsz::b512, int32_t>()), 16U);
        EXPECT_EQ((avxvector<bsz::b512, int32_t>::hp_size), 8U);
        EXPECT_EQ((avxvector<bsz::b512, int32_t>::tsz), 16U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b512, int32_t>), typeid(__m512i));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b512, int32_t>), typeid(__m256i));
        EXPECT_EQ((avxvector_v<bsz::b512, int32_t>), 16U);
        EXPECT_EQ((avxvector<bsz::b512, int32_t>()), 16U);
        EXPECT_EQ((avxvector_half_v<bsz::b512, int32_t>), 8U);
        EXPECT_EQ((hsz_v<bsz::b512, int32_t>), 8U);
        EXPECT_EQ((tsz_v<bsz::b512, int32_t>), 16U);

        // bsz::b512 int64_t
        EXPECT_EQ(typeid(avxvector<bsz::b512, int64_t>::type), typeid(__m512i));
        EXPECT_EQ(typeid(avxvector<bsz::b512, int64_t>::half_type), typeid(__m256i));
        EXPECT_EQ((avxvector<bsz::b512, int64_t>::p_size), 8U);
        EXPECT_EQ((avxvector<bsz::b512, int64_t>()), 8U);
        EXPECT_EQ((avxvector<bsz::b512, int64_t>::hp_size), 4U);
        EXPECT_EQ((avxvector<bsz::b512, int64_t>::tsz), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b512, int64_t>), typeid(__m512i));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b512, int64_t>), typeid(__m256i));
        EXPECT_EQ((avxvector_v<bsz::b512, int64_t>), 8U);
        EXPECT_EQ((avxvector<bsz::b512, int64_t>()), 8U);
        EXPECT_EQ((avxvector_half_v<bsz::b512, int64_t>), 4U);
        EXPECT_EQ((hsz_v<bsz::b512, int64_t>), 4U);
        EXPECT_EQ((tsz_v<bsz::b512, int64_t>), 8U);
    }

    void kt_ctypes_512()
    {
        /*
        * These bsz::b512 bit tests check that the correct complex data type
        * and packed sizes are "returned".
        */
        // bsz::b512 cfloat
        EXPECT_EQ(typeid(avxvector<bsz::b512, cfloat>::type), typeid(__m512));
        EXPECT_EQ(typeid(avxvector<bsz::b512, cfloat>::half_type), typeid(__m256));
        EXPECT_EQ((avxvector<bsz::b512, cfloat>::p_size), 16U);
        EXPECT_EQ((avxvector<bsz::b512, cfloat>()), 16U);
        EXPECT_EQ((avxvector<bsz::b512, cfloat>::hp_size), 8U);
        EXPECT_EQ((avxvector<bsz::b512, cfloat>::tsz), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b512, cfloat>), typeid(__m512));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b512, cfloat>), typeid(__m256));
        EXPECT_EQ((avxvector_v<bsz::b512, cfloat>), 16U);
        EXPECT_EQ((avxvector<bsz::b512, cfloat>()), 16U);
        EXPECT_EQ((avxvector_half_v<bsz::b512, cfloat>), 8U);
        EXPECT_EQ((hsz_v<bsz::b512, cfloat>), 8U);
        EXPECT_EQ((tsz_v<bsz::b512, cfloat>), 8U);

        // bsz::b512 cdouble
        EXPECT_EQ(typeid(avxvector<bsz::b512, cdouble>::type), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector<bsz::b512, cdouble>::half_type), typeid(__m256d));
        EXPECT_EQ((avxvector<bsz::b512, cdouble>::p_size), 8U);
        EXPECT_EQ((avxvector<bsz::b512, cdouble>()), 8U);
        EXPECT_EQ((avxvector<bsz::b512, cdouble>::hp_size), 4U);
        EXPECT_EQ((avxvector<bsz::b512, cdouble>::tsz), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b512, cdouble>), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b512, cdouble>), typeid(__m256d));
        EXPECT_EQ((avxvector_v<bsz::b512, cdouble>), 8U);
        EXPECT_EQ((avxvector<bsz::b512, cdouble>()), 8U);
        EXPECT_EQ((avxvector_half_v<bsz::b512, cdouble>), 4U);
        EXPECT_EQ((hsz_v<bsz::b512, cdouble>), 4U);
        EXPECT_EQ((tsz_v<bsz::b512, cdouble>), 4U);
    }
#endif

    template <bsz SZ, typename SUF>
    void kt_loadu_p_test()
    {
        auto       data = D.get_data<SUF>();
        const auto sz   = tsz_v<SZ, SUF>;

        SUF                 *c;
        avxvector_t<SZ, SUF> w;
        w = kt_loadu_p<SZ, SUF>(data);
        c = reinterpret_cast<SUF *>(&w);

        EXPECT_EQ_VEC(sz, c, data);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_load_p_test()
    {
        const auto sz    = tsz_v<SZ, SUF>;
        size_t     align = 32; // Default alignment for AVX is 32 bytes

        if constexpr(SZ == bsz::b512) // 512 instruction expect the data to be 64 byte aligned
            align = 64;
        else if constexpr(SZ == bsz::b128) // 128 instruction expect the data to be 16 byte aligned
            align = 16;

        SUF *aligned_data
            = static_cast<SUF *>(::operator new(sizeof(SUF) * sz, std::align_val_t{align}));

        // Initialize the aligned memory
        for(size_t i = 0; i < sz; ++i)
        {
            aligned_data[i] = D.get_data<SUF>()[i];
        }

        SUF                 *c;
        avxvector_t<SZ, SUF> w;
        w = kt_load_p<SZ, SUF>(aligned_data);
        c = reinterpret_cast<SUF *>(&w);

        EXPECT_EQ_VEC(sz, c, aligned_data);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;

        ::operator delete(aligned_data, std::align_val_t{align});
    }

    template <bsz SZ, typename SUF>
    void kt_setzero_p_test()
    {
        const size_t sz = tsz_v<SZ, SUF>;
        const SUF    zero_v[sz]{};

        auto v = kt_setzero_p<SZ, SUF>();
        auto c = reinterpret_cast<SUF *>(&v);

        EXPECT_EQ_VEC(sz, c, zero_v);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_set1_p_test()
    {
        const auto sz = tsz_v<SZ, SUF>;

        SUF ref = D.get_data<SUF>()[0];
        SUF ref_v[sz];

        for(size_t itr = 0; itr < sz; ++itr)
        {
            ref_v[itr] = ref;
        }

        SUF                 *c;
        avxvector_t<SZ, SUF> w;
        w = kt_set1_p<SZ, SUF>(ref);
        c = reinterpret_cast<SUF *>(&w);

        EXPECT_EQ_VEC(sz, c, ref_v);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_add_p_test()
    {
        const size_t         sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s, as, bs;
        SUF                  refs[sz];

        SUF        n    = D.get_data<SUF>()[0];
        const SUF *data = (D.get_data<SUF>() + 4);

        as = kt_loadu_p<SZ, SUF>(data);
        bs = kt_set1_p<SZ, SUF>(n);
        s  = kt_add_p<SZ, SUF>(as, bs);

        for(size_t i = 0; i < sz; i++)
        {
            refs[i] = data[i] + n;
        }

        auto res_ptr = reinterpret_cast<SUF *>(&s);
        expect_eq_vec(sz, res_ptr, refs);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_sub_p_test()
    {
        const size_t         sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s, as, bs;
        SUF                  refs[sz];

        SUF        n    = D.get_data<SUF>()[1];
        const SUF *data = (D.get_data<SUF>() + 3);

        as = kt_loadu_p<SZ, SUF>(data);
        bs = kt_set1_p<SZ, SUF>(n);
        s  = kt_sub_p<SZ, SUF>(as, bs);
        for(size_t i = 0; i < sz; i++)
        {
            refs[i] = data[i] - n;
        }

        auto res_ptr = reinterpret_cast<SUF *>(&s);
        expect_eq_vec(sz, res_ptr, refs);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_mul_p_test()
    {
        size_t               sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s, as, bs;
        SUF                  refs[sz];

        SUF        n    = D.get_data<SUF>()[2];
        const SUF *data = (D.get_data<SUF>() + 2);

        as = kt_loadu_p<SZ, SUF>(data);
        bs = kt_set1_p<SZ, SUF>(n);
        s  = kt_mul_p<SZ, SUF>(as, bs);
        for(size_t i = 0; i < sz; i++)
        {
            refs[i] = data[i] * n;
        }

        auto res_ptr = reinterpret_cast<SUF *>(&s);
        expect_eq_vec(sz, res_ptr, refs);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_fmadd_p_test()
    {
        size_t               sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s, as, bs;
        SUF                  refs[sz];

        const SUF    *data = D.get_data<SUF>();
        const size_t *idx  = D.map;

        as = kt_loadu_p<SZ, SUF>(data + 2);
        bs = kt_set_p<SZ, SUF>(data, idx);
        s  = kt_set_p<SZ, SUF>(data, idx + 4);
        s  = kt_fmadd_p<SZ, SUF>(as, bs, s);

        for(size_t i = 0; i < sz; i++)
        {
            refs[i] = data[i + 2] * data[idx[i]] + data[idx[4 + i]];
        }

        auto res_ptr = reinterpret_cast<SUF *>(&s);
        expect_eq_vec(sz, res_ptr, refs);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_fmsub_p_test()
    {
        size_t               sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s, as, bs;
        SUF                  refs[sz];

        const SUF    *data = D.get_data<SUF>();
        const size_t *idx  = D.map;

        as = kt_loadu_p<SZ, SUF>(data);
        bs = kt_set_p<SZ, SUF>(data, idx);
        s  = kt_set_p<SZ, SUF>(data, idx + 4);
        s  = kt_fmsub_p<SZ, SUF>(as, bs, s);

        for(size_t i = 0; i < sz; i++)
        {
            refs[i] = data[i] * data[idx[i]] - data[idx[4 + i]];
        }

        auto res_ptr = reinterpret_cast<SUF *>(&s);
        expect_eq_vec(sz, res_ptr, refs);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_set_p_test()
    {
        const size_t         sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s;
        SUF                  refs[sz];

        const SUF    *data = D.get_data<SUF>();
        const size_t *idx  = D.map;

        s = kt_set_p<SZ, SUF>(data, idx + 2);

        for(size_t i = 0; i < sz; i++)
        {
            refs[i] = data[idx[2 + i]];
        }

        auto res_ptr = reinterpret_cast<SUF *>(&s);
        EXPECT_EQ_VEC(sz, res_ptr, refs);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    /*
     * Test "maskz_set" zero-masked version of load_u (aka maskz_loadu) to load
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors.
     * K specifies the how much zero-padding is done
     * at the end of the vector, so K = 1 for 256d vector would load [0 x3 x2 x1]
     */
#define kt_maskz_set_p_param_dir(SZ, SUF, S, EXT, K, B)                                 \
    {                                                                                   \
        const size_t         n = tsz_v<SZ, SUF>;                                        \
        avxvector_t<SZ, SUF> v = kt_maskz_set_p<SZ, SUF, EXT, K>(D.v##S, (size_t)B##U); \
        SUF                  ve[n];                                                     \
        SUF                 *pv = nullptr;                                              \
        for(size_t i = 0; i < n; i++)                                                   \
        {                                                                               \
            if(int(i - K) >= 0)                                                         \
                ve[i] = 0;                                                              \
            else                                                                        \
                ve[i] = D.v##S[B + i];                                                  \
        }                                                                               \
        pv = reinterpret_cast<SUF *>(&v);                                               \
        if constexpr(std::is_same_v<SUF, cdouble> || std::is_same_v<SUF, cfloat>)       \
        {                                                                               \
            EXPECT_COMPLEX_EQ_VEC(n, pv, ve);                                           \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            EXPECT_EQ_VEC(n, pv, ve);                                                   \
        }                                                                               \
    }
    /*
     * Test "maskz_set" zero-masked version of "set" to indirectly load
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors.
     * K specifies the how much zero-padding is done
     * at the end of the vector, so K = 1 for 256d vector would load
     * [0 x[m[3]] x[m[2]] x[m[1]]
     * Note: for indirect addressing EXT can be ANY
     */
#define kt_maskz_set_p_param_indir(SZ, SUF, S, EXT, K, B)                            \
    {                                                                                \
        const size_t n = tsz_v<SZ, SUF>;                                             \
                                                                                     \
        avxvector_t<SZ, SUF> v = kt_maskz_set_p<SZ, SUF, EXT, K>(D.v##S, &D.map[B]); \
        SUF                  ve[n];                                                  \
        SUF                 *pv = nullptr;                                           \
        for(size_t i = 0; i < n; i++)                                                \
        {                                                                            \
            if(int(i - K) >= 0)                                                      \
                ve[i] = 0;                                                           \
            else                                                                     \
                ve[i] = D.v##S[D.map[B + i]];                                        \
        }                                                                            \
        pv = reinterpret_cast<SUF *>(&v);                                            \
        if constexpr(std::is_same_v<SUF, cdouble> || std::is_same_v<SUF, cfloat>)    \
        {                                                                            \
            EXPECT_COMPLEX_EQ_VEC(n, pv, ve);                                        \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            EXPECT_EQ_VEC(n, v, ve);                                                 \
        }                                                                            \
    }

    /*
     * Test out of bound access in mask indirect access
     */
#define kt_maskz_set_p_param_indir_out_of_bound(SZ, SUF, S, EXT)                             \
    {                                                                                        \
        const size_t         n        = tsz_v<SZ, SUF>;                                      \
        size_t               mock_idx = 1000000;                                             \
        avxvector_t<SZ, SUF> v        = kt_maskz_set_p<SZ, SUF, EXT, -1>(D.v##S, &mock_idx); \
        SUF                  ve[n];                                                          \
        SUF                 *pv = nullptr;                                                   \
        for(size_t i = 0; i < n; i++)                                                        \
                                                                                             \
            ve[i] = 0;                                                                       \
                                                                                             \
        pv = reinterpret_cast<SUF *>(&v);                                                    \
        if constexpr(std::is_same_v<SUF, cdouble> || std::is_same_v<SUF, cfloat>)            \
        {                                                                                    \
            EXPECT_COMPLEX_EQ_VEC(n, pv, ve);                                                \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            EXPECT_EQ_VEC(n, v, ve);                                                         \
        }                                                                                    \
    }

#ifdef KT_AVX2_BUILD
    void kt_maskz_set_p_128_avx()
    {
        // ===================================================
        // DIRECT (can have any extension other than AVX512VL)
        //====================================================
        // bsz::b128/float -> 4
        kt_maskz_set_p_param_dir(bsz::b128, float, s, AVX, 1, 1);
        kt_maskz_set_p_param_dir(bsz::b128, float, s, AVX2, 2, 0);
        kt_maskz_set_p_param_dir(bsz::b128, float, s, AVX512F, 3, 2);
        kt_maskz_set_p_param_dir(bsz::b128, float, s, AVX, 4, 1);

        // This must trigger a warning under AVX512F (bsz::b128 bit __mask8)
        // kt_maskz_set_p_param_dir(bsz::b128, float, s, AVX, 9);
        // Test to ensure the memory is not touched
        kt_maskz_set_p_param_dir(bsz::b128, float, s, AVX, -1, 10000000);

        // bsz::b128 double -> 2
        kt_maskz_set_p_param_dir(bsz::b128, double, d, AVX512DQ, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b128, double, d, AVX512F, 2, 1);
        // This also triggers a warning
        // kt_maskz_set_p_param_dir(bsz::b128, double, d, AVX, 5);
        // Test to ensure the memory is not touched
        kt_maskz_set_p_param_dir(bsz::b128, double, d, AVX, -1, 10000000);

        // bsz::b128 cfloat -> 2
        kt_maskz_set_p_param_dir(bsz::b128, cfloat, c, AVX2, 1, 1);
        kt_maskz_set_p_param_dir(bsz::b128, cfloat, c, AVX, 2, 0);

        // Test to ensure the memory is not touched
        kt_maskz_set_p_param_dir(bsz::b128, cfloat, c, AVX512F, -1, 10000000);

        // bsz::b128 cdouble -> 1
        kt_maskz_set_p_param_dir(bsz::b128, cdouble, z, AVX, 1, 0);

        // Test to ensure the memory is not touched
        kt_maskz_set_p_param_dir(bsz::b128, cdouble, z, AVX2, -1, 10000000);

        // =================================
        // INDIRECT (can have any extension)
        // =================================
        kt_maskz_set_p_param_indir(bsz::b128, float, s, AVX, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b128, float, s, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(bsz::b128, float, s, AVX512VL, 3, 0);
        kt_maskz_set_p_param_indir(bsz::b128, float, s, AVX, 4, 1);

        kt_maskz_set_p_param_indir(bsz::b128, double, d, AVX, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b128, double, d, AVX512DQ, 2, 1);

        kt_maskz_set_p_param_indir(bsz::b128, cfloat, c, AVX, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b128, cfloat, c, AVX512F, 2, 1);

        kt_maskz_set_p_param_indir(bsz::b128, cdouble, z, AVX, 1, 1);

        // Out of bound access tests
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b128, float, s, AVX);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b128, double, d, AVX);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b128, cfloat, c, AVX);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b128, cdouble, z, AVX);
    }

    void kt_maskz_set_p_256_avx()
    {
        // ===============
        // DIRECT
        //================
        // bsz::b256/float -> 8
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 1, 1);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 2, 0);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 3, 2);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 4, 1);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 5, 0);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 6, 1);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 7, 0);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 8, 1);
        // This must trigger a warning under AVX512F (bsz::b256 bit __mask8)
        // kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, 9);
        // Test to ensure the memory is not touched
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX, -1, 10000000);

        // bsz::b256 double -> 4
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 2, 1);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 3, 3);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 4, 2);
        // This also triggers a warning
        // kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 5);
        // Test to ensure the memory is not touched
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, -1, 10000000);

        // bsz::b256 cfloat -> 4
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, 1, 1);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, 2, 0);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, 3, 4);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, 4, 0);

        // Test to ensure the memory is not touched
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, -1, 10000000);

        // bsz::b256 cdouble -> 2
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX, 2, 4);
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX, 2, 0);

        // Test to ensure the memory is not touched
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX, -1, 10000000);

        // =================================
        // INDIRECT (can have any extension)
        // =================================
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX512VL, 3, 0);
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX, 4, 1);
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX, 5, 0);
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX, 6, 2);
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX, 7, 0);
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX, 8, 0);
        kt_maskz_set_p_param_indir(bsz::b256, float, s, AVX, 9, 0);

        kt_maskz_set_p_param_indir(bsz::b256, double, d, AVX, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b256, double, d, AVX512DQ, 0, 1);
        kt_maskz_set_p_param_indir(bsz::b256, double, d, AVX, 3, 0);
        kt_maskz_set_p_param_indir(bsz::b256, double, d, AVX, 4, 3);
        kt_maskz_set_p_param_indir(bsz::b256, double, d, AVX, 5, 0);

        kt_maskz_set_p_param_indir(bsz::b256, cfloat, c, AVX, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b256, cfloat, c, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(bsz::b256, cfloat, c, AVX512VL, 0, 0);
        kt_maskz_set_p_param_indir(bsz::b256, cfloat, c, AVX, 4, 1);
        kt_maskz_set_p_param_indir(bsz::b256, cfloat, c, AVX, 3, 3);

        kt_maskz_set_p_param_indir(bsz::b256, cdouble, z, AVX, 1, 1);
        kt_maskz_set_p_param_indir(bsz::b256, cdouble, z, AVX512DQ, 2, 0);
        kt_maskz_set_p_param_indir(bsz::b256, cdouble, z, AVX, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b256, cdouble, z, AVX, 2, 2);
        kt_maskz_set_p_param_indir(bsz::b256, cdouble, z, AVX512DQ, 0, 0);

        // Out of bound access tests
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b256, float, s, AVX2);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b256, double, d, AVX2);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b256, cfloat, c, AVX2);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b256, cdouble, z, AVX2);
    }
#else
    void kt_maskz_set_p_256_AVX512vl()
    {
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX512VL, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX512VL, 2, 5);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX512VL, 3, 1);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX512VL, 4, 0);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX512VL, 5, 1);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX512VL, 6, 2);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX512VL, 7, 1);
        kt_maskz_set_p_param_dir(bsz::b256, float, s, AVX512VL, 8, 0);

        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX512VL, 1, 1);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX512VL, 2, 0);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX512VL, 3, 1);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX512VL, 4, 2);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX512VL, 5, 1);

        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX512VL, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX512VL, 2, 5);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX512VL, 3, 2);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX512VL, 4, 0);

        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX512VL, 1, 1);
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX512VL, 2, 0);
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX512VL, 3, 4);
    }

    void kt_maskz_set_p_512_AVX512f()
    {
        // Direct
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 2, 0);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 3, 1);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 4, 1);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 6, 1);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 7, 1);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 8, 1);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 9, 1);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 9, 1);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 10, 0);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 11, 0);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 12, 0);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 13, 0);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 14, 0);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 15, 0);
        kt_maskz_set_p_param_dir(bsz::b512, float, s, AVX512F, 16, 0);

        kt_maskz_set_p_param_dir(bsz::b512, double, d, AVX512F, 1, 3);
        kt_maskz_set_p_param_dir(bsz::b512, double, d, AVX512F, 2, 2);
        kt_maskz_set_p_param_dir(bsz::b512, double, d, AVX512F, 3, 4);
        kt_maskz_set_p_param_dir(bsz::b512, double, d, AVX512F, 4, 0);
        kt_maskz_set_p_param_dir(bsz::b512, double, d, AVX512F, 5, 1);
        kt_maskz_set_p_param_dir(bsz::b512, double, d, AVX512F, 6, 0);
        kt_maskz_set_p_param_dir(bsz::b512, double, d, AVX512F, 7, 0);
        kt_maskz_set_p_param_dir(bsz::b512, double, d, AVX512F, 8, 0);

        kt_maskz_set_p_param_dir(bsz::b512, cfloat, c, AVX512F, 1, 5);
        kt_maskz_set_p_param_dir(bsz::b512, cfloat, c, AVX512F, 2, 0);
        kt_maskz_set_p_param_dir(bsz::b512, cfloat, c, AVX512F, 3, 2);
        kt_maskz_set_p_param_dir(bsz::b512, cfloat, c, AVX512F, 4, 0);

        kt_maskz_set_p_param_dir(bsz::b512, cdouble, z, AVX512F, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b512, cdouble, z, AVX512F, 2, 2);
        kt_maskz_set_p_param_dir(bsz::b512, cdouble, z, AVX512F, 3, 2);

        // Indirect
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 1, 1);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 3, 3);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 4, 0);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 5, 1);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 6, 0);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 7, 0);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 8, 3);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 9, 0);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 10, 0);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 11, 5);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 12, 0);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 13, 2);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 14, 0);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 15, 0);
        kt_maskz_set_p_param_indir(bsz::b512, float, s, AVX512F, 16, 0);

        kt_maskz_set_p_param_indir(bsz::b512, double, d, AVX512F, 1, 5);
        kt_maskz_set_p_param_indir(bsz::b512, double, d, AVX512F, 2, 3);
        kt_maskz_set_p_param_indir(bsz::b512, double, d, AVX512F, 3, 0);
        kt_maskz_set_p_param_indir(bsz::b512, double, d, AVX512F, 4, 1);
        kt_maskz_set_p_param_indir(bsz::b512, double, d, AVX512F, 5, 2);
        kt_maskz_set_p_param_indir(bsz::b512, double, d, AVX512F, 6, 4);
        kt_maskz_set_p_param_indir(bsz::b512, double, d, AVX512F, 7, 2);
        kt_maskz_set_p_param_indir(bsz::b512, double, d, AVX512F, 8, 0);

        kt_maskz_set_p_param_indir(bsz::b512, cfloat, c, AVX512F, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b512, cfloat, c, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(bsz::b512, cfloat, c, AVX512F, 3, 1);
        kt_maskz_set_p_param_indir(bsz::b512, cfloat, c, AVX512F, 4, 3);
        kt_maskz_set_p_param_indir(bsz::b512, cfloat, c, AVX512F, 5, 1);
        kt_maskz_set_p_param_indir(bsz::b512, cfloat, c, AVX512F, 6, 0);
        kt_maskz_set_p_param_indir(bsz::b512, cfloat, c, AVX512F, 7, 1);
        kt_maskz_set_p_param_indir(bsz::b512, cfloat, c, AVX512F, 8, 2);

        kt_maskz_set_p_param_indir(bsz::b512, cdouble, z, AVX512F, 1, 0);
        kt_maskz_set_p_param_indir(bsz::b512, cdouble, z, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(bsz::b512, cdouble, z, AVX512F, 3, 0);
        kt_maskz_set_p_param_indir(bsz::b512, cdouble, z, AVX512F, 4, 3);

        // Out of bound access tests
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b512, float, s, AVX512F);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b512, double, d, AVX512F);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b512, cfloat, c, AVX512F);
        kt_maskz_set_p_param_indir_out_of_bound(bsz::b512, cdouble, z, AVX512F);
    }
#endif

    template <bsz SZ, typename SUF>
    void kt_hsum_p_test()
    {
        const size_t         sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> vs;
        SUF                  sums{}, refs{};

        const SUF *data = D.get_data<SUF>();

        vs   = kt_loadu_p<SZ, SUF>(data);
        sums = kt_hsum_p<SZ, SUF>(vs);

        for(size_t i = 0; i < sz; i++)
        {
            refs += data[i];
        }

        expect_eq<SUF>(sums, refs);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_conj_p_test()
    {

        constexpr size_t     sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> vs;
        SUF                  refs[sz];

        // Float
        vs = kt_loadu_p<SZ, SUF>(D.get_data<SUF>());
        vs = kt_conj_p<SZ, SUF>(vs);

        for(size_t i = 0; i < sz; i++)
        {
            if constexpr(kt_type_is_real<SUF>())
                refs[i] = D.get_data<SUF>()[i];
            else
                refs[i] = std::conj(D.get_data<SUF>()[i]);
        }

        auto *pv = reinterpret_cast<SUF *>(&vs);
        expect_eq_vec(sz, refs, pv);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_dot_p_test()
    {
        const size_t sz = tsz_v<SZ, SUF>;

        const SUF *data = D.get_data<SUF>();

        avxvector_t<SZ, SUF> s1   = kt_loadu_p<SZ, SUF>(data);
        avxvector_t<SZ, SUF> s2   = kt_loadu_p<SZ, SUF>(data + 1);
        SUF                  refs = 0.0f;

        SUF sdot = kt_dot_p<SZ, SUF>(s1, s2);

        for(size_t i = 0; i < sz; i++)
            refs += data[0 + i] * data[1 + i];

        expect_eq<SUF>(sdot, refs);
    }

    template <bsz SZ, typename SUF>
    void kt_cdot_p_test()
    {
        size_t sz = tsz_v<SZ, SUF>;

        const SUF *data = D.get_data<SUF>();

        avxvector_t<SZ, SUF> s1 = kt_loadu_p<SZ, SUF>(data);
        avxvector_t<SZ, SUF> s2 = kt_loadu_p<SZ, SUF>(data + 1);
        SUF                  refs{};

        SUF sdot = kt_cdot_p<SZ, SUF>(s1, s2);
        for(size_t i = 0; i < sz; i++)
        {
            if constexpr(kt_type_is_real<SUF>())
                refs += data[0 + i] * data[1 + i];
            else
                refs += data[0 + i] * std::conj(data[1 + i]);
        }

        expect_eq(refs, sdot);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_storeu_p_test()
    {
        constexpr size_t     sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s;

        // Smart pointer to manage memory
        // Used to dynamic memory allocation to suppress a potential compiler bug
        std::unique_ptr<SUF[]> refs(new SUF[sz]);
        std::unique_ptr<SUF[]> vss(new SUF[sz]);

        const SUF *data = D.get_data<SUF>();

        s = kt_loadu_p<SZ, SUF>(data + 3);
        kt_storeu_p<SZ, SUF>(vss.get(), s);

        for(size_t i = 0; i < sz; i++)
        {
            refs[i] = data[i + 3];
        }

        expect_eq_vec(sz, vss.get(), refs.get());

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_fmadd_B_test()
    {
        // In case of b128 and complex, we use the kt_fmadd_p_test
        // to test the kt_fmadd_B, because the latter is not implemented
        if constexpr(SZ == bsz::b128 && !kt_type_is_real<SUF>())
        {
            return kt_fmadd_p_test<SZ, SUF>();
        }

        // Blocked fmadd_B tests work differently for real and complex types
        if constexpr(kt_type_is_real<SUF>())
        {
            size_t               sz = tsz_v<SZ, SUF>;
            avxvector_t<SZ, SUF> s, as, bs, s_;
            SUF                  refs[sz];

            const SUF    *data = D.get_data<SUF>();
            const size_t *idx  = D.map;

            std::vector<SUF> refs_(sz, data[2]);

            as = kt_loadu_p<SZ, SUF>(data);
            bs = kt_set_p<SZ, SUF>(data, idx);
            s  = kt_set_p<SZ, SUF>(data, idx + 4);
            s_ = kt_set1_p<SZ, SUF>(data[2]);

            kt_fmadd_B<SZ, SUF>(as, bs, s, s_);

            for(size_t i = 0; i < sz; i++)
            {
                refs[i] = data[i] * data[idx[i]] + data[idx[4 + i]];
            }

            auto res_ptr = reinterpret_cast<SUF *>(&s);
            expect_eq_vec(sz, res_ptr, refs);

            if(::testing::Test::HasFailure())
                std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
        }
        else
        {
            constexpr size_t     sz = tsz_v<SZ, SUF>;
            avxvector_t<SZ, SUF> scr, sci, ac, bc, cc, s1;
            SUF                  refc[sz], refsci[sz], refscr[sz];
            const SUF           *vc1  = D.get_data<SUF>();
            const SUF           *vc2  = D.get_data<SUF>() + 1;
            const SUF           *data = D.get_data<SUF>();

            ac  = kt_loadu_p<SZ, SUF>(vc1 + 2);
            bc  = kt_loadu_p<SZ, SUF>(vc1);
            cc  = kt_loadu_p<SZ, SUF>(vc2);
            s1  = kt_set1_p<SZ, SUF>({1.0f, 1.0f});
            scr = kt_setzero_p<SZ, SUF>();
            sci = kt_setzero_p<SZ, SUF>();
            kt_fmadd_B<SZ, SUF>(s1, cc, scr, sci);

            for(size_t i = 0; i < sz; i++)
            {
                refc[i] = std::complex(vc2[i].real(), vc2[i].imag());
            }

            SUF *pc = reinterpret_cast<SUF *>(&scr);
            expect_eq_vec(sz, pc, refc);

            for(size_t i = 0; i < sz; i++)
            {
                refc[i] = std::complex(vc2[i].imag(), vc2[i].real());
            }

            pc = reinterpret_cast<SUF *>(&sci);
            expect_eq_vec(sz, pc, refc);

            for(size_t i = 0; i < sz; i++)
            {
                refscr[i] = {data[2 + i].real() * vc1[i].real() + scr[2 * i],
                             data[2 + i].imag() * vc1[i].imag() + scr[2 * i + 1]};
            }
            for(size_t i = 0; i < sz; i++)
            {
                refsci[i] = {data[2 + i].real() * vc1[i].imag() + sci[2 * i],
                             data[2 + i].imag() * vc1[i].real() + sci[2 * i + 1]};
            }
            kt_fmadd_B<SZ, SUF>(ac, bc, scr, sci);

            pc = reinterpret_cast<SUF *>(&scr);
            expect_eq_vec(sz, pc, refscr);

            pc = reinterpret_cast<SUF *>(&sci);
            expect_eq_vec(sz, pc, refsci);
        }
    }

    template <bsz SZ, typename SUF>
    void kt_hsum_B_test()
    {
        // In case of b128 and complex, we use the kt_hsum_p_test
        // to test the kt_hsum_B, because the latter is not implemented
        if constexpr(SZ == bsz::b128 && !kt_type_is_real<SUF>())
        {
            return kt_hsum_p_test<SZ, SUF>();
        }

        if constexpr(kt_type_is_real<SUF>())
        {
            const size_t         sz = tsz_v<SZ, SUF>;
            avxvector_t<SZ, SUF> as, bs, s, s_;
            SUF                  sums{}, refs{};

            const SUF    *data = D.get_data<SUF>();
            const size_t *idx  = D.map;

            const SUF n = data[2];

            as = kt_loadu_p<SZ, SUF>(data);
            bs = kt_set_p<SZ, SUF>(data, idx + 4);
            s  = kt_set1_p<SZ, SUF>(n);
            s_ = kt_setzero_p<SZ, SUF>();

            kt_fmadd_B<SZ, SUF>(as, bs, s, s_);

            sums = kt_hsum_B<SZ, SUF>(s, s_);

            for(size_t i = 0; i < sz; i++)
            {
                refs += data[i] * data[idx[4 + i]] + n;
            }

            expect_eq(sums, refs);
        }
        else
        {
            const size_t         sz = tsz_v<SZ, SUF>;
            avxvector_t<SZ, SUF> az, bz, z, z_;
            SUF                  sumz{}, refz{};

            const SUF    *data = D.get_data<SUF>();
            const size_t *idx  = D.map;

            az = kt_loadu_p<SZ, SUF>(data);
            bz = kt_set_p<SZ, SUF>(data, idx + 3);

            z  = kt_setzero_p<SZ, SUF>();
            z_ = kt_setzero_p<SZ, SUF>();
            // (z, z_) <- az * bz
            kt_fmadd_B<SZ, SUF>(bz, az, z, z_);
            sumz = kt_hsum_B<SZ, SUF>(z, z_);
            for(size_t i = 0; i < sz; i++)
            {
                // Test also commutativity
                //      bz                 * az
                refz += data[idx[3 + i]] * data[i];
            }
            expect_eq(sumz, refz);
        }

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_max_p_test()
    {
        size_t sz = tsz_v<SZ, SUF>;

        const SUF *data = D.get_data<SUF>();

        avxvector_t<SZ, SUF> s1 = kt_loadu_p<SZ, SUF>(data);
        avxvector_t<SZ, SUF> s2 = kt_loadu_p<SZ, SUF>(data + 2);

        SUF ref_ress[sz];

        avxvector_t<SZ, SUF> ress = kt_max_p<SZ, SUF>(s1, s2);

        for(size_t i = 0; i < sz; i++)
        {
            ref_ress[i] = (std::max)(data[0 + i], data[2 + i]);
        }

        auto *pv = reinterpret_cast<SUF *>(&ress);

        expect_eq_vec(sz, ref_ress, pv);
    }

    template <bsz SZ, typename SUF>
    void kt_div_p_test()
    {
        size_t               sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s, as, bs;
        SUF                  refs[sz];

        SUF        n    = D.get_data<SUF>()[0];
        const SUF *data = D.get_data<SUF>();

        as = kt_loadu_p<SZ, SUF>(data + 1);
        bs = kt_set1_p<SZ, SUF>(n);
        s  = kt_div_p<SZ, SUF>(as, bs);
        for(size_t i = 0; i < sz; i++)
        {
            refs[i] = data[i + 1] / n;
        }

        auto res_ptr = reinterpret_cast<SUF *>(&s);
        expect_eq_vec<SUF>(sz, res_ptr, refs);

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_pow2_p_test()
    {
        using base_t = typename kt_dt<SUF>::base_type;

        size_t               sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> s, as;

        const SUF *data = D.get_data<SUF>();

        as = kt_loadu_p<SZ, SUF>(data);
        s  = kt_pow2_p<SZ, SUF>(as);

        if constexpr(kt_type_is_real<SUF>())
        {
            SUF refs[sz];

            for(size_t i = 0; i < sz; i++)
            {
                refs[i] = data[i] * data[i];
            }

            auto res_ptr = reinterpret_cast<SUF *>(&s);
            expect_eq_vec<SUF>(sz, res_ptr, refs);
        }
        else
        {
            // The comparison is done on the base type
            base_t refs[sz * 2];

            // In case of complex number, the real and complex
            // parts are set to the result of pow2.
            for(size_t i = 0; i < sz; i++)
            {
                base_t temp = (data[i].real() * data[i].real()) + (data[i].imag() * data[i].imag());

                refs[(2 * i)]     = temp;
                refs[(2 * i) + 1] = temp;
            }

            auto res_ptr = reinterpret_cast<base_t *>(&s);
            expect_eq_vec<base_t>(sz * 2, res_ptr, refs);
        }

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }

    template <bsz SZ, typename SUF>
    void kt_scatter_p_test()
    {
        constexpr size_t     sz = tsz_v<SZ, SUF>;
        avxvector_t<SZ, SUF> v;
        const SUF           *data = D.get_data<SUF>();
        const size_t        *idx  = D.map;

        size_t max_idx = 0;

        // Get the maximum index from idx array
        // This maximum index is used to size the output array
        for(size_t i = 0; i < sz; ++i)
        {
            max_idx = (std::max)(max_idx, *(idx + i));
        }

        // Output and reference arrays
        // 3 output arrays to test fused add, sub and no op
        // Size of the array is max_idx + 1 to accommodate the maximum index element
        std::vector<std::vector<SUF>> out(3, std::vector<SUF>(max_idx + 1, 0)),
            ref(3, std::vector<SUF>(max_idx + 1, 0));

        // Load vector from data
        v = kt_loadu_p<SZ, SUF>(data);

        // Scatter to out using idx
        kt_scatter_p<SZ, SUF, fused_op::ADD>(v, out[0].data(), idx);
        kt_scatter_p<SZ, SUF, fused_op::SUB>(v, out[1].data(), idx);
        kt_scatter_p<SZ, SUF>(v, out[2].data(), idx);

        // Reference: out[idx[i]] (op)= data[i]
        // Reference scatter with fused add and sub
        // Copy the contents of the vector (data) to the reference
        for(size_t i = 0; i < sz; i++)
        {
            ref[0][idx[i]] += data[i];
            ref[1][idx[i]] -= data[i];
            ref[2][idx[i]] = data[i];
        }

        expect_eq_vec(ref[0].size(), out[0].data(), ref[0].data());
        expect_eq_vec(ref[1].size(), out[1].data(), ref[1].data());
        expect_eq_vec(ref[2].size(), out[2].data(), ref[2].data());

        if(::testing::Test::HasFailure())
            std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
    }
}

// Test instantiation macros for real types
#define KT_TEST_INSTANTIATE_FOR_REAL(func, SZ) \
    template void func<SZ, float>();           \
    template void func<SZ, double>();

// Test instantiation macros for complex types
#define KT_TEST_INSTANTIATE_FOR_COMPLEX(func, SZ) \
    template void func<SZ, cfloat>();             \
    template void func<SZ, cdouble>();

// Test instantiation macros for all data types
#define KT_TEST_INSTANTIATE_FOR_ALL_TYPES(func, SZ) \
    KT_TEST_INSTANTIATE_FOR_REAL(func, SZ)          \
    KT_TEST_INSTANTIATE_FOR_COMPLEX(func, SZ)

#define KT_TEST_INSTANTIATE_FOR_INT(func, SZ) \
    template void func<SZ, int32_t>();        \
    template void func<SZ, int64_t>();

// Test instantiation macros for all data types during AVX2 build
#ifdef KT_AVX2_BUILD
#define KT_INSTANTIATE_TEST(func)                       \
    KT_TEST_INSTANTIATE_FOR_ALL_TYPES(func, bsz::b128); \
    KT_TEST_INSTANTIATE_FOR_ALL_TYPES(func, get_bsz());

#define KT_INSTANTIATE_TEST_REAL(func)             \
    KT_TEST_INSTANTIATE_FOR_REAL(func, bsz::b128); \
    KT_TEST_INSTANTIATE_FOR_REAL(func, get_bsz());

#define KT_INSTANTIATE_TEST_INT(func)             \
    KT_TEST_INSTANTIATE_FOR_INT(func, bsz::b128); \
    KT_TEST_INSTANTIATE_FOR_INT(func, get_bsz());
#else
// Test instantiation macros for all data types during AVX512 build
#define KT_INSTANTIATE_TEST(func) KT_TEST_INSTANTIATE_FOR_ALL_TYPES(func, get_bsz());

#define KT_INSTANTIATE_TEST_REAL(func) KT_TEST_INSTANTIATE_FOR_REAL(func, get_bsz());

#define KT_INSTANTIATE_TEST_INT(func) KT_TEST_INSTANTIATE_FOR_INT(func, get_bsz());
#endif

KT_INSTANTIATE_TEST(TestsKT::kt_loadu_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_load_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_setzero_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_set1_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_add_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_sub_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_mul_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_fmadd_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_fmsub_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_set_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_hsum_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_conj_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_dot_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_cdot_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_storeu_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_fmadd_B_test);
KT_INSTANTIATE_TEST(TestsKT::kt_hsum_B_test);
KT_INSTANTIATE_TEST(TestsKT::kt_div_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_pow2_p_test);
KT_INSTANTIATE_TEST(TestsKT::kt_scatter_p_test);

// Operations that only support real types
KT_INSTANTIATE_TEST_REAL(TestsKT::kt_max_p_test);

// Operations that support integer types
KT_INSTANTIATE_TEST_INT(TestsKT::kt_loadu_p_test);
KT_INSTANTIATE_TEST_INT(TestsKT::kt_load_p_test);
KT_INSTANTIATE_TEST_INT(TestsKT::kt_setzero_p_test);
KT_INSTANTIATE_TEST_INT(TestsKT::kt_set1_p_test);
KT_INSTANTIATE_TEST_INT(TestsKT::kt_set_p_test);