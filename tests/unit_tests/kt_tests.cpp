/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

// Note: Since it uses KT outside the scope of the library, AVX512 tests will be
// executed only if this file is compiled with appropriate AVX512 flags
#include "common_data_utils.h"
#include "gtest/gtest.h"

using kt_int_t = size_t;

#include "kernel-templates/kernel_templates.hpp"

#include <complex>
#include <iostream>
#include <typeinfo>

using namespace kernel_templates;
using namespace std::complex_literals;

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
        std::complex<float>  vc[16]{ 2.25f -1.5if,  4.25f - 2if,    6.125f -3if,    8.25f -4if,
                                     1.5f  -5if,    3.5f  - 6.25if, 7.75f  -7if,    9.25f -8if,
                                    -2.5f  -1.5if, -3.25f - 2if,   -5.5f   -3if,   -7.25f -4if,
                                    -9.75f -5if,   -2.2f  - 6if,   -4.75f  -7.5if, -6.0f  -8.125if};
        std::complex<double> vz[8]{  1.25  -12i,    0.5   - 21.0i,  0.125  -13.0i,  3.5   -4.5i,
                                     5.25  -8.125i, 8.5   - 6.75i,  9.5    -7.25i,  2.125 -3.0i};
        // clang-format on
    };

    const KTTCommonData D;

    TEST(KT_L0, KT_BASE_T_CHECK)
    {
        EXPECT_TRUE(kt_is_base_t_float<float>());
        EXPECT_TRUE(kt_is_base_t_float<cfloat>());
        EXPECT_FALSE(kt_is_base_t_float<double>());
        EXPECT_FALSE(kt_is_base_t_float<cdouble>());

        EXPECT_TRUE(kt_is_base_t_double<double>());
        EXPECT_TRUE(kt_is_base_t_double<cdouble>());
        EXPECT_FALSE(kt_is_base_t_double<float>());
        EXPECT_FALSE(kt_is_base_t_double<cfloat>());
    }

    TEST(KT_L0, KT_IS_SAME)
    {
        EXPECT_TRUE((kt_is_same<bsz::b256, bsz::b256, double, double>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b256, double, cdouble>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b512, double, double>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b512, float, cfloat>()));
    }

    TEST(KT_L0, KT_TYPES_256)
    {
        /*
         * These bsz::b256 bit tests check that the correct data type
         * and packed sizes are "returned".
         */
        // bsz::b256 float
        EXPECT_EQ(typeid(avxvector<bsz::b256, float>::type), typeid(__m256));
        EXPECT_EQ(typeid(avxvector<bsz::b256, float>::half_type), typeid(__m128));
        EXPECT_EQ((avxvector<bsz::b256, float>::value), 8U);
        EXPECT_EQ((avxvector<bsz::b256, float>()), 8U);
        EXPECT_EQ((avxvector<bsz::b256, float>::half), 4U);
        EXPECT_EQ((avxvector<bsz::b256, float>::count), 8U);

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
        EXPECT_EQ((avxvector<bsz::b256, double>::value), 4U);
        EXPECT_EQ((avxvector<bsz::b256, double>()), 4U);
        EXPECT_EQ((avxvector<bsz::b256, double>::half), 2U);
        EXPECT_EQ((avxvector<bsz::b256, double>::count), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b256, double>), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b256, double>), typeid(__m128d));
        EXPECT_EQ((avxvector_v<bsz::b256, double>), 4U);
        EXPECT_EQ((avxvector<bsz::b256, double>()), 4U);
        EXPECT_EQ((avxvector_half_v<bsz::b256, double>), 2U);
        EXPECT_EQ((hsz_v<bsz::b256, double>), 2U);
        EXPECT_EQ((tsz_v<bsz::b256, double>), 4U);
    }
#ifdef __AVX512F__
    TEST(KT_L0, KT_TYPES_512)
    {
        /*
         * These bsz::b512 bit tests check that the correct data type
         * and packed sizes are "returned".
         */
        // bsz::b512 float
        EXPECT_EQ(typeid(avxvector<bsz::b512, float>::type), typeid(__m512));
        EXPECT_EQ(typeid(avxvector<bsz::b512, float>::half_type), typeid(__m256));
        EXPECT_EQ((avxvector<bsz::b512, float>::value), 16U);
        EXPECT_EQ((avxvector<bsz::b512, float>()), 16U);
        EXPECT_EQ((avxvector<bsz::b512, float>::half), 8U);
        EXPECT_EQ((avxvector<bsz::b512, float>::count), 16U);

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
        EXPECT_EQ((avxvector<bsz::b512, double>::value), 8U);
        EXPECT_EQ((avxvector<bsz::b512, double>()), 8U);
        EXPECT_EQ((avxvector<bsz::b512, double>::half), 4U);
        EXPECT_EQ((avxvector<bsz::b512, double>::count), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b512, double>), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b512, double>), typeid(__m256d));
        EXPECT_EQ((avxvector_v<bsz::b512, double>), 8U);
        EXPECT_EQ((avxvector<bsz::b512, double>()), 8U);
        EXPECT_EQ((avxvector_half_v<bsz::b512, double>), 4U);
        EXPECT_EQ((hsz_v<bsz::b512, double>), 4U);
        EXPECT_EQ((tsz_v<bsz::b512, double>), 8U);
    }
#endif

    TEST(KT_L0, KT_CTYPES_256)
    {
        /*
         * These bsz::b256 bit tests check that the correct complex data type
         * and packed sizes are "returned".
         */
        // bsz::b256 cfloat
        EXPECT_EQ(typeid(avxvector<bsz::b256, cfloat>::type), typeid(__m256));
        EXPECT_EQ(typeid(avxvector<bsz::b256, cfloat>::half_type), typeid(__m128));
        EXPECT_EQ((avxvector<bsz::b256, cfloat>::value), 8U);
        EXPECT_EQ((avxvector<bsz::b256, cfloat>()), 8U);
        EXPECT_EQ((avxvector<bsz::b256, cfloat>::half), 4U);
        EXPECT_EQ((avxvector<bsz::b256, cfloat>::count), 4U);

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
        EXPECT_EQ((avxvector<bsz::b256, cdouble>::value), 4U);
        EXPECT_EQ((avxvector<bsz::b256, cdouble>()), 4U);
        EXPECT_EQ((avxvector<bsz::b256, cdouble>::half), 2U);
        EXPECT_EQ((avxvector<bsz::b256, cdouble>::count), 2U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<bsz::b256, cdouble>), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector_half_t<bsz::b256, cdouble>), typeid(__m128d));
        EXPECT_EQ((avxvector_v<bsz::b256, cdouble>), 4U);
        EXPECT_EQ((avxvector<bsz::b256, cdouble>()), 4U);
        EXPECT_EQ((avxvector_half_v<bsz::b256, cdouble>), 2U);
        EXPECT_EQ((hsz_v<bsz::b256, cdouble>), 2U);
        EXPECT_EQ((tsz_v<bsz::b256, cdouble>), 2U);
    }
#ifdef __AVX512F__
    TEST(KT_L0, KT_CTYPES_512)
    {
        /*
         * These bsz::b512 bit tests check that the correct complex data type
         * and packed sizes are "returned".
         */
        // bsz::b512 cfloat
        EXPECT_EQ(typeid(avxvector<bsz::b512, cfloat>::type), typeid(__m512));
        EXPECT_EQ(typeid(avxvector<bsz::b512, cfloat>::half_type), typeid(__m256));
        EXPECT_EQ((avxvector<bsz::b512, cfloat>::value), 16U);
        EXPECT_EQ((avxvector<bsz::b512, cfloat>()), 16U);
        EXPECT_EQ((avxvector<bsz::b512, cfloat>::half), 8U);
        EXPECT_EQ((avxvector<bsz::b512, cfloat>::count), 8U);

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
        EXPECT_EQ((avxvector<bsz::b512, cdouble>::value), 8U);
        EXPECT_EQ((avxvector<bsz::b512, cdouble>()), 8U);
        EXPECT_EQ((avxvector<bsz::b512, cdouble>::half), 4U);
        EXPECT_EQ((avxvector<bsz::b512, cdouble>::count), 4U);

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

    /*
     * Test loadu intrinsic to load 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_loadu_p_256)
    {
        EXPECT_EQ_VEC(
            (tsz_v<bsz::b256, float>), (kt_loadu_p<bsz::b256, float>(&D.vs[3])), &D.vs[3]);
        EXPECT_EQ_VEC(
            (tsz_v<bsz::b256, double>), (kt_loadu_p<bsz::b256, double>(&D.vd[5])), &D.vd[5]);

        std::complex<float>           *c;
        avxvector_t<bsz::b256, cfloat> w;
        w = kt_loadu_p<bsz::b256, cfloat>(&D.vc[5]);
        c = reinterpret_cast<std::complex<float> *>(&w);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<bsz::b256, cfloat>), c, &D.vc[5]);

        std::complex<double>           *z;
        avxvector_t<bsz::b256, cdouble> v;
        v = kt_loadu_p<bsz::b256, cdouble>(&D.vz[1]);
        z = reinterpret_cast<std::complex<double> *>(&v);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<bsz::b256, cdouble>), z, &D.vz[1]);
    }
#ifdef __AVX512F__
    TEST(KT_L0, kt_loadu_p_512)
    {
        EXPECT_EQ_VEC(
            (tsz_v<bsz::b512, float>), (kt_loadu_p<bsz::b512, float>(&D.vs[0])), &D.vs[0]);
        EXPECT_EQ_VEC(
            (tsz_v<bsz::b512, double>), (kt_loadu_p<bsz::b512, double>(&D.vd[0])), &D.vd[0]);

        std::complex<float>           *c;
        avxvector_t<bsz::b512, cfloat> w;
        w = kt_loadu_p<bsz::b512, cfloat>(&D.vc[0]);
        c = reinterpret_cast<std::complex<float> *>(&w);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<bsz::b512, cfloat>), c, &D.vc[0]);

        std::complex<double>           *z;
        avxvector_t<bsz::b512, cdouble> v;
        v = kt_loadu_p<bsz::b512, cdouble>(&D.vz[0]);
        z = reinterpret_cast<std::complex<double> *>(&v);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<bsz::b512, cdouble>), z, &D.vz[0]);
    }
#endif

    /*
     * Test setzero intrinsic to zero-out 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_setzero_p_256)
    {
        const size_t ns        = tsz_v<bsz::b256, float>;
        const size_t nd        = tsz_v<bsz::b256, double>;
        const float  szero[ns] = {0.f};
        const double dzero[nd] = {0.0};

        EXPECT_EQ_VEC(ns, (kt_setzero_p<bsz::b256, float>()), szero);
        EXPECT_EQ_VEC(nd, (kt_setzero_p<bsz::b256, double>()), dzero);

        // Complex checks are reinterpreted as double-packed reals
        EXPECT_EQ_VEC((tsz_v<bsz::b256, cfloat>), (kt_setzero_p<bsz::b256, cfloat>()), szero);
        EXPECT_EQ_VEC((tsz_v<bsz::b256, cdouble>), (kt_setzero_p<bsz::b256, cdouble>()), dzero);
    }
#ifdef __AVX512F__
    TEST(KT_L0, kt_setzero_p_512)
    {
        const size_t ns        = tsz_v<bsz::b512, float>;
        const size_t nd        = tsz_v<bsz::b512, double>;
        const float  szero[ns] = {0.f};
        const double dzero[nd] = {0.0};

        EXPECT_EQ_VEC((tsz_v<bsz::b512, float>), (kt_setzero_p<bsz::b512, float>()), szero);
        EXPECT_EQ_VEC((tsz_v<bsz::b512, double>), (kt_setzero_p<bsz::b512, double>()), dzero);

        // Complex checks are reinterpreted as double-packed reals
        EXPECT_EQ_VEC((tsz_v<bsz::b512, cfloat>), (kt_setzero_p<bsz::b512, cfloat>()), szero);
        EXPECT_EQ_VEC((tsz_v<bsz::b512, cdouble>), (kt_setzero_p<bsz::b512, cdouble>()), dzero);
    }
#endif

    /*
     * Test set1 intrinsic to load a scalat into 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_set1_p_256)
    {
        float   refs[] = {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};
        double  refd[] = {5.0, 5.0, 5.0, 5.0};
        cfloat  refc[] = {3.f + 4.0if, 3.f + 4.0if, 3.f + 4.0if, 3.f + 4.0if};
        cdouble refz[] = {2.0 + 5.0i, 2.0 + 5.0i};

        EXPECT_EQ_VEC((tsz_v<bsz::b256, float>), (kt_set1_p<bsz::b256, float>(4.0f)), refs);
        EXPECT_EQ_VEC((tsz_v<bsz::b256, double>), (kt_set1_p<bsz::b256, double>(5.0)), refd);

        std::complex<float>           *c;
        avxvector_t<bsz::b256, cfloat> vc;
        vc = kt_set1_p<bsz::b256, cfloat>(3.f + 4.0if);
        c  = reinterpret_cast<std::complex<float> *>(&vc);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<bsz::b256, cfloat>), c, refc);

        std::complex<double>           *z;
        avxvector_t<bsz::b256, cdouble> vz;
        vz = kt_set1_p<bsz::b256, cdouble>(2. + 5.0i);
        z  = reinterpret_cast<std::complex<double> *>(&vz);
        EXPECT_EQ_VEC((tsz_v<bsz::b256, cdouble>), z, refz);
    }
#ifdef __AVX512F__
    TEST(KT_L0, kt_set1_p_512)
    {
        float refs[]{
            4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f};
        double  refd[]{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
        cfloat  refc[]{2.1f + 4.5if,
                       2.1f + 4.5if,
                       2.1f + 4.5if,
                       2.1f + 4.5if,
                       2.1f + 4.5if,
                       2.1f + 4.5if,
                       2.1f + 4.5if,
                       2.1f + 4.5if};
        cdouble refz[]{2.2 + 5.1i, 2.2 + 5.1i, 2.2 + 5.1i, 2.2 + 5.1i};

        EXPECT_EQ_VEC((tsz_v<bsz::b512, float>), (kt_set1_p<bsz::b512, float>(4.0f)), refs);
        EXPECT_EQ_VEC((tsz_v<bsz::b512, double>), (kt_set1_p<bsz::b512, double>(5.0)), refd);

        std::complex<float>           *c;
        avxvector_t<bsz::b512, cfloat> vc;
        vc = kt_set1_p<bsz::b512, cfloat>(2.1f + 4.5if);
        c  = reinterpret_cast<std::complex<float> *>(&vc);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<bsz::b512, cfloat>), c, refc);

        std::complex<double>           *z;
        avxvector_t<bsz::b512, cdouble> vz;
        vz = kt_set1_p<bsz::b512, cdouble>(2.2 + 5.1i);
        z  = reinterpret_cast<std::complex<double> *>(&vz);
        EXPECT_EQ_VEC((tsz_v<bsz::b512, cdouble>), z, refz);
    }
#endif

    /*
     * Test add intrinsic to sum two: 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_add_p_256)
    {
        const size_t                   ns = tsz_v<bsz::b256, float>;
        const size_t                   nd = tsz_v<bsz::b256, double>;
        avxvector_t<bsz::b256, float>  s, as, bs;
        avxvector_t<bsz::b256, double> d, ad, bd;
        float                          refs[8];
        double                         refd[4];

        as = kt_loadu_p<bsz::b256, float>(&D.vs[2]);
        bs = kt_set1_p<bsz::b256, float>(1.0);
        s  = kt_add_p<bsz::b256, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[2 + i] + 1.0f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b256, double>(&D.vd[2]);
        bd = kt_set1_p<bsz::b256, double>(1.0);
        d  = kt_add_p<bsz::b256, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[2 + i] + 1.0;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex
        const size_t                    nc = tsz_v<bsz::b256, cfloat>;
        const size_t                    nz = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, cfloat>  c, ac, bc;
        avxvector_t<bsz::b256, cdouble> z, az, bz;
        cfloat                          refc[4];
        cdouble                         refz[2];
        std::complex<float>            *pc;
        std::complex<double>           *pz;

        ac = kt_loadu_p<bsz::b256, cfloat>(&D.vc[2]);
        bc = kt_set1_p<bsz::b256, cfloat>(1.0f + 5.if);
        c  = kt_add_p<bsz::b256, cfloat>(ac, bc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] + (1.0f + 5.if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&c);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        az = kt_loadu_p<bsz::b256, cdouble>(&D.vz[2]);
        bz = kt_set1_p<bsz::b256, cdouble>(3.0 + 5.5i);
        z  = kt_add_p<bsz::b256, cdouble>(az, bz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] + (3.0 + 5.5i);
        }
        pz = reinterpret_cast<std::complex<double> *>(&z);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_add_p_512)
    {
        const size_t                   ns = tsz_v<bsz::b512, float>;
        const size_t                   nd = tsz_v<bsz::b512, double>;
        avxvector_t<bsz::b512, float>  s, as, bs;
        avxvector_t<bsz::b512, double> d, ad, bd;
        float                          refs[16];
        double                         refd[8];

        as = kt_loadu_p<bsz::b512, float>(&D.vs[0]);
        bs = kt_set1_p<bsz::b512, float>(1.0f);
        s  = kt_add_p<bsz::b512, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] + 1.0f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b512, double>(&D.vd[0]);
        bd = kt_set1_p<bsz::b512, double>(1.0);
        d  = kt_add_p<bsz::b512, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] + 1.0;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex type testing
        size_t                          nc = tsz_v<bsz::b512, cfloat>;
        size_t                          nz = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, cfloat>  c, ac, bc;
        avxvector_t<bsz::b512, cdouble> z, az, bz;
        cfloat                          refc[8];
        cdouble                         refz[4];
        std::complex<float>            *pc;
        std::complex<double>           *pz;

        ac = kt_loadu_p<bsz::b512, cfloat>(&D.vc[2]);
        bc = kt_set1_p<bsz::b512, cfloat>(1.0f + 5.if);
        c  = kt_add_p<bsz::b512, cfloat>(ac, bc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] + (1.0f + 5.if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&c);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        az = kt_loadu_p<bsz::b512, cdouble>(&D.vz[2]);
        bz = kt_set1_p<bsz::b512, cdouble>(3.0 + 5.5i);
        z  = kt_add_p<bsz::b512, cdouble>(az, bz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] + (3.0 + 5.5i);
        }
        pz = reinterpret_cast<std::complex<double> *>(&z);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }
#endif

    /*
     * Test intrinsic to subtract two: 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_sub_p_256)
    {
        const size_t                   ns = tsz_v<bsz::b256, float>;
        const size_t                   nd = tsz_v<bsz::b256, double>;
        avxvector_t<bsz::b256, float>  s, as, bs;
        avxvector_t<bsz::b256, double> d, ad, bd;
        float                          refs[8];
        double                         refd[4];

        as = kt_loadu_p<bsz::b256, float>(&D.vs[2]);
        bs = kt_set1_p<bsz::b256, float>(1.0);
        s  = kt_sub_p<bsz::b256, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[2 + i] - 1.0f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b256, double>(&D.vd[2]);
        bd = kt_set1_p<bsz::b256, double>(1.0);
        d  = kt_sub_p<bsz::b256, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[2 + i] - 1.0;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex
        const size_t                    nc = tsz_v<bsz::b256, cfloat>;
        const size_t                    nz = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, cfloat>  c, ac, bc;
        avxvector_t<bsz::b256, cdouble> z, az, bz;
        cfloat                          refc[4];
        cdouble                         refz[2];
        std::complex<float>            *pc;
        std::complex<double>           *pz;

        ac = kt_loadu_p<bsz::b256, cfloat>(&D.vc[2]);
        bc = kt_set1_p<bsz::b256, cfloat>(1.0f + 5.if);
        c  = kt_sub_p<bsz::b256, cfloat>(ac, bc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] - (1.0f + 5.if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&c);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        az = kt_loadu_p<bsz::b256, cdouble>(&D.vz[2]);
        bz = kt_set1_p<bsz::b256, cdouble>(3.0 + 5.5i);
        z  = kt_sub_p<bsz::b256, cdouble>(az, bz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] - (3.0 + 5.5i);
        }
        pz = reinterpret_cast<std::complex<double> *>(&z);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_sub_p_512)
    {
        const size_t                   ns = tsz_v<bsz::b512, float>;
        const size_t                   nd = tsz_v<bsz::b512, double>;
        avxvector_t<bsz::b512, float>  s, as, bs;
        avxvector_t<bsz::b512, double> d, ad, bd;
        float                          refs[16];
        double                         refd[8];

        as = kt_loadu_p<bsz::b512, float>(&D.vs[0]);
        bs = kt_set1_p<bsz::b512, float>(1.0f);
        s  = kt_sub_p<bsz::b512, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] - 1.0f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b512, double>(&D.vd[0]);
        bd = kt_set1_p<bsz::b512, double>(1.0);
        d  = kt_sub_p<bsz::b512, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] - 1.0;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex type testing
        size_t                          nc = tsz_v<bsz::b512, cfloat>;
        size_t                          nz = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, cfloat>  c, ac, bc;
        avxvector_t<bsz::b512, cdouble> z, az, bz;
        cfloat                          refc[8];
        cdouble                         refz[4];
        std::complex<float>            *pc;
        std::complex<double>           *pz;

        ac = kt_loadu_p<bsz::b512, cfloat>(&D.vc[2]);
        bc = kt_set1_p<bsz::b512, cfloat>(1.0f + 5.if);
        c  = kt_sub_p<bsz::b512, cfloat>(ac, bc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] - (1.0f + 5.if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&c);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        az = kt_loadu_p<bsz::b512, cdouble>(&D.vz[2]);
        bz = kt_set1_p<bsz::b512, cdouble>(3.0 + 5.5i);
        z  = kt_sub_p<bsz::b512, cdouble>(az, bz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] - (3.0 + 5.5i);
        }
        pz = reinterpret_cast<std::complex<double> *>(&z);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }
#endif

    /*
     * Test mul intrinsic to multiply two 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_mul_p_256)
    {
        size_t                         ns = tsz_v<bsz::b256, float>;
        size_t                         nd = tsz_v<bsz::b256, double>;
        avxvector_t<bsz::b256, float>  s, as, bs;
        avxvector_t<bsz::b256, double> d, ad, bd;
        float                          refs[8];
        double                         refd[4];

        as = kt_loadu_p<bsz::b256, float>(&D.vs[2]);
        bs = kt_set1_p<bsz::b256, float>(2.75f);
        s  = kt_mul_p<bsz::b256, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[2 + i] * 2.75f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b256, double>(&D.vd[2]);
        bd = kt_set1_p<bsz::b256, double>(2.75);
        d  = kt_mul_p<bsz::b256, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[2 + i] * 2.75;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // complex<float> mul_p
        const size_t                   nc = tsz_v<bsz::b256, cfloat>;
        avxvector_t<bsz::b256, cfloat> ac = kt_loadu_p<bsz::b256, cfloat>(&D.vc[2]);
        avxvector_t<bsz::b256, cfloat> bc = kt_set1_p<bsz::b256, cfloat>(2.5f + 4.5if);
        avxvector_t<bsz::b256, cfloat> sc = kt_mul_p<bsz::b256, cfloat>(ac, bc);
        std::complex<float>            refc[nc];
        std::complex<float>           *pc;
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * (2.5f + 4.5if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // complex<double> mul_p
        const size_t                    nz = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, cdouble> az = kt_loadu_p<bsz::b256, cdouble>(&D.vz[2]);
        avxvector_t<bsz::b256, cdouble> bz = kt_set1_p<bsz::b256, cdouble>(2.125 + 3.5i);
        avxvector_t<bsz::b256, cdouble> cz = kt_mul_p<bsz::b256, cdouble>(az, bz);
        std::complex<double>            refz[nz];
        std::complex<double>           *pz;
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * (2.125 + 3.5i);
        }
        pz = reinterpret_cast<std::complex<double> *>(&cz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_mul_p_512)
    {
        size_t                         ns = tsz_v<bsz::b512, float>;
        size_t                         nd = tsz_v<bsz::b512, double>;
        avxvector_t<bsz::b512, float>  s, as, bs;
        avxvector_t<bsz::b512, double> d, ad, bd;
        float                          refs[16];
        double                         refd[8];

        as = kt_loadu_p<bsz::b512, float>(&D.vs[0]);
        bs = kt_set1_p<bsz::b512, float>(3.3f);
        s  = kt_mul_p<bsz::b512, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * 3.3f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b512, double>(&D.vd[0]);
        bd = kt_set1_p<bsz::b512, double>(3.5);
        d  = kt_mul_p<bsz::b512, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * 3.5;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // complex<float> mul_p
        const size_t                   nc = tsz_v<bsz::b512, cfloat>;
        avxvector_t<bsz::b512, cfloat> ac = kt_loadu_p<bsz::b512, cfloat>(&D.vc[2]);
        avxvector_t<bsz::b512, cfloat> bc = kt_set1_p<bsz::b512, cfloat>(2.5f + 4.125if);
        avxvector_t<bsz::b512, cfloat> sc = kt_mul_p<bsz::b512, cfloat>(ac, bc);
        std::complex<float>            refc[nc];
        std::complex<float>           *pc;
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * (2.5f + 4.125if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // complex<double> mul_p
        avxvector_t<bsz::b512, cdouble> az = kt_loadu_p<bsz::b512, cdouble>(&D.vz[2]);
        avxvector_t<bsz::b512, cdouble> bz = kt_set1_p<bsz::b512, cdouble>(2.75 + 3.5i);
        avxvector_t<bsz::b512, cdouble> dz = kt_mul_p<bsz::b512, cdouble>(az, bz);
        const size_t                    nz = tsz_v<bsz::b512, cdouble>;
        std::complex<double>            refz[nz];
        std::complex<double>           *pz;
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * (2.75 + 3.5i);
        }
        pz = reinterpret_cast<std::complex<double> *>(&dz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }
#endif

    /*
     * Test fmadd intrinsic to fuse-multiply-add three
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_fmadd_p_256)
    {
        const size_t                   ns = tsz_v<bsz::b256, float>;
        const size_t                   nd = tsz_v<bsz::b256, double>;
        avxvector_t<bsz::b256, float>  s, as, bs;
        avxvector_t<bsz::b256, double> d, ad, bd;
        float                          refs[8];
        double                         refd[4];

        as = kt_loadu_p<bsz::b256, float>(&D.vs[2]);
        bs = kt_set_p<bsz::b256, float>(D.vs, D.map);
        s  = kt_set_p<bsz::b256, float>(D.vs, &D.map[4]);
        s  = kt_fmadd_p<bsz::b256, float>(as, bs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[2 + i] * D.vs[D.map[i]] + D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b256, double>(&D.vd[1]);
        bd = kt_set_p<bsz::b256, double>(D.vd, D.map);
        d  = kt_set_p<bsz::b256, double>(D.vd, &D.map[2]);
        d  = kt_fmadd_p<bsz::b256, double>(ad, bd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[1 + i] * D.vd[D.map[i]] + D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex<float> FMADD
        constexpr size_t              nc = tsz_v<bsz::b256, cfloat>;
        avxvector_t<bsz::b256, float> sc, ac, bc, cc;
        cfloat                        refc[nc];
        const cfloat                  vc1[nc]{1.f + 1.if, 2.f + 3if, 0.f - 4if, 1.25f + 3.5if};
        const cfloat                  vc2[nc]{-2.f + 3.if, -1.f - 2if, 3.f + 1if, 2.5f - 5.75if};

        ac = kt_loadu_p<bsz::b256, cfloat>(&D.vc[2]);
        bc = kt_loadu_p<bsz::b256, cfloat>(vc1);
        cc = kt_loadu_p<bsz::b256, cfloat>(vc2);
        sc = kt_fmadd_p<bsz::b256, cfloat>(ac, bc, cc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * vc1[i] + vc2[i];
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // Complex<double> FMADD
        constexpr size_t               nz = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, double> sz, az, bz, cz;
        cdouble                        refz[nz];
        const cdouble                  vz1[nz]{1. + 1.i, 2. - 4i};
        const cdouble                  vz2[nz]{-3. - 2i, 2.5 + 1.5i};

        az = kt_loadu_p<bsz::b256, cdouble>(&D.vz[2]);
        bz = kt_loadu_p<bsz::b256, cdouble>(vz1);
        cz = kt_loadu_p<bsz::b256, cdouble>(vz2);
        sz = kt_fmadd_p<bsz::b256, cdouble>(az, bz, cz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * vz1[i] + vz2[i];
        }
        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&sz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_fmadd_p_512)
    {
        size_t                         ns = tsz_v<bsz::b512, float>;
        size_t                         nd = tsz_v<bsz::b512, double>;
        avxvector_t<bsz::b512, float>  s, as, bs;
        avxvector_t<bsz::b512, double> d, ad, bd;
        float                          refs[ns];
        double                         refd[nd];

        as = kt_loadu_p<bsz::b512, float>(&D.vs[0]);
        bs = kt_set_p<bsz::b512, float>(D.vs, D.map);
        s  = kt_set_p<bsz::b512, float>(D.vs, &D.map[4]);
        s  = kt_fmadd_p<bsz::b512, float>(as, bs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * D.vs[D.map[i]] + D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b512, double>(&D.vd[0]);
        bd = kt_set_p<bsz::b512, double>(D.vd, D.map);
        d  = kt_set_p<bsz::b512, double>(D.vd, &D.map[2]);
        d  = kt_fmadd_p<bsz::b512, double>(ad, bd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * D.vd[D.map[i]] + D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex<float> FMADD
        constexpr size_t              nc = tsz_v<bsz::b512, cfloat>;
        avxvector_t<bsz::b512, float> sc, ac, bc, cc;
        cfloat                        refc[nc];
        const cfloat                  vc1[nc]{1.f + 1.if,
                                              2.f + 3if,
                                              0.f - 4if,
                                              1.2f + 0.125if,
                                              2.f + 4.if,
                                              4.f + 5if,
                                              2.f - 4if,
                                              2.75f + 4.5if};
        const cfloat                  vc2[nc]{-1.f + 1.if,
                                              -1.f - 2if,
                                              3.f + 1if,
                                              2.5f - 1.75if,
                                              3.f + 3.if,
                                              7.f + 8if,
                                              1.f - 3if,
                                              3.5f + 0.5if};

        ac = kt_loadu_p<bsz::b512, cfloat>(&D.vc[2]);
        bc = kt_loadu_p<bsz::b512, cfloat>(vc1);
        cc = kt_loadu_p<bsz::b512, cfloat>(vc2);
        sc = kt_fmadd_p<bsz::b512, cfloat>(ac, bc, cc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * vc1[i] + vc2[i];
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // Complex<double> FMADD
        constexpr size_t               nz = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, double> sz, az, bz, cz;
        cdouble                        refz[nz];
        const cdouble                  vz1[nz]{1. + 1.i, 2. - 4i, 2. + 3.i, 7. - 5i};
        const cdouble                  vz2[nz]{-3. - 2i, 2.5 + 1.5i, 1. + 7.i, 5. - 8i};

        az = kt_loadu_p<bsz::b512, cdouble>(&D.vz[2]);
        bz = kt_loadu_p<bsz::b512, cdouble>(vz1);
        cz = kt_loadu_p<bsz::b512, cdouble>(vz2);
        sz = kt_fmadd_p<bsz::b512, cdouble>(az, bz, cz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * vz1[i] + vz2[i];
        }

        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&sz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }
#endif

    /*
     * Test fmsub intrinsic to fused-multiply-subtract three
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_fmsub_p_256)
    {
        const size_t                   ns = tsz_v<bsz::b256, float>;
        const size_t                   nd = tsz_v<bsz::b256, double>;
        avxvector_t<bsz::b256, float>  s, as, bs;
        avxvector_t<bsz::b256, double> d, ad, bd;
        float                          refs[8];
        double                         refd[4];

        as = kt_loadu_p<bsz::b256, float>(&D.vs[2]);
        bs = kt_set_p<bsz::b256, float>(D.vs, D.map);
        s  = kt_set_p<bsz::b256, float>(D.vs, &D.map[4]);
        s  = kt_fmsub_p<bsz::b256, float>(as, bs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[2 + i] * D.vs[D.map[i]] - D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b256, double>(&D.vd[1]);
        bd = kt_set_p<bsz::b256, double>(D.vd, D.map);
        d  = kt_set_p<bsz::b256, double>(D.vd, &D.map[2]);
        d  = kt_fmsub_p<bsz::b256, double>(ad, bd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[1 + i] * D.vd[D.map[i]] - D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex<float> FMSUB
        constexpr size_t              nc = tsz_v<bsz::b256, cfloat>;
        avxvector_t<bsz::b256, float> sc, ac, bc, cc;
        cfloat                        refc[nc];
        const cfloat                  vc1[nc]{1.f + 1.if, 2.f + 3if, 0.f - 4if, 1.25f + 3.5if};
        const cfloat                  vc2[nc]{-2.f + 3.if, -1.f - 2if, 3.f + 1if, 2.5f - 5.75if};

        ac = kt_loadu_p<bsz::b256, cfloat>(&D.vc[2]);
        bc = kt_loadu_p<bsz::b256, cfloat>(vc1);
        cc = kt_loadu_p<bsz::b256, cfloat>(vc2);
        sc = kt_fmsub_p<bsz::b256, cfloat>(ac, bc, cc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * vc1[i] - vc2[i];
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // Complex<double> FMSUB
        constexpr size_t               nz = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, double> sz, az, bz, cz;
        cdouble                        refz[nz];
        const cdouble                  vz1[nz]{1. + 1.i, 2. - 4i};
        const cdouble                  vz2[nz]{-3. - 2i, 2.5 + 1.5i};

        az = kt_loadu_p<bsz::b256, cdouble>(&D.vz[2]);
        bz = kt_loadu_p<bsz::b256, cdouble>(vz1);
        cz = kt_loadu_p<bsz::b256, cdouble>(vz2);
        sz = kt_fmsub_p<bsz::b256, cdouble>(az, bz, cz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * vz1[i] - vz2[i];
        }
        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&sz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_fmsub_p_512)
    {
        size_t                         ns = tsz_v<bsz::b512, float>;
        size_t                         nd = tsz_v<bsz::b512, double>;
        avxvector_t<bsz::b512, float>  s, as, bs;
        avxvector_t<bsz::b512, double> d, ad, bd;
        float                          refs[ns];
        double                         refd[nd];

        as = kt_loadu_p<bsz::b512, float>(&D.vs[0]);
        bs = kt_set_p<bsz::b512, float>(D.vs, D.map);
        s  = kt_set_p<bsz::b512, float>(D.vs, &D.map[4]);
        s  = kt_fmsub_p<bsz::b512, float>(as, bs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * D.vs[D.map[i]] - D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<bsz::b512, double>(&D.vd[0]);
        bd = kt_set_p<bsz::b512, double>(D.vd, D.map);
        d  = kt_set_p<bsz::b512, double>(D.vd, &D.map[2]);
        d  = kt_fmsub_p<bsz::b512, double>(ad, bd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * D.vd[D.map[i]] - D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex<float> FMSUB
        constexpr size_t              nc = tsz_v<bsz::b512, cfloat>;
        avxvector_t<bsz::b512, float> sc, ac, bc, cc;
        cfloat                        refc[nc];
        const cfloat                  vc1[nc]{1.f + 1.if,
                                              2.f + 3if,
                                              0.f - 4if,
                                              1.2f + 0.125if,
                                              2.f + 4.if,
                                              4.f + 5if,
                                              2.f - 4if,
                                              2.75f + 4.5if};
        const cfloat                  vc2[nc]{-1.f + 1.if,
                                              -1.f - 2if,
                                              3.f + 1if,
                                              2.5f - 1.75if,
                                              3.f + 3.if,
                                              7.f + 8if,
                                              1.f - 3if,
                                              3.5f + 0.5if};

        ac = kt_loadu_p<bsz::b512, cfloat>(&D.vc[2]);
        bc = kt_loadu_p<bsz::b512, cfloat>(vc1);
        cc = kt_loadu_p<bsz::b512, cfloat>(vc2);
        sc = kt_fmsub_p<bsz::b512, cfloat>(ac, bc, cc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * vc1[i] - vc2[i];
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // Complex<double> FMSUB
        constexpr size_t               nz = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, double> sz, az, bz, cz;
        cdouble                        refz[nz];
        const cdouble                  vz1[nz]{1. + 1.i, 2. - 4i, 2. + 3.i, 7. - 5i};
        const cdouble                  vz2[nz]{-3. - 2i, 2.5 + 1.5i, 1. + 7.i, 5. - 8i};

        az = kt_loadu_p<bsz::b512, cdouble>(&D.vz[2]);
        bz = kt_loadu_p<bsz::b512, cdouble>(vz1);
        cz = kt_loadu_p<bsz::b512, cdouble>(vz2);
        sz = kt_fmsub_p<bsz::b512, cdouble>(az, bz, cz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * vz1[i] - vz2[i];
        }

        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&sz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }
#endif

    /*
     * Test "set" intrinsic to indirectly load using a "map"
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_set_p_256)
    {
        const size_t                    ns = tsz_v<bsz::b256, float>;
        const size_t                    nd = tsz_v<bsz::b256, double>;
        const size_t                    nc = tsz_v<bsz::b256, cfloat>;
        const size_t                    nz = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, float>   s;
        avxvector_t<bsz::b256, double>  d;
        avxvector_t<bsz::b256, cfloat>  c;
        avxvector_t<bsz::b256, cdouble> z;
        float                           refs[ns];
        double                          refd[nd];
        cfloat                          refc[nc];
        cdouble                         refz[nz];
        cfloat                         *pc;
        cdouble                        *pz;

        s = kt_set_p<bsz::b256, float>(D.vs, &D.map[1]);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[D.map[1 + i]];
        }
        EXPECT_EQ_VEC(ns, s, refs);

        d = kt_set_p<bsz::b256, double>(D.vd, &D.map[2]);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[D.map[2 + i]];
        }
        EXPECT_EQ_VEC(nd, d, refd);

        c = kt_set_p<bsz::b256, cfloat>(D.vc, &D.map[5]);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[D.map[5 + i]];
        }
        pc = reinterpret_cast<cfloat *>(&c);
        EXPECT_COMPLEX_EQ_VEC(nc, pc, refc);

        z = kt_set_p<bsz::b256, cdouble>(D.vz, &D.map[3]);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[D.map[3 + i]];
        }
        pz = reinterpret_cast<cdouble *>(&z);
        EXPECT_COMPLEX_EQ_VEC(nz, pz, refz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_set_p_512)
    {
        const size_t                    ns = tsz_v<bsz::b512, float>;
        const size_t                    nd = tsz_v<bsz::b512, double>;
        const size_t                    nc = tsz_v<bsz::b512, cfloat>;
        const size_t                    nz = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, float>   s;
        avxvector_t<bsz::b512, double>  d;
        avxvector_t<bsz::b512, cfloat>  c;
        avxvector_t<bsz::b512, cdouble> z;
        float                           refs[ns];
        double                          refd[nd];
        cfloat                          refc[nc];
        cdouble                         refz[nz];
        cfloat                         *pc;
        cdouble                        *pz;

        s = kt_set_p<bsz::b512, float>(D.vs, &D.map[2]);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[D.map[2 + i]];
        }
        EXPECT_EQ_VEC(ns, s, refs);

        d = kt_set_p<bsz::b512, double>(D.vd, &D.map[3]);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[D.map[3 + i]];
        }
        EXPECT_EQ_VEC(nd, d, refd);

        c = kt_set_p<bsz::b512, cfloat>(D.vc, &D.map[5]);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[D.map[5 + i]];
        }
        pc = reinterpret_cast<cfloat *>(&c);
        EXPECT_COMPLEX_EQ_VEC(nc, pc, refc);

        z = kt_set_p<bsz::b512, cdouble>(D.vz, &D.map[0]);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[D.map[0 + i]];
        }
        pz = reinterpret_cast<cdouble *>(&z);
        EXPECT_COMPLEX_EQ_VEC(nz, pz, refz);
    }
#endif

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
        const size_t         n = tsz_v<SZ, SUF>;                                     \
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

    TEST(KT_L0, kt_maskz_set_p_256_AVX)
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

        // bsz::b256 double -> 4
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 2, 1);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 3, 3);
        kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 4, 2);
        // This also triggers a warning
        // kt_maskz_set_p_param_dir(bsz::b256, double, d, AVX, 5);

        // bsz::b256 cfloat -> 4
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, 1, 1);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, 2, 0);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, 3, 4);
        kt_maskz_set_p_param_dir(bsz::b256, cfloat, c, AVX, 4, 0);

        // bsz::b256 cdouble -> 2
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX, 1, 0);
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX, 2, 4);
        kt_maskz_set_p_param_dir(bsz::b256, cdouble, z, AVX, 2, 0);

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
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_maskz_set_p_256_AVX512VL)
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
#endif

#ifdef __AVX512F__
    TEST(KT_L0, kt_maskz_set_p_512_AVX512F)
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
    }
#endif

    /*
     * Test "hsum" intrinsic to horizontally-reduce via summation
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_hsum_p_256)
    {
        const size_t                   ns = tsz_v<bsz::b256, float>;
        const size_t                   nd = tsz_v<bsz::b256, double>;
        avxvector_t<bsz::b256, float>  vs;
        avxvector_t<bsz::b256, double> vd;
        float                          sums, refs = 0.0f;
        double                         sumd, refd = 0.0;

        vs   = kt_loadu_p<bsz::b256, float>(D.vs);
        sums = kt_hsum_p<bsz::b256, float>(vs);
        for(size_t i = 0; i < ns; i++)
        {
            refs += D.vs[i];
        }
        EXPECT_FLOAT_EQ(sums, refs);

        vd   = kt_loadu_p<bsz::b256, double>(D.vd);
        sumd = kt_hsum_p<bsz::b256, double>(vd);
        for(size_t i = 0; i < nd; i++)
        {
            refd += D.vd[i];
        }
        EXPECT_DOUBLE_EQ(sumd, refd);

        const size_t                    nc = tsz_v<bsz::b256, cfloat>;
        const size_t                    nz = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, cfloat>  vc;
        avxvector_t<bsz::b256, cdouble> vz;
        cfloat                          sumc, refc = {0.0f, 0.0f};
        cdouble                         sumz, refz = {0.0, 0.0};

        vc   = kt_loadu_p<bsz::b256, cfloat>(D.vc);
        sumc = kt_hsum_p<bsz::b256, cfloat>(vc);
        for(size_t i = 0; i < nc; i++)
        {
            refc += D.vc[i];
        }
        EXPECT_COMPLEX_FLOAT_EQ(sumc, refc);

        vz   = kt_loadu_p<bsz::b256, cdouble>(D.vz);
        sumz = kt_hsum_p<bsz::b256, cdouble>(vz);
        for(size_t i = 0; i < nz; i++)
        {
            refz += D.vz[i];
        }
        EXPECT_COMPLEX_DOUBLE_EQ(sumz, refz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_hsum_p_512)
    {
        const size_t                   ns = tsz_v<bsz::b512, float>;
        const size_t                   nd = tsz_v<bsz::b512, double>;
        avxvector_t<bsz::b512, float>  vs;
        avxvector_t<bsz::b512, double> vd;
        float                          sums, refs = 0.0f;
        double                         sumd, refd = 0.0;

        vs   = kt_loadu_p<bsz::b512, float>(D.vs);
        sums = kt_hsum_p<bsz::b512, float>(vs);
        for(size_t i = 0; i < ns; i++)
        {
            refs += D.vs[i];
        }
        EXPECT_FLOAT_EQ(sums, refs);

        vd   = kt_loadu_p<bsz::b512, double>(D.vd);
        sumd = kt_hsum_p<bsz::b512, double>(vd);
        for(size_t i = 0; i < nd; i++)
        {
            refd += D.vd[i];
        }
        EXPECT_DOUBLE_EQ(sumd, refd);

        const size_t                    nc = tsz_v<bsz::b512, cfloat>;
        const size_t                    nz = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, cfloat>  vc;
        avxvector_t<bsz::b512, cdouble> vz;
        cfloat                          sumz, refz = {0.0f, 0.0f};
        cdouble                         sumc, refc = {0.0, 0.0};

        vc   = kt_loadu_p<bsz::b512, cfloat>(D.vc);
        sumz = kt_hsum_p<bsz::b512, cfloat>(vc);
        for(size_t i = 0; i < nc; i++)
        {
            refz += D.vc[i];
        }
        EXPECT_COMPLEX_FLOAT_EQ(sumz, refz);

        vz   = kt_loadu_p<bsz::b512, cdouble>(D.vz);
        sumc = kt_hsum_p<bsz::b512, cdouble>(vz);
        for(size_t i = 0; i < nz; i++)
        {
            refc += D.vz[i];
        }
        EXPECT_COMPLEX_DOUBLE_EQ(sumc, refc);
    }
#endif

    /*
     * Test "dot-product" intrinsic on
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_conj_p_256)
    {
        constexpr size_t ns = tsz_v<bsz::b256, float>;
        constexpr size_t nd = tsz_v<bsz::b256, double>;
        constexpr size_t nz = tsz_v<bsz::b256, cdouble>;
        constexpr size_t nc = tsz_v<bsz::b256, cfloat>;

        avxvector_t<bsz::b256, float>   vs, ress;
        avxvector_t<bsz::b256, double>  vd, resd;
        avxvector_t<bsz::b256, cfloat>  vc, resc;
        avxvector_t<bsz::b256, cdouble> vz, resz;

        float   refs[ns];
        cfloat  refc[nc];
        double  refd[nd];
        cdouble refz[nz];

        // Float
        vs   = kt_loadu_p<bsz::b256, float>(D.vs);
        ress = kt_conj_p<bsz::b256, float>(vs);

        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i];
        }

        EXPECT_EQ_VEC(ns, refs, ress);

        // Double
        vd   = kt_loadu_p<bsz::b256, double>(D.vd);
        resd = kt_conj_p<bsz::b256, double>(vd);

        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i];
        }

        EXPECT_EQ_VEC(nd, refd, resd);

        // Cfloat
        vc   = kt_loadu_p<bsz::b256, cfloat>(D.vc);
        resc = kt_conj_p<bsz::b256, cfloat>(vc);

        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = conj(D.vc[i]);
        }

        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&resc);
        EXPECT_COMPLEX_EQ_VEC(nc, refc, pc);

        // CDouble
        vz   = kt_loadu_p<bsz::b256, cdouble>(D.vz);
        resz = kt_conj_p<bsz::b256, cdouble>(vz);

        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = conj(D.vz[i]);
        }

        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&resz);
        EXPECT_COMPLEX_EQ_VEC(nz, refz, pz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_conj_p_512)
    {
        constexpr size_t                ns = tsz_v<bsz::b512, float>;
        constexpr size_t                nd = tsz_v<bsz::b512, double>;
        constexpr size_t                nz = tsz_v<bsz::b512, cdouble>;
        constexpr size_t                nc = tsz_v<bsz::b512, cfloat>;
        avxvector_t<bsz::b512, float>   vs;
        avxvector_t<bsz::b512, double>  vd;
        avxvector_t<bsz::b512, cfloat>  vc;
        avxvector_t<bsz::b512, cdouble> vz;

        float   refs[ns];
        double  refd[nd];
        cfloat  refc[nc];
        cdouble refz[nz];

        // Float
        vs = kt_loadu_p<bsz::b512, float>(D.vs);
        vs = kt_conj_p<bsz::b512, float>(vs);

        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i];
        }

        EXPECT_EQ_VEC(ns, refs, vs);

        // Double
        vd = kt_loadu_p<bsz::b512, double>(D.vd);
        vd = kt_conj_p<bsz::b512, double>(vd);

        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i];
        }

        EXPECT_EQ_VEC(nd, refd, vd);

        // Cfloat
        vc = kt_loadu_p<bsz::b512, cfloat>(&D.vc[0]);
        vc = kt_conj_p<bsz::b512, cfloat>(vc);

        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = conj(D.vc[i]);
        }

        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&vc);
        EXPECT_COMPLEX_EQ_VEC(nc, refc, pc);

        // CDouble
        vz = kt_loadu_p<bsz::b512, cdouble>(&D.vz[0]);
        vz = kt_conj_p<bsz::b512, cdouble>(vz);

        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = conj(D.vz[i]);
        }

        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&vz);
        EXPECT_COMPLEX_EQ_VEC(nz, refz, pz);
    }
#endif

    /*
     * Test "dot-product" intrinsic on two
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L1, kt_dot_p_256)
    {
        size_t                         ns   = tsz_v<bsz::b256, float>;
        size_t                         nd   = tsz_v<bsz::b256, double>;
        avxvector_t<bsz::b256, float>  s1   = kt_loadu_p<bsz::b256, float>(&D.vs[3]);
        avxvector_t<bsz::b256, float>  s2   = kt_loadu_p<bsz::b256, float>(&D.vs[5]);
        avxvector_t<bsz::b256, double> d1   = kt_loadu_p<bsz::b256, double>(&D.vd[1]);
        avxvector_t<bsz::b256, double> d2   = kt_loadu_p<bsz::b256, double>(&D.vd[2]);
        float                          refs = 0.0f;
        double                         refd = 0.0;

        float sdot = kt_dot_p<bsz::b256, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[3 + i] * D.vs[5 + i];
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_dot_p<bsz::b256, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[1 + i] * D.vd[2 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);

        size_t                          nc   = tsz_v<bsz::b256, cfloat>;
        size_t                          nz   = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, cfloat>  c1   = kt_loadu_p<bsz::b256, cfloat>(&D.vc[3]);
        avxvector_t<bsz::b256, cfloat>  c2   = kt_loadu_p<bsz::b256, cfloat>(&D.vc[5]);
        avxvector_t<bsz::b256, cdouble> z1   = kt_loadu_p<bsz::b256, cdouble>(&D.vz[2]);
        avxvector_t<bsz::b256, cdouble> z2   = kt_loadu_p<bsz::b256, cdouble>(&D.vz[4]);
        cfloat                          refc = {0.0f, 0.0f};
        cdouble                         refz = {0.0, 0.0};

        cfloat cdot = kt_dot_p<bsz::b256, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * D.vc[5 + i];
        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_dot_p<bsz::b256, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[2 + i] * D.vz[4 + i];
        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }

#ifdef __AVX512F__
    TEST(KT_L1, kt_dot_p_512)
    {
        const size_t                   ns   = tsz_v<bsz::b512, float>;
        const size_t                   nd   = tsz_v<bsz::b512, double>;
        avxvector_t<bsz::b512, float>  s1   = kt_loadu_p<bsz::b512, float>(&D.vs[0]);
        avxvector_t<bsz::b512, float>  s2   = kt_loadu_p<bsz::b512, float>(&D.vs[0]);
        avxvector_t<bsz::b512, double> d1   = kt_loadu_p<bsz::b512, double>(&D.vd[0]);
        avxvector_t<bsz::b512, double> d2   = kt_loadu_p<bsz::b512, double>(&D.vd[0]);
        float                          refs = 0.0f;
        double                         refd = 0.0;

        float sdot = kt_dot_p<bsz::b512, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[0 + i] * D.vs[0 + i];
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_dot_p<bsz::b512, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[0 + i] * D.vd[0 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);

        const size_t                    nc   = tsz_v<bsz::b512, cfloat>;
        const size_t                    nz   = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, cfloat>  c1   = kt_loadu_p<bsz::b512, cfloat>(&D.vc[3]);
        avxvector_t<bsz::b512, cfloat>  c2   = kt_loadu_p<bsz::b512, cfloat>(&D.vc[5]);
        avxvector_t<bsz::b512, cdouble> z1   = kt_loadu_p<bsz::b512, cdouble>(&D.vz[1]);
        avxvector_t<bsz::b512, cdouble> z2   = kt_loadu_p<bsz::b512, cdouble>(&D.vz[0]);
        cfloat                          refc = {0.0f, 0.0f};
        cdouble                         refz = {0.0, 0.0};

        cfloat cdot = kt_dot_p<bsz::b512, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * D.vc[5 + i];
        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_dot_p<bsz::b512, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[1 + i] * D.vz[0 + i];
        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }
#endif

    /*
     * Test "complex dot-product" intrinsic on two
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L1, kt_cdot_p_256)
    {
        size_t                         ns   = tsz_v<bsz::b256, float>;
        size_t                         nd   = tsz_v<bsz::b256, double>;
        avxvector_t<bsz::b256, float>  s1   = kt_loadu_p<bsz::b256, float>(&D.vs[3]);
        avxvector_t<bsz::b256, float>  s2   = kt_loadu_p<bsz::b256, float>(&D.vs[5]);
        avxvector_t<bsz::b256, double> d1   = kt_loadu_p<bsz::b256, double>(&D.vd[3]);
        avxvector_t<bsz::b256, double> d2   = kt_loadu_p<bsz::b256, double>(&D.vd[2]);
        float                          refs = 0.0f;
        double                         refd = 0.0;

        float sdot = kt_cdot_p<bsz::b256, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[3 + i] * D.vs[5 + i];
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_cdot_p<bsz::b256, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[3 + i] * D.vd[2 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);

        size_t                          nc   = tsz_v<bsz::b256, cfloat>;
        size_t                          nz   = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, cfloat>  c1   = kt_loadu_p<bsz::b256, cfloat>(&D.vc[3]);
        avxvector_t<bsz::b256, cfloat>  c2   = kt_loadu_p<bsz::b256, cfloat>(&D.vc[5]);
        avxvector_t<bsz::b256, cdouble> z1   = kt_loadu_p<bsz::b256, cdouble>(&D.vz[3]);
        avxvector_t<bsz::b256, cdouble> z2   = kt_loadu_p<bsz::b256, cdouble>(&D.vz[4]);
        cfloat                          refc = {0.0f, 0.0f};
        cdouble                         refz = {0.0, 0.0};

        cfloat cdot = kt_cdot_p<bsz::b256, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * conj(D.vc[5 + i]);

        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_cdot_p<bsz::b256, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[3 + i] * conj(D.vz[4 + i]);

        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }

#ifdef __AVX512F__
    TEST(KT_L1, kt_cdot_p_512)
    {
        size_t                         ns   = tsz_v<bsz::b512, float>;
        size_t                         nd   = tsz_v<bsz::b512, double>;
        avxvector_t<bsz::b512, float>  s1   = kt_loadu_p<bsz::b512, float>(&D.vs[0]);
        avxvector_t<bsz::b512, float>  s2   = kt_loadu_p<bsz::b512, float>(&D.vs[1]);
        avxvector_t<bsz::b512, double> d1   = kt_loadu_p<bsz::b512, double>(&D.vd[2]);
        avxvector_t<bsz::b512, double> d2   = kt_loadu_p<bsz::b512, double>(&D.vd[0]);
        float                          refs = 0.0f;
        double                         refd = 0.0;

        float sdot = kt_cdot_p<bsz::b512, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[0 + i] * D.vs[1 + i];
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_cdot_p<bsz::b512, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[2 + i] * D.vd[0 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);

        size_t                          nc   = tsz_v<bsz::b512, cfloat>;
        size_t                          nz   = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, cfloat>  c1   = kt_loadu_p<bsz::b512, cfloat>(&D.vc[3]);
        avxvector_t<bsz::b512, cfloat>  c2   = kt_loadu_p<bsz::b512, cfloat>(&D.vc[5]);
        avxvector_t<bsz::b512, cdouble> z1   = kt_loadu_p<bsz::b512, cdouble>(&D.vz[3]);
        avxvector_t<bsz::b512, cdouble> z2   = kt_loadu_p<bsz::b512, cdouble>(&D.vz[2]);
        cfloat                          refc = {0.0f, 0.0f};
        cdouble                         refz = {0.0, 0.0};

        cfloat cdot = kt_cdot_p<bsz::b512, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * conj(D.vc[5 + i]);

        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_cdot_p<bsz::b512, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[3 + i] * conj(D.vz[2 + i]);

        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }
#endif
    /*
        Test "store" KT to store elements to memory
    */
    TEST(KT_L0, kt_storeu_p_256)
    {
        const size_t                    ns = tsz_v<bsz::b256, float>;
        const size_t                    nd = tsz_v<bsz::b256, double>;
        const size_t                    nc = tsz_v<bsz::b256, cfloat>;
        const size_t                    nz = tsz_v<bsz::b256, cdouble>;
        avxvector_t<bsz::b256, float>   s;
        avxvector_t<bsz::b256, double>  d;
        avxvector_t<bsz::b256, cfloat>  c;
        avxvector_t<bsz::b256, cdouble> z;
        float                           refs[ns], vs[ns];
        double                          refd[nd], vd[nd];
        cfloat                          refc[nc], vc[nc];
        cdouble                         refz[nz], vz[nz];

        s = kt_loadu_p<bsz::b256, float>(D.vs);
        kt_storeu_p<bsz::b256, float>(vs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i];
        }
        EXPECT_EQ_VEC(ns, vs, refs);

        d = kt_loadu_p<bsz::b256, double>(D.vd);
        kt_storeu_p<bsz::b256, double>(vd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i];
        }
        EXPECT_EQ_VEC(nd, vd, refd);

        c = kt_loadu_p<bsz::b256, cfloat>(D.vc);
        kt_storeu_p<bsz::b256, cfloat>(vc, c);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[i];
        }
        EXPECT_EQ_VEC(nc, vc, refc);

        z = kt_loadu_p<bsz::b256, cdouble>(D.vz);
        kt_storeu_p<bsz::b256, cdouble>(vz, z);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[i];
        }
        EXPECT_EQ_VEC(nz, vz, refz);
    }

#ifdef __AVX512F__
    TEST(KT_L0, kt_storeu_p_512)
    {
        constexpr size_t                ns = tsz_v<bsz::b512, float>;
        constexpr size_t                nd = tsz_v<bsz::b512, double>;
        constexpr size_t                nc = tsz_v<bsz::b512, cfloat>;
        constexpr size_t                nz = tsz_v<bsz::b512, cdouble>;
        avxvector_t<bsz::b512, float>   s;
        avxvector_t<bsz::b512, double>  d;
        avxvector_t<bsz::b512, cfloat>  c;
        avxvector_t<bsz::b512, cdouble> z;
        float                           refs[ns];
        float                           vss[ns];
        double                          refd[nd];
        double                          vdd[nd];
        cfloat                          vcc[nc];
        cdouble                         refz[nz];
        cdouble                         vzz[nz];

        cfloat *refc = new cfloat
            [nc]; // Used to dynamic memory allocation to suppress a potential compiler bug

        s = kt_loadu_p<bsz::b512, float>(&D.vs[3]);
        kt_storeu_p<bsz::b512, float>(vss, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i + 3];
        }
        EXPECT_EQ_VEC(ns, vss, refs);

        d = kt_loadu_p<bsz::b512, double>(&D.vd[3]);
        kt_storeu_p<bsz::b512, double>(vdd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i + 3];
        }
        EXPECT_EQ_VEC(nd, vdd, refd);

        c = kt_loadu_p<bsz::b512, cfloat>(&D.vc[3]);
        kt_storeu_p<bsz::b512, cfloat>(vcc, c);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[i + 3];
        }
        EXPECT_EQ_VEC(nc, vcc, refc);

        z = kt_loadu_p<bsz::b512, cdouble>(&D.vz[3]);
        kt_storeu_p<bsz::b512, cdouble>(vzz, z);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[i + 3];
        }
        EXPECT_EQ_VEC(nz, vzz, refz);

        delete[] refc;
    }
#endif
}
