/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse_kernel_templates.hpp"

#include <complex>
#include <iostream>
#include <typeinfo>

#define kt_int aoclsparse_int

using namespace kernel_templates;
using namespace std::complex_literals;

namespace TestsKT
{
    class KTTCommonData
    {
    public:
        size_t               map[32]{3,  1,  0,  2,  4,  7,  5,  6,  8,  15, 13, 9,  11, 10, 12, 14,
                                     29, 19, 18, 31, 20, 23, 22, 25, 24, 16, 21, 26, 27, 30, 28, 17};
        float                vs[32]{1.531f, 2.753f, 2.734f, 3.870f, 3.849f, 2.586f, 4.056f, 1.143f,
                                    2.276f, 1.437f, 3.003f, 1.972f, 3.851f, 1.793f, 3.357f, 1.221f,
                                    3.511f, 1.713f, 1.133f, 1.170f, 1.189f, 1.226f, 2.033f, 2.114f,
                                    1.766f, 1.447f, 2.382f, 5.192f, 4.172f, 3.173f, 1.735f, 3.031f};
        double               vd[16]{1.131,
                                    2.764,
                                    3.349,
                                    2.556,
                                    1.376,
                                    3.903,
                                    2.293,
                                    1.921,
                                    1.291,
                                    3.724,
                                    1.939,
                                    2.561,
                                    1.237,
                                    3.913,
                                    6.274,
                                    0.931};
        std::complex<float>  vc[16]{.11f - 1if,
                                    .22f - 2if,
                                    .33f - 3if,
                                    .44f - 4if,
                                    .55f - 5if,
                                    .66f - 6if,
                                    .77f - 7if,
                                    .88f - 8if,
                                    -.11f - 1if,
                                    -.22f - 2if,
                                    -.33f - 3if,
                                    -.44f - 4if,
                                    -.55f - 5if,
                                    -.66f - 6if,
                                    -.77f - 7if,
                                    -.88f - 8if};
        std::complex<double> vz[8]{.15 - 12i,
                                   .26 - 21i,
                                   .37 - 13i,
                                   .54 - 4.5i,
                                   .15 - 5.1i,
                                   .67 - 6.7i,
                                   .74 - 7.2i,
                                   .81 - 8.1i};
    };

    const KTTCommonData D;

    TEST(KT_L0, KT_IS_SAME)
    {
        EXPECT_TRUE((kt_is_same<256, 256, double, double>()));
        EXPECT_FALSE((kt_is_same<256, 256, double, cdouble>()));
        EXPECT_FALSE((kt_is_same<256, 512, double, double>()));
        EXPECT_FALSE((kt_is_same<256, 512, float, cfloat>()));
    }

    TEST(KT_L0, KT_TYPES_256)
    {
        /*
         * These 256 bit tests check that the correct data type
         * and packed sizes are "returned".
         */
        // 256 float
        EXPECT_EQ(typeid(avxvector<256, float>::type), typeid(__m256));
        EXPECT_EQ(typeid(avxvector<256, float>::half_type), typeid(__m128));
        EXPECT_EQ((avxvector<256, float>::value), 8U);
        EXPECT_EQ((avxvector<256, float>()), 8U);
        EXPECT_EQ((avxvector<256, float>::half), 4U);
        EXPECT_EQ((avxvector<256, float>::count), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<256, float>), typeid(__m256));
        EXPECT_EQ(typeid(avxvector_half_t<256, float>), typeid(__m128));
        EXPECT_EQ((avxvector_v<256, float>), 8U);
        EXPECT_EQ((avxvector<256, float>()), 8U);
        EXPECT_EQ((avxvector_half_v<256, float>), 4U);
        EXPECT_EQ((hsz_v<256, float>), 4U);
        EXPECT_EQ((tsz_v<256, float>), 8U);

        // 256 double
        EXPECT_EQ(typeid(avxvector<256, double>::type), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector<256, double>::half_type), typeid(__m128d));
        EXPECT_EQ((avxvector<256, double>::value), 4U);
        EXPECT_EQ((avxvector<256, double>()), 4U);
        EXPECT_EQ((avxvector<256, double>::half), 2U);
        EXPECT_EQ((avxvector<256, double>::count), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<256, double>), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector_half_t<256, double>), typeid(__m128d));
        EXPECT_EQ((avxvector_v<256, double>), 4U);
        EXPECT_EQ((avxvector<256, double>()), 4U);
        EXPECT_EQ((avxvector_half_v<256, double>), 2U);
        EXPECT_EQ((hsz_v<256, double>), 2U);
        EXPECT_EQ((tsz_v<256, double>), 4U);
    }
#ifdef USE_AVX512
    TEST(KT_L0, KT_TYPES_512)
    {
        /*
         * These 512 bit tests check that the correct data type
         * and packed sizes are "returned".
         */
        // 512 float
        EXPECT_EQ(typeid(avxvector<512, float>::type), typeid(__m512));
        EXPECT_EQ(typeid(avxvector<512, float>::half_type), typeid(__m256));
        EXPECT_EQ((avxvector<512, float>::value), 16U);
        EXPECT_EQ((avxvector<512, float>()), 16U);
        EXPECT_EQ((avxvector<512, float>::half), 8U);
        EXPECT_EQ((avxvector<512, float>::count), 16U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<512, float>), typeid(__m512));
        EXPECT_EQ(typeid(avxvector_half_t<512, float>), typeid(__m256));
        EXPECT_EQ((avxvector_v<512, float>), 16U);
        EXPECT_EQ((avxvector<512, float>()), 16U);
        EXPECT_EQ((avxvector_half_v<512, float>), 8U);
        EXPECT_EQ((hsz_v<512, float>), 8U);
        EXPECT_EQ((tsz_v<512, float>), 16U);

        // 512 double
        EXPECT_EQ(typeid(avxvector<512, double>::type), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector<512, double>::half_type), typeid(__m256d));
        EXPECT_EQ((avxvector<512, double>::value), 8U);
        EXPECT_EQ((avxvector<512, double>()), 8U);
        EXPECT_EQ((avxvector<512, double>::half), 4U);
        EXPECT_EQ((avxvector<512, double>::count), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<512, double>), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector_half_t<512, double>), typeid(__m256d));
        EXPECT_EQ((avxvector_v<512, double>), 8U);
        EXPECT_EQ((avxvector<512, double>()), 8U);
        EXPECT_EQ((avxvector_half_v<512, double>), 4U);
        EXPECT_EQ((hsz_v<512, double>), 4U);
        EXPECT_EQ((tsz_v<512, double>), 8U);
    }
#endif

    TEST(KT_L0, KT_CTYPES_256)
    {
        /*
         * These 256 bit tests check that the correct complex data type
         * and packed sizes are "returned".
         */
        // 256 cfloat
        EXPECT_EQ(typeid(avxvector<256, cfloat>::type), typeid(__m256));
        EXPECT_EQ(typeid(avxvector<256, cfloat>::half_type), typeid(__m128));
        EXPECT_EQ((avxvector<256, cfloat>::value), 8U);
        EXPECT_EQ((avxvector<256, cfloat>()), 8U);
        EXPECT_EQ((avxvector<256, cfloat>::half), 4U);
        EXPECT_EQ((avxvector<256, cfloat>::count), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<256, cfloat>), typeid(__m256));
        EXPECT_EQ(typeid(avxvector_half_t<256, cfloat>), typeid(__m128));
        EXPECT_EQ((avxvector_v<256, cfloat>), 8U);
        EXPECT_EQ((avxvector<256, cfloat>()), 8U);
        EXPECT_EQ((avxvector_half_v<256, cfloat>), 4U);
        EXPECT_EQ((hsz_v<256, cfloat>), 4U);
        EXPECT_EQ((tsz_v<256, cfloat>), 4U);

        // 256 cdouble
        EXPECT_EQ(typeid(avxvector<256, cdouble>::type), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector<256, cdouble>::half_type), typeid(__m128d));
        EXPECT_EQ((avxvector<256, cdouble>::value), 4U);
        EXPECT_EQ((avxvector<256, cdouble>()), 4U);
        EXPECT_EQ((avxvector<256, cdouble>::half), 2U);
        EXPECT_EQ((avxvector<256, cdouble>::count), 2U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<256, cdouble>), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector_half_t<256, cdouble>), typeid(__m128d));
        EXPECT_EQ((avxvector_v<256, cdouble>), 4U);
        EXPECT_EQ((avxvector<256, cdouble>()), 4U);
        EXPECT_EQ((avxvector_half_v<256, cdouble>), 2U);
        EXPECT_EQ((hsz_v<256, cdouble>), 2U);
        EXPECT_EQ((tsz_v<256, cdouble>), 2U);
    }
#ifdef USE_AVX512
    TEST(KT_L0, KT_CTYPES_512)
    {
        /*
         * These 512 bit tests check that the correct complex data type
         * and packed sizes are "returned".
         */
        // 512 cfloat
        EXPECT_EQ(typeid(avxvector<512, cfloat>::type), typeid(__m512));
        EXPECT_EQ(typeid(avxvector<512, cfloat>::half_type), typeid(__m256));
        EXPECT_EQ((avxvector<512, cfloat>::value), 16U);
        EXPECT_EQ((avxvector<512, cfloat>()), 16U);
        EXPECT_EQ((avxvector<512, cfloat>::half), 8U);
        EXPECT_EQ((avxvector<512, cfloat>::count), 8U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<512, cfloat>), typeid(__m512));
        EXPECT_EQ(typeid(avxvector_half_t<512, cfloat>), typeid(__m256));
        EXPECT_EQ((avxvector_v<512, cfloat>), 16U);
        EXPECT_EQ((avxvector<512, cfloat>()), 16U);
        EXPECT_EQ((avxvector_half_v<512, cfloat>), 8U);
        EXPECT_EQ((hsz_v<512, cfloat>), 8U);
        EXPECT_EQ((tsz_v<512, cfloat>), 8U);

        // 512 cdouble
        EXPECT_EQ(typeid(avxvector<512, cdouble>::type), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector<512, cdouble>::half_type), typeid(__m256d));
        EXPECT_EQ((avxvector<512, cdouble>::value), 8U);
        EXPECT_EQ((avxvector<512, cdouble>()), 8U);
        EXPECT_EQ((avxvector<512, cdouble>::half), 4U);
        EXPECT_EQ((avxvector<512, cdouble>::count), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<512, cdouble>), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector_half_t<512, cdouble>), typeid(__m256d));
        EXPECT_EQ((avxvector_v<512, cdouble>), 8U);
        EXPECT_EQ((avxvector<512, cdouble>()), 8U);
        EXPECT_EQ((avxvector_half_v<512, cdouble>), 4U);
        EXPECT_EQ((hsz_v<512, cdouble>), 4U);
        EXPECT_EQ((tsz_v<512, cdouble>), 4U);
    }
#endif

    /*
     * Test loadu intrinsic to load 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_loadu_p_256)
    {
        EXPECT_EQ_VEC((tsz_v<256, float>), (kt_loadu_p<256, float>(&D.vs[3])), &D.vs[3]);
        EXPECT_EQ_VEC((tsz_v<256, double>), (kt_loadu_p<256, double>(&D.vd[5])), &D.vd[5]);

        std::complex<float>     *c;
        avxvector_t<256, cfloat> w;
        w = kt_loadu_p<256, cfloat>(&D.vc[5]);
        c = reinterpret_cast<std::complex<float> *>(&w);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<256, cfloat>), c, &D.vc[5]);

        std::complex<double>     *z;
        avxvector_t<256, cdouble> v;
        v = kt_loadu_p<256, cdouble>(&D.vz[1]);
        z = reinterpret_cast<std::complex<double> *>(&v);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<256, cdouble>), z, &D.vz[1]);
    }
#ifdef USE_AVX512
    TEST(KT_L0, kt_loadu_p_512)
    {
        EXPECT_EQ_VEC((tsz_v<512, float>), (kt_loadu_p<512, float>(&D.vs[0])), &D.vs[0]);
        EXPECT_EQ_VEC((tsz_v<512, double>), (kt_loadu_p<512, double>(&D.vd[0])), &D.vd[0]);

        std::complex<float>     *c;
        avxvector_t<512, cfloat> w;
        w = kt_loadu_p<512, cfloat>(&D.vc[0]);
        c = reinterpret_cast<std::complex<float> *>(&w);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<512, cfloat>), c, &D.vc[0]);

        std::complex<double>     *z;
        avxvector_t<512, cdouble> v;
        v = kt_loadu_p<512, cdouble>(&D.vz[0]);
        z = reinterpret_cast<std::complex<double> *>(&v);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<512, cdouble>), z, &D.vz[0]);
    }
#endif

    /*
     * Test setzero intrinsic to zero-out 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_setzero_p_256)
    {
        const size_t ns        = tsz_v<256, float>;
        const size_t nd        = tsz_v<256, double>;
        const float  szero[ns] = {0.f};
        const double dzero[nd] = {0.0};

        EXPECT_EQ_VEC(ns, (kt_setzero_p<256, float>()), szero);
        EXPECT_EQ_VEC(nd, (kt_setzero_p<256, double>()), dzero);

        // Complex checks are reinterpreted as double-packed reals
        EXPECT_EQ_VEC((tsz_v<256, cfloat>), (kt_setzero_p<256, cfloat>()), szero);
        EXPECT_EQ_VEC((tsz_v<256, cdouble>), (kt_setzero_p<256, cdouble>()), dzero);
    }
#ifdef USE_AVX512
    TEST(KT_L0, kt_setzero_p_512)
    {
        const size_t ns        = tsz_v<512, float>;
        const size_t nd        = tsz_v<512, double>;
        const float  szero[ns] = {0.f};
        const double dzero[nd] = {0.0};

        EXPECT_EQ_VEC((tsz_v<512, float>), (kt_setzero_p<512, float>()), szero);
        EXPECT_EQ_VEC((tsz_v<512, double>), (kt_setzero_p<512, double>()), dzero);

        // Complex checks are reinterpreted as double-packed reals
        EXPECT_EQ_VEC((tsz_v<512, cfloat>), (kt_setzero_p<512, cfloat>()), szero);
        EXPECT_EQ_VEC((tsz_v<512, cdouble>), (kt_setzero_p<512, cdouble>()), dzero);
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

        EXPECT_EQ_VEC((tsz_v<256, float>), (kt_set1_p<256, float>(4.0f)), refs);
        EXPECT_EQ_VEC((tsz_v<256, double>), (kt_set1_p<256, double>(5.0)), refd);

        std::complex<float>     *c;
        avxvector_t<256, cfloat> vc;
        vc = kt_set1_p<256, cfloat>(3.f + 4.0if);
        c  = reinterpret_cast<std::complex<float> *>(&vc);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<256, cfloat>), c, refc);

        std::complex<double>     *z;
        avxvector_t<256, cdouble> vz;
        vz = kt_set1_p<256, cdouble>(2. + 5.0i);
        z  = reinterpret_cast<std::complex<double> *>(&vz);
        EXPECT_EQ_VEC((tsz_v<256, cdouble>), z, refz);
    }
#ifdef USE_AVX512
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

        EXPECT_EQ_VEC((tsz_v<512, float>), (kt_set1_p<512, float>(4.0f)), refs);
        EXPECT_EQ_VEC((tsz_v<512, double>), (kt_set1_p<512, double>(5.0)), refd);

        std::complex<float>     *c;
        avxvector_t<512, cfloat> vc;
        vc = kt_set1_p<512, cfloat>(2.1f + 4.5if);
        c  = reinterpret_cast<std::complex<float> *>(&vc);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<512, cfloat>), c, refc);

        std::complex<double>     *z;
        avxvector_t<512, cdouble> vz;
        vz = kt_set1_p<512, cdouble>(2.2 + 5.1i);
        z  = reinterpret_cast<std::complex<double> *>(&vz);
        EXPECT_EQ_VEC((tsz_v<512, cdouble>), z, refz);
    }
#endif

    /*
     * Test add intrinsic to sum two: 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_add_p_256)
    {
        const size_t             ns = tsz_v<256, float>;
        const size_t             nd = tsz_v<256, double>;
        avxvector_t<256, float>  s, as, bs;
        avxvector_t<256, double> d, ad, bd;
        float                    refs[8];
        double                   refd[4];

        as = kt_loadu_p<256, float>(&D.vs[2]);
        bs = kt_set1_p<256, float>(1.0);
        s  = kt_add_p(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[2 + i] + 1.0f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<256, double>(&D.vd[2]);
        bd = kt_set1_p<256, double>(1.0);
        d  = kt_add_p(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[2 + i] + 1.0;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex
        const size_t              nc = tsz_v<256, cfloat>;
        const size_t              nz = tsz_v<256, cdouble>;
        avxvector_t<256, cfloat>  c, ac, bc;
        avxvector_t<256, cdouble> z, az, bz;
        cfloat                    refc[4];
        cdouble                   refz[2];
        std::complex<float>      *pc;
        std::complex<double>     *pz;

        ac = kt_loadu_p<256, cfloat>(&D.vc[2]);
        bc = kt_set1_p<256, cfloat>(1.0f + 5.if);
        c  = kt_add_p(ac, bc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] + (1.0f + 5.if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&c);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        az = kt_loadu_p<256, cdouble>(&D.vz[2]);
        bz = kt_set1_p<256, cdouble>(3.0 + 5.5i);
        z  = kt_add_p(az, bz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] + (3.0 + 5.5i);
        }
        pz = reinterpret_cast<std::complex<double> *>(&z);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_add_p_512)
    {
        const size_t             ns = tsz_v<512, float>;
        const size_t             nd = tsz_v<512, double>;
        avxvector_t<512, float>  s, as, bs;
        avxvector_t<512, double> d, ad, bd;
        float                    refs[16];
        double                   refd[8];

        as = kt_loadu_p<512, float>(&D.vs[0]);
        bs = kt_set1_p<512, float>(1.0f);
        s  = kt_add_p(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] + 1.0f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<512, double>(&D.vd[0]);
        bd = kt_set1_p<512, double>(1.0);
        d  = kt_add_p(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] + 1.0;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex type testing
        size_t                    nc = tsz_v<512, cfloat>;
        size_t                    nz = tsz_v<512, cdouble>;
        avxvector_t<512, cfloat>  c, ac, bc;
        avxvector_t<512, cdouble> z, az, bz;
        cfloat                    refc[8];
        cdouble                   refz[4];
        std::complex<float>      *pc;
        std::complex<double>     *pz;

        ac = kt_loadu_p<512, cfloat>(&D.vc[2]);
        bc = kt_set1_p<512, cfloat>(1.0f + 5.if);
        c  = kt_add_p(ac, bc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] + (1.0f + 5.if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&c);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        az = kt_loadu_p<512, cdouble>(&D.vz[2]);
        bz = kt_set1_p<512, cdouble>(3.0 + 5.5i);
        z  = kt_add_p(az, bz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] + (3.0 + 5.5i);
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
        size_t                   ns = tsz_v<256, float>;
        size_t                   nd = tsz_v<256, double>;
        avxvector_t<256, float>  s, as, bs;
        avxvector_t<256, double> d, ad, bd;
        float                    refs[8];
        double                   refd[4];

        as = kt_loadu_p<256, float>(&D.vs[2]);
        bs = kt_set1_p<256, float>(2.7f);
        s  = kt_mul_p<256, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[2 + i] * 2.7f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<256, double>(&D.vd[2]);
        bd = kt_set1_p<256, double>(2.7);
        d  = kt_mul_p<256, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[2 + i] * 2.7;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // complex<float> mul_p
        const size_t             nc = tsz_v<256, cfloat>;
        avxvector_t<256, cfloat> ac = kt_loadu_p<256, cfloat>(&D.vc[2]);
        avxvector_t<256, cfloat> bc = kt_set1_p<256, cfloat>(2.6f + 4.4if);
        avxvector_t<256, cfloat> sc = kt_mul_p<256, cfloat>(ac, bc);
        std::complex<float>      refc[nc];
        std::complex<float>     *pc;
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * (2.6f + 4.4if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // complex<double> mul_p
        const size_t              nz = tsz_v<256, cdouble>;
        avxvector_t<256, cdouble> az = kt_loadu_p<256, cdouble>(&D.vz[2]);
        avxvector_t<256, cdouble> bz = kt_set1_p<256, cdouble>(2.7 + 3.5i);
        avxvector_t<256, cdouble> cz = kt_mul_p<256, cdouble>(az, bz);
        std::complex<double>      refz[nz];
        std::complex<double>     *pz;
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * (2.7 + 3.5i);
        }
        pz = reinterpret_cast<std::complex<double> *>(&cz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_mul_p_512)
    {
        size_t                   ns = tsz_v<512, float>;
        size_t                   nd = tsz_v<512, double>;
        avxvector_t<512, float>  s, as, bs;
        avxvector_t<512, double> d, ad, bd;
        float                    refs[16];
        double                   refd[8];

        as = kt_loadu_p<512, float>(&D.vs[0]);
        bs = kt_set1_p<512, float>(3.3f);
        s  = kt_mul_p<512, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * 3.3f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<512, double>(&D.vd[0]);
        bd = kt_set1_p<512, double>(3.3);
        d  = kt_mul_p<512, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * 3.3;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // complex<float> mul_p
        const size_t             nc = tsz_v<512, cfloat>;
        avxvector_t<512, cfloat> ac = kt_loadu_p<512, cfloat>(&D.vc[2]);
        avxvector_t<512, cfloat> bc = kt_set1_p<512, cfloat>(2.6f + 4.4if);
        avxvector_t<512, cfloat> sc = kt_mul_p<512, cfloat>(ac, bc);
        std::complex<float>      refc[nc];
        std::complex<float>     *pc;
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * (2.6f + 4.4if);
        }
        pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // complex<double> mul_p
        avxvector_t<512, cdouble> az = kt_loadu_p<512, cdouble>(&D.vz[2]);
        avxvector_t<512, cdouble> bz = kt_set1_p<512, cdouble>(2.7 + 3.5i);
        avxvector_t<512, cdouble> dz = kt_mul_p<512, cdouble>(az, bz);
        const size_t              nz = tsz_v<512, cdouble>;
        std::complex<double>      refz[nz];
        std::complex<double>     *pz;
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * (2.7 + 3.5i);
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
        const size_t             ns = tsz_v<256, float>;
        const size_t             nd = tsz_v<256, double>;
        avxvector_t<256, float>  s, as, bs;
        avxvector_t<256, double> d, ad, bd;
        float                    refs[8];
        double                   refd[4];

        as = kt_loadu_p<256, float>(&D.vs[2]);
        bs = kt_set_p<256, float>(D.vs, D.map);
        s  = kt_set_p<256, float>(D.vs, &D.map[4]);
        s  = kt_fmadd_p<256, float>(as, bs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[2 + i] * D.vs[D.map[i]] + D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<256, double>(&D.vd[1]);
        bd = kt_set_p<256, double>(D.vd, D.map);
        d  = kt_set_p<256, double>(D.vd, &D.map[2]);
        d  = kt_fmadd_p<256, double>(ad, bd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[1 + i] * D.vd[D.map[i]] + D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex<float> FMADD
        constexpr size_t        nc = tsz_v<256, cfloat>;
        avxvector_t<256, float> sc, ac, bc, cc;
        cfloat                  refc[nc];
        const cfloat            vc1[nc]{1.f + 1.if, 2.f + 3if, 0.f - 4if, 1.2f + 0.6if};
        const cfloat            vc2[nc]{-2.f + 3.if, -1.f - 2if, 3.f + 1if, 2.5f - 1.6if};

        ac = kt_loadu_p<256, cfloat>(&D.vc[2]);
        bc = kt_loadu_p<256, cfloat>(vc1);
        cc = kt_loadu_p<256, cfloat>(vc2);
        sc = kt_fmadd_p<256, cfloat>(ac, bc, cc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * vc1[i] + vc2[i];
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // Complex<double> FMADD
        constexpr size_t         nz = tsz_v<256, cdouble>;
        avxvector_t<256, double> sz, az, bz, cz;
        cdouble                  refz[nz];
        const cdouble            vz1[nz]{1. + 1.i, 2. - 4i};
        const cdouble            vz2[nz]{-3. - 2i, 2.5 + 1.6i};

        az = kt_loadu_p<256, cdouble>(&D.vz[2]);
        bz = kt_loadu_p<256, cdouble>(vz1);
        cz = kt_loadu_p<256, cdouble>(vz2);
        sz = kt_fmadd_p<256, cdouble>(az, bz, cz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * vz1[i] + vz2[i];
        }
        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&sz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_fmadd_p_512)
    {
        size_t                   ns = tsz_v<512, float>;
        size_t                   nd = tsz_v<512, double>;
        avxvector_t<512, float>  s, as, bs;
        avxvector_t<512, double> d, ad, bd;
        float                    refs[ns];
        double                   refd[nd];

        as = kt_loadu_p<512, float>(&D.vs[0]);
        bs = kt_set_p<512, float>(D.vs, D.map);
        s  = kt_set_p<512, float>(D.vs, &D.map[4]);
        s  = kt_fmadd_p<512, float>(as, bs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * D.vs[D.map[i]] + D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<512, double>(&D.vd[0]);
        bd = kt_set_p<512, double>(D.vd, D.map);
        d  = kt_set_p<512, double>(D.vd, &D.map[2]);
        d  = kt_fmadd_p<512, double>(ad, bd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * D.vd[D.map[i]] + D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex<float> FMADD
        constexpr size_t        nc = tsz_v<512, cfloat>;
        avxvector_t<512, float> sc, ac, bc, cc;
        cfloat                  refc[nc];
        const cfloat            vc1[nc]{1.f + 1.if,
                                        2.f + 3if,
                                        0.f - 4if,
                                        1.2f + 0.6if,
                                        2.f + 4.if,
                                        4.f + 5if,
                                        2.f - 4if,
                                        2.4f + 4.6if};
        const cfloat            vc2[nc]{-1.f + 1.if,
                                        -1.f - 2if,
                                        3.f + 1if,
                                        2.5f - 1.6if,
                                        3.f + 3.if,
                                        7.f + 8if,
                                        1.f - 3if,
                                        3.2f + 0.5if};

        ac = kt_loadu_p<512, cfloat>(&D.vc[2]);
        bc = kt_loadu_p<512, cfloat>(vc1);
        cc = kt_loadu_p<512, cfloat>(vc2);
        sc = kt_fmadd_p<512, cfloat>(ac, bc, cc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * vc1[i] + vc2[i];
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // Complex<double> FMADD
        constexpr size_t         nz = tsz_v<512, cdouble>;
        avxvector_t<512, double> sz, az, bz, cz;
        cdouble                  refz[nz];
        const cdouble            vz1[nz]{1. + 1.i, 2. - 4i, 2. + 3.i, 7. - 5i};
        const cdouble            vz2[nz]{-3. - 2i, 2.5 + 1.6i, 1. + 7.i, 5. - 8i};

        az = kt_loadu_p<512, cdouble>(&D.vz[2]);
        bz = kt_loadu_p<512, cdouble>(vz1);
        cz = kt_loadu_p<512, cdouble>(vz2);
        sz = kt_fmadd_p<512, cdouble>(az, bz, cz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * vz1[i] + vz2[i];
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
        const size_t              ns = tsz_v<256, float>;
        const size_t              nd = tsz_v<256, double>;
        const size_t              nc = tsz_v<256, cfloat>;
        const size_t              nz = tsz_v<256, cdouble>;
        avxvector_t<256, float>   s;
        avxvector_t<256, double>  d;
        avxvector_t<256, cfloat>  c;
        avxvector_t<256, cdouble> z;
        float                     refs[ns];
        double                    refd[nd];
        cfloat                    refc[nc];
        cdouble                   refz[nz];
        cfloat                   *pc;
        cdouble                  *pz;

        s = kt_set_p<256, float>(D.vs, &D.map[1]);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[D.map[1 + i]];
        }
        EXPECT_EQ_VEC(ns, s, refs);

        d = kt_set_p<256, double>(D.vd, &D.map[2]);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[D.map[2 + i]];
        }
        EXPECT_EQ_VEC(nd, d, refd);

        c = kt_set_p<256, cfloat>(D.vc, &D.map[5]);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[D.map[5 + i]];
        }
        pc = reinterpret_cast<cfloat *>(&c);
        EXPECT_COMPLEX_EQ_VEC(nc, pc, refc);

        z = kt_set_p<256, cdouble>(D.vz, &D.map[3]);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[D.map[3 + i]];
        }
        pz = reinterpret_cast<cdouble *>(&z);
        EXPECT_COMPLEX_EQ_VEC(nz, pz, refz);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_set_p_512)
    {
        const size_t              ns = tsz_v<512, float>;
        const size_t              nd = tsz_v<512, double>;
        const size_t              nc = tsz_v<512, cfloat>;
        const size_t              nz = tsz_v<512, cdouble>;
        avxvector_t<512, float>   s;
        avxvector_t<512, double>  d;
        avxvector_t<512, cfloat>  c;
        avxvector_t<512, cdouble> z;
        float                     refs[ns];
        double                    refd[nd];
        cfloat                    refc[nc];
        cdouble                   refz[nz];
        cfloat                   *pc;
        cdouble                  *pz;

        s = kt_set_p<512, float>(D.vs, &D.map[2]);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[D.map[2 + i]];
        }
        EXPECT_EQ_VEC(ns, s, refs);

        d = kt_set_p<512, double>(D.vd, &D.map[3]);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[D.map[3 + i]];
        }
        EXPECT_EQ_VEC(nd, d, refd);

        c = kt_set_p<512, cfloat>(D.vc, &D.map[5]);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[D.map[5 + i]];
        }
        pc = reinterpret_cast<cfloat *>(&c);
        EXPECT_COMPLEX_EQ_VEC(nc, pc, refc);

        z = kt_set_p<512, cdouble>(D.vz, &D.map[0]);
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
        // 256/float -> 8
        kt_maskz_set_p_param_dir(256, float, s, AVX, 1, 1);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 2, 0);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 3, 2);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 4, 1);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 5, 0);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 6, 1);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 7, 0);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 8, 1);
        // This must trigger a warning under AVX512F (256 bit __mask8)
        // kt_maskz_set_p_param_dir(256, float, s, AVX, 9);

        // 256 double -> 4
        kt_maskz_set_p_param_dir(256, double, d, AVX, 1, 0);
        kt_maskz_set_p_param_dir(256, double, d, AVX, 2, 1);
        kt_maskz_set_p_param_dir(256, double, d, AVX, 3, 3);
        kt_maskz_set_p_param_dir(256, double, d, AVX, 4, 2);
        // This also triggers a warning
        // kt_maskz_set_p_param_dir(256, double, d, AVX, 5);

        // 256 cfloat -> 4
        kt_maskz_set_p_param_dir(256, cfloat, c, AVX, 1, 1);
        kt_maskz_set_p_param_dir(256, cfloat, c, AVX, 2, 0);
        kt_maskz_set_p_param_dir(256, cfloat, c, AVX, 3, 4);
        kt_maskz_set_p_param_dir(256, cfloat, c, AVX, 4, 0);

        // 256 cdouble -> 2
        kt_maskz_set_p_param_dir(256, cdouble, z, AVX, 1, 0);
        kt_maskz_set_p_param_dir(256, cdouble, z, AVX, 2, 4);
        kt_maskz_set_p_param_dir(256, cdouble, z, AVX, 2, 0);

        // =================================
        // INDIRECT (can have any extension)
        // =================================
        kt_maskz_set_p_param_indir(256, float, s, AVX, 1, 0);
        kt_maskz_set_p_param_indir(256, float, s, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(256, float, s, AVX512VL, 3, 0);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 4, 1);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 5, 0);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 6, 2);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 7, 0);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 8, 0);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 9, 0);

        kt_maskz_set_p_param_indir(256, double, d, AVX, 1, 0);
        kt_maskz_set_p_param_indir(256, double, d, AVX512DQ, 0, 1);
        kt_maskz_set_p_param_indir(256, double, d, AVX, 3, 0);
        kt_maskz_set_p_param_indir(256, double, d, AVX, 4, 3);
        kt_maskz_set_p_param_indir(256, double, d, AVX, 5, 0);

        kt_maskz_set_p_param_indir(256, cfloat, c, AVX, 1, 0);
        kt_maskz_set_p_param_indir(256, cfloat, c, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(256, cfloat, c, AVX512VL, 0, 0);
        kt_maskz_set_p_param_indir(256, cfloat, c, AVX, 4, 1);
        kt_maskz_set_p_param_indir(256, cfloat, c, AVX, 3, 3);

        kt_maskz_set_p_param_indir(256, cdouble, z, AVX, 1, 1);
        kt_maskz_set_p_param_indir(256, cdouble, z, AVX512DQ, 2, 0);
        kt_maskz_set_p_param_indir(256, cdouble, z, AVX, 1, 0);
        kt_maskz_set_p_param_indir(256, cdouble, z, AVX, 2, 2);
        kt_maskz_set_p_param_indir(256, cdouble, z, AVX512DQ, 0, 0);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_maskz_set_p_256_AVX512VL)
    {
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 1, 0);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 2, 5);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 3, 1);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 4, 0);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 5, 1);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 6, 2);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 7, 1);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 8, 0);

        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 1, 1);
        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 2, 0);
        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 3, 1);
        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 4, 2);
        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 5, 1);

        kt_maskz_set_p_param_dir(256, cfloat, c, AVX512VL, 1, 0);
        kt_maskz_set_p_param_dir(256, cfloat, c, AVX512VL, 2, 5);
        kt_maskz_set_p_param_dir(256, cfloat, c, AVX512VL, 3, 2);
        kt_maskz_set_p_param_dir(256, cfloat, c, AVX512VL, 4, 0);

        kt_maskz_set_p_param_dir(256, cdouble, z, AVX512VL, 1, 1);
        kt_maskz_set_p_param_dir(256, cdouble, z, AVX512VL, 2, 0);
        kt_maskz_set_p_param_dir(256, cdouble, z, AVX512VL, 3, 4);
    }
#endif

#ifdef USE_AVX512
    TEST(KT_L0, kt_maskz_set_p_512_AVX512F)
    {
        // Direct
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 1, 0);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 2, 0);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 3, 1);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 4, 1);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 6, 1);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 7, 1);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 8, 1);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 9, 1);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 9, 1);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 10, 0);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 11, 0);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 12, 0);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 13, 0);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 14, 0);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 15, 0);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 16, 0);

        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 1, 3);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 2, 2);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 3, 4);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 4, 0);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 5, 1);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 6, 0);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 7, 0);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 8, 0);

        kt_maskz_set_p_param_dir(512, cfloat, c, AVX512F, 1, 5);
        kt_maskz_set_p_param_dir(512, cfloat, c, AVX512F, 2, 0);
        kt_maskz_set_p_param_dir(512, cfloat, c, AVX512F, 3, 2);
        kt_maskz_set_p_param_dir(512, cfloat, c, AVX512F, 4, 0);

        kt_maskz_set_p_param_dir(512, cdouble, z, AVX512F, 1, 0);
        kt_maskz_set_p_param_dir(512, cdouble, z, AVX512F, 2, 2);
        kt_maskz_set_p_param_dir(512, cdouble, z, AVX512F, 3, 2);

        // Indirect
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 1, 1);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 3, 3);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 4, 0);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 5, 1);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 6, 0);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 7, 0);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 8, 3);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 9, 0);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 10, 0);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 11, 5);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 12, 0);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 13, 2);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 14, 0);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 15, 0);
        kt_maskz_set_p_param_indir(512, float, s, AVX512F, 16, 0);

        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 1, 5);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 2, 3);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 3, 0);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 4, 1);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 5, 2);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 6, 4);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 7, 2);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 8, 0);

        kt_maskz_set_p_param_indir(512, cfloat, c, AVX512F, 1, 0);
        kt_maskz_set_p_param_indir(512, cfloat, c, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(512, cfloat, c, AVX512F, 3, 1);
        kt_maskz_set_p_param_indir(512, cfloat, c, AVX512F, 4, 3);
        kt_maskz_set_p_param_indir(512, cfloat, c, AVX512F, 5, 1);
        kt_maskz_set_p_param_indir(512, cfloat, c, AVX512F, 6, 0);
        kt_maskz_set_p_param_indir(512, cfloat, c, AVX512F, 7, 1);
        kt_maskz_set_p_param_indir(512, cfloat, c, AVX512F, 8, 2);

        kt_maskz_set_p_param_indir(512, cdouble, z, AVX512F, 1, 0);
        kt_maskz_set_p_param_indir(512, cdouble, z, AVX512F, 2, 1);
        kt_maskz_set_p_param_indir(512, cdouble, z, AVX512F, 3, 0);
        kt_maskz_set_p_param_indir(512, cdouble, z, AVX512F, 4, 3);
    }
#endif

    /*
     * Test "hsum" intrinsic to horizontally-reduce via summation
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L0, kt_hsum_p_256)
    {
        const size_t             ns = tsz_v<256, float>;
        const size_t             nd = tsz_v<256, double>;
        avxvector_t<256, float>  vs;
        avxvector_t<256, double> vd;
        float                    sums, refs = 0.0f;
        double                   sumd, refd = 0.0;

        vs   = kt_loadu_p<256, float>(D.vs);
        sums = kt_hsum_p<256, float>(vs);
        for(size_t i = 0; i < ns; i++)
        {
            refs += D.vs[i];
        }
        EXPECT_FLOAT_EQ(sums, refs);

        vd   = kt_loadu_p<256, double>(D.vd);
        sumd = kt_hsum_p<256, double>(vd);
        for(size_t i = 0; i < nd; i++)
        {
            refd += D.vd[i];
        }
        EXPECT_DOUBLE_EQ(sumd, refd);

        const size_t              nc = tsz_v<256, cfloat>;
        const size_t              nz = tsz_v<256, cdouble>;
        avxvector_t<256, cfloat>  vc;
        avxvector_t<256, cdouble> vz;
        cfloat                    sumc, refc = {0.0f, 0.0f};
        cdouble                   sumz, refz = {0.0, 0.0};

        vc   = kt_loadu_p<256, cfloat>(D.vc);
        sumc = kt_hsum_p<256, cfloat>(vc);
        for(size_t i = 0; i < nc; i++)
        {
            refc += D.vc[i];
        }
        EXPECT_COMPLEX_FLOAT_EQ(sumc, refc);

        vz   = kt_loadu_p<256, cdouble>(D.vz);
        sumz = kt_hsum_p<256, cdouble>(vz);
        for(size_t i = 0; i < nz; i++)
        {
            refz += D.vz[i];
        }
        EXPECT_COMPLEX_DOUBLE_EQ(sumz, refz);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_hsum_p_512)
    {
        const size_t             ns = tsz_v<512, float>;
        const size_t             nd = tsz_v<512, double>;
        avxvector_t<512, float>  vs;
        avxvector_t<512, double> vd;
        float                    sums, refs = 0.0f;
        double                   sumd, refd = 0.0;

        vs   = kt_loadu_p<512, float>(D.vs);
        sums = kt_hsum_p<512, float>(vs);
        for(size_t i = 0; i < ns; i++)
        {
            refs += D.vs[i];
        }
        EXPECT_FLOAT_EQ(sums, refs);

        vd   = kt_loadu_p<512, double>(D.vd);
        sumd = kt_hsum_p<512, double>(vd);
        for(size_t i = 0; i < nd; i++)
        {
            refd += D.vd[i];
        }
        EXPECT_DOUBLE_EQ(sumd, refd);

        const size_t              nc = tsz_v<512, cfloat>;
        const size_t              nz = tsz_v<512, cdouble>;
        avxvector_t<512, cfloat>  vc;
        avxvector_t<512, cdouble> vz;
        cfloat                    sumz, refz = {0.0f, 0.0f};
        cdouble                   sumc, refc = {0.0, 0.0};

        vc   = kt_loadu_p<512, cfloat>(D.vc);
        sumz = kt_hsum_p<512, cfloat>(vc);
        for(size_t i = 0; i < nc; i++)
        {
            refz += D.vc[i];
        }
        EXPECT_COMPLEX_FLOAT_EQ(sumz, refz);

        vz   = kt_loadu_p<512, cdouble>(D.vz);
        sumc = kt_hsum_p<512, cdouble>(vz);
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
        constexpr size_t ns = tsz_v<256, float>;
        constexpr size_t nd = tsz_v<256, double>;
        constexpr size_t nz = tsz_v<256, cdouble>;
        constexpr size_t nc = tsz_v<256, cfloat>;

        avxvector_t<256, float>   vs, ress;
        avxvector_t<256, double>  vd, resd;
        avxvector_t<256, cfloat>  vc, resc;
        avxvector_t<256, cdouble> vz, resz;

        float   refs[ns];
        cfloat  refc[nc];
        double  refd[nd];
        cdouble refz[nz];

        // Float
        vs   = kt_loadu_p<256, float>(D.vs);
        ress = kt_conj_p<256, float>(vs);

        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i];
        }

        EXPECT_EQ_VEC(ns, refs, ress);

        // Double
        vd   = kt_loadu_p<256, double>(D.vd);
        resd = kt_conj_p<256, double>(vd);

        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i];
        }

        EXPECT_EQ_VEC(nd, refd, resd);

        // Cfloat
        vc   = kt_loadu_p<256, cfloat>(D.vc);
        resc = kt_conj_p<256, cfloat>(vc);

        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = conj(D.vc[i]);
        }

        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&resc);
        EXPECT_COMPLEX_EQ_VEC(nc, refc, pc);

        // CDouble
        vz   = kt_loadu_p<256, cdouble>(D.vz);
        resz = kt_conj_p<256, cdouble>(vz);

        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = conj(D.vz[i]);
        }

        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&resz);
        EXPECT_COMPLEX_EQ_VEC(nz, refz, pz);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_conj_p_512)
    {
        constexpr size_t          ns = tsz_v<512, float>;
        constexpr size_t          nd = tsz_v<512, double>;
        constexpr size_t          nz = tsz_v<512, cdouble>;
        constexpr size_t          nc = tsz_v<512, cfloat>;
        avxvector_t<512, float>   vs;
        avxvector_t<512, double>  vd;
        avxvector_t<512, cfloat>  vc;
        avxvector_t<512, cdouble> vz;

        float   refs[ns];
        double  refd[nd];
        cfloat  refc[nc];
        cdouble refz[nz];

        // Float
        vs = kt_loadu_p<512, float>(D.vs);
        vs = kt_conj_p<512, float>(vs);

        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i];
        }

        EXPECT_EQ_VEC(ns, refs, vs);

        // Double
        vd = kt_loadu_p<512, double>(D.vd);
        vd = kt_conj_p<512, double>(vd);

        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i];
        }

        EXPECT_EQ_VEC(nd, refd, vd);

        // Cfloat
        vc = kt_loadu_p<512, cfloat>(&D.vc[0]);
        vc = kt_conj_p<512, cfloat>(vc);

        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = conj(D.vc[i]);
        }

        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&vc);
        EXPECT_COMPLEX_EQ_VEC(nc, refc, pc);

        // CDouble
        vz = kt_loadu_p<512, cdouble>(&D.vz[0]);
        vz = kt_conj_p<512, cdouble>(vz);

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
        size_t                   ns   = tsz_v<256, float>;
        size_t                   nd   = tsz_v<256, double>;
        avxvector_t<256, float>  s1   = kt_loadu_p<256, float>(&D.vs[3]);
        avxvector_t<256, float>  s2   = kt_loadu_p<256, float>(&D.vs[5]);
        avxvector_t<256, double> d1   = kt_loadu_p<256, double>(&D.vd[1]);
        avxvector_t<256, double> d2   = kt_loadu_p<256, double>(&D.vd[2]);
        float                    refs = 0.0f;
        double                   refd = 0.0;

        float sdot = kt_dot_p<256, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[3 + i] * D.vs[5 + i];
        EXPECT_FLOAT_EQ(refs, sdot);
        // Test convenience wrapper
        sdot = kt_dot_p(s1, s2);
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_dot_p<256, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[1 + i] * D.vd[2 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);
        // Test convenience wrapper
        ddot = kt_dot_p(d1, d2);
        EXPECT_DOUBLE_EQ(refd, ddot);

        size_t                    nc   = tsz_v<256, cfloat>;
        size_t                    nz   = tsz_v<256, cdouble>;
        avxvector_t<256, cfloat>  c1   = kt_loadu_p<256, cfloat>(&D.vc[3]);
        avxvector_t<256, cfloat>  c2   = kt_loadu_p<256, cfloat>(&D.vc[5]);
        avxvector_t<256, cdouble> z1   = kt_loadu_p<256, cdouble>(&D.vz[2]);
        avxvector_t<256, cdouble> z2   = kt_loadu_p<256, cdouble>(&D.vz[4]);
        cfloat                    refc = {0.0f, 0.0f};
        cdouble                   refz = {0.0, 0.0};

        cfloat cdot = kt_dot_p<256, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * D.vc[5 + i];
        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_dot_p<256, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[2 + i] * D.vz[4 + i];
        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }

#ifdef USE_AVX512
    TEST(KT_L1, kt_dot_p_512)
    {
        const size_t             ns   = tsz_v<512, float>;
        const size_t             nd   = tsz_v<512, double>;
        avxvector_t<512, float>  s1   = kt_loadu_p<512, float>(&D.vs[0]);
        avxvector_t<512, float>  s2   = kt_loadu_p<512, float>(&D.vs[0]);
        avxvector_t<512, double> d1   = kt_loadu_p<512, double>(&D.vd[0]);
        avxvector_t<512, double> d2   = kt_loadu_p<512, double>(&D.vd[0]);
        float                    refs = 0.0f;
        double                   refd = 0.0;

        float sdot = kt_dot_p<512, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[0 + i] * D.vs[0 + i];
        EXPECT_FLOAT_EQ(refs, sdot);
        // Test convenience wrapper
        sdot = kt_dot_p(s1, s2);
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_dot_p<512, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[0 + i] * D.vd[0 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);
        // Test convenience wrapper
        ddot = kt_dot_p(d1, d2);
        EXPECT_DOUBLE_EQ(refd, ddot);

        const size_t              nc   = tsz_v<512, cfloat>;
        const size_t              nz   = tsz_v<512, cdouble>;
        avxvector_t<512, cfloat>  c1   = kt_loadu_p<512, cfloat>(&D.vc[3]);
        avxvector_t<512, cfloat>  c2   = kt_loadu_p<512, cfloat>(&D.vc[5]);
        avxvector_t<512, cdouble> z1   = kt_loadu_p<512, cdouble>(&D.vz[1]);
        avxvector_t<512, cdouble> z2   = kt_loadu_p<512, cdouble>(&D.vz[0]);
        cfloat                    refc = {0.0f, 0.0f};
        cdouble                   refz = {0.0, 0.0};

        cfloat cdot = kt_dot_p<512, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * D.vc[5 + i];
        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);
        // Don't check convenience wrapper

        cdouble zdot = kt_dot_p<512, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[1 + i] * D.vz[0 + i];
        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
        // Don't check convenience wrapper
    }
#endif

    /*
     * Test "complex dot-product" intrinsic on two
     * 2 (cdouble), 4 (cfloat), 4 (double), 8 (floats) length vectors
     */
    TEST(KT_L1, kt_cdot_p_256)
    {
        size_t                   ns   = tsz_v<256, float>;
        size_t                   nd   = tsz_v<256, double>;
        avxvector_t<256, float>  s1   = kt_loadu_p<256, float>(&D.vs[3]);
        avxvector_t<256, float>  s2   = kt_loadu_p<256, float>(&D.vs[5]);
        avxvector_t<256, double> d1   = kt_loadu_p<256, double>(&D.vd[3]);
        avxvector_t<256, double> d2   = kt_loadu_p<256, double>(&D.vd[2]);
        float                    refs = 0.0f;
        double                   refd = 0.0;

        float sdot = kt_cdot_p<256, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[3 + i] * D.vs[5 + i];
        EXPECT_FLOAT_EQ(refs, sdot);
        // Test convenience wrapper
        sdot = kt_dot_p(s1, s2);
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_cdot_p<256, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[3 + i] * D.vd[2 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);
        // Test convenience wrapper
        ddot = kt_dot_p(d1, d2);
        EXPECT_DOUBLE_EQ(refd, ddot);

        size_t                    nc   = tsz_v<256, cfloat>;
        size_t                    nz   = tsz_v<256, cdouble>;
        avxvector_t<256, cfloat>  c1   = kt_loadu_p<256, cfloat>(&D.vc[3]);
        avxvector_t<256, cfloat>  c2   = kt_loadu_p<256, cfloat>(&D.vc[5]);
        avxvector_t<256, cdouble> z1   = kt_loadu_p<256, cdouble>(&D.vz[3]);
        avxvector_t<256, cdouble> z2   = kt_loadu_p<256, cdouble>(&D.vz[4]);
        cfloat                    refc = {0.0f, 0.0f};
        cdouble                   refz = {0.0, 0.0};

        cfloat cdot = kt_cdot_p<256, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * conj(D.vc[5 + i]);

        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_cdot_p<256, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[3 + i] * conj(D.vz[4 + i]);

        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }

#ifdef USE_AVX512
    TEST(KT_L1, kt_cdot_p_512)
    {
        size_t                   ns   = tsz_v<512, float>;
        size_t                   nd   = tsz_v<512, double>;
        avxvector_t<512, float>  s1   = kt_loadu_p<512, float>(&D.vs[0]);
        avxvector_t<512, float>  s2   = kt_loadu_p<512, float>(&D.vs[1]);
        avxvector_t<512, double> d1   = kt_loadu_p<512, double>(&D.vd[2]);
        avxvector_t<512, double> d2   = kt_loadu_p<512, double>(&D.vd[0]);
        float                    refs = 0.0f;
        double                   refd = 0.0;

        float sdot = kt_cdot_p<512, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[0 + i] * D.vs[1 + i];
        EXPECT_FLOAT_EQ(refs, sdot);
        // Test convenience wrapper
        sdot = kt_dot_p(s1, s2);
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_cdot_p<512, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[2 + i] * D.vd[0 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);
        // Test convenience wrapper
        ddot = kt_dot_p(d1, d2);
        EXPECT_DOUBLE_EQ(refd, ddot);

        size_t                    nc   = tsz_v<512, cfloat>;
        size_t                    nz   = tsz_v<512, cdouble>;
        avxvector_t<512, cfloat>  c1   = kt_loadu_p<512, cfloat>(&D.vc[3]);
        avxvector_t<512, cfloat>  c2   = kt_loadu_p<512, cfloat>(&D.vc[5]);
        avxvector_t<512, cdouble> z1   = kt_loadu_p<512, cdouble>(&D.vz[3]);
        avxvector_t<512, cdouble> z2   = kt_loadu_p<512, cdouble>(&D.vz[2]);
        cfloat                    refc = {0.0f, 0.0f};
        cdouble                   refz = {0.0, 0.0};

        cfloat cdot = kt_cdot_p<512, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * conj(D.vc[5 + i]);

        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_cdot_p<512, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[3 + i] * conj(D.vz[2 + i]);

        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }
#endif
}
