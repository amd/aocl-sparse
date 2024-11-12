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
#include "common_data_utils.h"
#include "gtest/gtest.h"

using kt_int_t = size_t;

#include "kernel-templates/kernel_templates.hpp"

#include <complex>
#include <iostream>
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
        std::complex<float>  vc[16]{ {2.25f, -1.5f},  {4.25f, -2.0f},  {6.125f, -3.0f}, {8.25f, -4.0f},
                                     {1.5f, -5.0f},   {3.5f, -6.25f},  {7.75f, -7.0f},  {9.25f, -8.0f},
                                     {-2.5f, -1.5f},  {-3.25f, -2.0f}, {-5.5f, -3.0f},  {-7.25f, -4.0f},
                                     {-9.75f, -5.0f}, {-2.2f, -6.0f},  {-4.75f, -7.5f}, {-6.0f,  -8.125f}};
        std::complex<double> vz[8]{  {1.25, -12},     {0.5, -21.0},    {0.125, -13.0},  {3.5,   -4.5},
                                     {5.25, -8.125},  {8.5, -6.75},    {9.5, -7.25},    {2.125, -3.0}};
        // clang-format on
    };

    const KTTCommonData D;

    // These functions only need to be defined in AVX2 builds
#ifndef __AVX512F__
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
    }

    void kt_is_same_test()
    {
        EXPECT_TRUE((kt_is_same<bsz::b256, bsz::b256, double, double>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b256, double, cdouble>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b512, double, double>()));
        EXPECT_FALSE((kt_is_same<bsz::b256, bsz::b512, float, cfloat>()));
    }
#endif

#ifdef __AVX512F__
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

#else
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
#endif

    template <bsz SZ>
    void kt_loadu_p_test()
    {
        EXPECT_EQ_VEC((tsz_v<SZ, float>), (kt_loadu_p<SZ, float>(&D.vs[0])), &D.vs[0]);
        EXPECT_EQ_VEC((tsz_v<SZ, double>), (kt_loadu_p<SZ, double>(&D.vd[0])), &D.vd[0]);

        std::complex<float>    *c;
        avxvector_t<SZ, cfloat> w;
        w = kt_loadu_p<SZ, cfloat>(&D.vc[0]);
        c = reinterpret_cast<std::complex<float> *>(&w);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<SZ, cfloat>), c, &D.vc[0]);

        std::complex<double>    *z;
        avxvector_t<SZ, cdouble> v;
        v = kt_loadu_p<SZ, cdouble>(&D.vz[0]);
        z = reinterpret_cast<std::complex<double> *>(&v);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<SZ, cdouble>), z, &D.vz[0]);
    }

    template <bsz SZ>
    void kt_setzero_p_test()
    {
        const size_t ns        = tsz_v<SZ, float>;
        const size_t nd        = tsz_v<SZ, double>;
        const float  szero[ns] = {0.f};
        const double dzero[nd] = {0.0};

        EXPECT_EQ_VEC((tsz_v<SZ, float>), (kt_setzero_p<SZ, float>()), szero);
        EXPECT_EQ_VEC((tsz_v<SZ, double>), (kt_setzero_p<SZ, double>()), dzero);

        // Complex checks are reinterpreted as double-packed reals
        EXPECT_EQ_VEC((tsz_v<SZ, cfloat>), (kt_setzero_p<SZ, cfloat>()), szero);
        EXPECT_EQ_VEC((tsz_v<SZ, cdouble>), (kt_setzero_p<SZ, cdouble>()), dzero);
    }

    template <bsz SZ>
    void kt_set1_p_test()
    {

        float refs[]{
            4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f};
        double  refd[]{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
        cfloat  tc(2.1f, 4.5f);
        cdouble tz(2.2f, 5.1f);

        cfloat refc[8];

        for(size_t i = 0; i < 8; ++i)
            refc[i] = tc;

        cdouble refz[4];

        for(size_t i = 0; i < 4; ++i)
            refz[i] = tz;

        EXPECT_EQ_VEC((tsz_v<SZ, float>), (kt_set1_p<SZ, float>(4.0f)), refs);
        EXPECT_EQ_VEC((tsz_v<SZ, double>), (kt_set1_p<SZ, double>(5.0)), refd);

        std::complex<float>    *c;
        avxvector_t<SZ, cfloat> vc;
        vc = kt_set1_p<SZ, cfloat>(tc);
        c  = reinterpret_cast<std::complex<float> *>(&vc);
        // Check in complex space
        EXPECT_EQ_VEC((tsz_v<SZ, cfloat>), c, refc);

        std::complex<double>    *z;
        avxvector_t<SZ, cdouble> vz;
        vz = kt_set1_p<SZ, cdouble>(tz);
        z  = reinterpret_cast<std::complex<double> *>(&vz);
        EXPECT_EQ_VEC((tsz_v<SZ, cdouble>), z, refz);
    }

    template <bsz SZ>
    void kt_add_p_test()
    {
        const size_t            ns = tsz_v<SZ, float>;
        const size_t            nd = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  s, as, bs;
        avxvector_t<SZ, double> d, ad, bd;
        float                   refs[ns];
        double                  refd[nd];

        as = kt_loadu_p<SZ, float>(&D.vs[0]);
        bs = kt_set1_p<SZ, float>(1.0f);
        s  = kt_add_p<SZ, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] + 1.0f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<SZ, double>(&D.vd[0]);
        bd = kt_set1_p<SZ, double>(1.0);
        d  = kt_add_p<SZ, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] + 1.0;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex type testing
        size_t                   nc = tsz_v<SZ, cfloat>;
        size_t                   nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, cfloat>  c, ac, bc;
        avxvector_t<SZ, cdouble> z, az, bz;
        cfloat                   refc[nc];
        cdouble                  refz[nz];
        std::complex<float>     *pc;
        std::complex<double>    *pz;
        std::complex<float>      tc(1.0f, 5.0f);
        std::complex<double>     tz(3.0, 5.5);

        ac = kt_loadu_p<SZ, cfloat>(&D.vc[2]);
        bc = kt_set1_p<SZ, cfloat>(tc);
        c  = kt_add_p<SZ, cfloat>(ac, bc);

        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] + tc;
        }
        pc = reinterpret_cast<std::complex<float> *>(&c);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        az = kt_loadu_p<SZ, cdouble>(&D.vz[2]);
        bz = kt_set1_p<SZ, cdouble>(tz);
        z  = kt_add_p<SZ, cdouble>(az, bz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] + (tz);
        }
        pz = reinterpret_cast<std::complex<double> *>(&z);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

    template <bsz SZ>
    void kt_sub_p_test()
    {
        const size_t            ns = tsz_v<SZ, float>;
        const size_t            nd = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  s, as, bs;
        avxvector_t<SZ, double> d, ad, bd;
        float                   refs[ns];
        double                  refd[nd];

        as = kt_loadu_p<SZ, float>(&D.vs[0]);
        bs = kt_set1_p<SZ, float>(1.0f);
        s  = kt_sub_p<SZ, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] - 1.0f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<SZ, double>(&D.vd[0]);
        bd = kt_set1_p<SZ, double>(1.0);
        d  = kt_sub_p<SZ, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] - 1.0;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex type testing
        size_t                   nc = tsz_v<SZ, cfloat>;
        size_t                   nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, cfloat>  c, ac, bc;
        avxvector_t<SZ, cdouble> z, az, bz;
        cfloat                   refc[8];
        cdouble                  refz[4];
        std::complex<float>     *pc;
        std::complex<double>    *pz;
        std::complex<float>      tc(1.0f, 5.0f);
        std::complex<double>     tz(3.0, 5.5);

        ac = kt_loadu_p<SZ, cfloat>(&D.vc[2]);
        bc = kt_set1_p<SZ, cfloat>(tc);
        c  = kt_sub_p<SZ, cfloat>(ac, bc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] - (tc);
        }
        pc = reinterpret_cast<std::complex<float> *>(&c);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        az = kt_loadu_p<SZ, cdouble>(&D.vz[2]);
        bz = kt_set1_p<SZ, cdouble>(tz);
        z  = kt_sub_p<SZ, cdouble>(az, bz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] - (tz);
        }
        pz = reinterpret_cast<std::complex<double> *>(&z);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

    template <bsz SZ>
    void kt_mul_p_test()
    {
        size_t                  ns = tsz_v<SZ, float>;
        size_t                  nd = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  s, as, bs;
        avxvector_t<SZ, double> d, ad, bd;
        float                   refs[ns];
        double                  refd[nd];

        as = kt_loadu_p<SZ, float>(&D.vs[0]);
        bs = kt_set1_p<SZ, float>(3.3f);
        s  = kt_mul_p<SZ, float>(as, bs);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * 3.3f;
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<SZ, double>(&D.vd[0]);
        bd = kt_set1_p<SZ, double>(3.5);
        d  = kt_mul_p<SZ, double>(ad, bd);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * 3.5;
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // complex<float> mul_p
        const size_t            nc = tsz_v<SZ, cfloat>;
        avxvector_t<SZ, cfloat> ac = kt_loadu_p<SZ, cfloat>(&D.vc[2]);
        avxvector_t<SZ, cfloat> bc = kt_set1_p<SZ, cfloat>(D.vc[0]);
        avxvector_t<SZ, cfloat> sc = kt_mul_p<SZ, cfloat>(ac, bc);
        std::complex<float>     refc[nc];
        std::complex<float>    *pc;
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * D.vc[0];
        }
        pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // complex<double> mul_p
        avxvector_t<SZ, cdouble> az = kt_loadu_p<SZ, cdouble>(&D.vz[2]);
        avxvector_t<SZ, cdouble> bz = kt_set1_p<SZ, cdouble>(D.vz[0]);
        avxvector_t<SZ, cdouble> dz = kt_mul_p<SZ, cdouble>(az, bz);
        const size_t             nz = tsz_v<SZ, cdouble>;
        std::complex<double>     refz[nz];
        std::complex<double>    *pz;
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * D.vz[0];
            ;
        }
        pz = reinterpret_cast<std::complex<double> *>(&dz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz)
    }

    template <bsz SZ>
    void kt_fmadd_p_test()
    {
        size_t                  ns = tsz_v<SZ, float>;
        size_t                  nd = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  s, as, bs;
        avxvector_t<SZ, double> d, ad, bd;
        float                   refs[ns];
        double                  refd[nd];

        as = kt_loadu_p<SZ, float>(&D.vs[0]);
        bs = kt_set_p<SZ, float>(D.vs, D.map);
        s  = kt_set_p<SZ, float>(D.vs, &D.map[4]);
        s  = kt_fmadd_p<SZ, float>(as, bs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * D.vs[D.map[i]] + D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<SZ, double>(&D.vd[0]);
        bd = kt_set_p<SZ, double>(D.vd, D.map);
        d  = kt_set_p<SZ, double>(D.vd, &D.map[2]);
        d  = kt_fmadd_p<SZ, double>(ad, bd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * D.vd[D.map[i]] + D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex<float> FMADD
        constexpr size_t       nc = tsz_v<SZ, cfloat>;
        avxvector_t<SZ, float> sc, ac, bc, cc;
        cfloat                 refc[nc];
        const cfloat          *vc1 = D.vc;
        const cfloat          *vc2 = D.vc + 1;

        ac = kt_loadu_p<SZ, cfloat>(&D.vc[2]);
        bc = kt_loadu_p<SZ, cfloat>(vc1);
        cc = kt_loadu_p<SZ, cfloat>(vc2);
        sc = kt_fmadd_p<SZ, cfloat>(ac, bc, cc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * vc1[i] + vc2[i];
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // Complex<double> FMADD
        constexpr size_t        nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, double> sz, az, bz, cz;
        cdouble                 refz[nz];
        const cdouble          *vz1 = D.vz;
        const cdouble          *vz2 = D.vz + 1;

        az = kt_loadu_p<SZ, cdouble>(&D.vz[2]);
        bz = kt_loadu_p<SZ, cdouble>(vz1);
        cz = kt_loadu_p<SZ, cdouble>(vz2);
        sz = kt_fmadd_p<SZ, cdouble>(az, bz, cz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * vz1[i] + vz2[i];
        }

        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&sz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

    template <bsz SZ>
    void kt_fmsub_p_test()
    {
        size_t                  ns = tsz_v<SZ, float>;
        size_t                  nd = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  s, as, bs;
        avxvector_t<SZ, double> d, ad, bd;
        float                   refs[ns];
        double                  refd[nd];

        as = kt_loadu_p<SZ, float>(&D.vs[0]);
        bs = kt_set_p<SZ, float>(D.vs, D.map);
        s  = kt_set_p<SZ, float>(D.vs, &D.map[4]);
        s  = kt_fmsub_p<SZ, float>(as, bs, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * D.vs[D.map[i]] - D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);

        ad = kt_loadu_p<SZ, double>(&D.vd[0]);
        bd = kt_set_p<SZ, double>(D.vd, D.map);
        d  = kt_set_p<SZ, double>(D.vd, &D.map[2]);
        d  = kt_fmsub_p<SZ, double>(ad, bd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * D.vd[D.map[i]] - D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);

        // Complex<float> FMSUB
        constexpr size_t       nc = tsz_v<SZ, cfloat>;
        avxvector_t<SZ, float> sc, ac, bc, cc;
        cfloat                 refc[nc];
        const cfloat          *vc1 = D.vc;
        const cfloat          *vc2 = D.vc + 1;

        ac = kt_loadu_p<SZ, cfloat>(&D.vc[2]);
        bc = kt_loadu_p<SZ, cfloat>(vc1);
        cc = kt_loadu_p<SZ, cfloat>(vc2);
        sc = kt_fmsub_p<SZ, cfloat>(ac, bc, cc);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[2 + i] * vc1[i] - vc2[i];
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&sc);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        // Complex<double> FMSUB
        constexpr size_t        nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, double> sz, az, bz, cz;
        cdouble                 refz[nz];
        const cdouble          *vz1 = D.vz;
        const cdouble          *vz2 = D.vz + 1;

        az = kt_loadu_p<SZ, cdouble>(&D.vz[2]);
        bz = kt_loadu_p<SZ, cdouble>(vz1);
        cz = kt_loadu_p<SZ, cdouble>(vz2);
        sz = kt_fmsub_p<SZ, cdouble>(az, bz, cz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[2 + i] * vz1[i] - vz2[i];
        }

        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&sz);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
    }

    template <bsz SZ>
    void kt_set_p_test()
    {
        const size_t             ns = tsz_v<SZ, float>;
        const size_t             nd = tsz_v<SZ, double>;
        const size_t             nc = tsz_v<SZ, cfloat>;
        const size_t             nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, float>   s;
        avxvector_t<SZ, double>  d;
        avxvector_t<SZ, cfloat>  c;
        avxvector_t<SZ, cdouble> z;
        float                    refs[ns];
        double                   refd[nd];
        cfloat                   refc[nc];
        cdouble                  refz[nz];
        cfloat                  *pc;
        cdouble                 *pz;

        s = kt_set_p<SZ, float>(D.vs, &D.map[2]);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[D.map[2 + i]];
        }
        EXPECT_EQ_VEC(ns, s, refs);

        d = kt_set_p<SZ, double>(D.vd, &D.map[3]);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[D.map[3 + i]];
        }
        EXPECT_EQ_VEC(nd, d, refd);

        c = kt_set_p<SZ, cfloat>(D.vc, &D.map[5]);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[D.map[5 + i]];
        }
        pc = reinterpret_cast<cfloat *>(&c);
        EXPECT_COMPLEX_EQ_VEC(nc, pc, refc);

        z = kt_set_p<SZ, cdouble>(D.vz, &D.map[0]);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[D.map[0 + i]];
        }
        pz = reinterpret_cast<cdouble *>(&z);
        EXPECT_COMPLEX_EQ_VEC(nz, pz, refz);
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

#ifdef __AVX512F__
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

#else
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
#endif

    template <bsz SZ>
    void kt_hsum_p_test()
    {
        const size_t            ns = tsz_v<SZ, float>;
        const size_t            nd = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  vs;
        avxvector_t<SZ, double> vd;
        float                   sums, refs = 0.0f;
        double                  sumd, refd = 0.0;

        vs   = kt_loadu_p<SZ, float>(D.vs);
        sums = kt_hsum_p<SZ, float>(vs);
        for(size_t i = 0; i < ns; i++)
        {
            refs += D.vs[i];
        }
        EXPECT_FLOAT_EQ(sums, refs);

        vd   = kt_loadu_p<SZ, double>(D.vd);
        sumd = kt_hsum_p<SZ, double>(vd);
        for(size_t i = 0; i < nd; i++)
        {
            refd += D.vd[i];
        }
        EXPECT_DOUBLE_EQ(sumd, refd);

        const size_t             nc = tsz_v<SZ, cfloat>;
        const size_t             nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, cfloat>  vc;
        avxvector_t<SZ, cdouble> vz;
        cfloat                   sumz, refz = {0.0f, 0.0f};
        cdouble                  sumc, refc = {0.0, 0.0};

        vc   = kt_loadu_p<SZ, cfloat>(D.vc);
        sumz = kt_hsum_p<SZ, cfloat>(vc);
        for(size_t i = 0; i < nc; i++)
        {
            refz += D.vc[i];
        }
        EXPECT_COMPLEX_FLOAT_EQ(sumz, refz);

        vz   = kt_loadu_p<SZ, cdouble>(D.vz);
        sumc = kt_hsum_p<SZ, cdouble>(vz);
        for(size_t i = 0; i < nz; i++)
        {
            refc += D.vz[i];
        }
        EXPECT_COMPLEX_DOUBLE_EQ(sumc, refc);
    }

    template <bsz SZ>
    void kt_conj_p_test()
    {

        constexpr size_t         ns = tsz_v<SZ, float>;
        constexpr size_t         nd = tsz_v<SZ, double>;
        constexpr size_t         nz = tsz_v<SZ, cdouble>;
        constexpr size_t         nc = tsz_v<SZ, cfloat>;
        avxvector_t<SZ, float>   vs;
        avxvector_t<SZ, double>  vd;
        avxvector_t<SZ, cfloat>  vc;
        avxvector_t<SZ, cdouble> vz;

        float   refs[ns];
        double  refd[nd];
        cfloat  refc[nc];
        cdouble refz[nz];

        // Float
        vs = kt_loadu_p<SZ, float>(D.vs);
        vs = kt_conj_p<SZ, float>(vs);

        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i];
        }

        EXPECT_EQ_VEC(ns, refs, vs);

        // Double
        vd = kt_loadu_p<SZ, double>(D.vd);
        vd = kt_conj_p<SZ, double>(vd);

        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i];
        }

        EXPECT_EQ_VEC(nd, refd, vd);

        // Cfloat
        vc = kt_loadu_p<SZ, cfloat>(&D.vc[0]);
        vc = kt_conj_p<SZ, cfloat>(vc);

        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = conj(D.vc[i]);
        }

        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&vc);
        EXPECT_COMPLEX_EQ_VEC(nc, refc, pc);

        // CDouble
        vz = kt_loadu_p<SZ, cdouble>(&D.vz[0]);
        vz = kt_conj_p<SZ, cdouble>(vz);

        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = conj(D.vz[i]);
        }

        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&vz);
        EXPECT_COMPLEX_EQ_VEC(nz, refz, pz);
    }

    template <bsz SZ>
    void kt_dot_p_test()
    {
        const size_t            ns   = tsz_v<SZ, float>;
        const size_t            nd   = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  s1   = kt_loadu_p<SZ, float>(&D.vs[0]);
        avxvector_t<SZ, float>  s2   = kt_loadu_p<SZ, float>(&D.vs[0]);
        avxvector_t<SZ, double> d1   = kt_loadu_p<SZ, double>(&D.vd[0]);
        avxvector_t<SZ, double> d2   = kt_loadu_p<SZ, double>(&D.vd[0]);
        float                   refs = 0.0f;
        double                  refd = 0.0;

        float sdot = kt_dot_p<SZ, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[0 + i] * D.vs[0 + i];
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_dot_p<SZ, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[0 + i] * D.vd[0 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);

        const size_t             nc   = tsz_v<SZ, cfloat>;
        const size_t             nz   = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, cfloat>  c1   = kt_loadu_p<SZ, cfloat>(&D.vc[3]);
        avxvector_t<SZ, cfloat>  c2   = kt_loadu_p<SZ, cfloat>(&D.vc[5]);
        avxvector_t<SZ, cdouble> z1   = kt_loadu_p<SZ, cdouble>(&D.vz[1]);
        avxvector_t<SZ, cdouble> z2   = kt_loadu_p<SZ, cdouble>(&D.vz[0]);
        cfloat                   refc = {0.0f, 0.0f};
        cdouble                  refz = {0.0, 0.0};

        cfloat cdot = kt_dot_p<SZ, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * D.vc[5 + i];
        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_dot_p<SZ, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[1 + i] * D.vz[0 + i];
        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }

    template <bsz SZ>
    void kt_cdot_p_test()
    {
        size_t                  ns   = tsz_v<SZ, float>;
        size_t                  nd   = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  s1   = kt_loadu_p<SZ, float>(&D.vs[0]);
        avxvector_t<SZ, float>  s2   = kt_loadu_p<SZ, float>(&D.vs[1]);
        avxvector_t<SZ, double> d1   = kt_loadu_p<SZ, double>(&D.vd[2]);
        avxvector_t<SZ, double> d2   = kt_loadu_p<SZ, double>(&D.vd[0]);
        float                   refs = 0.0f;
        double                  refd = 0.0;

        float sdot = kt_cdot_p<SZ, float>(s1, s2);
        for(size_t i = 0; i < ns; i++)
            refs += D.vs[0 + i] * D.vs[1 + i];
        EXPECT_FLOAT_EQ(refs, sdot);

        double ddot = kt_cdot_p<SZ, double>(d1, d2);
        for(size_t i = 0; i < nd; i++)
            refd += D.vd[2 + i] * D.vd[0 + i];
        EXPECT_DOUBLE_EQ(refd, ddot);

        size_t                   nc   = tsz_v<SZ, cfloat>;
        size_t                   nz   = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, cfloat>  c1   = kt_loadu_p<SZ, cfloat>(&D.vc[3]);
        avxvector_t<SZ, cfloat>  c2   = kt_loadu_p<SZ, cfloat>(&D.vc[5]);
        avxvector_t<SZ, cdouble> z1   = kt_loadu_p<SZ, cdouble>(&D.vz[3]);
        avxvector_t<SZ, cdouble> z2   = kt_loadu_p<SZ, cdouble>(&D.vz[2]);
        cfloat                   refc = {0.0f, 0.0f};
        cdouble                  refz = {0.0, 0.0};

        cfloat cdot = kt_cdot_p<SZ, cfloat>(c1, c2);
        for(size_t i = 0; i < nc; i++)
            refc += D.vc[3 + i] * conj(D.vc[5 + i]);

        EXPECT_COMPLEX_FLOAT_EQ(refc, cdot);

        cdouble zdot = kt_cdot_p<SZ, cdouble>(z1, z2);
        for(size_t i = 0; i < nz; i++)
            refz += D.vz[3 + i] * conj(D.vz[2 + i]);

        EXPECT_COMPLEX_DOUBLE_EQ(refz, zdot);
    }

    template <bsz SZ>
    void kt_storeu_p_test()
    {
        constexpr size_t         ns = tsz_v<SZ, float>;
        constexpr size_t         nd = tsz_v<SZ, double>;
        constexpr size_t         nc = tsz_v<SZ, cfloat>;
        constexpr size_t         nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, float>   s;
        avxvector_t<SZ, double>  d;
        avxvector_t<SZ, cfloat>  c;
        avxvector_t<SZ, cdouble> z;
        float                    refs[ns];
        float                    vss[ns];
        double                   refd[nd];
        double                   vdd[nd];

        // Used to dynamic memory allocation to suppress a potential compiler bug
        cfloat  *refc = new cfloat[nc];
        cfloat  *vcc  = new cfloat[nc];
        cdouble *refz = new cdouble[nz];
        cdouble *vzz  = new cdouble[nz];

        s = kt_loadu_p<SZ, float>(&D.vs[3]);
        kt_storeu_p<SZ, float>(vss, s);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i + 3];
        }
        EXPECT_EQ_VEC(ns, vss, refs);

        d = kt_loadu_p<SZ, double>(&D.vd[3]);
        kt_storeu_p<SZ, double>(vdd, d);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i + 3];
        }
        EXPECT_EQ_VEC(nd, vdd, refd);

        c = kt_loadu_p<SZ, cfloat>(&D.vc[3]);
        kt_storeu_p<SZ, cfloat>(vcc, c);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = D.vc[i + 3];
        }
        EXPECT_EQ_VEC(nc, vcc, refc);

        z = kt_loadu_p<SZ, cdouble>(&D.vz[3]);
        kt_storeu_p<SZ, cdouble>(vzz, z);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = D.vz[i + 3];
        }
        EXPECT_EQ_VEC(nz, vzz, refz);

        delete[] refc;
        delete[] refz;
        delete[] vcc;
        delete[] vzz;
    }

    template <bsz SZ>
    void kt_fmadd_B_test()
    {
        size_t                  ns = tsz_v<SZ, float>;
        size_t                  nd = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  s, as, bs, s_;
        avxvector_t<SZ, double> d, ad, bd, d_;
        float                   refs[ns];
        std::vector<float>      refs_(ns, -9.0f);
        double                  refd[nd];
        std::vector<double>     refd_(ns, -8.0);

        as = kt_loadu_p<SZ, float>(&D.vs[0]);
        bs = kt_set_p<SZ, float>(D.vs, D.map);
        s  = kt_set_p<SZ, float>(D.vs, &D.map[4]);
        s_ = kt_set1_p<SZ, float>(-9.0f);
        kt_fmadd_B<SZ, float>(as, bs, s, s_);
        for(size_t i = 0; i < ns; i++)
        {
            refs[i] = D.vs[i] * D.vs[D.map[i]] + D.vs[D.map[4 + i]];
        }
        EXPECT_FLOAT_EQ_VEC(ns, s, refs);
        EXPECT_EQ_VEC(ns, s_, refs_.data());

        ad = kt_loadu_p<SZ, double>(&D.vd[0]);
        bd = kt_set_p<SZ, double>(D.vd, D.map);
        d  = kt_set_p<SZ, double>(D.vd, &D.map[2]);
        d_ = kt_set1_p<SZ, double>(-8.0);
        kt_fmadd_B<SZ, double>(ad, bd, d, d_);
        for(size_t i = 0; i < nd; i++)
        {
            refd[i] = D.vd[i] * D.vd[D.map[i]] + D.vd[D.map[2 + i]];
        }
        EXPECT_DOUBLE_EQ_VEC(nd, d, refd);
        EXPECT_DOUBLE_EQ_VEC(nd, d_, refd_);

        // Complex<float> FMADD ================================================
        constexpr size_t       nc = tsz_v<SZ, cfloat>;
        avxvector_t<SZ, float> scr, sci, ac, bc, cc, s1;
        cfloat                 refc[nc], refsci[nc], refscr[nc];
        const cfloat          *vc1 = D.vc;
        const cfloat          *vc2 = D.vc + 1;

        ac  = kt_loadu_p<SZ, cfloat>(&D.vc[2]);
        bc  = kt_loadu_p<SZ, cfloat>(vc1);
        cc  = kt_loadu_p<SZ, cfloat>(vc2);
        s1  = kt_set1_p<SZ, cfloat>({1.0f, 1.0f});
        scr = kt_setzero_p<SZ, cfloat>();
        sci = kt_setzero_p<SZ, cfloat>();
        kt_fmadd_B<SZ, cfloat>(s1, cc, scr, sci);
        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = std::complex(vc2[i].real(), vc2[i].imag());
        }
        std::complex<float> *pc = reinterpret_cast<std::complex<float> *>(&scr);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        for(size_t i = 0; i < nc; i++)
        {
            refc[i] = std::complex(vc2[i].imag(), vc2[i].real());
        }
        pc = reinterpret_cast<std::complex<float> *>(&sci);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refc);

        for(size_t i = 0; i < nc; i++)
        {
            refscr[i] = {D.vc[2 + i].real() * vc1[i].real() + scr[2 * i],
                         D.vc[2 + i].imag() * vc1[i].imag() + scr[2 * i + 1]};
        }
        for(size_t i = 0; i < nc; i++)
        {
            refsci[i] = {D.vc[2 + i].real() * vc1[i].imag() + sci[2 * i],
                         D.vc[2 + i].imag() * vc1[i].real() + sci[2 * i + 1]};
        }
        kt_fmadd_B<SZ, cfloat>(ac, bc, scr, sci);
        pc = reinterpret_cast<std::complex<float> *>(&scr);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refscr);
        pc = reinterpret_cast<std::complex<float> *>(&sci);
        EXPECT_COMPLEX_FLOAT_EQ_VEC(nc, pc, refsci);

        // Complex<double> FMADD ===============================================
        constexpr size_t        nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, double> az, bz, cz, dcr, dci, d1;
        cdouble                 refz[nz], refdzr[nz], refdzi[nz];
        const cdouble          *vz1 = D.vz;
        const cdouble          *vz2 = D.vz + 1;

        az  = kt_loadu_p<SZ, cdouble>(&D.vz[2]);
        bz  = kt_loadu_p<SZ, cdouble>(vz1);
        cz  = kt_loadu_p<SZ, cdouble>(vz2);
        d1  = kt_set1_p<SZ, cdouble>({1.0, 1.0});
        dcr = kt_setzero_p<SZ, cdouble>();
        dci = kt_setzero_p<SZ, cdouble>();
        kt_fmadd_B<SZ, cdouble>(d1, cz, dcr, dci);

        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = vz2[i];
        }
        std::complex<double> *pz = reinterpret_cast<std::complex<double> *>(&dcr);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);
        for(size_t i = 0; i < nz; i++)
        {
            refz[i] = std::complex<double>(vz2[i].imag(), vz2[i].real());
        }
        pz = reinterpret_cast<std::complex<double> *>(&dci);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refz);

        for(size_t i = 0; i < nz; i++)
        {
            refdzr[i] = {D.vz[2 + i].real() * vz1[i].real() + dcr[2 * i],
                         D.vz[2 + i].imag() * vz1[i].imag() + dcr[2 * i + 1]};
        }
        for(size_t i = 0; i < nz; i++)
        {
            refdzi[i] = {D.vz[2 + i].real() * vz1[i].imag() + dci[2 * i],
                         D.vz[2 + i].imag() * vz1[i].real() + dci[2 * i + 1]};
        }

        kt_fmadd_B<SZ, cdouble>(az, bz, dcr, dci);

        pz = reinterpret_cast<std::complex<double> *>(&dcr);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refdzr);

        pz = reinterpret_cast<std::complex<double> *>(&dci);
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(nz, pz, refdzi);
    }

    template <bsz SZ>
    void kt_hsum_B_test()
    {
        const size_t            ns = tsz_v<SZ, float>;
        const size_t            nd = tsz_v<SZ, double>;
        avxvector_t<SZ, float>  as, bs, s, s_;
        avxvector_t<SZ, double> ad, bd, d, d_;
        float                   sums, refs = 0.0f;
        double                  sumd, refd = 0.0;

        // Float ==============================================================
        as = kt_loadu_p<SZ, float>(D.vs);
        bs = kt_set_p<SZ, float>(D.vs, &D.map[4]);
        s  = kt_set1_p<SZ, float>(-3.0f);
        s_ = kt_setzero_p<SZ, float>();
        kt_fmadd_B<SZ, float>(as, bs, s, s_);
        sums = kt_hsum_B<SZ, float>(s, s_);
        for(size_t i = 0; i < ns; i++)
        {
            refs += D.vs[i] * D.vs[D.map[4 + i]] - 3.f;
        }
        EXPECT_FLOAT_EQ(sums, refs);

        // Double ==============================================================
        ad = kt_loadu_p<SZ, double>(D.vd);
        bd = kt_set_p<SZ, double>(D.vd, &D.map[2]);
        d  = kt_set1_p<SZ, double>(-5.0);
        d_ = kt_set1_p<SZ, double>(1.3);
        kt_fmadd_B<SZ, double>(bd, ad, d, d_);
        sumd = kt_hsum_B<SZ, double>(d, d_);
        for(size_t i = 0; i < nd; i++)
        {
            refd += D.vd[i] * D.vd[D.map[2 + i]] - 5.0;
        }
        EXPECT_DOUBLE_EQ(sumd, refd);

        const size_t             nc = tsz_v<SZ, cfloat>;
        const size_t             nz = tsz_v<SZ, cdouble>;
        avxvector_t<SZ, cfloat>  ac, bc, c, c_, qc;
        avxvector_t<SZ, cdouble> az, bz, z, z_;
        cfloat                   sumc, refc = {0.0f, 0.0f};
        cdouble                  sumz, refz = {0.0, 0.0};

        // cdouble ==============================================================
        az = kt_loadu_p<SZ, cdouble>(D.vz);
        bz = kt_set_p<SZ, cdouble>(D.vz, &D.map[3]);
        // auto c1 = std::complex(1.0, 5.0);
        // auto c0 = std::complex(0.0);
        // cdouble t1[]{c1, c0, c0, c0};
        // cdouble t2[]{std::complex(2.0) * c1, c0, c0, c0};
        // az = kt_loadu_p<SZ, cdouble>(t1);
        // bz = kt_loadu_p<SZ, cdouble>(t2);
        z  = kt_setzero_p<SZ, cdouble>();
        z_ = kt_setzero_p<SZ, cdouble>();
        // (z, z_) <- az * bz
        kt_fmadd_B<SZ, cdouble>(bz, az, z, z_);
        sumz = kt_hsum_B<SZ, cdouble>(z, z_);
        for(size_t i = 0; i < nz; i++)
        {
            // Test also commutativity
            //      bz                 * az
            refz += D.vz[D.map[3 + i]] * D.vz[i];
        }
        EXPECT_COMPLEX_DOUBLE_EQ(sumz, refz);

        // cfloat ==============================================================
        ac = kt_loadu_p<SZ, cfloat>(D.vc);
        bc = kt_set_p<SZ, cfloat>(D.vc, &D.map[2]);
        qc = kt_set_p<SZ, cfloat>(D.vc, &D.map[5]);
        c  = kt_setzero_p<SZ, cfloat>();
        c_ = kt_setzero_p<SZ, cfloat>();
        kt_fmadd_B<SZ, cfloat>(ac, bc, c, c_);
        kt_fmadd_B<SZ, cfloat>(qc, bc, c, c_);
        sumc = kt_hsum_B<SZ, cfloat>(c, c_);
        for(size_t i = 0; i < nc; i++)
        {
            // test also accumulation TODO
            //      (qc                 + ac      ) * bc
            refc += (D.vc[D.map[5 + i]] + D.vc[i]) * D.vc[D.map[2 + i]];
        }
        EXPECT_COMPLEX_FLOAT_EQ(sumc, refc);
    }

}

// Instantiation
template void TestsKT::kt_loadu_p_test<get_bsz()>();
template void TestsKT::kt_setzero_p_test<get_bsz()>();
template void TestsKT::kt_set1_p_test<get_bsz()>();
template void TestsKT::kt_add_p_test<get_bsz()>();
template void TestsKT::kt_sub_p_test<get_bsz()>();
template void TestsKT::kt_mul_p_test<get_bsz()>();
template void TestsKT::kt_fmadd_p_test<get_bsz()>();
template void TestsKT::kt_fmsub_p_test<get_bsz()>();
template void TestsKT::kt_set_p_test<get_bsz()>();
template void TestsKT::kt_hsum_p_test<get_bsz()>();
template void TestsKT::kt_conj_p_test<get_bsz()>();
template void TestsKT::kt_dot_p_test<get_bsz()>();
template void TestsKT::kt_cdot_p_test<get_bsz()>();
template void TestsKT::kt_storeu_p_test<get_bsz()>();
template void TestsKT::kt_fmadd_B_test<get_bsz()>();
template void TestsKT::kt_hsum_B_test<get_bsz()>();
