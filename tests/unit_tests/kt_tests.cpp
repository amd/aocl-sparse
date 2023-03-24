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

/* NOTE ON COMPILER FLAGS
 * Some test in this file rely on the compiler to "not-optimize-out"
 * some of the instructions and is expected to be compiled with -O0 flag.
 */

#include "aoclsparse.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse_kernel_templates.hpp"

#include <iostream>
#include <typeinfo>

#define kt_int aoclsparse_int

using namespace kernel_templates;

namespace TestsKT
{
    class KTTCommonData
    {
    public:
        size_t map[32] = {3,  1,  0,  2,  4,  7,  5,  6,  8,  15, 13, 9,  11, 10, 12, 14,
                          29, 19, 18, 31, 20, 23, 22, 25, 24, 16, 21, 26, 27, 30, 28, 17};
        double vd[16]  = {1.131,
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
        float  vs[32]  = {1.531f, 2.753f, 2.734f, 3.870f, 3.849f, 2.586f, 4.056f, 1.143f,
                          2.276f, 1.437f, 3.003f, 1.972f, 3.851f, 1.793f, 3.357f, 1.221f,
                          3.511f, 1.713f, 1.133f, 1.170f, 1.189f, 1.226f, 2.033f, 2.114f,
                          1.766f, 1.447f, 2.382f, 5.192f, 4.172f, 3.173f, 1.735f, 3.031f};
    };

    const KTTCommonData D;

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

        // helpers
        EXPECT_EQ(typeid(avxvector_t<256, float>), typeid(__m256));
        EXPECT_EQ(typeid(avxvector_half_t<256, float>), typeid(__m128));
        EXPECT_EQ((avxvector_v<256, float>), 8U);
        EXPECT_EQ((avxvector<256, float>()), 8U);
        EXPECT_EQ((psz_v<256, float>), 8U);
        EXPECT_EQ((avxvector_half_v<256, float>), 4U);
        EXPECT_EQ((hsz_v<256, float>), 4U);

        // 256 double
        EXPECT_EQ(typeid(avxvector<256, double>::type), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector<256, double>::half_type), typeid(__m128d));
        EXPECT_EQ((avxvector<256, double>::value), 4U);
        EXPECT_EQ((avxvector<256, double>()), 4U);
        EXPECT_EQ((avxvector<256, double>::half), 2U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<256, double>), typeid(__m256d));
        EXPECT_EQ(typeid(avxvector_half_t<256, double>), typeid(__m128d));
        EXPECT_EQ((avxvector_v<256, double>), 4U);
        EXPECT_EQ((avxvector<256, double>()), 4U);
        EXPECT_EQ((psz_v<256, double>), 4U);
        EXPECT_EQ((avxvector_half_v<256, double>), 2U);
        EXPECT_EQ((hsz_v<256, double>), 2U);
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

        // helpers
        EXPECT_EQ(typeid(avxvector_t<512, float>), typeid(__m512));
        EXPECT_EQ(typeid(avxvector_half_t<512, float>), typeid(__m256));
        EXPECT_EQ((avxvector_v<512, float>), 16U);
        EXPECT_EQ((avxvector<512, float>()), 16U);
        EXPECT_EQ((psz_v<512, float>), 16U);
        EXPECT_EQ((avxvector_half_v<512, float>), 8U);
        EXPECT_EQ((hsz_v<512, float>), 8U);

        // 512 double
        EXPECT_EQ(typeid(avxvector<512, double>::type), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector<512, double>::half_type), typeid(__m256d));
        EXPECT_EQ((avxvector<512, double>::value), 8U);
        EXPECT_EQ((avxvector<512, double>()), 8U);
        EXPECT_EQ((avxvector<512, double>::half), 4U);

        // helpers
        EXPECT_EQ(typeid(avxvector_t<512, double>), typeid(__m512d));
        EXPECT_EQ(typeid(avxvector_half_t<512, double>), typeid(__m256d));
        EXPECT_EQ((avxvector_v<512, double>), 8U);
        EXPECT_EQ((avxvector<512, double>()), 8U);
        EXPECT_EQ((psz_v<512, double>), 8U);
        EXPECT_EQ((avxvector_half_v<512, double>), 4U);
        EXPECT_EQ((hsz_v<512, double>), 4U);
    }
#endif

    /*
     * Test loadu intrinsic to load 8/16/32 reals
     */
    TEST(KT_L0, kt_loadu_p_256)
    {
        EXPECT_EQ_VEC_ERR((psz_v<256, float>), (kt_loadu_p<256, float>(&D.vs[3])), &D.vs[3]);
        EXPECT_EQ_VEC_ERR((psz_v<256, double>), (kt_loadu_p<256, double>(&D.vd[5])), &D.vd[5]);
    }
#ifdef USE_AVX512
    TEST(KT_L0, kt_loadu_p_512)
    {
        EXPECT_EQ_VEC_ERR((psz_v<512, float>), (kt_loadu_p<512, float>(&D.vs[0])), &D.vs[0]);
        EXPECT_EQ_VEC_ERR((psz_v<512, double>), (kt_loadu_p<512, double>(&D.vd[0])), &D.vd[0]);
    }
#endif

    /*
     * Test setzero intrinsic to load 8/16/32 zeros into a vector
     */
    TEST(KT_L0, kt_setzero_p_256)
    {
        const float  zs[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const double zd[] = {0.0, 0.0, 0.0, 0.0};

        EXPECT_EQ_VEC_ERR((psz_v<256, float>), (kt_setzero_p<256, float>()), zs);
        EXPECT_EQ_VEC_ERR((psz_v<256, double>), (kt_setzero_p<256, double>()), zd);
    }
#ifdef USE_AVX512
    TEST(KT_L0, kt_setzero_p_512)
    {
        float  zs[] = {0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f};
        double zd[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        EXPECT_EQ_VEC_ERR((psz_v<512, float>), (kt_setzero_p<512, float>()), zs);
        EXPECT_EQ_VEC_ERR((psz_v<512, double>), (kt_setzero_p<512, double>()), zd);
    }
#endif

    /*
     * Test set1 intrinsic to load a scalar into 8/16/32 length vectors
     */
    TEST(KT_L0, kt_set1_p_256)
    {
        float  zs[] = {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};
        double zd[] = {5.0, 5.0, 5.0, 5.0};

        EXPECT_EQ_VEC_ERR((avxvector_v<256, float>), (kt_set1_p<256, float>(4.0f)), zs);
        EXPECT_EQ_VEC_ERR((avxvector_v<256, double>), (kt_set1_p<256, double>(5.0)), zd);
    }
#ifdef USE_AVX512
    TEST(KT_L0, kt_set1_p_512)
    {
        float  zs[] = {4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f,
                       4.0f};
        double zd[] = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};

        EXPECT_EQ_VEC_ERR((psz_v<512, float>), (kt_set1_p<512, float>(4.0f)), zs);
        EXPECT_EQ_VEC_ERR((psz_v<512, double>), (kt_set1_p<512, double>(5.0)), zd);
    }
#endif

    /*
     * Test add intrinsic to add two 8/16/32 length vectors
     */
    TEST(KT_L0, kt_add_p_256)
    {
        size_t                   szs = psz_v<256, float>;
        size_t                   szd = psz_v<256, double>;
        avxvector_t<256, float>  s, as, bs;
        avxvector_t<256, double> d, ad, bd;
        float                    zs[8];
        double                   zd[4];

        as = kt_loadu_p<256, float>(&D.vs[2]);
        bs = kt_set1_p<256, float>(1.0);
        s  = kt_add_p(as, bs);
        for(size_t i = 0; i < szs; i++)
        {
            zs[i] = D.vs[2 + i] + 1.0f;
        }

        EXPECT_EQ_VEC_ERR(szs, s, zs);

        ad = kt_loadu_p<256, double>(&D.vd[2]);
        bd = kt_set1_p<256, double>(1.0);
        d  = kt_add_p(ad, bd);
        for(size_t i = 0; i < szd; i++)
        {
            zd[i] = D.vd[2 + i] + 1.0;
        }

        EXPECT_EQ_VEC_ERR(szd, d, zd);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_add_p_512)
    {
        size_t                   szs = psz_v<512, float>;
        size_t                   szd = psz_v<512, double>;
        avxvector_t<512, float>  s, as, bs;
        avxvector_t<512, double> d, ad, bd;
        float                    zs[16];
        double                   zd[8];

        as = kt_loadu_p<512, float>(&D.vs[0]);
        bs = kt_set1_p<512, float>(1.0f);
        s  = kt_add_p(as, bs);
        for(size_t i = 0; i < szs; i++)
        {
            zs[i] = D.vs[i] + 1.0f;
        }

        EXPECT_EQ_VEC_ERR(szs, s, zs);

        ad = kt_loadu_p<512, double>(&D.vd[0]);
        bd = kt_set1_p<512, double>(1.0);
        d  = kt_add_p(ad, bd);
        for(size_t i = 0; i < szd; i++)
        {
            zd[i] = D.vd[i] + 1.0;
        }

        EXPECT_EQ_VEC_ERR(szd, d, zd);
    }
#endif

    /*
     * Test mul intrinsic to multiply two 8/16/32 length vectors
     */
    TEST(KT_L0, kt_mul_p_256)
    {
        size_t                   szs = psz_v<256, float>;
        size_t                   szd = psz_v<256, double>;
        avxvector_t<256, float>  s, as, bs;
        avxvector_t<256, double> d, ad, bd;
        float                    zs[8];
        double                   zd[4];

        as = kt_loadu_p<256, float>(&D.vs[2]);
        bs = kt_set1_p<256, float>(2.7f);
        s  = kt_mul_p(as, bs);
        for(size_t i = 0; i < szs; i++)
        {
            zs[i] = D.vs[2 + i] * 2.7f;
        }

        EXPECT_EQ_VEC_ERR(szs, s, zs);

        ad = kt_loadu_p<256, double>(&D.vd[2]);
        bd = kt_set1_p<256, double>(2.7);
        d  = kt_mul_p(ad, bd);
        for(size_t i = 0; i < szd; i++)
        {
            zd[i] = D.vd[2 + i] * 2.7;
        }

        EXPECT_EQ_VEC_ERR(szd, d, zd);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_mul_p_512)
    {
        size_t                   szs = psz_v<512, float>;
        size_t                   szd = psz_v<512, double>;
        avxvector_t<512, float>  s, as, bs;
        avxvector_t<512, double> d, ad, bd;
        float                    zs[16];
        double                   zd[8];

        as = kt_loadu_p<512, float>(&D.vs[0]);
        bs = kt_set1_p<512, float>(3.3f);
        s  = kt_mul_p(as, bs);
        for(size_t i = 0; i < szs; i++)
        {
            zs[i] = D.vs[i] * 3.3f;
        }

        EXPECT_EQ_VEC_ERR(szs, s, zs);

        ad = kt_loadu_p<512, double>(&D.vd[0]);
        bd = kt_set1_p<512, double>(3.3);
        d  = kt_mul_p(ad, bd);
        for(size_t i = 0; i < szd; i++)
        {
            zd[i] = D.vd[i] * 3.3;
        }

        EXPECT_EQ_VEC_ERR(szd, d, zd);
    }
#endif

    /*
     * Test fmadd intrinsic to fuse-multiply-add three 8/16/32 length vectors
     */
    TEST(KT_L0, kt_fmadd_p_256)
    {
        size_t                   szs = psz_v<256, float>;
        size_t                   szd = psz_v<256, double>;
        avxvector_t<256, float>  s, as, bs;
        avxvector_t<256, double> d, ad, bd;
        float                    zs[8];
        double                   zd[4];

        as = kt_loadu_p<256, float>(&D.vs[2]);
        bs = kt_set1_p<256, float>(2.0f);
        s  = kt_set1_p<256, float>(3.0f);
        s  = kt_fmadd_p(as, bs, s);
        for(size_t i = 0; i < szs; i++)
        {
            zs[i] = D.vs[2 + i] * 2.0f + 3.0f;
        }

        EXPECT_EQ_VEC_ERR(szs, s, zs);

        ad = kt_loadu_p<256, double>(&D.vd[2]);
        bd = kt_set1_p<256, double>(2.0);
        d  = kt_set1_p<256, double>(7.0);
        d  = kt_fmadd_p(ad, bd, d);
        for(size_t i = 0; i < szd; i++)
        {
            zd[i] = D.vd[2 + i] * 2.0 + 7.0;
        }

        EXPECT_EQ_VEC_ERR(szd, d, zd);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_fmadd_p_512)
    {
        size_t                   szs = avxvector_v<512, float>;
        size_t                   szd = avxvector_v<512, double>;
        avxvector_t<512, float>  s, as, bs;
        avxvector_t<512, double> d, ad, bd;
        float                    zs[16];
        double                   zd[8];

        as = kt_loadu_p<512, float>(&D.vs[0]);
        bs = kt_set1_p<512, float>(0.02f);
        s  = kt_set1_p<512, float>(4.0f);
        s  = kt_fmadd_p(as, bs, s);
        for(size_t i = 0; i < szs; i++)
        {
            zs[i] = D.vs[i] * 0.02f + 4.0f;
        }

        EXPECT_EQ_VEC_ERR(szs, s, zs);

        ad = kt_loadu_p<512, double>(&D.vd[0]);
        bd = kt_set1_p<512, double>(0.04);
        d  = kt_set1_p<512, double>(2.0);
        d  = kt_fmadd_p(ad, bd, d);
        for(size_t i = 0; i < szd; i++)
        {
            zd[i] = D.vd[i] * 0.04 + 2.0;
        }

        EXPECT_EQ_VEC_ERR(szd, d, zd);
    }
#endif

    /*
     * Test "set" intrinsic to indirectly load 8/16/32 reals
     * using a "map"
     */
    TEST(KT_L0, kt_set_p_256)
    {
        size_t                   szs = psz_v<256, float>;
        size_t                   szd = psz_v<256, double>;
        avxvector_t<256, float>  s;
        avxvector_t<256, double> d;
        float                    zs[8];
        double                   zd[4];

        s = kt_set_p<256, float>(D.vs, &D.map[1]);
        for(size_t i = 0; i < szs; i++)
        {
            zs[i] = D.vs[D.map[1 + i]];
        }
        EXPECT_EQ_VEC_ERR(szs, s, zs);

        d = kt_set_p<256, double>(D.vd, &D.map[2]);
        for(size_t i = 0; i < szd; i++)
        {
            zd[i] = D.vd[D.map[2 + i]];
        }

        EXPECT_EQ_VEC_ERR(szd, d, zd);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_set_p_512)
    {
        size_t                   szs = psz_v<512, float>;
        size_t                   szd = psz_v<512, double>;
        avxvector_t<512, float>  s;
        avxvector_t<512, double> d;
        float                    zs[16];
        double                   zd[8];

        s = kt_set_p<512, float>(D.vs, &D.map[2]);
        for(size_t i = 0; i < szs; i++)
        {
            zs[i] = D.vs[D.map[2 + i]];
        }
        EXPECT_EQ_VEC_ERR(szs, s, zs);

        d = kt_set_p<512, double>(D.vd, &D.map[3]);
        for(size_t i = 0; i < szd; i++)
        {
            zd[i] = D.vd[D.map[3 + i]];
        }
        EXPECT_EQ_VEC_ERR(szd, d, zd);
    }
#endif

    /*
     * Test "maskz_set" zero-masked version of load_u (aka maskz_loadu) 
     * to load 8/16/32 reals. K specifies the how much zero-padding is done at the
     * end of the vector, so K = 1 for 256d vector would load [0 x3 x2 x1]
     */
#define kt_maskz_set_p_param_dir(SZ, SUF, S, EXT, K)                                  \
    {                                                                                 \
        const size_t         n = avxvector<SZ, SUF>();                                \
        avxvector_t<SZ, SUF> v = kt_maskz_set_p<SZ, SUF, EXT, K>(D.v##S, (size_t)1U); \
        SUF                  ve[n];                                                   \
        for(size_t i = 0; i < n; i++)                                                 \
        {                                                                             \
            if(i >= K)                                                                \
                ve[i] = 0.0;                                                          \
            else                                                                      \
                ve[i] = D.v##S[1 + i];                                                \
        }                                                                             \
        EXPECT_EQ_VEC_ERR(n, v, ve);                                                  \
    }
    /*
     * Test "maskz_set" zero-masked version of "set" to indirectly
     * load 8/16/32 reals. K specifies the how much zero-padding is done at the end
     * of the vector, so K = 1 for 256d vector would load [0 x[m[3]] x[m[2]] x[m[1]]
     * Note: for indirect addressing EXT can be any
     */
#define kt_maskz_set_p_param_indir(SZ, SUF, S, EXT, K)                               \
    {                                                                                \
        const size_t         n = avxvector<SZ, SUF>();                               \
        avxvector_t<SZ, SUF> v = kt_maskz_set_p<SZ, SUF, EXT, K>(D.v##S, &D.map[0]); \
        SUF                  ve[n];                                                  \
        for(size_t i = 0; i < n; i++)                                                \
        {                                                                            \
            if(i >= K)                                                               \
                ve[i] = 0.0;                                                         \
            else                                                                     \
                ve[i] = D.v##S[D.map[0 + i]];                                        \
        }                                                                            \
        EXPECT_EQ_VEC_ERR(n, v, ve);                                                 \
    }

    TEST(KT_L0, kt_maskz_set_p_256_AVX)
    {
        kt_maskz_set_p_param_dir(256, float, s, AVX, 1);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 2);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 3);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 4);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 5);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 6);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 7);
        kt_maskz_set_p_param_dir(256, float, s, AVX, 8);
        // This must trigger a warning under AVX512F (256 bit __mask8)
        // kt_maskz_set_p_param_dir(256, float, s, AVX, 9);

        kt_maskz_set_p_param_dir(256, double, d, AVX, 1);
        kt_maskz_set_p_param_dir(256, double, d, AVX, 2);
        kt_maskz_set_p_param_dir(256, double, d, AVX, 3);
        kt_maskz_set_p_param_dir(256, double, d, AVX, 4);
        // This also triggers a warning
        // kt_maskz_set_p_param_dir(256, double, d, AVX, 5);

        // indirect can have any extension
        kt_maskz_set_p_param_indir(256, float, s, AVX, 1);
        kt_maskz_set_p_param_indir(256, float, s, AVX512F, 2);
        kt_maskz_set_p_param_indir(256, float, s, AVX512VL, 3);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 4);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 5);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 6);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 7);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 8);
        kt_maskz_set_p_param_indir(256, float, s, AVX, 9);

        kt_maskz_set_p_param_indir(256, double, d, AVX, 1);
        kt_maskz_set_p_param_indir(256, double, d, AVX512DQ, 2);
        kt_maskz_set_p_param_indir(256, double, d, AVX, 3);
        kt_maskz_set_p_param_indir(256, double, d, AVX, 4);
        kt_maskz_set_p_param_indir(256, double, d, AVX, 5);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_maskz_set_p_256_AVX512VL)
    {
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 1);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 2);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 3);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 4);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 5);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 6);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 7);
        kt_maskz_set_p_param_dir(256, float, s, AVX512VL, 8);

        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 1);
        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 2);
        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 3);
        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 4);
        kt_maskz_set_p_param_dir(256, double, d, AVX512VL, 5);
    }
#endif

#ifdef USE_AVX512
    TEST(KT_L0, kt_maskz_set_p_512_AVX512F)
    {
        // Direct
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 1);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 2);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 3);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 4);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 6);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 7);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 8);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 9);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 9);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 10);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 11);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 12);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 13);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 14);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 15);
        kt_maskz_set_p_param_dir(512, float, s, AVX512F, 16);

        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 1);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 2);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 3);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 4);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 5);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 6);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 7);
        kt_maskz_set_p_param_dir(512, double, d, AVX512F, 8);

        // Indirect
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 1);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 2);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 3);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 4);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 6);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 7);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 8);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 9);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 9);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 10);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 11);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 12);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 13);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 14);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 15);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 16);

        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 1);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 2);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 3);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 4);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 5);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 6);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 7);
        kt_maskz_set_p_param_indir(512, double, d, AVX512F, 8);
    }
#endif

    /*
     * Test "hsum" intrinsic to horizontally-reduce via summation
     * vectors of length 8/16/32
     */
    TEST(KT_L0, kt_hsum_p_256)
    {
        avxvector_t<256, float>  vs;
        avxvector_t<256, double> vd;
        float                    sums, sums_exp = 0.0f;
        double                   sumd, sumd_exp = 0.0;

        vs   = kt_loadu_p<256, float>(D.vs);
        sums = kt_hsum_p(vs);

        for(size_t i = 0; i < avxvector_v<256, float>; i++)
        {
            sums_exp += D.vs[i];
        }
        EXPECT_FLOAT_EQ(sums, sums_exp);

        vd   = kt_loadu_p<256, double>(D.vd);
        sumd = kt_hsum_p(vd);
        for(size_t i = 0; i < avxvector_v<256, double>; i++)
        {
            sumd_exp += D.vd[i];
        }

        EXPECT_DOUBLE_EQ(sumd, sumd_exp);
    }

#ifdef USE_AVX512
    TEST(KT_L0, kt_hsum_p_512)
    {
        avxvector_t<512, float>  vs;
        avxvector_t<512, double> vd;
        float                    sums, sums_exp = 0.0f;
        double                   sumd, sumd_exp = 0.0;
        size_t                   size;

        vs   = kt_loadu_p<512, float>(D.vs);
        sums = kt_hsum_p(vs);

        size = avxvector<512, float>();
        for(size_t i = 0; i < size; i++)
        {
            sums_exp += D.vs[i];
        }
        EXPECT_FLOAT_EQ(sums, sums_exp);

        vd   = kt_loadu_p<512, double>(D.vd);
        sumd = kt_hsum_p(vd);

        size = avxvector<512, double>();
        for(size_t i = 0; i < size; i++)
        {
            sumd_exp += D.vd[i];
        }

        EXPECT_DOUBLE_EQ(sumd, sumd_exp);
    }
#endif

    /*
     * Test "dot-product" intrinsic on vectors of length 8/16/32
     */
    TEST(KT_L1, kt_dot_p_256)
    {
        size_t                   szs = psz_v<256, float>;
        size_t                   szd = psz_v<256, double>;
        avxvector_t<256, float>  s1  = kt_loadu_p<256, float>(&D.vs[3]);
        avxvector_t<256, float>  s2  = kt_loadu_p<256, float>(&D.vs[5]);
        avxvector_t<256, double> d1  = kt_loadu_p<256, double>(&D.vd[3]);
        avxvector_t<256, double> d2  = kt_loadu_p<256, double>(&D.vd[5]);
        float                    s   = 0.0f;
        double                   d   = 0.0;

        float sdot = kt_dot_p<256, float>(s1, s2);
        for(size_t i = 0; i < szs; i++)
            s += D.vs[3 + i] * D.vs[5 + i];
        EXPECT_FLOAT_EQ(s, sdot);
        // Test convenience wrapper
        sdot = kt_dot_p(s1, s2);
        EXPECT_FLOAT_EQ(s, sdot);

        double ddot = kt_dot_p<256, double>(d1, d2);
        for(size_t i = 0; i < szd; i++)
            d += D.vd[3 + i] * D.vd[5 + i];
        EXPECT_DOUBLE_EQ(d, ddot);
        // Test convenience wrapper
        ddot = kt_dot_p(d1, d2);
        EXPECT_DOUBLE_EQ(d, ddot);
    }

#ifdef USE_AVX512
    TEST(KT_L1, kt_dot_p_512)
    {
        size_t                   szs = psz_v<512, float>;
        size_t                   szd = psz_v<512, double>;
        avxvector_t<512, float>  s1  = kt_loadu_p<512, float>(&D.vs[0]);
        avxvector_t<512, float>  s2  = kt_loadu_p<512, float>(&D.vs[0]);
        avxvector_t<512, double> d1  = kt_loadu_p<512, double>(&D.vd[0]);
        avxvector_t<512, double> d2  = kt_loadu_p<512, double>(&D.vd[0]);
        float                    s   = 0.0f;
        double                   d   = 0.0;

        float sdot = kt_dot_p<512, float>(s1, s2);
        for(size_t i = 0; i < szs; i++)
            s += D.vs[0 + i] * D.vs[0 + i];
        EXPECT_FLOAT_EQ(s, sdot);
        // Test convenience wrapper
        sdot = kt_dot_p(s1, s2);
        EXPECT_FLOAT_EQ(s, sdot);

        double ddot = kt_dot_p<512, double>(d1, d2);
        for(size_t i = 0; i < szd; i++)
            d += D.vd[0 + i] * D.vd[0 + i];
        EXPECT_DOUBLE_EQ(d, ddot);
        // Test convenience wrapper
        ddot = kt_dot_p(d1, d2);
        EXPECT_DOUBLE_EQ(d, ddot);
    }
#endif
}
