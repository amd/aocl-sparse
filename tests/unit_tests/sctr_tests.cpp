/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "gtest/gtest.h"
#include "aoclsparse.hpp"

#include <complex>
#include <iostream>
#include <vector>

namespace
{

    template <typename T>
    void init(aoclsparse_int              &nnz,
              std::vector<aoclsparse_int> &indx,
              std::vector<T>              &x,
              std::vector<T>              &y,
              std::vector<T>              &y_exp,
              bool                         len)
    {
        indx.assign({3, 0, 6});

        // to test scatter when nnz is less than or equal to the size of the vectors: indx, x
        if(len)
            nnz = 3;
        else
            nnz = 2;
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            x.assign({{1, 1}, {2, 2}, {3, 3}});
            y.assign({{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
            if(len)
                y_exp.assign({{2, 2}, {0, 0}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {3, 3}});
            else
                y_exp.assign({{2, 2}, {0, 0}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}});
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            x.assign({1, 2, 3});
            y.assign({0, 0, 0, 0, 0, 0, 0});
            if(len)
                y_exp.assign({2, 0, 0, 1, 0, 0, 3});
            else
                y_exp.assign({2, 0, 0, 1, 0, 0, 0});
        }
    }
    template <typename T>
    void init_strided_data(aoclsparse_int &m,
                           aoclsparse_int &n,
                           aoclsparse_int &nnz,
                           aoclsparse_int &stride,
                           std::vector<T> &x, //input
                           std::vector<T> &y, //output
                           std::vector<T> &y_gold) //reference output
    {
        m      = 7;
        n      = 7;
        nnz    = 49;
        stride = m + 5;
        y_gold.resize(stride * n);
        y.resize(stride * n, 0);
        x.resize(m);

        y_gold
            = {bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0), bc((T)6.0), bc((T)7.0),
               bc((T)0),   bc((T)0),   bc((T)0),   bc((T)0),   bc((T)0),   bc((T)1.0), bc((T)2.0),
               bc((T)3.0), bc((T)4.0), bc((T)5.0), bc((T)6.0), bc((T)7.0), bc((T)0),   bc((T)0),
               bc((T)0),   bc((T)0),   bc((T)0),   bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0),
               bc((T)5.0), bc((T)6.0), bc((T)7.0), bc((T)0),   bc((T)0),   bc((T)0),   bc((T)0),
               bc((T)0),   bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0), bc((T)6.0),
               bc((T)7.0), bc((T)0),   bc((T)0),   bc((T)0),   bc((T)0),   bc((T)0),   bc((T)1.0),
               bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0), bc((T)6.0), bc((T)7.0), bc((T)0),
               bc((T)0),   bc((T)0),   bc((T)0),   bc((T)0),   bc((T)1.0), bc((T)2.0), bc((T)3.0),
               bc((T)4.0), bc((T)5.0), bc((T)6.0), bc((T)7.0), bc((T)0),   bc((T)0),   bc((T)0),
               bc((T)0),   bc((T)0),   bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0),
               bc((T)6.0), bc((T)7.0), bc((T)0),   bc((T)0),   bc((T)0),   bc((T)0),   bc((T)0)};
    }

    // Several tests in one when nullptr or invalid input is passed instead of valid data
    template <typename T>
    void test_aoclsparse_sctr_invalid()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;
        init(nnz, indx, x, y, y_exp, true);

        EXPECT_EQ((aoclsparse_sctr<T>(nnz, nullptr, indx.data(), y.data(), -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), nullptr, y.data(), -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), nullptr, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_sctr<T>(-1, x.data(), indx.data(), y.data(), -1)),
                  aoclsparse_status_invalid_size);
        indx[0] = -1;
        EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), -1)),
                  aoclsparse_status_invalid_index_value);
    }

    template <typename T>
    void test_aoclsparse_sctr_success()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        for(bool len : {true, false})
        {
            init(nnz, indx, x, y, y_exp, len);
            aoclsparse_int sz = y_exp.size();
            if constexpr(std::is_same_v<T, double>)
            {
                EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), -1)),
                          aoclsparse_status_success);
                EXPECT_DOUBLE_EQ_VEC(sz, y, y_exp);
            }
            if constexpr(std::is_same_v<T, float>)
            {
                EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), -1)),
                          aoclsparse_status_success);
                EXPECT_FLOAT_EQ_VEC(sz, y, y_exp);
            }
            if constexpr(std::is_same_v<T, std::complex<double>>)
            {
                EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), -1)),
                          aoclsparse_status_success);
                EXPECT_COMPLEX_DOUBLE_EQ_VEC(sz, y, y_exp);
            }
            if constexpr(std::is_same_v<T, std::complex<float>>)
            {
                EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), -1)),
                          aoclsparse_status_success);
                EXPECT_COMPLEX_FLOAT_EQ_VEC(sz, y, y_exp);
            }
            EXPECT_EQ((aoclsparse_sctr<T>(0, x.data(), indx.data(), y.data(), -1)),
                      aoclsparse_status_success);
        }
    }

    // testing aoclsparse_*_complex types
    template <typename T>
    void test_aoclsparse_sctr_success_struct()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        for(bool len : {true, false})
        {
            init(nnz, indx, x, y, y_exp, len);

            aoclsparse_int sz = y_exp.size();
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                std::vector<std::complex<double>> *ty, *ty_exp;
                ty     = (std::vector<std::complex<double>> *)&y;
                ty_exp = (std::vector<std::complex<double>> *)&y_exp;
                EXPECT_EQ((aoclsparse_sctr<std::complex<double>>(nnz,
                                                                 (std::complex<double> *)x.data(),
                                                                 indx.data(),
                                                                 (std::complex<double> *)y.data(),
                                                                 -1)),
                          aoclsparse_status_success);
                EXPECT_COMPLEX_DOUBLE_EQ_VEC(sz, (*ty), (*ty_exp));
            }
            else if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                std::vector<std::complex<float>> *ty, *ty_exp;
                ty     = (std::vector<std::complex<float>> *)&y;
                ty_exp = (std::vector<std::complex<float>> *)&y_exp;
                EXPECT_EQ((aoclsparse_sctr<std::complex<float>>(nnz,
                                                                (std::complex<float> *)x.data(),
                                                                indx.data(),
                                                                (std::complex<float> *)y.data(),
                                                                -1)),
                          aoclsparse_status_success);
                EXPECT_COMPLEX_FLOAT_EQ_VEC(sz, (*ty), (*ty_exp));
            }
        }
    }
    //Scatter with stride
    // Positive test case with checking output correctness
    template <typename T>
    void test_sctrs_success()
    {
        aoclsparse_int            m, n, nnz;
        aoclsparse_int            stride;
        std::vector<T>            y_gold;
        std::vector<T>            y;
        std::vector<T>            x;
        T                         xtol;
        decltype(std::real(xtol)) tol;
        xtol = (T)0;
        tol  = std::real(xtol); // get the tolerance.

        init_strided_data(m, n, nnz, stride, x, y, y_gold);
        for(aoclsparse_int col = 0; col < n; col++)
        {
            T val = col + 1;
            x     = {
                bc((T)val), bc((T)val), bc((T)val), bc((T)val), bc((T)val), bc((T)val), bc((T)val)};

            // expect success
            EXPECT_EQ(aoclsparse_sctrs<T>(n, &x[0], stride, &y[col], 0 /*REF KERNEL ID*/),
                      aoclsparse_status_success);
        }
        if constexpr(std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, std::complex<double>>)
        {
            EXPECT_COMPLEX_ARR_NEAR(m * n, y, y_gold, expected_precision<decltype(tol)>(tol));
        }
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            EXPECT_ARR_NEAR(m * n, y, y_gold, expected_precision<decltype(tol)>(tol));
        }
    }
    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_sctrs_nullptr()
    {
        aoclsparse_int m, n, nnz;
        aoclsparse_int stride;
        std::vector<T> y_gold;
        std::vector<T> y;
        std::vector<T> x;

        init_strided_data(m, n, nnz, stride, x, y, y_gold);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_sctrs<T>(n, nullptr, stride, &y[0], 0 /*REF KERNEL ID*/),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sctrs<T>(n, &x[0], stride, nullptr, 0 /*REF KERNEL ID*/),
                  aoclsparse_status_invalid_pointer);
    }
    // tests with wrong scalar data n, m, nnz
    template <typename T>
    void test_sctrs_wrong_size()
    {
        aoclsparse_int m, n, nnz;
        aoclsparse_int stride;
        std::vector<T> y_gold;
        std::vector<T> y;
        std::vector<T> x;

        init_strided_data(m, n, nnz, stride, x, y, y_gold);

        // In turns pass negative nnz
        // and expect invalid size error
        EXPECT_EQ(aoclsparse_sctrs<T>(-3, &x[0], stride, &y[0], 0 /*REF KERNEL ID*/),
                  aoclsparse_status_invalid_size);

        EXPECT_EQ(aoclsparse_sctrs<T>(n, &x[0], -2, &y[0], 0 /*REF KERNEL ID*/),
                  aoclsparse_status_invalid_size);
    }
    // zero vector size is valid - just do nothing
    template <typename T>
    void test_sctrs_do_nothing()
    {
        aoclsparse_int m, n, nnz;
        aoclsparse_int stride;
        std::vector<T> y_gold;
        std::vector<T> y;
        std::vector<T> x;

        init_strided_data(m, n, nnz, stride, x, y, y_gold);

        // pass zero nnz
        // and expect success
        EXPECT_EQ(aoclsparse_sctrs<T>(0, &x[0], stride, &y[0], 0 /*REF KERNEL ID*/),
                  aoclsparse_status_success);
    }

    TEST(sctr, NullArgCDouble)
    {
        test_aoclsparse_sctr_invalid<std::complex<double>>();
    }
    TEST(sctr, NullArgCFloat)
    {
        test_aoclsparse_sctr_invalid<std::complex<float>>();
    }

    TEST(sctr, NullArgDouble)
    {
        test_aoclsparse_sctr_invalid<double>();
    }
    TEST(sctr, NullArgFloat)
    {
        test_aoclsparse_sctr_invalid<float>();
    }

    TEST(sctr, SuccessArgDouble)
    {
        test_aoclsparse_sctr_success<double>();
    }
    TEST(sctr, SuccessArgFloat)
    {
        test_aoclsparse_sctr_success<float>();
    }
    TEST(sctr, SuccessArgCDouble)
    {
        test_aoclsparse_sctr_success<std::complex<double>>();
    }
    TEST(sctr, SuccessArgCFloat)
    {
        test_aoclsparse_sctr_success<std::complex<float>>();
    }
    TEST(sctr, SuccessArgCStructDouble)
    {
        test_aoclsparse_sctr_success_struct<aoclsparse_double_complex>();
    }
    TEST(sctr, SuccessArgCStructFloat)
    {
        test_aoclsparse_sctr_success_struct<aoclsparse_float_complex>();
    }

    //Scatter with stride
    TEST(sctrs, SuccessArgDouble)
    {
        test_sctrs_success<double>();
    }
    TEST(sctrs, SuccessArgFloat)
    {
        test_sctrs_success<float>();
    }
    TEST(sctrs, SuccessArgCDouble)
    {
        test_sctrs_success<std::complex<double>>();
    }
    TEST(sctrs, SuccessArgCFloat)
    {
        test_sctrs_success<std::complex<float>>();
    }
    TEST(sctrs, NullArgDouble)
    {
        test_sctrs_nullptr<double>();
    }
    TEST(sctrs, NullArgFloat)
    {
        test_sctrs_nullptr<float>();
    }
    TEST(sctrs, NullArgCDouble)
    {
        test_sctrs_nullptr<std::complex<double>>();
    }
    TEST(sctrs, NullArgCFloat)
    {
        test_sctrs_nullptr<std::complex<float>>();
    }
    TEST(sctrs, WrongSizeDouble)
    {
        test_sctrs_wrong_size<double>();
    }
    TEST(sctrs, WrongSizeFloat)
    {
        test_sctrs_wrong_size<float>();
    }
    TEST(sctrs, WrongSizeCDouble)
    {
        test_sctrs_wrong_size<std::complex<double>>();
    }
    TEST(sctrs, WrongSizeCFloat)
    {
        test_sctrs_wrong_size<std::complex<float>>();
    }
    TEST(sctrs, DoNothingDouble)
    {
        test_sctrs_do_nothing<double>();
    }
    TEST(sctrs, DoNothingFloat)
    {
        test_sctrs_do_nothing<float>();
    }
    TEST(sctrs, DoNothingCDouble)
    {
        test_sctrs_do_nothing<std::complex<double>>();
    }
    TEST(sctrs, DoNothingCFloat)
    {
        test_sctrs_do_nothing<std::complex<float>>();
    }
} // namespace
