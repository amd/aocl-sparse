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
#include "aoclsparse_dot.hpp"

#include <complex>
#include <vector>

namespace
{

    template <typename T>
    void init(aoclsparse_int              &nnz,
              std::vector<aoclsparse_int> &indx,
              std::vector<T>              &x,
              std::vector<T>              &y,
              T                           &dot_exp,
              T                           &dotc_exp)
    {
        nnz = 3;
        indx.assign({6, 1, 4}); // test of unordered index
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {

            dot_exp  = {-5, 23};
            dotc_exp = {23, 5};
            x.assign({{1, 1}, {2, 2}, {3, 3}});
            y.assign(
                {{1.2, -1}, {1, 2}, {2.1, -3}, {1, 0}, {2, 3}, {-1, -7}, {1, 1}, {0, 2}, {-2, 3}});
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            dot_exp = dotc_exp = 17;
            x.assign({1, 2, 3});
            y.assign({-4.7, 2, -1.3, 5, 4, 3, 1, 6});
        }
    }

    // Stress the reference template implementation
    template <typename T>
    void test_dotp_ref(void)
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        T                           dot_exp;
        T                           dotc_exp;
        T                           dot;

        init(nnz, indx, x, y, dot_exp, dotc_exp);

        for(bool conj : {true, false})
        {
            dotp_ref(nnz, x.data(), indx.data(), y.data(), &dot, conj, -1);
            EXPECT_EQ((conj ? dotc_exp : dot_exp), dot);
        }
    }

    // Several tests in one when nullptr is passed instead of valid data
    template <typename T>
    void test_dotp_nullptr()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        T                           dot_exp;
        T                           dotc_exp;
        T                           dot;
        init(nnz, indx, x, y, dot_exp, dotc_exp);

        EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                      nnz, nullptr, indx.data(), y.data(), &dot, false, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                      nnz, x.data(), nullptr, y.data(), &dot, false, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                      nnz, x.data(), indx.data(), nullptr, &dot, false, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                      nnz, x.data(), indx.data(), y.data(), nullptr, false, -1)),
                  aoclsparse_status_invalid_pointer);
    }

    template <typename T>
    void test_dotp_invalidsize()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        T                           dot_exp;
        T                           dotc_exp;
        T                           dot;
        init(nnz, indx, x, y, dot_exp, dotc_exp);

        if constexpr(std::is_same_v<T, std::complex<double>>)
        {
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          0, x.data(), indx.data(), y.data(), &dot, false, -1)),
                      aoclsparse_status_invalid_size);
            EXPECT_COMPLEX_DOUBLE_EQ(dot, 0);
        }
        else if constexpr(std::is_same_v<T, std::complex<float>>)
        {
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          0, x.data(), indx.data(), y.data(), &dot, false, -1)),
                      aoclsparse_status_invalid_size);
            EXPECT_COMPLEX_FLOAT_EQ(dot, 0);
        }
        else
        {
            EXPECT_EQ((aoclsparse_dot<T, T>(0, x.data(), indx.data(), y.data(), &dot, false, -1)),
                      0);
        }
        if constexpr(std::is_same_v<T, std::complex<double>>)
        {
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          -1, x.data(), indx.data(), y.data(), &dot, false, -1)),
                      aoclsparse_status_invalid_size);
            EXPECT_COMPLEX_DOUBLE_EQ(dot, 0);
        }
        else if constexpr(std::is_same_v<T, std::complex<float>>)
        {
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          -1, x.data(), indx.data(), y.data(), &dot, false, -1)),
                      aoclsparse_status_invalid_size);
            EXPECT_COMPLEX_FLOAT_EQ(dot, 0);
        }
        else
        {
            EXPECT_EQ((aoclsparse_dot<T, T>(-1, x.data(), indx.data(), y.data(), &dot, false, -1)),
                      0);
        }
    }

    template <typename T>
    void test_dotp_success()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        T                           dot_exp;
        T                           dotc_exp;
        T                           dot;

        init(nnz, indx, x, y, dot_exp, dotc_exp);

        if constexpr(std::is_same_v<T, double>)
        {
            EXPECT_DOUBLE_EQ(
                (aoclsparse_dot<T, T>(nnz, x.data(), indx.data(), y.data(), &dot, false, -1)),
                dot_exp);
        }
        if constexpr(std::is_same_v<T, float>)
        {
            EXPECT_FLOAT_EQ(
                (aoclsparse_dot<T, T>(nnz, x.data(), indx.data(), y.data(), &dot, false, -1)),
                dot_exp);
        }

        if constexpr(std::is_same_v<T, std::complex<double>>)
        {
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          nnz, x.data(), indx.data(), y.data(), &dot, false, -1)),
                      aoclsparse_status_success);
            EXPECT_COMPLEX_DOUBLE_EQ(dot, dot_exp);
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          nnz, x.data(), indx.data(), y.data(), &dot, true, -1)),
                      aoclsparse_status_success);
            EXPECT_COMPLEX_DOUBLE_EQ(dot, dotc_exp);
        }
        else if constexpr(std::is_same_v<T, std::complex<float>>)
        {
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          nnz, x.data(), indx.data(), y.data(), &dot, false, -1)),
                      aoclsparse_status_success);
            EXPECT_COMPLEX_FLOAT_EQ(dot, dot_exp);
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          nnz, x.data(), indx.data(), y.data(), &dot, true, -1)),
                      aoclsparse_status_success);
            EXPECT_COMPLEX_FLOAT_EQ(dot, dotc_exp);
        }
    }

    // testing aoclsparse_*_complex types
    template <typename T>
    void test_dotp_success_struct()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        T                           dot_exp;
        T                           dotc_exp;
        T                           dot;

        init(nnz, indx, x, y, dot_exp, dotc_exp);

        if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            std::complex<double> *tdot, *tdot_exp, *tdotc_exp;
            tdot      = (std::complex<double> *)&dot;
            tdotc_exp = (std::complex<double> *)&dotc_exp;
            tdot_exp  = (std::complex<double> *)&dot_exp;

            EXPECT_EQ((aoclsparse_dot<std::complex<double>, aoclsparse_status>(
                          nnz,
                          (std::complex<double> *)x.data(),
                          indx.data(),
                          (std::complex<double> *)y.data(),
                          (std::complex<double> *)&dot,
                          false,
                          -1)),
                      aoclsparse_status_success);
            EXPECT_COMPLEX_DOUBLE_EQ(*tdot, *tdot_exp);
            EXPECT_EQ((aoclsparse_dot<std::complex<double>, aoclsparse_status>(
                          nnz,
                          (std::complex<double> *)x.data(),
                          indx.data(),
                          (std::complex<double> *)y.data(),
                          (std::complex<double> *)&dot,
                          true,
                          -1)),
                      aoclsparse_status_success);
            EXPECT_COMPLEX_DOUBLE_EQ(*tdot, *tdotc_exp);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            std::complex<float> *tdot, *tdot_exp, *tdotc_exp;
            tdot      = (std::complex<float> *)&dot;
            tdotc_exp = (std::complex<float> *)&dotc_exp;
            tdot_exp  = (std::complex<float> *)&dot_exp;

            EXPECT_EQ((aoclsparse_dot<std::complex<float>, aoclsparse_status>(
                          nnz,
                          (std::complex<float> *)x.data(),
                          indx.data(),
                          (std::complex<float> *)y.data(),
                          (std::complex<float> *)&dot,
                          false,
                          -1)),
                      aoclsparse_status_success);
            EXPECT_COMPLEX_FLOAT_EQ(*tdot, *tdot_exp);
            EXPECT_EQ((aoclsparse_dot<std::complex<float>, aoclsparse_status>(
                          nnz,
                          (std::complex<float> *)x.data(),
                          indx.data(),
                          (std::complex<float> *)y.data(),
                          (std::complex<float> *)&dot,
                          true,
                          -1)),
                      aoclsparse_status_success);
            EXPECT_COMPLEX_FLOAT_EQ(*tdot, *tdotc_exp);
        }
    }

    TEST(dot, RefImplAll)
    {
        test_dotp_ref<float>();
        test_dotp_ref<double>();
        test_dotp_ref<std::complex<float>>();
        test_dotp_ref<std::complex<double>>();
    }

    TEST(dot, NullArgCDouble)
    {
        test_dotp_nullptr<std::complex<double>>();
    }
    TEST(dot, NullArgCFloat)
    {
        test_dotp_nullptr<std::complex<float>>();
    }
    TEST(dot, InvalidArgDouble)
    {
        test_dotp_invalidsize<double>();
    }
    TEST(dot, InvalidArgFloat)
    {
        test_dotp_invalidsize<float>();
    }
    TEST(dot, InvalidArgCDouble)
    {
        test_dotp_invalidsize<std::complex<double>>();
    }
    TEST(dot, InvalidArgCFloat)
    {
        test_dotp_invalidsize<std::complex<float>>();
    }
    TEST(dot, SuccessArgDouble)
    {
        test_dotp_success<double>();
    }
    TEST(dot, SuccessArgFloat)
    {
        test_dotp_success<float>();
    }
    TEST(dot, SuccessArgCDouble)
    {
        test_dotp_success<std::complex<double>>();
    }
    TEST(dot, SuccessArgCFloat)
    {
        test_dotp_success<std::complex<float>>();
    }
    TEST(dot, SuccessArgCStructDouble)
    {
        test_dotp_success_struct<aoclsparse_double_complex>();
    }
    TEST(dot, SuccessArgCStructFloat)
    {
        test_dotp_success_struct<aoclsparse_float_complex>();
    }

} // namespace
