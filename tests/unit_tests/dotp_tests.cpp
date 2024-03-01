/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
        nnz = 18;

        // Test unordered list
        indx.assign({6, 1, 4, 20, 2, 3, 7, 8, 10, 12, 13, 15, 16, 18, 0, 14, 5, 11});

        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            dotc_exp = {29.3, -233.3}; // Expected result for conjugate
            dot_exp  = {233.3, 29.3}; // Expected result for no-conjugate

            // clang-format off
            x.assign({{1, 1},   {2, 2},   {3, 3},   {4, 4},   {5, 5},   {6, 6},   {7, 7},
                      {8, 8},   {9, 9},   {10, 10}, {11, 11}, {12, 12}, {13, 13}, {14, 14},
                      {15, 15}, {16, 16}, {17, 17}, {18, 18}});
            // clang-format on

            y.assign({{1.2, -1}, {1, 2},  {2.1, -3}, {1, 0},  {2, 3},    {-1, -7}, {1, 1},
                      {0, 2},    {-2, 3}, {1.2, -1}, {1, 2},  {2.1, -3}, {1, 0},   {2, 3},
                      {-1, -7},  {1, 1},  {0, 2},    {-2, 3}, {3, 4},    {5, 1},   {1, 4}});
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            dot_exp = dotc_exp = 271.5; // Expected result for conjugate and no-conjugate
            x.assign({1, 0, 3, 4, 0, 0, 7, 8, 0, 10, 0, 12, 0, 0, 15, 16, 0, 18});
            y.assign({-4.7, 2,   -1.3, 5,  4, 3, 1, 6, -7, 12,  -3,
                      0.5,  4.5, 3.5,  15, 2, 8, 2, 9, 10, 6.25});
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
            dotp_ref(nnz, x.data(), indx.data(), y.data(), &dot, conj);
            expect_eq<T>((conj ? dotc_exp : dot_exp), dot);
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

    template <typename T, int KID = -1>
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

        std::cerr << "Kernel ID : " << KID << std::endl;

        EXPECT_EQ((aoclsparse_dotp<T>(nnz, x.data(), indx.data(), y.data(), &dot, false, KID)),
                  aoclsparse_status_success);

        expect_eq<T>(dot, dot_exp);

        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            // In case of complex, test conjugate dot.
            EXPECT_EQ((aoclsparse_dotp<T>(nnz, x.data(), indx.data(), y.data(), &dot, true, KID)),
                      aoclsparse_status_success);
            expect_eq<T>(dot, dotc_exp);
        }
    }

    // testing aoclsparse_*_complex types
    template <typename T, int KID = -1>
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

        /*  Determine the complex type
         *  If T is aoclsparse_double_complex, complex_t will be std::complex<double>.
         *  Else, complex_t will be std::complex<float>.
        */
        using complex_t = std::conditional_t<std::is_same_v<T, aoclsparse_double_complex>,
                                             std::complex<double>,
                                             std::complex<float>>;

        complex_t *tdot, *tdot_exp, *tdotc_exp;
        tdot      = (complex_t *)&dot;
        tdotc_exp = (complex_t *)&dotc_exp;
        tdot_exp  = (complex_t *)&dot_exp;

        std::cerr << "Kernel ID : " << KID << std::endl;
        EXPECT_EQ((aoclsparse_dotp<complex_t>(nnz,
                                              (complex_t *)x.data(),
                                              indx.data(),
                                              (complex_t *)y.data(),
                                              (complex_t *)&dot,
                                              false,
                                              KID)),
                  aoclsparse_status_success);
        expect_eq<complex_t>(*tdot, *tdot_exp);

        EXPECT_EQ((aoclsparse_dotp<complex_t>(nnz,
                                              (complex_t *)x.data(),
                                              indx.data(),
                                              (complex_t *)y.data(),
                                              (complex_t *)&dot,
                                              true,
                                              KID)),
                  aoclsparse_status_success);
        expect_eq<complex_t>(*tdot, *tdotc_exp);
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
        // Test the default API
        test_dotp_success<double>();

        // Explicitly, request the kernels
        test_dotp_success<double, 0>();
        test_dotp_success<double, 1>();
        test_dotp_success<double, 2>();
    }
    TEST(dot, SuccessArgFloat)
    {
        // Test the default API
        test_dotp_success<float>();

        // Explicitly, request the kernels
        test_dotp_success<float, 0>();
        test_dotp_success<float, 1>();
        test_dotp_success<float, 2>();
    }
    TEST(dot, SuccessArgCDouble)
    {
        // Test the default API
        test_dotp_success<std::complex<double>>();

        // Explicitly, request the kernels
        test_dotp_success<std::complex<double>, 0>();
        test_dotp_success<std::complex<double>, 1>();
        test_dotp_success<std::complex<double>, 2>();
    }
    TEST(dot, SuccessArgCFloat)
    {
        // Test the default API
        test_dotp_success<std::complex<float>>();

        // Explicitly, request the kernels
        test_dotp_success<std::complex<float>, 0>();
        test_dotp_success<std::complex<float>, 1>();
        test_dotp_success<std::complex<float>, 2>();
    }
    TEST(dot, SuccessArgCStructDouble)
    {
        // Test the default API
        test_dotp_success_struct<aoclsparse_double_complex>();

        // Explicitly, request the kernels
        test_dotp_success_struct<aoclsparse_double_complex, 0>();
        test_dotp_success_struct<aoclsparse_double_complex, 1>();
        test_dotp_success_struct<aoclsparse_double_complex, 2>();
    }
    TEST(dot, SuccessArgCStructFloat)
    {
        // Test the default API
        test_dotp_success_struct<aoclsparse_float_complex>();

        // Explicitly, request the kernels
        test_dotp_success_struct<aoclsparse_float_complex, 0>();
        test_dotp_success_struct<aoclsparse_float_complex, 1>();
        test_dotp_success_struct<aoclsparse_float_complex, 2>();
    }

} // namespace
