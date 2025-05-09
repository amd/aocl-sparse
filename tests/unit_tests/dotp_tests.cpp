/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_interface.hpp"

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
        SCOPED_TRACE(std::string("test_dotp_ref for ") + std::string(typeid(T).name()));

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
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
            }
            else
            {
                EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                              nnz, x.data(), indx.data(), y.data(), &dot, conj, 0)),
                          aoclsparse_status_success);
                expect_eq<T>((conj ? dotc_exp : dot_exp), dot);
            }
        }
    }

    // Several tests in one when nullptr is passed instead of valid data
    // Also tests for Invalid KID
    template <typename T>
    void test_dotp_nullptr()
    {
        SCOPED_TRACE(std::string("test_dotp_nullptr for ") + std::string(typeid(T).name()));

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

        // Invalid KID test
        EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                      nnz, x.data(), indx.data(), y.data(), &dot, false, 999)),
                  aoclsparse_status_invalid_kid);
    }

    template <typename T>
    void test_dotp_invalidsize()
    {
        SCOPED_TRACE(std::string("test_dotp_invalidsize for ") + std::string(typeid(T).name()));

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
        SCOPED_TRACE(std::string("test_dotp_success for ") + std::string(typeid(T).name())
                     + std::string("datatype with KID = ") + std::to_string(KID));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        T                           dot_exp;
        T                           dotc_exp;
        T                           dot;

        init(nnz, indx, x, y, dot_exp, dotc_exp);

        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_float_complex>
                     || std::is_same_v<T, aoclsparse_double_complex>)
        {
            // In case of complex, test conjugate dot.
            EXPECT_EQ((aoclsparse_dot<T, aoclsparse_status>(
                          nnz, x.data(), indx.data(), y.data(), &dot, true, KID)),
                      aoclsparse_status_success);

            using U = typename get_data_type<T>::type;

            U *tdot, *tdotc_exp;

            tdot      = reinterpret_cast<U *>(&dot);
            tdotc_exp = reinterpret_cast<U *>(&dotc_exp);

            expect_eq<U>(*tdot, *tdotc_exp);
        }
        else
        {
            expect_eq<T>(
                (aoclsparse_dot<T, T>(nnz, x.data(), indx.data(), y.data(), &dot, false, KID)),
                dot_exp);
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
        // Test the default API
        test_dotp_success<double>();

        // Explicitly, request the kernels
        test_dotp_success<double, 0>();
        test_dotp_success<double, 1>();
        test_dotp_success<double, 2>();
        if(can_exec_avx512_tests())
            test_dotp_success<double, 3>();
    }
    TEST(dot, SuccessArgFloat)
    {
        // Test the default API
        test_dotp_success<float>();

        // Explicitly, request the kernels
        test_dotp_success<float, 0>();
        test_dotp_success<float, 1>();
        test_dotp_success<float, 2>();
        if(can_exec_avx512_tests())
            test_dotp_success<float, 3>();
    }
    TEST(dot, SuccessArgCDouble)
    {
        // Test the default API
        test_dotp_success<std::complex<double>>();

        // Explicitly, request the kernels
        test_dotp_success<std::complex<double>, 0>();
        test_dotp_success<std::complex<double>, 1>();
        test_dotp_success<std::complex<double>, 2>();
        if(can_exec_avx512_tests())
            test_dotp_success<std::complex<double>, 3>();
    }
    TEST(dot, SuccessArgCFloat)
    {
        // Test the default API
        test_dotp_success<std::complex<float>>();

        // Explicitly, request the kernels
        test_dotp_success<std::complex<float>, 0>();
        test_dotp_success<std::complex<float>, 1>();
        test_dotp_success<std::complex<float>, 2>();
        if(can_exec_avx512_tests())
            test_dotp_success<std::complex<float>, 3>();
    }
    TEST(dot, SuccessArgCStructDouble)
    {
        // Test the default API
        test_dotp_success<aoclsparse_double_complex>();

        // Explicitly, request the kernels
        test_dotp_success<aoclsparse_double_complex, 0>();
        test_dotp_success<aoclsparse_double_complex, 1>();
        test_dotp_success<aoclsparse_double_complex, 2>();
        if(can_exec_avx512_tests())
            test_dotp_success<aoclsparse_double_complex, 3>();
    }
    TEST(dot, SuccessArgCStructFloat)
    {
        // Test the default API
        test_dotp_success<aoclsparse_float_complex>();

        // Explicitly, request the kernels
        test_dotp_success<aoclsparse_float_complex, 0>();
        test_dotp_success<aoclsparse_float_complex, 1>();
        test_dotp_success<aoclsparse_float_complex, 2>();
        if(can_exec_avx512_tests())
            test_dotp_success<aoclsparse_float_complex, 3>();
    }

} // namespace
