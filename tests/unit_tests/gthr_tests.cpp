/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

namespace
{

    template <typename T>
    void init(aoclsparse_int              &nnz,
              std::vector<T>              &x,
              std::vector<T>              &y,
              std::vector<aoclsparse_int> &indx,
              std::vector<T>              &x_exp,
              std::vector<T>              &y_exp)
    {
        std::vector<aoclsparse_int> tindx
            = {0, 3, 5, 1, 7, 12, 2, 6, 8, 9, 10, 11, 4, 13, 15, 16, 14, 18};
        indx = tindx;
        nnz  = 18;
        x.resize(nnz);
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            // clang-format off
            y.assign({{1, 1},   {1, 2},   {2, 3},   {3, 4},   {4, 5},   {5, 6},
                      {6, 7},   {7, 8},   {8, 9},   {9, 10},  {10, 11}, {11, 12},
                      {12, 13}, {13, 14}, {14, 15}, {15, 16}, {16, 17}, {17, 18},
                      {18, 19}, {19, 20}, {20, 21}, {21, 22}});
            y_exp.assign({{0, 0}, {0, 0},   {0, 0},   {0, 0}, {0, 0}, {0, 0},
                          {0, 0}, {0, 0},   {0, 0},   {0, 0}, {0, 0}, {0, 0},
                          {0, 0}, {0, 0},   {0, 0},   {0, 0}, {0, 0}, {17, 18},
                          {0, 0}, {19, 20}, {20, 21}, {21, 22}});
            x_exp.assign({{1, 1},   {3, 4},  {5, 6},   {1, 2},   {7, 8}, {12, 13},   {2, 3},   {6, 7},
                          {8, 9},   {9, 10}, {10, 11}, {11, 12}, {4, 5}, {13, 14}, {15, 16}, {16, 17},
                          {14, 15}, {18, 19}});
            // clang-format on
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            y.assign(
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22});
            x_exp.assign({1, 4, 6, 2, 8, 13, 3, 7, 9, 10, 11, 12, 5, 14, 16, 17, 15, 19});
            y_exp.assign({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 20, 21, 22});
        }
    }
    template <typename T>
    void init_strided_data(aoclsparse_int &m,
                           aoclsparse_int &n,
                           aoclsparse_int &nnz,
                           aoclsparse_int &stride,
                           std::vector<T> &x,
                           std::vector<T> &y,
                           std::vector<T> &x_gold)
    {
        m      = 7;
        n      = 7;
        nnz    = 49;
        stride = m + 5;
        y.resize(stride * n);
        x.resize(m);
        x_gold.resize(m);

        y = {bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0), bc((T)6.0), bc((T)7.0),
             bc((T)0),   bc((T)0),   bc((T)34),  bc((T)0),   bc((T)13),  bc((T)1.0), bc((T)2.0),
             bc((T)3.0), bc((T)4.0), bc((T)5.0), bc((T)6.0), bc((T)7.0), bc((T)0),   bc((T)0),
             bc((T)34),  bc((T)0),   bc((T)13),  bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0),
             bc((T)5.0), bc((T)6.0), bc((T)7.0), bc((T)0),   bc((T)0),   bc((T)34),  bc((T)0),
             bc((T)13),  bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0), bc((T)6.0),
             bc((T)7.0), bc((T)0),   bc((T)0),   bc((T)34),  bc((T)0),   bc((T)13),  bc((T)1.0),
             bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0), bc((T)6.0), bc((T)7.0), bc((T)0),
             bc((T)0),   bc((T)34),  bc((T)0),   bc((T)13),  bc((T)1.0), bc((T)2.0), bc((T)3.0),
             bc((T)4.0), bc((T)5.0), bc((T)6.0), bc((T)7.0), bc((T)0),   bc((T)0),   bc((T)34),
             bc((T)0),   bc((T)13),  bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0),
             bc((T)6.0), bc((T)7.0), bc((T)0),   bc((T)0),   bc((T)34),  bc((T)0),   bc((T)13)};
    }

    // Positive test case with checking output correctness
    template <typename T, int KID>
    void test_gthr_success()
    {
        SCOPED_TRACE(std::string("test_gthr_success for ") + std::string(typeid(T).name())
                     + std::string("datatype with KID = ") + std::to_string(KID));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        aoclsparse_status res = aoclsparse_gthr(nnz, y.data(), x.data(), indx.data(), KID);

        // expect success
        EXPECT_EQ(res, aoclsparse_status_success);
        if constexpr(std::is_same_v<T, double>)
        {
            EXPECT_DOUBLE_EQ_VEC(nnz, x.data(), x_exp.data());
        }
        if constexpr(std::is_same_v<T, float>)
        {
            EXPECT_FLOAT_EQ_VEC(nnz, x.data(), x_exp.data());
        }
        if constexpr(std::is_same_v<T, std::complex<double>>)
        {
            EXPECT_COMPLEX_DOUBLE_EQ_VEC(nnz, x.data(), x_exp.data());
        }
        if constexpr(std::is_same_v<T, std::complex<float>>)
        {
            EXPECT_COMPLEX_FLOAT_EQ_VEC(nnz, x.data(), x_exp.data());
        }
    }

    // Positive test case with checking output correctness
    template <typename T, int KID>
    void test_gthrz_success()
    {
        SCOPED_TRACE(std::string("test_gthrz_success for ") + std::string(typeid(T).name())
                     + std::string("datatype with KID = ") + std::to_string(KID));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        aoclsparse_status res = aoclsparse_gthrz(nnz, y.data(), x.data(), indx.data(), KID);

        // expect success
        EXPECT_EQ(res, aoclsparse_status_success);
        if constexpr(std::is_same_v<T, double>)
        {
            EXPECT_DOUBLE_EQ_VEC(nnz, x.data(), x_exp.data());
            EXPECT_DOUBLE_EQ_VEC((int)y.size(), y.data(), y_exp.data());
        }
        if constexpr(std::is_same_v<T, float>)
        {
            EXPECT_FLOAT_EQ_VEC(nnz, x.data(), x_exp.data());
            EXPECT_FLOAT_EQ_VEC((int)y.size(), y.data(), y_exp.data());
        }
        if constexpr(std::is_same_v<T, std::complex<double>>)
        {
            EXPECT_COMPLEX_DOUBLE_EQ_VEC(nnz, x.data(), x_exp.data());
            EXPECT_COMPLEX_DOUBLE_EQ_VEC((int)y.size(), y.data(), y_exp.data());
        }
        if constexpr(std::is_same_v<T, std::complex<float>>)
        {
            EXPECT_COMPLEX_FLOAT_EQ_VEC(nnz, x.data(), x_exp.data());
            EXPECT_COMPLEX_FLOAT_EQ_VEC((int)y.size(), y.data(), y_exp.data());
        }
    }

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_gthr_nullptr()
    {
        SCOPED_TRACE(std::string("test_gthr_nullptr for ") + std::string(typeid(T).name()));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_gthr<T>(nnz, nullptr, x.data(), indx.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_gthr<T>(nnz, y.data(), nullptr, indx.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_gthr<T>(nnz, y.data(), x.data(), nullptr),
                  aoclsparse_status_invalid_pointer);
    }
    // tests with wrong scalar data n, m, nnz
    template <typename T>
    void test_gthr_wrong_size()
    {
        SCOPED_TRACE(std::string("test_gthr_wrong_size for ") + std::string(typeid(T).name()));

        // Sparse index vector
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        // In turns pass negative nnz
        // and expect invalid size error
        EXPECT_EQ(aoclsparse_gthr<T>(-3, y.data(), x.data(), indx.data()),
                  aoclsparse_status_invalid_size);
    }
    // zero vector size is valid - just do nothing
    template <typename T>
    void test_gthr_do_nothing()
    {
        SCOPED_TRACE(std::string("test_gthr_do_nothing for ") + std::string(typeid(T).name()));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        // pass zero nnz
        // and expect success
        EXPECT_EQ(aoclsparse_gthr<T>(0, y.data(), x.data(), indx.data()),
                  aoclsparse_status_success);
    }

    // Negative value in index array return error
    // Additionally tests for invalid KID
    template <typename T>
    void test_gthr_invalid_index()
    {
        SCOPED_TRACE(std::string("test_gthr_invalid_index for ") + std::string(typeid(T).name()));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        aoclsparse_status res = aoclsparse_gthr(nnz, y.data(), x.data(), indx.data(), 999);
        EXPECT_EQ(res, aoclsparse_status_invalid_kid);

        indx[0] = -3;

        // Pass negative value in index array
        // and expect aoclsparse_status_invalid_index_value
        // This tests is only valid for reference kernel
        res = aoclsparse_gthr(nnz, y.data(), x.data(), indx.data(), 0);
        EXPECT_EQ(res, aoclsparse_status_invalid_index_value);
    }

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_gthrz_nullptr()
    {
        SCOPED_TRACE(std::string("test_gthrz_nullptr for ") + std::string(typeid(T).name()));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_gthrz<T>(nnz, nullptr, x.data(), indx.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_gthrz<T>(nnz, y.data(), nullptr, indx.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_gthrz<T>(nnz, y.data(), x.data(), nullptr),
                  aoclsparse_status_invalid_pointer);
    }
    // tests with wrong scalar data n, m, nnz
    template <typename T>
    void test_gthrz_wrong_size()
    {
        SCOPED_TRACE(std::string("test_gthrz_wrong_size for ") + std::string(typeid(T).name()));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        // In turns pass negative nnz
        // and expect invalid size error
        EXPECT_EQ(aoclsparse_gthrz<T>(-3, y.data(), x.data(), indx.data()),
                  aoclsparse_status_invalid_size);
    }
    // zero vector size is valid - just do nothing
    template <typename T>
    void test_gthrz_do_nothing()
    {
        SCOPED_TRACE(std::string("test_gthrz_do_nothing for ") + std::string(typeid(T).name()));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        // pass zero nnz
        // and expect success
        EXPECT_EQ(aoclsparse_gthrz<T>(0, y.data(), x.data(), indx.data()),
                  aoclsparse_status_success);
    }

    // Negative value in index array return error
    // Tests for Invalid KID
    template <typename T>
    void test_gthrz_invalid_index()
    {
        SCOPED_TRACE(std::string("test_gthrz_invalid_index for ") + std::string(typeid(T).name()));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              x_exp;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        init(nnz, x, y, indx, x_exp, y_exp);

        // Invoke with invalid KID
        aoclsparse_status res = aoclsparse_gthrz(nnz, y.data(), x.data(), indx.data(), 9999);
        EXPECT_EQ(res, aoclsparse_status_invalid_kid);

        indx[0] = -3;

        // Pass negative value in index array
        // and expect aoclsparse_status_invalid_index_value
        // This tests is only valid for reference kernel
        res = aoclsparse_gthrz(nnz, y.data(), x.data(), indx.data(), 0);
        EXPECT_EQ(res, aoclsparse_status_invalid_index_value);
    }

    //Gather with stride
    // Positive test case with checking output correctness
    template <typename T, int KID>
    void test_gthrs_success()
    {
        SCOPED_TRACE(std::string("test_gthrs_success for ") + std::string(typeid(T).name()));

        aoclsparse_int m, n, nnz;
        aoclsparse_int stride;
        std::vector<T> x_gold;
        std::vector<T> y;
        std::vector<T> x;
        T              xtol;
        tolerance_t<T> tol;
        xtol = (T)0;
        tol  = std::real(xtol); // get the tolerance.

        init_strided_data(m, n, nnz, stride, x, y, x_gold);
        for(aoclsparse_int col = 0; col < n; col++)
        {
            T val = col + 1;

            aoclsparse_status res = aoclsparse_gthrs(n, &y[col], &x[0], stride, KID);

            // expect success
            EXPECT_EQ(res, aoclsparse_status_success);

            x_gold = {
                bc((T)val), bc((T)val), bc((T)val), bc((T)val), bc((T)val), bc((T)val), bc((T)val)};
            if(tol <= 0.0)
                tol = 10;
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                EXPECT_COMPLEX_ARR_NEAR(m, x, x_gold, expected_precision<decltype(tol)>(tol));
            }
            else
            {
                EXPECT_ARR_NEAR(m, &x[0], &x_gold[0], expected_precision<decltype(tol)>(tol));
            }
        }
    }
    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_gthrs_nullptr()
    {
        SCOPED_TRACE(std::string("test_gthrs_nullptr for ") + std::string(typeid(T).name()));

        aoclsparse_int m, n, nnz;
        aoclsparse_int stride;
        std::vector<T> x_gold;
        std::vector<T> y;
        std::vector<T> x;

        init_strided_data(m, n, nnz, stride, x, y, x_gold);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_gthrs<T>(nnz, nullptr, x.data(), stride),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_gthrs<T>(nnz, y.data(), nullptr, stride),
                  aoclsparse_status_invalid_pointer);

        // Invalid KID
        EXPECT_EQ(aoclsparse_gthrs<T>(nnz, y.data(), x.data(), stride, 9999),
                  aoclsparse_status_invalid_kid);
    }
    // tests with wrong scalar data n, m, nnz
    template <typename T>
    void test_gthrs_wrong_size()
    {
        SCOPED_TRACE(std::string("test_gthrs_wrong_size for ") + std::string(typeid(T).name()));

        aoclsparse_int m, n, nnz;
        aoclsparse_int stride;
        std::vector<T> x_gold;
        std::vector<T> y;
        std::vector<T> x;

        init_strided_data(m, n, nnz, stride, x, y, x_gold);

        // In turns pass negative nnz
        // and expect invalid size error
        EXPECT_EQ(aoclsparse_gthrs<T>(-3, y.data(), x.data(), stride),
                  aoclsparse_status_invalid_size);

        EXPECT_EQ(aoclsparse_gthrs<T>(nnz, y.data(), x.data(), -1), aoclsparse_status_invalid_size);
    }
    // zero vector size is valid - just do nothing
    template <typename T>
    void test_gthrs_do_nothing()
    {
        SCOPED_TRACE(std::string("test_gthrs_do_nothing for ") + std::string(typeid(T).name()));

        aoclsparse_int m, n, nnz;
        aoclsparse_int stride;
        std::vector<T> x_gold;
        std::vector<T> y;
        std::vector<T> x;

        init_strided_data(m, n, nnz, stride, x, y, x_gold);

        // pass zero nnz
        // and expect success
        EXPECT_EQ(aoclsparse_gthrs<T>(0, y.data(), x.data(), stride), aoclsparse_status_success);
    }
    TEST(gthr, SuccessDouble)
    {
        test_gthr_success<double, 0>();
        test_gthr_success<double, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthr_success<double, 2>();
            test_gthr_success<double, 3>();
        }
    }
    TEST(gthr, SuccessFloat)
    {
        test_gthr_success<float, 0>();
        test_gthr_success<float, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthr_success<float, 2>();
            test_gthr_success<float, 3>();
        }
    }
    TEST(gthr, SuccessCDouble)
    {
        test_gthr_success<std::complex<double>, 0>();
        test_gthr_success<std::complex<double>, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthr_success<std::complex<double>, 2>();
            test_gthr_success<std::complex<double>, 3>();
        }
        test_gthr_success<aoclsparse_double_complex, 0>();
        test_gthr_success<aoclsparse_double_complex, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthr_success<aoclsparse_double_complex, 2>();
            test_gthr_success<aoclsparse_double_complex, 3>();
        }
    }
    TEST(gthr, SuccessCFloat)
    {
        test_gthr_success<std::complex<float>, 0>();
        test_gthr_success<std::complex<float>, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthr_success<std::complex<float>, 2>();
            test_gthr_success<std::complex<float>, 3>();
        }
        test_gthr_success<aoclsparse_float_complex, 0>();
        test_gthr_success<aoclsparse_float_complex, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthr_success<aoclsparse_float_complex, 2>();
            test_gthr_success<aoclsparse_float_complex, 3>();
        }
    }

    TEST(gthr, NullArgDouble)
    {
        test_gthr_nullptr<double>();
    }
    TEST(gthr, NullArgFloat)
    {
        test_gthr_nullptr<float>();
    }
    TEST(gthr, NullArgCDouble)
    {
        test_gthr_nullptr<std::complex<double>>();
    }
    TEST(gthr, NullArgCFloat)
    {
        test_gthr_nullptr<std::complex<float>>();
    }

    TEST(gthr, WrongSizeDouble)
    {
        test_gthr_wrong_size<double>();
    }
    TEST(gthr, WrongSizeFloat)
    {
        test_gthr_wrong_size<float>();
    }
    TEST(gthr, WrongSizeCDouble)
    {
        test_gthr_wrong_size<std::complex<double>>();
    }
    TEST(gthr, WrongSizeCFloat)
    {
        test_gthr_wrong_size<std::complex<float>>();
    }

    TEST(gthr, DoNothingDouble)
    {
        test_gthr_do_nothing<double>();
    }
    TEST(gthr, DoNothingFloat)
    {
        test_gthr_do_nothing<float>();
    }
    TEST(gthr, DoNothingCDouble)
    {
        test_gthr_do_nothing<std::complex<double>>();
    }
    TEST(gthr, DoNothingCFloat)
    {
        test_gthr_do_nothing<std::complex<float>>();
    }

    TEST(gthr, InvalidIndxDouble)
    {
        test_gthr_invalid_index<double>();
    }
    TEST(gthr, InvalidIndxFloat)
    {
        test_gthr_invalid_index<float>();
    }
    TEST(gthr, InvalidIndxCDouble)
    {
        test_gthr_invalid_index<std::complex<double>>();
    }
    TEST(gthr, InvalidIndxCFloat)
    {
        test_gthr_invalid_index<std::complex<float>>();
    }

    TEST(gthrz, SuccessDouble)
    {
        test_gthrz_success<double, 0>();
        test_gthrz_success<double, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrz_success<double, 2>();
            test_gthrz_success<double, 3>();
        }
    }
    TEST(gthrz, SuccessFloat)
    {
        test_gthrz_success<float, 0>();
        test_gthrz_success<float, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrz_success<float, 2>();
            test_gthrz_success<float, 3>();
        }
    }
    TEST(gthrz, SuccessCDouble)
    {
        test_gthrz_success<std::complex<double>, 0>();
        test_gthrz_success<std::complex<double>, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrz_success<std::complex<double>, 2>();
            test_gthrz_success<std::complex<double>, 3>();
        }

        test_gthrz_success<aoclsparse_double_complex, 0>();
        test_gthrz_success<aoclsparse_double_complex, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrz_success<aoclsparse_double_complex, 2>();
            test_gthrz_success<aoclsparse_double_complex, 3>();
        }
    }
    TEST(gthrz, SuccessCFloat)
    {
        test_gthrz_success<std::complex<float>, 0>();
        test_gthrz_success<std::complex<float>, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrz_success<std::complex<float>, 2>();
            test_gthrz_success<std::complex<float>, 3>();
        }
        test_gthrz_success<aoclsparse_float_complex, 0>();
        test_gthrz_success<aoclsparse_float_complex, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrz_success<aoclsparse_float_complex, 2>();
            test_gthrz_success<aoclsparse_float_complex, 3>();
        }
    }

    TEST(gthrz, NullArgDouble)
    {
        test_gthrz_nullptr<double>();
    }
    TEST(gthrz, NullArgFloat)
    {
        test_gthrz_nullptr<float>();
    }
    TEST(gthrz, NullArgCDouble)
    {
        test_gthrz_nullptr<std::complex<double>>();
    }
    TEST(gthrz, NullArgCFloat)
    {
        test_gthrz_nullptr<std::complex<float>>();
    }

    TEST(gthrz, WrongSizeDouble)
    {
        test_gthrz_wrong_size<double>();
    }
    TEST(gthrz, WrongSizeFloat)
    {
        test_gthrz_wrong_size<float>();
    }
    TEST(gthrz, WrongSizeCDouble)
    {
        test_gthrz_wrong_size<std::complex<double>>();
    }
    TEST(gthrz, WrongSizeCFloat)
    {
        test_gthrz_wrong_size<std::complex<float>>();
    }

    TEST(gthrz, DoNothingDouble)
    {
        test_gthrz_do_nothing<double>();
    }
    TEST(gthrz, DoNothingFloat)
    {
        test_gthrz_do_nothing<float>();
    }
    TEST(gthrz, DoNothingCDouble)
    {
        test_gthrz_do_nothing<std::complex<double>>();
    }
    TEST(gthrz, DoNothingCFloat)
    {
        test_gthrz_do_nothing<std::complex<float>>();
    }

    TEST(gthrz, InvalidIndxDouble)
    {
        test_gthrz_invalid_index<double>();
    }
    TEST(gthrz, InvalidIndxFloat)
    {
        test_gthrz_invalid_index<float>();
    }
    TEST(gthrz, InvalidIndxCDouble)
    {
        test_gthrz_invalid_index<std::complex<double>>();
    }
    TEST(gthrz, InvalidIndxCFloat)
    {
        test_gthrz_invalid_index<std::complex<float>>();
    }

    //gather with stride
    TEST(gthrs, SuccessDouble)
    {
        test_gthrs_success<double, 0>();
        test_gthrs_success<double, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrs_success<double, 2>();
            test_gthrs_success<double, 3>();
        }
    }
    TEST(gthrs, SuccessFloat)
    {
        test_gthrs_success<float, 0>();
        test_gthrs_success<float, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrs_success<float, 2>();
            test_gthrs_success<float, 3>();
        }
    }
    TEST(gthrs, SuccessCDouble)
    {
        test_gthrs_success<std::complex<double>, 0>();
        test_gthrs_success<std::complex<double>, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrs_success<std::complex<double>, 2>();
            test_gthrs_success<std::complex<double>, 3>();
        }
    }
    TEST(gthrs, SuccessCFloat)
    {
        test_gthrs_success<std::complex<float>, 0>();
        test_gthrs_success<std::complex<float>, 1>();
        if(can_exec_avx512_tests())
        {
            test_gthrs_success<std::complex<float>, 2>();
            test_gthrs_success<std::complex<float>, 3>();
        }
    }

    TEST(gthrs, NullArgDouble)
    {
        test_gthrs_nullptr<double>();
    }
    TEST(gthrs, NullArgFloat)
    {
        test_gthrs_nullptr<float>();
    }
    TEST(gthrs, NullArgCDouble)
    {
        test_gthrs_nullptr<std::complex<double>>();
    }
    TEST(gthrs, NullArgCFloat)
    {
        test_gthrs_nullptr<std::complex<float>>();
    }

    TEST(gthrs, WrongSizeDouble)
    {
        test_gthrs_wrong_size<double>();
    }
    TEST(gthrs, WrongSizeFloat)
    {
        test_gthrs_wrong_size<float>();
    }
    TEST(gthrs, WrongSizeCDouble)
    {
        test_gthrs_wrong_size<std::complex<double>>();
    }
    TEST(gthrs, WrongSizeCFloat)
    {
        test_gthrs_wrong_size<std::complex<float>>();
    }

    TEST(gthrs, DoNothingDouble)
    {
        test_gthrs_do_nothing<double>();
    }
    TEST(gthrs, DoNothingFloat)
    {
        test_gthrs_do_nothing<float>();
    }
    TEST(gthrs, DoNothingCDouble)
    {
        test_gthrs_do_nothing<std::complex<double>>();
    }
    TEST(gthrs, DoNothingCFloat)
    {
        test_gthrs_do_nothing<std::complex<float>>();
    }
} // namespace
