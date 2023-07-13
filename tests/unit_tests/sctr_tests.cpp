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
                std::vector<std::complex<double>> *tx, *ty, *ty_exp;
                tx     = (std::vector<std::complex<double>> *)&x;
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
                std::vector<std::complex<float>> *tx, *ty, *ty_exp;
                tx     = (std::vector<std::complex<float>> *)&x;
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

} // namespace
