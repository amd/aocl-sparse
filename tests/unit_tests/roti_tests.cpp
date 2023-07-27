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
              std::vector<T>              &x_exp,
              std::vector<T>              &y_exp,
              T                           &c,
              T                           &s,
              int                          id)
    {
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            switch(id)
            {
            case 0:
                nnz = 3;
                c   = -2;
                s   = 2;
                indx.assign({0, 3, 6});
                x.assign({1, 4, 8});
                y.assign({1, 0, 0, 4, 0, 0, 8});
                x_exp.assign({0, 0, 0});
                y_exp.assign({-4, 0, 0, -16, 0, 0, -32});
                break;
            case 1:
                nnz = 3;
                c   = -4.5;
                s   = 3.5;
                indx.assign({0, 3, 6});
                x.assign({1, 4, 8});
                y.assign({1, 0, 0, 4, 0, 0, 8});
                x_exp.assign({-1, -4, -8});
                y_exp.assign({-8, 0, 0, -32, 0, 0, -64});
                break;
            case 2:
                nnz = 3;
                c   = -4.5;
                s   = 3.5;
                indx.assign({0, 3, 6});
                x.assign({4.75, -2.5, 7});
                y.assign({4.75, 0, 0, -2.5, 0, 0, 7});
                x_exp.assign({-4.75, 2.50, -7});
                y_exp.assign({-38, 0, 0, 20, 0, 0, -56});
                break;
            case 3:
                nnz = 5;
                c   = -4.5;
                s   = 3.5;
                indx.assign({0, 3, 6, 7, 9});
                x.assign({-0.75, 4, -9.5, 46, 1.25});
                y.assign({-0.75, 0, 0, 4, 0, 0, -9.5, 46, 0, 1.25});
                x_exp.assign({0.75, -4, 9.5, -46, -1.25});
                y_exp.assign({6, 0, 0, -32, 0, 0, 76, -368, 0, -10});
                break;

            case 4:
                nnz = 5;
                c   = 2;
                s   = 2;
                indx.assign({0, 3, 6, 7, 9});
                x.assign({-0.75, 4, -9.5, 46, 1.25});
                y.assign({-0.75, 0, 0, 4, 0, 0, -9.5, 46, 0, 1.25});
                x_exp.assign({-3, 16, -38, 184, 5});
                y_exp.assign({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
                break;
            }
        }
    }

    // Several tests in one when nullptr is passed instead of valid data
    template <typename T>
    void test_roti_nullptr()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;
        std::vector<T>              x_exp;
        T                           c;
        T                           s;

        init(nnz, indx, x, y, x_exp, y_exp, c, s, 3);

        EXPECT_EQ((aoclsparse_roti<T>(nnz, nullptr, indx.data(), y.data(), c, s, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_roti<T>(nnz, x.data(), nullptr, y.data(), c, s, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_roti<T>(nnz, x.data(), indx.data(), nullptr, c, s, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_roti<T>(-1, x.data(), indx.data(), y.data(), c, s, -1)),
                  aoclsparse_status_invalid_size);
        indx[0] = -1;
        EXPECT_EQ((aoclsparse_roti<T>(nnz, x.data(), indx.data(), y.data(), c, s, -1)),
                  aoclsparse_status_invalid_index_value);
    }

    template <typename T>
    void test_roti_success()
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;
        std::vector<T>              x_exp;
        T                           c;
        T                           s;

        init(nnz, indx, x, y, x_exp, y_exp, c, s, 4);
        EXPECT_EQ((aoclsparse_roti<T>(0, x.data(), indx.data(), y.data(), c, s, -1)),
                  aoclsparse_status_success);

        for(aoclsparse_int id = 0; id < 5; id++)
        {
            init(nnz, indx, x, y, x_exp, y_exp, c, s, id);
            aoclsparse_int y_sz = y_exp.size();
            aoclsparse_int x_sz = x_exp.size();

            EXPECT_EQ((aoclsparse_roti<T>(nnz, x.data(), indx.data(), y.data(), c, s, -1)),
                      aoclsparse_status_success);

            if constexpr(std::is_same_v<T, float>)
            {
                EXPECT_FLOAT_EQ_VEC(y_sz, y, y_exp);
                EXPECT_FLOAT_EQ_VEC(x_sz, x, x_exp);
            }
            if constexpr(std::is_same_v<T, double>)
            {
                EXPECT_DOUBLE_EQ_VEC(y_sz, y, y_exp);
                EXPECT_DOUBLE_EQ_VEC(x_sz, x, x_exp);
            }
        }
    }

    TEST(roti, NullArgDouble)
    {
        test_roti_nullptr<double>();
    }
    TEST(roti, NullArgFloat)
    {
        test_roti_nullptr<float>();
    }

    TEST(roti, SuccessArgDouble)
    {
        test_roti_success<double>();
    }
    TEST(roti, SuccessArgFloat)
    {
        test_roti_success<float>();
    }

} // namespace
