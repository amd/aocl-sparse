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
#include "aoclsparse_utils.hpp"

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

        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            indx.assign({0, 3, 5, 1, 7, 12, 2, 6, 8, 9});

            // to test scatter when nnz is less than or equal to the size of the vectors: indx, x
            if(len)
                nnz = 10;
            else
                nnz = 9;

            // clang-format off
            x.assign(
                {{1, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10}});
            y.assign(13, aoclsparse_numeric::zero<T>());

            if(len)
                y_exp.assign({{1, 1}, {3, 4}, {6, 7},  {1, 2}, {0, 0}, {2, 3}, {7, 8},
                              {4, 5}, {8, 9}, {9, 10}, {0, 0}, {0, 0}, {5, 6}});
            else
                y_exp.assign({{1, 1}, {3, 4}, {6, 7}, {1, 2}, {0, 0}, {2, 3}, {7, 8},
                              {4, 5}, {8, 9}, {0, 0}, {0, 0}, {0, 0}, {5, 6}});
            // clang-format on
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            indx.assign({1, 5, 13, 14, 6, 8, 9, 3, 7, 2, 10, 0, 15, 12, 4, 11, 16});

            // to test scatter when nnz is less than or equal to the size of the vectors: indx, x
            if(len)
                nnz = 17;
            else
                nnz = 10;

            x.assign({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
            y.assign({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
            if(len)
                y_exp.assign({12, 1, 10, 8, 15, 2, 5, 9, 6, 7, 11, 16, 14, 3, 4, 13, 17});
            else
                y_exp.assign({0, 1, 10, 8, 0, 2, 5, 9, 6, 7, 0, 0, 0, 3, 4, 0, 0});
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
        SCOPED_TRACE(std::string("test_roti_nullptr for ") + std::string(typeid(T).name()));

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

        EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), 999)),
                  aoclsparse_status_invalid_kid);

        indx[0] = -1;

        // Invalid indices test can only be done on reference kernels
        EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), 0)),
                  aoclsparse_status_invalid_index_value);
    }

    template <typename T, int KID>
    void test_aoclsparse_sctr_success()
    {
        SCOPED_TRACE(std::string("test_aoclsparse_sctr_success for ")
                     + std::string(typeid(T).name()) + std::string("datatype with KID = ")
                     + std::to_string(KID));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        for(bool len : {true, false})
        {
            init(nnz, indx, x, y, y_exp, len);
            aoclsparse_int sz = y_exp.size();

            EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), KID)),
                      aoclsparse_status_success);

            expect_eq_vec<T>(sz, y.data(), y_exp.data());

            // Early return test
            EXPECT_EQ((aoclsparse_sctr<T>(0, x.data(), indx.data(), y.data(), KID)),
                      aoclsparse_status_success);
        }
    }

    // testing aoclsparse_*_complex types
    template <typename T, int KID>
    void test_aoclsparse_sctr_success_struct()
    {
        SCOPED_TRACE(std::string("test_aoclsparse_sctr_success_struct for ")
                     + std::string(typeid(T).name()) + std::string("datatype with KID = ")
                     + std::to_string(KID));

        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        /*  Determine the complex type
         *  If T is aoclsparse_double_complex, complex_t will be std::complex<double>.
         *  Else, complex_t will be std::complex<float>.
        */
        using complex_t = std::conditional_t<std::is_same_v<T, aoclsparse_double_complex>,
                                             std::complex<double>,
                                             std::complex<float>>;

        for(bool len : {true, false})
        {
            init(nnz, indx, x, y, y_exp, len);

            aoclsparse_int sz = y_exp.size();

            EXPECT_EQ((aoclsparse_sctr<T>(nnz, x.data(), indx.data(), y.data(), KID)),
                      aoclsparse_status_success);

            expect_eq_vec<complex_t>(sz, (complex_t *)y.data(), (complex_t *)y_exp.data());
        }
    }
    //Scatter with stride
    // Positive test case with checking output correctness
    template <typename T, int KID>
    void test_sctrs_success()
    {
        SCOPED_TRACE(std::string("test_sctrs_success for ") + std::string(typeid(T).name())
                     + std::string("datatype with KID = ") + std::to_string(KID));

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

            aoclsparse_status res = aoclsparse_sctrs<T>(n, &x[0], stride, &y[col], KID);

            // expect success
            EXPECT_EQ(res, aoclsparse_status_success);
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
        SCOPED_TRACE(std::string("test_sctrs_nullptr for ") + std::string(typeid(T).name()));

        aoclsparse_int m, n, nnz;
        aoclsparse_int stride;
        std::vector<T> y_gold;
        std::vector<T> y;
        std::vector<T> x;

        init_strided_data(m, n, nnz, stride, x, y, y_gold);

        EXPECT_EQ(aoclsparse_sctrs<T>(n, &x[0], stride, &y[0], 999 /*INVALID KERNEL ID*/),
                  aoclsparse_status_invalid_kid);

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
        SCOPED_TRACE(std::string("test_sctrs_wrong_size for ") + std::string(typeid(T).name()));

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
        SCOPED_TRACE(std::string("test_sctrs_do_nothing for ") + std::string(typeid(T).name()));

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
        test_aoclsparse_sctr_success<double, 0>();
        test_aoclsparse_sctr_success<double, 1>();
        if(can_exec_avx512_tests())
            test_aoclsparse_sctr_success<double, 2>();
    }
    TEST(sctr, SuccessArgFloat)
    {
        test_aoclsparse_sctr_success<float, 0>();
        test_aoclsparse_sctr_success<float, 1>();
        if(can_exec_avx512_tests())
            test_aoclsparse_sctr_success<float, 2>();
    }
    TEST(sctr, SuccessArgCDouble)
    {
        test_aoclsparse_sctr_success<std::complex<double>, 0>();
        test_aoclsparse_sctr_success<std::complex<double>, 1>();
        if(can_exec_avx512_tests())
            test_aoclsparse_sctr_success<std::complex<double>, 2>();
    }
    TEST(sctr, SuccessArgCFloat)
    {
        test_aoclsparse_sctr_success<std::complex<float>, 0>();
        test_aoclsparse_sctr_success<std::complex<float>, 1>();
        if(can_exec_avx512_tests())
            test_aoclsparse_sctr_success<std::complex<float>, 2>();
    }
    TEST(sctr, SuccessArgCStructDouble)
    {
        test_aoclsparse_sctr_success_struct<aoclsparse_double_complex, 0>();
        test_aoclsparse_sctr_success_struct<aoclsparse_double_complex, 1>();
        if(can_exec_avx512_tests())
            test_aoclsparse_sctr_success_struct<aoclsparse_double_complex, 2>();
    }
    TEST(sctr, SuccessArgCStructFloat)
    {
        test_aoclsparse_sctr_success_struct<aoclsparse_float_complex, 0>();
        test_aoclsparse_sctr_success_struct<aoclsparse_float_complex, 1>();
        if(can_exec_avx512_tests())
            test_aoclsparse_sctr_success_struct<aoclsparse_float_complex, 2>();
    }

    //Scatter with stride
    TEST(sctrs, SuccessArgDouble)
    {
        test_sctrs_success<double, 0>();
        test_sctrs_success<double, 1>();
        if(can_exec_avx512_tests())
            test_sctrs_success<double, 2>();
    }
    TEST(sctrs, SuccessArgFloat)
    {
        test_sctrs_success<float, 0>();
        test_sctrs_success<float, 1>();
        if(can_exec_avx512_tests())
            test_sctrs_success<float, 2>();
    }
    TEST(sctrs, SuccessArgCDouble)
    {
        test_sctrs_success<std::complex<double>, 0>();
        test_sctrs_success<std::complex<double>, 1>();
        if(can_exec_avx512_tests())
            test_sctrs_success<std::complex<double>, 2>();
    }
    TEST(sctrs, SuccessArgCFloat)
    {
        test_sctrs_success<std::complex<float>, 0>();
        test_sctrs_success<std::complex<float>, 1>();
        if(can_exec_avx512_tests())
            test_sctrs_success<std::complex<float>, 2>();
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
