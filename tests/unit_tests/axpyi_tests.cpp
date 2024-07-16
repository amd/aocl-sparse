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

#include <complex>
#include <vector>

namespace
{

    template <typename T>
    void init(aoclsparse_int              &nnz,
              T                           &a,
              std::vector<T>              &x,
              std::vector<aoclsparse_int> &indx,
              std::vector<T>              &y,
              std::vector<T>              &y_exp,
              bool                         len)
    {
        // to test when nnz is less than or equal to the size of the vectors: indx, x
        if(len)
            nnz = 4;
        else
            nnz = 2;
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            a = {10, 20};
            x.assign({{1, 2}, {2, 3}, {4, 1}, {5, 1}});
            y.assign(
                {{11, 22}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {65, -999}});
            if(len)
                y_exp.assign({{41, 132},
                              {0, 0},
                              {0, 0},
                              {-30, 40},
                              {0, 0},
                              {0, 0},
                              {-40, 70},
                              {0, 0},
                              {85, -909}});
            else
                y_exp.assign({{11, 22},
                              {0, 0},
                              {0, 0},
                              {-30, 40},
                              {0, 0},
                              {0, 0},
                              {-40, 70},
                              {0, 0},
                              {65, -999}});
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            a = 30;
            x.assign({1, 2, 3, 4});
            y.assign({31, 0, 0, 0, 0, 0, 0, 0, 7});
            if(len)
                y_exp.assign({151, 0, 0, 30, 0, 0, 60, 0, 97});
            else
                y_exp.assign({31, 0, 0, 30, 0, 0, 60, 0, 7});
        }
        indx.assign({3, 6, 8, 0});
    }

    template <typename T>
    void test_aoclsparse_axpyi_invalid()
    {
        T                           a;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;
        init(nnz, a, x, indx, y, y_exp, true);

        if constexpr(std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            EXPECT_EQ((aoclsparse_caxpyi(nnz, nullptr, x.data(), indx.data(), y.data())),
                      aoclsparse_status_invalid_pointer);
        }
        else if constexpr(std::is_same_v<T, std::complex<double>>
                          || std::is_same_v<T, aoclsparse_double_complex>)
        {
            EXPECT_EQ((aoclsparse_zaxpyi(nnz, nullptr, x.data(), indx.data(), y.data())),
                      aoclsparse_status_invalid_pointer);
        }
        EXPECT_EQ((aoclsparse_axpyi<T>(nnz, a, nullptr, indx.data(), y.data(), -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_axpyi<T>(nnz, a, x.data(), nullptr, y.data(), -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_axpyi<T>(nnz, a, x.data(), indx.data(), nullptr, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_axpyi<T>(-1, a, x.data(), indx.data(), y.data(), -1)),
                  aoclsparse_status_invalid_size);
        // Only reference kernel checks for invalid index entries
        indx[0] = -1;
        aoclsparse_int kid
            = 0; // Invalid index checks are only done for the reference implementation
        EXPECT_EQ((aoclsparse_axpyi<T>(nnz, a, x.data(), indx.data(), y.data(), 0)),
                  aoclsparse_status_invalid_index_value);
    }

    // Test private / public axpyi interfaces
    // pub = false will call directly axpyi_t along with kid
    // pub = true calls saxpyi, daxpyi, ... without kid taking effect.
    template <typename T>
    void test_aoclsparse_axpyi_success(bool pub = true, aoclsparse_int kid = -1)
    {
        T                           a;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> indx;
        std::vector<T>              x;
        std::vector<T>              y;
        std::vector<T>              y_exp;

        // true: access boundary indices also in y, false: doesn't access boundary indices of y
        for(bool len : {true, false})
        {
            init(nnz, a, x, indx, y, y_exp, len);
            aoclsparse_int y_exp_size = y_exp.size();
            if(pub)
            {
                EXPECT_EQ((aoclsparse_axpyi<T>(nnz, a, x.data(), indx.data(), y.data(), kid)),
                          aoclsparse_status_success);
            }
            else
            {
                // take care of aliasing aoclsparse_xxx_complex to std::complex<xxx>
                using U = typename get_data_type<T>::type;
                EXPECT_EQ((aoclsparse_axpyi<U>(
                              nnz, *((U *)&a), (U *)x.data(), indx.data(), (U *)y.data(), kid)),
                          aoclsparse_status_success);
            }

            if constexpr(std::is_same_v<T, float>)
            {
                EXPECT_FLOAT_EQ_VEC(y_exp_size, y, y_exp);
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                EXPECT_DOUBLE_EQ_VEC(y_exp_size, y, y_exp);
            }
            else if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                EXPECT_COMPLEX_FLOAT_EQ_VEC(y_exp_size,
                                            ((std::complex<float> *)y.data()),
                                            ((std::complex<float> *)y_exp.data()));
            }
            else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                EXPECT_COMPLEX_DOUBLE_EQ_VEC(y_exp_size,
                                             ((std::complex<double> *)y.data()),
                                             ((std::complex<double> *)y_exp.data()));
            }
            else if constexpr(std::is_same_v<T, std::complex<float>>)
            {
                EXPECT_COMPLEX_FLOAT_EQ_VEC(y_exp_size, y, y_exp);
            }
            else if constexpr(std::is_same_v<T, std::complex<double>>)
            {
                EXPECT_COMPLEX_DOUBLE_EQ_VEC(y_exp_size, y, y_exp);
            }

            EXPECT_EQ((aoclsparse_axpyi<T>(0, a, x.data(), indx.data(), y.data(), kid)),
                      aoclsparse_status_success);
        }
    }

    TEST(axpyi, InvalidParms)
    {
        test_aoclsparse_axpyi_invalid<float>();
        test_aoclsparse_axpyi_invalid<double>();
        test_aoclsparse_axpyi_invalid<std::complex<float>>();
        test_aoclsparse_axpyi_invalid<std::complex<double>>();
    }

    TEST(axpyi, SuccessFloat)
    { // public path
        test_aoclsparse_axpyi_success<float>();
        // private path
        test_aoclsparse_axpyi_success<float>(false, 0);
    }
    TEST(axpyi, SuccessDouble)
    { // public
        test_aoclsparse_axpyi_success<double>();
        // private
        test_aoclsparse_axpyi_success<double>(false, 1);
    }
    TEST(axpyi, SuccessCFloat)
    { // public
        test_aoclsparse_axpyi_success<std::complex<float>>();
        // private
        test_aoclsparse_axpyi_success<std::complex<float>>(false, 1);
    }
    TEST(axpyi, SuccessCDouble)
    { // public
        test_aoclsparse_axpyi_success<std::complex<double>>();
        // private
#ifdef USE_AVX512
        test_aoclsparse_axpyi_success<std::complex<double>>(false, 2);
#else
        test_aoclsparse_axpyi_success<std::complex<double>>(false, 0);
#endif
    }
    TEST(axpyi, SuccessCStructFloat)
    { // public
        test_aoclsparse_axpyi_success<aoclsparse_float_complex>();
        // private
        test_aoclsparse_axpyi_success<aoclsparse_float_complex>(false, 1);
    }
    TEST(axpyi, SuccessCStructDouble)
    { // public
        test_aoclsparse_axpyi_success<aoclsparse_double_complex>();
        // private
#ifdef USE_AVX512
        test_aoclsparse_axpyi_success<aoclsparse_double_complex>(false, 2);
#else
        test_aoclsparse_axpyi_success<aoclsparse_double_complex>(false, 0);
#endif
    }

} // namespace
