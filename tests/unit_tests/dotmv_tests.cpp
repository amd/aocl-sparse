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

namespace
{
    template <typename T>
    void verify_y_d(std::vector<T> y, std::vector<T> y_exp, T d, T d_exp)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            EXPECT_FLOAT_EQ_VEC(y.size(), y, y_exp);
            EXPECT_EQ(d, d_exp);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            EXPECT_DOUBLE_EQ_VEC(y.size(), y, y_exp);
            EXPECT_EQ(d, d_exp);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            EXPECT_COMPLEX_FLOAT_EQ_VEC(
                y.size(), ((std::complex<float> *)y.data()), ((std::complex<float> *)y_exp.data()));
            EXPECT_COMPLEX_FLOAT_EQ((*((std::complex<float> *)&d)),
                                    (*((std::complex<float> *)&d_exp)));
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            EXPECT_COMPLEX_DOUBLE_EQ_VEC(y.size(),
                                         ((std::complex<double> *)y.data()),
                                         ((std::complex<double> *)y_exp.data()));
            EXPECT_COMPLEX_DOUBLE_EQ((*((std::complex<double> *)&d)),
                                     (*((std::complex<double> *)&d_exp)));
        }
    }

    template <typename T>
    void init_csr_mat(aoclsparse_matrix           &A,
                      std::vector<aoclsparse_int> &idx_ptr,
                      std::vector<aoclsparse_int> &idx,
                      std::vector<T>              &val,
                      aoclsparse_matrix_type       type,
                      aoclsparse_index_base        base,
                      aoclsparse_fill_mode         fill_mode)
    {
        aoclsparse_int m, n, nnz;
        // Initialise matrix. Only lower part is used for general matrix
        //  | 0  1  2  3  4
        // -|---------------
        // 0| 1  0  0  2  0
        // 1| 0  3  0  5  0
        // 2| 0  0  4  0  0
        // 3| 2  5  0  6  7
        // 4| 0  0  0  7  0
        // 5| 0  0  0  0  0
        if(type == aoclsparse_matrix_type_general)
            m = 6, n = 5, nnz = 7;
        else if((type == aoclsparse_matrix_type_symmetric)
                || (type == aoclsparse_matrix_type_hermitian))
            m = 5, n = 5, nnz = 7;
        else if(type == aoclsparse_matrix_type_triangular)
            m = 5, n = 5, nnz = 10;

        if(type == aoclsparse_matrix_type_triangular)
        {
            idx_ptr.assign({0 + base, 2 + base, 4 + base, 5 + base, 9 + base, 10 + base});
            idx.assign({0 + base,
                        3 + base,
                        1 + base,
                        3 + base,
                        2 + base,
                        0 + base,
                        4 + base,
                        3 + base,
                        1 + base,
                        3 + base});
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                         || std::is_same_v<T, aoclsparse_float_complex>)
            {
                val.assign({{1.0, 1.0},
                            {2.0, 2.0},
                            {3.0, 3.0},
                            {5.0, 5.0},
                            {4.0, 4.0},
                            {2.0, 2.0},
                            {7.0, 7.0},
                            {6.0, 6.0},
                            {5.0, 5.0},
                            {7.0, 7.0}});
            }
            else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                val.assign({1.0, 2.0, 3.0, 5.0, 4.0, 2.0, 7.0, 6.0, 5.0, 7.0});
            }
        }
        else
        {
            if(fill_mode == aoclsparse_fill_mode_lower)
            {
                idx_ptr.assign({0 + base, 1 + base, 2 + base, 3 + base, 6 + base, 7 + base});
                idx.assign({0 + base, 1 + base, 2 + base, 0 + base, 1 + base, 3 + base, 3 + base});
            }
            else
            {
                idx_ptr.assign({0 + base, 2 + base, 4 + base, 5 + base, 7 + base, 7 + base});
                idx.assign({0 + base, 3 + base, 1 + base, 3 + base, 2 + base, 3 + base, 4 + base});
            }
            if(type == aoclsparse_matrix_type_general)
                idx_ptr.push_back(idx_ptr[idx_ptr.size() - 1]);

            if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                         || std::is_same_v<T, aoclsparse_float_complex>)
            {
                if(type == aoclsparse_matrix_type_hermitian)
                { // diagonal elements should be real
                    if(fill_mode == aoclsparse_fill_mode_lower)
                        val.assign({{1.0, 0.0},
                                    {3.0, 0.0},
                                    {4.0, 0.0},
                                    {2.0, 2.0},
                                    {5.0, 5.0},
                                    {6.0, 0.0},
                                    {7.0, 7.0}});
                    else
                        val.assign({{1.0, 0.0},
                                    {2.0, 2.0},
                                    {3.0, 0.0},
                                    {5.0, 5.0},
                                    {4.0, 0.0},
                                    {6.0, 0.0},
                                    {7.0, 7.0}});
                }
                else
                {
                    if(fill_mode == aoclsparse_fill_mode_lower)
                        val.assign({{1.0, 1.0},
                                    {3.0, 3.0},
                                    {4.0, 4.0},
                                    {2.0, 2.0},
                                    {5.0, 5.0},
                                    {6.0, 6.0},
                                    {7.0, 7.0}});
                    else
                        val.assign({{1.0, 1.0},
                                    {2.0, 2.0},
                                    {3.0, 3.0},
                                    {5.0, 5.0},
                                    {4.0, 4.0},
                                    {6.0, 6.0},
                                    {7.0, 7.0}});
                }
            }
            else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                if(fill_mode == aoclsparse_fill_mode_lower)
                    val.assign({1.0, 3.0, 4.0, 2.0, 5.0, 6.0, 7.0});
                else
                    val.assign({1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0});
            }
        }
        ASSERT_EQ(
            aoclsparse_create_csr<T>(A, base, m, n, nnz, idx_ptr.data(), idx.data(), val.data()),
            aoclsparse_status_success);
    }

    template <typename T>
    void init(aoclsparse_matrix           &A,
              std::vector<aoclsparse_int> &idx_ptr,
              std::vector<aoclsparse_int> &idx,
              std::vector<T>              &val,
              T                           &alpha,
              T                           &beta,
              std::vector<T>              &y,
              std::vector<T>              &x,
              std::vector<T>              &y_exp,
              T                           &d_exp,
              aoclsparse_matrix_type       type,
              aoclsparse_index_base        base,
              aoclsparse_fill_mode         fill_mode,
              aoclsparse_operation         op)
    {
        // fill CSR data
        init_csr_mat<T>(A, idx_ptr, idx, val, type, base, fill_mode);

        // fill expected output, alpha, beta
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1.0, 1.0}, beta = {2.0, 2.0};
            y.assign({{1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}, {1.0, 1.0}});
            x.assign({{2.0, 2.0}, {2.0, 2.0}, {2.0, 2.0}, {2.0, 2.0}, {2.0, 2.0}});

            if(type == aoclsparse_matrix_type_symmetric)
            {
                if(op == aoclsparse_operation_conjugate_transpose)
                {
                    d_exp = {712.0, 40.0};
                    y_exp.assign(
                        {{12.0, 16.0}, {32.0, 36.0}, {16.0, 20.0}, {80.0, 84.0}, {28.0, 32.0}});
                }
                else
                {
                    d_exp = {40.0, 712.0};
                    y_exp.assign({{-12.0, 16.0},
                                  {-32.0, 36.0},
                                  {-16.0, 20.0},
                                  {-80.0, 84.0},
                                  {-28.0, 32.0}});
                }
            }
            else if(type == aoclsparse_matrix_type_hermitian)
            {
                d_exp = {376.0, 376.0};
                if(fill_mode == aoclsparse_fill_mode_lower)
                {
                    if((op == aoclsparse_operation_none)
                       || (op == aoclsparse_operation_conjugate_transpose))
                    {
                        y_exp.assign(
                            {{8.0, 16.0}, {20.0, 36.0}, {0.0, 20.0}, {0.0, 84.0}, {-28.0, 32.0}});
                    }
                    else
                    {
                        y_exp.assign(
                            {{-8.0, 16.0}, {-20.0, 36.0}, {0.0, 20.0}, {0.0, 84.0}, {28.0, 32.0}});
                    }
                }
                else
                {
                    if((op == aoclsparse_operation_none)
                       || (op == aoclsparse_operation_conjugate_transpose))
                    {
                        y_exp.assign(
                            {{-8.0, 16.0}, {-20.0, 36.0}, {0.0, 20.0}, {0.0, 84.0}, {28.0, 32.0}});
                    }
                    else
                    {
                        y_exp.assign(
                            {{8.0, 16.0}, {20.0, 36.0}, {0.0, 20.0}, {0.0, 84.0}, {-28.0, 32.0}});
                    }
                }
            }
            else
            {
                if(op == aoclsparse_operation_none)
                {
                    d_exp = {40.0, 488.0};
                    if(fill_mode == aoclsparse_fill_mode_lower)
                        y_exp.assign({{-4.0, 8.0},
                                      {-12.0, 16.0},
                                      {-16.0, 20.0},
                                      {-52.0, 56.0},
                                      {-28.0, 32.0}});
                    else
                        y_exp.assign({{-12.0, 16.0},
                                      {-32.0, 36.0},
                                      {-16.0, 20.0},
                                      {-52.0, 56.0},
                                      {0.0, 4.0}});
                }
                else if(op == aoclsparse_operation_transpose)
                {
                    d_exp = {40.0, 488.0};
                    if(fill_mode == aoclsparse_fill_mode_lower)
                        y_exp.assign({{-12.0, 16.0},
                                      {-32.0, 36.0},
                                      {-16.0, 20.0},
                                      {-52.0, 56.0},
                                      {0.0, 4.0}});
                    else
                        y_exp.assign({{-4.0, 8.0},
                                      {-12.0, 16.0},
                                      {-16.0, 20.0},
                                      {-52.0, 56.0},
                                      {-28.0, 32.0}});
                }
                else // aoclsparse_operation_conjugate_transpose
                {
                    d_exp = {488.0, 40};
                    if(fill_mode == aoclsparse_fill_mode_lower)
                    {
                        y_exp.assign(
                            {{12.0, 16.0}, {32.0, 36.0}, {16.0, 20.0}, {52.0, 56.0}, {0.0, 4.0}});
                    }
                    else
                    {
                        y_exp.assign(
                            {{4.0, 8.0}, {12.0, 16.0}, {16.0, 20.0}, {52.0, 56.0}, {28.0, 32.0}});
                    }
                }
                if(type == aoclsparse_matrix_type_general)
                {
                    if(op == aoclsparse_operation_none)
                    {
                        // size of y=m (6), x=n (5)
                        y.push_back({1.0, 1.0});
                        y_exp.push_back({0.0, 4.0});
                    }
                    else
                    {
                        // transpose op: size of y=n (5), x=m (6)
                        x.push_back({2.0, 2.0});
                    }
                }
            }
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            alpha = 1.0, beta = 2.0;
            x.assign({2.0, 2.0, 2.0, 2.0, 2.0});
            y.assign({1.0, 1.0, 1.0, 1.0, 1.0});
            if(type == aoclsparse_matrix_type_symmetric)
            {
                d_exp = 188.0;
                y_exp.assign({8.0, 18.0, 10.0, 42.0, 16.0});
            }
            else
            {
                d_exp = 132.0;
                if(((op == aoclsparse_operation_none) && (fill_mode == aoclsparse_fill_mode_lower))
                   || (((op == aoclsparse_operation_transpose)
                        || (op == aoclsparse_operation_conjugate_transpose))
                       && (fill_mode == aoclsparse_fill_mode_upper)))
                {
                    y_exp.assign({4.0, 8.0, 10.0, 28.0, 16.0});
                }
                else
                {
                    y_exp.assign({8.0, 18.0, 10.0, 28.0, 2.0});
                }
                if(type == aoclsparse_matrix_type_general)
                {
                    if(op == aoclsparse_operation_none)
                    {
                        // size of y=m (6), x=n (5)
                        y.push_back(1.0);
                        y_exp.push_back(2.0);
                    }
                    else
                    {
                        // transpose op: size of y=n (5), x=m (6)
                        x.push_back(2.0);
                    }
                }
            }
        }
    }

    template <typename T>
    void test_dotmv_nullptr()
    {
        aoclsparse_operation        op = aoclsparse_operation_none;
        T                           alpha, beta;
        std::vector<T>              x, y, y_exp;
        T                           d, d_exp;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        std::vector<aoclsparse_int> idx_ptr, idx;
        std::vector<T>              val;

        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        init<T>(A,
                idx_ptr,
                idx,
                val,
                alpha,
                beta,
                y,
                x,
                y_exp,
                d_exp,
                aoclsparse_matrix_type_general,
                base,
                aoclsparse_fill_mode_lower,
                aoclsparse_operation_none);

        // pass nullptr and expect pointer error
        EXPECT_EQ(aoclsparse_dotmv(op, alpha, nullptr, descr, x.data(), beta, y.data(), &d),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_dotmv(op, alpha, A, nullptr, x.data(), beta, y.data(), &d),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_dotmv<T>(op, alpha, A, descr, nullptr, beta, y.data(), &d),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_dotmv<T>(op, alpha, A, descr, x.data(), beta, nullptr, &d),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_dotmv<T>(op, alpha, A, descr, x.data(), beta, y.data(), nullptr),
                  aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_wrong_type_base_size()
    {
        aoclsparse_operation        op = aoclsparse_operation_none;
        T                           alpha, beta;
        std::vector<T>              x, y, y_exp;
        T                           d, d_exp;
        aoclsparse_index_base       base  = aoclsparse_index_base_zero;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        std::vector<aoclsparse_int> idx_ptr, idx;
        std::vector<T>              val;

        init<T>(A,
                idx_ptr,
                idx,
                val,
                alpha,
                beta,
                y,
                x,
                y_exp,
                d_exp,
                aoclsparse_matrix_type_general,
                base,
                aoclsparse_fill_mode_lower,
                aoclsparse_operation_none);
        aoclsparse_create_mat_descr(&descr);

        // wrong data type
        if(std::is_same_v<T, double>)
        {
            A->val_type = aoclsparse_smat;
        }
        else if(std::is_same_v<T, float>)
        {
            A->val_type = aoclsparse_cmat;
        }
        else if(std::is_same_v<T, aoclsparse_float_complex>)
        {
            A->val_type = aoclsparse_zmat;
        }
        else if(std::is_same_v<T, aoclsparse_double_complex>)
        {
            A->val_type = aoclsparse_dmat;
        }
        EXPECT_EQ(aoclsparse_dotmv(op, alpha, A, descr, x.data(), beta, y.data(), &d),
                  aoclsparse_status_wrong_type);

        if(std::is_same_v<T, double>)
        {
            A->val_type = aoclsparse_dmat;
        }
        else if(std::is_same_v<T, float>)
        {
            A->val_type = aoclsparse_smat;
        }
        else if(std::is_same_v<T, aoclsparse_float_complex>)
        {
            A->val_type = aoclsparse_cmat;
        }
        else if(std::is_same_v<T, aoclsparse_double_complex>)
        {
            A->val_type = aoclsparse_zmat;
        }
        // wrong size for symmetric matrix
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(aoclsparse_dotmv(op, alpha, A, descr, x.data(), beta, y.data(), &d),
                  aoclsparse_status_invalid_size);

        // base in matrix and descr doesn't match
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
        EXPECT_EQ(aoclsparse_dotmv(op, alpha, A, descr, x.data(), beta, y.data(), &d),
                  aoclsparse_status_invalid_value);

        // invalid base
        aoclsparse_set_mat_index_base(descr, (aoclsparse_index_base)2);
        EXPECT_EQ(aoclsparse_dotmv(op, alpha, A, descr, x.data(), beta, y.data(), &d),
                  aoclsparse_status_invalid_value);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }

    template <typename T>
    void test_dotmv(std::string            testcase,
                    aoclsparse_matrix_type type,
                    aoclsparse_index_base  base,
                    aoclsparse_fill_mode   fill_mode,
                    aoclsparse_operation   op,
                    aoclsparse_status      status)
    {
        T                           alpha, beta;
        std::vector<T>              x, y, y_exp;
        T                           d, d_exp;
        std::vector<aoclsparse_int> idx_ptr, idx;
        std::vector<T>              val;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;

        init<T>(A, idx_ptr, idx, val, alpha, beta, y, x, y_exp, d_exp, type, base, fill_mode, op);
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, base);
        aoclsparse_set_mat_type(descr, type);
        aoclsparse_set_mat_fill_mode(descr, fill_mode);
        {
            SCOPED_TRACE(testcase);
            EXPECT_EQ(aoclsparse_dotmv(op, alpha, A, descr, x.data(), beta, y.data(), &d), status);
            if(status == aoclsparse_status_success)
                verify_y_d(y, y_exp, d, d_exp);

            aoclsparse_destroy_mat_descr(descr);
            EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        }
    }

    void test_dotmv_combs(std::string            testcase,
                          aoclsparse_matrix_type type,
                          aoclsparse_index_base  base)
    {
        aoclsparse_status status = aoclsparse_status_success;

        // 1) Testcases for double data type
        // hermitian matrix are for complex data type
        if(type == aoclsparse_matrix_type_hermitian)
            status = aoclsparse_status_not_implemented;
        test_dotmv<double>(testcase + "-11",
                           type,
                           base,
                           aoclsparse_fill_mode_lower,
                           aoclsparse_operation_none,
                           status);
        test_dotmv<double>(testcase + "-12",
                           type,
                           base,
                           aoclsparse_fill_mode_upper,
                           aoclsparse_operation_none,
                           status);
        test_dotmv<double>(testcase + "-13",
                           type,
                           base,
                           aoclsparse_fill_mode_lower,
                           aoclsparse_operation_transpose,
                           status);
        test_dotmv<double>(testcase + "-14",
                           type,
                           base,
                           aoclsparse_fill_mode_upper,
                           aoclsparse_operation_transpose,
                           status);
        test_dotmv<double>(testcase + "-15",
                           type,
                           base,
                           aoclsparse_fill_mode_lower,
                           aoclsparse_operation_conjugate_transpose,
                           status);
        test_dotmv<double>(testcase + "-16",
                           type,
                           base,
                           aoclsparse_fill_mode_upper,
                           aoclsparse_operation_conjugate_transpose,
                           status);

        // 2) Testcases for float data type
        status = aoclsparse_status_success;
        // hermitian matrix are for complex data type
        if(type == aoclsparse_matrix_type_hermitian)
            status = aoclsparse_status_not_implemented;
        test_dotmv<float>(testcase + "-21",
                          type,
                          base,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_operation_none,
                          status);
        test_dotmv<float>(testcase + "-22",
                          type,
                          base,
                          aoclsparse_fill_mode_upper,
                          aoclsparse_operation_none,
                          status);
        test_dotmv<float>(testcase + "-23",
                          type,
                          base,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_operation_transpose,
                          status);
        test_dotmv<float>(testcase + "-24",
                          type,
                          base,
                          aoclsparse_fill_mode_upper,
                          aoclsparse_operation_transpose,
                          status);
        test_dotmv<float>(testcase + "-25",
                          type,
                          base,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_operation_conjugate_transpose,
                          status);
        test_dotmv<float>(testcase + "-26",
                          type,
                          base,
                          aoclsparse_fill_mode_upper,
                          aoclsparse_operation_conjugate_transpose,
                          status);

        // 3) Testcases for complex float data type
        status = aoclsparse_status_success;
        test_dotmv<aoclsparse_float_complex>(testcase + "-31",
                                             type,
                                             base,
                                             aoclsparse_fill_mode_lower,
                                             aoclsparse_operation_none,
                                             status);
        test_dotmv<aoclsparse_float_complex>(testcase + "-32",
                                             type,
                                             base,
                                             aoclsparse_fill_mode_upper,
                                             aoclsparse_operation_none,
                                             status);
        test_dotmv<aoclsparse_float_complex>(testcase + "-33",
                                             type,
                                             base,
                                             aoclsparse_fill_mode_lower,
                                             aoclsparse_operation_transpose,
                                             status);
        test_dotmv<aoclsparse_float_complex>(testcase + "-34",
                                             type,
                                             base,
                                             aoclsparse_fill_mode_upper,
                                             aoclsparse_operation_transpose,
                                             status);
        test_dotmv<aoclsparse_float_complex>(testcase + "-35",
                                             type,
                                             base,
                                             aoclsparse_fill_mode_lower,
                                             aoclsparse_operation_conjugate_transpose,
                                             status);
        test_dotmv<aoclsparse_float_complex>(testcase + "-36",
                                             type,
                                             base,
                                             aoclsparse_fill_mode_upper,
                                             aoclsparse_operation_conjugate_transpose,
                                             status);

        // 4) Testcases for complex double data type
        status = aoclsparse_status_success;
        test_dotmv<aoclsparse_double_complex>(testcase + "-41",
                                              type,
                                              base,
                                              aoclsparse_fill_mode_lower,
                                              aoclsparse_operation_none,
                                              status);
        test_dotmv<aoclsparse_double_complex>(testcase + "-42",
                                              type,
                                              base,
                                              aoclsparse_fill_mode_upper,
                                              aoclsparse_operation_none,
                                              status);
        test_dotmv<aoclsparse_double_complex>(testcase + "-43",
                                              type,
                                              base,
                                              aoclsparse_fill_mode_lower,
                                              aoclsparse_operation_transpose,
                                              status);
        test_dotmv<aoclsparse_double_complex>(testcase + "-44",
                                              type,
                                              base,
                                              aoclsparse_fill_mode_upper,
                                              aoclsparse_operation_transpose,
                                              status);
        test_dotmv<aoclsparse_double_complex>(testcase + "-45",
                                              type,
                                              base,
                                              aoclsparse_fill_mode_lower,
                                              aoclsparse_operation_conjugate_transpose,
                                              status);
        test_dotmv<aoclsparse_double_complex>(testcase + "-46",
                                              type,
                                              base,
                                              aoclsparse_fill_mode_upper,
                                              aoclsparse_operation_conjugate_transpose,
                                              status);
    }

    template <typename T>
    void test_zero_size_mat(aoclsparse_index_base base)
    {
        aoclsparse_int              m, n, nnz;
        T                           alpha, beta;
        std::vector<aoclsparse_int> csr_row_ptr, csr_col_ind;
        std::vector<T>              x, y, y_exp, csr_val;
        T                           d, d_exp;

        aoclsparse_matrix    A;
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, base);

        csr_row_ptr.assign({base, base});
        csr_col_ind.assign({base});

        // 1) m=1, n=0, nnz=0
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            d     = {100.0, 100.0};
            alpha = {1.0, 1.0}, beta = {2.0, 2.0};
            y.assign({{1.0, 1.0}});
            x.assign({{2.0, 2.0}});
            csr_val.assign({{10.0, 10.0}});
            d_exp = {0.0, 0.0};
            y_exp.assign({{0.0, 4.0}});
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            d     = 100.0;
            alpha = 1.0, beta = 2.0;
            x.assign({2.0});
            y.assign({1.0});
            csr_val.assign({10.0});
            d_exp = 0.0;
            y_exp.assign({2.0});
        }
        m = 1, n = 0, nnz = 0;
        {
            SCOPED_TRACE("1. m = 1, n = 0, nnz = 0");
            ASSERT_EQ(
                aoclsparse_create_csr<T>(
                    A, base, m, n, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_dotmv(
                          aoclsparse_operation_none, alpha, A, descr, x.data(), beta, y.data(), &d),
                      aoclsparse_status_success);
            verify_y_d(y, y_exp, d, d_exp);
            EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        }

        // 2) m=0, n=1, nnz=0
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            d = {100.0, 100.0};
            y.assign({{1.0, 1.0}});
            x.assign({{2.0, 2.0}});
            y_exp.assign({{1.0, 1.0}});
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            d = 100.0;
            x.assign({2.0});
            y.assign({1.0});
            y_exp.assign({1.0});
        }
        m = 0, n = 1, nnz = 0;
        {
            SCOPED_TRACE("1. m = 0, n = 1, nnz = 0");
            ASSERT_EQ(
                aoclsparse_create_csr<T>(
                    A, base, m, n, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_dotmv(
                          aoclsparse_operation_none, alpha, A, descr, x.data(), beta, y.data(), &d),
                      aoclsparse_status_success);
            verify_y_d(y, y_exp, d, d_exp);
            EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        }

        // 3) m=1, n=1, nnz=0
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            d = {100.0, 100.0};
            y.assign({{1.0, 1.0}});
            x.assign({{2.0, 2.0}});
            d_exp = {8.0, 8.0};
            y_exp.assign({{0.0, 4.0}});
        }
        else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            d = 100.0;
            x.assign({2.0});
            y.assign({1.0});
            y_exp.assign({2.0});
            d_exp = 4.0;
        }
        m = 1, n = 1, nnz = 0;
        {
            SCOPED_TRACE("1. m = 1, n = 1, nnz = 0");
            ASSERT_EQ(
                aoclsparse_create_csr<T>(
                    A, base, m, n, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_dotmv(
                          aoclsparse_operation_none, alpha, A, descr, x.data(), beta, y.data(), &d),
                      aoclsparse_status_success);

            verify_y_d(y, y_exp, d, d_exp);
            EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
            aoclsparse_destroy_mat_descr(descr);
        }
    }

    TEST(dotmv, NullArg)
    {
        test_dotmv_nullptr<float>();
        test_dotmv_nullptr<double>();
        test_dotmv_nullptr<aoclsparse_float_complex>();
        test_dotmv_nullptr<aoclsparse_double_complex>();
    }

    TEST(dotmv, WrongTypeSize)
    {
        test_mv_wrong_type_base_size<float>();
        test_mv_wrong_type_base_size<double>();
        test_mv_wrong_type_base_size<aoclsparse_float_complex>();
        test_mv_wrong_type_base_size<aoclsparse_double_complex>();
    }

    TEST(dotmv, NotImpl)
    {
        // CSC matrix not supported for dotmv
        aoclsparse_int              m = 2, n = 2, nnz = 2;
        std::vector<aoclsparse_int> col_ptr = {0, 1, 2};
        std::vector<aoclsparse_int> row_idx = {0, 1};
        std::vector<float>          val     = {10.0, 33.0};
        std::vector<float>          x       = {2.0, 2.0, 2.0};
        std::vector<float>          y       = {1.0, 1.0};
        aoclsparse_matrix           A       = nullptr;
        aoclsparse_mat_descr        descr   = nullptr;
        float                       d       = 111.0;
        float                       alpha = 1.0, beta = 2.0;

        ASSERT_EQ(aoclsparse_createcsc(A,
                                       aoclsparse_index_base_zero,
                                       m,
                                       n,
                                       nnz,
                                       col_ptr.data(),
                                       row_idx.data(),
                                       val.data()),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);

        EXPECT_EQ(aoclsparse_dotmv(
                      aoclsparse_operation_none, alpha, A, descr, x.data(), beta, y.data(), &d),
                  aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }

    TEST(dotmv, ZeroSizeMatSuccess)
    {
        test_zero_size_mat<double>(aoclsparse_index_base_zero);
        test_zero_size_mat<double>(aoclsparse_index_base_one);

        test_zero_size_mat<float>(aoclsparse_index_base_zero);
        test_zero_size_mat<float>(aoclsparse_index_base_one);

        test_zero_size_mat<aoclsparse_float_complex>(aoclsparse_index_base_zero);
        test_zero_size_mat<aoclsparse_float_complex>(aoclsparse_index_base_one);

        test_zero_size_mat<aoclsparse_double_complex>(aoclsparse_index_base_zero);
        test_zero_size_mat<aoclsparse_double_complex>(aoclsparse_index_base_one);
    }

    TEST(dotmv, SuccessGeneral)
    {
        test_dotmv_combs(
            "SuccessGeneral-ZeroBase", aoclsparse_matrix_type_general, aoclsparse_index_base_zero);
        test_dotmv_combs(
            "SuccessGeneral-OneBase", aoclsparse_matrix_type_general, aoclsparse_index_base_one);
    }

    TEST(dotmv, SuccessSymm)
    {
        test_dotmv_combs(
            "SuccessSymm-ZeroBase", aoclsparse_matrix_type_symmetric, aoclsparse_index_base_zero);
        test_dotmv_combs(
            "SuccessSymm-OneBase", aoclsparse_matrix_type_symmetric, aoclsparse_index_base_one);
    }

    TEST(dotmv, SuccessTraingular)
    {
        test_dotmv_combs("SuccessTraingular-ZeroBase",
                         aoclsparse_matrix_type_triangular,
                         aoclsparse_index_base_zero);
        test_dotmv_combs("SuccessTraingular-OneBase",
                         aoclsparse_matrix_type_triangular,
                         aoclsparse_index_base_one);
    }

    TEST(dotmv, SuccessHerm)
    {
        test_dotmv_combs(
            "SuccessHerm-ZeroBase", aoclsparse_matrix_type_hermitian, aoclsparse_index_base_zero);
        test_dotmv_combs(
            "SuccessHerm-OneBase", aoclsparse_matrix_type_hermitian, aoclsparse_index_base_one);
    }

} // namespace