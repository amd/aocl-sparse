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

    // Several tests in one when nullptr is passed instead of valid data
    template <typename T>
    void test_diamv_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 5, NNZ = 5;
        aoclsparse_int       dia_num_diag = 1;
        T                    alpha        = 1.0;
        T                    beta         = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[M];

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_int dia_offset[] = {0};
        T              dia_val[]    = {0.1, 0.22, 3.1, 1.0, -1.1};

        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, nullptr, M, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, N, NNZ, nullptr, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_diamv<T>(
                      trans, &alpha, M, N, NNZ, dia_val, nullptr, dia_num_diag, descr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, N, NNZ, dia_val, dia_offset, dia_num_diag, nullptr, x, &beta, y),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_diamv<T>(trans,
                                      &alpha,
                                      M,
                                      N,
                                      NNZ,
                                      dia_val,
                                      dia_offset,
                                      dia_num_diag,
                                      descr,
                                      nullptr,
                                      &beta,
                                      y),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, nullptr, y),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_diamv<T>(trans,
                                      &alpha,
                                      M,
                                      N,
                                      NNZ,
                                      dia_val,
                                      dia_offset,
                                      dia_num_diag,
                                      descr,
                                      x,
                                      &beta,
                                      nullptr),
                  aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
    }

    template <typename T>
    void test_diamv_wrong_type_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 5, NNZ = 5;
        aoclsparse_int       dia_num_diag = 1;
        T                    alpha        = 1.0;
        T                    beta         = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[M];

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_int dia_offset[] = {0};
        T              dia_val[]    = {0.1, 0.22, 3.1, 1.0, -1.1};

        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, -1, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_invalid_size);
        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, -1, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_diamv<T>(
                      trans, &alpha, M, N, NNZ, dia_val, dia_offset, -1, descr, x, &beta, y),
                  aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
    }

    template <typename T>
    void test_diamv_not_implemented()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 5, NNZ = 5;
        aoclsparse_int       dia_num_diag = 1;
        T                    alpha        = 1.0;
        T                    beta         = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[M];

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_int dia_offset[] = {0};
        T              dia_val[]    = {0.1, 0.22, 3.1, 1.0, -1.1};

        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_not_implemented);
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);
        trans = aoclsparse_operation_transpose;
        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_not_implemented);
        trans = aoclsparse_operation_none;
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_hermitian);
        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descr);
    }

    template <typename T>
    void test_diamv_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 5, NNZ = 5;
        aoclsparse_int       dia_num_diag = 1;
        T                    alpha        = 1.0;
        T                    beta         = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T exp_y[] = {1.0, 2.0, 3.0, 4.0, 5.0};

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_int dia_offset[] = {0};
        T              dia_val[]    = {0.1, 0.22, 3.1, 1.0, -1.1};

        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, 0, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y);
        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, 0, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y);
        EXPECT_EQ(aoclsparse_diamv<T>(
                      trans, &alpha, M, N, NNZ, dia_val, dia_offset, 0, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y);
        aoclsparse_destroy_mat_descr(descr);
    }

    TEST(diamv, NullArgDouble)
    {
        test_diamv_nullptr<double>();
    }
    TEST(diamv, NullArgFloat)
    {
        test_diamv_nullptr<float>();
    }

    TEST(diamv, WrongSizeDouble)
    {
        test_diamv_wrong_type_size<double>();
    }
    TEST(diamv, WrongSizeFloat)
    {
        test_diamv_wrong_type_size<float>();
    }
    TEST(diamv, NotImplDouble)
    {
        test_diamv_not_implemented<double>();
    }
    TEST(diamv, NotImplFloat)
    {
        test_diamv_not_implemented<float>();
    }
    TEST(diamv, DoNothingDouble)
    {
        test_diamv_do_nothing<double>();
    }
    TEST(diamv, DoNothingFloat)
    {
        test_diamv_do_nothing<float>();
    }

} // namespace
