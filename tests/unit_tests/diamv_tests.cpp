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
    void test_diamv_baseOneCSRInput()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 5, NNZ = 7;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T                    x[5]      = {1.0, 2.0, 3.0, 4.0, 5.0};
        T                    y[5]      = {0};
        T                    y_gold[5] = {6.00, 2.00, 13.00, 9.00, 40.00};
        aoclsparse_mat_descr descr;

        aoclsparse_int csr_row_ptr[6] = {1, 2, 3, 5, 7, 8}; //one-based indexing
        aoclsparse_int csr_col_ind[7] = {1, 2, 2, 3, 1, 4, 4}; //one-based indexing
        T              csr_val[7]     = {6.00, 1.00, 2.00, 3.00, 5.00, 1.00, 10.00};

        std::vector<aoclsparse_int> dia_offset;
        std::vector<T>              dia_val;
        aoclsparse_int              dia_num_diag;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        ASSERT_EQ(
            aoclsparse_csr2dia_ndiag(M, N, descr, NNZ, csr_row_ptr, csr_col_ind, &dia_num_diag),
            aoclsparse_status_success);

        aoclsparse_int size    = (M > N) ? M : N;
        aoclsparse_int nnz_dia = size * dia_num_diag;
        // Allocate DIA matrix
        dia_offset.resize(dia_num_diag);
        dia_val.resize(nnz_dia);
        // Convert CSR matrix to DIA
        ASSERT_EQ(aoclsparse_csr2dia(M,
                                     N,
                                     descr,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     dia_num_diag,
                                     dia_offset.data(),
                                     dia_val.data()),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_diamv<T>(trans,
                                      &alpha,
                                      M,
                                      N,
                                      NNZ,
                                      dia_val.data(),
                                      dia_offset.data(),
                                      dia_num_diag,
                                      descr,
                                      x,
                                      &beta,
                                      y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_diamv_baseOneDiaInput()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 5, NNZ = 12;
        aoclsparse_int       dia_num_diag = 5;
        T                    alpha        = 1.0;
        T                    beta         = 0.0;
        // Initialise vectors
        T                    x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T                    y[5];
        T                    y_gold[5]          = {9.00, 16.00, 54.00, 80.00, 32.00};
        int                  invalid_index_base = 2;
        aoclsparse_mat_descr descr;

        //assign y[] with NaN value to verify tests with zero beta
        for(int i = 0; i < M; i++)
        {
            y[i] = std::numeric_limits<double>::quiet_NaN();
        }

        // diagonal offset array is a distance array, that indicates where the nnz values
        // are wrt  diagonal +ve value means above diagonal and -ve means below diagonal
        aoclsparse_int dia_offset[5] = {-3, -2, -1, 0, 1};
        T              dia_val[25]
            = {0.00, 0.00, 0.00, 5.00, 1.00, 0.00, 0.00, 3.00, 0.00, 0.00, 0.00, 0.00, 1.00,
               9.00, 0.00, 9.00, 8.00, 7.00, 7.00, 6.00, 0.00, 0.00, 7.00, 4.00, 0.00};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        descr->base = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(
            aoclsparse_diamv<T>(
                trans, &alpha, M, N, NNZ, dia_val, dia_offset, dia_num_diag, descr, x, &beta, y),
            aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
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
        T x[]     = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[]     = {1.0, 2.0, 3.0, 4.0, 5.0};
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

    TEST(diamv, BaseOneDoubleDiaInput)
    {
        test_diamv_baseOneDiaInput<double>();
    }
    TEST(diamv, BaseOneFloatDiaInput)
    {
        test_diamv_baseOneDiaInput<double>();
    }
    TEST(diamv, BaseOneDoubleCSRInput)
    {
        test_diamv_baseOneCSRInput<double>();
    }
    TEST(diamv, BaseOneFloatCSRInput)
    {
        test_diamv_baseOneCSRInput<float>();
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
