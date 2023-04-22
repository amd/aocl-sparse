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
    void test_mv_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 5, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[M];

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_index_base base = aoclsparse_index_base_zero;

        // Initialise matrix
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  4  0  0
        //  0  5  0  6  7
        //  0  0  0  0  8
        aoclsparse_int    csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int    csr_col_ind[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T                 csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};
        aoclsparse_matrix A;
        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        // pass nullptr and expect pointer error
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, nullptr, descr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, nullptr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, nullptr, &beta, y),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, nullptr),
                  aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_wrong_type_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 4, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0};
        T y[M];

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_index_base base = aoclsparse_index_base_zero;

        aoclsparse_int    csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int    csr_col_ind[] = {0, 3, 1, 2, 1, 2, 3, 1};
        T                 csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};
        aoclsparse_matrix A;
        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        if(std::is_same_v<T, double>)
        {
            A->val_type = aoclsparse_smat;
        }
        else if(std::is_same_v<T, float>)
        {
            A->val_type = aoclsparse_dmat;
        }
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_wrong_type);

        if(std::is_same_v<T, double>)
        {
            A->val_type = aoclsparse_dmat;
        }
        else if(std::is_same_v<T, float>)
        {
            A->val_type = aoclsparse_smat;
        }
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_invalid_value);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_not_implemented()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 5, N = 5, NNZ = 8;
        T              alpha = 1.0;
        T              beta  = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[M];

        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);

        aoclsparse_int    csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int    csr_col_ind[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T                 csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};
        aoclsparse_matrix A;
        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_not_implemented);

        trans = aoclsparse_operation_transpose;
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_not_implemented);

        trans = aoclsparse_operation_none;
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_hermitian);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 1, N = 1, NNZ = 1;
        T              alpha = 1.0;
        T              beta  = 0.0;
        // Initialise vectors
        T x[N];
        T y[M];

        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);

        aoclsparse_int    csr_row_ptr[] = {0};
        aoclsparse_int    csr_col_ind[] = {0};
        T                 csr_val[]     = {0};
        aoclsparse_matrix AM0, AN0;
        aoclsparse_create_csr<T>(AM0, base, 0, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);
        aoclsparse_create_csr<T>(AN0, base, M, 0, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, AM0, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, AN0, descr, x, &beta, y),
                  aoclsparse_status_success);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(AM0), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(AN0), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_success()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 4, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0};
        T y[M];
        T exp_y_l[] = {1, 6, 12, 56, 8};
        T exp_y_u[] = {9, 6, 12, 28, 0};

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_index_base base = aoclsparse_index_base_zero;

        aoclsparse_int    csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int    csr_col_ind[] = {0, 3, 1, 2, 1, 2, 3, 1};
        T                 csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};
        aoclsparse_matrix A;
        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y_l);

        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y_u);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }

    TEST(mv, NullArgDouble)
    {
        test_mv_nullptr<double>();
    }
    TEST(mv, NullArgFloat)
    {
        test_mv_nullptr<float>();
    }
    TEST(mv, WrongSizeDouble)
    {
        test_mv_wrong_type_size<double>();
    }
    TEST(mv, WrongSizeFloat)
    {
        test_mv_wrong_type_size<float>();
    }
    TEST(mv, NotImplDouble)
    {
        test_mv_not_implemented<double>();
    }
    TEST(mv, NotImplFloat)
    {
        test_mv_not_implemented<float>();
    }
    TEST(mv, DoNothingDouble)
    {
        test_mv_do_nothing<double>();
    }
    TEST(mv, DoNothingFloat)
    {
        test_mv_do_nothing<float>();
    }

    TEST(mv, SuccessDouble)
    {
        test_mv_success<double>();
    }
    TEST(mv, SuccessFloat)
    {
        test_mv_success<float>();
    }

} // namespace
