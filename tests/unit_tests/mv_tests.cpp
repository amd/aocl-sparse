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
#include "aoclsparse_reference.hpp"

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
        T exp_y_l[] = {1, 6, 12, 56, 16};
        T exp_y_u[] = {9, 6, 12, 28, 0};

        for(int i = 0; i < M; i++)
        {
            y[i] = 0.0;
        }

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
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);

        for(int i = 0; i < M; i++)
        {
            y[i] = 0.0;
        }
        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y_u);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_trianglular_transpose()
    {
        aoclsparse_operation trans;
        int                  invalid_trans = 114;
        aoclsparse_int       M = 5, N = 5, NNZ = 14;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T                     x[5]      = {1.0, 2.0, 3.0, 4.0, 5.0};
        T                     y[5]      = {0.0};
        T                     y_gold[5] = {0.0};
        aoclsparse_mat_descr  descr;
        aoclsparse_index_base base = aoclsparse_index_base_zero;

        aoclsparse_int csr_row_ptr[] = {0, 4, 6, 9, 11, 14};
        aoclsparse_int csr_col_ind[] = {0, 2, 3, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4};
        T              csr_val[]     = {1, 5, 3, 9, 3, 1, 2, 4, 6, 7, 6, 13, 9, 8};

        aoclsparse_matrix      A;
        aoclsparse_matrix_type mattype;
        aoclsparse_fill_mode   fill;
        aoclsparse_diag_type   diag;

        //assign y[] with NaN value to verify tests with zero beta
        for(int i = 0; i < M; i++)
        {
            y[i] = std::numeric_limits<double>::quiet_NaN();
        }
        trans   = aoclsparse_operation_transpose;
        mattype = aoclsparse_matrix_type_triangular;
        fill    = aoclsparse_fill_mode_lower;
        diag    = aoclsparse_diag_type_non_unit;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        //CASE 1: Check transpose case for lower triangular SPMV, with alpha=1.0, beta=0.0
        /*
            1	0	5	3	9
            0	3	0	1	0
            2	0	4	0	6
            0	7	0	6	0
            13	0	9	0	8
        */
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descr, diag), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_EQ(
            ref_csrmvtrgt(
                alpha, M, N, csr_val, csr_col_ind, csr_row_ptr, fill, diag, base, x, beta, y_gold),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        //CASE 2: Check transpose case for lower triangular SPMV, with alpha=5.1, beta=3.2
        alpha = 5.1;
        beta  = 3.2;
        //assign y[] and y_gold_2[] with non-zero initial value to verify tests with non-zero beta
        for(int i = 0; i < M; i++)
        {
            y[i]      = 10. + i;
            y_gold[i] = 10. + i;
        }
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_EQ(
            ref_csrmvtrgt(
                alpha, M, N, csr_val, csr_col_ind, csr_row_ptr, fill, diag, base, x, beta, y_gold),
            aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);

        //CASE 3: Check transpose case for upper triangular SPMV, with alpha=1.0, beta=0.0
        alpha = 1.0;
        beta  = 0.0;
        /*
            1	0	5	3	9
            0	3	0	1	0
            2	0	4	0	6
            0	7	0	6	0
            13	0	9	0	8
        */
        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);
        fill = aoclsparse_fill_mode_upper;
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        //assign y[] with NaN value to verify tests with zero beta
        for(int i = 0; i < M; i++)
        {
            y[i]      = std::numeric_limits<double>::quiet_NaN();
            y_gold[i] = 0.0;
        }

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_EQ(
            ref_csrmvtrgt(
                alpha, M, N, csr_val, csr_col_ind, csr_row_ptr, fill, diag, base, x, beta, y_gold),
            aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        //CASE 4: Check transpose case for upper triangular SPMV, with alpha=5.1, beta=3.2
        alpha = 5.1;
        beta  = 3.2;
        //assign y[] and y_gold_2[] with non-zero initial value to verify tests with non-zero beta
        for(int i = 0; i < M; i++)
        {
            y[i]      = 10. + i;
            y_gold[i] = 10. + i;
        }
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_EQ(
            ref_csrmvtrgt(
                alpha, M, N, csr_val, csr_col_ind, csr_row_ptr, fill, diag, base, x, beta, y_gold),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        //CASE 5: Check for invalid transpose value
        trans = (aoclsparse_operation)invalid_trans;
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_symm_transpose()
    {
        aoclsparse_operation trans;
        aoclsparse_int       M = 8, N = 8, NNZ = 18;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[8]      = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        T y[8]      = {0.0};
        T y_gold[8] = {0.0};

        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_mat_descr  descr;
        aoclsparse_matrix     A;
        //symmetric matrix with lower triangle
        aoclsparse_int csr_row_ptr[] = {0, 1, 2, 5, 6, 8, 11, 15, 18};
        aoclsparse_int csr_col_ind[] = {0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
        T              csr_val[]     = {19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9};

        //the same matrix above in tranposed state, so it is upper triangle only
        aoclsparse_int csc_col_ptr[] = {0, 4, 7, 9, 11, 14, 16, 17, 18};
        aoclsparse_int csc_row_ind[] = {0, 2, 5, 6, 1, 2, 4, 2, 7, 3, 6, 4, 5, 6, 5, 7, 6, 7};
        T              csc_val[]
            = {19., 1., 2., 7., 10., 8., 2., 11., 5., 13., 9., 11., 1., 5., 9., 5., 12., 9.};

        trans = aoclsparse_operation_transpose;

        //assign y[] with NaN value to verify tests with zero beta
        for(int i = 0; i < M; i++)
        {
            y[i] = std::numeric_limits<double>::quiet_NaN();
        }

        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_EQ(ref_csrmvsym(alpha,
                               M,
                               csc_val,
                               csc_row_ind,
                               csc_col_ptr,
                               aoclsparse_fill_mode_upper,
                               aoclsparse_diag_type_non_unit,
                               base,
                               x,
                               beta,
                               y_gold),
                  aoclsparse_status_success);

        //output of symmetric-spmv on CSR-Sparse matrix of lower triangle is compared against
        //symmetric-spmv on CSC-Sparse or transposed CSR matrix of upper triangle, which means
        //SPMV and SPMV-Transposed on symmetric sparse matrices yield same results
        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_conjugate_transpose()
    {
        aoclsparse_operation trans;
        aoclsparse_int       M = 5, N = 5, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[5] = {0.0};

        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_mat_descr  descr;
        aoclsparse_matrix     A;
        aoclsparse_int        csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int        csr_col_ind[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T                     csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};

        trans = aoclsparse_operation_conjugate_transpose;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        aoclsparse_create_csr<T>(A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        //TODO: conjugate transpose should work without any error, once the support is added.
        //The expected return code should be aoclsparse_status_success and it should work.
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_not_implemented);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
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
    TEST(mv, TriangTransposeDouble)
    {
        test_mv_trianglular_transpose<double>();
    }
    TEST(mv, TriangTransposeFloat)
    {
        test_mv_trianglular_transpose<float>();
    }
    TEST(mv, SymmTransposeDouble)
    {
        test_mv_symm_transpose<double>();
    }
    TEST(mv, SymmTransposeFloat)
    {
        test_mv_symm_transpose<float>();
    }
    TEST(mv, ConjugateTransposeDouble)
    {
        test_mv_conjugate_transpose<double>();
    }
    TEST(mv, ConjugateTransposeFloat)
    {
        test_mv_conjugate_transpose<float>();
    }
} // namespace
