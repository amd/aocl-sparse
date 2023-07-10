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

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_csrmv_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 3, nnz = 1;
        T                    csr_val[]     = {42.};
        aoclsparse_int       csr_col_ind[] = {1};
        aoclsparse_int       csr_row_ptr[] = {0, 0, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        //EXPECT_EQ(aoclsparse_csrmv<T>(trans, &alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
        //      aoclsparse_status_invalid_pointer);
        //FIXME crashes: EXPECT_EQ(aoclsparse_csrmv<T>(trans, nullptr, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
        //      aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, n, nnz, nullptr, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmv<T>(
                      trans, &alpha, m, n, nnz, csr_val, nullptr, csr_row_ptr, descr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmv<T>(
                      trans, &alpha, m, n, nnz, csr_val, csr_col_ind, nullptr, descr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, nullptr, x, &beta, y),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmv<T>(trans,
                                      &alpha,
                                      m,
                                      n,
                                      nnz,
                                      csr_val,
                                      csr_col_ind,
                                      csr_row_ptr,
                                      descr,
                                      nullptr,
                                      &beta,
                                      y),
                  aoclsparse_status_invalid_pointer);
        //FIXME crashes: EXPECT_EQ(aoclsparse_csrmv<T>(trans, &alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, nullptr, y),
        //      aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmv<T>(trans,
                                      &alpha,
                                      m,
                                      n,
                                      nnz,
                                      csr_val,
                                      csr_col_ind,
                                      csr_row_ptr,
                                      descr,
                                      x,
                                      &beta,
                                      nullptr),
                  aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
    }

    // tests with wrong scalar data n, m, nnz
    template <typename T>
    void test_csrmv_wrong_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 3, nnz = 1, wrong = -1;
        T                    csr_val[]     = {42.};
        aoclsparse_int       csr_col_ind[] = {1};
        aoclsparse_int       csr_row_ptr[] = {0, 0, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);

        // In turns pass wrong size in place of n, m, nnz
        //EXPECT_EQ(aoclsparse_csrmv<T>(trans, &alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
        //      aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_csrmv<T>(trans,
                                      &alpha,
                                      wrong,
                                      n,
                                      nnz,
                                      csr_val,
                                      csr_col_ind,
                                      csr_row_ptr,
                                      descr,
                                      x,
                                      &beta,
                                      y),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_csrmv<T>(trans,
                                      &alpha,
                                      m,
                                      wrong,
                                      nnz,
                                      csr_val,
                                      csr_col_ind,
                                      csr_row_ptr,
                                      descr,
                                      x,
                                      &beta,
                                      y),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, n, wrong, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
    }

    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_csrmv_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 3, nnz = 1;
        T                    csr_val[]     = {42.};
        aoclsparse_int       csr_col_ind[] = {1};
        aoclsparse_int       csr_row_ptr[] = {0, 0, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);

        // Passing zero size matrix should be OK
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, 0, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, 0, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, n, 0, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);

        aoclsparse_destroy_mat_descr(descr);
    }
    template <typename T>
    void test_csrmv_baseOneIndexing()
    {
        aoclsparse_operation trans              = aoclsparse_operation_none;
        int                  invalid_index_base = 2;
        aoclsparse_int       M = 5, N = 5, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[M];
        T y_gold[] = {9.00, 6.00, 12.00, 69.00, 40.00};

        aoclsparse_index_base base = aoclsparse_index_base_one;
        aoclsparse_mat_descr  descr;
        aoclsparse_int        csr_row_ptr[] = {1, 3, 4, 5, 8, 9};
        aoclsparse_int        csr_col_ind[] = {1, 4, 2, 3, 2, 4, 5, 5};
        T                     csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, M, N, NNZ, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        descr->base = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, M, N, NNZ, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_invalid_value);

        aoclsparse_destroy_mat_descr(descr);
    }
    template <typename T>
    void test_csrmv_transpose()
    {
        aoclsparse_operation trans;
        int                  invalid_trans = 114;
        aoclsparse_int       M = 5, N = 5, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[5]      = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[5]      = {0.0};
        T y_gold[5] = {0.0};

        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_mat_descr  descr;
        aoclsparse_int        csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int        csr_col_ind[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T                     csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};

        trans = aoclsparse_operation_transpose;

        //assign y[] with NaN value to verify tests with zero beta
        for(int i = 0; i < M; i++)
        {
            y[i] = std::numeric_limits<double>::quiet_NaN();
        }

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, M, N, NNZ, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);

        EXPECT_EQ(ref_csrmvt(alpha, M, N, csr_val, csr_col_ind, csr_row_ptr, base, x, beta, y_gold),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        // Reset y, y_gold to known values as they will be used with nonzero beta
        for(aoclsparse_int i = 0; i < N; i++)
        {
            y[i]      = 10. + i;
            y_gold[i] = 10. + i;
        }
        alpha = 5.1;
        beta  = 3.2;
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, M, N, NNZ, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_EQ(ref_csrmvt(alpha, M, N, csr_val, csr_col_ind, csr_row_ptr, base, x, beta, y_gold),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        //Check for invalid transpose value
        trans = (aoclsparse_operation)invalid_trans;
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, M, N, NNZ, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_csrmv_symm_transpose()
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

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, M, N, NNZ, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
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
    }
    template <typename T>
    void test_csrmv_conjugate_transpose()
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
        aoclsparse_int        csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int        csr_col_ind[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T                     csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};

        trans = aoclsparse_operation_conjugate_transpose;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        //TODO: conjugate transpose should work without any error, once the support is added.
        //The expected return code should be aoclsparse_status_success and it should work.
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, M, N, NNZ, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_not_implemented);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_csrmv_symmetric_baseone()
    {
        aoclsparse_operation trans;
        aoclsparse_int       M = 8, N = 8, NNZ = 18;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[8]      = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        T y[8]      = {0.0};
        T y_gold[8] = {0.0};

        aoclsparse_index_base base = aoclsparse_index_base_one;
        aoclsparse_mat_descr  descr;
        //symmetric matrix with lower triangle
        aoclsparse_int csr_row_ptr[] = {1, 2, 3, 6, 7, 9, 12, 16, 19};
        aoclsparse_int csr_col_ind[] = {1, 2, 1, 2, 3, 4, 2, 5, 1, 5, 6, 1, 4, 5, 7, 3, 6, 8};
        T              csr_val[]     = {19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9};

        trans = aoclsparse_operation_none;

        //assign y[] with NaN value to verify tests with zero beta
        for(int i = 0; i < M; i++)
        {
            y[i]      = std::numeric_limits<double>::quiet_NaN();
            y_gold[i] = std::numeric_limits<double>::quiet_NaN();
        }

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, M, N, NNZ, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);

        EXPECT_EQ(ref_csrmvsym(alpha,
                               M,
                               csr_val,
                               csr_col_ind,
                               csr_row_ptr,
                               aoclsparse_fill_mode_lower,
                               aoclsparse_diag_type_non_unit,
                               base,
                               x,
                               beta,
                               y_gold),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    //TODO add:
    // * positive tests with special predefined matrices
    // * positive tests with unsorted CSR
    // * whole branch for symmetric matrix computation
    // * not supported/implemented
    // * invalid array data (but we don't test these right now, e.g., col_ind out of bounds)
    // * nnz not matching row_ptr
    //

    TEST(csrmv, NullArgDouble)
    {
        test_csrmv_nullptr<double>();
    }
    TEST(csrmv, NullArgFloat)
    {
        test_csrmv_nullptr<float>();
    }

    TEST(csrmv, WrongSizeDouble)
    {
        test_csrmv_wrong_size<double>();
    }
    TEST(csrmv, WrongSizeFloat)
    {
        test_csrmv_wrong_size<float>();
    }

    TEST(csrmv, DoNothingDouble)
    {
        test_csrmv_do_nothing<double>();
    }
    TEST(csrmv, DoNothingFloat)
    {
        test_csrmv_do_nothing<float>();
    }
    TEST(csrmv, TransposeDouble)
    {
        test_csrmv_transpose<double>();
    }
    TEST(csrmv, TransposeFloat)
    {
        test_csrmv_transpose<float>();
    }
    TEST(csrmv, SymmTransposeDouble)
    {
        test_csrmv_symm_transpose<double>();
    }
    TEST(csrmv, SymmTransposeFloat)
    {
        test_csrmv_symm_transpose<float>();
    }
    TEST(csrmv, ConjugateTransposeDouble)
    {
        test_csrmv_conjugate_transpose<double>();
    }
    TEST(csrmv, ConjugateTransposeFloat)
    {
        test_csrmv_conjugate_transpose<float>();
    }
    TEST(csrmv, BaseOneDouble)
    {
        test_csrmv_baseOneIndexing<double>();
    }
    TEST(csrmv, BaseOneFloat)
    {
        test_csrmv_baseOneIndexing<float>();
    }
    TEST(csrmv, SymmetricBaseOneDouble)
    {
        test_csrmv_symmetric_baseone<double>();
    }
    TEST(csrmv, SymmetricBaseOneFloat)
    {
        test_csrmv_symmetric_baseone<float>();
    }
} // namespace
