/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"
#include "aoclsparse_reference.hpp"

#include <algorithm>

namespace
{
    template <typename T>
    void test_mv_gen(std::string testcase, aoclsparse_matrix_sort sort, aoclsparse_index_base base)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_int              M, N, NNZ;
        std::vector<aoclsparse_int> row_ptr_L, row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L, col_idx_U;
        std::vector<T>              val_L, val_U, x, y, exp_y;

        init_tcsr_matrix<T>(
            M, N, NNZ, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U, sort, base);

        T alpha, beta;
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            alpha = 1.0;
            beta  = 0.0;
            x.assign({1.0, 2.0, 3.0, 4.0});
            y.assign({1.0, 1.0, 1.0, 1.0});
            exp_y.assign({17.0, 8.0, -13.0, 59.0});
        }
        else if constexpr(std::is_same_v<T, std::complex<double>>
                          || std::is_same_v<T, std::complex<float>>
                          || std::is_same_v<T, aoclsparse_double_complex>
                          || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 1};
            beta  = {0, 0};
            x.assign({{1, 1}, {2, 2}, {3, 3}, {4, 4}});
            y.assign({{1, 1}, {1, 1}, {1, 1}, {1, 1}});
        }

        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, base);

        // Create TCSR matrix
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            M,
                                            N,
                                            NNZ,
                                            row_ptr_L.data(),
                                            row_ptr_U.data(),
                                            col_idx_L.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_success);
        // Full matrix spmv
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_status    status;
        // supports only double datatypes
        status = aoclsparse_mv<T>(trans, &alpha, A, descr, x.data(), &beta, y.data());
        if constexpr(std::is_same_v<T, double>)
        {
            EXPECT_EQ(status, aoclsparse_status_success);
            EXPECT_DOUBLE_EQ_VEC(M, y, exp_y);
        }
        else
            EXPECT_EQ(status, aoclsparse_status_not_implemented);

        // destroy matrix and descr
        aoclsparse_destroy_mat_descr(descr);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_triangular(std::string testcase, aoclsparse_index_base base)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_int              M, N, NNZ;
        std::vector<aoclsparse_int> row_ptr_L, row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L, col_idx_U;
        std::vector<T>              val_L, val_U, x, y;

        init_tcsr_matrix(M,
                         N,
                         NNZ,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         base);
        x.assign({1.0, 2.0, 3.0, 4.0});
        y.assign({1.0, 1.0, 1.0, 1.0});

        aoclsparse_operation op_n  = aoclsparse_operation_none;
        aoclsparse_operation op_t  = aoclsparse_operation_transpose;
        T                    alpha = 1.0;
        T                    beta  = 0.0;

        std::vector<T> exp_y_L({-1.0, 8.0, -13.0, 59.0});
        std::vector<T> exp_y_strict_L({0.0, 0.0, 5.0, 23.0});
        std::vector<T> exp_y_U({17.0, 8.0, -18.0, 36.0});
        std::vector<T> exp_y_strict_U({18.0, 0.0, 0.0, 0.0});
        std::vector<T> exp_y_trans_L({42.0, 40.0, -18.0, 36.0});
        std::vector<T> exp_y_trans_U({-1.0, 8.0, -16.0, 39.0});

        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, base);
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);

        // Create TCSR matrix
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            M,
                                            N,
                                            NNZ,
                                            row_ptr_L.data(),
                                            row_ptr_U.data(),
                                            col_idx_L.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_success);

        // triangular matrix type - fill lower
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        EXPECT_EQ(aoclsparse_mv<T>(op_n, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y, exp_y_L);

        // triangular matrix type - strictly lower
        aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_zero);
        EXPECT_EQ(aoclsparse_mv<T>(op_n, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y, exp_y_strict_L);

        // triangular matrix type - fill lower - transpose
        aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);
        EXPECT_EQ(aoclsparse_mv<T>(op_t, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y, exp_y_trans_L);

        // triangular matrix type - upper spmv
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        EXPECT_EQ(aoclsparse_mv<T>(op_n, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y, exp_y_U);

        // triangular matrix type - strictly upper spmv
        aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_zero);
        EXPECT_EQ(aoclsparse_mv<T>(op_n, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y, exp_y_strict_U);

        // triangular matrix type - fill upper - transpose
        aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);
        EXPECT_EQ(aoclsparse_mv<T>(op_t, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y, exp_y_trans_U);

        // destroy matrix and descr
        aoclsparse_destroy_mat_descr(descr);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_symmetric(std::string testcase, aoclsparse_index_base b)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_int              M = 4, N = 4, NNZ = 6;
        std::vector<aoclsparse_int> row_ptr_L, row_ptr_U, row_ptr;
        std::vector<aoclsparse_int> col_idx_L, col_idx_U, col_idx;
        std::vector<T>              val, x, y, exp_y;

        // Initialize matrix
        // 1 2 0 0
        // 2 3 4 0
        // 0 4 5 0
        // 0 0 0 6

        row_ptr_L.assign({b, 1 + b, 3 + b, 5 + b, 6 + b});
        row_ptr_U.assign({b, 2 + b, 4 + b, 5 + b, 6 + b});
        row_ptr.assign({b, 1 + b, 2 + b, 3 + b, 4 + b}); // diag row ptr
        col_idx_L.assign({b, b, 1 + b, 1 + b, 2 + b, 3 + b});
        col_idx_U.assign({b, 1 + b, 1 + b, 2 + b, 2 + b, 3 + b});
        col_idx.assign({b, 1 + b, 2 + b, 3 + b}); // indices of diagonals
        val.assign({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        x.assign({1.0, 2.0, 3.0, 4.0});
        y.assign({1.0, 1.0, 1.0, 1.0});
        exp_y.assign({5.0, 20.0, 23.0, 24.0});

        aoclsparse_operation trans = aoclsparse_operation_none;
        T                    alpha = 1.0;
        T                    beta  = 0.0;

        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, b);
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);

        // Symmetric matrix type - fill lower
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            b,
                                            M,
                                            N,
                                            NNZ,
                                            row_ptr_L.data(),
                                            row_ptr.data(),
                                            col_idx_L.data(),
                                            col_idx.data(),
                                            val.data(),
                                            val.data()),
                  aoclsparse_status_success);

        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y, exp_y);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);

        // Symmetric matrix type - fill upper
        ASSERT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            b,
                                            M,
                                            N,
                                            NNZ,
                                            row_ptr.data(),
                                            row_ptr_U.data(),
                                            col_idx.data(),
                                            col_idx_U.data(),
                                            val.data(),
                                            val.data()),
                  aoclsparse_status_success);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y, exp_y);

        // destroy matrix and descr
        aoclsparse_destroy_mat_descr(descr);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    template <typename T>
    void test_nullptr(std::string testcase, aoclsparse_index_base base)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_int              M, N, NNZ;
        std::vector<aoclsparse_int> row_ptr_L, row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L, col_idx_U;
        std::vector<T>              val_L, val_U, x, y;

        init_tcsr_matrix(M,
                         N,
                         NNZ,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         base);
        x.assign({1.0, 2.0, 3.0, 4.0});
        y.assign({1.0, 1.0, 1.0, 1.0});

        aoclsparse_operation trans = aoclsparse_operation_none;
        T                    alpha = 1.0;
        T                    beta  = 0.0;

        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, base);
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);

        // Create TCSR matrix
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            M,
                                            N,
                                            NNZ,
                                            row_ptr_L.data(),
                                            row_ptr_U.data(),
                                            col_idx_L.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_success);

        // pass nullptr and expect pointer
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, nullptr, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, nullptr, x.data(), &beta, y.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, nullptr, &beta, y.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x.data(), &beta, nullptr),
                  aoclsparse_status_invalid_pointer);

        // destroy matrix and descr
        aoclsparse_destroy_mat_descr(descr);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    template <typename T>
    void test_optimize_tcsr(std::string testcase, aoclsparse_index_base b)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_int              M, N, NNZ;
        std::vector<aoclsparse_int> row_ptr_L, row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L, col_idx_U;
        std::vector<T>              val_L, val_U, x, y;

        init_tcsr_matrix(M,
                         N,
                         NNZ,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         b);
        x.assign({1.0, 2.0, 3.0, 4.0});
        y.assign({1.0, 1.0, 1.0, 1.0});

        aoclsparse_operation        trans = aoclsparse_operation_none;
        T                           alpha = 1.0;
        T                           beta  = 0.0;
        std::vector<aoclsparse_int> exp_idiag({b, 1 + b, 3 + b, 6 + b});
        std::vector<aoclsparse_int> exp_iurow({1 + b, 4 + b, 5 + b, 6 + b});

        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, b);
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);

        // Create TCSR matrix
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            b,
                                            M,
                                            N,
                                            NNZ,
                                            row_ptr_L.data(),
                                            row_ptr_U.data(),
                                            col_idx_L.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_success);

        // Call optimize
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

        // Check for idiag and iurow
        EXPECT_EQ_VEC(M, A->tcsr_mat.idiag, exp_idiag);
        EXPECT_EQ_VEC(M, A->tcsr_mat.iurow, exp_iurow);

        // Full matrix spmv
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);

        // Destroy matrix and descr
        aoclsparse_destroy_mat_descr(descr);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    TEST(tcsrmv, General)
    {
        test_mv_gen<double>("SuccessGeneral-FullySorted-Double",
                            aoclsparse_fully_sorted,
                            aoclsparse_index_base_zero);
        test_mv_gen<double>("SuccessGeneral-PartiallySorted-Double",
                            aoclsparse_partially_sorted,
                            aoclsparse_index_base_zero);
        test_mv_gen<double>(
            "SuccessGeneral-Double-OneBase", aoclsparse_fully_sorted, aoclsparse_index_base_one);
        test_mv_gen<float>(
            "NotImplemented-General-Float", aoclsparse_fully_sorted, aoclsparse_index_base_zero);
        test_mv_gen<aoclsparse_float_complex>("NotImplementedGeneral-Complex-Float",
                                              aoclsparse_fully_sorted,
                                              aoclsparse_index_base_zero);
        test_mv_gen<aoclsparse_double_complex>("NotImplementedGeneral-Complex-Double",
                                               aoclsparse_fully_sorted,
                                               aoclsparse_index_base_zero);
    }

    TEST(tcsrmv, SuccessTriangular)
    {
        test_mv_triangular<double>("SuccessTriangular-Double-ZeroBase", aoclsparse_index_base_zero);
        test_mv_triangular<float>("SuccessTriangular-Float-ZeroBase", aoclsparse_index_base_zero);
        test_mv_triangular<double>("SuccessTriangular-Double-OneBase", aoclsparse_index_base_one);
        test_mv_triangular<float>("SuccessTriangular-Float-OneBase", aoclsparse_index_base_one);
    }
    TEST(tcsrmv, SuccessSymmetric)
    {
        test_mv_symmetric<double>("SuccessSymmetric-Double-ZeroBase", aoclsparse_index_base_zero);
        test_mv_symmetric<float>("SuccessSymmetric-Float-ZeroBase", aoclsparse_index_base_zero);
        test_mv_symmetric<double>("SuccessSymmetric-Double-OneBase", aoclsparse_index_base_one);
        test_mv_symmetric<float>("SuccessSymmetric-Float-OneBase", aoclsparse_index_base_one);
    }
    TEST(tcsrmv, NullArg)
    {
        test_nullptr<double>("NullArgument", aoclsparse_index_base_zero);
        test_nullptr<float>("NullArgument", aoclsparse_index_base_zero);
        test_nullptr<double>("NullArgument", aoclsparse_index_base_one);
        test_nullptr<float>("NullArgument", aoclsparse_index_base_one);
    }
    TEST(tcsrmv, OptmizeTCSR)
    {
        test_optimize_tcsr<double>("OptimizeTCSR - iurow, idiag checks",
                                   aoclsparse_index_base_zero);
        test_optimize_tcsr<float>("OptimizeTCSR - iurow, idiag checks", aoclsparse_index_base_zero);
        test_optimize_tcsr<double>("OptimizeTCSR - iurow, idiag checks", aoclsparse_index_base_one);
        test_optimize_tcsr<float>("OptimizeTCSR - iurow, idiag checks", aoclsparse_index_base_one);
    }
    // All the other checks are addressed in mv_tests.cpp
}