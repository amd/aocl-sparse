/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <complex>
#include <iostream>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "blis.hh"
#pragma GCC diagnostic pop
#include "cblas.hh"
namespace
{
    aoclsparse_order      col  = aoclsparse_order_column;
    aoclsparse_order      row  = aoclsparse_order_row;
    aoclsparse_operation  op_t = aoclsparse_operation_transpose;
    aoclsparse_operation  op_h = aoclsparse_operation_conjugate_transpose;
    aoclsparse_operation  op_n = aoclsparse_operation_none;
    aoclsparse_index_base zero = aoclsparse_index_base_zero;
    aoclsparse_index_base one  = aoclsparse_index_base_one;

    // wrong sizes
    template <typename T>
    void test_wrong_size(aoclsparse_matrix    &A,
                         aoclsparse_matrix    &B,
                         aoclsparse_mat_descr &descrA,
                         aoclsparse_mat_descr &descrB,
                         std::vector<T>       &dense_c)
    {
        aoclsparse_operation op_a, op_b;
        T                    alpha, beta;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
        }
        else
        {
            alpha = 3.0;
            beta  = -2.0;
        }

        op_a = op_n;
        op_b = op_n;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, A->m, -1),
                  aoclsparse_status_invalid_size);
        op_a = op_n;
        op_b = op_t;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, A->m, -1),
                  aoclsparse_status_invalid_size);
        op_a = op_t;
        op_b = op_t;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, A->m, -1),
                  aoclsparse_status_invalid_size);

        op_a = op_t;
        op_b = op_n;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, A->m, -1),
                  aoclsparse_status_invalid_size);

        op_a = op_n;
        op_b = op_n;
        A->m = 5;
        A->n = 3;
        B->m = 3;
        B->n = 3;
        EXPECT_EQ(
            aoclsparse_sp2md(
                op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, B->n - 1, -1),
            aoclsparse_status_invalid_size);
        EXPECT_EQ(
            aoclsparse_sp2md(
                op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), col, A->m - 1, -1),
            aoclsparse_status_invalid_size);
    }
    template <typename T>
    void test_invalid_value(aoclsparse_matrix    &A,
                            aoclsparse_matrix    &B,
                            aoclsparse_mat_descr &descrA,
                            aoclsparse_mat_descr &descrB,
                            std::vector<T>       &dense_c)
    {
        aoclsparse_operation op_a, op_b;
        T                    alpha, beta;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
        }
        else
        {
            alpha = 3.0;
            beta  = -2.0;
        }

        op_a    = op_n;
        op_b    = op_n;
        A->m    = 5;
        A->n    = 3;
        B->m    = 3;
        B->n    = 3;
        A->base = one;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, B->n, -1),
                  aoclsparse_status_invalid_value);
        A->base = zero;
        B->base = one;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), col, A->m, -1),
                  aoclsparse_status_invalid_value);
        A->base = zero;
        B->base = zero;
        EXPECT_EQ(aoclsparse_sp2md(op_a,
                                   descrA,
                                   A,
                                   op_b,
                                   descrB,
                                   B,
                                   alpha,
                                   beta,
                                   dense_c.data(),
                                   (aoclsparse_order)3,
                                   A->m,
                                   -1),
                  aoclsparse_status_invalid_value);
    }
    // not implemented
    template <typename T>
    void test_not_impl(aoclsparse_matrix    &A,
                       aoclsparse_matrix    &B,
                       aoclsparse_mat_descr &descrA,
                       aoclsparse_mat_descr &descrB,
                       std::vector<T>       &dense_c)
    {
        aoclsparse_operation op_a, op_b;
        T                    alpha, beta;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
        }
        else
        {
            alpha = 3.0;
            beta  = -2.0;
        }

        op_a        = op_n;
        op_b        = op_n;
        A->mat_type = aoclsparse_csc_mat;
        B->mat_type = aoclsparse_csc_mat;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, B->m, -1),
                  aoclsparse_status_not_implemented);

        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, B->m, -1),
                  aoclsparse_status_not_implemented);

        aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_general);
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, B->m, -1),
                  aoclsparse_status_not_implemented);
    }

    // wrong type
    template <typename T>
    void test_wrong_type(aoclsparse_matrix    &A,
                         aoclsparse_matrix    &B,
                         aoclsparse_mat_descr &descrA,
                         aoclsparse_mat_descr &descrB,
                         std::vector<T>       &dense_c)
    {
        aoclsparse_operation op_a, op_b;
        T                    alpha, beta;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
        }
        else
        {
            alpha = 3.0;
            beta  = -2.0;
        }

        op_a        = op_n;
        op_b        = op_n;
        A->val_type = aoclsparse_cmat;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, B->m, -1),
                  aoclsparse_status_wrong_type);
        B->val_type = aoclsparse_cmat;
        A->val_type = aoclsparse_dmat;
        EXPECT_EQ(aoclsparse_sp2md(
                      op_a, descrA, A, op_b, descrB, B, alpha, beta, dense_c.data(), row, B->m, -1),
                  aoclsparse_status_wrong_type);
    }

    // test failure cases
    template <typename T>
    void test_failures(aoclsparse_int id)
    {
        aoclsparse_seedrand();
        aoclsparse_int              m_a = 5, n_a = 3, m_b = 4, n_b = 6;
        aoclsparse_int              nnz_a = 10, nnz_b = 8;
        aoclsparse_index_base       b_a = zero, b_b = zero;
        std::vector<T>              dense_c;
        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        bool                        issymm = true;
        ASSERT_EQ(aoclsparse_init_csr_matrix(
                      row_ptr_a,
                      col_ind_a,
                      val_a,
                      m_a,
                      n_a,
                      nnz_a,
                      b_a,
                      aoclsparse_matrix_random, /*random matrix, diagonal dominance not guaranteed*/
                      nullptr, /*no file to be read*/
                      issymm, /*unused for random matrix generation*/
                      true, /*unused for random matrix generation*/
                      aoclsparse_fully_sorted), /*fully sorted value and col index buffers*/
                  aoclsparse_status_success);
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, b_a, m_a, n_a, nnz_a, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

        std::vector<T>              val_b;
        std::vector<aoclsparse_int> col_ind_b;
        std::vector<aoclsparse_int> row_ptr_b;
        ASSERT_EQ(aoclsparse_init_csr_matrix(
                      row_ptr_b,
                      col_ind_b,
                      val_b,
                      m_b,
                      n_b,
                      nnz_b,
                      b_b,
                      aoclsparse_matrix_random, /*random matrix, diagonal dominance not guaranteed*/
                      nullptr, /*no file to be read*/
                      issymm, /*unused for random matrix generation*/
                      true, /*unused for random matrix generation*/
                      aoclsparse_fully_sorted), /*fully sorted value and col index buffers*/
                  aoclsparse_status_success);
        aoclsparse_matrix B;
        ASSERT_EQ(aoclsparse_create_csr(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);
        dense_c.resize(m_a * n_b);
        if(id == 0)
            test_wrong_size(A, B, descrA, descrB, dense_c);
        else if(id == 1)
            test_wrong_type(A, B, descrA, descrB, dense_c);
        else if(id == 2)
            test_not_impl(A, B, descrA, descrB, dense_c);
        else if(id == 3)
            test_invalid_value(A, B, descrA, descrB, dense_c);

        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
    }

    // test success cases
    template <typename T>
    void test_sp2md_success(aoclsparse_int        m_a,
                            aoclsparse_int        n_a,
                            aoclsparse_int        m_b,
                            aoclsparse_int        n_b,
                            aoclsparse_int        nnz_a,
                            aoclsparse_int        nnz_b,
                            aoclsparse_index_base b_a,
                            aoclsparse_index_base b_b,
                            aoclsparse_operation  op_a,
                            aoclsparse_operation  op_b,
                            aoclsparse_order      layout,
                            aoclsparse_int        ldc    = -1,
                            aoclsparse_int        offset = 0,
                            aoclsparse_int        scalar = 0)
    {
        aoclsparse_int m_c = 0, n_c = 0, lda, ldb, dense_c_sz;
        CBLAS_ORDER    blis_layout;
        if(layout == aoclsparse_order_row)
        {
            blis_layout = CblasRowMajor;
        }
        else
        {
            blis_layout = CblasColMajor;
        }

        aoclsparse_seedrand();

        std::vector<T> dense_a(m_a * n_a), dense_b(m_b * n_b), dense_c, dense_c_exp;
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        bool                        issymm = true;
        ASSERT_EQ(aoclsparse_init_csr_matrix(
                      row_ptr_a,
                      col_ind_a,
                      val_a,
                      m_a,
                      n_a,
                      nnz_a,
                      b_a,
                      aoclsparse_matrix_random, /*random matrix, diagonal dominance not guaranteed*/
                      nullptr, /*no file to be read*/
                      issymm, /*unused for random matrix generation*/
                      true, /*unused for random matrix generation*/
                      aoclsparse_fully_sorted), /*fully sorted value and col index buffers*/
                  aoclsparse_status_success);

        if(val_a.size() == 0)
            val_a.reserve(1);
        if(col_ind_a.size() == 0)
            col_ind_a.reserve(1);
        if(row_ptr_a.size() == 0)
            row_ptr_a.reserve(1);
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, b_a, m_a, n_a, nnz_a, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

        std::vector<T>              val_b;
        std::vector<aoclsparse_int> col_ind_b;
        std::vector<aoclsparse_int> row_ptr_b;
        ASSERT_EQ(aoclsparse_init_csr_matrix(
                      row_ptr_b,
                      col_ind_b,
                      val_b,
                      m_b,
                      n_b,
                      nnz_b,
                      b_b,
                      aoclsparse_matrix_random, /*random matrix, diagonal dominance not guaranteed*/
                      nullptr, /*no file to be read*/
                      issymm, /*unused for random matrix generation*/
                      true, /*unused for random matrix generation*/
                      aoclsparse_fully_sorted), /*fully sorted value and col index buffers*/
                  aoclsparse_status_success);
        aoclsparse_matrix B;

        if(val_b.size() == 0)
            val_b.reserve(1);
        if(col_ind_b.size() == 0)
            col_ind_b.reserve(1);
        if(row_ptr_b.size() == 0)
            row_ptr_b.reserve(1);

        ASSERT_EQ(aoclsparse_create_csr(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);
        aoclsparse_int m, n, k = 0;
        if((op_a == aoclsparse_operation_none) && (op_b == aoclsparse_operation_none))
        {
            m_c = A->m;
            n_c = B->n;
            k   = n_a;
        }
        else if((op_a != aoclsparse_operation_none) && (op_b == aoclsparse_operation_none))
        {
            m_c = A->n;
            n_c = B->n;
            k   = m_a;
        }
        else if((op_a != aoclsparse_operation_none) && (op_b != aoclsparse_operation_none))
        {
            m_c = A->n;
            n_c = B->m;
            k   = m_a;
        }
        else if((op_a == aoclsparse_operation_none) && (op_b != aoclsparse_operation_none))
        {
            m_c = A->m;
            n_c = B->m;
            k   = n_a;
        }
        m = m_c;
        n = n_c;

        if(layout == aoclsparse_order_row)
        {
            if(ldc == -1)
                ldc = n_c;
            dense_c.resize(m_c * ldc);
            aoclsparse_init<T>(dense_c, m_c, ldc, m_c);
            dense_c_sz = m_c * ldc;
            lda        = n_a;
            ldb        = n_b;
        }
        else
        {
            if(ldc == -1)
                ldc = m_c;
            dense_c.resize(ldc * n_c);
            aoclsparse_init<T>(dense_c, ldc, n_c, ldc);
            dense_c_sz = ldc * n_c;
            lda        = m_a;
            ldb        = m_b;
        }

        dense_c_exp = dense_c;
        if(dense_c.size() == 0)
        {
            dense_c.reserve(1);
            dense_c_exp.reserve(1);
        }

        T alpha, beta;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            switch(scalar)
            {
            case 0:
                alpha = {-1, 2};
                beta  = {2, -1};
                break;
            case 1:
                alpha = {0, 0};
                beta  = {2, -1};
                break;
            case 2:
                alpha = {0, 0};
                beta  = {0, 0};
                break;
            case 3:
                alpha = {0, 0};
                beta  = {1, 0};
                break;
            case 4:
                alpha = {1, 0};
                beta  = {0, 0};
                break;
            }
        }
        else
        {
            switch(scalar)
            {
            case 0:
                alpha = 3.0;
                beta  = -2.0;
                break;
            case 1:
                alpha = 0.;
                beta  = -2.0;
                break;
            case 2:
                alpha = 0.;
                beta  = 0.;
                break;
            case 3:
                alpha = 0.;
                beta  = 1.0;
                break;
            case 4:
                alpha = 1.;
                beta  = 0.0;
                break;
            }
        }

        EXPECT_EQ(aoclsparse_sp2md(op_a,
                                   descrA,
                                   A,
                                   op_b,
                                   descrB,
                                   B,
                                   alpha,
                                   beta,
                                   dense_c.data() + offset,
                                   layout,
                                   ldc,
                                   -1),
                  aoclsparse_status_success);

        aoclsparse_csr2dense(m_a,
                             n_a,
                             descrA,
                             val_a.data(),
                             row_ptr_a.data(),
                             col_ind_a.data(),
                             dense_a.data(),
                             lda,
                             layout);
        aoclsparse_csr2dense(m_b,
                             n_b,
                             descrB,
                             val_b.data(),
                             row_ptr_b.data(),
                             col_ind_b.data(),
                             dense_b.data(),
                             ldb,
                             layout);

        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            blis::gemm(blis_layout,
                       (CBLAS_TRANSPOSE)op_a,
                       (CBLAS_TRANSPOSE)op_b,
                       (int64_t)m,
                       (int64_t)n,
                       (int64_t)k,
                       *reinterpret_cast<const std::complex<float> *>(&alpha),
                       (std::complex<float> const *)dense_a.data(),
                       (int64_t)lda,
                       (std::complex<float> const *)dense_b.data(),
                       (int64_t)ldb,
                       *reinterpret_cast<const std::complex<float> *>(&beta),
                       (std::complex<float> *)dense_c_exp.data() + offset,
                       (int64_t)ldc);
            EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                    ((std::complex<float> *)dense_c.data()),
                                    ((std::complex<float> *)dense_c_exp.data()),
                                    abserr);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            blis::gemm(blis_layout,
                       (CBLAS_TRANSPOSE)op_a,
                       (CBLAS_TRANSPOSE)op_b,
                       (int64_t)m,
                       (int64_t)n,
                       (int64_t)k,
                       *reinterpret_cast<const std::complex<double> *>(&alpha),
                       (std::complex<double> const *)dense_a.data(),
                       (int64_t)lda,
                       (std::complex<double> const *)dense_b.data(),
                       (int64_t)ldb,
                       *reinterpret_cast<const std::complex<double> *>(&beta),
                       (std::complex<double> *)dense_c_exp.data() + offset,
                       (int64_t)ldc);
            EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                    ((std::complex<double> *)dense_c.data()),
                                    ((std::complex<double> *)dense_c_exp.data()),
                                    abserr);
        }

        else
        {
            blis::gemm(blis_layout,
                       (CBLAS_TRANSPOSE)op_a,
                       (CBLAS_TRANSPOSE)op_b,
                       (int64_t)m,
                       (int64_t)n,
                       (int64_t)k,
                       (T)alpha,
                       (T const *)dense_a.data(),
                       (int64_t)lda,
                       (T const *)dense_b.data(),
                       (int64_t)ldb,
                       (T)beta,
                       (T *)dense_c_exp.data() + offset,
                       (int64_t)ldc);
            EXPECT_ARR_NEAR(dense_c_sz, dense_c.data(), dense_c_exp.data(), abserr);
        }

        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
    }

    TEST(sp2md, wrongSize)
    {
        test_failures<float>(0);
        test_failures<double>(0);
        test_failures<aoclsparse_float_complex>(0);
        test_failures<aoclsparse_double_complex>(0);
    }

    TEST(sp2md, wrongType)
    {
        test_failures<float>(1);
        test_failures<double>(1);
    }

    TEST(sp2md, InvalidVal)
    {
        test_failures<float>(3);
        test_failures<double>(3);
        test_failures<aoclsparse_float_complex>(3);
        test_failures<aoclsparse_double_complex>(3);
    }

    TEST(sp2md, NotImpl)
    {
        test_failures<float>(2);
        test_failures<double>(2);
        test_failures<aoclsparse_float_complex>(2);
        test_failures<aoclsparse_double_complex>(2);
    }

    TEST(sp2md, SuccessTypeDouble)
    {
        test_sp2md_success<double>(6, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col);
        test_sp2md_success<double>(6, 4, 8, 4, 10, 8, one, zero, op_n, op_t, col);
        test_sp2md_success<double>(6, 4, 5, 6, 12, 11, zero, zero, op_t, op_t, col);
        test_sp2md_success<double>(6, 4, 5, 6, 12, 11, zero, one, op_t, op_t, row);
        test_sp2md_success<double>(5, 4, 5, 6, 11, 17, one, one, op_t, op_n, row);
        test_sp2md_success<double>(6, 4, 5, 6, 12, 11, zero, one, op_h, op_t, row);
        test_sp2md_success<double>(6, 4, 5, 6, 12, 11, zero, one, op_h, op_h, col);

        // corner cases
        test_sp2md_success<double>(1, 4, 4, 1, 3, 2, zero, zero, op_n, op_n, col);
        test_sp2md_success<double>(1, 1, 1, 4, 1, 2, zero, zero, op_n, op_n, col);
        test_sp2md_success<double>(4, 1, 4, 1, 3, 2, zero, zero, op_t, op_n, col);
        test_sp2md_success<double>(1, 4, 4, 1, 0, 0, zero, zero, op_n, op_n, col);
        test_sp2md_success<double>(1, 4, 4, 1, 3, 0, zero, zero, op_n, op_n, col);
    }

    TEST(sp2md, SuccessTypeFloat)
    {
        test_sp2md_success<float>(6, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col);
        test_sp2md_success<float>(6, 4, 8, 4, 10, 8, one, zero, op_n, op_t, col);
        test_sp2md_success<float>(6, 4, 5, 6, 12, 11, zero, zero, op_t, op_t, col);
        test_sp2md_success<float>(6, 4, 5, 6, 12, 11, zero, one, op_t, op_t, row);
        test_sp2md_success<float>(5, 4, 5, 6, 11, 17, one, one, op_t, op_n, row);
        test_sp2md_success<float>(6, 4, 5, 6, 12, 11, zero, one, op_h, op_t, row);
        test_sp2md_success<float>(6, 4, 5, 6, 12, 11, zero, one, op_h, op_h, col);
    }

    TEST(sp2md, SuccessAlphaBeta)
    {
        test_sp2md_success<float>(
            6, 4, 4, 3, 10, 8, zero, zero, op_n, op_n, col, -1, 0, 1); // alpha = 0
        test_sp2md_success<float>(
            6, 3, 3, 5, 10, 8, zero, zero, op_n, op_n, col, -1, 0, 2); // alpha = 0, beta = 0
        test_sp2md_success<float>(
            4, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col, -1, 0, 3); // alpha = 0, beta = 1
        test_sp2md_success<aoclsparse_double_complex>(
            7, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col, -1, 0, 1); // alpha = 0
        test_sp2md_success<aoclsparse_double_complex>(
            6, 2, 2, 5, 10, 8, zero, zero, op_n, op_n, col, -1, 0, 2); // alpha = 0, beta = 0
        test_sp2md_success<aoclsparse_double_complex>(
            6, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col, -1, 0, 3); // alpha = 0, beta = 1
    }

    TEST(sp2md, SuccessTypeCFloat)
    {
        test_sp2md_success<aoclsparse_float_complex>(
            6, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col);
        test_sp2md_success<aoclsparse_float_complex>(6, 4, 8, 4, 10, 8, one, zero, op_n, op_t, col);
        test_sp2md_success<aoclsparse_float_complex>(
            6, 4, 5, 6, 12, 11, zero, zero, op_t, op_t, col);
        test_sp2md_success<aoclsparse_float_complex>(
            6, 4, 5, 6, 12, 11, zero, one, op_t, op_t, row);
        test_sp2md_success<aoclsparse_float_complex>(5, 4, 5, 6, 11, 17, one, one, op_t, op_n, row);
        test_sp2md_success<aoclsparse_float_complex>(
            6, 4, 5, 6, 12, 11, zero, one, op_h, op_t, row);
        test_sp2md_success<aoclsparse_float_complex>(
            6, 4, 5, 6, 12, 11, zero, one, op_h, op_h, col);
    }

    // This test simulates spmm calls with alpha = 1 and beta = 0, op_n is transpose_none
    TEST(sp2md, SuccessSPMMDSim)
    {
        test_sp2md_success<double>(6, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col, -1, 0, 4);
        test_sp2md_success<double>(5, 4, 5, 6, 11, 17, one, one, op_t, op_n, row, -1, 0, 4);
        test_sp2md_success<double>(5, 4, 5, 6, 11, 17, one, one, op_h, op_n, row, -1, 0, 4);

        test_sp2md_success<aoclsparse_float_complex>(
            6, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col, -1, 0, 4);
        test_sp2md_success<aoclsparse_float_complex>(
            5, 4, 5, 6, 11, 17, one, one, op_t, op_n, row, -1, 0, 4);
        test_sp2md_success<aoclsparse_float_complex>(
            6, 4, 6, 5, 12, 11, zero, one, op_h, op_n, row, -1, 0, 4);
    }

    TEST(sp2md, SuccessTypeCDouble)
    {
        test_sp2md_success<aoclsparse_double_complex>(
            6, 4, 4, 5, 10, 8, zero, zero, op_n, op_n, col);
        test_sp2md_success<aoclsparse_double_complex>(
            6, 4, 8, 4, 10, 8, one, zero, op_n, op_t, col);
        test_sp2md_success<aoclsparse_double_complex>(
            6, 4, 5, 6, 12, 11, zero, zero, op_t, op_t, col);
        test_sp2md_success<aoclsparse_double_complex>(
            6, 4, 5, 6, 12, 11, zero, one, op_t, op_t, row);
        test_sp2md_success<aoclsparse_double_complex>(
            5, 4, 5, 6, 11, 17, one, one, op_t, op_n, row);
        test_sp2md_success<aoclsparse_double_complex>(
            6, 4, 5, 6, 12, 11, zero, one, op_h, op_t, row);
        test_sp2md_success<aoclsparse_double_complex>(
            6, 4, 5, 6, 12, 11, zero, one, op_h, op_h, col);
    }

    TEST(sp2md, Successldc)
    {
        test_sp2md_success<float>(5, 4, 5, 6, 11, 17, one, one, op_t, op_n, col, 10);
        test_sp2md_success<float>(
            5, 4, 5, 6, 11, 17, one, one, op_t, op_n, col, 10, 4); // Changing the C starting window
        test_sp2md_success<aoclsparse_double_complex>(
            11, 7, 11, 8, 31, 17, one, one, op_t, op_n, row, 12);

        test_sp2md_success<aoclsparse_float_complex>(11,
                                                     7,
                                                     11,
                                                     8,
                                                     31,
                                                     17,
                                                     one,
                                                     one,
                                                     op_t,
                                                     op_n,
                                                     row,
                                                     17,
                                                     7); // Changing the C starting window
    }

} // namespace
