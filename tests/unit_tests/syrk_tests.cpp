/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "blis.hh"
#include "cblas.hh"
#pragma GCC diagnostic pop

namespace
{
    aoclsparse_operation  op_t = aoclsparse_operation_transpose;
    aoclsparse_operation  op_h = aoclsparse_operation_conjugate_transpose;
    aoclsparse_operation  op_n = aoclsparse_operation_none;
    aoclsparse_index_base zero = aoclsparse_index_base_zero;
    aoclsparse_index_base one  = aoclsparse_index_base_one;

    // Structure holding the source arrays for matrix A
    template <typename T>
    struct mats
    {
        // CSR arrays for matrix A
        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        // CSC arrays for matrix A (populated when format is csc_mat)
        std::vector<T>              csc_val_a;
        std::vector<aoclsparse_int> csc_row_ind_a;
        std::vector<aoclsparse_int> csc_col_ptr_a;
    };

    // Generate a random matrix A as CSR or CSC format
    // use_csc_a: false (default CSR) or true (CSC format for matrix A)
    template <typename T>
    void gen_A(aoclsparse_int        m_a,
               aoclsparse_int        n_a,
               aoclsparse_int        nnz_a,
               aoclsparse_index_base b_a,
               mats<T>              &src,
               aoclsparse_matrix    &A,
               bool                  use_csc_a = false)
    {
        std::vector<aoclsparse_int> coo_row; // don't need to be preserved, we want only CSR
        // Randomly generate A matrix as CSR first
        aoclsparse_matrix A_csr = NULL;
        ASSERT_EQ(aoclsparse_init_matrix_random(b_a,
                                                m_a,
                                                n_a,
                                                nnz_a,
                                                aoclsparse_csr_mat,
                                                coo_row,
                                                src.col_ind_a,
                                                src.val_a,
                                                src.row_ptr_a,
                                                A_csr),
                  aoclsparse_status_success);

        if(use_csc_a)
        {
            // Convert CSR A to CSC format
            aoclsparse_mat_descr descr_conv;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descr_conv), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descr_conv, b_a), aoclsparse_status_success);

            src.csc_col_ptr_a.resize(n_a + 1);
            src.csc_row_ind_a.resize(nnz_a);
            src.csc_val_a.resize(nnz_a);

            ASSERT_EQ(aoclsparse_csr2csc(m_a,
                                         n_a,
                                         nnz_a,
                                         descr_conv,
                                         b_a,
                                         src.row_ptr_a.data(),
                                         src.col_ind_a.data(),
                                         src.val_a.data(),
                                         src.csc_row_ind_a.data(),
                                         src.csc_col_ptr_a.data(),
                                         src.csc_val_a.data()),
                      aoclsparse_status_success);
            aoclsparse_destroy_mat_descr(descr_conv);

            // Create CSC matrix A
            ASSERT_EQ(aoclsparse_create_csc(&A,
                                            b_a,
                                            m_a,
                                            n_a,
                                            nnz_a,
                                            src.csc_col_ptr_a.data(),
                                            src.csc_row_ind_a.data(),
                                            src.csc_val_a.data()),
                      aoclsparse_status_success);
            // Destroy the temporary CSR matrix
            aoclsparse_destroy(&A_csr);
        }
        else
        {
            // Use CSR format as-is
            A = A_csr;
        }
    }

    // Null test
    template <typename T>
    void test_null()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        aoclsparse_matrix    C;
        aoclsparse_operation op = op_n;
        EXPECT_EQ(aoclsparse_syrk(op, nullptr, &C), aoclsparse_status_invalid_pointer);

        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        std::vector<aoclsparse_int> coo_row;
        aoclsparse_int              m_a = 4, n_a = 2, nnz_a = 7;
        aoclsparse_matrix           A = NULL;
        ASSERT_EQ(
            aoclsparse_init_matrix_random(
                zero, m_a, n_a, nnz_a, aoclsparse_csr_mat, coo_row, col_ind_a, val_a, row_ptr_a, A),
            aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_syrk(op, A, nullptr), aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&A);
    }

    // CSC A with C=nullptr: verifies null-C guard fires regardless of input format
    void test_null_csc_c()
    {
        aoclsparse_int              m_a = 2, n_a = 3, nnz_a = 2;
        std::vector<double>         val_a     = {1.0, 2.0};
        std::vector<aoclsparse_int> row_ind_a = {0, 1};
        std::vector<aoclsparse_int> col_ptr_a = {0, 1, 1, 2};
        aoclsparse_matrix           A;
        ASSERT_EQ(aoclsparse_create_csc(
                      &A, zero, m_a, n_a, nnz_a, col_ptr_a.data(), row_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_syrk(op_n, A, nullptr), aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&A);
    }

    // transpose for conjugate not implemented
    template <typename T>
    void test_not_impl_ops()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        aoclsparse_int m_a   = 4;
        aoclsparse_int n_a   = 2;
        aoclsparse_int nnz_a = 4;
        std::vector<T> val_a;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            val_a.assign({{1, 1}, {1, 2}, {2, 3}, {4, 2}});
        }
        else
        {
            val_a.assign({1, 2, 3, 4});
        }
        std::vector<aoclsparse_int> col_ind_a = {0, 1, 0, 1};
        std::vector<aoclsparse_int> row_ptr_a = {0, 1, 2, 2, 4};
        aoclsparse_matrix           A;
        aoclsparse_matrix           C;
        aoclsparse_operation        op = op_n;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, zero, m_a, n_a, nnz_a, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);

        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            op = op_t;
        }
        else
        {
            op = op_h;
        }
        EXPECT_EQ(aoclsparse_syrk(op, A, &C), aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A);
    }

    // unsorted column index test
    template <typename T>
    void test_unsorted_col_ind()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        aoclsparse_int m_a   = 4;
        aoclsparse_int n_a   = 2;
        aoclsparse_int nnz_a = 4;
        std::vector<T> val_a;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            val_a.assign({{1, 1}, {1, 2}, {2, 3}, {4, 2}});
        }
        else
        {
            val_a.assign({1, 2, 3, 4});
        }
        std::vector<aoclsparse_int> col_ind_a = {0, 1, 1, 0};
        std::vector<aoclsparse_int> row_ptr_a = {0, 1, 2, 2, 4};
        aoclsparse_matrix           A;
        aoclsparse_matrix           C;
        aoclsparse_operation        op;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            op = op_h;
        }
        else
        {
            op = op_t;
        }
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, zero, m_a, n_a, nnz_a, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_syrk(op, A, &C), aoclsparse_status_unsorted_input);
        aoclsparse_destroy(&A);
    }

    // CSC matrix with unsorted row indices within a column + op_none should return unsorted_input
    template <typename T>
    void test_csc_unsorted_col_ind()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        // 4x2 CSC: col 0 has rows {0,3}, col 1 has rows {1,0} — col 1 is unsorted
        aoclsparse_int m_a = 4, n_a = 2, nnz_a = 4;
        std::vector<T> val_a;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
            val_a.assign({{1, 1}, {1, 2}, {2, 3}, {4, 2}});
        else
            val_a.assign({1, 2, 3, 4});

        std::vector<aoclsparse_int> row_ind_a = {0, 3, 1, 0}; // col 1: {1,0} unsorted
        std::vector<aoclsparse_int> col_ptr_a = {0, 2, 4};
        aoclsparse_matrix           A, C;
        ASSERT_EQ(aoclsparse_create_csc(
                      &A, zero, m_a, n_a, nnz_a, col_ptr_a.data(), row_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_syrk(op_n, A, &C), aoclsparse_status_unsorted_input);
        aoclsparse_destroy(&A);
    }

    // Complex CSC + op_t: not implemented (non-Hermitian result)
    template <typename T>
    void test_csc_complex_op_t_not_impl()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        aoclsparse_int m_a   = 4;
        aoclsparse_int n_a   = 2;
        aoclsparse_int nnz_a = 4;
        std::vector<T> val_a;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            val_a.assign({{1, 1}, {1, 2}, {2, 3}, {4, 2}});
        }
        else
        {
            val_a.assign({1, 2, 3, 4});
        }
        std::vector<aoclsparse_int> row_ind_a = {0, 3, 1, 3};
        std::vector<aoclsparse_int> col_ptr_a = {0, 2, 4};
        aoclsparse_matrix           A;
        aoclsparse_matrix           C;
        aoclsparse_operation        op = op_t; // CSC + op_transpose should fail
        ASSERT_EQ(aoclsparse_create_csc(
                      &A, zero, m_a, n_a, nnz_a, col_ptr_a.data(), row_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_syrk(op, A, &C), aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A);
    }

    // op value outside the valid enum range
    template <typename T>
    void test_invalid_op()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        aoclsparse_int              m_a = 3, n_a = 2, nnz_a = 3;
        std::vector<T>              val_a     = {1, 2, 3};
        std::vector<aoclsparse_int> col_ind_a = {0, 1, 0};
        std::vector<aoclsparse_int> row_ptr_a = {0, 1, 2, 3};
        aoclsparse_matrix           A, C;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, zero, m_a, n_a, nnz_a, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_syrk(static_cast<aoclsparse_operation>(99), A, &C),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);
    }

    // test success cases; use_csc_a=true to test CSC input path
    template <typename T>
    void test_syrk_success(aoclsparse_int        m_a,
                           aoclsparse_int        n_a,
                           aoclsparse_int        nnz_a,
                           aoclsparse_index_base b_a,
                           aoclsparse_operation  op_a,
                           bool                  use_csc_a = false)
    {
        std::ostringstream tname;
        tname << "Success test, type " << typeid(T).name() << ", A " << m_a << "x" << n_a
              << " nnz=" << nnz_a << " " << b_a << "-base, op " << op_a
              << (use_csc_a ? " (CSC)" : " (CSR)");
        SCOPED_TRACE(tname.str());

        aoclsparse_int m_c, n_c, op_n_a, lda, ldc;
        CBLAS_ORDER    blis_layout;
        blis_layout           = CblasRowMajor;
        const CBLAS_UPLO uplo = CblasUpper;
        T                alpha, beta;

        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
        }
        else
        {
            alpha = 1.;
            beta  = 0.;
        }
        aoclsparse_seedrand();
        std::vector<T> dense_a(m_a * n_a), dense_c, dense_c_exp;
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        mats<T>           src;
        aoclsparse_matrix A = NULL;
        gen_A(m_a, n_a, nnz_a, b_a, src, A, use_csc_a);

        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

        lda = n_a;
        if(op_a == op_n)
        {
            m_c    = m_a;
            op_n_a = n_a;
        }
        else
        {
            m_c    = n_a;
            op_n_a = m_a;
        }
        ldc                       = m_c;
        n_c                       = m_c;
        aoclsparse_int dense_c_sz = m_c * m_c;
        dense_c.resize(dense_c_sz);

        // need to initialize the dense matrix to zero for later validation
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            for(aoclsparse_int i = 0; i < dense_c_sz; i++)
                dense_c[i] = {0, 0};
        }
        else
        {
            for(aoclsparse_int i = 0; i < dense_c_sz; i++)
                dense_c[i] = 0;
        }

        dense_c_exp = dense_c;
        if(dense_c.size() == 0)
        {
            dense_c.reserve(1);
            dense_c_exp.reserve(1);
        }

        aoclsparse_matrix     C;
        aoclsparse_int        nnz_c;
        aoclsparse_int       *row_ptr_c = NULL;
        aoclsparse_int       *col_ind_c = NULL;
        T                    *val_c     = NULL;
        aoclsparse_index_base base_c    = b_a;
        EXPECT_EQ(aoclsparse_syrk(op_a, A, &C), aoclsparse_status_success);
        aoclsparse_mat_descr descrC;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrC), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descrC, aoclsparse_fill_mode_upper),
                  aoclsparse_status_success);
        // Export resultant C matrix and Convert to Dense
        ASSERT_EQ(
            aoclsparse_export_csr(C, &base_c, &m_c, &n_c, &nnz_c, &row_ptr_c, &col_ind_c, &val_c),
            aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrC, base_c), aoclsparse_status_success);

        // Verify output is upper-triangular: every stored entry must be on or above the diagonal
        for(aoclsparse_int i = 0; i < m_c; i++)
            for(aoclsparse_int idx = row_ptr_c[i] - base_c; idx < row_ptr_c[i + 1] - base_c; ++idx)
                EXPECT_GE(col_ind_c[idx], i + base_c) << "sub-diagonal entry at row " << i;

        aoclsparse_csr2dense(m_c,
                             m_c,
                             descrC,
                             val_c,
                             row_ptr_c,
                             col_ind_c,
                             dense_c.data(),
                             n_c,
                             aoclsparse_order_row);

        // For CSC input, we use CSR arrays for dense conversion since
        // aoclsparse_csr2dense works with CSR format
        aoclsparse_csr2dense(m_a,
                             n_a,
                             descrA,
                             src.val_a.data(),
                             src.row_ptr_a.data(),
                             src.col_ind_a.data(),
                             dense_a.data(),
                             n_a,
                             aoclsparse_order_row);

        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            blis::herk(blis_layout,
                       (CBLAS_UPLO)uplo,
                       (CBLAS_TRANSPOSE)op_a,
                       (int64_t)m_c,
                       (int64_t)op_n_a,
                       alpha.real,
                       (std::complex<float> const *)dense_a.data(),
                       (int64_t)lda,
                       beta.real,
                       (std::complex<float> *)dense_c_exp.data(),
                       (int64_t)ldc);
            EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                    ((std::complex<float> *)dense_c.data()),
                                    ((std::complex<float> *)dense_c_exp.data()),
                                    abserr);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            blis::herk(blis_layout,
                       (CBLAS_UPLO)uplo,
                       (CBLAS_TRANSPOSE)op_a,
                       (int64_t)m_c,
                       (int64_t)op_n_a,
                       alpha.real,
                       (std::complex<double> const *)dense_a.data(),
                       (int64_t)lda,
                       beta.real,
                       (std::complex<double> *)dense_c_exp.data(),
                       (int64_t)ldc);
            EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                    ((std::complex<double> *)dense_c.data()),
                                    ((std::complex<double> *)dense_c_exp.data()),
                                    abserr);
        }
        else
        {
            blis::syrk(blis_layout,
                       (CBLAS_UPLO)uplo,
                       (CBLAS_TRANSPOSE)op_a,
                       (int64_t)m_c,
                       (int64_t)op_n_a,
                       (T)alpha,
                       (T const *)dense_a.data(),
                       (int64_t)lda,
                       (T)beta,
                       (T *)dense_c_exp.data(),
                       (int64_t)ldc);
            EXPECT_ARR_NEAR(dense_c_sz, dense_c.data(), dense_c_exp.data(), abserr);
        }

        aoclsparse_destroy(&A);
        aoclsparse_destroy(&C);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy_mat_descr(descrC);
    }

    // CSC A with nnz=0: verify quick-return produces correct empty C dimensions
    void test_syrk_csc_empty()
    {
        aoclsparse_int              m_a = 4, n_a = 3;
        std::vector<aoclsparse_int> col_ptr(n_a + 1, 0);
        std::vector<aoclsparse_int> dummy_row(1, 0);
        std::vector<double>         dummy_val(1, 0.0);
        aoclsparse_matrix           A = nullptr, C = nullptr;
        ASSERT_EQ(aoclsparse_create_csc(
                      &A, zero, m_a, n_a, 0, col_ptr.data(), dummy_row.data(), dummy_val.data()),
                  aoclsparse_status_success);
        // op_none: C is m_a x m_a
        EXPECT_EQ(aoclsparse_syrk(op_n, A, &C), aoclsparse_status_success);
        ASSERT_NE(C, nullptr);
        EXPECT_EQ(C->nnz, 0);
        EXPECT_EQ(C->m, m_a);
        aoclsparse_destroy(&C);
        // op_h: C is n_a x n_a
        EXPECT_EQ(aoclsparse_syrk(op_h, A, &C), aoclsparse_status_success);
        ASSERT_NE(C, nullptr);
        EXPECT_EQ(C->nnz, 0);
        EXPECT_EQ(C->m, n_a);
        aoclsparse_destroy(&C);
        aoclsparse_destroy(&A);
    }

    TEST(syrk, NullArg)
    {
        test_null<float>();
        test_null<double>();
        test_null_csc_c();
    }

    TEST(syrk, NotImplMat)
    {
        test_csc_complex_op_t_not_impl<aoclsparse_double_complex>();
    }

    TEST(syrk, NotImplOp)
    {
        test_not_impl_ops<aoclsparse_float_complex>();
    }

    TEST(syrk, InvalidOp)
    {
        test_invalid_op<double>();
    }

    TEST(syrk, UnsortedCol)
    {
        test_unsorted_col_ind<float>();
    }

    TEST(syrk, UnsortedColCSC)
    {
        test_csc_unsorted_col_ind<double>();
    }

    TEST(syrk, EmptyC)
    {
        test_syrk_success<double>(5, 4, 0, zero, op_t);
        test_syrk_success<double>(1, 47, 0, one, op_n);
        test_syrk_success<float>(0, 0, 0, zero, op_n);
        test_syrk_success<aoclsparse_float_complex>(1, 34, 0, one, op_h);
        test_syrk_success<aoclsparse_double_complex>(1, 5, 0, zero, op_h);
    }

    TEST(syrk, EmptyCSC)
    {
        test_syrk_csc_empty();
    }

    TEST(syrk, SuccessTypeDouble)
    {
        test_syrk_success<double>(6, 4, 10, zero, op_n);
        test_syrk_success<double>(10, 5, 20, zero, op_n);
        test_syrk_success<double>(10, 10, 20, zero, op_n);
        test_syrk_success<double>(4, 15, 22, zero, op_n);
        test_syrk_success<double>(4, 15, 22, zero, op_n);
        test_syrk_success<double>(10, 13, 50, one, op_n);
        test_syrk_success<double>(2, 13, 10, one, op_n);
        test_syrk_success<double>(8, 6, 10, one, op_n);
        test_syrk_success<double>(4, 5, 12, zero, op_t);
        test_syrk_success<double>(10, 5, 20, zero, op_t);
        test_syrk_success<double>(10, 10, 20, zero, op_t);
        test_syrk_success<double>(10, 10, 20, one, op_t);
        test_syrk_success<double>(4, 5, 12, one, op_t);
        test_syrk_success<double>(10, 10, 20, one, op_h);
    }

    TEST(syrk, SuccessTypeFloat)
    {
        test_syrk_success<float>(6, 4, 10, zero, op_n);
        test_syrk_success<float>(1, 4, 3, zero, op_n);
        test_syrk_success<float>(11, 1, 11, zero, op_n);
        test_syrk_success<float>(1, 1, 1, zero, op_n);
        test_syrk_success<float>(6, 4, 10, zero, op_h);
    }

    TEST(syrk, SuccessTypeCFloat)
    {
        test_syrk_success<aoclsparse_float_complex>(6, 4, 10, zero, op_n);
        test_syrk_success<aoclsparse_float_complex>(6, 6, 17, zero, op_n);
        test_syrk_success<aoclsparse_float_complex>(6, 6, 17, one, op_n);
        test_syrk_success<aoclsparse_float_complex>(1, 1, 1, zero, op_n);
        test_syrk_success<aoclsparse_float_complex>(1, 1, 1, one, op_n);
        // m < n: hits aat_dense_row path
        test_syrk_success<aoclsparse_float_complex>(2, 8, 4, zero, op_n);
        test_syrk_success<aoclsparse_float_complex>(3, 12, 9, one, op_n);

        test_syrk_success<aoclsparse_float_complex>(6, 4, 10, zero, op_h);
        test_syrk_success<aoclsparse_float_complex>(6, 4, 10, one, op_h);
        test_syrk_success<aoclsparse_float_complex>(4, 6, 11, zero, op_h);
        test_syrk_success<aoclsparse_float_complex>(4, 6, 11, one, op_h);
        test_syrk_success<aoclsparse_float_complex>(2, 2, 3, zero, op_h);
        test_syrk_success<aoclsparse_float_complex>(2, 2, 3, one, op_h);
    }

    TEST(syrk, SuccessTypeCDouble)
    {
        test_syrk_success<aoclsparse_double_complex>(1, 4, 2, zero, op_n);
        test_syrk_success<aoclsparse_double_complex>(6, 4, 10, zero, op_n);
        test_syrk_success<aoclsparse_double_complex>(5, 4, 3, zero, op_n);
        // m < n: hits aat_dense_row path
        test_syrk_success<aoclsparse_double_complex>(2, 8, 4, zero, op_n);
        test_syrk_success<aoclsparse_double_complex>(3, 12, 9, one, op_n);

        test_syrk_success<aoclsparse_double_complex>(6, 4, 10, zero, op_h);
        test_syrk_success<aoclsparse_double_complex>(6, 4, 10, one, op_h);
        test_syrk_success<aoclsparse_double_complex>(4, 6, 11, zero, op_h);
        test_syrk_success<aoclsparse_double_complex>(4, 6, 11, one, op_h);
        test_syrk_success<aoclsparse_double_complex>(2, 2, 3, zero, op_h);
        test_syrk_success<aoclsparse_double_complex>(2, 2, 3, one, op_h);
    }

    TEST(syrk, SuccessCSCOpTrans)
    {
        test_syrk_success<double>(6, 4, 10, zero, op_t, true);
        test_syrk_success<double>(10, 5, 20, zero, op_t, true);
        test_syrk_success<double>(10, 10, 20, zero, op_t, true);
        test_syrk_success<double>(4, 15, 22, one, op_t, true);
        test_syrk_success<double>(8, 6, 10, one, op_t, true);
        test_syrk_success<float>(6, 4, 10, zero, op_t, true);
        test_syrk_success<float>(11, 1, 11, one, op_t, true);
    }

    TEST(syrk, SuccessCSCOpConjTrans)
    {
        test_syrk_success<double>(6, 4, 10, zero, op_h, true);
        test_syrk_success<double>(4, 15, 22, one, op_h, true);
        test_syrk_success<double>(8, 6, 10, one, op_h, true);
        test_syrk_success<float>(6, 4, 10, zero, op_h, true);
        test_syrk_success<float>(11, 1, 11, one, op_h, true);
        test_syrk_success<aoclsparse_float_complex>(6, 4, 10, zero, op_h, true);
        test_syrk_success<aoclsparse_float_complex>(4, 6, 11, one, op_h, true);
        test_syrk_success<aoclsparse_float_complex>(5, 8, 15, zero, op_h, true);
        test_syrk_success<aoclsparse_double_complex>(6, 4, 10, zero, op_h, true);
        test_syrk_success<aoclsparse_double_complex>(5, 8, 15, one, op_h, true);
        test_syrk_success<aoclsparse_double_complex>(10, 10, 20, zero, op_h, true);
    }

    // Tests for CSC input on matrix A with op_none
    TEST(syrk, SuccessCSCDouble)
    {
        test_syrk_success<double>(6, 4, 10, zero, op_n, true);
        test_syrk_success<double>(10, 5, 20, zero, op_n, true);
        test_syrk_success<double>(10, 10, 20, zero, op_n, true);
        test_syrk_success<double>(4, 15, 22, one, op_n, true);
        test_syrk_success<double>(8, 6, 10, one, op_n, true);
    }

    TEST(syrk, SuccessCSCFloat)
    {
        test_syrk_success<float>(6, 4, 10, zero, op_n, true);
        test_syrk_success<float>(1, 4, 3, zero, op_n, true);
        test_syrk_success<float>(11, 1, 11, one, op_n, true);
    }

    TEST(syrk, SuccessCSCCFloat)
    {
        test_syrk_success<aoclsparse_float_complex>(6, 4, 10, zero, op_n, true);
        test_syrk_success<aoclsparse_float_complex>(5, 8, 15, one, op_n, true);
        test_syrk_success<aoclsparse_float_complex>(6, 6, 17, zero, op_n, true);
    }

    TEST(syrk, SuccessCSCCDouble)
    {
        test_syrk_success<aoclsparse_double_complex>(6, 4, 10, zero, op_n, true);
        test_syrk_success<aoclsparse_double_complex>(10, 10, 20, zero, op_n, true);
        test_syrk_success<aoclsparse_double_complex>(8, 5, 12, one, op_n, true);
    }

} // namespace
