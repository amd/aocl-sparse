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

    // transpose for conjugate not impelented
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

    // wrong matrix type
    template <typename T>
    void test_not_impl_mat_type()
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
        aoclsparse_operation        op = op_n;
        ASSERT_EQ(aoclsparse_create_csc(
                      &A, zero, m_a, n_a, nnz_a, col_ptr_a.data(), row_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_syrk(op, A, &C), aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A);
    }

    // test success cases
    template <typename T>
    void test_syrk_success(aoclsparse_int        m_a,
                           aoclsparse_int        n_a,
                           aoclsparse_int        nnz_a,
                           aoclsparse_index_base b_a,
                           aoclsparse_operation  op_a,
                           aoclsparse_int        offset = 0)
    {
        std::ostringstream tname;
        tname << "Success test, type " << typeid(T).name() << ", A " << m_a << "x" << n_a
              << " nnz=" << nnz_a << " " << b_a << "-base, op " << op_a;
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

        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        std::vector<aoclsparse_int> coo_row;
        aoclsparse_matrix           A;
        ASSERT_EQ(
            aoclsparse_init_matrix_random(
                b_a, m_a, n_a, nnz_a, aoclsparse_csr_mat, coo_row, col_ind_a, val_a, row_ptr_a, A),
            aoclsparse_status_success);

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

        aoclsparse_csr2dense(m_c,
                             m_c,
                             descrC,
                             val_c,
                             row_ptr_c,
                             col_ind_c,
                             dense_c.data(),
                             n_c,
                             aoclsparse_order_row);

        aoclsparse_csr2dense(m_a,
                             n_a,
                             descrA,
                             val_a.data(),
                             row_ptr_a.data(),
                             col_ind_a.data(),
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
                       (std::complex<float> *)dense_c_exp.data() + offset,
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
                       (std::complex<double> *)dense_c_exp.data() + offset,
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
                       (T *)dense_c_exp.data() + offset,
                       (int64_t)ldc);
            EXPECT_ARR_NEAR(dense_c_sz, dense_c.data(), dense_c_exp.data(), abserr);
        }

        aoclsparse_destroy(&A);
        aoclsparse_destroy(&C);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy_mat_descr(descrC);
    }

    TEST(syrk, NullArg)
    {
        test_null<float>();
        test_null<double>();
        test_null<aoclsparse_float_complex>();
        test_null<aoclsparse_double_complex>();
    }

    TEST(syrk, NotImplMat)
    {
        test_not_impl_mat_type<float>();
        test_not_impl_mat_type<double>();
        test_not_impl_mat_type<aoclsparse_float_complex>();
        test_not_impl_mat_type<aoclsparse_double_complex>();
    }

    TEST(syrk, NotImplOp)
    {
        test_not_impl_ops<aoclsparse_float_complex>();
        test_not_impl_ops<aoclsparse_double_complex>();
    }

    TEST(syrk, UnsortedCol)
    {
        test_unsorted_col_ind<float>();
        test_unsorted_col_ind<aoclsparse_double_complex>();
    }

    TEST(syrk, EmptyC)
    {
        test_syrk_success<double>(5, 4, 0, zero, op_t);
        test_syrk_success<double>(1, 47, 0, one, op_n);
        test_syrk_success<float>(0, 0, 0, zero, op_n);
        test_syrk_success<aoclsparse_float_complex>(1, 34, 0, one, op_h);
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
    }

} // namespace
