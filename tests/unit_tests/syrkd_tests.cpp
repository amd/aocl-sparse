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
    aoclsparse_order      col  = aoclsparse_order_column;
    aoclsparse_order      row  = aoclsparse_order_row;
    aoclsparse_operation  op_t = aoclsparse_operation_transpose;
    aoclsparse_operation  op_h = aoclsparse_operation_conjugate_transpose;
    aoclsparse_operation  op_n = aoclsparse_operation_none;
    aoclsparse_index_base zero = aoclsparse_index_base_zero;
    aoclsparse_index_base one  = aoclsparse_index_base_one;

    // tests null args
    template <typename T>
    void test_null()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        T                   *C  = (T *)malloc(sizeof(T) * 1);
        aoclsparse_operation op = op_n;
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

        ASSERT_EQ(aoclsparse_create_csr(
                      &A, zero, m_a, n_a, nnz_a, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_syrkd(op, nullptr, alpha, beta, C, row, 2),
                  aoclsparse_status_invalid_pointer);
        free(C);
        C = nullptr;
        EXPECT_EQ(aoclsparse_syrkd(op, A, alpha, beta, C, row, 2),
                  aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&A);
    }

    // wrong type test
    template <typename T>
    void test_wrong_type()
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
        T                          *C = (T *)malloc(sizeof(T) * 1);
        T                           alpha, beta;
        aoclsparse_operation        op = op_n;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, zero, m_a, n_a, nnz_a, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);

        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
            op    = op_t;
        }
        else
        {
            alpha = 3.0;
            beta  = -2.0;
            op    = op_h;
        }
        EXPECT_EQ(aoclsparse_syrkd(op, A, alpha, beta, C, row, 10),
                  aoclsparse_status_not_implemented);
        op = op_n;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            A->val_type = aoclsparse_dmat;
        }
        else
        {
            A->val_type = aoclsparse_cmat;
        }
        EXPECT_EQ(aoclsparse_syrkd(op, A, alpha, beta, C, row, 10), aoclsparse_status_wrong_type);
        free(C);
        aoclsparse_destroy(&A);
    }

    // tests not implemented configs
    template <typename T>
    void test_not_impl()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        aoclsparse_int m_a   = 4;
        aoclsparse_int n_a   = 2;
        aoclsparse_int nnz_a = 4;
        std::vector<T> val_a;
        T              alpha, beta;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
            val_a.assign({{1, 1}, {1, 2}, {2, 3}, {4, 2}});
        }
        else
        {
            val_a.assign({1, 2, 3, 4});
            alpha = 3.0;
            beta  = -2.0;
        }
        std::vector<aoclsparse_int> row_ind_a = {0, 3, 1, 3};
        std::vector<aoclsparse_int> col_ptr_a = {0, 2, 4};
        aoclsparse_matrix           A;
        T                          *C  = (T *)malloc(sizeof(T) * 1);
        aoclsparse_operation        op = op_n;
        ASSERT_EQ(aoclsparse_create_csc(
                      &A, zero, m_a, n_a, nnz_a, col_ptr_a.data(), row_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_syrkd(op, A, alpha, beta, C, row, 10),
                  aoclsparse_status_not_implemented);
        free(C);
        aoclsparse_destroy(&A);
    }

    template <typename T>
    void test_not_impl_ops()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());

        T              alpha = aoclsparse_numeric::zero<T>();
        T              beta  = aoclsparse_numeric::zero<T>();
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
        T                          *C  = (T *)malloc(sizeof(T) * 1);
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
        EXPECT_EQ(aoclsparse_syrkd(op, A, alpha, beta, C, row, 10),
                  aoclsparse_status_not_implemented);
        free(C);
        aoclsparse_destroy(&A);
    }

    // unsorted column index test
    template <typename T>
    void test_unsorted_col_ind()
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name();
        SCOPED_TRACE(tname.str());
        T              alpha = aoclsparse_numeric::zero<T>();
        T              beta  = aoclsparse_numeric::zero<T>();
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
        T                          *C = (T *)malloc(sizeof(T) * 1);
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

        EXPECT_EQ(aoclsparse_syrkd(op, A, alpha, beta, C, row, 10),
                  aoclsparse_status_unsorted_input);
        free(C);
        aoclsparse_destroy(&A);
    }

    // test success cases
    template <typename T>
    void test_syrkd_success(aoclsparse_int        m_a,
                            aoclsparse_int        n_a,
                            aoclsparse_int        nnz_a,
                            aoclsparse_index_base b_a,
                            aoclsparse_operation  op_a,
                            aoclsparse_order      layout,
                            aoclsparse_int        ldc    = -1,
                            aoclsparse_int        offset = 0,
                            aoclsparse_int        scalar = 0)
    {

        std::ostringstream tname;
        tname << "Success test, type " << typeid(T).name() << ", A " << m_a << "x" << n_a
              << " nnz=" << nnz_a << " " << b_a << "-base, op " << op_a << " ldc= " << ldc;
        SCOPED_TRACE(tname.str());

        aoclsparse_int m_c, op_n_a, lda;
        CBLAS_ORDER    blis_layout;
        if(layout == aoclsparse_order_row)
        {
            blis_layout = CblasRowMajor;
        }
        else
        {
            blis_layout = CblasColMajor;
        }
        const CBLAS_UPLO uplo = CblasUpper;

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

        aoclsparse_seedrand();
        std::vector<T> dense_a(m_a * n_a), dense_c, C, dense_c_exp;
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

        if(layout == aoclsparse_order_row)
        {
            lda = n_a;
        }
        else
        {
            lda = m_a;
        }
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
        if(ldc == -1)
            ldc = m_c;
        aoclsparse_int dense_c_sz = ldc * m_c;
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
        C           = dense_c;
        if(dense_c.size() == 0)
        {
            dense_c.reserve(1);
            dense_c_exp.reserve(1);
        }

        EXPECT_EQ(aoclsparse_syrkd(op_a, A, alpha, beta, C.data() + offset, layout, ldc),
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
                                    ((std::complex<float> *)C.data()),
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
                                    ((std::complex<double> *)C.data()),
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
            EXPECT_ARR_NEAR(dense_c_sz, C.data(), dense_c_exp.data(), abserr);
        }

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descrA);
    }

    TEST(syrkd, NullArg)
    {
        test_null<float>();
        test_null<double>();
        test_null<aoclsparse_float_complex>();
        test_null<aoclsparse_double_complex>();
    }
    TEST(syrkd, WrongType)
    {
        test_wrong_type<aoclsparse_float_complex>();
        test_wrong_type<aoclsparse_double_complex>();
    }

    TEST(syrkd, NotImpl)
    {
        test_not_impl<float>();
        test_not_impl<double>();
        test_not_impl<aoclsparse_float_complex>();
        test_not_impl<aoclsparse_double_complex>();
    }

    TEST(syrkd, NotImplOp)
    {
        test_not_impl_ops<aoclsparse_float_complex>();
        test_not_impl_ops<aoclsparse_double_complex>();
    }

    TEST(syrkd, UnsortedCol)
    {
        test_unsorted_col_ind<float>();
        test_unsorted_col_ind<aoclsparse_double_complex>();
    }

    TEST(syrkd, EmptyC)
    {
        test_syrkd_success<double>(6, 4, 0, zero, op_t, row, -1, 0, 0);
        test_syrkd_success<double>(1, 47, 0, zero, op_n, row, -1, 0, 0);
        // This test is failing with: lapack2flame: On entry to SSYRK , parameter number  7 had an illegal value
        //test_syrkd_success<float>(0, 0, 0, zero, op_n, row, -1, 0, 0);
        test_syrkd_success<aoclsparse_float_complex>(1, 34, 0, one, op_h, row, -1, 0, 0);
    }

    TEST(syrkd, SuccessTypeDouble)
    {
        // parameters: m_a, n_a, nnz_a, base, op_a, layout, ldc, offset (starting position of C),
        //             scalar (for selecting alpha and beta)

        test_syrkd_success<double>(6, 4, 10, zero, op_n, row, -1, 0, 0);
        test_syrkd_success<double>(6, 4, 10, zero, op_n, col, -1, 0, 0);
        test_syrkd_success<double>(3, 6, 10, zero, op_n, row, 10, 2, 0);
        test_syrkd_success<double>(7, 2, 10, one, op_n, col, 13, 4, 0);
        test_syrkd_success<double>(10, 10, 10, zero, op_n, row, 16, 2, 0);
        test_syrkd_success<double>(12, 12, 50, one, op_n, col, 20, 4, 0);
        test_syrkd_success<double>(7, 2, 10, zero, op_t, col, 13, 4, 0);
        test_syrkd_success<double>(10, 10, 10, zero, op_t, row, 16, 2, 0);
        test_syrkd_success<double>(12, 12, 50, zero, op_t, col, 20, 4, 0);
        test_syrkd_success<double>(10, 5, 20, one, op_t, row, 16, 2, 1);
        test_syrkd_success<double>(10, 5, 20, one, op_t, row, 16, 2, 2);
    }

    TEST(syrkd, SuccessTypeFloat)
    {
        test_syrkd_success<float>(6, 4, 10, zero, op_n, row, -1, 0, 0);
        test_syrkd_success<float>(1, 4, 3, zero, op_n, row, -1, 0, 0);
        test_syrkd_success<float>(11, 1, 11, zero, op_n, col, -1, 0, 3);
        test_syrkd_success<float>(11, 4, 21, zero, op_t, col, -1, 0, 3);
        test_syrkd_success<float>(11, 4, 21, zero, op_h, col, 7, 3, 3);
        test_syrkd_success<float>(1, 1, 1, zero, op_n, row, -1, 0, 4);
        test_syrkd_success<float>(10, 10, 10, one, op_t, row, -1, 0, 0);
    }

    TEST(syrkd, SuccessTypeCFloat)
    {
        test_syrkd_success<aoclsparse_float_complex>(6, 4, 10, zero, op_n, row, -1, 0, 0);
        test_syrkd_success<aoclsparse_float_complex>(6, 6, 17, zero, op_n, row, -1, 0, 0);
        test_syrkd_success<aoclsparse_float_complex>(6, 6, 17, one, op_n, row, -1, 0, 0);
        test_syrkd_success<aoclsparse_float_complex>(1, 1, 1, zero, op_n, row, -1, 0, 0);
        test_syrkd_success<aoclsparse_float_complex>(1, 1, 1, one, op_n, row, -1, 0, 0);

        test_syrkd_success<aoclsparse_float_complex>(6, 4, 10, zero, op_h, row, -1, 0, 0);
        test_syrkd_success<aoclsparse_float_complex>(6, 4, 10, one, op_h, row, -1, 0, 1);
        test_syrkd_success<aoclsparse_float_complex>(4, 6, 11, zero, op_h, row, -1, 0, 4);
        test_syrkd_success<aoclsparse_float_complex>(4, 6, 11, one, op_h, row, -1, 0, 0);
        test_syrkd_success<aoclsparse_float_complex>(2, 2, 3, zero, op_h, row, -1, 0, 0);
        test_syrkd_success<aoclsparse_float_complex>(2, 2, 3, one, op_h, row, -1, 0, 0);
    }

    TEST(syrkd, SuccessTypeCDouble)
    {
        test_syrkd_success<aoclsparse_double_complex>(1, 4, 2, zero, op_n, col, -1, 0, 0);
        test_syrkd_success<aoclsparse_double_complex>(6, 4, 10, zero, op_n, col, -1, 0, 2);
        test_syrkd_success<aoclsparse_double_complex>(5, 4, 3, zero, op_n, col, -1, 0, 0);
        test_syrkd_success<aoclsparse_double_complex>(4, 7, 13, zero, op_h, col, -1, 0, 0);
    }

} // namespace
