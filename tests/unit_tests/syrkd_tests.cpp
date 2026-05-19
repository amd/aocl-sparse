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
    aoclsparse_order      col  = aoclsparse_order_column;
    aoclsparse_order      row  = aoclsparse_order_row;
    aoclsparse_operation  op_t = aoclsparse_operation_transpose;
    aoclsparse_operation  op_h = aoclsparse_operation_conjugate_transpose;
    aoclsparse_operation  op_n = aoclsparse_operation_none;
    aoclsparse_index_base zero = aoclsparse_index_base_zero;
    aoclsparse_index_base one  = aoclsparse_index_base_one;

    // Structure holding the source arrays for matrix A (CSR and optional CSC)
    template <typename T>
    struct syrkd_mats
    {
        // CSR arrays
        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        // CSC arrays (populated by test_syrkd_csc_success only)
        std::vector<aoclsparse_int> csc_col_ptr;
        std::vector<aoclsparse_int> csc_row_ind;
        std::vector<T>              csc_val;
    };
    // Set alpha and beta scalars based on selector index
    template <typename T>
    void syrkd_init_scalars(aoclsparse_int scalar, T &alpha, T &beta)
    {
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
            default:
                alpha = {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};
                beta  = {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};
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
            default:
                alpha = std::numeric_limits<double>::quiet_NaN();
                beta  = std::numeric_limits<double>::quiet_NaN();
                break;
            }
        }
    }
    // Generate a random sparse CSR matrix and populate src arrays
    template <typename T>
    void syrkd_gen_A(aoclsparse_int        m_a,
                     aoclsparse_int        n_a,
                     aoclsparse_int        nnz_a,
                     aoclsparse_index_base b_a,
                     syrkd_mats<T>        &src,
                     aoclsparse_matrix    &A,
                     aoclsparse_mat_descr &descrA)
    {
        std::vector<aoclsparse_int> coo_row;
        ASSERT_EQ(aoclsparse_init_matrix_random(b_a,
                                                m_a,
                                                n_a,
                                                nnz_a,
                                                aoclsparse_csr_mat,
                                                coo_row,
                                                src.col_ind_a,
                                                src.val_a,
                                                src.row_ptr_a,
                                                A),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);
    }

    // Compute dense reference result using BLIS syrk/herk.
    // Converts CSR A to dense, then calls the appropriate BLIS routine.
    // Does NOT contain any EXPECT_* assertions — comparison stays in the test body.
    template <typename T>
    void syrkd_compute_dense_ref(aoclsparse_int        m_a,
                                 aoclsparse_int        n_a,
                                 aoclsparse_index_base b_a,
                                 syrkd_mats<T>        &src,
                                 aoclsparse_operation  op_a,
                                 aoclsparse_order      layout,
                                 aoclsparse_int        m_c,
                                 aoclsparse_int        op_n_a,
                                 T                     alpha,
                                 T                     beta,
                                 std::vector<T>       &dense_c_exp,
                                 aoclsparse_int        offset,
                                 aoclsparse_int        ldc)
    {
        aoclsparse_int lda;
        if(layout == aoclsparse_order_row)
            lda = n_a;
        else
            lda = m_a;

        CBLAS_ORDER blis_layout = (layout == aoclsparse_order_row) ? CblasRowMajor : CblasColMajor;
        const CBLAS_UPLO uplo   = CblasUpper;

        // Convert CSR → dense A
        aoclsparse_mat_descr descrRef;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrRef), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrRef, b_a), aoclsparse_status_success);

        std::vector<T> dense_a(m_a * n_a);
        aoclsparse_csr2dense(m_a,
                             n_a,
                             descrRef,
                             src.val_a.data(),
                             src.row_ptr_a.data(),
                             src.col_ind_a.data(),
                             dense_a.data(),
                             lda,
                             layout);
        aoclsparse_destroy_mat_descr(descrRef);

        // Dispatch to BLIS syrk (real) or herk (complex)
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
        }
    }

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

        aoclsparse_int m_c, op_n_a;

        T alpha, beta;
        syrkd_init_scalars<T>(scalar, alpha, beta);

        aoclsparse_seedrand();
        std::vector<T> dense_c, C, dense_c_exp;
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        syrkd_mats<T>        src;
        aoclsparse_matrix    A;
        aoclsparse_mat_descr descrA;
        syrkd_gen_A<T>(m_a, n_a, nnz_a, b_a, src, A, descrA);
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

        syrkd_compute_dense_ref<T>(
            m_a, n_a, b_a, src, op_a, layout, m_c, op_n_a, alpha, beta, dense_c_exp, offset, ldc);

        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                    ((std::complex<float> *)C.data()),
                                    ((std::complex<float> *)dense_c_exp.data()),
                                    abserr);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                    ((std::complex<double> *)C.data()),
                                    ((std::complex<double> *)dense_c_exp.data()),
                                    abserr);
        }
        else
        {
            EXPECT_ARR_NEAR(dense_c_sz, C.data(), dense_c_exp.data(), abserr);
        }

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descrA);
    }

    // test CSC input success cases
    template <typename T>
    void test_syrkd_csc_success(aoclsparse_int        m_a,
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
        tname << "CSC Success test, type " << typeid(T).name() << ", A " << m_a << "x" << n_a
              << " nnz=" << nnz_a << " " << b_a << "-base, op " << op_a << " ldc= " << ldc;
        SCOPED_TRACE(tname.str());

        aoclsparse_int m_c, op_n_a;

        T alpha, beta;
        syrkd_init_scalars<T>(scalar, alpha, beta);

        aoclsparse_seedrand();
        std::vector<T> dense_c, C, dense_c_exp;
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        // Step 1: Generate random CSR matrix
        syrkd_mats<T>        src;
        aoclsparse_matrix    A_csr;
        aoclsparse_mat_descr descrA;
        syrkd_gen_A<T>(m_a, n_a, nnz_a, b_a, src, A_csr, descrA);

        // Step 2: Convert CSR → CSC arrays
        src.csc_col_ptr.resize(n_a + 1);
        // Ensure at least 1 element so .data() is non-null (create_csc checks pointers even when nnz=0)
        src.csc_row_ind.resize((std::max)(nnz_a, (aoclsparse_int)1));
        src.csc_val.resize((std::max)(nnz_a, (aoclsparse_int)1));

        if(nnz_a > 0)
        {
            ASSERT_EQ(aoclsparse_csr2csc(m_a,
                                         n_a,
                                         nnz_a,
                                         descrA,
                                         b_a,
                                         src.row_ptr_a.data(),
                                         src.col_ind_a.data(),
                                         src.val_a.data(),
                                         src.csc_row_ind.data(),
                                         src.csc_col_ptr.data(),
                                         src.csc_val.data()),
                      aoclsparse_status_success);
        }
        else
        {
            // Empty matrix: csc_col_ptr is all base-index values
            aoclsparse_int base = (b_a == aoclsparse_index_base_one) ? 1 : 0;
            std::fill(src.csc_col_ptr.begin(), src.csc_col_ptr.end(), base);
        }

        // Step 3: Create CSC matrix handle
        aoclsparse_matrix A_csc;
        ASSERT_EQ(aoclsparse_create_csc<T>(&A_csc,
                                           b_a,
                                           m_a,
                                           n_a,
                                           nnz_a,
                                           src.csc_col_ptr.data(),
                                           src.csc_row_ind.data(),
                                           src.csc_val.data()),
                  aoclsparse_status_success);

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

        // Step 4: Call syrkd with CSC matrix
        EXPECT_EQ(aoclsparse_syrkd(op_a, A_csc, alpha, beta, C.data() + offset, layout, ldc),
                  aoclsparse_status_success);

        // Step 5: Compute dense reference using original CSR data
        syrkd_compute_dense_ref<T>(
            m_a, n_a, b_a, src, op_a, layout, m_c, op_n_a, alpha, beta, dense_c_exp, offset, ldc);

        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                    ((std::complex<float> *)C.data()),
                                    ((std::complex<float> *)dense_c_exp.data()),
                                    abserr);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                    ((std::complex<double> *)C.data()),
                                    ((std::complex<double> *)dense_c_exp.data()),
                                    abserr);
        }
        else
        {
            EXPECT_ARR_NEAR(dense_c_sz, C.data(), dense_c_exp.data(), abserr);
        }

        aoclsparse_destroy(&A_csc);
        aoclsparse_destroy(&A_csr);
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

        // CSC empty-matrix coverage
        test_syrkd_csc_success<double>(6, 4, 0, zero, op_t, row, -1, 0, 0);
        test_syrkd_csc_success<double>(1, 47, 0, zero, op_n, row, -1, 0, 0);
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

    // CSC input success tests — validates CSC A produces same result as CSR A
    TEST(syrkd, CSCSuccessTypeDouble)
    {
        // op_none: CSC → effective_op=transpose, no csr2csc needed
        test_syrkd_csc_success<double>(6, 4, 10, zero, op_n, row, -1, 0, 0);
        test_syrkd_csc_success<double>(6, 4, 10, zero, op_n, col, -1, 0, 0);
        test_syrkd_csc_success<double>(3, 6, 10, zero, op_n, row, 10, 2, 0);
        // op_transpose: CSC → effective_op=none, uses csr2csc internally
        test_syrkd_csc_success<double>(7, 2, 10, zero, op_t, col, 13, 4, 0);
        test_syrkd_csc_success<double>(10, 10, 10, zero, op_t, row, 16, 2, 0);
        // op_conj_trans (real = same as transpose)
        test_syrkd_csc_success<double>(12, 12, 50, zero, op_h, col, 20, 4, 0);
        // with alpha/beta scalars
        test_syrkd_csc_success<double>(10, 5, 20, one, op_t, row, 16, 2, 1);
        test_syrkd_csc_success<double>(10, 5, 20, one, op_n, row, 16, 2, 2);
        // 1-based index coverage
        test_syrkd_csc_success<double>(6, 4, 10, one, op_n, row, -1, 0, 0);
        test_syrkd_csc_success<double>(7, 2, 10, one, op_t, col, 13, 4, 0);
        test_syrkd_csc_success<double>(12, 12, 50, one, op_h, col, 20, 4, 0);
    }

    TEST(syrkd, CSCSuccessTypeFloat)
    {
        test_syrkd_csc_success<float>(6, 4, 10, zero, op_n, row, -1, 0, 0);
        test_syrkd_csc_success<float>(11, 4, 21, zero, op_t, col, -1, 0, 3);
        test_syrkd_csc_success<float>(11, 4, 21, zero, op_h, col, 7, 3, 3);
    }

    TEST(syrkd, CSCSuccessTypeCFloat)
    {
        test_syrkd_csc_success<aoclsparse_float_complex>(6, 4, 10, zero, op_n, row, -1, 0, 0);
        test_syrkd_csc_success<aoclsparse_float_complex>(6, 4, 10, zero, op_h, row, -1, 0, 0);
        test_syrkd_csc_success<aoclsparse_float_complex>(4, 6, 11, zero, op_h, row, -1, 0, 4);
    }

    TEST(syrkd, CSCSuccessTypeCDouble)
    {
        // op_none: CSC → effective_op=transpose
        test_syrkd_csc_success<aoclsparse_double_complex>(6, 4, 10, zero, op_n, col, -1, 0, 0);
        test_syrkd_csc_success<aoclsparse_double_complex>(5, 4, 3, zero, op_n, col, -1, 0, 0);
        // op_conj_trans: CSC → effective_op=none, conj in online_atb
        test_syrkd_csc_success<aoclsparse_double_complex>(4, 7, 13, zero, op_h, col, -1, 0, 0);
        test_syrkd_csc_success<aoclsparse_double_complex>(1, 4, 2, zero, op_h, row, -1, 0, 0);
    }

} // namespace
