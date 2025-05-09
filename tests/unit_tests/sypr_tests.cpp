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
#include <type_traits>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "blis.hh"
#pragma GCC diagnostic pop

namespace
{
    // structure holding the source arrays for 2 matrices
    template <typename T>
    struct mats
    {
        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        std::vector<T>              val_b;
        std::vector<aoclsparse_int> col_ind_b;
        std::vector<aoclsparse_int> row_ptr_b;
    };

    // generate two random matrices of given dimensions
    // and descriptor for B to be symmetric/hermitian (if complex)
    // and return all the source arrays as std::vectors in 'src'
    template <typename T>
    void gen_AB(aoclsparse_int        m_a,
                aoclsparse_int        n_a,
                aoclsparse_int        m_b,
                aoclsparse_int        n_b,
                aoclsparse_int        nnz_a,
                aoclsparse_int        nnz_b,
                aoclsparse_index_base b_a,
                aoclsparse_index_base b_b,
                mats<T>              &src,
                aoclsparse_matrix    &A,
                aoclsparse_matrix    &B,
                aoclsparse_mat_descr &descrB)
    {
        std::vector<aoclsparse_int> coo_row; // don't need to be preserved, we want only CSR
        // Randomly generate A matrix
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

        // Randomly generate B matrix
        ASSERT_EQ(aoclsparse_init_matrix_random(b_b,
                                                m_b,
                                                n_b,
                                                nnz_b,
                                                aoclsparse_csr_mat,
                                                coo_row,
                                                src.col_ind_b,
                                                src.val_b,
                                                src.row_ptr_b,
                                                B),
                  aoclsparse_status_success);
        // Remove imaginary part from diagonal element in B matrix to make it hermitian
        if constexpr(!(std::is_same_v<T, double> || std::is_same_v<T, float>))
        {
            for(aoclsparse_int i = 0; i < m_b; i++)
                for(aoclsparse_int j = src.row_ptr_b[i] - b_b; j < src.row_ptr_b[i + 1] - b_b; j++)
                {
                    if(src.col_ind_b[j] - b_b == i)
                        src.val_b[j].imag = 0.0;
                }
        }

        // descriptor B matching the base nad symmetric/hermitian
        // the fill triangle doesn't matter because the matrix is random
        // so both have something in
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            ASSERT_EQ(aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_symmetric),
                      aoclsparse_status_success);
        else
            ASSERT_EQ(aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_hermitian),
                      aoclsparse_status_success);
    }

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_sypr_nullptr(aoclsparse_int        m_a,
                           aoclsparse_int        n_a,
                           aoclsparse_int        m_b,
                           aoclsparse_int        n_b,
                           aoclsparse_int        nnz_a,
                           aoclsparse_int        nnz_b,
                           aoclsparse_index_base b_a,
                           aoclsparse_index_base b_b,
                           aoclsparse_operation  op_a)
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name() << ", A " << m_a << "x" << n_a << " B " << m_b
              << "x" << n_b;
        SCOPED_TRACE(tname.str());

        aoclsparse_request request = aoclsparse_stage_full_computation;
        aoclsparse_seedrand();

        mats<T>              src;
        aoclsparse_matrix    A      = NULL;
        aoclsparse_matrix    B      = NULL;
        aoclsparse_mat_descr descrB = NULL;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B, descrB);

        aoclsparse_matrix C = NULL;
        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_sypr(op_a, nullptr, B, descrB, &C, request),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sypr(op_a, A, nullptr, descrB, &C, request),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, nullptr, &C, request),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, nullptr, request),
                  aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }
    // Quick return with success when size 0 matrix is passed
    template <typename T>
    void test_sypr_do_nothing(aoclsparse_int        m_a,
                              aoclsparse_int        n_a,
                              aoclsparse_int        m_b,
                              aoclsparse_int        n_b,
                              aoclsparse_int        nnz_a,
                              aoclsparse_int        nnz_b,
                              aoclsparse_index_base b_a,
                              aoclsparse_index_base b_b,
                              aoclsparse_operation  op_a)
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name() << ", A " << m_a << "x" << n_a << " B " << m_b
              << "x" << n_b;
        SCOPED_TRACE(tname.str());

        aoclsparse_request request = aoclsparse_stage_full_computation;
        aoclsparse_seedrand();

        mats<T>              src;
        aoclsparse_matrix    A      = NULL;
        aoclsparse_matrix    B      = NULL;
        aoclsparse_mat_descr descrB = NULL;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B, descrB);

        aoclsparse_matrix C = NULL;

        A->m = 0;
        A->n = 0;
        B->m = 0;
        B->n = 0;
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request), aoclsparse_status_success);
        ASSERT_NE(C, nullptr);
        EXPECT_EQ(C->m, 0);
        EXPECT_EQ(C->nnz, 0);
        aoclsparse_destroy(&C);

        // Check for non-null C matrix pointer after empty matrix inputs
        A->m   = m_a;
        A->n   = n_a;
        B->m   = m_b;
        B->n   = n_b;
        A->nnz = 0;
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request), aoclsparse_status_success);
        ASSERT_NE(C, nullptr);
        EXPECT_EQ(C->m, op_a == aoclsparse_operation_none ? m_a : n_a);
        EXPECT_EQ(C->nnz, 0);
        aoclsparse_destroy(&C);

        A->nnz = nnz_a;
        B->nnz = 0;
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request), aoclsparse_status_success);
        ASSERT_NE(C, nullptr);
        EXPECT_EQ(C->m, op_a == aoclsparse_operation_none ? m_a : n_a);
        EXPECT_EQ(C->nnz, 0);
        aoclsparse_destroy(&C);

        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrB);
    }

    // tests for Wrong size
    template <typename T>
    void test_sypr_wrong_size(aoclsparse_int        m_a,
                              aoclsparse_int        n_a,
                              aoclsparse_int        m_b,
                              aoclsparse_int        n_b,
                              aoclsparse_int        nnz_a,
                              aoclsparse_int        nnz_b,
                              aoclsparse_index_base b_a,
                              aoclsparse_index_base b_b,
                              aoclsparse_operation  op_a)
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name() << ", A " << m_a << "x" << n_a << " B " << m_b
              << "x" << n_b;
        SCOPED_TRACE(tname.str());

        aoclsparse_request request = aoclsparse_stage_full_computation;
        aoclsparse_seedrand();

        mats<T>              src;
        aoclsparse_matrix    A      = NULL;
        aoclsparse_matrix    B      = NULL;
        aoclsparse_mat_descr descrB = NULL;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B, descrB);

        aoclsparse_matrix C = NULL;
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request), aoclsparse_status_invalid_size);
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&C);
    }
    // tests for Invalid base value and wrong matrix B type
    template <typename T>
    void test_sypr_invalid_value(aoclsparse_int        m_a,
                                 aoclsparse_int        n_a,
                                 aoclsparse_int        m_b,
                                 aoclsparse_int        n_b,
                                 aoclsparse_int        nnz_a,
                                 aoclsparse_int        nnz_b,
                                 aoclsparse_index_base b_a,
                                 aoclsparse_index_base b_b,
                                 aoclsparse_operation  op_a,
                                 aoclsparse_int        stage)
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name() << ", A " << m_a << "x" << n_a << " B " << m_b
              << "x" << n_b;
        SCOPED_TRACE(tname.str());

        aoclsparse_request request = aoclsparse_stage_full_computation;
        aoclsparse_seedrand();

        mats<T>              src;
        aoclsparse_matrix    A      = NULL;
        aoclsparse_matrix    B      = NULL;
        aoclsparse_mat_descr descrB = NULL;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B, descrB);

        aoclsparse_matrix C = NULL;

        // invalid request
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, (aoclsparse_request)1234),
                  aoclsparse_status_invalid_value);

        // invalid op
        EXPECT_EQ(aoclsparse_sypr((aoclsparse_operation)678, A, B, descrB, &C, request),
                  aoclsparse_status_invalid_value);

        // unsupported diag_type
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descrB, aoclsparse_diag_type_unit),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, aoclsparse_stage_full_computation),
                  aoclsparse_status_not_implemented);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descrB, aoclsparse_diag_type_non_unit),
                  aoclsparse_status_success);

        // Invalid base for A matrix
        A->base = (aoclsparse_index_base)3;
        if(stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
        }
        else if(stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
        }
        aoclsparse_destroy(&C);

        // Invalid base for B matrix
        A->base      = b_a;
        descrB->base = (aoclsparse_index_base)3;
        if(stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
        }
        else if(stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
        }

        // Invalid matrix B type(general)
        ASSERT_EQ(aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_general),
                  aoclsparse_status_success);
        descrB->base = b_b;
        if(stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
        }
        else if(stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_invalid_value);
        }
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&C);
    }
    // tests for settings not implemented
    template <typename T>
    void test_sypr_not_implemented(aoclsparse_int        m_a,
                                   aoclsparse_int        n_a,
                                   aoclsparse_int        m_b,
                                   aoclsparse_int        n_b,
                                   aoclsparse_int        nnz_a,
                                   aoclsparse_int        nnz_b,
                                   aoclsparse_index_base b_a,
                                   aoclsparse_index_base b_b,
                                   aoclsparse_operation  op_a,
                                   aoclsparse_int        stage)
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name() << ", A " << m_a << "x" << n_a << " B " << m_b
              << "x" << n_b;
        SCOPED_TRACE(tname.str());

        aoclsparse_request request = aoclsparse_stage_full_computation;
        aoclsparse_seedrand();

        //Randomly generate A matrix
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

        //Randomly generate B matrix
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
        ASSERT_EQ(aoclsparse_create_csc(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            ASSERT_EQ(aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_symmetric),
                      aoclsparse_status_success);
        else
            ASSERT_EQ(aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_hermitian),
                      aoclsparse_status_success);

        // and expect not_implemented for CSC for matrix B
        aoclsparse_matrix C = NULL;
        if(stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_not_implemented);
        }
        else if(stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_not_implemented);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                      aoclsparse_status_not_implemented);
        }
        aoclsparse_destroy(&C);

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }

    void test_sypr_wrong_datatype()
    {
        aoclsparse_operation  op_a = aoclsparse_operation_none;
        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_request    request;
        aoclsparse_int        m = 2, k = 3, n = 3, nnzA = 1, nnzB = 3;
        float                 csr_valA[]     = {42.};
        aoclsparse_int        csr_col_indA[] = {1};
        aoclsparse_int        csr_row_ptrA[] = {0, 0, 1};
        double                csr_valB[]     = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[] = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[] = {0, 1, 2, 3};
        aoclsparse_mat_descr  descrB;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descrB);

        aoclsparse_matrix A;
        aoclsparse_create_scsr(&A, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix B;
        aoclsparse_create_dcsr(&B, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix C = NULL;
        request             = aoclsparse_stage_full_computation;
        // For float A and double B matrices, invoke sypr
        // and expect wrong type error
        EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request), aoclsparse_status_wrong_type);

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }

    // Dense Reference for sypr
    template <typename T>
    void ref_sypr(aoclsparse_int       m,
                  aoclsparse_int       n,
                  const T             *dense_a,
                  const T             *dense_b,
                  T                   *dense_c_exp,
                  aoclsparse_operation op_a,
                  aoclsparse_fill_mode fill_b)
    {
        std::vector<T> dense_t(m * n);
        T              alpha, beta;
        // Invoke Dense GEMM for Dense A and B matrices
        // Compare Dense GEMM output and sypr output(converted to Dense)
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};

            if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                if((op_a == aoclsparse_operation_transpose)
                   || (op_a == aoclsparse_operation_conjugate_transpose))
                {
                    // Ah * B * A
                    // B * A = T
                    blis::hemm(CblasRowMajor,
                               CblasLeft,
                               (fill_b == 1) ? CblasUpper : CblasLower,
                               (int64_t)m,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<float> *>(&alpha),
                               (std::complex<float> const *)dense_b,
                               (int64_t)m,
                               (std::complex<float> const *)dense_a,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<float> *>(&beta),
                               (std::complex<float> *)dense_t.data(),
                               (int64_t)n);

                    // Ah * T
                    blis::gemm(CblasRowMajor,
                               CblasConjTrans,
                               CblasNoTrans,
                               (int64_t)n,
                               (int64_t)n,
                               (int64_t)m,
                               *reinterpret_cast<const std::complex<float> *>(&alpha),
                               (std::complex<float> const *)dense_a,
                               (int64_t)n,
                               (std::complex<float> const *)dense_t.data(),
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<float> *>(&beta),
                               (std::complex<float> *)dense_c_exp,
                               (int64_t)n);
                }
                else
                {
                    // A * B * Ah
                    blis::hemm(CblasRowMajor,
                               CblasRight,
                               (fill_b == 1) ? CblasUpper : CblasLower,
                               (int64_t)m,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<float> *>(&alpha),
                               (std::complex<float> const *)dense_b,
                               (int64_t)n,
                               (std::complex<float> const *)dense_a,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<float> *>(&beta),
                               (std::complex<float> *)dense_t.data(),
                               (int64_t)n);

                    blis::gemm(CblasRowMajor,
                               CblasNoTrans,
                               CblasConjTrans,
                               (int64_t)m,
                               (int64_t)m,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<float> *>(&alpha),
                               (std::complex<float> const *)dense_t.data(),
                               (int64_t)n,
                               (std::complex<float> const *)dense_a,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<float> *>(&beta),
                               (std::complex<float> *)dense_c_exp,
                               (int64_t)m);
                }
            }
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                if((op_a == aoclsparse_operation_transpose)
                   || (op_a == aoclsparse_operation_conjugate_transpose))
                {
                    // Ah * B * A
                    // B * A = T
                    blis::hemm(CblasRowMajor,
                               CblasLeft,
                               (fill_b == 1) ? CblasUpper : CblasLower,
                               (int64_t)m,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<double> *>(&alpha),
                               (std::complex<double> const *)dense_b,
                               (int64_t)m,
                               (std::complex<double> const *)dense_a,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<double> *>(&beta),
                               (std::complex<double> *)dense_t.data(),
                               (int64_t)n);

                    // Ah * T
                    blis::gemm(CblasRowMajor,
                               CblasConjTrans,
                               CblasNoTrans,
                               (int64_t)n,
                               (int64_t)n,
                               (int64_t)m,
                               *reinterpret_cast<const std::complex<double> *>(&alpha),
                               (std::complex<double> const *)dense_a,
                               (int64_t)n,
                               (std::complex<double> const *)dense_t.data(),
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<double> *>(&beta),
                               (std::complex<double> *)dense_c_exp,
                               (int64_t)n);
                }
                else
                {
                    // A * B * Ah
                    blis::hemm(CblasRowMajor,
                               CblasRight,
                               (fill_b == 1) ? CblasUpper : CblasLower,
                               (int64_t)m,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<double> *>(&alpha),
                               (std::complex<double> const *)dense_b,
                               (int64_t)n,
                               (std::complex<double> const *)dense_a,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<double> *>(&beta),
                               (std::complex<double> *)dense_t.data(),
                               (int64_t)n);

                    blis::gemm(CblasRowMajor,
                               CblasNoTrans,
                               CblasConjTrans,
                               (int64_t)m,
                               (int64_t)m,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<double> *>(&alpha),
                               (std::complex<double> const *)dense_t.data(),
                               (int64_t)n,
                               (std::complex<double> const *)dense_a,
                               (int64_t)n,
                               *reinterpret_cast<const std::complex<double> *>(&beta),
                               (std::complex<double> *)dense_c_exp,
                               (int64_t)m);
                }
            }
        }
        else
        {
            if((op_a == aoclsparse_operation_transpose)
               || (op_a == aoclsparse_operation_conjugate_transpose))
            {
                // At * B * A
                alpha = 1;
                beta  = 0;
                // B * A = T
                blis::symm(CblasRowMajor,
                           CblasLeft,
                           (fill_b == 1) ? CblasUpper : CblasLower,
                           (int64_t)m,
                           (int64_t)n,
                           (T)alpha,
                           (T const *)dense_b,
                           (int64_t)m,
                           (T const *)dense_a,
                           (int64_t)n,
                           (T)beta,
                           (T *)dense_t.data(),
                           (int64_t)n);

                // At * T
                blis::gemm(CblasRowMajor,
                           CblasTrans,
                           CblasNoTrans,
                           (int64_t)n,
                           (int64_t)n,
                           (int64_t)m,
                           (T)alpha,
                           (T const *)dense_a,
                           (int64_t)n,
                           (T const *)dense_t.data(),
                           (int64_t)n,
                           (T)beta,
                           (T *)dense_c_exp,
                           (int64_t)n);
            }
            else if(op_a == aoclsparse_operation_none)
            {
                // A * B * At
                alpha = 1;
                beta  = 0;
                // A * B = T
                blis::symm(CblasRowMajor,
                           CblasRight,
                           (fill_b == 1) ? CblasUpper : CblasLower,
                           (int64_t)m,
                           (int64_t)n,
                           (T)alpha,
                           (T const *)dense_b,
                           (int64_t)n,
                           (T const *)dense_a,
                           (int64_t)n,
                           (T)beta,
                           (T *)dense_t.data(),
                           (int64_t)n);
                // T * At
                blis::gemm(CblasRowMajor,
                           CblasNoTrans,
                           CblasTrans,
                           (int64_t)m,
                           (int64_t)m,
                           (int64_t)n,
                           (T)alpha,
                           (T const *)dense_t.data(),
                           (int64_t)n,
                           (T const *)dense_a,
                           (int64_t)n,
                           (T)beta,
                           (T *)dense_c_exp,
                           (int64_t)m);
            }
        }
    }
    // Test for success and verify results against Dense GEMM results.
    template <typename T>
    void test_sypr_success(std::string           test_message,
                           aoclsparse_int        m_a,
                           aoclsparse_int        n_a,
                           aoclsparse_int        m_b,
                           aoclsparse_int        n_b,
                           aoclsparse_int        nnz_a,
                           aoclsparse_int        nnz_b,
                           aoclsparse_index_base b_a,
                           aoclsparse_index_base b_b,
                           aoclsparse_operation  op_a,
                           aoclsparse_fill_mode  fill_b,
                           aoclsparse_int        stage)
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name() << ", A " << m_a << "x" << n_a << " B " << m_b
              << "x" << n_b;
        SCOPED_TRACE(tname.str());

        aoclsparse_int        m_c, n_c, nnz_c;
        aoclsparse_int       *row_ptr_c = NULL;
        aoclsparse_int       *col_ind_c = NULL;
        T                    *val_c     = NULL;
        aoclsparse_index_base base_c;
        aoclsparse_seedrand();

        std::vector<T> dense_a(m_a * n_a), dense_b(m_b * n_b), dense_c, dense_c_exp;
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        mats<T>              src;
        aoclsparse_matrix    A      = NULL;
        aoclsparse_matrix    B      = NULL;
        aoclsparse_mat_descr descrB = NULL;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B, descrB);

        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descrB, fill_b), aoclsparse_status_success);

        {
            SCOPED_TRACE(test_message);
            // Expect success from sypr
            aoclsparse_matrix C = NULL;
            if(stage == 0)
            {
                aoclsparse_request request = aoclsparse_stage_nnz_count;
                EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                          aoclsparse_status_success);
                request = aoclsparse_stage_finalize;
                EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                          aoclsparse_status_success);
            }
            else if(stage == 1)
            {
                aoclsparse_request request = aoclsparse_stage_full_computation;
                EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request),
                          aoclsparse_status_success);
            }
            // Export resultant C matrix and Convert to Dense
            ASSERT_EQ(aoclsparse_export_csr(
                          C, &base_c, &m_c, &n_c, &nnz_c, &row_ptr_c, &col_ind_c, &val_c),
                      aoclsparse_status_success);

            // check expected dimensions
            if(op_a == aoclsparse_operation_none)
            {
                EXPECT_EQ(m_c, m_a);
                EXPECT_EQ(n_c, m_a);
            }
            else
            {
                EXPECT_EQ(m_c, n_a);
                EXPECT_EQ(n_c, n_a);
            }

            aoclsparse_mat_descr descrC;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descrC), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descrC, base_c), aoclsparse_status_success);
            dense_c.resize(m_c * n_c);
            dense_c_exp.resize(m_c * n_c);
            aoclsparse_csr2dense(m_c,
                                 n_c,
                                 descrC,
                                 val_c,
                                 row_ptr_c,
                                 col_ind_c,
                                 dense_c.data(),
                                 n_c,
                                 aoclsparse_order_row);

            // Convert input matrices A and B into Dense for invoking Dense GEMM
            aoclsparse_csr2dense(m_a,
                                 n_a,
                                 descrA,
                                 src.val_a.data(),
                                 src.row_ptr_a.data(),
                                 src.col_ind_a.data(),
                                 dense_a.data(),
                                 n_a,
                                 aoclsparse_order_row);
            aoclsparse_csr2dense(m_b,
                                 n_b,
                                 descrB,
                                 src.val_b.data(),
                                 src.row_ptr_b.data(),
                                 src.col_ind_b.data(),
                                 dense_b.data(),
                                 n_b,
                                 aoclsparse_order_row);
            // Invoke Dense GEMM for Dense A and B matrices
            ref_sypr(m_a, n_a, dense_a.data(), dense_b.data(), dense_c_exp.data(), op_a, fill_b);

            // Compare Dense GEMM output and sypr output(converted to Dense)
            if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                EXPECT_COMPLEX_TRIMAT_NEAR(m_c,
                                           n_c,
                                           n_c,
                                           ((std::complex<float> *)dense_c.data()),
                                           ((std::complex<float> *)dense_c_exp.data()),
                                           abserr);
            }
            else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                EXPECT_COMPLEX_TRIMAT_NEAR(m_c,
                                           n_c,
                                           n_c,
                                           ((std::complex<double> *)dense_c.data()),
                                           ((std::complex<double> *)dense_c_exp.data()),
                                           abserr);
            }
            else // double or float
            {
                EXPECT_TRIMAT_NEAR(m_c, n_c, n_c, dense_c.data(), dense_c_exp.data(), abserr);
            }
            aoclsparse_destroy(&C);
            aoclsparse_destroy_mat_descr(descrC);
            aoclsparse_destroy_mat_descr(descrB);
            aoclsparse_destroy(&B);
            aoclsparse_destroy_mat_descr(descrA);
            aoclsparse_destroy(&A);
        }
    }

    // Test for success and verify results against Dense GEMM results.
    template <typename T>
    void test_sypr_finalize(std::string           test_message,
                            aoclsparse_int        m_a,
                            aoclsparse_int        n_a,
                            aoclsparse_int        m_b,
                            aoclsparse_int        n_b,
                            aoclsparse_int        nnz_a,
                            aoclsparse_int        nnz_b,
                            aoclsparse_index_base b_a,
                            aoclsparse_index_base b_b,
                            aoclsparse_operation  op_a,
                            aoclsparse_fill_mode  fill_b)
    {
        std::ostringstream tname;
        tname << "Test type " << typeid(T).name() << ", A " << m_a << "x" << n_a << " B " << m_b
              << "x" << n_b;
        SCOPED_TRACE(tname.str());

        aoclsparse_int        m_c, n_c, nnz_c;
        aoclsparse_int       *row_ptr_c = NULL;
        aoclsparse_int       *col_ind_c = NULL;
        T                    *val_c     = NULL;
        aoclsparse_index_base base_c;
        aoclsparse_seedrand();

        std::vector<T> dense_a(m_a * n_a), dense_b(m_b * n_b), dense_c, dense_c_exp;
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        mats<T>              src;
        aoclsparse_matrix    A      = NULL;
        aoclsparse_matrix    B      = NULL;
        aoclsparse_mat_descr descrB = NULL;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B, descrB);

        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descrB, fill_b), aoclsparse_status_success);

        {
            SCOPED_TRACE(test_message);
            // Expect success from sypr
            aoclsparse_matrix  C       = NULL;
            aoclsparse_request request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request), aoclsparse_status_success);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request), aoclsparse_status_success);

            // Modify the values of A matix value arrays.
            for(aoclsparse_int i = 0; i < A->nnz; i++)
                ((T *)A->csr_mat.csr_val)[i] = random_generator_normal<T>();

            // Invoke sypr with finalize stage alone.
            // Expect success as C matrix created in previous invocation
            // is reused to update the value array alone.
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sypr(op_a, A, B, descrB, &C, request), aoclsparse_status_success);

            // Export resultant C matrix and Convert to Dense
            ASSERT_EQ(aoclsparse_export_csr(
                          C, &base_c, &m_c, &n_c, &nnz_c, &row_ptr_c, &col_ind_c, &val_c),
                      aoclsparse_status_success);

            aoclsparse_mat_descr descrC;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descrC), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descrC, base_c), aoclsparse_status_success);
            dense_c.resize(m_c * n_c);
            dense_c_exp.resize(m_c * n_c);
            aoclsparse_csr2dense(m_c,
                                 n_c,
                                 descrC,
                                 val_c,
                                 row_ptr_c,
                                 col_ind_c,
                                 dense_c.data(),
                                 n_c,
                                 aoclsparse_order_row);

            // Convert input matrices A and B into Dense for invoking Dense GEMM
            aoclsparse_csr2dense(m_a,
                                 n_a,
                                 descrA,
                                 src.val_a.data(),
                                 src.row_ptr_a.data(),
                                 src.col_ind_a.data(),
                                 dense_a.data(),
                                 n_a,
                                 aoclsparse_order_row);
            aoclsparse_csr2dense(m_b,
                                 n_b,
                                 descrB,
                                 src.val_b.data(),
                                 src.row_ptr_b.data(),
                                 src.col_ind_b.data(),
                                 dense_b.data(),
                                 n_b,
                                 aoclsparse_order_row);
            // Invoke Dense GEMM for Dense A and B matrices
            ref_sypr(m_a, n_a, dense_a.data(), dense_b.data(), dense_c_exp.data(), op_a, fill_b);

            // Compare Dense GEMM output and sypr output(converted to Dense)
            if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                EXPECT_COMPLEX_TRIMAT_NEAR(m_c,
                                           n_c,
                                           n_c,
                                           ((std::complex<float> *)dense_c.data()),
                                           ((std::complex<float> *)dense_c_exp.data()),
                                           abserr);
            }
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                EXPECT_COMPLEX_TRIMAT_NEAR(m_c,
                                           n_c,
                                           n_c,
                                           ((std::complex<double> *)dense_c.data()),
                                           ((std::complex<double> *)dense_c_exp.data()),
                                           abserr);
            }
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                EXPECT_TRIMAT_NEAR(m_c, n_c, n_c, dense_c.data(), dense_c_exp.data(), abserr);
            }
            aoclsparse_destroy(&C);
            aoclsparse_destroy_mat_descr(descrC);
            aoclsparse_destroy_mat_descr(descrB);
            aoclsparse_destroy(&B);
            aoclsparse_destroy_mat_descr(descrA);
            aoclsparse_destroy(&A);
        }
    }

    TEST(sypr, NullArgAll)
    {
        test_sypr_nullptr<double>(3,
                                  2,
                                  4,
                                  2,
                                  2,
                                  5,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_operation_transpose);
        test_sypr_nullptr<float>(3,
                                 2,
                                 4,
                                 3,
                                 2,
                                 5,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_operation_transpose);
        test_sypr_nullptr<aoclsparse_double_complex>(3,
                                                     2,
                                                     4,
                                                     2,
                                                     2,
                                                     5,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_operation_transpose);
        test_sypr_nullptr<aoclsparse_float_complex>(3,
                                                    2,
                                                    4,
                                                    2,
                                                    2,
                                                    5,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_operation_transpose);
    }
    TEST(sypr, DoNothingAll)
    {
        test_sypr_do_nothing<double>(4,
                                     4,
                                     4,
                                     4,
                                     7,
                                     9,
                                     aoclsparse_index_base_zero,
                                     aoclsparse_index_base_zero,
                                     aoclsparse_operation_none);
        test_sypr_do_nothing<float>(4,
                                    4,
                                    4,
                                    4,
                                    7,
                                    9,
                                    aoclsparse_index_base_zero,
                                    aoclsparse_index_base_zero,
                                    aoclsparse_operation_transpose);
        test_sypr_do_nothing<aoclsparse_double_complex>(4,
                                                        4,
                                                        4,
                                                        4,
                                                        7,
                                                        9,
                                                        aoclsparse_index_base_zero,
                                                        aoclsparse_index_base_zero,
                                                        aoclsparse_operation_none);
        test_sypr_do_nothing<aoclsparse_float_complex>(4,
                                                       4,
                                                       4,
                                                       4,
                                                       7,
                                                       9,
                                                       aoclsparse_index_base_zero,
                                                       aoclsparse_index_base_zero,
                                                       aoclsparse_operation_transpose);
    }
    TEST(sypr, WrongSizeAll)
    {
        test_sypr_wrong_size<double>(3,
                                     2,
                                     4,
                                     5,
                                     2,
                                     5,
                                     aoclsparse_index_base_zero,
                                     aoclsparse_index_base_zero,
                                     aoclsparse_operation_none);
        test_sypr_wrong_size<float>(4,
                                    5,
                                    5,
                                    5,
                                    7,
                                    9,
                                    aoclsparse_index_base_one,
                                    aoclsparse_index_base_zero,
                                    aoclsparse_operation_transpose);
        test_sypr_wrong_size<aoclsparse_double_complex>(3,
                                                        4,
                                                        4,
                                                        4,
                                                        2,
                                                        5,
                                                        aoclsparse_index_base_zero,
                                                        aoclsparse_index_base_one,
                                                        aoclsparse_operation_conjugate_transpose);
        test_sypr_wrong_size<aoclsparse_float_complex>(5,
                                                       4,
                                                       5,
                                                       5,
                                                       7,
                                                       9,
                                                       aoclsparse_index_base_zero,
                                                       aoclsparse_index_base_zero,
                                                       aoclsparse_operation_none);
    }

    TEST(sypr, InvalidValueAll)
    {
        test_sypr_invalid_value<double>(4,
                                        5,
                                        4,
                                        4,
                                        7,
                                        9,
                                        aoclsparse_index_base_one,
                                        aoclsparse_index_base_zero,
                                        aoclsparse_operation_transpose,
                                        0);
        test_sypr_invalid_value<float>(5,
                                       3,
                                       3,
                                       3,
                                       7,
                                       9,
                                       aoclsparse_index_base_zero,
                                       aoclsparse_index_base_zero,
                                       aoclsparse_operation_none,
                                       1);
        test_sypr_invalid_value<aoclsparse_double_complex>(5,
                                                           4,
                                                           5,
                                                           5,
                                                           7,
                                                           9,
                                                           aoclsparse_index_base_zero,
                                                           aoclsparse_index_base_zero,
                                                           aoclsparse_operation_conjugate_transpose,
                                                           0);
        test_sypr_invalid_value<aoclsparse_float_complex>(3,
                                                          4,
                                                          4,
                                                          4,
                                                          7,
                                                          9,
                                                          aoclsparse_index_base_one,
                                                          aoclsparse_index_base_one,
                                                          aoclsparse_operation_none,
                                                          1);
    }
    TEST(sypr, NotImplAll)
    {
        test_sypr_not_implemented<double>(8,
                                          6,
                                          8,
                                          8,
                                          7,
                                          9,
                                          aoclsparse_index_base_zero,
                                          aoclsparse_index_base_zero,
                                          aoclsparse_operation_transpose,
                                          0);
        test_sypr_not_implemented<float>(3,
                                         5,
                                         5,
                                         5,
                                         7,
                                         9,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_operation_none,
                                         1);
        test_sypr_not_implemented<aoclsparse_double_complex>(
            7,
            4,
            7,
            7,
            7,
            9,
            aoclsparse_index_base_zero,
            aoclsparse_index_base_zero,
            aoclsparse_operation_conjugate_transpose,
            0);
        test_sypr_not_implemented<aoclsparse_float_complex>(5,
                                                            6,
                                                            6,
                                                            6,
                                                            7,
                                                            9,
                                                            aoclsparse_index_base_zero,
                                                            aoclsparse_index_base_zero,
                                                            aoclsparse_operation_none,
                                                            1);
    }
    TEST(sypr, WrongType)
    {
        test_sypr_wrong_datatype();
    }
    TEST(sypr, SuccessTypeDouble)
    {
        test_sypr_success<double>("ba = 1, bb = 1, opA = T, fillB = L, 1 stage",
                                  24,
                                  24,
                                  24,
                                  24,
                                  100,
                                  80,
                                  aoclsparse_index_base_one,
                                  aoclsparse_index_base_one,
                                  aoclsparse_operation_transpose,
                                  aoclsparse_fill_mode_lower,
                                  0);
        test_sypr_success<double>("ba = 0, bb = 1, opA = N, fillB = U, 2 stage",
                                  25,
                                  36,
                                  36,
                                  36,
                                  100,
                                  150,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_one,
                                  aoclsparse_operation_none,
                                  aoclsparse_fill_mode_upper,
                                  1);
        test_sypr_success<double>("ba = 0, bb = 0, opA = H, fillB = U, 2 stage",
                                  15,
                                  36,
                                  15,
                                  15,
                                  100,
                                  13,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_operation_conjugate_transpose,
                                  aoclsparse_fill_mode_upper,
                                  1);
    }
    TEST(sypr, SuccessTypeFloat)
    {
        test_sypr_success<float>("ba = 0, bb = 0, opA = T, fillB = U, 2 stage",
                                 14,
                                 23,
                                 14,
                                 14,
                                 110,
                                 171,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_operation_transpose,
                                 aoclsparse_fill_mode_upper,
                                 1);
        test_sypr_success<float>("ba = 1, bb = 0, opA = N, fillB = L, 1 stage",
                                 24,
                                 15,
                                 15,
                                 15,
                                 111,
                                 87,
                                 aoclsparse_index_base_one,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_operation_none,
                                 aoclsparse_fill_mode_lower,
                                 0);
    }
    TEST(sypr, SuccessTypeCDouble)
    {
        test_sypr_success<aoclsparse_double_complex>("ba = 0, bb = 1, opA = T, fillB = L, 1 stage",
                                                     17,
                                                     15,
                                                     17,
                                                     17,
                                                     100,
                                                     80,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_index_base_one,
                                                     aoclsparse_operation_transpose,
                                                     aoclsparse_fill_mode_lower,
                                                     0);
        test_sypr_success<aoclsparse_double_complex>("ba = 0, bb = 0, opA = N, fillB = U, 2 stage",
                                                     20,
                                                     16,
                                                     16,
                                                     16,
                                                     80,
                                                     95,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_operation_none,
                                                     aoclsparse_fill_mode_upper,
                                                     1);
    }
    TEST(sypr, SuccessTypeCFloat)
    {
        test_sypr_success<aoclsparse_float_complex>("ba = 1, bb = 1, opA = N, fillB = U, 2 stage",
                                                    18,
                                                    19,
                                                    19,
                                                    19,
                                                    132,
                                                    122,
                                                    aoclsparse_index_base_one,
                                                    aoclsparse_index_base_one,
                                                    aoclsparse_operation_none,
                                                    aoclsparse_fill_mode_upper,
                                                    1);
        test_sypr_success<aoclsparse_float_complex>("ba = 1, bb = 0, opA = H, fillB = L, 1 stage",
                                                    16,
                                                    12,
                                                    16,
                                                    16,
                                                    56,
                                                    43,
                                                    aoclsparse_index_base_one,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_operation_conjugate_transpose,
                                                    aoclsparse_fill_mode_lower,
                                                    0);
    }
    TEST(sypr, FinalizeAll)
    {
        test_sypr_finalize<float>("float ba = 0, bb = 1, opA = H, fillB = L, 1 stage",
                                  9,
                                  10,
                                  9,
                                  9,
                                  55,
                                  63,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_one,
                                  aoclsparse_operation_conjugate_transpose,
                                  aoclsparse_fill_mode_lower);
        test_sypr_finalize<aoclsparse_float_complex>(
            "cfloat, ba = 1, bb = 0, opA = N, fillB = U, 1 stage",
            15,
            17,
            17,
            17,
            105,
            71,
            aoclsparse_index_base_one,
            aoclsparse_index_base_zero,
            aoclsparse_operation_none,
            aoclsparse_fill_mode_upper);
        test_sypr_finalize<aoclsparse_double_complex>(
            "cdouble, empty C, ba = 1, bb = 1, opA = N, fillB = U, 1 stage",
            3,
            11,
            11,
            11,
            20,
            0,
            aoclsparse_index_base_one,
            aoclsparse_index_base_one,
            aoclsparse_operation_none,
            aoclsparse_fill_mode_upper);
    }
} // namespace
