/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "blis.hh"
#include "cblas.hh"
#pragma GCC diagnostic pop

namespace
{
    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_sp2m_nullptr(aoclsparse_int        m_a,
                           aoclsparse_int        n_a,
                           aoclsparse_int        m_b,
                           aoclsparse_int        n_b,
                           aoclsparse_int        nnz_a,
                           aoclsparse_int        nnz_b,
                           aoclsparse_index_base b_a,
                           aoclsparse_index_base b_b,
                           aoclsparse_operation  op_a,
                           aoclsparse_operation  op_b)
    {
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
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

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
        ASSERT_EQ(aoclsparse_create_csr(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);

        aoclsparse_matrix C = NULL;
        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_sp2m(op_a, nullptr, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, nullptr, B, request, &C),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, nullptr, op_b, descrB, B, request, &C),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, nullptr, request, &C),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, nullptr),
                  aoclsparse_status_invalid_pointer);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }

    // Quick return with success when size 0 matrix is passed
    template <typename T>
    void test_sp2m_do_nothing(aoclsparse_int        m_a,
                              aoclsparse_int        n_a,
                              aoclsparse_int        m_b,
                              aoclsparse_int        n_b,
                              aoclsparse_int        nnz_a,
                              aoclsparse_int        nnz_b,
                              aoclsparse_index_base b_a,
                              aoclsparse_index_base b_b,
                              aoclsparse_operation  op_a,
                              aoclsparse_operation  op_b)
    {
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
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

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
        ASSERT_EQ(aoclsparse_create_csr(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);

        aoclsparse_matrix C = NULL;

        A->m = 0;
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_success);
        aoclsparse_destroy(&C);

        A->m = m_a;
        B->n = 0;
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_success);
        aoclsparse_destroy(&C);

        // Check for non-null C matrix pointer after empty matrix inputs
        B->n   = n_b;
        A->nnz = 0;
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_success);
        EXPECT_NE(C, nullptr);
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&C);
    }

    // tests for Wrong size
    template <typename T>
    void test_sp2m_wrong_size(aoclsparse_int        m_a,
                              aoclsparse_int        n_a,
                              aoclsparse_int        m_b,
                              aoclsparse_int        n_b,
                              aoclsparse_int        nnz_a,
                              aoclsparse_int        nnz_b,
                              aoclsparse_index_base b_a,
                              aoclsparse_index_base b_b,
                              aoclsparse_operation  op_a,
                              aoclsparse_operation  op_b)
    {
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
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

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
        ASSERT_EQ(aoclsparse_create_csr(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);

        aoclsparse_matrix C = NULL;
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_invalid_size);
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&C);
    }
    // tests for Invalid base value
    template <typename T>
    void test_sp2m_invalid_base(aoclsparse_int        m_a,
                                aoclsparse_int        n_a,
                                aoclsparse_int        m_b,
                                aoclsparse_int        n_b,
                                aoclsparse_int        nnz_a,
                                aoclsparse_int        nnz_b,
                                aoclsparse_index_base b_a,
                                aoclsparse_index_base b_b,
                                aoclsparse_operation  op_a,
                                aoclsparse_operation  op_b,
                                aoclsparse_int        stage)
    {
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
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

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
        ASSERT_EQ(aoclsparse_create_csr(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);

        // Invalid base for A matrix
        descrA->base        = (aoclsparse_index_base)3;
        aoclsparse_matrix C = NULL;
        if(stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_invalid_value);
        }
        else if(stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_invalid_value);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_invalid_value);
        }
        aoclsparse_destroy(&C);

        // Invalid base for B matrix
        descrA->base = b_a;
        descrB->base = (aoclsparse_index_base)3;
        if(stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_invalid_value);
        }
        else if(stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_invalid_value);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_invalid_value);
        }
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&C);
    }
    // tests for settings not implemented
    template <typename T>
    void test_sp2m_not_implemented(aoclsparse_int        m_a,
                                   aoclsparse_int        n_a,
                                   aoclsparse_int        m_b,
                                   aoclsparse_int        n_b,
                                   aoclsparse_int        nnz_a,
                                   aoclsparse_int        nnz_b,
                                   aoclsparse_index_base b_a,
                                   aoclsparse_index_base b_b,
                                   aoclsparse_operation  op_a,
                                   aoclsparse_operation  op_b,
                                   aoclsparse_int        stage)
    {
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
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

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
        ASSERT_EQ(aoclsparse_create_csr(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);

        // and expect not_implemented for !aoclsparse_matrix_type_general for matrix A and B
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_symmetric);
        aoclsparse_matrix C = NULL;
        if(stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_not_implemented);
        }
        else if(stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_not_implemented);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_not_implemented);
        }
        aoclsparse_destroy(&C);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_general);
        aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_symmetric);
        if(stage == 0)
        {
            request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_not_implemented);
        }
        else if(stage == 1)
        {
            request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_not_implemented);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_not_implemented);
        }

        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }
    // CSC input not implemented for sp2m
    template <typename T>
    void test_sp2m_csc_not_impl()
    {
        aoclsparse_index_base base    = aoclsparse_index_base_zero;
        aoclsparse_operation  op_a    = aoclsparse_operation_none;
        aoclsparse_operation  op_b    = aoclsparse_operation_none;
        aoclsparse_request    request = aoclsparse_stage_full_computation;

        // Small 3x3 matrix with 4 nnz
        aoclsparse_int m = 3, n = 3, nnz = 4;
        std::vector<T> val;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
            val.assign({{1, 1}, {1, 2}, {2, 3}, {4, 2}});
        else
            val.assign({1, 2, 3, 4});

        // CSC arrays
        std::vector<aoclsparse_int> row_ind = {0, 2, 1, 2};
        std::vector<aoclsparse_int> col_ptr = {0, 2, 3, 4};
        // CSR arrays (same sparsity, different layout)
        std::vector<aoclsparse_int> col_ind = {0, 1, 0, 2};
        std::vector<aoclsparse_int> row_ptr = {0, 2, 3, 4};

        aoclsparse_mat_descr descrA, descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);

        // Test 1: A=CSC, B=CSR
        aoclsparse_matrix A_csc, B_csr, C = NULL;
        ASSERT_EQ(aoclsparse_create_csc<T>(
                      &A_csc, base, m, n, nnz, col_ptr.data(), row_ind.data(), val.data()),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(
                      &B_csr, base, m, n, nnz, row_ptr.data(), col_ind.data(), val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A_csc, op_b, descrB, B_csr, request, &C),
                  aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A_csc);
        aoclsparse_destroy(&B_csr);
        aoclsparse_destroy(&C);

        // Test 2: A=CSR, B=CSC
        C = NULL;
        aoclsparse_matrix A_csr, B_csc;
        ASSERT_EQ(aoclsparse_create_csr<T>(
                      &A_csr, base, m, n, nnz, row_ptr.data(), col_ind.data(), val.data()),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csc<T>(
                      &B_csc, base, m, n, nnz, col_ptr.data(), row_ind.data(), val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A_csr, op_b, descrB, B_csc, request, &C),
                  aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A_csr);
        aoclsparse_destroy(&B_csc);
        aoclsparse_destroy(&C);

        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy_mat_descr(descrB);
    }

    // Test to verify sp2m actual functionality with CSC input (double only).
    // Uses a non-symmetric 3x3 matrix so CSR vs CSC interpretation gives
    // different results. The mathematically correct result of
    //   C = A * B  (with both A and B given as CSC)
    // is compared against the actual output from sp2m.
    // Note: test_sp2m_success uses random matrices + blis::gemm reference;
    //       here we use a hand-computed reference for a small known matrix.
    void test_sp2m_csc_functionality()
    {
        aoclsparse_index_base base    = aoclsparse_index_base_zero;
        aoclsparse_operation  op_a    = aoclsparse_operation_none;
        aoclsparse_operation  op_b    = aoclsparse_operation_none;
        aoclsparse_request    request = aoclsparse_stage_full_computation;

        // Define a non-symmetric 3x3 CSC matrix A:
        //     [1  3  0]
        // A = [0  2  0]
        //     [0  0  4]
        // CSC: col_ptr={0,1,3,4}, row_ind={0,0,1,2}, val={1,3,2,4}
        aoclsparse_int              m = 3, n = 3, nnz = 4;
        std::vector<double>         val_a     = {1, 3, 2, 4};
        std::vector<aoclsparse_int> col_ptr_a = {0, 1, 3, 4};
        std::vector<aoclsparse_int> row_ind_a = {0, 0, 1, 2};

        // B = 3x3 identity in CSR for simplicity, so C = A * I = A
        aoclsparse_int              nnz_b     = 3;
        std::vector<double>         val_b     = {1, 1, 1};
        std::vector<aoclsparse_int> col_ind_b = {0, 1, 2};
        std::vector<aoclsparse_int> row_ptr_b = {0, 1, 2, 3};

        // Correct dense result: A * I = A
        //     [1  3  0]
        //     [0  2  0]
        //     [0  0  4]
        // Flattened row-major:
        std::vector<double> dense_c_exp = {1, 3, 0, 0, 2, 0, 0, 0, 4};

        // If sp2m misinterprets CSC A as CSR (reads A^T as A), it computes
        // A^T * I = A^T:
        //     [1  0  0]
        //     [3  2  0]   <-- incorrect
        //     [0  0  4]

        aoclsparse_mat_descr descrA, descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csc<double>(
                      &A, base, m, n, nnz, col_ptr_a.data(), row_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);

        aoclsparse_matrix B;
        ASSERT_EQ(aoclsparse_create_csr<double>(
                      &B, base, m, n, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);

        aoclsparse_matrix C      = NULL;
        aoclsparse_status status = aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C);

        // If sp2m rejects CSC input (after DOID guard is enabled), expect not_implemented.
        // If sp2m processes CSC input, validate output against correct reference.
        if(status == aoclsparse_status_success)
        {
            // Export and convert C to dense for comparison
            aoclsparse_int        m_c, n_c, nnz_c;
            aoclsparse_int       *row_ptr_c = NULL;
            aoclsparse_int       *col_ind_c = NULL;
            double               *val_c     = NULL;
            aoclsparse_index_base base_c;

            ASSERT_EQ(aoclsparse_export_csr(
                          C, &base_c, &m_c, &n_c, &nnz_c, &row_ptr_c, &col_ind_c, &val_c),
                      aoclsparse_status_success);

            aoclsparse_mat_descr descrC;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descrC), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descrC, base_c), aoclsparse_status_success);

            std::vector<double> dense_c(m_c * n_c, 0.0);
            aoclsparse_csr2dense(m_c,
                                 n_c,
                                 descrC,
                                 val_c,
                                 row_ptr_c,
                                 col_ind_c,
                                 dense_c.data(),
                                 n_c,
                                 aoclsparse_order_row);

            tolerance_t<double> abserr = sqrt(std::numeric_limits<tolerance_t<double>>::epsilon());
            EXPECT_ARR_NEAR(m_c * n_c, dense_c.data(), dense_c_exp.data(), abserr);

            aoclsparse_destroy_mat_descr(descrC);
        }
        else
        {
            EXPECT_EQ(status, aoclsparse_status_not_implemented);
        }

        aoclsparse_destroy(&C);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
    }

    void test_sp2m_wrong_datatype()
    {
        aoclsparse_operation  op_a = aoclsparse_operation_none;
        aoclsparse_operation  op_b = aoclsparse_operation_none;
        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_request    request;
        aoclsparse_int        m = 2, k = 3, n = 2, nnzA = 1, nnzB = 3;
        float                 csr_valA[]     = {42.};
        aoclsparse_int        csr_col_indA[] = {1};
        aoclsparse_int        csr_row_ptrA[] = {0, 0, 1};
        double                csr_valB[]     = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[] = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[] = {0, 1, 2, 3};
        aoclsparse_mat_descr  descrA, descrB;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_create_mat_descr(&descrB);

        aoclsparse_matrix A;
        aoclsparse_create_scsr(&A, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix B;
        aoclsparse_create_dcsr(&B, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix C = NULL;
        request             = aoclsparse_stage_full_computation;
        // For float A and double B matrices, invoke sp2m
        // and expect wrong type error
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_wrong_type);

        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }
    // Test for success and verify results against Dense GEMM results.
    template <typename T>
    void test_sp2m_success(aoclsparse_int        m_a,
                           aoclsparse_int        n_a,
                           aoclsparse_int        m_b,
                           aoclsparse_int        n_b,
                           aoclsparse_int        nnz_a,
                           aoclsparse_int        nnz_b,
                           aoclsparse_index_base b_a,
                           aoclsparse_index_base b_b,
                           aoclsparse_operation  op_a,
                           aoclsparse_operation  op_b,
                           aoclsparse_int        stage)
    {
        aoclsparse_int        m_c, n_c, nnz_c;
        aoclsparse_int       *row_ptr_c = NULL;
        aoclsparse_int       *col_ind_c = NULL;
        T                    *val_c     = NULL;
        aoclsparse_index_base base_c;
        aoclsparse_seedrand();

        std::vector<T> dense_a(m_a * n_a), dense_b(m_b * n_b), dense_c, dense_c_exp;
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

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
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

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
        ASSERT_EQ(aoclsparse_create_csr(
                      &B, b_b, m_b, n_b, nnz_b, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, b_b), aoclsparse_status_success);

        // Expect success from sp2m
        aoclsparse_matrix C = NULL;
        if(stage == 0)
        {
            aoclsparse_request request = aoclsparse_stage_nnz_count;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_success);
            request = aoclsparse_stage_finalize;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_success);
        }
        else if(stage == 1)
        {
            aoclsparse_request request = aoclsparse_stage_full_computation;
            EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                      aoclsparse_status_success);
        }
        // Export resultant C matrix and Convert to Dense
        ASSERT_EQ(
            aoclsparse_export_csr(C, &base_c, &m_c, &n_c, &nnz_c, &row_ptr_c, &col_ind_c, &val_c),
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
                             val_a.data(),
                             row_ptr_a.data(),
                             col_ind_a.data(),
                             dense_a.data(),
                             n_a,
                             aoclsparse_order_row);
        aoclsparse_csr2dense(m_b,
                             n_b,
                             descrB,
                             val_b.data(),
                             row_ptr_b.data(),
                             col_ind_b.data(),
                             dense_b.data(),
                             n_b,
                             aoclsparse_order_row);
        aoclsparse_int m = m_c, n = n_c, k = -1;
        if((op_a == aoclsparse_operation_none) && (op_b == aoclsparse_operation_none))
        {
            k = n_a; // m_b
        }
        else if(((op_a == aoclsparse_operation_transpose)
                 || (op_a == aoclsparse_operation_conjugate_transpose))
                && (op_b == aoclsparse_operation_none))
        {
            k = m_a; // m_b
        }
        else if((op_a == aoclsparse_operation_none)
                && ((op_b == aoclsparse_operation_transpose)
                    || (op_b == aoclsparse_operation_conjugate_transpose)))
        {
            k = n_a; // n_b
        }
        else if(((op_a == aoclsparse_operation_transpose)
                 || (op_a == aoclsparse_operation_conjugate_transpose))
                && ((op_b == aoclsparse_operation_transpose)
                    || (op_b == aoclsparse_operation_conjugate_transpose)))
        {
            k = m_a; // n_b
        }
        T alpha, beta;
        // Invoke Dense GEMM for Dense A and B matrices
        // Compare Dense GEMM output and sp2m output(converted to Dense)
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};

            if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                blis::gemm(CblasRowMajor,
                           (CBLAS_TRANSPOSE)op_a,
                           (CBLAS_TRANSPOSE)op_b,
                           (int64_t)m,
                           (int64_t)n,
                           (int64_t)k,
                           *reinterpret_cast<const std::complex<float> *>(&alpha),
                           (std::complex<float> const *)dense_a.data(),
                           (int64_t)n_a,
                           (std::complex<float> const *)dense_b.data(),
                           (int64_t)n_b,
                           *reinterpret_cast<const std::complex<float> *>(&beta),
                           (std::complex<float> *)dense_c_exp.data(),
                           (int64_t)n_c);
                EXPECT_COMPLEX_ARR_NEAR(m_c * n_c,
                                        ((std::complex<float> *)dense_c.data()),
                                        ((std::complex<float> *)dense_c_exp.data()),
                                        abserr);
            }
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                blis::gemm(CblasRowMajor,
                           (CBLAS_TRANSPOSE)op_a,
                           (CBLAS_TRANSPOSE)op_b,
                           (int64_t)m,
                           (int64_t)n,
                           (int64_t)k,
                           *reinterpret_cast<const std::complex<double> *>(&alpha),
                           (std::complex<double> const *)dense_a.data(),
                           (int64_t)n_a,
                           (std::complex<double> const *)dense_b.data(),
                           (int64_t)n_b,
                           *reinterpret_cast<const std::complex<double> *>(&beta),
                           (std::complex<double> *)dense_c_exp.data(),
                           (int64_t)n_c);
                EXPECT_COMPLEX_ARR_NEAR(m_c * n_c,
                                        ((std::complex<double> *)dense_c.data()),
                                        ((std::complex<double> *)dense_c_exp.data()),
                                        abserr);
            }
        }
        else
        {
            alpha = 1;
            beta  = 0;
            blis::gemm(CblasRowMajor,
                       (CBLAS_TRANSPOSE)op_a,
                       (CBLAS_TRANSPOSE)op_b,
                       (int64_t)m,
                       (int64_t)n,
                       (int64_t)k,
                       (T)alpha,
                       (T const *)dense_a.data(),
                       (int64_t)n_a,
                       (T const *)dense_b.data(),
                       (int64_t)n_b,
                       (T)beta,
                       (T *)dense_c_exp.data(),
                       (int64_t)n_c);
            EXPECT_ARR_NEAR(m_c * n_c, dense_c.data(), dense_c_exp.data(), abserr);
        }
        aoclsparse_destroy(&C);
        aoclsparse_destroy_mat_descr(descrC);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
    }

    // Invoke nnz_count stage once and finalise stage twice with change in only value arrays of
    // A and B matrices. sp2m should return success and results should match as well.
    template <typename T>
    void test_sp2m_finalize(aoclsparse_int        m_a,
                            aoclsparse_int        n_a,
                            aoclsparse_int        m_b,
                            aoclsparse_int        n_b,
                            aoclsparse_int        nnz_a,
                            aoclsparse_int        nnz_b,
                            aoclsparse_index_base b_a,
                            aoclsparse_index_base b_b,
                            aoclsparse_operation  op_a,
                            aoclsparse_operation  op_b)
    {
        aoclsparse_int        m_c, n_c, nnz_c;
        aoclsparse_int       *row_ptr_c = NULL;
        aoclsparse_int       *col_ind_c = NULL;
        T                    *val_c     = NULL;
        aoclsparse_index_base base_c;
        aoclsparse_seedrand();

        std::vector<T> dense_a(m_a * n_a), dense_b(m_b * n_b), dense_c, dense_c_exp;
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

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
        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);

        //Randomly generate A matrix
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

        // Invoke sp2m with nnz_count followed by finalize stage.
        aoclsparse_matrix  C       = NULL;
        aoclsparse_request request = aoclsparse_stage_nnz_count;
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_success);
        request = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_success);

        aoclsparse::csr *A_csr = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
        aoclsparse::csr *B_csr = dynamic_cast<aoclsparse::csr *>(B->mats[0]);
        EXPECT_NE(A_csr, nullptr);
        EXPECT_NE(B_csr, nullptr);
        // Modify the values of A and B matix value arrays.
        for(aoclsparse_int i = 0; i < A->nnz; i++)
            ((T *)A_csr->val)[i] = random_generator_normal<T>();
        for(aoclsparse_int i = 0; i < B->nnz; i++)
            ((T *)B_csr->val)[i] = random_generator_normal<T>();

        // Invoke sp2m with finalize stage alone.
        // Expect success as C matrix created in previous invocation
        // is reused to update the value array alone.
        request = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_sp2m(op_a, descrA, A, op_b, descrB, B, request, &C),
                  aoclsparse_status_success);

        // Export resultant C matrix and Convert to Dense
        ASSERT_EQ(
            aoclsparse_export_csr(C, &base_c, &m_c, &n_c, &nnz_c, &row_ptr_c, &col_ind_c, &val_c),
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
                             val_a.data(),
                             row_ptr_a.data(),
                             col_ind_a.data(),
                             dense_a.data(),
                             n_a,
                             aoclsparse_order_row);
        aoclsparse_csr2dense(m_b,
                             n_b,
                             descrB,
                             val_b.data(),
                             row_ptr_b.data(),
                             col_ind_b.data(),
                             dense_b.data(),
                             n_b,
                             aoclsparse_order_row);
        aoclsparse_int m = m_c, n = n_c, k = -1;
        if((op_a == aoclsparse_operation_none) && (op_b == aoclsparse_operation_none))
        {
            k = n_a; // m_b
        }
        else if(((op_a == aoclsparse_operation_transpose)
                 || (op_a == aoclsparse_operation_conjugate_transpose))
                && (op_b == aoclsparse_operation_none))
        {
            k = m_a; // m_b
        }
        else if((op_a == aoclsparse_operation_none)
                && ((op_b == aoclsparse_operation_transpose)
                    || (op_b == aoclsparse_operation_conjugate_transpose)))
        {
            k = n_a; // n_b
        }
        else if(((op_a == aoclsparse_operation_transpose)
                 || (op_a == aoclsparse_operation_conjugate_transpose))
                && ((op_b == aoclsparse_operation_transpose)
                    || (op_b == aoclsparse_operation_conjugate_transpose)))
        {
            k = m_a; // n_b
        }
        T alpha, beta;
        // Invoke Dense GEMM for Dense A and B matrices
        // Compare Dense GEMM output and sp2m output(converted to Dense)
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};

            if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                blis::gemm(CblasRowMajor,
                           (CBLAS_TRANSPOSE)op_a,
                           (CBLAS_TRANSPOSE)op_b,
                           (int64_t)m,
                           (int64_t)n,
                           (int64_t)k,
                           *reinterpret_cast<const std::complex<float> *>(&alpha),
                           (std::complex<float> const *)dense_a.data(),
                           (int64_t)n_a,
                           (std::complex<float> const *)dense_b.data(),
                           (int64_t)n_b,
                           *reinterpret_cast<const std::complex<float> *>(&beta),
                           (std::complex<float> *)dense_c_exp.data(),
                           (int64_t)n_c);
                EXPECT_COMPLEX_ARR_NEAR(m_c * n_c,
                                        ((std::complex<float> *)dense_c.data()),
                                        ((std::complex<float> *)dense_c_exp.data()),
                                        abserr);
            }
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                blis::gemm(CblasRowMajor,
                           (CBLAS_TRANSPOSE)op_a,
                           (CBLAS_TRANSPOSE)op_b,
                           (int64_t)m,
                           (int64_t)n,
                           (int64_t)k,
                           *reinterpret_cast<const std::complex<double> *>(&alpha),
                           (std::complex<double> const *)dense_a.data(),
                           (int64_t)n_a,
                           (std::complex<double> const *)dense_b.data(),
                           (int64_t)n_b,
                           *reinterpret_cast<const std::complex<double> *>(&beta),
                           (std::complex<double> *)dense_c_exp.data(),
                           (int64_t)n_c);
                EXPECT_COMPLEX_ARR_NEAR(m_c * n_c,
                                        ((std::complex<double> *)dense_c.data()),
                                        ((std::complex<double> *)dense_c_exp.data()),
                                        abserr);
            }
        }
        else
        {
            alpha = 1;
            beta  = 0;
            blis::gemm(CblasRowMajor,
                       (CBLAS_TRANSPOSE)op_a,
                       (CBLAS_TRANSPOSE)op_b,
                       (int64_t)m,
                       (int64_t)n,
                       (int64_t)k,
                       (T)alpha,
                       (T const *)dense_a.data(),
                       (int64_t)n_a,
                       (T const *)dense_b.data(),
                       (int64_t)n_b,
                       (T)beta,
                       (T *)dense_c_exp.data(),
                       (int64_t)n_c);
            EXPECT_ARR_NEAR(m_c * n_c, dense_c.data(), dense_c_exp.data(), abserr);
        }
        aoclsparse_destroy(&C);
        aoclsparse_destroy_mat_descr(descrC);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
    }

    TEST(sp2m, NullArgAll)
    {
        test_sp2m_nullptr<double>(3,
                                  2,
                                  4,
                                  2,
                                  2,
                                  5,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_one,
                                  aoclsparse_operation_none,
                                  aoclsparse_operation_transpose);
        test_sp2m_nullptr<float>(3,
                                 2,
                                 4,
                                 3,
                                 2,
                                 5,
                                 aoclsparse_index_base_one,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_operation_transpose,
                                 aoclsparse_operation_transpose);
        test_sp2m_nullptr<aoclsparse_double_complex>(3,
                                                     2,
                                                     4,
                                                     2,
                                                     2,
                                                     5,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_index_base_one,
                                                     aoclsparse_operation_none,
                                                     aoclsparse_operation_transpose);
        test_sp2m_nullptr<aoclsparse_float_complex>(3,
                                                    2,
                                                    4,
                                                    2,
                                                    2,
                                                    5,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_index_base_one,
                                                    aoclsparse_operation_none,
                                                    aoclsparse_operation_transpose);
    }
    TEST(sp2m, DoNothingAll)
    {
        test_sp2m_do_nothing<double>(5,
                                     4,
                                     4,
                                     5,
                                     7,
                                     9,
                                     aoclsparse_index_base_zero,
                                     aoclsparse_index_base_one,
                                     aoclsparse_operation_none,
                                     aoclsparse_operation_none);
        test_sp2m_do_nothing<float>(3,
                                    4,
                                    4,
                                    5,
                                    7,
                                    9,
                                    aoclsparse_index_base_one,
                                    aoclsparse_index_base_zero,
                                    aoclsparse_operation_none,
                                    aoclsparse_operation_none);
        test_sp2m_do_nothing<aoclsparse_double_complex>(5,
                                                        4,
                                                        4,
                                                        6,
                                                        7,
                                                        9,
                                                        aoclsparse_index_base_zero,
                                                        aoclsparse_index_base_one,
                                                        aoclsparse_operation_none,
                                                        aoclsparse_operation_none);
        test_sp2m_do_nothing<aoclsparse_float_complex>(3,
                                                       4,
                                                       4,
                                                       5,
                                                       7,
                                                       9,
                                                       aoclsparse_index_base_one,
                                                       aoclsparse_index_base_zero,
                                                       aoclsparse_operation_none,
                                                       aoclsparse_operation_none);
    }
    TEST(sp2m, WrongSizeAll)
    {
        test_sp2m_wrong_size<double>(3,
                                     2,
                                     4,
                                     5,
                                     2,
                                     5,
                                     aoclsparse_index_base_zero,
                                     aoclsparse_index_base_zero,
                                     aoclsparse_operation_none,
                                     aoclsparse_operation_none);
        test_sp2m_wrong_size<float>(3,
                                    4,
                                    4,
                                    5,
                                    7,
                                    9,
                                    aoclsparse_index_base_zero,
                                    aoclsparse_index_base_one,
                                    aoclsparse_operation_transpose,
                                    aoclsparse_operation_none);
        test_sp2m_wrong_size<aoclsparse_double_complex>(3,
                                                        2,
                                                        2,
                                                        5,
                                                        2,
                                                        5,
                                                        aoclsparse_index_base_zero,
                                                        aoclsparse_index_base_one,
                                                        aoclsparse_operation_conjugate_transpose,
                                                        aoclsparse_operation_none);
        test_sp2m_wrong_size<aoclsparse_float_complex>(3,
                                                       4,
                                                       4,
                                                       5,
                                                       7,
                                                       9,
                                                       aoclsparse_index_base_one,
                                                       aoclsparse_index_base_one,
                                                       aoclsparse_operation_transpose,
                                                       aoclsparse_operation_conjugate_transpose);
    }

    TEST(sp2m, WrongBaseAll)
    {
        test_sp2m_invalid_base<double>(5,
                                       4,
                                       4,
                                       5,
                                       7,
                                       9,
                                       aoclsparse_index_base_zero,
                                       aoclsparse_index_base_one,
                                       aoclsparse_operation_transpose,
                                       aoclsparse_operation_conjugate_transpose,
                                       0);
        test_sp2m_invalid_base<float>(3,
                                      4,
                                      3,
                                      5,
                                      7,
                                      9,
                                      aoclsparse_index_base_one,
                                      aoclsparse_index_base_zero,
                                      aoclsparse_operation_transpose,
                                      aoclsparse_operation_none,
                                      1);
        test_sp2m_invalid_base<aoclsparse_double_complex>(5,
                                                          4,
                                                          5,
                                                          4,
                                                          7,
                                                          9,
                                                          aoclsparse_index_base_zero,
                                                          aoclsparse_index_base_one,
                                                          aoclsparse_operation_none,
                                                          aoclsparse_operation_conjugate_transpose,
                                                          0);
        test_sp2m_invalid_base<aoclsparse_float_complex>(3,
                                                         4,
                                                         4,
                                                         5,
                                                         7,
                                                         9,
                                                         aoclsparse_index_base_one,
                                                         aoclsparse_index_base_zero,
                                                         aoclsparse_operation_none,
                                                         aoclsparse_operation_none,
                                                         1);
    }
    TEST(sp2m, NotImplAll)
    {
        test_sp2m_not_implemented<double>(5,
                                          6,
                                          4,
                                          5,
                                          7,
                                          9,
                                          aoclsparse_index_base_zero,
                                          aoclsparse_index_base_one,
                                          aoclsparse_operation_transpose,
                                          aoclsparse_operation_conjugate_transpose,
                                          0);
        test_sp2m_not_implemented<float>(3,
                                         6,
                                         3,
                                         5,
                                         7,
                                         9,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_operation_transpose,
                                         aoclsparse_operation_none,
                                         1);
        test_sp2m_not_implemented<aoclsparse_double_complex>(
            5,
            4,
            6,
            4,
            7,
            9,
            aoclsparse_index_base_zero,
            aoclsparse_index_base_one,
            aoclsparse_operation_none,
            aoclsparse_operation_conjugate_transpose,
            0);
        test_sp2m_not_implemented<aoclsparse_float_complex>(5,
                                                            4,
                                                            4,
                                                            6,
                                                            7,
                                                            9,
                                                            aoclsparse_index_base_one,
                                                            aoclsparse_index_base_zero,
                                                            aoclsparse_operation_none,
                                                            aoclsparse_operation_none,
                                                            1);
    }
    TEST(sp2m, CSCNotImpl)
    {
        test_sp2m_csc_not_impl<double>();
        test_sp2m_csc_not_impl<aoclsparse_double_complex>();
    }
    TEST(sp2m, CSCFunctionalityDouble)
    {
        test_sp2m_csc_functionality();
    }
    TEST(sp2m, WrongType)
    {
        test_sp2m_wrong_datatype();
    }
    TEST(sp2m, SuccessTypeDouble)
    {
        test_sp2m_success<double>(4,
                                  4,
                                  4,
                                  4,
                                  10,
                                  8,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_operation_none,
                                  aoclsparse_operation_transpose,
                                  0);
        test_sp2m_success<double>(5,
                                  6,
                                  6,
                                  7,
                                  10,
                                  15,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_one,
                                  aoclsparse_operation_none,
                                  aoclsparse_operation_none,
                                  1);
    }
    TEST(sp2m, SuccessTypeFloat)
    {
        test_sp2m_success<float>(5,
                                 6,
                                 5,
                                 8,
                                 11,
                                 7,
                                 aoclsparse_index_base_one,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_operation_transpose,
                                 aoclsparse_operation_none,
                                 1);
        test_sp2m_success<float>(5,
                                 4,
                                 5,
                                 6,
                                 11,
                                 17,
                                 aoclsparse_index_base_one,
                                 aoclsparse_index_base_one,
                                 aoclsparse_operation_conjugate_transpose,
                                 aoclsparse_operation_none,
                                 0);
    }
    TEST(sp2m, SuccessTypeCDouble)
    {
        test_sp2m_success<aoclsparse_double_complex>(6,
                                                     8,
                                                     5,
                                                     6,
                                                     11,
                                                     17,
                                                     aoclsparse_index_base_one,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_operation_conjugate_transpose,
                                                     aoclsparse_operation_transpose,
                                                     0);
        test_sp2m_success<aoclsparse_double_complex>(5,
                                                     4,
                                                     6,
                                                     4,
                                                     11,
                                                     17,
                                                     aoclsparse_index_base_one,
                                                     aoclsparse_index_base_one,
                                                     aoclsparse_operation_none,
                                                     aoclsparse_operation_transpose,
                                                     1);
    }
    TEST(sp2m, SuccessTypeCFloat)
    {
        test_sp2m_success<aoclsparse_float_complex>(6,
                                                    7,
                                                    8,
                                                    6,
                                                    12,
                                                    15,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_index_base_one,
                                                    aoclsparse_operation_conjugate_transpose,
                                                    aoclsparse_operation_conjugate_transpose,
                                                    1);
        test_sp2m_success<aoclsparse_float_complex>(5,
                                                    7,
                                                    6,
                                                    5,
                                                    10,
                                                    15,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_operation_transpose,
                                                    aoclsparse_operation_conjugate_transpose,
                                                    0);
    }
    TEST(sp2m, FinalizeAll)
    {
        test_sp2m_finalize<float>(6,
                                  4,
                                  5,
                                  6,
                                  12,
                                  12,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_one,
                                  aoclsparse_operation_conjugate_transpose,
                                  aoclsparse_operation_conjugate_transpose);
        test_sp2m_finalize<aoclsparse_float_complex>(5,
                                                     7,
                                                     6,
                                                     5,
                                                     15,
                                                     15,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_operation_transpose,
                                                     aoclsparse_operation_conjugate_transpose);
    }
} // namespace
