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
#include "aoclsparse_reference.hpp"

#include <algorithm>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "blis.hh"
#pragma GCC diagnostic pop

namespace
{
    aoclsparse_operation   op_t    = aoclsparse_operation_transpose;
    aoclsparse_operation   op_h    = aoclsparse_operation_conjugate_transpose;
    aoclsparse_operation   op_n    = aoclsparse_operation_none;
    aoclsparse_index_base  zero    = aoclsparse_index_base_zero;
    aoclsparse_index_base  one     = aoclsparse_index_base_one;
    aoclsparse_fill_mode   fl_up   = aoclsparse_fill_mode_upper;
    aoclsparse_fill_mode   fl_lo   = aoclsparse_fill_mode_lower;
    aoclsparse_diag_type   diag_u  = aoclsparse_diag_type_unit;
    aoclsparse_diag_type   diag_nu = aoclsparse_diag_type_non_unit;
    aoclsparse_diag_type   diag_z  = aoclsparse_diag_type_zero;
    aoclsparse_matrix_type mat_s   = aoclsparse_matrix_type_symmetric;
    aoclsparse_matrix_type mat_t   = aoclsparse_matrix_type_triangular;
    aoclsparse_matrix_type mat_h   = aoclsparse_matrix_type_hermitian;

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
        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

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
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
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
        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

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
                  aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_base_indexing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_matrix    A;
        int                  invalid_index_base = 2;
        aoclsparse_int       M = 5, N = 5, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T                    x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T                    y[M];
        aoclsparse_mat_descr descr;

        aoclsparse_int csr_row_ptr[] = {1, 3, 4, 5, 8, 9};
        aoclsparse_int csr_col_ind[] = {1, 4, 2, 3, 2, 4, 5, 5};
        T              csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_create_csr<T>(
                      &A, aoclsparse_index_base_one, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);

        descr->base = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_base_index_mismatch()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_matrix    A;
        aoclsparse_int       M = 5, N = 5, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T                    x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T                    y[M];
        aoclsparse_mat_descr descr;

        aoclsparse_int csr_row_ptr[] = {1, 3, 4, 5, 8, 9};
        aoclsparse_int csr_col_ind[] = {1, 4, 2, 3, 2, 4, 5, 5};
        T              csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};

        // TEST CASE 1: descriptor base is base-zero and aoclsparse_matrix base is base-one
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(
                      &A, aoclsparse_index_base_one, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_invalid_value);

        aoclsparse_matrix A_2;
        aoclsparse_int    csr_row_ptr_2[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int    csr_col_ind_2[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T                 csr_val_2[]     = {1, 2, 3, 4, 5, 6, 7, 8};
        // TEST CASE 2: descriptor base is base-one and aoclsparse_matrix base is base-zero
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(&A_2,
                                           aoclsparse_index_base_zero,
                                           M,
                                           N,
                                           NNZ,
                                           csr_row_ptr_2,
                                           csr_col_ind_2,
                                           csr_val_2),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A_2, descr, x, &beta, y),
                  aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_destroy(&A_2), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
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
        aoclsparse_set_mat_index_base(descr, base);

        aoclsparse_int    csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int    csr_col_ind[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T                 csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};
        aoclsparse_matrix A;
        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        trans = aoclsparse_operation_none;
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_hermitian);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }
    template <typename T>
    void test_invalid_kid()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 5, N = 5, NNZ = 8;
        T              alpha = {1.0, 0.0};
        T              beta  = {0.0, 0.0};
        // Initialise vectors
        T x[5] = {{0, 1}, {2, 1}, {3, 0}, {4, -1}, {5, 2}};
        T y[M];

        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, base);

        aoclsparse_int csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int csr_col_ind[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T csr_val[] = {{1, 1}, {2, 0}, {2, 2}, {1, -1}, {1, -1}, {2, 0}, {2, 3}, {5, 5}};
        aoclsparse_matrix A;
        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        trans = aoclsparse_operation_none;
        // hint AVX512 kernel, during execution on Zen3 should return error
        EXPECT_EQ(aoclsparse_set_mv_hint_kid(A, trans, descr, 0 /*calls*/, 3 /*kid AVX512*/),
                  aoclsparse_status_success);
        if(!can_exec_avx512_tests())
        {
            EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                      aoclsparse_status_invalid_kid);
        }

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 1, N = 1;
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

        aoclsparse_int    csr_row_ptr[] = {0, 0};
        aoclsparse_int    csr_col_ind[] = {0};
        T                 csr_val[]     = {0};
        aoclsparse_matrix AM0, AN0;
        aoclsparse_create_csr<T>(&AM0, base, 0, N, 0, csr_row_ptr, csr_col_ind, csr_val);
        aoclsparse_create_csr<T>(&AN0, base, M, 0, 0, csr_row_ptr, csr_col_ind, csr_val);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, AM0, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, AN0, descr, x, &beta, y),
                  aoclsparse_status_success);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(&AM0), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&AN0), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_success()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 5, N = 4, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[4]       = {1.0, 2.0, 3.0, 4.0};
        T y[5]       = {0};
        T exp_y_l[5] = {1, 6, 12, 56, 16};
        T exp_y_u[5] = {9, 6, 12, 28, 0};

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_index_base base = aoclsparse_index_base_zero;

        aoclsparse_int    csr_row_ptr[6] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int    csr_col_ind[8] = {0, 3, 1, 2, 1, 2, 3, 1};
        T                 csr_val[8]     = {1, 2, 3, 4, 5, 6, 7, 8};
        aoclsparse_matrix A;
        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y_l);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);

        for(int i = 0; i < M; i++)
        {
            y[i] = 0.0;
        }
        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y_u);

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    template <typename T>
    void test_mvtrg_success(aoclsparse_int         m_a,
                            aoclsparse_int         n_a,
                            aoclsparse_int         nnz_a,
                            aoclsparse_operation   op_a,
                            aoclsparse_matrix_type mat_type,
                            aoclsparse_fill_mode   fill,
                            aoclsparse_diag_type   diag,
                            aoclsparse_index_base  b_a)
    {

        aoclsparse_seedrand();
        std::vector<T> y, y_ref, x;
        std::vector<T> z(m_a), z_t(n_a);
        T              alpha = 1, beta = 2.0;
        if(op_a != aoclsparse_operation_none)
        {
            x = z_t;
            y = z;
            aoclsparse_init<T>(x, 1, n_a, 1);
            aoclsparse_init<T>(y, 1, m_a, 1);
            y_ref = y;
        }
        else
        {
            x = z;
            y = z_t;
            aoclsparse_init<T>(x, 1, m_a, 1);
            aoclsparse_init<T>(y, 1, n_a, 1);
            y_ref = y;
        }
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
        ASSERT_EQ(aoclsparse_set_mat_type(descrA, mat_type), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descrA, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descrA, diag), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, descrA, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(ref_csrmvtrg(op_a,
                               alpha,
                               m_a,
                               n_a,
                               val_a.data(),
                               col_ind_a.data(),
                               row_ptr_a.data(),
                               fill,
                               diag,
                               b_a,
                               x.data(),
                               beta,
                               y_ref.data()),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(m_a, y, y_ref, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descrA), aoclsparse_status_success);
    }

    template <typename T>
    void test_mv_trianglular_strict()
    {
        aoclsparse_operation trans;
        aoclsparse_int       M = 5, N = 5, NNZ = 14;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T                     x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T                     y[5] = {0.0};
        aoclsparse_mat_descr  descr;
        aoclsparse_index_base base = aoclsparse_index_base_zero;

        /*      Matrix A
            1	0	5	3	9
            0	3	0	1	0
            2	0	4	0	6
            0	7	0	6	0
            13	0	9	0	8
        */
        aoclsparse_int csr_row_ptr[] = {0, 4, 6, 9, 11, 14};
        aoclsparse_int csr_col_ind[] = {0, 2, 3, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4};
        T              csr_val[]     = {1, 5, 3, 9, 3, 1, 2, 4, 6, 7, 6, 13, 9, 8};
        T              y_exp_lower[] = {0, 0, 2, 14, 40}; //{0, 0, 6, 28, 40};
        T              y_exp_upper[] = {72, 4.0, 30, 0, 0};

        aoclsparse_matrix      A;
        aoclsparse_matrix_type mattype;
        aoclsparse_fill_mode   fill;
        aoclsparse_diag_type   diag;

        trans   = aoclsparse_operation_none;
        mattype = aoclsparse_matrix_type_triangular;
        fill    = aoclsparse_fill_mode_lower;
        diag    = aoclsparse_diag_type_zero; // strictly lower

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descr, diag), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_exp_lower, expected_precision<T>());

        fill = aoclsparse_fill_mode_upper; // strictly upper given diag = aoclsparse_diag_type_zero
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_exp_upper, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

    template <typename T>
    void test_mvt_trianglular_strict()
    {
        aoclsparse_operation trans;
        aoclsparse_int       M = 5, N = 5, NNZ = 14;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T                     x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T                     y[5] = {0.0};
        aoclsparse_mat_descr  descr;
        aoclsparse_index_base base = aoclsparse_index_base_zero;

        /*      Matrix A
            1	0	5	3	9
            0	3	0	1	0
            2	0	4	0	6
            0	7	0	6	0
            13	0	9	0	8
        */
        aoclsparse_int csr_row_ptr[] = {0, 4, 6, 9, 11, 14};
        aoclsparse_int csr_col_ind[] = {0, 2, 3, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4};
        T              csr_val[]     = {1, 5, 3, 9, 3, 1, 2, 4, 6, 7, 6, 13, 9, 8};
        T              y_exp_lower[] = {71, 28, 45, 0, 0};
        T              y_exp_upper[] = {0, 0, 5, 5, 27};

        aoclsparse_matrix      A;
        aoclsparse_matrix_type mattype;
        aoclsparse_fill_mode   fill;
        aoclsparse_diag_type   diag;

        trans   = aoclsparse_operation_transpose;
        mattype = aoclsparse_matrix_type_triangular;
        fill    = aoclsparse_fill_mode_lower;
        diag    = aoclsparse_diag_type_zero; // strictly lower

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descr, diag), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_exp_lower, expected_precision<T>());

        fill = aoclsparse_fill_mode_upper; // strictly upper given diag = aoclsparse_diag_type_zero
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_exp_upper, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
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

        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

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

        EXPECT_EQ(ref_csrmvtrg(trans,
                               alpha,
                               M,
                               N,
                               csr_val,
                               csr_col_ind,
                               csr_row_ptr,
                               fill,
                               diag,
                               base,
                               x,
                               beta,
                               y_gold),
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
        EXPECT_EQ(ref_csrmvtrg(trans,
                               alpha,
                               M,
                               N,
                               csr_val,
                               csr_col_ind,
                               csr_row_ptr,
                               fill,
                               diag,
                               base,
                               x,
                               beta,
                               y_gold),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);

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
        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);
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
        EXPECT_EQ(ref_csrmvtrg(trans,
                               alpha,
                               M,
                               N,
                               csr_val,
                               csr_col_ind,
                               csr_row_ptr,
                               fill,
                               diag,
                               base,
                               x,
                               beta,
                               y_gold),
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
        EXPECT_EQ(ref_csrmvtrg(trans,
                               alpha,
                               M,
                               N,
                               csr_val,
                               csr_col_ind,
                               csr_row_ptr,
                               fill,
                               diag,
                               base,
                               x,
                               beta,
                               y_gold),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        //CASE 5: Check for invalid transpose value
        trans = (aoclsparse_operation)invalid_trans;
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_symm(std::string testcase, aoclsparse_index_base base)
    {
        SCOPED_TRACE(testcase);

        aoclsparse_operation trans;
        aoclsparse_int       M = 8, N = 8, NNZ = 19;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[8]      = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        T y[8]      = {0.0};
        T y_gold[8] = {0.0};

        aoclsparse_mat_descr descr;
        aoclsparse_matrix    A;

        //symmetric matrix with lower triangle
        aoclsparse_int csr_row_ptr[] = {0, 1, 2, 6, 7, 9, 12, 16, 19};
        aoclsparse_int csr_col_ind[] = {0, 1, 0, 3, 2, 1, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
        T              csr_val[] = {19, 10, 1, 13, 11, 13, 8, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9};

        if(base == aoclsparse_index_base_one)
        {
            for(aoclsparse_int i = 0; i < M + 1; i++)
                csr_row_ptr[i] += base;
            for(aoclsparse_int i = 0; i < NNZ; i++)
                csr_col_ind[i] += base;
        }

        trans = aoclsparse_operation_none;

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

        ASSERT_EQ(aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_mv_hint(A, trans, descr, 1), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_EQ(ref_csrmvsym(trans,
                               alpha,
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
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_symm_complex(std::string testcase, aoclsparse_index_base base)
    {
        SCOPED_TRACE(testcase);
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        aoclsparse_int M = 8, N = 8, NNZ = 18;
        T              alpha = {1.0, 0.0};
        T              beta  = {0.0, 0.0};
        // Initialise vectors
        T x[8]      = {{1.0, 1.0},
                       {2.0, 1.0},
                       {3.0, 3.0},
                       {4.0, 4.0},
                       {5.0, 5.0},
                       {6.0, 6.0},
                       {7.0, 7.0},
                       {8.0, 8.0}};
        T y[8]      = {{0, 0}};
        T y_gold[8] = {{0, 0}};

        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_mat_descr descr;
        aoclsparse_matrix    A;

        //symmetric matrix with lower triangle
        aoclsparse_int csr_row_ptr[] = {0, 1, 2, 5, 6, 8, 11, 15, 18};
        aoclsparse_int csr_col_ind[] = {0, 1, 0, 2, 1, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
        T              csr_val[]     = {{19, 19},
                                        {10, 10},
                                        {1, 1},
                                        {11, 11},
                                        {8, 8},
                                        {13, 13},
                                        {2, 2},
                                        {11, 11},
                                        {2, 2},
                                        {1, 1},
                                        {9, 9},
                                        {7, 7},
                                        {9, 9},
                                        {5, 5},
                                        {12, 12},
                                        {5, 5},
                                        {5, 5},
                                        {9, 9}};

        if(base == aoclsparse_index_base_one)
        {
            for(aoclsparse_int i = 0; i < M + 1; i++)
                csr_row_ptr[i] += base;
            for(aoclsparse_int i = 0; i < NNZ; i++)
                csr_col_ind[i] += base;
        }

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_mv_hint(A, trans, descr, 1), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);

        const std::complex<double> alphap = *reinterpret_cast<const std::complex<double> *>(&alpha);
        const std::complex<double> betap  = *reinterpret_cast<const std::complex<double> *>(&beta);

        EXPECT_EQ(ref_csrmvsym(trans,
                               alphap,
                               M,
                               (std::complex<double> *)csr_val,
                               csr_col_ind,
                               csr_row_ptr,
                               aoclsparse_fill_mode_lower,
                               aoclsparse_diag_type_non_unit,
                               base,
                               (std::complex<double> *)x,
                               betap,
                               (std::complex<double> *)y_gold),
                  aoclsparse_status_success);
        EXPECT_COMPLEX_ARR_NEAR(
            M, ((std::complex<double> *)y), ((std::complex<double> *)y_gold), abserr);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
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

        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_EQ(ref_csrmvsym(trans,
                               alpha,
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
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }
    template <typename T>
    void test_mv_conjugate_transpose()
    {
        aoclsparse_operation trans;
        aoclsparse_int       M = 5, N = 5, NNZ = 8;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T x[5]     = {1.0, 2.0, 3.0, 4.0, 5.0};
        T y[5]     = {0.0};
        T y_exp[5] = {1, 26, 12, 26, 68};

        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_mat_descr  descr;
        aoclsparse_matrix     A;
        aoclsparse_int        csr_row_ptr[] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int        csr_col_ind[] = {0, 3, 1, 2, 1, 3, 4, 4};
        T                     csr_val[]     = {1, 2, 3, 4, 5, 6, 7, 8};

        trans = aoclsparse_operation_conjugate_transpose;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        // Note: real conjugate transpose is same transpose.
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_exp, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    template <typename T>
    void mv_cmplx_success_sq5x5_all_ops(aoclsparse_operation op = aoclsparse_operation_none)
    {
        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        // matrix A
        /*
        1+i     0       0       0       2
        0       2+2i    1-i     0       0
        0       1-i     3+3i    0       2+3i
        0       0       0       4+4i    0
        2       0       2+3i    0       5+5i
        */

        aoclsparse_int M = 5, N = 5, NNZ = 11;
        T              alpha = {1.0, 0.0};
        T              beta  = {0.0, 0.0};
        // Initialise vectors
        T x[5] = {{0, 1}, {2, 1}, {3, 0}, {4, -1}, {5, 2}};
        //        T y_exp[5]      = {{9, 5}, {5, 3}, {16, 17}, {20, 12}, {21, 46}};
        T y[5]      = {{0, 0}};
        T y_gold[5] = {{0, 0}};

        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_mat_descr  descr;
        aoclsparse_matrix     A;
        aoclsparse_int        csr_row_ptr[] = {0, 2, 4, 7, 8, 11};
        aoclsparse_int        csr_col_ind[] = {0, 4, 1, 2, 1, 2, 4, 3, 0, 2, 4};
        T                     csr_val[]     = {{1, 1},
                                               {2, 0},
                                               {2, 2},
                                               {1, -1},
                                               {1, -1},
                                               {3, 3},
                                               {2, 3},
                                               {4, 4},
                                               {2, 0},
                                               {2, 3},
                                               {5, 5}};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        EXPECT_EQ(aoclsparse_mv<T>(op, &alpha, A, descr, x, &beta, y), aoclsparse_status_success);

        const std::complex<double> alphap = *reinterpret_cast<const std::complex<double> *>(&alpha);
        const std::complex<double> betap  = *reinterpret_cast<const std::complex<double> *>(&beta);

        EXPECT_EQ(ref_csrmvgen(op,
                               alphap,
                               M,
                               N,
                               (std::complex<double> *)csr_val,
                               csr_col_ind,
                               csr_row_ptr,
                               base,
                               (std::complex<double> *)x,
                               betap,
                               (std::complex<double> *)y_gold),
                  aoclsparse_status_success);

        EXPECT_COMPLEX_ARR_NEAR(
            M, ((std::complex<double> *)y), ((std::complex<double> *)y_gold), abserr);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    // test success cases
    template <typename T>
    void mv_cmplx_success(aoclsparse_int         m_a,
                          aoclsparse_int         n_a,
                          aoclsparse_int         nnz_a,
                          aoclsparse_index_base  b_a,
                          aoclsparse_operation   op_a,
                          aoclsparse_int         id       = 0,
                          aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_general,
                          aoclsparse_fill_mode   fill     = aoclsparse_fill_mode_lower,
                          aoclsparse_diag_type   diag     = aoclsparse_diag_type_non_unit)
    {
        tolerance_t<T>  abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());
        CBLAS_ORDER     layout = CblasRowMajor;
        CBLAS_TRANSPOSE trans  = CblasNoTrans;
        aoclsparse_int  ydim;
        if(op_a == op_t)
        {
            trans = CblasTrans;
        }
        else if(op_a == op_h)
        {
            trans = CblasConjTrans;
        }
        else if(op_a == op_n)
        {
            trans = CblasNoTrans;
        }

        aoclsparse_seedrand();
        std::vector<T> y, y_ref, x;
        std::vector<T> z(m_a), z_t(n_a);
        T              alpha = {1, 0}, beta = {0, 0};
        if(id == 1)
        {
            alpha = {1, 1};
            beta  = {-1, 2};
        }
        else if(id == 2)
        {
            alpha = {1, -2};
            beta  = {0, 0};
        }
        if(op_a == aoclsparse_operation_none)
        {
            x = z_t;
            y = z;
            aoclsparse_init<T>(x, 1, n_a, 1);
            aoclsparse_init<T>(y, 1, m_a, 1);
            y_ref = y;
            ydim  = m_a;
        }
        else
        {
            x = z;
            y = z_t;
            aoclsparse_init<T>(x, 1, m_a, 1);
            aoclsparse_init<T>(y, 1, n_a, 1);
            y_ref = y;
            ydim  = n_a;
        }
        aoclsparse_seedrand();
        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        std::vector<T>              dense_A(m_a * n_a);
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
        ASSERT_EQ(aoclsparse_set_mat_type(descrA, mat_type), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descrA, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descrA, diag), aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_csr2dense(m_a,
                                       n_a,
                                       descrA,
                                       val_a.data(),
                                       row_ptr_a.data(),
                                       col_ind_a.data(),
                                       dense_A.data(),
                                       n_a,
                                       aoclsparse_order_row),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, descrA, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        blis::gemv(layout,
                   trans,
                   (int64_t)m_a,
                   (int64_t)n_a,
                   *reinterpret_cast<const std::complex<double> *>(&alpha),
                   (std::complex<double> *)dense_A.data(),
                   (int64_t)n_a,
                   (std::complex<double> *)x.data(),
                   1,
                   *reinterpret_cast<const std::complex<double> *>(&beta),
                   (std::complex<double> *)y_ref.data(),
                   1);

        EXPECT_COMPLEX_ARR_NEAR(ydim,
                                ((std::complex<double> *)y.data()),
                                ((std::complex<double> *)y_ref.data()),
                                abserr);

        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
    }

    // test failures cases
    template <typename T>
    void mv_cmplx_failures(aoclsparse_int         m_a,
                           aoclsparse_int         n_a,
                           aoclsparse_int         nnz_a,
                           aoclsparse_index_base  b_a,
                           aoclsparse_operation   op_a,
                           aoclsparse_int         id       = 0,
                           aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_general,
                           aoclsparse_fill_mode   fill     = aoclsparse_fill_mode_lower,
                           aoclsparse_diag_type   diag     = aoclsparse_diag_type_non_unit)
    {
        aoclsparse_seedrand();
        std::vector<T> y, x;
        std::vector<T> z(m_a), z_t(n_a);
        T              alpha = {1, 0}, beta = {0, 0};
        if(op_a == aoclsparse_operation_none)
        {
            x = z_t;
            y = z;
            aoclsparse_init<T>(x, 1, n_a, 1);
            aoclsparse_init<T>(y, 1, m_a, 1);
        }
        else
        {
            x = z;
            y = z_t;
            aoclsparse_init<T>(x, 1, m_a, 1);
            aoclsparse_init<T>(y, 1, n_a, 1);
        }
        aoclsparse_seedrand();
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
        ASSERT_EQ(aoclsparse_set_mat_type(descrA, mat_type), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descrA, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descrA, diag), aoclsparse_status_success);

        // test null pointers
        switch(id)
        {
        case 0: // test null cases
            EXPECT_EQ(aoclsparse_mv<T>(op_a, nullptr, A, descrA, x.data(), &beta, y.data()),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, nullptr, x.data(), &beta, y.data()),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, descrA, nullptr, &beta, y.data()),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, descrA, x.data(), nullptr, y.data()),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, descrA, x.data(), &beta, nullptr),
                      aoclsparse_status_invalid_pointer);
            break;
        case 1: // test invalid type / value cases
            ASSERT_EQ(aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_symmetric),
                      aoclsparse_status_success);
            A->m = A->n + 1;
            EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, descrA, x.data(), &beta, y.data()),
                      aoclsparse_status_invalid_size);
            A->m            = A->n;
            A->input_format = aoclsparse_ell_mat;
            EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, descrA, x.data(), &beta, y.data()),
                      aoclsparse_status_not_implemented);
            A->input_format  = aoclsparse_csr_mat;
            A->mats[0]->base = (aoclsparse_index_base)2;
            EXPECT_EQ(aoclsparse_mv<T>(op_a, &alpha, A, descrA, x.data(), &beta, y.data()),
                      aoclsparse_status_invalid_value);

            break;
        default:
            break;
        }
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
    }

    template <typename T>
    void test_optmv_empty_rows()
    {
        aoclsparse_operation trans = aoclsparse_operation_transpose;
        aoclsparse_int       M = 5, N = 5, NNZ = 1;
        T                    alpha = 1.0;
        T                    beta  = 0.0;
        // Initialise vectors
        T                     x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        T                     y[5] = {0.0};
        aoclsparse_mat_descr  descr;
        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_matrix     A;

        /*      Matrix A
            0	0	0	0	0
            0	0	0	0	0
            0	0	1	0	0
            0	0	0	0	0
            0	0	0	0	0
        */
        aoclsparse_int csr_row_ptr[] = {0, 0, 0, 1, 1, 1};
        aoclsparse_int csr_col_ind[] = {2};
        T              csr_val[]     = {1};
        T              y_exp[]       = {0, 0, 3, 0, 0};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(trans, &alpha, A, descr, x, &beta, y),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_exp, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    // Test CSC Matrix Vector multiplication with Transpose operation of triangular matrix type
    // Provide a CSC matrix and test with transpose and fill mode switching.
    // Expectation: Should correctly flip fill mode and produce correct results.
    template <typename T>
    void test_mv_csc_success()
    {
        aoclsparse_int M = 5, N = 4, NNZ = 8;
        T              alpha = 1.0;
        T              beta  = 0.0;
        // Initialise vectors
        std::vector<T> x(N, 0.0), y, y_ref;

        x[0] = 1.0;
        x[1] = 2.0;
        x[2] = 3.0;
        x[3] = 4.0;

        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_index_base base = aoclsparse_index_base_zero;

        aoclsparse_int    csr_row_ptr[6] = {0, 2, 3, 4, 7, 8};
        aoclsparse_int    csr_col_ind[8] = {0, 3, 1, 2, 1, 2, 3, 1};
        T                 csr_val[8]     = {1, 2, 3, 4, 5, 6, 7, 8};
        aoclsparse_matrix A;
        aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val);

        // Convert CSR to CSC
        std::vector<aoclsparse_int> csc_col_ptr(N + 1);
        std::vector<aoclsparse_int> csc_row_ind(NNZ);
        std::vector<T>              csc_val(NNZ);

        ASSERT_EQ(aoclsparse_csr2csc(M,
                                     N,
                                     NNZ,
                                     descr,
                                     base,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     csc_row_ind.data(),
                                     csc_col_ptr.data(),
                                     csc_val.data()),
                  aoclsparse_status_success);

        aoclsparse_matrix A_csc;
        ASSERT_EQ(
            aoclsparse_create_csc<T>(
                &A_csc, base, M, N, NNZ, csc_col_ptr.data(), csc_row_ind.data(), csc_val.data()),
            aoclsparse_status_success);

        // Set matrix type and fill mode
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);

        // Reference result using CSR (y_ref and y are size M)
        y.assign(M, 0.0);
        y_ref.assign(M, 0.0);
        EXPECT_EQ(aoclsparse_mv<T>(
                      aoclsparse_operation_none, &alpha, A, descr, x.data(), &beta, y_ref.data()),
                  aoclsparse_status_success);

        // Result using CSC
        y.assign(M, 0.0);
        EXPECT_EQ(aoclsparse_mv<T>(
                      aoclsparse_operation_none, &alpha, A_csc, descr, x.data(), &beta, y.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(M, y.data(), y_ref.data());

        // Now test transpose operation
        // For transpose, input x should be size M, output y should be size N
        std::vector<T> x_tr(M, 0.0), y_tr(N, 0.0), y_ref_tr(N, 0.0);
        x_tr[0] = 1.0;
        x_tr[1] = 2.0;
        x_tr[2] = 3.0;
        x_tr[3] = 4.0;
        x_tr[4] = 5.0;

        EXPECT_EQ(aoclsparse_mv<T>(aoclsparse_operation_transpose,
                                   &alpha,
                                   A,
                                   descr,
                                   x_tr.data(),
                                   &beta,
                                   y_ref_tr.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_mv<T>(aoclsparse_operation_transpose,
                                   &alpha,
                                   A_csc,
                                   descr,
                                   x_tr.data(),
                                   &beta,
                                   y_tr.data()),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(N, y_tr.data(), y_ref_tr.data());

        aoclsparse_destroy_mat_descr(descr);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A_csc), aoclsparse_status_success);
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
    TEST(mv, BaseOneDouble)
    {
        test_mv_base_indexing<double>();
    }
    TEST(mv, BaseOneFloat)
    {
        test_mv_base_indexing<float>();
    }
    TEST(mv, NotImplDouble)
    {
        test_mv_not_implemented<double>();
    }
    TEST(mv, NotImplFloat)
    {
        test_mv_not_implemented<float>();
    }
    TEST(mv, InvalidKID)
    {
        test_invalid_kid<aoclsparse_double_complex>();
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
        //GTEST_SKIP() << "Skipping since implementation does not exist";
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
    TEST(mv, SymmDouble)
    {
        test_mv_symm<double>("SuccessSymmetric-Double-ZeroBase", aoclsparse_index_base_zero);
        test_mv_symm<double>("SuccessSymmetric-Double-OneBase", aoclsparse_index_base_one);
    }
    TEST(mv, SymmFloat)
    {
        test_mv_symm<float>("SuccessSymmetric-Float-ZeroBase", aoclsparse_index_base_zero);
        test_mv_symm<float>("SuccessSymmetric-Float-OneBase", aoclsparse_index_base_one);
    }
    TEST(mv, SymmComplexDouble)
    {
        test_mv_symm_complex<aoclsparse_double_complex>("SuccessSymmetric-ComplexDouble-ZeroBase",
                                                        aoclsparse_index_base_zero);
        test_mv_symm_complex<aoclsparse_double_complex>("SuccessSymmetric-ComplexDouble-OneBase",
                                                        aoclsparse_index_base_one);
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
    TEST(mv, BaseIndexismatchDouble)
    {
        test_mv_base_index_mismatch<double>();
    }
    TEST(mv, BaseIndexismatchFloat)
    {
        test_mv_base_index_mismatch<float>();
    }
    TEST(mv, CmplxSuccessSq5x5AllOpN)
    {
        //mv_cmplx_success_sq5x5_all_ops<std::complex<double>>();
        //mv_cmplx_success_sq5x5_all_ops<aoclsparse_float_complex>();
        mv_cmplx_success_sq5x5_all_ops<aoclsparse_double_complex>(op_n);
    }
    TEST(mv, CmplxSuccessSq5x5AllOpT)
    {
        mv_cmplx_success_sq5x5_all_ops<aoclsparse_double_complex>(op_t);
    }
    TEST(mv, CmplxSuccessSq5x5AllOpH)
    {
        mv_cmplx_success_sq5x5_all_ops<aoclsparse_double_complex>(op_h);
    }
    TEST(mv, CmplxGSuccessOpN)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 2, 3, zero, op_n, 0);
        mv_cmplx_success<aoclsparse_double_complex>(10, 12, 40, zero, op_n, 0);
        mv_cmplx_success<aoclsparse_double_complex>(8, 1, 4, zero, op_n, 0);
        mv_cmplx_success<aoclsparse_double_complex>(1, 9, 9, zero, op_n, 0);
        mv_cmplx_success<aoclsparse_double_complex>(7, 9, 19, one, op_n, 0);
        mv_cmplx_success<aoclsparse_double_complex>(7, 9, 29, one, op_n, 1);
        mv_cmplx_success<aoclsparse_double_complex>(11, 9, 39, one, op_n, 2);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 8, one, op_n, 0);
        mv_cmplx_success<aoclsparse_double_complex>(7, 7, 38, zero, op_n, 0);
        mv_cmplx_success<aoclsparse_double_complex>(8, 10, 49, one, op_n, 0);
        mv_cmplx_success<aoclsparse_double_complex>(10, 7, 45, zero, op_n, 0);
    }

    TEST(mv, CmplxGSuccessOpT)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 2, 3, zero, op_t, 0);
        mv_cmplx_success<aoclsparse_double_complex>(10, 12, 40, zero, op_t, 0);
        mv_cmplx_success<aoclsparse_double_complex>(8, 1, 4, zero, op_t, 0);
        mv_cmplx_success<aoclsparse_double_complex>(1, 9, 9, zero, op_t, 0);
        mv_cmplx_success<aoclsparse_double_complex>(7, 9, 19, one, op_t, 0);
        mv_cmplx_success<aoclsparse_double_complex>(7, 9, 29, one, op_t, 1);
        mv_cmplx_success<aoclsparse_double_complex>(11, 9, 39, one, op_t, 2);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 8, one, op_t, 0);
    }
    TEST(mv, CmplxGSuccessOpH)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 2, 3, zero, op_h, 0);
        mv_cmplx_success<aoclsparse_double_complex>(10, 12, 40, zero, op_h, 0);
        mv_cmplx_success<aoclsparse_double_complex>(8, 1, 4, zero, op_h, 0);
        mv_cmplx_success<aoclsparse_double_complex>(1, 9, 9, zero, op_h, 0);
        mv_cmplx_success<aoclsparse_double_complex>(7, 9, 19, one, op_h, 0);
        mv_cmplx_success<aoclsparse_double_complex>(7, 9, 29, one, op_h, 1);
        mv_cmplx_success<aoclsparse_double_complex>(11, 9, 39, one, op_h, 2);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 8, one, op_h, 0);
    }

    // ToDo: One-base testing missing as csr2dense doesn't support it at present
    TEST(mv, CmplxSymSuccessOpN)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_n, 0, mat_s, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_n, 0, mat_s, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_n, 0, mat_s, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_n, 0, mat_s, fl_up, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 20, zero, op_n, 0, mat_s, fl_up, diag_u);
    }
    TEST(mv, CmplxSymSuccessOpT)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_t, 0, mat_s, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_t, 0, mat_s, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_t, 0, mat_s, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_t, 0, mat_s, fl_up, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 20, zero, op_t, 0, mat_s, fl_up, diag_u);
    }
    TEST(mv, CmplxSymSuccessOpH)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_h, 0, mat_s, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_h, 0, mat_s, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_h, 0, mat_s, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_h, 0, mat_s, fl_up, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 20, zero, op_h, 0, mat_s, fl_up, diag_u);
    }
    TEST(mv, CmplxTriSuccessOpN)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_n, 0, mat_t, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_n, 0, mat_t, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_n, 0, mat_t, fl_lo, diag_z);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_n, 0, mat_t, fl_lo, diag_u);

        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_n, 0, mat_t, fl_up, diag_nu);
    }
    TEST(mv, CmplxTriSuccessOpT)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_t, 0, mat_t, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_t, 0, mat_t, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_t, 0, mat_t, fl_up, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_t, 0, mat_t, fl_lo, diag_u);
    }
    TEST(mv, CmplxTriSuccessOpH)
    {
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_h, 0, mat_t, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_h, 0, mat_t, fl_lo, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_h, 0, mat_t, fl_up, diag_nu);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_h, 0, mat_t, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_h, 0, mat_t, fl_lo, diag_z);
    }

    TEST(mv, CmplxHermSuccessOpN)
    {
        // Need to pass only unit diagonals as a hermitian matrix needs real diagonals
        // Our random generation doesn't support it
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_n, 0, mat_h, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_n, 0, mat_h, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_n, 0, mat_h, fl_up, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_n, 0, mat_h, fl_up, diag_z);

        // The below is not correct, but still passes as the op in non-transpose
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_n, 0, mat_h, fl_up, diag_nu);

        // Test cases when alpha != (1,0) and beta != (0,0)
        mv_cmplx_success<aoclsparse_double_complex>(
            11, 11, 43, zero, op_n, 1, mat_h, fl_up, diag_z);
        mv_cmplx_success<aoclsparse_double_complex>(
            13, 13, 71, zero, op_n, 2, mat_h, fl_lo, diag_z);
    }

    TEST(mv, CmplxHermSuccessOpH)
    {
        // Need to pass only unit diagonals as a hermitian matrix needs real diagonals
        // Our random generation doesn't support it
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_h, 0, mat_h, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_h, 0, mat_h, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_h, 0, mat_h, fl_up, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_h, 0, mat_h, fl_up, diag_z);
        // the below test fails as expected as diag is complex
        //        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_h, 0, mat_h, fl_up, diag_nu);
    }

    TEST(mv, CmplxNull)
    {
        mv_cmplx_failures<aoclsparse_double_complex>(8, 8, 27, zero, op_h, 0, mat_h, fl_up, diag_u);
    }

    TEST(mv, CmplxInvalid)
    {
        mv_cmplx_failures<aoclsparse_double_complex>(8, 8, 27, zero, op_h, 1, mat_h, fl_up, diag_u);
    }
    TEST(mv, SuccessDoubleCSC)
    {
        test_mv_csc_success<double>();
    }
    TEST(mv, SuccessFloatCSC)
    {
        test_mv_csc_success<float>();
    }

    /* Transpose operation for a hermitian matrix is not implemented
 * The below tests will fail
 */
    /*
    TEST(mv, CmplxHermSuccessOpT)
    {
        // Need to pass only unit diagonals as a hermitian matrix needs real diagonals
        // Our random generation doesn't support it
        mv_cmplx_success<aoclsparse_double_complex>(3, 3, 3, zero, op_t, 0, mat_h, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(
            10, 10, 37, zero, op_h, 0, mat_h, fl_lo, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_t, 0, mat_h, fl_up, diag_u);
        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_t, 0, mat_h, fl_up, diag_z);
        // the below test fails as expected as diag is complex
//        mv_cmplx_success<aoclsparse_double_complex>(8, 8, 27, zero, op_t, 0, mat_h, fl_up, diag_nu);
    }
*/

    TEST(mv, TriStrictDouble)
    {
        test_mv_trianglular_strict<double>();
    }
    TEST(mv, TriTransposeStrictDouble)
    {
        test_mvt_trianglular_strict<double>();
    }

    TEST(mv, Tri_Sq_LoStr_DoubleSuccess)
    {
        aoclsparse_operation   op       = aoclsparse_operation_none;
        aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_triangular;
        aoclsparse_fill_mode   fill     = aoclsparse_fill_mode_lower;
        aoclsparse_diag_type   diag     = aoclsparse_diag_type_zero;
        aoclsparse_index_base  b_a      = aoclsparse_index_base_zero;

        // inputs: m_a, n_a, nnz, ...
        test_mvtrg_success<double>(10, 10, 30, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(1, 1, 1, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(5, 5, 5, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(8, 8, 5, op, mat_type, fill, diag, b_a);
    }
    TEST(mv, Tri_Sq_UpStr_DoubleSuccess)
    {
        aoclsparse_operation   op       = aoclsparse_operation_none;
        aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_triangular;
        aoclsparse_fill_mode   fill     = aoclsparse_fill_mode_upper;
        aoclsparse_diag_type   diag     = aoclsparse_diag_type_zero;
        aoclsparse_index_base  b_a      = aoclsparse_index_base_zero;

        // inputs: m_a, n_a, nnz, ...
        test_mvtrg_success<double>(10, 10, 30, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(1, 1, 1, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(5, 5, 5, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(8, 8, 5, op, mat_type, fill, diag, b_a);
    }
    TEST(mv, TriT_Sq_LoStr_DoubleSuccess)
    {
        aoclsparse_operation   op       = aoclsparse_operation_transpose;
        aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_triangular;
        aoclsparse_fill_mode   fill     = aoclsparse_fill_mode_lower;
        aoclsparse_diag_type   diag     = aoclsparse_diag_type_zero;
        aoclsparse_index_base  b_a      = aoclsparse_index_base_zero;

        // inputs: m_a, n_a, nnz, ...
        test_mvtrg_success<double>(10, 10, 30, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(1, 1, 1, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(5, 5, 5, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(8, 8, 5, op, mat_type, fill, diag, b_a);
    }
    TEST(mv, TriT_Sq_UpStr_DoubleSuccess)
    {
        aoclsparse_operation   op       = aoclsparse_operation_transpose;
        aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_triangular;
        aoclsparse_fill_mode   fill     = aoclsparse_fill_mode_upper;
        aoclsparse_diag_type   diag     = aoclsparse_diag_type_zero;
        aoclsparse_index_base  b_a      = aoclsparse_index_base_zero;

        // inputs: m_a, n_a, nnz, ...
        test_mvtrg_success<double>(10, 10, 30, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(1, 1, 1, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(5, 5, 5, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(8, 8, 5, op, mat_type, fill, diag, b_a);
    }
    TEST(mv, Tri_NSq_LoStr_DoubleSuccess)
    {
        aoclsparse_operation   op       = aoclsparse_operation_none;
        aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_triangular;
        aoclsparse_fill_mode   fill     = aoclsparse_fill_mode_lower;
        aoclsparse_diag_type   diag     = aoclsparse_diag_type_zero;
        aoclsparse_index_base  b_a      = aoclsparse_index_base_zero;

        // inputs: m_a, n_a, nnz, ...
        test_mvtrg_success<double>(7, 10, 30, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(1, 4, 3, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(12, 24, 103, op, mat_type, fill, diag, b_a);
        // The below tests fail
        // test_mvtrg_success<double>(5, 3, 5, op, mat_type, fill, diag, b_a);
        //test_mvtrg_success<double>(11, 8, 35, op, mat_type, fill, diag, b_a);
    }
    /* all tests fail
    TEST(mv, Tri_NSq_UpStr_DoubleSuccess)
    {
        aoclsparse_operation   op       = aoclsparse_operation_none;
        aoclsparse_matrix_type mat_type = aoclsparse_matrix_type_triangular;
        aoclsparse_fill_mode   fill     = aoclsparse_fill_mode_upper;
        aoclsparse_diag_type   diag     = aoclsparse_diag_type_zero;
        aoclsparse_index_base  b_a      = aoclsparse_index_base_zero;

        // inputs: m_a, n_a, nnz, ...
        test_mvtrg_success<double>(7, 10, 30, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(1, 4, 3, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(5, 3, 5, op, mat_type, fill, diag, b_a);
        test_mvtrg_success<double>(11, 8, 35, op, mat_type, fill, diag, b_a);
    }
    */
    // testing optimize functionality (in the context of SpMV) when
    // some of the matrix rows are empty
    TEST(mv, OptEmptyRows)
    {
        test_optmv_empty_rows<double>();
    }
    /*---------------------------------------------------------------------------------*/
    //                          Extreme value testing for spmv
    /*---------------------------------------------------------------------------------*/
    template <typename T>
    void mv_extreme_test_driver(linear_system_id  id,
                                aoclsparse_int    mtype,
                                aoclsparse_int    fmode,
                                aoclsparse_int    transp,
                                aoclsparse_int    op1,
                                aoclsparse_int    op2,
                                aoclsparse_int    ou_range,
                                aoclsparse_status spmv_ext_status = aoclsparse_status_success)
    {
        aoclsparse_status    status;
        std::string          title;
        T                    alpha;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        std::vector<T>       b, y;
        std::vector<T>       x;
        std::vector<T>       xref;
        T                    xtol, beta;
        aoclsparse_operation trans;
        // permanent storage of matrix data
        std::vector<T>                 aval;
        std::vector<aoclsparse_int>    icola;
        std::vector<aoclsparse_int>    icrowa;
        std::array<aoclsparse_int, 10> iparm{0};
        std::array<T, 10>              dparm;
        aoclsparse_status              exp_status;
        aoclsparse_index_base          base = aoclsparse_index_base_zero;
        decltype(std::real(xtol))      tol;
        /*
         * Below parameters are inputs to create_linear_system(), which are needed to populate correct reference vectors
         * OUTPUT: iparm[3] should contain the type of result value
         * OUTPUT: dparm[0] = beta
         * INPUT: iparm[4] = type of special value in operand 1 in csr_val
         * INPUT: iparm[5] = type of special value in x[]
         * INPUT: iparm[6] = matrix type
         * INPUT: iparm[7] = fill mode
         * INPUT: iparm[8] = tranpose mode
         * INPUT: iparm[9] = overflow/underflow range
         */
        //output param: iparm[3] should contain the type of result value
        iparm[4] = op1;
        iparm[5] = op2;

        iparm[6] = mtype;
        iparm[7] = fmode;
        iparm[8] = transp;
        iparm[9] = ou_range;
        status   = create_linear_system<T>(id,
                                         title,
                                         trans,
                                         A,
                                         descr,
                                         base,
                                         alpha,
                                         b,
                                         x,
                                         xref,
                                         xtol,
                                         icrowa,
                                         icola,
                                         aval,
                                         iparm,
                                         dparm,
                                         exp_status);
        ASSERT_EQ(status, aoclsparse_status_success)
            << "Error: could not find linear system id " << id << "!";
        beta                   = dparm[0];
        const aoclsparse_int n = A->n;

#if(VERBOSE > 0)
        std::string oplabel{""};
        switch(trans)
        {
        case aoclsparse_operation_none:
            oplabel = "None";
            break;
        case aoclsparse_operation_transpose:
            oplabel = "Transpose";
            break;
        case aoclsparse_operation_conjugate_transpose:
            oplabel = "Hermitian Transpose";
            break;
        }
        std::string dtype = "unknown";
        if(typeid(T) == typeid(double))
        {
            dtype = "double";
        }
        else if(typeid(T) == typeid(float))
        {
            dtype = "float";
        }
        if(typeid(T) == typeid(std::complex<double>))
        {
            dtype = "cdouble";
        }
        else if(typeid(T) == typeid(std::complex<float>))
        {
            dtype = "cfloat";
        }
        const bool unit = descr->diag_type == aoclsparse_diag_type_unit;
        std::cout << "Problem id: " << id << " \"" << title << "\"" << std::endl;
        std::cout << "Configuration: <" << dtype << "> unit=" << (unit ? "Unit" : "Non-unit")
                  << " op=" << oplabel << ">" << std::endl;
#endif
        //validate symgs output
        y   = std::move(xref);
        tol = std::real(xtol); // get the tolerance.
        if(tol <= 0)
            tol = 10;
        tolerance_t<T> abs_error = expected_precision<decltype(tol)>(tol);
        status                   = aoclsparse_mv<T>(trans, &alpha, A, descr, &x[0], &beta, &y[0]);
        ASSERT_EQ(status, spmv_ext_status)
            << "Test failed with unexpected return from aoclsparse_mv";

        if(status == aoclsparse_status_success)
        {
            EXPECT_ARR_MATCH(T, n, &b[0], &y[0], abs_error, abs_error);
        }
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
#define LT 0
#define UT 1
#define SLT 2 //Strictly Lower Triangle
#define SUT 3 //Strictly Upper Triangle

#define GEN 0
#define SYM 1
#define HERMIT 2
#define TRIANG 3

#define NT 0
#define TRANS 1

#define NAN_INF_TEST -1
#define FLOW_EDGE_WITHIN 0
#define FLOW_OUTOF 1

    typedef struct
    {
        linear_system_id id;
        std::string      testname;
        aoclsparse_int   matrix_type;
        aoclsparse_int   fill_mode;
        aoclsparse_int   transp;
        aoclsparse_int   op1;
        aoclsparse_int   op2;
        aoclsparse_int   ou_range;
    } extreme_value_list_t;
#undef ADD_TEST
#define ADD_TEST(ID, MTYPE, FMODE, TRANSP, OPERAND1, OPERAND2, FLOW_RANGE)                     \
    {                                                                                          \
        ID, #ID "/" #MTYPE "/" #FMODE "/" #TRANSP "/" #OPERAND1 "/" #OPERAND2 "/" #FLOW_RANGE, \
            MTYPE, FMODE, TRANSP, OPERAND1, OPERAND2, FLOW_RANGE                               \
    }

    extreme_value_list_t extreme_value_list[] = {
        //General matrices with variations in transpose/Non-transpose and 3 cases of extreme values
        //  1. NaN * any_number = NaN
        ADD_TEST(EXT_G5, TRIANG, SUT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_G5, TRIANG, SLT, TRANS, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_G5, GEN, LT, TRANS, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(
            EXT_G5_B0, GEN, LT, TRANS, ET_NAN, ET_NUM, NAN_INF_TEST), //BETA ZERO CASE for Transpose

        ADD_TEST(EXT_H5, HERMIT, LT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5, HERMIT, UT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5, TRIANG, SUT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5, TRIANG, SLT, TRANS, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5_B0, HERMIT, LT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5_B0, HERMIT, UT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        //  2. Inf * any_number = Inf
        ADD_TEST(EXT_G5, TRIANG, SUT, NT, ET_INF, ET_NUM, NAN_INF_TEST),

        ADD_TEST(EXT_H5, HERMIT, LT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5, HERMIT, UT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5, TRIANG, SUT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5_B0, HERMIT, LT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_H5_B0, HERMIT, UT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        //  3. Inf * zero = NaN
        ADD_TEST(EXT_G5, TRIANG, SUT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_G5, TRIANG, SLT, TRANS, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_G5, GEN, LT, TRANS, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_G5_B0, GEN, LT, TRANS, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_H5, HERMIT, LT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_H5, HERMIT, UT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_H5, TRIANG, SLT, TRANS, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_H5_B0, HERMIT, LT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_H5_B0, HERMIT, UT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        //Symmetric matrices with variations in transpose/Non-transpose, lower/upper
        // and 3 cases of extreme values
        //  1. NaN * any_number = NaN
        ADD_TEST(EXT_S5, SYM, LT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, LT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, UT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, UT, NT, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, LT, TRANS, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, LT, TRANS, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, UT, TRANS, ET_NAN, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, UT, TRANS, ET_NAN, ET_NUM, NAN_INF_TEST),
        //  2. Inf * any_number = Inf
        ADD_TEST(EXT_S5, SYM, LT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, LT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, UT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, UT, NT, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, LT, TRANS, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, LT, TRANS, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, UT, TRANS, ET_INF, ET_NUM, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, UT, TRANS, ET_INF, ET_NUM, NAN_INF_TEST),
        //  3. Inf * zero = NaN
        ADD_TEST(EXT_S5, SYM, LT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, LT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, UT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, UT, NT, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, LT, TRANS, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, LT, TRANS, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_S5, SYM, UT, TRANS, ET_INF, ET_ZERO, NAN_INF_TEST),
        ADD_TEST(EXT_S5_B0, SYM, UT, TRANS, ET_INF, ET_ZERO, NAN_INF_TEST),
        //Positive Overflow for 2 cases:
        // edge of DBL_MAX/FLT_MAX but within bounds and well outside DBL_MAX/FLT_MAX bound
        ADD_TEST(EXT_G5, GEN, LT, NT, ET_POVRFLOW, ET_POVRFLOW, FLOW_EDGE_WITHIN),
        ADD_TEST(EXT_G5, GEN, LT, NT, ET_POVRFLOW, ET_POVRFLOW, FLOW_OUTOF),
        //Negative Overflow for 2 cases:
        // edge of DBL_MAX/FLT_MAX but within bounds and well outside DBL_MAX/FLT_MAX bound
        ADD_TEST(EXT_G5, GEN, LT, NT, ET_NOVRFLOW, ET_NOVRFLOW, FLOW_EDGE_WITHIN),
        ADD_TEST(EXT_G5, GEN, LT, NT, ET_NOVRFLOW, ET_NOVRFLOW, FLOW_OUTOF),
        //Positive Underflow for 2 cases:
        // edge of DBL_MAX/FLT_MAX but within bounds and well outside DBL_MAX/FLT_MAX bound
        ADD_TEST(EXT_NSYMM_5, GEN, LT, NT, ET_PUNDRFLOW, ET_PUNDRFLOW, FLOW_EDGE_WITHIN),
        ADD_TEST(EXT_NSYMM_5, GEN, LT, NT, ET_PUNDRFLOW, ET_PUNDRFLOW, FLOW_OUTOF),
        //Positive Underflow for 2 cases:
        // edge of DBL_MAX/FLT_MAX but within bounds and well outside DBL_MAX/FLT_MAX bound
        ADD_TEST(EXT_NSYMM_5, GEN, LT, NT, ET_NUNDRFLOW, ET_NUNDRFLOW, FLOW_EDGE_WITHIN),
        ADD_TEST(EXT_NSYMM_5, GEN, LT, NT, ET_NUNDRFLOW, ET_NUNDRFLOW, FLOW_OUTOF),
    };

    void PrintTo(const extreme_value_list_t &param, ::std::ostream *os)
    {
        *os << param.testname;
    }
    class PosDouble : public testing::TestWithParam<extreme_value_list_t>
    {
    };
    TEST_P(PosDouble, spmv)
    {
        const linear_system_id id     = GetParam().id;
        aoclsparse_int         mtype  = GetParam().matrix_type;
        const aoclsparse_int   fmode  = GetParam().fill_mode;
        aoclsparse_int         transp = GetParam().transp;

        aoclsparse_int op1      = GetParam().op1;
        aoclsparse_int op2      = GetParam().op2;
        aoclsparse_int ou_range = GetParam().ou_range;

        const aoclsparse_status spmv_ext_status = aoclsparse_status_success;
        //database populates real only data when ldouble or float variants are called, so
        //force matrix type to symmetric since the data that is fed is symmetric
        if(mtype == aoclsparse_matrix_type_hermitian)
        {
            mtype = SYM;
        }
        if(transp == NT)
        {
            transp = (aoclsparse_int)aoclsparse_operation_none;
        }
        else if(transp == TRANS)
        {
            transp = (aoclsparse_int)aoclsparse_operation_transpose;
        }
#if(VERBOSE > 0)
        std::cout << "Pos/Double/ExtremeValues test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif

        mv_extreme_test_driver<double>(
            id, mtype, fmode, transp, op1, op2, ou_range, spmv_ext_status);
    }
    INSTANTIATE_TEST_SUITE_P(ExtSuite, PosDouble, ::testing::ValuesIn(extreme_value_list));

    class PosFloat : public testing::TestWithParam<extreme_value_list_t>
    {
    };
    TEST_P(PosFloat, spmv)
    {
        const linear_system_id id       = GetParam().id;
        aoclsparse_int         mtype    = GetParam().matrix_type;
        const aoclsparse_int   fmode    = GetParam().fill_mode;
        aoclsparse_int         transp   = GetParam().transp;
        aoclsparse_int         op1      = GetParam().op1;
        aoclsparse_int         op2      = GetParam().op2;
        aoclsparse_int         ou_range = GetParam().ou_range;

        const aoclsparse_status spmv_ext_status = aoclsparse_status_success;

        //database populates real only data when Double or Float variants are called, so
        //force matrix type to symmetric since the data thatis fed is symmetric
        if(mtype == aoclsparse_matrix_type_hermitian)
        {
            mtype = SYM;
        }

        if(transp == NT)
        {
            transp = (aoclsparse_int)aoclsparse_operation_none;
        }
        else if(transp == TRANS)
        {
            transp = (aoclsparse_int)aoclsparse_operation_transpose;
        }
#if(VERBOSE > 0)
        std::cout << "Pos/Float/ExtremeValues test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        mv_extreme_test_driver<float>(
            id, mtype, fmode, transp, op1, op2, ou_range, spmv_ext_status);
    }
    INSTANTIATE_TEST_SUITE_P(ExtSuite, PosFloat, ::testing::ValuesIn(extreme_value_list));

    class PosCplxDouble : public testing::TestWithParam<extreme_value_list_t>
    {
    };
    TEST_P(PosCplxDouble, spmv)
    {
        const linear_system_id id     = GetParam().id;
        aoclsparse_int         mtype  = GetParam().matrix_type;
        const aoclsparse_int   fmode  = GetParam().fill_mode;
        aoclsparse_int         transp = GetParam().transp;

        aoclsparse_int op1      = GetParam().op1;
        aoclsparse_int op2      = GetParam().op2;
        aoclsparse_int ou_range = GetParam().ou_range;

        const aoclsparse_status spmv_ext_status = aoclsparse_status_success;

        if(transp == NT)
        {
            transp = (aoclsparse_int)aoclsparse_operation_none;
        }
        else if(transp == TRANS)
        {
            //check for complex-conjugate tranpose, if matrix type is hermitian, else simple transpose
            if(mtype == aoclsparse_matrix_type_hermitian)
            {
                transp = (aoclsparse_int)aoclsparse_operation_conjugate_transpose;
            }
            else
            {
                transp = (aoclsparse_int)aoclsparse_operation_transpose;
            }
        }
#if(VERBOSE > 0)
        std::cout << "Pos/CplxDouble/ExtremeValues test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        /*
            exclude complex-double cases of Inf*number, overflow and underflow
            Reason: The order of multiplication of spmv between reference and library kernel matters to decide
            whether the result is (inf + i . NaN) or (inf + i . inf)
        */
        if(op1 != ET_INF && op1 != ET_POVRFLOW && op1 != ET_PUNDRFLOW && op1 != ET_NOVRFLOW
           && op1 != ET_NUNDRFLOW)
        {
            mv_extreme_test_driver<std::complex<double>>(
                id, mtype, fmode, transp, op1, op2, ou_range, spmv_ext_status);
        }
        else
        {
            std::cout << "complex cases of Inf*number, overflow and underflow are disabled due to "
                         "varying spmv "
                         "result involving infinity and a complex number.\n";
        }
    }
    INSTANTIATE_TEST_SUITE_P(ExtSuite, PosCplxDouble, ::testing::ValuesIn(extreme_value_list));

    class PosCplxFloat : public testing::TestWithParam<extreme_value_list_t>
    {
    };
    TEST_P(PosCplxFloat, spmv)
    {
        const linear_system_id id       = GetParam().id;
        aoclsparse_int         mtype    = GetParam().matrix_type;
        const aoclsparse_int   fmode    = GetParam().fill_mode;
        aoclsparse_int         transp   = GetParam().transp;
        aoclsparse_int         op1      = GetParam().op1;
        aoclsparse_int         op2      = GetParam().op2;
        aoclsparse_int         ou_range = GetParam().ou_range;

        const aoclsparse_status spmv_ext_status = aoclsparse_status_success;

        if(transp == NT)
        {
            transp = (aoclsparse_int)aoclsparse_operation_none;
        }
        else if(transp == TRANS)
        {
            //check for complex-conjugate tranpose, if matrix type is hermitian, else simple transpose
            if(mtype == aoclsparse_matrix_type_hermitian)
            {
                transp = (aoclsparse_int)aoclsparse_operation_conjugate_transpose;
            }
            else
            {
                transp = (aoclsparse_int)aoclsparse_operation_transpose;
            }
        }
#if(VERBOSE > 0)
        std::cout << "Pos/CplxFloat/ExtremeValues test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        /*
            exclude complex-float cases of Inf*number, overflow and underflow
            Reason: The order of multiplication of spmv between reference and library kernel matters to decide
            whether the result is (inf + i . NaN) or (inf + i . inf)
        */
        if(op1 != ET_INF && op1 != ET_POVRFLOW && op1 != ET_PUNDRFLOW && op1 != ET_NOVRFLOW
           && op1 != ET_NUNDRFLOW)
        {
            mv_extreme_test_driver<std::complex<float>>(
                id, mtype, fmode, transp, op1, op2, ou_range, spmv_ext_status);
        }
        else
        {
            std::cout << "complex cases of Inf*number, overflow and underflow are disabled due to "
                         "varying spmv "
                         "result involving infinity and a complex number.\n";
        }
    }
    INSTANTIATE_TEST_SUITE_P(ExtSuite, PosCplxFloat, ::testing::ValuesIn(extreme_value_list));
    /*---------------------------------------------------------------------------------*/
    /*---------------------------------------------------------------------------------*/
} // namespace
