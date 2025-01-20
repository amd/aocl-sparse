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

/*
 * Unit-tests for Successive Over-Relaxation solver (aoclsparse_sorv)
 */
#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"

// Ignore compiler warning from BLIS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wunused-function"

#include "cblas.hh"

// Restore
#pragma GCC diagnostic pop

namespace
{
    template <typename T>
    void test_null_pointers()
    {
        aoclsparse_matrix    A         = nullptr;
        aoclsparse_mat_descr descr     = nullptr;
        aoclsparse_int       row_ptr[] = {0, 3, 7, 10, 13};
        aoclsparse_int       col_idx[] = {0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 0, 2, 3};
        T val[]          = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 8.0, 9.0, 4.0, -2.0, 1.0, -7.0, 5.0};
        T b[]            = {2.0, 21.0, -12.0, -6.0};
        T x[]            = {0.0, 0.0, 0.0, 0.0};
        T omega          = 0.5;
        T alpha          = 1.0;
        aoclsparse_int M = 4;
        aoclsparse_int N = 4;
        aoclsparse_int NNZ = 13;

        ASSERT_EQ(
            aoclsparse_create_csr(&A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val),
            aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, nullptr, omega, alpha, x, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, nullptr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, (T *)nullptr, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, (const T *)nullptr),
            aoclsparse_status_invalid_pointer);

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }
    TEST(Sorv, NullPointers)
    {
        test_null_pointers<float>();
        test_null_pointers<double>();
    }
    TEST(Sorv, WrongType)
    {
        aoclsparse_matrix    A         = nullptr;
        aoclsparse_mat_descr descr     = nullptr;
        aoclsparse_int       row_ptr[] = {0, 3, 7, 10, 13};
        aoclsparse_int       col_idx[] = {0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 0, 2, 3};
        double val[]     = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 8.0, 9.0, 4.0, -2.0, 1.0, -7.0, 5.0};
        float  b[]       = {2.0, 21.0, -12.0, -6.0};
        float  x[]       = {0.0, 0.0, 0.0, 0.0};
        float  omega     = 0.5;
        float  alpha     = 1.0;
        aoclsparse_int M = 4;
        aoclsparse_int N = 4;
        aoclsparse_int NNZ = 13;

        ASSERT_EQ(
            aoclsparse_create_csr(&A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val),
            aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        // checking for matrix val type (double) and function datatype (float) mismatch
        EXPECT_EQ(aoclsparse_ssorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_wrong_type);

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }
    TEST(Sorv, NegativeMatrixSize)
    {
        aoclsparse_matrix    A         = nullptr;
        aoclsparse_mat_descr descr     = nullptr;
        aoclsparse_int       row_ptr[] = {0, 3, 7, 10, 13};
        aoclsparse_int       col_idx[] = {0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 0, 2, 3};
        double val[]     = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 8.0, 9.0, 4.0, -2.0, 1.0, -7.0, 5.0};
        double b[]       = {2.0, 21.0, -12.0, -6.0};
        double x[]       = {0.0, 0.0, 0.0, 0.0};
        double omega     = 0.5;
        double alpha     = 1.0;
        aoclsparse_int M = 4;
        aoclsparse_int N = 4;
        aoclsparse_int NNZ = 13;

        ASSERT_EQ(
            aoclsparse_create_csr(&A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val),
            aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        A->m = -4;
        A->n = -4;
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_value);

        A->nnz = -4;
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_value);

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }
    TEST(Sorv, NotSquareMatrix)
    {
        aoclsparse_matrix    A         = nullptr;
        aoclsparse_mat_descr descr     = nullptr;
        aoclsparse_int       row_ptr[] = {0, 3, 7, 10, 13, 13};
        aoclsparse_int       col_idx[] = {0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 0, 2, 3};
        double val[]     = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 8.0, 9.0, 4.0, -2.0, 1.0, -7.0, 5.0};
        double b[]       = {2.0, 21.0, -12.0, -6.0};
        double x[]       = {0.0, 0.0, 0.0, 0.0};
        double omega     = 0.5;
        double alpha     = 1.0;
        aoclsparse_int M = 5;
        aoclsparse_int N = 4;
        aoclsparse_int NNZ = 13;

        ASSERT_EQ(
            aoclsparse_create_csr(&A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val),
            aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_size);
        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }
    TEST(Sorv, InvalidValues)
    {
        aoclsparse_matrix    A         = nullptr;
        aoclsparse_mat_descr descr     = nullptr;
        aoclsparse_int       row_ptr[] = {0, 3, 7, 10, 13};
        aoclsparse_int       col_idx[] = {0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 0, 2, 3};
        double val[]     = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 8.0, 9.0, 4.0, -2.0, 1.0, -7.0, 5.0};
        double b[]       = {2.0, 21.0, -12.0, -6.0};
        double x[]       = {0.0, 0.0, 0.0, 0.0};
        double omega     = 0.5;
        double alpha     = 1.0;
        aoclsparse_int M = 4;
        aoclsparse_int N = 4;
        aoclsparse_int NNZ = 13;

        ASSERT_EQ(
            aoclsparse_create_csr(&A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val),
            aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        // sor operation in invalid
        EXPECT_EQ(aoclsparse_sorv((aoclsparse_sor_type)10, descr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_value);

        // descr->base is invalid
        descr->base = (aoclsparse_index_base)5;
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_value);

        // descr->base deosn't match A->base
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_value);
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }
    TEST(Sorv, ZeroSizeMatrix) // check if x is unchanged
    {
        aoclsparse_matrix    A         = nullptr;
        aoclsparse_mat_descr descr     = nullptr;
        aoclsparse_int       row_ptr[] = {0, 0, 0, 0};
        aoclsparse_int       col_idx[] = {0};
        double               val[]     = {0.0};
        double               b[]       = {0.0, 0.0, 0.0, 0.0};
        double               x[]       = {0.0, 0.0, 0.0, 0.0};
        double               omega     = 0.5;
        double               alpha     = 1.0;
        aoclsparse_int       M         = 3;
        aoclsparse_int       N         = 3;
        aoclsparse_int       NNZ       = 0;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        // m=3, n=3, nnz=0 : fail as no diagonals present to solve for x
        ASSERT_EQ(
            aoclsparse_create_csr(&A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val),
            aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);

        // m=0, n=0, nnz=0: success as size of x is also 0.
        M = 0;
        N = 0;
        ASSERT_EQ(
            aoclsparse_create_csr(&A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val),
            aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_success);
        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }
    TEST(Sorv, NotImplemented)
    {
        aoclsparse_matrix    A         = nullptr;
        aoclsparse_mat_descr descr     = nullptr;
        aoclsparse_int       M         = 4;
        aoclsparse_int       N         = 4;
        aoclsparse_int       NNZ       = 6;
        aoclsparse_int       row_ptr[] = {0, 1, 2, 3, 6};
        aoclsparse_int       col_idx[] = {0, 1, 2, 0, 1, 3};
        double               val_d[]   = {1.0, 3.0, 4.0, 2.0, 5.0, 6.0};
        double               b_d[]     = {2.0, 21.0, -12.0, -6.0};
        double               x_d[]     = {0.0, 0.0, 0.0, 0.0};
        double               omega_d   = 0.5;
        double               alpha_d   = 1.0;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val_d),
                  aoclsparse_status_success);

        // backward sor
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_backward, descr, A, omega_d, alpha_d, x_d, b_d),
                  aoclsparse_status_not_implemented);
        // symmetric sor
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_symmetric, descr, A, omega_d, alpha_d, x_d, b_d),
                  aoclsparse_status_not_implemented);

        // symmetric matrix
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega_d, alpha_d, x_d, b_d),
                  aoclsparse_status_not_implemented);
        //triangular matrix
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega_d, alpha_d, x_d, b_d),
                  aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A);

        // complex float hermitian matrix
        aoclsparse_float_complex b_cf[]   = {{2.0, 1.0}, {21.0, 1.0}, {-12.0, 1.0}, {-6.0, 1.0}};
        aoclsparse_float_complex x_cf[]   = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
        aoclsparse_float_complex omega_cf = {0.5, 0.5};
        aoclsparse_float_complex alpha_cf = {1.0, 1.0};
        aoclsparse_float_complex val_cf[]
            = {{1.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 0.0}};
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_hermitian),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val_cf),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega_cf, alpha_cf, x_cf, b_cf),
                  aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A);

        // complex double hermitian matrix
        aoclsparse_double_complex b_cd[]   = {{2.0, 1.0}, {21.0, 1.0}, {-12.0, 1.0}, {-6.0, 1.0}};
        aoclsparse_double_complex x_cd[]   = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
        aoclsparse_double_complex omega_cd = {0.5, 0.5};
        aoclsparse_double_complex alpha_cd = {1.0, 1.0};
        aoclsparse_double_complex val_cd[]
            = {{1.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {2.0, 2.0}, {5.0, 5.0}, {6.0, 0.0}};
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr, col_idx, val_cd),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega_cd, alpha_cd, x_cd, b_cd),
                  aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }
    TEST(Sorv, InvalidDiagonal)
    {
        double               omega = 0.5;
        double               alpha = 1.0;
        aoclsparse_int       M     = 4;
        aoclsparse_int       N     = 4;
        aoclsparse_int       NNZ   = 12;
        double               b[]   = {2.0, 21.0, -12.0, -6.0};
        double               x[]   = {0.0, 0.0, 0.0, 0.0};
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        // missing diagonal in 3rd row
        aoclsparse_int row_ptr_1[] = {0, 3, 7, 9, 12};
        aoclsparse_int col_idx_1[] = {0, 1, 2, 0, 1, 2, 3, 1, 3, 0, 2, 3};
        double val_1[] = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 8.0, 9.0, -2.0, 1.0, -7.0, 5.0};
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr_1, col_idx_1, val_1),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x, b),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);

        // duplicate diagonal in 2nd row (non-zero followed by non-zero)
        NNZ                        = 13;
        aoclsparse_int row_ptr_2[] = {0, 3, 7, 10, 13};
        aoclsparse_int col_idx_2[] = {0, 1, 2, 0, 1, 2, 1, 1, 2, 3, 0, 2, 3};
        double val_2[] = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 8.0, 9.0, -2.0, 1.0, -7.0, 5.0};
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr_2, col_idx_2, val_2),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);

        // duplicate diagonal in 2nd row (non-zero followed by zero)
        aoclsparse_int row_ptr_3[] = {0, 3, 7, 10, 13};
        aoclsparse_int col_idx_3[] = {0, 1, 2, 0, 1, 2, 1, 1, 2, 3, 0, 2, 3};
        double val_3[] = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 0.0, 9.0, -2.0, 1.0, -7.0, 5.0};
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr_3, col_idx_3, val_3),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);

        // duplicate diagonal in 2nd row (first is zero )
        aoclsparse_int row_ptr_4[] = {0, 3, 7, 10, 13};
        aoclsparse_int col_idx_4[] = {0, 1, 2, 0, 1, 2, 1, 1, 2, 3, 0, 2, 3};
        double         val_4[] = {4.0, -1.0, -6.0, -5.0, 0.0, 10.0, 8.0, 9.0, -2.0, 1.0, -7.0, 5.0};
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr_4, col_idx_4, val_4),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);

        aoclsparse_destroy_mat_descr(descr);
    }
    TEST(Sorv, GeneralMatrixSuccess)
    {
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        double val[]  = {4.0, -1.0, -6.0, -5.0, -4.0, 10.0, 8.0, 9.0, 4.0, -2.0, 1.0, -7.0, 5.0};
        double b[]    = {2.0, 21.0, -12.0, -6.0};
        double x_0B[] = {1.0, -0.5, -2.0, 3.7};
        // x_gold generated from octave
        double x_gold_iter1[] = {-0.8125, -1.1671874999999998, -0.26191406249999982, 1.14791015625};
        double x_gold_iter1_alpha0[]
            = {0.250000000000, -2.781250000000, 1.628906250000, 0.51523437500000002};
        double x_gold_iter10[]
            = {2.8668745958572917, -2.0001324279196497, 1.9725100983350874, 0.97782651833978285};
        double         omega = 0.5;
        double         alpha = 1.0;
        aoclsparse_int M     = 4;
        aoclsparse_int N     = 4;
        aoclsparse_int NNZ   = 13;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        // zero base indexing. Verifying result after 10 iterations.
        aoclsparse_int row_ptr_0B[] = {0, 3, 7, 10, 13};
        aoclsparse_int col_idx_0B[] = {0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 0, 2, 3};
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr_0B, col_idx_0B, val),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x_0B, b),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, x_0B, x_gold_iter1, expected_precision<double>(10.0));
        for(aoclsparse_int i = 0; i < 9; i++)
        {
            EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x_0B, b),
                      aoclsparse_status_success);
        }
        EXPECT_ARR_NEAR(M, x_0B, x_gold_iter10, expected_precision<double>(10.0));
        aoclsparse_destroy(&A);

        // testing when alpha = 0
        alpha = 0.0;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_zero, M, N, NNZ, row_ptr_0B, col_idx_0B, val),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x_0B, b),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, x_0B, x_gold_iter1_alpha0, expected_precision<double>(10.0));
        aoclsparse_destroy(&A);
        // restoring alpha
        alpha = 1.0;

        // 1-base indexing. Testing for convergence
        aoclsparse_int row_ptr_1B[] = {1, 4, 8, 11, 14};
        aoclsparse_int col_idx_1B[] = {1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 1, 3, 4};
        double         x_1B[]       = {1.0, -0.5, -2.0, 3.7};
        double         b_est[]      = {0.0, 0.0, 0.0, 0.0};
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, aoclsparse_index_base_one, M, N, NNZ, row_ptr_1B, col_idx_1B, val),
                  aoclsparse_status_success);
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
        std::cout.precision(12);
        for(aoclsparse_int i = 1; i <= 46; i++)
        {
            EXPECT_EQ(aoclsparse_sorv(aoclsparse_sor_forward, descr, A, omega, alpha, x_1B, b),
                      aoclsparse_status_success);
            // std::cout << "Iteration " << i << " : " << x_1B[0] << " " << x_1B[1] << " " << x_1B[2]
            //           << " " << x_1B[3] << std::endl;
        }
        EXPECT_EQ(ref_csrmvgen(aoclsparse_operation_none,
                               alpha,
                               M,
                               N,
                               val,
                               col_idx_1B,
                               row_ptr_1B,
                               aoclsparse_index_base_one,
                               x_1B,
                               0.0,
                               b_est),
                  aoclsparse_status_success);

        double ap = 1.0, bp = -1.0, rnorm = 0.0;
        // b_est = b - b_est
        blis::cblas_axpby(M, ap, b, 1, bp, b_est, 1);
        // b_est = || b_gold - b_est ||
        rnorm = blis::cblas_nrm2(M, b_est, 1);
        /*
          x_1B converges after 41 iterations to have a tolerance of 1e-6 and
          46 iterations to have a tolerance of 1e-7 (expected_precision<double>(10.0))
        */
        EXPECT_NEAR(rnorm, 0.0, expected_precision<double>(10.0));
        aoclsparse_destroy(&A);

        aoclsparse_destroy_mat_descr(descr);
    }
}
