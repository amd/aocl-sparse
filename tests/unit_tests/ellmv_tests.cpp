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
#include "aoclsparse_interface.hpp"
#include "aoclsparse_reference.hpp"
namespace
{

    // Several tests in one when nullptr is passed instead of valid data
    template <typename T>
    void test_ellmv_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 2, N = 3, nnz = 1;
        aoclsparse_int       ell_width     = 1;
        aoclsparse_int       ell_col_ind[] = {-1, 1};
        T                    ell_val[]     = {0., 42.};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        //     EXPECT_EQ(aoclsparse_ellmv(trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y), aoclsparse_status_success);

        // pass nullptr and expect pointer error
        EXPECT_EQ(
            aoclsparse_ellmv<T>(
                trans, &alpha, M, N, nnz, nullptr, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_ellmv<T>(
                      trans, &alpha, M, N, nnz, ell_val, nullptr, ell_width, descr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_ellmv<T>(trans,
                                      &alpha,
                                      M,
                                      N,
                                      nnz,
                                      ell_val,
                                      ell_col_ind,
                                      ell_width,
                                      descr,
                                      nullptr,
                                      &beta,
                                      y),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_ellmv<T>(trans,
                                      &alpha,
                                      M,
                                      N,
                                      nnz,
                                      ell_val,
                                      ell_col_ind,
                                      ell_width,
                                      descr,
                                      x,
                                      &beta,
                                      nullptr),
                  aoclsparse_status_invalid_pointer);

        // FIXME: crashes EXPECT_EQ(aoclsparse_ellmv<T>(trans, nullptr, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y), aoclsparse_status_invalid_pointer);
        // FIXME: crashes EXPECT_EQ(aoclsparse_ellmv<T>(trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, nullptr, y), aoclsparse_status_invalid_pointer);

        EXPECT_EQ(
            aoclsparse_ellmv<T>(
                trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, nullptr, x, &beta, y),
            aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
    }

    template <typename T>
    void test_ellmv_wrong_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int       M = 2, N = 3, nnz = 1;
        aoclsparse_int       ell_width     = 1;
        aoclsparse_int       ell_col_ind[] = {-1, 1};
        T                    ell_val[]     = {0., 42.};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        EXPECT_EQ(
            aoclsparse_ellmv(
                trans, &alpha, -1, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_size);

        EXPECT_EQ(
            aoclsparse_ellmv(
                trans, &alpha, M, -1, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_size);

        EXPECT_EQ(aoclsparse_ellmv(
                      trans, &alpha, M, N, nnz, ell_val, ell_col_ind, -1, descr, x, &beta, y),
                  aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
    }
    template <typename T>
    void test_ellmv_not_implemented()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int                M = 2, N = 3, nnz = 1;
        aoclsparse_int                ell_width     = 1;
        aoclsparse_int                ell_col_ind[] = {-1, 1};
        T                             ell_val[]     = {0., 42.};
        T                             alpha = 2.3, beta = 11.2;
        T                             x[] = {1.0, -2.0, 3.0};
        T                             y[] = {0.1, 0.2};
        struct _aoclsparse_mat_descr *descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(
            aoclsparse_ellmv(
                trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_not_implemented);

        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
        trans = aoclsparse_operation_transpose;
        EXPECT_EQ(
            aoclsparse_ellmv(
                trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descr);
    }
    template <typename T>
    void test_ellmv_baseOneCSRInput()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 3, N = 3, NNZ = 4;
        T              alpha = 1.0, beta = 0.0;

        aoclsparse_int csr_row_ptr[] = {1, 2, 3, 5}; //one-based indexing
        aoclsparse_int csr_col_ind[] = {1, 2, 1, 3}; //one-based indexing
        T              csr_val[]     = {8.00, 5.00, 7.00, 7.00};
        // Initialise vectors
        T x[] = {1.0, 2.0, 3.0};
        T y[M];
        T y_gold[] = {8.00, 10.00, 28.00};

        std::vector<aoclsparse_int> ell_col_ind;
        std::vector<T>              ell_val;
        aoclsparse_int              ell_width;

        aoclsparse_mat_descr descr;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_csr2ell_width(M, NNZ, csr_row_ptr, &ell_width),
                  aoclsparse_status_success);
        ell_col_ind.resize(ell_width * M);
        ell_val.resize(ell_width * M);

        ASSERT_EQ(aoclsparse_csr2ell(M,
                                     descr,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     ell_col_ind.data(),
                                     ell_val.data(),
                                     ell_width),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_ellmv(trans,
                                   &alpha,
                                   M,
                                   N,
                                   NNZ,
                                   ell_val.data(),
                                   ell_col_ind.data(),
                                   ell_width,
                                   descr,
                                   x,
                                   &beta,
                                   y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_ellmv_baseOneEllInput()
    {
        aoclsparse_operation trans              = aoclsparse_operation_none;
        int                  invalid_index_base = 2;
        aoclsparse_int       M = 3, N = 3, NNZ = 4;
        aoclsparse_int       ell_width     = 2;
        aoclsparse_int       ell_col_ind[] = {1, -1, 2, -1, 1, 3};
        //aoclsparse_int        ell_col_ind[] = {0,-1,1,-1,0,2};
        T                    ell_val[] = {8.00, 0.00, 5.00, 0.00, 7.00, 7.00};
        T                    alpha = 1.0, beta = 0.0;
        T                    x[3]      = {1.0, 2.0, 3.0};
        T                    y[3]      = {0};
        T                    y_gold[3] = {8.00, 10.00, 28.00};
        aoclsparse_mat_descr descr;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_ellmv(
                trans, &alpha, M, N, NNZ, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        descr->base = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(
            aoclsparse_ellmv<T>(
                trans, &alpha, M, N, NNZ, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_ellmv_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int       M = 2, N = 3, nnz = 1;
        aoclsparse_int       ell_width     = 1;
        aoclsparse_int       ell_col_ind[] = {-1, 1};
        T                    ell_val[]     = {0., 42.};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        EXPECT_EQ(
            aoclsparse_ellmv(
                trans, &alpha, 0, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_size);

        EXPECT_EQ(
            aoclsparse_ellmv(
                trans, &alpha, M, 0, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_size);
        EXPECT_EQ(
            aoclsparse_ellmv(trans, &alpha, M, 0, nnz, ell_val, ell_col_ind, 0, descr, x, &beta, y),
            aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_ellmv(trans, &alpha, M, N, nnz, ell_val, ell_col_ind, 0, descr, x, &beta, y),
            aoclsparse_status_success);

        aoclsparse_destroy_mat_descr(descr);
    }

    // tests to cover ell related conversion routines
    // includes cases related to nullptr and wrong sizes
    template <typename T>
    void test_ellmv_conversion()
    {

        aoclsparse_int M = 4, nnz = 6;
        aoclsparse_int ell_width         = 3;
        T              csr_val[]         = {42., 1, 5, 1, -1, 2};
        aoclsparse_int csr_col_ind[]     = {0, 1, 2, 1, 2, 2};
        aoclsparse_int csr_row_ptr[]     = {0, 3, 4, 5, 6};
        aoclsparse_int ell_col_ind[]     = {0, 1, 2, 1, -1, -1, 2, -1, -1, 2, -1, -1};
        T              ell_val[]         = {42., 1, 5, 1, 0, 0, -1, 0, 0, 2, 0, 0};
        aoclsparse_int ell_m             = 3;
        aoclsparse_int csr_row_idx_map[] = {0};

        aoclsparse_mat_descr  descr;
        aoclsparse_index_base base = aoclsparse_index_base_zero;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_csr2ell_width(-1, nnz, csr_row_ptr, &ell_width),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_csr2ell_width(M, nnz, nullptr, &ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ell_width(M, nnz, csr_row_ptr, nullptr),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_csr2ellthyb_width(-1, nnz, csr_row_ptr, &ell_m, &ell_width),
                  aoclsparse_status_invalid_size);

        EXPECT_EQ(aoclsparse_csr2ellthyb_width(M, nnz, nullptr, &ell_m, &ell_width),
                  aoclsparse_status_invalid_pointer);
        // Fails this test
        //      EXPECT_EQ(aoclsparse_csr2ellthyb_width(M, nnz, csr_row_ptr, nullptr, &ell_width), aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellthyb_width(0, nnz, csr_row_ptr, &ell_m, &ell_width),
                  aoclsparse_status_success);
        // resetting the values as before
        ell_width = 3;
        ell_m     = 3;

        EXPECT_EQ(aoclsparse_csr2ellthyb_width(M, nnz, csr_row_ptr, &ell_m, nullptr),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_csr2ell<T>(
                      0, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_csr2ell<T>(
                -1, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width),
            aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_csr2ell<T>(
                      M, descr, nullptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ell<T>(
                      M, descr, csr_row_ptr, nullptr, csr_val, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ell<T>(
                      M, descr, csr_row_ptr, csr_col_ind, nullptr, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ell<T>(
                      M, descr, csr_row_ptr, csr_col_ind, csr_val, nullptr, ell_val, ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ell<T>(
                      M, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, nullptr, ell_width),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_csr2ellt<T>(
                      0, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_csr2ellt<T>(
                -1, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width),
            aoclsparse_status_invalid_size);

        EXPECT_EQ(aoclsparse_csr2ellt<T>(
                      M, descr, nullptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellt<T>(
                      M, descr, csr_row_ptr, nullptr, csr_val, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellt<T>(
                      M, descr, csr_row_ptr, csr_col_ind, nullptr, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_csr2ellt<T>(
                      M, descr, csr_row_ptr, csr_col_ind, csr_val, nullptr, ell_val, ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellt<T>(
                      M, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, nullptr, ell_width),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_csr2ellt<T>(
                      M, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            base,
                                            &ell_m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            csr_row_idx_map,
                                            ell_col_ind,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            base,
                                            nullptr,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            csr_row_idx_map,
                                            ell_col_ind,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            base,
                                            &ell_m,
                                            nullptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            csr_row_idx_map,
                                            ell_col_ind,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            base,
                                            &ell_m,
                                            csr_row_ptr,
                                            nullptr,
                                            csr_val,
                                            nullptr,
                                            csr_row_idx_map,
                                            ell_col_ind,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            base,
                                            &ell_m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            nullptr,
                                            nullptr,
                                            csr_row_idx_map,
                                            ell_col_ind,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            base,
                                            &ell_m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            nullptr,
                                            ell_col_ind,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            base,
                                            &ell_m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            csr_row_idx_map,
                                            nullptr,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            base,
                                            &ell_m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            csr_row_idx_map,
                                            ell_col_ind,
                                            nullptr,
                                            ell_width),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(-1,
                                            base,
                                            &ell_m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            csr_row_idx_map,
                                            ell_col_ind,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_invalid_size);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

    template <typename T>
    void test_elltmv()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 5, N = 3, nnz = 5;
        aoclsparse_int ell_width     = 1;
        aoclsparse_int ell_col_ind[] = {0, 1, 2, 2, 2};
        T              ell_val[]     = {1, 1, 1, 1, 1};
        T              alpha = 2, beta = 2;
        T              x[]      = {1.0, 2.0, 5.0};
        T              y[]      = {1.0, 1.0, 1.0, 1.0, 1.0};
        T              exp_y1[] = {4.0, 6.0, 12.0, 12.0, 12.0};
        T              exp_y2[] = {1.0, 2.0, 5.0, 5.0, 5.0};

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        // test elltmv success case
        //ToDo: add failure cases
        EXPECT_EQ(
            aoclsparse_elltmv<T>(
                trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y1);
        alpha = 1, beta = 0;
        EXPECT_EQ(
            aoclsparse_elltmv<T>(
                trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y2);

        aoclsparse_destroy_mat_descr(descr);
    }
    template <typename T>
    void test_elltmv_baseOneCSRInput()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 3, N = 3, NNZ = 4;
        T                    alpha = 1.0, beta = 0.0;

        aoclsparse_int csr_row_ptr[] = {1, 2, 3, 5}; //one-based indexing
        aoclsparse_int csr_col_ind[] = {1, 2, 1, 3}; //one-based indexing
        T              csr_val[]     = {8.00, 5.00, 7.00, 7.00};
        // Initialise vectors
        T                           x[] = {1.0, 2.0, 3.0};
        T                           y[M];
        T                           y_gold[] = {8.00, 10.00, 28.00};
        std::vector<aoclsparse_int> ell_col_ind;
        std::vector<T>              ell_val;
        aoclsparse_int              ell_width;
        aoclsparse_mat_descr        descr;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_csr2ell_width(M, NNZ, csr_row_ptr, &ell_width),
                  aoclsparse_status_success);
        ell_col_ind.resize(ell_width * M);
        ell_val.resize(ell_width * M);

        ASSERT_EQ(aoclsparse_csr2ellt(M,
                                      descr,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      csr_val,
                                      ell_col_ind.data(),
                                      ell_val.data(),
                                      ell_width),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_elltmv(trans,
                                    &alpha,
                                    M,
                                    N,
                                    NNZ,
                                    ell_val.data(),
                                    ell_col_ind.data(),
                                    ell_width,
                                    descr,
                                    x,
                                    &beta,
                                    y),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_elltmv_baseOneElltInput()
    {
        aoclsparse_operation trans              = aoclsparse_operation_none;
        int                  invalid_index_base = 2;
        aoclsparse_int       M = 5, N = 3, NNZ = 5;
        aoclsparse_int       ell_width = 1;
        //aoclsparse_int ell_col_ind[5] = {0, 1, 2, 2, 2};
        aoclsparse_int       ell_col_ind[5] = {1, 2, 3, 3, 3};
        T                    ell_val[5]     = {1, 1, 1, 1, 1};
        T                    alpha = 1, beta = 0;
        T                    x[3]      = {1.0, 2.0, 5.0};
        T                    y[5]      = {0};
        T                    y_gold[5] = {1.0, 2.0, 5.0, 5.0, 5.0};
        aoclsparse_mat_descr descr;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_elltmv(
                trans, &alpha, M, N, NNZ, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        descr->base = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(
            aoclsparse_elltmv(
                trans, &alpha, M, N, NNZ, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_elltmv_not_implemented()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int                M = 2, N = 3, nnz = 1;
        aoclsparse_int                ell_width     = 1;
        aoclsparse_int                ell_col_ind[] = {-1, 1};
        T                             ell_val[]     = {0., 42.};
        T                             alpha = 2.3, beta = 11.2;
        T                             x[] = {1.0, -2.0, 3.0};
        T                             y[] = {0.1, 0.2};
        struct _aoclsparse_mat_descr *descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(
            aoclsparse_elltmv(
                trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_not_implemented);

        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
        trans = aoclsparse_operation_transpose;
        EXPECT_EQ(
            aoclsparse_elltmv(
                trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descr);
    }

    template <typename T>
    void test_elltmv_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 2, N = 3, nnz = 1;
        aoclsparse_int       ell_width     = 1;
        aoclsparse_int       ell_col_ind[] = {-1, 1};
        T                    ell_val[]     = {0., 42.};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        //     EXPECT_EQ(aoclsparse_ellmv(trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y), aoclsparse_status_success);

        // pass nullptr and expect pointer error
        EXPECT_EQ(
            aoclsparse_elltmv<T>(
                trans, &alpha, M, N, nnz, nullptr, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_elltmv<T>(
                      trans, &alpha, M, N, nnz, ell_val, nullptr, ell_width, descr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_elltmv<T>(trans,
                                       &alpha,
                                       M,
                                       N,
                                       nnz,
                                       ell_val,
                                       ell_col_ind,
                                       ell_width,
                                       descr,
                                       nullptr,
                                       &beta,
                                       y),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_elltmv<T>(trans,
                                       &alpha,
                                       M,
                                       N,
                                       nnz,
                                       ell_val,
                                       ell_col_ind,
                                       ell_width,
                                       descr,
                                       x,
                                       &beta,
                                       nullptr),
                  aoclsparse_status_invalid_pointer);

        // FIXME: crashes EXPECT_EQ(aoclsparse_ellmv<T>(trans, nullptr, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y), aoclsparse_status_invalid_pointer);
        // FIXME: crashes EXPECT_EQ(aoclsparse_ellmv<T>(trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, nullptr, y), aoclsparse_status_invalid_pointer);

        EXPECT_EQ(
            aoclsparse_elltmv<T>(
                trans, &alpha, M, N, nnz, ell_val, ell_col_ind, ell_width, nullptr, x, &beta, y),
            aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
    }

    template <typename T>
    void test_elltmv_wrong_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int       M = 2, N = 3, nnz = 1;
        aoclsparse_int       ell_width     = 1;
        aoclsparse_int       ell_col_ind[] = {-1, 1};
        T                    ell_val[]     = {0., 42.};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        EXPECT_EQ(
            aoclsparse_elltmv(
                trans, &alpha, -1, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_size);

        EXPECT_EQ(
            aoclsparse_elltmv(
                trans, &alpha, M, -1, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_size);

        EXPECT_EQ(aoclsparse_elltmv(
                      trans, &alpha, M, N, nnz, ell_val, ell_col_ind, -1, descr, x, &beta, y),
                  aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
    }

    template <typename T>
    void test_elltmv_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int       M = 2, N = 3, nnz = 1;
        aoclsparse_int       ell_width     = 1;
        aoclsparse_int       ell_col_ind[] = {-1, 1};
        T                    ell_val[]     = {0., 42.};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);
        EXPECT_EQ(
            aoclsparse_elltmv(
                trans, &alpha, 0, N, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_size);

        EXPECT_EQ(
            aoclsparse_elltmv(
                trans, &alpha, M, 0, nnz, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_elltmv(
                      trans, &alpha, M, 0, nnz, ell_val, ell_col_ind, 0, descr, x, &beta, y),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_elltmv(
                      trans, &alpha, M, N, nnz, ell_val, ell_col_ind, 0, descr, x, &beta, y),
                  aoclsparse_status_success);

        aoclsparse_destroy_mat_descr(descr);
    }

    template <typename T>
    void test_ellthybmv()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 5, N = 6, nnz = 10;
        aoclsparse_int ell_width     = 1;
        T              csr_val[]     = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        aoclsparse_int csr_col_ind[] = {0, 1, 0, 1, 2, 3, 4, 5, 2, 2};
        aoclsparse_int csr_row_ptr[] = {0, 1, 2, 8, 9, 10};
        aoclsparse_int ell_col_ind[] = {0, 1, 2, 2, 2};
        T              ell_val[]     = {1, 1, 0, 1, 1};
        aoclsparse_int ell_m         = 4;

        aoclsparse_int csr_row_idx_map[] = {2};
        T              alpha = 2, beta = 2;
        T              x[]      = {2.0, 3.0, 1.0, -1.0, 2.0, 1.0};
        T              y[]      = {1, 1, 1, 1, 1};
        T              exp_y1[] = {6, 8, 18, 4, 4};
        T              exp_y2[] = {2, 3, 8, 1, 1};

        aoclsparse_mat_descr descr;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        // test ellthybmv success case
        //ToDo: add failure cases
        EXPECT_EQ(aoclsparse_ellthybmv<T>(trans,
                                          &alpha,
                                          M,
                                          N,
                                          nnz,
                                          ell_val,
                                          ell_col_ind,
                                          ell_width,
                                          ell_m,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          nullptr,
                                          csr_row_idx_map,
                                          descr,
                                          x,
                                          &beta,
                                          y),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y1);
        alpha = 1, beta = 0;
        EXPECT_EQ(aoclsparse_ellthybmv<T>(trans,
                                          &alpha,
                                          M,
                                          N,
                                          nnz,
                                          ell_val,
                                          ell_col_ind,
                                          ell_width,
                                          ell_m,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          nullptr,
                                          csr_row_idx_map,
                                          descr,
                                          x,
                                          &beta,
                                          y),
                  aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(5, y, exp_y2);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_ellthybmv_baseZeroCSRInput()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 3, N = 3, NNZ = 4;
        T                    alpha = 1.0, beta = 0.0;

        aoclsparse_int csr_row_ptr[4] = {0, 1, 2, 4}; //zero-based indexing
        aoclsparse_int csr_col_ind[4] = {0, 1, 0, 2}; //one-based indexing
        T              csr_val[4]     = {8.00, 5.00, 7.00, 7.00};
        // Initialise vectors
        T                           x[3] = {1.0, 2.0, 3.0};
        T                           y[3];
        T                           y_gold[3] = {0};
        std::vector<aoclsparse_int> csr_row_idx_map;
        std::vector<aoclsparse_int> ell_col_ind;
        std::vector<T>              ell_val;
        aoclsparse_int              ell_width, ell_m;
        aoclsparse_mat_descr        descr;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        //compute reference spmv output
        ASSERT_EQ(ref_csrmvgen(
                      trans, alpha, M, N, csr_val, csr_col_ind, csr_row_ptr, base, x, beta, y_gold),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_csr2ellthyb_width(M, NNZ, csr_row_ptr, &ell_m, &ell_width),
                  aoclsparse_status_success);
        ell_col_ind.resize(ell_width * M);
        ell_val.resize(ell_width * M);
        csr_row_idx_map.resize(M - ell_m);

        ASSERT_EQ(aoclsparse_csr2ellthyb(M,
                                         base,
                                         &ell_m,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         csr_val,
                                         NULL,
                                         csr_row_idx_map.data(),
                                         ell_col_ind.data(),
                                         ell_val.data(),
                                         ell_width),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_ellthybmv<T>(trans,
                                          &alpha,
                                          M,
                                          N,
                                          NNZ,
                                          ell_val.data(),
                                          ell_col_ind.data(),
                                          ell_width,
                                          ell_m,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          nullptr,
                                          csr_row_idx_map.data(),
                                          descr,
                                          x,
                                          &beta,
                                          y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_ellthybmv_baseOneCSRInput()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       M = 3, N = 3, NNZ = 4;
        T                    alpha = 1.0, beta = 0.0;

        aoclsparse_int csr_row_ptr[4] = {1, 2, 3, 5}; //one-based indexing
        aoclsparse_int csr_col_ind[4] = {1, 2, 1, 3}; //one-based indexing
        T              csr_val[4]     = {8.00, 5.00, 7.00, 7.00};
        // Initialise vectors
        T                           x[3] = {1.0, 2.0, 3.0};
        T                           y[3];
        T                           y_gold[3] = {0};
        std::vector<aoclsparse_int> csr_row_idx_map;
        std::vector<aoclsparse_int> ell_col_ind;
        std::vector<T>              ell_val;
        aoclsparse_int              ell_width, ell_m;
        aoclsparse_mat_descr        descr;
        aoclsparse_index_base       base = aoclsparse_index_base_one;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        //compute reference spmv output
        ASSERT_EQ(ref_csrmvgen(
                      trans, alpha, M, N, csr_val, csr_col_ind, csr_row_ptr, base, x, beta, y_gold),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_csr2ellthyb_width(M, NNZ, csr_row_ptr, &ell_m, &ell_width),
                  aoclsparse_status_success);
        ell_col_ind.resize(ell_width * M);
        ell_val.resize(ell_width * M);
        csr_row_idx_map.resize(M - ell_m);

        ASSERT_EQ(aoclsparse_csr2ellthyb(M,
                                         base,
                                         &ell_m,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         csr_val,
                                         NULL,
                                         csr_row_idx_map.data(),
                                         ell_col_ind.data(),
                                         ell_val.data(),
                                         ell_width),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_ellthybmv<T>(trans,
                                          &alpha,
                                          M,
                                          N,
                                          NNZ,
                                          ell_val.data(),
                                          ell_col_ind.data(),
                                          ell_width,
                                          ell_m,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          nullptr,
                                          csr_row_idx_map.data(),
                                          descr,
                                          x,
                                          &beta,
                                          y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_ellthybmv_baseOneElltHybInput()
    {
        aoclsparse_operation trans              = aoclsparse_operation_none;
        int                  invalid_index_base = 2;
        aoclsparse_int       M = 5, N = 6, NNZ = 10;
        aoclsparse_int       ell_width     = 1;
        T                    csr_val[]     = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        aoclsparse_int       csr_col_ind[] = {1, 2, 1, 2, 3, 4, 5, 6, 3, 3};
        aoclsparse_int       csr_row_ptr[] = {1, 2, 3, 9, 10, 11};
        aoclsparse_int       ell_col_ind[] = {1, 2, 3, 3, 3};
        T                    ell_val[]     = {1, 1, 0, 1, 1};
        aoclsparse_int       ell_m         = 4;

        aoclsparse_int csr_row_idx_map[] = {2};
        T              alpha = 1.0, beta = 0.0;
        T              x[]       = {2.0, 3.0, 1.0, -1.0, 2.0, 1.0};
        T              y[5]      = {0};
        T              y_gold[5] = {2, 3, 8, 1, 1};

        aoclsparse_mat_descr descr;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_ellthybmv<T>(trans,
                                          &alpha,
                                          M,
                                          N,
                                          NNZ,
                                          ell_val,
                                          ell_col_ind,
                                          ell_width,
                                          ell_m,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          nullptr,
                                          csr_row_idx_map,
                                          descr,
                                          x,
                                          &beta,
                                          y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        descr->base = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(
            aoclsparse_ellmv<T>(
                trans, &alpha, M, N, NNZ, ell_val, ell_col_ind, ell_width, descr, x, &beta, y),
            aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_ellthybmv_wrong_ellidx()
    {
        aoclsparse_int M             = 4;
        aoclsparse_int ell_width     = 2;
        T              csr_val[]     = {42., 1, 5, 1, -1, 2};
        aoclsparse_int csr_col_ind[] = {0, 1, 0, 1, 2, 1};
        aoclsparse_int csr_row_ptr[] = {0, 1, 2, 5, 6};
        aoclsparse_int ell_col_ind[]
            = {1209091209, -129821982, 9120912, 9120912, -7, 18, -1201123, 11};
        T              ell_val[]         = {42., 1, 5, 1, 1, 2, 3, 4};
        aoclsparse_int ell_m             = -1;
        aoclsparse_int csr_row_idx_map[] = {0};

        aoclsparse_int exp_ell_col_ind[] = {0, 1, 2, 1, 0, 1, 2, 1};
        T              exp_ell_val[]     = {42., 1, 0, 2, 0, 0, 0, 0};
        aoclsparse_int exp_ell_m         = 3;

        aoclsparse_int exp_csr_row_idx_map[] = {2};

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        EXPECT_EQ(aoclsparse_csr2ellthyb<T>(M,
                                            aoclsparse_index_base_zero,
                                            &ell_m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            csr_row_idx_map,
                                            ell_col_ind,
                                            ell_val,
                                            ell_width),
                  aoclsparse_status_success);
        EXPECT_EQ_VEC(8, ell_col_ind, exp_ell_col_ind);
        EXPECT_DOUBLE_EQ_VEC(8, ell_val, exp_ell_val);
        EXPECT_EQ(ell_m, exp_ell_m);
        EXPECT_EQ_VEC(1, csr_row_idx_map, exp_csr_row_idx_map);

        aoclsparse_destroy_mat_descr(descr);
    }

    TEST(ellmv, NullArgDouble)
    {
        test_ellmv_nullptr<double>();
    }
    TEST(ellmv, NullArgFloat)
    {
        test_ellmv_nullptr<float>();
    }

    TEST(ellmv, WrongSizeDouble)
    {
        test_ellmv_wrong_size<double>();
    }
    TEST(ellmv, WrongSizeFloat)
    {
        test_ellmv_wrong_size<float>();
    }
    TEST(ellmv, NotImplDouble)
    {
        test_ellmv_not_implemented<double>();
    }
    TEST(ellmv, NotImplFloat)
    {
        test_ellmv_not_implemented<float>();
    }
    TEST(ellmv, DoNothingDouble)
    {
        test_ellmv_do_nothing<double>();
    }
    TEST(ellmv, DoNothingFloat)
    {
        test_ellmv_do_nothing<float>();
    }

    TEST(ellmv, ELLTMVNotImplDouble)
    {
        test_elltmv_not_implemented<double>();
    }
    TEST(ellmv, ELLTMVNotImplFloat)
    {
        test_elltmv_not_implemented<float>();
    }
    TEST(ellmv, ELLTNullArgDouble)
    {
        test_elltmv_nullptr<double>();
    }
    TEST(ellmv, ELLTNullArgFloat)
    {
        test_elltmv_nullptr<float>();
    }

    TEST(ellmv, ELLTWrongSizeDouble)
    {
        test_elltmv_wrong_size<double>();
    }
    TEST(ellmv, ELLTWrongSizeFloat)
    {
        test_elltmv_wrong_size<float>();
    }
    TEST(ellmv, ELLTDoNothingDouble)
    {
        test_elltmv_do_nothing<double>();
    }
    TEST(ellmv, ELLTDoNothingFloat)
    {
        test_elltmv_do_nothing<float>();
    }

    TEST(ellmv, ConversionDouble)
    {
        test_ellmv_conversion<double>();
    }
    TEST(ellmv, ConversionFloat)
    {
        test_ellmv_conversion<float>();
    }

    TEST(ellmv, ELLTSuccessDouble)
    {
        test_elltmv<double>();
    }
    TEST(ellmv, ELLTSuccessFloat)
    {
        test_elltmv<float>();
    }
    TEST(ellmv, ELLTHYBSuccessDouble)
    {
        test_ellthybmv<double>();
    }

    TEST(ellmv, ELLTHYBWrongIdx)
    {
        test_ellthybmv_wrong_ellidx<double>();
    }

    //Base One-Indexing Tests

    //ELL-MV
    TEST(ellmv, BaseOneDoubleCSRInput)
    {
        test_ellmv_baseOneCSRInput<double>();
    }
    TEST(ellmv, BaseOneFloatCSRInput)
    {
        test_ellmv_baseOneCSRInput<float>();
    }
    TEST(ellmv, BaseOneDoubleEllInput)
    {
        test_ellmv_baseOneEllInput<double>();
    }
    TEST(ellmv, BaseOneFloatEllInput)
    {
        test_ellmv_baseOneEllInput<float>();
    }
    //ELL-T-MV
    TEST(ellmv, ELLTMVBaseOneDoubleCSRInput)
    {
        test_elltmv_baseOneCSRInput<double>();
    }
    TEST(ellmv, ELLTMVBaseOneFloatCSRInput)
    {
        test_elltmv_baseOneCSRInput<float>();
    }
    TEST(ellmv, ELLTMVBaseOneDoubleElltInput)
    {
        test_elltmv_baseOneElltInput<double>();
    }
    TEST(ellmv, ELLTMVBaseOneFloatElltInput)
    {
        test_elltmv_baseOneElltInput<float>();
    }
    //ELL-T-HYB-MV
    TEST(ellmv, ELLTHYBBaseZeroDoubleCSRInput)
    {
        test_ellthybmv_baseZeroCSRInput<double>();
    }
    TEST(ellmv, ELLTHYBBaseOneDoubleCSRInput)
    {
        test_ellthybmv_baseOneCSRInput<double>();
    }
    TEST(ellmv, ELLTHYBBaseOneDoubleElltHybInput)
    {
        test_ellthybmv_baseOneElltHybInput<double>();
    }
} // namespace
