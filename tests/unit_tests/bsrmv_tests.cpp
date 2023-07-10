/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse.hpp"

namespace
{
    template <typename T>
    void test_bsrmv_baseZeroIndexing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 5, N = 5, bsr_dim = 2;
        T              alpha = 1.0;
        T              beta  = 0.0;

        aoclsparse_mat_descr descr;
        aoclsparse_int       csr_row_ptr[] = {0, 1, 2, 4, 6, 7};
        aoclsparse_int       csr_col_ind[] = {0, 1, 1, 2, 0, 3, 3};
        T                    csr_val[]     = {
            6.00,
            1.00,
            2.00,
            3.00,
            5.00,
            1.00,
            10.00,
        };
        // Initialize data
        T x[6]      = {1, 2, 3, 4, 5, 6};
        T y[6]      = {0};
        T y_gold[6] = {6.00, 2.00, 13.00, 9.00, 40.00, 0};
        // Update BSR block dimensions from generated matrix
        aoclsparse_int mb = 3;
        aoclsparse_int nb = 3;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        // Convert CSR to BSR
        aoclsparse_int              nnzb;
        std::vector<aoclsparse_int> bsr_row_ptr(mb + 1);

        ASSERT_EQ(aoclsparse_csr2bsr_nnz(
                      M, N, descr, csr_row_ptr, csr_col_ind, bsr_dim, bsr_row_ptr.data(), &nnzb),
                  aoclsparse_status_success);

        std::vector<aoclsparse_int> bsr_col_ind(nnzb);
        std::vector<T>              bsr_val(nnzb * bsr_dim * bsr_dim);
        ASSERT_EQ(aoclsparse_csr2bsr<T>(M,
                                        N,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        bsr_dim,
                                        bsr_val.data(),
                                        bsr_row_ptr.data(),
                                        bsr_col_ind.data()),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_bsrmv(trans,
                                   &alpha,
                                   mb,
                                   nb,
                                   bsr_dim,
                                   bsr_val.data(),
                                   bsr_col_ind.data(),
                                   bsr_row_ptr.data(),
                                   descr,
                                   x,
                                   &beta,
                                   y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_bsrmv_baseOneCSRInput()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 5, N = 5, bsr_dim = 2;
        T              alpha = 1.0;
        T              beta  = 0.0;

        aoclsparse_mat_descr descr;
        aoclsparse_int       csr_row_ptr[] = {1, 2, 3, 5, 7, 8};
        aoclsparse_int       csr_col_ind[] = {1, 2, 2, 3, 1, 4, 4};
        T                    csr_val[]     = {
            6.00,
            1.00,
            2.00,
            3.00,
            5.00,
            1.00,
            10.00,
        };
        /*
            BSR block dimension
                mb = (M + bsr_dim - 1) / bsr_dim = (5+2-1)/2 = 6/2 = 3
                nb = (N + bsr_dim - 1) / bsr_dim = (5+2-1)/2 = 6/2 = 3
            x vector length: nb * bsr_dim = 3 * 2 = 6
            y vector length: mb * bsr_dim = 3 * 2 = 6
            the final bsrmv and reference spmv output(y_gold) are validated for length 'M' only
        */
        aoclsparse_int mb        = 3;
        aoclsparse_int nb        = 3;
        T              x[6]      = {1, 2, 3, 4, 5, 6};
        T              y[6]      = {0};
        T              y_gold[6] = {6.00, 2.00, 13.00, 9.00, 40.00, 0};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        // Convert CSR to BSR
        aoclsparse_int              nnzb;
        std::vector<aoclsparse_int> bsr_row_ptr(mb + 1);

        ASSERT_EQ(aoclsparse_csr2bsr_nnz(
                      M, N, descr, csr_row_ptr, csr_col_ind, bsr_dim, bsr_row_ptr.data(), &nnzb),
                  aoclsparse_status_success);

        std::vector<aoclsparse_int> bsr_col_ind(nnzb);
        std::vector<T>              bsr_val(nnzb * bsr_dim * bsr_dim);
        ASSERT_EQ(aoclsparse_csr2bsr<T>(M,
                                        N,
                                        descr,
                                        csr_val,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        bsr_dim,
                                        bsr_val.data(),
                                        bsr_row_ptr.data(),
                                        bsr_col_ind.data()),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_bsrmv(trans,
                                   &alpha,
                                   mb,
                                   nb,
                                   bsr_dim,
                                   bsr_val.data(),
                                   bsr_col_ind.data(),
                                   bsr_row_ptr.data(),
                                   descr,
                                   x,
                                   &beta,
                                   y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_bsrmv_baseOneBSRInput()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int M = 3, bsr_dim = 2;
        aoclsparse_int mb    = 2;
        aoclsparse_int nb    = 2;
        T              alpha = 1.0;
        T              beta  = 0.0;

        aoclsparse_mat_descr descr;
        aoclsparse_int       bsr_row_ptr[3] = {1, 2, 4};
        aoclsparse_int       bsr_col_ind[3] = {1, 1, 2};
        T bsr_val[12] = {8.00, 0.00, 0.00, 5.00, 7.00, 0.00, 0.00, 0.00, 7.00, 0.00, 0.00, 0.00};
        T x[4]        = {1, 2, 3, 4};
        T y[4]        = {0};
        T y_gold[4]   = {8.00, 10.00, 28.00, -0.96};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_bsrmv(trans,
                                   &alpha,
                                   mb,
                                   nb,
                                   bsr_dim,
                                   bsr_val,
                                   bsr_col_ind,
                                   bsr_row_ptr,
                                   descr,
                                   x,
                                   &beta,
                                   y),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M, y, y_gold, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    TEST(bsrmv, BaseZeroDouble)
    {
        test_bsrmv_baseZeroIndexing<double>();
    }
    TEST(bsrmv, BaseZeroFloat)
    {
        test_bsrmv_baseZeroIndexing<float>();
    }

    TEST(bsrmv, BaseOneDoubleCSRInput)
    {
        test_bsrmv_baseOneCSRInput<double>();
    }
    TEST(bsrmv, BaseOneFloatCSRInput)
    {
        test_bsrmv_baseOneCSRInput<float>();
    }
    TEST(bsrmv, BaseOneDoubleBSRInput)
    {
        test_bsrmv_baseOneBSRInput<double>();
    }
    TEST(bsrmv, BaseOneFloatBSRInput)
    {
        test_bsrmv_baseOneBSRInput<float>();
    }
} // namespace
