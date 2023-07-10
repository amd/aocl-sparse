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

#include <algorithm>
#include <cmath>
#include <limits.h>

namespace
{
    template <typename T>
    void test_csr_to_csc()
    {
        aoclsparse_int M = 5, N = 5, NNZ = 16;

        aoclsparse_mat_descr  descr;
        aoclsparse_index_base base, baseCSC;
        aoclsparse_int        csr_row_ptr[6]  = {0, 3, 6, 10, 12, 16};
        aoclsparse_int        csr_col_ind[16] = {0, 1, 4, 1, 2, 4, 0, 1, 2, 3, 2, 3, 0, 1, 2, 4};
        T                     csr_val[16]     = {1.00,
                                                 1.00,
                                                 4.00,
                                                 2.00,
                                                 4.00,
                                                 1.00,
                                                 2.00,
                                                 1.00,
                                                 8.00,
                                                 2.00,
                                                 4.00,
                                                 1.00,
                                                 3.00,
                                                 6.00,
                                                 2.00,
                                                 1.00};

        aoclsparse_int csc_col_ptr[6]  = {0};
        aoclsparse_int csc_row_ind[16] = {0};
        T              csc_val[16]     = {0.0};

        //the reference output in CSC format for final comparisons
        aoclsparse_int csc_base0_col_ptr_gold[6] = {0, 3, 7, 11, 13, 16};
        aoclsparse_int csc_base0_row_ind_gold[16]
            = {0, 2, 4, 0, 1, 2, 4, 1, 2, 3, 4, 2, 3, 0, 1, 4};
        T csc_val_gold[16] = {1.00,
                              2.00,
                              3.00,
                              1.00,
                              2.00,
                              1.00,
                              6.00,
                              4.00,
                              8.00,
                              4.00,
                              2.00,
                              2.00,
                              1.00,
                              4.00,
                              1.00,
                              1.00};

        aoclsparse_int        csc_base1_col_ptr_gold[6];
        aoclsparse_int        csc_base1_row_ind_gold[16];
        aoclsparse_index_base base1_gold = aoclsparse_index_base_one;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        //update reference data since the output of
        //csr2csc preserves input base-indexing, which is 1-based
        std::transform(csc_base0_col_ptr_gold,
                       csc_base0_col_ptr_gold + N + 1,
                       csc_base1_col_ptr_gold,
                       [base1_gold](aoclsparse_int &d) { return d + base1_gold; });
        std::transform(csc_base0_row_ind_gold,
                       csc_base0_row_ind_gold + NNZ,
                       csc_base1_row_ind_gold,
                       [base1_gold](aoclsparse_int &d) { return d + base1_gold; });

        //TEST CASE 1: input csr: base-0, output csc: base-0
        base = aoclsparse_index_base_zero;
        //Output-base index of csc buffer
        baseCSC = aoclsparse_index_base_zero;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        // Convert CSR to CSC
        ASSERT_EQ(aoclsparse_csr2csc(M,
                                     N,
                                     NNZ,
                                     descr,
                                     baseCSC,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     csc_row_ind,
                                     csc_col_ptr,
                                     csc_val),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M + 1, csc_col_ptr, csc_base0_col_ptr_gold, expected_precision<T>());
        EXPECT_ARR_NEAR(NNZ, csc_row_ind, csc_base0_row_ind_gold, expected_precision<T>());
        EXPECT_ARR_NEAR(NNZ, csc_val, csc_val_gold, expected_precision<T>());

        //TEST CASE 2: input csr: base-0, output csc: base-1
        std::fill(csc_col_ptr, csc_col_ptr + M + 1, 0);
        std::fill(csc_row_ind, csc_row_ind + NNZ, 0);
        std::fill(csc_val, csc_val + NNZ, 0.0);
        //Output-base index of csc buffer
        baseCSC = aoclsparse_index_base_one;

        // Convert CSR to CSC
        ASSERT_EQ(aoclsparse_csr2csc(M,
                                     N,
                                     NNZ,
                                     descr,
                                     baseCSC,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     csc_row_ind,
                                     csc_col_ptr,
                                     csc_val),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M + 1, csc_col_ptr, csc_base1_col_ptr_gold, expected_precision<T>());
        EXPECT_ARR_NEAR(NNZ, csc_row_ind, csc_base1_row_ind_gold, expected_precision<T>());
        EXPECT_ARR_NEAR(NNZ, csc_val, csc_val_gold, expected_precision<T>());

        //TEST CASE 3: input csr: base-1, output csc: base-0
        std::fill(csc_col_ptr, csc_col_ptr + M + 1, 0);
        std::fill(csc_row_ind, csc_row_ind + NNZ, 0);
        std::fill(csc_val, csc_val + NNZ, 0.0);
        base = aoclsparse_index_base_one;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        //rebuild indices for 1-based indexing and then call csr2csc
        std::transform(csr_row_ptr, csr_row_ptr + M + 1, csr_row_ptr, [base](aoclsparse_int &d) {
            return d + base;
        });
        std::transform(csr_col_ind, csr_col_ind + NNZ, csr_col_ind, [base](aoclsparse_int &d) {
            return d + base;
        });
        //Output-base index of csc buffer
        baseCSC = aoclsparse_index_base_zero;
        // Convert CSR to CSC
        ASSERT_EQ(aoclsparse_csr2csc(M,
                                     N,
                                     NNZ,
                                     descr,
                                     baseCSC,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     csc_row_ind,
                                     csc_col_ptr,
                                     csc_val),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M + 1, csc_col_ptr, csc_base0_col_ptr_gold, expected_precision<T>());
        EXPECT_ARR_NEAR(NNZ, csc_row_ind, csc_base0_row_ind_gold, expected_precision<T>());
        EXPECT_ARR_NEAR(NNZ, csc_val, csc_val_gold, expected_precision<T>());

        //TEST CASE 4: input csr: base-1, output csc: base-1
        std::fill(csc_col_ptr, csc_col_ptr + M + 1, 0);
        std::fill(csc_row_ind, csc_row_ind + NNZ, 0);
        std::fill(csc_val, csc_val + NNZ, 0.0);
        //Output-base index of csc buffer
        baseCSC = aoclsparse_index_base_one;

        // Convert CSR to CSC
        ASSERT_EQ(aoclsparse_csr2csc(M,
                                     N,
                                     NNZ,
                                     descr,
                                     baseCSC,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     csc_row_ind,
                                     csc_col_ptr,
                                     csc_val),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(M + 1, csc_col_ptr, csc_base1_col_ptr_gold, expected_precision<T>());
        EXPECT_ARR_NEAR(NNZ, csc_row_ind, csc_base1_row_ind_gold, expected_precision<T>());
        EXPECT_ARR_NEAR(NNZ, csc_val, csc_val_gold, expected_precision<T>());

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_csr_to_dense()
    {
        aoclsparse_int        M = 5, N = 5, NNZ = 16;
        aoclsparse_int        ld;
        aoclsparse_mat_descr  descr;
        aoclsparse_index_base base;
        aoclsparse_order      order;
        aoclsparse_int        csr_row_ptr[6]  = {0, 3, 6, 10, 12, 16};
        aoclsparse_int        csr_col_ind[16] = {0, 1, 4, 1, 2, 4, 0, 1, 2, 3, 2, 3, 0, 1, 2, 4};
        T                     csr_val[16]     = {1.00,
                                                 1.00,
                                                 4.00,
                                                 2.00,
                                                 4.00,
                                                 1.00,
                                                 2.00,
                                                 1.00,
                                                 8.00,
                                                 2.00,
                                                 4.00,
                                                 1.00,
                                                 3.00,
                                                 6.00,
                                                 2.00,
                                                 1.00};
        T                     dense_rowmajor_gold[25] = {
            1, 1, 0, 0, 4, 0, 2, 4, 0, 1, 2, 1, 8, 2, 0, 0, 0, 4, 1, 0, 3, 6, 2, 0, 1,
        };

        T dense_colmajor_gold[25] = {
            1, 0, 2, 0, 3, 1, 2, 1, 0, 6, 0, 4, 8, 4, 2, 0, 0, 2, 1, 0, 4, 1, 0, 0, 1,
        };
        T dense_rowmajor_gold_ld[40] = {
            1, 1, 0, 0, 4, 0, 0, 0, 0, 2, 4, 0, 1, 0, 0, 0, 2, 1, 8, 2,
            0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 3, 6, 2, 0, 1, 0, 0, 0,
        };

        T dense_mat[25]    = {-1};
        T dense_mat_ld[40] = {0.0};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        base = aoclsparse_index_base_zero;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        //TEST CASE 1: input csr: base-0, output order: row-major
        order = aoclsparse_order_row;
        ld    = N;
        // Convert CSR to Dense
        ASSERT_EQ(aoclsparse_csr2dense(
                      M, N, descr, csr_val, csr_row_ptr, csr_col_ind, dense_mat, ld, order),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR((M * N), dense_mat, dense_rowmajor_gold, expected_precision<T>());

        //TEST CASE 1.1: input csr: base-0, output order: row-major, leading dimension = 8 (matrix's N = 5)
        order = aoclsparse_order_row;
        ld    = N + 3;
        // Convert CSR to Dense
        ASSERT_EQ(aoclsparse_csr2dense(
                      M, N, descr, csr_val, csr_row_ptr, csr_col_ind, dense_mat_ld, ld, order),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR((M * ld), dense_mat_ld, dense_rowmajor_gold_ld, expected_precision<T>());

        //TEST CASE 2: input csr: base-0, output order: column-major
        std::fill(dense_mat, dense_mat + (M * N), -1.0);
        order = aoclsparse_order_column;
        ld    = M;
        // Convert CSR to Dense
        ASSERT_EQ(aoclsparse_csr2dense(
                      M, N, descr, csr_val, csr_row_ptr, csr_col_ind, dense_mat, ld, order),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR((M * N), dense_mat, dense_colmajor_gold, expected_precision<T>());

        //TEST CASE 3: input csr: base-1, output order: row-major
        std::fill(dense_mat, dense_mat + (M * N), -1.0);
        base = aoclsparse_index_base_zero;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        //rebuild indices for 1-based indexing and then call csr2dense
        std::transform(csr_row_ptr, csr_row_ptr + M + 1, csr_row_ptr, [base](aoclsparse_int &d) {
            return d + base;
        });
        std::transform(csr_col_ind, csr_col_ind + NNZ, csr_col_ind, [base](aoclsparse_int &d) {
            return d + base;
        });
        order = aoclsparse_order_row;
        ld    = N;
        // Convert CSR to Dense
        ASSERT_EQ(aoclsparse_csr2dense(
                      M, N, descr, csr_val, csr_row_ptr, csr_col_ind, dense_mat, ld, order),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR((M * N), dense_mat, dense_rowmajor_gold, expected_precision<T>());

        //TEST CASE 4: input csr: base-1, output order: column-major
        std::fill(dense_mat, dense_mat + (M * N), -1.0);
        order = aoclsparse_order_column;
        ld    = M;
        // Convert CSR to Dense
        ASSERT_EQ(aoclsparse_csr2dense(
                      M, N, descr, csr_val, csr_row_ptr, csr_col_ind, dense_mat, ld, order),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR((M * N), dense_mat, dense_colmajor_gold, expected_precision<T>());
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    TEST(convert, ConversionCSR2CSCDouble)
    {
        test_csr_to_csc<double>();
    }
    TEST(convert, ConversionCSR2CSCFloat)
    {
        test_csr_to_csc<float>();
    }
    TEST(convert, ConversionCSR2DenseDouble)
    {
        test_csr_to_dense<double>();
    }
    TEST(convert, ConversionCSR2DenseFloat)
    {
        test_csr_to_dense<float>();
    }

} // namespace
