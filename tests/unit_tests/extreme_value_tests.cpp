/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_reference.hpp"
#include "aoclsparse_utility.hpp"

namespace
{

    template <typename T>
    void init(aoclsparse_int              &A_m,
              aoclsparse_int              &A_nnz,
              aoclsparse_int              &B_m,
              aoclsparse_int              &B_nnz,
              std::vector<T>              &A_val,
              std::vector<aoclsparse_int> &A_col_ind,
              std::vector<aoclsparse_int> &A_row_ptr,
              std::vector<T>              &B_val,
              std::vector<aoclsparse_int> &B_col_ind,
              std::vector<aoclsparse_int> &B_row_ptr,
              std::vector<T>              &B_dense)
    {
        A_m = B_m = 7;
        A_nnz     = 17;
        B_nnz     = 21;
        A_val.assign({aoclsparse_numeric::quiet_NaN<T>(),
                      1.000000e+00,
                      1.000000e+00,
                      aoclsparse_numeric::infinity<T>(),
                      1.000000e+00,
                      aoclsparse_numeric::infinity<T>(),
                      1.000000e+00,
                      aoclsparse_numeric::infinity<T>(),
                      1.000000e+00,
                      1.000000e+00,
                      (std::numeric_limits<double>::max)(),
                      (std::numeric_limits<double>::max)(),
                      1.000000e+00,
                      1.000000e+00,
                      1.000000e+00,
                      (std::numeric_limits<double>::min)(),
                      1.000000e+00});

        A_col_ind.assign({0, 2, 1, 0, 1, 2, 4, 2, 3, 4, 0, 2, 4, 5, 6, 0, 6});
        A_row_ptr.assign({0, 2, 3, 7, 10, 13, 15, 17});
        B_val.assign({1.000000e+00, 1.000000e+00, 8.988500e-24, 1.000000e+00,  1.000000e+00,
                      1.000000e+00, 1.000000e+00, 1.000000e+00, -1.000000e+00, 1.000000e+00,
                      1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00,  8.988500e+10,
                      1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00,  1.000000e+00,
                      1.000000e+00});
        B_col_ind.assign({0, 2, 4, 5, 1, 2, 4, 0, 2, 1, 2, 3, 0, 1, 2, 4, 6, 1, 5, 0, 3});
        B_dense.assign({1.000000e+00, 0.000000e+00,  1.000000e+00, 0.000000e+00, 8.988500e-24,
                        1.000000e+00, 0.000000e+00,  0.000000e+00, 1.000000e+00, 1.000000e+00,
                        0.000000e+00, 1.000000e+00,  0.000000e+00, 0.000000e+00, 1.000000e+00,
                        0.000000e+00, -1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00,  1.000000e+00, 1.000000e+00, 1.000000e+00,
                        0.000000e+00, 0.000000e+00,  0.000000e+00, 1.000000e+00, 1.000000e+00,
                        8.988500e+10, 0.000000e+00,  1.000000e+00, 0.000000e+00, 1.000000e+00,
                        0.000000e+00, 1.000000e+00,  0.000000e+00, 0.000000e+00, 0.000000e+00,
                        1.000000e+00, 0.000000e+00,  1.000000e+00, 0.000000e+00, 0.000000e+00,
                        1.000000e+00, 0.000000e+00,  0.000000e+00, 0.000000e+00});
        B_row_ptr.assign({0, 4, 7, 9, 12, 17, 19, 21});
    }

    template <typename T>
    void test_ev_add()
    {
        std::vector<T>              A_val;
        std::vector<aoclsparse_int> A_col_ind;
        std::vector<aoclsparse_int> A_row_ptr;
        std::vector<T>              B_val;
        std::vector<aoclsparse_int> B_col_ind;
        std::vector<aoclsparse_int> B_row_ptr;
        aoclsparse_int              A_m, A_nnz, B_m, B_nnz;
        aoclsparse_matrix           A_mat = nullptr, B_mat = nullptr, C_mat = nullptr;
        std::vector<T>              C_exp_val;
        std::vector<aoclsparse_int> C_exp_col_ind;
        std::vector<aoclsparse_int> C_exp_row_ptr;
        std::vector<T>              B_dense;
        C_exp_val.assign({aoclsparse_numeric::quiet_NaN<T>(),
                          2.000000e+00,
                          8.988500e-24,
                          1.000000e+00,
                          2.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          aoclsparse_numeric::infinity<T>(),
                          1.000000e+00,
                          aoclsparse_numeric::infinity<T>(),
                          1.000000e+00,
                          1.000000e+00,
                          aoclsparse_numeric::infinity<T>(),
                          2.000000e+00,
                          1.000000e+00,
                          (std::numeric_limits<double>::max)(),
                          1.000000e+00,
                          (std::numeric_limits<double>::max)(),
                          2.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          2.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          1.000000e+00});
        C_exp_col_ind.assign(
            {0, 2, 4, 5, 1, 2, 4, 0, 1, 2, 4, 1, 2, 3, 4, 0, 1, 2, 4, 6, 1, 5, 6, 0, 3, 6});
        C_exp_row_ptr.assign({0, 4, 7, 11, 15, 20, 23, 26});
        init(A_m,
             A_nnz,
             B_m,
             B_nnz,
             A_val,
             A_col_ind,
             A_row_ptr,
             B_val,
             B_col_ind,
             B_row_ptr,
             B_dense);
        ASSERT_EQ(aoclsparse_create_csr(&A_mat,
                                        aoclsparse_index_base_zero,
                                        A_m,
                                        A_m,
                                        A_nnz,
                                        A_row_ptr.data(),
                                        A_col_ind.data(),
                                        A_val.data()),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr(&B_mat,
                                        aoclsparse_index_base_zero,
                                        B_m,
                                        B_m,
                                        B_nnz,
                                        B_row_ptr.data(),
                                        B_col_ind.data(),
                                        B_val.data()),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_none, A_mat, 1., B_mat, &C_mat),
                  aoclsparse_status_success);
        std::vector<aoclsparse_int> row_ptr, col_ind;
        std::vector<T>              val;
        aoclsparse_int              n, m, nnz;
        aoclsparse_index_base       b;

        EXPECT_EQ(aocl_csr_sorted_export(C_mat, b, m, n, nnz, row_ptr, col_ind, val),
                  aoclsparse_status_success);

        double         tol       = 1;
        tolerance_t<T> abs_error = expected_precision<decltype(tol)>(tol);
        // As this test is related to extreme value functionality, we only compare values
        // and assume rest of the CSR arrays are correct
        EXPECT_ARR_MATCH(T, nnz, &val[0], &C_exp_val[0], abs_error, abs_error);
        EXPECT_EQ(aoclsparse_destroy(&A_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&B_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&C_mat), aoclsparse_status_success);
    }

    template <typename T>
    void test_ev_sp2m()
    {
        std::vector<T>              A_val;
        std::vector<aoclsparse_int> A_col_ind;
        std::vector<aoclsparse_int> A_row_ptr;
        std::vector<T>              B_val;
        std::vector<aoclsparse_int> B_col_ind;
        std::vector<aoclsparse_int> B_row_ptr;
        aoclsparse_int              A_m, A_nnz, B_m, B_nnz;
        aoclsparse_matrix           A_mat = nullptr, B_mat = nullptr, C_mat = nullptr;
        std::vector<T>              C_exp_val;
        std::vector<aoclsparse_int> C_exp_col_ind;
        std::vector<aoclsparse_int> C_exp_row_ptr;
        std::vector<T>              B_dense;

        T tmp = /*1.615856e+285; */ (std::numeric_limits<double>::max)() * 8.988500e-24
                + 1.000000e+00;
        C_exp_val.assign({aoclsparse_numeric::quiet_NaN<T>(),
                          aoclsparse_numeric::quiet_NaN<T>(),
                          aoclsparse_numeric::quiet_NaN<T>(),
                          aoclsparse_numeric::quiet_NaN<T>(),
                          1.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          aoclsparse_numeric::infinity<T>(),
                          2.000000e+00,
                          aoclsparse_numeric::quiet_NaN<T>(),
                          aoclsparse_numeric::infinity<T>(),
                          aoclsparse_numeric::infinity<T>(),
                          1.000000e+00,
                          aoclsparse_numeric::infinity<T>(),
                          2.000000e+00,
                          -aoclsparse_numeric::infinity<T>(),
                          1.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          aoclsparse_numeric::infinity<T>(),
                          1.000000e+00,
                          8.988500e+10,
                          tmp,
                          (std::numeric_limits<double>::max)(),
                          1.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          1.000000e+00,
                          (std::numeric_limits<double>::min)(),
                          1.000000e+00,
                          0.000000e+00,
                          (std::numeric_limits<double>::min)()});
        C_exp_col_ind.assign({0, 2, 4, 5, 1, 2, 4, 0, 1, 2, 4, 5, 6, 0, 1, 2, 3,
                              4, 6, 0, 1, 2, 4, 5, 6, 0, 1, 3, 5, 0, 2, 3, 5});
        C_exp_row_ptr.assign({0, 4, 7, 13, 19, 25, 29, 33});

        init(A_m,
             A_nnz,
             B_m,
             B_nnz,
             A_val,
             A_col_ind,
             A_row_ptr,
             B_val,
             B_col_ind,
             B_row_ptr,
             B_dense);
        ASSERT_EQ(aoclsparse_create_csr(&A_mat,
                                        aoclsparse_index_base_zero,
                                        A_m,
                                        A_m,
                                        A_nnz,
                                        A_row_ptr.data(),
                                        A_col_ind.data(),
                                        A_val.data()),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr(&B_mat,
                                        aoclsparse_index_base_zero,
                                        B_m,
                                        B_m,
                                        B_nnz,
                                        B_row_ptr.data(),
                                        B_col_ind.data(),
                                        B_val.data()),
                  aoclsparse_status_success);

        aoclsparse_mat_descr descrA;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        aoclsparse_mat_descr descrB;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        aoclsparse_request request = aoclsparse_stage_full_computation;
        EXPECT_EQ(aoclsparse_sp2m(aoclsparse_operation_none,
                                  descrA,
                                  A_mat,
                                  aoclsparse_operation_none,
                                  descrB,
                                  B_mat,
                                  request,
                                  &C_mat),
                  aoclsparse_status_success);

        std::vector<aoclsparse_int> row_ptr, col_ind;
        std::vector<T>              val;
        aoclsparse_int              n, m, nnz;
        aoclsparse_index_base       b;

        EXPECT_EQ(aocl_csr_sorted_export(C_mat, b, m, n, nnz, row_ptr, col_ind, val),
                  aoclsparse_status_success);

        double         tol       = 1;
        tolerance_t<T> abs_error = expected_precision<decltype(tol)>(tol);

        // As this test is related to extreme value functionality, we only compare values
        // and assume rest of the CSR arrays are correct
        EXPECT_ARR_MATCH(T, nnz, &C_exp_val[0], &val[0], abs_error, abs_error);

        EXPECT_EQ(aoclsparse_destroy(&A_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&B_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&C_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descrB), aoclsparse_status_success);
    }

    TEST(add, EVDouble)
    {
        test_ev_add<double>();
    }
    TEST(sp2m, EVDouble)
    {
        test_ev_sp2m<double>();
    }
} // namespace
