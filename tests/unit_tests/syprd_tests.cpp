/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

namespace
{
    // Function to cover multiple initializations for both real and complex data types
    template <typename T>
    void init(aoclsparse_operation        &op,
              aoclsparse_order            &order,
              aoclsparse_int              &m,
              aoclsparse_int              &k,
              aoclsparse_int              &nnz,
              std::vector<T>              &csr_val,
              std::vector<aoclsparse_int> &csr_col_ind,
              std::vector<aoclsparse_int> &csr_row_ptr,
              T                           &alpha,
              T                           &beta,
              std::vector<T>              &B,
              std::vector<T>              &C,
              std::vector<T>              &C_exp,
              aoclsparse_int               id,
              aoclsparse_int               b = 0)
    {
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            switch(id)
            {
            // Tests inputs for both square and non-square matrices
            case 0:
                m = 3, k = 2, nnz = 6, alpha = 1, beta = 0;
                csr_val.assign({7, 1, 1, 4, 2, 4});
                csr_row_ptr.assign({0, 2, 4, 6});
                csr_col_ind.assign({0, 1, 0, 1, 0, 1});
                if(op == aoclsparse_operation_none)
                {
                    C.assign({0, 0, 0, 0, 0, 0, 0, 0, 0});
                    B.assign({4, 7, 8, 6});
                }
                else
                {
                    C.assign({0, 0, 0, 0});
                    B.assign({4, 7, 8, 6, 4, 6, 7, 3, 10});
                }

                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({314, 0, 0, 284, 164, 0, 320, 200, 240});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({532, 0, 544, 428});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({300, 255, 290, 0, 156, 188, 0, 0, 224});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({586, 639, 0, 540});
                }
                break;
            case 1:
                m = 2, k = 4, nnz = 6, alpha = 0, beta = 2;
                csr_val.assign({9, 2, 7, 10, 8, 9});
                csr_row_ptr.assign({0, 2, 6});
                csr_col_ind.assign({0, 2, 0, 1, 2, 3});
                if(op == aoclsparse_operation_none)
                {
                    C.assign({1, 2, 3, 2});
                    B.assign({4, 7, 8, 6, 4, 6, 7, 3, 10, 2, 3, 8, 1, 10, 4, 7});
                }
                else
                {
                    C.assign({4, 1, 2, 3, 5, 2, 3, 3, 1, 5, 4, 2, 1, 2, 3, 2});
                    B.assign({4, 7, 8, 6});
                }

                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({2, 2, 6, 4});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({8, 1, 2, 3, 10, 4, 3, 3, 2, 10, 8, 2, 2, 4, 6, 4});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({2, 4, 3, 4});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({8, 2, 4, 6, 5, 4, 6, 6, 1, 5, 8, 4, 1, 2, 3, 4});
                }
                break;
            case 2:
                m = 5, k = 3, nnz = 8, alpha = 3, beta = 2;
                csr_val.assign({8, 1, 9, 1, 7, 8, 3, 4});
                csr_row_ptr.assign({0, 1, 2, 4, 7, 8});
                csr_col_ind.assign({0, 0, 0, 1, 0, 1, 2, 0});
                if(op == aoclsparse_operation_none)
                {
                    C.assign({2, 3, 3, 1, 5, 4, 2, 1, 2, 3, 2, 2, 4,
                              3, 5, 3, 1, 3, 4, 3, 1, 5, 3, 3, 4});
                    B.assign({4, 7, 8, 6, 4, 6, 7, 3, 10});
                }
                else
                {
                    C.assign({1, 3, 4, 3, 1, 5, 3, 3, 4});
                    B.assign({4,  7, 8, 6, 4, 6, 7, 3, 10, 2, 3,  8, 1,
                              10, 4, 7, 1, 7, 3, 7, 2, 9,  8, 10, 3});
                }

                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({772,  3,    3,   1,    5,   104,  16,   1,   2,
                                      3,    1012, 130, 1316, 3,   5,    2334, 293, 2874,
                                      4964, 3,    386, 58,   510, 1170, 200});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({12683, 3, 4, 4716, 917, 5, 1635, 285, 89});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({772,  102,  1038, 2594, 394,  4,   16, 131, 328,
                                      54,   2,    2,    1370, 3219, 526, 3,  1,   3,
                                      5858, 1302, 1,    5,    3,    3,   200});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({14645, 5220, 1781, 3, 1061, 316, 3, 3, 89});
                }
                break;
            case 3:
                m = 4, k = 4, nnz = 12, alpha = 3.5, beta = 2.2;
                csr_val.assign({3, 1, 4, 8, 2, 4, 8, 5, 10, 10, 1, 10});
                csr_row_ptr.assign({0, 3, 4, 8, 12});
                csr_col_ind.assign({0, 1, 2, 1, 0, 1, 2, 3, 0, 1, 2, 3});

                C.assign({1, 2, 3, 2, 2, 4, 3, 5, 3, 1, 3, 4, 3, 1, 5, 3});
                B.assign({4, 7, 8, 6, 4, 6, 7, 3, 10, 2, 3, 8, 1, 10, 4, 7});

                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({1297.2,
                                      2,
                                      3,
                                      2,
                                      732.4,
                                      1352.8,
                                      3,
                                      5,
                                      2502.1,
                                      2746.2,
                                      6065.1,
                                      4,
                                      4045.6,
                                      5658.2,
                                      11323,
                                      17587.1});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({3810.2,
                                      2,
                                      3,
                                      2,
                                      7298.4,
                                      11726.8,
                                      3,
                                      5,
                                      3006.1,
                                      3974.7,
                                      3419.1,
                                      4,
                                      4171.6,
                                      7212.2,
                                      2706,
                                      4119.1});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                    {
                        C_exp.assign({1332.2,
                                      1544.4,
                                      3265.1,
                                      5719.9,
                                      2,
                                      1352.8,
                                      3058.6,
                                      4687,
                                      3,
                                      1,
                                      7619.1,
                                      13056.8,
                                      3,
                                      1,
                                      5,
                                      18777.1});
                    }
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({5336.2,
                                      7312.4,
                                      4682.6,
                                      5569.4,
                                      2,
                                      10508.8,
                                      6964.6,
                                      7361,
                                      3,
                                      1,
                                      3335.1,
                                      4453.8,
                                      3,
                                      1,
                                      5,
                                      5519.1});
                }
                break;
            // Random test input to check wrong size, type and do-nothing case.
            case 6:
                order = aoclsparse_order_row;
                m = 2, k = 3, nnz = 1;
                csr_val.assign({42.});
                csr_col_ind.assign({1});
                csr_row_ptr.assign({0, 0, 1});
                alpha = 2.3, beta = 11.2;
                B.assign({1.0, -2.0, 3.0, 4.0, 5.0, -6.0});
                C.assign({0.1, 0.2, 0.3, 0.4});
                break;
            }
        }
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            switch(id)
            {
            case 0:
                m   = 3;
                k   = 3;
                nnz = 4;
                csr_val.assign({{2, 2}, {5, 2}, {3, 8}, {8, 4}});
                csr_col_ind.assign({1, 0, 0, 1});
                csr_row_ptr.assign({0, 2, 3, 4});
                alpha = {1, 0};
                beta  = {2, 2};

                if(order == aoclsparse_order_column)
                    B.assign({{-1, 0},
                              {-2, 7},
                              {3, 0},
                              {-2, -7},
                              {5, 0},
                              {-6, 0},
                              {3, 0},
                              {-6, 0},
                              {3, 0}});
                if(order == aoclsparse_order_row)
                    B.assign({{-1, 0},
                              {-2, -7},
                              {3, 0},
                              {-2, 7},
                              {5, 0},
                              {-6, 0},
                              {3, 0},
                              {-6, 0},
                              {3, 0}});

                C.assign({{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{-129, 0},
                                      {0, 0},
                                      {0, 0},
                                      {-5, 208},
                                      {-73, 0},
                                      {0, 0},
                                      {-4, -288},
                                      {252, -496},
                                      {400, 0}});
                    if(op == aoclsparse_operation_conjugate_transpose
                       || op == aoclsparse_operation_transpose)
                        C_exp.assign({{688, 0},
                                      {0, 0},
                                      {0, 0},
                                      {-180, 492},
                                      {376, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0}});
                }
                if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{-129, 0},
                                      {-5, 208},
                                      {-4, -288},
                                      {0, 0},
                                      {-73, 0},
                                      {252, -496},
                                      {0, 0},
                                      {0, 0},
                                      {400, 0}});
                    if(op == aoclsparse_operation_conjugate_transpose
                       || op == aoclsparse_operation_transpose)
                        C_exp.assign({{688, 0},
                                      {-180, 492},
                                      {0, 0},
                                      {0, 0},
                                      {376, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0}});
                }
                break;
            case 1:
                m = 3, k = 2, nnz = 4;
                csr_val.assign({{-2, 1}, {3, 2}, {5, -3}, {8, 4}});
                csr_col_ind.assign({1, 0, 0, 1});
                csr_row_ptr.assign({0, 2, 3, 4});
                alpha = {1, 0};
                beta  = {2, 2};
                if(op == aoclsparse_operation_none)
                {
                    if(order == aoclsparse_order_column)
                        B.assign({{1, 0}, {22, 5}, {22, -5}, {-113, 00}});
                    if(order == aoclsparse_order_row)
                        B.assign({{1, 0}, {22, -5}, {22, 5}, {-113, 00}});
                    C.assign(
                        {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
                }
                if(op == aoclsparse_operation_conjugate_transpose
                   || op == aoclsparse_operation_transpose)
                {
                    if(order == aoclsparse_order_column)
                        B.assign({{1, 0},
                                  {22, 5},
                                  {54, -20},
                                  {22, -5},
                                  {7.0, 0},
                                  {-2, -25},
                                  {54, 20},
                                  {-2, 25},
                                  {-1, 0}});
                    if(order == aoclsparse_order_row)
                        B.assign({{1, 0},
                                  {22, -5},
                                  {54, 20},
                                  {22, 5},
                                  {7.0, 0},
                                  {-2, 25},
                                  {54, -20},
                                  {-2, -25},
                                  {-1, 0}});
                    C.assign({{0, 0}, {0, 0}, {0, 0}, {0, 0}});
                }

                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{-798, 0},
                                      {0, 0},
                                      {0, 0},
                                      {-272, -68},
                                      {34, 0},
                                      {0, 0},
                                      {2080, -1880},
                                      {396, -1108},
                                      {-9040, 0}});

                    if(op == aoclsparse_operation_conjugate_transpose
                       || op == aoclsparse_operation_transpose)
                        C_exp.assign({{457, 0}, {0, 0}, {367, 956}, {-731, 0}});
                }
                if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{-798, 0},
                                      {-272, -68},
                                      {2080, -1880},
                                      {0, 0},
                                      {34, 0},
                                      {396, -1108},
                                      {0, 0},
                                      {0, 0},
                                      {-9040, 0}});
                    if(op == aoclsparse_operation_conjugate_transpose
                       || op == aoclsparse_operation_transpose)
                        C_exp.assign({{457, 0}, {367, 956}, {0, 0}, {-731, 0}});
                }
                break;
            }
        }
        for(aoclsparse_int i = 0; i < (aoclsparse_int)csr_row_ptr.size(); i++)
            csr_row_ptr[i] += b;
        for(aoclsparse_int i = 0; i < (aoclsparse_int)csr_col_ind.size(); i++)
            csr_col_ind[i] += b;
    }

    // Function to set the ldb, ldc and find dimensions for matrices B & C
    void set_syprd_dim(aoclsparse_operation &op,
                       aoclsparse_int       &m,
                       aoclsparse_int       &k,
                       aoclsparse_int       &A_m,
                       aoclsparse_int       &A_n,
                       aoclsparse_int       &B_m,
                       aoclsparse_int       &B_n,
                       aoclsparse_int       &C_m,
                       aoclsparse_int       &C_n,
                       aoclsparse_int       &ldb,
                       aoclsparse_int       &ldc)
    {
        A_m = (op == aoclsparse_operation_none ? m : k);
        A_n = (op == aoclsparse_operation_none ? k : m);
        B_m = A_n;
        B_n = B_m;
        C_m = A_m;
        C_n = C_m;
        ldb = op == aoclsparse_operation_none ? B_m : B_n;
        ldc = op == aoclsparse_operation_none ? C_m : C_n;
    }

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_syprd_nullptr()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

        init<T>(
            op, order, m, k, nnz, csr_val, csr_col_ind, csr_row_ptr, alpha, beta, B, C, C_exp, 0);
        // Set values of ldb, ldc and matrix dimenstions of C matrix
        set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_syprd<T>(
                      op, nullptr, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_syprd<T>(op, A, nullptr, order, ldb, alpha, beta, C.data(), order, ldc),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_syprd<T>(op, A, B.data(), order, ldb, alpha, beta, nullptr, order, ldc),
            aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(&A);
    }

    // tests for Wrong size
    template <typename T>
    void test_syprd_wrong_size()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

        init<T>(
            op, order, m, k, nnz, csr_val, csr_col_ind, csr_row_ptr, alpha, beta, B, C, C_exp, 6);
        // Set values of ldb, ldc and matrix dimenstions of C matrix
        set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        // expect invalid size for wrong ldb
        EXPECT_EQ(
            aoclsparse_syprd<T>(op, A, B.data(), order, k - 1, alpha, beta, C.data(), order, ldc),
            aoclsparse_status_invalid_size);

        // expect invalid size for wrong ldc
        EXPECT_EQ(
            aoclsparse_syprd<T>(op, A, B.data(), order, ldb, alpha, beta, C.data(), order, m - 1),
            aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(&A);
    }

    // tests to check invalid operation
    template <typename T>
    void test_syprd_invalid_operation()
    {
        aoclsparse_int              m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        aoclsparse_operation        op;
        aoclsparse_order            order = aoclsparse_order_row;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;

        init<T>(
            op, order, m, k, nnz, csr_val, csr_col_ind, csr_row_ptr, alpha, beta, B, C, C_exp, 0);
        aoclsparse_mat_descr descr;
        // Set values of ldb, ldc and matrix dimenstions of C matrix
        set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
        C.resize(C_m * C_n);
        B.resize(B_m * B_n);

        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        aoclsparse_matrix A;
        ASSERT_EQ(
            aoclsparse_create_csr(
                &A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), (T *)csr_val.data()),
            aoclsparse_status_success);

        // Expect to return invalid operation when matrix B and C have different ordering
        op = aoclsparse_operation_none;
        EXPECT_EQ(
            aoclsparse_syprd<T>(
                op, A, B.data(), order, ldb, alpha, beta, C.data(), aoclsparse_order_column, ldc),
            aoclsparse_status_invalid_operation);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(&A);
    }

    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_syprd_do_nothing()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

        init<T>(
            op, order, m, k, nnz, csr_val, csr_col_ind, csr_row_ptr, alpha, beta, B, C, C_exp, 6);
        // Set values of ldb, ldc and matrix dimenstions of C matrix
        set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
        aoclsparse_int       csr_row_ptr_zeros[] = {0, 0, 0};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        // expect success for m=0
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, 0, k, 0, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_syprd<T>(op, A, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
            aoclsparse_status_success);
        aoclsparse_destroy(&A);

        // expect success for k=0
        C_exp.assign({1.12, 2.24, 0.3, 4.48}); //For k = 0, expect to return (beta * C)
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, m, 0, 0, csr_row_ptr_zeros, csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_syprd<T>(op, A, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
            aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(C_m * C_n, C, C_exp);
        aoclsparse_destroy(&A);

        // expect success for alpha = 0 & beta = 1
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);
        alpha = 0.0;
        beta  = 1.0;
        EXPECT_EQ(
            aoclsparse_syprd<T>(op, A, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
            aoclsparse_status_success);

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }

    // tests for ldb and ldc greater than minimum
    template <typename T>
    void test_syprd_greater_ld()
    {
        aoclsparse_index_base       base  = aoclsparse_index_base_zero;
        aoclsparse_operation        op    = aoclsparse_operation_none;
        aoclsparse_order            order = aoclsparse_order_column;
        aoclsparse_int              m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

        init<T>(
            op, order, m, k, nnz, csr_val, csr_col_ind, csr_row_ptr, alpha, beta, B, C, C_exp, 0);
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        B.assign({2, 3, 3, 1, 5, 4, 2, 1, 2, 3, 2, 2, 4, 3, 5, 3, 1, 3, 4, 3, 1, 5, 3, 3, 4});
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);
        for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
        {
            // Set values of ldb, ldc and matrix dimenstions of C matrix
            set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
            C = B;
            if(order == aoclsparse_order_row)
                C_exp.assign({144, 117, 134, 1, 5, 4, 90, 104, 2, 3, 2, 2, 120,
                              3,   5,   3,   1, 3, 4, 3,  1,   5, 3, 3, 4});
            else
                C_exp.assign({172, 3, 3, 1, 5, 175, 106, 1, 2, 3, 194, 128, 152,
                              3,   5, 3, 1, 3, 4,   3,   1, 5, 3, 3,   4});

            // expect success for ldb = ldb+2 and ldc = ldc+2
            EXPECT_EQ(aoclsparse_syprd(
                          op, A, B.data(), order, ldb + 2, alpha, beta, C.data(), order, ldc + 2),
                      aoclsparse_status_success);
            if constexpr(std::is_same_v<T, double>)
                EXPECT_DOUBLE_EQ_VEC(C_exp.size(), C, C_exp);
            if constexpr(std::is_same_v<T, float>)
                EXPECT_FLOAT_EQ_VEC(C_exp.size(), C, C_exp);
        }
        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(&A);
    }

    // tests for Wrong type
    template <typename T>
    void test_syprd_wrongtype()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

        init<T>(
            op, order, m, k, nnz, csr_val, csr_col_ind, csr_row_ptr, alpha, beta, B, C, C_exp, 6);
        // Set values of ldb, ldc and matrix dimenstions of C matrix
        set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        aoclsparse_matrix A;
        ASSERT_EQ(
            aoclsparse_create_csr(
                &A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), (T *)csr_val.data()),
            aoclsparse_status_success);
        if constexpr(std::is_same_v<T, double>)
        {
            // expect wrong type error for invoking syprd for single precision with double csr_val
            EXPECT_EQ(aoclsparse_ssyprd(op,
                                        A,
                                        (float *)B.data(),
                                        order,
                                        ldb,
                                        alpha,
                                        beta,
                                        (float *)C.data(),
                                        order,
                                        ldc),
                      aoclsparse_status_wrong_type);

            aoclsparse_destroy_mat_descr(descr);
            aoclsparse_destroy(&A);
        }

        if constexpr(std::is_same_v<T, float>)
        {
            // expect wrong type error for invoking syprd for double precision with float csr_val
            EXPECT_EQ(aoclsparse_dsyprd(op,
                                        A,
                                        (double *)B.data(),
                                        order,
                                        ldb,
                                        alpha,
                                        beta,
                                        (double *)C.data(),
                                        order,
                                        ldc),
                      aoclsparse_status_wrong_type);

            aoclsparse_destroy_mat_descr(descr);
            aoclsparse_destroy(&A);
        }
    }

    // test for success cases
    template <typename T>
    void test_syprd_success(aoclsparse_index_base base = aoclsparse_index_base_zero)
    {
        aoclsparse_int              m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;
        aoclsparse_int              b = 0;

        if(base == aoclsparse_index_base_zero)
            b = 0;
        else
            b = 1;

        //Test for real types
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            for(int id = 0; id < 4; id++)
            {
                for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                {
                    for(aoclsparse_operation op :
                        {aoclsparse_operation_none, aoclsparse_operation_transpose})
                    {
                        //Initialize inputs for test
                        init<T>(op,
                                order,
                                m,
                                k,
                                nnz,
                                csr_val,
                                csr_col_ind,
                                csr_row_ptr,
                                alpha,
                                beta,
                                B,
                                C,
                                C_exp,
                                id,
                                b);

                        // Set values of ldb, ldc and matrix dimenstions of C matrix
                        set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

                        aoclsparse_mat_descr descr;
                        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
                        // and test aoclsparse_index_base to both aoclsparse_index_base_zero, aoclsparse_index_base_one.
                        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
                        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base),
                                  aoclsparse_status_success);

                        aoclsparse_matrix A;
                        ASSERT_EQ(aoclsparse_create_csr(&A,
                                                        base,
                                                        m,
                                                        k,
                                                        nnz,
                                                        csr_row_ptr.data(),
                                                        csr_col_ind.data(),
                                                        csr_val.data()),
                                  aoclsparse_status_success);
                        EXPECT_EQ(
                            aoclsparse_syprd(
                                op, A, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
                            aoclsparse_status_success);
                        if constexpr(std::is_same_v<T, double>)
                            EXPECT_DOUBLE_EQ_VEC(C_m * C_n, C, C_exp);
                        if constexpr(std::is_same_v<T, float>)
                            EXPECT_FLOAT_EQ_VEC(C_m * C_n, C, C_exp);

                        aoclsparse_destroy_mat_descr(descr);
                        aoclsparse_destroy(&A);
                    }
                }
            }
        }
        //Test for complex types
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            for(int id = 0; id < 2; id++)
            {
                for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                {
                    for(aoclsparse_operation op :
                        {aoclsparse_operation_none, aoclsparse_operation_conjugate_transpose})

                    {
                        //Initialize inputs for test
                        init<T>(op,
                                order,
                                m,
                                k,
                                nnz,
                                csr_val,
                                csr_col_ind,
                                csr_row_ptr,
                                alpha,
                                beta,
                                B,
                                C,
                                C_exp,
                                id,
                                b);

                        // Set values of ldb, ldc and matrix dimenstions of C matrix
                        set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
                        C.resize(C_m * C_n);

                        aoclsparse_mat_descr descr;
                        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
                        // and test aoclsparse_index_base to both aoclsparse_index_base_zero, aoclsparse_index_base_one.
                        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
                        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base),
                                  aoclsparse_status_success);
                        aoclsparse_matrix A;

                        ASSERT_EQ(aoclsparse_create_csr(&A,
                                                        base,
                                                        m,
                                                        k,
                                                        nnz,
                                                        csr_row_ptr.data(),
                                                        csr_col_ind.data(),
                                                        (T *)csr_val.data()),
                                  aoclsparse_status_success);
                        EXPECT_EQ(
                            aoclsparse_syprd(
                                op, A, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
                            aoclsparse_status_success);
                        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
                        {
                            std::vector<std::complex<float>> *res, *res_exp;
                            res     = (std::vector<std::complex<float>> *)&C;
                            res_exp = (std::vector<std::complex<float>> *)&C_exp;
                            EXPECT_COMPLEX_FLOAT_EQ_VEC(C_m * C_n, (*res), (*res_exp));
                        }
                        if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
                        {
                            std::vector<std::complex<double>> *res, *res_exp;
                            res     = (std::vector<std::complex<double>> *)&C;
                            res_exp = (std::vector<std::complex<double>> *)&C_exp;
                            EXPECT_COMPLEX_DOUBLE_EQ_VEC(C_m * C_n, (*res), (*res_exp));
                        }

                        aoclsparse_destroy_mat_descr(descr);
                        aoclsparse_destroy(&A);
                    }
                }
            }
        }
    }

    TEST(syprd, NullArgDouble)
    {
        test_syprd_nullptr<double>();
    }
    TEST(syprd, NullArgFloat)
    {
        test_syprd_nullptr<float>();
    }
    TEST(syprd, NullArgComplexDouble)
    {
        test_syprd_nullptr<aoclsparse_double_complex>();
    }
    TEST(syprd, NullArgComplexFloat)
    {
        test_syprd_nullptr<aoclsparse_float_complex>();
    }
    TEST(syprd, WrongSizeDouble)
    {
        test_syprd_wrong_size<double>();
    }
    TEST(syprd, WrongSizeFloat)
    {
        test_syprd_wrong_size<float>();
    }

    TEST(syprd, InvalidOpComplexDouble)
    {
        test_syprd_invalid_operation<aoclsparse_double_complex>();
    }
    TEST(syprd, InvalidOpComplexFloat)
    {
        test_syprd_invalid_operation<aoclsparse_float_complex>();
    }

    TEST(syprd, DoNothingDouble)
    {
        test_syprd_do_nothing<double>();
    }
    TEST(syprd, DoNothingFloat)
    {
        test_syprd_do_nothing<float>();
    }

    TEST(syprd, GreaterLDDouble)
    {
        test_syprd_greater_ld<double>();
    }
    TEST(syprd, GreaterLDFloat)
    {
        test_syprd_greater_ld<float>();
    }
    TEST(syprd, WrongTypeDouble)
    {
        test_syprd_wrongtype<double>();
    }
    TEST(syprd, WrongTypeFloat)
    {
        test_syprd_wrongtype<float>();
    }
    TEST(syprd, SuccessTypeDouble)
    {
        test_syprd_success<double>();
        test_syprd_success<double>(aoclsparse_index_base_one);
    }
    TEST(syprd, SuccessTypeFloat)
    {
        test_syprd_success<float>();
        test_syprd_success<float>(aoclsparse_index_base_one);
    }
    TEST(syprd, SuccessTypeComplexFloat)
    {
        test_syprd_success<aoclsparse_float_complex>();
        test_syprd_success<aoclsparse_float_complex>(aoclsparse_index_base_one);
    }

    TEST(syprd, SuccessTypeComplexDouble)
    {
        test_syprd_success<aoclsparse_double_complex>();
        test_syprd_success<aoclsparse_double_complex>(aoclsparse_index_base_one);
    }
} // namespace
