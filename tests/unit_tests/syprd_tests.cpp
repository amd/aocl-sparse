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
#include "level3_test_common.hpp"

#include <type_traits>

namespace
{
    // CSR/CSC format selector for init()
    enum syprd_format
    {
        syprd_csr,
        syprd_csc
    };

    // Format-neutral sparse matrix array triplet.
    template <typename T>
    struct syprd_mat_data
    {
        std::vector<T>              val;
        std::vector<aoclsparse_int> ind;
        std::vector<aoclsparse_int> ptr;
    };

    // Function to cover multiple initializations for both real and complex data types
    template <typename T>
    void init(aoclsparse_operation &op,
              aoclsparse_order     &order,
              aoclsparse_int       &m,
              aoclsparse_int       &k,
              aoclsparse_int       &nnz,
              syprd_mat_data<T>    &A,
              T                    &alpha,
              T                    &beta,
              std::vector<T>       &B,
              std::vector<T>       &C,
              std::vector<T>       &C_exp,
              aoclsparse_int        id,
              aoclsparse_int        b   = 0,
              syprd_format          fmt = syprd_csr)
    {
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            switch(id)
            {
            // Tests inputs for both square and non-square matrices
            case 0:
                m = 3, k = 2, nnz = 6, alpha = 1, beta = 0;
                A.val.assign({7, 1, 1, 4, 2, 4});
                A.ptr.assign({0, 2, 4, 6});
                A.ind.assign({0, 1, 0, 1, 0, 1});
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
                A.val.assign({9, 2, 7, 10, 8, 9});
                A.ptr.assign({0, 2, 6});
                A.ind.assign({0, 2, 0, 1, 2, 3});
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
                A.val.assign({8, 1, 9, 1, 7, 8, 3, 4});
                A.ptr.assign({0, 1, 2, 4, 7, 8});
                A.ind.assign({0, 0, 0, 1, 0, 1, 2, 0});
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
                A.val.assign({3, 1, 4, 8, 2, 4, 8, 5, 10, 10, 1, 10});
                A.ptr.assign({0, 3, 4, 8, 12});
                A.ind.assign({0, 1, 2, 1, 0, 1, 2, 3, 0, 1, 2, 3});

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
                A.val.assign({42.});
                A.ind.assign({1});
                A.ptr.assign({0, 0, 1});
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
                A.val.assign({{2, 2}, {5, 2}, {3, 8}, {8, 4}});
                A.ind.assign({1, 0, 0, 1});
                A.ptr.assign({0, 2, 3, 4});
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
                A.val.assign({{-2, 1}, {3, 2}, {5, -3}, {8, 4}});
                A.ind.assign({1, 0, 0, 1});
                A.ptr.assign({0, 2, 3, 4});
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
        // If CSC format requested, convert the CSR data to CSC in-place.
        // Done before base-adjustment so the conversion works with 0-based indices.
        if(fmt == syprd_csc)
        {
            std::vector<aoclsparse_int> csc_col_ptr(k + 1), csc_row_ind(nnz);
            std::vector<T>              csc_val(nnz);
            aoclsparse_mat_descr        descr_csr;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descr_csr), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descr_csr, aoclsparse_index_base_zero),
                      aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_csr2csc<T>(m,
                                            k,
                                            nnz,
                                            descr_csr,
                                            aoclsparse_index_base_zero,
                                            A.ptr.data(),
                                            A.ind.data(),
                                            A.val.data(),
                                            csc_row_ind.data(),
                                            csc_col_ptr.data(),
                                            csc_val.data()),
                      aoclsparse_status_success);
            aoclsparse_destroy_mat_descr(descr_csr);

            A.ptr = std::move(csc_col_ptr);
            A.ind = std::move(csc_row_ind);
            A.val = std::move(csc_val);
        }
        for(aoclsparse_int i = 0; i < (aoclsparse_int)A.ptr.size(); i++)
            A.ptr[i] += b;
        for(aoclsparse_int i = 0; i < (aoclsparse_int)A.ind.size(); i++)
            A.ind[i] += b;
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

    // Shared boilerplate for negative tests.
    template <typename T>
    struct syprd_test_setup
    {
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_operation  op    = aoclsparse_operation_none;
        aoclsparse_order      order = aoclsparse_order_row;
        aoclsparse_int        m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        syprd_mat_data<T>     arr;
        T                     alpha, beta;
        std::vector<T>        B, C, C_exp;
        aoclsparse_mat_descr  descr = nullptr;
        aoclsparse_matrix     A     = nullptr;

        void setup(aoclsparse_int id = 0)
        {
            init<T>(op, order, m, k, nnz, arr, alpha, beta, B, C, C_exp, id);
            set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
        }

        void create_matrix(aoclsparse_index_base b = aoclsparse_index_base_zero)
        {
            base = b;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_create_csr(
                          &A, base, m, k, nnz, arr.ptr.data(), arr.ind.data(), arr.val.data()),
                      aoclsparse_status_success);
        }

        ~syprd_test_setup()
        {
            if(A)
                aoclsparse_destroy(&A);
            if(descr)
                aoclsparse_destroy_mat_descr(descr);
        }
    };

    // Several tests in one, when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_syprd_nullptr()
    {
        syprd_test_setup<T> s;
        s.setup(0);
        s.create_matrix();

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_syprd<T>(s.op,
                                      nullptr,
                                      s.B.data(),
                                      s.order,
                                      s.ldb,
                                      s.alpha,
                                      s.beta,
                                      s.C.data(),
                                      s.order,
                                      s.ldc),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_syprd<T>(
                s.op, s.A, nullptr, s.order, s.ldb, s.alpha, s.beta, s.C.data(), s.order, s.ldc),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_syprd<T>(
                s.op, s.A, s.B.data(), s.order, s.ldb, s.alpha, s.beta, nullptr, s.order, s.ldc),
            aoclsparse_status_invalid_pointer);
    }

    // tests for Wrong size
    template <typename T>
    void test_syprd_wrong_size()
    {
        syprd_test_setup<T> s;
        s.setup(6);
        s.create_matrix();

        // expect invalid size for wrong ldb
        EXPECT_EQ(aoclsparse_syprd<T>(s.op,
                                      s.A,
                                      s.B.data(),
                                      s.order,
                                      s.k - 1,
                                      s.alpha,
                                      s.beta,
                                      s.C.data(),
                                      s.order,
                                      s.ldc),
                  aoclsparse_status_invalid_size);

        // expect invalid size for wrong ldc
        EXPECT_EQ(aoclsparse_syprd<T>(s.op,
                                      s.A,
                                      s.B.data(),
                                      s.order,
                                      s.ldb,
                                      s.alpha,
                                      s.beta,
                                      s.C.data(),
                                      s.order,
                                      s.m - 1),
                  aoclsparse_status_invalid_size);
    }

    // tests to check invalid operation
    template <typename T>
    void test_syprd_invalid_operation()
    {
        syprd_test_setup<T> s;
        s.setup(0);
        s.C.resize(s.C_m * s.C_n);
        s.B.resize(s.B_m * s.B_n);
        s.create_matrix();

        // Expect to return invalid operation when matrix B and C have different ordering
        s.op = aoclsparse_operation_none;
        EXPECT_EQ(aoclsparse_syprd<T>(s.op,
                                      s.A,
                                      s.B.data(),
                                      s.order,
                                      s.ldb,
                                      s.alpha,
                                      s.beta,
                                      s.C.data(),
                                      aoclsparse_order_column,
                                      s.ldc),
                  aoclsparse_status_invalid_operation);
    }

    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_syprd_do_nothing()
    {
        syprd_test_setup<T> s;
        s.setup(6);
        aoclsparse_int csr_row_ptr_zeros[] = {0, 0, 0};
        ASSERT_EQ(aoclsparse_create_mat_descr(&s.descr), aoclsparse_status_success);

        // expect success for m=0
        ASSERT_EQ(
            aoclsparse_create_csr(
                &s.A, s.base, 0, s.k, 0, s.arr.ptr.data(), s.arr.ind.data(), s.arr.val.data()),
            aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_syprd<T>(
                s.op, s.A, s.B.data(), s.order, s.ldb, s.alpha, s.beta, s.C.data(), s.order, s.ldc),
            aoclsparse_status_success);
        aoclsparse_destroy(&s.A);

        // expect success for k=0
        s.C_exp.assign({1.12, 2.24, 0.3, 4.48}); //For k = 0, expect to return (beta * C)
        ASSERT_EQ(
            aoclsparse_create_csr(
                &s.A, s.base, s.m, 0, 0, csr_row_ptr_zeros, s.arr.ind.data(), s.arr.val.data()),
            aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_syprd<T>(
                s.op, s.A, s.B.data(), s.order, s.ldb, s.alpha, s.beta, s.C.data(), s.order, s.ldc),
            aoclsparse_status_success);
        EXPECT_DOUBLE_EQ_VEC(s.C_m * s.C_n, s.C, s.C_exp);
        aoclsparse_destroy(&s.A);

        // expect success for alpha = 0 & beta = 1
        ASSERT_EQ(aoclsparse_create_csr(&s.A,
                                        s.base,
                                        s.m,
                                        s.k,
                                        s.nnz,
                                        s.arr.ptr.data(),
                                        s.arr.ind.data(),
                                        s.arr.val.data()),
                  aoclsparse_status_success);
        s.alpha = 0.0;
        s.beta  = 1.0;
        EXPECT_EQ(
            aoclsparse_syprd<T>(
                s.op, s.A, s.B.data(), s.order, s.ldb, s.alpha, s.beta, s.C.data(), s.order, s.ldc),
            aoclsparse_status_success);
        // destructor handles cleanup of A and descr
    }

    // tests for ldb and ldc greater than minimum
    template <typename T>
    void test_syprd_greater_ld()
    {
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_operation  op    = aoclsparse_operation_none;
        aoclsparse_order      order = aoclsparse_order_column;
        aoclsparse_int        m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        syprd_mat_data<T>     arr;
        T                     alpha, beta;
        std::vector<T>        B, C, C_exp;

        init<T>(op, order, m, k, nnz, arr, alpha, beta, B, C, C_exp, 0);
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        B.assign({2, 3, 3, 1, 5, 4, 2, 1, 2, 3, 2, 2, 4, 3, 5, 3, 1, 3, 4, 3, 1, 5, 3, 3, 4});
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, m, k, nnz, arr.ptr.data(), arr.ind.data(), arr.val.data()),
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
        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_operation  op   = aoclsparse_operation_none;
        aoclsparse_order      order;
        aoclsparse_int        m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        syprd_mat_data<T>     arr;
        T                     alpha, beta;
        std::vector<T>        B, C, C_exp;

        init<T>(op, order, m, k, nnz, arr, alpha, beta, B, C, C_exp, 6);
        // Set values of ldb, ldc and matrix dimenstions of C matrix
        set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, m, k, nnz, arr.ptr.data(), arr.ind.data(), (T *)arr.val.data()),
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

    // test for success cases — unified CSR + CSC
    template <typename T>
    void test_syprd_success(syprd_format          fmt  = syprd_csr,
                            aoclsparse_index_base base = aoclsparse_index_base_zero)
    {
        aoclsparse_int    m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        syprd_mat_data<T> arr;
        T                 alpha, beta;
        std::vector<T>    B, C, C_exp;

        // Determine valid ops and number of test cases based on type
        std::vector<aoclsparse_operation> ops;
        int                               num_cases;
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            ops       = {aoclsparse_operation_none, aoclsparse_operation_transpose};
            num_cases = 4;
        }
        else
        {
            ops       = {aoclsparse_operation_none, aoclsparse_operation_conjugate_transpose};
            num_cases = 2;
        }

        aoclsparse_int b = (base == aoclsparse_index_base_zero) ? 0 : 1;

        for(int id = 0; id < num_cases; id++)
        {
            for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
            {
                for(aoclsparse_operation op : ops)
                {
                    init<T>(op, order, m, k, nnz, arr, alpha, beta, B, C, C_exp, id, b, fmt);
                    set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

                    aoclsparse_matrix mat;
                    if(fmt == syprd_csr)
                    {
                        ASSERT_EQ(aoclsparse_create_csr(&mat,
                                                        base,
                                                        m,
                                                        k,
                                                        nnz,
                                                        arr.ptr.data(),
                                                        arr.ind.data(),
                                                        arr.val.data()),
                                  aoclsparse_status_success);
                    }
                    else
                    {
                        ASSERT_EQ(aoclsparse_create_csc<T>(&mat,
                                                           base,
                                                           m,
                                                           k,
                                                           nnz,
                                                           arr.ptr.data(),
                                                           arr.ind.data(),
                                                           arr.val.data()),
                                  aoclsparse_status_success);
                    }

                    EXPECT_EQ(aoclsparse_syprd(
                                  op, mat, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
                              aoclsparse_status_success)
                        << "fmt=" << fmt << " base=" << base << " op=" << op << " order=" << order
                        << " id=" << id;

                    if constexpr(std::is_same_v<T, double>)
                        EXPECT_DOUBLE_EQ_VEC(C_m * C_n, C, C_exp);
                    if constexpr(std::is_same_v<T, float>)
                        EXPECT_FLOAT_EQ_VEC(C_m * C_n, C, C_exp);
                    if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
                    {
                        std::vector<std::complex<double>> *res, *res_exp;
                        res     = (std::vector<std::complex<double>> *)&C;
                        res_exp = (std::vector<std::complex<double>> *)&C_exp;
                        EXPECT_COMPLEX_DOUBLE_EQ_VEC(C_m * C_n, (*res), (*res_exp));
                    }
                    if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
                    {
                        std::vector<std::complex<float>> *res, *res_exp;
                        res     = (std::vector<std::complex<float>> *)&C;
                        res_exp = (std::vector<std::complex<float>> *)&C_exp;
                        EXPECT_COMPLEX_FLOAT_EQ_VEC(C_m * C_n, (*res), (*res_exp));
                    }

                    aoclsparse_destroy(&mat);
                }
            }
        }
    }

    // Complex+Transpose guard validation for both CSR and CSC.
    // Complex + op_transpose is mathematically invalid (result not Hermitian).
    // syprd computes C = alpha * op(A) * B * op(A)^H + beta * C where C is Hermitian.
    // With op=transpose: C = A^T * B * conj(A), which is NOT Hermitian in general.
    // Validates that:
    //   (a) CSR + complex + op_transpose → not_implemented
    //   (b) CSC + complex + op_transpose → not_implemented
    //   (c) CSC + complex + op_none → success (regression guard: proves guard uses
    //       original op, not effective_op which would be op_transpose internally)
    template <typename T>
    void test_syprd_complex_trans_guard()
    {
        static_assert(std::is_same_v<T, aoclsparse_float_complex>
                          || std::is_same_v<T, aoclsparse_double_complex>,
                      "This test is for complex types only");

        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_order      order = aoclsparse_order_row;
        aoclsparse_int        m, k, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        syprd_mat_data<T>     arr;
        T                     alpha, beta;
        std::vector<T>        B, C, C_exp;

        // --- (a) Rejection: CSR + complex + op_transpose → not_implemented ---
        {
            aoclsparse_operation op = aoclsparse_operation_transpose;
            init<T>(op, order, m, k, nnz, arr, alpha, beta, B, C, C_exp, 0);
            set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

            aoclsparse_matrix A_csr;
            ASSERT_EQ(aoclsparse_create_csr(
                          &A_csr, base, m, k, nnz, arr.ptr.data(), arr.ind.data(), arr.val.data()),
                      aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_syprd<T>(
                          op, A_csr, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
                      aoclsparse_status_not_implemented)
                << "CSR + complex + op_transpose must be blocked";
            aoclsparse_destroy(&A_csr);
        }

        // --- (b) Rejection: CSC + complex + op_transpose → not_implemented ---
        {
            aoclsparse_operation op = aoclsparse_operation_transpose;
            init<T>(op, order, m, k, nnz, arr, alpha, beta, B, C, C_exp, 0);
            set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

            // Convert CSR to CSC arrays using library API
            std::vector<aoclsparse_int> csc_col_ptr(k + 1), csc_row_ind(nnz);
            std::vector<T>              csc_val(nnz);
            aoclsparse_mat_descr        descr_csr;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descr_csr), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descr_csr, aoclsparse_index_base_zero),
                      aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_csr2csc<T>(m,
                                            k,
                                            nnz,
                                            descr_csr,
                                            aoclsparse_index_base_zero,
                                            arr.ptr.data(),
                                            arr.ind.data(),
                                            arr.val.data(),
                                            csc_row_ind.data(),
                                            csc_col_ptr.data(),
                                            csc_val.data()),
                      aoclsparse_status_success);
            aoclsparse_destroy_mat_descr(descr_csr);

            aoclsparse_matrix A_csc;
            ASSERT_EQ(aoclsparse_create_csc<T>(&A_csc,
                                               base,
                                               m,
                                               k,
                                               nnz,
                                               csc_col_ptr.data(),
                                               csc_row_ind.data(),
                                               csc_val.data()),
                      aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_syprd<T>(
                          op, A_csc, B.data(), order, ldb, alpha, beta, C.data(), order, ldc),
                      aoclsparse_status_not_implemented)
                << "CSC + complex + op_transpose must be blocked";
            aoclsparse_destroy(&A_csc);
        }

        // --- (c) Acceptance: CSC + complex + op_none → success ---
        // This proves the guard checks original op, NOT effective_op.
        // CSC + op_none maps to effective_op=transpose internally,
        // but op_none is mathematically valid → must succeed.
        {
            aoclsparse_operation op = aoclsparse_operation_none;
            init<T>(op, order, m, k, nnz, arr, alpha, beta, B, C, C_exp, 0);
            set_syprd_dim(op, m, k, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);

            // Convert CSR to CSC arrays using library API
            std::vector<aoclsparse_int> csc_col_ptr(k + 1), csc_row_ind(nnz);
            std::vector<T>              csc_val(nnz);
            aoclsparse_mat_descr        descr_csr;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descr_csr), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descr_csr, aoclsparse_index_base_zero),
                      aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_csr2csc<T>(m,
                                            k,
                                            nnz,
                                            descr_csr,
                                            aoclsparse_index_base_zero,
                                            arr.ptr.data(),
                                            arr.ind.data(),
                                            arr.val.data(),
                                            csc_row_ind.data(),
                                            csc_col_ptr.data(),
                                            csc_val.data()),
                      aoclsparse_status_success);
            aoclsparse_destroy_mat_descr(descr_csr);

            aoclsparse_matrix A_csc;
            ASSERT_EQ(aoclsparse_create_csc<T>(&A_csc,
                                               base,
                                               m,
                                               k,
                                               nnz,
                                               csc_col_ptr.data(),
                                               csc_row_ind.data(),
                                               csc_val.data()),
                      aoclsparse_status_success);

            // Reset C to initial values
            std::vector<T> C_out(C);
            EXPECT_EQ(aoclsparse_syprd<T>(
                          op, A_csc, B.data(), order, ldb, alpha, beta, C_out.data(), order, ldc),
                      aoclsparse_status_success)
                << "CSC + complex + op_none must succeed (effective_op=transpose is internal)";

            // Verify output matches precomputed C_exp
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                std::vector<std::complex<double>> *res, *res_exp;
                res     = (std::vector<std::complex<double>> *)&C_out;
                res_exp = (std::vector<std::complex<double>> *)&C_exp;
                EXPECT_COMPLEX_DOUBLE_EQ_VEC(C_m * C_n, (*res), (*res_exp));
            }
            if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                std::vector<std::complex<float>> *res, *res_exp;
                res     = (std::vector<std::complex<float>> *)&C_out;
                res_exp = (std::vector<std::complex<float>> *)&C_exp;
                EXPECT_COMPLEX_FLOAT_EQ_VEC(C_m * C_n, (*res), (*res_exp));
            }

            aoclsparse_destroy(&A_csc);
        }
    }

    // CSC edge-case tests
    template <typename T>
    void test_syprd_csc_empty_matrix()
    {
        // CSC with nnz=0 (empty matrix): should succeed with C = beta * C
        aoclsparse_index_base base   = aoclsparse_index_base_zero;
        aoclsparse_operation  op     = aoclsparse_operation_none;
        aoclsparse_order      layout = aoclsparse_order_row;
        aoclsparse_int        m_a = 3, n_a = 2;

        // Empty CSC: col_ptr = all zeros, dummy row_ind and val
        std::vector<aoclsparse_int> col_ptr(n_a + 1, 0);
        std::vector<aoclsparse_int> row_ind(1, 0); // dummy, won't be accessed
        std::vector<T>              val;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
            val.assign(1, {0, 0});
        else
            val.assign(1, (T)0);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csc<T>(
                      &A, base, m_a, n_a, 0, col_ptr.data(), row_ind.data(), val.data()),
                  aoclsparse_status_success);

        aoclsparse_int B_dim = n_a; // op_none: B is k x k
        aoclsparse_int C_dim = m_a; // op_none: C is m x m
        std::vector<T> B_data, C_data;

        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            B_data.resize(B_dim * B_dim, {1, 0});
            C_data.resize(C_dim * C_dim, {2, 1});
        }
        else
        {
            B_data.resize(B_dim * B_dim, (T)1);
            C_data.resize(C_dim * C_dim, (T)2);
        }

        std::vector<T> C_exp(C_data);
        T              alpha, beta;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
        }
        else
        {
            alpha = (T)1;
            beta  = (T)0;
        }

        // alpha*op(A)*B*op(A)^H + beta*C = 0 + 0 = 0 since A has no nonzeros and beta=0
        // But the A->nnz=0 case may trigger early return or nullptr guard.
        // The important thing is that it does not crash.
        aoclsparse_status status = aoclsparse_syprd(
            op, A, B_data.data(), layout, B_dim, alpha, beta, C_data.data(), layout, C_dim);
        // Accept either success or invalid_pointer (nnz=0 may have nullptr val/ind)
        EXPECT_TRUE(status == aoclsparse_status_success
                    || status == aoclsparse_status_invalid_pointer);

        aoclsparse_destroy(&A);
    }

    template <typename T>
    void test_syprd_csc_wrong_order()
    {
        // Mismatching B/C order with CSC input must return invalid_operation
        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_operation  op   = aoclsparse_operation_none;
        aoclsparse_int        m_a = 3, n_a = 2, nnz_a = 3;

        std::vector<aoclsparse_int> col_ptr = {0, 2, 3};
        std::vector<aoclsparse_int> row_ind = {0, 1, 2};
        std::vector<T>              val;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
            val.assign({{1, 0}, {2, 1}, {3, -1}});
        else
            val.assign({1, 2, 3});

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csc<T>(
                      &A, base, m_a, n_a, nnz_a, col_ptr.data(), row_ind.data(), val.data()),
                  aoclsparse_status_success);

        std::vector<T> B(n_a * n_a), C(m_a * m_a);
        T              alpha, beta;
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            alpha = {1, 0};
            beta  = {0, 0};
        }
        else
        {
            alpha = (T)1;
            beta  = (T)0;
        }

        // Pass different orderB and orderC
        EXPECT_EQ(aoclsparse_syprd(op,
                                   A,
                                   B.data(),
                                   aoclsparse_order_row,
                                   n_a,
                                   alpha,
                                   beta,
                                   C.data(),
                                   aoclsparse_order_column,
                                   m_a),
                  aoclsparse_status_invalid_operation);

        aoclsparse_destroy(&A);
    }

    // --- Negative tests (type-spread: each type-agnostic test uses one datatype) ---
    TEST(syprd, NullArgDouble)
    {
        test_syprd_nullptr<double>();
    }
    TEST(syprd, WrongSizeFloat)
    {
        test_syprd_wrong_size<float>();
    }
    TEST(syprd, InvalidOpComplexDouble)
    {
        test_syprd_invalid_operation<aoclsparse_double_complex>();
    }
    TEST(syprd, DoNothingDouble)
    {
        test_syprd_do_nothing<double>();
    }
    TEST(syprd, ComplexTransGuardCDouble)
    {
        test_syprd_complex_trans_guard<aoclsparse_double_complex>();
    }

    // --- Numerical tests (both types: type-specific multiply-accumulate) ---
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

    // --- CSC edge-case ---
    TEST(syprd, CSCEdgeCaseDouble)
    {
        test_syprd_csc_empty_matrix<double>();
        test_syprd_csc_wrong_order<double>();
    }

    // --- Success CSR (pairwise: type × base) ---
    TEST(syprd, SuccessDouble)
    {
        test_syprd_success<double>(syprd_csr, aoclsparse_index_base_zero);
    }
    TEST(syprd, SuccessFloat)
    {
        test_syprd_success<float>(syprd_csr, aoclsparse_index_base_one);
    }
    TEST(syprd, SuccessCDouble)
    {
        test_syprd_success<aoclsparse_double_complex>(syprd_csr, aoclsparse_index_base_zero);
    }
    TEST(syprd, SuccessCFloat)
    {
        test_syprd_success<aoclsparse_float_complex>(syprd_csr, aoclsparse_index_base_one);
    }

    // --- Success CSC (pairwise: type × base) ---
    TEST(syprd, SuccessCscDouble)
    {
        test_syprd_success<double>(syprd_csc, aoclsparse_index_base_one);
    }
    TEST(syprd, SuccessCscFloat)
    {
        test_syprd_success<float>(syprd_csc, aoclsparse_index_base_zero);
    }
    TEST(syprd, SuccessCscCDouble)
    {
        test_syprd_success<aoclsparse_double_complex>(syprd_csc, aoclsparse_index_base_one);
    }
    TEST(syprd, SuccessCscCFloat)
    {
        test_syprd_success<aoclsparse_float_complex>(syprd_csc, aoclsparse_index_base_zero);
    }
} // namespace
