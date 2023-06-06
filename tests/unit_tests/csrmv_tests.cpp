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

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_csrmv_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 3, nnz = 1;
        T                    csr_val[]     = {42.};
        aoclsparse_int       csr_col_ind[] = {1};
        aoclsparse_int       csr_row_ptr[] = {0, 0, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        //EXPECT_EQ(aoclsparse_csrmv<T>(trans, &alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
        //      aoclsparse_status_invalid_pointer);
        //FIXME crashes: EXPECT_EQ(aoclsparse_csrmv<T>(trans, nullptr, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
        //      aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, n, nnz, nullptr, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmv<T>(
                      trans, &alpha, m, n, nnz, csr_val, nullptr, csr_row_ptr, descr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmv<T>(
                      trans, &alpha, m, n, nnz, csr_val, csr_col_ind, nullptr, descr, x, &beta, y),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, nullptr, x, &beta, y),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmv<T>(trans,
                                      &alpha,
                                      m,
                                      n,
                                      nnz,
                                      csr_val,
                                      csr_col_ind,
                                      csr_row_ptr,
                                      descr,
                                      nullptr,
                                      &beta,
                                      y),
                  aoclsparse_status_invalid_pointer);
        //FIXME crashes: EXPECT_EQ(aoclsparse_csrmv<T>(trans, &alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, nullptr, y),
        //      aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmv<T>(trans,
                                      &alpha,
                                      m,
                                      n,
                                      nnz,
                                      csr_val,
                                      csr_col_ind,
                                      csr_row_ptr,
                                      descr,
                                      x,
                                      &beta,
                                      nullptr),
                  aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
    }

    // tests with wrong scalar data n, m, nnz
    template <typename T>
    void test_csrmv_wrong_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 3, nnz = 1, wrong = -1;
        T                    csr_val[]     = {42.};
        aoclsparse_int       csr_col_ind[] = {1};
        aoclsparse_int       csr_row_ptr[] = {0, 0, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);

        // In turns pass wrong size in place of n, m, nnz
        //EXPECT_EQ(aoclsparse_csrmv<T>(trans, &alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
        //      aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_csrmv<T>(trans,
                                      &alpha,
                                      wrong,
                                      n,
                                      nnz,
                                      csr_val,
                                      csr_col_ind,
                                      csr_row_ptr,
                                      descr,
                                      x,
                                      &beta,
                                      y),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_csrmv<T>(trans,
                                      &alpha,
                                      m,
                                      wrong,
                                      nnz,
                                      csr_val,
                                      csr_col_ind,
                                      csr_row_ptr,
                                      descr,
                                      x,
                                      &beta,
                                      y),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, n, wrong, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
    }

    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_csrmv_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 3, nnz = 1;
        T                    csr_val[]     = {42.};
        aoclsparse_int       csr_col_ind[] = {1};
        aoclsparse_int       csr_row_ptr[] = {0, 0, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);

        // Passing zero size matrix should be OK
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, 0, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, 0, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_csrmv<T>(
                trans, &alpha, m, n, 0, csr_val, csr_col_ind, csr_row_ptr, descr, x, &beta, y),
            aoclsparse_status_success);

        aoclsparse_destroy_mat_descr(descr);
    }

    //TODO add:
    // * positive tests with special predefined matrices
    // * positive tests with unsorted CSR
    // * whole branch for symmetric matrix computation
    // * not supported/implemented
    // * invalid array data (but we don't test these right now, e.g., col_ind out of bounds)
    // * nnz not matching row_ptr
    //

    TEST(csrmv, NullArgDouble)
    {
        test_csrmv_nullptr<double>();
    }
    TEST(csrmv, NullArgFloat)
    {
        test_csrmv_nullptr<float>();
    }

    TEST(csrmv, WrongSizeDouble)
    {
        test_csrmv_wrong_size<double>();
    }
    TEST(csrmv, WrongSizeFloat)
    {
        test_csrmv_wrong_size<float>();
    }

    TEST(csrmv, DoNothingDouble)
    {
        test_csrmv_do_nothing<double>();
    }
    TEST(csrmv, DoNothingFloat)
    {
        test_csrmv_do_nothing<float>();
    }

} // namespace
