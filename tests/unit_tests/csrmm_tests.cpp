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
    void test_csrmm_nullptr()
    {
        aoclsparse_operation  trans = aoclsparse_operation_none;
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_order      order = aoclsparse_order_column;
        aoclsparse_int        m = 2, k = 3, n = 2, nnz = 1;
        T                     csr_val[]     = {42.};
        aoclsparse_int        csr_col_ind[] = {1};
        aoclsparse_int        csr_row_ptr[] = {0, 0, 1};
        T                     alpha = 2.3, beta = 11.2;
        T                     B[] = {1.0, -2.0, 3.0, 4.0, 5.0, -6.0};
        T                     C[] = {0.1, 0.2, 0.3, 0.4};
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_matrix csr;
        aoclsparse_create_csr(csr, base, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, nullptr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, nullptr, descr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, nullptr, csr, descr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k, nullptr, C, m),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, nullptr, n, k, &beta, C, m),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k, &beta, nullptr, m),
                  aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(csr);
    }

    // tests for Wrong size
    template <typename T>
    void test_csrmm_wrong_size()
    {
        aoclsparse_operation  trans = aoclsparse_operation_none;
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_order      order = aoclsparse_order_column;
        aoclsparse_int        m = 2, k = 3, n = 2, nnz = 1;
        T                     csr_val[]     = {42.};
        aoclsparse_int        csr_col_ind[] = {1};
        aoclsparse_int        csr_row_ptr[] = {0, 0, 1};
        T                     alpha = 2.3, beta = 11.2;
        T                     B[] = {1.0, -2.0, 3.0, 4.0, 5.0, -6.0};
        T                     C[] = {0.1, 0.2, 0.3, 0.4};
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_matrix csr;
        aoclsparse_create_csr(csr, base, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val);

        // and expect invalid size for wrong ldb
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k - 1, &beta, C, m),
                  aoclsparse_status_invalid_size);

        // and expect invalid size for wrong ldc
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k, &beta, C, m - 1),
                  aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(csr);
    }

    template <typename T>
    void test_csrmm_rowOrder_baseOne()
    {
        aoclsparse_operation transA = aoclsparse_operation_none;
        aoclsparse_order     order  = aoclsparse_order_row;
        aoclsparse_int       M = 3, K = 3, N = 3, NNZ = 4;
        T                    csr_val[]     = {8.00, 5.00, 7.00, 7.00};
        aoclsparse_int       csr_col_ind[] = {1, 2, 1, 3};
        aoclsparse_int       csr_row_ptr[] = {1, 2, 3, 5};
        T                    alpha = 1.0, beta = 0.0;
        T                    B[9] = {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00};
        T                    C[9] = {0};
        T C_gold[9]               = {8.00, 16.00, 24.00, 20.00, 25.00, 30.00, 56.00, 70.00, 84.00};
        aoclsparse_mat_descr descr;

        aoclsparse_int B_n   = N;
        aoclsparse_int C_m   = M;
        aoclsparse_int C_n   = N;
        aoclsparse_int ldb   = order == aoclsparse_order_column ? K : N;
        aoclsparse_int ldc   = order == aoclsparse_order_column ? M : N;
        aoclsparse_int nrowC = order == aoclsparse_order_column ? ldc : C_m;
        aoclsparse_int ncolC = order == aoclsparse_order_column ? C_n : ldc;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        aoclsparse_matrix csr;
        ASSERT_EQ(aoclsparse_create_csr(csr,
                                        aoclsparse_index_base_one,
                                        (transA == aoclsparse_operation_none ? M : K),
                                        (transA == aoclsparse_operation_none ? K : M),
                                        NNZ,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        csr_val),
                  aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_csrmm<T>(transA, &alpha, csr, descr, order, B, B_n, ldb, &beta, C, ldc),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR((nrowC * ncolC), C, C_gold, expected_precision<T>(10.0));

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csr), aoclsparse_status_success);
    }
    template <typename T>
    void test_csrmm_colOrder_baseOne()
    {
        aoclsparse_operation transA = aoclsparse_operation_none;
        aoclsparse_order     order  = aoclsparse_order_column;
        aoclsparse_int       M = 3, K = 3, N = 3, NNZ = 4;
        T                    csr_val[]     = {8.00, 5.00, 7.00, 7.00};
        aoclsparse_int       csr_col_ind[] = {1, 2, 1, 3};
        aoclsparse_int       csr_row_ptr[] = {1, 2, 3, 5};
        T                    alpha = 1.0, beta = 0.0;
        T                    B[9] = {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00};
        T                    C[9] = {0};
        T C_gold[9]               = {8.00, 10.00, 28.00, 32.00, 25.00, 70.00, 56.00, 40.00, 112.00};
        aoclsparse_mat_descr descr;

        aoclsparse_int B_n   = N;
        aoclsparse_int C_m   = M;
        aoclsparse_int C_n   = N;
        aoclsparse_int ldb   = order == aoclsparse_order_column ? K : N;
        aoclsparse_int ldc   = order == aoclsparse_order_column ? M : N;
        aoclsparse_int nrowC = order == aoclsparse_order_column ? ldc : C_m;
        aoclsparse_int ncolC = order == aoclsparse_order_column ? C_n : ldc;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        aoclsparse_matrix csr;
        ASSERT_EQ(aoclsparse_create_csr(csr,
                                        aoclsparse_index_base_one,
                                        (transA == aoclsparse_operation_none ? M : K),
                                        (transA == aoclsparse_operation_none ? K : M),
                                        NNZ,
                                        csr_row_ptr,
                                        csr_col_ind,
                                        csr_val),
                  aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_csrmm<T>(transA, &alpha, csr, descr, order, B, B_n, ldb, &beta, C, ldc),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR((nrowC * ncolC), C, C_gold, expected_precision<T>(10.0));

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csr), aoclsparse_status_success);
    }

    // tests for settings not implemented
    template <typename T>
    void test_csrmm_not_implemented()
    {
        aoclsparse_operation  trans = aoclsparse_operation_none;
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_order      order = aoclsparse_order_column;
        aoclsparse_int        m = 2, k = 3, n = 2, nnz = 1;
        T                     csr_val[]     = {42.};
        aoclsparse_int        csr_col_ind[] = {1};
        aoclsparse_int        csr_row_ptr[] = {0, 0, 1};
        T                     alpha = 2.3, beta = 11.2;
        T                     B[] = {1.0, -2.0, 3.0, 4.0, 5.0, -6.0};
        T                     C[] = {0.1, 0.2, 0.3, 0.4};
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_matrix csr;
        aoclsparse_create_csr(csr, base, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val);

        // and expect not_implemented for aoclsparse_operation_transpose
        EXPECT_EQ(
            aoclsparse_csrmm<T>(
                aoclsparse_operation_transpose, &alpha, csr, descr, order, B, n, k, &beta, C, m),
            aoclsparse_status_not_implemented);

        // and expect not_implemented for !aoclsparse_matrix_type_general
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(csr);
    }

    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_csrmm_do_nothing()
    {
        // tests for settings not implemented
        aoclsparse_operation  trans = aoclsparse_operation_none;
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_order      order = aoclsparse_order_column;
        aoclsparse_int        m = 2, k = 3, n = 2, nnz = 1;
        T                     csr_val[]     = {42.};
        aoclsparse_int        csr_col_ind[] = {1};
        aoclsparse_int        csr_row_ptr[] = {0, 0, 1};
        T                     alpha = 2.3, beta = 11.2;
        T                     B[] = {1.0, -2.0, 3.0, 4.0, 5.0, -6.0};
        T                     C[] = {0.1, 0.2, 0.3, 0.4};
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_matrix csr;

        // expect success for m=0
        aoclsparse_create_csr(csr, base, 0, k, nnz, csr_row_ptr, csr_col_ind, csr_val);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_success);
        aoclsparse_destroy(csr);

        // and expect success for k=0
        aoclsparse_create_csr(csr, base, m, 0, nnz, csr_row_ptr, csr_col_ind, csr_val);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_success);
        aoclsparse_destroy(csr);

        // and expect success for n=0
        aoclsparse_create_csr(csr, base, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val);
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, 0, k, &beta, C, m),
                  aoclsparse_status_success);

        // and expect success for alpha = 0 & beta = 1
        alpha = 0.0;
        beta  = 1.0;
        EXPECT_EQ(aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_success);

        aoclsparse_destroy(csr);
        aoclsparse_destroy_mat_descr(descr);
    }

    // tests for ldb and ldc greater than minimum
    template <typename T>
    void test_csrmm_greater_ld()
    {
        aoclsparse_operation  trans = aoclsparse_operation_none;
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_order      order = aoclsparse_order_column;
        aoclsparse_int        m = 2, k = 3, n = 2, nnz = 1;
        T                     csr_val[]     = {42.};
        aoclsparse_int        csr_col_ind[] = {1};
        aoclsparse_int        csr_row_ptr[] = {0, 0, 1};
        T                     alpha = 2.3, beta = 11.2;
        T                     B[] = {1.0, -2.0, 3.0, 0, 0, 0, 4.0, 5.0, -6.0, 0, 0, 0};
        T                     C[] = {0.1, 0.2, 0, 0, 0.3, 0.4, 0, 0};
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_matrix csr;
        aoclsparse_create_csr(csr, base, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val);

        // and expect success for ldb = k*2 and ldc = m*2
        EXPECT_EQ(
            aoclsparse_csrmm<T>(trans, &alpha, csr, descr, order, B, n, k * 2, &beta, C, m * 2),
            aoclsparse_status_success);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(csr);
    }

    // tests for wrong datatype of matrix value (float)
    void test_csrmm_wrongtype_float()
    {
        aoclsparse_operation  trans = aoclsparse_operation_none;
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_order      order = aoclsparse_order_column;
        aoclsparse_int        m = 2, k = 3, n = 2, nnz = 1;
        float                 csr_val[]     = {42.};
        aoclsparse_int        csr_col_ind[] = {1};
        aoclsparse_int        csr_row_ptr[] = {0, 0, 1};
        double                alpha = 2.3, beta = 11.2;
        double                B[] = {1.0, -2.0, 3.0, 4.0, 5.0, -6.0};
        double                C[] = {0.1, 0.2, 0.3, 0.4};
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_matrix csr;
        aoclsparse_create_scsr(csr, base, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val);

        // expect wrong type error for invoking csrmm for double precision with float csr_val
        EXPECT_EQ(aoclsparse_dcsrmm(trans, &alpha, csr, descr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_wrong_type);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(csr);
    }

    // tests for wrong datatype of matrix value (double)
    void test_csrmm_wrongtype_double()
    {
        aoclsparse_operation  trans = aoclsparse_operation_none;
        aoclsparse_index_base base  = aoclsparse_index_base_zero;
        aoclsparse_order      order = aoclsparse_order_column;
        aoclsparse_int        m = 2, k = 3, n = 2, nnz = 1;
        double                csr_val[]     = {42.};
        aoclsparse_int        csr_col_ind[] = {1};
        aoclsparse_int        csr_row_ptr[] = {0, 0, 1};
        float                 alpha = 2.3, beta = 11.2;
        float                 B[] = {1.0, -2.0, 3.0, 4.0, 5.0, -6.0};
        float                 C[] = {0.1, 0.2, 0.3, 0.4};
        aoclsparse_mat_descr  descr;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        aoclsparse_matrix csr;
        aoclsparse_create_dcsr(csr, base, m, k, nnz, csr_row_ptr, csr_col_ind, csr_val);

        // expect wrong type error for invoking csrmm for single precision with double csr_val
        EXPECT_EQ(aoclsparse_scsrmm(trans, &alpha, csr, descr, order, B, n, k, &beta, C, m),
                  aoclsparse_status_wrong_type);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(csr);
    }

    TEST(csrmm, NullArgDouble)
    {
        test_csrmm_nullptr<double>();
    }
    TEST(csrmm, NullArgFloat)
    {
        test_csrmm_nullptr<float>();
    }

    TEST(csrmm, WrongSizeDouble)
    {
        test_csrmm_wrong_size<double>();
    }
    TEST(csrmm, WrongSizeFloat)
    {
        test_csrmm_wrong_size<float>();
    }

    TEST(csrmm, BaseOneDoubleRO)
    {
        test_csrmm_rowOrder_baseOne<double>();
    }
    TEST(csrmm, BaseOneFloatRO)
    {
        test_csrmm_rowOrder_baseOne<float>();
    }
    TEST(csrmm, BaseOneDoubleCO)
    {
        test_csrmm_colOrder_baseOne<double>();
    }
    TEST(csrmm, BaseOneFloatCO)
    {
        test_csrmm_colOrder_baseOne<float>();
    }

    TEST(csrmm, NotImplDouble)
    {
        test_csrmm_not_implemented<double>();
    }
    TEST(csrmm, NotImplFloat)
    {
        test_csrmm_not_implemented<float>();
    }

    TEST(csrmm, DoNothingDouble)
    {
        test_csrmm_do_nothing<double>();
    }
    TEST(csrmm, DoNothingFloat)
    {
        test_csrmm_do_nothing<float>();
    }

    TEST(csrmm, GreaterLDDouble)
    {
        test_csrmm_greater_ld<double>();
    }
    TEST(csrmm, GreaterLDFloat)
    {
        test_csrmm_greater_ld<float>();
    }

    TEST(csrmm, WrongTypeDouble)
    {
        test_csrmm_wrongtype_double();
    }
    TEST(csrmm, WrongTypeFloat)
    {
        test_csrmm_wrongtype_float();
    }
} // namespace
