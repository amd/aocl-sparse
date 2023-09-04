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
    void test_csr2m_nullptr()
    {
        aoclsparse_operation  transA = aoclsparse_operation_none;
        aoclsparse_operation  transB = aoclsparse_operation_none;
        aoclsparse_index_base base   = aoclsparse_index_base_zero;
        aoclsparse_request    request;
        aoclsparse_int        m = 2, k = 3, n = 2, nnzA = 1, nnzB = 3;
        T                     csr_valA[]     = {42.};
        aoclsparse_int        csr_col_indA[] = {1};
        aoclsparse_int        csr_row_ptrA[] = {0, 0, 1};
        T                     csr_valB[]     = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[] = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[] = {0, 1, 2, 3};
        aoclsparse_mat_descr  descrA, descrB;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_create_mat_descr(&descrB);

        aoclsparse_matrix csrA;
        aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix csrB;
        aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix csrC = NULL;
        request                = aoclsparse_stage_full_computation;
        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, nullptr, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, nullptr, csrB, request, &csrC),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_csr2m<T>(transA, descrA, nullptr, transB, descrB, csrB, request, &csrC),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, nullptr, request, &csrC),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, nullptr),
                  aoclsparse_status_invalid_pointer);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);
    }
    // Quick return with success when size 0 matrix is passed

    template <typename T>
    void test_csr2m_do_nothing()
    {
        aoclsparse_operation  transA = aoclsparse_operation_none;
        aoclsparse_operation  transB = aoclsparse_operation_none;
        aoclsparse_index_base base   = aoclsparse_index_base_zero;
        aoclsparse_request    request;
        aoclsparse_int        m = 2, k = 3, n = 2, nnzA = 1, nnzA_zero = 0, nnzB = 3, nnzB_zero = 0;
        T                     csr_valA[]           = {42.};
        aoclsparse_int        csr_col_indA[]       = {1};
        aoclsparse_int        csr_row_ptrA[]       = {0, 0, 1};
        aoclsparse_int        csr_row_ptrA_zeros[] = {0, 0, 0};
        T                     csr_valB[]           = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[]       = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[]       = {0, 1, 2, 3};
        aoclsparse_int        csr_row_ptrB_zeros[] = {0, 0, 0, 0};
        aoclsparse_mat_descr  descrA, descrB;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_create_mat_descr(&descrB);

        aoclsparse_matrix csrA;
        aoclsparse_create_csr(
            csrA, base, 0, k, nnzA_zero, csr_row_ptrA_zeros, csr_col_indA, csr_valA);
        aoclsparse_matrix csrB;
        aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix csrC = NULL;
        request                = aoclsparse_stage_full_computation;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);

        aoclsparse_create_csr(
            csrA, base, m, 0, nnzA_zero, csr_row_ptrA_zeros, csr_col_indA, csr_valA);
        aoclsparse_create_csr(csrB, base, 0, n, nnzB_zero, csr_row_ptrB, csr_col_indB, csr_valB);
        request = aoclsparse_stage_full_computation;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);

        aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_create_csr(
            csrB, base, k, 0, nnzB_zero, csr_row_ptrB_zeros, csr_col_indB, csr_valB);
        request = aoclsparse_stage_full_computation;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);

        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy_mat_descr(descrB);
    }

    // tests for Wrong size
    template <typename T>
    void test_csr2m_wrong_size()
    {
        aoclsparse_operation  transA = aoclsparse_operation_none;
        aoclsparse_operation  transB = aoclsparse_operation_none;
        aoclsparse_index_base base   = aoclsparse_index_base_zero;
        aoclsparse_request    request;
        aoclsparse_int        m = 2, k = 3, n = 4, nnzA = 1, nnzB = 3;
        T                     csr_valA[]     = {42.};
        aoclsparse_int        csr_col_indA[] = {1};
        aoclsparse_int        csr_row_ptrA[] = {0, 0, 1};
        T                     csr_valB[]     = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[] = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[] = {0, 1, 2, 3};
        aoclsparse_mat_descr  descrA, descrB;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_create_mat_descr(&descrB);

        // expect aoclsparse_status_invalid_value for csrA->n != csrB->m
        aoclsparse_matrix csrA;
        ASSERT_EQ(
            aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);
        aoclsparse_matrix csrB;
        ASSERT_EQ(
            aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        csrB->m                = k - 1;
        aoclsparse_matrix csrC = NULL;
        request                = aoclsparse_stage_full_computation;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_size);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy(csrB);

        // expect aoclsparse_status_invalid_value for csrC->m = 0 , csrC->n = 0
        // TBD csr arrays for C matrix need to be defined.
        ASSERT_EQ(
            aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(csrC, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        csrC->m   = 0;
        csrC->n   = 0;
        csrC->nnz = 0;
        request   = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_size);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);

        // expect aoclsparse_status_invalid_value for invalid request
        ASSERT_EQ(
            aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        request = (aoclsparse_request)3;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);
    }
    template <typename T>
    void test_csr2m_1step_baseOne()
    {
        aoclsparse_operation  transA = aoclsparse_operation_none;
        aoclsparse_operation  transB = aoclsparse_operation_none;
        aoclsparse_index_base base;
        int                   invalid_index_base = 2;
        aoclsparse_request    request;
        aoclsparse_int        m = 3, k = 3, n = 3, nnzA = 4, nnzB = 4;

        aoclsparse_int    csr_row_ptrA[4] = {1, 2, 3, 5};
        aoclsparse_int    csr_col_indA[4] = {1, 2, 1, 3};
        T                 csr_valA[4]     = {8.00, 5.00, 7.00, 7.00};
        aoclsparse_matrix csrA;

        aoclsparse_int    nnz_C_gold      = 5;
        aoclsparse_int    csr_row_ptrB[4] = {1, 2, 3, 5};
        aoclsparse_int    csr_col_indB[4] = {1, 1, 2, 3};
        T                 csr_valB[4]     = {7.00, 9.00, 6.00, 2.00};
        aoclsparse_matrix csrB;

        //since the output csrC is in zero-based indexing, the pre-computed csr_gold is defined as such
        aoclsparse_int csr_row_ptr_C_gold[4] = {0, 1, 2, 5}; //size = C_M + 1
        aoclsparse_int csr_col_ind_C_gold[5] = {0, 0, 0, 1, 2}; //size = nnz_C
        T              csr_val_C_gold[5]     = {56.00, 45.00, 49.00, 42.00, 14.00}; //size = nnz_C

        aoclsparse_mat_descr descrA, descrB;
        aoclsparse_matrix    csrC          = NULL;
        aoclsparse_int      *csr_row_ptr_C = NULL;
        aoclsparse_int      *csr_col_ind_C = NULL;
        T                   *csr_val_C     = NULL;
        aoclsparse_int       C_M, C_N;
        aoclsparse_int       nnz_C;

        request = aoclsparse_stage_full_computation;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);

        //A -> 1-base, B -> 1-base
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrA, aoclsparse_index_base_one, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrB, aoclsparse_index_base_one, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_export_csr(
                      csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C),
                  aoclsparse_status_success);
        EXPECT_EQ(nnz_C, nnz_C_gold);
        EXPECT_ARR_NEAR((C_M + 1), csr_row_ptr_C, csr_row_ptr_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_col_ind_C, csr_col_ind_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_val_C, csr_val_C_gold, expected_precision<T>(10.0));
        EXPECT_EQ(aoclsparse_destroy(csrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrB), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrC), aoclsparse_status_success);

        //A -> 1-base, B -> 0-base
        csrA          = NULL;
        csrB          = NULL;
        csrC          = NULL;
        csr_row_ptr_C = NULL;
        csr_col_ind_C = NULL;
        csr_val_C     = NULL;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        //Convert csrB arrays to 0-based
        for(int i = 0; i < k + 1; i++)
        {
            csr_row_ptrB[i] = csr_row_ptrB[i] - 1;
        }
        for(int i = 0; i < nnzB; i++)
        {
            csr_col_indB[i] = csr_col_indB[i] - 1;
        }
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrA, aoclsparse_index_base_one, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrB, aoclsparse_index_base_zero, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_export_csr(
                      csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C),
                  aoclsparse_status_success);
        EXPECT_EQ(nnz_C, nnz_C_gold);
        EXPECT_ARR_NEAR((C_M + 1), csr_row_ptr_C, csr_row_ptr_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_col_ind_C, csr_col_ind_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_val_C, csr_val_C_gold, expected_precision<T>(10.0));
        EXPECT_EQ(aoclsparse_destroy(csrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrB), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrC), aoclsparse_status_success);

        //A -> 0-base, B -> 1-base
        csrA          = NULL;
        csrB          = NULL;
        csrC          = NULL;
        csr_row_ptr_C = NULL;
        csr_col_ind_C = NULL;
        csr_val_C     = NULL;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        //Convert csrA arrays to 0-based
        for(int i = 0; i < m + 1; i++)
        {
            csr_row_ptrA[i] = csr_row_ptrA[i] - 1;
        }
        for(int i = 0; i < nnzA; i++)
        {
            csr_col_indA[i] = csr_col_indA[i] - 1;
        }
        //Convert csrB arrays to 1-based
        for(int i = 0; i < k + 1; i++)
        {
            csr_row_ptrB[i] = csr_row_ptrB[i] + 1;
        }
        for(int i = 0; i < nnzB; i++)
        {
            csr_col_indB[i] = csr_col_indB[i] + 1;
        }
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrA, aoclsparse_index_base_zero, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrB, aoclsparse_index_base_one, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_export_csr(
                      csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C),
                  aoclsparse_status_success);
        EXPECT_EQ(nnz_C, nnz_C_gold);
        EXPECT_ARR_NEAR((C_M + 1), csr_row_ptr_C, csr_row_ptr_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_col_ind_C, csr_col_ind_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_val_C, csr_val_C_gold, expected_precision<T>(10.0));
        EXPECT_EQ(aoclsparse_destroy(csrC), aoclsparse_status_success);

        //check for invalid base-index value
        csrC          = NULL;
        csr_row_ptr_C = NULL;
        csr_col_ind_C = NULL;
        csr_val_C     = NULL;
        descrA->base  = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        descrB->base = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_destroy(csrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrB), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descrB), aoclsparse_status_success);
    }
    template <typename T>
    void test_csr2m_2step_baseOne()
    {
        aoclsparse_operation  transA = aoclsparse_operation_none;
        aoclsparse_operation  transB = aoclsparse_operation_none;
        aoclsparse_index_base base;
        int                   invalid_index_base = 2;
        aoclsparse_request    request;
        aoclsparse_int        m = 3, k = 3, n = 3, nnzA = 4, nnzB = 4;

        aoclsparse_int    csr_row_ptrA[4] = {1, 2, 3, 5};
        aoclsparse_int    csr_col_indA[4] = {1, 2, 1, 3};
        T                 csr_valA[4]     = {8.00, 5.00, 7.00, 7.00};
        aoclsparse_matrix csrA;

        aoclsparse_int    nnz_C_gold      = 5;
        aoclsparse_int    csr_row_ptrB[4] = {1, 2, 3, 5};
        aoclsparse_int    csr_col_indB[4] = {1, 1, 2, 3};
        T                 csr_valB[4]     = {7.00, 9.00, 6.00, 2.00};
        aoclsparse_matrix csrB;

        //since the output csrC is in zero-based indexing, the pre-computed csr_gold is defined as such
        aoclsparse_int csr_row_ptr_C_gold[4] = {0, 1, 2, 5}; //size = C_M + 1
        aoclsparse_int csr_col_ind_C_gold[5] = {0, 0, 0, 1, 2}; //size = nnz_C
        T              csr_val_C_gold[5]     = {56.00, 45.00, 49.00, 42.00, 14.00}; //size = nnz_C

        aoclsparse_mat_descr descrA, descrB;
        aoclsparse_matrix    csrC          = NULL;
        aoclsparse_int      *csr_row_ptr_C = NULL;
        aoclsparse_int      *csr_col_ind_C = NULL;
        T                   *csr_val_C     = NULL;
        aoclsparse_int       C_M, C_N;
        aoclsparse_int       nnz_C;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_mat_descr(&descrB), aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        ASSERT_EQ(
            aoclsparse_create_csr(
                csrA, aoclsparse_index_base_one, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);

        ASSERT_EQ(
            aoclsparse_create_csr(
                csrB, aoclsparse_index_base_one, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);

        //Step-1
        request = aoclsparse_stage_nnz_count;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);

        //Step-2
        request = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_export_csr(
                      csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C),
                  aoclsparse_status_success);

        EXPECT_EQ(nnz_C, nnz_C_gold);
        EXPECT_ARR_NEAR((C_M + 1), csr_row_ptr_C, csr_row_ptr_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_col_ind_C, csr_col_ind_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_val_C, csr_val_C_gold, expected_precision<T>(10.0));
        EXPECT_EQ(aoclsparse_destroy(csrC), aoclsparse_status_success);

        //check for invalid base-index value
        csrC          = NULL;
        csr_row_ptr_C = NULL;
        csr_col_ind_C = NULL;
        csr_val_C     = NULL;
        descrA->base  = (aoclsparse_index_base)invalid_index_base;
        //Step-1
        request = aoclsparse_stage_nnz_count;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        //Step-2
        request = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        descrB->base = (aoclsparse_index_base)invalid_index_base;
        //Step-1
        request = aoclsparse_stage_nnz_count;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        //Step-2
        request = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_destroy(csrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrB), aoclsparse_status_success);

        //A -> 1-base, B -> 0-base
        csrA          = NULL;
        csrB          = NULL;
        csrC          = NULL;
        csr_row_ptr_C = NULL;
        csr_col_ind_C = NULL;
        csr_val_C     = NULL;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        //Convert csrB arrays to 0-based
        for(int i = 0; i < k + 1; i++)
        {
            csr_row_ptrB[i] = csr_row_ptrB[i] - 1;
        }
        for(int i = 0; i < nnzB; i++)
        {
            csr_col_indB[i] = csr_col_indB[i] - 1;
        }
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrA, aoclsparse_index_base_one, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrB, aoclsparse_index_base_zero, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        //Step-1
        request = aoclsparse_stage_nnz_count;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        //Step-2
        request = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_export_csr(
                      csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C),
                  aoclsparse_status_success);
        EXPECT_EQ(nnz_C, nnz_C_gold);
        EXPECT_ARR_NEAR((C_M + 1), csr_row_ptr_C, csr_row_ptr_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_col_ind_C, csr_col_ind_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_val_C, csr_val_C_gold, expected_precision<T>(10.0));
        EXPECT_EQ(aoclsparse_destroy(csrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrB), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrC), aoclsparse_status_success);

        //A -> 0-base, B -> 1-base
        csrA          = NULL;
        csrB          = NULL;
        csrC          = NULL;
        csr_row_ptr_C = NULL;
        csr_col_ind_C = NULL;
        csr_val_C     = NULL;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        //Convert csrA arrays to 0-based
        for(int i = 0; i < m + 1; i++)
        {
            csr_row_ptrA[i] = csr_row_ptrA[i] - 1;
        }
        for(int i = 0; i < nnzA; i++)
        {
            csr_col_indA[i] = csr_col_indA[i] - 1;
        }
        //Convert csrB arrays to 1-based
        for(int i = 0; i < k + 1; i++)
        {
            csr_row_ptrB[i] = csr_row_ptrB[i] + 1;
        }
        for(int i = 0; i < nnzB; i++)
        {
            csr_col_indB[i] = csr_col_indB[i] + 1;
        }
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrA, aoclsparse_index_base_zero, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA),
            aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(
                csrB, aoclsparse_index_base_one, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB),
            aoclsparse_status_success);
        //Step-1
        request = aoclsparse_stage_nnz_count;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        //Step-2
        request = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_export_csr(
                      csrC, &base, &C_M, &C_N, &nnz_C, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C),
                  aoclsparse_status_success);
        EXPECT_EQ(nnz_C, nnz_C_gold);
        EXPECT_ARR_NEAR((C_M + 1), csr_row_ptr_C, csr_row_ptr_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_col_ind_C, csr_col_ind_C_gold, expected_precision<T>(10.0));
        EXPECT_ARR_NEAR(nnz_C, csr_val_C, csr_val_C_gold, expected_precision<T>(10.0));
        EXPECT_EQ(aoclsparse_destroy(csrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrB), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(csrC), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descrA), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descrB), aoclsparse_status_success);
    }
    // tests for settings not implemented
    template <typename T>
    void test_csr2m_not_implemented()
    {
        aoclsparse_operation  transA = aoclsparse_operation_none;
        aoclsparse_operation  transB = aoclsparse_operation_none;
        aoclsparse_index_base base   = aoclsparse_index_base_zero;
        aoclsparse_request    request;
        aoclsparse_int        m = 2, k = 3, n = 2, nnzA = 1, nnzB = 3;
        T                     csr_valA[]     = {42.};
        aoclsparse_int        csr_col_indA[] = {1};
        aoclsparse_int        csr_row_ptrA[] = {0, 0, 1};
        T                     csr_valB[]     = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[] = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[] = {0, 1, 2, 3};
        aoclsparse_mat_descr  descrA, descrB;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_create_mat_descr(&descrB);

        // and expect not_implemented for !aoclsparse_matrix_type_general for matrix A and B
        aoclsparse_matrix csrA;
        aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix csrB;
        aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix csrC = NULL;
        request                = aoclsparse_stage_full_computation;

        aoclsparse_set_mat_index_base(descrB, aoclsparse_index_base_zero);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_not_implemented);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_general);
        aoclsparse_set_mat_type(descrB, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);
    }

    void test_csr2m_wrong_datatype_float()
    {
        aoclsparse_operation  transA = aoclsparse_operation_none;
        aoclsparse_operation  transB = aoclsparse_operation_none;
        aoclsparse_index_base base   = aoclsparse_index_base_zero;
        aoclsparse_request    request;
        aoclsparse_int        m = 2, k = 3, n = 2, nnzA = 1, nnzB = 3;
        float                 csr_valA[]     = {42.};
        aoclsparse_int        csr_col_indA[] = {1};
        aoclsparse_int        csr_row_ptrA[] = {0, 0, 1};
        float                 csr_valB[]     = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[] = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[] = {0, 1, 2, 3};
        aoclsparse_mat_descr  descrA, descrB;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_create_mat_descr(&descrB);

        aoclsparse_matrix csrA;
        aoclsparse_create_scsr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix csrB;
        aoclsparse_create_scsr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix csrC = NULL;
        request                = aoclsparse_stage_full_computation;
        // For float date type matrices, invoke csr2m for double precision
        // and expect wrong type error
        EXPECT_EQ(aoclsparse_dcsr2m(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_wrong_type);

        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);
    }
    void test_csr2m_wrong_datatype_double()
    {
        aoclsparse_operation  transA = aoclsparse_operation_none;
        aoclsparse_operation  transB = aoclsparse_operation_none;
        aoclsparse_index_base base   = aoclsparse_index_base_zero;
        aoclsparse_request    request;
        aoclsparse_int        m = 2, k = 3, n = 2, nnzA = 1, nnzB = 3;
        double                csr_valA[]     = {42.};
        aoclsparse_int        csr_col_indA[] = {1};
        aoclsparse_int        csr_row_ptrA[] = {0, 0, 1};
        double                csr_valB[]     = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[] = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[] = {0, 1, 2, 3};
        aoclsparse_mat_descr  descrA, descrB;
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_create_mat_descr(&descrB);

        aoclsparse_matrix csrA;
        aoclsparse_create_dcsr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix csrB;
        aoclsparse_create_dcsr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix csrC = NULL;
        request                = aoclsparse_stage_full_computation;
        // For double date type matrices, invoke csr2m for single precision
        // and expect wrong type error
        EXPECT_EQ(aoclsparse_scsr2m(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_wrong_type);

        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);
    }
    TEST(csr2m, NullArgDouble)
    {
        test_csr2m_nullptr<double>();
    }
    TEST(csr2m, NullArgFloat)
    {
        test_csr2m_nullptr<float>();
    }
    TEST(csr2m, DoNothingFloat)
    {
        test_csr2m_do_nothing<float>();
    }
    TEST(csr2m, DoNothingDouble)
    {
        test_csr2m_do_nothing<double>();
    }
    TEST(csr2m, WrongSizeDouble)
    {
        test_csr2m_wrong_size<double>();
    }
    TEST(csr2m, WrongSizeFloat)
    {
        test_csr2m_wrong_size<float>();
    }

    TEST(csr2m, BaseOneDouble1Step)
    {
        test_csr2m_1step_baseOne<double>();
    }
    TEST(csr2m, BaseOneFloat1Step)
    {
        test_csr2m_1step_baseOne<float>();
    }
    TEST(csr2m, BaseOneDouble2Step)
    {
        test_csr2m_2step_baseOne<double>();
    }
    TEST(csr2m, BaseOneFloat2Step)
    {
        test_csr2m_2step_baseOne<float>();
    }

    TEST(csr2m, NotImplDouble)
    {
        test_csr2m_not_implemented<double>();
    }
    TEST(csr2m, NotImplFloat)
    {
        test_csr2m_not_implemented<float>();
    }
    TEST(csr2m, WrongTypeFloat)
    {
        test_csr2m_wrong_datatype_float();
    }
    TEST(csr2m, WrongTypeDouble)
    {
        test_csr2m_wrong_datatype_double();
    }
} // namespace
