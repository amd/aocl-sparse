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

    // tests for Wrong size
    template <typename T>
    void test_csr2m_wrong_size()
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

        // expect aoclsparse_status_invalid_value for csrA->n != csrB->m
        aoclsparse_matrix csrA;
        aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix csrB;
        aoclsparse_create_csr(csrB, base, k - 1, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix csrC = NULL;
        request                = aoclsparse_stage_full_computation;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy(csrB);

        // expect aoclsparse_status_invalid_value for csrC->m = 0 , csrC->n = 0
        // TBD csr arrays for C matrix need to be defined.
        aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_create_csr(csrC, base, 0, 0, 0, csr_row_ptrB, csr_col_indB, csr_valB);
        request = aoclsparse_stage_finalize;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);

        // expect aoclsparse_status_invalid_value for invalid request
        aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        request = (aoclsparse_request)3;
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(csrA);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(csrB);
        aoclsparse_destroy(csrC);
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

        // and expect not_implemented for aoclsparse_operation_transpose(transA & transB)
        aoclsparse_matrix csrA;
        aoclsparse_create_csr(csrA, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix csrB;
        aoclsparse_create_csr(csrB, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix csrC = NULL;
        request                = aoclsparse_stage_full_computation;
        EXPECT_EQ(
            aoclsparse_csr2m<T>(
                aoclsparse_operation_transpose, descrA, csrA, transB, descrB, csrB, request, &csrC),
            aoclsparse_status_not_implemented);
        EXPECT_EQ(
            aoclsparse_csr2m<T>(
                transA, descrA, csrA, aoclsparse_operation_transpose, descrB, csrB, request, &csrC),
            aoclsparse_status_not_implemented);

        // FIX ME : Code doesnt check for descr attributes of matrix B
        // and expect not_implemented for aoclsparse_index_base_one
        aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_one);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_not_implemented);

        // and expect not_implemented for !aoclsparse_matrix_type_general
        aoclsparse_set_mat_index_base(descrA, aoclsparse_index_base_zero);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(aoclsparse_csr2m<T>(transA, descrA, csrA, transB, descrB, csrB, request, &csrC),
                  aoclsparse_status_not_implemented);

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
    TEST(csr2m, WrongSizeDouble)
    {
        test_csr2m_wrong_size<double>();
    }
    TEST(csr2m, WrongSizeFloat)
    {
        test_csr2m_wrong_size<float>();
    }

    TEST(csr2m, NotImplDouble)
    {
        test_csr2m_not_implemented<double>();
    }
    TEST(csr2m, NotImplFloat)
    {
        test_csr2m_not_implemented<float>();
    }

} // namespace
