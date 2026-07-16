/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "blis.hh"
#include "cblas.hh"
#pragma GCC diagnostic pop
#include "level3_test_common.hpp"

namespace
{
    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_spmm_nullptr(aoclsparse_int        m_a,
                           aoclsparse_int        n_a,
                           aoclsparse_int        m_b,
                           aoclsparse_int        n_b,
                           aoclsparse_int        nnz_a,
                           aoclsparse_int        nnz_b,
                           aoclsparse_index_base b_a,
                           aoclsparse_index_base b_b,
                           aoclsparse_operation  op_a)
    {
        SCOPED_TRACE(make_test_name<T>(m_a, n_a, m_b, n_b));
        aoclsparse_seedrand();
        spmm_mats<T>      src;
        aoclsparse_matrix A = nullptr, B = nullptr;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B);
        aoclsparse_matrix C = NULL;
        EXPECT_EQ(aoclsparse_spmm(op_a, nullptr, B, &C), aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_spmm(op_a, A, nullptr, &C), aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_spmm(op_a, A, B, nullptr), aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }

    template <typename T>
    void test_spmm_do_nothing(aoclsparse_int        m_a,
                              aoclsparse_int        n_a,
                              aoclsparse_int        m_b,
                              aoclsparse_int        n_b,
                              aoclsparse_int        nnz_a,
                              aoclsparse_int        nnz_b,
                              aoclsparse_index_base b_a,
                              aoclsparse_index_base b_b,
                              aoclsparse_operation  op_a)
    {
        SCOPED_TRACE(make_test_name<T>(m_a, n_a, m_b, n_b));
        aoclsparse_seedrand();
        spmm_mats<T>      src;
        aoclsparse_matrix A, B;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B);

        aoclsparse_matrix C = NULL;
        A->m                = 0;
        EXPECT_EQ(aoclsparse_spmm(op_a, A, B, &C), aoclsparse_status_success);
        aoclsparse_destroy(&C);

        A->m = m_a;
        B->n = 0;
        EXPECT_EQ(aoclsparse_spmm(op_a, A, B, &C), aoclsparse_status_success);
        aoclsparse_destroy(&C);

        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
    }

    // tests for Wrong size
    template <typename T>
    void test_spmm_wrong_size(aoclsparse_int        m_a,
                              aoclsparse_int        n_a,
                              aoclsparse_int        m_b,
                              aoclsparse_int        n_b,
                              aoclsparse_int        nnz_a,
                              aoclsparse_int        nnz_b,
                              aoclsparse_index_base b_a,
                              aoclsparse_index_base b_b,
                              aoclsparse_operation  op_a)
    {
        SCOPED_TRACE(make_test_name<T>(m_a, n_a, m_b, n_b));
        aoclsparse_seedrand();
        spmm_mats<T>      src;
        aoclsparse_matrix A, B;
        gen_AB(m_a, n_a, m_b, n_b, nnz_a, nnz_b, b_a, b_b, src, A, B);

        aoclsparse_matrix C = NULL;
        EXPECT_EQ(aoclsparse_spmm(op_a, A, B, &C), aoclsparse_status_invalid_size);
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }

    // non-CSR input format — exercises the input_format guard in sp2m template
    template <typename T>
    void test_spmm_not_implemented(aoclsparse_index_base base)
    {
        aoclsparse_int              m = 4, k = 4, n = 4, nnz = 0;
        std::vector<aoclsparse_int> row_ind(1, 0), col_ind_v(1, 0);
        std::vector<T>              val(1);
        std::vector<aoclsparse_int> csr_row_ptr(m + 1, 0);

        aoclsparse_matrix A_coo = nullptr, B_csr = nullptr, C = nullptr;
        ASSERT_EQ(aoclsparse_create_coo<T>(
                      &A_coo, base, m, k, nnz, row_ind.data(), col_ind_v.data(), val.data()),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(
                      &B_csr, base, k, n, nnz, csr_row_ptr.data(), col_ind_v.data(), val.data()),
                  aoclsparse_status_success);
        // COO A, CSR B
        EXPECT_EQ(aoclsparse_spmm(aoclsparse_operation_none, A_coo, B_csr, &C),
                  aoclsparse_status_not_implemented);
        aoclsparse_destroy(&C);
        // CSR A, COO B
        EXPECT_EQ(aoclsparse_spmm(aoclsparse_operation_none, B_csr, A_coo, &C),
                  aoclsparse_status_not_implemented);
        aoclsparse_destroy(&A_coo);
        aoclsparse_destroy(&B_csr);
        aoclsparse_destroy(&C);
    }

    void test_spmm_wrong_datatype()
    {
        aoclsparse_operation  opA  = aoclsparse_operation_none;
        aoclsparse_index_base base = aoclsparse_index_base_zero;
        aoclsparse_int        m = 2, k = 3, n = 2, nnzA = 1, nnzB = 3;
        float                 csr_valA[]     = {42.};
        aoclsparse_int        csr_col_indA[] = {1};
        aoclsparse_int        csr_row_ptrA[] = {0, 0, 1};
        double                csr_valB[]     = {42., 21., 11.};
        aoclsparse_int        csr_col_indB[] = {1, 0, 1};
        aoclsparse_int        csr_row_ptrB[] = {0, 1, 2, 3};

        aoclsparse_matrix A;
        aoclsparse_create_scsr(&A, base, m, k, nnzA, csr_row_ptrA, csr_col_indA, csr_valA);
        aoclsparse_matrix B;
        aoclsparse_create_dcsr(&B, base, k, n, nnzB, csr_row_ptrB, csr_col_indB, csr_valB);
        aoclsparse_matrix C = NULL;
        // For float A and double B matrices, invoke spmm
        // and expect wrong type error
        EXPECT_EQ(aoclsparse_spmm(opA, A, B, &C), aoclsparse_status_wrong_type);

        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&C);
    }

    // Test for success and verify results against Dense GEMM results.
    // use_csr_a/b=false creates CSC handles instead of CSR.
    template <typename T>
    void test_spmm_success(aoclsparse_int        m_a,
                           aoclsparse_int        n_a,
                           aoclsparse_int        m_b,
                           aoclsparse_int        n_b,
                           aoclsparse_int        nnz_a,
                           aoclsparse_int        nnz_b,
                           aoclsparse_index_base b_a,
                           aoclsparse_index_base b_b,
                           aoclsparse_operation  op_a,
                           bool                  use_csr_a = true,
                           bool                  use_csr_b = true)
    {
        SCOPED_TRACE(make_test_name<T>(m_a, n_a, m_b, n_b) + " A=" + (use_csr_a ? "CSR" : "CSC")
                     + " B=" + (use_csr_b ? "CSR" : "CSC"));
        aoclsparse_seedrand();
        tolerance_t<T>       abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());
        spmm_mats<T>         src;
        aoclsparse_matrix    A, B, C = NULL;
        aoclsparse_mat_descr descrA, descrB;
        gen_AB(m_a,
               n_a,
               m_b,
               n_b,
               nnz_a,
               nnz_b,
               b_a,
               b_b,
               src,
               A,
               B,
               &descrA,
               &descrB,
               use_csr_a,
               use_csr_b);

        EXPECT_EQ(aoclsparse_spmm(op_a, A, B, &C), aoclsparse_status_success);

        aoclsparse_int m_c, n_c;
        std::vector<T> dense_c;
        export_and_dense<T>(C, m_c, n_c, dense_c);

        std::vector<T> dense_a(m_a * n_a), dense_b(m_b * n_b);
        aoclsparse_csr2dense(m_a,
                             n_a,
                             descrA,
                             src.val_a.data(),
                             src.row_ptr_a.data(),
                             src.col_ind_a.data(),
                             dense_a.data(),
                             n_a,
                             aoclsparse_order_row);
        aoclsparse_csr2dense(m_b,
                             n_b,
                             descrB,
                             src.val_b.data(),
                             src.row_ptr_b.data(),
                             src.col_ind_b.data(),
                             dense_b.data(),
                             n_b,
                             aoclsparse_order_row);
        auto dense_c_exp = gemm_ref<T>(m_c, n_c, m_a, n_a, n_b, dense_a, dense_b, op_a);
        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            EXPECT_COMPLEX_ARR_NEAR(m_c * n_c,
                                    ((std::complex<float> *)dense_c.data()),
                                    ((std::complex<float> *)dense_c_exp.data()),
                                    abserr);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            EXPECT_COMPLEX_ARR_NEAR(m_c * n_c,
                                    ((std::complex<double> *)dense_c.data()),
                                    ((std::complex<double> *)dense_c_exp.data()),
                                    abserr);
        }
        else
        {
            EXPECT_ARR_NEAR(m_c * n_c, dense_c.data(), dense_c_exp.data(), abserr);
        }

        aoclsparse_destroy(&C);
        aoclsparse_destroy_mat_descr(descrB);
        aoclsparse_destroy(&B);
        aoclsparse_destroy_mat_descr(descrA);
        aoclsparse_destroy(&A);
    }

    TEST(spmm, NullArgAll)
    {
        test_spmm_nullptr<double>(3,
                                  2,
                                  2,
                                  4,
                                  2,
                                  5,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_one,
                                  aoclsparse_operation_none);
        test_spmm_nullptr<float>(3,
                                 2,
                                 3,
                                 4,
                                 2,
                                 4,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_index_base_one,
                                 aoclsparse_operation_conjugate_transpose);
    }
    TEST(spmm, DoNothingAll)
    {
        test_spmm_do_nothing<aoclsparse_double_complex>(5,
                                                        4,
                                                        4,
                                                        5,
                                                        7,
                                                        9,
                                                        aoclsparse_index_base_zero,
                                                        aoclsparse_index_base_one,
                                                        aoclsparse_operation_none);
        test_spmm_do_nothing<aoclsparse_float_complex>(3,
                                                       4,
                                                       4,
                                                       5,
                                                       7,
                                                       9,
                                                       aoclsparse_index_base_one,
                                                       aoclsparse_index_base_zero,
                                                       aoclsparse_operation_none);
    }
    TEST(spmm, WrongSizeAll)
    {
        test_spmm_wrong_size<float>(3,
                                    2,
                                    3,
                                    5,
                                    2,
                                    5,
                                    aoclsparse_index_base_zero,
                                    aoclsparse_index_base_zero,
                                    aoclsparse_operation_none);
        test_spmm_wrong_size<aoclsparse_double_complex>(3,
                                                        4,
                                                        6,
                                                        5,
                                                        7,
                                                        9,
                                                        aoclsparse_index_base_one,
                                                        aoclsparse_index_base_one,
                                                        aoclsparse_operation_conjugate_transpose);
    }

    TEST(spmm, WrongType)
    {
        test_spmm_wrong_datatype();
    }
    TEST(spmm, NotImplAll)
    {
        constexpr auto base0 = aoclsparse_index_base_zero;
        test_spmm_not_implemented<double>(base0);
        test_spmm_not_implemented<aoclsparse_float_complex>(base0);
    }
    TEST(spmm, SuccessTypeDouble)
    {
        test_spmm_success<double>(4,
                                  4,
                                  4,
                                  4,
                                  10,
                                  8,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_operation_transpose);
        test_spmm_success<double>(5,
                                  6,
                                  6,
                                  7,
                                  10,
                                  15,
                                  aoclsparse_index_base_zero,
                                  aoclsparse_index_base_one,
                                  aoclsparse_operation_none);
    }
    TEST(spmm, SuccessTypeFloat)
    {
        test_spmm_success<float>(5,
                                 5,
                                 5,
                                 5,
                                 11,
                                 7,
                                 aoclsparse_index_base_one,
                                 aoclsparse_index_base_zero,
                                 aoclsparse_operation_transpose);
        test_spmm_success<float>(5,
                                 4,
                                 5,
                                 6,
                                 11,
                                 17,
                                 aoclsparse_index_base_one,
                                 aoclsparse_index_base_one,
                                 aoclsparse_operation_conjugate_transpose);
    }
    TEST(spmm, SuccessTypeCDouble)
    {
        test_spmm_success<aoclsparse_double_complex>(5,
                                                     5,
                                                     5,
                                                     5,
                                                     11,
                                                     17,
                                                     aoclsparse_index_base_one,
                                                     aoclsparse_index_base_zero,
                                                     aoclsparse_operation_conjugate_transpose);
        test_spmm_success<aoclsparse_double_complex>(5,
                                                     4,
                                                     4,
                                                     6,
                                                     11,
                                                     17,
                                                     aoclsparse_index_base_one,
                                                     aoclsparse_index_base_one,
                                                     aoclsparse_operation_none);
    }
    TEST(spmm, SuccessTypeCFloat)
    {
        test_spmm_success<aoclsparse_float_complex>(6,
                                                    6,
                                                    6,
                                                    6,
                                                    12,
                                                    15,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_index_base_one,
                                                    aoclsparse_operation_conjugate_transpose);
        test_spmm_success<aoclsparse_float_complex>(5,
                                                    7,
                                                    5,
                                                    6,
                                                    10,
                                                    15,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_index_base_zero,
                                                    aoclsparse_operation_conjugate_transpose);
    }
    TEST(spmm, CSCInputSuccess)
    {
        constexpr auto base0    = aoclsparse_index_base_zero;
        constexpr auto base1    = aoclsparse_index_base_one;
        constexpr auto op_none  = aoclsparse_operation_none;
        constexpr auto op_trans = aoclsparse_operation_transpose;
        constexpr auto op_conj  = aoclsparse_operation_conjugate_transpose;

        // CSR×CSC: op_none
        test_spmm_success<double>(4, 4, 4, 4, 8, 8, base0, base0, op_none, true, false);
        test_spmm_success<float>(3, 5, 5, 4, 7, 9, base1, base0, op_none, true, false);
        // CSC×CSR: op_none and op_trans
        test_spmm_success<double>(4, 4, 4, 4, 8, 8, base0, base0, op_none, false, true);
        test_spmm_success<double>(5, 4, 5, 3, 9, 7, base0, base0, op_trans, false, true);
        test_spmm_success<float>(3, 5, 3, 4, 7, 7, base1, base0, op_trans, false, true);
        // CSC×CSC: op_trans
        test_spmm_success<double>(4, 5, 4, 3, 9, 7, base0, base0, op_trans, false, false);
        // complex: CSC×CSR op_none and op_conj, CSC×CSC op_conj
        test_spmm_success<aoclsparse_double_complex>(
            4, 4, 4, 4, 8, 8, base0, base0, op_none, false, true);
        test_spmm_success<aoclsparse_double_complex>(
            4, 4, 4, 4, 8, 8, base0, base0, op_conj, false, true);
        test_spmm_success<aoclsparse_float_complex>(
            3, 5, 3, 4, 7, 7, base0, base1, op_conj, false, false);
    }
} // namespace
