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
#include "aoclsparse_auxiliary.h"
#include "aoclsparse_convert.h"
#include "aoclsparse_types.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_datatype2string.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_reference.hpp"
#include "aoclsparse_utility.hpp"

#include <algorithm>
#include <vector>

namespace
{
    typedef struct
    {
        const char           *testname;
        aoclsparse_operation  op;
        aoclsparse_index_base base_A;
        aoclsparse_index_base base_B;
        aoclsparse_int        M;
        aoclsparse_int        N;
        aoclsparse_int        nnzA;
        aoclsparse_int        nnzB;
    } AddCSRParam;

    template <typename T>
    void test_csr_add(aoclsparse_operation  op,
                      aoclsparse_index_base base_A,
                      aoclsparse_index_base base_B,
                      aoclsparse_int        M,
                      aoclsparse_int        N,
                      aoclsparse_int        nnz_A,
                      aoclsparse_int        nnz_B)
    {
        aoclsparse_matrix           src_mat_A = nullptr, src_mat_B = nullptr, dest_mat = nullptr;
        std::vector<aoclsparse_int> A_row, A_col, B_row, B_col, C_row_ref, C_col_ref;
        std::vector<T>              A_val, B_val, C_val_ref;
        aoclsparse_int              B_m = M, B_n = N, C_nnz = nnz_A + nnz_B;
        const char                  filename[] = "";
        bool                        issymm, general = true;
        aoclsparse_matrix_sort      sort = aoclsparse_unsorted; //aoclsparse_fully_sorted;

        aoclsparse_seedrand();
        T alpha = random_generator_normal<T>();

        ASSERT_EQ(aoclsparse_init_csr_matrix(A_row,
                                             A_col,
                                             A_val,
                                             M,
                                             N,
                                             nnz_A,
                                             base_A,
                                             aoclsparse_matrix_random,
                                             filename,
                                             issymm,
                                             general,
                                             sort),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr(
                      &src_mat_A, base_A, M, N, nnz_A, A_row.data(), A_col.data(), A_val.data()),
                  aoclsparse_status_success);

        if(op != aoclsparse_operation_none)
        {
            std::swap(B_m, B_n);
        }

        ASSERT_EQ(aoclsparse_init_csr_matrix(B_row,
                                             B_col,
                                             B_val,
                                             B_m,
                                             B_n,
                                             nnz_B,
                                             base_B,
                                             aoclsparse_matrix_random,
                                             filename,
                                             issymm,
                                             general,
                                             sort),
                  aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr(
                &src_mat_B, base_B, B_m, B_n, nnz_B, B_row.data(), B_col.data(), B_val.data()),
            aoclsparse_status_success);

        C_row_ref.resize(B_m + 1);
        if(C_nnz > 0)
        {
            C_col_ref.resize(C_nnz);
            C_val_ref.resize(C_nnz);
        }
        else
        {
            C_col_ref.reserve(1);
            C_val_ref.reserve(1);
        }

        ASSERT_EQ(ref_add(op,
                          base_A,
                          M,
                          N,
                          A_row.data(),
                          A_col.data(),
                          A_val.data(),
                          alpha,
                          base_B,
                          B_m,
                          B_n,
                          B_row.data(),
                          B_col.data(),
                          B_val.data(),
                          C_nnz,
                          C_row_ref.data(),
                          C_col_ref.data(),
                          C_val_ref.data()),
                  aoclsparse_status_success);

        std::vector<aoclsparse_int> row_ptr, col_ind;
        std::vector<T>              val;
        aoclsparse_int              n, m, nnz;
        aoclsparse_index_base       b;

        EXPECT_EQ(aoclsparse_add(op, src_mat_A, alpha, src_mat_B, &dest_mat),
                  aoclsparse_status_success);

        EXPECT_EQ(aocl_csr_sorted_export(dest_mat, b, m, n, nnz, row_ptr, col_ind, val),
                  aoclsparse_status_success);

        EXPECT_EQ(csrmat_check(B_m,
                               B_n,
                               C_nnz,
                               base_A,
                               C_row_ref,
                               C_col_ref,
                               C_val_ref,
                               m,
                               n,
                               nnz,
                               b,
                               row_ptr,
                               col_ind,
                               val),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat_A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat_B), aoclsparse_status_success);
    }

    // List of all desired negative tests
    const AddCSRParam AddCSRValues[] = {{"CSR_00B",
                                         aoclsparse_operation_none,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         10,
                                         11,
                                         15,
                                         5},
                                        {"CSR_10B_dense",
                                         aoclsparse_operation_none,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         10,
                                         11,
                                         95,
                                         50},
                                        {"CSR_01B_thin",
                                         aoclsparse_operation_none,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         1,
                                         100,
                                         6,
                                         12},
                                        {"CSR_11B_tall",
                                         aoclsparse_operation_none,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         97,
                                         1,
                                         25,
                                         93},
                                        {"CSR_00B_0nnzA",
                                         aoclsparse_operation_none,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         0,
                                         2},
                                        {"CSR_11B_0nnzB",
                                         aoclsparse_operation_none,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         4,
                                         9,
                                         34,
                                         0},
                                        {"CSRT_00B",
                                         aoclsparse_operation_transpose,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         6,
                                         4,
                                         5,
                                         22},
                                        {"CSRT_10B_tall",
                                         aoclsparse_operation_transpose,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         67,
                                         3,
                                         60,
                                         10},
                                        {"CSRT_01B_very_sparse",
                                         aoclsparse_operation_transpose,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         31,
                                         43,
                                         5,
                                         6},
                                        {"CSRT_11B_0nnzA",
                                         aoclsparse_operation_transpose,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         13,
                                         9,
                                         0,
                                         34},
                                        {"CSRT_10B_0nnz_both",
                                         aoclsparse_operation_transpose,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         2,
                                         2,
                                         0,
                                         0},
                                        {"CSRT_11B_1x1",
                                         aoclsparse_operation_transpose,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         1,
                                         1,
                                         1,
                                         1},
                                        {"CSRCT_00B_0nnzB",
                                         aoclsparse_operation_conjugate_transpose,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         8,
                                         7,
                                         5,
                                         0},
                                        {"CSRCT_01B",
                                         aoclsparse_operation_conjugate_transpose,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         5,
                                         9,
                                         15,
                                         23},
                                        {"CSRCT_11B_square",
                                         aoclsparse_operation_conjugate_transpose,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         9,
                                         9,
                                         70,
                                         53},
                                        {"CSRT_11B_tiny",
                                         aoclsparse_operation_transpose,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         1,
                                         4,
                                         1,
                                         0}};

    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const AddCSRParam &param, std::ostream *os)
    {
        *os << param.testname;
    }

    class CSR : public testing::TestWithParam<AddCSRParam>
    {
    };

    // tests with double type
    TEST_P(CSR, Double)
    {
        const AddCSRParam &param = GetParam();
        test_csr_add<double>(
            param.op, param.base_A, param.base_B, param.M, param.N, param.nnzA, param.nnzB);
    }

    // tests with float type
    TEST_P(CSR, Float)
    {
        const AddCSRParam &param = GetParam();
        test_csr_add<float>(
            param.op, param.base_A, param.base_B, param.M, param.N, param.nnzA, param.nnzB);
    }

    // tests with comlex double type
    TEST_P(CSR, ComplexDouble)
    {
        const AddCSRParam &param = GetParam();
        test_csr_add<aoclsparse_double_complex>(
            param.op, param.base_A, param.base_B, param.M, param.N, param.nnzA, param.nnzB);
    }

    // tests with complex float type
    TEST_P(CSR, ComplexFloat)
    {
        const AddCSRParam &param = GetParam();
        test_csr_add<aoclsparse_float_complex>(
            param.op, param.base_A, param.base_B, param.M, param.N, param.nnzA, param.nnzB);
    }

    INSTANTIATE_TEST_SUITE_P(AddTestSuite, CSR, testing::ValuesIn(AddCSRValues));

    TEST(AddValidationSuite, MatrixDimensionTest)
    {
        aoclsparse_matrix           A = nullptr, B = nullptr, C = nullptr;
        std::vector<aoclsparse_int> row_ptr(6, 0), col_ptr(1, 0);
        aoclsparse_int              m = 5, n = 4, nnz = 0;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        float                      *val  = new float;
        EXPECT_EQ(aoclsparse_create_csr(&A, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(&B, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_transpose, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_conjugate_transpose, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&B), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&C), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(&A, base, n, m, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(&B, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_none, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&B), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&C), aoclsparse_status_success);
        delete val;
    }

    TEST(AddValidationSuite, NonInitMatrix)
    {
        aoclsparse_matrix           A = nullptr, B = nullptr, C = nullptr;
        std::vector<aoclsparse_int> row_ptr(6, 0), col_ptr(1, 0);
        aoclsparse_int              m = 5, n = 4, nnz = 0;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        float                      *val  = new float;
        EXPECT_EQ(aoclsparse_create_csr(&A, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_none, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        A = nullptr;
        EXPECT_EQ(aoclsparse_create_csr(&B, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_transpose, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_destroy(&B), aoclsparse_status_success);
        delete val;
    }

    TEST(AddValidationSuite, WrongMatrixFormatTest)
    {
        aoclsparse_matrix           A = nullptr, B = nullptr, C = nullptr;
        std::vector<aoclsparse_int> row_ptr(6, 0), col_ptr(1, 0);
        aoclsparse_int              m = 5, n = 5, nnz = 0;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        float                      *val  = new float;
        EXPECT_EQ(aoclsparse_create_coo(&A, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(&B, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_transpose, A, 0.1f, B, &C),
                  aoclsparse_status_not_implemented);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&B), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&C), aoclsparse_status_success);
        delete val;
    }

    TEST(AddValidationSuite, WrongMatrixTypeTest)
    {
        aoclsparse_matrix           A = nullptr, B = nullptr, C = nullptr;
        std::vector<aoclsparse_int> row_ptr(6, 0), col_ptr(1, 0);
        aoclsparse_int              m = 5, n = 5, nnz = 0;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        float                      *fval = new float;
        double                     *dval = new double;
        EXPECT_EQ(aoclsparse_create_csr(&A, base, m, n, nnz, row_ptr.data(), col_ptr.data(), dval),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(&B, base, m, n, nnz, row_ptr.data(), col_ptr.data(), fval),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_transpose, A, 0.1f, B, &C),
                  aoclsparse_status_wrong_type);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&B), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&C), aoclsparse_status_success);
        delete dval;
        delete fval;
    }

} // namespacecd
