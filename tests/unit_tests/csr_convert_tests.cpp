/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_datatype2string.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"
#include "aoclsparse_random.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace
{
    typedef struct
    {
        const char                   *testname;
        aoclsparse_operation          op;
        aoclsparse_index_base         base;
        aoclsparse_int                m;
        aoclsparse_int                n;
        aoclsparse_int                nnz;
        aoclsparse_matrix_format_type format_type;
        aoclsparse::doid              doid = aoclsparse::doid::len;
    } ConvertCSRParam;

    template <typename T>
    void test_convert_csr(aoclsparse_operation          op,
                          aoclsparse_index_base         base,
                          aoclsparse_int                m,
                          aoclsparse_int                n,
                          aoclsparse_int                nnz,
                          aoclsparse_matrix_format_type format_type,
                          aoclsparse::doid              doid)
    {
        aoclsparse_seedrand();

        aoclsparse_matrix src_mat = nullptr, dest_mat = nullptr;

        std::vector<aoclsparse_int> coo_row, coo_col, ptr;
        std::vector<T>              val;

        EXPECT_EQ(aoclsparse_init_matrix_random(
                      base, m, n, nnz, format_type, coo_row, coo_col, val, ptr, src_mat, doid),
                  aoclsparse_status_success);

        aoclsparse_int       *dest_row_ptr = nullptr;
        aoclsparse_int       *dest_col_ind = nullptr;
        T                    *dest_val     = nullptr;
        aoclsparse_int        dest_n, dest_m, dest_nnz;
        aoclsparse_index_base dest_b;

        EXPECT_EQ(aoclsparse_convert_csr(src_mat, op, &dest_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_export_csr(dest_mat,
                                        &dest_b,
                                        &dest_m,
                                        &dest_n,
                                        &dest_nnz,
                                        &dest_row_ptr,
                                        &dest_col_ind,
                                        &dest_val),
                  aoclsparse_status_success);

        // destroy src_mat before we start manipulating arrays it was created from (coo_row/col, val, ptr)
        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);

        if(op != aoclsparse_operation_none)
        {
            ASSERT_EQ(dest_m, n);
            ASSERT_EQ(dest_n, m);
            swap(coo_col, coo_row);
            // Conjugate if necessary
            if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                         || std::is_same_v<T, aoclsparse_double_complex>)
                if(op == aoclsparse_operation_conjugate_transpose)
                {
                    for(aoclsparse_int i = 0; i < nnz; ++i)
                        val[i].imag = -val[i].imag;
                }
        }
        else
        {
            ASSERT_EQ(dest_m, m);
            ASSERT_EQ(dest_n, n);
        }
        ASSERT_EQ(dest_b, base);
        ASSERT_EQ(dest_nnz, nnz);

        // Generate expected results from input COO & sorting
        std::vector<aoclsparse_int> exp_col(nnz), exp_row_ptr;
        std::vector<T>              exp_val(nnz);

        // sort to CSR order
        std::vector<aoclsparse_int> idxs(nnz);
        std::iota(idxs.begin(), idxs.end(), 0);
        sort(idxs.begin(), idxs.end(), [coo_col, coo_row](auto a, auto b) {
            if(coo_row[a] == coo_row[b])
                return coo_col[a] < coo_col[b];
            return coo_row[a] < coo_row[b];
        });
        std::sort(coo_row.begin(), coo_row.end());
        for(aoclsparse_int i = 0; i < nnz; i++)
        {
            exp_col[i] = coo_col[idxs[i]];
            exp_val[i] = val[idxs[i]];
        }
        coo_to_csr(dest_m, nnz, coo_row, exp_row_ptr, base);

        // compare
        EXPECT_EQ_VEC(dest_m + 1, exp_row_ptr, dest_row_ptr);
        EXPECT_EQ_VEC(nnz, exp_col, dest_col_ind);
        tolerance_t<T> abs_error = expected_precision<tolerance_t<T>>();
        EXPECT_ARR_MATCH(T, nnz, exp_val.data(), dest_val, 0. /*rel_error*/, abs_error);

        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
    }

#define NOOP aoclsparse_operation_none
#define TRANS aoclsparse_operation_transpose
#define HERM aoclsparse_operation_conjugate_transpose

#define B0 aoclsparse_index_base_zero
#define B1 aoclsparse_index_base_one

#define COO aoclsparse_coo_mat
#define CSR aoclsparse_csr_mat

    // List of all desired negative tests
    const ConvertCSRParam ConvertCSRValues[] = {
        {"cooToCsr__0NNZ_0b", NOOP, B0, 1, 1, 0, COO},
        {"cscToCsr__0NNZ_1b", NOOP, B1, 10, 1, 0, CSR, aoclsparse::doid::gt},
        {"csrToCsr__0NNZ_0b", NOOP, B0, 1, 5, 0, CSR},
        {"cooToCsrT_0NNZ_1b", TRANS, B1, 4, 3, 0, COO},
        {"cscToCsrH_0NNZ_0b", HERM, B0, 4, 30, 0, CSR, aoclsparse::doid::gt},
        {"csrToCsrT_0NNZ_1b", TRANS, B1, 5, 1, 0, CSR},

        {"cooToCsr__0b", NOOP, B0, 1, 11, 10, COO},
        {"cscToCsr__0b", NOOP, B0, 1, 11, 5, CSR, aoclsparse::doid::gt},
        {"csrToCsr__0b", NOOP, B0, 1, 1, 1, CSR},

        {"cooToCsrT_1b", TRANS, B1, 7, 3, 13, COO},
        {"cscToCsrT_0b", TRANS, B0, 7, 5, 2, CSR, aoclsparse::doid::gt},
        {"csrToCsrT_1b", TRANS, B1, 1, 3, 3, CSR},

        {"cooToCsrH_0b", HERM, B0, 16, 10, 100, COO},
        {"cscToCsrH_1b", HERM, B1, 7, 7, 20, CSR, aoclsparse::doid::gt},
        {"csrToCsrH_0b", HERM, B0, 6, 10, 40, CSR},

        {"cooToCsr__1B", NOOP, B1, 6, 10, 1, COO},
        {"cscToCsr__1B", NOOP, B1, 6, 4, 10, CSR, aoclsparse::doid::gt},
        {"csrToCsr__1B", NOOP, B1, 6, 10, 10, CSR},
    };

    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const ConvertCSRParam &param, std::ostream *os)
    {
        *os << param.testname;
    }

    class Pos : public testing::TestWithParam<ConvertCSRParam>
    {
    };

    // tests with double type
    TEST_P(Pos, Double)
    {
        const ConvertCSRParam &param = GetParam();
        test_convert_csr<double>(
            param.op, param.base, param.m, param.n, param.nnz, param.format_type, param.doid);
    }

    // tests with float type
    TEST_P(Pos, Float)
    {
        const ConvertCSRParam &param = GetParam();
        test_convert_csr<float>(
            param.op, param.base, param.m, param.n, param.nnz, param.format_type, param.doid);
    }

    // tests with double type
    TEST_P(Pos, ComplexDouble)
    {
        const ConvertCSRParam &param = GetParam();
        test_convert_csr<aoclsparse_double_complex>(
            param.op, param.base, param.m, param.n, param.nnz, param.format_type, param.doid);
    }

    // tests with float type
    TEST_P(Pos, ComplexFloat)
    {
        const ConvertCSRParam &param = GetParam();
        test_convert_csr<aoclsparse_float_complex>(
            param.op, param.base, param.m, param.n, param.nnz, param.format_type, param.doid);
    }

    INSTANTIATE_TEST_SUITE_P(ConvertCSRTestSuite, Pos, testing::ValuesIn(ConvertCSRValues));

    TEST(ConvertCSRErrorSuite, UinitializedMatrix)
    {
        aoclsparse_matrix src_mat = nullptr, dest_mat = nullptr;
        EXPECT_EQ(aoclsparse_convert_csr(src_mat, aoclsparse_operation_none, &dest_mat),
                  aoclsparse_status_invalid_pointer);
    }

#if NEW_FORMAT

    TEST(ConvertCSRErrorSuite, NotImplementedMatrixFormat)
    {
        // TODO uncomment the checks once a format comes up which we will not support

        // aoclsparse_matrix src_mat, dest_mat;
        // aoclsparse_create_ell_csr_hyb(src_mat, 3, 2, ell_col_ind, ell_col_ind, val_float);
        // aoclsparse_create_csr(
        //     &dest_mat, aoclsparse_index_base_zero, 4, 3, 3, csr_row_ptr, csr_col_ind, val_float);

        // EXPECT_EQ(aoclsparse_convert_csr(src_mat, aoclsparse_operation_none, &dest_mat),
        //           aoclsparse_status_not_implemented);
        // EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        // EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
#endif
    TEST(ConvertCSR, UnsortedCoo2Csr)
    {
        aoclsparse_index_base       base = aoclsparse_index_base_one;
        aoclsparse_int              m = 5, n = 5, nnz = 9;
        std::vector<aoclsparse_int> coo_row_idx = {1, 2, 3, 4, 5, 1, 2, 4, 5};
        std::vector<aoclsparse_int> coo_col_idx = {1, 2, 3, 4, 5, 5, 4, 2, 1};
        std::vector<double>         coo_val     = {1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 2.0, 1.0};
        aoclsparse_matrix           src_mat, dest_mat;
        aoclsparse_int             *dest_row_ptr = nullptr;
        aoclsparse_int             *dest_col_idx = nullptr;
        double                     *dest_val     = nullptr;
        aoclsparse_int              dest_n, dest_m, dest_nnz;
        aoclsparse_index_base       dest_b;
        /*      1       2       3       4       5
            ______________________________________
            1|  1.0     0.0     0.0     0.0     5.0
            2|  0.0     2.0     0.0     4.0     0.0
            3|  0.0     0.0     3.0     0.0     0.0
            4|  0.0     2.0     0.0     4.0     0.0
            5|  1.0     0.0     0.0     0.0     5.0
        */
        ASSERT_EQ(
            aoclsparse_create_coo(
                &src_mat, base, m, n, nnz, coo_row_idx.data(), coo_col_idx.data(), coo_val.data()),
            aoclsparse_status_success);

        std::vector<aoclsparse_int> csr_row_ptr = {1, 3, 5, 6, 8, 10};
        std::vector<aoclsparse_int> csr_col_idx = {1, 5, 2, 4, 3, 4, 2, 5, 1};
        // op = aoclsparse_operation_none
        std::vector<double> csr_val_1 = {1.0, 5.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 1.0};
        EXPECT_EQ(aoclsparse_convert_csr(src_mat, aoclsparse_operation_none, &dest_mat),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_export_csr(dest_mat,
                                        &dest_b,
                                        &dest_m,
                                        &dest_n,
                                        &dest_nnz,
                                        &dest_row_ptr,
                                        &dest_col_idx,
                                        &dest_val),
                  aoclsparse_status_success);
        EXPECT_EQ(dest_b, base);
        EXPECT_EQ(dest_m, m);
        EXPECT_EQ(dest_n, n);
        EXPECT_EQ(dest_nnz, nnz);
        EXPECT_EQ_VEC(dest_nnz, csr_col_idx, dest_col_idx);
        EXPECT_EQ_VEC(dest_m + 1, csr_row_ptr, dest_row_ptr);
        EXPECT_EQ_VEC(dest_nnz, csr_val_1, dest_val);
        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);

        // op = aoclsparse_operation_transpose
        std::vector<double> csr_val_2 = {1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0};
        EXPECT_EQ(aoclsparse_convert_csr(src_mat, aoclsparse_operation_transpose, &dest_mat),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_export_csr(dest_mat,
                                        &dest_b,
                                        &dest_m,
                                        &dest_n,
                                        &dest_nnz,
                                        &dest_row_ptr,
                                        &dest_col_idx,
                                        &dest_val),
                  aoclsparse_status_success);
        EXPECT_EQ(dest_b, base);
        EXPECT_EQ(dest_m, m);
        EXPECT_EQ(dest_n, n);
        EXPECT_EQ(dest_nnz, nnz);
        EXPECT_EQ_VEC(dest_nnz, csr_col_idx, dest_col_idx);
        EXPECT_EQ_VEC(dest_m + 1, csr_row_ptr, dest_row_ptr);
        EXPECT_EQ_VEC(dest_nnz, csr_val_2, dest_val);
        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
    TEST(ConvertCSR, UnsortedCsc2Csr)
    {
        aoclsparse_index_base       base = aoclsparse_index_base_one;
        aoclsparse_int              m = 5, n = 5, nnz = 9;
        std::vector<aoclsparse_int> csc_col_ptr = {1, 3, 5, 6, 8, 10};
        std::vector<aoclsparse_int> csc_row_idx = {1, 5, 2, 4, 3, 4, 2, 5, 1};
        std::vector<double>         csc_val     = {1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0};
        aoclsparse_matrix           src_mat, dest_mat;
        aoclsparse_int             *dest_row_ptr = nullptr;
        aoclsparse_int             *dest_col_idx = nullptr;
        double                     *dest_val     = nullptr;
        aoclsparse_int              dest_n, dest_m, dest_nnz;
        aoclsparse_index_base       dest_b;
        /*      1       2       3       4       5
            ______________________________________
            1|  1.0     0.0     0.0     0.0     5.0
            2|  0.0     2.0     0.0     4.0     0.0
            3|  0.0     0.0     3.0     0.0     0.0
            4|  0.0     2.0     0.0     4.0     0.0
            5|  1.0     0.0     0.0     0.0     5.0
        */
        ASSERT_EQ(
            aoclsparse_create_csc(
                &src_mat, base, m, n, nnz, csc_col_ptr.data(), csc_row_idx.data(), csc_val.data()),
            aoclsparse_status_success);

        // op = aoclsparse_operation_none
        std::vector<aoclsparse_int> csr_row_ptr = {1, 3, 5, 6, 8, 10};
        std::vector<aoclsparse_int> csr_col_idx = {1, 5, 2, 4, 3, 2, 4, 1, 5};
        std::vector<double>         csr_val     = {1.0, 5.0, 2.0, 4.0, 3.0, 2.0, 4.0, 1.0, 5.0};
        EXPECT_EQ(aoclsparse_convert_csr(src_mat, aoclsparse_operation_none, &dest_mat),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_export_csr(dest_mat,
                                        &dest_b,
                                        &dest_m,
                                        &dest_n,
                                        &dest_nnz,
                                        &dest_row_ptr,
                                        &dest_col_idx,
                                        &dest_val),
                  aoclsparse_status_success);
        EXPECT_EQ(dest_b, base);
        EXPECT_EQ(dest_m, m);
        EXPECT_EQ(dest_n, n);
        EXPECT_EQ(dest_nnz, nnz);
        EXPECT_EQ_VEC(dest_nnz, csr_col_idx, dest_col_idx);
        EXPECT_EQ_VEC(dest_m + 1, csr_row_ptr, dest_row_ptr);
        EXPECT_EQ_VEC(dest_nnz, csr_val, dest_val);
        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
} // namespace
