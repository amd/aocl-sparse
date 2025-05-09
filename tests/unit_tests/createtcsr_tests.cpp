/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_interface.hpp"

#include <algorithm>
#include <complex>
#include <vector>

namespace
{
    template <typename T>
    void test_success(std::string            testcase,
                      aoclsparse_matrix_sort sorted,
                      aoclsparse_index_base  base)
    {

        SCOPED_TRACE(testcase);
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr_L;
        std::vector<aoclsparse_int> row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L;
        std::vector<aoclsparse_int> col_idx_U;
        std::vector<T>              val_L;
        std::vector<T>              val_U;
        aoclsparse_matrix           A;

        init_tcsr_matrix(
            m, n, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U, sorted, base);

        EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            m,
                                            n,
                                            nnz,
                                            row_ptr_L.data(),
                                            row_ptr_U.data(),
                                            col_idx_L.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_success);

        aoclsparse_int lnnz = row_ptr_L[m] - base; // (L + D)nnz
        aoclsparse_int unnz = row_ptr_U[m] - base; // (D + U)nnz

        EXPECT_EQ_VEC(m + 1, row_ptr_L, A->tcsr_mat.row_ptr_L);
        EXPECT_EQ_VEC(m + 1, row_ptr_U, A->tcsr_mat.row_ptr_U);
        EXPECT_EQ_VEC(lnnz, col_idx_L, A->tcsr_mat.col_idx_L);
        EXPECT_EQ_VEC(unnz, col_idx_U, A->tcsr_mat.col_idx_U);

        EXPECT_EQ(m, A->m);
        EXPECT_EQ(n, A->n);
        EXPECT_EQ(nnz, A->nnz);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    template <typename T>
    void test_nullptr(void)
    {
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr_L;
        std::vector<aoclsparse_int> row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L;
        std::vector<aoclsparse_int> col_idx_U;
        std::vector<T>              val_L;
        std::vector<T>              val_U;
        aoclsparse_matrix           A;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;

        init_tcsr_matrix(m,
                         n,
                         nnz,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         base);

        // pass nullptr and expect pointer error
        // A = nullptr
        {
            SCOPED_TRACE("Nullptr A");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(nullptr,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_pointer);
        }
        // null row_ptr_L
        {
            SCOPED_TRACE("Nullptr row_ptr_L");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                nullptr,
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_pointer);
        }
        // null row_ptr_U
        {
            SCOPED_TRACE("Nullptr row_ptr_U");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                nullptr,
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_pointer);
        }
        // null col_idx_L
        {
            SCOPED_TRACE("Nullptr col_idx_L");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                nullptr,
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_pointer);
        }
        // null col_idx_U
        {
            SCOPED_TRACE("Nullptr col_idx_U");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                nullptr,
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_pointer);
        }
        // null val_L
        {
            SCOPED_TRACE("Nullptr val_L");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                nullptr,
                                                val_U.data()),
                      aoclsparse_status_invalid_pointer);
        }
        // null val_U
        {
            SCOPED_TRACE("Nullptr val_U");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                nullptr),
                      aoclsparse_status_invalid_pointer);
        }
    }

    template <typename T>
    void test_zero_dimension(void)
    {
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr_L;
        std::vector<aoclsparse_int> row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L;
        std::vector<aoclsparse_int> col_idx_U;
        std::vector<T>              val_L;
        std::vector<T>              val_U;
        aoclsparse_matrix           A;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;

        init_tcsr_matrix(m,
                         n,
                         nnz,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         base);

        // m = 0, n = 0, nnz = 0
        {
            SCOPED_TRACE("Zero Dimension - m = 0, n = 0, nnz = 0");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                0,
                                                0,
                                                0,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, true);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
    }

    template <typename T>
    void test_invalid_size(void)
    {
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr_L;
        std::vector<aoclsparse_int> row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L;
        std::vector<aoclsparse_int> col_idx_U;
        std::vector<T>              val_L;
        std::vector<T>              val_U;
        aoclsparse_matrix           A;
        aoclsparse_index_base       base = aoclsparse_index_base_zero;

        init_tcsr_matrix(m,
                         n,
                         nnz,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         base);

        // Rectangle matrix
        // m=4, n=5, nnz=9
        {
            SCOPED_TRACE("Invalid rectangular matrix");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                5,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_size);
        }

        // m = -1, n = 4, nnz = 9
        {
            SCOPED_TRACE("Invalid size m");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                -1,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_size);
        }
        // m = 4, n = -1, nnz = 9
        {
            SCOPED_TRACE("Invalid size n");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                -1,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_size);
        }
        // m = 4, n = 4, nnz = -1
        {
            SCOPED_TRACE("Invalid size nnz");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                -1,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_size);
        }
    }

    template <typename T>
    void test_unsorted(aoclsparse_index_base base)
    {
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr_L;
        std::vector<aoclsparse_int> row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L;
        std::vector<aoclsparse_int> col_idx_U;
        std::vector<T>              val_L;
        std::vector<T>              val_U;
        aoclsparse_matrix           A;
        std::vector<aoclsparse_int> col_unsorted_L, col_unsorted_U;

        init_tcsr_matrix(m,
                         n,
                         nnz,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_unsorted,
                         base);

        // both lower and upper triangles unsorted
        {
            SCOPED_TRACE("Unsorted matrix");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_unsorted_input);
        }

        // lower unsorted, upper sorted
        col_unsorted_U    = col_idx_U;
        col_unsorted_U[0] = base;
        col_unsorted_U[1] = 2 + base;
        {
            SCOPED_TRACE("Unsorted lower triangular matrix");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_unsorted_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_unsorted_input);
        }

        // lower sorted, upper unsorted
        col_unsorted_L    = col_idx_L;
        col_unsorted_L[4] = base;
        col_unsorted_L[6] = 3 + base;
        {
            SCOPED_TRACE("Unsorted upper triangular matrix");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_unsorted_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_unsorted_input);
        }
    }

    template <typename T>
    void test_missing_diag(aoclsparse_index_base base)
    {
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr_L;
        std::vector<aoclsparse_int> row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L;
        std::vector<aoclsparse_int> col_idx_U;
        std::vector<T>              val_L;
        std::vector<T>              val_U;
        aoclsparse_matrix           A;
        std::vector<aoclsparse_int> col, col_U;

        init_tcsr_matrix(m,
                         n,
                         nnz,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         base);

        // missing diagonal in the lower triangle
        col    = col_idx_L;
        col[6] = 2 + base;
        {
            SCOPED_TRACE("Missing diagonal in the lower triangle");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_value);
        }

        // missing diagonal in the upper triangle
        col_U    = col_idx_U;
        col_U[0] = 1 + base;
        {
            SCOPED_TRACE("Missing diagonal in the upper triangle");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row_ptr_L.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_value);
        }
    }

    template <typename T>
    void test_invalid_value(aoclsparse_index_base base)
    {
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr_L;
        std::vector<aoclsparse_int> row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L;
        std::vector<aoclsparse_int> col_idx_U;
        std::vector<T>              val_L;
        std::vector<T>              val_U;
        aoclsparse_matrix           A;
        std::vector<aoclsparse_int> row, col;

        init_tcsr_matrix(m,
                         n,
                         nnz,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         base);

        // ideally row ptr[0] = base
        row    = row_ptr_L;
        row[0] = 2;
        {
            SCOPED_TRACE("Invalid Value, row_ptr[0]!=base");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_value);
        }
        // ideally row ptr order should be non-decreasing
        row    = row_ptr_L;
        row[2] = base;
        {
            SCOPED_TRACE("Invalid Value, order of row_ptr not non-decreasing");
            EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                                base,
                                                m,
                                                n,
                                                nnz,
                                                row.data(),
                                                row_ptr_U.data(),
                                                col_idx_L.data(),
                                                col_idx_U.data(),
                                                val_L.data(),
                                                val_U.data()),
                      aoclsparse_status_invalid_value);
        }
    }

    template <typename T>
    void test_invalid_index(aoclsparse_index_base base)
    {
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr_L;
        std::vector<aoclsparse_int> row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L;
        std::vector<aoclsparse_int> col_idx_U;
        std::vector<T>              val_L;
        std::vector<T>              val_U;
        aoclsparse_matrix           A;
        std::vector<aoclsparse_int> row, col;

        init_tcsr_matrix(m,
                         n,
                         nnz,
                         row_ptr_L,
                         row_ptr_U,
                         col_idx_L,
                         col_idx_U,
                         val_L,
                         val_U,
                         aoclsparse_fully_sorted,
                         base);

        // col ind >= n, invalid col index
        col    = col_idx_L;
        col[0] = n + base;
        EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            m,
                                            n,
                                            nnz,
                                            row_ptr_L.data(),
                                            row_ptr_U.data(),
                                            col.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_invalid_index_value);
        // col ind < base, invalid col index
        col    = col_idx_L;
        col[0] = base - 1;
        EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            m,
                                            n,
                                            nnz,
                                            row_ptr_L.data(),
                                            row_ptr_U.data(),
                                            col.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_invalid_index_value);
        // different base index for row ptr and col idx
        if(base == aoclsparse_index_base_zero)
            col.assign({1, 2, 1, 3, 1, 2, 4}); // one based col idx
        else
            col.assign({0, 1, 0, 2, 0, 1, 3}); // zero based col idx
        EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            m,
                                            n,
                                            nnz,
                                            row_ptr_L.data(),
                                            row_ptr_U.data(),
                                            col.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_invalid_index_value);
        // upper triangular element in the lower part
        row.assign({base, 1 + base, 2 + base, 5 + base, 7 + base});
        col.assign({base, 1 + base, base, 2 + base, 3 + base, 1 + base, 3 + base});
        EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            m,
                                            n,
                                            nnz,
                                            row.data(),
                                            row_ptr_U.data(),
                                            col.data(),
                                            col_idx_U.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_invalid_index_value);
        // lower triangular element in the upper part
        row.assign({base, 2 + base, 4 + base, 5 + base, 6 + base});
        col.assign({base, 2 + base, 0 + base, 1 + base, 2 + base, 3 + base});
        EXPECT_EQ(aoclsparse_create_tcsr<T>(&A,
                                            base,
                                            m,
                                            n,
                                            nnz,
                                            row_ptr_L.data(),
                                            row.data(),
                                            col_idx_L.data(),
                                            col.data(),
                                            val_L.data(),
                                            val_U.data()),
                  aoclsparse_status_invalid_index_value);
    }

    TEST(createtcsr, SuccessAll)
    {
        // FullDiag + Fully Sorted + Zero Base
        test_success<float>("Success-Float-FullySorted-ZeroBase",
                            aoclsparse_fully_sorted,
                            aoclsparse_index_base_zero);
        test_success<double>("Success-Double-FullySorted-ZeroBase",
                             aoclsparse_fully_sorted,
                             aoclsparse_index_base_zero);
        test_success<aoclsparse_float_complex>("Success-ComplexFloat-FullySorted-ZeroBase",
                                               aoclsparse_fully_sorted,
                                               aoclsparse_index_base_zero);
        test_success<aoclsparse_double_complex>("Success-ComplexDouble-FullySorted-ZeroBase",
                                                aoclsparse_fully_sorted,
                                                aoclsparse_index_base_zero);

        // FullDiag + Fully Sorted + One Base
        test_success<float>("Success-Float-FullySorted-OneBase",
                            aoclsparse_fully_sorted,
                            aoclsparse_index_base_one);
        test_success<double>("Success-Double-FullySorted-OneBase",
                             aoclsparse_fully_sorted,
                             aoclsparse_index_base_one);
        test_success<aoclsparse_float_complex>("Success-ComplexFloat-FullySorted-OneBase",
                                               aoclsparse_fully_sorted,
                                               aoclsparse_index_base_one);
        test_success<aoclsparse_double_complex>("Success-ComplexDouble-FullySorted-OneBase",
                                                aoclsparse_fully_sorted,
                                                aoclsparse_index_base_one);

        // FullDiag + Partially Sorted + Zero Base
        test_success<float>("Success-Float-PartiallySorted-ZeroBase",
                            aoclsparse_partially_sorted,
                            aoclsparse_index_base_zero);
        test_success<double>("Success-Double-PartiallySorted-ZeroBase",
                             aoclsparse_partially_sorted,
                             aoclsparse_index_base_zero);
        test_success<aoclsparse_float_complex>("Success-ComplexFloat-PartiallySorted-ZeroBase",
                                               aoclsparse_partially_sorted,
                                               aoclsparse_index_base_zero);
        test_success<aoclsparse_double_complex>("Success-ComplexDouble-PartiallySorted-ZeroBase",
                                                aoclsparse_partially_sorted,
                                                aoclsparse_index_base_zero);

        // FullDiag + Partially Sorted + one Base
        test_success<float>("Success-Float-PartiallySorted-OneBase",
                            aoclsparse_partially_sorted,
                            aoclsparse_index_base_one);
        test_success<double>("Success-Double-PartiallySorted-OneBase",
                             aoclsparse_partially_sorted,
                             aoclsparse_index_base_one);
        test_success<aoclsparse_float_complex>("Success-ComplexFloat-PartiallySorted-OneBase",
                                               aoclsparse_partially_sorted,
                                               aoclsparse_index_base_one);
        test_success<aoclsparse_double_complex>("Success-ComplexDouble-PartiallySorted-OneBase",
                                                aoclsparse_partially_sorted,
                                                aoclsparse_index_base_one);
    }

    TEST(createtcsr, NullArg)
    {
        test_nullptr<float>();
        test_nullptr<double>();
    }

    TEST(createtcsr, ZeroDimensionMatrix)
    {
        test_zero_dimension<float>();
        test_zero_dimension<double>();
    }

    TEST(createtcsr, InvalidSize)
    {
        test_invalid_size<float>();
        test_invalid_size<double>();
    }

    TEST(createtcsr, UnsortedMatrix)
    {
        // base-0
        test_unsorted<float>(aoclsparse_index_base_zero);
        test_unsorted<double>(aoclsparse_index_base_zero);
        // base-1
        test_unsorted<float>(aoclsparse_index_base_one);
        test_unsorted<double>(aoclsparse_index_base_one);
    }

    TEST(createtcsr, MissingDiag)
    {
        // base-0
        test_missing_diag<float>(aoclsparse_index_base_zero);
        test_missing_diag<double>(aoclsparse_index_base_zero);
        // base-1
        test_missing_diag<float>(aoclsparse_index_base_one);
        test_missing_diag<double>(aoclsparse_index_base_one);
    }

    TEST(createtcsr, InvalidValue)
    {
        // base-0
        test_invalid_value<float>(aoclsparse_index_base_zero);
        test_invalid_value<double>(aoclsparse_index_base_zero);
        // base-1
        test_invalid_value<float>(aoclsparse_index_base_one);
        test_invalid_value<double>(aoclsparse_index_base_one);
    }

    TEST(createtcsr, InvalidIndex)
    {
        // base-0
        test_invalid_index<float>(aoclsparse_index_base_zero);
        test_invalid_index<double>(aoclsparse_index_base_zero);
        // base-1
        test_invalid_index<float>(aoclsparse_index_base_one);
        test_invalid_index<double>(aoclsparse_index_base_one);
    }
}
