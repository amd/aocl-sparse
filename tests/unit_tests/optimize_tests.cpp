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
#include "aoclsparse.h"
#include "aoclsparse_mat_structures.h"
#include "gtest/gtest.h"

namespace
{
    TEST(OptimizeTest, NullPtr)
    {
        aoclsparse_matrix A = NULL;

        // Matrix is NULL
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_pointer);

        float          val[]     = {2.0f, 8.0f};
        aoclsparse_int col_idx[] = {0, 1};
        aoclsparse_int row_ptr[] = {0, 1, 2};
        ASSERT_EQ(
            aoclsparse_create_scsr(&A, aoclsparse_index_base_zero, 2, 2, 2, row_ptr, col_idx, val),
            aoclsparse_status_success);

        // NULL CSR row pointers
        A->csr_mat.csr_row_ptr = nullptr;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_pointer);

        // NULL CSR col pointers
        A->csr_mat.csr_row_ptr = row_ptr;
        A->csr_mat.csr_col_ptr = nullptr;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_pointer);

        // NULL CSR val pointers
        A->csr_mat.csr_col_ptr = col_idx;
        A->csr_mat.csr_val     = nullptr;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_pointer);

        aoclsparse_destroy(&A);
    }

    TEST(OptimizeTest, InvalidValues)
    {
        aoclsparse_matrix A = NULL;

        float          val[]     = {2.0f, 8.0f, 10.0f};
        aoclsparse_int col_idx[] = {0, 0, 1};
        aoclsparse_int row_ptr[] = {0, 1, 3};
        ASSERT_EQ(
            aoclsparse_create_scsr(&A, aoclsparse_index_base_zero, 2, 2, 3, row_ptr, col_idx, val),
            aoclsparse_status_success);

        // Invalid row (M)
        A->m = -2;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_size);

        // Invalid col (N)
        A->m = 2;
        A->n = -2;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_size);

        // Invalid nnz
        A->n   = 2;
        A->nnz = -2;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_size);

        // Invalid row_ptr values
        A->nnz = 3;
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        // a) row_ptr[0] is invalid
        aoclsparse_int row_ptr0[] = {1, 1, 3};
        A->csr_mat.csr_row_ptr    = row_ptr0;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_value);

        // b) row_ptr[nnz] is invalid. nnz=3
        aoclsparse_int row_ptr1[] = {0, 1, 2};
        A->csr_mat.csr_row_ptr    = row_ptr1;
        ASSERT_EQ(aoclsparse_set_sv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_value);

        // c) last 2 row_ptr values are not okay
        aoclsparse_int row_ptr2[] = {0, 4, 3};
        A->csr_mat.csr_row_ptr    = row_ptr2;
        ASSERT_EQ(aoclsparse_set_sv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_value);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(&A);
    }

} // namespace
