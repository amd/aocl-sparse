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

    template <typename T>
    void test_ordered_mat(aoclsparse_matrix mat, aoclsparse_int *idx_exp, T *val_exp)
    {
        aoclsparse_int *idx = NULL;
        T              *val = NULL;

        if(mat->input_format == aoclsparse_csr_mat)
        {
            idx = mat->csr_mat.csr_col_ptr;
            val = static_cast<T *>(mat->csr_mat.csr_val);
        }
        else
        {
            idx = mat->csc_mat.row_idx;
            val = static_cast<T *>(mat->csc_mat.val);
        }
        EXPECT_EQ_VEC(mat->nnz, idx, idx_exp);
        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            EXPECT_COMPLEX_FLOAT_EQ_VEC(
                mat->nnz, ((std::complex<float> *)val), ((std::complex<float> *)val_exp));
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            EXPECT_COMPLEX_DOUBLE_EQ_VEC(
                mat->nnz, ((std::complex<double> *)val), ((std::complex<double> *)val_exp));
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            EXPECT_DOUBLE_EQ_VEC(mat->nnz, val, val_exp);
        }
        else
        {
            EXPECT_FLOAT_EQ_VEC(mat->nnz, val, val_exp);
        }
    }

    TEST(order, NullMatrixFailure)
    {
        EXPECT_EQ(aoclsparse_order_mat(nullptr), aoclsparse_status_invalid_pointer);
    }

    TEST(order, CooMatrixFailure)
    {
        aoclsparse_matrix mat       = NULL;
        aoclsparse_int    row_idx[] = {0, 1, 2, 3};
        aoclsparse_int    col_idx[] = {0, 3, 1, 3};
        double            vald[]    = {1, 6, 8, 4};

        ASSERT_EQ(
            aoclsparse_createcoo(mat, aoclsparse_index_base_zero, 4, 5, 4, row_idx, col_idx, vald),
            aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_not_implemented);
        aoclsparse_destroy(mat);
    }

    TEST(order, NullArrayPointersFailure)
    {
        aoclsparse_matrix mat       = NULL;
        aoclsparse_int    row_ptr[] = {0, 1, 2, 3, 4, 4};
        aoclsparse_int    col_idx[] = {0, 3, 1, 3};
        double            vald[]    = {1, 6, 8, 4};

        // CSR matrix pointers are NULL
        ASSERT_EQ(
            aoclsparse_create_csr(mat, aoclsparse_index_base_zero, 5, 4, 4, row_ptr, col_idx, vald),
            aoclsparse_status_success);

        // a) csr_row_ptr is NULL
        mat->csr_mat.csr_row_ptr = NULL;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_pointer);

        // b) csr_col_idx is NULL
        mat->csr_mat.csr_row_ptr = row_ptr;
        mat->csr_mat.csr_col_ptr = NULL;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_pointer);

        // c) csr_val is NULL
        mat->csr_mat.csr_col_ptr = col_idx;
        mat->csr_mat.csr_val     = NULL;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(mat);

        // CSC matrix pointers are NULL
        aoclsparse_int col_ptr[] = {0, 1, 2, 3, 4, 4};
        aoclsparse_int row_idx[] = {0, 3, 1, 3};
        ASSERT_EQ(
            aoclsparse_createcsc(mat, aoclsparse_index_base_zero, 4, 5, 4, col_ptr, row_idx, vald),
            aoclsparse_status_success);

        // a) col_ptr is NULL
        mat->csc_mat.col_ptr = NULL;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_pointer);

        // b) row_idx is NULL
        mat->csc_mat.col_ptr = col_ptr;
        mat->csc_mat.row_idx = NULL;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_pointer);

        // c) val is NULL
        mat->csc_mat.row_idx = row_idx;
        mat->csc_mat.val     = NULL;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(mat);
    }

    TEST(order, DoNothing)
    {
        aoclsparse_matrix         mat       = NULL;
        aoclsparse_int            row_ptr[] = {1, 1, 1, 1, 1, 1};
        aoclsparse_int            col_idx[1];
        aoclsparse_double_complex valc[1];
        double                    vald[1];

        // complex 5x40, nnz=0
        ASSERT_EQ(
            aoclsparse_create_csr(mat, aoclsparse_index_base_one, 5, 40, 0, row_ptr, col_idx, valc),
            aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);

        aoclsparse_destroy(mat);

        // double 3x0
        ASSERT_EQ(
            aoclsparse_create_csr(mat, aoclsparse_index_base_one, 3, 0, 0, row_ptr, col_idx, vald),
            aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);

        aoclsparse_destroy(mat);
    }

    TEST(order, InvalidDimensions)
    {
        aoclsparse_matrix mat       = NULL;
        aoclsparse_int    row_ptr[] = {0, 1, 2, 3, 4, 4};
        aoclsparse_int    col_idx[] = {0, 3, 1, 3};
        float             valf[]    = {1, 6, 8, 4};

        ASSERT_EQ(
            aoclsparse_create_csr(mat, aoclsparse_index_base_zero, 5, 4, 4, row_ptr, col_idx, valf),
            aoclsparse_status_success);

        // It is impossible to create normally any of these matricies
        // so let's build them artificially
        // a) m<0
        mat->m = -1;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_value);

        // b) n<0
        mat->m = 5;
        mat->n = -2;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_value);

        // c) nnz<0
        mat->n   = 4;
        mat->nnz = -1;
        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_invalid_value);
        aoclsparse_destroy(mat);
    }

    TEST(order, InputCsrIsOptimized)
    {
        aoclsparse_matrix mat       = NULL;
        aoclsparse_int    row_ptr[] = {0, 1, 3, 7, 10, 10};
        // col indices are not ordered but lower trianglular elements are
        // followed by diagonal followed by upper triangular elements
        aoclsparse_int       col_idx[]     = {0, 0, 1, 1, 0, 2, 3, 2, 1, 3};
        float                valf[]        = {1, 1, 2, 2, 1, 3, 4, 2, 1, 3};
        aoclsparse_int       col_idx_exp[] = {0, 0, 1, 0, 1, 2, 3, 1, 2, 3};
        float                valf_exp[]    = {1, 1, 2, 1, 2, 3, 4, 1, 2, 3};
        aoclsparse_mat_descr descr;

        ASSERT_EQ(aoclsparse_create_csr(
                      mat, aoclsparse_index_base_zero, 5, 4, 10, row_ptr, col_idx, valf),
                  aoclsparse_status_success);

        // existing user csr pointers are already clean. opt_csr_is_user = true.
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(mat, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(mat), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, col_idx_exp, valf_exp);
        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(mat);
    }

    TEST(order, AllRowOneNnzSuc)
    {
        aoclsparse_matrix mat        = NULL;
        aoclsparse_int    row_ptr[]  = {0, 1, 2, 3, 4, 5};
        aoclsparse_int    col_idx[]  = {0, 1, 2, 3, 4};
        double            vald1[]    = {10, 8, 6, 4, 2};
        aoclsparse_int    idx_exp[]  = {0, 1, 2, 3, 4};
        double            vald_exp[] = {10, 8, 6, 4, 2};

        // CSR matrix
        ASSERT_EQ(aoclsparse_create_csr(
                      mat, aoclsparse_index_base_zero, 5, 5, 5, row_ptr, col_idx, vald1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, idx_exp, vald_exp);
        aoclsparse_destroy(mat);

        // CSC matrix
        aoclsparse_int col_ptr[] = {0, 1, 2, 3, 4, 5};
        aoclsparse_int row_idx[] = {0, 1, 2, 3, 4};
        double         vald2[]   = {10, 8, 6, 4, 2};
        ASSERT_EQ(
            aoclsparse_createcsc(mat, aoclsparse_index_base_zero, 5, 5, 5, col_ptr, row_idx, vald2),
            aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, idx_exp, vald_exp);
        aoclsparse_destroy(mat);
    }

    TEST(order, AllRowTwoNnzSuc)
    {
        // input is not ordered
        aoclsparse_matrix mat        = NULL;
        float             valf_exp[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        // a) 0-based indexing for CSR
        aoclsparse_int row_ptr1[]     = {0, 2, 4, 6, 8, 10};
        aoclsparse_int col_idx1[]     = {1, 0, 2, 1, 3, 2, 4, 3, 4, 0};
        float          valf1[]        = {2, 1, 4, 3, 6, 5, 8, 7, 10, 9};
        aoclsparse_int col_idx_exp1[] = {0, 1, 1, 2, 2, 3, 3, 4, 0, 4};

        ASSERT_EQ(aoclsparse_create_csr(
                      mat, aoclsparse_index_base_zero, 5, 5, 10, row_ptr1, col_idx1, valf1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, col_idx_exp1, valf_exp);
        aoclsparse_destroy(mat);

        // b) 1-based indexing for CSC
        aoclsparse_int col_ptr[]     = {1, 3, 5, 7, 9, 11};
        aoclsparse_int row_idx[]     = {2, 1, 3, 2, 4, 3, 5, 4, 5, 1};
        float          valf2[]       = {2, 1, 4, 3, 6, 5, 8, 7, 10, 9};
        aoclsparse_int row_idx_exp[] = {1, 2, 2, 3, 3, 4, 4, 5, 1, 5};

        ASSERT_EQ(aoclsparse_create_csr(
                      mat, aoclsparse_index_base_one, 5, 5, 10, col_ptr, row_idx, valf2),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, row_idx_exp, valf_exp);
        aoclsparse_destroy(mat);
    }

    TEST(order, GeneralCsrSuccess)
    {
        aoclsparse_matrix mat           = NULL;
        aoclsparse_int    col_idx_exp[] = {0, 1, 1, 2, 3, 4, 0, 1, 3, 4};

        // a) Variable number of nnz in each CSC col. user data is not sorted
        aoclsparse_int            col_ptr[] = {0, 2, 3, 6, 10, 10};
        aoclsparse_int            row_idx[] = {1, 0, 1, 2, 4, 3, 3, 1, 0, 4};
        aoclsparse_double_complex valcd[]
            = {{10, 10}, {9, 9}, {8, 8}, {7, 7}, {6, 6}, {5, 5}, {4, 4}, {3, 3}, {2, 2}, {1, 1}};
        aoclsparse_double_complex valcd_exp[]
            = {{9, 9}, {10, 10}, {8, 8}, {7, 7}, {5, 5}, {6, 6}, {2, 2}, {3, 3}, {4, 4}, {1, 1}};

        ASSERT_EQ(aoclsparse_createcsc(
                      mat, aoclsparse_index_base_zero, 5, 5, 10, col_ptr, row_idx, valcd),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, col_idx_exp, valcd_exp);
        aoclsparse_destroy(mat);

        // b) variable nnz in each CSR row. user data is not sorted and we have optimized data
        aoclsparse_int       row_ptr[]  = {0, 2, 3, 6, 10, 10};
        aoclsparse_int       col_idx[]  = {1, 0, 1, 2, 4, 3, 3, 1, 0, 4};
        float                valf[]     = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        float                valf_exp[] = {9, 10, 8, 7, 5, 6, 2, 3, 4, 1};
        aoclsparse_mat_descr descr;
        ASSERT_EQ(aoclsparse_create_csr(
                      mat, aoclsparse_index_base_zero, 5, 5, 10, row_ptr, col_idx, valf),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(mat, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(mat), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, col_idx_exp, valf_exp);
        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(mat);
    }

    TEST(order, SortedCsrSuccess)
    {
        // Input CSR is already sorted
        aoclsparse_matrix        mat       = NULL;
        aoclsparse_int           row_ptr[] = {0, 2, 4, 6, 8, 10};
        aoclsparse_int           col_idx[] = {0, 1, 1, 2, 2, 3, 3, 4, 0, 4};
        aoclsparse_float_complex valcf1[]
            = {{10, 10}, {9, 9}, {8, 8}, {7, 7}, {6, 6}, {5, 5}, {4, 4}, {3, 3}, {2, 2}, {1, 1}};
        aoclsparse_int           idx_exp[] = {0, 1, 1, 2, 2, 3, 3, 4, 0, 4};
        aoclsparse_float_complex valcf_exp[]
            = {{10, 10}, {9, 9}, {8, 8}, {7, 7}, {6, 6}, {5, 5}, {4, 4}, {3, 3}, {2, 2}, {1, 1}};

        ASSERT_EQ(aoclsparse_create_csr(
                      mat, aoclsparse_index_base_zero, 5, 5, 10, row_ptr, col_idx, valcf1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, idx_exp, valcf_exp);
        aoclsparse_destroy(mat);

        // Input CSC is already sorted
        aoclsparse_int           col_ptr[] = {0, 2, 4, 6, 8, 10};
        aoclsparse_int           row_idx[] = {0, 1, 1, 2, 2, 3, 3, 4, 0, 4};
        aoclsparse_float_complex valcf2[]
            = {{10, 10}, {9, 9}, {8, 8}, {7, 7}, {6, 6}, {5, 5}, {4, 4}, {3, 3}, {2, 2}, {1, 1}};

        ASSERT_EQ(aoclsparse_createcsc(
                      mat, aoclsparse_index_base_zero, 5, 5, 10, col_ptr, row_idx, valcf2),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_order_mat(mat), aoclsparse_status_success);
        test_ordered_mat(mat, idx_exp, valcf_exp);
        aoclsparse_destroy(mat);
    }

} // namespace
