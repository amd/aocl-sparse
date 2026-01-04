/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <complex>
#include <vector>
#define TRANSFORM_BASE(base, ptr, idx)                                                        \
    std::transform(                                                                           \
        ptr.begin(), ptr.end(), ptr.begin(), [base](aoclsparse_int &d) { return d + base; }); \
    std::transform(                                                                           \
        idx.begin(), idx.end(), idx.begin(), [base](aoclsparse_int &d) { return d + base; });

namespace
{

    template <typename T>
    aoclsparse_status init(aoclsparse_int              &m,
                           aoclsparse_int              &n,
                           aoclsparse_int              &block_dim,
                           std::vector<aoclsparse_int> &row_ptr,
                           std::vector<aoclsparse_int> &col_idx,
                           std::vector<T>              &val,
                           aoclsparse_index_base        base,
                           aoclsparse_int               id = 0)
    {
        // m = block rows, n = block columns, block_dim = size of the block
        switch(id)
        {
        // Input to create a BSR matrix with block size 2x2, having 2 rows and 6 columns
        case 0:
            m = 1, n = 3, block_dim = 2;
            row_ptr.assign({0, 3});
            col_idx.assign({0, 1, 2});
            // We are not validating the values populated. We are only allocating the vector.
            val.resize(row_ptr[m] * block_dim * block_dim);

            //rebuild indices using one-based indexing
            TRANSFORM_BASE(base, row_ptr, col_idx);

            break;
        // Input to create a BSR matrix with block size 3x3, having 6 rows and 6 columns
        case 1:
            m = 2, n = 2, block_dim = 3;
            row_ptr.assign({0, 2, 4});
            col_idx.assign({0, 1, 0, 1});
            // We are not validating the values populated, so we are only allocating the vector
            val.resize(row_ptr[m] * block_dim * block_dim);

            //rebuild indices using one-based indexing
            TRANSFORM_BASE(base, row_ptr, col_idx);
            break;
        default:
            break;
        }
        return aoclsparse_status_success;
    }

    template <typename T>
    void test_success(std::string testcase)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              block_dim;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_idx;
        std::vector<T>              val;
        aoclsparse_matrix           A;

        for(aoclsparse_int id = 0; id < 2; id++)
        {
            for(aoclsparse_index_base base :
                {aoclsparse_index_base_zero, aoclsparse_index_base_one})
            {
                for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                {
                    SCOPED_TRACE(testcase + ":: Base: " + std::to_string(base)
                                 + ", Order: " + std::to_string(order));
                    EXPECT_EQ(init(m, n, block_dim, row_ptr, col_idx, val, base, id),
                              aoclsparse_status_success);
                    nnz = (row_ptr[m] - base);

                    EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                                       base,
                                                       order,
                                                       m,
                                                       n,
                                                       block_dim,
                                                       row_ptr.data(),
                                                       col_idx.data(),
                                                       val.data(),
                                                       false),
                              aoclsparse_status_success);
                    aoclsparse::bsr *bsr_mat = dynamic_cast<aoclsparse::bsr *>(A->mats[0]);
                    EXPECT_EQ_VEC(
                        m + 1, (aoclsparse_int *)bsr_mat->ptr, (aoclsparse_int *)row_ptr.data());
                    EXPECT_EQ_VEC(
                        nnz, (aoclsparse_int *)bsr_mat->ind, (aoclsparse_int *)col_idx.data());
                    EXPECT_EQ(m, bsr_mat->bm);
                    EXPECT_EQ(n, bsr_mat->bn);
                    EXPECT_EQ(nnz, bsr_mat->bnnz);
                    EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
                }
            }
        }
    }

    template <typename T>
    void test_nullptr(std::string testcase)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              block_dim;
        aoclsparse_order            order = aoclsparse_order_row;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_idx;
        std::vector<T>              val;
        aoclsparse_matrix           A;

        EXPECT_EQ(init(m, n, block_dim, row_ptr, col_idx, val, base), aoclsparse_status_success);
        {
            SCOPED_TRACE(testcase + ":: nullptr A");
            EXPECT_EQ(aoclsparse_create_bsr<T>(nullptr,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               true),
                      aoclsparse_status_invalid_pointer);
        }
        {
            SCOPED_TRACE(testcase + ":: nullptr row_ptr");
            EXPECT_EQ(
                aoclsparse_create_bsr<T>(
                    &A, base, order, m, n, block_dim, nullptr, col_idx.data(), val.data(), false),
                aoclsparse_status_invalid_pointer);
        }
        {
            SCOPED_TRACE(testcase + ":: nullptr col_idx");
            EXPECT_EQ(
                aoclsparse_create_bsr<T>(
                    &A, base, order, m, n, block_dim, row_ptr.data(), nullptr, val.data(), false),
                aoclsparse_status_invalid_pointer);
        }
        {
            SCOPED_TRACE(testcase + ":: nullptr val");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               nullptr,
                                               false),
                      aoclsparse_status_invalid_pointer);
        }
    }

    template <typename T>
    void test_invalid_input(std::string testcase)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        aoclsparse_int              block_dim;
        aoclsparse_order            order = aoclsparse_order_row;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_idx;
        std::vector<T>              val;
        aoclsparse_matrix           A;

        EXPECT_EQ(init(m, n, block_dim, row_ptr, col_idx, val, base), aoclsparse_status_success);
        nnz = row_ptr[m];
        {
            SCOPED_TRACE(testcase + ":: invalid size n");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               -1,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_invalid_size);
        }
        EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                           base,
                                           order,
                                           -1,
                                           n,
                                           block_dim,
                                           row_ptr.data(),
                                           col_idx.data(),
                                           val.data(),
                                           false),
                  aoclsparse_status_invalid_size);
        // Testing negative bnnz
        row_ptr[m] = -1;
        {
            SCOPED_TRACE(testcase + ":: negative bnnz");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_invalid_size);
        }
        // Restore the row_ptr with original value
        row_ptr[m] = nnz;
        // incorrect block size = 0
        {
            SCOPED_TRACE(testcase + ":: invalid block size");
            EXPECT_EQ(
                aoclsparse_create_bsr<T>(
                    &A, base, order, m, n, 0, row_ptr.data(), col_idx.data(), val.data(), false),
                aoclsparse_status_invalid_value);
        }
        // invalid column index for zero-based indexing
        col_idx[2] = 3; // should be between 0 and 2
        {
            SCOPED_TRACE(testcase + ":: invalid column index for zero-based indexing");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_invalid_index_value);
        }
        // Test for one-based indexing with row-major input
        base = aoclsparse_index_base_one;
        {
            SCOPED_TRACE(testcase
                         + ":: invalid block size for one-based indexing with row-major input");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_invalid_value);
        }
        // rebuild indices for 1-based indexing and then test for invalid column index
        TRANSFORM_BASE(base, row_ptr, col_idx);

        // Test for column major ordering with zero-based index input
        col_idx[2] = 0; // should be between 1 and 3
        {
            SCOPED_TRACE(
                testcase
                + ":: invalid column index for column major ordering with zero-based indexing");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_invalid_index_value);
        }
        row_ptr[0] = 0; // should be greater than 1
        {
            SCOPED_TRACE(
                testcase
                + ":: invalid row pointer for column major ordering with zero-based indexing");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_invalid_value);
        }
        // out-of bound column index for one-based indexing
        row_ptr.assign({1, 4});
        col_idx.assign({1, 5, 8});
        {
            SCOPED_TRACE(testcase + ":: out-of bound column index for one-based indexing");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_invalid_index_value);
        }
    }

    template <typename T>
    void test_zero_dimension(std::string testcase)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              block_dim;
        aoclsparse_order            order = aoclsparse_order_row;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_idx;
        std::vector<T>              val;
        aoclsparse_matrix           A;

        // m = 1, n = 3, nnz = 12
        EXPECT_EQ(init(m, n, block_dim, row_ptr, col_idx, val, base), aoclsparse_status_success);

        // 1) 1*3 , nnz=0
        row_ptr.assign({0, 0});
        {
            SCOPED_TRACE(testcase + ":: zero dimension nnz=0");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        // 2) 0*7 , nnz=0
        m = 0;
        n = 7;
        row_ptr.assign({0});
        col_idx.assign({0});
        {
            SCOPED_TRACE(testcase + ":: zero dimension nnz=0, m=0");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        // 3) 2*0 , nnz=0
        m = 2;
        n = 0;
        row_ptr.assign({0, 0, 0});
        {
            SCOPED_TRACE(testcase + ":: zero dimension nnz=0, n=0");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }

        // 4) 0*0 , nnz=0
        m = 0;
        n = 0;
        row_ptr.assign({0});
        col_idx.assign({0});
        {
            SCOPED_TRACE(testcase + ":: zero dimension nnz=0, m=0, n=0");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        // 5) 2*0 , nnz=3
        // This is a negative test case which is expected to fail
        m = 2;
        n = 0;
        row_ptr.assign({0, 2, 3});
        col_idx.assign({0, 0, 0});
        {
            SCOPED_TRACE(testcase + ":: negative test with zero dimension n=0");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_idx.data(),
                                               val.data(),
                                               false),
                      aoclsparse_status_invalid_index_value);
        }
    }

    template <typename T>
    aoclsparse_status test_sorted_fulldiag(std::string testcase, aoclsparse_index_base base)
    {
        SCOPED_TRACE(testcase);
        // test dataset
        aoclsparse_int m, n;
        m                                     = 4;
        n                                     = 4;
        aoclsparse_int              block_dim = 2;
        aoclsparse_order            order;
        std::vector<aoclsparse_int> row_ptr, col_fully_sorted, col_partially_sorted, col_unsorted;
        std::vector<T>              val_fully_sorted, val_partially_sorted, val_un_sorted;

        if(base == aoclsparse_index_base_zero)
        {
            order = aoclsparse_order_row;
            row_ptr.assign({0, 3, 4, 6, 9});
            col_fully_sorted.assign({0, 2, 3, 1, 0, 2, 0, 1, 3});
            col_partially_sorted.assign({0, 3, 2, 1, 0, 2, 1, 0, 3});
            col_unsorted.assign({2, 0, 3, 1, 0, 2, 3, 1, 0});
        }
        else
        {
            order = aoclsparse_order_column;
            row_ptr.assign({1, 4, 5, 7, 10});
            col_fully_sorted.assign({1, 3, 4, 2, 1, 3, 1, 2, 4});
            col_partially_sorted.assign({1, 4, 3, 2, 1, 3, 2, 1, 4});
            col_unsorted.assign({3, 1, 4, 2, 1, 3, 4, 2, 1});
        }

        // We are not validating the values populated. We are only allocating the vector.
        try
        {
            val_fully_sorted.resize(row_ptr[m] * block_dim * block_dim);
            val_partially_sorted.resize(row_ptr[m] * block_dim * block_dim);
            val_un_sorted.resize(row_ptr[m] * block_dim * block_dim);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
        aoclsparse_matrix A;
        {
            // fully sorted + full diag
            SCOPED_TRACE(testcase + ":: fully sorted + full diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_fully_sorted.data(),
                                               val_fully_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }

        {
            // fully sorted + missing diag
            col_fully_sorted[5] = 3 + base;
            SCOPED_TRACE(testcase + ":: fully sorted + missing diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_fully_sorted.data(),
                                               val_fully_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // partially sorted + full diag
            SCOPED_TRACE(testcase + ":: partially sorted + full diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_partially_sorted.data(),
                                               val_partially_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_partially_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // unsorted + full diag
            SCOPED_TRACE(testcase + ":: unsorted + full diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_unsorted.data(),
                                               val_un_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_unsorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // unsorted + missing diag
            col_unsorted[5] = 0 + base;
            SCOPED_TRACE(testcase + ":: unsorted + missing diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_unsorted.data(),
                                               val_un_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_unsorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // fully sorted + full diag + check for zero-based indexing
            col_fully_sorted[5] = 3 + base;
            SCOPED_TRACE(testcase + ":: fully sorted + full diag with zero-based indexing");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_fully_sorted.data(),
                                               val_fully_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // partially sorted + missing diag
            col_partially_sorted[5] = 3 + base;
            SCOPED_TRACE(testcase + ":: partially sorted + missing diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_partially_sorted.data(),
                                               val_partially_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_partially_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // unsorted + full diag
            col_unsorted[5] = 3 + base;
            SCOPED_TRACE(testcase + ":: unsorted + full diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_unsorted.data(),
                                               val_un_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_unsorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // unsorted + missing diag
            col_unsorted[5] = 0 + base;
            SCOPED_TRACE(testcase + ":: unsorted + missing diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_unsorted.data(),
                                               val_un_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_unsorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // fully sorted + full diag + check for one-based indexing
            col_fully_sorted[5] = 3 + base;
            SCOPED_TRACE(testcase + ":: fully sorted + full diag with one-based indexing");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_fully_sorted.data(),
                                               val_fully_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_fully_sorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // unsorted + full diag
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_unsorted.data(),
                                               val_un_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_unsorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        {
            // unsorted + missing diag
            col_unsorted[5] = 0 + base;
            SCOPED_TRACE(testcase + ":: unsorted + missing diag");
            EXPECT_EQ(aoclsparse_create_bsr<T>(&A,
                                               base,
                                               order,
                                               m,
                                               n,
                                               block_dim,
                                               row_ptr.data(),
                                               col_unsorted.data(),
                                               val_un_sorted.data(),
                                               false),
                      aoclsparse_status_success);
            EXPECT_EQ(A->sort, aoclsparse_unsorted);
            EXPECT_EQ(A->fulldiag, false);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
        return aoclsparse_status_success;
    }

    TEST(createbsr, NullArgAll)
    {
        test_nullptr<float>("CreateBSR-TestNullArg-float");
        test_nullptr<double>("CreateBSR-TestNullArg-double");
        test_nullptr<aoclsparse_float_complex>("CreateBSR-TestNullArg-floatComplex");
        test_nullptr<aoclsparse_double_complex>("CreateBSR-TestNullArg-doubleComplex");
    }
    TEST(createbsr, InvalidInputAll)
    {
        test_invalid_input<float>("CreateBSR-InvalidInput-float");
        test_invalid_input<double>("CreateBSR-InvalidInput-double");
        test_invalid_input<aoclsparse_float_complex>("CreateBSR-InvalidInput-floatComplex");
        test_invalid_input<aoclsparse_double_complex>("CreateBSR-InvalidInput-doubleComplex");
    }
    TEST(createbsr, SuccessAll)
    {
        test_success<float>("CreateBSR-Success-float");
        test_success<double>("CreateBSR-Success-double");
        test_success<aoclsparse_float_complex>("CreateBSR-Success-floatComplex");
        test_success<aoclsparse_double_complex>("CreateBSR-Success-doubleComplex");
    }
    TEST(createbsr, ZeroDimensionMatrix)
    {
        test_zero_dimension<float>("CreateBSR-ZeroDimension-float");
        test_zero_dimension<double>("CreateBSR-ZeroDimension-double");
        test_zero_dimension<aoclsparse_float_complex>("CreateBSR-ZeroDimension-floatComplex");
        test_zero_dimension<aoclsparse_double_complex>("CreateBSR-ZeroDimension-doubleComplex");
    }
    TEST(createbsr, CheckSortedDiag)
    {
        test_sorted_fulldiag<float>("CreateBSR-CheckSortedDiag-float", aoclsparse_index_base_zero);
        test_sorted_fulldiag<double>("CreateBSR-CheckSortedDiag-double",
                                     aoclsparse_index_base_zero);
        test_sorted_fulldiag<aoclsparse_float_complex>("CreateBSR-CheckSortedDiag-floatComplex",
                                                       aoclsparse_index_base_one);
        test_sorted_fulldiag<aoclsparse_double_complex>("CreateBSR-CheckSortedDiag-doubleComplex",
                                                        aoclsparse_index_base_one);
    }
} // namespace
