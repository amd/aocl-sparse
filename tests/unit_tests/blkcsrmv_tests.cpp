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
#include "common_data_utils.h"
#include "aoclsparse.hpp"

#if USE_AVX512
namespace
{

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_blkcsrmv_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 8, nnz = 3;
        T                    blk_csr_val[] = {3, 2, 1};
        aoclsparse_int       blk_col_ind[] = {1};
        aoclsparse_int       blk_row_ptr[] = {0, 1, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[] = {1.0, -2.0, 3.0};
        T                    y[] = {0.1, 0.2};
        aoclsparse_mat_descr descr;
        aoclsparse_int       nRowsblk[3] = {1, 2, 4};
        uint8_t              masks[1];
        // aoclsparse_create_mat_descr set aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        aoclsparse_create_mat_descr(&descr);

        for(int i = 0; i < 3; i++)
        {
            // In turns pass nullptr in every single pointer argument
            // and expect pointer error
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             nnz,
                                             nullptr,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             nnz,
                                             masks,
                                             nullptr,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             nullptr,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             nullptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             nullptr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             nullptr,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             nullptr,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            //FIXME crashes: EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans, nullptr, m, n, nnz, masks, blk_csr_val, blk_col_ind, blk_row_ptr, descr, x, &beta, y, nRowsblk[i]),
            //      aoclsparse_status_invalid_pointer);
            //FIXME crashes: EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans, &alpha, m, n, nnz, masks, blk_csr_val, blk_col_ind, blk_row_ptr, descr, x, nullptr, y, nRowsblk[i]),
            //      aoclsparse_status_invalid_pointer);
        }

        aoclsparse_destroy_mat_descr(descr);
    }

    // tests with wrong scalar data n, m, nnz, masks and incorrect wrong block size
    template <typename T>
    void test_blkcsrmv_wrong_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 8, nnz = 3, wrong = -1;
        T                    blk_csr_val[] = {3, 2, 1};
        aoclsparse_int       blk_col_ind[] = {1};
        aoclsparse_int       blk_row_ptr[] = {0, 1, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[]         = {1.0, -2.0, 3.0};
        T                    y[]         = {0.1, 0.2};
        aoclsparse_int       nRowsblk[3] = {1, 2, 4};
        uint8_t              masks[]     = {100, 25};
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);

        for(int i = 0; i < 3; i++)
        {
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             5,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_size);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             wrong,
                                             n,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_size);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             wrong,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_size);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             wrong,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_invalid_size);
        }

        //test with invalid block size 
	EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                 &alpha,
                                 m,
                                 n,
                                 nnz,
                                 masks,
                                 blk_csr_val,
                                 blk_col_ind,
                                 blk_row_ptr,
                                 descr,
                                 x,
                                 &beta,
                                 y,
                                 wrong),
        aoclsparse_status_invalid_size);
	EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                 &alpha,
                                 m,
                                 n,
                                 nnz,
                                 masks,
                                 blk_csr_val,
                                 blk_col_ind,
                                 blk_row_ptr,
                                 descr,
                                 x,
                                 &beta,
                                 y,
                                 5),
        aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
    }

    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_blkcsrmv_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 2, n = 8, nnz = 3;
        T                    blk_csr_val[] = {3, 2, 1};
        aoclsparse_int       blk_col_ind[] = {1};
        aoclsparse_int       blk_row_ptr[] = {0, 1, 1};
        T                    alpha = 2.3, beta = 11.2;
        T                    x[]         = {1.0, -2.0, 3.0};
        T                    y[]         = {0.1, 0.2};
        aoclsparse_int       nRowsblk[3] = {1, 2, 4};
        uint8_t              masks[]     = {100, 25};
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);

        // Passing zero size matrix should be OK
        for(int i = 0; i < 3; i++)
        {
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             0,
                                             n,
                                             nnz,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans,
                                             &alpha,
                                             m,
                                             n,
                                             0,
                                             masks,
                                             blk_csr_val,
                                             blk_col_ind,
                                             blk_row_ptr,
                                             descr,
                                             x,
                                             &beta,
                                             y,
                                             nRowsblk[i]),
                      aoclsparse_status_success);
        }

        aoclsparse_destroy_mat_descr(descr);
    }

    
    //Test cases for analysis and conversion routines
    template <typename T>
    void test_blkcsrmv_conversion()
    {
        aoclsparse_int       m = 2, n = 8, nnz = 3;
        T                    csr_val[]     = {3, 2, 1};
        aoclsparse_int       csr_col_ind[] = {1};
        aoclsparse_int       csr_row_ptr[] = {0, 1, 1};
        T                    blk_csr_val[nnz];
        aoclsparse_int       blk_col_ind[nnz];
        aoclsparse_int       blk_row_ptr[nnz];
        aoclsparse_int       nRowsblk[3] = {1, 2, 4};
        aoclsparse_int       total_blks;
        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);

        //Test cases to identify failures in findng optimal block size -- used from optmv
        EXPECT_EQ(aoclsparse_opt_blksize(-1, nnz, csr_row_ptr, csr_col_ind, &total_blks),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_opt_blksize(m, -1, csr_row_ptr, csr_col_ind, &total_blks),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_opt_blksize(m, nnz, nullptr, csr_col_ind, &total_blks),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_opt_blksize(m, nnz, csr_row_ptr, nullptr, &total_blks),
                  aoclsparse_status_invalid_pointer);

        //Test cases in conversion routines
        for(int i = 0; i < 3; i++)
        {
            uint8_t masks[nnz * nRowsblk[i]];

            EXPECT_EQ(aoclsparse_csr2blkcsr(m, n,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_success);

            //Check for nullptr cases in conversion routine
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            nullptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr,
                                            nullptr,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            nullptr,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            nullptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            nullptr,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            nullptr,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            nullptr,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_pointer);

            //Check for wrong sizes in conversion routine
            EXPECT_EQ(aoclsparse_csr2blkcsr(-1,
                                            n,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_size);
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            -1,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_size);
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            -1,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i]),
                      aoclsparse_status_invalid_size);
        }

        aoclsparse_destroy_mat_descr(descr);
    }

    //TODO add:
    // * not supported/implemented
    // * invalid array data (but we don't test these right now, e.g., col_ind out of bounds)
    // * nnz not matching row_ptr
    //

    TEST(blkcsrmv, NullArgDouble)
    {
        test_blkcsrmv_nullptr<double>();
    }

    TEST(blkcsrmv, WrongSizeDouble)
    {
        test_blkcsrmv_wrong_size<double>();
    }

    TEST(blkcsrmv, DoNothingDouble)
    {
        test_blkcsrmv_do_nothing<double>();
    }

    TEST(blkcsrmv, Conversion)
    {
        test_blkcsrmv_conversion<double>();
    }

} // namespace
#endif
