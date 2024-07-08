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
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"

//aocl utils
#include "Au/Cpuid/X86Cpu.hh"

namespace
{
    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_blkcsrmv_nullptr(aoclsparse_status blkcsrmv_status = aoclsparse_status_success)
    {
        aoclsparse_status    status;
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
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with null mask ptr input";
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status)
                << "Test failed with unexpected return from aoclsparse_blkcsrmv with null "
                   "blk_csr_val ptr input";
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status)
                << "Test failed with unexpected return from aoclsparse_blkcsrmv with null "
                   "blk_col_ind ptr input";
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status)
                << "Test failed with unexpected return from aoclsparse_blkcsrmv with null "
                   "blk_row_ptr ptr input";
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with null descr ptr input";
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with null x ptr input";
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with null y ptr input";
            //FIXME crashes: EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans, nullptr, m, n, nnz, masks, blk_csr_val, blk_col_ind, blk_row_ptr, descr, x, &beta, y, nRowsblk[i]),
            //      aoclsparse_status_invalid_pointer);
            //FIXME crashes: EXPECT_EQ(aoclsparse_blkcsrmv<T>(trans, &alpha, m, n, nnz, masks, blk_csr_val, blk_col_ind, blk_row_ptr, descr, x, nullptr, y, nRowsblk[i]),
            //      aoclsparse_status_invalid_pointer);
        }

        aoclsparse_destroy_mat_descr(descr);
    }

    // tests with wrong scalar data n, m, nnz, masks and incorrect wrong block size
    template <typename T>
    void test_blkcsrmv_wrong_size(aoclsparse_status blkcsrmv_status = aoclsparse_status_success)
    {
        aoclsparse_status    status;
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
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status)
                << "Test failed with unexpected return from aoclsparse_blkcsrmv with wrong n input";

            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with negative m input";

            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with negative n input";

            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with negative nnz input";
        }

        //test with invalid block size
        status = aoclsparse_blkcsrmv<T>(trans,
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
                                        wrong);
        ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                              "aoclsparse_blkcsrmv with negative block size input";

        status = aoclsparse_blkcsrmv<T>(trans,
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
                                        5);
        ASSERT_EQ(status, blkcsrmv_status)
            << "Test failed with unexpected return from aoclsparse_blkcsrmv with unsupported block "
               "size input";

        aoclsparse_destroy_mat_descr(descr);
    }

    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_blkcsrmv_do_nothing(aoclsparse_status blkcsrmv_status = aoclsparse_status_success)
    {
        aoclsparse_status    status;
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
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status)
                << "Test failed with unexpected return from aoclsparse_blkcsrmv with zero size m";
            status = aoclsparse_blkcsrmv<T>(trans,
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
                                            nRowsblk[i]);
            ASSERT_EQ(status, blkcsrmv_status)
                << "Test failed with unexpected return from aoclsparse_blkcsrmv with zero size nnz";
        }

        aoclsparse_destroy_mat_descr(descr);
    }
    // check for invalid base index
    template <typename T>
    void test_blkcsrmv_invalid_base(aoclsparse_status blkcsrmv_status = aoclsparse_status_success)
    {
        aoclsparse_status    status;
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_mat_descr descr;
        int                  invalid_index_base = 2;
        int                  iB;
        aoclsparse_int       nRowsblk[3] = {1, 2, 4};

        aoclsparse_int              m = 6, n = 8, nnz = 14;
        T                           alpha = 1.0, beta = 0.0;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        T                           x[8] = {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00};
        T                           y[6] = {0};

        //Initialize block width
        std::vector<aoclsparse_int> blk_row_ptr;
        std::vector<aoclsparse_int> blk_col_ind;
        std::vector<T>              blk_csr_val;
        std::vector<uint8_t>        masks;

        csr_row_ptr.assign({0, 1, 2, 2, 10, 12, 14});
        csr_col_ind.assign({0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 0, 1});
        csr_val.assign(
            {8.00, 2.00, 3.00, 3.00, 3.00, 6.00, 10.00, 9.00, 6.00, 2.00, 2.00, 3.00, 2.00, 6.00});

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A = nullptr;
        ASSERT_EQ(
            create_aoclsparse_matrix<T>(A, descr, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
            aoclsparse_status_success);

        //check if base index value is invalid
        iB = 2; //nRowsblk = 4
        std::fill(blk_row_ptr.begin(), blk_row_ptr.end(), 0);
        std::fill(blk_col_ind.begin(), blk_col_ind.end(), 0);
        std::fill(blk_csr_val.begin(), blk_csr_val.end(), 0.0);
        std::fill(masks.begin(), masks.end(), 0);

        descr->base = (aoclsparse_index_base)invalid_index_base;
        status      = aoclsparse_blkcsrmv<T>(trans,
                                        &alpha,
                                        m,
                                        n,
                                        nnz,
                                        masks.data(),
                                        blk_csr_val.data(),
                                        blk_col_ind.data(),
                                        blk_row_ptr.data(),
                                        descr,
                                        x,
                                        &beta,
                                        y,
                                        nRowsblk[iB]);
        ASSERT_EQ(status, blkcsrmv_status)
            << "Test failed to validate invalid base-index from aoclsparse_blkcsrmv";

        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_blkcsrmv_baseOneBlkCSRInput(aoclsparse_status blkcsrmv_status
                                          = aoclsparse_status_success)
    {
        aoclsparse_status    status;
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_int       m = 6, n = 8, nnz = 14;
        T blk_csr_val[30] = {8.00, 2.00, 3.00, 3.00, 3.00, 6.00, 10.00, 9.00, 6.00, 2.00,
                             2.00, 3.00, 2.00, 6.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00,
                             0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  0.00, 0.00, 0.00};
        aoclsparse_int       blk_col_ind[14] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        aoclsparse_int       blk_row_ptr[7]  = {1, 2, 2, 3, 3, 4, 4};
        T                    alpha = 1.0, beta = 0.0;
        T                    x[8]      = {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00};
        T                    y[6]      = {0};
        T                    y_gold[6] = {8.00, 2.00, 0.00, 204.00, 23.00, 14.00};
        aoclsparse_mat_descr descr;
        aoclsparse_int       nRowsblk  = 2;
        uint8_t              masks[30] = {1, 1, 0, 255, 24, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0,   0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        status = aoclsparse_blkcsrmv<T>(trans,
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
                                        nRowsblk);
        ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                              "aoclsparse_blkcsrmv with baseOne Block CSR input";
        if(status == aoclsparse_status_success)
        {
            EXPECT_ARR_NEAR(m, y, y_gold, expected_precision<T>(10.0));
        }
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_blkcsrmv_baseOneCSRInput(aoclsparse_status blkcsrmv_status
                                       = aoclsparse_status_success)
    {
        aoclsparse_status    status;
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_mat_descr descr;
        aoclsparse_int       nRowsblk[3] = {1, 2, 4};
        int                  iB;

        aoclsparse_int              m = 6, n = 8, nnz = 14;
        T                           alpha = 1.0, beta = 0.0;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        T                           x[8]      = {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00};
        T                           y[6]      = {0};
        T                           y_gold[6] = {8.00, 2.00, 0.00, 204.00, 23.00, 14.00};

        std::vector<aoclsparse_int> blk_col_ind;
        std::vector<aoclsparse_int> blk_row_ptr;
        std::vector<T>              blk_csr_val;
        std::vector<uint8_t>        masks;
        //Initialize block width
        const aoclsparse_int blk_width = 8;

        csr_row_ptr.assign({1, 2, 3, 3, 11, 13, 15});
        csr_col_ind.assign({1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 4, 5, 1, 2});
        csr_val.assign(
            {8.00, 2.00, 3.00, 3.00, 3.00, 6.00, 10.00, 9.00, 6.00, 2.00, 2.00, 3.00, 2.00, 6.00});

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);

        aoclsparse_matrix A = nullptr;
        ASSERT_EQ(
            create_aoclsparse_matrix<T>(A, descr, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
            aoclsparse_status_success);

        for(iB = 0; iB < 3; iB++)
        {
            blk_row_ptr.resize(m + 1);
            blk_col_ind.resize(nnz);
            blk_csr_val.resize((nnz + (nRowsblk[iB] * blk_width)));
            masks.resize((nnz + (nRowsblk[iB] * blk_width)));

            std::fill(blk_row_ptr.begin(), blk_row_ptr.end(), 0);
            std::fill(blk_col_ind.begin(), blk_col_ind.end(), 0);
            std::fill(blk_csr_val.begin(), blk_csr_val.end(), 0.0);
            std::fill(masks.begin(), masks.end(), 0);

            /*
                since BlkCSR spmv is decided in Sparse-Analysis-Framework's (SAF) optimize module, direct
                call to csr2blkcsr conversion routine below is made to write unit tests for Block-CSRMV
                cases
            */
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr.data(),
                                            csr_col_ind.data(),
                                            csr_val.data(),
                                            blk_row_ptr.data(),
                                            blk_col_ind.data(),
                                            blk_csr_val.data(),
                                            masks.data(),
                                            nRowsblk[iB],
                                            aoclsparse_index_base_one),
                      aoclsparse_status_success);

            status = aoclsparse_blkcsrmv<T>(trans,
                                            &alpha,
                                            m,
                                            n,
                                            nnz,
                                            masks.data(),
                                            blk_csr_val.data(),
                                            blk_col_ind.data(),
                                            blk_row_ptr.data(),
                                            descr,
                                            x,
                                            &beta,
                                            y,
                                            nRowsblk[iB]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with baseOne CSR input";
            if(status == aoclsparse_status_success)
            {
                EXPECT_ARR_NEAR(m, y, y_gold, expected_precision<T>(10.0));
            }
        }
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    template <typename T>
    void test_blkcsrmv_baseZeroCSRInput(aoclsparse_status blkcsrmv_status
                                        = aoclsparse_status_success)
    {
        aoclsparse_status    status;
        aoclsparse_operation trans = aoclsparse_operation_none;
        aoclsparse_mat_descr descr;
        int                  iB;
        aoclsparse_int       nRowsblk[3] = {1, 2, 4};

        aoclsparse_int              m = 6, n = 8, nnz = 14;
        T                           alpha = 1.0, beta = 0.0;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        T                           x[8]      = {1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00};
        T                           y[6]      = {0};
        T                           y_gold[6] = {8.00, 2.00, 0.00, 204.00, 23.00, 14.00};

        //Initialize block width
        const aoclsparse_int        blk_width = 8;
        std::vector<aoclsparse_int> blk_row_ptr;
        std::vector<aoclsparse_int> blk_col_ind;
        std::vector<T>              blk_csr_val;
        std::vector<uint8_t>        masks;

        csr_row_ptr.assign({0, 1, 2, 2, 10, 12, 14});
        csr_col_ind.assign({0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 0, 1});
        csr_val.assign(
            {8.00, 2.00, 3.00, 3.00, 3.00, 6.00, 10.00, 9.00, 6.00, 2.00, 2.00, 3.00, 2.00, 6.00});

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        aoclsparse_matrix A = nullptr;
        ASSERT_EQ(
            create_aoclsparse_matrix<T>(A, descr, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
            aoclsparse_status_success);

        for(iB = 0; iB < 3; iB++)
        {
            blk_row_ptr.resize(m + 1);
            blk_col_ind.resize(nnz);
            blk_csr_val.resize((nnz + (nRowsblk[iB] * blk_width)));
            masks.resize((nnz + (nRowsblk[iB] * blk_width)));

            std::fill(blk_row_ptr.begin(), blk_row_ptr.end(), 0);
            std::fill(blk_col_ind.begin(), blk_col_ind.end(), 0);
            std::fill(blk_csr_val.begin(), blk_csr_val.end(), 0.0);
            std::fill(masks.begin(), masks.end(), 0);

            //Function to convert csr to blkcsr
            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr.data(),
                                            csr_col_ind.data(),
                                            csr_val.data(),
                                            blk_row_ptr.data(),
                                            blk_col_ind.data(),
                                            blk_csr_val.data(),
                                            masks.data(),
                                            nRowsblk[iB],
                                            aoclsparse_index_base_zero),
                      aoclsparse_status_success);

            status = aoclsparse_blkcsrmv<T>(trans,
                                            &alpha,
                                            m,
                                            n,
                                            nnz,
                                            masks.data(),
                                            blk_csr_val.data(),
                                            blk_col_ind.data(),
                                            blk_row_ptr.data(),
                                            descr,
                                            x,
                                            &beta,
                                            y,
                                            nRowsblk[iB]);
            ASSERT_EQ(status, blkcsrmv_status) << "Test failed with unexpected return from "
                                                  "aoclsparse_blkcsrmv with baseZero CSR input";
            if(status == aoclsparse_status_success)
            {
                EXPECT_ARR_NEAR(m, y, y_gold, expected_precision<T>(10.0));
            }
        }

        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
    //Test cases for analysis and conversion routines
    template <typename T>
    void test_blkcsrmv_conversion()
    {
        aoclsparse_int        m = 2, n = 8, nnz = 3;
        T                     csr_val[]     = {3, 2, 1};
        aoclsparse_int        csr_col_ind[] = {1};
        aoclsparse_int        csr_row_ptr[] = {0, 1, 1};
        T                     blk_csr_val[nnz];
        aoclsparse_int        blk_col_ind[nnz];
        aoclsparse_int        blk_row_ptr[nnz];
        aoclsparse_int        nRowsblk[3] = {1, 2, 4};
        aoclsparse_int        total_blks;
        aoclsparse_mat_descr  descr;
        aoclsparse_index_base base = aoclsparse_index_base_zero;

        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        //Test cases to identify failures in findng optimal block size -- used from optmv
        EXPECT_EQ(aoclsparse_opt_blksize(-1, nnz, base, csr_row_ptr, csr_col_ind, &total_blks),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_opt_blksize(m, -1, base, csr_row_ptr, csr_col_ind, &total_blks),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_opt_blksize(m, nnz, base, nullptr, csr_col_ind, &total_blks),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_opt_blksize(m, nnz, base, csr_row_ptr, nullptr, &total_blks),
                  aoclsparse_status_invalid_pointer);

        //Test cases in conversion routines
        for(int i = 0; i < 3; i++)
        {
            uint8_t masks[nnz * nRowsblk[i]];

            EXPECT_EQ(aoclsparse_csr2blkcsr(m,
                                            n,
                                            nnz,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            blk_row_ptr,
                                            blk_col_ind,
                                            blk_csr_val,
                                            masks,
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
                                            nRowsblk[i],
                                            descr->base),
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
        Au::X86Cpu Cpu = {0};
        bool okblk = Cpu.hasFlag(Au::ECpuidFlag::avx512f) && Cpu.hasFlag(Au::ECpuidFlag::avx512vl);
        const aoclsparse_status blkcsrmv_status
            = okblk ? aoclsparse_status_invalid_pointer : aoclsparse_status_not_implemented;
        test_blkcsrmv_nullptr<double>(blkcsrmv_status);
    }
    TEST(blkcsrmv, WrongSizeDouble)
    {
        Au::X86Cpu Cpu = {0};
        bool okblk = Cpu.hasFlag(Au::ECpuidFlag::avx512f) && Cpu.hasFlag(Au::ECpuidFlag::avx512vl);
        const aoclsparse_status blkcsrmv_status
            = okblk ? aoclsparse_status_invalid_size : aoclsparse_status_not_implemented;
        test_blkcsrmv_wrong_size<double>(blkcsrmv_status);
    }

    TEST(blkcsrmv, DoNothingDouble)
    {
        Au::X86Cpu Cpu = {0};
        bool okblk = Cpu.hasFlag(Au::ECpuidFlag::avx512f) && Cpu.hasFlag(Au::ECpuidFlag::avx512vl);
        const aoclsparse_status blkcsrmv_status
            = okblk ? aoclsparse_status_success : aoclsparse_status_not_implemented;
        test_blkcsrmv_do_nothing<double>(blkcsrmv_status);
    }

    TEST(blkcsrmv, InvalidBase)
    {
        Au::X86Cpu Cpu = {0};
        bool okblk = Cpu.hasFlag(Au::ECpuidFlag::avx512f) && Cpu.hasFlag(Au::ECpuidFlag::avx512vl);
        const aoclsparse_status blkcsrmv_status
            = okblk ? aoclsparse_status_invalid_value : aoclsparse_status_not_implemented;
        test_blkcsrmv_invalid_base<double>(blkcsrmv_status);
    }
    TEST(blkcsrmv, Conversion)
    {
        test_blkcsrmv_conversion<double>();
    }

    TEST(blkcsrmv, AVX512BaseOneDoubleBlkCSRInput)
    {
        Au::X86Cpu Cpu = {0};
        bool okblk = Cpu.hasFlag(Au::ECpuidFlag::avx512f) && Cpu.hasFlag(Au::ECpuidFlag::avx512vl);
        const aoclsparse_status blkcsrmv_status
            = okblk ? aoclsparse_status_success : aoclsparse_status_not_implemented;
        test_blkcsrmv_baseOneBlkCSRInput<double>(blkcsrmv_status);
    }

    TEST(blkcsrmv, AVX512BaseOneDoubleCSRInput)
    {
        Au::X86Cpu Cpu = {0};
        bool okblk = Cpu.hasFlag(Au::ECpuidFlag::avx512f) && Cpu.hasFlag(Au::ECpuidFlag::avx512vl);
        const aoclsparse_status blkcsrmv_status
            = okblk ? aoclsparse_status_success : aoclsparse_status_not_implemented;
        test_blkcsrmv_baseOneCSRInput<double>(blkcsrmv_status);
    }
    TEST(blkcsrmv, AVX512BaseZeroDoubleCSRInput)
    {
        Au::X86Cpu Cpu = {0};
        bool okblk = Cpu.hasFlag(Au::ECpuidFlag::avx512f) && Cpu.hasFlag(Au::ECpuidFlag::avx512vl);
        const aoclsparse_status blkcsrmv_status
            = okblk ? aoclsparse_status_success : aoclsparse_status_not_implemented;
        test_blkcsrmv_baseZeroCSRInput<double>(blkcsrmv_status);
    }
} // namespace
