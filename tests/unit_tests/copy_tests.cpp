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
#include "aoclsparse.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"

namespace
{
    void test_compare_matrix(const aoclsparse_matrix src, const aoclsparse_matrix dest)
    {
        void *src_val;
        void *dest_val;
        EXPECT_EQ(src->m, dest->m);
        EXPECT_EQ(src->n, dest->n);
        EXPECT_EQ(src->nnz, dest->nnz);
        EXPECT_EQ(src->mats[0]->base, dest->mats[0]->base);
        EXPECT_EQ(src->input_format, dest->input_format);
        EXPECT_EQ(src->val_type, dest->val_type);

        if(src->input_format == aoclsparse_csr_mat)
        {
            aoclsparse::csr *src_csr  = dynamic_cast<aoclsparse::csr *>(src->mats[0]);
            aoclsparse::csr *dest_csr = dynamic_cast<aoclsparse::csr *>(dest->mats[0]);
            EXPECT_NE(src_csr, nullptr);
            EXPECT_NE(dest_csr, nullptr);
            EXPECT_EQ_VEC(src_csr->m + 1, src_csr->ptr, dest_csr->ptr);
            EXPECT_EQ_VEC(src_csr->nnz, src_csr->ind, dest_csr->ind);
            src_val  = src_csr->val;
            dest_val = dest_csr->val;
        }
        else if(src->input_format == aoclsparse_coo_mat)
        {
            aoclsparse::coo *src_mat  = dynamic_cast<aoclsparse::coo *>(src->mats[0]);
            aoclsparse::coo *dest_mat = dynamic_cast<aoclsparse::coo *>(dest->mats[0]);
            EXPECT_EQ_VEC(src->nnz, src_mat->row_ind, dest_mat->row_ind);
            EXPECT_EQ_VEC(src->nnz, src_mat->col_ind, dest_mat->col_ind);
            src_val  = src_mat->val;
            dest_val = dest_mat->val;
        }
        if(src->val_type == aoclsparse_smat)
        {
            EXPECT_FLOAT_EQ_VEC(
                src->nnz, static_cast<float *>(src_val), static_cast<float *>(dest_val));
        }
        else if(src->val_type == aoclsparse_dmat)
        {
            EXPECT_DOUBLE_EQ_VEC(
                src->nnz, static_cast<double *>(src_val), static_cast<double *>(dest_val));
        }
        else if(src->val_type == aoclsparse_cmat)
        {
            EXPECT_COMPLEX_FLOAT_EQ_VEC(src->nnz,
                                        static_cast<std::complex<float> *>(src_val),
                                        static_cast<std::complex<float> *>(dest_val));
        }
        else if(src->val_type == aoclsparse_zmat)
        {
            EXPECT_COMPLEX_DOUBLE_EQ_VEC(src->nnz,
                                         static_cast<std::complex<double> *>(src_val),
                                         static_cast<std::complex<double> *>(dest_val));
        }
    }

    TEST(CopyTest, CopyCsr)
    {
        aoclsparse_matrix    src   = NULL;
        aoclsparse_matrix    dest  = NULL;
        aoclsparse_int       m     = 2;
        aoclsparse_int       n     = 3;
        aoclsparse_int       nnz   = 2;
        float                val[] = {2.0f, 8.0f};
        aoclsparse_mat_descr descr = NULL;

        // 0-based CSR matrix
        aoclsparse_int col_idx0[] = {0, 1};
        aoclsparse_int row_ptr0[] = {0, 1, 2};
        ASSERT_EQ(aoclsparse_create_scsr(
                      &src, aoclsparse_index_base_zero, m, n, nnz, row_ptr0, col_idx0, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);

        // 1-based CSR matrix
        aoclsparse_int col_idx1[] = {1, 2};
        aoclsparse_int row_ptr1[] = {1, 2, 3};
        ASSERT_EQ(aoclsparse_create_scsr(
                      &src, aoclsparse_index_base_one, m, n, nnz, row_ptr1, col_idx1, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);

        // 0-base matrix, m=0, n=3, nnz = 0
        aoclsparse_int col_idx2[] = {0, 0};
        aoclsparse_int row_ptr2[] = {0};
        ASSERT_EQ(aoclsparse_create_scsr(
                      &src, aoclsparse_index_base_zero, 0, n, 0, row_ptr2, col_idx2, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);
    }

    TEST(CopyTest, CopyCsc)
    {
        aoclsparse_matrix    src   = NULL;
        aoclsparse_matrix    dest  = NULL;
        aoclsparse_int       m     = 2;
        aoclsparse_int       n     = 3;
        aoclsparse_int       nnz   = 2;
        double               val[] = {2.0, 8.0};
        aoclsparse_mat_descr descr = NULL;

        // 0-based CSC matrix
        aoclsparse_int row_idx0[] = {0, 1};
        aoclsparse_int col_ptr0[] = {0, 1, 2, 2};
        ASSERT_EQ(aoclsparse_create_dcsc(
                      &src, aoclsparse_index_base_zero, m, n, nnz, col_ptr0, row_idx0, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);

        // 1-based CSC matrix
        aoclsparse_int row_idx1[] = {1, 2};
        aoclsparse_int col_ptr1[] = {1, 2, 3, 3};
        ASSERT_EQ(aoclsparse_create_dcsc(
                      &src, aoclsparse_index_base_one, m, n, nnz, col_ptr1, row_idx1, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);

        // 0-based matrix, m=2, n=0, nnz=0
        aoclsparse_int row_idx2[] = {0, 0};
        aoclsparse_int col_ptr2[] = {0};
        ASSERT_EQ(aoclsparse_create_dcsc(
                      &src, aoclsparse_index_base_zero, m, 0, 0, col_ptr2, row_idx2, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);
    }

    TEST(CopyTest, CopyCoo)
    {
        aoclsparse_matrix        src   = NULL;
        aoclsparse_matrix        dest  = NULL;
        aoclsparse_int           m     = 2;
        aoclsparse_int           n     = 3;
        aoclsparse_int           nnz   = 2;
        aoclsparse_float_complex val[] = {{2.0f, 10.0f}, {8.0f, 50.0f}};
        aoclsparse_mat_descr     descr = NULL;

        // 0-based COO matrix
        aoclsparse_int row_idx0[] = {0, 1};
        aoclsparse_int col_idx0[] = {0, 1};
        ASSERT_EQ(aoclsparse_create_ccoo(
                      &src, aoclsparse_index_base_zero, m, n, nnz, row_idx0, col_idx0, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);

        // 1-based COO matrix
        aoclsparse_int row_idx1[] = {1, 2};
        aoclsparse_int col_idx1[] = {1, 2};
        ASSERT_EQ(aoclsparse_create_ccoo(
                      &src, aoclsparse_index_base_one, m, n, nnz, row_idx1, col_idx1, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);

        // 0-based matrix, nnz=0
        aoclsparse_int row_idx2[] = {0, 0};
        aoclsparse_int col_idx2[] = {0, 0};
        ASSERT_EQ(aoclsparse_create_ccoo(
                      &src, aoclsparse_index_base_zero, m, n, 0, row_idx2, col_idx2, val),
                  aoclsparse_status_success);
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_success);
        test_compare_matrix(src, dest);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);
    }

    TEST(CopyTest, CsrMiscCases)
    {
        aoclsparse_matrix    src       = NULL;
        aoclsparse_matrix    dest      = NULL;
        aoclsparse_int       m         = 2;
        aoclsparse_int       n         = 3;
        aoclsparse_int       nnz       = 2;
        float                val[]     = {10.0f, 50.0f};
        aoclsparse_int       col_idx[] = {0, 1};
        aoclsparse_int       row_ptr[] = {0, 1, 2};
        aoclsparse_mat_descr descr     = NULL;

        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_create_scsr(
                      &src, aoclsparse_index_base_zero, m, n, nnz, row_ptr, col_idx, val),
                  aoclsparse_status_success);
        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(src->mats[0]);
        EXPECT_NE(csr_mat, nullptr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        // NULL src matrix
        EXPECT_EQ(aoclsparse_copy(NULL, descr, &dest), aoclsparse_status_invalid_pointer);

        // NULL descr matrix
        //EXPECT_EQ(aoclsparse_copy(src, NULL, &dest), aoclsparse_status_invalid_pointer);

        // NULL dest matrix
        EXPECT_EQ(aoclsparse_copy(src, descr, NULL), aoclsparse_status_invalid_pointer);

        // same dest matrix as src
        EXPECT_EQ(aoclsparse_copy(src, descr, &src), aoclsparse_status_invalid_pointer);

        // nnz < 0
        src->nnz = -1;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_size);

        // m < 0
        src->nnz = nnz;
        src->m   = -1;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_size);

        // val array is NULL
        src->m       = m;
        csr_mat->val = NULL;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);
    }

    TEST(CopyTest, CscMiscCases)
    {
        aoclsparse_matrix         src       = NULL;
        aoclsparse_matrix         dest      = NULL;
        aoclsparse_int            m         = 2;
        aoclsparse_int            n         = 3;
        aoclsparse_int            nnz       = 2;
        aoclsparse_double_complex val[]     = {{2.0, 10.0}, {8.0, 50.0}};
        aoclsparse_int            row_idx[] = {0, 1};
        aoclsparse_int            col_ptr[] = {0, 1, 2, 2};
        aoclsparse_mat_descr      descr     = NULL;

        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_create_zcsc(
                      &src, aoclsparse_index_base_zero, m, n, nnz, col_ptr, row_idx, val),
                  aoclsparse_status_success);
        aoclsparse::csr *csc_mat = dynamic_cast<aoclsparse::csr *>(src->mats[0]);
        EXPECT_NE(csc_mat, nullptr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        // NULL src matrix
        EXPECT_EQ(aoclsparse_copy(NULL, descr, &dest), aoclsparse_status_invalid_pointer);

        // NULL descr matrix
        //EXPECT_EQ(aoclsparse_copy(src, NULL, &dest), aoclsparse_status_invalid_pointer);

        // nnz < 0
        src->nnz = -1;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_size);

        // n < 0
        src->nnz = nnz;
        src->n   = -1;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_size);

        // col_ptr array is NULL
        src->n       = n;
        csc_mat->ptr = NULL;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_pointer);

        // row_idx array is NULL
        csc_mat->ptr = col_ptr;
        csc_mat->ind = NULL;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_pointer);

        // val array is NULL
        csc_mat->ind = row_idx;
        csc_mat->val = NULL;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);
    }

    TEST(CopyTest, CooMiscCases)
    {
        aoclsparse_matrix    src       = NULL;
        aoclsparse_matrix    dest      = NULL;
        aoclsparse_int       m         = 2;
        aoclsparse_int       n         = 3;
        aoclsparse_int       nnz       = 2;
        float                val[]     = {2.0f, 8.0f};
        aoclsparse_int       row_idx[] = {0, 1};
        aoclsparse_int       col_idx[] = {0, 1};
        aoclsparse_mat_descr descr     = NULL;

        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_create_scoo(
                      &src, aoclsparse_index_base_zero, m, n, nnz, row_idx, col_idx, val),
                  aoclsparse_status_success);

        aoclsparse::coo *coo_mat = dynamic_cast<aoclsparse::coo *>(src->mats[0]);
        EXPECT_NE(coo_mat, nullptr);
        ASSERT_EQ(aoclsparse_set_sv_hint(src, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        // nnz < 0
        src->nnz = -1;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_size);

        // col_ind array is NULL
        src->nnz         = nnz;
        coo_mat->col_ind = NULL;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_pointer);

        // row_ind array is NULL
        coo_mat->col_ind = col_idx;
        coo_mat->row_ind = NULL;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_pointer);

        // val array is NULL
        coo_mat->row_ind = row_idx;
        coo_mat->val     = NULL;
        EXPECT_EQ(aoclsparse_copy(src, descr, &dest), aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&src);
        aoclsparse_destroy(&dest);
        aoclsparse_destroy_mat_descr(descr);
    }
} // namespace
