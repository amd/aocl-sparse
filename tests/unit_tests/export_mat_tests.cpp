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
#include "aoclsparse_analysis.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_types.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"

namespace
{
    template <typename T>
    void verify_export_data(aoclsparse_matrix_format_type mat_type,
                            aoclsparse_int                in_m,
                            aoclsparse_int                in_n,
                            aoclsparse_int                in_nnz,
                            aoclsparse_index_base         in_base,
                            const aoclsparse_int         *in_idx_ptr,
                            const aoclsparse_int         *in_indices,
                            const T                      *in_val,
                            aoclsparse_int                out_m,
                            aoclsparse_int                out_n,
                            aoclsparse_int                out_nnz,
                            aoclsparse_index_base         out_base,
                            const aoclsparse_int         *out_idx_ptr,
                            const aoclsparse_int         *out_indices,
                            const T                      *out_val)
    {
        EXPECT_EQ(in_m, out_m);
        EXPECT_EQ(in_n, out_n);
        EXPECT_EQ(in_nnz, out_nnz);
        EXPECT_EQ(in_base, out_base);

        EXPECT_EQ_VEC(out_nnz, in_indices, out_indices);
        if(mat_type == aoclsparse_csr_mat)
        {
            EXPECT_EQ_VEC(out_m + 1, in_idx_ptr, out_idx_ptr);
        }
        else if(mat_type == aoclsparse_csc_mat)
        {
            EXPECT_EQ_VEC(out_n + 1, in_idx_ptr, out_idx_ptr);
        }

        if constexpr(std::is_same_v<T, float>)
        {
            EXPECT_FLOAT_EQ_VEC(out_nnz, in_val, out_val);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            EXPECT_DOUBLE_EQ_VEC(out_nnz, in_val, out_val);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            EXPECT_COMPLEX_FLOAT_EQ_VEC(out_nnz,
                                        static_cast<std::complex<float> *>(in_val),
                                        static_cast<std::complex<float> *>(out_val));
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            EXPECT_COMPLEX_DOUBLE_EQ_VEC(out_nnz,
                                         static_cast<std::complex<double> *>(in_val),
                                         static_cast<std::complex<double> *>(out_val));
        }
    }

    TEST(ExportMatTest, CSR_NullArgs)
    {
        aoclsparse_matrix          A             = NULL;
        float                      valf[]        = {2.0f, 8.0f, 6.0f};
        aoclsparse_int             csr_col_idx[] = {0, 1, 1};
        aoclsparse_int             csr_row_ptr[] = {0, 2, 3};
        aoclsparse_int             m             = 2;
        aoclsparse_int             n             = 2;
        aoclsparse_int             nnz           = 3;
        aoclsparse_index_base      in_base       = aoclsparse_index_base_zero, out_base;
        aoclsparse_int            *out_row_ptr   = NULL;
        aoclsparse_int            *out_col_idx   = NULL;
        float                     *out_valf      = NULL;
        double                    *out_vald      = NULL;
        double                   **out_vald_null = NULL;
        aoclsparse_float_complex  *out_valcf     = NULL;
        aoclsparse_double_complex *out_valcd     = NULL;
        aoclsparse_int             out_m, out_n, out_nnz;
        aoclsparse_int            *tmp_arr;

        // 1) Matrix is NULL
        EXPECT_EQ(
            aoclsparse_export_csr(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
            aoclsparse_status_invalid_pointer);

        // 2) OUT param is NULL
        ASSERT_EQ(aoclsparse_create_csr(&A, in_base, m, n, nnz, csr_row_ptr, csr_col_idx, valf),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_export_csr(
                      A, NULL, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_vald),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csr(
                      A, &out_base, NULL, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valcf),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csr(
                      A, &out_base, &out_m, NULL, &out_nnz, &out_row_ptr, &out_col_idx, &out_valcd),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csr(
                      A, &out_base, &out_m, &out_n, NULL, &out_row_ptr, &out_col_idx, &out_valf),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csr(
                      A, &out_base, &out_m, &out_n, &out_nnz, NULL, &out_col_idx, &out_vald),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csr(
                      A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, NULL, &out_valf),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_export_csr(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, out_vald_null),
            aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&A);

        // 3) Matrix is in CSR format(unoptimized). But one of the csr_mat pointer is NULL.
        ASSERT_EQ(aoclsparse_create_csr(&A, in_base, m, n, nnz, csr_row_ptr, csr_col_idx, valf),
                  aoclsparse_status_success);
        // 3.a) csr_row_ptr is NULL
        tmp_arr                = A->csr_mat.csr_row_ptr;
        A->csr_mat.csr_row_ptr = NULL;
        EXPECT_EQ(
            aoclsparse_export_csr(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
            aoclsparse_status_invalid_value);
        // 3.b) csr_col_ptr is NULL
        A->csr_mat.csr_row_ptr = tmp_arr;
        tmp_arr                = A->csr_mat.csr_col_ptr;
        A->csr_mat.csr_col_ptr = NULL;
        EXPECT_EQ(
            aoclsparse_export_csr(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
            aoclsparse_status_invalid_value);
        // 3.c) csr_val is NULL
        A->csr_mat.csr_col_ptr = tmp_arr;
        A->csr_mat.csr_val     = NULL;
        EXPECT_EQ(
            aoclsparse_export_csr(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
            aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);
    }
    TEST(ExportMatTest, CSR_WrongInput)
    {
        aoclsparse_matrix         A             = NULL;
        float                     valf[]        = {2.0f, 8.0f, 6.0f};
        double                    vald[]        = {2.0, 8.0, 6.0};
        aoclsparse_int            csc_row_idx[] = {0, 1, 1};
        aoclsparse_int            csc_col_ptr[] = {0, 2, 3};
        aoclsparse_int            csr_col_idx[] = {0, 1, 1};
        aoclsparse_int            csr_row_ptr[] = {0, 2, 3};
        aoclsparse_int            m             = 2;
        aoclsparse_int            n             = 2;
        aoclsparse_int            nnz           = 3;
        aoclsparse_index_base     in_base       = aoclsparse_index_base_zero, out_base;
        aoclsparse_int           *out_row_ptr   = NULL;
        aoclsparse_int           *out_col_idx   = NULL;
        double                   *out_vald      = NULL;
        aoclsparse_float_complex *out_valcf     = NULL;
        aoclsparse_int            out_m, out_n, out_nnz;

        // 1) Matrix is in CSC format but want to export as CSR
        ASSERT_EQ(aoclsparse_create_csc(&A, in_base, m, n, nnz, csc_col_ptr, csc_row_idx, vald),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_export_csr(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_vald),
            aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);

        // 2) Matric data type is float but want to export as complex float
        ASSERT_EQ(aoclsparse_create_csr(&A, in_base, m, n, nnz, csr_row_ptr, csr_col_idx, valf),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_export_csr(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valcf),
            aoclsparse_status_wrong_type);
        aoclsparse_destroy(&A);
    }
    TEST(ExportMatTest, CSR_Success)
    {
        aoclsparse_matrix     A           = NULL;
        aoclsparse_int        m           = 5;
        aoclsparse_int        n           = 5;
        aoclsparse_int        nnz         = 7;
        aoclsparse_index_base in_base     = aoclsparse_index_base_zero, out_base;
        aoclsparse_int       *out_row_ptr = NULL;
        aoclsparse_int       *out_col_idx = NULL;
        float                *out_valf    = NULL;
        aoclsparse_int        out_m, out_n, out_nnz;
        aoclsparse_int       *tmp_arr;
        void                 *val;

        // Initialise matrix
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  0  0  0
        //  0  5  0  6  7
        //  0  0  0  0  8
        aoclsparse_int       csr_row_ptr[] = {0, 2, 3, 3, 6, 7};
        aoclsparse_int       csr_col_idx[] = {0, 3, 1, 3, 4, 1, 4};
        float                valf[]        = {1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 5.0f, 8.0f};
        aoclsparse_mat_descr descr;

        // 1) Matrix is in CSR format(unoptimized). Export csr_mat.
        {
            SCOPED_TRACE("1. CSR unoptimized");
            ASSERT_EQ(aoclsparse_create_csr(&A, in_base, m, n, nnz, csr_row_ptr, csr_col_idx, valf),
                      aoclsparse_status_success);
            EXPECT_EQ(
                aoclsparse_export_csr(
                    A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
                aoclsparse_status_success);
            verify_export_data(aoclsparse_csr_mat,
                               m,
                               n,
                               nnz,
                               in_base,
                               csr_row_ptr,
                               csr_col_idx,
                               valf,
                               out_m,
                               out_n,
                               out_nnz,
                               out_base,
                               out_row_ptr,
                               out_col_idx,
                               out_valf);
            aoclsparse_destroy(&A);
        }

        // 2) Matrix is in CSR format(1-base, optimized). Export optimized CSR.
        {
            SCOPED_TRACE("2. CSR 1-base, optimized");
            in_base                              = aoclsparse_index_base_one;
            aoclsparse_int        csr_row_ptr1[] = {1, 3, 4, 4, 7, 8};
            aoclsparse_int        csr_col_idx1[] = {1, 4, 2, 4, 5, 2, 5};
            float                 valf1[]        = {1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 5.0f, 8.0f};
            aoclsparse_index_base exp_base       = aoclsparse_index_base_zero;
            aoclsparse_int        exp_nnz        = 8;
            aoclsparse_int        exp_row_ptr1[] = {0, 2, 3, 4, 7, 8};
            aoclsparse_int        exp_col_idx1[] = {0, 3, 1, 2, 1, 3, 4, 4};
            float                 exp_valf1[]    = {1.0f, 2.0f, 3.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            aoclsparse_create_mat_descr(&descr);
            ASSERT_EQ(
                aoclsparse_create_csr(&A, in_base, m, n, nnz, csr_row_ptr1, csr_col_idx1, valf1),
                aoclsparse_status_success);
            aoclsparse_set_mv_hint(A, aoclsparse_operation_none, descr, 1);
            aoclsparse_optimize(A);
            EXPECT_EQ(
                aoclsparse_export_csr(
                    A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
                aoclsparse_status_success);
            aoclsparse_destroy_mat_descr(descr);
            verify_export_data(aoclsparse_csr_mat,
                               m,
                               n,
                               exp_nnz,
                               exp_base,
                               exp_row_ptr1,
                               exp_col_idx1,
                               exp_valf1,
                               out_m,
                               out_n,
                               out_nnz,
                               out_base,
                               out_row_ptr,
                               out_col_idx,
                               out_valf);
            aoclsparse_destroy(&A);
        }

        // 3) Matrix is in CSR format(optimized) but one of the opt_csr_mat is NULL.
        in_base                       = aoclsparse_index_base_zero;
        aoclsparse_int csr_row_ptr2[] = {0, 2, 3, 3, 6, 7};
        aoclsparse_int csr_col_idx2[] = {0, 3, 1, 3, 4, 1, 4};
        float          valf2[]        = {1.0f, 2.0f, 3.0f, 6.0f, 7.0f, 5.0f, 8.0f};
        aoclsparse_create_mat_descr(&descr);
        ASSERT_EQ(aoclsparse_create_csr(&A, in_base, m, n, nnz, csr_row_ptr2, csr_col_idx2, valf2),
                  aoclsparse_status_success);
        aoclsparse_set_mv_hint(A, aoclsparse_operation_none, descr, 1);
        aoclsparse_optimize(A);
        // 3.a) opt_csr_mat.csr_row_ptr is NULL. Return pointers from csr_mat.
        {
            SCOPED_TRACE("3.a CSR 1-base, optimized, hacked NULL row_ptr");
            tmp_arr                    = A->opt_csr_mat.csr_row_ptr;
            A->opt_csr_mat.csr_row_ptr = NULL;
            EXPECT_EQ(
                aoclsparse_export_csr(
                    A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
                aoclsparse_status_success);
            verify_export_data(aoclsparse_csr_mat,
                               m,
                               n,
                               nnz,
                               in_base,
                               csr_row_ptr2,
                               csr_col_idx2,
                               valf2,
                               out_m,
                               out_n,
                               out_nnz,
                               out_base,
                               out_row_ptr,
                               out_col_idx,
                               out_valf);
            A->opt_csr_mat.csr_row_ptr = tmp_arr;
        }

        // 3.b) opt_csr_mat.csr_col_ptr is NULL. Return pointers from csr_mat.
        {
            SCOPED_TRACE("3.b CSR 1-base, optimized, hacked NULL col_ptr");
            tmp_arr                    = A->opt_csr_mat.csr_col_ptr;
            A->opt_csr_mat.csr_col_ptr = NULL;
            EXPECT_EQ(
                aoclsparse_export_csr(
                    A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
                aoclsparse_status_success);
            verify_export_data(aoclsparse_csr_mat,
                               m,
                               n,
                               nnz,
                               in_base,
                               csr_row_ptr2,
                               csr_col_idx2,
                               valf2,
                               out_m,
                               out_n,
                               out_nnz,
                               out_base,
                               out_row_ptr,
                               out_col_idx,
                               out_valf);
            A->opt_csr_mat.csr_col_ptr = tmp_arr;
        }

        // 3.c) opt_csr_mat.csr_col is NULL. Return pointers from csr_mat.
        {
            SCOPED_TRACE("3.c CSR 1-base, optimized, hacked NULL val");
            val                    = A->opt_csr_mat.csr_val;
            A->opt_csr_mat.csr_val = NULL;
            EXPECT_EQ(
                aoclsparse_export_csr(
                    A, &out_base, &out_m, &out_n, &out_nnz, &out_row_ptr, &out_col_idx, &out_valf),
                aoclsparse_status_success);
            verify_export_data(aoclsparse_csr_mat,
                               m,
                               n,
                               nnz,
                               in_base,
                               csr_row_ptr2,
                               csr_col_idx2,
                               valf2,
                               out_m,
                               out_n,
                               out_nnz,
                               out_base,
                               out_row_ptr,
                               out_col_idx,
                               out_valf);
            A->opt_csr_mat.csr_val = val;
            aoclsparse_destroy_mat_descr(descr);
            aoclsparse_destroy(&A);
        }
    }

    TEST(ExportMatTest, CSC_NullArgs)
    {
        aoclsparse_matrix          A             = NULL;
        float                      valf[]        = {2.0f, 8.0f, 6.0f};
        aoclsparse_int             csc_row_idx[] = {1, 2, 2};
        aoclsparse_int             csc_col_ptr[] = {1, 3, 4};
        aoclsparse_int             m             = 2;
        aoclsparse_int             n             = 2;
        aoclsparse_int             nnz           = 3;
        aoclsparse_index_base      out_base;
        aoclsparse_int            *out_col_ptr   = NULL;
        aoclsparse_int            *out_row_idx   = NULL;
        float                     *out_valf      = NULL;
        double                    *out_vald      = NULL;
        double                   **out_vald_null = NULL;
        aoclsparse_float_complex  *out_valcf     = NULL;
        aoclsparse_double_complex *out_valcd     = NULL;
        aoclsparse_int             out_m, out_n, out_nnz;
        aoclsparse_int            *tmp_arr;

        // 1) Matrix is NULL
        EXPECT_EQ(
            aoclsparse_export_csc(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_valf),
            aoclsparse_status_invalid_pointer);

        // 2) OUT param is NULL
        ASSERT_EQ(aoclsparse_create_csc(
                      &A, aoclsparse_index_base_one, m, n, nnz, csc_col_ptr, csc_row_idx, valf),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_export_csc(
                      A, NULL, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_valf),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csc(
                      A, &out_base, NULL, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_vald),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csc(
                      A, &out_base, &out_m, NULL, &out_nnz, &out_col_ptr, &out_row_idx, &out_vald),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csc(
                      A, &out_base, &out_m, &out_n, NULL, &out_col_ptr, &out_row_idx, &out_valf),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csc(
                      A, &out_base, &out_m, &out_n, &out_nnz, NULL, &out_row_idx, &out_valcf),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_export_csc(
                      A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, NULL, &out_valcd),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_export_csc(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, out_vald_null),
            aoclsparse_status_invalid_pointer);
        aoclsparse_destroy(&A);

        // 3) Matrix is on CSC format but one of the csc_mat pointer is NULL
        ASSERT_EQ(aoclsparse_create_csc(
                      &A, aoclsparse_index_base_one, m, n, nnz, csc_col_ptr, csc_row_idx, valf),
                  aoclsparse_status_success);
        tmp_arr            = A->csc_mat.col_ptr;
        A->csc_mat.col_ptr = NULL;
        EXPECT_EQ(
            aoclsparse_export_csc(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_valf),
            aoclsparse_status_invalid_value);
        A->csc_mat.col_ptr = tmp_arr;
        tmp_arr            = A->csc_mat.row_idx;
        A->csc_mat.row_idx = NULL;
        EXPECT_EQ(
            aoclsparse_export_csc(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_valf),
            aoclsparse_status_invalid_value);
        A->csc_mat.row_idx = tmp_arr;
        A->csc_mat.val     = NULL;
        EXPECT_EQ(
            aoclsparse_export_csc(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_valf),
            aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);
    }
    TEST(ExportMatTest, CSC_WrongInput)
    {
        aoclsparse_matrix          A             = NULL;
        float                      valf[]        = {2.0f, 8.0f, 6.0f};
        aoclsparse_int             csr_col_idx[] = {1, 1, 2};
        aoclsparse_int             csr_row_ptr[] = {1, 2, 4};
        aoclsparse_int             csc_row_idx[] = {1, 2, 2};
        aoclsparse_int             csc_col_ptr[] = {1, 3, 4};
        aoclsparse_int             m             = 2;
        aoclsparse_int             n             = 2;
        aoclsparse_int             nnz           = 3;
        aoclsparse_index_base      in_base       = aoclsparse_index_base_one, out_base;
        aoclsparse_int            *out_col_ptr   = NULL;
        aoclsparse_int            *out_row_idx   = NULL;
        float                     *out_valf      = NULL;
        aoclsparse_double_complex *out_valcd     = NULL;
        aoclsparse_int             out_m, out_n, out_nnz;

        // 1) Matrix is in CSR format but want to export as CSC
        ASSERT_EQ(aoclsparse_create_csr(&A, in_base, m, n, nnz, csr_row_ptr, csr_col_idx, valf),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_export_csc(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_valf),
            aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);

        // 2) Matric data type is float but want to export as complex double
        ASSERT_EQ(aoclsparse_create_csc(&A, in_base, m, n, nnz, csc_col_ptr, csc_row_idx, valf),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_export_csc(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_valcd),
            aoclsparse_status_wrong_type);
        aoclsparse_destroy(&A);
    }
    TEST(ExportMatTest, CSC_Success)
    {
        aoclsparse_matrix     A             = NULL;
        float                 valf[]        = {2.0f, 8.0f, 6.0f};
        aoclsparse_int        csc_row_idx[] = {1, 2, 2};
        aoclsparse_int        csc_col_ptr[] = {1, 3, 4};
        aoclsparse_int        m             = 2;
        aoclsparse_int        n             = 2;
        aoclsparse_int        nnz           = 3;
        aoclsparse_index_base in_base       = aoclsparse_index_base_one, out_base;
        aoclsparse_int       *out_col_ptr   = NULL;
        aoclsparse_int       *out_row_idx   = NULL;
        float                *out_valf      = NULL;
        aoclsparse_int        out_m, out_n, out_nnz;

        // Matrix is in CSC format and want to export as CSC
        ASSERT_EQ(aoclsparse_create_csc(&A, in_base, m, n, nnz, csc_col_ptr, csc_row_idx, valf),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_export_csc(
                A, &out_base, &out_m, &out_n, &out_nnz, &out_col_ptr, &out_row_idx, &out_valf),
            aoclsparse_status_success);
        verify_export_data(aoclsparse_csc_mat,
                           m,
                           n,
                           nnz,
                           in_base,
                           csc_col_ptr,
                           csc_row_idx,
                           valf,
                           out_m,
                           out_n,
                           out_nnz,
                           out_base,
                           out_col_ptr,
                           out_row_idx,
                           out_valf);
        aoclsparse_destroy(&A);
    }

} // namespace
