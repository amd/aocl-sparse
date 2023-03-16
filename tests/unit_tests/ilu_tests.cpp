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

#define VERBOSE 1

namespace
{

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_ilu_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A               = nullptr;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        aoclsparse_mat_descr        descr;
        T                           x[5] = {1.0};
        T                           b[5] = {1.0};
        m                                = 5;
        n                                = 5;
        nnz                              = 8;
        csr_row_ptr.assign({0, 2, 3, 4, 7, 8});
        csr_col_ind.assign({0, 3, 1, 2, 1, 3, 4, 4});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        aoclsparse_create_mat_descr(&descr);

        ASSERT_EQ(create_aoclsparse_matrix<T>(A, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        // In turns pass nullptr in every single pointer argument
        /*
            trans -> can be checked for invalid values
            A -> can be checked for nullptr
            descr -> can be checked for nullptr
            precond_csr_val -> is a output argument, will be passed as nullptr and expect LU factors in it
            approx_inv_diag ->  unused argument
            x -> can be checked for nullptr
            b -> can be checked for nullptr
        */
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, nullptr, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_ilu_smoother<T>(trans, A, nullptr, &precond_csr_val, approx_inv_diag, x, b),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A, descr, &precond_csr_val, approx_inv_diag, nullptr, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A, descr, &precond_csr_val, approx_inv_diag, x, nullptr),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }

    // tests with wrong scalar data n, m, nnz
    template <typename T>
    void test_ilu_wrong_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A_n_wrong       = nullptr;
        aoclsparse_matrix           A_m_wrong       = nullptr;
        aoclsparse_matrix           A_nnz_wrong     = nullptr;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        aoclsparse_int              wrong = -1;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        aoclsparse_mat_descr        descr;
        T                           x[5] = {1.0};
        T                           b[5] = {1.0};

        m   = 5;
        n   = 5;
        nnz = 8;
        csr_row_ptr.assign({0, 2, 3, 4, 7, 8});
        csr_col_ind.assign({0, 3, 1, 2, 1, 3, 4, 4});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        aoclsparse_create_mat_descr(&descr);

        ASSERT_EQ(create_aoclsparse_matrix<T>(
                      A_n_wrong, m, wrong, nnz, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_invalid_size);
        ASSERT_EQ(create_aoclsparse_matrix<T>(
                      A_m_wrong, wrong, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_invalid_size);
        ASSERT_EQ(create_aoclsparse_matrix<T>(
                      A_nnz_wrong, m, n, wrong, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_invalid_size);

        // aoclsparse_matrix "A" which contains members m,n and nnz are validated during matrix creation.
        // the below call should return aoclsparse_status_invalid_pointer, since matrix "A" is nullptr
        // which never got created due to wrong m,n,nnz values

        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_n_wrong, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_m_wrong, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_nnz_wrong, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A_n_wrong), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A_m_wrong), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A_nnz_wrong), aoclsparse_status_success);
    }
    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_ilu_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A_n_zero        = nullptr;
        aoclsparse_matrix           A_m_zero        = nullptr;
        aoclsparse_matrix           A_nnz_zero      = nullptr;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        aoclsparse_int              zero = 0;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        aoclsparse_mat_descr        descr;
        T                           x[5] = {1.0};
        T                           b[5] = {1.0};

        m   = 5;
        n   = 5;
        nnz = 8;
        csr_row_ptr.assign({0, 2, 3, 4, 7, 8});
        csr_col_ind.assign({0, 3, 1, 2, 1, 3, 4, 4});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        aoclsparse_create_mat_descr(&descr);

        /*
            pass zero arguments for m, n and nnz to test the creation API.
        */
        ASSERT_EQ(
            create_aoclsparse_matrix<T>(A_n_zero, m, zero, nnz, csr_row_ptr, csr_col_ind, csr_val),
            aoclsparse_status_success);
        ASSERT_EQ(
            create_aoclsparse_matrix<T>(A_m_zero, zero, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
            aoclsparse_status_success);
        ASSERT_EQ(
            create_aoclsparse_matrix<T>(A_nnz_zero, m, n, zero, csr_row_ptr, csr_col_ind, csr_val),
            aoclsparse_status_success);

        /*
            to check if the ILU API exits gracefully with success
            when the values zero are passed for m, n and nnz
        */
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_n_zero, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_m_zero, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_nnz_zero, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A_n_zero), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A_m_zero), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A_nnz_zero), aoclsparse_status_success);
    }
    // test not-implemented/supported scenarios
    template <typename T>
    void test_ilu_unsupported()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A               = nullptr;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        aoclsparse_mat_descr        descr;
        T                           x[5] = {1.0};
        T                           b[5] = {1.0};
        m                                = 5;
        n                                = 5;
        nnz                              = 8;
        csr_row_ptr.assign({0, 2, 3, 4, 7, 8});
        csr_col_ind.assign({0, 3, 1, 2, 1, 3, 4, 4});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        aoclsparse_create_mat_descr(&descr);

        ASSERT_EQ(create_aoclsparse_matrix<T>(A, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);
        
        /*
            check if the transpose operation is supported
        */
        trans = aoclsparse_operation_transpose;
        EXPECT_EQ(
            aoclsparse_ilu_smoother<T>(trans, A, descr, &precond_csr_val, approx_inv_diag, x, b),
            aoclsparse_status_not_implemented);
        /*
            check if the One-Based Indexing is supported
        */
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one);
        EXPECT_EQ(
            aoclsparse_ilu_smoother<T>(trans, A, descr, &precond_csr_val, approx_inv_diag, x, b),
            aoclsparse_status_not_implemented);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }
    // test by passing predefined matrices
    template <typename T>
    void test_ilu_predefined_matrices()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        T                           init_x = 1.0;
        T                           alpha = 1.0, beta = 0.0;
        aoclsparse_mat_descr        descr;
        T                          *x       = NULL;
        T                          *b       = NULL;

        /*
            use the 5 matrices from data utils to test positive scenarios
        */
        enum matrix_id              mids[5] = {sample_gmres_mat_01,
                                               sample_gmres_mat_02,
                                               sample_cg_mat,
                                               N5_full_sorted,
                                               N5_full_unsorted};
        for(aoclsparse_int idx = 0; idx < 5; idx++)
        {
            A = nullptr;
            x = nullptr;
            b = nullptr;
            descr = nullptr;

            ASSERT_EQ(
                create_matrix(
                    mids[idx], m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, A, descr, VERBOSE),
                aoclsparse_status_success);

            b = (T *)malloc(sizeof(T) * n);
            ASSERT_NE(b, nullptr);
            x = (T *)malloc(sizeof(T) * n);
            ASSERT_NE(x, nullptr);

            for(aoclsparse_int i = 0; i < n; i++)
            {
                x[i] = init_x;
            }

            // Reference SPMV CSR implementation
            for(aoclsparse_int i = 0; i < n; i++)
            {
                T result = 0.0;
                for(aoclsparse_int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
                {
                    result += alpha * csr_val[j] * x[csr_col_ind[j]];
                }
                b[i] = (beta * b[i]) + result;
            }

            EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                          trans, A, descr, &precond_csr_val, approx_inv_diag, x, b),
                      aoclsparse_status_success);

            free(b);
            b = NULL;
            free(x);
            x = NULL;
            EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        }
    }
    //TODO add:
    // * invalid array data (but we don't test these right now, e.g., col_ind out of bounds)
    // * nnz not matching row_ptr

    TEST(ilu, NullArgDouble)
    {
        test_ilu_nullptr<double>();
    }
    TEST(ilu, NullArgFloat)
    {
        test_ilu_nullptr<float>();
    }

    TEST(ilu, WrongSizeDouble)
    {
        test_ilu_wrong_size<double>();
    }
    TEST(ilu, WrongSizeFloat)
    {
        test_ilu_wrong_size<float>();
    }

    TEST(ilu, DoNothingDouble)
    {
        test_ilu_do_nothing<double>();
    }
    TEST(ilu, DoNothingFloat)
    {
        test_ilu_do_nothing<float>();
    }

    TEST(ilu, UnsupportedDouble)
    {
        test_ilu_unsupported<double>();
    }
    TEST(ilu, UnsupportedFloat)
    {
        test_ilu_unsupported<float>();
    }

    TEST(ilu, PredefinedMatricesDouble)
    {
        test_ilu_predefined_matrices<double>();
    }
    TEST(ilu, PredefinedMatricesFloat)
    {
        test_ilu_predefined_matrices<float>();
    }
} // namespace
