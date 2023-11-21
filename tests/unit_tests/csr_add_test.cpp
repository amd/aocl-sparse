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
#include "aoclsparse_auxiliary.h"
#include "aoclsparse_convert.h"
#include "aoclsparse_types.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"

#include <algorithm>
#include <vector>

using namespace std;

namespace
{
    template <typename T, typename F>
    static bool array_complex_match(T *a, F *b, int size, bool isConj)
    {
        T *x = reinterpret_cast<T *>(b);
        for(int i = 0; i < size; i++)
        {
            if(a[i].real != x[i].real && a[i].imag != x[i].imag * (isConj ? -1 : 1))
                return false;
        }
        return true;
    }

    template <typename T, typename F>
    static bool array_match(T *a, F *b, int size)
    {
        T *x = reinterpret_cast<T *>(b);
        for(int i = 0; i < size; i++)
        {
            if(a[i] != x[i])
                return false;
        }
        return true;
    }

    void startup(vector<aoclsparse_int> &A_row,
                 vector<aoclsparse_int> &A_col,
                 vector<aoclsparse_int> &B_row,
                 vector<aoclsparse_int> &B_col,
                 bool                    is_transpose,
                 bool                    isAOneBase,
                 bool                    isBOneBase,
                 bool                    isAZeroNNZ,
                 bool                    isBZeroNNZ,
                 bool                    getTranspose = false)
    {
        if(is_transpose)
        {
            if(isAZeroNNZ)
            {
                A_row.assign({0, 0, 0, 0, 0});
                A_col.assign({0});
            }
            else
            {
                A_row.assign({0, 1, 3, 3, 4});
                A_col.assign({0, 0, 1, 2});
            }
        }
        else
        {
            if(isAZeroNNZ)
            {
                A_row.assign({0, 0, 0, 0});
                A_col.assign({0});
            }
            else
            {
                A_row.assign({0, 2, 3, 4});
                A_col.assign({0, 1, 1, 3});
            }
            if(getTranspose)
            {
                if(isAOneBase)
                {
                    transform(
                        A_row.begin(), A_row.end(), A_row.begin(), [&](auto x) { return x + 1; });
                    transform(
                        A_col.begin(), A_col.end(), A_col.begin(), [&](auto x) { return x + 1; });
                }
                return;
            }
        }

        if(isBZeroNNZ)
        {
            B_row.assign({0, 0, 0, 0});
            B_col.assign({0});
        }
        else
        {
            B_row.assign({0, 2, 4, 5});
            B_col.assign({0, 1, 1, 2, 3});
        }
        if(isAOneBase)
        {
            transform(A_row.begin(), A_row.end(), A_row.begin(), [&](auto x) { return x + 1; });
            transform(A_col.begin(), A_col.end(), A_col.begin(), [&](auto x) { return x + 1; });
        }
        if(isBOneBase)
        {
            transform(B_row.begin(), B_row.end(), B_row.begin(), [&](auto x) { return x + 1; });
            transform(B_col.begin(), B_col.end(), B_col.begin(), [&](auto x) { return x + 1; });
        }
    }

    typedef struct
    {
        const char           *testname;
        aoclsparse_operation  op;
        bool                  isTransposed;
        bool                  isAZeroNNZ;
        bool                  isBZeroNNZ;
        aoclsparse_index_base base_A;
        aoclsparse_index_base base_B;
        aoclsparse_int        M;
        aoclsparse_int        N;
        aoclsparse_int        nnz;
    } AddCSRParam;

    double                    A_val_double[]          = {1, 3, 2, 5};
    float                     A_val_float[]           = {1, 3, 2, 5};
    aoclsparse_float_complex  A_val_complexf[]        = {{1, 1}, {3, 1}, {4, 2}, {1, 5}};
    aoclsparse_double_complex A_val_complexd[]        = {{1, 1}, {3, 1}, {4, 2}, {1, 5}};
    double                    B_val_double[]          = {4, 9, 3, 5, 1};
    float                     B_val_float[]           = {4, 9, 3, 5, 1};
    aoclsparse_float_complex  B_val_complexf[]        = {{4, 2}, {9, 3}, {1, 3}, {2, 5}, {4, 1}};
    aoclsparse_double_complex B_val_complexd[]        = {{4, 2}, {9, 3}, {1, 3}, {2, 5}, {4, 1}};
    double                    C_val_double[]          = {5, 12, 5, 5, 6};
    float                     C_val_float[]           = {5, 12, 5, 5, 6};
    aoclsparse_float_complex  C_0nnz_BMat_complexf[]  = {{0, 2}, {2, 4}, {2, 6}, {-4, 6}};
    aoclsparse_double_complex C_0nnz_BMat_complexd[]  = {{0, 2}, {2, 4}, {2, 6}, {-4, 6}};
    aoclsparse_float_complex  CConj_0nnz_BMat_compf[] = {{2, 0}, {4, 2}, {6, 2}, {6, -4}};
    aoclsparse_double_complex CConj_0nnz_BMat_compd[] = {{2, 0}, {4, 2}, {6, 2}, {6, -4}};
    aoclsparse_float_complex  C_val_complexf[]        = {{4, 4}, {11, 7}, {3, 9}, {2, 5}, {0, 7}};
    aoclsparse_double_complex C_val_complexd[]        = {{4, 4}, {11, 7}, {3, 9}, {2, 5}, {0, 7}};
    aoclsparse_float_complex  CConj_val_complexf[]    = {{6, 2}, {13, 5}, {7, 5}, {2, 5}, {10, -3}};
    aoclsparse_double_complex CConj_val_complexd[]    = {{6, 2}, {13, 5}, {7, 5}, {2, 5}, {10, -3}};

    template <typename T>
    void test_csr_add(aoclsparse_operation  op,
                      bool                  isTransposed,
                      bool                  isAZeroNNZ,
                      bool                  isBZeroNNZ,
                      aoclsparse_index_base base_A,
                      aoclsparse_index_base base_B,
                      aoclsparse_int        M,
                      aoclsparse_int        N,
                      aoclsparse_int        NNZ,
                      T                    *Val,
                      T                    *Val_B,
                      T                    *Val_C,
                      T                     alpha)
    {
        aoclsparse_matrix      src_mat_A = nullptr, src_mat_B = nullptr, dest_mat = nullptr;
        vector<aoclsparse_int> A_row, A_col, B_row, B_col;
        startup(A_row, A_col, B_row, B_col, isTransposed, base_A, base_B, isAZeroNNZ, isBZeroNNZ);
        int A_nnz = 0, B_nnz = 0;
        if(!isAZeroNNZ)
            A_nnz = 4;
        if(!isBZeroNNZ)
            B_nnz = 5;

        if(isTransposed)

            EXPECT_EQ(aoclsparse_create_csr(
                          &src_mat_A, base_A, N, M, A_nnz, A_row.data(), A_col.data(), Val),
                      aoclsparse_status_success);
        else
            EXPECT_EQ(aoclsparse_create_csr(
                          &src_mat_A, base_A, M, N, A_nnz, A_row.data(), A_col.data(), Val),
                      aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(
                      &src_mat_B, base_B, M, N, B_nnz, B_row.data(), B_col.data(), Val_B),
                  aoclsparse_status_success);

        aoclsparse_int       *row_ptr = nullptr;
        aoclsparse_int       *col_ind = nullptr;
        T                    *val     = nullptr;
        aoclsparse_int        n, m, nnz;
        aoclsparse_index_base b;

        EXPECT_EQ(aoclsparse_add(op, src_mat_A, alpha, src_mat_B, &dest_mat),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_export_csr(dest_mat, &b, &m, &n, &nnz, &row_ptr, &col_ind, &val),
                  aoclsparse_status_success);
        EXPECT_EQ(b, base_A);
        EXPECT_EQ(nnz, NNZ);
        if(base_A != base_B)
        {
            if(base_A)
            {
                transform(B_row.begin(), B_row.end(), B_row.begin(), [](auto x) { return x + 1; });
                transform(B_col.begin(), B_col.end(), B_col.begin(), [](auto x) { return x + 1; });
            }
            else
            {
                transform(B_row.begin(), B_row.end(), B_row.begin(), [](auto x) { return x - 1; });
                transform(B_col.begin(), B_col.end(), B_col.begin(), [](auto x) { return x - 1; });
            }
        }
        if(!isBZeroNNZ)
        {
            EXPECT_EQ_VEC(M + 1, row_ptr, B_row);
            EXPECT_EQ_VEC(NNZ, col_ind, B_col);
        }
        else
        {
            if(!isTransposed)
            {
                EXPECT_EQ_VEC(M + 1, row_ptr, A_row);
                EXPECT_EQ_VEC(NNZ, col_ind, A_col);
            }
            else
            {
                startup(A_row,
                        A_col,
                        B_row,
                        B_col,
                        false,
                        base_A,
                        base_B,
                        isAZeroNNZ,
                        isBZeroNNZ,
                        true);
                EXPECT_EQ_VEC(M + 1, row_ptr, A_row);
                EXPECT_EQ_VEC(NNZ, col_ind, A_col);
            }
        }
        EXPECT_TRUE(array_match(Val_C, val, NNZ));

        if(isTransposed)
        {
            EXPECT_EQ(m, M);
            EXPECT_EQ(n, N);
        }
        else
        {
            EXPECT_EQ(m, M);
            EXPECT_EQ(n, N);
        }
        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat_A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat_B), aoclsparse_status_success);
    }

    template <typename T>
    void test_csr_add_complex(aoclsparse_operation  op,
                              bool                  isTransposed,
                              bool                  isAZeroNNZ,
                              bool                  isBZeroNNZ,
                              aoclsparse_index_base base_A,
                              aoclsparse_index_base base_B,
                              aoclsparse_int        M,
                              aoclsparse_int        N,
                              aoclsparse_int        NNZ,
                              T                    *Val,
                              T                    *Val_B,
                              T                    *Val_C,
                              T                     alpha)
    {
        aoclsparse_matrix      src_mat_A = nullptr, src_mat_B = nullptr, dest_mat = nullptr;
        vector<aoclsparse_int> A_row, A_col, B_row, B_col;
        startup(A_row, A_col, B_row, B_col, isTransposed, base_A, base_B, isAZeroNNZ, isBZeroNNZ);
        int A_nnz = 0, B_nnz = 0;
        if(!isAZeroNNZ)
            A_nnz = 4;
        if(!isBZeroNNZ)
            B_nnz = 5;

        if(isTransposed)
            EXPECT_EQ(aoclsparse_create_csr(
                          &src_mat_A, base_A, N, M, A_nnz, A_row.data(), A_col.data(), Val),
                      aoclsparse_status_success);
        else
            EXPECT_EQ(aoclsparse_create_csr(
                          &src_mat_A, base_A, M, N, A_nnz, A_row.data(), A_col.data(), Val),
                      aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(
                      &src_mat_B, base_B, M, N, B_nnz, B_row.data(), B_col.data(), Val_B),
                  aoclsparse_status_success);

        aoclsparse_int       *row_ptr = nullptr;
        aoclsparse_int       *col_ind = nullptr;
        T                    *val     = nullptr;
        aoclsparse_int        n, m, nnz;
        aoclsparse_index_base b;

        EXPECT_EQ(aoclsparse_add(op, src_mat_A, alpha, src_mat_B, &dest_mat),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_export_csr(dest_mat, &b, &m, &n, &nnz, &row_ptr, &col_ind, &val),
                  aoclsparse_status_success);
        EXPECT_EQ(b, base_A);
        EXPECT_EQ(nnz, NNZ);
        if(base_A != base_B)
        {
            if(base_A)
            {
                transform(B_row.begin(), B_row.end(), B_row.begin(), [](auto x) { return x + 1; });
                transform(B_col.begin(), B_col.end(), B_col.begin(), [](auto x) { return x + 1; });
            }
            else
            {
                transform(B_row.begin(), B_row.end(), B_row.begin(), [](auto x) { return x - 1; });
                transform(B_col.begin(), B_col.end(), B_col.begin(), [](auto x) { return x - 1; });
            }
        }
        if(!isBZeroNNZ)
        {
            EXPECT_EQ_VEC(M + 1, row_ptr, B_row);
            EXPECT_EQ_VEC(NNZ, col_ind, B_col);
        }
        else
        {
            if(!isTransposed)
            {
                EXPECT_EQ_VEC(M + 1, row_ptr, A_row);
                EXPECT_EQ_VEC(NNZ, col_ind, A_col);
            }
            else
            {
                startup(A_row,
                        A_col,
                        B_row,
                        B_col,
                        false,
                        base_A,
                        base_B,
                        isAZeroNNZ,
                        isBZeroNNZ,
                        true);
                EXPECT_EQ_VEC(M + 1, row_ptr, A_row);
                EXPECT_EQ_VEC(NNZ, col_ind, A_col);
            }
        }
        EXPECT_TRUE(
            array_complex_match(Val_C, val, NNZ, op == aoclsparse_operation_conjugate_transpose));

        if(isTransposed)
        {
            EXPECT_EQ(m, M);
            EXPECT_EQ(n, N);
        }
        else
        {
            EXPECT_EQ(m, M);
            EXPECT_EQ(n, N);
        }
        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat_A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat_B), aoclsparse_status_success);
    }

    // List of all desired negative tests
    const AddCSRParam AddCSRValues[] = {{"CSR_0B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         false,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSR_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         false,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSR_BMat_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         false,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSR_1B_BMat_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         false,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSR_0nnz_0B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         true,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSR_0nnz_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         true,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSR_0nnz_0B_BMat_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         true,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSR_0nnz_1B_BMat_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         true,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSRT_0B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         false,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSRT_1B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         false,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSRT_0B_BMat_1B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         false,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSRT_1B_BMat_1B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         false,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSRT_0nnz_0B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         true,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSRT_0nnz_1B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         true,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSRT_0nnz_0B_BMat_1B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         true,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSRT_0nnz_1B_BMat_1B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         true,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSRCT_0B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         false,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSRCT_1B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         false,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSRCT_BMat_1B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         false,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSRCT_1B_BMat_1B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         false,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSRCT_0nnz_0B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         true,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSRCT_0nnz_1B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         true,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         5},
                                        {"CSRCT_0nnz_0B_BMat_1B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         true,
                                         false,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSRCT_0nnz_1B_BMat_1B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         true,
                                         false,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         5},
                                        {"CSR_0nnz_Bmat_0B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         false,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         4},
                                        {"CSR_0nnz_Bmat_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         false,
                                         true,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         4},
                                        {"CSR_0nnz_BMat_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         false,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         4},
                                        {"CSR_0nnz_1B_BMat_1B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         false,
                                         true,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         4},
                                        {"CSR_0nnz_Bmat_0nnz_0B_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         true,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         0},
                                        {"CSR_0nnz_1B_Bmat_0nnz_ADD",
                                         aoclsparse_operation_none,
                                         false,
                                         true,
                                         true,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         0},
                                        {"CSRT_0nnz_1B_Bmat_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         false,
                                         true,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         4},
                                        {"CSRT_0nnz_Bmat_0B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         false,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         4},
                                        {"CSRT_0nnz_Bmat_1B_BMat_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         false,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         4},
                                        {"CSRT_0nnz_Bmat_0nnz_0B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         true,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         0},
                                        {"CSRT_0nnz_Bmat_0nnz_1B_ADD",
                                         aoclsparse_operation_transpose,
                                         true,
                                         true,
                                         true,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         0},
                                        {"CSRCT_0nnz_Bmat_1B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         false,
                                         true,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         4},
                                        {"CSRCT_0nnz_Bmat_0B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         false,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         4},
                                        {"CSRCT_0nnz_Bmat_1B_BMat_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         false,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_one,
                                         3,
                                         4,
                                         4},
                                        {"CSRCT_0nnz_Bmat_0nnz_0B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         true,
                                         true,
                                         aoclsparse_index_base_zero,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         0},
                                        {"CSRCT_0nnz_Bmat_0nnz_1B_ADD",
                                         aoclsparse_operation_conjugate_transpose,
                                         true,
                                         true,
                                         true,
                                         aoclsparse_index_base_one,
                                         aoclsparse_index_base_zero,
                                         3,
                                         4,
                                         0}};

    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const AddCSRParam &param, std::ostream *os)
    {
        *os << param.testname;
    }

    class AddCSRSuite : public testing::TestWithParam<AddCSRParam>
    {
    };

    // tests with double type
    TEST_P(AddCSRSuite, Double)
    {
        const AddCSRParam &param = GetParam();
        double            *c_val = nullptr;
        if(!param.isAZeroNNZ)
        {
            if(param.isBZeroNNZ)
            {
                c_val = A_val_double;
            }
            else
            {
                c_val = C_val_double;
            }
        }
        else
        {
            if(!param.isBZeroNNZ)
                c_val = B_val_double;
        }
        test_csr_add(param.op,
                     param.isTransposed,
                     param.isAZeroNNZ,
                     param.isBZeroNNZ,
                     param.base_A,
                     param.base_B,
                     param.M,
                     param.N,
                     param.nnz,
                     A_val_double,
                     B_val_double,
                     c_val,
                     1.0);
    }

    // tests with float type
    TEST_P(AddCSRSuite, Float)
    {
        const AddCSRParam &param = GetParam();
        float             *c_val = nullptr;
        if(!param.isAZeroNNZ)
        {
            if(param.isBZeroNNZ)
            {
                c_val = A_val_float;
            }
            else
            {
                c_val = C_val_float;
            }
        }
        else
        {
            if(!param.isBZeroNNZ)
                c_val = B_val_float;
        }
        test_csr_add(param.op,
                     param.isTransposed,
                     param.isAZeroNNZ,
                     param.isBZeroNNZ,
                     param.base_A,
                     param.base_B,
                     param.M,
                     param.N,
                     param.nnz,
                     A_val_float,
                     B_val_float,
                     c_val,
                     1.0f);
    }

    // tests with double type
    TEST_P(AddCSRSuite, ComplexDouble)
    {
        const AddCSRParam         &param = GetParam();
        aoclsparse_double_complex *c_val = nullptr;
        if(!param.isAZeroNNZ)
        {
            if(param.isBZeroNNZ)
            {
                if(param.op == aoclsparse_operation_conjugate_transpose)
                    c_val = CConj_0nnz_BMat_compd;
                else
                    c_val = C_0nnz_BMat_complexd;
            }
            else
            {

                if(param.op == aoclsparse_operation_conjugate_transpose)
                    c_val = CConj_val_complexd;
                else
                    c_val = C_val_complexd;
            }
        }
        else
        {
            if(!param.isBZeroNNZ)
                c_val = B_val_complexd;
        }
        test_csr_add_complex(param.op,
                             param.isTransposed,
                             param.isAZeroNNZ,
                             param.isBZeroNNZ,
                             param.base_A,
                             param.base_B,
                             param.M,
                             param.N,
                             param.nnz,
                             A_val_complexd,
                             B_val_complexd,
                             c_val,
                             {1.0, 1.0});
    }

    // tests with float type
    TEST_P(AddCSRSuite, ComplexFloat)
    {
        const AddCSRParam        &param = GetParam();
        aoclsparse_float_complex *c_val = nullptr;
        if(!param.isAZeroNNZ)
        {
            if(param.isBZeroNNZ)
            {
                if(param.op == aoclsparse_operation_conjugate_transpose)
                    c_val = CConj_0nnz_BMat_compf;
                else
                    c_val = C_0nnz_BMat_complexf;
            }
            else
            {

                if(param.op == aoclsparse_operation_conjugate_transpose)
                    c_val = CConj_val_complexf;
                else
                    c_val = C_val_complexf;
            }
        }
        else
        {
            if(!param.isBZeroNNZ)
                c_val = B_val_complexf;
        }
        test_csr_add_complex(param.op,
                             param.isTransposed,
                             param.isAZeroNNZ,
                             param.isBZeroNNZ,
                             param.base_A,
                             param.base_B,
                             param.M,
                             param.N,
                             param.nnz,
                             A_val_complexf,
                             B_val_complexf,
                             c_val,
                             {1.0f, 1.0f});
    }

    INSTANTIATE_TEST_SUITE_P(ADDTestSuite, AddCSRSuite, testing::ValuesIn(AddCSRValues));

    TEST(AddValidationSuite, MatrixDimensionTest)
    {
        aoclsparse_matrix      A = nullptr, B = nullptr, C = nullptr;
        vector<aoclsparse_int> row_ptr(6, 0), col_ptr(1, 0);
        aoclsparse_int         m = 5, n = 4, nnz = 0;
        aoclsparse_index_base  base = aoclsparse_index_base_zero;
        float                 *val  = new float;
        EXPECT_EQ(aoclsparse_create_csr(&A, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(&B, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_transpose, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_conjugate_transpose, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&B), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&C), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(&A, base, n, m, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr(&B, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_none, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&B), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&C), aoclsparse_status_success);
        delete val;
    }

    TEST(AddValidationSuite, NonInitMatrix)
    {
        aoclsparse_matrix      A = nullptr, B = nullptr, C = nullptr;
        vector<aoclsparse_int> row_ptr(6, 0), col_ptr(1, 0);
        aoclsparse_int         m = 5, n = 4, nnz = 0;
        aoclsparse_index_base  base = aoclsparse_index_base_zero;
        float                 *val  = new float;
        EXPECT_EQ(aoclsparse_create_csr(&A, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_none, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        A = nullptr;
        EXPECT_EQ(aoclsparse_create_csr(&B, base, m, n, nnz, row_ptr.data(), col_ptr.data(), val),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_add(aoclsparse_operation_transpose, A, 0.1f, B, &C),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_destroy(&B), aoclsparse_status_success);
        delete val;
    }

} // namespacecd
