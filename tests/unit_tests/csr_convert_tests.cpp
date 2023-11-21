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
#include "createcsc_ut_functions.hpp"

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

    typedef struct
    {
        const char                *testname;
        aoclsparse_status          status_exp;
        aoclsparse_operation       op;
        bool                       isTransposed;
        bool                       isZeroNNZ;
        aoclsparse_index_base      base;
        aoclsparse_int             M;
        aoclsparse_int             N;
        aoclsparse_int             nnz;
        std::string                format_type;
        double                    *vald;
        float                     *valf;
        aoclsparse_double_complex *val_cd;
        aoclsparse_float_complex  *val_cf;
    } ConvertCSRParam;

    // TODO use when we have a way to validate transposed matrix
    aoclsparse_int csrt_row_ptr[]    = {0, 1, 2, 3};
    aoclsparse_int csrt_col_ind[]    = {0, 1, 3};
    aoclsparse_int csrt_1b_row_ptr[] = {1, 2, 3, 4};
    aoclsparse_int csrt_1b_col_ind[] = {1, 2, 4};

    // test dataset
    aoclsparse_int csr_row_ptr[]          = {0, 1, 2, 2, 3};
    aoclsparse_int csr_col_ind[]          = {0, 1, 2};
    aoclsparse_int csr_1b_row_ptr[]       = {1, 2, 3, 3, 4};
    aoclsparse_int csr_1b_col_ind[]       = {1, 2, 3};
    aoclsparse_int csr_0b_0nnz_row_ptr[]  = {0, 0, 0, 0, 0};
    aoclsparse_int csr_1b_0nnz_row_ptr[]  = {1, 1, 1, 1, 1};
    aoclsparse_int csrt_0b_0nnz_row_ptr[] = {0, 0, 0, 0};
    aoclsparse_int csrt_1b_0nnz_row_ptr[] = {1, 1, 1, 1};

    aoclsparse_int coo_row_ind[]    = {0, 1, 3};
    aoclsparse_int coo_col_ind[]    = {0, 1, 2};
    aoclsparse_int coo_1b_row_ind[] = {1, 2, 4};
    aoclsparse_int coo_1b_col_ind[] = {1, 2, 3};

    aoclsparse_int csc_row_ind[]    = {0, 1, 3};
    aoclsparse_int csc_col_ptr[]    = {0, 1, 2, 3};
    aoclsparse_int csc_1b_row_ind[] = {1, 2, 4};
    aoclsparse_int csc_1b_col_ptr[] = {1, 2, 3, 4};

    double                    val_double[]   = {1.000000000000, 6.000000000000, 8.000000000000};
    float                     val_float[]    = {1.000000, 6.000000, 8.000000};
    aoclsparse_float_complex  val_complexf[] = {{1, 2}, {2, 3}, {3, 4}};
    aoclsparse_double_complex val_complexd[] = {{1, 2}, {2, 3}, {3, 4}};

    template <typename T>
    void test_convert_csr(aoclsparse_status     status_exp,
                          aoclsparse_operation  op,
                          bool                  isTransposed,
                          bool                  isZeroNNZ,
                          aoclsparse_index_base base,
                          aoclsparse_int        M,
                          aoclsparse_int        N,
                          aoclsparse_int        NNZ,
                          std::string           format_type,
                          T                    *Val)
    {
        aoclsparse_matrix src_mat = nullptr, dest_mat;
        if(format_type == "coo")
        {
            if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ(
                    aoclsparse_create_coo(&src_mat, base, M, N, NNZ, coo_row_ind, coo_col_ind, Val),
                    aoclsparse_status_success);
            }
            else
            {
                EXPECT_EQ(aoclsparse_create_coo(
                              &src_mat, base, M, N, NNZ, coo_1b_row_ind, coo_1b_col_ind, Val),
                          aoclsparse_status_success);
            }
        }
        else if(format_type == "csc")
        {
            if(isZeroNNZ)
            {
                if(base == aoclsparse_index_base_zero)
                {
                    EXPECT_EQ(
                        aoclsparse_create_csc(
                            &src_mat, base, M, N, NNZ, csrt_0b_0nnz_row_ptr, csc_row_ind, Val),
                        aoclsparse_status_success);
                }
                else
                {
                    EXPECT_EQ(
                        aoclsparse_create_csc(
                            &src_mat, base, M, N, NNZ, csrt_1b_0nnz_row_ptr, csc_row_ind, Val),
                        aoclsparse_status_success);
                }
            }
            else if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ(
                    aoclsparse_create_csc(&src_mat, base, M, N, NNZ, csc_col_ptr, csc_row_ind, Val),
                    aoclsparse_status_success);
            }
            else
            {
                EXPECT_EQ(aoclsparse_create_csc(
                              &src_mat, base, M, N, NNZ, csc_1b_col_ptr, csc_1b_row_ind, Val),
                          aoclsparse_status_success);
            }
        }
        else
        {
            if(isZeroNNZ)
            {
                if(base == aoclsparse_index_base_zero)
                {
                    EXPECT_EQ(aoclsparse_create_csr(
                                  &src_mat, base, M, N, NNZ, csr_0b_0nnz_row_ptr, csr_col_ind, Val),
                              aoclsparse_status_success);
                }
                else
                {
                    EXPECT_EQ(aoclsparse_create_csr(
                                  &src_mat, base, M, N, NNZ, csr_1b_0nnz_row_ptr, csr_col_ind, Val),
                              aoclsparse_status_success);
                }
            }
            else if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ(
                    aoclsparse_create_csr(&src_mat, base, M, N, NNZ, csr_row_ptr, csr_col_ind, Val),
                    aoclsparse_status_success);
            }
            else
            {
                EXPECT_EQ(aoclsparse_create_csr(
                              &src_mat, base, M, N, NNZ, csr_1b_row_ptr, csr_1b_col_ind, Val),
                          aoclsparse_status_success);
            }
        }
        aoclsparse_int       *row_ptr = nullptr;
        aoclsparse_int       *col_ind = nullptr;
        T                    *val     = nullptr;
        aoclsparse_int        n, m, nnz;
        aoclsparse_index_base b;

        EXPECT_EQ(aoclsparse_convert_csr(src_mat, op, &dest_mat), status_exp);
        EXPECT_EQ(aoclsparse_export_csr(dest_mat, &b, &m, &n, &nnz, &row_ptr, &col_ind, &val),
                  status_exp);
        EXPECT_EQ(b, base);
        EXPECT_EQ(nnz, NNZ);
        if(isTransposed)
        {
            if(isZeroNNZ)
            {
                if(base == aoclsparse_index_base_zero)
                {
                    EXPECT_EQ_VEC(N + 1, row_ptr, csrt_0b_0nnz_row_ptr);
                }
                else
                {
                    EXPECT_EQ_VEC(N + 1, row_ptr, csrt_1b_0nnz_row_ptr);
                }
            }
            else if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ_VEC(N + 1, row_ptr, csrt_row_ptr);
                EXPECT_EQ_VEC(NNZ, col_ind, csrt_col_ind);
            }
            else
            {
                EXPECT_EQ_VEC(N + 1, row_ptr, csrt_1b_row_ptr);
                EXPECT_EQ_VEC(NNZ, col_ind, csrt_1b_col_ind);
            }
            EXPECT_EQ(m, N);
            EXPECT_EQ(n, M);
        }
        else
        {
            if(isZeroNNZ)
            {
                if(base == aoclsparse_index_base_zero)
                {
                    EXPECT_EQ_VEC(M + 1, row_ptr, csr_0b_0nnz_row_ptr);
                }
                else
                {
                    EXPECT_EQ_VEC(M + 1, row_ptr, csr_1b_0nnz_row_ptr);
                }
            }
            else if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ_VEC(M + 1, row_ptr, csr_row_ptr);
                EXPECT_EQ_VEC(NNZ, col_ind, csr_col_ind);
            }
            else
            {
                EXPECT_EQ_VEC(M + 1, row_ptr, csr_1b_row_ptr);
                EXPECT_EQ_VEC(NNZ, col_ind, csr_1b_col_ind);
            }
            EXPECT_EQ(m, M);
            EXPECT_EQ(n, N);
        }
        if(std::is_same_v<T, double>)
            EXPECT_DOUBLE_EQ_VEC(NNZ, Val, val);
        if(std::is_same_v<T, float>)
            EXPECT_FLOAT_EQ_VEC(NNZ, Val, val);
        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

    template <typename T>
    void test_convert_csr_complex(aoclsparse_status     status_exp,
                                  aoclsparse_operation  op,
                                  bool                  isTransposed,
                                  bool                  isZeroNNZ,
                                  aoclsparse_index_base base,
                                  aoclsparse_int        M,
                                  aoclsparse_int        N,
                                  aoclsparse_int        NNZ,
                                  std::string           format_type,
                                  T                    *Val)
    {
        aoclsparse_matrix src_mat = nullptr, dest_mat;
        if(format_type == "coo")
        {
            if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ(
                    aoclsparse_create_coo(&src_mat, base, M, N, NNZ, coo_row_ind, coo_col_ind, Val),
                    aoclsparse_status_success);
            }
            else
            {
                EXPECT_EQ(aoclsparse_create_coo(
                              &src_mat, base, M, N, NNZ, coo_1b_row_ind, coo_1b_col_ind, Val),
                          aoclsparse_status_success);
            }
        }
        else if(format_type == "csc")
        {
            if(isZeroNNZ)
            {
                if(base == aoclsparse_index_base_zero)
                {
                    EXPECT_EQ(
                        aoclsparse_create_csc(
                            &src_mat, base, M, N, NNZ, csrt_0b_0nnz_row_ptr, csc_row_ind, Val),
                        aoclsparse_status_success);
                }
                else
                {
                    EXPECT_EQ(
                        aoclsparse_create_csc(
                            &src_mat, base, M, N, NNZ, csrt_1b_0nnz_row_ptr, csc_row_ind, Val),
                        aoclsparse_status_success);
                }
            }
            else if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ(
                    aoclsparse_create_csc(&src_mat, base, M, N, NNZ, csc_col_ptr, csc_row_ind, Val),
                    aoclsparse_status_success);
            }
            else
            {
                EXPECT_EQ(aoclsparse_create_csc(
                              &src_mat, base, M, N, NNZ, csc_1b_col_ptr, csc_1b_row_ind, Val),
                          aoclsparse_status_success);
            }
        }
        else
        {
            if(isZeroNNZ)
            {
                if(base == aoclsparse_index_base_zero)
                {
                    EXPECT_EQ(aoclsparse_create_csr(
                                  &src_mat, base, M, N, NNZ, csr_0b_0nnz_row_ptr, csr_col_ind, Val),
                              aoclsparse_status_success);
                }
                else
                {
                    EXPECT_EQ(aoclsparse_create_csr(
                                  &src_mat, base, M, N, NNZ, csr_1b_0nnz_row_ptr, csr_col_ind, Val),
                              aoclsparse_status_success);
                }
            }
            else if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ(
                    aoclsparse_create_csr(&src_mat, base, M, N, NNZ, csr_row_ptr, csr_col_ind, Val),
                    aoclsparse_status_success);
            }
            else
            {
                EXPECT_EQ(aoclsparse_create_csr(
                              &src_mat, base, M, N, NNZ, csr_1b_row_ptr, csr_1b_col_ind, Val),
                          aoclsparse_status_success);
            }
        }
        aoclsparse_int       *row_ptr = nullptr;
        aoclsparse_int       *col_ind = nullptr;
        T                    *val     = nullptr;
        aoclsparse_int        n, m, nnz;
        aoclsparse_index_base b;

        EXPECT_EQ(aoclsparse_convert_csr(src_mat, op, &dest_mat), status_exp);
        EXPECT_EQ(aoclsparse_export_csr(dest_mat, &b, &m, &n, &nnz, &row_ptr, &col_ind, &val),
                  status_exp);
        EXPECT_EQ(b, base);
        EXPECT_EQ(nnz, NNZ);
        if(isTransposed)
        {
            if(isZeroNNZ)
            {
                if(base == aoclsparse_index_base_zero)
                {
                    EXPECT_EQ_VEC(N + 1, row_ptr, csrt_0b_0nnz_row_ptr);
                }
                else
                {
                    EXPECT_EQ_VEC(N + 1, row_ptr, csrt_1b_0nnz_row_ptr);
                }
            }
            else if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ_VEC(N + 1, row_ptr, csrt_row_ptr);
                EXPECT_EQ_VEC(NNZ, col_ind, csrt_col_ind);
            }
            else
            {
                EXPECT_EQ_VEC(N + 1, row_ptr, csrt_1b_row_ptr);
                EXPECT_EQ_VEC(NNZ, col_ind, csrt_1b_col_ind);
            }
            EXPECT_EQ(m, N);
            EXPECT_EQ(n, M);
        }
        else
        {
            if(isZeroNNZ)
            {
                if(base == aoclsparse_index_base_zero)
                {
                    EXPECT_EQ_VEC(M + 1, row_ptr, csr_0b_0nnz_row_ptr);
                }
                else
                {
                    EXPECT_EQ_VEC(M + 1, row_ptr, csr_1b_0nnz_row_ptr);
                }
            }
            else if(base == aoclsparse_index_base_zero)
            {
                EXPECT_EQ_VEC(M + 1, row_ptr, csr_row_ptr);
                EXPECT_EQ_VEC(NNZ, col_ind, csr_col_ind);
            }
            else
            {
                EXPECT_EQ_VEC(M + 1, row_ptr, csr_1b_row_ptr);
                EXPECT_EQ_VEC(NNZ, col_ind, csr_1b_col_ind);
            }
            EXPECT_EQ(m, M);
            EXPECT_EQ(n, N);
        }
        EXPECT_TRUE(
            array_complex_match(Val, val, NNZ, (op == aoclsparse_operation_conjugate_transpose)));
        EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

    // List of all desired negative tests
    const ConvertCSRParam ConvertCSRValues[] = {{"cooToCsr_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 true,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 0,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsr_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 true,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 0,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsr_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 true,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 0,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsr_1B_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 true,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 0,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsr_1B_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 true,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 0,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsr_1B_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 true,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 0,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsrT_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 true,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 0,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsrT_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 true,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 0,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsrT_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 true,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 0,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsrT_1B_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 true,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 0,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsrT_1B_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 true,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 0,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsrT_1B_0NNZ_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 true,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 0,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsrConvert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsr_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsr_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsrTransConvert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsrTransConvert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsrTransConvert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsrConjTransConvert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_conjugate_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsrConjTransConvert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_conjugate_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsrConjTransConvert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_conjugate_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_zero,
                                                 4,
                                                 3,
                                                 3,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsr_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsr_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsr_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_none,
                                                 false,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsrTrans_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsrTrans_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsrTrans_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cooToCsrConjTrans_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_conjugate_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "coo",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"cscToCsrConjTrans_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_conjugate_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "csc",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf},
                                                {"csrToCsrConjTrans_1B_Convert",
                                                 aoclsparse_status_success,
                                                 aoclsparse_operation_conjugate_transpose,
                                                 true,
                                                 false,
                                                 aoclsparse_index_base_one,
                                                 4,
                                                 3,
                                                 3,
                                                 "csr",
                                                 val_double,
                                                 val_float,
                                                 val_complexd,
                                                 val_complexf}};

    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const ConvertCSRParam &param, std::ostream *os)
    {
        *os << param.testname;
    }

    class ConvertCSRSuite : public testing::TestWithParam<ConvertCSRParam>
    {
    };

    // tests with double type
    TEST_P(ConvertCSRSuite, Double)
    {
        const ConvertCSRParam &param = GetParam();
        test_convert_csr(param.status_exp,
                         param.op,
                         param.isTransposed,
                         param.isZeroNNZ,
                         param.base,
                         param.M,
                         param.N,
                         param.nnz,
                         param.format_type,
                         param.vald);
    }

    // tests with float type
    TEST_P(ConvertCSRSuite, Float)
    {
        const ConvertCSRParam &param = GetParam();
        test_convert_csr(param.status_exp,
                         param.op,
                         param.isTransposed,
                         param.isZeroNNZ,
                         param.base,
                         param.M,
                         param.N,
                         param.nnz,
                         param.format_type,
                         param.valf);
    }

    // tests with double type
    TEST_P(ConvertCSRSuite, ComplexDouble)
    {
        const ConvertCSRParam &param = GetParam();
        test_convert_csr_complex(param.status_exp,
                                 param.op,
                                 param.isTransposed,
                                 param.isZeroNNZ,
                                 param.base,
                                 param.M,
                                 param.N,
                                 param.nnz,
                                 param.format_type,
                                 param.val_cd);
    }

    // tests with float type
    TEST_P(ConvertCSRSuite, ComplexFloat)
    {
        const ConvertCSRParam &param = GetParam();
        test_convert_csr_complex(param.status_exp,
                                 param.op,
                                 param.isTransposed,
                                 param.isZeroNNZ,
                                 param.base,
                                 param.M,
                                 param.N,
                                 param.nnz,
                                 param.format_type,
                                 param.val_cf);
    }

    INSTANTIATE_TEST_SUITE_P(ConvertCSRTestSuite,
                             ConvertCSRSuite,
                             testing::ValuesIn(ConvertCSRValues));

    TEST(ConvertCSRErrorSuite, UinitializedMatrix)
    {
        aoclsparse_matrix src_mat = nullptr, dest_mat = nullptr;
        EXPECT_EQ(aoclsparse_convert_csr(src_mat, aoclsparse_operation_none, &dest_mat),
                  aoclsparse_status_invalid_pointer);
    }

    TEST(ConvertCSRErrorSuite, NotImplementedMatrixFormat)
    {
        // TODO uncomment the checks once a format comes up which we will not support

        // aoclsparse_matrix src_mat, dest_mat;
        // aoclsparse_create_ell_csr_hyb(src_mat, 3, 2, ell_col_ind, ell_col_ind, val_float);
        // aoclsparse_create_csr(
        //     &dest_mat, aoclsparse_index_base_zero, 4, 3, 3, csr_row_ptr, csr_col_ind, val_float);

        // EXPECT_EQ(aoclsparse_convert_csr(src_mat, aoclsparse_operation_none, &dest_mat),
        //           aoclsparse_status_not_implemented);
        // EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
        // EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
} // namespace
