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
#include "aoclsparse_types.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"
#include "createcsc_ut_functions.hpp"

namespace
{
    typedef struct
    {
        const char           *testname;
        aoclsparse_status     status_exp;
        aoclsparse_index_base base;
        aoclsparse_int        M;
        aoclsparse_int        N;
        aoclsparse_int        nnz;
        aoclsparse_int       *col_ptr;
        aoclsparse_int       *row_idx;
        double               *vald;
        float                *valf;
    } CreateCSCParamType;

    // test dataset
    aoclsparse_int col_ptr[]         = {0, 1, 2, 3, 4};
    aoclsparse_int row_idx[]         = {0, 3, 1, 3};
    aoclsparse_int col_ptr_onebase[] = {1, 2, 3, 4, 5};
    double         val_double[] = {1.000000000000, 6.000000000000, 8.000000000000, 4.000000000000};
    float          val_float[]  = {1.000000, 6.000000, 8.000000, 4.000000};
    aoclsparse_int row_idx_invalid_zerobase[] = {0, 4, 1, 3};
    aoclsparse_int row_idx_invalid_onebase[]  = {0, 4, 2, 4};

    // List of all desired negative tests
    const CreateCSCParamType CreateCSCErrorValues[] = {{"InvalidRowSize",
                                                        aoclsparse_status_invalid_size,
                                                        aoclsparse_index_base_zero,
                                                        -1,
                                                        4,
                                                        4,
                                                        col_ptr,
                                                        row_idx,
                                                        val_double,
                                                        val_float},
                                                       {"InvalidColSize",
                                                        aoclsparse_status_invalid_size,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        -2,
                                                        4,
                                                        col_ptr,
                                                        row_idx,
                                                        val_double,
                                                        val_float},
                                                       {"InvalidNnz",
                                                        aoclsparse_status_invalid_size,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        -4,
                                                        col_ptr,
                                                        row_idx,
                                                        val_double,
                                                        val_float},
                                                       {"NullColPtr",
                                                        aoclsparse_status_invalid_pointer,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        nullptr,
                                                        row_idx,
                                                        val_double,
                                                        val_float},
                                                       {"NullRowPtr",
                                                        aoclsparse_status_invalid_pointer,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        col_ptr,
                                                        nullptr,
                                                        val_double,
                                                        val_float},
                                                       {"NullValuePtr",
                                                        aoclsparse_status_invalid_pointer,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        col_ptr,
                                                        row_idx,
                                                        nullptr,
                                                        nullptr},
                                                       {"ZeroBase_InvalidRowPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        col_ptr,
                                                        row_idx_invalid_zerobase,
                                                        val_double,
                                                        val_float},
                                                       {"OneBase_InvalidRowPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_one,
                                                        4,
                                                        4,
                                                        4,
                                                        col_ptr_onebase,
                                                        row_idx_invalid_onebase,
                                                        val_double,
                                                        val_float}};

    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const CreateCSCParamType &param, ::std::ostream *os)
    {
        *os << param.testname;
    }

    class CreateCSCTestErr : public testing::TestWithParam<CreateCSCParamType>
    {
    };

    // Error tests with double type
    TEST_P(CreateCSCTestErr, Double)
    {
        const CreateCSCParamType &param = GetParam();
        test_create_dcsc(param.status_exp,
                         param.base,
                         param.M,
                         param.N,
                         param.nnz,
                         param.col_ptr,
                         param.row_idx,
                         param.vald);
    }

    // Error tests with float type
    TEST_P(CreateCSCTestErr, Float)
    {
        const CreateCSCParamType &param = GetParam();
        test_create_scsc(param.status_exp,
                         param.base,
                         param.M,
                         param.N,
                         param.nnz,
                         param.col_ptr,
                         param.row_idx,
                         param.valf);
    }

    INSTANTIATE_TEST_SUITE_P(CreateCSCSuiteErr,
                             CreateCSCTestErr,
                             testing::ValuesIn(CreateCSCErrorValues));

    TEST(CreateCSCTest, CreateCSCTestDoubleSuccess)
    {
        test_create_dcsc(aoclsparse_status_success,
                         aoclsparse_index_base_zero,
                         4,
                         4,
                         4,
                         col_ptr,
                         row_idx,
                         val_double);
    }
    TEST(CreateCSCTest, CreateCSCTestFloatSuccess)
    {
        test_create_scsc(aoclsparse_status_success,
                         aoclsparse_index_base_zero,
                         4,
                         4,
                         4,
                         col_ptr,
                         row_idx,
                         val_float);
    }
    TEST(CreateCSCTest, CreateCSCTestComplexDoubleSuccess)
    {
        aoclsparse_matrix         mat      = NULL;
        aoclsparse_double_complex val_cd[] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};

        EXPECT_EQ(aoclsparse_createcsc(
                      mat, aoclsparse_index_base_zero, 4, 4, 4, col_ptr, row_idx, val_cd),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_status_success, aoclsparse_destroy(mat));
    }
    TEST(CreateCSCTest, CreateCSCTestComplexFloatSuccess)
    {
        aoclsparse_matrix        mat      = NULL;
        aoclsparse_float_complex val_cf[] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
        EXPECT_EQ(aoclsparse_createcsc(
                      mat, aoclsparse_index_base_zero, 4, 4, 4, col_ptr, row_idx, val_cf),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_status_success, aoclsparse_destroy(mat));
    }
} // namespace
