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
#include "aoclsparse_types.h"
#include "gtest/gtest.h"
#include "aoclsparse_interface.hpp"

namespace
{

    template <typename T>
    void test_create_coo(aoclsparse_status     status_exp,
                         aoclsparse_index_base base,
                         aoclsparse_int        M,
                         aoclsparse_int        N,
                         aoclsparse_int        nnz,
                         aoclsparse_int       *row_ind,
                         aoclsparse_int       *col_ind,
                         T                    *val)
    {
        aoclsparse_matrix mat = NULL;
        aoclsparse_status status;

        EXPECT_EQ(status = aoclsparse_create_coo(&mat, base, M, N, nnz, row_ind, col_ind, val),
                  status_exp);

        if(status == aoclsparse_status_success)
        {
            EXPECT_EQ(aoclsparse_status_success, aoclsparse_destroy(&mat));
        }
    }

    typedef struct
    {
        const char                *testname;
        aoclsparse_status          status_exp;
        aoclsparse_index_base      base;
        aoclsparse_int             M;
        aoclsparse_int             N;
        aoclsparse_int             nnz;
        aoclsparse_int            *row_ptr;
        aoclsparse_int            *col_ptr;
        double                    *vald;
        float                     *valf;
        aoclsparse_double_complex *valcd;
        aoclsparse_float_complex  *valcf;
    } CreateCOOParamType;

    // test dataset
    aoclsparse_int col_ind[]     = {0, 1, 2, 3};
    aoclsparse_int row_ind[]     = {0, 3, 1, 3};
    aoclsparse_int col_ind_2x2[] = {0, 1};
    aoclsparse_int row_ind_2x2[] = {0, 0};
    double         val_double[]  = {1.000000000000, 6.000000000000, 8.000000000000, 4.000000000000};
    float          val_float[]   = {1.000000, 6.000000, 8.000000, 4.000000};
    aoclsparse_double_complex val_cdouble[4]{{.1, 1}, {.2, -2}, {.3, -3}, {.4, 4}};
    aoclsparse_float_complex  val_cfloat[4]{{.11, -1}, {.22, -2}, {.33, -3}, {.44, -4}};
    aoclsparse_int            row_ind_invalid_zerobase[]          = {0, 4, 1, 3};
    aoclsparse_int            row_ind_invalid_onebase[]           = {0, 4, 2, 4};
    aoclsparse_int            col_ind_invalid_zerobase[]          = {0, 1, 4, 3};
    aoclsparse_int            col_ind_invalid_onebase[]           = {0, 3, 1, 3};
    aoclsparse_int            row_invalid_negative_ind_zerobase[] = {-1, 34};
    aoclsparse_int            row_invalid_negative_ind_onebase[]  = {15, -100000000};
    aoclsparse_int            col_invalid_negative_ind_zerobase[] = {-1, 11};
    aoclsparse_int            col_invalid_negative_ind_onebase[]  = {2, -45436};

    // List of all desired negative tests
    const CreateCOOParamType CreateCOOErrorValues[] = {{"Error_InvalidRowSize",
                                                        aoclsparse_status_invalid_size,
                                                        aoclsparse_index_base_zero,
                                                        -1,
                                                        4,
                                                        4,
                                                        row_ind,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_InvalidColSize",
                                                        aoclsparse_status_invalid_size,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        -2,
                                                        4,
                                                        row_ind,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_InvalidNnz",
                                                        aoclsparse_status_invalid_size,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        -4,
                                                        row_ind,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_NullRowPtr",
                                                        aoclsparse_status_invalid_pointer,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        nullptr,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_NullColPtr",
                                                        aoclsparse_status_invalid_pointer,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        row_ind,
                                                        nullptr,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_NullValuePtr",
                                                        aoclsparse_status_invalid_pointer,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        row_ind,
                                                        col_ind,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr},
                                                       {"Error_0B_InvalidRowPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        row_ind_invalid_zerobase,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_1B_InvalidRowPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_one,
                                                        4,
                                                        4,
                                                        4,
                                                        row_ind_invalid_onebase,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_0B_InvalidColPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        row_ind,
                                                        col_ind_invalid_zerobase,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_1B_InvalidColPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_one,
                                                        4,
                                                        4,
                                                        4,
                                                        row_ind,
                                                        col_ind_invalid_onebase,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_1B_NegativeColPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_one,
                                                        10,
                                                        3,
                                                        2,
                                                        row_ind,
                                                        col_invalid_negative_ind_onebase,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_0B_NegativeColPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_one,
                                                        42,
                                                        12,
                                                        2,
                                                        row_ind,
                                                        col_invalid_negative_ind_zerobase,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_1B_NegativeRowPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_one,
                                                        16,
                                                        2,
                                                        2,
                                                        row_invalid_negative_ind_onebase,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_0B_NegativeRowPtrValue",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_one,
                                                        42,
                                                        4,
                                                        2,
                                                        row_invalid_negative_ind_zerobase,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Success_CreateCOOTest_4x4",
                                                        aoclsparse_status_success,
                                                        aoclsparse_index_base_zero,
                                                        4,
                                                        4,
                                                        4,
                                                        row_ind,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Success_CreateCOOTest_2x2",
                                                        aoclsparse_status_success,
                                                        aoclsparse_index_base_zero,
                                                        2,
                                                        2,
                                                        2,
                                                        row_ind_2x2,
                                                        col_ind_2x2,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Success_CreateCOOTest_1x1",
                                                        aoclsparse_status_success,
                                                        aoclsparse_index_base_zero,
                                                        1,
                                                        1,
                                                        1,
                                                        row_ind_2x2,
                                                        col_ind_2x2,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat},
                                                       {"Error_InvalidRowInd_2x4mat",
                                                        aoclsparse_status_invalid_index_value,
                                                        aoclsparse_index_base_zero,
                                                        2,
                                                        4,
                                                        2,
                                                        row_ind,
                                                        col_ind,
                                                        val_double,
                                                        val_float,
                                                        val_cdouble,
                                                        val_cfloat}};

    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const CreateCOOParamType &param, ::std::ostream *os)
    {
        *os << param.testname;
    }

    class CreateCOOTest : public testing::TestWithParam<CreateCOOParamType>
    {
    };

    // Error tests with double type
    TEST_P(CreateCOOTest, Double)
    {
        const CreateCOOParamType &param = GetParam();
        test_create_coo(param.status_exp,
                        param.base,
                        param.M,
                        param.N,
                        param.nnz,
                        param.row_ptr,
                        param.col_ptr,
                        param.vald);
    }

    // Error tests with float type
    TEST_P(CreateCOOTest, Float)
    {
        const CreateCOOParamType &param = GetParam();
        test_create_coo(param.status_exp,
                        param.base,
                        param.M,
                        param.N,
                        param.nnz,
                        param.row_ptr,
                        param.col_ptr,
                        param.valf);
    }

    // Error tests with double type
    TEST_P(CreateCOOTest, ComplexDouble)
    {
        const CreateCOOParamType &param = GetParam();
        test_create_coo(param.status_exp,
                        param.base,
                        param.M,
                        param.N,
                        param.nnz,
                        param.row_ptr,
                        param.col_ptr,
                        param.valcd);
    }

    // Error tests with float type
    TEST_P(CreateCOOTest, ComplexFloat)
    {
        const CreateCOOParamType &param = GetParam();
        test_create_coo(param.status_exp,
                        param.base,
                        param.M,
                        param.N,
                        param.nnz,
                        param.row_ptr,
                        param.col_ptr,
                        param.valcf);
    }

    INSTANTIATE_TEST_SUITE_P(CreateCOOSuite,
                             CreateCOOTest,
                             testing::ValuesIn(CreateCOOErrorValues));

} // namespace
