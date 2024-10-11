
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
#include "aoclsparse_auxiliary.h"
#include "aoclsparse_types.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"
#include "aoclsparse_datatype2string.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_random.hpp"

#include <algorithm>
#include <numeric>
#include <vector>
namespace
{
    typedef struct
    {
        const char                   *testname;
        aoclsparse_index_base         base;
        aoclsparse_int                M;
        aoclsparse_int                N;
        aoclsparse_int                nnz;
        aoclsparse_matrix_format_type format_type;
    } SetValueParam;

    template <typename T>
    void test_set_value(aoclsparse_index_base         base,
                        aoclsparse_int                M,
                        aoclsparse_int                N,
                        aoclsparse_int                NNZ,
                        aoclsparse_matrix_format_type format_type)
    {
        aoclsparse_matrix src_mat = nullptr;
        aoclsparse_seedrand();
        std::vector<aoclsparse_int> coo_row, coo_col, ptr;
        std::vector<T>              val;

        EXPECT_EQ(aoclsparse_init_matrix_random(
                      base, M, N, NNZ, format_type, coo_row, coo_col, val, ptr, src_mat),
                  aoclsparse_status_success);

        // random index to be used to set new value
        aoclsparse_int ridx = random_generator<aoclsparse_int>(0, NNZ - 1);

        // getting values of coordiante to be changed using random index
        aoclsparse_int check_row = coo_row[ridx], check_col = coo_col[ridx];
        T              check_val = random_generator_normal<T>();

        EXPECT_EQ(aoclsparse_set_value(src_mat, check_row, check_col, check_val),
                  aoclsparse_status_success);

        // here check is done on val variable as we have passed
        // this array pointer to the matrix so change will be reflected here also
        if constexpr(std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, std::complex<double>>)
        {
            EXPECT_EQ(std::real(val[ridx]), std::real(check_val));
            EXPECT_EQ(std::imag(val[ridx]), std::imag(check_val));
        }
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            EXPECT_EQ(val[ridx], check_val);
        }

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

    // List of all desired positive tests
    const SetValueParam SetValuesCases[]
        = {{"set_coo_0B", aoclsparse_index_base_zero, 10, 11, 20, aoclsparse_coo_mat},
           {"set_csc_0B", aoclsparse_index_base_zero, 5, 22, 1, aoclsparse_csc_mat},
           {"set_csr_0B", aoclsparse_index_base_zero, 1, 1, 1, aoclsparse_csr_mat},
           {"set_coo_1B", aoclsparse_index_base_one, 2, 1, 1, aoclsparse_coo_mat},
           {"set_csc_1B", aoclsparse_index_base_one, 3, 4, 11, aoclsparse_csc_mat},
           {"set_csr_1B", aoclsparse_index_base_one, 12, 5, 40, aoclsparse_csr_mat}};

    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const SetValueParam &param, std::ostream *os)
    {
        *os << param.testname;
    }

    class RndOK : public testing::TestWithParam<SetValueParam>
    {
    };

    // tests with double type
    TEST_P(RndOK, Double)
    {
        const SetValueParam &param = GetParam();
        test_set_value<double>(param.base, param.M, param.N, param.nnz, param.format_type);
    }

    // tests with float type
    TEST_P(RndOK, Float)
    {
        const SetValueParam &param = GetParam();
        test_set_value<float>(param.base, param.M, param.N, param.nnz, param.format_type);
    }

    // tests with double type
    TEST_P(RndOK, ComplexDouble)
    {
        const SetValueParam &param = GetParam();
        test_set_value<aoclsparse_double_complex>(
            param.base, param.M, param.N, param.nnz, param.format_type);
    }

    // tests with float type
    TEST_P(RndOK, ComplexFloat)
    {
        const SetValueParam &param = GetParam();
        test_set_value<aoclsparse_float_complex>(
            param.base, param.M, param.N, param.nnz, param.format_type);
    }

    INSTANTIATE_TEST_SUITE_P(SetValueTestSuite, RndOK, testing::ValuesIn(SetValuesCases));

    TEST(SetValueErrorSuite, UinitializedMatrix)
    {
        aoclsparse_matrix src_mat = nullptr;
        EXPECT_EQ(aoclsparse_set_value(src_mat, 1, 1, 1.0f), aoclsparse_status_invalid_pointer);
    }

    TEST(SetValueErrorSuite, InvalidValueMatrix0B)
    {
        aoclsparse_matrix src_mat = nullptr;

        std::vector<aoclsparse_int> row_ptr(1, 0), col_ptr(1, 0);
        std::vector<float>          val(1, 1.0f);

        aoclsparse_create_scoo(&src_mat,
                               aoclsparse_index_base_zero,
                               1,
                               7,
                               1,
                               row_ptr.data(),
                               col_ptr.data(),
                               val.data());

        EXPECT_EQ(aoclsparse_set_value(src_mat, 0, 7, 2.0f), aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_set_value(src_mat, 1, 1, 2.0f), aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

    TEST(SetValueErrorSuite, InvalidValueMatrix1B)
    {
        aoclsparse_seedrand();
        aoclsparse_matrix           src_mat = nullptr;
        aoclsparse_int              M = 4, N = 3, NNZ = 10;
        std::vector<aoclsparse_int> coo_row, coo_col, ptr;
        std::vector<double>         val;

        EXPECT_EQ(aoclsparse_init_matrix_random<double>(aoclsparse_index_base_one,
                                                        M,
                                                        N,
                                                        NNZ,
                                                        aoclsparse_csr_mat,
                                                        coo_row,
                                                        coo_col,
                                                        val,
                                                        ptr,
                                                        src_mat),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_set_value(src_mat, 0, 2, 3.14), aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_set_value(src_mat, 2, 0, 3.14), aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_set_value(src_mat, 5, 2, 3.14), aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_set_value(src_mat, 2, 4, 3.14), aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

    TEST(SetValueErrorSuite, InvalidValueMatrixEdgeCase)
    {
        aoclsparse_matrix src_mat = nullptr;

        std::vector<aoclsparse_int> row_ptr(1, 0), col_ptr(1, 0);
        std::vector<float>          val(1, 1.0f);

        aoclsparse_create_scoo(&src_mat,
                               aoclsparse_index_base_zero,
                               1,
                               7,
                               1,
                               row_ptr.data(),
                               col_ptr.data(),
                               val.data());

        EXPECT_EQ(aoclsparse_set_value(src_mat, 1, 0, 2.0f), aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_set_value(src_mat, 1, 1, 2.0f), aoclsparse_status_invalid_value);
        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

    TEST(SetValueErrorSuite, InvalidIndexMatrixCOO)
    {
        aoclsparse_matrix src_mat = nullptr;

        std::vector<aoclsparse_int> row_ptr(1, 0), col_ptr(1, 0);
        std::vector<float>          val(1, 1.0f);

        aoclsparse_create_scoo(&src_mat,
                               aoclsparse_index_base_zero,
                               1,
                               7,
                               1,
                               row_ptr.data(),
                               col_ptr.data(),
                               val.data());

        EXPECT_EQ(aoclsparse_set_value(src_mat, 0, 2, 2.0f), aoclsparse_status_invalid_index_value);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

    TEST(SetValueErrorSuite, InvalidIndexMatrixCSR)
    {
        aoclsparse_matrix src_mat = nullptr;

        aoclsparse_int row_ptr[] = {1, 2, 3, 5}, col_idx[] = {1, 3, 1, 4};
        double         val[] = {1.1, 2.3, 3.1, 3.4};

        aoclsparse_create_dcsr(&src_mat, aoclsparse_index_base_one, 3, 4, 4, row_ptr, col_idx, val);

        EXPECT_EQ(aoclsparse_set_value(src_mat, 1, 2, 2.0), aoclsparse_status_invalid_index_value);
        EXPECT_EQ(aoclsparse_set_value(src_mat, 3, 2, 2.0), aoclsparse_status_invalid_index_value);
        EXPECT_EQ(aoclsparse_set_value(src_mat, 3, 3, 2.0), aoclsparse_status_invalid_index_value);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

    TEST(SetValueErrorSuite, WrongTypeMatrix)
    {
        aoclsparse_matrix src_mat = nullptr;

        std::vector<aoclsparse_int> row_ptr(1, 0), col_ptr(1, 0);
        std::vector<float>          val(1, 1.0f);

        aoclsparse_create_scoo(&src_mat,
                               aoclsparse_index_base_zero,
                               1,
                               7,
                               1,
                               row_ptr.data(),
                               col_ptr.data(),
                               val.data());

        EXPECT_EQ(aoclsparse_set_value(src_mat, 0, 0, 2.0), aoclsparse_status_wrong_type);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }

#if 0
    // TODO : This needs to be activated when we can test with different matrix
    TEST(SetValueErrorSuite, NotImplementedMatrix)
    {
        aoclsparse_matrix src_mat = nullptr;

        std::vector<aoclsparse_int> row_ptr(1, 0), col_ptr(1, 0);
        std::vector<float>          val(1, 1.0f);

        aoclsparse_create_scoo(&src_mat,
                               aoclsparse_index_base_zero,
                               1,
                               7,
                               1,
                               row_ptr.data(),
                               col_ptr.data(),
                               val.data());
        src_mat->input_format = aoclsparse_ell_mat;

        EXPECT_EQ(aoclsparse_set_value(src_mat, 0, 2, 2.0f), aoclsparse_status_not_implemented);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
#endif
} // namespace
