
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
    } UpdateValuesParam;

    template <typename T>
    void test_update_values(aoclsparse_index_base         base,
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

        aoclsparse_int       *t_row_ptr = nullptr;
        aoclsparse_int       *t_col_ind = nullptr;
        T                    *t_new_val = nullptr;
        aoclsparse_int        n, m, nnz;
        aoclsparse_index_base b;
        // getting random new values for matrix update
        std::vector<T> newVal;
        // reserve at least one element so that newVal.data() is nonzero
        // and the test works even for NNZ=0 case
        newVal.reserve((std::max)((aoclsparse_int)1, NNZ));
        for(int i = 0; i < NNZ; i++)
            newVal.push_back(random_generator_normal<T>());

        EXPECT_EQ(aoclsparse_update_values(src_mat, NNZ, newVal.data()), aoclsparse_status_success);
        switch(format_type)
        {
        case aoclsparse_csr_mat:
            EXPECT_EQ(aoclsparse_export_csr(
                          src_mat, &b, &m, &n, &nnz, &t_row_ptr, &t_col_ind, &t_new_val),
                      aoclsparse_status_success);
            break;
        case aoclsparse_csc_mat:
            EXPECT_EQ(aoclsparse_export_csc(
                          src_mat, &b, &m, &n, &nnz, &t_row_ptr, &t_col_ind, &t_new_val),
                      aoclsparse_status_success);
            break;
        case aoclsparse_coo_mat:
            EXPECT_EQ(aoclsparse_export_coo(
                          src_mat, &b, &m, &n, &nnz, &t_row_ptr, &t_col_ind, &t_new_val),
                      aoclsparse_status_success);
            break;
        default:
            FAIL() << "Unsupported matrix format.";
            return;
        }

        // here check is done on val variable as we have passed
        // this array pointer to the matrix so change will be reflected here also
        if constexpr(std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, std::complex<double>>)
        {
            EXPECT_COMPLEX_EQ_VEC(NNZ, newVal, t_new_val);
        }
        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            EXPECT_COMPLEX_EQ_VEC(NNZ,
                                  reinterpret_cast<std::complex<float> *>(newVal.data()),
                                  reinterpret_cast<std::complex<float> *>(t_new_val));
        }
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            EXPECT_COMPLEX_EQ_VEC(NNZ,
                                  reinterpret_cast<std::complex<double> *>(newVal.data()),
                                  reinterpret_cast<std::complex<double> *>(t_new_val));
        }
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            EXPECT_EQ_VEC(NNZ, t_new_val, newVal);
        }
        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
    // List of all desired positive tests
    const UpdateValuesParam UpdateValuessCases[]
        = {{"update_coo_0B", aoclsparse_index_base_zero, 10, 11, 15, aoclsparse_coo_mat},
           {"update_csc_1B", aoclsparse_index_base_one, 4, 3, 1, aoclsparse_csc_mat},
           {"update_csr_0B", aoclsparse_index_base_zero, 3, 9, 27, aoclsparse_csr_mat},
           {"update_csc_0B", aoclsparse_index_base_zero, 7, 3, 3, aoclsparse_csc_mat}};
    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const UpdateValuesParam &param, std::ostream *os)
    {
        *os << param.testname;
    }
    class RndOK : public testing::TestWithParam<UpdateValuesParam>
    {
    };
    TEST_P(RndOK, Double)
    {
        const UpdateValuesParam &param = GetParam();
        test_update_values<double>(param.base, param.M, param.N, param.nnz, param.format_type);
    }
    TEST_P(RndOK, Float)
    {
        const UpdateValuesParam &param = GetParam();
        test_update_values<float>(param.base, param.M, param.N, param.nnz, param.format_type);
    }
    TEST_P(RndOK, ComplexDouble)
    {
        const UpdateValuesParam &param = GetParam();
        test_update_values<aoclsparse_double_complex>(
            param.base, param.M, param.N, param.nnz, param.format_type);
    }
    TEST_P(RndOK, ComplexFloat)
    {
        const UpdateValuesParam &param = GetParam();
        test_update_values<aoclsparse_float_complex>(
            param.base, param.M, param.N, param.nnz, param.format_type);
    }
    INSTANTIATE_TEST_SUITE_P(UpdateValuesTestSuite, RndOK, testing::ValuesIn(UpdateValuessCases));

    TEST(UpdateValuesErrorSuite, UinitializedMatrix)
    {
        aoclsparse_matrix src_mat = nullptr;
        float             newVal[1];
        EXPECT_EQ(aoclsparse_update_values(src_mat, 1, newVal), aoclsparse_status_invalid_pointer);
    }
    TEST(UpdateValuesErrorSuite, UinitializedVal)
    {
        aoclsparse_matrix src_mat = nullptr;

        aoclsparse_int row_ptr[] = {1, 2, 3, 5}, col_idx[] = {1, 3, 1, 4};
        double         val[]  = {1.1, 2.3, 3.1, 3.4};
        double        *newVal = nullptr;

        EXPECT_EQ(aoclsparse_create_dcsr(
                      &src_mat, aoclsparse_index_base_one, 3, 4, 4, row_ptr, col_idx, val),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_update_values(src_mat, 4, newVal), aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
    TEST(UpdateValuesErrorSuite, InvalidSizeVal)
    {
        aoclsparse_matrix src_mat   = nullptr;
        aoclsparse_int    row_ptr[] = {1, 2, 3, 5}, col_idx[] = {1, 3, 1, 4};
        double            val[]    = {1.1, 2.3, 3.1, 3.4};
        double            newVal[] = {1., 2., 3., 4., 5.};

        EXPECT_EQ(aoclsparse_create_dcsr(
                      &src_mat, aoclsparse_index_base_one, 3, 4, 4, row_ptr, col_idx, val),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_update_values(src_mat, 5, newVal), aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_update_values(src_mat, 3, newVal), aoclsparse_status_invalid_size);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
    TEST(UpdateValuesErrorSuite, WrongTypeMatrix)
    {
        aoclsparse_matrix        src_mat   = nullptr;
        aoclsparse_int           row_ptr[] = {2}, col_ptr[] = {4};
        aoclsparse_float_complex val[]    = {{3., 6.}};
        double                   newVal[] = {2.0};
        EXPECT_EQ(aoclsparse_create_ccoo(
                      &src_mat, aoclsparse_index_base_zero, 4, 5, 1, row_ptr, col_ptr, val),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_update_values(src_mat, 1, newVal), aoclsparse_status_wrong_type);

        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
#if 0
    // TODO : This needs to be activated when we can test with different matrix
    TEST(UpdateValuesErrorSuite, NotImplementedMatrix)
    {
        aoclsparse_matrix src_mat = nullptr;
        std::vector<aoclsparse_int> row_ptr(1, 0), col_ptr(1, 0);
        std::vector<float>          val(1, 1.0f);
        EXPECT_EQ(aoclsparse_create_scoo(&src_mat,
                               aoclsparse_index_base_zero,
                               1,
                               7,
                               1,
                               row_ptr.data(),
                               col_ptr.data(),
                               val.data()), aoclsparse_status_success);
        src_mat->input_format = aoclsparse_ell_mat;
        EXPECT_EQ(aoclsparse_update_values(src_mat, 0, 2.0f), aoclsparse_status_not_implemented);
        EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
    }
#endif
} // namespace
