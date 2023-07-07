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
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"

#include <algorithm>
#include <complex>
#include <vector>

namespace
{

    template <typename T>
    void init(aoclsparse_int              &m,
              aoclsparse_int              &n,
              aoclsparse_int              &nnz,
              std::vector<aoclsparse_int> &row_ptr,
              std::vector<aoclsparse_int> &col_idx,
              std::vector<T>              &val)
    {
        m = 5, n = 7, nnz = 8;
        row_ptr.assign({0, 2, 3, 4, 7, 8});
        col_idx.assign({0, 6, 1, 5, 1, 3, 4, 6});
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            val.assign({1, 2, 3, 4, 5, 6, 7, 8});
        }
        else if constexpr(std::is_same_v<T, std::complex<double>>
                          || std::is_same_v<T, std::complex<float>>
                          || std::is_same_v<T, aoclsparse_double_complex>
                          || std::is_same_v<T, aoclsparse_float_complex>)
        {
            val.assign(
                {{1, -2}, {-1, 0.2}, {0.3, 0}, {4, -5}, {1.5, 2}, {6, 0}, {1.7, -1}, {0.8, -1}});
        }
    }

    template <typename T>
    void test_success(void)
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_idx;
        std::vector<T>              val;
        aoclsparse_matrix           A;

        init(m, n, nnz, row_ptr, col_idx, val);

        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ_VEC(
            m + 1, (aoclsparse_int *)A->csr_mat.csr_row_ptr, (aoclsparse_int *)row_ptr.data());
        EXPECT_EQ_VEC(
            nnz, (aoclsparse_int *)A->csr_mat.csr_col_ptr, (aoclsparse_int *)col_idx.data());
        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            std::complex<float> *tmp1 = (std::complex<float> *)val.data();
            std::complex<float> *tmp2 = (std::complex<float> *)A->csr_mat.csr_val;
            EXPECT_COMPLEX_FLOAT_EQ_VEC(nnz, tmp1, tmp2);
        }
        else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            std::complex<double> *tmp1 = (std::complex<double> *)val.data();
            std::complex<double> *tmp2 = (std::complex<double> *)A->csr_mat.csr_val;
            EXPECT_COMPLEX_DOUBLE_EQ_VEC(nnz, tmp1, tmp2);
        }
        else
        {
            EXPECT_EQ_VEC(nnz, (T *)A->csr_mat.csr_val, (T *)val.data());
        }
        EXPECT_EQ(m, A->m);
        EXPECT_EQ(n, A->n);
        EXPECT_EQ(nnz, A->nnz);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    }

    template <typename T>
    void test_nullptr(void)
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_idx;
        std::vector<T>              val;
        aoclsparse_matrix           A;

        init(m, n, nnz, row_ptr, col_idx, val);

        EXPECT_EQ(aoclsparse_create_csr<T>(A, base, m, n, nnz, nullptr, col_idx.data(), val.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_create_csr<T>(A, base, m, n, nnz, row_ptr.data(), nullptr, val.data()),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_create_csr<T>(A, base, m, n, nnz, row_ptr.data(), col_idx.data(), nullptr),
            aoclsparse_status_invalid_pointer);
    }

    template <typename T>
    void test_invalid_input(void)
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_idx;
        std::vector<T>              val;
        aoclsparse_matrix           A;

        init(m, n, nnz, row_ptr, col_idx, val);

        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, -1, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_invalid_size);
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, -1, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_invalid_size);

        /*
        FIXME: Fails now, need to uncomment this test after other validation related commits are merged
        EXPECT_EQ(
            aoclsparse_create_csr<T>(A, base, m, n, 0, row_ptr.data(), col_idx.data(), val.data()),
            aoclsparse_status_invalid_value);
*/
        EXPECT_EQ(
            aoclsparse_create_csr<T>(A, base, m, n, -1, row_ptr.data(), col_idx.data(), val.data()),
            aoclsparse_status_invalid_size);
        // invalid column index for zero-based indexing
        col_idx[2] = 7; // should be between 0 and 6
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_invalid_index_value);

        base = aoclsparse_index_base_one;
        //rebuild indices for 1-based indexing and then test for invalid column index
        transform(row_ptr.begin(), row_ptr.end(), row_ptr.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        transform(col_idx.begin(), col_idx.end(), col_idx.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // invalid column index for one-based indexing
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_invalid_index_value);
    }

    template <typename T>
    void test_zero_dimension(void)
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_int              m;
        aoclsparse_int              n;
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_idx;
        std::vector<T>              val;
        aoclsparse_matrix           A;

        // m = 5, n = 7, nnz = 8
        init(m, n, nnz, row_ptr, col_idx, val);

        // 1) 5*7 , nnz=0
        nnz = 0;
        row_ptr.assign({0, 0, 0, 0, 0, 0});
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);

        // 2) 0*7 , nnz=0
        m = 0;
        n = 7;
        row_ptr.assign({0});
        col_idx.assign({0});
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);

        // 3) 2*0 , nnz=0
        m   = 2;
        n   = 0;
        nnz = 0;
        row_ptr.assign({0, 0, 0});
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);

        // 4) 0*0 , nnz=0
        m   = 0;
        n   = 0;
        nnz = 0;
        row_ptr.assign({0});
        col_idx.assign({0});
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);

        // 5) 2*0 , nnz=3
        m   = 2;
        n   = 0;
        nnz = 3;
        row_ptr.assign({0, 2, 3});
        col_idx.assign({0, 0, 0});
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      A, base, m, n, nnz, row_ptr.data(), col_idx.data(), val.data()),
                  aoclsparse_status_invalid_index_value);
    }

    TEST(createcsr, NullArgAll)
    {
        test_nullptr<float>();
        test_nullptr<double>();
        test_nullptr<aoclsparse_float_complex>();
        test_nullptr<aoclsparse_double_complex>();
    }
    TEST(createcsr, InvalidInputAll)
    {
        test_invalid_input<float>();
        //test_invalid_input<double>();
        //test_invalid_input<aoclsparse_float_complex>();
        //test_invalid_input<aoclsparse_double_complex>();
    }
    TEST(createcsr, SuccessAll)
    {
        test_success<float>();
        test_success<double>();
        test_success<aoclsparse_float_complex>();
        test_success<aoclsparse_double_complex>();
    }
    TEST(createcsr, ZeroDimensionMatrix)
    {
        test_zero_dimension<float>();
        test_zero_dimension<double>();
        test_zero_dimension<aoclsparse_float_complex>();
        test_zero_dimension<aoclsparse_double_complex>();
    }

} // namespace
