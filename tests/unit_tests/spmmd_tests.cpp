/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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
#include "aoclsparse_interface.hpp"

#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
    // tests for nullptr
    template <typename T>
    void spmmd_nullptr()
    {
        //        aoclsparse_int m, n, k, nnz_a, nnz_b;
        aoclsparse_matrix    A, B;
        aoclsparse_operation op     = aoclsparse_operation_none;
        aoclsparse_order     layout = aoclsparse_order_row;
        std::vector<T>       C;
        aoclsparse_int       ldc = 0;

        EXPECT_EQ((aoclsparse_spmmd<T>(op, nullptr, B, layout, C.data(), ldc, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_spmmd<T>(op, A, nullptr, layout, C.data(), ldc, -1)),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_spmmd<T>(op, A, B, layout, nullptr, ldc, -1)),
                  aoclsparse_status_invalid_pointer);
    }

    // tests a variety of cases w.r.t row-major ordering
    template <typename T>
    void spmmd_testall(aoclsparse_int ba,
                       aoclsparse_int bb,
                       aoclsparse_int row_maj,
                       aoclsparse_int ldc = -1)
    {
        aoclsparse_index_base base_a = aoclsparse_index_base_zero;
        aoclsparse_index_base base_b = aoclsparse_index_base_zero;
        aoclsparse_order      layout = aoclsparse_order_row;
        aoclsparse_matrix     A;
        aoclsparse_matrix     B;
        aoclsparse_int        m, n, k, nnz_A, nnz_B;
        m     = 3;
        k     = 3;
        nnz_A = 6;
        nnz_B = 4;
        std::vector<aoclsparse_int> row_ptr_a, col_ind_a, row_ptr_b, col_ind_b;
        std::vector<T>              val_a, val_b;
        std::vector<T>              c(64);
        std::vector<T>              c_exp(64);
        std::vector<T>              c_t_exp(64);
        std::vector<T>              c_h_exp(64);
        std::vector<T>              t1(64);
        std::vector<T>              t2(64);
        std::vector<T>              t3(64);

        memset((void *)c_exp.data(), 0, sizeof(T) * 64);
        memset((void *)c_t_exp.data(), 0, sizeof(T) * 64);
        memset((void *)c_h_exp.data(), 0, sizeof(T) * 64);

        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        if(ba)
            base_a = aoclsparse_index_base_one;
        if(bb)
            base_b = aoclsparse_index_base_one;
        if(!row_maj)
            layout = aoclsparse_order_column;

        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>
                     || std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            n = 2;
            // Matrix A
            // 	1+2i    i       2-i
            // 	0       2i      0
            // 	0       0.1+4i  3
            row_ptr_a.assign({0 + ba, 3 + ba, 4 + ba, 6 + ba});
            col_ind_a.assign({0 + ba, 1 + ba, 2 + ba, 1 + ba, 1 + ba, 2 + ba});
            val_a.assign({{1, 2}, {0, 1}, {2, -1}, {0, 2}, {0.1, 4}, {3, 0}});

            // Matrix B
            // 	1-i    0
            // 	0      i
            // 	4+2i   3
            row_ptr_b.assign({0 + bb, 1 + bb, 2 + bb, 4 + bb});
            col_ind_b.assign({0 + bb, 1 + bb, 0 + bb, 1 + bb});
            val_b.assign({{1, -1}, {0, 1}, {4, 2}, {3, 0}});

            // expected output matrices
            // c dimension 3x2
            t1.assign({{13, 1}, {5, -3}, {0, 0}, {-2, 0}, {12, 6}, {5, 0.1}});
            t2.assign({{3, 1}, {0, 0}, {-6.6, 17.2}, {-1.7, 12}, {13, 3}, {9, 0}});
            t3.assign({{-1, -3}, {0, 0}, {7.4, -16.8}, {2.3, -12}, {15, 5}, {9, 0}});
        }
        else
        {
            n = 3;
            // Matrix A
            // 	1  0  2
            // 	0  0  3
            // 	4  5  6
            row_ptr_a.assign({0 + ba, 2 + ba, 3 + ba, 6 + ba});
            col_ind_a.assign({0 + ba, 2 + ba, 2 + ba, 0 + ba, 1 + ba, 2 + ba});
            val_a.assign({1, 2, 3, 4, 5, 6});

            // Matrix B
            // 	1  2  0
            // 	0  0  3
            // 	0  4  0
            row_ptr_b.assign({0 + bb, 2 + bb, 3 + bb, 4 + bb});
            col_ind_b.assign({0 + bb, 1 + bb, 2 + bb, 1 + bb});
            val_b.assign({1, 2, 3, 4});

            // expected output
            t1.assign({1, 10, 0, 0, 12, 0, 4, 32, 15});
            t2.assign({1, 18, 0, 0, 20, 0, 2, 28, 9});
            t3 = t2;
        }
        if(ldc == -1)
        {
            if(row_maj)
                ldc = n;
            else
                ldc = m;
        }
        if(!row_maj)
        {
            for(aoclsparse_int i = 0; i < m; i++)
            {
                for(aoclsparse_int j = 0; j < n; j++)
                {
                    c_exp[i + ldc * j]   = t1[i * n + j];
                    c_t_exp[i + ldc * j] = t2[i * n + j];
                    c_h_exp[i + ldc * j] = t3[i * n + j];
                }
            }
        }
        else
        {
            for(aoclsparse_int i = 0; i < m; i++)
            {
                for(aoclsparse_int j = 0; j < n; j++)
                {
                    c_exp[i * ldc + j]   = t1[i * n + j];
                    c_t_exp[i * ldc + j] = t2[i * n + j];
                    c_h_exp[i * ldc + j] = t3[i * n + j];
                }
            }
        }
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      &A, base_a, m, k, nnz_A, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_create_csr<T>(
                      &B, base_b, k, n, nnz_B, row_ptr_b.data(), col_ind_b.data(), val_b.data()),
                  aoclsparse_status_success);

        // testing nullptrs
        EXPECT_EQ(
            (aoclsparse_spmmd<T>(aoclsparse_operation_none, nullptr, B, layout, c.data(), ldc, -1)),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            (aoclsparse_spmmd<T>(aoclsparse_operation_none, A, nullptr, layout, c.data(), ldc, -1)),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ((aoclsparse_spmmd<T>(aoclsparse_operation_none, A, B, layout, nullptr, ldc, -1)),
                  aoclsparse_status_invalid_pointer);

        for(aoclsparse_operation op : {aoclsparse_operation_none,
                                       aoclsparse_operation_transpose,
                                       aoclsparse_operation_conjugate_transpose})
        {

            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>
                         || std::is_same_v<T, aoclsparse_double_complex>
                         || std::is_same_v<T, aoclsparse_float_complex>)
            {
                EXPECT_EQ((aoclsparse_spmmd<T>(op, A, B, layout, c.data(), ldc, -1)),
                          aoclsparse_status_success);

                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, aoclsparse_float_complex>)
                {
                    if(op == aoclsparse_operation_none)
                    {
                        EXPECT_COMPLEX_ARR_NEAR(64,
                                                ((std::complex<float> *)c.data()),
                                                ((std::complex<float> *)c_exp.data()),
                                                abserr);
                    }
                    else if(op == aoclsparse_operation_transpose)
                    {
                        EXPECT_COMPLEX_ARR_NEAR(64,
                                                ((std::complex<float> *)c.data()),
                                                ((std::complex<float> *)c_t_exp.data()),
                                                abserr);
                    }
                    else if(op == aoclsparse_operation_conjugate_transpose)
                    {
                        EXPECT_COMPLEX_ARR_NEAR(64,
                                                ((std::complex<float> *)c.data()),
                                                ((std::complex<float> *)c_h_exp.data()),
                                                abserr);
                    }
                }
                if constexpr(std::is_same_v<T, std::complex<double>>
                             || std::is_same_v<T, aoclsparse_double_complex>)
                {
                    if(op == aoclsparse_operation_none)
                    {
                        EXPECT_COMPLEX_ARR_NEAR(64,
                                                ((std::complex<double> *)c.data()),
                                                ((std::complex<double> *)c_exp.data()),
                                                abserr);
                    }
                    else if(op == aoclsparse_operation_transpose)
                    {
                        EXPECT_COMPLEX_ARR_NEAR(64,
                                                ((std::complex<double> *)c.data()),
                                                ((std::complex<double> *)c_t_exp.data()),
                                                abserr);
                    }
                    else if(op == aoclsparse_operation_conjugate_transpose)
                    {
                        EXPECT_COMPLEX_ARR_NEAR(64,
                                                ((std::complex<double> *)c.data()),
                                                ((std::complex<double> *)c_h_exp.data()),
                                                abserr);
                    }
                }
            }
            else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                EXPECT_EQ((aoclsparse_spmmd<T>(op, A, B, layout, c.data(), ldc, -1)),
                          aoclsparse_status_success);
                // ToDo: check the output using vector EXPECT macro
                if(op == aoclsparse_operation_none)
                {
                    EXPECT_ARR_NEAR(64, c.data(), c_exp.data(), abserr);
                }
                else if(op == aoclsparse_operation_transpose)
                {
                    EXPECT_ARR_NEAR(64, c.data(), c_t_exp.data(), abserr);
                }
                else if(op == aoclsparse_operation_conjugate_transpose)
                {
                    EXPECT_ARR_NEAR(64, c.data(), c_h_exp.data(), abserr);
                }
            }
        }
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
    }

    TEST(spmmd, row_A0B0)
    {
        spmmd_testall<float>(0, 0, 1);
        spmmd_testall<double>(0, 0, 1);
        spmmd_testall<aoclsparse_float_complex>(0, 0, 1);
        spmmd_testall<aoclsparse_double_complex>(0, 0, 1);
    }

    TEST(spmmd, row_A0B1)
    {
        spmmd_testall<float>(0, 1, 1);
        spmmd_testall<double>(0, 1, 1);
        spmmd_testall<aoclsparse_float_complex>(0, 1, 1);
        spmmd_testall<aoclsparse_double_complex>(0, 1, 1);
    }
    TEST(spmmd, row_A1B0)
    {
        spmmd_testall<float>(1, 0, 1);
        spmmd_testall<double>(1, 0, 1);
        spmmd_testall<aoclsparse_float_complex>(1, 0, 1);
        spmmd_testall<aoclsparse_double_complex>(1, 0, 1);
    }
    /*
    TEST(spmmd, row_A1B1)
    {
        spmmd_testall<float>(1, 1, 1);
        spmmd_testall<double>(1, 1, 1);
        spmmd_testall<aoclsparse_float_complex>(1, 1, 1);
        spmmd_testall<aoclsparse_double_complex>(1, 1, 1);
    }

    TEST(spmmd, col_A0B0)
    {
        spmmd_testall<float>(0, 0, 0);
        spmmd_testall<double>(0, 0, 0);
        spmmd_testall<aoclsparse_float_complex>(0, 0, 0);
        spmmd_testall<aoclsparse_double_complex>(0, 0, 0);
    }

    TEST(spmmd, col_A0B1)
    {
        spmmd_testall<float>(0, 1, 0);
        spmmd_testall<double>(0, 1, 0);
        spmmd_testall<aoclsparse_float_complex>(0, 1, 0);
        spmmd_testall<aoclsparse_double_complex>(0, 1, 0);
    }
    TEST(spmmd, col_A1B0)
    {
        spmmd_testall<float>(1, 0, 0);
        spmmd_testall<double>(1, 0, 0);
        spmmd_testall<aoclsparse_float_complex>(1, 0, 0);
        spmmd_testall<aoclsparse_double_complex>(1, 0, 0);
    }
    TEST(spmmd, col_A1B1)
    {
        spmmd_testall<float>(1, 1, 0);
        spmmd_testall<double>(1, 1, 0);
        spmmd_testall<aoclsparse_float_complex>(1, 1, 0);
        spmmd_testall<aoclsparse_double_complex>(1, 1, 1);
    }

    TEST(spmmd, row_A0B0_ldc)
    {
        spmmd_testall<float>(0, 0, 1, 8);
        spmmd_testall<double>(0, 0, 1, 6);
        spmmd_testall<aoclsparse_float_complex>(0, 0, 1, 4);
        spmmd_testall<aoclsparse_double_complex>(0, 0, 1, 5);
    }

    TEST(spmmd, col_A1B1_ldc)
    {
        spmmd_testall<float>(1, 1, 0, 7);
        spmmd_testall<double>(1, 1, 0, 3);
        spmmd_testall<aoclsparse_float_complex>(1, 1, 0, 6);
        spmmd_testall<aoclsparse_double_complex>(1, 1, 1, 4);
    }
*/
} // namespace
