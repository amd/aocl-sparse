/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <vector>
template <typename T>
struct sol_opt_csr
{
    std::vector<aoclsparse_int> ptr, ind, idiag;
    std::vector<T>              diag_val, non_unit_diag_val, unit_diag_val, zero_diag_val;
};

namespace
{
    template <typename T>
    void generate_test_matrix(aoclsparse_int               matrix_id,
                              aoclsparse_int              &m,
                              aoclsparse_int              &n,
                              aoclsparse_int              &nnz,
                              std::vector<aoclsparse_int> &row_ptr,
                              std::vector<aoclsparse_int> &col_ind,
                              std::vector<T>              &val,
                              sol_opt_csr<T>              &sol_opt_csr_t,
                              aoclsparse_index_base        base,
                              aoclsparse::doid             doid)
    {
        switch(matrix_id)
        {
        case 0:
            /* 3x3 matrix:
            * [ 0 0 1 ]
            * [ 4 3 2 ]
            * [ 5 0 0 ]
            * Similar matrix for the complex types.
            */
            m = 3, n = 3, nnz = 5;
            col_ind.assign({2, 0, 1, 2, 0});
            row_ptr.assign({0, 1, 4, 5});
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                val.assign({1.0, 4.0, 3.0, 2.0, 5.0});
            }
            else if constexpr(std::is_same_v<T, std::complex<double>>
                              || std::is_same_v<T, std::complex<float>>
                              || std::is_same_v<T, aoclsparse_double_complex>
                              || std::is_same_v<T, aoclsparse_float_complex>)
            {
                val.assign({{1, -2}, {4, 2}, {3, -5}, {2, 2}, {5, -10}});
            }
            TRANSFORM_BASE(base, row_ptr, col_ind);

            // Populate the output vectors
            switch(doid)
            {
            case aoclsparse::doid::sl:
            case aoclsparse::doid::slc:
                sol_opt_csr_t.ptr.assign({0, 3, 5, 7});
                sol_opt_csr_t.ind.assign({0, 1, 2, 0, 1, 0, 2});
                sol_opt_csr_t.idiag.assign({0, 4, 6});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({0, 4, 5, 4, 3, 5, 0});
                    sol_opt_csr_t.unit_diag_val.assign({1, 4, 5, 4, 1, 5, 1});
                    sol_opt_csr_t.zero_diag_val.assign({0, 4, 5, 4, 0, 5, 0});
                    sol_opt_csr_t.diag_val.assign({0, 3, 0});
                }
                else
                {
                    if(doid == aoclsparse::doid::sl)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign(
                            {{0, 0}, {4, 2}, {5, -10}, {4, 2}, {3, -5}, {5, -10}, {0, 0}});
                        sol_opt_csr_t.unit_diag_val.assign(
                            {{1, 0}, {4, 2}, {5, -10}, {4, 2}, {1, 0}, {5, -10}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign(
                            {{0, 0}, {4, 2}, {5, -10}, {4, 2}, {0, 0}, {5, -10}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{0, 0}, {3, -5}, {0, 0}});
                    }
                    else if(doid == aoclsparse::doid::slc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign(
                            {{0, 0}, {4, -2}, {5, 10}, {4, -2}, {3, 5}, {5, 10}, {0, 0}});
                        sol_opt_csr_t.unit_diag_val.assign(
                            {{1, 0}, {4, -2}, {5, 10}, {4, -2}, {1, 0}, {5, 10}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign(
                            {{0, 0}, {4, -2}, {5, 10}, {4, -2}, {0, 0}, {5, 10}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{0, 0}, {3, 5}, {0, 0}});
                    }
                }
                break;
            case aoclsparse::doid::su:
            case aoclsparse::doid::suc:
                sol_opt_csr_t.ptr.assign({0, 2, 4, 7});
                sol_opt_csr_t.ind.assign({0, 2, 1, 2, 0, 1, 2});
                sol_opt_csr_t.idiag.assign({0, 2, 6});

                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({0, 1, 3, 2, 1, 2, 0});
                    sol_opt_csr_t.unit_diag_val.assign({1, 1, 1, 2, 1, 2, 1});
                    sol_opt_csr_t.zero_diag_val.assign({0, 1, 0, 2, 1, 2, 0});
                    sol_opt_csr_t.diag_val.assign({0, 3, 0});
                }
                else
                {
                    if(doid == aoclsparse::doid::su)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign(
                            {{0, 0}, {1, -2}, {3, -5}, {2, 2}, {1, -2}, {2, 2}, {0, 0}});
                        sol_opt_csr_t.unit_diag_val.assign(
                            {{1, 0}, {1, -2}, {1, 0}, {2, 2}, {1, -2}, {2, 2}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign(
                            {{0, 0}, {1, -2}, {0, 0}, {2, 2}, {1, -2}, {2, 2}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{0, 0}, {3, -5}, {0, 0}});
                    }
                    else if(doid == aoclsparse::doid::suc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign(
                            {{0, 0}, {1, 2}, {3, 5}, {2, -2}, {1, 2}, {2, -2}, {0, 0}});
                        sol_opt_csr_t.unit_diag_val.assign(
                            {{1, 0}, {1, 2}, {1, 0}, {2, -2}, {1, 2}, {2, -2}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign(
                            {{0, 0}, {1, 2}, {0, 0}, {2, -2}, {1, 2}, {2, -2}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{0, 0}, {3, 5}, {0, 0}});
                    }
                }
                break;
            case aoclsparse::doid::hl:
            case aoclsparse::doid::hlc:
                sol_opt_csr_t.ptr.assign({0, 3, 5, 7});
                sol_opt_csr_t.ind.assign({0, 1, 2, 0, 1, 0, 2});
                sol_opt_csr_t.idiag.assign({0, 4, 6});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({0, 4, 5, 4, 3, 5, 0});
                    sol_opt_csr_t.diag_val.assign({0, 3, 0});
                }
                else
                {
                    if(doid == aoclsparse::doid::hl)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign(
                            {{0, 0}, {4, -2}, {5, 10}, {4, 2}, {3, 0}, {5, -10}, {0, 0}});
                        sol_opt_csr_t.unit_diag_val.assign(
                            {{1, 0}, {4, -2}, {5, 10}, {4, 2}, {1, 0}, {5, -10}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign(
                            {{0, 0}, {4, -2}, {5, 10}, {4, 2}, {0, 0}, {5, -10}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{0, 0}, {3, 0}, {0, 0}});
                    }
                    else if(doid == aoclsparse::doid::hlc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign(
                            {{0, 0}, {4, 2}, {5, -10}, {4, -2}, {3, 0}, {5, 10}, {0, 0}});
                        sol_opt_csr_t.unit_diag_val.assign(
                            {{1, 0}, {4, 2}, {5, -10}, {4, -2}, {1, 0}, {5, 10}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign(
                            {{0, 0}, {4, 2}, {5, -10}, {4, -2}, {0, 0}, {5, 10}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{0, 0}, {3, 0}, {0, 0}});
                    }
                }
                break;
            case aoclsparse::doid::hu:
            case aoclsparse::doid::huc:
                sol_opt_csr_t.ptr.assign({0, 2, 4, 7});
                sol_opt_csr_t.ind.assign({0, 2, 1, 2, 0, 1, 2});
                sol_opt_csr_t.idiag.assign({0, 2, 6});

                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({0, 1, 3, 2, 1, 2, 0});
                    sol_opt_csr_t.diag_val.assign({0, 3, 0});
                }
                else
                {
                    if(doid == aoclsparse::doid::hu)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign(
                            {{0, 0}, {1, -2}, {3, 0}, {2, 2}, {1, 2}, {2, -2}, {0, 0}});
                        sol_opt_csr_t.unit_diag_val.assign(
                            {{1, 0}, {1, -2}, {1, 0}, {2, 2}, {1, 2}, {2, -2}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign(
                            {{0, 0}, {1, -2}, {0, 0}, {2, 2}, {1, 2}, {2, -2}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{0, 0}, {3, 0}, {0, 0}});
                    }
                    else if(doid == aoclsparse::doid::huc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign(
                            {{0, 0}, {1, 2}, {3, 0}, {2, -2}, {1, -2}, {2, 2}, {0, 0}});
                        sol_opt_csr_t.unit_diag_val.assign(
                            {{1, 0}, {1, 2}, {1, 0}, {2, -2}, {1, -2}, {2, 2}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign(
                            {{0, 0}, {1, 2}, {0, 0}, {2, -2}, {1, -2}, {2, 2}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{0, 0}, {3, 0}, {0, 0}});
                    }
                }
                break;
            default:
                break;
            }
            break;
        case 1:
            /* 4x4 matrix with diagonal dominance
            * [ 1 2 3 4 ]
            * [ 0 5 0 0 ]
            * [ 0 0 6 0 ]
            * [ 10 9 8 7]
            * Similar matrix for the complex types.
            */
            m = 4, n = 4, nnz = 10;
            col_ind.assign({3, 2, 0, 1, 1, 2, 3, 2, 1, 0});
            row_ptr.assign({0, 4, 5, 6, 10});
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                val.assign({4.0, 3.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
            }
            else if constexpr(std::is_same_v<T, std::complex<double>>
                              || std::is_same_v<T, std::complex<float>>
                              || std::is_same_v<T, aoclsparse_double_complex>
                              || std::is_same_v<T, aoclsparse_float_complex>)
            {
                val.assign({{4, -2},
                            {3, 1},
                            {1, -2},
                            {2, 6},
                            {5, -9},
                            {6, 8},
                            {7, -3},
                            {8, 5},
                            {9, -9},
                            {10, 1}});
            }
            TRANSFORM_BASE(base, row_ptr, col_ind);

            // Populate the output vectors
            switch(doid)
            {
            case aoclsparse::doid::sl:
            case aoclsparse::doid::slc:
                sol_opt_csr_t.ptr.assign({0, 2, 4, 6, 10});
                sol_opt_csr_t.ind.assign({0, 3, 1, 3, 2, 3, 2, 1, 0, 3});
                sol_opt_csr_t.idiag.assign({0, 2, 4, 9});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({1, 10, 5, 9, 6, 8, 8, 9, 10, 7});
                    sol_opt_csr_t.unit_diag_val.assign({1, 10, 1, 9, 1, 8, 8, 9, 10, 1});
                    sol_opt_csr_t.zero_diag_val.assign({0, 10, 0, 9, 0, 8, 8, 9, 10, 0});
                    sol_opt_csr_t.diag_val.assign({1, 5, 6, 7});
                }
                else
                {
                    if(doid == aoclsparse::doid::sl)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, -2},
                                                                {10, 1},
                                                                {5, -9},
                                                                {9, -9},
                                                                {6, 8},
                                                                {8, 5},
                                                                {8, 5},
                                                                {9, -9},
                                                                {10, 1},
                                                                {7, -3}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0},
                                                            {10, 1},
                                                            {1, 0},
                                                            {9, -9},
                                                            {1, 0},
                                                            {8, 5},
                                                            {8, 5},
                                                            {9, -9},
                                                            {10, 1},
                                                            {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0},
                                                            {10, 1},
                                                            {0, 0},
                                                            {9, -9},
                                                            {0, 0},
                                                            {8, 5},
                                                            {8, 5},
                                                            {9, -9},
                                                            {10, 1},
                                                            {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, -2}, {5, -9}, {6, 8}, {7, -3}});
                    }
                    else if(doid == aoclsparse::doid::slc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, 2},
                                                                {10, -1},
                                                                {5, 9},
                                                                {9, 9},
                                                                {6, -8},
                                                                {8, -5},
                                                                {8, -5},
                                                                {9, 9},
                                                                {10, -1},
                                                                {7, 3}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0},
                                                            {10, -1},
                                                            {1, 0},
                                                            {9, 9},
                                                            {1, 0},
                                                            {8, -5},
                                                            {8, -5},
                                                            {9, 9},
                                                            {10, -1},
                                                            {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0},
                                                            {10, -1},
                                                            {0, 0},
                                                            {9, 9},
                                                            {0, 0},
                                                            {8, -5},
                                                            {8, -5},
                                                            {9, 9},
                                                            {10, -1},
                                                            {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, 2}, {5, 9}, {6, -8}, {7, 3}});
                    }
                }
                break;
            case aoclsparse::doid::su:
            case aoclsparse::doid::suc:
                sol_opt_csr_t.ptr.assign({0, 4, 6, 8, 10});
                sol_opt_csr_t.ind.assign({0, 3, 2, 1, 0, 1, 0, 2, 0, 3});
                sol_opt_csr_t.idiag.assign({0, 5, 7, 9});

                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({1, 4, 3, 2, 2, 5, 3, 6, 4, 7});
                    sol_opt_csr_t.unit_diag_val.assign({1, 4, 3, 2, 2, 1, 3, 1, 4, 1});
                    sol_opt_csr_t.zero_diag_val.assign({0, 4, 3, 2, 2, 0, 3, 0, 4, 0});
                    sol_opt_csr_t.diag_val.assign({1, 5, 6, 7});
                }
                else
                {
                    if(doid == aoclsparse::doid::su)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, -2},
                                                                {4, -2},
                                                                {3, 1},
                                                                {2, 6},
                                                                {2, 6},
                                                                {5, -9},
                                                                {3, 1},
                                                                {6, 8},
                                                                {4, -2},
                                                                {7, -3}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0},
                                                            {4, -2},
                                                            {3, 1},
                                                            {2, 6},
                                                            {2, 6},
                                                            {1, 0},
                                                            {3, 1},
                                                            {1, 0},
                                                            {4, -2},
                                                            {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0},
                                                            {4, -2},
                                                            {3, 1},
                                                            {2, 6},
                                                            {2, 6},
                                                            {0, 0},
                                                            {3, 1},
                                                            {0, 0},
                                                            {4, -2},
                                                            {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, -2}, {5, -9}, {6, 8}, {7, -3}});
                    }
                    else if(doid == aoclsparse::doid::suc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, 2},
                                                                {4, 2},
                                                                {3, -1},
                                                                {2, -6},
                                                                {2, -6},
                                                                {5, 9},
                                                                {3, -1},
                                                                {6, -8},
                                                                {4, 2},
                                                                {7, 3}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0},
                                                            {4, 2},
                                                            {3, -1},
                                                            {2, -6},
                                                            {2, -6},
                                                            {1, 0},
                                                            {3, -1},
                                                            {1, 0},
                                                            {4, 2},
                                                            {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0},
                                                            {4, 2},
                                                            {3, -1},
                                                            {2, -6},
                                                            {2, -6},
                                                            {0, 0},
                                                            {3, -1},
                                                            {0, 0},
                                                            {4, 2},
                                                            {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, 2}, {5, 9}, {6, -8}, {7, 3}});
                    }
                }
                break;
            case aoclsparse::doid::hl:
            case aoclsparse::doid::hlc:
                sol_opt_csr_t.ptr.assign({0, 2, 4, 6, 10});
                sol_opt_csr_t.ind.assign({0, 3, 1, 3, 2, 3, 2, 1, 0, 3});
                sol_opt_csr_t.idiag.assign({0, 2, 4, 9});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({1, 10, 5, 9, 6, 8, 8, 9, 10, 7});
                    sol_opt_csr_t.diag_val.assign({1, 5, 6, 7});
                }
                else
                {
                    if(doid == aoclsparse::doid::hl)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, 0},
                                                                {10, -1},
                                                                {5, 0},
                                                                {9, 9},
                                                                {6, 0},
                                                                {8, -5},
                                                                {8, 5},
                                                                {9, -9},
                                                                {10, 1},
                                                                {7, 0}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0},
                                                            {10, -1},
                                                            {1, 0},
                                                            {9, 9},
                                                            {1, 0},
                                                            {8, -5},
                                                            {8, 5},
                                                            {9, -9},
                                                            {10, 1},
                                                            {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0},
                                                            {10, -1},
                                                            {0, 0},
                                                            {9, 9},
                                                            {0, 0},
                                                            {8, -5},
                                                            {8, 5},
                                                            {9, -9},
                                                            {10, 1},
                                                            {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, 0}, {5, 0}, {6, 0}, {7, 0}});
                    }
                    else if(doid == aoclsparse::doid::hlc)
                    {

                        sol_opt_csr_t.non_unit_diag_val.assign({{1, 0},
                                                                {10, 1},
                                                                {5, 0},
                                                                {9, -9},
                                                                {6, 0},
                                                                {8, 5},
                                                                {8, -5},
                                                                {9, 9},
                                                                {10, -1},
                                                                {7, 0}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0},
                                                            {10, 1},
                                                            {1, 0},
                                                            {9, -9},
                                                            {1, 0},
                                                            {8, 5},
                                                            {8, -5},
                                                            {9, 9},
                                                            {10, -1},
                                                            {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0},
                                                            {10, 1},
                                                            {0, 0},
                                                            {9, -9},
                                                            {0, 0},
                                                            {8, 5},
                                                            {8, -5},
                                                            {9, 9},
                                                            {10, -1},
                                                            {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, 0}, {5, 0}, {6, 0}, {7, 0}});
                    }
                }
                break;
            case aoclsparse::doid::hu:
            case aoclsparse::doid::huc:
                sol_opt_csr_t.ptr.assign({0, 4, 6, 8, 10});
                sol_opt_csr_t.ind.assign({0, 3, 2, 1, 0, 1, 0, 2, 0, 3});
                sol_opt_csr_t.idiag.assign({0, 5, 7, 9});

                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({1, 4, 3, 2, 2, 5, 3, 6, 4, 7});
                    sol_opt_csr_t.diag_val.assign({1, 5, 6, 7});
                }
                else
                {
                    if(doid == aoclsparse::doid::hu)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, 0},
                                                                {4, -2},
                                                                {3, 1},
                                                                {2, 6},
                                                                {2, -6},
                                                                {5, 0},
                                                                {3, -1},
                                                                {6, 0},
                                                                {4, 2},
                                                                {7, 0}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0},
                                                            {4, -2},
                                                            {3, 1},
                                                            {2, 6},
                                                            {2, -6},
                                                            {1, 0},
                                                            {3, -1},
                                                            {1, 0},
                                                            {4, 2},
                                                            {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0},
                                                            {4, -2},
                                                            {3, 1},
                                                            {2, 6},
                                                            {2, -6},
                                                            {0, 0},
                                                            {3, -1},
                                                            {0, 0},
                                                            {4, 2},
                                                            {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, 0}, {5, 0}, {6, 0}, {7, 0}});
                    }
                    else if(doid == aoclsparse::doid::huc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, 0},
                                                                {4, 2},
                                                                {3, -1},
                                                                {2, -6},
                                                                {2, 6},
                                                                {5, 0},
                                                                {3, 1},
                                                                {6, 0},
                                                                {4, -2},
                                                                {7, 0}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0},
                                                            {4, 2},
                                                            {3, -1},
                                                            {2, -6},
                                                            {2, 6},
                                                            {1, 0},
                                                            {3, 1},
                                                            {1, 0},
                                                            {4, -2},
                                                            {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0},
                                                            {4, 2},
                                                            {3, -1},
                                                            {2, -6},
                                                            {2, 6},
                                                            {0, 0},
                                                            {3, 1},
                                                            {0, 0},
                                                            {4, -2},
                                                            {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, 0}, {5, 0}, {6, 0}, {7, 0}});
                    }
                }
                break;
            default:
                break;
            }
            break;
        case 2:
            // Singleton matrix
            m = 1, n = 1, nnz = 1;
            col_ind.assign({0});
            row_ptr.assign({0, 1});
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                val.assign({4.0});
            }
            else if constexpr(std::is_same_v<T, std::complex<double>>
                              || std::is_same_v<T, std::complex<float>>
                              || std::is_same_v<T, aoclsparse_double_complex>
                              || std::is_same_v<T, aoclsparse_float_complex>)
            {
                val.assign({{4, -2}});
            }
            TRANSFORM_BASE(base, row_ptr, col_ind);

            // Populate the output vectors
            switch(doid)
            {
            case aoclsparse::doid::sl:
            case aoclsparse::doid::slc:
            case aoclsparse::doid::su:
            case aoclsparse::doid::suc:
            case aoclsparse::doid::hl:
            case aoclsparse::doid::hlc:
            case aoclsparse::doid::hu:
            case aoclsparse::doid::huc:
                sol_opt_csr_t.ptr.assign({0, 1});
                sol_opt_csr_t.ind.assign({0});
                sol_opt_csr_t.idiag.assign({0});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({4});
                    sol_opt_csr_t.unit_diag_val.assign({1});
                    sol_opt_csr_t.zero_diag_val.assign({0});
                    sol_opt_csr_t.diag_val.assign({4});
                }
                else
                {
                    if(doid == aoclsparse::doid::sl || doid == aoclsparse::doid::su)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{4, -2}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0}});
                        sol_opt_csr_t.diag_val.assign({{4, -2}});
                    }
                    else if(doid == aoclsparse::doid::slc || doid == aoclsparse::doid::suc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{4, 2}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0}});
                        sol_opt_csr_t.diag_val.assign({{4, 2}});
                    }
                    else if(doid == aoclsparse::doid::hl || doid == aoclsparse::doid::hu
                            || doid == aoclsparse::doid::hlc || doid == aoclsparse::doid::huc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{4, 0}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0}});
                        sol_opt_csr_t.diag_val.assign({{4, 0}});
                    }
                }
                break;
            default:
                break;
            }
            break;
        case 3:
            /* 3x3 matrix with only diagonals:
            * [ 1 0 0 ]
            * [ 0 2 0 ]
            * [ 0 0 3 ]
            * Similar matrix for the complex types.
            */
            m = 3, n = 3, nnz = 3;
            col_ind.assign({0, 1, 2});
            row_ptr.assign({0, 1, 2, 3});
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                val.assign({1.0, 2.0, 3.0});
            }
            else if constexpr(std::is_same_v<T, std::complex<double>>
                              || std::is_same_v<T, std::complex<float>>
                              || std::is_same_v<T, aoclsparse_double_complex>
                              || std::is_same_v<T, aoclsparse_float_complex>)
            {
                val.assign({{1, -2}, {2, 2}, {3, -10}});
            }
            TRANSFORM_BASE(base, row_ptr, col_ind);

            // Populate the output vectors
            switch(doid)
            {
            case aoclsparse::doid::sl:
            case aoclsparse::doid::slc:
            case aoclsparse::doid::su:
            case aoclsparse::doid::suc:
                sol_opt_csr_t.ptr.assign({0, 1, 2, 3});
                sol_opt_csr_t.ind.assign({0, 1, 2});
                sol_opt_csr_t.idiag.assign({0, 1, 2});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({1, 2, 3});
                    sol_opt_csr_t.unit_diag_val.assign({1, 1, 1});
                    sol_opt_csr_t.zero_diag_val.assign({0, 0, 0});
                    sol_opt_csr_t.diag_val.assign({1, 2, 3});
                }
                else
                {
                    if(doid == aoclsparse::doid::su || doid == aoclsparse::doid::sl)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, -2}, {2, 2}, {3, -10}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0}, {1, 0}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0}, {0, 0}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, -2}, {2, 2}, {3, -10}});
                    }
                    else if(doid == aoclsparse::doid::suc || doid == aoclsparse::doid::slc)
                    {
                        sol_opt_csr_t.non_unit_diag_val.assign({{1, 2}, {2, -2}, {3, 10}});
                        sol_opt_csr_t.unit_diag_val.assign({{1, 0}, {1, 0}, {1, 0}});
                        sol_opt_csr_t.zero_diag_val.assign({{0, 0}, {0, 0}, {0, 0}});
                        sol_opt_csr_t.diag_val.assign({{1, 2}, {2, -2}, {3, 10}});
                    }
                }
                break;
            case aoclsparse::doid::hl:
            case aoclsparse::doid::hlc:
            case aoclsparse::doid::hu:
            case aoclsparse::doid::huc:
                sol_opt_csr_t.ptr.assign({0, 1, 2, 3});
                sol_opt_csr_t.ind.assign({0, 1, 2});
                sol_opt_csr_t.idiag.assign({0, 1, 2});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({1, 2, 3});
                    sol_opt_csr_t.diag_val.assign({1, 2, 3});
                }
                else
                {
                    sol_opt_csr_t.non_unit_diag_val.assign({{1, 0}, {2, 0}, {3, 0}});
                    sol_opt_csr_t.unit_diag_val.assign({{1, 0}, {1, 0}, {1, 0}});
                    sol_opt_csr_t.zero_diag_val.assign({{0, 0}, {0, 0}, {0, 0}});
                    sol_opt_csr_t.diag_val.assign({{1, 0}, {2, 0}, {3, 0}});
                }
                break;
            default:
                break;
            }
            break;
        default:
            break;
        }
    }

    /**
     * @brief Test aoclsparse_optimize to transform U/L triangle
     * of the matrix into full symmetric/hermitian matrices.
     */
    template <typename T>
    void test_opt_symm_herm_matrix(int matrix_id)
    {
        std::vector<aoclsparse_int> row_ptr, col_ind;
        std::vector<T>              values;
        aoclsparse_int              m, n, nnz;
        aoclsparse_matrix           A;
        sol_opt_csr<T>              sol_opt_csr_t;
        T                           alpha, beta;
        T                          *val_exp;

        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            alpha = 1.0;
            beta  = 0.0;
        }
        else
        {
            alpha = {1.0, 0.0};
            beta  = {0.0, 0.0};
        }
        for(aoclsparse_matrix_type mat_type :
            {aoclsparse_matrix_type_symmetric, aoclsparse_matrix_type_hermitian})
        {
            for(aoclsparse_fill_mode fill_mode :
                {aoclsparse_fill_mode_lower, aoclsparse_fill_mode_upper})
            {
                for(aoclsparse_index_base base :
                    {aoclsparse_index_base_zero, aoclsparse_index_base_one})
                {
                    for(aoclsparse_operation op : {aoclsparse_operation_none,
                                                   aoclsparse_operation_transpose,
                                                   aoclsparse_operation_conjugate_transpose})
                    {
                        for(aoclsparse_diag_type dtype : {aoclsparse_diag_type_unit,
                                                          aoclsparse_diag_type_zero,
                                                          aoclsparse_diag_type_non_unit})
                        {

                            // Create matrix descriptor
                            aoclsparse_mat_descr descr;
                            ASSERT_EQ(aoclsparse_create_mat_descr(&descr),
                                      aoclsparse_status_success);
                            ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base),
                                      aoclsparse_status_success);

                            // Set matrix as symmetric
                            ASSERT_EQ(aoclsparse_set_mat_type(descr, mat_type),
                                      aoclsparse_status_success);
                            ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill_mode),
                                      aoclsparse_status_success);
                            ASSERT_EQ(aoclsparse_set_mat_diag_type(descr, dtype),
                                      aoclsparse_status_success);

                            aoclsparse::doid lcl_doid;
                            lcl_doid = aoclsparse::get_doid<T>(descr, op);
                            // Generate test matrix
                            generate_test_matrix<T>(matrix_id,
                                                    m,
                                                    n,
                                                    nnz,
                                                    row_ptr,
                                                    col_ind,
                                                    values,
                                                    sol_opt_csr_t,
                                                    base,
                                                    lcl_doid);

                            // Set expected values based on the matrix diag type
                            if(dtype == aoclsparse_diag_type_unit)
                                val_exp = sol_opt_csr_t.unit_diag_val.data();
                            else if(dtype == aoclsparse_diag_type_zero)
                                val_exp = sol_opt_csr_t.zero_diag_val.data();
                            else if(dtype == aoclsparse_diag_type_non_unit)
                                val_exp = sol_opt_csr_t.non_unit_diag_val.data();
                            T x[m], y[m];

                            ASSERT_EQ(aoclsparse_create_csr(&A,
                                                            base,
                                                            m,
                                                            n,
                                                            nnz,
                                                            row_ptr.data(),
                                                            col_ind.data(),
                                                            values.data()),
                                      aoclsparse_status_success);

                            // Set optimization hint
                            ASSERT_EQ(aoclsparse_set_mv_hint(A, op, descr, 1),
                                      aoclsparse_status_success);

                            // Test aoclsparse_optimize function
                            EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

                            for(auto mat : A->mats)
                            {
                                aoclsparse::csr *csr_m = dynamic_cast<aoclsparse::csr *>(mat);
                                if(csr_m != nullptr && mat->mat_type == aoclsparse_csr_mat
                                   && mat->doid == lcl_doid)
                                {
                                    EXPECT_EQ_VEC(m + 1, sol_opt_csr_t.ptr.data(), csr_m->ptr);
                                    EXPECT_EQ_VEC(
                                        sol_opt_csr_t.ptr[m], sol_opt_csr_t.ind.data(), csr_m->ind);
                                    EXPECT_EQ_VEC(m, sol_opt_csr_t.idiag.data(), csr_m->idiag);
                                    /*
                                     * For real types with hermitian matrix, we don't need to call
                                     * aoclsparse_mv. We just compare the values directly from
                                     * the optimize() routine and return.
                                     */
                                    if constexpr(std::is_same_v<T, double>
                                                 || std::is_same_v<T, float>)
                                        if(mat_type == aoclsparse_matrix_type_hermitian)
                                        {
                                            EXPECT_EQ_VEC(sol_opt_csr_t.ptr[m],
                                                          sol_opt_csr_t.non_unit_diag_val.data(),
                                                          (T *)csr_m->val);
                                            EXPECT_EQ_VEC(m,
                                                          sol_opt_csr_t.diag_val.data(),
                                                          (T *)csr_m->diag_val);
                                            break;
                                        }
                                    /*
                                     * Call aoclsparse_mv to trigger internal optimization routines
                                     * that set matrix diagonals according to the specified descriptor type.
                                     * This call ensures the diagonal elements are correctly processed
                                     * based on whether the matrix is unit, non-unit, or zero diagonal.
                                     * We're primarily validating the values of internal data structures, not the MV result.
                                     */
                                    EXPECT_EQ(aoclsparse_mv<T>(op, &alpha, A, descr, x, &beta, y),
                                              aoclsparse_status_success);
                                    if constexpr(std::is_same_v<T, double>
                                                 || std::is_same_v<T, float>)
                                    {
                                        EXPECT_EQ_VEC(
                                            sol_opt_csr_t.ptr[m], val_exp, (T *)csr_m->val);
                                        EXPECT_EQ_VEC(
                                            m, sol_opt_csr_t.diag_val.data(), (T *)csr_m->diag_val);
                                    }
                                    else if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                                                      || std::is_same_v<T, std::complex<float>>)
                                    {
                                        EXPECT_COMPLEX_EQ_VEC(
                                            sol_opt_csr_t.ptr[m],
                                            reinterpret_cast<std::complex<float> *>(val_exp),
                                            reinterpret_cast<std::complex<float> *>(
                                                (T *)csr_m->val));
                                        EXPECT_COMPLEX_EQ_VEC(
                                            m,
                                            reinterpret_cast<std::complex<float> *>(
                                                sol_opt_csr_t.diag_val.data()),
                                            reinterpret_cast<std::complex<float> *>(
                                                (T *)csr_m->diag_val));
                                    }
                                    else if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                                                      || std::is_same_v<T, std::complex<double>>)
                                    {
                                        EXPECT_COMPLEX_EQ_VEC(
                                            sol_opt_csr_t.ptr[m],
                                            reinterpret_cast<std::complex<double> *>(val_exp),
                                            reinterpret_cast<std::complex<double> *>(
                                                (T *)csr_m->val));
                                        EXPECT_COMPLEX_EQ_VEC(
                                            m,
                                            reinterpret_cast<std::complex<double> *>(
                                                sol_opt_csr_t.diag_val.data()),
                                            reinterpret_cast<std::complex<double> *>(
                                                (T *)csr_m->diag_val));
                                    }
                                }
                            }

                            // Cleanup
                            aoclsparse_destroy_mat_descr(descr);
                            ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
                        }
                    }
                }
            }
        }
    }

    TEST(optimize_symm_herm_tests, Success)
    {
        for(aoclsparse_int mat_id = 0; mat_id < 4; mat_id++)
        {
            test_opt_symm_herm_matrix<float>(mat_id);
            test_opt_symm_herm_matrix<double>(mat_id);
            test_opt_symm_herm_matrix<aoclsparse_float_complex>(mat_id);
            test_opt_symm_herm_matrix<aoclsparse_double_complex>(mat_id);
            test_opt_symm_herm_matrix<std::complex<float>>(mat_id);
            test_opt_symm_herm_matrix<std::complex<double>>(mat_id);
        }
    }

    TEST(optimize_symm_herm_tests, ErrorConditions)
    {
        // Test null pointer
        EXPECT_EQ(aoclsparse_optimize(nullptr), aoclsparse_status_invalid_pointer);

        // Test with invalid matrix (create and modify to be invalid)
        std::vector<aoclsparse_int> row_ptr = {0, 2};
        std::vector<aoclsparse_int> col_ind = {0, 1};
        std::vector<double>         values  = {1.0, 2.0};

        aoclsparse_matrix A;
        // Create a matrix with values m = 1, n =2
        ASSERT_EQ(aoclsparse_create_csr(&A,
                                        aoclsparse_index_base_zero,
                                        1,
                                        2,
                                        2,
                                        row_ptr.data(),
                                        col_ind.data(),
                                        values.data()),
                  aoclsparse_status_success);

        aoclsparse_mat_descr descr;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        // Set matrix as symmetric with invalid values (m = 1, n =2)
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric),
                  aoclsparse_status_success);
        // Set optimization hint
        ASSERT_EQ(aoclsparse_set_mv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);

        // Test aoclsparse_optimize function with invalid matrix dimension
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_size);

        // Make matrix invalid by setting negative dimensions
        A->m = -1;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_size);

        // Restore valid dimensions but set invalid nnz
        A->m   = 2;
        A->nnz = -1;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_size);

        // Restore valid nnz and test with null pointers
        A->nnz                   = 2;
        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
        csr_mat->ptr             = nullptr;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_pointer);

        // Restore valid row pointer
        csr_mat->ptr = row_ptr.data();
        csr_mat->ind = nullptr;
        EXPECT_EQ(aoclsparse_optimize(A), aoclsparse_status_invalid_pointer);

        // Cleanup
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

} // namespace
