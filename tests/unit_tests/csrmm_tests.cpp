/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_init.hpp"
#include "aoclsparse_interface.hpp"

#include <algorithm>
#include <sstream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "blis.hh"
#include "cblas.hh"
#pragma GCC diagnostic pop
#include "level3_test_common.hpp"

namespace
{
    aoclsparse_order      col  = aoclsparse_order_column;
    aoclsparse_order      row  = aoclsparse_order_row;
    aoclsparse_operation  op_t = aoclsparse_operation_transpose;
    aoclsparse_operation  op_h = aoclsparse_operation_conjugate_transpose;
    aoclsparse_operation  op_n = aoclsparse_operation_none;
    aoclsparse_index_base zero = aoclsparse_index_base_zero;
    aoclsparse_index_base one  = aoclsparse_index_base_one;

    // Function to cover multiple initializations for both real and complex data types
    template <typename T>
    void init(aoclsparse_operation        &op,
              aoclsparse_order            &order,
              aoclsparse_int              &m,
              aoclsparse_int              &k,
              aoclsparse_int              &n,
              aoclsparse_int              &nnz,
              std::vector<T>              &csr_val,
              std::vector<aoclsparse_int> &csr_col_ind,
              std::vector<aoclsparse_int> &csr_row_ptr,
              T                           &alpha,
              T                           &beta,
              std::vector<T>              &B,
              std::vector<T>              &C,
              std::vector<T>              &C_exp,
              aoclsparse_index_base        base,
              aoclsparse_int               id,
              aoclsparse_matrix_type       mat_type = aoclsparse_matrix_type_general,
              aoclsparse_fill_mode         fill     = aoclsparse_fill_mode_lower,
              aoclsparse_diag_type         diag     = aoclsparse_diag_type_non_unit)
    {
        // Boolean variables to detect the matrix types -- Symmetric, Upper/Lower triangular, Non-unit/Unit diagonal.
        // Symmetric/Hermitian upper triangular and non unit diagonal matrix
        const bool symut_non_unit = (mat_type == aoclsparse_matrix_type_symmetric
                                     || mat_type == aoclsparse_matrix_type_hermitian)
                                    && (fill == aoclsparse_fill_mode_upper)
                                    && (diag == aoclsparse_diag_type_non_unit);
        // Symmetric/Hermitian lower triangular and non unit diagonal matrix
        const bool symlt_non_unit = (mat_type == aoclsparse_matrix_type_symmetric
                                     || mat_type == aoclsparse_matrix_type_hermitian)
                                    && (fill == aoclsparse_fill_mode_lower)
                                    && (diag == aoclsparse_diag_type_non_unit);
        // Symmetric/Hermitian upper triangular and unit diagonal matrix
        const bool symut_unit = (mat_type == aoclsparse_matrix_type_symmetric
                                 || mat_type == aoclsparse_matrix_type_hermitian)
                                && (fill == aoclsparse_fill_mode_upper)
                                && (diag == aoclsparse_diag_type_unit);
        // Symmetric/Hermitian lower triangular and unit diagonal matrix
        const bool symlt_unit = (mat_type == aoclsparse_matrix_type_symmetric
                                 || mat_type == aoclsparse_matrix_type_hermitian)
                                && (fill == aoclsparse_fill_mode_lower)
                                && (diag == aoclsparse_diag_type_unit);

        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            switch(id)
            {
            // Test cases to check with square matrices
            case 0:
                m = 3, k = 3, n = 3, nnz = 4;
                csr_val.assign({42., 0.2, 4.6, -8});
                csr_col_ind.assign({1, 2, 0, 2});
                csr_row_ptr.assign({0, 2, 3, 4});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                alpha = 0, beta = -3.2;
                C.assign({1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 1.0, -2.0, 3.0});
                B.assign({-1.0, -2.7, 3.0, 4.5, 5.8, -6.0, 1.0, -2.0, 3.0});
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({-3.200000,
                                      6.400000,
                                      -9.600000,
                                      -12.800000,
                                      -16,
                                      19.200000,
                                      -3.200000,
                                      6.400000,
                                      -9.600000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-3.200000,
                                      6.400000,
                                      -9.600000,
                                      -12.800000,
                                      -16,
                                      19.200000,
                                      -3.200000,
                                      6.400000,
                                      -9.600000});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({-3.200000,
                                      6.400000,
                                      -9.600000,
                                      -12.800000,
                                      -16,
                                      19.200000,
                                      -3.200000,
                                      6.400000,
                                      -9.600000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-3.200000,
                                      6.400000,
                                      -9.600000,
                                      -12.800000,
                                      -16,
                                      19.200000,
                                      -3.200000,
                                      6.400000,
                                      -9.600000});
                }
                break;
            case 1:
                m   = 5;
                k   = 5;
                n   = 5;
                nnz = 8;
                csr_val.assign({42., 2, 4, 8, 10, 12, 14, 16});
                csr_col_ind.assign({1, 3, 1, 4, 2, 2, 3, 4});
                csr_row_ptr.assign({0, 2, 3, 4, 5, 8});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                alpha = 3.0, beta = 2.5;
                C.assign({1.0,  -2.0, 3.0, 4.0, 5.0,  -6.0, 1.0,  -2.0, 3.0, 4.0, 5.0,  -6.0, 1.0,
                          -2.0, 3.0,  4.0, 5.0, -6.0, 1.0,  -2.0, 3.0,  4.0, 5.0, -6.0, 10});
                B.assign({1.0,  -2.0, 3.0, 4.0, 5.0,  -6.0, 1.0,  -2.0, 3.0, 4.0, 5.0,  -6.0, 1.0,
                          -2.0, 3.0,  4.0, 5.0, -6.0, 1.0,  -2.0, 3.0,  4.0, 5.0, -6.0, 10});
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({-225.500000, -29,       127.500000, 100,         528.500000,
                                      129,         14.500000, 91,         -52.500000,  256,
                                      -755.500000, -87,       74.500000,  25,          103.500000,
                                      646,         72.500000, -63,        -177.500000, -275,
                                      475.500000,  58,        252.500000, 135,         433});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({2.500000,  97,          307.500000, 226,        324.500000,
                                      -15,       -741.500000, 229,        139.500000, 154,
                                      12.500000, 543,         50.500000,  151,        175.500000,
                                      10,        576.500000,  -57,        -57.500000, -245,
                                      7.500000,  436,         192.500000, 423,        625});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({-729.500000, 151,         -280.500000, 394,        504.500000,
                                      -87,         14.500000,   -29,         43.500000,  58,
                                      84.500000,   81,          122.500000,  -149,       247.500000,
                                      160,         -167.500000, 15,          -57.500000, 85,
                                      499.500000,  196,         36.500000,   -333,       529});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({2.500000,   -5,          7.500000,   10,          12.500000,
                                      39,         -237.500000, 349,        547.500000,  688,
                                      240.500000, 279,         2.500000,   -191,        307.500000,
                                      142,        168.500000,  213,        -225.500000, 445,
                                      271.500000, 58,          276.500000, -351,        577});
                }
                break;
            case 2:
                m = 7, k = 7, n = 7, nnz = 12;
                csr_val.assign({42., 2, 4, 8, 42., 0.2, 4.6, -8, 10, 12, 14, 16});
                csr_col_ind.assign({0, 5, 1, 4, 2, 5, 0, 1, 3, 6, 3, 4});
                csr_row_ptr.assign({0, 2, 4, 6, 7, 9, 10, 12});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                alpha = 2, beta = -1.5;
                C.assign({0});
                B.assign({1.0, -2.0, 3.0,  4.0,   5.0, -6.0, 1.0, -2.0, 3.0, 4.0,
                          5.0, -6.0, 1.0,  -2.0,  3.0, 4.0,  5.0, -6.0, 1.0, -2.0,
                          3.0, 4.0,  5.0,  -6.0,  10,  49,   0.2, 4.6,  -8,  1,
                          3,   12,   48,   26,    34,  25,   47,  10,   60,  542.0,
                          22,  42,   8.66, 42.89, 0.2, 4.6,  -8,  10,   124});
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({60,         64,          249.600000, 9.200000,    112,
                                      24,         272,         -164,       -72,         336.400000,
                                      -18.400000, 52,          -48,        -52,         244,
                                      48,         419.200000,  27.600000,  -184,        72,
                                      -136,       336.800000,  824,        -503.920000, 36.800000,
                                      120,        110.400000,  1848,       -568,        776,
                                      262.400000, -73.600000,  224,        816,         1872,
                                      2188,       9048,        848.800000, 230,         448,
                                      1008,       19024,       767.440000, 215.120000,  20.800000,
                                      79.672000,  -594.240000, 2976,       -127.200000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({120.800000, -96,         252,        128,         0,
                                      5.200000,   -144,        -122,       120,         336,
                                      -176,       -16,         -6.400000,  24,          196.800000,
                                      16,         420,         104,        160,         14,
                                      -48,        428,         -744,       -504,        1108.800000,
                                      227.200000, 13.600000,   4.800000,   -561.600000, -760,
                                      252,        1912,        1104,       -30.800000,  624,
                                      2652,       -8296,       840,        12016,       2096,
                                      104,        528,         769.760000, 471.120000,  16.800000,
                                      3312,       4654.240000, 34.720000,  240});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({184,        20,          292,        576,        2588,
                                      -416,       252,         -144,       40,         80,
                                      232,        720,         424,        528,        262,
                                      354.800000, 424,         -480,       300.800000, -159.200000,
                                      268.800000, 9.200000,    -18.400000, 27.600000,  36.800000,
                                      46,         -55.200000,  9.200000,   112,        52,
                                      -184,       120,         1076,       -12,        124,
                                      207.840000, 1029.360000, 4.800000,   110.400000, -192,
                                      240,        2976,        -144,       172,        -72,
                                      664,        2908,        837.600000, 1216.800000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({120.800000,  -122,       196.800000,  428,        870.800000,
                                      -502.160000, 126.320000, 112,         8,          -16,
                                      -152,        -816,       -408,        -560,       252,
                                      336,         420,        -504,        84,         -168,
                                      252,         82.480000,  1220.920000, 65.600000,  368.800000,
                                      736,         800,        4152,        245.120000, 1420.480000,
                                      70.400000,   227.200000, -352,        336,        3936,
                                      5.200000,    -6.400000,  14,          13.600000,  20.400000,
                                      -24.800000,  5.200000,   600,         1128,       240,
                                      1440,        13008,      528,         1008});
                }
                break;
            //Test cases to check with non square matrices
            case 3:
                m = 4, k = 3, n = 2, nnz = 3;
                csr_val.assign({2, 4, 8});
                csr_col_ind.assign({0, 1, 2});
                csr_row_ptr.assign({0, 0, 1, 2, 3});
                alpha = -4.5, beta = 11.0;
                C.assign({0});
                B.assign({3.0, 7.0, 3.0, 1.0, 5.0, 2.0, 4.0, 3.0, 1.0, 2.0, 1.0, 3.0});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({0, -27, -126, -108, 0, -9, -90, -72});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-63, -54, -36, -18, -72, -108});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({0, 0, -27, -63, -54, -18, -180, -72});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-27, -9, -90, -36, -144, -108});
                }
                break;
            case 4:
                m = 2, k = 4, n = 7, nnz = 4, alpha = -5.5, beta = 6.0;
                csr_val.assign({3.0, 6.0, 4.0, 3.0});
                csr_col_ind.assign({0, 1, 0, 1});
                csr_row_ptr.assign({0, 2, 4});
                B.assign({3.0, 7.0, 3.0, 1.0, 5.0, 2.0, 4.0, 3.0, 1.0, 2.0, 1.0, 3.0, 5.0, 2.0,
                          1.0, 1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 5.0});
                C.assign({0});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({-280.500000,
                                      -181.500000,
                                      -148.500000,
                                      -143,
                                      -82.500000,
                                      -55,
                                      -148.500000,
                                      -143,
                                      -66,
                                      -60.500000,
                                      -99,
                                      -77,
                                      -82.500000,
                                      -55});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-203.500000, -214.500000, 0, 0, -71.500000, -115.500000, 0, 0,
                                      -126.500000, -198,        0, 0, -132,       -181.500000, 0, 0,
                                      -60.500000,  -66,         0, 0, -82.500000, -82.500000,  0, 0,
                                      -126.500000, -198,        0, 0});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({-148.500000,
                                      -148.500000,
                                      -115.500000,
                                      -49.500000,
                                      -181.500000,
                                      -198,
                                      -132,
                                      -115.500000,
                                      -170.500000,
                                      -99,
                                      -38.500000,
                                      -159.500000,
                                      -126.500000,
                                      -121});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-115.500000, -137.500000, -93.500000, -38.500000,
                                      -148.500000, -143,        -110,       -148.500000,
                                      -247.500000, -132,        -49.500000, -214.500000,
                                      -148.500000, -165,        0,          0,
                                      0,           0,           0,          0,
                                      0,           0,           0,          0,
                                      0,           0,           0,          0});
                }
                break;
            case 5:
                m = 3, k = 4, n = 5, nnz = 3;
                csr_val.assign({8.00, 5.00, 7.00});
                csr_col_ind.assign({1, 2, 1});
                csr_row_ptr.assign({1, 2, 3, 4});
                alpha = 1.0, beta = 0.0;
                B.assign({1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,   9.00, 8.00,
                          10.00, 28.00, 32.00, 25.00, 70.00, 56.00, 40.00, 112.00, 8.00, 10.00});
                C.assign({0.0});
                if(order == aoclsparse_order_column)
                    C_exp.assign({8.00,
                                  10.00,
                                  7.00,
                                  40.00,
                                  30.00,
                                  35.00,
                                  72.00,
                                  40.00,
                                  63.00,
                                  256.00,
                                  125.00,
                                  224.00,
                                  320.00,
                                  560.00,
                                  280.00});
                else
                    C_exp.assign({8.00,
                                  16.00,
                                  24.00,
                                  32.00,
                                  40.00,
                                  30.00,
                                  35.00,
                                  40.00,
                                  45.00,
                                  40.00,
                                  7.00,
                                  14.00,
                                  21.00,
                                  28.00,
                                  35.00});
                break;
            //Test case to check failure and do nothing cases
            case 6:
                order = aoclsparse_order_column;
                m = 2, k = 3, n = 2, nnz = 1;
                csr_val.assign({42.});
                csr_col_ind.assign({1});
                csr_row_ptr.assign({0, 0, 1});
                alpha = 2.3, beta = 11.2;
                B.assign({1.0, -2.0, 3.0, 4.0, 5.0, -6.0});
                C.assign({0.1, 0.2, 0.3, 0.4});
                break;

            // Test cases for symmetric matrices (only the requested triangle is processed). Case 7 and 8.
            case 7:
                m = 3, k = 3, n = 3, nnz = 9, alpha = 2, beta = 2;
                csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8, 9});
                csr_col_ind.assign({0, 1, 2, 0, 1, 2, 0, 1, 2});
                csr_row_ptr.assign({0, 3, 6, 9});
                B.assign({3.0, 2.0, 2.0, 4.0, 2.0, 3.0, 4.0, 5.0, 4.0});
                C.assign({1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 1.0, -2.0, 3.0});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });

                // Note: Since symmetric matrices are equal to its transpose, we are using the same result to compare transpose cases.
                if(symut_non_unit)
                {
                    if(order == aoclsparse_order_column)
                        C_exp.assign({28, 52, 84, 42, 82, 90, 54, 110, 162});
                    else
                        C_exp.assign({48, 38, 46, 108, 98, 74, 140, 122, 126});
                }
                if(symut_unit)
                {
                    if(order == aoclsparse_order_column)
                        C_exp.assign({28, 36, 52, 42, 66, 42, 54, 70, 98});
                    else
                        C_exp.assign({48, 38, 46, 76, 82, 50, 76, 42, 62});
                }
                if(symlt_non_unit)
                {
                    if(order == aoclsparse_order_column)
                        C_exp.assign({52, 72, 116, 74, 110, 130, 106, 142, 214});
                    else
                        C_exp.assign({96, 86, 90, 136, 126, 98, 180, 146, 154});
                }
                if(symlt_unit)
                {
                    if(order == aoclsparse_order_column)
                        C_exp.assign({52, 56, 84, 74, 94, 82, 106, 102, 150});
                    else
                        C_exp.assign({96, 86, 90, 104, 110, 74, 116, 66, 90});
                }
                break;
            case 8:
                m   = 5;
                k   = 5;
                n   = 5;
                nnz = 8;
                csr_val.assign({42., 2, 4, 8, 10, 12, 14, 16});
                csr_col_ind.assign({1, 3, 1, 4, 2, 2, 3, 4});
                csr_row_ptr.assign({0, 2, 3, 4, 5, 8});
                alpha = 1, beta = 0;
                B.assign({1.0,  -2.0, 3.0, 4.0, 5.0,  -6.0, 1.0,  -2.0, 3.0, 4.0, 5.0,  -6.0, 1.0,
                          -2.0, 3.0,  4.0, 5.0, -6.0, 1.0,  -2.0, 3.0,  4.0, 5.0, -6.0, 10});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });

                // Note: Since symmetric matrices are equal to its transpose, we are using the same result to compare transpose cases.
                if(symut_non_unit)
                {
                    if(order == aoclsparse_order_column)
                        C_exp.assign({-76, 34, 40,  2,   104, 48, -248, 32,  -12, 48, -256, 186, 24,
                                      10,  56, 212, 188, -16, 8,  -80,  156, 142, 80, 6,    200});
                    else
                        C_exp.assign({-244, 52, -96, 128, 164, 18, -80, 118, 180, 226, 24,   32, 40,
                                      -48,  80, 2,   -4,  6,   8,  10,  88,  16,  88,  -112, 184});
                }
                if(symut_unit)
                {
                    if(order == aoclsparse_order_column)
                        C_exp.assign({-75, 40,   43,  6,   29, 42, -251, 30,  -9,
                                      -12, -251, 204, 25,  8,  11, 216,  173, -22,
                                      9,   -50,  159, 130, 85, 0,  50});
                    else
                        C_exp.assign({-243, 50, -93, 132, 169, 36, -83, 124, 171, 214, 29,  26, 41,
                                      -50,  83, 6,   1,   0,   9,  8,   43,  -44, 13,  -22, 34});
                }

                if(symlt_non_unit)
                {
                    if(order == aoclsparse_order_column)
                        C_exp.assign({0,  -8, 100, 100, 172, 0,   4,   78, 36, 82, 0,   -24, 16,
                                      52, 32, 0,   20,  -14, -88, -90, 0,  16, 60, 190, 136});
                    else
                        C_exp.assign({0,   0,   0,  0,  0,  -24,  4,   -8,  12, 16, 76,   98, 0,
                                      -62, 100, 92, -4, 80, -104, 170, 164, 62, 8,  -106, 168});
                }
                if(symlt_unit)
                {
                    if(order == aoclsparse_order_column)
                        C_exp.assign({1,  -2,  103, 104, 97,  -6,  1,   76, 39, 22, 5,   -6, 17,
                                      50, -13, 4,   5,   -20, -87, -60, 3,  4,  65, 184, -14});
                    else
                        C_exp.assign({1,   -2,  3,  4, 5,  -6,   1,   -2,  3, 4,   81,  92, 1,
                                      -64, 103, 96, 1, 74, -103, 168, 119, 2, -67, -16, 18});
                }
                break;

            // CSC datasets: csr_col_ind holds col_ptr, csr_row_ptr holds row_ind.
            case 9:
                // 3x3 general CSC matrix; A = [[1,3,0],[0,2,0],[0,0,4]]
                m = 3, k = 3, n = 3, nnz = 4, alpha = 1, beta = 0;
                csr_col_ind.assign({0, 1, 3, 4});
                csr_row_ptr.assign({0, 0, 1, 2});
                csr_val.assign({1, 3, 2, 4});
                B.assign({1, 0, 0, 0, 1, 0, 0, 0, 1});
                C.assign(9, T{});
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                // op_none -> C=A; op_t/op_h -> C=A^T (same for real)
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({1, 0, 0, 3, 2, 0, 0, 0, 4});
                    else
                        C_exp.assign({1, 3, 0, 0, 2, 0, 0, 0, 4});
                }
                else
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({1, 3, 0, 0, 2, 0, 0, 0, 4});
                    else
                        C_exp.assign({1, 0, 0, 3, 2, 0, 0, 0, 4});
                }
                break;
            case 10:
                // 3x3 symmetric CSC (full matrix); underlying A = [[1,2,3],[4,5,6],[7,8,9]]
                // CSC fill=upper expands upper of A; CSC fill=lower expands lower of A
                m = 3, k = 3, n = 3, nnz = 9, alpha = 1, beta = 0;
                csr_col_ind.assign({0, 3, 6, 9});
                csr_row_ptr.assign({0, 1, 2, 0, 1, 2, 0, 1, 2});
                csr_val.assign({1, 4, 7, 2, 5, 8, 3, 6, 9});
                B.assign({1, 0, 0, 0, 1, 0, 0, 0, 1});
                C.assign(9, T{});
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                // All ops equivalent for symmetric; result symmetric so row=col C_exp
                if(symut_non_unit)
                    C_exp.assign({1, 2, 3, 2, 5, 6, 3, 6, 9});
                if(symut_unit)
                    C_exp.assign({1, 2, 3, 2, 1, 6, 3, 6, 1});
                if(symlt_non_unit)
                    C_exp.assign({1, 4, 7, 4, 5, 8, 7, 8, 9});
                if(symlt_unit)
                    C_exp.assign({1, 4, 7, 4, 1, 8, 7, 8, 1});
                break;
            case 11:
                // 4x3 non-square general CSC; same A, B, C_exp as CSR id=3
                m = 4, k = 3, n = 2, nnz = 3;
                alpha = -4.5, beta = 11.0;
                csr_col_ind.assign({0, 1, 2, 3}); // col_ptr for CSC
                csr_row_ptr.assign({1, 2, 3}); // row_ind for CSC
                csr_val.assign({2, 4, 8});
                B.assign({3.0, 7.0, 3.0, 1.0, 5.0, 2.0, 4.0, 3.0, 1.0, 2.0, 1.0, 3.0});
                C.assign({0});
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({0, -27, -126, -108, 0, -9, -90, -72});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-63, -54, -36, -18, -72, -108});
                }
                else
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({0, 0, -27, -63, -54, -18, -180, -72});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-27, -9, -90, -36, -144, -108});
                }
                break;
            }
        }
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            switch(id)
            {
            case 0:
                m   = 3;
                k   = 3;
                n   = 3;
                nnz = 4;
                csr_val.assign({{42, 2}, {5, 2}, {3, 8}, {-81, 55}});
                csr_col_ind.assign({1, 2, 0, 2});
                csr_row_ptr.assign({0, 2, 3, 4});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                alpha = {0, 0};
                beta  = {2, 1};
                C.assign(
                    {{-1, 0}, {-2, 7}, {3, 0}, {4, 5}, {5, 8}, {-6, 0}, {1, 0}, {-2, 0}, {3, 0}});
                B.assign(
                    {{-1, 0}, {-2, 7}, {3, 0}, {4, 5}, {5, 8}, {-6, 0}, {1, 0}, {-2, 0}, {3, 0}});
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{-2, -1},
                                      {-11, 12},
                                      {6, 3},
                                      {3, 14},
                                      {2, 21},
                                      {-12, -6},
                                      {2, 1},
                                      {-4, -2},
                                      {6, 3}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{-2, -1},
                                      {-11, 12},
                                      {6, 3},
                                      {3, 14},
                                      {2, 21},
                                      {-12, -6},
                                      {2, 1},
                                      {-4, -2},
                                      {6, 3}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({{-2, -1},
                                      {-11, 12},
                                      {6, 3},
                                      {3, 14},
                                      {2, 21},
                                      {-12, -6},
                                      {2, 1},
                                      {-4, -2},
                                      {6, 3}});
                }
                if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{-2, -1},
                                      {-11, 12},
                                      {6, 3},
                                      {3, 14},
                                      {2, 21},
                                      {-12, -6},
                                      {2, 1},
                                      {-4, -2},
                                      {6, 3}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{-2, -1},
                                      {-11, 12},
                                      {6, 3},
                                      {3, 14},
                                      {2, 21},
                                      {-12, -6},
                                      {2, 1},
                                      {-4, -2},
                                      {6, 3}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({{-2, -1},
                                      {-11, 12},
                                      {6, 3},
                                      {3, 14},
                                      {2, 21},
                                      {-12, -6},
                                      {2, 1},
                                      {-4, -2},
                                      {6, 3}});
                }
                break;
            case 1:
                m   = 5;
                k   = 5;
                n   = 5;
                nnz = 8;
                csr_val.assign(
                    {{2, 52}, {52, 20}, {-13, 8}, {-8, 5}, {-2, 20}, {5, 12}, {5, 40}, {-1, 5}});
                csr_col_ind.assign({1, 3, 1, 4, 2, 2, 3, 4});
                csr_row_ptr.assign({0, 2, 3, 4, 5, 8});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                alpha = {1, -2};
                beta  = {-5.2, 7};
                C.assign({{0, 0}});
                B.assign({{1, 2},     {22, -5},  {54, 20},   {-113, 80}, {7.0, 56},
                          {-2, 25},   {3, 2},    {54, 80},   {-1, 5},    {87, -51},
                          {10, 240},  {24, 523}, {528, 206}, {-13, 8.0}, {-89, 57},
                          {-23, 233}, {55, 28},  {5, 10},    {-1, 5},    {87, -51.0},
                          {1, 2},     {2, 52},   {52, 20},   {-13, 8},   {-8, 5}});
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign(
                            {{-1104, 17378}, {236, 733},     {-1162, 259},    {1572, 2056},
                             {-10808, 4651}, {550, 900},     {-59, 108},      {1245, 1725},
                             {132, 4336},    {2311, 2973},   {-23084, 58418}, {-17710, 2385},
                             {-1375, -1755}, {15120, 20500}, {12355, 7210},   {4814, 6152},
                             {-787, 1954},   {1245, 1725},   {-50, 500},      {1030, 845},
                             {-2808, 7436},  {-1762, 224},   {-121, -158},    {1496, 2008},
                             {16, 963}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign(
                            {{-0, 0},         {246, 993},     {-6123, 1966},  {-825, 5070},
                             {-641, 1727},    {-0, 0},        {-1471, 2662},  {2467, -1139},
                             {10841, 743},    {-432, 1444},   {-0, 0},        {-28170, 28305},
                             {-3381, 1467},   {11805, 23415}, {-4470, 11390}, {-0, 0},
                             {-14409, 25548}, {2467, -1139},  {26381, 21643}, {940, 275},
                             {-0, 0},         {-1752, 484},   {-928, 121},    {-570, 285},
                             {-423, 1121}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign(
                            {{-0, 0},         {-538, 281},    {7125, -2770},  {2535, -4650},
                             {-1101, -403},   {-0, 0},        {1481, -2442},  {-2653, -1139},
                             {-5999, 23},     {-2962, -546},  {-0, 0},        {2310, -39775},
                             {3619, 747},     {39405, 1815},  {-10620, 1740}, {-0, 0},
                             {13295, -22300}, {-2653, -1139}, {19541, 5123},  {-1310, 175},
                             {-0, 0},         {-984, -1732},  {936, -47},     {1110, -75},
                             {-1053, 181}});
                }
                if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{16044, 25922},  {7634, -1688},  {3184, 11572}, {-18, 1026},
                                      {15390, -13230}, {-856, 7},      {-59, 108},    {-2558, 2076},
                                      {-173, -19},     {1995, 2805},   {-40, 25},     {-1088, 146},
                                      {-316, 1132},    {-194, -257},   {-121, -158},  {-5380, 9360},
                                      {-11640, 20450}, {15120, 20500}, {-686, -8},    {-4750, 30},
                                      {-9140, 26120},  {3139, 19727},  {15353, 8574}, {-801, 582},
                                      {6123, -261}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{-0, 0},        {-0, 0},        {-0, 0},        {-0, 0},
                                      {-0, 0},        {-846, 267},    {2513, 634},    {2206, 6788},
                                      {-15991, 3037}, {49, 9077},     {-6441, 8362},  {1372, 3896},
                                      {1418, 1184},   {-551, 372},    {4288, 279},    {285, 300},
                                      {214, 2172},    {10468, 564},   {-5021, 17142}, {4518, 4749},
                                      {-5025, 715},   {-11281, 2032}, {-2942, 12044}, {-367, -276},
                                      {-1482, -1766}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign(
                            {{-0, 0},        {-0, 0},        {-0, 0},        {-0, 0},
                             {-0, 0},        {-382, -1021},  {-2647, -726},  {-7394, -6412},
                             {15945, -1995}, {817, -3059},   {4719, -9478},  {-756, -3088},
                             {-598, -2024},  {545, -60},     {-4208, 831},   {285, -300},
                             {2094, -6788},  {228, -10556},  {9939, 15022},  {7878, -171},
                             {-2825, -4235}, {-6051, -9728}, {-12282, 1724}, {313, -336},
                             {1078, -2036}});
                }
                break;
            // Test cases for symmetric matrices (only the requested triangle is processed). Case 2 and 3.
            case 2:
                m = 3, k = 3, n = 3, nnz = 9;
                csr_val.assign(
                    {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10}});
                csr_col_ind.assign({0, 1, 2, 0, 1, 2, 0, 1, 2});
                csr_row_ptr.assign({0, 3, 6, 9});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                B.assign({{1, 2},
                          {22, -5},
                          {54, 20},
                          {-113, 80},
                          {7.0, 56},
                          {-2, 25},
                          {3, 2},
                          {54, 80},
                          {-1, 5}});
                alpha = {2, -2};
                beta  = {3, 2};
                C.assign(
                    {{-1, 0}, {-2, 7}, {3, 0}, {4, 5}, {5, 8}, {-6, 0}, {1, 0}, {-2, 0}, {3, 0}});
                if(symut_non_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign({{945, 394},
                                          {1844, 601},
                                          {2613, 818},
                                          {-956, 1197},
                                          {-1351, 2500},
                                          {-1816, 3298},
                                          {373, 996},
                                          {1012, 2018},
                                          {1243, 2416}});
                        else
                            C_exp.assign({{85, -1018},
                                          {256, -1915},
                                          {513, -2666},
                                          {1608, 337},
                                          {2897, 740},
                                          {3724, 1142},
                                          {773, -716},
                                          {1792, -1370},
                                          {2195, -1576}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign({{-1253, 620},
                                          {676, 1833},
                                          {199, 548},
                                          {-2564, 1637},
                                          {1515, 3462},
                                          {352, 970},
                                          {-2975, 1968},
                                          {2274, 4588},
                                          {575, 1228}});

                        else
                            C_exp.assign({{1055, 912},
                                          {1464, -1203},
                                          {347, -384},
                                          {2052, 2253},
                                          {3095, -2006},
                                          {760, -582},
                                          {2405, 2644},
                                          {4254, -2808},
                                          {1027, -760}});
                    }
                }
                if(symut_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign({{949, 382},
                                          {1384, 613},
                                          {749, -118},
                                          {-184, 1329},
                                          {-1267, 1352},
                                          {-1644, 2406},
                                          {369, 976},
                                          {252, 202},
                                          {1299, 2240}});

                        else
                            C_exp.assign({{81, -1006},
                                          {444, -1495},
                                          {9, -642},
                                          {836, 205},
                                          {1805, 1104},
                                          {2816, 1170},
                                          {777, -696},
                                          {408, 30},
                                          {2011, -1592}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign({{-1249, 608},
                                          {568, 1765},
                                          {63, 252},
                                          {16, 489},
                                          {1599, 2314},
                                          {492, 478},
                                          {-3075, 1884},
                                          {650, 1492},
                                          {631, 1052}});
                        else
                            C_exp.assign({{1051, 924},
                                          {1572, -1135},
                                          {483, -88},
                                          {0, 313},
                                          {2003, -1642},
                                          {252, -522},
                                          {2345, 2760},
                                          {1590, -544},
                                          {843, -776}});
                    }
                }

                if(symlt_non_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign({{1985, 674},
                                          {2284, 777},
                                          {2805, 810},
                                          {-932, 2045},
                                          {-2271, 3340},
                                          {-3568, 5026},
                                          {789, 1716},
                                          {1028, 2074},
                                          {1723, 3088}});

                        else
                            C_exp.assign({{365, -2058},
                                          {432, -2355},
                                          {505, -2858},
                                          {2456, 313},
                                          {3737, 1660},
                                          {5452, 2894},
                                          {1493, -1132},
                                          {1848, -1386},
                                          {2867, -2056}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign({{-2109, 1292},
                                          {1596, 3561},
                                          {167, 828},
                                          {-2532, 1669},
                                          {2123, 4062},
                                          {776, 1170},
                                          {-3863, 2640},
                                          {2682, 4956},
                                          {1423, 1748}});
                        else
                            C_exp.assign({{1727, 1768},
                                          {3192, -2123},
                                          {627, -352},
                                          {2084, 2221},
                                          {3695, -2614},
                                          {960, -1006},
                                          {3077, 3532},
                                          {4622, -3216},
                                          {1547, -1608}});
                    }
                }
                if(symlt_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign({{1989, 662},
                                          {1824, 789},
                                          {941, -126},
                                          {-160, 2177},
                                          {-2187, 2192},
                                          {-3396, 4134},
                                          {785, 1696},
                                          {268, 258},
                                          {1779, 2912}});

                        else
                            C_exp.assign({{361, -2046},
                                          {620, -1935},
                                          {1, -834},
                                          {1684, 181},
                                          {2645, 2024},
                                          {4544, 2922},
                                          {1497, -1112},
                                          {464, 14},
                                          {2683, -2072}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign({{-2105, 1280},
                                          {1488, 3493},
                                          {31, 532},
                                          {48, 521},
                                          {2207, 2914},
                                          {916, 678},
                                          {-3963, 2556},
                                          {1058, 1860},
                                          {1479, 1572}});
                        else
                            C_exp.assign({{1723, 1780},
                                          {3300, -2055},
                                          {763, -56},
                                          {32, 281},
                                          {2603, -2250},
                                          {452, -946},
                                          {3017, 3648},
                                          {1958, -952},
                                          {1363, -1624}});
                    }
                }
                break;
            case 3:
                m   = 5;
                k   = 5;
                n   = 5;
                nnz = 8;
                csr_val.assign(
                    {{2, 52}, {52, 20}, {-13, 8}, {-8, 5}, {-2, 20}, {5, 12}, {5, 40}, {-1, 5}});
                csr_col_ind.assign({1, 3, 1, 4, 2, 2, 3, 4});
                csr_row_ptr.assign({0, 2, 3, 4, 5, 8});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                alpha = {1, -2};
                beta  = {-5, 7};
                C.assign({{1, 2},     {22, -5},  {54, 20},   {-113, 80}, {7.0, 56},
                          {-2, 25},   {3, 2},    {54, 80},   {-1, 5},    {87, -51},
                          {10, 240},  {24, 523}, {528, 206}, {-13, 8.0}, {-89, 57},
                          {-23, 233}, {55, 28},  {5, 10},    {-1, 5},    {87, -51.0},
                          {1, 2},     {2, 52},   {52, 20},   {-13, 8},   {-8, 5}});
                B.assign({{1, 2},     {22, -5},  {54, 20},   {-113, 80}, {7.0, 56},
                          {-2, 25},   {3, 2},    {54, 80},   {-1, 5},    {87, -51},
                          {10, 240},  {24, 523}, {528, 206}, {-13, 8.0}, {-89, 57},
                          {-23, 233}, {55, 28},  {5, 10},    {-1, 5},    {87, -51.0},
                          {1, 2},     {2, 52},   {52, 20},   {-13, 8},   {-8, 5}});
                if(symut_non_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign(
                                {{-1123, 17375},  {171, 1172},    {-1572, 537},    {265, -1091},
                                 {-1068, 1496},   {385, 761},     {-1500, 2673},   {415, 1703},
                                 {1886, 2436},    {-510, 2308},   {-24814, 57288}, {-31951, 25858},
                                 {-5457, 911},    {21089, 21109}, {-4424, 10482},  {3298, 4826},
                                 {-14880, 25793}, {1150, 1710},   {17426, 23336},  {862, 1139},
                                 {-2827, 7433},   {-2126, 238},   {-521, 106},     {269, -31},
                                 {-418, 1040}});
                        else
                            C_exp.assign(
                                {{6021, 14247},   {-613, 460},    {-1152, -653},   {265, -1291},
                                 {-1528, -634},   {249, -327},    {1452, -2431},   {-1835, 1853},
                                 {3046, 516},     {-3040, 318},   {25946, -54112}, {-1471, -42222},
                                 {-3107, 661},    {29889, 1509},  {-10574, 832},   {-4950, -7078},
                                 {12824, -22055}, {-1100, 1860},  {28586, 5616},   {-1388, 1039},
                                 {3525, -3711},   {-1358, -1978}, {-311, 86},      {269, -231},
                                 {-1048, 100}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign(
                                {{16025, 25919},  {7559, -1509},  {2774, 11850},  {-13, -165},
                                 {14963, -13461}, {-1011, 128},   {2484, 645},    {1376, 6766},
                                 {-16021, 3005},  {-29, 9941},    {-1770, -1105}, {-4869, -2301},
                                 {-4398, 3798},   {-185, -388},   {-75, -1066},   {-1256, -1226},
                                 {1133, -2063},   {6553, -2711},  {-3706, 16820}, {5270, 5428},
                                 {-5044, 712},    {-11655, 1786}, {-3342, 12308}, {-358, -407},
                                 {-1477, -1847}});

                        else
                            C_exp.assign(
                                {{30201, 3207},    {3863, -6677},  {-138, -11406}, {995, -1461},
                                 {-17437, -11301}, {-547, -1160},  {-2676, -715},  {-8224, -6434},
                                 {15915, -2027},   {739, -2195},   {-1770, -1155}, {-4389, -3361},
                                 {-5238, 2878},    {155, -418},    {135, -1086},   {-1256, -1426},
                                 {-827, -2543},    {3033, -6471},  {8534, 14940},  {6950, 668},
                                 {-2844, -4238},   {-6425, -9974}, {-12682, 1988}, {322, -467},
                                 {1083, -2117}});
                    }
                }
                if(symut_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign(
                                {{-1118, 17375},  {-53, 390},     {-1478, 449},    {312, -785},
                                 {-620, 985},     {433, 790},     {-1434, 2561},   {629, 1675},
                                 {1895, 2443},    {-1665, 1933},  {-24324, 57508}, {-13171, 23948},
                                 {-4517, 61},     {21092, 21143}, {-3199, 10827},  {3741, 5105},
                                 {-13982, 23757}, {1175, 1710},   {17435, 23343},  {-293, 764},
                                 {-2822, 7433},   {-258, 62},     {-429, 22},      {272, 3},
                                 {-309, 1072}});

                        else
                            C_exp.assign(
                                {{6026, 14247},   {-53, -130},   {-1058, -741},   {312, -985},
                                 {-1500, 45},     {297, -298},   {1582, -2431},   {-1621, 1825},
                                 {3055, 523},     {-1945, -207}, {26436, -53892}, {9709, -27012},
                                 {-2167, -189},   {29892, 1543}, {-11699, 1427},  {-4507, -6799},
                                 {15034, -22315}, {-1075, 1860}, {28595, 5623},   {-293, 514},
                                 {3530, -3711},   {-258, -458},  {-219, 2},       {272, -197},
                                 {-1149, 152}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign(
                                {{16030, 25919},  {7571, -1558},  {2868, 11762},  {34, 141},
                                 {15082, -13419}, {-107, 150},    {2550, 533},    {4148, 4662},
                                 {-15839, 3031},  {-2039, 6911},  {-1280, -885},  {-3799, -1826},
                                 {-3458, 2948},   {-182, -354},   {-50, -831},    {-813, -947},
                                 {1244, -2145},   {6578, -2711},  {-3697, 16827}, {5255, 5203},
                                 {-5034, 687},    {-11203, 1352}, {-3578, 11680}, {-182, -354},
                                 {-1368, -1815}});
                        else
                            C_exp.assign(
                                {{30206, 3207},    {3875, -6726},  {-44, -11494},  {1042, -1155},
                                 {-17318, -11259}, {-107, -370},   {-2546, -715},  {-5004, -5114},
                                 {15985, -1857},   {2329, -5465},  {-1280, -935},  {-3319, -2886},
                                 {-4298, 2028},    {158, -384},    {160, -851},    {-813, -1147},
                                 {-716, -2625},    {3058, -6471},  {8543, 14947},  {6935, 443},
                                 {-2834, -4213},   {-6453, -9348}, {-12078, 2280}, {158, -384},
                                 {982, -2065}});
                    }
                }

                if(symlt_non_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign(
                                {{-19, -3},      {161, 912},    {-6533, 2244},  {492, 5835},
                                 {-11235, 4420}, {-165, -139},  {-88, 119},     {1637, -1161},
                                 {9027, 2579},   {2233, 3837},  {-1730, -1130}, {-21491, -62},
                                 {-7463, 4133},  {5854, 22544}, {12401, 6302},  {-1516, -1326},
                                 {-1258, 2199},  {2372, -1154}, {8845, -1257},  {952, 1709},
                                 {-19, -3},      {-2136, -22},  {-1328, 385},   {675, 2062},
                                 {21, 882}});

                        else
                            C_exp.assign({{-19, -3},      {-623, 720},     {6715, -2492},
                                          {332, -7445},   {11553, -2786},  {-165, -139},
                                          {-152, 7},      {-3483, -1161},  {-10093, -4781},
                                          {-129, -1869},  {-1730, -1130},  {-13891, -17182},
                                          {-463, 3413},   {-9346, -17056}, {-2929, -16748},
                                          {-1516, -1326}, {-2570, 423},    {-2748, -1154},
                                          {-9155, -1057}, {-738, 539},     {-19, -3},
                                          {-1368, -1718}, {536, 217},      {-1005, -1778},
                                          {935, -1586}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign(
                                {{-19, -3},       {-75, 179},     {-410, 278},   {5, -1191},
                                 {-427, -231},    {-1021, -132},  {-88, 119},    {-3388, 2054},
                                 {-203, -51},     {1917, 3669},   {-8171, 7232}, {-2409, 1449},
                                 {-2664, 3850},   {-542, 241},    {4334, -629},  {-6871, 8234},
                                 {-13501, 25175}, {18845, 23745}, {-2061, 250},  {-5658, 1079},
                                 {-9159, 26117},  {2765, 19481},  {14953, 8838}, {-792, 451},
                                 {6128, -342}});
                        else
                            C_exp.assign(
                                {{-19, -3},       {-75, 179},       {-410, 278},     {5, -1191},
                                 {-427, -231},    {-557, -900},     {-152, 7},       {-3836, -1370},
                                 {-91, -195},     {-1683, 3909},    {2989, -10608},  {-4537, -5535},
                                 {-4680, 642},    {554, -191},      {-4162, -77},    {1929, -11766},
                                 {9339, -26105},  {-21875, -21215}, {2019, -110},    {5422, -81},
                                 {18441, -21133}, {8085, -16139},   {-6287, -16642}, {924, -371},
                                 {-6022, 238}});
                    }
                }
                if(symlt_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign(
                                {{-14, -3},      {-63, 130},    {-6439, 2156}, {539, 6141},
                                 {-10787, 3909}, {-117, -110},  {-22, 7},      {1851, -1189},
                                 {9036, 2586},   {1078, 3462},  {-1240, -910}, {-2711, -1972},
                                 {-6523, 3283},  {5857, 22578}, {13626, 6647}, {-1073, -1047},
                                 {-360, 163},    {2397, -1154}, {8854, -1250}, {-203, 1334},
                                 {-14, -3},      {-268, -198},  {-1236, 301},  {678, 2096},
                                 {130, 914}});

                        else
                            C_exp.assign(
                                {{-14, -3},       {-63, 130},      {6809, -2580},   {379, -7139},
                                 {11581, -2107},  {-117, -110},    {-22, 7},        {-3269, -1189},
                                 {-10084, -4774}, {966, -2394},    {-1240, -910},   {-2711, -1972},
                                 {477, 2563},     {-9343, -17022}, {-4054, -16153}, {-1073, -1047},
                                 {-360, 163},     {-2723, -1154},  {-9146, -1050},  {357, 14},
                                 {-14, -3},       {-268, -198},    {628, 133},      {-1002, -1744},
                                 {834, -1534}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none || op == aoclsparse_operation_transpose)
                            C_exp.assign(
                                {{-14, -3},       {-63, 130},     {-316, 190},   {52, -885},
                                 {-308, -189},    {-117, -110},   {-22, 7},      {-616, -50},
                                 {-21, -25},      {-93, 639},     {-7681, 7452}, {-1339, 1924},
                                 {-1724, 3000},   {-539, 275},    {4359, -394},  {-6428, 8513},
                                 {-13390, 25093}, {18870, 23745}, {-2052, 257},  {-5673, 854},
                                 {-9149, 26092},  {3217, 19047},  {14717, 8210}, {-616, 504},
                                 {6237, -310}});
                        else
                            C_exp.assign(
                                {{-14, -3},       {-63, 130},       {-316, 190},     {52, -885},
                                 {-308, -189},    {-117, -110},     {-22, 7},        {-616, -50},
                                 {-21, -25},      {-93, 639},       {3479, -10388},  {-3467, -5060},
                                 {-3740, -208},   {557, -157},      {-4137, 158},    {2372, -11487},
                                 {9450, -26187},  {-21850, -21215}, {2028, -103},    {5407, -306},
                                 {18451, -21108}, {8057, -15513},   {-5683, -16350}, {760, -288},
                                 {-6123, 290}});
                    }
                }
                break;

            // Test case for Hermitian matrix
            case 4:
                m = 3, k = 3, n = 3, nnz = 9;
                csr_val.assign(
                    {{1, 0}, {2, 3}, {3, 4}, {4, 5}, {5, 0}, {6, 7}, {7, 8}, {8, 9}, {9, 0}});
                csr_col_ind.assign({0, 1, 2, 0, 1, 2, 0, 1, 2});
                csr_row_ptr.assign({0, 3, 6, 9});
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                B.assign({{1, 2},
                          {22, -5},
                          {54, 20},
                          {-113, 80},
                          {7.0, 56},
                          {-2, 25},
                          {3, 2},
                          {54, 80},
                          {-1, 5}});
                alpha = {2, -2};
                beta  = {3, 2};
                C.assign(
                    {{-1, 0}, {-2, 7}, {3, 0}, {4, 5}, {5, 8}, {-6, 0}, {1, 0}, {-2, 0}, {3, 0}});
                if(symut_non_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            C_exp.assign({{949.0, 382.0},
                                          {1532.0, 361.0},
                                          {1193.0, -1186.0},
                                          {-184.0, 1329.0},
                                          {1553.0, 2140.0},
                                          {3184.0, 1602.0},
                                          {369.0, 976.0},
                                          {1312.0, 350.0},
                                          {2075.0, -1496.0}});
                        else
                            C_exp.assign({{81.0, -1006.0},
                                          {568.0, -1675.0},
                                          {1933.0, -662.0},
                                          {836.0, 205.0},
                                          {-7.0, 1100.0},
                                          {-1276.0, 2838.0},
                                          {777.0, -696.0},
                                          {1492.0, 298.0},
                                          {1363.0, 2336.0}});
                    }
                    else //Row major
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            C_exp.assign({{-1249.0, 608.0},
                                          {568.0, 1765.0},
                                          {63.0, 252.0},
                                          {-236.0, 1997.0},
                                          {1779.0, 2502.0},
                                          {268.0, -194.0},
                                          {2425.0, 2744.0},
                                          {3734.0, -128.0},
                                          {907.0, -680.0}});
                        else
                            C_exp.assign({{1051.0, 924.0},
                                          {1572.0, -1135.0},
                                          {483.0, -88.0},
                                          {-276.0, 1893.0},
                                          {2831.0, -1046.0},
                                          {844.0, 582.0},
                                          {-2995.0, 1868.0},
                                          {2794.0, 1908.0},
                                          {695.0, 1148.0}});
                    }
                }
                if(symut_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            C_exp.assign({{949, 382},
                                          {1396, 577},
                                          {9, -642},
                                          {-184, 1329},
                                          {1049, 1748},
                                          {2816, 1170},
                                          {369, 976},
                                          {240, 142},
                                          {2011, -1592}});
                        else
                            C_exp.assign({{81, -1006},
                                          {432, -1459},
                                          {749, -118},
                                          {836, 205},
                                          {-511, 708},
                                          {-1644, 2406},
                                          {777, -696},
                                          {420, 90},
                                          {1299, 2240}});
                    }
                    else //Row major
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            C_exp.assign({{-1249.0, 608.0},
                                          {568.0, 1765.0},
                                          {63.0, 252.0},
                                          {28.0, 453.0},
                                          {1275.0, 2110.0},
                                          {84.0, -410.0},
                                          {2345.0, 2760.0},
                                          {1590.0, -544.0},
                                          {843.0, -776.0}});
                        else
                            C_exp.assign({{1051.0, 924.0},
                                          {1572.0, -1135.0},
                                          {483.0, -88.0},
                                          {-12.0, 349.0},
                                          {2327.0, -1438.0},
                                          {660.0, 366.0},
                                          {-3075.0, 1884.0},
                                          {650.0, 1492.0},
                                          {631.0, 1052.0}});
                    }
                }
                if(symlt_non_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            C_exp.assign({{361.0, -2046.0},
                                          {736.0, -2091.0},
                                          {2125.0, -670.0},
                                          {1684.0, 181.0},
                                          {-711.0, 1756.0},
                                          {-3028.0, 4566.0},
                                          {1497.0, -1112.0},
                                          {1556.0, 322.0},
                                          {1843.0, 3008.0}});
                        else
                            C_exp.assign({{1989.0, 662.0},
                                          {1980.0, 513.0},
                                          {1185.0, -1378.0},
                                          {-160.0, 2177.0},
                                          {2177.0, 3244.0},
                                          {4912.0, 3354.0},
                                          {785.0, 1696.0},
                                          {1320.0, 366.0},
                                          {2747.0, -1976.0}});
                    }
                    else //Row major
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            C_exp.assign({{1723.0, 1780.0},
                                          {3300.0, -2055.0},
                                          {763.0, -56.0},
                                          {-252.0, 1885.0},
                                          {3647.0, -1518.0},
                                          {1316.0, 750.0},
                                          {-3883.0, 2540.0},
                                          {3202.0, 2276.0},
                                          {1543.0, 1668.0}});
                        else
                            C_exp.assign({{-2105.0, 1280.0},
                                          {1488.0, 3493.0},
                                          {31.0, 532.0},
                                          {-196.0, 2005.0},
                                          {2171.0, 2966.0},
                                          {420.0, -586.0},
                                          {3097.0, 3632.0},
                                          {4102.0, -536.0},
                                          {1427.0, -1528.0}});
                    }
                }
                if(symlt_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            C_exp.assign({{361, -2046},
                                          {600, -1875},
                                          {941, -126},
                                          {1684, 181},
                                          {-1215, 1364},
                                          {-3396, 4134},
                                          {1497, -1112},
                                          {484, 114},
                                          {1779, 2912}});
                        else
                            C_exp.assign({{1989, 662},
                                          {1844, 729},
                                          {1, -834},
                                          {-160, 2177},
                                          {1673, 2852},
                                          {4544, 2922},
                                          {785, 1696},
                                          {248, 158},
                                          {2683, -2072}});
                    }
                    else //Row major
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            C_exp.assign({{1723.0, 1780.0},
                                          {3300.0, -2055.0},
                                          {763.0, -56.0},
                                          {12.0, 341.0},
                                          {3143.0, -1910.0},
                                          {1132.0, 534.0},
                                          {-3963.0, 2556.0},
                                          {1058.0, 1860.0},
                                          {1479.0, 1572.0}});
                        else
                            C_exp.assign({{-2105.0, 1280.0},
                                          {1488.0, 3493.0},
                                          {31.0, 532.0},
                                          {68.0, 461.0},
                                          {1667.0, 2574.0},
                                          {236.0, -802.0},
                                          {3017.0, 3648.0},
                                          {1958.0, -952.0},
                                          {1363.0, -1624.0}});
                    }
                }
                break;

            // CSC datasets: csr_col_ind holds col_ptr, csr_row_ptr holds row_ind.
            case 5:
                // 3x3 general CSC, complex; A = [[(1,1),(3,0),0],[0,(2,2),0],[0,0,(4,0)]]
                m = 3, k = 3, n = 3, nnz = 4;
                alpha = T{1, 0}, beta = T{0, 0};
                csr_col_ind.assign({0, 1, 3, 4});
                csr_row_ptr.assign({0, 0, 1, 2});
                csr_val.assign({{1, 1}, {3, 0}, {2, 2}, {4, 0}});
                B.assign({{1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}});
                C.assign(9, T{});
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                // op_none -> A; op_t -> A^T; op_h -> A^H=conj(A^T)
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{1, 1},
                                      {0, 0},
                                      {0, 0},
                                      {3, 0},
                                      {2, 2},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {4, 0}});
                    else if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{1, 1},
                                      {3, 0},
                                      {0, 0},
                                      {0, 0},
                                      {2, 2},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {4, 0}});
                    else
                        C_exp.assign({{1, -1},
                                      {3, 0},
                                      {0, 0},
                                      {0, 0},
                                      {2, -2},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {4, 0}});
                }
                else
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{1, 1},
                                      {3, 0},
                                      {0, 0},
                                      {0, 0},
                                      {2, 2},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {4, 0}});
                    else if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{1, 1},
                                      {0, 0},
                                      {0, 0},
                                      {3, 0},
                                      {2, 2},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {4, 0}});
                    else
                        C_exp.assign({{1, -1},
                                      {0, 0},
                                      {0, 0},
                                      {3, 0},
                                      {2, -2},
                                      {0, 0},
                                      {0, 0},
                                      {0, 0},
                                      {4, 0}});
                }
                break;
            case 6:
                // 3x3 symmetric CSC (full, real-valued complex); same A as real id=10
                // CSC fill=upper -> upper of A; CSC fill=lower -> lower of A
                m = 3, k = 3, n = 3, nnz = 9;
                alpha = T{1, 0}, beta = T{0, 0};
                csr_col_ind.assign({0, 3, 6, 9});
                csr_row_ptr.assign({0, 1, 2, 0, 1, 2, 0, 1, 2});
                csr_val.assign(
                    {{1, 0}, {4, 0}, {7, 0}, {2, 0}, {5, 0}, {8, 0}, {3, 0}, {6, 0}, {9, 0}});
                B.assign({{1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}});
                C.assign(9, T{});
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                // All ops equivalent for symmetric (all-real vals); row=col C_exp since result symmetric
                if(symut_non_unit)
                    C_exp.assign(
                        {{1, 0}, {2, 0}, {3, 0}, {2, 0}, {5, 0}, {6, 0}, {3, 0}, {6, 0}, {9, 0}});
                if(symut_unit)
                    C_exp.assign(
                        {{1, 0}, {2, 0}, {3, 0}, {2, 0}, {1, 0}, {6, 0}, {3, 0}, {6, 0}, {1, 0}});
                if(symlt_non_unit)
                    C_exp.assign(
                        {{1, 0}, {4, 0}, {7, 0}, {4, 0}, {5, 0}, {8, 0}, {7, 0}, {8, 0}, {9, 0}});
                if(symlt_unit)
                    C_exp.assign(
                        {{1, 0}, {4, 0}, {7, 0}, {4, 0}, {1, 0}, {8, 0}, {7, 0}, {8, 0}, {1, 0}});
                break;
            case 7:
                // 3x3 hermitian CSC (full); A = [[(4,0),(2,-1),(1,2)],[(2,1),(3,0),(0,0)],[(1,-2),(0,0),(5,0)]]
                // A^T is also Hermitian; both fill modes expand to A^T for op_none/op_h, A for op_t
                m = 3, k = 3, n = 3, nnz = 9;
                alpha = T{1, 0}, beta = T{0, 0};
                csr_col_ind.assign({0, 3, 6, 9});
                csr_row_ptr.assign({0, 1, 2, 0, 1, 2, 0, 1, 2});
                // columns of A: col0={(4,0),(2,1),(1,-2)}, col1={(2,-1),(3,0),(0,0)}, col2={(1,2),(0,0),(5,0)}
                csr_val.assign(
                    {{4, 0}, {2, 1}, {1, -2}, {2, -1}, {3, 0}, {0, 0}, {1, 2}, {0, 0}, {5, 0}});
                B.assign({{1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}});
                C.assign(9, T{});
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                // Both fills produce same result (A is Hermitian, A^H=A); diag=unit replaces diagonal with 1
                // op_none/op_h: C = A*I = A; op_t: C = A^T*I = A^T
                // row-major flat: A = {(4,0),(2,-1),(1,2),(2,1),(3,0),(0,0),(1,-2),(0,0),(5,0)}
                // col-major flat: A = {(4,0),(2,1),(1,-2),(2,-1),(3,0),(0,0),(1,2),(0,0),(5,0)}
                if(symut_non_unit || symlt_non_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            // C = A, stored col-major
                            C_exp.assign({{4, 0},
                                          {2, 1},
                                          {1, -2},
                                          {2, -1},
                                          {3, 0},
                                          {0, 0},
                                          {1, 2},
                                          {0, 0},
                                          {5, 0}});
                        else
                            // C = A^T, stored col-major
                            C_exp.assign({{4, 0},
                                          {2, -1},
                                          {1, 2},
                                          {2, 1},
                                          {3, 0},
                                          {0, 0},
                                          {1, -2},
                                          {0, 0},
                                          {5, 0}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            // C = A, stored row-major
                            C_exp.assign({{4, 0},
                                          {2, -1},
                                          {1, 2},
                                          {2, 1},
                                          {3, 0},
                                          {0, 0},
                                          {1, -2},
                                          {0, 0},
                                          {5, 0}});
                        else
                            // C = A^T, stored row-major
                            C_exp.assign({{4, 0},
                                          {2, 1},
                                          {1, -2},
                                          {2, -1},
                                          {3, 0},
                                          {0, 0},
                                          {1, 2},
                                          {0, 0},
                                          {5, 0}});
                    }
                }
                if(symut_unit || symlt_unit)
                {
                    if(order == aoclsparse_order_column)
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            // C = A (unit diag), stored col-major
                            C_exp.assign({{1, 0},
                                          {2, 1},
                                          {1, -2},
                                          {2, -1},
                                          {1, 0},
                                          {0, 0},
                                          {1, 2},
                                          {0, 0},
                                          {1, 0}});
                        else
                            // C = A^T (unit diag), stored col-major
                            C_exp.assign({{1, 0},
                                          {2, -1},
                                          {1, 2},
                                          {2, 1},
                                          {1, 0},
                                          {0, 0},
                                          {1, -2},
                                          {0, 0},
                                          {1, 0}});
                    }
                    else
                    {
                        if(op == aoclsparse_operation_none
                           || op == aoclsparse_operation_conjugate_transpose)
                            // C = A (unit diag), stored row-major
                            C_exp.assign({{1, 0},
                                          {2, -1},
                                          {1, 2},
                                          {2, 1},
                                          {1, 0},
                                          {0, 0},
                                          {1, -2},
                                          {0, 0},
                                          {1, 0}});
                        else
                            // C = A^T (unit diag), stored row-major
                            C_exp.assign({{1, 0},
                                          {2, 1},
                                          {1, -2},
                                          {2, -1},
                                          {1, 0},
                                          {0, 0},
                                          {1, 2},
                                          {0, 0},
                                          {1, 0}});
                    }
                }
                break;
            case 8:
                // 4x3 non-square general CSC (complex); A has entries at rows 1,2,3 cols 0,1,2
                m = 4, k = 3, n = 2, nnz = 3;
                alpha = {1, 0};
                beta  = {0, 0};
                csr_col_ind.assign({0, 1, 2, 3}); // col_ptr for CSC
                csr_row_ptr.assign({1, 2, 3}); // row_ind for CSC
                csr_val.assign({{2, 1}, {4, 2}, {8, 4}});
                B.assign(12, T{1, 0});
                C.assign({T{0, 0}});
                transform(csr_col_ind.begin(),
                          csr_col_ind.end(),
                          csr_col_ind.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                transform(csr_row_ptr.begin(),
                          csr_row_ptr.end(),
                          csr_row_ptr.begin(),
                          [base](aoclsparse_int &d) { return d + base; });
                if(order == aoclsparse_order_column)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign(
                            {{0, 0}, {2, 1}, {4, 2}, {8, 4}, {0, 0}, {2, 1}, {4, 2}, {8, 4}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{2, 1}, {4, 2}, {8, 4}, {2, 1}, {4, 2}, {8, 4}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({{2, -1}, {4, -2}, {8, -4}, {2, -1}, {4, -2}, {8, -4}});
                }
                else
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign(
                            {{0, 0}, {0, 0}, {2, 1}, {2, 1}, {4, 2}, {4, 2}, {8, 4}, {8, 4}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{2, 1}, {2, 1}, {4, 2}, {4, 2}, {8, 4}, {8, 4}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({{2, -1}, {2, -1}, {4, -2}, {4, -2}, {8, -4}, {8, -4}});
                }
                break;
            }
        }
    }

    // Function to set the ldb, ldc values depending on sizes m, k, n
    void set_mm_dim(aoclsparse_operation &op,
                    aoclsparse_order     &order,
                    aoclsparse_int       &m,
                    aoclsparse_int       &k,
                    aoclsparse_int       &n,
                    aoclsparse_int       &A_m,
                    aoclsparse_int       &A_n,
                    aoclsparse_int       &B_m,
                    aoclsparse_int       &B_n,
                    aoclsparse_int       &C_m,
                    aoclsparse_int       &C_n,
                    aoclsparse_int       &ldb,
                    aoclsparse_int       &ldc)
    {
        A_m = (op == aoclsparse_operation_none ? m : k);
        A_n = (op == aoclsparse_operation_none ? k : m);
        B_m = (op == aoclsparse_operation_none ? k : m);
        B_n = n;
        C_m = A_m;
        C_n = B_n;
        ldb = (order == aoclsparse_order_column ? B_m : B_n);
        ldc = (order == aoclsparse_order_column ? C_m : C_n);
    }

    // Consolidated invalid-argument and quick-return tests.
    void test_csrmm_invalid()
    {
        aoclsparse_index_base       zero      = aoclsparse_index_base_zero;
        aoclsparse_index_base       one       = aoclsparse_index_base_one;
        aoclsparse_operation        op_n      = aoclsparse_operation_none;
        aoclsparse_operation        op_t      = aoclsparse_operation_transpose;
        aoclsparse_order            col_order = aoclsparse_order_column;
        aoclsparse_int              m, k, n, nnz;
        std::vector<float>          val, B, C, C_exp;
        std::vector<aoclsparse_int> col, row;
        float                       alpha, beta;
        aoclsparse_mat_descr        descr;
        aoclsparse_matrix           A;

        // id=6: m=2 k=3 n=2, col-major, csr_val={42}, row_ptr={0,0,1}, col_ind={1}
        // col-major op=none: min ldb=k=3, min ldc=m=2
        init<float>(
            op_n, col_order, m, k, n, nnz, val, col, row, alpha, beta, B, C, C_exp, zero, 6);

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(
            aoclsparse_create_csr<float>(&A, zero, m, k, nnz, row.data(), col.data(), val.data()),
            aoclsparse_status_success);

        // Group 1: nullptr arguments
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, nullptr, descr, col_order, B.data(), n, k, beta, C.data(), m, 0),
                  aoclsparse_status_invalid_pointer)
            << "null A";
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, nullptr, col_order, B.data(), n, k, beta, C.data(), m, 0),
                  aoclsparse_status_invalid_pointer)
            << "null descr";
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, nullptr, n, k, beta, C.data(), m, 0),
                  aoclsparse_status_invalid_pointer)
            << "null B";
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, B.data(), n, k, beta, nullptr, m, 0),
                  aoclsparse_status_invalid_pointer)
            << "null C";

        // Group 2: invalid enum arguments
        EXPECT_EQ(aoclsparse_csrmm<float>((aoclsparse_operation)99,
                                          alpha,
                                          A,
                                          descr,
                                          col_order,
                                          B.data(),
                                          n,
                                          k,
                                          beta,
                                          C.data(),
                                          m,
                                          0),
                  aoclsparse_status_invalid_value)
            << "bad op";
        EXPECT_EQ(
            aoclsparse_csrmm<float>(
                op_n, alpha, A, descr, (aoclsparse_order)2, B.data(), n, k, beta, C.data(), m, 0),
            aoclsparse_status_invalid_value)
            << "bad order";

        // Group 3: index base mismatch — descr=one vs CSR=zero
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, one), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, B.data(), n, k, beta, C.data(), m, 0),
                  aoclsparse_status_invalid_value)
            << "base mismatch";
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, zero), aoclsparse_status_success);

        // Group 4: wrong ldb/ldc for op=none col-major (min ldb=k=3, min ldc=m=2)
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, B.data(), n, k - 1, beta, C.data(), m, 0),
                  aoclsparse_status_invalid_size)
            << "ldb too small op=none";
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, B.data(), n, k, beta, C.data(), m - 1, 0),
                  aoclsparse_status_invalid_size)
            << "ldc too small op=none";

        // Group 5: wrong ldb/ldc for op=trans col-major (min ldb=m=2, min ldc=k=3)
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_t, alpha, A, descr, col_order, B.data(), n, m - 1, beta, C.data(), k, 0),
                  aoclsparse_status_invalid_size)
            << "ldb too small op=trans";
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_t, alpha, A, descr, col_order, B.data(), n, m, beta, C.data(), k - 1, 0),
                  aoclsparse_status_invalid_size)
            << "ldc too small op=trans";

        // Group 6: ELL format — not_implemented; restore to CSR
        A->input_format = aoclsparse_ell_mat;
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, B.data(), n, k, beta, C.data(), m, 0),
                  aoclsparse_status_not_implemented)
            << "ELL format";
        A->input_format = aoclsparse_csr_mat;

        // Group 7 (no-op): real hermitian CSR is supported (treated as symmetric)
        // after BLUE phase fixes; not_implemented is no longer expected here.

        // Group 8: wrong scalar type — double matrix passed to scsrmm
        {
            double            dval[] = {42.0};
            aoclsparse_matrix A_d;
            ASSERT_EQ(
                aoclsparse_create_csr<double>(&A_d, zero, m, k, nnz, row.data(), col.data(), dval),
                aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_scsrmm(
                          op_n, 1.f, A_d, descr, col_order, B.data(), n, k, 0.f, C.data(), m),
                      aoclsparse_status_wrong_type)
                << "double matrix in scsrmm";
            aoclsparse_destroy(&A_d);
        }

        // Group 9: CSC matrix — ldb too small fires invalid_size
        {
            // 3x3 CSC: col-major op=none requires ldb >= 3
            aoclsparse_int    csc_col[] = {0, 1, 3, 4};
            aoclsparse_int    csc_row[] = {0, 0, 1, 2};
            float             csc_val[] = {1.f, 3.f, 2.f, 4.f};
            aoclsparse_matrix A_csc;
            ASSERT_EQ(
                aoclsparse_create_csc<float>(&A_csc, zero, 3, 3, 4, csc_col, csc_row, csc_val),
                aoclsparse_status_success);
            EXPECT_EQ(
                aoclsparse_csrmm<float>(
                    op_n, alpha, A_csc, descr, col_order, B.data(), n, 2, beta, C.data(), 3, 0),
                aoclsparse_status_invalid_size)
                << "CSC ldb too small";
            aoclsparse_destroy(&A_csc);
        }

        // Group 10: quick returns — all must succeed
        A->m = 0;
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, B.data(), n, k, beta, C.data(), m, 0),
                  aoclsparse_status_success)
            << "m=0";
        A->m = m;
        A->n = 0;
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, B.data(), n, k, beta, C.data(), m, 0),
                  aoclsparse_status_success)
            << "n=0";
        A->n = k;
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, alpha, A, descr, col_order, B.data(), 0, k, beta, C.data(), m, 0),
                  aoclsparse_status_success)
            << "n_rhs=0";
        EXPECT_EQ(aoclsparse_csrmm<float>(
                      op_n, 0.f, A, descr, col_order, B.data(), n, k, 1.f, C.data(), m, 0),
                  aoclsparse_status_success)
            << "alpha=0 beta=1";

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(&A);
    }

    // tests for ldb and ldc greater than minimum
    template <typename T>
    void test_csrmm_greater_ld()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              kid = 1;
        aoclsparse_int              m, k, n, nnz;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;
        if(can_exec_avx512_tests())
            kid = 3;

        init<T>(op,
                order,
                m,
                k,
                n,
                nnz,
                csr_val,
                csr_col_ind,
                csr_row_ptr,
                alpha,
                beta,
                B,
                C,
                C_exp,
                base,
                6);
        B.assign({1.0, -2.0, 3.0, 0, 0, 0, 4.0, 5.0, -6.0, 0, 0, 0});
        C.assign({0.1, 0.2, 0, 0, 0.3, 0.4, 0, 0});
        C_exp.assign({1.120000, -190.960000, 0, 0, 3.360000, 487.480000, 0, 0});
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        // expect success for ldb = k*2 and ldc = m*2
        EXPECT_EQ(aoclsparse_csrmm<T>(
                      op, alpha, A, descr, order, B.data(), n, k * 2, beta, C.data(), m * 2, kid),
                  aoclsparse_status_success);
        if constexpr(std::is_same_v<T, double>)
            EXPECT_DOUBLE_EQ_VEC(m * 2 * n, C, C_exp);
        if constexpr(std::is_same_v<T, float>)
            EXPECT_FLOAT_EQ_VEC(m * 2 * n, C, C_exp);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(&A);
    }

    // Single inner kernel used by all csrmm success test functions.
    template <typename T>
    void run_csrmm_case(aoclsparse_operation    op,
                        aoclsparse_order        order,
                        aoclsparse_index_base   base,
                        aoclsparse_int          kid,
                        aoclsparse_memory_usage mem_policy,
                        aoclsparse_matrix_type  mat_type,
                        aoclsparse_fill_mode    fill,
                        aoclsparse_diag_type    diag,
                        bool                    is_csc,
                        int                     id)
    {
        // clang-format off
        auto op_s    = op    == aoclsparse_operation_none             ? "none"
                     : op    == aoclsparse_operation_transpose        ? "trans"
                                                                      : "conj_trans";
        auto ord_s   = order == aoclsparse_order_row                  ? "row"   : "col";
        auto base_s  = base  == aoclsparse_index_base_zero            ? "zero"  : "one";
        auto mem_s   = mem_policy == aoclsparse_memory_usage_minimal  ? "min"   : "unres";
        auto mat_s   = mat_type == aoclsparse_matrix_type_general     ? "gen"
                     : mat_type == aoclsparse_matrix_type_symmetric   ? "sym"   : "herm";
        auto fill_s  = fill  == aoclsparse_fill_mode_lower            ? "lower" : "upper";
        auto diag_s  = diag  == aoclsparse_diag_type_non_unit         ? "non_unit" : "unit";
        // clang-format on
        std::ostringstream trace;
        trace << "id=" << id << " op=" << op_s << " ord=" << ord_s << " base=" << base_s
              << " kid=" << kid << " mem=" << mem_s << " mat=" << mat_s << " fill=" << fill_s
              << " diag=" << diag_s << " fmt=" << (is_csc ? "csc" : "csr");
        SCOPED_TRACE(trace.str());

        aoclsparse_int              m, k, n, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind, csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B, C, C_exp;

        init<T>(op,
                order,
                m,
                k,
                n,
                nnz,
                csr_val,
                csr_col_ind,
                csr_row_ptr,
                alpha,
                beta,
                B,
                C,
                C_exp,
                base,
                id,
                mat_type,
                fill,
                diag);
        set_mm_dim(op, order, m, k, n, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
        C.resize(C_m * C_n);

        aoclsparse_mat_descr descr;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mat_type), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descr, diag), aoclsparse_status_success);

        aoclsparse_matrix A;
        if(is_csc)
            // CSC convention in init<T>: csr_col_ind=col_ptr, csr_row_ptr=row_ind
            ASSERT_EQ(
                aoclsparse_create_csc<T>(
                    &A, base, m, k, nnz, csr_col_ind.data(), csr_row_ptr.data(), csr_val.data()),
                aoclsparse_status_success);
        else
            ASSERT_EQ(
                aoclsparse_create_csr(
                    &A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_memory_hint(A, mem_policy), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mm_hint(A, op, descr, 1000), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_csrmm<T>(
                      op, alpha, A, descr, order, B.data(), C_n, ldb, beta, C.data(), ldc, kid),
                  aoclsparse_status_success);

        if constexpr(std::is_same_v<T, double>)
            EXPECT_DOUBLE_EQ_VEC(C_m * C_n, C, C_exp);
        if constexpr(std::is_same_v<T, float>)
            EXPECT_FLOAT_EQ_VEC(C_m * C_n, C, C_exp);
        if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
        {
            const aoclsparse_int       sz = C_m * C_n;
            const std::complex<float> *res_ptr
                = reinterpret_cast<const std::complex<float> *>(C.data());
            const std::complex<float> *exp_ptr
                = reinterpret_cast<const std::complex<float> *>(C_exp.data());
            std::vector<std::complex<float>> res_c(res_ptr, res_ptr + sz);
            std::vector<std::complex<float>> exp_c(exp_ptr, exp_ptr + sz);
            EXPECT_COMPLEX_FLOAT_EQ_VEC(sz, res_c, exp_c);
        }
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
        {
            const aoclsparse_int        sz = C_m * C_n;
            const std::complex<double> *res_ptr
                = reinterpret_cast<const std::complex<double> *>(C.data());
            const std::complex<double> *exp_ptr
                = reinterpret_cast<const std::complex<double> *>(C_exp.data());
            std::vector<std::complex<double>> res_c(res_ptr, res_ptr + sz);
            std::vector<std::complex<double>> exp_c(exp_ptr, exp_ptr + sz);
            EXPECT_COMPLEX_DOUBLE_EQ_VEC(sz, res_c, exp_c);
        }

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(&A);
    }

    // General CSR tests. Fixed: base=zero, kid=0. Spot-checks: base=one and kid=1.
    template <typename T>
    void test_csrmm_success()
    {
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            for(int id = 0; id < 5; id++)
                for(aoclsparse_operation op : {aoclsparse_operation_none,
                                               aoclsparse_operation_transpose,
                                               aoclsparse_operation_conjugate_transpose})
                    for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                        for(aoclsparse_memory_usage mem_policy :
                            {aoclsparse_memory_usage_minimal, aoclsparse_memory_usage_unrestricted})
                            run_csrmm_case<T>(op,
                                              order,
                                              aoclsparse_index_base_zero,
                                              0,
                                              mem_policy,
                                              aoclsparse_matrix_type_general,
                                              aoclsparse_fill_mode_lower,
                                              aoclsparse_diag_type_non_unit,
                                              false,
                                              id);
            // Spot-check: one-based index
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_general,
                              aoclsparse_fill_mode_lower,
                              aoclsparse_diag_type_non_unit,
                              false,
                              0);
            // Spot-check: kid=1 (always valid on all machines)
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_zero,
                              1,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_general,
                              aoclsparse_fill_mode_lower,
                              aoclsparse_diag_type_non_unit,
                              false,
                              0);
        }
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            for(int id = 0; id < 2; id++)
                for(aoclsparse_operation op : {aoclsparse_operation_none,
                                               aoclsparse_operation_transpose,
                                               aoclsparse_operation_conjugate_transpose})
                    for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                        for(aoclsparse_memory_usage mem_policy :
                            {aoclsparse_memory_usage_minimal, aoclsparse_memory_usage_unrestricted})
                            run_csrmm_case<T>(op,
                                              order,
                                              aoclsparse_index_base_zero,
                                              0,
                                              mem_policy,
                                              aoclsparse_matrix_type_general,
                                              aoclsparse_fill_mode_lower,
                                              aoclsparse_diag_type_non_unit,
                                              false,
                                              id);
            // Spot-check: one-based index
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_general,
                              aoclsparse_fill_mode_lower,
                              aoclsparse_diag_type_non_unit,
                              false,
                              0);
            // Spot-check: kid=1 (always valid on all machines)
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_zero,
                              1,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_general,
                              aoclsparse_fill_mode_lower,
                              aoclsparse_diag_type_non_unit,
                              false,
                              0);
        }
    }

    // Symmetric and Hermitian CSR tests. Fixed: base=zero, kid=0. Spot-check: base=one per sub-type.
    template <typename T>
    void test_csrmm_symm_success()
    {
        // Real symmetric: ids 7-8
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            for(int id = 7; id < 9; id++)
                for(aoclsparse_fill_mode fill :
                    {aoclsparse_fill_mode_upper, aoclsparse_fill_mode_lower})
                    for(aoclsparse_diag_type diag :
                        {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})
                        for(aoclsparse_operation op : {aoclsparse_operation_none,
                                                       aoclsparse_operation_transpose,
                                                       aoclsparse_operation_conjugate_transpose})
                            for(aoclsparse_order order :
                                {aoclsparse_order_row, aoclsparse_order_column})
                                for(aoclsparse_memory_usage mem_policy :
                                    {aoclsparse_memory_usage_minimal,
                                     aoclsparse_memory_usage_unrestricted})
                                    run_csrmm_case<T>(op,
                                                      order,
                                                      aoclsparse_index_base_zero,
                                                      0,
                                                      mem_policy,
                                                      aoclsparse_matrix_type_symmetric,
                                                      fill,
                                                      diag,
                                                      false,
                                                      id);
            // Spot-check: one-based index
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_symmetric,
                              aoclsparse_fill_mode_upper,
                              aoclsparse_diag_type_non_unit,
                              false,
                              7);
        }
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            // Complex symmetric: ids 2-3
            for(int id = 2; id < 4; id++)
                for(aoclsparse_fill_mode fill :
                    {aoclsparse_fill_mode_upper, aoclsparse_fill_mode_lower})
                    for(aoclsparse_diag_type diag :
                        {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})
                        for(aoclsparse_operation op : {aoclsparse_operation_none,
                                                       aoclsparse_operation_transpose,
                                                       aoclsparse_operation_conjugate_transpose})
                            for(aoclsparse_order order :
                                {aoclsparse_order_row, aoclsparse_order_column})
                                for(aoclsparse_memory_usage mem_policy :
                                    {aoclsparse_memory_usage_minimal,
                                     aoclsparse_memory_usage_unrestricted})
                                    run_csrmm_case<T>(op,
                                                      order,
                                                      aoclsparse_index_base_zero,
                                                      0,
                                                      mem_policy,
                                                      aoclsparse_matrix_type_symmetric,
                                                      fill,
                                                      diag,
                                                      false,
                                                      id);
            // Spot-check: one-based index
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_symmetric,
                              aoclsparse_fill_mode_upper,
                              aoclsparse_diag_type_non_unit,
                              false,
                              2);
            // Complex Hermitian: id=4 hardcoded
            for(aoclsparse_fill_mode fill :
                {aoclsparse_fill_mode_upper, aoclsparse_fill_mode_lower})
                for(aoclsparse_diag_type diag :
                    {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})
                    for(aoclsparse_operation op : {aoclsparse_operation_none,
                                                   aoclsparse_operation_transpose,
                                                   aoclsparse_operation_conjugate_transpose})
                        for(aoclsparse_order order :
                            {aoclsparse_order_column, aoclsparse_order_row})
                            for(aoclsparse_memory_usage mem_policy :
                                {aoclsparse_memory_usage_minimal,
                                 aoclsparse_memory_usage_unrestricted})
                                run_csrmm_case<T>(op,
                                                  order,
                                                  aoclsparse_index_base_zero,
                                                  0,
                                                  mem_policy,
                                                  aoclsparse_matrix_type_hermitian,
                                                  fill,
                                                  diag,
                                                  false,
                                                  4);
            // Spot-check: one-based index
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_column,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_hermitian,
                              aoclsparse_fill_mode_upper,
                              aoclsparse_diag_type_non_unit,
                              false,
                              4);
        }
    }

    // CSC general, symmetric, and hermitian tests using datasets from init<T>().
    template <typename T>
    void test_csrmm_csc_success()
    {
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            // General CSC (id=9)
            for(aoclsparse_operation op : {aoclsparse_operation_none,
                                           aoclsparse_operation_transpose,
                                           aoclsparse_operation_conjugate_transpose})
                for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                    for(aoclsparse_memory_usage mem :
                        {aoclsparse_memory_usage_minimal, aoclsparse_memory_usage_unrestricted})
                        run_csrmm_case<T>(op,
                                          order,
                                          aoclsparse_index_base_zero,
                                          0,
                                          mem,
                                          aoclsparse_matrix_type_general,
                                          aoclsparse_fill_mode_lower,
                                          aoclsparse_diag_type_non_unit,
                                          true,
                                          9);
            // Symmetric CSC (id=10)
            for(aoclsparse_fill_mode fill :
                {aoclsparse_fill_mode_upper, aoclsparse_fill_mode_lower})
                for(aoclsparse_diag_type diag :
                    {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})
                    for(aoclsparse_operation op : {aoclsparse_operation_none,
                                                   aoclsparse_operation_transpose,
                                                   aoclsparse_operation_conjugate_transpose})
                        for(aoclsparse_order order :
                            {aoclsparse_order_row, aoclsparse_order_column})
                            for(aoclsparse_memory_usage mem :
                                {aoclsparse_memory_usage_minimal,
                                 aoclsparse_memory_usage_unrestricted})
                                run_csrmm_case<T>(op,
                                                  order,
                                                  aoclsparse_index_base_zero,
                                                  0,
                                                  mem,
                                                  aoclsparse_matrix_type_symmetric,
                                                  fill,
                                                  diag,
                                                  true,
                                                  10);
            // Spot-check: one-based index on general CSC
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_general,
                              aoclsparse_fill_mode_lower,
                              aoclsparse_diag_type_non_unit,
                              true,
                              9);
            // Spot-check: one-based index on symmetric CSC
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_symmetric,
                              aoclsparse_fill_mode_upper,
                              aoclsparse_diag_type_non_unit,
                              true,
                              10);
            // Spot-check: kid=1 on general CSC
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_zero,
                              1,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_general,
                              aoclsparse_fill_mode_lower,
                              aoclsparse_diag_type_non_unit,
                              true,
                              9);
        }
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            // General CSC (id=5)
            for(aoclsparse_operation op : {aoclsparse_operation_none,
                                           aoclsparse_operation_transpose,
                                           aoclsparse_operation_conjugate_transpose})
                for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                    for(aoclsparse_memory_usage mem :
                        {aoclsparse_memory_usage_minimal, aoclsparse_memory_usage_unrestricted})
                        run_csrmm_case<T>(op,
                                          order,
                                          aoclsparse_index_base_zero,
                                          0,
                                          mem,
                                          aoclsparse_matrix_type_general,
                                          aoclsparse_fill_mode_lower,
                                          aoclsparse_diag_type_non_unit,
                                          true,
                                          5);
            // Symmetric CSC (id=6)
            for(aoclsparse_fill_mode fill :
                {aoclsparse_fill_mode_upper, aoclsparse_fill_mode_lower})
                for(aoclsparse_diag_type diag :
                    {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})
                    for(aoclsparse_operation op : {aoclsparse_operation_none,
                                                   aoclsparse_operation_transpose,
                                                   aoclsparse_operation_conjugate_transpose})
                        for(aoclsparse_order order :
                            {aoclsparse_order_row, aoclsparse_order_column})
                            for(aoclsparse_memory_usage mem :
                                {aoclsparse_memory_usage_minimal,
                                 aoclsparse_memory_usage_unrestricted})
                                run_csrmm_case<T>(op,
                                                  order,
                                                  aoclsparse_index_base_zero,
                                                  0,
                                                  mem,
                                                  aoclsparse_matrix_type_symmetric,
                                                  fill,
                                                  diag,
                                                  true,
                                                  6);
            // Hermitian CSC (id=7)
            for(aoclsparse_fill_mode fill :
                {aoclsparse_fill_mode_upper, aoclsparse_fill_mode_lower})
                for(aoclsparse_diag_type diag :
                    {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})
                    for(aoclsparse_operation op : {aoclsparse_operation_none,
                                                   aoclsparse_operation_transpose,
                                                   aoclsparse_operation_conjugate_transpose})
                        for(aoclsparse_order order :
                            {aoclsparse_order_column, aoclsparse_order_row})
                            for(aoclsparse_memory_usage mem :
                                {aoclsparse_memory_usage_minimal,
                                 aoclsparse_memory_usage_unrestricted})
                                run_csrmm_case<T>(op,
                                                  order,
                                                  aoclsparse_index_base_zero,
                                                  0,
                                                  mem,
                                                  aoclsparse_matrix_type_hermitian,
                                                  fill,
                                                  diag,
                                                  true,
                                                  7);
            // Spot-check: one-based index on general CSC
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_general,
                              aoclsparse_fill_mode_lower,
                              aoclsparse_diag_type_non_unit,
                              true,
                              5);
            // Spot-check: one-based index on symmetric CSC
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_symmetric,
                              aoclsparse_fill_mode_upper,
                              aoclsparse_diag_type_non_unit,
                              true,
                              6);
            // Spot-check: one-based index on hermitian CSC
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_one,
                              0,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_hermitian,
                              aoclsparse_fill_mode_upper,
                              aoclsparse_diag_type_non_unit,
                              true,
                              7);
            // Spot-check: kid=1 on general CSC
            run_csrmm_case<T>(aoclsparse_operation_none,
                              aoclsparse_order_row,
                              aoclsparse_index_base_zero,
                              1,
                              aoclsparse_memory_usage_minimal,
                              aoclsparse_matrix_type_general,
                              aoclsparse_fill_mode_lower,
                              aoclsparse_diag_type_non_unit,
                              true,
                              5);
        }
    }

    // Non-square CSC: diagonal spread across op × order × base × mem × kid.
    template <typename T>
    void test_csrmm_csc_nonsquare()
    {
        const aoclsparse_index_base zero = aoclsparse_index_base_zero;
        const aoclsparse_index_base one  = aoclsparse_index_base_one;
        // real id=11, complex id=8
        const int id = (std::is_same_v<T, aoclsparse_float_complex>
                        || std::is_same_v<T, aoclsparse_double_complex>)
                           ? 8
                           : 11;

        // 1. none  / row / zero / unrestricted / kid=0
        run_csrmm_case<T>(aoclsparse_operation_none,
                          aoclsparse_order_row,
                          zero,
                          0,
                          aoclsparse_memory_usage_unrestricted,
                          aoclsparse_matrix_type_general,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_diag_type_non_unit,
                          true,
                          id);
        // 2. trans / col / one  / minimal       / kid=0
        run_csrmm_case<T>(aoclsparse_operation_transpose,
                          aoclsparse_order_column,
                          one,
                          0,
                          aoclsparse_memory_usage_minimal,
                          aoclsparse_matrix_type_general,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_diag_type_non_unit,
                          true,
                          id);
        // 3. conj_t / row / zero / minimal      / kid=1
        run_csrmm_case<T>(aoclsparse_operation_conjugate_transpose,
                          aoclsparse_order_row,
                          zero,
                          1,
                          aoclsparse_memory_usage_minimal,
                          aoclsparse_matrix_type_general,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_diag_type_non_unit,
                          true,
                          id);
        // 4. none  / col / zero / unrestricted  / kid=0
        run_csrmm_case<T>(aoclsparse_operation_none,
                          aoclsparse_order_column,
                          zero,
                          0,
                          aoclsparse_memory_usage_unrestricted,
                          aoclsparse_matrix_type_general,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_diag_type_non_unit,
                          true,
                          id);
        // 5. trans / row / one  / unrestricted  / kid=0
        run_csrmm_case<T>(aoclsparse_operation_transpose,
                          aoclsparse_order_row,
                          one,
                          0,
                          aoclsparse_memory_usage_unrestricted,
                          aoclsparse_matrix_type_general,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_diag_type_non_unit,
                          true,
                          id);
        // 6. conj_t / col / zero / minimal      / kid=0
        run_csrmm_case<T>(aoclsparse_operation_conjugate_transpose,
                          aoclsparse_order_column,
                          zero,
                          0,
                          aoclsparse_memory_usage_minimal,
                          aoclsparse_matrix_type_general,
                          aoclsparse_fill_mode_lower,
                          aoclsparse_diag_type_non_unit,
                          true,
                          id);
    }

    // Test for success and verify results against Dense GEMM results.
    template <typename T>
    void test_csrmm_all_success(aoclsparse_int             m_a,
                                aoclsparse_int             n_a,
                                aoclsparse_int             nnz_a,
                                aoclsparse_index_base      b_a,
                                aoclsparse_operation       op_a,
                                const aoclsparse_mat_descr descr,
                                aoclsparse_order           layout,
                                aoclsparse_int             column,
                                aoclsparse_int             ldb,
                                aoclsparse_int             ldc,
                                aoclsparse_int             offset = 0,
                                aoclsparse_int             scalar = 0,
                                aoclsparse_int             kid    = 0)
    {
        aoclsparse_int               m_c, n_c, m_b, n_b, lda, dense_c_sz;
        const aoclsparse_fill_mode   fill     = descr->fill_mode;
        const aoclsparse_diag_type   diag     = descr->diag_type;
        const aoclsparse_matrix_type mat_type = descr->type;
        aoclsparse_int               m, n, k;
        aoclsparse_operation         op_b = aoclsparse_operation_none;
        aoclsparse_matrix_init       mat_init_type
            = aoclsparse_matrix_random; // random matrix generation by default
        if(mat_type == aoclsparse_matrix_type_hermitian)
        {
            mat_init_type = aoclsparse_matrix_random_herm_diag_dom;
        }

        CBLAS_ORDER blis_layout;
        aoclsparse_seedrand();

        if(op_a == aoclsparse_operation_none)
        {
            m_b = n_a;
            m_c = m_a;
        }
        else
        {
            m_b = m_a;
            m_c = n_a;
        }
        n_b = n_c = column;

        m = m_c;
        n = column;
        k = m_b;
        std::vector<T> dense_a(m_a * n_a), dense_b(m_b * n_b), dense_c, dense_c_exp, dense_c_bkp;
        if(layout == aoclsparse_order_row)
        {
            blis_layout = CblasRowMajor;
            lda         = n_a;
            dense_b.resize(m_b * ldb);
            aoclsparse_init<T>(dense_b, m_b, ldb, m_b);
            dense_c.resize(m_c * ldc);
            aoclsparse_init<T>(dense_c, m_c, ldc, m_c);
            dense_c_sz = m_c * ldc;
            dense_c_bkp.resize(dense_c_sz);
            dense_c_bkp = dense_c;
        }
        else
        {
            blis_layout = CblasColMajor;
            lda         = m_a;
            dense_b.resize(ldb * n_b);
            aoclsparse_init<T>(dense_b, ldb, n_b, ldb);
            dense_c.resize(ldc * n_c);
            aoclsparse_init<T>(dense_c, ldc, n_c, ldc);
            dense_c_sz = ldc * n_c;
            dense_c_bkp.resize(dense_c_sz);
            dense_c_bkp = dense_c;
        }

        tolerance_t<T> abserr = sqrt(std::numeric_limits<tolerance_t<T>>::epsilon());

        std::vector<T>              val_a;
        std::vector<aoclsparse_int> col_ind_a;
        std::vector<aoclsparse_int> row_ptr_a;
        bool                        issymm;
        ASSERT_EQ(aoclsparse_init_csr_matrix(row_ptr_a,
                                             col_ind_a,
                                             val_a,
                                             m_a,
                                             n_a,
                                             nnz_a,
                                             b_a,
                                             mat_init_type,
                                             "",
                                             issymm,
                                             true,
                                             aoclsparse_fully_sorted),
                  aoclsparse_status_success);
        if(val_a.size() == 0)
            val_a.reserve(1);
        if(col_ind_a.size() == 0)
            col_ind_a.reserve(1);
        if(row_ptr_a.size() == 0)
            row_ptr_a.reserve(1);
        aoclsparse_matrix A;

        for(aoclsparse_memory_usage mem_policy :
            {aoclsparse_memory_usage_minimal, aoclsparse_memory_usage_unrestricted})
        {
            dense_c = dense_c_bkp;
            ASSERT_EQ(
                aoclsparse_create_csr(
                    &A, b_a, m_a, n_a, nnz_a, row_ptr_a.data(), col_ind_a.data(), val_a.data()),
                aoclsparse_status_success);
            aoclsparse_mat_descr descrA;
            ASSERT_EQ(aoclsparse_create_mat_descr(&descrA), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descrA, b_a), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_type(descrA, mat_type), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_diag_type(descrA, diag), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_fill_mode(descrA, fill), aoclsparse_status_success);

            ASSERT_EQ(aoclsparse_set_memory_hint(A, mem_policy), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mm_hint(A, op_a, descrA, 1000 /*Hint*/),
                      aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

            dense_c_exp = dense_c;
            if(dense_c.size() == 0)
            {
                dense_c.reserve(1);
                dense_c_exp.reserve(1);
            }

            T alpha, beta;
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                         || std::is_same_v<T, aoclsparse_float_complex>)
            {
                switch(scalar)
                {
                case 0:
                    alpha = {-1, 2};
                    beta  = {2, -1};
                    break;
                case 1:
                    alpha = {0, 0};
                    beta  = {2, -1};
                    break;
                case 2:
                    alpha = {0, 0};
                    beta  = {0, 0};
                    break;
                case 3:
                    alpha = {0, 0};
                    beta  = {1, 0};
                    break;
                case 4:
                    alpha = {1, 0};
                    beta  = {0, 0};
                    break;
                }
            }
            else
            {
                switch(scalar)
                {
                case 0:
                    alpha = 3.0;
                    beta  = -2.0;
                    break;
                case 1:
                    alpha = 0.;
                    beta  = -2.0;
                    break;
                case 2:
                    alpha = 0.;
                    beta  = 0.;
                    break;
                case 3:
                    alpha = 0.;
                    beta  = 1.0;
                    break;
                case 4:
                    alpha = 1.;
                    beta  = 0.0;
                    break;
                }
            }
            EXPECT_EQ(aoclsparse_csrmm<T>(op_a,
                                          alpha,
                                          A,
                                          descrA,
                                          layout,
                                          dense_b.data(),
                                          column,
                                          ldb,
                                          beta,
                                          dense_c.data() + offset,
                                          ldc,
                                          kid),
                      aoclsparse_status_success);

            EXPECT_EQ(aoclsparse_csr2dense(m_a,
                                           n_a,
                                           descrA,
                                           val_a.data(),
                                           row_ptr_a.data(),
                                           col_ind_a.data(),
                                           dense_a.data(),
                                           lda,
                                           layout),
                      aoclsparse_status_success);

            if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
            {
                blis::gemm(blis_layout,
                           (CBLAS_TRANSPOSE)op_a,
                           (CBLAS_TRANSPOSE)op_b,
                           (int64_t)m,
                           (int64_t)n,
                           (int64_t)k,
                           *reinterpret_cast<const std::complex<float> *>(&alpha),
                           (std::complex<float> const *)dense_a.data(),
                           (int64_t)lda,
                           (std::complex<float> const *)dense_b.data(),
                           (int64_t)ldb,
                           *reinterpret_cast<const std::complex<float> *>(&beta),
                           (std::complex<float> *)dense_c_exp.data() + offset,
                           (int64_t)ldc);
                EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                        ((std::complex<float> *)dense_c.data()),
                                        ((std::complex<float> *)dense_c_exp.data()),
                                        abserr);
            }
            else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
            {
                blis::gemm(blis_layout,
                           (CBLAS_TRANSPOSE)op_a,
                           (CBLAS_TRANSPOSE)op_b,
                           (int64_t)m,
                           (int64_t)n,
                           (int64_t)k,
                           *reinterpret_cast<const std::complex<double> *>(&alpha),
                           (std::complex<double> const *)dense_a.data(),
                           (int64_t)lda,
                           (std::complex<double> const *)dense_b.data(),
                           (int64_t)ldb,
                           *reinterpret_cast<const std::complex<double> *>(&beta),
                           (std::complex<double> *)dense_c_exp.data() + offset,
                           (int64_t)ldc);

                EXPECT_COMPLEX_ARR_NEAR(dense_c_sz,
                                        ((std::complex<double> *)dense_c.data()),
                                        ((std::complex<double> *)dense_c_exp.data()),
                                        abserr);
            }

            else
            {
                blis::gemm(blis_layout,
                           (CBLAS_TRANSPOSE)op_a,
                           (CBLAS_TRANSPOSE)op_b,
                           (int64_t)m,
                           (int64_t)n,
                           (int64_t)k,
                           (T)alpha,
                           (T const *)dense_a.data(),
                           (int64_t)lda,
                           (T const *)dense_b.data(),
                           (int64_t)ldb,
                           (T)beta,
                           (T *)dense_c_exp.data() + offset,
                           (int64_t)ldc);
                EXPECT_ARR_NEAR(dense_c_sz, dense_c.data(), dense_c_exp.data(), abserr);
            }

            aoclsparse_destroy_mat_descr(descrA);
            aoclsparse_destroy(&A);
        }
    }

    TEST(csrmm, InvalidChecks)
    {
        test_csrmm_invalid();
    }

    TEST(csrmm, GreaterLDDouble)
    {
        test_csrmm_greater_ld<double>();
    }
    TEST(csrmm, GreaterLDFloat)
    {
        test_csrmm_greater_ld<float>();
    }
    TEST(csrmm, SuccessTypeDouble)
    {
        test_csrmm_success<double>();
    }
    TEST(csrmm, SuccessTypeFloat)
    {
        test_csrmm_success<float>();
    }
    TEST(csrmm, SuccessTypeComplexFloat)
    {
        test_csrmm_success<aoclsparse_float_complex>();
    }

    TEST(csrmm, SuccessTypeComplexDouble)
    {
        test_csrmm_success<aoclsparse_double_complex>();
    }
    TEST(csrmm, SuccessTypeSymmDouble)
    {
        test_csrmm_symm_success<double>();
    }
    TEST(csrmm, SuccessTypeSymmFloat)
    {
        test_csrmm_symm_success<float>();
    }
    TEST(csrmm, SuccessTypeSymmFloatComplex)
    {
        test_csrmm_symm_success<aoclsparse_float_complex>();
    }
    TEST(csrmm, SuccessTypeSymmDoubleComplex)
    {
        test_csrmm_symm_success<aoclsparse_double_complex>();
    }

    TEST(csrmm, SuccessDoubleGRnd)
    {
        aoclsparse_mat_descr descrA;
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_set_mat_index_base(descrA, zero);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_general);
        aoclsparse_set_mat_diag_type(descrA, aoclsparse_diag_type_non_unit);
        aoclsparse_set_mat_fill_mode(descrA, aoclsparse_fill_mode_lower);

        // m_a, n_a, nnz_a, b_a, op_a, descr, layout, column, ldb, ldc, offset=0, scalar=0
        test_csrmm_all_success<double>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 0);
        test_csrmm_all_success<double>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<double>(3, 4, 8, one, op_n, descrA, row, 2, 4, 6, 0, 0, 2);
            test_csrmm_all_success<double>(3, 4, 8, one, op_n, descrA, row, 2, 4, 6, 0, 0, 3);
        }
        test_csrmm_all_success<double>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 1, 0);
        test_csrmm_all_success<double>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 2, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<double>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 3, 2);
            test_csrmm_all_success<double>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 3, 3);
        }
        test_csrmm_all_success<double>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 4, 0);
        test_csrmm_all_success<double>(1, 4, 3, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<double>(1, 1, 1, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 2);
            test_csrmm_all_success<double>(1, 1, 1, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 3);
        }
        test_csrmm_all_success<double>(10, 10, 33, zero, op_n, descrA, row, 2, 8, 7, 0, 0, 0);
        test_csrmm_all_success<double>(4, 3, 8, zero, op_t, descrA, row, 2, 4, 6, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<double>(3, 4, 8, one, op_t, descrA, row, 2, 4, 6, 0, 0, 2);
            test_csrmm_all_success<double>(3, 4, 8, one, op_t, descrA, row, 2, 4, 6, 0, 0, 3);
        }
        test_csrmm_all_success<double>(5, 8, 18, one, op_t, descrA, row, 4, 4, 4, 0, 0, 0);

        aoclsparse_destroy_mat_descr(descrA);
    }
    TEST(csrmm, SuccessFloatGRnd)
    {
        aoclsparse_mat_descr descrA;
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_set_mat_index_base(descrA, zero);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_general);
        aoclsparse_set_mat_diag_type(descrA, aoclsparse_diag_type_non_unit);
        aoclsparse_set_mat_fill_mode(descrA, aoclsparse_fill_mode_lower);

        // m_a, n_a, nnz_a, b_a, op_a, descr, layout, column, ldb, ldc, offset=0, scalar=0, kid
        test_csrmm_all_success<float>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 0);
        test_csrmm_all_success<float>(3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<float>(1, 4, 3, one, op_n, descrA, row, 2, 4, 6, 0, 0, 2);
            test_csrmm_all_success<float>(1, 4, 3, one, op_n, descrA, row, 2, 4, 6, 0, 0, 3);
        }

        test_csrmm_all_success<float>(1, 1, 1, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 0);
        test_csrmm_all_success<float>(10, 10, 33, zero, op_n, descrA, row, 2, 8, 7, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<float>(11, 4, 18, zero, op_t, descrA, row, 2, 4, 6, 0, 0, 2);
            test_csrmm_all_success<float>(11, 4, 18, zero, op_t, descrA, row, 2, 4, 6, 0, 0, 3);
        }

        test_csrmm_all_success<float>(3, 4, 8, one, op_t, descrA, row, 2, 4, 6, 0, 0, 0);
        test_csrmm_all_success<float>(5, 8, 18, one, op_t, descrA, row, 4, 4, 4, 0, 0, 1);

        aoclsparse_destroy_mat_descr(descrA);
    }
    TEST(csrmm, SuccessCDoubleGRnd)
    {
        aoclsparse_mat_descr descrA;
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_set_mat_index_base(descrA, zero);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_general);
        aoclsparse_set_mat_diag_type(descrA, aoclsparse_diag_type_non_unit);
        aoclsparse_set_mat_fill_mode(descrA, aoclsparse_fill_mode_lower);

        // m_a, n_a, nnz_a, b_a, op_a, descr, layout, column, ldb, ldc, offset=0, scalar=0, kid
        test_csrmm_all_success<aoclsparse_double_complex>(
            3, 4, 8, zero, op_n, descrA, row, 2, 4, 2, 0, 0, 0);
        test_csrmm_all_success<aoclsparse_double_complex>(
            3, 1, 3, zero, op_n, descrA, row, 2, 2, 6, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<aoclsparse_double_complex>(
                3, 3, 8, one, op_n, descrA, row, 2, 4, 5, 0, 0, 2);
            test_csrmm_all_success<aoclsparse_double_complex>(
                3, 3, 8, one, op_n, descrA, row, 2, 4, 5, 0, 0, 3);
        }
        test_csrmm_all_success<aoclsparse_double_complex>(
            1, 4, 4, zero, op_n, descrA, row, 4, 4, 6, 0, 1, 0);
        test_csrmm_all_success<aoclsparse_double_complex>(
            4, 1, 4, zero, op_n, descrA, row, 2, 4, 6, 0, 2, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<aoclsparse_double_complex>(
                3, 4, 8, zero, op_n, descrA, row, 1, 4, 6, 0, 3, 2);
            test_csrmm_all_success<aoclsparse_double_complex>(
                3, 4, 8, zero, op_n, descrA, row, 1, 4, 6, 0, 3, 3);
        }
        test_csrmm_all_success<aoclsparse_double_complex>(
            3, 4, 8, zero, op_n, descrA, row, 3, 4, 6, 0, 4, 0);
        test_csrmm_all_success<aoclsparse_double_complex>(
            1, 4, 3, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<aoclsparse_double_complex>(
                1, 1, 1, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 2);
            test_csrmm_all_success<aoclsparse_double_complex>(
                1, 1, 1, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 3);
        }
        test_csrmm_all_success<aoclsparse_double_complex>(
            10, 10, 33, zero, op_n, descrA, row, 2, 8, 7, 0, 0, 0);
        test_csrmm_all_success<aoclsparse_double_complex>(
            4, 3, 8, zero, op_t, descrA, row, 2, 4, 6, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<aoclsparse_double_complex>(
                3, 4, 8, one, op_t, descrA, row, 2, 4, 6, 0, 0, 2);
            test_csrmm_all_success<aoclsparse_double_complex>(
                3, 4, 8, one, op_t, descrA, row, 2, 4, 6, 0, 0, 3);
        }
        test_csrmm_all_success<aoclsparse_double_complex>(
            5, 8, 18, one, op_t, descrA, row, 4, 4, 4, 0, 0, 0);
        test_csrmm_all_success<aoclsparse_double_complex>(
            3, 4, 8, zero, op_n, descrA, row, 2, 4, 6, 0, 0, 1);
        if(can_exec_avx512_tests())
        {
            test_csrmm_all_success<aoclsparse_double_complex>(
                3, 4, 8, zero, op_h, descrA, row, 2, 4, 6, 0, 0, 2);
            test_csrmm_all_success<aoclsparse_double_complex>(
                3, 4, 8, zero, op_h, descrA, row, 2, 4, 6, 0, 0, 3);
        }
        test_csrmm_all_success<aoclsparse_double_complex>(
            5, 6, 13, zero, op_n, descrA, row, 2, 5, 7, 2, 0, 0);
        test_csrmm_all_success<aoclsparse_double_complex>(
            5, 6, 13, zero, op_n, descrA, col, 2, 15, 7, 2, 0, 1);
        aoclsparse_destroy_mat_descr(descrA);
    }

    TEST(csrmm, SuccessDoubleSymRnd)
    {
        aoclsparse_mat_descr descrA;
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_set_mat_index_base(descrA, zero);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_symmetric);

        for(aoclsparse_fill_mode fill : {aoclsparse_fill_mode_lower, aoclsparse_fill_mode_upper})
        {
            aoclsparse_set_mat_fill_mode(descrA, fill);
            for(aoclsparse_diag_type diag_type :
                {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})

            {
                aoclsparse_set_mat_diag_type(descrA, diag_type);
                for(aoclsparse_order order : {row, col})

                {
                    // m_a, n_a, nnz_a, b_a, op_a, descr, layout, column, ldb, ldc, offset=0, scalar=0
                    test_csrmm_all_success<double>(4, 4, 8, zero, op_n, descrA, order, 2, 14, 16);
                    test_csrmm_all_success<double>(7, 7, 28, zero, op_n, descrA, order, 5, 10, 10);
                    test_csrmm_all_success<double>(7, 7, 28, one, op_t, descrA, order, 5, 15, 11);
                    test_csrmm_all_success<double>(
                        7, 7, 28, one, op_t, descrA, order, 5, 18, 18, 4, 2);
                    test_csrmm_all_success<double>(
                        11, 11, 28, zero, op_t, descrA, order, 5, 25, 16, 0, 1);
                    test_csrmm_all_success<double>(1, 1, 1, zero, op_n, descrA, order, 2, 13, 13);
                }
            }
        }
        aoclsparse_destroy_mat_descr(descrA);
    }

    TEST(csrmm, SuccessCDoubleSymRnd)
    {
        aoclsparse_mat_descr descrA;
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_set_mat_index_base(descrA, zero);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_symmetric);

        for(aoclsparse_fill_mode fill : {aoclsparse_fill_mode_lower, aoclsparse_fill_mode_upper})
        {
            aoclsparse_set_mat_fill_mode(descrA, fill);
            for(aoclsparse_diag_type diag_type :
                {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})
            {
                aoclsparse_set_mat_diag_type(descrA, diag_type);
                for(aoclsparse_order order : {row, col})
                {
                    // m_a, n_a, nnz_a, b_a, op_a, descr, layout, column, ldb, ldc, offset=0, scalar=0
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        3, 3, 9, zero, op_n, descrA, order, 2, 14, 16);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        7, 7, 28, zero, op_n, descrA, order, 5, 10, 10);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        7, 7, 28, one, op_h, descrA, order, 5, 15, 11);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        7, 7, 28, one, op_t, descrA, order, 5, 18, 18, 4, 2);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        11, 11, 28, zero, op_t, descrA, order, 5, 25, 16, 0, 1);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        1, 1, 1, zero, op_n, descrA, order, 2, 13, 13);
                }
            }
        }
        aoclsparse_destroy_mat_descr(descrA);
    }

    TEST(csrmm, SuccessCDoubleHermRnd)
    {
        aoclsparse_mat_descr descrA;
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_set_mat_index_base(descrA, zero);
        aoclsparse_set_mat_type(descrA, aoclsparse_matrix_type_hermitian);

        for(aoclsparse_fill_mode fill : {aoclsparse_fill_mode_lower, aoclsparse_fill_mode_upper})
        {
            aoclsparse_set_mat_fill_mode(descrA, fill);
            for(aoclsparse_diag_type diag_type :
                {aoclsparse_diag_type_non_unit, aoclsparse_diag_type_unit})

            {
                aoclsparse_set_mat_diag_type(descrA, diag_type);
                for(aoclsparse_order order : {row}) //, col})
                {
                    // m_a, n_a, nnz_a, b_a, op_a, descr, layout, column, ldb, ldc, offset=0, scalar=0
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        4, 4, 8, zero, op_n, descrA, order, 2, 4, 4);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        3, 3, 9, zero, op_h, descrA, order, 3, 3, 3);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        4, 4, 8, zero, op_t, descrA, order, 2, 4, 4);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        7, 7, 28, zero, op_t, descrA, order, 5, 10, 10);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        7, 7, 28, one, op_h, descrA, order, 5, 15, 11);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        7, 7, 28, one, op_h, descrA, order, 5, 18, 18, 4, 2);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        11, 11, 28, zero, op_h, descrA, order, 5, 25, 16, 0, 1);
                    test_csrmm_all_success<aoclsparse_double_complex>(
                        1, 1, 1, zero, op_n, descrA, order, 2, 13, 13);
                }
            }
        }
        aoclsparse_destroy_mat_descr(descrA);
    }

    TEST(csrmm, SuccessCFloatSymAndHermRnd)
    {
        aoclsparse_mat_descr descrA;
        aoclsparse_create_mat_descr(&descrA);
        aoclsparse_set_mat_index_base(descrA, zero);
        aoclsparse_set_mat_diag_type(descrA, aoclsparse_diag_type_non_unit);
        for(aoclsparse_fill_mode fill : {aoclsparse_fill_mode_lower})
        {
            aoclsparse_set_mat_fill_mode(descrA, fill);
            for(aoclsparse_matrix_type mat_type :
                {aoclsparse_matrix_type_symmetric, aoclsparse_matrix_type_hermitian})
            {
                aoclsparse_set_mat_type(descrA, mat_type);
                for(aoclsparse_order order : {row})
                {
                    // m_a, n_a, nnz_a, b_a, op_a, descr, layout, column, ldb, ldc, offset=0, scalar=0
                    test_csrmm_all_success<aoclsparse_float_complex>(
                        3, 3, 9, zero, op_n, descrA, order, 2, 14, 16);
                    test_csrmm_all_success<aoclsparse_float_complex>(
                        7, 7, 28, zero, op_n, descrA, order, 5, 10, 10);
                    test_csrmm_all_success<aoclsparse_float_complex>(
                        7, 7, 28, one, op_h, descrA, order, 5, 15, 11);
                    test_csrmm_all_success<aoclsparse_float_complex>(
                        7, 7, 28, one, op_t, descrA, order, 5, 18, 18, 4, 2);
                    test_csrmm_all_success<aoclsparse_float_complex>(
                        11, 11, 28, zero, op_t, descrA, order, 5, 25, 16, 0, 1);
                    test_csrmm_all_success<aoclsparse_float_complex>(
                        1, 1, 1, zero, op_n, descrA, order, 2, 13, 13);
                }
            }
        }
        aoclsparse_destroy_mat_descr(descrA);
    }

    TEST(csrmm, CSCSuccessFloat)
    {
        test_csrmm_csc_success<float>();
    }
    TEST(csrmm, CSCSuccessDouble)
    {
        test_csrmm_csc_success<double>();
    }
    TEST(csrmm, CSCSuccessFloatComplex)
    {
        test_csrmm_csc_success<aoclsparse_float_complex>();
    }
    TEST(csrmm, CSCSuccessDoubleComplex)
    {
        test_csrmm_csc_success<aoclsparse_double_complex>();
    }
    // RED: non-square CSC — fails before fix (R-1: invalid_size for col-maj op=none; R-4: OOB under ASan)
    TEST(csrmm, CSCNonSquareDouble)
    {
        test_csrmm_csc_nonsquare<double>();
    }
    TEST(csrmm, CSCNonSquareFloat)
    {
        test_csrmm_csc_nonsquare<float>();
    }
    TEST(csrmm, CSCNonSquareDoubleComplex)
    {
        test_csrmm_csc_nonsquare<aoclsparse_double_complex>();
    }
    TEST(csrmm, CSCNonSquareFloatComplex)
    {
        test_csrmm_csc_nonsquare<aoclsparse_float_complex>();
    }
    // Regression: general-type optimized matrix with diag mismatch must not return invalid_pointer
    TEST(csrmm, GeneralOptimizedDiagMismatch)
    {
        // 3x3 general CSR matrix
        //  [2  0  1]
        //  [0  3  0]
        //  [4  0  5]
        aoclsparse_int              m = 3, k = 3, n = 1, nnz = 5;
        std::vector<double>         val     = {2.0, 1.0, 3.0, 4.0, 5.0};
        std::vector<aoclsparse_int> col_ind = {0, 2, 1, 0, 2};
        std::vector<aoclsparse_int> row_ptr = {0, 2, 3, 5};
        std::vector<double>         B(k, 1.0);
        std::vector<double>         C(m, 0.0);

        aoclsparse_mat_descr descr;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general),
                  aoclsparse_status_success);
        // unit diag differs from the default mtx_diag (non_unit) stored post-optimize,
        // triggering aoclsparse_set_mat_diag on a general matrix whose diag_val == nullptr
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_unit),
                  aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(&A,
                                        aoclsparse_index_base_zero,
                                        m,
                                        k,
                                        nnz,
                                        row_ptr.data(),
                                        col_ind.data(),
                                        val.data()),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mm_hint(A, aoclsparse_operation_none, descr, 1000),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

        // Must return success, not aoclsparse_status_invalid_pointer
        EXPECT_EQ(aoclsparse_csrmm(aoclsparse_operation_none,
                                   1.0,
                                   A,
                                   descr,
                                   aoclsparse_order_column,
                                   B.data(),
                                   n,
                                   k,
                                   0.0,
                                   C.data(),
                                   m,
                                   0),
                  aoclsparse_status_success);

        aoclsparse_destroy(&A);
        aoclsparse_destroy_mat_descr(descr);
    }
} // namespace
