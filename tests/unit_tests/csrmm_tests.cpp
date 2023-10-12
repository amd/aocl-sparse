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
namespace
{
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
              aoclsparse_int               id)
    {
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
                                      -16.000000,
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
                                      -16.000000,
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
                                      -16.000000,
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
                                      -16.000000,
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
                        C_exp.assign(
                            {-225.500000, -29.000000, 127.500000, 100.000000,  528.500000,
                             129.000000,  14.500000,  91.000000,  -52.500000,  256.000000,
                             -755.500000, -87.000000, 74.500000,  25.000000,   103.500000,
                             646.000000,  72.500000,  -63.000000, -177.500000, -275.000000,
                             475.500000,  58.000000,  252.500000, 135.000000,  433.000000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({2.500000,   97.000000,   307.500000, 226.000000, 324.500000,
                                      -15.000000, -741.500000, 229.000000, 139.500000, 154.000000,
                                      12.500000,  543.000000,  50.500000,  151.000000, 175.500000,
                                      10.000000,  576.500000,  -57.000000, -57.500000, -245.000000,
                                      7.500000,   436.000000,  192.500000, 423.000000, 625.000000});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign(
                            {-729.500000, 151.000000,  -280.500000, 394.000000,  504.500000,
                             -87.000000,  14.500000,   -29.000000,  43.500000,   58.000000,
                             84.500000,   81.000000,   122.500000,  -149.000000, 247.500000,
                             160.000000,  -167.500000, 15.000000,   -57.500000,  85.000000,
                             499.500000,  196.000000,  36.500000,   -333.000000, 529.000000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign(
                            {2.500000,   -5.000000,   7.500000,   10.000000,   12.500000,
                             39.000000,  -237.500000, 349.000000, 547.500000,  688.000000,
                             240.500000, 279.000000,  2.500000,   -191.000000, 307.500000,
                             142.000000, 168.500000,  213.000000, -225.500000, 445.000000,
                             271.500000, 58.000000,   276.500000, -351.000000, 577.000000});
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
                        C_exp.assign(
                            {60.000000,   64.000000,    249.600000,  9.200000,    112.000000,
                             24.000000,   272.000000,   -164.000000, -72.000000,  336.400000,
                             -18.400000,  52.000000,    -48.000000,  -52.000000,  244.000000,
                             48.000000,   419.200000,   27.600000,   -184.000000, 72.000000,
                             -136.000000, 336.800000,   824.000000,  -503.920000, 36.800000,
                             120.000000,  110.400000,   1848.000000, -568.000000, 776.000000,
                             262.400000,  -73.600000,   224.000000,  816.000000,  1872.000000,
                             2188.000000, 9048.000000,  848.800000,  230.000000,  448.000000,
                             1008.000000, 19024.000000, 767.440000,  215.120000,  20.800000,
                             79.672000,   -594.240000,  2976.000000, -127.200000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign(
                            {120.800000,  -96.000000,   252.000000,  128.000000,   0.000000,
                             5.200000,    -144.000000,  -122.000000, 120.000000,   336.000000,
                             -176.000000, -16.000000,   -6.400000,   24.000000,    196.800000,
                             16.000000,   420.000000,   104.000000,  160.000000,   14.000000,
                             -48.000000,  428.000000,   -744.000000, -504.000000,  1108.800000,
                             227.200000,  13.600000,    4.800000,    -561.600000,  -760.000000,
                             252.000000,  1912.000000,  1104.000000, -30.800000,   624.000000,
                             2652.000000, -8296.000000, 840.000000,  12016.000000, 2096.000000,
                             104.000000,  528.000000,   769.760000,  471.120000,   16.800000,
                             3312.000000, 4654.240000,  34.720000,   240.000000});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign(
                            {184.000000,  20.000000,   292.000000,  576.000000, 2588.000000,
                             -416.000000, 252.000000,  -144.000000, 40.000000,  80.000000,
                             232.000000,  720.000000,  424.000000,  528.000000, 262.000000,
                             354.800000,  424.000000,  -480.000000, 300.800000, -159.200000,
                             268.800000,  9.200000,    -18.400000,  27.600000,  36.800000,
                             46.000000,   -55.200000,  9.200000,    112.000000, 52.000000,
                             -184.000000, 120.000000,  1076.000000, -12.000000, 124.000000,
                             207.840000,  1029.360000, 4.800000,    110.400000, -192.000000,
                             240.000000,  2976.000000, -144.000000, 172.000000, -72.000000,
                             664.000000,  2908.000000, 837.600000,  1216.800000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign(
                            {120.800000,  -122.000000,  196.800000,  428.000000,  870.800000,
                             -502.160000, 126.320000,   112.000000,  8.000000,    -16.000000,
                             -152.000000, -816.000000,  -408.000000, -560.000000, 252.000000,
                             336.000000,  420.000000,   -504.000000, 84.000000,   -168.000000,
                             252.000000,  82.480000,    1220.920000, 65.600000,   368.800000,
                             736.000000,  800.000000,   4152.000000, 245.120000,  1420.480000,
                             70.400000,   227.200000,   -352.000000, 336.000000,  3936.000000,
                             5.200000,    -6.400000,    14.000000,   13.600000,   20.400000,
                             -24.800000,  5.200000,     600.000000,  1128.000000, 240.000000,
                             1440.000000, 13008.000000, 528.000000,  1008.000000});
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
                        C_exp.assign({0.000000,
                                      -27.000000,
                                      -126.000000,
                                      -108.000000,
                                      0.000000,
                                      -9.000000,
                                      -90.000000,
                                      -72.000000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-63.000000,
                                      -54.000000,
                                      -36.000000,
                                      -18.000000,
                                      -72.000000,
                                      -108.000000});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({0.000000,
                                      0.000000,
                                      -27.000000,
                                      -63.000000,
                                      -54.000000,
                                      -18.000000,
                                      -180.000000,
                                      -72.000000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-27.000000,
                                      -9.000000,
                                      -90.000000,
                                      -36.000000,
                                      -144.000000,
                                      -108.000000});
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
                                      -143.000000,
                                      -82.500000,
                                      -55.000000,
                                      -148.500000,
                                      -143.000000,
                                      -66.000000,
                                      -60.500000,
                                      -99.000000,
                                      -77.000000,
                                      -82.500000,
                                      -55.000000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-203.500000, -214.500000, 0.000000, 0.000000,
                                      -71.500000,  -115.500000, 0.000000, 0.000000,
                                      -126.500000, -198.000000, 0.000000, 0.000000,
                                      -132.000000, -181.500000, 0.000000, 0.000000,
                                      -60.500000,  -66.000000,  0.000000, 0.000000,
                                      -82.500000,  -82.500000,  0.000000, 0.000000,
                                      -126.500000, -198.000000, 0.000000, 0.000000});
                }
                else if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({-148.500000,
                                      -148.500000,
                                      -115.500000,
                                      -49.500000,
                                      -181.500000,
                                      -198.000000,
                                      -132.000000,
                                      -115.500000,
                                      -170.500000,
                                      -99.000000,
                                      -38.500000,
                                      -159.500000,
                                      -126.500000,
                                      -121.000000});
                    if(op == aoclsparse_operation_transpose
                       || op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({-115.500000, -137.500000, -93.500000,  -38.500000,
                                      -148.500000, -143.000000, -110.000000, -148.500000,
                                      -247.500000, -132.000000, -49.500000,  -214.500000,
                                      -148.500000, -165.000000, 0.000000,    0.000000,
                                      0.000000,    0.000000,    0.000000,    0.000000,
                                      0.000000,    0.000000,    0.000000,    0.000000,
                                      0.000000,    0.000000,    0.000000,    0.000000});
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
                        C_exp.assign({{-2.000000, -1.000000},
                                      {-11.000000, 12.000000},
                                      {6.000000, 3.000000},
                                      {3.000000, 14.000000},
                                      {2.000000, 21.000000},
                                      {-12.000000, -6.000000},
                                      {2.000000, 1.000000},
                                      {-4.000000, -2.000000},
                                      {6.000000, 3.000000}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{-2.000000, -1.000000},
                                      {-11.000000, 12.000000},
                                      {6.000000, 3.000000},
                                      {3.000000, 14.000000},
                                      {2.000000, 21.000000},
                                      {-12.000000, -6.000000},
                                      {2.000000, 1.000000},
                                      {-4.000000, -2.000000},
                                      {6.000000, 3.000000}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({{-2.000000, -1.000000},
                                      {-11.000000, 12.000000},
                                      {6.000000, 3.000000},
                                      {3.000000, 14.000000},
                                      {2.000000, 21.000000},
                                      {-12.000000, -6.000000},
                                      {2.000000, 1.000000},
                                      {-4.000000, -2.000000},
                                      {6.000000, 3.000000}});
                }
                if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{-2.000000, -1.000000},
                                      {-11.000000, 12.000000},
                                      {6.000000, 3.000000},
                                      {3.000000, 14.000000},
                                      {2.000000, 21.000000},
                                      {-12.000000, -6.000000},
                                      {2.000000, 1.000000},
                                      {-4.000000, -2.000000},
                                      {6.000000, 3.000000}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{-2.000000, -1.000000},
                                      {-11.000000, 12.000000},
                                      {6.000000, 3.000000},
                                      {3.000000, 14.000000},
                                      {2.000000, 21.000000},
                                      {-12.000000, -6.000000},
                                      {2.000000, 1.000000},
                                      {-4.000000, -2.000000},
                                      {6.000000, 3.000000}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({{-2.000000, -1.000000},
                                      {-11.000000, 12.000000},
                                      {6.000000, 3.000000},
                                      {3.000000, 14.000000},
                                      {2.000000, 21.000000},
                                      {-12.000000, -6.000000},
                                      {2.000000, 1.000000},
                                      {-4.000000, -2.000000},
                                      {6.000000, 3.000000}});
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
                        C_exp.assign({{-1104.000000, 17378.000000},  {236.000000, 733.000000},
                                      {-1162.000000, 259.000000},    {1572.000000, 2056.000000},
                                      {-10808.000000, 4651.000000},  {550.000000, 900.000000},
                                      {-59.000000, 108.000000},      {1245.000000, 1725.000000},
                                      {132.000000, 4336.000000},     {2311.000000, 2973.000000},
                                      {-23084.000000, 58418.000000}, {-17710.000000, 2385.000000},
                                      {-1375.000000, -1755.000000},  {15120.000000, 20500.000000},
                                      {12355.000000, 7210.000000},   {4814.000000, 6152.000000},
                                      {-787.000000, 1954.000000},    {1245.000000, 1725.000000},
                                      {-50.000000, 500.000000},      {1030.000000, 845.000000},
                                      {-2808.000000, 7436.000000},   {-1762.000000, 224.000000},
                                      {-121.000000, -158.000000},    {1496.000000, 2008.000000},
                                      {16.000000, 963.000000}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{-0.000000, 0.000000},         {246.000000, 993.000000},
                                      {-6123.000000, 1966.000000},   {-825.000000, 5070.000000},
                                      {-641.000000, 1727.000000},    {-0.000000, 0.000000},
                                      {-1471.000000, 2662.000000},   {2467.000000, -1139.000000},
                                      {10841.000000, 743.000000},    {-432.000000, 1444.000000},
                                      {-0.000000, 0.000000},         {-28170.000000, 28305.000000},
                                      {-3381.000000, 1467.000000},   {11805.000000, 23415.000000},
                                      {-4470.000000, 11390.000000},  {-0.000000, 0.000000},
                                      {-14409.000000, 25548.000000}, {2467.000000, -1139.000000},
                                      {26381.000000, 21643.000000},  {940.000000, 275.000000},
                                      {-0.000000, 0.000000},         {-1752.000000, 484.000000},
                                      {-928.000000, 121.000000},     {-570.000000, 285.000000},
                                      {-423.000000, 1121.000000}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({{-0.000000, 0.000000},         {-538.000000, 281.000000},
                                      {7125.000000, -2770.000000},   {2535.000000, -4650.000000},
                                      {-1101.000000, -403.000000},   {-0.000000, 0.000000},
                                      {1481.000000, -2442.000000},   {-2653.000000, -1139.000000},
                                      {-5999.000000, 23.000000},     {-2962.000000, -546.000000},
                                      {-0.000000, 0.000000},         {2310.000000, -39775.000000},
                                      {3619.000000, 747.000000},     {39405.000000, 1815.000000},
                                      {-10620.000000, 1740.000000},  {-0.000000, 0.000000},
                                      {13295.000000, -22300.000000}, {-2653.000000, -1139.000000},
                                      {19541.000000, 5123.000000},   {-1310.000000, 175.000000},
                                      {-0.000000, 0.000000},         {-984.000000, -1732.000000},
                                      {936.000000, -47.000000},      {1110.000000, -75.000000},
                                      {-1053.000000, 181.000000}});
                }
                if(order == aoclsparse_order_row)
                {
                    if(op == aoclsparse_operation_none)
                        C_exp.assign({{16044.000000, 25922.000000},  {7634.000000, -1688.000000},
                                      {3184.000000, 11572.000000},   {-18.000000, 1026.000000},
                                      {15390.000000, -13230.000000}, {-856.000000, 7.000000},
                                      {-59.000000, 108.000000},      {-2558.000000, 2076.000000},
                                      {-173.000000, -19.000000},     {1995.000000, 2805.000000},
                                      {-40.000000, 25.000000},       {-1088.000000, 146.000000},
                                      {-316.000000, 1132.000000},    {-194.000000, -257.000000},
                                      {-121.000000, -158.000000},    {-5380.000000, 9360.000000},
                                      {-11640.000000, 20450.000000}, {15120.000000, 20500.000000},
                                      {-686.000000, -8.000000},      {-4750.000000, 30.000000},
                                      {-9140.000000, 26120.000000},  {3139.000000, 19727.000000},
                                      {15353.000000, 8574.000000},   {-801.000000, 582.000000},
                                      {6123.000000, -261.000000}});
                    if(op == aoclsparse_operation_transpose)
                        C_exp.assign({{-0.000000, 0.000000},        {-0.000000, 0.000000},
                                      {-0.000000, 0.000000},        {-0.000000, 0.000000},
                                      {-0.000000, 0.000000},        {-846.000000, 267.000000},
                                      {2513.000000, 634.000000},    {2206.000000, 6788.000000},
                                      {-15991.000000, 3037.000000}, {49.000000, 9077.000000},
                                      {-6441.000000, 8362.000000},  {1372.000000, 3896.000000},
                                      {1418.000000, 1184.000000},   {-551.000000, 372.000000},
                                      {4288.000000, 279.000000},    {285.000000, 300.000000},
                                      {214.000000, 2172.000000},    {10468.000000, 564.000000},
                                      {-5021.000000, 17142.000000}, {4518.000000, 4749.000000},
                                      {-5025.000000, 715.000000},   {-11281.000000, 2032.000000},
                                      {-2942.000000, 12044.000000}, {-367.000000, -276.000000},
                                      {-1482.000000, -1766.000000}});
                    if(op == aoclsparse_operation_conjugate_transpose)
                        C_exp.assign({{-0.000000, 0.000000},        {-0.000000, 0.000000},
                                      {-0.000000, 0.000000},        {-0.000000, 0.000000},
                                      {-0.000000, 0.000000},        {-382.000000, -1021.000000},
                                      {-2647.000000, -726.000000},  {-7394.000000, -6412.000000},
                                      {15945.000000, -1995.000000}, {817.000000, -3059.000000},
                                      {4719.000000, -9478.000000},  {-756.000000, -3088.000000},
                                      {-598.000000, -2024.000000},  {545.000000, -60.000000},
                                      {-4208.000000, 831.000000},   {285.000000, -300.000000},
                                      {2094.000000, -6788.000000},  {228.000000, -10556.000000},
                                      {9939.000000, 15022.000000},  {7878.000000, -171.000000},
                                      {-2825.000000, -4235.000000}, {-6051.000000, -9728.000000},
                                      {-12282.000000, 1724.000000}, {313.000000, -336.000000},
                                      {1078.000000, -2036.000000}});
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

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_csrmm_nullptr()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, n, nnz;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

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
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        // In turns pass nullptr in every single pointer argument
        // and expect pointer error
        EXPECT_EQ(
            aoclsparse_csrmm<T>(op, alpha, A, nullptr, order, B.data(), n, k, beta, C.data(), m),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmm<T>(
                      op, alpha, nullptr, descr, order, B.data(), n, k, beta, C.data(), m),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmm<T>(op, alpha, A, descr, order, nullptr, n, k, beta, C.data(), m),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_csrmm<T>(op, alpha, A, descr, order, B.data(), n, k, beta, nullptr, m),
                  aoclsparse_status_invalid_pointer);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(A);
    }

    // tests for Wrong size
    template <typename T>
    void test_csrmm_wrong_size()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, n, nnz;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

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
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        // expect invalid size for wrong ldb
        EXPECT_EQ(
            aoclsparse_csrmm<T>(op, alpha, A, descr, order, B.data(), n, k - 1, beta, C.data(), m),
            aoclsparse_status_invalid_size);

        // expect invalid size for wrong ldc
        EXPECT_EQ(
            aoclsparse_csrmm<T>(op, alpha, A, descr, order, B.data(), n, k, beta, C.data(), m - 1),
            aoclsparse_status_invalid_size);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(A);
    }

    template <typename T>
    void test_csrmm_baseOne()
    {
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_index_base       base = aoclsparse_index_base_one;
        aoclsparse_int              m, k, n, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

        for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
        {
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
                    5);
            // Set values of ldb, ldc and matrix dimenstions of C matrix
            set_mm_dim(op, order, m, k, n, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
            C.resize(C_m * C_n);

            aoclsparse_mat_descr descr;

            ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

            aoclsparse_matrix A;
            ASSERT_EQ(
                aoclsparse_create_csr(
                    A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                aoclsparse_status_success);

            EXPECT_EQ(aoclsparse_csrmm<T>(
                          op, alpha, A, descr, order, B.data(), C_n, ldb, beta, C.data(), ldc),
                      aoclsparse_status_success);

            EXPECT_ARR_NEAR((C_m * C_n), C, C_exp, expected_precision<T>(10.0));
            EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        }
    }

    // tests for settings not implemented
    template <typename T>
    void test_csrmm_not_implemented()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, n, nnz;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

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
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        // expect not_implemented for !aoclsparse_matrix_type_general
        aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        EXPECT_EQ(
            aoclsparse_csrmm<T>(op, alpha, A, descr, order, B.data(), n, k, beta, C.data(), m),
            aoclsparse_status_not_implemented);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(A);
    }

    // zero matrix size is valid - just do nothing
    template <typename T>
    void test_csrmm_do_nothing()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, n, nnz;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

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
        aoclsparse_int       csr_row_ptr_zeros[] = {0, 0, 0};
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        // expect success for m=0
        ASSERT_EQ(aoclsparse_create_csr(
                      A, base, 0, k, 0, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_csrmm<T>(op, alpha, A, descr, order, B.data(), n, k, beta, C.data(), m),
            aoclsparse_status_success);
        aoclsparse_destroy(A);

        // expect success for k=0
        ASSERT_EQ(aoclsparse_create_csr(
                      A, base, m, 0, 0, csr_row_ptr_zeros, csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_csrmm<T>(op, alpha, A, descr, order, B.data(), n, k, beta, C.data(), m),
            aoclsparse_status_success);
        aoclsparse_destroy(A);

        // expect success for n=0
        ASSERT_EQ(aoclsparse_create_csr(
                      A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);
        EXPECT_EQ(
            aoclsparse_csrmm<T>(op, alpha, A, descr, order, B.data(), 0, k, beta, C.data(), m),
            aoclsparse_status_success);

        // expect success for alpha = 0 & beta = 1
        alpha = 0.0;
        beta  = 1.0;
        EXPECT_EQ(
            aoclsparse_csrmm<T>(op, alpha, A, descr, order, B.data(), n, k, beta, C.data(), m),
            aoclsparse_status_success);

        aoclsparse_destroy(A);
        aoclsparse_destroy_mat_descr(descr);
    }

    // tests for ldb and ldc greater than minimum
    template <typename T>
    void test_csrmm_greater_ld()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, n, nnz;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

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
        C_exp.assign(
            {1.120000, -190.960000, 0.000000, 0.000000, 3.360000, 487.480000, 0.000000, 0.000000});
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);

        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr(
                      A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);

        // expect success for ldb = k*2 and ldc = m*2
        EXPECT_EQ(aoclsparse_csrmm<T>(
                      op, alpha, A, descr, order, B.data(), n, k * 2, beta, C.data(), m * 2),
                  aoclsparse_status_success);
        if constexpr(std::is_same_v<T, double>)
            EXPECT_DOUBLE_EQ_VEC(m * 2 * n, C, C_exp);
        if constexpr(std::is_same_v<T, float>)
            EXPECT_FLOAT_EQ_VEC(m * 2 * n, C, C_exp);

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(A);
    }

    template <typename T>
    void test_csrmm_wrongtype()
    {
        aoclsparse_index_base       base = aoclsparse_index_base_zero;
        aoclsparse_operation        op   = aoclsparse_operation_none;
        aoclsparse_order            order;
        aoclsparse_int              m, k, n, nnz;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

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
                1);
        aoclsparse_mat_descr descr;
        // aoclsparse_create_mat_descr sets aoclsparse_matrix_type to aoclsparse_matrix_type_general
        // and aoclsparse_index_base to aoclsparse_index_base_zero.
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        aoclsparse_matrix A;
        ASSERT_EQ(
            aoclsparse_create_csr(
                A, base, m, k, nnz, csr_row_ptr.data(), csr_col_ind.data(), (T *)csr_val.data()),
            aoclsparse_status_success);

        if constexpr(std::is_same_v<T, double>)
        {
            // expect wrong type error for invoking csrmm for single precision with double csr_val
            EXPECT_EQ(aoclsparse_scsrmm(op,
                                        alpha,
                                        A,
                                        descr,
                                        order,
                                        (float *)B.data(),
                                        n,
                                        k,
                                        beta,
                                        (float *)C.data(),
                                        m),
                      aoclsparse_status_wrong_type);
        }

        if constexpr(std::is_same_v<T, float>)
        {
            // expect wrong type error for invoking csrmm for double precision with float csr_val
            EXPECT_EQ(aoclsparse_dcsrmm(op,
                                        alpha,
                                        A,
                                        descr,
                                        order,
                                        (double *)B.data(),
                                        n,
                                        k,
                                        beta,
                                        (double *)C.data(),
                                        m),
                      aoclsparse_status_wrong_type);
        }

        aoclsparse_destroy_mat_descr(descr);
        aoclsparse_destroy(A);
    }

    template <typename T>
    void test_csrmm_success()
    {
        aoclsparse_int              m, k, n, nnz, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<aoclsparse_int> csr_row_ptr;
        T                           alpha, beta;
        std::vector<T>              B;
        std::vector<T>              C;
        std::vector<T>              C_exp;

        //Test for real types
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
        {
            for(int id = 0; id < 5; id++)
            {
                for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                {
                    for(aoclsparse_index_base base :
                        {aoclsparse_index_base_zero, aoclsparse_index_base_one})
                    {
                        for(aoclsparse_operation op : {aoclsparse_operation_none,
                                                       aoclsparse_operation_transpose,
                                                       aoclsparse_operation_conjugate_transpose})
                        {
                            //Initialize inputs for test
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
                                    id);

                            // Set values of ldb, ldc and matrix dimenstions of C matrix
                            set_mm_dim(op, order, m, k, n, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
                            C.resize(C_m * C_n);

                            aoclsparse_mat_descr descr;
                            ASSERT_EQ(aoclsparse_create_mat_descr(&descr),
                                      aoclsparse_status_success);
                            ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base),
                                      aoclsparse_status_success);

                            aoclsparse_matrix A;
                            ASSERT_EQ(aoclsparse_create_csr(A,
                                                            base,
                                                            m,
                                                            k,
                                                            nnz,
                                                            csr_row_ptr.data(),
                                                            csr_col_ind.data(),
                                                            csr_val.data()),
                                      aoclsparse_status_success);
                            EXPECT_EQ(aoclsparse_csrmm<T>(op,
                                                          alpha,
                                                          A,
                                                          descr,
                                                          order,
                                                          B.data(),
                                                          C_n,
                                                          ldb,
                                                          beta,
                                                          C.data(),
                                                          ldc),
                                      aoclsparse_status_success);
                            if constexpr(std::is_same_v<T, double>)
                                EXPECT_DOUBLE_EQ_VEC(C_m * C_n, C, C_exp);
                            if constexpr(std::is_same_v<T, float>)
                                EXPECT_FLOAT_EQ_VEC(C_m * C_n, C, C_exp);

                            aoclsparse_destroy_mat_descr(descr);
                            aoclsparse_destroy(A);
                        }
                    }
                }
            }
        }
        //Test for complex types
        if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                     || std::is_same_v<T, aoclsparse_float_complex>)
        {
            for(int id = 0; id < 2; id++)
            {
                for(aoclsparse_operation op : {aoclsparse_operation_none,
                                               aoclsparse_operation_transpose,
                                               aoclsparse_operation_conjugate_transpose})
                {
                    //Initialize inputs for test
                    for(aoclsparse_order order : {aoclsparse_order_row, aoclsparse_order_column})
                    {
                        for(aoclsparse_index_base base :
                            {aoclsparse_index_base_zero, aoclsparse_index_base_one})
                        {
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
                                    id);

                            // Set values of ldb, ldc and matrix dimenstions of C matrix
                            set_mm_dim(op, order, m, k, n, A_m, A_n, B_m, B_n, C_m, C_n, ldb, ldc);
                            C.resize(C_m * C_n);

                            aoclsparse_mat_descr descr;
                            ASSERT_EQ(aoclsparse_create_mat_descr(&descr),
                                      aoclsparse_status_success);
                            ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base),
                                      aoclsparse_status_success);

                            aoclsparse_matrix A;

                            ASSERT_EQ(aoclsparse_create_csr(A,
                                                            base,
                                                            m,
                                                            k,
                                                            nnz,
                                                            csr_row_ptr.data(),
                                                            csr_col_ind.data(),
                                                            (T *)csr_val.data()),
                                      aoclsparse_status_success);
                            EXPECT_EQ(aoclsparse_csrmm<T>(op,
                                                          alpha,
                                                          A,
                                                          descr,
                                                          order,
                                                          B.data(),
                                                          C_n,
                                                          ldb,
                                                          beta,
                                                          C.data(),
                                                          ldc),
                                      aoclsparse_status_success);
                            if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
                            {
                                std::vector<std::complex<float>> *res, *res_exp;
                                res     = (std::vector<std::complex<float>> *)&C;
                                res_exp = (std::vector<std::complex<float>> *)&C_exp;
                                EXPECT_COMPLEX_FLOAT_EQ_VEC(C_m * C_n, (*res), (*res_exp));
                            }
                            if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
                            {
                                std::vector<std::complex<double>> *res, *res_exp;
                                res     = (std::vector<std::complex<double>> *)&C;
                                res_exp = (std::vector<std::complex<double>> *)&C_exp;
                                EXPECT_COMPLEX_DOUBLE_EQ_VEC(C_m * C_n, (*res), (*res_exp));
                            }

                            aoclsparse_destroy_mat_descr(descr);
                            aoclsparse_destroy(A);
                        }
                    }
                }
            }
        }
    }

    TEST(csrmm, NullArgDouble)
    {
        test_csrmm_nullptr<double>();
    }
    TEST(csrmm, NullArgFloat)
    {
        test_csrmm_nullptr<float>();
    }

    TEST(csrmm, WrongSizeDouble)
    {
        test_csrmm_wrong_size<double>();
    }
    TEST(csrmm, WrongSizeFloat)
    {
        test_csrmm_wrong_size<float>();
    }

    TEST(csrmm, BaseOneDouble)
    {
        test_csrmm_baseOne<double>();
    }
    TEST(csrmm, BaseOneFloat)
    {
        test_csrmm_baseOne<float>();
    }

    TEST(csrmm, NotImplDouble)
    {
        test_csrmm_not_implemented<double>();
    }
    TEST(csrmm, NotImplFloat)
    {
        test_csrmm_not_implemented<float>();
    }

    TEST(csrmm, DoNothingDouble)
    {
        test_csrmm_do_nothing<double>();
    }
    TEST(csrmm, DoNothingFloat)
    {
        test_csrmm_do_nothing<float>();
    }

    TEST(csrmm, GreaterLDDouble)
    {
        test_csrmm_greater_ld<double>();
    }
    TEST(csrmm, GreaterLDFloat)
    {
        test_csrmm_greater_ld<float>();
    }
    TEST(csrmm, WrongTypeDouble)
    {
        test_csrmm_wrongtype<double>();
    }
    TEST(csrmm, WrongTypeFloat)
    {
        test_csrmm_wrongtype<float>();
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
} // namespace
