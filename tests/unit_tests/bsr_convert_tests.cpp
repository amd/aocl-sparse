/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <complex>
#include <vector>

#define CONJ(len, val)                         \
    for(long unsigned int i = 0; i < len; i++) \
        val[i] = aoclsparse::conj(val[i]);

namespace
{
    template <typename T>
    aoclsparse_status init(aoclsparse_int              &m,
                           aoclsparse_int              &n,
                           std::vector<aoclsparse_int> &row_ptr,
                           std::vector<aoclsparse_int> &col_ind,
                           std::vector<T>              &csr_val,
                           aoclsparse_int               block_dim,
                           aoclsparse_index_base        base,
                           aoclsparse_order             block_order,
                           aoclsparse_operation         op,
                           std::vector<aoclsparse_int> &bsr_ptr_exp,
                           std::vector<aoclsparse_int> &bsr_ind_exp,
                           std::vector<T>              &bsr_val_exp,
                           aoclsparse_int               id = 0)
    {
        switch(id)
        {
        case 0:
            // 0 15 4 0 0
            // 3 1 0 -3 0
            // 6 0 0 5 7
            // 0 4 -8 0 0
            // 0 9 0 12 0
            m = 5, n = 5;
            row_ptr.assign({0, 2, 5, 8, 10, 12});
            col_ind.assign({1, 2, 3, 1, 0, 0, 3, 4, 1, 2, 3, 1});
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                csr_val.assign({15.0, 4.0, -3.0, 1.0, 3.0, 6.0, 5.0, 7.0, 4.0, -8.0, 12.0, 9.0});
            else
                csr_val.assign({{15.0, 1.0},
                                {4.0, 2.0},
                                {-3.0, 3.0},
                                {1.0, 4.0},
                                {3.0, 5.0},
                                {6.0, 6.0},
                                {5.0, 7.0},
                                {7.0, 8.0},
                                {4.0, 9.0},
                                {-8.0, 10.0},
                                {12.0, 11.0},
                                {9.0, 12.0}});
            TRANSFORM_BASE(base, row_ptr, col_ind);
            if(op == aoclsparse_operation_none)
            {
                switch(block_dim)
                {
                case 1:
                    bsr_ptr_exp.assign({0, 2, 5, 8, 10, 12});
                    bsr_ind_exp.assign({1, 2, 0, 1, 3, 0, 3, 4, 1, 2, 1, 3});
                    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                        bsr_val_exp.assign(
                            {15.0, 4.0, 3.0, 1.0, -3.0, 6.0, 5.0, 7.0, 4.0, -8.0, 9.0, 12.0});
                    else
                        bsr_val_exp.assign({{15.0, 1.0},
                                            {4.0, 2.0},
                                            {3.0, 5.0},
                                            {1.0, 4.0},
                                            {-3.0, 3.0},
                                            {6.0, 6.0},
                                            {5.0, 7.0},
                                            {7.0, 8.0},
                                            {4.0, 9.0},
                                            {-8.0, 10.0},
                                            {9.0, 12.0},
                                            {12.0, 11.0}});
                    break;
                case 2:
                    bsr_ptr_exp.assign({0, 2, 5, 7});
                    bsr_ind_exp.assign({0, 1, 0, 1, 2, 0, 1});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0,  15, 3, 1, 4, 0, 0, -3, 6, 0, 0, 4,  0, 5,
                                                -8, 0,  7, 0, 0, 0, 0, 9,  0, 0, 0, 12, 0, 0});

                        else
                            bsr_val_exp.assign({{0, 0},     {15.0, 1.0},  {3.0, 5.0},   {1.0, 4.0},
                                                {4.0, 2.0}, {0, 0},       {0, 0},       {-3.0, 3.0},
                                                {6.0, 6.0}, {0, 0},       {0, 0},       {4.0, 9.0},
                                                {0, 0},     {5.0, 7.0},   {-8.0, 10.0}, {0, 0},
                                                {7.0, 8.0}, {0, 0},       {0, 0},       {0, 0},
                                                {0, 0},     {9.0, 12.0},  {0, 0},       {0, 0},
                                                {0, 0},     {12.0, 11.0}, {0, 0},       {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 3, 15, 1, 4, 0, 0, -3, 6, 0, 0, 4, 0,  -8,
                                                5, 0, 7,  0, 0, 0, 0, 0,  9, 0, 0, 0, 12, 0});
                        else
                            bsr_val_exp.assign({
                                {0, 0}, {3.0, 5.0},   {15.0, 1.0}, {1.0, 4.0},   {4.0, 2.0},
                                {0, 0}, {0, 0},       {-3.0, 3.0}, {6.0, 6.0},   {0, 0},
                                {0, 0}, {4.0, 9.0},   {0, 0},      {-8.0, 10.0}, {5.0, 7.0},
                                {0, 0}, {7.0, 8.0},   {0, 0},      {0, 0},       {0, 0},
                                {0, 0}, {0, 0},       {9.0, 12.0}, {0, 0},       {0, 0},
                                {0, 0}, {12.0, 11.0}, {0, 0},
                            });

                    break;
                case 3:
                    bsr_ptr_exp.assign({0, 2, 4});
                    bsr_ind_exp.assign({0, 1, 0, 1});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0,  15, 4, 3, 1, 0, 6,  0, 0,  0, 0, 0,
                                                -3, 0,  0, 5, 7, 0, 0,  4, -8, 0, 9, 0,
                                                0,  0,  0, 0, 0, 0, 12, 0, 0,  0, 0, 0});
                        else
                            bsr_val_exp.assign({
                                {0, 0},       {15.0, 1.0}, {4.0, 2.0},  {3.0, 5.0}, {1.0, 4.0},
                                {0, 0},       {6.0, 6.0},  {0, 0},      {0, 0},     {0, 0},
                                {0, 0},       {0, 0},      {-3.0, 3.0}, {0, 0},     {0, 0},
                                {5.0, 7.0},   {7.0, 8.0},  {0, 0},      {0, 0},     {4.0, 9.0},
                                {-8.0, 10.0}, {0, 0},      {9.0, 12.0}, {0, 0},     {0, 0},
                                {0, 0},       {0, 0},      {0, 0},      {0, 0},     {0, 0},
                                {12.0, 11.0}, {0, 0},      {0, 0},      {0, 0},     {0, 0},
                                {0, 0},
                            });
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0,  3, 6, 15, 1,  0, 4, 0, 0, 0, -3, 5,
                                                0,  0, 7, 0,  0,  0, 0, 0, 0, 4, 9,  0,
                                                -8, 0, 0, 0,  12, 0, 0, 0, 0, 0, 0,  0});
                        else
                            bsr_val_exp.assign({{0, 0},       {3.0, 5.0}, {6.0, 6.0},  {15.0, 1.0},
                                                {1.0, 4.0},   {0, 0},     {4.0, 2.0},  {0, 0},
                                                {0, 0},       {0, 0},     {-3.0, 3.0}, {5.0, 7.0},
                                                {0, 0},       {0, 0},     {7.0, 8.0},  {0, 0},
                                                {0, 0},       {0, 0},     {0, 0},      {0, 0},
                                                {0, 0},       {4.0, 9.0}, {9.0, 12.0}, {0, 0},
                                                {-8.0, 10.0}, {0, 0},     {0, 0},      {0, 0},
                                                {12.0, 11.0}, {0, 0},     {0, 0},      {0, 0},
                                                {0, 0},       {0, 0},     {0, 0},      {0, 0}});
                    break;
                case 4:
                    bsr_ptr_exp.assign({0, 2, 3});
                    bsr_ind_exp.assign({0, 1, 0});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 15, 4,  0, 3, 1, 0, -3, 6, 0, 0, 5,
                                                0, 4,  -8, 0, 0, 0, 0, 0,  0, 0, 0, 0,
                                                7, 0,  0,  0, 0, 0, 0, 0,  0, 9, 0, 12,
                                                0, 0,  0,  0, 0, 0, 0, 0,  0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{0, 0},       {15.0, 1.0}, {4.0, 2.0},  {0, 0},      {3.0, 5.0},
                                 {1.0, 4.0},   {0, 0},      {-3.0, 3.0}, {6.0, 6.0},  {0, 0},
                                 {0, 0},       {5.0, 7.0},  {0, 0},      {4.0, 9.0},  {-8.0, 10.0},
                                 {0, 0},       {0, 0},      {0, 0},      {0, 0},      {0, 0},
                                 {0, 0},       {0, 0},      {0, 0},      {0, 0},      {7.0, 8.0},
                                 {0, 0},       {0, 0},      {0, 0},      {0, 0},      {0, 0},
                                 {0, 0},       {0, 0},      {0, 0},      {9.0, 12.0}, {0, 0},
                                 {12.0, 11.0}, {0, 0},      {0, 0},      {0, 0},      {0, 0},
                                 {0, 0},       {0, 0},      {0, 0},      {0, 0},      {0, 0},
                                 {0, 0},       {0, 0},      {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 3,  6, 0, 15, 1, 0, 4, 4,  0, 0, -8,
                                                0, -3, 5, 0, 0,  0, 7, 0, 0,  0, 0, 0,
                                                0, 0,  0, 0, 0,  0, 0, 0, 0,  0, 0, 0,
                                                9, 0,  0, 0, 0,  0, 0, 0, 12, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{0, 0},     {3.0, 5.0},   {6.0, 6.0}, {0, 0},      {15.0, 1.0},
                                 {1.0, 4.0}, {0, 0},       {4.0, 9.0}, {4.0, 2.0},  {0, 0},
                                 {0, 0},     {-8.0, 10.0}, {0, 0},     {-3.0, 3.0}, {5.0, 7.0},
                                 {0, 0},     {0, 0},       {0, 0},     {7.0, 8.0},  {0, 0},
                                 {0, 0},     {0, 0},       {0, 0},     {0, 0},      {0, 0},
                                 {0, 0},     {0, 0},       {0, 0},     {0, 0},      {0, 0},
                                 {0, 0},     {0, 0},       {0, 0},     {0, 0},      {0, 0},
                                 {0, 0},     {9.0, 12.0},  {0, 0},     {0, 0},      {0, 0},
                                 {0, 0},     {0, 0},       {0, 0},     {0, 0},      {12.0, 11.0},
                                 {0, 0},     {0, 0},       {0, 0}});
                    break;
                default:
                    return aoclsparse_status_invalid_value;
                }
            }
            else //if(op == aoclsparse_operation_transpose)
            {
                switch(block_dim)
                {
                case 1:
                    bsr_ptr_exp.assign({0, 2, 6, 8, 11, 12});
                    bsr_ind_exp.assign({1, 2, 0, 1, 3, 4, 0, 3, 1, 2, 4, 2});
                    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                        bsr_val_exp.assign({3, 6, 15, 1, 4, 9, 4, -8, -3, 5, 12, 7});
                    else
                        bsr_val_exp.assign({{3.0, 5.0},
                                            {6.0, 6.0},
                                            {15.0, 1.0},
                                            {1.0, 4.0},
                                            {4.0, 9.0},
                                            {9.0, 12.0},
                                            {4.0, 2.0},
                                            {-8.0, 10.0},
                                            {-3.0, 3.0},
                                            {5.0, 7.0},
                                            {12.0, 11.0},
                                            {7.0, 8.0}});
                    break;
                case 2:
                    bsr_ptr_exp.assign({0, 3, 6, 7});
                    bsr_ind_exp.assign({0, 1, 2, 0, 1, 2, 1});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 3,  15, 1,  6, 0, 0, 4, 0,  0, 9, 0, 4, 0,
                                                0, -3, 0,  -8, 5, 0, 0, 0, 12, 0, 7, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{0, 0},     {3.0, 5.0},   {15.0, 1.0},  {1.0, 4.0},
                                                {6.0, 6.0}, {0, 0},       {0, 0},       {4.0, 9.0},
                                                {0, 0},     {0, 0},       {9.0, 12.0},  {0, 0},
                                                {4.0, 2.0}, {0, 0},       {0, 0},       {-3.0, 3.0},
                                                {0, 0},     {-8.0, 10.0}, {5.0, 7.0},   {0, 0},
                                                {0, 0},     {0, 0},       {12.0, 11.0}, {0, 0},
                                                {7.0, 8.0}, {0, 0},       {0, 0},       {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 15, 3, 1, 6,  0, 0, 4,  0, 9, 0, 0, 4, 0,
                                                0, -3, 0, 5, -8, 0, 0, 12, 0, 0, 7, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{0, 0},     {15.0, 1.0},  {3.0, 5.0},   {1.0, 4.0},
                                                {6.0, 6.0}, {0, 0},       {0, 0},       {4.0, 9.0},
                                                {0, 0},     {9.0, 12.0},  {0, 0},       {0, 0},
                                                {4.0, 2.0}, {0, 0},       {0, 0},       {-3.0, 3.0},
                                                {0, 0},     {5.0, 7.0},   {-8.0, 10.0}, {0, 0},
                                                {0, 0},     {12.0, 11.0}, {0, 0},       {0, 0},
                                                {7.0, 8.0}, {0, 0},       {0, 0},       {0, 0}});

                    break;
                case 3:
                    bsr_ptr_exp.assign({0, 2, 4});
                    bsr_ind_exp.assign({0, 1, 0, 1});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 3, 6, 15, 1,  0, 4, 0,  0, 0, 0, 0,
                                                4, 9, 0, -8, 0,  0, 0, -3, 5, 0, 0, 7,
                                                0, 0, 0, 0,  12, 0, 0, 0,  0, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{0, 0},       {3.0, 5.0},  {6.0, 6.0}, {15.0, 1.0},
                                                {1.0, 4.0},   {0, 0},      {4.0, 2.0}, {0, 0},
                                                {0, 0},       {0, 0},      {0, 0},     {0, 0},
                                                {4.0, 9.0},   {9.0, 12.0}, {0, 0},     {-8.0, 10.0},
                                                {0, 0},       {0, 0},      {0, 0},     {-3.0, 3.0},
                                                {5.0, 7.0},   {0, 0},      {0, 0},     {7.0, 8.0},
                                                {0, 0},       {0, 0},      {0, 0},     {0, 0},
                                                {12.0, 11.0}, {0, 0},      {0, 0},     {0, 0},
                                                {0, 0},       {0, 0},      {0, 0},     {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 15, 4, 3, 1, 0, 6,  0, 0, 0,  4, -8,
                                                0, 9,  0, 0, 0, 0, 0,  0, 0, -3, 0, 0,
                                                5, 7,  0, 0, 0, 0, 12, 0, 0, 0,  0, 0});
                        else
                            bsr_val_exp.assign({{0, 0},     {15.0, 1.0}, {4.0, 2.0},   {3.0, 5.0},
                                                {1.0, 4.0}, {0, 0},      {6.0, 6.0},   {0, 0},
                                                {0, 0},     {0, 0},      {4.0, 9.0},   {-8.0, 10.0},
                                                {0, 0},     {9.0, 12.0}, {0, 0},       {0, 0},
                                                {0, 0},     {0, 0},      {0, 0},       {0, 0},
                                                {0, 0},     {-3.0, 3.0}, {0, 0},       {0, 0},
                                                {5.0, 7.0}, {7.0, 8.0},  {0, 0},       {0, 0},
                                                {0, 0},     {0, 0},      {12.0, 11.0}, {0, 0},
                                                {0, 0},     {0, 0},      {0, 0},       {0, 0}});
                    break;
                case 4:
                    bsr_ptr_exp.assign({0, 2, 3});
                    bsr_ind_exp.assign({0, 1, 0});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 3,  6, 0, 15, 1, 0, 4, 4, 0, 0, -8,
                                                0, -3, 5, 0, 0,  0, 0, 0, 9, 0, 0, 0,
                                                0, 0,  0, 0, 12, 0, 0, 0, 0, 0, 7, 0,
                                                0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{0, 0},      {3.0, 5.0},   {6.0, 6.0}, {0, 0},       {15.0, 1.0},
                                 {1.0, 4.0},  {0, 0},       {4.0, 9.0}, {4.0, 2.0},   {0, 0},
                                 {0, 0},      {-8.0, 10.0}, {0, 0},     {-3.0, 3.0},  {5.0, 7.0},
                                 {0, 0},      {0, 0},       {0, 0},     {0, 0},       {0, 0},
                                 {9.0, 12.0}, {0, 0},       {0, 0},     {0, 0},       {0, 0},
                                 {0, 0},      {0, 0},       {0, 0},     {12.0, 11.0}, {0, 0},
                                 {0, 0},      {0, 0},       {0, 0},     {0, 0},       {7.0, 8.0},
                                 {0, 0},      {0, 0},       {0, 0},     {0, 0},       {0, 0},
                                 {0, 0},      {0, 0},       {0, 0},     {0, 0},       {0, 0},
                                 {0, 0},      {0, 0},       {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({0, 15, 4,  0, 3, 1, 0, -3, 6, 0, 0, 5,
                                                0, 4,  -8, 0, 0, 9, 0, 12, 0, 0, 0, 0,
                                                0, 0,  0,  0, 0, 0, 0, 0,  0, 0, 0, 0,
                                                0, 0,  0,  0, 7, 0, 0, 0,  0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{0, 0},     {15.0, 1.0}, {4.0, 2.0},  {0, 0},     {3.0, 5.0},
                                 {1.0, 4.0}, {0, 0},      {-3.0, 3.0}, {6.0, 6.0}, {0, 0},
                                 {0, 0},     {5.0, 7.0},  {0, 0},      {4.0, 9.0}, {-8.0, 10.0},
                                 {0, 0},     {0, 0},      {9.0, 12.0}, {0, 0},     {12.0, 11.0},
                                 {0, 0},     {0, 0},      {0, 0},      {0, 0},     {0, 0},
                                 {0, 0},     {0, 0},      {0, 0},      {0, 0},     {0, 0},
                                 {0, 0},     {0, 0},      {0, 0},      {0, 0},     {0, 0},
                                 {0, 0},     {0, 0},      {0, 0},      {0, 0},     {0, 0},
                                 {7.0, 8.0}, {0, 0},      {0, 0},      {0, 0},     {0, 0},
                                 {0, 0},     {0, 0},      {0, 0}});
                    break;
                default:
                    return aoclsparse_status_invalid_value;
                }
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    if(op == aoclsparse_operation_conjugate_transpose)
                        CONJ(bsr_val_exp.size(), bsr_val_exp);
            }
            break;
        case 1:
            // same as N5_full_sorted with 2 added columns
            //  1  0  0  2  0  1  0
            //  0  3  0  0  0  2  0
            //  0  0  4  0  0  0  0
            //  0  5  0  6  7  0  3
            //  0  0  0  0  8  4  5
            n       = 7;
            m       = 5;
            row_ptr = {0, 3, 5, 6, 10, 13};
            col_ind = {0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6};
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                csr_val = {1, 2, 1, 3, 2, 4, 5, 6, 7, 3, 8, 4, 5};
            else
                csr_val = {{1, 1},
                           {2, 2},
                           {1, 3},
                           {3, 4},
                           {2, 5},
                           {4, 6},
                           {5, 7},
                           {6, 8},
                           {7, 9},
                           {3, 10},
                           {8, 11},
                           {4, 12},
                           {5, 13}};
            TRANSFORM_BASE(base, row_ptr, col_ind);
            if(op == aoclsparse_operation_none)
            {
                switch(block_dim)
                {
                case 1:
                    bsr_ptr_exp.assign({0, 3, 5, 6, 10, 13});
                    bsr_ind_exp.assign({0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6});
                    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                        bsr_val_exp.assign({1, 2, 1, 3, 2, 4, 5, 6, 7, 3, 8, 4, 5});
                    else
                        bsr_val_exp.assign({{1, 1},
                                            {2, 2},
                                            {1, 3},
                                            {3, 4},
                                            {2, 5},
                                            {4, 6},
                                            {5, 7},
                                            {6, 8},
                                            {7, 9},
                                            {3, 10},
                                            {8, 11},
                                            {4, 12},
                                            {5, 13}});
                    break;
                case 2:
                    bsr_ptr_exp.assign({0, 3, 7, 9});
                    bsr_ind_exp.assign({0, 1, 2, 0, 1, 2, 3, 2, 3});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 3, 0, 2, 0, 0, 0, 1, 0, 2,
                                                0, 0, 0, 5, 4, 0, 0, 6, 0, 0, 7, 0,
                                                0, 0, 3, 0, 8, 4, 0, 0, 5, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{1, 1}, {0, 0}, {0, 0},  {3, 4}, {0, 0},  {2, 2},
                                                {0, 0}, {0, 0}, {0, 0},  {1, 3}, {0, 0},  {2, 5},
                                                {0, 0}, {0, 0}, {0, 0},  {5, 7}, {4, 6},  {0, 0},
                                                {0, 0}, {6, 8}, {0, 0},  {0, 0}, {7, 9},  {0, 0},
                                                {0, 0}, {0, 0}, {3, 10}, {0, 0}, {8, 11}, {4, 12},
                                                {0, 0}, {0, 0}, {5, 13}, {0, 0}, {0, 0},  {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 3, 0, 0, 2, 0, 0, 0, 1, 2,
                                                0, 0, 0, 5, 4, 0, 0, 6, 0, 7, 0, 0,
                                                0, 3, 0, 0, 8, 0, 4, 0, 5, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1},  {0, 0},  {0, 0}, {3, 4}, {0, 0},  {0, 0}, {2, 2},  {0, 0},
                                 {0, 0},  {0, 0},  {1, 3}, {2, 5}, {0, 0},  {0, 0}, {0, 0},  {5, 7},
                                 {4, 6},  {0, 0},  {0, 0}, {6, 8}, {0, 0},  {7, 9}, {0, 0},  {0, 0},
                                 {0, 0},  {3, 10}, {0, 0}, {0, 0}, {8, 11}, {0, 0}, {4, 12}, {0, 0},
                                 {5, 13}, {0, 0},  {0, 0}, {0, 0}});
                    break;
                case 3:
                    bsr_ptr_exp.assign({0, 2, 5});
                    bsr_ind_exp.assign({0, 1, 0, 1, 2});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 0, 3, 0, 0, 0, 4, 2, 0, 1, 0, 0, 2,
                                                0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 7, 0,
                                                0, 8, 4, 0, 0, 0, 3, 0, 0, 5, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1},  {0, 0}, {0, 0}, {0, 0}, {3, 4},  {0, 0}, {0, 0}, {0, 0},
                                 {4, 6},  {2, 2}, {0, 0}, {1, 3}, {0, 0},  {0, 0}, {2, 5}, {0, 0},
                                 {0, 0},  {0, 0}, {0, 0}, {5, 7}, {0, 0},  {0, 0}, {0, 0}, {0, 0},
                                 {0, 0},  {0, 0}, {0, 0}, {6, 8}, {7, 9},  {0, 0}, {0, 0}, {8, 11},
                                 {4, 12}, {0, 0}, {0, 0}, {0, 0}, {3, 10}, {0, 0}, {0, 0}, {5, 13},
                                 {0, 0},  {0, 0}, {0, 0}, {0, 0}, {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 0, 3, 0, 0, 0, 4, 2, 0, 0, 0, 0, 0,
                                                1, 2, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 6, 0, 0,
                                                7, 8, 0, 0, 4, 0, 3, 5, 0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1}, {0, 0}, {0, 0},  {0, 0}, {3, 4},  {0, 0},  {0, 0}, {0, 0},
                                 {4, 6}, {2, 2}, {0, 0},  {0, 0}, {0, 0},  {0, 0},  {0, 0}, {1, 3},
                                 {2, 5}, {0, 0}, {0, 0},  {0, 0}, {0, 0},  {5, 7},  {0, 0}, {0, 0},
                                 {0, 0}, {0, 0}, {0, 0},  {6, 8}, {0, 0},  {0, 0},  {7, 9}, {8, 11},
                                 {0, 0}, {0, 0}, {4, 12}, {0, 0}, {3, 10}, {5, 13}, {0, 0}, {0, 0},
                                 {0, 0}, {0, 0}, {0, 0},  {0, 0}, {0, 0}});
                    break;
                case 4:
                    bsr_ptr_exp.assign({0, 2, 3});
                    bsr_ind_exp.assign({0, 1, 1});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 2, 0, 3, 0, 0, 0, 0, 4, 0, 0, 5, 0, 6,
                                                0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 7, 0, 3, 0,
                                                8, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1}, {0, 0}, {0, 0},  {2, 2}, {0, 0},  {3, 4},  {0, 0},
                                 {0, 0}, {0, 0}, {0, 0},  {4, 6}, {0, 0},  {0, 0},  {5, 7},
                                 {0, 0}, {6, 8}, {0, 0},  {1, 3}, {0, 0},  {0, 0},  {0, 0},
                                 {2, 5}, {0, 0}, {0, 0},  {0, 0}, {0, 0},  {0, 0},  {0, 0},
                                 {7, 9}, {0, 0}, {3, 10}, {0, 0}, {8, 11}, {4, 12}, {5, 13},
                                 {0, 0}, {0, 0}, {0, 0},  {0, 0}, {0, 0},  {0, 0},  {0, 0},
                                 {0, 0}, {0, 0}, {0, 0},  {0, 0}, {0, 0},  {0, 0}});

                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 0, 0, 3, 0, 5, 0, 0, 4, 0, 2, 0, 0, 6,
                                                0, 0, 0, 7, 1, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
                                                8, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1}, {0, 0},  {0, 0}, {0, 0}, {0, 0},  {3, 4},  {0, 0},
                                 {5, 7}, {0, 0},  {0, 0}, {4, 6}, {0, 0},  {2, 2},  {0, 0},
                                 {0, 0}, {6, 8},  {0, 0}, {0, 0}, {0, 0},  {7, 9},  {1, 3},
                                 {2, 5}, {0, 0},  {0, 0}, {0, 0}, {0, 0},  {0, 0},  {3, 10},
                                 {0, 0}, {0, 0},  {0, 0}, {0, 0}, {8, 11}, {0, 0},  {0, 0},
                                 {0, 0}, {4, 12}, {0, 0}, {0, 0}, {0, 0},  {5, 13}, {0, 0},
                                 {0, 0}, {0, 0},  {0, 0}, {0, 0}, {0, 0},  {0, 0}});
                    break;
                default:
                    return aoclsparse_status_invalid_value;
                }
            }
            else //if(op == aoclsparse_operation_transpose)
            {
                switch(block_dim)
                {
                case 1:
                    bsr_ptr_exp.assign({0, 1, 3, 4, 6, 8});
                    bsr_ind_exp.assign({0, 1, 3, 2, 0, 3, 3, 4, 0, 1, 4, 3, 4});
                    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                        bsr_val_exp.assign({1, 3, 5, 4, 2, 6, 7, 8, 1, 2, 4, 3, 5});
                    else
                        bsr_val_exp.assign({{1, 1},
                                            {3, 4},
                                            {5, 7},
                                            {4, 6},
                                            {2, 2},
                                            {6, 8},
                                            {7, 9},
                                            {8, 11},
                                            {1, 3},
                                            {2, 5},
                                            {4, 12},
                                            {3, 10},
                                            {5, 13}});
                    break;
                case 2:
                    bsr_ptr_exp.assign({0, 2, 4, 7});
                    bsr_ind_exp.assign({0, 1, 0, 1, 0, 1, 2, 1, 2});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 3, 0, 0, 0, 5, 0, 0, 2, 0,
                                                4, 0, 0, 6, 0, 0, 1, 2, 0, 7, 0, 0,
                                                8, 0, 4, 0, 0, 3, 0, 0, 5, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{1, 1},  {0, 0}, {0, 0},  {3, 4}, {0, 0}, {0, 0},
                                                {0, 0},  {5, 7}, {0, 0},  {0, 0}, {2, 2}, {0, 0},
                                                {4, 6},  {0, 0}, {0, 0},  {6, 8}, {0, 0}, {0, 0},
                                                {1, 3},  {2, 5}, {0, 0},  {7, 9}, {0, 0}, {0, 0},
                                                {8, 11}, {0, 0}, {4, 12}, {0, 0}, {0, 0}, {3, 10},
                                                {0, 0},  {0, 0}, {5, 13}, {0, 0}, {0, 0}, {0, 0}});
                    else // if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 3, 0, 0, 0, 5, 0, 2, 0, 0,
                                                4, 0, 0, 6, 0, 1, 0, 2, 0, 0, 7, 0,
                                                8, 4, 0, 0, 0, 0, 3, 0, 5, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{1, 1},  {0, 0},  {0, 0},  {3, 4}, {0, 0}, {0, 0},
                                                {0, 0},  {5, 7},  {0, 0},  {2, 2}, {0, 0}, {0, 0},
                                                {4, 6},  {0, 0},  {0, 0},  {6, 8}, {0, 0}, {1, 3},
                                                {0, 0},  {2, 5},  {0, 0},  {0, 0}, {7, 9}, {0, 0},
                                                {8, 11}, {4, 12}, {0, 0},  {0, 0}, {0, 0}, {0, 0},
                                                {3, 10}, {0, 0},  {5, 13}, {0, 0}, {0, 0}, {0, 0}});

                    break;
                case 3:
                    bsr_ptr_exp.assign({0, 2, 4});
                    bsr_ind_exp.assign({0, 1, 0, 1, 1});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0,
                                                0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 2, 0, 6, 0, 0,
                                                7, 8, 0, 0, 4, 0, 3, 5, 0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1}, {0, 0}, {0, 0},  {0, 0}, {3, 4},  {0, 0},  {0, 0}, {0, 0},
                                 {4, 6}, {0, 0}, {0, 0},  {0, 0}, {5, 7},  {0, 0},  {0, 0}, {0, 0},
                                 {0, 0}, {0, 0}, {2, 2},  {0, 0}, {0, 0},  {0, 0},  {0, 0}, {0, 0},
                                 {1, 3}, {2, 5}, {0, 0},  {6, 8}, {0, 0},  {0, 0},  {7, 9}, {8, 11},
                                 {0, 0}, {0, 0}, {4, 12}, {0, 0}, {3, 10}, {5, 13}, {0, 0}, {0, 0},
                                 {0, 0}, {0, 0}, {0, 0},  {0, 0}, {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0,
                                                0, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 0, 6, 7, 0,
                                                0, 8, 4, 0, 0, 0, 3, 0, 0, 5, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1},  {0, 0}, {0, 0}, {0, 0}, {3, 4},  {0, 0}, {0, 0}, {0, 0},
                                 {4, 6},  {0, 0}, {5, 7}, {0, 0}, {0, 0},  {0, 0}, {0, 0}, {0, 0},
                                 {0, 0},  {0, 0}, {2, 2}, {0, 0}, {1, 3},  {0, 0}, {0, 0}, {2, 5},
                                 {0, 0},  {0, 0}, {0, 0}, {6, 8}, {7, 9},  {0, 0}, {0, 0}, {8, 11},
                                 {4, 12}, {0, 0}, {0, 0}, {0, 0}, {3, 10}, {0, 0}, {0, 0}, {5, 13},
                                 {0, 0},  {0, 0}, {0, 0}, {0, 0}, {0, 0}});
                    break;
                case 4:
                    bsr_ptr_exp.assign({0, 1, 3});
                    bsr_ind_exp.assign({0, 0, 1});
                    if(block_order == aoclsparse_order_row)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 0, 0, 3, 0, 5, 0, 0, 4, 0, 2, 0, 0, 6,
                                                0, 0, 0, 7, 1, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
                                                8, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1}, {0, 0},  {0, 0}, {0, 0}, {0, 0},  {3, 4},  {0, 0},
                                 {5, 7}, {0, 0},  {0, 0}, {4, 6}, {0, 0},  {2, 2},  {0, 0},
                                 {0, 0}, {6, 8},  {0, 0}, {0, 0}, {0, 0},  {7, 9},  {1, 3},
                                 {2, 5}, {0, 0},  {0, 0}, {0, 0}, {0, 0},  {0, 0},  {3, 10},
                                 {0, 0}, {0, 0},  {0, 0}, {0, 0}, {8, 11}, {0, 0},  {0, 0},
                                 {0, 0}, {4, 12}, {0, 0}, {0, 0}, {0, 0},  {5, 13}, {0, 0},
                                 {0, 0}, {0, 0},  {0, 0}, {0, 0}, {0, 0},  {0, 0}});
                    else //if(block_order == aoclsparse_order_column)
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({1, 0, 0, 2, 0, 3, 0, 0, 0, 0, 4, 0, 0, 5, 0, 6,
                                                0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 7, 0, 3, 0,
                                                8, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign(
                                {{1, 1}, {0, 0}, {0, 0},  {2, 2}, {0, 0},  {3, 4},  {0, 0},
                                 {0, 0}, {0, 0}, {0, 0},  {4, 6}, {0, 0},  {0, 0},  {5, 7},
                                 {0, 0}, {6, 8}, {0, 0},  {1, 3}, {0, 0},  {0, 0},  {0, 0},
                                 {2, 5}, {0, 0}, {0, 0},  {0, 0}, {0, 0},  {0, 0},  {0, 0},
                                 {7, 9}, {0, 0}, {3, 10}, {0, 0}, {8, 11}, {4, 12}, {5, 13},
                                 {0, 0}, {0, 0}, {0, 0},  {0, 0}, {0, 0},  {0, 0},  {0, 0},
                                 {0, 0}, {0, 0}, {0, 0},  {0, 0}, {0, 0},  {0, 0}});
                    break;
                default:
                    return aoclsparse_status_invalid_value;
                }
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    if(op == aoclsparse_operation_conjugate_transpose)
                        CONJ(bsr_val_exp.size(), bsr_val_exp);
            }
            break;
        case 2:
            // Singleton matrix
            m = 1, n = 1; //, nnz = 1;
            col_ind.assign({0});
            row_ptr.assign({0, 1});
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                csr_val.assign({4.0});
            else
                csr_val.assign({{4, -2}});
            TRANSFORM_BASE(base, row_ptr, col_ind);
            switch(block_dim)
            {
            case 1:
                bsr_ptr_exp.assign({0, 1});
                bsr_ind_exp.assign({0});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                    bsr_val_exp.assign({4.0});
                else
                    bsr_val_exp.assign({{4, -2}});
                break;
            case 2:
                bsr_ptr_exp.assign({0, 1});
                bsr_ind_exp.assign({0});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                    bsr_val_exp.assign({4.0, 0, 0, 0});
                else
                    bsr_val_exp.assign({{4, -2}});
                break;
            case 3:
                bsr_ptr_exp.assign({0, 1});
                bsr_ind_exp.assign({0});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                    bsr_val_exp.assign({4.0, 0, 0, 0, 0, 0, 0, 0, 0});
                else
                    bsr_val_exp.assign(
                        {{4, -2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}});
                break;
            case 4:
                bsr_ptr_exp.assign({0, 1});
                bsr_ind_exp.assign({0});
                if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                    bsr_val_exp.assign({4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
                else
                    bsr_val_exp.assign({{4, -2},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0},
                                        {0, 0}});
                break;
            default:
                return aoclsparse_status_invalid_value;
            }
            if constexpr(std::is_same_v<T, std::complex<float>>
                         || std::is_same_v<T, std::complex<double>>)
                if(op == aoclsparse_operation_conjugate_transpose)
                    CONJ(bsr_val_exp.size(), bsr_val_exp);

            break;
        case 3:
            // Input 4x3 matrix
            //  2  7  1
            //  0  5  0
            //  3  0  4
            //  0  6  0
            n       = 3;
            m       = 4;
            row_ptr = {0, 3, 4, 6, 7};
            col_ind = {0, 1, 2, 1, 0, 2, 1};
            if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                csr_val = {2, 7, 1, 5, 3, 4, 6};
            else
                csr_val = {{2, 1}, {7, 7}, {1, 2}, {5, 3}, {3, 4}, {4, 5}, {6, 6}};

            TRANSFORM_BASE(base, row_ptr, col_ind);

            if(op == aoclsparse_operation_none)
            {
                switch(block_dim)
                {
                case 1:
                    bsr_ptr_exp.assign({0, 3, 4, 6, 7});
                    bsr_ind_exp.assign({0, 1, 2, 1, 0, 2, 1});
                    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                        bsr_val_exp.assign({2, 7, 1, 5, 3, 4, 6});
                    else
                        bsr_val_exp.assign(
                            {{2, 1}, {7, 7}, {1, 2}, {5, 3}, {3, 4}, {4, 5}, {6, 6}});
                    break;
                case 2:
                    /*
                     * Input matrix (4x3)
                     *  [2  7  1]
                     *  [0  5  0]
                     *  [3  0  4]
                     *  [0  6  0]
                     *
                     * Block representation with block_dim = 2:
                     * BSR structure: 2 block rows, 2 block columns
                     * Block structure:
                     * +-------+-------+
                     * | B(0,0)| B(0,1)|  <- Block row 0
                     * +-------+-------+
                     * | B(1,0)| B(1,1)|  <- Block row 1
                     * +-------+-------+
                     *   ^          ^
                     *  Block   Block
                     *  col 0   col 1
                     * Block row 0 has 2 blocks: B(0,0) and B(0,1) (padded (0,1))
                     * Block row 1 has 2 blocks: B(1,0) and B(1,1) (padded (1,1))
                     * Individual blocks:
                     * B(0,0) = [2 7]   B(0,1) = [1]  (padded: [1 0]
                     *          [0 5]            [0]           [0 0])
                     *
                     * B(1,0) = [3 0]   B(1,1) = [4]  (padded: [4 0]
                     *          [0 6]            [0]           [0 0])
                     */
                    bsr_ptr_exp.assign({0, 2, 4});
                    bsr_ind_exp.assign({0, 1, 0, 1});

                    if(block_order == aoclsparse_order_row)
                        /* Row-major storage within blocks:
                         *
                         * B(0,0): [2 7] -> stored as [2, 7, 0, 5]
                         *         [0 5]
                         *
                         * B(0,1): [1 0] -> stored as [1, 0, 0, 0]
                         *         [0 0]
                         *
                         * B(1,0): [3 0] -> stored as [3, 0, 0, 6]
                         *         [0 6]
                         *
                         * B(1,1): [4 0] -> stored as [4, 0, 0, 0]
                         *         [0 0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({2, 7, 0, 5, 1, 0, 0, 0, 3, 0, 0, 6, 4, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {7, 7},
                                                {0, 0},
                                                {5, 3},
                                                {1, 2},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {3, 4},
                                                {0, 0},
                                                {0, 0},
                                                {6, 6},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    else // if(block_order == aoclsparse_order_column)
                        /* Column-major storage within blocks:
                         *
                         * B(0,0): [2 7] -> stored as [2, 0, 7, 5]
                         *         [0 5]
                         *
                         * B(0,1): [1 0] -> stored as [1, 0, 0, 0]
                         *         [0 0]
                         *
                         * B(1,0): [3 0] -> stored as [3, 0, 0, 6]
                         *         [0 6]
                         *
                         * B(1,1): [4 0] -> stored as [4, 0, 0, 0]
                         *         [0 0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({2, 0, 7, 5, 1, 0, 0, 0, 3, 0, 0, 6, 4, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {0, 0},
                                                {7, 7},
                                                {5, 3},
                                                {1, 2},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {3, 4},
                                                {0, 0},
                                                {0, 0},
                                                {6, 6},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    break;
                case 3:
                    /*
                     * Input matrix (4x3)
                     *  [2  7  1]
                     *  [0  5  0]
                     *  [3  0  4]
                     *  [0  6  0]
                     *
                     * Block representation with block_dim = 3:
                     * BSR structure: 2 block rows, 1 block columns
                     * Block structure:
                     * +-------+
                     * | B(0,0)|  <- Block row 0
                     * +-------+
                     * | B(1,0)|  <- Block row 1
                     * +-------+
                     *   ^
                     *  Block
                     *  col 0
                     * Block row 0 has 1 block: B(0,0)
                     * Block row 1 has 1 block: B(1,0) (padded (1,0))
                     * Individual blocks:
                     * B(0,0) = [2 7 1]
                     *          [0 5 0]
                     *          [3 0 4]
                     *
                     * B(1,0) = [0 6 0]   (padded:  [0 6 0]
                     *                              [0 0 0]
                     *                              [0 0 0])
                     */
                    bsr_ptr_exp.assign({0, 1, 2});
                    bsr_ind_exp.assign({0, 0});

                    if(block_order == aoclsparse_order_row)
                        /* Row-major storage within blocks:
                         *
                         * B(0,0): [2 7 1] -> stored as [2, 7, 1, 0, 5, 0, 3, 0, 4]
                         *         [0 5 0]
                         *         [3 0 4]
                         *
                         * B(1,0): [0 6 0] -> stored as [0, 6, 0, 0, 0, 0, 0, 0, 0]
                         *         [0 0 0]
                         *         [0 0 0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign(
                                {2, 7, 1, 0, 5, 0, 3, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {7, 7},
                                                {1, 2},
                                                {0, 0},
                                                {5, 3},
                                                {0, 0},
                                                {3, 4},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {6, 6},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    else // if(block_order == aoclsparse_order_column)
                        /* Column-major storage within blocks:
                         *
                         * B(0,0): [2 7 1] -> stored as [2, 0, 3, 7, 5, 0, 1, 0, 4]
                         *         [0 5 0]
                         *         [3 0 4]
                         *
                         * B(1,0): [0 6 0] -> stored as [0, 0, 0, 6, 0, 0, 0, 0, 0]
                         *         [0 0 0]
                         *         [0 0 0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign(
                                {2, 0, 3, 7, 5, 0, 1, 0, 4, 0, 0, 0, 6, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {0, 0},
                                                {3, 4},
                                                {7, 7},
                                                {5, 3},
                                                {0, 0},
                                                {1, 2},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {6, 6},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    break;

                case 4:
                    /*
                     * Input matrix (4x3)
                     *  [2  7  1]
                     *  [0  5  0]
                     *  [3  0  4]
                     *  [0  6  0]
                     *
                     * Block representation with block_dim = 4:
                     * BSR structure: 1 block rows, 1 block columns
                     * Block structure:
                     * +-------+
                     * | B(0,0)|  <- Block row 0
                     * +-------+
                     *   ^
                     *  Block
                     *  col 0
                     * Block row 0 has 1 block: B(0,0) (padded (0,0))
                     * Single 4x4 block contains the entire 4x3 matrix (with padded column) as follows:
                     * Individual blocks:
                     * B(0,0): [2 7 1 0]
                     *         [0 5 0 0]
                     *         [3 0 4 0]
                     *         [0 6 0 0]
                     */
                    bsr_ptr_exp.assign({0, 1});
                    bsr_ind_exp.assign({0});

                    if(block_order == aoclsparse_order_row)
                        /* Row-major storage within blocks:
                         *
                         * B(0,0): [2 7 1 0] -> stored as [2, 7, 1, 0, 0, 5, 0, 0, 3, 0, 4, 0, 0, 6, 0, 0]
                         *         [0 5 0 0]
                         *         [3 0 4 0]
                         *         [0 6 0 0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({2, 7, 1, 0, 0, 5, 0, 0, 3, 0, 4, 0, 0, 6, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {7, 7},
                                                {1, 2},
                                                {0, 0},
                                                {0, 0},
                                                {5, 3},
                                                {0, 0},
                                                {0, 0},
                                                {3, 4},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {6, 6},
                                                {0, 0},
                                                {0, 0}});
                    else // if(block_order == aoclsparse_order_column)
                        /* Column-major storage within blocks:
                         *
                         * B(0,0): [2 7 1 0] -> stored as [2, 0, 3, 0, 7, 5, 0, 6, 1, 0, 4, 0, 0, 0, 0, 0]
                         *         [0 5 0 0]
                         *         [3 0 4 0]
                         *         [0 6 0 0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({2, 0, 3, 0, 7, 5, 0, 6, 1, 0, 4, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {0, 0},
                                                {3, 4},
                                                {0, 0},
                                                {7, 7},
                                                {5, 3},
                                                {0, 0},
                                                {6, 6},
                                                {1, 2},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    break;
                default:
                    return aoclsparse_status_invalid_value;
                }
            }
            else // if(op == aoclsparse_operation_transpose)
            {
                // Transposed matrix (3x4):
                //  2  0  3  0
                //  7  5  0  6
                //  1  0  4  0
                switch(block_dim)
                {
                case 1:
                    bsr_ptr_exp.assign({0, 2, 5, 7}); // Updated for transpose: 3x4 matrix
                    bsr_ind_exp.assign({0, 2, 0, 1, 3, 0, 2}); // Updated column indices
                    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                        bsr_val_exp.assign({2, 3, 7, 5, 6, 1, 4}); // Updated values
                    else
                        bsr_val_exp.assign(
                            {{2, 1}, {3, 4}, {7, 7}, {5, 3}, {6, 6}, {1, 2}, {4, 5}});
                    break;
                case 2:
                    /*
                     * Transposed input matrix (3x4):
                     *  [2  0  3  0]
                     *  [7  5  0  6]
                     *  [1  0  4  0]
                     *
                     * Block representation with block_dim = 2:
                     * Transposed matrix is divided into 2x2 blocks: 2 block row, 2 block columns
                     * Block structure:
                     * +-------+-------+
                     * | B(0,0)| B(0,1)|  <- Block row 0
                     * +-------+-------+
                     * | B(1,0)| B(1,1)|  <- Block row 1
                     * +-------+-------+
                     *   ^          ^
                     *  Block   Block
                     *  col 0   col 1
                     * Block row 0 has 2 blocks: B(0,0) and B(0,1)
                     * Block row 1 has 2 blocks: B(1,0) and B(1,1) (padded (1,0), (1,1))
                     * Individual blocks:
                     * B(0,0) = [2 0]   B(0,1) = [3 0]
                     *          [7 5]            [0 6]
                     *
                     * B(1,0) = [1 0]   B(1,1) = [4 0]
                     *
                     * Padded:
                     * B(1,0) = [1 0]   B(1,1) = [4 0]
                     *          [0 0]            [0 0]
                     */
                    bsr_ptr_exp.assign({0, 2, 4}); // Both block rows have 2 blocks each
                    bsr_ind_exp.assign({0, 1, 0, 1}); // All 4 blocks included

                    if(block_order == aoclsparse_order_row)
                        /* Row-major storage for transposed matrix blocks:
                         *
                         * B(0,0): [2 0] -> stored as [2, 0, 7, 5]
                         *         [7 5]
                         *
                         * B(0,1): [3 0] -> stored as [3, 0, 0, 6]
                         *         [0 6]
                         *
                         * B(1,0): [1 0] -> stored as [1, 0, 0, 0]
                         *         [0 0]
                         *
                         * B(1,1): [4 0] -> stored as [4, 0, 0, 0]
                         *         [0 0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({2, 0, 7, 5, 3, 0, 0, 6, 1, 0, 0, 0, 4, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {0, 0},
                                                {7, 7},
                                                {5, 3},
                                                {3, 4},
                                                {0, 0},
                                                {0, 0},
                                                {6, 6},
                                                {1, 2},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    else // if(block_order == aoclsparse_order_column)
                        /* Column-major storage for transposed matrix blocks:
                         *
                         * B(0,0): [2 0] -> stored as [2, 7, 0, 5]
                         *         [7 5]
                         *
                         * B(0,1): [3 0] -> stored as [3, 0, 0, 6]
                         *         [0 6]
                         *
                         * B(1,0): [1 0] -> stored as [1, 0, 0, 0]
                         *         [0 0]
                         *
                         * B(1,1): [4 0] -> stored as [4, 0, 0, 0]
                         *         [0 0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({2, 7, 0, 5, 3, 0, 0, 6, 1, 0, 0, 0, 4, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {7, 7},
                                                {0, 0},
                                                {5, 3},
                                                {3, 4},
                                                {0, 0},
                                                {0, 0},
                                                {6, 6},
                                                {1, 2},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    break;
                case 3:
                    /*
                     * Transposed input matrix (3x4):
                     *  [2  0  3  0]
                     *  [7  5  0  6]
                     *  [1  0  4  0]
                     *
                     * Block representation with block_dim = 3:
                     * Transposed matrix is divided into 3x3 blocks: 1 block row, 2 block columns
                     *
                     * Block structure:
                     * +-------------+-------------+
                     * |   B(0,0)    |   B(0,1)    |  <- Block row 0
                     * +-------------+-------------+
                     *       ^             ^
                     * Block  col 0   Block col 1
                     * Block row 0 has 2 blocks: B(0,0) and B(0,1) (padded (0, 1))
                     * Individual blocks:
                     * B(0,0) = [2 0 3]   B(0,1) = [0]  (padded: [0 0 0]
                     *          [7 5 0]            [6]           [6 0 0]
                     *          [1 0 4]            [0]           [0 0 0])
                     */
                    bsr_ptr_exp.assign({0, 2});
                    bsr_ind_exp.assign({0, 1});

                    if(block_order == aoclsparse_order_row)
                        /* Row-major storage for transposed matrix blocks:
                         *
                         * B(0,0): [2 0 3] -> stored as [2, 0, 3, 7, 5, 0, 1, 0, 4]
                         *         [7 5 0]
                         *         [1 0 4]
                         *
                         * B(0,1): [0] -> stored as [0, 0, 0, 6, 0, 0, 0, 0, 0] (padded)
                         *         [6]
                         *         [0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign(
                                {2, 0, 3, 7, 5, 0, 1, 0, 4, 0, 0, 0, 6, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {0, 0},
                                                {3, 4},
                                                {7, 7},
                                                {5, 3},
                                                {0, 0},
                                                {1, 2},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {6, 6},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    else // if(block_order == aoclsparse_order_column)
                        /* Column-major storage for transposed matrix blocks:
                         *
                         * B(0,0): [2 0 3] -> stored as [2, 7, 1, 0, 5, 0, 3, 0, 4]
                         *         [7 5 0]
                         *         [1 0 4]
                         *
                         * B(0,1): [0] -> stored as [0, 6, 0, 0, 0, 0, 0, 0, 0] (padded)
                         *         [6]
                         *         [0]
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign(
                                {2, 7, 1, 0, 5, 0, 3, 0, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {7, 7},
                                                {1, 2},
                                                {0, 0},
                                                {5, 3},
                                                {0, 0},
                                                {3, 4},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {6, 6},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    break;

                case 4:
                    /*
                     * Transposed input matrix (3x4):
                     *  [2  0  3  0]
                     *  [7  5  0  6]
                     *  [1  0  4  0]
                     *
                     * Block representation with block_dim = 4:
                     * Transposed matrix is divided into 4x4 blocks: 1 block row, 1 block columns
                     *
                     * Block structure:
                     * +-------------+
                     * |   B(0,0)    |  <- Block row 0
                     * +-------------+
                     *       ^
                     * Block  col 0
                     * Single 4x4 block contains the entire 3x4 matrix (with padded row) as follows:
                     * Individual blocks:
                     * B(0,0) = [2 0 3 0]
                     *          [7 5 0 6]
                     *          [1 0 4 0]
                     *          [0 0 0 0]
                     */
                    bsr_ptr_exp.assign({0, 1});
                    bsr_ind_exp.assign({0});

                    if(block_order == aoclsparse_order_row)
                        /* Row-major storage for transposed matrix:
                         *
                         * B(0,0): [2  0  3  0]  -> stored as [2, 0, 3, 0, 7, 5, 0, 6, 1, 0, 4, 0, 0, 0, 0, 0]
                         *         [7  5  0  6]
                         *         [1  0  4  0]
                         *         [0  0  0  0]  (padded row)
                         *
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({2, 0, 3, 0, 7, 5, 0, 6, 1, 0, 4, 0, 0, 0, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {0, 0},
                                                {3, 4},
                                                {0, 0},
                                                {7, 7},
                                                {5, 3},
                                                {0, 0},
                                                {6, 6},
                                                {1, 2},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0},
                                                {0, 0}});
                    else // if(block_order == aoclsparse_order_column)
                        /* Column-major storage for transposed matrix:
                         *
                         * B(0,0): [2  0  3  0]  -> stored as [2, 7, 1, 0, 0, 5, 0, 0, 3, 0, 4, 0, 0, 6, 0, 0]
                         *         [7  5  0  6]
                         *         [1  0  4  0]
                         *         [0  0  0  0]  (padded row)
                         *
                         */
                        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
                            bsr_val_exp.assign({2, 7, 1, 0, 0, 5, 0, 0, 3, 0, 4, 0, 0, 6, 0, 0});
                        else
                            bsr_val_exp.assign({{2, 1},
                                                {7, 7},
                                                {1, 2},
                                                {0, 0},
                                                {0, 0},
                                                {5, 3},
                                                {0, 0},
                                                {0, 0},
                                                {3, 4},
                                                {0, 0},
                                                {4, 5},
                                                {0, 0},
                                                {0, 0},
                                                {6, 6},
                                                {0, 0},
                                                {0, 0}});
                    break;
                default:
                    return aoclsparse_status_invalid_value;
                }
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    if(op == aoclsparse_operation_conjugate_transpose)
                        CONJ(bsr_val_exp.size(), bsr_val_exp);
            }
            break;
        }

        TRANSFORM_BASE(base, bsr_ptr_exp, bsr_ind_exp);
        return aoclsparse_status_success;
    }

    //Test for success cases in aoclsparse_convert_bsr
    template <typename T>
    void test_convert_bsr_success(std::string testcase)
    {
        aoclsparse_int M, N;
        // Initialize data using init function
        std::vector<aoclsparse_int> row_ptr, col_ind;
        std::vector<T>              csr_val;
        std::vector<aoclsparse_int> bsr_ptr_exp, bsr_ind_exp;
        std::vector<T>              bsr_val_exp;

        for(aoclsparse_int id = 0; id < 4; id++)
        {
            for(aoclsparse_operation op : {aoclsparse_operation_none,
                                           aoclsparse_operation_transpose,
                                           aoclsparse_operation_conjugate_transpose})
            {
                for(aoclsparse_index_base base :
                    {aoclsparse_index_base_zero, aoclsparse_index_base_one})
                {
                    for(aoclsparse_order block_order :
                        {aoclsparse_order_row, aoclsparse_order_column})
                    {
                        for(aoclsparse_int block_dim : {1, 2, 3, 4})
                        {
                            SCOPED_TRACE(testcase + ", Operation: " + std::to_string(op)
                                         + ":: Base: " + std::to_string(base)
                                         + ", Block order: " + std::to_string(block_order)
                                         + ",Block dimension: " + std::to_string(block_dim));
                            // Initialize test data based on id, block_dim, base, block_order, and operation
                            init<T>(M,
                                    N,
                                    row_ptr,
                                    col_ind,
                                    csr_val,
                                    block_dim,
                                    base,
                                    block_order,
                                    op,
                                    bsr_ptr_exp,
                                    bsr_ind_exp,
                                    bsr_val_exp,
                                    id);

                            // source matrix handle
                            aoclsparse_matrix A;
                            // destination matrix handle
                            aoclsparse_matrix dst_mat = nullptr;

                            // Create a CSR matrix with the test data
                            ASSERT_EQ(aoclsparse_create_csr(&A,
                                                            base,
                                                            M,
                                                            N,
                                                            row_ptr[M] - base,
                                                            row_ptr.data(),
                                                            col_ind.data(),
                                                            csr_val.data()),
                                      aoclsparse_status_success);

                            // Convert the CSR matrix to BSR format with specified block dimension, order and operation
                            EXPECT_EQ(
                                aoclsparse_convert_bsr(A, block_dim, block_order, op, &dst_mat),
                                aoclsparse_status_success);

                            // Validate the destination matrix was created properly
                            ASSERT_NE(dst_mat, nullptr) << "Destination matrix is null";
                            ASSERT_FALSE(dst_mat->mats.empty())
                                << "Destination matrix has no internal matrices";
                            ASSERT_NE(dst_mat->mats[0], nullptr) << "Internal matrix is null";

                            aoclsparse::bsr *bsr_mat
                                = dynamic_cast<aoclsparse::bsr *>(dst_mat->mats[0]);

                            // Adding nullptr check for bsr_mat
                            ASSERT_NE(bsr_mat, nullptr) << "The destination matrix handle is null.";

                            // Verify the values in the converted BSR matrix match the expected values
                            expect_eq_vec<T>(
                                bsr_val_exp.size(), (T *)bsr_mat->val, (T *)bsr_val_exp.data());

                            // Verify the row/block pointers in the BSR matrix match the expected values
                            EXPECT_EQ_VEC(bsr_ptr_exp.size(), bsr_mat->ptr, bsr_ptr_exp.data());

                            // Verify the column indices in the BSR matrix match the expected values
                            EXPECT_EQ_VEC(bsr_ind_exp.size(), bsr_mat->ind, bsr_ind_exp.data());

                            // Clean up resources by destroying the source and destination matrix
                            aoclsparse_destroy(&A);
                            aoclsparse_destroy(&dst_mat);
                        }
                    }
                }
            }
        }
    }

    // Test for error conditions in aoclsparse_convert_bsr
    TEST(BsrConvertTests, ErrorConditions)
    {
        aoclsparse_matrix     A       = nullptr;
        aoclsparse_matrix     dst_mat = nullptr;
        aoclsparse_int        M = 5, N = 5, NNZ = 12;
        aoclsparse_index_base base        = aoclsparse_index_base_zero;
        aoclsparse_order      block_order = aoclsparse_order_row;
        aoclsparse_operation  op          = aoclsparse_operation_none;
        aoclsparse_int        block_dim   = 2;

        // Test data
        std::vector<aoclsparse_int> row_ptr = {0, 2, 5, 8, 10, 12};
        std::vector<aoclsparse_int> col_ind = {1, 2, 3, 1, 0, 0, 3, 4, 1, 2, 3, 1};
        std::vector<double>         csr_val
            = {15.0, 4.0, -3.0, 1.0, 3.0, 6.0, 5.0, 7.0, 4.0, -8.0, 12.0, 9.0};

        // Create a valid CSR matrix for testing
        ASSERT_EQ(aoclsparse_create_csr(
                      &A, base, M, N, NNZ, row_ptr.data(), col_ind.data(), csr_val.data()),
                  aoclsparse_status_success);
        {
            SCOPED_TRACE(std::string("BsrConvertTests - ErrorConditions")
                         + "Null input or output matrix pointers");
            // Test 1: Null input matrix
            EXPECT_EQ(aoclsparse_convert_bsr(nullptr, block_dim, block_order, op, &dst_mat),
                      aoclsparse_status_invalid_pointer);

            // Test 2: Null output matrix pointer
            EXPECT_EQ(aoclsparse_convert_bsr(A, block_dim, block_order, op, nullptr),
                      aoclsparse_status_invalid_pointer);
        }

        {
            SCOPED_TRACE(std::string("BsrConvertTests - ErrorConditions")
                         + "Invalid operation type");
            // Test 3: Invalid operation type
            EXPECT_EQ(aoclsparse_convert_bsr(
                          A, block_dim, block_order, (aoclsparse_operation)99, &dst_mat),
                      aoclsparse_status_not_implemented);
        }

        // Test 4: Non-CSR input matrix
        aoclsparse_matrix B = nullptr;
        aoclsparse_destroy(&A);

        {
            SCOPED_TRACE(std::string("BsrConvertTests - ErrorConditions") + "Non-CSR input matrix");

            // Create a CSR matrix as input
            ASSERT_EQ(aoclsparse_create_csr(
                          &A, base, M, N, NNZ, row_ptr.data(), col_ind.data(), csr_val.data()),
                      aoclsparse_status_success);

            // First convert to BSR to get a BSR matrix
            ASSERT_EQ(aoclsparse_convert_bsr(A, block_dim, block_order, op, &B),
                      aoclsparse_status_success);

            // Now try to convert from BSR to BSR (should fail)
            EXPECT_EQ(aoclsparse_convert_bsr(B, block_dim, block_order, op, &dst_mat),
                      aoclsparse_status_not_implemented);
        }
        // Cleanup
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&B);
        aoclsparse_destroy(&dst_mat);
    }

    //Test edge cases in aoclsparse_convert_bsr
    template <typename T>
    void edge_cases(std::string testcase)
    {
        aoclsparse_matrix           A;
        aoclsparse_matrix           dst_mat;
        aoclsparse_int              M = 0, N = 0, NNZ = 0;
        std::vector<aoclsparse_int> row_ptr = {0}, col_ind = {0};
        std::vector<T>              csr_val     = {0};
        aoclsparse_order            block_order = aoclsparse_order_row;
        aoclsparse_operation        op          = aoclsparse_operation_none;
        // Initialize an empty matrix A
        ASSERT_EQ(aoclsparse_create_csr(&A,
                                        aoclsparse_index_base_zero,
                                        M,
                                        N,
                                        NNZ,
                                        row_ptr.data(),
                                        col_ind.data(),
                                        csr_val.data()),
                  aoclsparse_status_success);

        SCOPED_TRACE(testcase + " - with valid block dimension 1");
        EXPECT_EQ(aoclsparse_convert_bsr(A, 1, block_order, op, &dst_mat),
                  aoclsparse_status_success);
        SCOPED_TRACE(testcase + " - with invalid block dimension 0");
        // With invalid block dimension
        EXPECT_EQ(aoclsparse_convert_bsr(A, 0, block_order, op, &dst_mat),
                  aoclsparse_status_invalid_value);
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&dst_mat);
    }

    template <typename T>
    void test_convert_bsr_success_missing_blks(std::string testcase)
    {
        SCOPED_TRACE(testcase);
        aoclsparse_matrix           A;
        aoclsparse_matrix           dst_mat;
        aoclsparse_int              M = 4, N = 6, NNZ = 6;
        std::vector<aoclsparse_int> row_ptr{0, 1, 4, 6, 6}, col_ind{0, 1, 2, 3, 0, 4};
        std::vector<T>              csr_val{1, 2, 3, 4, 5, 6};
        aoclsparse_order            block_order = aoclsparse_order_row;
        aoclsparse_operation        op          = aoclsparse_operation_none;
        std::vector<aoclsparse_int> bsr_ptr_exp, bsr_ind_exp;
        std::vector<T>              bsr_val_exp;

        aoclsparse_int block_dim = 2;
        // Initialize expected BSR pointers and indices
        bsr_ptr_exp.assign({0, 2, 4});
        bsr_ind_exp.assign({0, 1, 0, 2});
        bsr_val_exp.assign({1, 0, 0, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 0, 0});

        ASSERT_EQ(aoclsparse_create_csr(&A,
                                        aoclsparse_index_base_zero,
                                        M,
                                        N,
                                        NNZ,
                                        row_ptr.data(),
                                        col_ind.data(),
                                        csr_val.data()),
                  aoclsparse_status_success);

        EXPECT_EQ(aoclsparse_convert_bsr(A, block_dim, block_order, op, &dst_mat),
                  aoclsparse_status_success);

        ASSERT_NE(dst_mat, nullptr) << "Destination matrix is null";
        ASSERT_FALSE(dst_mat->mats.empty()) << "Destination matrix has no internal matrices";
        ASSERT_NE(dst_mat->mats[0], nullptr) << "Internal matrix is null";

        aoclsparse::bsr *bsr_mat = dynamic_cast<aoclsparse::bsr *>(dst_mat->mats[0]);

        // Adding nullptr check for bsr_mat
        ASSERT_NE(bsr_mat, nullptr) << "The destination matrix handle is null.";

        // Verify the values in the converted BSR matrix match the expected values
        expect_eq_vec<T>(bsr_val_exp.size(), (T *)bsr_mat->val, (T *)bsr_val_exp.data());

        // Verify the row/block pointers in the BSR matrix match the expected values
        EXPECT_EQ_VEC(bsr_ptr_exp.size(), bsr_mat->ptr, bsr_ptr_exp.data());

        // Verify the column indices in the BSR matrix match the expected values
        EXPECT_EQ_VEC(bsr_ind_exp.size(), bsr_mat->ind, bsr_ind_exp.data());
        aoclsparse_destroy(&A);
        aoclsparse_destroy(&dst_mat);
    }

    TEST(BsrConvertTests, Success)
    {
        test_convert_bsr_success<double>("BsrConvertTests - Success - double");
        test_convert_bsr_success<float>("BsrConvertTests - Success - float");
        test_convert_bsr_success<std::complex<double>>(
            "BsrConvertTests - Success - complex<double>");
        test_convert_bsr_success<std::complex<float>>("BsrConvertTests - Success - complex<float>");
        test_convert_bsr_success_missing_blks<double>(
            "BsrConvertTests - missing blocks - Success - double -");
    }
    TEST(BsrConvertTests, EdgeCaseEmptyMatrix)
    {
        edge_cases<double>("BsrConvertTests - EdgeCaseEmptyMatrix - double");
        edge_cases<float>("BsrConvertTests - EdgeCaseEmptyMatrix - float");
        edge_cases<std::complex<double>>("BsrConvertTests - EdgeCaseEmptyMatrix - complex<double>");
        edge_cases<std::complex<float>>("BsrConvertTests - EdgeCaseEmptyMatrix - complex<float>");
    }

} // namespace