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

template <typename T>
void test_create_csc(aoclsparse_status     status_exp,
                     aoclsparse_index_base base,
                     aoclsparse_int        M,
                     aoclsparse_int        N,
                     aoclsparse_int        nnz,
                     aoclsparse_int       *col_ptr,
                     aoclsparse_int       *row_idx,
                     T                    *val)
{
    aoclsparse_matrix mat = NULL;
    aoclsparse_status status;

    EXPECT_EQ(status = aoclsparse_create_csc(&mat, base, M, N, nnz, col_ptr, row_idx, val),
              status_exp);

    if(status == aoclsparse_status_success)
    {
        EXPECT_EQ(aoclsparse_status_success, aoclsparse_destroy(&mat));
    }
}
