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
#ifndef AOCLSPARSE_HPP
#define AOCLSPARSE_HPP
#pragma once
#include "aoclsparse.h"

#include <string>
namespace aoclsparse
{
    // Extern declaration of L2 dispatcher(s)
    template <typename T>
    aoclsparse_status trsv(const aoclsparse_operation transpose,
                           const T                    alpha,
                           aoclsparse_matrix          A,
                           const aoclsparse_mat_descr descr,
                           const T                   *b,
                           const aoclsparse_int       incb,
                           T                         *x,
                           const aoclsparse_int       incx,
                           aoclsparse_int             kid = -1);

    template <typename T>
    aoclsparse_status mv(aoclsparse_operation       op,
                         const T                   *alpha,
                         aoclsparse_matrix          A,
                         const aoclsparse_mat_descr descr,
                         const T                   *x,
                         const T                   *beta,
                         T                         *y);

    template <typename T>
    aoclsparse_status create_csr(aoclsparse_matrix    *mat,
                                 aoclsparse_index_base base,
                                 aoclsparse_int        M,
                                 aoclsparse_int        N,
                                 aoclsparse_int        nnz,
                                 aoclsparse_int       *row_ptr,
                                 aoclsparse_int       *col_idx,
                                 T                    *val,
                                 bool                  fast_chck = false);

    template <typename T>
    aoclsparse_status sp2m(aoclsparse_operation       opA,
                           const aoclsparse_mat_descr descrA,
                           const aoclsparse_matrix    A,
                           aoclsparse_operation       opB,
                           const aoclsparse_mat_descr descrB,
                           const aoclsparse_matrix    B,
                           aoclsparse_request         request,
                           aoclsparse_matrix         *C);

    // Test wrappers for unit tests
    // All test wrappers will be prefixed with "t_"
    namespace test
    {
        template <typename T>
        aoclsparse_int dispatcher(std::string    dispatch,
                                  aoclsparse_int kid   = -1,
                                  aoclsparse_int begin = 0,
                                  aoclsparse_int end   = 0);
    } // namespace test_wrapper
} // namespace aoclsparse
#endif