/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_DOTMV_HPP
#define AOCLSPARSE_DOTMV_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_dense_dot.hpp"
#include "aoclsparse_l2.hpp"
#include "aoclsparse_mat_structures.hpp"

template <typename T>
aoclsparse_status aoclsparse_dotmv_t(const aoclsparse_operation op,
                                     const T                    alpha,
                                     aoclsparse_matrix          A,
                                     const aoclsparse_mat_descr descr,
                                     const T                   *x,
                                     const T                    beta,
                                     T                         *y,
                                     T                         *d,
                                     aoclsparse_int             kid = -1)
{
    if(d == nullptr || A == nullptr)
    {
        // Todo: All validations need to be done here
        return aoclsparse_status_invalid_pointer;
    }

    aoclsparse_status status = aoclsparse_mv_t(op, &alpha, A, descr, x, &beta, y);

    if(status)
    {
        return status;
    }

    /* Dot product needs x and y of same size but
     * op = non-transpose, size of y=m, x=n
     * op = transpose, size of y=n, x=m
     * hence, taking minimum of m and n
     */
    return aoclsparse::dense_dot((std::min)(A->m, A->n), x, y, d, kid);
}

#endif // AOCLSPARSE_DOTMV_HPP
