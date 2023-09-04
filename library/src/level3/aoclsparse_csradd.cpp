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

#include "aoclsparse_csradd.hpp"

aoclsparse_status aoclsparse_sadd(const aoclsparse_operation op,
                                  const aoclsparse_matrix    A,
                                  const float                alpha,
                                  const aoclsparse_matrix    B,
                                  aoclsparse_matrix         *C)
{
    return aoclsparse_add_t(op, A, alpha, B, C);
};

aoclsparse_status aoclsparse_dadd(const aoclsparse_operation op,
                                  const aoclsparse_matrix    A,
                                  const double               alpha,
                                  const aoclsparse_matrix    B,
                                  aoclsparse_matrix         *C)
{
    return aoclsparse_add_t(op, A, alpha, B, C);
};

aoclsparse_status aoclsparse_cadd(const aoclsparse_operation     op,
                                  const aoclsparse_matrix        A,
                                  const aoclsparse_float_complex alpha,
                                  const aoclsparse_matrix        B,
                                  aoclsparse_matrix             *C)
{
    return aoclsparse_add_t(op, A, std::complex<float>(alpha.real, alpha.imag), B, C);
};

aoclsparse_status aoclsparse_zadd(const aoclsparse_operation      op,
                                  const aoclsparse_matrix         A,
                                  const aoclsparse_double_complex alpha,
                                  const aoclsparse_matrix         B,
                                  aoclsparse_matrix              *C)
{
    return aoclsparse_add_t(op, A, std::complex<double>(alpha.real, alpha.imag), B, C);
};
