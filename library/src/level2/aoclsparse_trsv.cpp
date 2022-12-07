/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_trsv.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

extern "C" aoclsparse_status aoclsparse_strsv(aoclsparse_operation       trans,
                                              const float                alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              const float               *b,
                                              float                     *x)
{
    const aoclsparse_int kid = -1; /* auto */

    return aoclsparse_trsv(trans, alpha, A, descr, b, x, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsv(aoclsparse_operation       trans,
                                              const double               alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              const double              *b,
                                              double                    *x)
{
    const aoclsparse_int kid = -1; /* auto */

    return aoclsparse_trsv(trans, alpha, A, descr, b, x, kid);
}

extern "C" aoclsparse_status aoclsparse_strsv_kid(aoclsparse_operation        trans,
                                                   const float                alpha,
                                                   aoclsparse_matrix          A,
                                                   const aoclsparse_mat_descr descr,
                                                   const float               *b,
                                                   float                     *x,
                                                   const aoclsparse_int       kid)
{
    return aoclsparse_trsv(trans, alpha, A, descr, b, x, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsv_kid(aoclsparse_operation        trans,
                                                   const double               alpha,
                                                   aoclsparse_matrix          A,
                                                   const aoclsparse_mat_descr descr,
                                                   const double              *b,
                                                   double                    *x,
                                                   const aoclsparse_int       kid)
{
    return aoclsparse_trsv(trans, alpha, A, descr, b, x, kid);
}
