/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */

#include "aoclsparse.h"
#include "aoclsparse_syrkd.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Computes the product of a sparse matrix with its transpose and stores the result in a
 * newly allocated sparse matrix. The sparse matrices are in CSR storage format and supports s/d/c/z data types.
 */
extern "C" aoclsparse_status aoclsparse_ssyrkd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               float                      alpha,
                                               float                      beta,
                                               float                     *C,
                                               const aoclsparse_order     layout,
                                               aoclsparse_int             ldc)
{
    const aoclsparse_int kid = -1; /* auto */

    if(A == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_syrkd_t(op, A, alpha, beta, C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_dsyrkd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               double                     alpha,
                                               double                     beta,
                                               double                    *C,
                                               const aoclsparse_order     layout,
                                               aoclsparse_int             ldc)
{
    const aoclsparse_int kid = -1; /* auto */

    if(A == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    return aoclsparse_syrkd_t(op, A, alpha, beta, C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_csyrkd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               aoclsparse_float_complex   alpha,
                                               aoclsparse_float_complex   beta,
                                               aoclsparse_float_complex  *C,
                                               const aoclsparse_order     layout,
                                               aoclsparse_int             ldc)
{
    const aoclsparse_int kid = -1; /* auto */

    if(A == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    const std::complex<float> calpha{alpha.real, 0};
    const std::complex<float> cbeta{beta.real, 0};

    return aoclsparse_syrkd_t(op, A, calpha, cbeta, (std::complex<float> *)C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_zsyrkd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               aoclsparse_double_complex  alpha,
                                               aoclsparse_double_complex  beta,
                                               aoclsparse_double_complex *C,
                                               const aoclsparse_order     layout,
                                               aoclsparse_int             ldc)
{
    const aoclsparse_int kid = -1; /* auto */

    if(A == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    const std::complex<double> calpha{alpha.real, 0};
    const std::complex<double> cbeta{beta.real, 0};

    return aoclsparse_syrkd_t(op, A, calpha, cbeta, (std::complex<double> *)C, layout, ldc, kid);
}