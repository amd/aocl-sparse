/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_syprd.hpp"

#include <algorithm>
#include <cmath>

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Performs symmetric triple product of a sparse matrix and a dense matrix
 * for type float.
 */
extern "C" aoclsparse_status aoclsparse_ssyprd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const float               *B,
                                               const aoclsparse_order     orderB,
                                               const aoclsparse_int       ldb,
                                               const float                alpha,
                                               const float                beta,
                                               float                     *C,
                                               const aoclsparse_order     orderC,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1;
    // Check for valid pointer
    if(A == nullptr || B == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(!(A->val_type == aoclsparse_smat))
    {
        return aoclsparse_status_wrong_type;
    }

    return aoclsparse_syprd<float>(op, A, B, orderB, ldb, alpha, beta, C, orderC, ldc, kid);
}

/*
 * Performs symmetric triple product of a sparse matrix and a dense matrix
 * for type double.
 */
extern "C" aoclsparse_status aoclsparse_dsyprd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const double              *B,
                                               const aoclsparse_order     orderB,
                                               const aoclsparse_int       ldb,
                                               const double               alpha,
                                               const double               beta,
                                               double                    *C,
                                               const aoclsparse_order     orderC,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1;
    // Check for valid pointer
    if(A == nullptr || B == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(!(A->val_type == aoclsparse_dmat))
    {
        return aoclsparse_status_wrong_type;
    }

    return aoclsparse_syprd<double>(op, A, B, orderB, ldb, alpha, beta, C, orderC, ldc, kid);
}

/*
 * Performs symmetric triple product of a sparse matrix and a dense matrix
 * for type complex float.
 */
extern "C" aoclsparse_status aoclsparse_csyprd(const aoclsparse_operation      op,
                                               const aoclsparse_matrix         A,
                                               const aoclsparse_float_complex *B,
                                               const aoclsparse_order          orderB,
                                               const aoclsparse_int            ldb,
                                               const aoclsparse_float_complex  alpha,
                                               const aoclsparse_float_complex  beta,
                                               aoclsparse_float_complex       *C,
                                               const aoclsparse_order          orderC,
                                               const aoclsparse_int            ldc)
{
    const aoclsparse_int kid = -1;
    // Check for valid pointer
    if(A == nullptr || B == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(!(A->val_type == aoclsparse_cmat))
    {
        return aoclsparse_status_wrong_type;
    }

    const std::complex<float>  alphap = *reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float>  betap  = *reinterpret_cast<const std::complex<float> *>(&beta);
    const std::complex<float> *b      = reinterpret_cast<const std::complex<float> *>(B);
    std::complex<float>       *c      = reinterpret_cast<std::complex<float> *>(C);

    return aoclsparse_syprd<std::complex<float>>(
        op, A, b, orderB, ldb, alphap, betap, c, orderC, ldc, kid);
}

/*
 * Performs symmetric triple product of a sparse matrix and a dense matrix
 * for type complex double.
 */
extern "C" aoclsparse_status aoclsparse_zsyprd(const aoclsparse_operation       op,
                                               const aoclsparse_matrix          A,
                                               const aoclsparse_double_complex *B,
                                               const aoclsparse_order           orderB,
                                               const aoclsparse_int             ldb,
                                               const aoclsparse_double_complex  alpha,
                                               const aoclsparse_double_complex  beta,
                                               aoclsparse_double_complex       *C,
                                               const aoclsparse_order           orderC,
                                               const aoclsparse_int             ldc)
{
    const aoclsparse_int kid = -1;
    // Check for valid pointer
    if(A == nullptr || B == nullptr || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(!(A->val_type == aoclsparse_zmat))
    {
        return aoclsparse_status_wrong_type;
    }

    const std::complex<double>  alphap = *reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double>  betap  = *reinterpret_cast<const std::complex<double> *>(&beta);
    const std::complex<double> *b      = reinterpret_cast<const std::complex<double> *>(B);
    std::complex<double>       *c      = reinterpret_cast<std::complex<double> *>(C);

    return aoclsparse_syprd<std::complex<double>>(
        op, A, b, orderB, ldb, alphap, betap, c, orderC, ldc, kid);
}
