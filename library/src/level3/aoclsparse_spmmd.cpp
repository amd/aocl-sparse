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
 * ************************************************************************
 */

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_sp2md.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/*
 * Computes the product of two sparse matrices (CSR or CSC format) and stores
 * the result in a dense matrix. Supports s/d/c/z data types.
 */
extern "C" aoclsparse_status aoclsparse_sspmmd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_matrix    B,
                                               const aoclsparse_order     layout,
                                               float                     *C,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    if((nullptr == A) || (nullptr == B))
        return aoclsparse_status_invalid_pointer;

    aoclsparse::csr *raw_A_w = A->get_first_mtx_if_valid<aoclsparse::csr>();
    aoclsparse::csr *raw_B_w = B->get_first_mtx_if_valid<aoclsparse::csr>();
    if(!raw_A_w || !raw_B_w)
        return aoclsparse_status_not_implemented;

    _aoclsparse_mat_descr descrA;
    descrA.base = raw_A_w->base;
    _aoclsparse_mat_descr descrB;
    descrB.base = raw_B_w->base;

    const aoclsparse_operation op_B  = aoclsparse_operation_none;
    const float                alpha = 1.0f;
    const float                beta  = 0.0f;

    return aoclsparse_sp2md_t(op, &descrA, A, op_B, &descrB, B, alpha, beta, C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_dspmmd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_matrix    B,
                                               const aoclsparse_order     layout,
                                               double                    *C,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    if((nullptr == A) || (nullptr == B))
        return aoclsparse_status_invalid_pointer;

    aoclsparse::csr *raw_A_w = A->get_first_mtx_if_valid<aoclsparse::csr>();
    aoclsparse::csr *raw_B_w = B->get_first_mtx_if_valid<aoclsparse::csr>();
    if(!raw_A_w || !raw_B_w)
        return aoclsparse_status_not_implemented;

    _aoclsparse_mat_descr descrA;
    descrA.base = raw_A_w->base;
    _aoclsparse_mat_descr descrB;
    descrB.base = raw_B_w->base;

    const aoclsparse_operation op_B  = aoclsparse_operation_none;
    const double               alpha = 1.0;
    const double               beta  = 0.0;

    return aoclsparse_sp2md_t(op, &descrA, A, op_B, &descrB, B, alpha, beta, C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_cspmmd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_matrix    B,
                                               const aoclsparse_order     layout,
                                               aoclsparse_float_complex  *C,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    if((nullptr == A) || (nullptr == B))
        return aoclsparse_status_invalid_pointer;

    aoclsparse::csr *raw_A_w = A->get_first_mtx_if_valid<aoclsparse::csr>();
    aoclsparse::csr *raw_B_w = B->get_first_mtx_if_valid<aoclsparse::csr>();
    if(!raw_A_w || !raw_B_w)
        return aoclsparse_status_not_implemented;

    _aoclsparse_mat_descr descrA;
    descrA.base = raw_A_w->base;
    _aoclsparse_mat_descr descrB;
    descrB.base = raw_B_w->base;

    const aoclsparse_operation op_B  = aoclsparse_operation_none;
    const std::complex<float>  alpha = 1.0f;
    const std::complex<float>  beta  = 0.0f;

    return aoclsparse_sp2md_t(
        op, &descrA, A, op_B, &descrB, B, alpha, beta, (std::complex<float> *)C, layout, ldc, kid);
}

extern "C" aoclsparse_status aoclsparse_zspmmd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_matrix    B,
                                               const aoclsparse_order     layout,
                                               aoclsparse_double_complex *C,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    if((nullptr == A) || (nullptr == B))
        return aoclsparse_status_invalid_pointer;

    aoclsparse::csr *raw_A_w = A->get_first_mtx_if_valid<aoclsparse::csr>();
    aoclsparse::csr *raw_B_w = B->get_first_mtx_if_valid<aoclsparse::csr>();
    if(!raw_A_w || !raw_B_w)
        return aoclsparse_status_not_implemented;

    _aoclsparse_mat_descr descrA;
    descrA.base = raw_A_w->base;
    _aoclsparse_mat_descr descrB;
    descrB.base = raw_B_w->base;

    const aoclsparse_operation op_B  = aoclsparse_operation_none;
    const std::complex<double> alpha = 1.0;
    const std::complex<double> beta  = 0.0;

    return aoclsparse_sp2md_t(
        op, &descrA, A, op_B, &descrB, B, alpha, beta, (std::complex<double> *)C, layout, ldc, kid);
}
