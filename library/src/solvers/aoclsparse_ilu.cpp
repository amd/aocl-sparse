/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_ilu.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
extern "C" aoclsparse_status aoclsparse_silu_smoother(aoclsparse_operation          op,
                                                      aoclsparse_matrix             A,
                                                      const aoclsparse_mat_descr    descr,
                                                      float                       **precond_csr_val,
                                                      [[maybe_unused]] const float *approx_inv_diag,
                                                      float                        *x,
                                                      const float                  *b)
{
    return aoclsparse_ilu_template<float>(op, A, descr, precond_csr_val, x, b);
}

extern "C" aoclsparse_status
    aoclsparse_dilu_smoother(aoclsparse_operation           op,
                             aoclsparse_matrix              A,
                             const aoclsparse_mat_descr     descr,
                             double                       **precond_csr_val,
                             [[maybe_unused]] const double *approx_inv_diag,
                             double                        *x,
                             const double                  *b)
{
    return aoclsparse_ilu_template<double>(op, A, descr, precond_csr_val, x, b);
}
extern "C" aoclsparse_status
    aoclsparse_cilu_smoother(aoclsparse_operation                             op,
                             aoclsparse_matrix                                A,
                             const aoclsparse_mat_descr                       descr,
                             aoclsparse_float_complex                       **precond_csr_val,
                             [[maybe_unused]] const aoclsparse_float_complex *approx_inv_diag,
                             aoclsparse_float_complex                        *x,
                             const aoclsparse_float_complex                  *b)
{
    return aoclsparse_ilu_template<std::complex<float>>(
        op,
        A,
        descr,
        reinterpret_cast<std::complex<float> **>(precond_csr_val),
        reinterpret_cast<std::complex<float> *>(x),
        reinterpret_cast<const std::complex<float> *>(b));
}
extern "C" aoclsparse_status
    aoclsparse_zilu_smoother(aoclsparse_operation                              op,
                             aoclsparse_matrix                                 A,
                             const aoclsparse_mat_descr                        descr,
                             aoclsparse_double_complex                       **precond_csr_val,
                             [[maybe_unused]] const aoclsparse_double_complex *approx_inv_diag,
                             aoclsparse_double_complex                        *x,
                             const aoclsparse_double_complex                  *b)
{
    return aoclsparse_ilu_template<std::complex<double>>(
        op,
        A,
        descr,
        reinterpret_cast<std::complex<double> **>(precond_csr_val),
        reinterpret_cast<std::complex<double> *>(x),
        reinterpret_cast<const std::complex<double> *>(b));
}