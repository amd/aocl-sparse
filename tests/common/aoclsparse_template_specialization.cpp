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
#include "aoclsparse.hpp"

#include <complex>

template <>
aoclsparse_status aoclsparse_csr2m<double>(aoclsparse_operation       transA,
                                           const aoclsparse_mat_descr descrA,
                                           const aoclsparse_matrix    csrA,
                                           aoclsparse_operation       transB,
                                           const aoclsparse_mat_descr descrB,
                                           const aoclsparse_matrix    csrB,
                                           aoclsparse_request         request,
                                           aoclsparse_matrix         *csrC)
{
    return aoclsparse_dcsr2m(transA, descrA, csrA, transB, descrB, csrB, request, csrC);
}

template <>
aoclsparse_status aoclsparse_csr2m<float>(aoclsparse_operation       transA,
                                          const aoclsparse_mat_descr descrA,
                                          const aoclsparse_matrix    csrA,
                                          aoclsparse_operation       transB,
                                          const aoclsparse_mat_descr descrB,
                                          const aoclsparse_matrix    csrB,
                                          aoclsparse_request         request,
                                          aoclsparse_matrix         *csrC)
{
    return aoclsparse_scsr2m(transA, descrA, csrA, transB, descrB, csrB, request, csrC);
}

template <>
aoclsparse_status aoclsparse_csrmm(aoclsparse_operation       op,
                                   float                      alpha,
                                   const aoclsparse_matrix    A,
                                   const aoclsparse_mat_descr descr,
                                   aoclsparse_order           order,
                                   const float               *B,
                                   aoclsparse_int             n,
                                   aoclsparse_int             ldb,
                                   float                      beta,
                                   float                     *C,
                                   aoclsparse_int             ldc)
{
    return aoclsparse_scsrmm(op, alpha, A, descr, order, B, n, ldb, beta, C, ldc);
}

template <>
aoclsparse_status aoclsparse_csrmm(aoclsparse_operation       op,
                                   double                     alpha,
                                   const aoclsparse_matrix    A,
                                   const aoclsparse_mat_descr descr,
                                   aoclsparse_order           order,
                                   const double              *B,
                                   aoclsparse_int             n,
                                   aoclsparse_int             ldb,
                                   double                     beta,
                                   double                    *C,
                                   aoclsparse_int             ldc)
{
    return aoclsparse_dcsrmm(op, alpha, A, descr, order, B, n, ldb, beta, C, ldc);
}

template <>
aoclsparse_status aoclsparse_csrmm(aoclsparse_operation            op,
                                   aoclsparse_float_complex        alpha,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_mat_descr      descr,
                                   aoclsparse_order                order,
                                   const aoclsparse_float_complex *B,
                                   aoclsparse_int                  n,
                                   aoclsparse_int                  ldb,
                                   aoclsparse_float_complex        beta,
                                   aoclsparse_float_complex       *C,
                                   aoclsparse_int                  ldc)
{
    return aoclsparse_ccsrmm(op, alpha, A, descr, order, B, n, ldb, beta, C, ldc);
}

template <>
aoclsparse_status aoclsparse_csrmm(aoclsparse_operation             op,
                                   aoclsparse_double_complex        alpha,
                                   const aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr       descr,
                                   aoclsparse_order                 order,
                                   const aoclsparse_double_complex *B,
                                   aoclsparse_int                   n,
                                   aoclsparse_int                   ldb,
                                   aoclsparse_double_complex        beta,
                                   aoclsparse_double_complex       *C,
                                   aoclsparse_int                   ldc)
{
    return aoclsparse_zcsrmm(op, alpha, A, descr, order, B, n, ldb, beta, C, ldc);
}

template <>
aoclsparse_status aoclsparse_add(const aoclsparse_operation op,
                                 const aoclsparse_matrix    A,
                                 const float                alpha,
                                 const aoclsparse_matrix    B,
                                 aoclsparse_matrix         *C)
{
    return aoclsparse_sadd(op, A, alpha, B, C);
}

template <>
aoclsparse_status aoclsparse_add(const aoclsparse_operation op,
                                 const aoclsparse_matrix    A,
                                 const double               alpha,
                                 const aoclsparse_matrix    B,
                                 aoclsparse_matrix         *C)
{
    return aoclsparse_dadd(op, A, alpha, B, C);
}

template <>
aoclsparse_status aoclsparse_add(const aoclsparse_operation op,
                                 const aoclsparse_matrix    A,
                                 const std::complex<float>  alpha,
                                 const aoclsparse_matrix    B,
                                 aoclsparse_matrix         *C)
{
    return aoclsparse_cadd(
        op, A, aoclsparse_float_complex{std::real(alpha), std::imag(alpha)}, B, C);
}

template <>
aoclsparse_status aoclsparse_add(const aoclsparse_operation op,
                                 const aoclsparse_matrix    A,
                                 const std::complex<double> alpha,
                                 const aoclsparse_matrix    B,
                                 aoclsparse_matrix         *C)
{
    return aoclsparse_zadd(
        op, A, aoclsparse_double_complex{std::real(alpha), std::imag(alpha)}, B, C);
}

template <>
aoclsparse_status aoclsparse_add(const aoclsparse_operation     op,
                                 const aoclsparse_matrix        A,
                                 const aoclsparse_float_complex alpha,
                                 const aoclsparse_matrix        B,
                                 aoclsparse_matrix             *C)
{
    return aoclsparse_cadd(op, A, alpha, B, C);
}

template <>
aoclsparse_status aoclsparse_add(const aoclsparse_operation      op,
                                 const aoclsparse_matrix         A,
                                 const aoclsparse_double_complex alpha,
                                 const aoclsparse_matrix         B,
                                 aoclsparse_matrix              *C)
{
    return aoclsparse_zadd(op, A, alpha, B, C);
}

template <>
aoclsparse_status aoclsparse_spmmd(aoclsparse_operation            op,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_matrix         B,
                                   aoclsparse_order                layout,
                                   float                          *C,
                                   aoclsparse_int                  ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    return aoclsparse_sspmmd(op, A, B, layout, C, ldc);
}

template <>
aoclsparse_status aoclsparse_spmmd(aoclsparse_operation            op,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_matrix         B,
                                   aoclsparse_order                layout,
                                   double                         *C,
                                   aoclsparse_int                  ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    return aoclsparse_dspmmd(op, A, B, layout, C, ldc);
}

template <>
aoclsparse_status aoclsparse_spmmd(aoclsparse_operation            op,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_matrix         B,
                                   aoclsparse_order                layout,
                                   aoclsparse_float_complex       *C,
                                   aoclsparse_int                  ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    return aoclsparse_cspmmd(op, A, B, layout, C, ldc);
}

template <>
aoclsparse_status aoclsparse_spmmd(aoclsparse_operation            op,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_matrix         B,
                                   aoclsparse_order                layout,
                                   aoclsparse_double_complex      *C,
                                   aoclsparse_int                  ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    return aoclsparse_zspmmd(op, A, B, layout, C, ldc);
}
template <>
aoclsparse_status aoclsparse_sp2md(const aoclsparse_operation      opA,
                                   const aoclsparse_mat_descr      descrA,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_operation      opB,
                                   const aoclsparse_mat_descr      descrB,
                                   const aoclsparse_matrix         B,
                                   const float                     alpha,
                                   const float                     beta,
                                   float                          *C,
                                   const aoclsparse_order          layout,
                                   const aoclsparse_int            ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    return aoclsparse_ssp2md(opA, descrA, A, opB, descrB, B, alpha, beta, C, layout, ldc);
}

template <>
aoclsparse_status aoclsparse_sp2md(const aoclsparse_operation      opA,
                                   const aoclsparse_mat_descr      descrA,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_operation      opB,
                                   const aoclsparse_mat_descr      descrB,
                                   const aoclsparse_matrix         B,
                                   const double                    alpha,
                                   const double                    beta,
                                   double                         *C,
                                   const aoclsparse_order          layout,
                                   const aoclsparse_int            ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    return aoclsparse_dsp2md(opA, descrA, A, opB, descrB, B, alpha, beta, C, layout, ldc);
}

template <>
aoclsparse_status aoclsparse_sp2md(const aoclsparse_operation      opA,
                                   const aoclsparse_mat_descr      descrA,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_operation      opB,
                                   const aoclsparse_mat_descr      descrB,
                                   const aoclsparse_matrix         B,
                                   aoclsparse_float_complex        alpha,
                                   aoclsparse_float_complex        beta,
                                   aoclsparse_float_complex       *C,
                                   const aoclsparse_order          layout,
                                   const aoclsparse_int            ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    return aoclsparse_csp2md(opA, descrA, A, opB, descrB, B, alpha, beta, C, layout, ldc);
}

template <>
aoclsparse_status aoclsparse_sp2md(const aoclsparse_operation      opA,
                                   const aoclsparse_mat_descr      descrA,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_operation      opB,
                                   const aoclsparse_mat_descr      descrB,
                                   const aoclsparse_matrix         B,
                                   aoclsparse_double_complex       alpha,
                                   aoclsparse_double_complex       beta,
                                   aoclsparse_double_complex      *C,
                                   const aoclsparse_order          layout,
                                   const aoclsparse_int            ldc,
                                   [[maybe_unused]] aoclsparse_int kid)
{
    return aoclsparse_zsp2md(opA, descrA, A, opB, descrB, B, alpha, beta, C, layout, ldc);
}

template <>
aoclsparse_status aoclsparse_syrkd(aoclsparse_operation    op,
                                   const aoclsparse_matrix A,
                                   float                   alpha,
                                   float                   beta,
                                   float                  *C,
                                   aoclsparse_order        orderC,
                                   aoclsparse_int          ldc)
{
    return aoclsparse_ssyrkd(op, A, alpha, beta, C, orderC, ldc);
}

template <>
aoclsparse_status aoclsparse_syrkd(aoclsparse_operation    op,
                                   const aoclsparse_matrix A,
                                   double                  alpha,
                                   double                  beta,
                                   double                 *C,
                                   aoclsparse_order        orderC,
                                   aoclsparse_int          ldc)
{
    return aoclsparse_dsyrkd(op, A, alpha, beta, C, orderC, ldc);
}

template <>
aoclsparse_status aoclsparse_syrkd(aoclsparse_operation      op,
                                   const aoclsparse_matrix   A,
                                   aoclsparse_float_complex  alpha,
                                   aoclsparse_float_complex  beta,
                                   aoclsparse_float_complex *C,
                                   aoclsparse_order          orderC,
                                   aoclsparse_int            ldc)
{
    return aoclsparse_csyrkd(op, A, alpha, beta, C, orderC, ldc);
}

template <>
aoclsparse_status aoclsparse_syrkd(aoclsparse_operation       op,
                                   const aoclsparse_matrix    A,
                                   aoclsparse_double_complex  alpha,
                                   aoclsparse_double_complex  beta,
                                   aoclsparse_double_complex *C,
                                   aoclsparse_order           orderC,
                                   aoclsparse_int             ldc)
{
    return aoclsparse_zsyrkd(op, A, alpha, beta, C, orderC, ldc);
}

template <>
aoclsparse_status aoclsparse_syprd(aoclsparse_operation    op,
                                   const aoclsparse_matrix A,
                                   const float            *B,
                                   aoclsparse_order        orderB,
                                   aoclsparse_int          ldb,
                                   float                   alpha,
                                   float                   beta,
                                   float                  *C,
                                   aoclsparse_order        orderC,
                                   aoclsparse_int          ldc)
{
    return aoclsparse_ssyprd(op, A, B, orderB, ldb, alpha, beta, C, orderC, ldc);
}

template <>
aoclsparse_status aoclsparse_syprd(aoclsparse_operation    op,
                                   const aoclsparse_matrix A,
                                   const double           *B,
                                   aoclsparse_order        orderB,
                                   aoclsparse_int          ldb,
                                   double                  alpha,
                                   double                  beta,
                                   double                 *C,
                                   aoclsparse_order        orderC,
                                   aoclsparse_int          ldc)
{
    return aoclsparse_dsyprd(op, A, B, orderB, ldb, alpha, beta, C, orderC, ldc);
}

template <>
aoclsparse_status aoclsparse_syprd(aoclsparse_operation            op,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_float_complex *B,
                                   aoclsparse_order                orderB,
                                   aoclsparse_int                  ldb,
                                   aoclsparse_float_complex        alpha,
                                   aoclsparse_float_complex        beta,
                                   aoclsparse_float_complex       *C,
                                   aoclsparse_order                orderC,
                                   aoclsparse_int                  ldc)
{
    return aoclsparse_csyprd(op, A, B, orderB, ldb, alpha, beta, C, orderC, ldc);
}

template <>
aoclsparse_status aoclsparse_syprd(aoclsparse_operation             op,
                                   const aoclsparse_matrix          A,
                                   const aoclsparse_double_complex *B,
                                   aoclsparse_order                 orderB,
                                   aoclsparse_int                   ldb,
                                   aoclsparse_double_complex        alpha,
                                   aoclsparse_double_complex        beta,
                                   aoclsparse_double_complex       *C,
                                   aoclsparse_order                 orderC,
                                   aoclsparse_int                   ldc)
{
    return aoclsparse_zsyprd(op, A, B, orderB, ldb, alpha, beta, C, orderC, ldc);
}

template <>
aoclsparse_status aoclsparse_trsm_kid(const aoclsparse_operation trans,
                                      const float                alpha,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      aoclsparse_order           order,
                                      const float               *B,
                                      aoclsparse_int             n,
                                      aoclsparse_int             ldb,
                                      float                     *X,
                                      aoclsparse_int             ldx,
                                      const aoclsparse_int       kid)
{
    if(kid >= 0)
        return aoclsparse_strsm_kid(trans, alpha, A, descr, order, B, n, ldb, X, ldx, kid);
    else
        return aoclsparse_strsm(trans, alpha, A, descr, order, B, n, ldb, X, ldx);
}

template <>
aoclsparse_status aoclsparse_trsm_kid(const aoclsparse_operation trans,
                                      const double               alpha,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      aoclsparse_order           order,
                                      const double              *B,
                                      aoclsparse_int             n,
                                      aoclsparse_int             ldb,
                                      double                    *X,
                                      aoclsparse_int             ldx,
                                      const aoclsparse_int       kid)
{
    if(kid >= 0)
        return aoclsparse_dtrsm_kid(trans, alpha, A, descr, order, B, n, ldb, X, ldx, kid);
    else
        return aoclsparse_dtrsm(trans, alpha, A, descr, order, B, n, ldb, X, ldx);
}

template <>
aoclsparse_status aoclsparse_trsm_kid(const aoclsparse_operation trans,
                                      const std::complex<float>  alpha,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      aoclsparse_order           order,
                                      const std::complex<float> *B,
                                      aoclsparse_int             n,
                                      aoclsparse_int             ldb,
                                      std::complex<float>       *X,
                                      aoclsparse_int             ldx,
                                      const aoclsparse_int       kid)
{
    const aoclsparse_float_complex *alphap
        = reinterpret_cast<const aoclsparse_float_complex *>(&alpha);
    const aoclsparse_float_complex *Bp = reinterpret_cast<const aoclsparse_float_complex *>(B);
    aoclsparse_float_complex       *Xp = reinterpret_cast<aoclsparse_float_complex *>(X);
    if(kid >= 0)
        return aoclsparse_ctrsm_kid(trans, *alphap, A, descr, order, Bp, n, ldb, Xp, ldx, kid);
    else
        return aoclsparse_ctrsm(trans, *alphap, A, descr, order, Bp, n, ldb, Xp, ldx);
}
template <>
aoclsparse_status aoclsparse_trsm_kid(const aoclsparse_operation      trans,
                                      const aoclsparse_float_complex  alpha,
                                      aoclsparse_matrix               A,
                                      const aoclsparse_mat_descr      descr,
                                      aoclsparse_order                order,
                                      const aoclsparse_float_complex *B,
                                      aoclsparse_int                  n,
                                      aoclsparse_int                  ldb,
                                      aoclsparse_float_complex       *X,
                                      aoclsparse_int                  ldx,
                                      const aoclsparse_int            kid)
{
    if(kid >= 0)
        return aoclsparse_ctrsm_kid(trans, alpha, A, descr, order, B, n, ldb, X, ldx, kid);
    else
        return aoclsparse_ctrsm(trans, alpha, A, descr, order, B, n, ldb, X, ldx);
}

template <>
aoclsparse_status aoclsparse_trsm_kid(const aoclsparse_operation  trans,
                                      const std::complex<double>  alpha,
                                      aoclsparse_matrix           A,
                                      const aoclsparse_mat_descr  descr,
                                      aoclsparse_order            order,
                                      const std::complex<double> *B,
                                      aoclsparse_int              n,
                                      aoclsparse_int              ldb,
                                      std::complex<double>       *X,
                                      aoclsparse_int              ldx,
                                      const aoclsparse_int        kid)
{
    const aoclsparse_double_complex *alphap
        = reinterpret_cast<const aoclsparse_double_complex *>(&alpha);
    const aoclsparse_double_complex *Bp = reinterpret_cast<const aoclsparse_double_complex *>(B);
    aoclsparse_double_complex       *Xp = reinterpret_cast<aoclsparse_double_complex *>(X);
    if(kid >= 0)
        return aoclsparse_ztrsm_kid(trans, *alphap, A, descr, order, Bp, n, ldb, Xp, ldx, kid);
    else
        return aoclsparse_ztrsm(trans, *alphap, A, descr, order, Bp, n, ldb, Xp, ldx);
}

template <>
aoclsparse_status aoclsparse_trsm_kid(const aoclsparse_operation       trans,
                                      const aoclsparse_double_complex  alpha,
                                      aoclsparse_matrix                A,
                                      const aoclsparse_mat_descr       descr,
                                      aoclsparse_order                 order,
                                      const aoclsparse_double_complex *B,
                                      aoclsparse_int                   n,
                                      aoclsparse_int                   ldb,
                                      aoclsparse_double_complex       *X,
                                      aoclsparse_int                   ldx,
                                      const aoclsparse_int             kid)
{
    if(kid >= 0)
        return aoclsparse_ztrsm_kid(trans, alpha, A, descr, order, B, n, ldb, X, ldx, kid);
    else
        return aoclsparse_ztrsm(trans, alpha, A, descr, order, B, n, ldb, X, ldx);
}

template <>
aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                   const float               *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const float               *csr_val,
                                   const aoclsparse_int      *csr_col_ind,
                                   const aoclsparse_int      *csr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const float               *x,
                                   const float               *beta,
                                   float                     *y)
{
    return aoclsparse_scsrmv(
        trans, alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                   const double              *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const double              *csr_val,
                                   const aoclsparse_int      *csr_col_ind,
                                   const aoclsparse_int      *csr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const double              *x,
                                   const double              *beta,
                                   double                    *y)
{
    return aoclsparse_dcsrmv(
        trans, alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_blkcsrmv(aoclsparse_operation       trans,
                                      const double              *alpha,
                                      aoclsparse_int             m,
                                      aoclsparse_int             n,
                                      aoclsparse_int             nnz,
                                      const uint8_t             *masks,
                                      const double              *blk_csr_val,
                                      const aoclsparse_int      *blk_col_ind,
                                      const aoclsparse_int      *blk_row_ptr,
                                      const aoclsparse_mat_descr descr,
                                      const double              *x,
                                      const double              *beta,
                                      double                    *y,
                                      aoclsparse_int             nRowsblk)
{
    return aoclsparse_dblkcsrmv(trans,
                                alpha,
                                m,
                                n,
                                nnz,
                                masks,
                                blk_csr_val,
                                blk_col_ind,
                                blk_row_ptr,
                                descr,
                                x,
                                beta,
                                y,
                                nRowsblk);
}

template <>
aoclsparse_status aoclsparse_ellmv(aoclsparse_operation       trans,
                                   const float               *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const float               *ell_val,
                                   const aoclsparse_int      *ell_col_ind,
                                   const aoclsparse_int       ell_width,
                                   const aoclsparse_mat_descr descr,
                                   const float               *x,
                                   const float               *beta,
                                   float                     *y)
{
    return aoclsparse_sellmv(
        trans, alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_ellmv(aoclsparse_operation       trans,
                                   const double              *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const double              *ell_val,
                                   const aoclsparse_int      *ell_col_ind,
                                   const aoclsparse_int       ell_width,
                                   const aoclsparse_mat_descr descr,
                                   const double              *x,
                                   const double              *beta,
                                   double                    *y)
{
    return aoclsparse_dellmv(
        trans, alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_elltmv(aoclsparse_operation       trans,
                                    const float               *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const float               *ell_val,
                                    const aoclsparse_int      *ell_col_ind,
                                    const aoclsparse_int       ell_width,
                                    const aoclsparse_mat_descr descr,
                                    const float               *x,
                                    const float               *beta,
                                    float                     *y)
{
    return aoclsparse_selltmv(
        trans, alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_elltmv(aoclsparse_operation       trans,
                                    const double              *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const double              *ell_val,
                                    const aoclsparse_int      *ell_col_ind,
                                    const aoclsparse_int       ell_width,
                                    const aoclsparse_mat_descr descr,
                                    const double              *x,
                                    const double              *beta,
                                    double                    *y)
{
    return aoclsparse_delltmv(
        trans, alpha, m, n, nnz, ell_val, ell_col_ind, ell_width, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_ellthybmv(aoclsparse_operation       trans,
                                       const float               *alpha,
                                       aoclsparse_int             m,
                                       aoclsparse_int             n,
                                       aoclsparse_int             nnz,
                                       const float               *ell_val,
                                       const aoclsparse_int      *ell_col_ind,
                                       const aoclsparse_int       ell_width,
                                       const aoclsparse_int       ell_m,
                                       const float               *csr_val,
                                       const aoclsparse_int      *csr_row_ind,
                                       const aoclsparse_int      *csr_col_ind,
                                       aoclsparse_int            *row_idx_map,
                                       aoclsparse_int            *csr_row_idx_map,
                                       const aoclsparse_mat_descr descr,
                                       const float               *x,
                                       const float               *beta,
                                       float                     *y)
{
    return aoclsparse_sellthybmv(trans,
                                 alpha,
                                 m,
                                 n,
                                 nnz,
                                 ell_val,
                                 ell_col_ind,
                                 ell_width,
                                 ell_m,
                                 csr_val,
                                 csr_row_ind,
                                 csr_col_ind,
                                 row_idx_map,
                                 csr_row_idx_map,
                                 descr,
                                 x,
                                 beta,
                                 y);
}

template <>
aoclsparse_status aoclsparse_ellthybmv(aoclsparse_operation       trans,
                                       const double              *alpha,
                                       aoclsparse_int             m,
                                       aoclsparse_int             n,
                                       aoclsparse_int             nnz,
                                       const double              *ell_val,
                                       const aoclsparse_int      *ell_col_ind,
                                       const aoclsparse_int       ell_width,
                                       const aoclsparse_int       ell_m,
                                       const double              *csr_val,
                                       const aoclsparse_int      *csr_row_ind,
                                       const aoclsparse_int      *csr_col_ind,
                                       aoclsparse_int            *row_idx_map,
                                       aoclsparse_int            *csr_row_idx_map,
                                       const aoclsparse_mat_descr descr,
                                       const double              *x,
                                       const double              *beta,
                                       double                    *y)
{
    return aoclsparse_dellthybmv(trans,
                                 alpha,
                                 m,
                                 n,
                                 nnz,
                                 ell_val,
                                 ell_col_ind,
                                 ell_width,
                                 ell_m,
                                 csr_val,
                                 csr_row_ind,
                                 csr_col_ind,
                                 row_idx_map,
                                 csr_row_idx_map,
                                 descr,
                                 x,
                                 beta,
                                 y);
}

template <>
aoclsparse_status aoclsparse_mv(aoclsparse_operation       op,
                                const float               *alpha,
                                aoclsparse_matrix          A,
                                const aoclsparse_mat_descr descr,
                                const float               *x,
                                const float               *beta,
                                float                     *y)
{
    return aoclsparse_smv(op, alpha, A, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_mv(aoclsparse_operation       op,
                                const double              *alpha,
                                aoclsparse_matrix          A,
                                const aoclsparse_mat_descr descr,
                                const double              *x,
                                const double              *beta,
                                double                    *y)
{
    return aoclsparse_dmv(op, alpha, A, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_mv(aoclsparse_operation             op,
                                const aoclsparse_double_complex *alpha,
                                aoclsparse_matrix                A,
                                const aoclsparse_mat_descr       descr,
                                const aoclsparse_double_complex *x,
                                const aoclsparse_double_complex *beta,
                                aoclsparse_double_complex       *y)
{
    return aoclsparse_zmv(op, alpha, A, descr, x, beta, y);
}
template <>
aoclsparse_status aoclsparse_mv(aoclsparse_operation        op,
                                const std::complex<double> *alpha,
                                aoclsparse_matrix           A,
                                const aoclsparse_mat_descr  descr,
                                const std::complex<double> *x,
                                const std::complex<double> *beta,
                                std::complex<double>       *y)
{
    const aoclsparse_double_complex *palpha
        = reinterpret_cast<const aoclsparse_double_complex *>(alpha);
    const aoclsparse_double_complex *pbeta
        = reinterpret_cast<const aoclsparse_double_complex *>(beta);
    aoclsparse_double_complex       *py = reinterpret_cast<aoclsparse_double_complex *>(y);
    const aoclsparse_double_complex *px = reinterpret_cast<const aoclsparse_double_complex *>(x);
    return aoclsparse_zmv(op, palpha, A, descr, px, pbeta, py);
}
template <>
aoclsparse_status aoclsparse_mv(aoclsparse_operation            op,
                                const aoclsparse_float_complex *alpha,
                                aoclsparse_matrix               A,
                                const aoclsparse_mat_descr      descr,
                                const aoclsparse_float_complex *x,
                                const aoclsparse_float_complex *beta,
                                aoclsparse_float_complex       *y)
{
    return aoclsparse_cmv(op, alpha, A, descr, x, beta, y);
}
template <>
aoclsparse_status aoclsparse_mv(aoclsparse_operation       op,
                                const std::complex<float> *alpha,
                                aoclsparse_matrix          A,
                                const aoclsparse_mat_descr descr,
                                const std::complex<float> *x,
                                const std::complex<float> *beta,
                                std::complex<float>       *y)
{
    const aoclsparse_float_complex *palpha
        = reinterpret_cast<const aoclsparse_float_complex *>(alpha);
    const aoclsparse_float_complex *pbeta
        = reinterpret_cast<const aoclsparse_float_complex *>(beta);
    aoclsparse_float_complex       *py = reinterpret_cast<aoclsparse_float_complex *>(y);
    const aoclsparse_float_complex *px = reinterpret_cast<const aoclsparse_float_complex *>(x);
    return aoclsparse_cmv(op, palpha, A, descr, px, pbeta, py);
}
template <>
aoclsparse_status aoclsparse_diamv(aoclsparse_operation       trans,
                                   const float               *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const float               *dia_val,
                                   const aoclsparse_int      *dia_offset,
                                   aoclsparse_int             dia_num_diag,
                                   const aoclsparse_mat_descr descr,
                                   const float               *x,
                                   const float               *beta,
                                   float                     *y)
{
    return aoclsparse_sdiamv(
        trans, alpha, m, n, nnz, dia_val, dia_offset, dia_num_diag, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_diamv(aoclsparse_operation       trans,
                                   const double              *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const double              *dia_val,
                                   const aoclsparse_int      *dia_offset,
                                   aoclsparse_int             dia_num_diag,
                                   const aoclsparse_mat_descr descr,
                                   const double              *x,
                                   const double              *beta,
                                   double                    *y)
{
    return aoclsparse_ddiamv(
        trans, alpha, m, n, nnz, dia_val, dia_offset, dia_num_diag, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_bsrmv(aoclsparse_operation       trans,
                                   const float               *alpha,
                                   aoclsparse_int             mb,
                                   aoclsparse_int             nb,
                                   aoclsparse_int             bsr_dim,
                                   const float               *bsr_val,
                                   const aoclsparse_int      *bsr_col_ind,
                                   const aoclsparse_int      *bsr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const float               *x,
                                   const float               *beta,
                                   float                     *y)
{
    return aoclsparse_sbsrmv(
        trans, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_bsrmv(aoclsparse_operation       trans,
                                   const double              *alpha,
                                   aoclsparse_int             mb,
                                   aoclsparse_int             nb,
                                   aoclsparse_int             bsr_dim,
                                   const double              *bsr_val,
                                   const aoclsparse_int      *bsr_col_ind,
                                   const aoclsparse_int      *bsr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const double              *x,
                                   const double              *beta,
                                   double                    *y)
{
    return aoclsparse_dbsrmv(
        trans, alpha, mb, nb, bsr_dim, bsr_val, bsr_col_ind, bsr_row_ptr, descr, x, beta, y);
}

template <>
aoclsparse_status aoclsparse_csrsv(aoclsparse_operation       trans,
                                   const float               *alpha,
                                   aoclsparse_int             m,
                                   const float               *csr_val,
                                   const aoclsparse_int      *csr_col_ind,
                                   const aoclsparse_int      *csr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const float               *x,
                                   float                     *y)
{
    return aoclsparse_scsrsv(trans, alpha, m, csr_val, csr_col_ind, csr_row_ptr, descr, x, y);
}

template <>
aoclsparse_status aoclsparse_csrsv(aoclsparse_operation       trans,
                                   const double              *alpha,
                                   aoclsparse_int             m,
                                   const double              *csr_val,
                                   const aoclsparse_int      *csr_col_ind,
                                   const aoclsparse_int      *csr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const double              *x,
                                   double                    *y)
{
    return aoclsparse_dcsrsv(trans, alpha, m, csr_val, csr_col_ind, csr_row_ptr, descr, x, y);
}

template <>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int  nnz,
                                  const double         *y,
                                  double               *x,
                                  const aoclsparse_int *indx,
                                  aoclsparse_int        kid)
{
    if(kid == -1)
        return aoclsparse_dgthr(nnz, y, x, indx);
    else
        return aoclsparse_dgthr_kid(nnz, y, x, indx, kid);
}

template <>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int  nnz,
                                  const float          *y,
                                  float                *x,
                                  const aoclsparse_int *indx,
                                  aoclsparse_int        kid)
{
    if(kid == -1)
        return aoclsparse_sgthr(nnz, y, x, indx);
    else
        return aoclsparse_sgthr_kid(nnz, y, x, indx, kid);
}
template <>
aoclsparse_status aoclsparse_trsv_kid(const aoclsparse_operation trans,
                                      const double               alpha,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      const double              *b,
                                      double                    *x,
                                      const aoclsparse_int       kid)
{
    if(kid >= 0)
        return aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid);
    else
        return aoclsparse_dtrsv(trans, alpha, A, descr, b, x);
}

template <>
aoclsparse_status aoclsparse_trsv_kid(const aoclsparse_operation trans,
                                      const float                alpha,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      const float               *b,
                                      float                     *x,
                                      const aoclsparse_int       kid)
{
    if(kid >= 0)
        return aoclsparse_strsv_kid(trans, alpha, A, descr, b, x, kid);
    else
        return aoclsparse_strsv(trans, alpha, A, descr, b, x);
}
template <>
aoclsparse_status aoclsparse_trsv_kid(const aoclsparse_operation  trans,
                                      const std::complex<double>  alpha,
                                      aoclsparse_matrix           A,
                                      const aoclsparse_mat_descr  descr,
                                      const std::complex<double> *b,
                                      std::complex<double>       *x,
                                      const aoclsparse_int        kid)
{
    const aoclsparse_double_complex *palpha
        = reinterpret_cast<const aoclsparse_double_complex *>(&alpha);
    const aoclsparse_double_complex *pb = reinterpret_cast<const aoclsparse_double_complex *>(b);
    aoclsparse_double_complex       *px = reinterpret_cast<aoclsparse_double_complex *>(x);
    if(kid >= 0)
        return aoclsparse_ztrsv_kid(trans, *palpha, A, descr, pb, px, kid);
    else
        return aoclsparse_ztrsv(trans, *palpha, A, descr, pb, px);
}
template <>
aoclsparse_status aoclsparse_trsv_kid(const aoclsparse_operation       trans,
                                      const aoclsparse_double_complex  alpha,
                                      aoclsparse_matrix                A,
                                      const aoclsparse_mat_descr       descr,
                                      const aoclsparse_double_complex *b,
                                      aoclsparse_double_complex       *x,
                                      const aoclsparse_int             kid)
{
    if(kid >= 0)
        return aoclsparse_ztrsv_kid(trans, alpha, A, descr, b, x, kid);
    else
        return aoclsparse_ztrsv(trans, alpha, A, descr, b, x);
}
template <>
aoclsparse_status aoclsparse_trsv_kid(const aoclsparse_operation trans,
                                      const std::complex<float>  alpha,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      const std::complex<float> *b,
                                      std::complex<float>       *x,
                                      const aoclsparse_int       kid)
{
    const aoclsparse_float_complex *palpha
        = reinterpret_cast<const aoclsparse_float_complex *>(&alpha);
    const aoclsparse_float_complex *pb = reinterpret_cast<const aoclsparse_float_complex *>(b);
    aoclsparse_float_complex       *px = reinterpret_cast<aoclsparse_float_complex *>(x);
    if(kid >= 0)
        return aoclsparse_ctrsv_kid(trans, *palpha, A, descr, pb, px, kid);
    else
        return aoclsparse_ctrsv(trans, *palpha, A, descr, pb, px);
}
template <>
aoclsparse_status aoclsparse_trsv_kid(const aoclsparse_operation      trans,
                                      const aoclsparse_float_complex  alpha,
                                      aoclsparse_matrix               A,
                                      const aoclsparse_mat_descr      descr,
                                      const aoclsparse_float_complex *b,
                                      aoclsparse_float_complex       *x,
                                      const aoclsparse_int            kid)
{
    if(kid >= 0)
        return aoclsparse_ctrsv_kid(trans, alpha, A, descr, b, x, kid);
    else
        return aoclsparse_ctrsv(trans, alpha, A, descr, b, x);
}

template <>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int        nnz,
                                  const std::complex<double> *y,
                                  std::complex<double>       *x,
                                  const aoclsparse_int       *indx,
                                  aoclsparse_int              kid)
{
    if(kid == -1)
        return aoclsparse_zgthr(nnz, y, x, indx);
    else
        return aoclsparse_zgthr_kid(nnz, y, x, indx, kid);
}
template <>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int             nnz,
                                  const aoclsparse_double_complex *y,
                                  aoclsparse_double_complex       *x,
                                  const aoclsparse_int            *indx,
                                  aoclsparse_int                   kid)
{
    const std::complex<double> *py = reinterpret_cast<const std::complex<double> *>(y);
    std::complex<double>       *px = reinterpret_cast<std::complex<double> *>(x);
    if(kid == -1)
        return aoclsparse_zgthr(nnz, py, px, indx);
    else
        return aoclsparse_zgthr_kid(nnz, y, x, indx, kid);
}

template <>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int       nnz,
                                  const std::complex<float> *y,
                                  std::complex<float>       *x,
                                  const aoclsparse_int      *indx,
                                  aoclsparse_int             kid)
{
    if(kid == -1)
        return aoclsparse_cgthr(nnz, y, x, indx);
    else
        return aoclsparse_cgthr_kid(nnz, y, x, indx, kid);
}

template <>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int            nnz,
                                  const aoclsparse_float_complex *y,
                                  aoclsparse_float_complex       *x,
                                  const aoclsparse_int           *indx,
                                  aoclsparse_int                  kid)
{
    const std::complex<float> *py = reinterpret_cast<const std::complex<float> *>(y);
    std::complex<float>       *px = reinterpret_cast<std::complex<float> *>(x);
    if(kid == -1)
        return aoclsparse_cgthr(nnz, py, px, indx);
    else
        return aoclsparse_cgthr_kid(nnz, y, x, indx, kid);
}

template <>
aoclsparse_status aoclsparse_gthrz(
    const aoclsparse_int nnz, double *y, double *x, const aoclsparse_int *indx, aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_dgthrz(nnz, y, x, indx);
    else
        return aoclsparse_dgthrz_kid(nnz, y, x, indx, kid);
}

template <>
aoclsparse_status aoclsparse_gthrz(
    const aoclsparse_int nnz, float *y, float *x, const aoclsparse_int *indx, aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_sgthrz(nnz, y, x, indx);
    else
        return aoclsparse_sgthrz_kid(nnz, y, x, indx, kid);
}

template <>
aoclsparse_status aoclsparse_gthrz(const aoclsparse_int  nnz,
                                   std::complex<double> *y,
                                   std::complex<double> *x,
                                   const aoclsparse_int *indx,
                                   aoclsparse_int        kid)
{
    if(kid == -1)
        return aoclsparse_zgthrz(nnz, y, x, indx);
    else
        return aoclsparse_zgthrz_kid(nnz, y, x, indx, kid);
}
template <>
aoclsparse_status aoclsparse_gthrz(const aoclsparse_int       nnz,
                                   aoclsparse_double_complex *y,
                                   aoclsparse_double_complex *x,
                                   const aoclsparse_int      *indx,
                                   aoclsparse_int             kid)
{
    std::complex<double> *py = reinterpret_cast<std::complex<double> *>(y);
    std::complex<double> *px = reinterpret_cast<std::complex<double> *>(x);
    if(kid == -1)
        return aoclsparse_zgthrz(nnz, py, px, indx);
    else
        return aoclsparse_zgthrz_kid(nnz, y, x, indx, kid);
}

template <>
aoclsparse_status aoclsparse_gthrz(const aoclsparse_int  nnz,
                                   std::complex<float>  *y,
                                   std::complex<float>  *x,
                                   const aoclsparse_int *indx,
                                   aoclsparse_int        kid)
{
    if(kid == -1)
        return aoclsparse_cgthrz(nnz, y, x, indx);
    else
        return aoclsparse_cgthrz_kid(nnz, y, x, indx, kid);
}
template <>
aoclsparse_status aoclsparse_gthrz(const aoclsparse_int      nnz,
                                   aoclsparse_float_complex *y,
                                   aoclsparse_float_complex *x,
                                   const aoclsparse_int     *indx,
                                   aoclsparse_int            kid)
{
    std::complex<float> *py = reinterpret_cast<std::complex<float> *>(y);
    std::complex<float> *px = reinterpret_cast<std::complex<float> *>(x);
    if(kid == -1)
        return aoclsparse_cgthrz(nnz, py, px, indx);
    else
        return aoclsparse_cgthrz_kid(nnz, y, x, indx, kid);
}

template <>
aoclsparse_status aoclsparse_gthrs(
    const aoclsparse_int nnz, const double *y, double *x, aoclsparse_int stride, aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_dgthrs(nnz, y, x, stride);
    else
        return aoclsparse_dgthrs_kid(nnz, y, x, stride, kid);
}

template <>
aoclsparse_status aoclsparse_gthrs(
    const aoclsparse_int nnz, const float *y, float *x, aoclsparse_int stride, aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_sgthrs(nnz, y, x, stride);
    else
        return aoclsparse_sgthrs_kid(nnz, y, x, stride, kid);
}

template <>
aoclsparse_status aoclsparse_gthrs(const aoclsparse_int        nnz,
                                   const std::complex<double> *y,
                                   std::complex<double>       *x,
                                   aoclsparse_int              stride,
                                   aoclsparse_int              kid)
{
    if(kid == -1)
        return aoclsparse_zgthrs(nnz, y, x, stride);
    else
        return aoclsparse_zgthrs_kid(nnz, y, x, stride, kid);
}

template <>
aoclsparse_status aoclsparse_gthrs(const aoclsparse_int       nnz,
                                   const std::complex<float> *y,
                                   std::complex<float>       *x,
                                   aoclsparse_int             stride,
                                   aoclsparse_int             kid)
{
    if(kid == -1)
        return aoclsparse_cgthrs(nnz, y, x, stride);
    else
        return aoclsparse_cgthrs_kid(nnz, y, x, stride, kid);
}

template <>
aoclsparse_status aoclsparse_gthrs(const aoclsparse_int             nnz,
                                   const aoclsparse_double_complex *y,
                                   aoclsparse_double_complex       *x,
                                   aoclsparse_int                   stride,
                                   aoclsparse_int                   kid)
{
    if(kid == -1)
        return aoclsparse_zgthrs(nnz, y, x, stride);
    else
        return aoclsparse_zgthrs_kid(nnz, y, x, stride, kid);
}

template <>
aoclsparse_status aoclsparse_gthrs(const aoclsparse_int            nnz,
                                   const aoclsparse_float_complex *y,
                                   aoclsparse_float_complex       *x,
                                   aoclsparse_int                  stride,
                                   aoclsparse_int                  kid)
{
    if(kid == -1)
        return aoclsparse_cgthrs(nnz, y, x, stride);
    else
        return aoclsparse_cgthrs_kid(nnz, y, x, stride, kid);
}

template <>
aoclsparse_status aoclsparse_csr2ell(aoclsparse_int             m,
                                     const aoclsparse_mat_descr descr,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const float               *csr_val,
                                     aoclsparse_int            *ell_col_ind,
                                     float                     *ell_val,
                                     aoclsparse_int             ell_width)
{
    return aoclsparse_scsr2ell(
        m, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

template <>
aoclsparse_status aoclsparse_csr2ell(aoclsparse_int             m,
                                     const aoclsparse_mat_descr descr,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const double              *csr_val,
                                     aoclsparse_int            *ell_col_ind,
                                     double                    *ell_val,
                                     aoclsparse_int             ell_width)
{
    return aoclsparse_dcsr2ell(
        m, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

template <>
aoclsparse_status aoclsparse_csr2ellt(aoclsparse_int             m,
                                      const aoclsparse_mat_descr descr,
                                      const aoclsparse_int      *csr_row_ptr,
                                      const aoclsparse_int      *csr_col_ind,
                                      const float               *csr_val,
                                      aoclsparse_int            *ell_col_ind,
                                      float                     *ell_val,
                                      aoclsparse_int             ell_width)
{
    return aoclsparse_scsr2ellt(
        m, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

template <>
aoclsparse_status aoclsparse_csr2ellt(aoclsparse_int             m,
                                      const aoclsparse_mat_descr descr,
                                      const aoclsparse_int      *csr_row_ptr,
                                      const aoclsparse_int      *csr_col_ind,
                                      const double              *csr_val,
                                      aoclsparse_int            *ell_col_ind,
                                      double                    *ell_val,
                                      aoclsparse_int             ell_width)
{
    return aoclsparse_dcsr2ellt(
        m, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

template <>
aoclsparse_status aoclsparse_csr2ellthyb(aoclsparse_int        m,
                                         aoclsparse_index_base base,
                                         aoclsparse_int       *ell_m,
                                         const aoclsparse_int *csr_row_ptr,
                                         const aoclsparse_int *csr_col_ind,
                                         const float          *csr_val,
                                         aoclsparse_int       *row_idx_map,
                                         aoclsparse_int       *csr_row_idx_map,
                                         aoclsparse_int       *ell_col_ind,
                                         float                *ell_val,
                                         aoclsparse_int        ell_width)
{
    return aoclsparse_scsr2ellthyb(m,
                                   base,
                                   ell_m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   row_idx_map,
                                   csr_row_idx_map,
                                   ell_col_ind,
                                   ell_val,
                                   ell_width);
}

template <>
aoclsparse_status aoclsparse_csr2ellthyb(aoclsparse_int        m,
                                         aoclsparse_index_base base,
                                         aoclsparse_int       *ell_m,
                                         const aoclsparse_int *csr_row_ptr,
                                         const aoclsparse_int *csr_col_ind,
                                         const double         *csr_val,
                                         aoclsparse_int       *row_idx_map,
                                         aoclsparse_int       *csr_row_idx_map,
                                         aoclsparse_int       *ell_col_ind,
                                         double               *ell_val,
                                         aoclsparse_int        ell_width)
{
    return aoclsparse_dcsr2ellthyb(m,
                                   base,
                                   ell_m,
                                   csr_row_ptr,
                                   csr_col_ind,
                                   csr_val,
                                   row_idx_map,
                                   csr_row_idx_map,
                                   ell_col_ind,
                                   ell_val,
                                   ell_width);
}

template <>
aoclsparse_status aoclsparse_csr2dia(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     const aoclsparse_mat_descr descr,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const float               *csr_val,
                                     aoclsparse_int             dia_num_diag,
                                     aoclsparse_int            *dia_offset,
                                     float                     *dia_val)
{
    return aoclsparse_scsr2dia(
        m, n, descr, csr_row_ptr, csr_col_ind, csr_val, dia_num_diag, dia_offset, dia_val);
}

template <>
aoclsparse_status aoclsparse_csr2dia(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     const aoclsparse_mat_descr descr,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const double              *csr_val,
                                     aoclsparse_int             dia_num_diag,
                                     aoclsparse_int            *dia_offset,
                                     double                    *dia_val)
{
    return aoclsparse_dcsr2dia(
        m, n, descr, csr_row_ptr, csr_col_ind, csr_val, dia_num_diag, dia_offset, dia_val);
}

template <>
aoclsparse_status aoclsparse_csr2bsr(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     const aoclsparse_mat_descr descr,
                                     const float               *csr_val,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     aoclsparse_int             block_dim,
                                     float                     *bsr_val,
                                     aoclsparse_int            *bsr_row_ptr,
                                     aoclsparse_int            *bsr_col_ind)
{
    return aoclsparse_scsr2bsr(m,
                               n,
                               descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
}

template <>
aoclsparse_status aoclsparse_csr2bsr(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     const aoclsparse_mat_descr descr,
                                     const double              *csr_val,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     aoclsparse_int             block_dim,
                                     double                    *bsr_val,
                                     aoclsparse_int            *bsr_row_ptr,
                                     aoclsparse_int            *bsr_col_ind)
{
    return aoclsparse_dcsr2bsr(m,
                               n,
                               descr,
                               csr_val,
                               csr_row_ptr,
                               csr_col_ind,
                               block_dim,
                               bsr_val,
                               bsr_row_ptr,
                               bsr_col_ind);
}

template <>
aoclsparse_status aoclsparse_csr2csc(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     aoclsparse_int             nnz,
                                     const aoclsparse_mat_descr descr,
                                     aoclsparse_index_base      baseCSC,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const float               *csr_val,
                                     aoclsparse_int            *csc_row_ind,
                                     aoclsparse_int            *csc_col_ptr,
                                     float                     *csc_val)
{
    return aoclsparse_scsr2csc(m,
                               n,
                               nnz,
                               descr,
                               baseCSC,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               csc_row_ind,
                               csc_col_ptr,
                               csc_val);
}

template <>
aoclsparse_status aoclsparse_csr2csc(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     aoclsparse_int             nnz,
                                     const aoclsparse_mat_descr descr,
                                     aoclsparse_index_base      baseCSC,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const double              *csr_val,
                                     aoclsparse_int            *csc_row_ind,
                                     aoclsparse_int            *csc_col_ptr,
                                     double                    *csc_val)
{
    return aoclsparse_dcsr2csc(m,
                               n,
                               nnz,
                               descr,
                               baseCSC,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               csc_row_ind,
                               csc_col_ptr,
                               csc_val);
}

template <>
aoclsparse_status aoclsparse_csr2csc(aoclsparse_int                  m,
                                     aoclsparse_int                  n,
                                     aoclsparse_int                  nnz,
                                     const aoclsparse_mat_descr      descr,
                                     aoclsparse_index_base           baseCSC,
                                     const aoclsparse_int           *csr_row_ptr,
                                     const aoclsparse_int           *csr_col_ind,
                                     const aoclsparse_float_complex *csr_val,
                                     aoclsparse_int                 *csc_row_ind,
                                     aoclsparse_int                 *csc_col_ptr,
                                     aoclsparse_float_complex       *csc_val)
{
    return aoclsparse_ccsr2csc(m,
                               n,
                               nnz,
                               descr,
                               baseCSC,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               csc_row_ind,
                               csc_col_ptr,
                               csc_val);
}

template <>
aoclsparse_status aoclsparse_csr2csc(aoclsparse_int                   m,
                                     aoclsparse_int                   n,
                                     aoclsparse_int                   nnz,
                                     const aoclsparse_mat_descr       descr,
                                     aoclsparse_index_base            baseCSC,
                                     const aoclsparse_int            *csr_row_ptr,
                                     const aoclsparse_int            *csr_col_ind,
                                     const aoclsparse_double_complex *csr_val,
                                     aoclsparse_int                  *csc_row_ind,
                                     aoclsparse_int                  *csc_col_ptr,
                                     aoclsparse_double_complex       *csc_val)
{
    return aoclsparse_zcsr2csc(m,
                               n,
                               nnz,
                               descr,
                               baseCSC,
                               csr_row_ptr,
                               csr_col_ind,
                               csr_val,
                               csc_row_ind,
                               csc_col_ptr,
                               csc_val);
}

template <>
aoclsparse_status aoclsparse_csr2dense(aoclsparse_int             m,
                                       aoclsparse_int             n,
                                       const aoclsparse_mat_descr descr,
                                       const float               *csr_val,
                                       const aoclsparse_int      *csr_row_ptr,
                                       const aoclsparse_int      *csr_col_ind,
                                       float                     *A,
                                       aoclsparse_int             ld,
                                       aoclsparse_order           order)
{
    return aoclsparse_scsr2dense(m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld, order);
}

template <>
aoclsparse_status aoclsparse_csr2dense(aoclsparse_int             m,
                                       aoclsparse_int             n,
                                       const aoclsparse_mat_descr descr,
                                       const double              *csr_val,
                                       const aoclsparse_int      *csr_row_ptr,
                                       const aoclsparse_int      *csr_col_ind,
                                       double                    *A,
                                       aoclsparse_int             ld,
                                       aoclsparse_order           order)
{
    return aoclsparse_dcsr2dense(m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld, order);
}
template <>
aoclsparse_status aoclsparse_csr2dense(aoclsparse_int                  m,
                                       aoclsparse_int                  n,
                                       const aoclsparse_mat_descr      descr,
                                       const aoclsparse_float_complex *csr_val,
                                       const aoclsparse_int           *csr_row_ptr,
                                       const aoclsparse_int           *csr_col_ind,
                                       aoclsparse_float_complex       *A,
                                       aoclsparse_int                  ld,
                                       aoclsparse_order                order)
{
    return aoclsparse_ccsr2dense(m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld, order);
}

template <>
aoclsparse_status aoclsparse_csr2dense(aoclsparse_int                   m,
                                       aoclsparse_int                   n,
                                       const aoclsparse_mat_descr       descr,
                                       const aoclsparse_double_complex *csr_val,
                                       const aoclsparse_int            *csr_row_ptr,
                                       const aoclsparse_int            *csr_col_ind,
                                       aoclsparse_double_complex       *A,
                                       aoclsparse_int                   ld,
                                       aoclsparse_order                 order)
{
    return aoclsparse_zcsr2dense(m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld, order);
}

template <>
aoclsparse_status aoclsparse_ilu_smoother(aoclsparse_operation       op,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          float                    **precond_csr_val,
                                          const float               *approx_inv_diag,
                                          float                     *x,
                                          const float               *b)
{
    return aoclsparse_silu_smoother(op, A, descr, precond_csr_val, approx_inv_diag, x, b);
}

template <>
aoclsparse_status aoclsparse_ilu_smoother(aoclsparse_operation       op,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          double                   **precond_csr_val,
                                          const double              *approx_inv_diag,
                                          double                    *x,
                                          const double              *b)
{
    return aoclsparse_dilu_smoother(op, A, descr, precond_csr_val, approx_inv_diag, x, b);
}

template <>
aoclsparse_status aoclsparse_sorv(aoclsparse_sor_type        sor_type,
                                  const aoclsparse_mat_descr descr,
                                  const aoclsparse_matrix    A,
                                  float                      omega,
                                  float                      alpha,
                                  float                     *x,
                                  const float               *b)
{
    return aoclsparse_ssorv(sor_type, descr, A, omega, alpha, x, b);
}
template <>
aoclsparse_status aoclsparse_sorv(aoclsparse_sor_type        sor_type,
                                  const aoclsparse_mat_descr descr,
                                  const aoclsparse_matrix    A,
                                  double                     omega,
                                  double                     alpha,
                                  double                    *x,
                                  const double              *b)
{
    return aoclsparse_dsorv(sor_type, descr, A, omega, alpha, x, b);
}
template <>
aoclsparse_status aoclsparse_sorv(aoclsparse_sor_type             sor_type,
                                  const aoclsparse_mat_descr      descr,
                                  const aoclsparse_matrix         A,
                                  aoclsparse_float_complex        omega,
                                  aoclsparse_float_complex        alpha,
                                  aoclsparse_float_complex       *x,
                                  const aoclsparse_float_complex *b)
{
    return aoclsparse_csorv(sor_type, descr, A, omega, alpha, x, b);
}
template <>
aoclsparse_status aoclsparse_sorv(aoclsparse_sor_type              sor_type,
                                  const aoclsparse_mat_descr       descr,
                                  const aoclsparse_matrix          A,
                                  aoclsparse_double_complex        omega,
                                  aoclsparse_double_complex        alpha,
                                  aoclsparse_double_complex       *x,
                                  const aoclsparse_double_complex *b)
{
    return aoclsparse_zsorv(sor_type, descr, A, omega, alpha, x, b);
}

template <>
double aoclsparse_dot(const aoclsparse_int nnz,
                      const double *__restrict__ x,
                      const aoclsparse_int *__restrict__ indx,
                      const double *__restrict__ y,
                      [[maybe_unused]] double *__restrict__ dot,
                      [[maybe_unused]] bool                 conj,
                      [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
    {
        double dotp;
        dotp = aoclsparse_ddoti(nnz, x, indx, y);
        return dotp;
    }
    else
    {
        double dotp;
        dotp = aoclsparse_ddoti_kid(nnz, x, indx, y, kid);
        return dotp;
    }
}

template <>
float aoclsparse_dot(const aoclsparse_int nnz,
                     const float *__restrict__ x,
                     const aoclsparse_int *__restrict__ indx,
                     const float *__restrict__ y,
                     [[maybe_unused]] float *__restrict__ dot,
                     [[maybe_unused]] bool                 conj,
                     [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
    {
        float dotp;
        dotp = aoclsparse_sdoti(nnz, x, indx, y);
        return dotp;
    }
    else
    {
        float dotp;
        dotp = aoclsparse_sdoti_kid(nnz, x, indx, y, kid);
        return dotp;
    }
}

template <>
aoclsparse_status aoclsparse_dot(const aoclsparse_int nnz,
                                 const std::complex<float> *__restrict__ x,
                                 const aoclsparse_int *__restrict__ indx,
                                 const std::complex<float> *__restrict__ y,
                                 std::complex<float> *__restrict__ dot,
                                 bool                                  conj,
                                 [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
    {
        if(conj)
        {
            return aoclsparse_cdotci(nnz, x, indx, y, dot);
        }
        else
        {
            return aoclsparse_cdotui(nnz, x, indx, y, dot);
        }
    }
    else
    {
        if(conj)
        {
            return aoclsparse_cdotci_kid(nnz, x, indx, y, dot, kid);
        }
        else
        {
            return aoclsparse_cdotui_kid(nnz, x, indx, y, dot, kid);
        }
    }
}
template <>
aoclsparse_status aoclsparse_dot(const aoclsparse_int nnz,
                                 const aoclsparse_float_complex *__restrict__ x,
                                 const aoclsparse_int *__restrict__ indx,
                                 const aoclsparse_float_complex *__restrict__ y,
                                 aoclsparse_float_complex *__restrict__ dot,
                                 bool                                  conj,
                                 [[maybe_unused]] const aoclsparse_int kid)
{
    const std::complex<float> *px   = reinterpret_cast<const std::complex<float> *>(x);
    const std::complex<float> *py   = reinterpret_cast<const std::complex<float> *>(y);
    std::complex<float>       *pdot = reinterpret_cast<std::complex<float> *>(dot);

    if(kid == -1)
    {
        if(conj)
        {
            return aoclsparse_cdotci(nnz, px, indx, py, pdot);
        }
        else
        {
            return aoclsparse_cdotui(nnz, px, indx, py, pdot);
        }
    }
    else
    {
        if(conj)
        {
            return aoclsparse_cdotci_kid(nnz, px, indx, py, pdot, kid);
        }
        else
        {
            return aoclsparse_cdotui_kid(nnz, px, indx, py, pdot, kid);
        }
    }
}

template <>
aoclsparse_status aoclsparse_dot(const aoclsparse_int nnz,
                                 const std::complex<double> *__restrict__ x,
                                 const aoclsparse_int *__restrict__ indx,
                                 const std::complex<double> *__restrict__ y,
                                 std::complex<double> *__restrict__ dot,
                                 bool                                  conj,
                                 [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
    {
        if(conj)
        {
            return aoclsparse_zdotci(nnz, x, indx, y, dot);
        }
        else
        {
            return aoclsparse_zdotui(nnz, x, indx, y, dot);
        }
    }
    else
    {
        if(conj)
        {
            return aoclsparse_zdotci_kid(nnz, x, indx, y, dot, kid);
        }
        else
        {
            return aoclsparse_zdotui_kid(nnz, x, indx, y, dot, kid);
        }
    }
}
template <>
aoclsparse_status aoclsparse_dot(const aoclsparse_int nnz,
                                 const aoclsparse_double_complex *__restrict__ x,
                                 const aoclsparse_int *__restrict__ indx,
                                 const aoclsparse_double_complex *__restrict__ y,
                                 aoclsparse_double_complex *__restrict__ dot,
                                 bool                                  conj,
                                 [[maybe_unused]] const aoclsparse_int kid)
{
    const std::complex<double> *px   = reinterpret_cast<const std::complex<double> *>(x);
    const std::complex<double> *py   = reinterpret_cast<const std::complex<double> *>(y);
    std::complex<double>       *pdot = reinterpret_cast<std::complex<double> *>(dot);
    if(kid == -1)
    {
        if(conj)
        {
            return aoclsparse_zdotci(nnz, px, indx, py, pdot);
        }
        else
        {
            return aoclsparse_zdotui(nnz, px, indx, py, pdot);
        }
    }
    else
    {
        if(conj)
        {
            return aoclsparse_zdotci_kid(nnz, px, indx, py, pdot, kid);
        }
        else
        {
            return aoclsparse_zdotui_kid(nnz, px, indx, py, pdot, kid);
        }
    }
}

template <>
aoclsparse_status aoclsparse_roti(const aoclsparse_int nnz,
                                  double *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  double *__restrict__ y,
                                  const double                          c,
                                  const double                          s,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_droti(nnz, x, indx, y, c, s);
    else
        return aoclsparse_droti_kid(nnz, x, indx, y, c, s, kid);
}

template <>
aoclsparse_status aoclsparse_roti(const aoclsparse_int nnz,
                                  float *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  float *__restrict__ y,
                                  const float                           c,
                                  const float                           s,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_sroti(nnz, x, indx, y, c, s);
    else
        return aoclsparse_sroti_kid(nnz, x, indx, y, c, s, kid);
}

template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const double *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  double *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_dsctr(nnz, x, indx, y);
    else
        return aoclsparse_dsctr_kid(nnz, x, indx, y, kid);
}

template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const float *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  float *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_ssctr(nnz, x, indx, y);
    else
        return aoclsparse_ssctr_kid(nnz, x, indx, y, kid);
}

template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const std::complex<float> *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  std::complex<float> *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_csctr(nnz, x, indx, y);
    else
        return aoclsparse_csctr_kid(nnz, x, indx, y, kid);
}
template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const aoclsparse_float_complex *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  aoclsparse_float_complex *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    const std::complex<float> *px = reinterpret_cast<const std::complex<float> *>(x);
    std::complex<float>       *py = reinterpret_cast<std::complex<float> *>(y);
    if(kid == -1)
        return aoclsparse_csctr(nnz, px, indx, py);
    else
        return aoclsparse_csctr_kid(nnz, x, indx, y, kid);
}
template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const std::complex<double> *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  std::complex<double> *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_zsctr(nnz, x, indx, y);
    else
        return aoclsparse_zsctr_kid(nnz, x, indx, y, kid);
}
template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const aoclsparse_double_complex *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  aoclsparse_double_complex *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    const std::complex<double> *px = reinterpret_cast<const std::complex<double> *>(x);
    std::complex<double>       *py = reinterpret_cast<std::complex<double> *>(y);
    if(kid == -1)
        return aoclsparse_zsctr(nnz, px, indx, py);
    else
        return aoclsparse_zsctr_kid(nnz, x, indx, y, kid);
}

template <>
aoclsparse_status aoclsparse_sctrs(const aoclsparse_int nnz,
                                   const double *__restrict__ x,
                                   aoclsparse_int stride,
                                   double *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_dsctrs(nnz, x, stride, y);
    else
        return aoclsparse_dsctrs_kid(nnz, x, stride, y, kid);
}

template <>
aoclsparse_status aoclsparse_sctrs(const aoclsparse_int nnz,
                                   const float *__restrict__ x,
                                   aoclsparse_int stride,
                                   float *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_ssctrs(nnz, x, stride, y);
    else
        return aoclsparse_ssctrs_kid(nnz, x, stride, y, kid);
}

template <>
aoclsparse_status aoclsparse_sctrs(const aoclsparse_int nnz,
                                   const std::complex<float> *__restrict__ x,
                                   aoclsparse_int stride,
                                   std::complex<float> *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_csctrs(nnz, x, stride, y);
    else
        return aoclsparse_csctrs_kid(nnz, x, stride, y, kid);
}

template <>
aoclsparse_status aoclsparse_sctrs(const aoclsparse_int nnz,
                                   const std::complex<double> *__restrict__ x,
                                   aoclsparse_int stride,
                                   std::complex<double> *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_zsctrs(nnz, x, stride, y);
    else
        return aoclsparse_zsctrs_kid(nnz, x, stride, y, kid);
}

template <>
aoclsparse_status aoclsparse_sctrs(const aoclsparse_int nnz,
                                   const aoclsparse_float_complex *__restrict__ x,
                                   aoclsparse_int stride,
                                   aoclsparse_float_complex *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_csctrs(nnz, x, stride, y);
    else
        return aoclsparse_csctrs_kid(nnz, x, stride, y, kid);
}

template <>
aoclsparse_status aoclsparse_sctrs(const aoclsparse_int nnz,
                                   const aoclsparse_double_complex *__restrict__ x,
                                   aoclsparse_int stride,
                                   aoclsparse_double_complex *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_zsctrs(nnz, x, stride, y);
    else
        return aoclsparse_zsctrs_kid(nnz, x, stride, y, kid);
}

template <>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *row_ptr,
                                        aoclsparse_int       *col_idx,
                                        float                *val)
{
    return aoclsparse_create_scsr(mat, base, M, N, nnz, row_ptr, col_idx, val);
}
template <>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *row_ptr,
                                        aoclsparse_int       *col_idx,
                                        double               *val)
{
    return aoclsparse_create_dcsr(mat, base, M, N, nnz, row_ptr, col_idx, val);
}
template <>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix        *mat,
                                        aoclsparse_index_base     base,
                                        aoclsparse_int            M,
                                        aoclsparse_int            N,
                                        aoclsparse_int            nnz,
                                        aoclsparse_int           *row_ptr,
                                        aoclsparse_int           *col_idx,
                                        aoclsparse_float_complex *val)
{
    return aoclsparse_create_ccsr(mat, base, M, N, nnz, row_ptr, col_idx, val);
}
template <>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix         *mat,
                                        aoclsparse_index_base      base,
                                        aoclsparse_int             M,
                                        aoclsparse_int             N,
                                        aoclsparse_int             nnz,
                                        aoclsparse_int            *row_ptr,
                                        aoclsparse_int            *col_idx,
                                        aoclsparse_double_complex *val)
{
    return aoclsparse_create_zcsr(mat, base, M, N, nnz, row_ptr, col_idx, val);
}
template <>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *row_ptr,
                                        aoclsparse_int       *col_idx,
                                        std::complex<float>  *val)
{
    return aoclsparse_create_ccsr(
        mat, base, M, N, nnz, row_ptr, col_idx, reinterpret_cast<aoclsparse_float_complex *>(val));
}
template <>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *row_ptr,
                                        aoclsparse_int       *col_idx,
                                        std::complex<double> *val)
{
    return aoclsparse_create_zcsr(
        mat, base, M, N, nnz, row_ptr, col_idx, reinterpret_cast<aoclsparse_double_complex *>(val));
}

template <>
aoclsparse_status aoclsparse_create_tcsr(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ptr_L,
                                         aoclsparse_int             *row_ptr_U,
                                         aoclsparse_int             *col_idx_L,
                                         aoclsparse_int             *col_idx_U,
                                         float                      *val_L,
                                         float                      *val_U)
{
    return aoclsparse_create_stcsr(
        mat, base, M, N, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);
}
template <>
aoclsparse_status aoclsparse_create_tcsr(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ptr_L,
                                         aoclsparse_int             *row_ptr_U,
                                         aoclsparse_int             *col_idx_L,
                                         aoclsparse_int             *col_idx_U,
                                         double                     *val_L,
                                         double                     *val_U)
{
    return aoclsparse_create_dtcsr(
        mat, base, M, N, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);
}
template <>
aoclsparse_status aoclsparse_create_tcsr(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ptr_L,
                                         aoclsparse_int             *row_ptr_U,
                                         aoclsparse_int             *col_idx_L,
                                         aoclsparse_int             *col_idx_U,
                                         aoclsparse_float_complex   *val_L,
                                         aoclsparse_float_complex   *val_U)
{
    return aoclsparse_create_ctcsr(
        mat, base, M, N, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);
}
template <>
aoclsparse_status aoclsparse_create_tcsr(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ptr_L,
                                         aoclsparse_int             *row_ptr_U,
                                         aoclsparse_int             *col_idx_L,
                                         aoclsparse_int             *col_idx_U,
                                         aoclsparse_double_complex  *val_L,
                                         aoclsparse_double_complex  *val_U)
{
    return aoclsparse_create_ztcsr(
        mat, base, M, N, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);
}

template <>
aoclsparse_status aoclsparse_create_csc(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *col_ptr,
                                        aoclsparse_int       *row_idx,
                                        float                *val)
{
    return aoclsparse_create_scsc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}
template <>
aoclsparse_status aoclsparse_create_csc(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *col_ptr,
                                        aoclsparse_int       *row_idx,
                                        double               *val)
{
    return aoclsparse_create_dcsc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}
template <>
aoclsparse_status aoclsparse_create_csc(aoclsparse_matrix        *mat,
                                        aoclsparse_index_base     base,
                                        aoclsparse_int            M,
                                        aoclsparse_int            N,
                                        aoclsparse_int            nnz,
                                        aoclsparse_int           *col_ptr,
                                        aoclsparse_int           *row_idx,
                                        aoclsparse_float_complex *val)
{
    return aoclsparse_create_ccsc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}
template <>
aoclsparse_status aoclsparse_create_csc(aoclsparse_matrix         *mat,
                                        aoclsparse_index_base      base,
                                        aoclsparse_int             M,
                                        aoclsparse_int             N,
                                        aoclsparse_int             nnz,
                                        aoclsparse_int            *col_ptr,
                                        aoclsparse_int            *row_idx,
                                        aoclsparse_double_complex *val)
{
    return aoclsparse_create_zcsc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

template <>
aoclsparse_status aoclsparse_create_coo(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *row_ind,
                                        aoclsparse_int       *col_ind,
                                        float                *val)
{
    return aoclsparse_create_scoo(mat, base, M, N, nnz, row_ind, col_ind, val);
};

template <>
aoclsparse_status aoclsparse_create_coo(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *row_ind,
                                        aoclsparse_int       *col_ind,
                                        double               *val)
{
    return aoclsparse_create_dcoo(mat, base, M, N, nnz, row_ind, col_ind, val);
};
template <>
aoclsparse_status aoclsparse_create_coo(aoclsparse_matrix        *mat,
                                        aoclsparse_index_base     base,
                                        aoclsparse_int            M,
                                        aoclsparse_int            N,
                                        aoclsparse_int            nnz,
                                        aoclsparse_int           *row_ind,
                                        aoclsparse_int           *col_ind,
                                        aoclsparse_float_complex *val)
{
    return aoclsparse_create_ccoo(mat, base, M, N, nnz, row_ind, col_ind, val);
};
template <>
aoclsparse_status aoclsparse_create_coo(aoclsparse_matrix         *mat,
                                        aoclsparse_index_base      base,
                                        aoclsparse_int             M,
                                        aoclsparse_int             N,
                                        aoclsparse_int             nnz,
                                        aoclsparse_int            *row_ind,
                                        aoclsparse_int            *col_ind,
                                        aoclsparse_double_complex *val)
{
    return aoclsparse_create_zcoo(mat, base, M, N, nnz, row_ind, col_ind, val);
};

template <>
aoclsparse_status
    aoclsparse_set_value(aoclsparse_matrix mat, aoclsparse_int row, aoclsparse_int col, float val)
{
    return aoclsparse_sset_value(mat, row, col, val);
};

template <>
aoclsparse_status
    aoclsparse_set_value(aoclsparse_matrix mat, aoclsparse_int row, aoclsparse_int col, double val)
{
    return aoclsparse_dset_value(mat, row, col, val);
};

template <>
aoclsparse_status aoclsparse_set_value(aoclsparse_matrix        mat,
                                       aoclsparse_int           row,
                                       aoclsparse_int           col,
                                       aoclsparse_float_complex val)
{
    return aoclsparse_cset_value(mat, row, col, val);
};

template <>
aoclsparse_status aoclsparse_set_value(aoclsparse_matrix         mat,
                                       aoclsparse_int            row,
                                       aoclsparse_int            col,
                                       aoclsparse_double_complex val)
{
    return aoclsparse_zset_value(mat, row, col, val);
};

template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int nnz,
                                   float          a,
                                   const float *__restrict__ x,
                                   const aoclsparse_int *__restrict__ indx,
                                   float *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_saxpyi(nnz, a, x, indx, y);
    else
        return aoclsparse_saxpyi_kid(nnz, a, x, indx, y, kid);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int nnz,
                                   double         a,
                                   const double *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   double *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_daxpyi(nnz, a, x, indx, y);
    else
        return aoclsparse_daxpyi_kid(nnz, a, x, indx, y, kid);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int      nnz,
                                   std::complex<float> a,
                                   const std::complex<float> *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   std::complex<float> *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_caxpyi(nnz, &a, x, indx, y);
    else
        return aoclsparse_caxpyi_kid(nnz, &a, x, indx, y, kid);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int       nnz,
                                   std::complex<double> a,
                                   const std::complex<double> *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   std::complex<double> *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_zaxpyi(nnz, &a, x, indx, y);
    else
        return aoclsparse_zaxpyi_kid(nnz, &a, x, indx, y, kid);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int           nnz,
                                   aoclsparse_float_complex a,
                                   const aoclsparse_float_complex *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   aoclsparse_float_complex *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_caxpyi(nnz, &a, x, indx, y);
    else
        return aoclsparse_caxpyi_kid(nnz, &a, x, indx, y, kid);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int            nnz,
                                   aoclsparse_double_complex a,
                                   const aoclsparse_double_complex *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   aoclsparse_double_complex *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    if(kid == -1)
        return aoclsparse_zaxpyi(nnz, &a, x, indx, y);
    else
        return aoclsparse_zaxpyi_kid(nnz, &a, x, indx, y, kid);
}

template <>
aoclsparse_status aoclsparse_update_values(aoclsparse_matrix mat, aoclsparse_int len, float *val)
{
    return aoclsparse_supdate_values(mat, len, val);
}

template <>
aoclsparse_status aoclsparse_update_values(aoclsparse_matrix mat, aoclsparse_int len, double *val)
{
    return aoclsparse_dupdate_values(mat, len, val);
}

template <>
aoclsparse_status aoclsparse_update_values(aoclsparse_matrix         mat,
                                           aoclsparse_int            len,
                                           aoclsparse_float_complex *val)
{
    return aoclsparse_cupdate_values(mat, len, val);
}

template <>
aoclsparse_status aoclsparse_update_values(aoclsparse_matrix          mat,
                                           aoclsparse_int             len,
                                           aoclsparse_double_complex *val)
{
    return aoclsparse_zupdate_values(mat, len, val);
}

template <>
aoclsparse_status aoclsparse_export_csr(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **row_ptr,
                                        aoclsparse_int        **col_idx,
                                        float                 **val)
{
    return aoclsparse_export_scsr(mat, base, m, n, nnz, row_ptr, col_idx, val);
}
template <>
aoclsparse_status aoclsparse_export_csr(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **row_ptr,
                                        aoclsparse_int        **col_idx,
                                        double                **val)
{
    return aoclsparse_export_dcsr(mat, base, m, n, nnz, row_ptr, col_idx, val);
}
template <>
aoclsparse_status aoclsparse_export_csr(const aoclsparse_matrix    mat,
                                        aoclsparse_index_base     *base,
                                        aoclsparse_int            *m,
                                        aoclsparse_int            *n,
                                        aoclsparse_int            *nnz,
                                        aoclsparse_int           **row_ptr,
                                        aoclsparse_int           **col_idx,
                                        aoclsparse_float_complex **val)
{
    return aoclsparse_export_ccsr(mat, base, m, n, nnz, row_ptr, col_idx, val);
}
template <>
aoclsparse_status aoclsparse_export_csr(const aoclsparse_matrix     mat,
                                        aoclsparse_index_base      *base,
                                        aoclsparse_int             *m,
                                        aoclsparse_int             *n,
                                        aoclsparse_int             *nnz,
                                        aoclsparse_int            **row_ptr,
                                        aoclsparse_int            **col_idx,
                                        aoclsparse_double_complex **val)
{
    return aoclsparse_export_zcsr(mat, base, m, n, nnz, row_ptr, col_idx, val);
}

template <>
aoclsparse_status aoclsparse_export_csr(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **row_ptr,
                                        aoclsparse_int        **col_idx,
                                        std::complex<float>   **val)
{
    return aoclsparse_export_ccsr(
        mat, base, m, n, nnz, row_ptr, col_idx, reinterpret_cast<aoclsparse_float_complex **>(val));
}

template <>
aoclsparse_status aoclsparse_export_csr(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **row_ptr,
                                        aoclsparse_int        **col_idx,
                                        std::complex<double>  **val)
{
    return aoclsparse_export_zcsr(mat,
                                  base,
                                  m,
                                  n,
                                  nnz,
                                  row_ptr,
                                  col_idx,
                                  reinterpret_cast<aoclsparse_double_complex **>(val));
}

template <>
aoclsparse_status aoclsparse_export_csc(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **col_ptr,
                                        aoclsparse_int        **row_idx,
                                        float                 **val)
{
    return aoclsparse_export_scsc(mat, base, m, n, nnz, col_ptr, row_idx, val);
}
template <>
aoclsparse_status aoclsparse_export_csc(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **col_ptr,
                                        aoclsparse_int        **row_idx,
                                        double                **val)
{
    return aoclsparse_export_dcsc(mat, base, m, n, nnz, col_ptr, row_idx, val);
}

template <>
aoclsparse_status aoclsparse_export_csc(const aoclsparse_matrix    mat,
                                        aoclsparse_index_base     *base,
                                        aoclsparse_int            *m,
                                        aoclsparse_int            *n,
                                        aoclsparse_int            *nnz,
                                        aoclsparse_int           **col_ptr,
                                        aoclsparse_int           **row_idx,
                                        aoclsparse_float_complex **val)
{
    return aoclsparse_export_ccsc(mat, base, m, n, nnz, col_ptr, row_idx, val);
}
template <>
aoclsparse_status aoclsparse_export_csc(const aoclsparse_matrix     mat,
                                        aoclsparse_index_base      *base,
                                        aoclsparse_int             *m,
                                        aoclsparse_int             *n,
                                        aoclsparse_int             *nnz,
                                        aoclsparse_int            **col_ptr,
                                        aoclsparse_int            **row_idx,
                                        aoclsparse_double_complex **val)
{
    return aoclsparse_export_zcsc(mat, base, m, n, nnz, col_ptr, row_idx, val);
}

template <>
aoclsparse_status aoclsparse_export_coo(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **row_ptr,
                                        aoclsparse_int        **col_ptr,
                                        float                 **val)
{
    return aoclsparse_export_scoo(mat, base, m, n, nnz, row_ptr, col_ptr, val);
}
template <>
aoclsparse_status aoclsparse_export_coo(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **row_ptr,
                                        aoclsparse_int        **col_ptr,
                                        double                **val)
{
    return aoclsparse_export_dcoo(mat, base, m, n, nnz, row_ptr, col_ptr, val);
}
template <>
aoclsparse_status aoclsparse_export_coo(const aoclsparse_matrix    mat,
                                        aoclsparse_index_base     *base,
                                        aoclsparse_int            *m,
                                        aoclsparse_int            *n,
                                        aoclsparse_int            *nnz,
                                        aoclsparse_int           **row_ptr,
                                        aoclsparse_int           **col_ptr,
                                        aoclsparse_float_complex **val)
{
    return aoclsparse_export_ccoo(mat, base, m, n, nnz, row_ptr, col_ptr, val);
}
template <>
aoclsparse_status aoclsparse_export_coo(const aoclsparse_matrix     mat,
                                        aoclsparse_index_base      *base,
                                        aoclsparse_int             *m,
                                        aoclsparse_int             *n,
                                        aoclsparse_int             *nnz,
                                        aoclsparse_int            **row_ptr,
                                        aoclsparse_int            **col_ptr,
                                        aoclsparse_double_complex **val)
{
    return aoclsparse_export_zcoo(mat, base, m, n, nnz, row_ptr, col_ptr, val);
}

template <>
aoclsparse_status aoclsparse_dotmv(const aoclsparse_operation op,
                                   float                      alpha,
                                   aoclsparse_matrix          A,
                                   aoclsparse_mat_descr       descr,
                                   float                     *x,
                                   float                      beta,
                                   float                     *y,
                                   float                     *d)
{
    return aoclsparse_sdotmv(op, alpha, A, descr, x, beta, y, d);
}
template <>
aoclsparse_status aoclsparse_dotmv(const aoclsparse_operation op,
                                   double                     alpha,
                                   aoclsparse_matrix          A,
                                   aoclsparse_mat_descr       descr,
                                   double                    *x,
                                   double                     beta,
                                   double                    *y,
                                   double                    *d)
{
    return aoclsparse_ddotmv(op, alpha, A, descr, x, beta, y, d);
}
template <>
aoclsparse_status aoclsparse_dotmv(const aoclsparse_operation op,
                                   aoclsparse_float_complex   alpha,
                                   aoclsparse_matrix          A,
                                   aoclsparse_mat_descr       descr,
                                   aoclsparse_float_complex  *x,
                                   aoclsparse_float_complex   beta,
                                   aoclsparse_float_complex  *y,
                                   aoclsparse_float_complex  *d)
{
    return aoclsparse_cdotmv(op, alpha, A, descr, x, beta, y, d);
}
template <>
aoclsparse_status aoclsparse_dotmv(const aoclsparse_operation op,
                                   aoclsparse_double_complex  alpha,
                                   aoclsparse_matrix          A,
                                   aoclsparse_mat_descr       descr,
                                   aoclsparse_double_complex *x,
                                   aoclsparse_double_complex  beta,
                                   aoclsparse_double_complex *y,
                                   aoclsparse_double_complex *d)
{
    return aoclsparse_zdotmv(op, alpha, A, descr, x, beta, y, d);
}
template <>
aoclsparse_status aoclsparse_symgs(aoclsparse_operation       trans,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   const float                alpha,
                                   const float               *b,
                                   float                     *x)
{
    return aoclsparse_ssymgs(trans, A, descr, alpha, b, x);
}
template <>
aoclsparse_status aoclsparse_symgs(aoclsparse_operation       trans,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   const double               alpha,
                                   const double              *b,
                                   double                    *x)
{
    return aoclsparse_dsymgs(trans, A, descr, alpha, b, x);
}
template <>
aoclsparse_status aoclsparse_symgs(aoclsparse_operation       trans,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   const std::complex<float>  alpha,
                                   const std::complex<float> *b,
                                   std::complex<float>       *x)
{
    const aoclsparse_float_complex *palpha
        = reinterpret_cast<const aoclsparse_float_complex *>(&alpha);
    const aoclsparse_float_complex *pb = reinterpret_cast<const aoclsparse_float_complex *>(b);
    aoclsparse_float_complex       *px = reinterpret_cast<aoclsparse_float_complex *>(x);
    return aoclsparse_csymgs(trans, A, descr, *palpha, pb, px);
}
template <>
aoclsparse_status aoclsparse_symgs(aoclsparse_operation            trans,
                                   aoclsparse_matrix               A,
                                   const aoclsparse_mat_descr      descr,
                                   const aoclsparse_float_complex  alpha,
                                   const aoclsparse_float_complex *b,
                                   aoclsparse_float_complex       *x)
{
    return aoclsparse_csymgs(trans, A, descr, alpha, b, x);
}
template <>
aoclsparse_status aoclsparse_symgs(aoclsparse_operation        trans,
                                   aoclsparse_matrix           A,
                                   const aoclsparse_mat_descr  descr,
                                   const std::complex<double>  alpha,
                                   const std::complex<double> *b,
                                   std::complex<double>       *x)
{
    const aoclsparse_double_complex *palpha
        = reinterpret_cast<const aoclsparse_double_complex *>(&alpha);
    const aoclsparse_double_complex *pb = reinterpret_cast<const aoclsparse_double_complex *>(b);
    aoclsparse_double_complex       *px = reinterpret_cast<aoclsparse_double_complex *>(x);
    return aoclsparse_zsymgs(trans, A, descr, *palpha, pb, px);
}
template <>
aoclsparse_status aoclsparse_symgs(aoclsparse_operation             trans,
                                   aoclsparse_matrix                A,
                                   const aoclsparse_mat_descr       descr,
                                   const aoclsparse_double_complex  alpha,
                                   const aoclsparse_double_complex *b,
                                   aoclsparse_double_complex       *x)
{
    return aoclsparse_zsymgs(trans, A, descr, alpha, b, x);
}

//symgs + mv
template <>
aoclsparse_status aoclsparse_symgs_mv(aoclsparse_operation       trans,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      const float                alpha,
                                      const float               *b,
                                      float                     *x,
                                      float                     *y)
{
    return aoclsparse_ssymgs_mv(trans, A, descr, alpha, b, x, y);
}
template <>
aoclsparse_status aoclsparse_symgs_mv(aoclsparse_operation       trans,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      const double               alpha,
                                      const double              *b,
                                      double                    *x,
                                      double                    *y)
{
    return aoclsparse_dsymgs_mv(trans, A, descr, alpha, b, x, y);
}
template <>
aoclsparse_status aoclsparse_symgs_mv(aoclsparse_operation       trans,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      const std::complex<float>  alpha,
                                      const std::complex<float> *b,
                                      std::complex<float>       *x,
                                      std::complex<float>       *y)
{
    const aoclsparse_float_complex *palpha
        = reinterpret_cast<const aoclsparse_float_complex *>(&alpha);
    const aoclsparse_float_complex *pb = reinterpret_cast<const aoclsparse_float_complex *>(b);
    aoclsparse_float_complex       *px = reinterpret_cast<aoclsparse_float_complex *>(x);
    aoclsparse_float_complex       *py = reinterpret_cast<aoclsparse_float_complex *>(y);
    return aoclsparse_csymgs_mv(trans, A, descr, *palpha, pb, px, py);
}
template <>
aoclsparse_status aoclsparse_symgs_mv(aoclsparse_operation            trans,
                                      aoclsparse_matrix               A,
                                      const aoclsparse_mat_descr      descr,
                                      const aoclsparse_float_complex  alpha,
                                      const aoclsparse_float_complex *b,
                                      aoclsparse_float_complex       *x,
                                      aoclsparse_float_complex       *y)
{
    return aoclsparse_csymgs_mv(trans, A, descr, alpha, b, x, y);
}
template <>
aoclsparse_status aoclsparse_symgs_mv(aoclsparse_operation        trans,
                                      aoclsparse_matrix           A,
                                      const aoclsparse_mat_descr  descr,
                                      const std::complex<double>  alpha,
                                      const std::complex<double> *b,
                                      std::complex<double>       *x,
                                      std::complex<double>       *y)
{
    const aoclsparse_double_complex *palpha
        = reinterpret_cast<const aoclsparse_double_complex *>(&alpha);
    const aoclsparse_double_complex *pb = reinterpret_cast<const aoclsparse_double_complex *>(b);
    aoclsparse_double_complex       *px = reinterpret_cast<aoclsparse_double_complex *>(x);
    aoclsparse_double_complex       *py = reinterpret_cast<aoclsparse_double_complex *>(y);
    return aoclsparse_zsymgs_mv(trans, A, descr, *palpha, pb, px, py);
}
template <>
aoclsparse_status aoclsparse_symgs_mv(aoclsparse_operation             trans,
                                      aoclsparse_matrix                A,
                                      const aoclsparse_mat_descr       descr,
                                      const aoclsparse_double_complex  alpha,
                                      const aoclsparse_double_complex *b,
                                      aoclsparse_double_complex       *x,
                                      aoclsparse_double_complex       *y)
{
    return aoclsparse_zsymgs_mv(trans, A, descr, alpha, b, x, y);
}
