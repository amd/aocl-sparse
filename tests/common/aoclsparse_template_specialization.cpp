/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
aoclsparse_status aoclsparse_csrmm(aoclsparse_operation       trans,
                                   const float               *alpha,
                                   const aoclsparse_matrix    csr,
                                   const aoclsparse_mat_descr descr,
                                   aoclsparse_order           order,
                                   const float               *B,
                                   aoclsparse_int             n,
                                   aoclsparse_int             ldb,
                                   const float               *beta,
                                   float                     *C,
                                   aoclsparse_int             ldc)
{
    return aoclsparse_scsrmm(trans, alpha, csr, descr, order, B, n, ldb, beta, C, ldc);
}

template <>
aoclsparse_status aoclsparse_csrmm(aoclsparse_operation       trans,
                                   const double              *alpha,
                                   const aoclsparse_matrix    csr,
                                   const aoclsparse_mat_descr descr,
                                   aoclsparse_order           order,
                                   const double              *B,
                                   aoclsparse_int             n,
                                   aoclsparse_int             ldb,
                                   const double              *beta,
                                   double                    *C,
                                   aoclsparse_int             ldc)
{
    return aoclsparse_dcsrmm(trans, alpha, csr, descr, order, B, n, ldb, beta, C, ldc);
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
                                  const aoclsparse_int *indx)
{
    return aoclsparse_dgthr(nnz, y, x, indx);
}

template <>
aoclsparse_status
    aoclsparse_gthr(const aoclsparse_int nnz, const float *y, float *x, const aoclsparse_int *indx)
{
    return aoclsparse_sgthr(nnz, y, x, indx);
}

template <>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int        nnz,
                                  const std::complex<double> *y,
                                  std::complex<double>       *x,
                                  const aoclsparse_int       *indx)
{
    return aoclsparse_zgthr(nnz, y, x, indx);
}

template <>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int       nnz,
                                  const std::complex<float> *y,
                                  std::complex<float>       *x,
                                  const aoclsparse_int      *indx)
{
    return aoclsparse_cgthr(nnz, y, x, indx);
}

template <>
aoclsparse_status
    aoclsparse_gthrz(const aoclsparse_int nnz, double *y, double *x, const aoclsparse_int *indx)
{
    return aoclsparse_dgthrz(nnz, y, x, indx);
}

template <>
aoclsparse_status
    aoclsparse_gthrz(const aoclsparse_int nnz, float *y, float *x, const aoclsparse_int *indx)
{
    return aoclsparse_sgthrz(nnz, y, x, indx);
}

template <>
aoclsparse_status aoclsparse_gthrz(const aoclsparse_int  nnz,
                                   std::complex<double> *y,
                                   std::complex<double> *x,
                                   const aoclsparse_int *indx)
{
    return aoclsparse_zgthrz(nnz, y, x, indx);
}

template <>
aoclsparse_status aoclsparse_gthrz(const aoclsparse_int  nnz,
                                   std::complex<float>  *y,
                                   std::complex<float>  *x,
                                   const aoclsparse_int *indx)
{
    return aoclsparse_cgthrz(nnz, y, x, indx);
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
double aoclsparse_dot(const aoclsparse_int nnz,
                      const double *__restrict__ x,
                      const aoclsparse_int *__restrict__ indx,
                      const double *__restrict__ y,
                      [[maybe_unused]] double *__restrict__ dot,
                      [[maybe_unused]] bool                 conj,
                      [[maybe_unused]] const aoclsparse_int kid)
{
    double dotp;
    dotp = aoclsparse_ddoti(nnz, x, indx, y);
    return dotp;
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
    float dotp;
    dotp = aoclsparse_sdoti(nnz, x, indx, y);
    return dotp;
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
    if(conj)
    {
        return aoclsparse_cdotci(nnz, x, indx, y, dot);
    }
    else
    {
        return aoclsparse_cdotui(nnz, x, indx, y, dot);
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
    if(conj)
    {
        return aoclsparse_zdotci(nnz, x, indx, y, dot);
    }
    else
    {
        return aoclsparse_zdotui(nnz, x, indx, y, dot);
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
    return aoclsparse_droti(nnz, x, indx, y, c, s);
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
    return aoclsparse_sroti(nnz, x, indx, y, c, s);
}

template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const double *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  double *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_dsctr(nnz, x, indx, y);
}

template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const float *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  float *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_ssctr(nnz, x, indx, y);
}

template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const std::complex<float> *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  std::complex<float> *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_csctr(nnz, x, indx, y);
}

template <>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const std::complex<double> *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  std::complex<double> *__restrict__ y,
                                  [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_zsctr(nnz, x, indx, y);
}

template <>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix    &mat,
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
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix    &mat,
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
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix        &mat,
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
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix         &mat,
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
aoclsparse_status aoclsparse_createcsc(aoclsparse_matrix    &mat,
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
aoclsparse_status aoclsparse_createcsc(aoclsparse_matrix    &mat,
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
aoclsparse_status aoclsparse_createcsc(aoclsparse_matrix        &mat,
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
aoclsparse_status aoclsparse_createcsc(aoclsparse_matrix         &mat,
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
aoclsparse_status aoclsparse_createcoo(aoclsparse_matrix    &mat,
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
aoclsparse_status aoclsparse_createcoo(aoclsparse_matrix    &mat,
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
aoclsparse_status aoclsparse_createcoo(aoclsparse_matrix        &mat,
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
aoclsparse_status aoclsparse_createcoo(aoclsparse_matrix         &mat,
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
aoclsparse_status aoclsparse_axpyi(aoclsparse_int nnz,
                                   float          a,
                                   const float *__restrict__ x,
                                   const aoclsparse_int *__restrict__ indx,
                                   float *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_saxpyi(nnz, a, x, indx, y);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int nnz,
                                   double         a,
                                   const double *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   double *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_daxpyi(nnz, a, x, indx, y);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int      nnz,
                                   std::complex<float> a,
                                   const std::complex<float> *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   std::complex<float> *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_caxpyi(nnz, &a, x, indx, y);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int       nnz,
                                   std::complex<double> a,
                                   const std::complex<double> *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   std::complex<double> *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_zaxpyi(nnz, &a, x, indx, y);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int           nnz,
                                   aoclsparse_float_complex a,
                                   const aoclsparse_float_complex *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   aoclsparse_float_complex *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_caxpyi(nnz, &a, x, indx, y);
}
template <>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int            nnz,
                                   aoclsparse_double_complex a,
                                   const aoclsparse_double_complex *__restrict__ x,
                                   const aoclsparse_int *indx,
                                   aoclsparse_double_complex *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid)
{
    return aoclsparse_zaxpyi(nnz, &a, x, indx, y);
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
