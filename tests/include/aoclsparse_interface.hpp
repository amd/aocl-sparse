/* ************************************************************************
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc.
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

/*! \file
 *  \brief aoclsparse_interface.hpp exposes C++ templated Sparse Linear Algebra interface
 *  with only the precision templated.
 */

#pragma once
#ifndef AOCLSPARSE_INTERFACE_HPP
#define AOCLSPARSE_INTERFACE_HPP

#include "aoclsparse.h"
#include "aoclsparse_utils.hpp"

#include <complex>

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_csr2m(aoclsparse_operation       transA,
                                   const aoclsparse_mat_descr descrA,
                                   const aoclsparse_matrix    csrA,
                                   aoclsparse_operation       transB,
                                   const aoclsparse_mat_descr descrB,
                                   const aoclsparse_matrix    csrB,
                                   aoclsparse_request         request,
                                   aoclsparse_matrix         *csrC);

template <typename T>
aoclsparse_status aoclsparse_csrmm(aoclsparse_operation       op,
                                   T                          alpha,
                                   const aoclsparse_matrix    A,
                                   const aoclsparse_mat_descr descr,
                                   aoclsparse_order           order,
                                   const T                   *B,
                                   aoclsparse_int             n,
                                   aoclsparse_int             ldb,
                                   T                          beta,
                                   T                         *C,
                                   aoclsparse_int             ldc,
                                   const aoclsparse_int       kid);

template <typename T>
aoclsparse_status aoclsparse_add(const aoclsparse_operation op,
                                 const aoclsparse_matrix    A,
                                 const T                    alpha,
                                 const aoclsparse_matrix    B,
                                 aoclsparse_matrix         *C);

template <typename T>
aoclsparse_status aoclsparse_spmmd(const aoclsparse_operation      op,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_matrix         B,
                                   const aoclsparse_order          layout,
                                   T                              *C,
                                   aoclsparse_int                  ldc,
                                   [[maybe_unused]] aoclsparse_int kid);

template <typename T>
aoclsparse_status aoclsparse_sp2md(const aoclsparse_operation      opA,
                                   const aoclsparse_mat_descr      descrA,
                                   const aoclsparse_matrix         A,
                                   const aoclsparse_operation      opB,
                                   const aoclsparse_mat_descr      descrB,
                                   const aoclsparse_matrix         B,
                                   T                               alpha,
                                   T                               beta,
                                   T                              *C,
                                   aoclsparse_order                layout,
                                   aoclsparse_int                  ldc,
                                   [[maybe_unused]] aoclsparse_int kid);

template <typename T>
aoclsparse_status aoclsparse_syrkd(aoclsparse_operation    op,
                                   const aoclsparse_matrix A,
                                   T                       alpha,
                                   T                       beta,
                                   T                      *C,
                                   aoclsparse_order        orderC,
                                   aoclsparse_int          ldc);

template <typename T>
aoclsparse_status aoclsparse_syprd(aoclsparse_operation    op,
                                   const aoclsparse_matrix A,
                                   const T                *B,
                                   aoclsparse_order        orderB,
                                   aoclsparse_int          ldb,
                                   T                       alpha,
                                   T                       beta,
                                   T                      *C,
                                   aoclsparse_order        orderC,
                                   aoclsparse_int          ldc);

template <typename T>
aoclsparse_status aoclsparse_trsm_kid(const aoclsparse_operation trans,
                                      const T                    alpha,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      aoclsparse_order           order,
                                      const T                   *B,
                                      aoclsparse_int             n,
                                      aoclsparse_int             ldb,
                                      T                         *X,
                                      aoclsparse_int             ldx,
                                      const aoclsparse_int       kid);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                   const T                   *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const T                   *csr_val,
                                   const aoclsparse_int      *csr_col_ind,
                                   const aoclsparse_int      *csr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const T                   *x,
                                   const T                   *beta,
                                   T                         *y);

template <typename T>
aoclsparse_status aoclsparse_ellmv(aoclsparse_operation       trans,
                                   const T                   *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const T                   *ell_val,
                                   const aoclsparse_int      *ell_col_ind,
                                   const aoclsparse_int       ell_width,
                                   const aoclsparse_mat_descr descr,
                                   const T                   *x,
                                   const T                   *beta,
                                   T                         *y);

template <typename T>
aoclsparse_status aoclsparse_blkcsrmv(aoclsparse_operation       trans,
                                      const T                   *alpha,
                                      aoclsparse_int             m,
                                      aoclsparse_int             n,
                                      aoclsparse_int             nnz,
                                      const uint8_t             *masks,
                                      const T                   *blk_csr_val,
                                      const aoclsparse_int      *blk_col_ind,
                                      const aoclsparse_int      *blk_row_ptr,
                                      const aoclsparse_mat_descr descr,
                                      const T                   *x,
                                      const T                   *beta,
                                      T                         *y,
                                      aoclsparse_int             nRowsblk = 1);

template <typename T>
aoclsparse_status aoclsparse_elltmv(aoclsparse_operation       trans,
                                    const T                   *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const T                   *ell_val,
                                    const aoclsparse_int      *ell_col_ind,
                                    const aoclsparse_int       ell_width,
                                    const aoclsparse_mat_descr descr,
                                    const T                   *x,
                                    const T                   *beta,
                                    T                         *y);

template <typename T>
aoclsparse_status aoclsparse_ellthybmv(aoclsparse_operation       trans,
                                       const T                   *alpha,
                                       aoclsparse_int             m,
                                       aoclsparse_int             n,
                                       aoclsparse_int             nnz,
                                       const T                   *ell_val,
                                       const aoclsparse_int      *ell_col_ind,
                                       const aoclsparse_int       ell_width,
                                       const aoclsparse_int       ell_m,
                                       const T                   *csr_val,
                                       const aoclsparse_int      *csr_row_ind,
                                       const aoclsparse_int      *csr_col_ind,
                                       aoclsparse_int            *row_idx_map,
                                       aoclsparse_int            *csr_row_idx_map,
                                       const aoclsparse_mat_descr descr,
                                       const T                   *x,
                                       const T                   *beta,
                                       T                         *y);

template <typename T>
aoclsparse_status aoclsparse_mv(aoclsparse_operation       trans,
                                const T                   *alpha,
                                aoclsparse_matrix          A,
                                const aoclsparse_mat_descr descr,
                                const T                   *x,
                                const T                   *beta,
                                T                         *y);

template <typename T>
aoclsparse_status aoclsparse_diamv(aoclsparse_operation       trans,
                                   const T                   *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const T                   *dia_val,
                                   const aoclsparse_int      *dia_offset,
                                   aoclsparse_int             dia_num_diag,
                                   const aoclsparse_mat_descr descr,
                                   const T                   *x,
                                   const T                   *beta,
                                   T                         *y);

template <typename T>
aoclsparse_status aoclsparse_bsrmv(aoclsparse_operation       trans,
                                   const T                   *alpha,
                                   aoclsparse_int             mb,
                                   aoclsparse_int             nb,
                                   aoclsparse_int             bsr_dim,
                                   const T                   *bsr_val,
                                   const aoclsparse_int      *bsr_col_ind,
                                   const aoclsparse_int      *bsr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const T                   *x,
                                   const T                   *beta,
                                   T                         *y);
template <typename T>
aoclsparse_status aoclsparse_csrsv(aoclsparse_operation       trans,
                                   const T                   *alpha,
                                   aoclsparse_int             m,
                                   const T                   *csr_val,
                                   const aoclsparse_int      *csr_col_ind,
                                   const aoclsparse_int      *csr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const T                   *x,
                                   T                         *y);
template <typename T>
aoclsparse_status aoclsparse_trsv_kid(const aoclsparse_operation trans,
                                      const T                    alpha,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      const T                   *b,
                                      T                         *x,
                                      const aoclsparse_int       kid);
template <typename T>
aoclsparse_status aoclsparse_dotmv(const aoclsparse_operation op,
                                   T                          alpha,
                                   aoclsparse_matrix          A,
                                   aoclsparse_mat_descr       descr,
                                   T                         *x,
                                   T                          beta,
                                   T                         *y,
                                   T                         *d);

/*
 * ===========================================================================
 *    level 1 SPARSE
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_gthr(const aoclsparse_int  nnz,
                                  const T              *y,
                                  T                    *x,
                                  const aoclsparse_int *indx,
                                  aoclsparse_int        kid = -1);

template <typename T>
aoclsparse_status aoclsparse_gthrz(
    const aoclsparse_int nnz, T *y, T *x, const aoclsparse_int *indx, aoclsparse_int kid = -1);

template <typename T>
aoclsparse_status aoclsparse_gthrs(
    const aoclsparse_int nnz, const T *y, T *x, aoclsparse_int stride, aoclsparse_int kid = -1);

template <typename T, typename R>
R aoclsparse_dot(const aoclsparse_int nnz,
                 const T *__restrict__ x,
                 const aoclsparse_int *__restrict__ indx,
                 const T *__restrict__ y,
                 T *__restrict__ dot,
                 bool                 conj,
                 const aoclsparse_int kid);

template <typename T>
aoclsparse_status aoclsparse_axpyi(aoclsparse_int nnz,
                                   T              a,
                                   const T *__restrict__ x,
                                   const aoclsparse_int *__restrict__ indx,
                                   T *__restrict__ y,
                                   [[maybe_unused]] const aoclsparse_int kid);

/*
 * ===========================================================================
 *    Conversion
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_csr2ell(aoclsparse_int             m,
                                     const aoclsparse_mat_descr descr,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const T                   *csr_val,
                                     aoclsparse_int            *ell_col_ind,
                                     T                         *ell_val,
                                     aoclsparse_int             ell_width);

template <typename T>
aoclsparse_status aoclsparse_csr2ellt(aoclsparse_int             m,
                                      const aoclsparse_mat_descr descr,
                                      const aoclsparse_int      *csr_row_ptr,
                                      const aoclsparse_int      *csr_col_ind,
                                      const T                   *csr_val,
                                      aoclsparse_int            *ell_col_ind,
                                      T                         *ell_val,
                                      aoclsparse_int             ell_width);

template <typename T>
aoclsparse_status aoclsparse_csr2ellthyb(aoclsparse_int        m,
                                         aoclsparse_index_base base,
                                         aoclsparse_int       *ell_m,
                                         const aoclsparse_int *csr_row_ptr,
                                         const aoclsparse_int *csr_col_ind,
                                         const T              *csr_val,
                                         aoclsparse_int       *row_idx_map,
                                         aoclsparse_int       *csr_row_idx_map,
                                         aoclsparse_int       *ell_col_ind,
                                         T                    *ell_val,
                                         aoclsparse_int        ell_width);

template <typename T>
aoclsparse_status aoclsparse_csr2dia(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     const aoclsparse_mat_descr descr,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const T                   *csr_val,
                                     aoclsparse_int             dia_num_diag,
                                     aoclsparse_int            *dia_offset,
                                     T                         *dia_val);

template <typename T>
aoclsparse_status aoclsparse_csr2bsr(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     const aoclsparse_mat_descr descr,
                                     const T                   *csr_val,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     aoclsparse_int             block_dim,
                                     T                         *bsr_val,
                                     aoclsparse_int            *bsr_row_ptr,
                                     aoclsparse_int            *bsr_col_ind);

template <typename T>
aoclsparse_status aoclsparse_csr2csc(aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     aoclsparse_int             nnz,
                                     const aoclsparse_mat_descr descr,
                                     aoclsparse_index_base      baseCSC,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_int      *csr_col_ind,
                                     const T                   *csr_val,
                                     aoclsparse_int            *csc_row_ind,
                                     aoclsparse_int            *csc_col_ptr,
                                     T                         *csc_val);

template <typename T>
aoclsparse_status aoclsparse_csr2dense(aoclsparse_int             m,
                                       aoclsparse_int             n,
                                       const aoclsparse_mat_descr descr,
                                       const T                   *csr_val,
                                       const aoclsparse_int      *csr_row_ptr,
                                       const aoclsparse_int      *csr_col_ind,
                                       T                         *A,
                                       aoclsparse_int             ld,
                                       aoclsparse_order           order);

/*
 * ===========================================================================
 *    Sparse Solvers
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_ilu_smoother(aoclsparse_operation       trans,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          T                        **precond_csr_val,
                                          const T                   *approx_inv_diag,
                                          T                         *x,
                                          const T                   *b);

template <typename T>
aoclsparse_status aoclsparse_sorv(aoclsparse_sor_type        sor_type,
                                  const aoclsparse_mat_descr descr,
                                  const aoclsparse_matrix    A,
                                  T                          omega,
                                  T                          alpha,
                                  T                         *x,
                                  const T                   *b);

template <typename T>
aoclsparse_status aoclsparse_sctr(const aoclsparse_int nnz,
                                  const T *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  T *__restrict__ y,
                                  const aoclsparse_int kid = -1);

template <typename T>
aoclsparse_status aoclsparse_sctrs(const aoclsparse_int nnz,
                                   const T *__restrict__ x,
                                   aoclsparse_int stride,
                                   T *__restrict__ y,
                                   const aoclsparse_int kid = -1);

template <typename T>
aoclsparse_status aoclsparse_roti(const aoclsparse_int nnz,
                                  T *__restrict__ x,
                                  const aoclsparse_int *__restrict__ indx,
                                  T *__restrict__ y,
                                  const T              c,
                                  const T              s,
                                  const aoclsparse_int kid = -1);
/*
 * ===========================================================================
 *    Initialize
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        csr_nnz,
                                        aoclsparse_int       *csr_row_ptr,
                                        aoclsparse_int       *csr_col_ptr,
                                        T                    *csr_val);

template <typename T>
aoclsparse_status aoclsparse_create_bsr(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_order      order,
                                        aoclsparse_int        bM,
                                        aoclsparse_int        bN,
                                        aoclsparse_int        block_dim,
                                        aoclsparse_int       *row_ptr,
                                        aoclsparse_int       *col_idx,
                                        T                    *val,
                                        bool                  fast_chck);

template <typename T>
aoclsparse_status aoclsparse_create_tcsr(aoclsparse_matrix    *mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *row_ptr_L,
                                         aoclsparse_int       *row_ptr_U,
                                         aoclsparse_int       *col_idx_L,
                                         aoclsparse_int       *col_idx_U,
                                         T                    *val_L,
                                         T                    *val_U);

template <typename T>
aoclsparse_status aoclsparse_create_csc(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *col_ptr,
                                        aoclsparse_int       *row_idx,
                                        T                    *val);

template <typename T>
aoclsparse_status aoclsparse_create_coo(aoclsparse_matrix    *mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *row_ind,
                                        aoclsparse_int       *col_idx,
                                        T                    *val);

template <typename T>
aoclsparse_status
    aoclsparse_set_value(aoclsparse_matrix mat, aoclsparse_int row, aoclsparse_int col, T val);

template <typename T>
aoclsparse_status aoclsparse_export_csr(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **row_ptr,
                                        aoclsparse_int        **col_idx,
                                        T                     **val);

template <typename T>
aoclsparse_status aoclsparse_export_csc(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **col_ptr,
                                        aoclsparse_int        **row_idx,
                                        T                     **val);
template <typename T>
aoclsparse_status aoclsparse_symgs_kid(aoclsparse_operation       trans,
                                       aoclsparse_matrix          A,
                                       const aoclsparse_mat_descr descr,
                                       const T                    alpha,
                                       const T                   *b,
                                       T                         *x,
                                       const aoclsparse_int       kid);
template <typename T>
aoclsparse_status aoclsparse_symgs(aoclsparse_operation       trans,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   const T                    alpha,
                                   const T                   *b,
                                   T                         *x);
template <typename T>
aoclsparse_status aoclsparse_symgs_mv_kid(aoclsparse_operation       trans,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          const T                    alpha,
                                          const T                   *b,
                                          T                         *x,
                                          T                         *y,
                                          const aoclsparse_int       kid);
template <typename T>
aoclsparse_status aoclsparse_symgs_mv(aoclsparse_operation       trans,
                                      aoclsparse_matrix          A,
                                      const aoclsparse_mat_descr descr,
                                      const T                    alpha,
                                      const T                   *b,
                                      T                         *x,
                                      T                         *y);

template <typename T>
aoclsparse_status aoclsparse_export_coo(const aoclsparse_matrix mat,
                                        aoclsparse_index_base  *base,
                                        aoclsparse_int         *m,
                                        aoclsparse_int         *n,
                                        aoclsparse_int         *nnz,
                                        aoclsparse_int        **row_ptr,
                                        aoclsparse_int        **col_ptr,
                                        T                     **val);

template <typename T>
aoclsparse_status aoclsparse_update_values(aoclsparse_matrix mat, aoclsparse_int len, T *val);

template <typename T>
aoclsparse_status
    aoclsparse_itsol_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const T *b);

template <typename T>
aoclsparse_status aoclsparse_itsol_rci_solve(aoclsparse_itsol_handle   handle,
                                             aoclsparse_itsol_rci_job *ircomm,
                                             T                       **u,
                                             T                       **v,
                                             T                        *x,
                                             tolerance_t<T>            rinfo[100]);

#endif /*AOCLSPARSE_INTERFACE_HPP*/
