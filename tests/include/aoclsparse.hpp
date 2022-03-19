/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
 *  \brief aoclsparse.hpp exposes C++ templated Sparse Linear Algebra interface
 *  with only the precision templated.
 */

#pragma once
#ifndef AOCLSPARSE_HPP
#define AOCLSPARSE_HPP

#include <aoclsparse.h>

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_csr2m(
		aoclsparse_operation       transA,
		const aoclsparse_mat_descr descrA,
		const aoclsparse_mat_csr   csrA,
		aoclsparse_operation       transB,
		const aoclsparse_mat_descr descrB,
		const aoclsparse_mat_csr   csrB,
		aoclsparse_request         request,
		aoclsparse_mat_csr         *csrC);

template <typename T>
aoclsparse_status aoclsparse_csrmm(
	aoclsparse_operation       trans,
	const T*                   alpha,
	const aoclsparse_mat_csr   csr,
	const aoclsparse_mat_descr descr,
	aoclsparse_order           order,
	const T*                   B,
	aoclsparse_int             n,
	aoclsparse_int             ldb,
	const T*                   beta,
	T*                         C,
	aoclsparse_int             ldc);

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T>
aoclsparse_status aoclsparse_csrmv(
	aoclsparse_operation       trans,
	const T*                   alpha,
	aoclsparse_int             m,
	aoclsparse_int             n,
	aoclsparse_int             nnz,
	const T*                   csr_val,
	const aoclsparse_int*      csr_col_ind,
	const aoclsparse_int*      csr_row_ptr,
	const aoclsparse_mat_descr descr,
	const T*                   x,
	const T*                   beta,
	T*                         y);

template <typename T>
aoclsparse_status aoclsparse_ellmv(
	aoclsparse_operation       trans,
	const T*                   alpha,
	aoclsparse_int             m,
	aoclsparse_int             n,
	aoclsparse_int             nnz,
	const T*                   ell_val,
	const aoclsparse_int*      ell_col_ind,
	const aoclsparse_int       ell_width,
	const aoclsparse_mat_descr descr,
	const T*                   x,
	const T*                   beta,
	T*                         y);

template <typename T>
aoclsparse_status aoclsparse_elltmv(
        aoclsparse_operation       trans,
        const T*                   alpha,
        aoclsparse_int             m,
        aoclsparse_int             n,
        aoclsparse_int             nnz,
        const T*                   ell_val,
        const aoclsparse_int*      ell_col_ind,
        const aoclsparse_int       ell_width,
        const aoclsparse_mat_descr descr,
        const T*                   x,
        const T*                   beta,
        T*                         y);

template <typename T>
aoclsparse_status aoclsparse_ellthybmv(
        aoclsparse_operation       trans,
        const T*                   alpha,
        aoclsparse_int             m,
        aoclsparse_int             n,
        aoclsparse_int             nnz,
        const T*                   ell_val,
        const aoclsparse_int*      ell_col_ind,
        const aoclsparse_int       ell_width,
        const aoclsparse_int       ell_m,
        const T*               csr_val,
        const aoclsparse_int*      csr_row_ind,
        const aoclsparse_int*      csr_col_ind,
        aoclsparse_int*            row_idx_map,
        aoclsparse_int*            csr_row_idx_map,
        const aoclsparse_mat_descr descr,
        const T*                   x,
        const T*                   beta,
        T*                         y);

template <typename T>
aoclsparse_status aoclsparse_diamv(
	aoclsparse_operation       trans,
	const T*                   alpha,
	aoclsparse_int             m,
	aoclsparse_int             n,
	aoclsparse_int             nnz,
	const T*                   dia_val,
	const aoclsparse_int*      dia_offset,
	aoclsparse_int             dia_num_diag,
	const aoclsparse_mat_descr descr,
	const T*                   x,
	const T*                   beta,
	T*                         y );

template <typename T>
aoclsparse_status aoclsparse_bsrmv(
	aoclsparse_operation       trans,
	const T*                   alpha,
	aoclsparse_int             mb,
	aoclsparse_int             nb,
	aoclsparse_int             bsr_dim,
	const T*                   bsr_val,
	const aoclsparse_int*      bsr_col_ind,
	const aoclsparse_int*      bsr_row_ptr,
	const aoclsparse_mat_descr descr,
	const T*                   x,
	const T*                   beta,
	T*                         y);
template <typename T>
aoclsparse_status aoclsparse_csrsv(
	aoclsparse_operation       trans,
	const T*                   alpha,
	aoclsparse_int             m,
	const T*                   csr_val,
	const aoclsparse_int*      csr_col_ind,
	const aoclsparse_int*      csr_row_ptr,
	const aoclsparse_mat_descr descr,
	const T*                   x,
	T*                         y );

template <typename T>
aoclsparse_status aoclsparse_csr2ell(
	aoclsparse_int       m,
	const aoclsparse_int *csr_row_ptr,
	const aoclsparse_int *csr_col_ind,
	const T              *csr_val,
	aoclsparse_int       *ell_col_ind,
	T                    *ell_val,
	aoclsparse_int       ell_width);

template <typename T>
aoclsparse_status aoclsparse_csr2ellt(
        aoclsparse_int       m,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const T              *csr_val,
        aoclsparse_int       *ell_col_ind,
        T                    *ell_val,
        aoclsparse_int       ell_width);

template <typename T>
aoclsparse_status aoclsparse_csr2ellthyb(
        aoclsparse_int       m,
	aoclsparse_int       *ell_m,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const T              *csr_val,
       aoclsparse_int       *row_idx_map,
        aoclsparse_int       *csr_row_idx_map,
        aoclsparse_int       *ell_col_ind,
        T                    *ell_val,
        aoclsparse_int       ell_width);

template <typename T>
aoclsparse_status aoclsparse_csr2dia(
        aoclsparse_int       m,
        aoclsparse_int       n,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const T              *csr_val,
        aoclsparse_int       dia_num_diag,
        aoclsparse_int       *dia_offset,
        T                    *dia_val);

template <typename T>
aoclsparse_status aoclsparse_csr2bsr(
	aoclsparse_int        m,
	aoclsparse_int        n,
	const T*              csr_val,
	const aoclsparse_int* csr_row_ptr,
	const aoclsparse_int* csr_col_ind,
	aoclsparse_int        block_dim,
	T*                    bsr_val,
	aoclsparse_int*       bsr_row_ptr,
	aoclsparse_int*       bsr_col_ind);

template <typename T>
aoclsparse_status aoclsparse_csr2csc(
        aoclsparse_int       m,
        aoclsparse_int       n,
        aoclsparse_int       nnz,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const T              *csr_val,
        aoclsparse_int       *csc_row_ind,
        aoclsparse_int       *csc_col_ptr,
        T                    *csc_val);

template <typename T>
aoclsparse_status aoclsparse_csr2dense(
            aoclsparse_int             m,
            aoclsparse_int             n,
            const aoclsparse_mat_descr descr,
            const T*                   csr_val,
            const aoclsparse_int*      csr_row_ptr,
            const aoclsparse_int*      csr_col_ind,
            T*                         A,
            aoclsparse_int             ld,
            aoclsparse_order           order);

#endif /*AOCLSPARSE_HPP*/
