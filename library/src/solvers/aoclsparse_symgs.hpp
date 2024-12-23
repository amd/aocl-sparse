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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_SYMGS_HPP
#define AOCLSPARSE_SYMGS_HPP

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_analysis.hpp"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_l2.hpp"
#include "aoclsparse_mv.hpp"
#include "aoclsparse_utils.hpp"

#include <immintrin.h>
#include <type_traits>

#define KT_ADDRESS_TYPE aoclsparse_int
#include "aoclsparse_kernel_templates.hpp"
#undef KT_ADDRESS_TYPE
#if defined(_WIN32) || defined(_WIN64)
//Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

/* Symmetric Gauss Seidel Preconditioner (SYMGS)
 *  ============================================
 * Computes a single iteration of Gauss Seidel relaxation. This smoothing
 * algorithm typically is part of Krylov methods such as CG (Conjugate Gradient),
 * GMRES (Generalized Minimal Residual) to accelerate the convergence rate. A
 * flag 'fuse_mv' controls whether the final sparse product is computed using
 * the 'x' estimate.
 * Note: The SYMGS operation requires 2 working buffers which are allocated
 * as part of aoclsparse_optimize_symgs()
 *
 */

template <typename T>
aoclsparse_status aoclsparse_symgs(aoclsparse_operation       trans,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   const T                    alpha,
                                   const T                   *b,
                                   T                         *x,
                                   T                         *y,
                                   const aoclsparse_int       kid,
                                   bool                       fuse_mv);

/* Core computation of SYMGS,
 * Splitting A = L + D + U, where L and U are strictly lower and upper
 * triangular parts of the matrix A respectively, Symmetric Gauss Seidel (SYMGS)
 * is implemented as follows:
 *      1. compute (L + D)*x1 = b - alpha*U*x0;
 *              --- 1.1 spmv operation q = alpha*U*x0, which is the product of
 *                  strict upper triangle and initial x, scaled by alpha
 *              --- 1.2 a simple subtraction, r = b - q
 *              --- 1.3 trsv operation "(L+D).x1 = r", which is lower triangle
 *                  solve to compute "x1" using results from (1.2)
 *      2. solve (U + D)*x = b - L*x1;
 *              --- 2.1 spmv operation r = L*x1, which is the product of
 *                  strict lower triangle and x1 (from 1.3)
 *              --- 2.2 a simple subtraction, q = b - r
 *              --- 2.3 trsv operation "(U+D).x = q", which is upper triangle
 *                  solve to compute the final "x" using results from (2.2)
 *      3. compute: y = A. x
 *              --- if fuse_mv is enabled, symmetric gauss seidel preconditioner
 *                  is followed by a spmv operation between "x" (from 2.3) and
 *                  the original matrix "A"
 */

template <typename T>
aoclsparse_status symgs_ref(aoclsparse_operation       trans,
                            const T                    alpha,
                            aoclsparse_matrix          A,
                            const aoclsparse_mat_descr descr,
                            const T *__restrict__ b,
                            T *__restrict__ x,
                            T *__restrict__ y,
                            bool fuse_mv)
{
    _aoclsparse_mat_descr descr_cpy;
    aoclsparse_operation  trans_cpy;
    aoclsparse_operation  u_trans, l_trans;
    aoclsparse_diag_type  dtype_strict, dtype;
    aoclsparse_fill_mode  u_fmode, l_fmode;

    aoclsparse_int    avxversion;
    aoclsparse_status status;
    T                *r         = (T *)A->symgs_info.r;
    T                *q         = (T *)A->symgs_info.q;
    T                 beta      = aoclsparse_numeric::zero<T>();
    T                 alpha_one = (T)1;

    aoclsparse_copy_mat_descr(&descr_cpy, descr);
    // Use default AVX extension
    avxversion = -1;
    //Quick exit if the input matrix is triangular, using a single TRSV (+ Final SPMV)
    if(descr->type == aoclsparse_matrix_type_triangular)
    {
        /*
            Step 1: TRSV
            1. if Lower triangle only, solve (L + D)*x = b;
            2. if Upper triangle only, solve (U + D)*x = b;
        */
        status = aoclsparse_trsv(trans, alpha_one, A, descr, b, 1, x, 1, avxversion);
        if(status != aoclsparse_status_success)
            return status;

        if(fuse_mv)
        {
            /*
                Step 2: Sparse product
                y = A . x
            */
            status = aoclsparse_mv_t<T>(trans, &alpha_one, A, descr, x, &beta, y);
        }
        return status;
    }

    dtype        = aoclsparse_diag_type_non_unit;
    dtype_strict = aoclsparse_diag_type_zero;
    u_fmode = l_fmode = aoclsparse_fill_mode_lower;
    l_trans           = aoclsparse_operation_none;
    if(descr->type == aoclsparse_matrix_type_hermitian)
    {
        u_trans = aoclsparse_operation_conjugate_transpose;
    }
    else
    {
        u_trans = aoclsparse_operation_transpose;
    }
    //aoclsparse_matrix_type_symmetric: default setting is lower, update if upper below
    if(descr->type == aoclsparse_matrix_type_symmetric
       && descr->fill_mode == aoclsparse_fill_mode_upper)
    {
        //control using transpose operation without changing fill mode
        u_fmode = l_fmode = aoclsparse_fill_mode_upper;
        u_trans           = aoclsparse_operation_none;
        l_trans           = aoclsparse_operation_transpose;
    }
    else if(descr->type == aoclsparse_matrix_type_general && trans == aoclsparse_operation_none)
    {
        //general case and no transpose
        u_trans = l_trans = aoclsparse_operation_none;
        u_fmode           = aoclsparse_fill_mode_upper;
    }
    else if(descr->type == aoclsparse_matrix_type_general
            && trans == aoclsparse_operation_transpose)
    {
        //access triangles in general mode, as transpose
        u_trans = l_trans = aoclsparse_operation_transpose;
        l_fmode           = aoclsparse_fill_mode_upper;
        u_fmode           = aoclsparse_fill_mode_lower;
    }
    else if(descr->type == aoclsparse_matrix_type_hermitian
            && descr->fill_mode == aoclsparse_fill_mode_upper)
    {
        //control using transpose operation without changing fill mode
        u_fmode = l_fmode = aoclsparse_fill_mode_upper;
        u_trans           = aoclsparse_operation_none;
        l_trans           = aoclsparse_operation_conjugate_transpose;
    }
    aoclsparse_set_mat_type(&descr_cpy, aoclsparse_matrix_type_triangular);
    /*
        Step 1: (L+D) . x1 = b - alpha.U.x0
                U.x0 = alpha.U.x, where is x is initial vector
                q = alpha . U . x, where U is strictly upper triangle
                r = b - U.x0 = b - q, where q = U.x
    */
    set_symgs_matrix_properties(&descr_cpy, &trans_cpy, u_fmode, dtype_strict, u_trans);
    //Step 1.1: q = alpha.U.x0

    status = aoclsparse_mv_t(trans_cpy, &alpha, A, &descr_cpy, x, &beta, q);

    //Step 1.2: r = b - q = b - alpha.U.x0
    for(aoclsparse_int i = 0; i < A->m; i++)
    {
        r[i] = b[i] - q[i];
    }

    /*
        Step 1.3: (L+D) . x1 = r, Use TRSV(L) to solve for x1 ?
                (L+D)q = r
    */
    set_symgs_matrix_properties(&descr_cpy, &trans_cpy, l_fmode, dtype, l_trans);
    status = aoclsparse_trsv(trans_cpy, alpha_one, A, &descr_cpy, r, 1, q, 1, avxversion);
    if(status != aoclsparse_status_success)
        return status;
    /*
        Step 2
    */
    /*
        Step 2: (U + D)*x = b - L*x1;
                (U + D)*x = b - L*q;, x1 = q in our case
                    L*x1 = L*q ,where is q is intermediate output from step 1
                    r = L . q, where L is strictly lower triangle
                    q = b - L*x1 = b - r, where r = L*x1
                (U + D)*x = q;
    */
    set_symgs_matrix_properties(&descr_cpy, &trans_cpy, l_fmode, dtype_strict, l_trans);
    //Step 2.1: r = L.q = L.x1

    status = aoclsparse_mv_t(trans_cpy, &alpha_one, A, &descr_cpy, q, &beta, r);

    //Step 2.2: q = b - r = (b - L.x1)
    for(aoclsparse_int i = 0; i < A->m; i++)
    {
        q[i] = b[i] - r[i];
    }

    set_symgs_matrix_properties(&descr_cpy, &trans_cpy, u_fmode, dtype, u_trans);
    //Step 2.3: (U + D).x = q
    status = aoclsparse_trsv(trans_cpy, alpha_one, A, &descr_cpy, q, 1, x, 1, avxversion);
    if(status != aoclsparse_status_success)
        return status;

    if(fuse_mv)
    {
        /*
            Step 3: Sparse product
            y = A . x
        */
        status = aoclsparse_mv_t(trans, &alpha_one, A, descr, x, &beta, y);

        if(status != aoclsparse_status_success)
            return status;
    }

    return aoclsparse_status_success;
}
/*
 * symgs dispatcher
 * ===============
 */
template <typename T>
aoclsparse_status aoclsparse_symgs(
    aoclsparse_operation       trans, /* matrix operation */
    aoclsparse_matrix          A, /* matrix data */
    const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
    const T                    alpha, /* scalar for rescaling RHS */
    const T *__restrict__ b, /* RHS */
    T *__restrict__ x, /* solution */
    T *__restrict__ y, /* sparse product */
    const aoclsparse_int kid, /* Kernel ID request */
    bool fuse_mv) /*flag to determine if the final sparse product operation is needed*/
{
    aoclsparse_status status = aoclsparse_status_success;
    // Check pointer arguments
    if((nullptr == x) || (nullptr == b))
    {
        return aoclsparse_status_invalid_pointer;
    }
    //if spmv fusing enabled, check for y
    if((true == fuse_mv) && (nullptr == y))
    {
        return aoclsparse_status_invalid_pointer;
    }

    if((A == nullptr) || (descr == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }

    //Only CSR format is supported for SYMGS
    if(A->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_not_implemented;
    }

    // Check for base index incompatibility
    // Check if descriptor's index-base is valid (and A's index-base must be the same)
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    // There is an issue that zero-based indexing is defined in two separate places and
    // can lead to ambiguity, we check that both are consistent.
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }

    // Check transpose
    if((trans != aoclsparse_operation_none) && (trans != aoclsparse_operation_transpose)
       && (trans != aoclsparse_operation_conjugate_transpose))
    {
        return aoclsparse_status_invalid_value;
    }
    if(descr->fill_mode != aoclsparse_fill_mode_lower
       && descr->fill_mode != aoclsparse_fill_mode_upper)
        return aoclsparse_status_invalid_value;

    // Support to be added, at present returning not implemented
    if(descr->diag_type == aoclsparse_diag_type_unit)
        return aoclsparse_status_not_implemented;

    // General matrices requesting conjugate transpose are yet to be supported
    if((descr->type == aoclsparse_matrix_type_general)
       && (trans == aoclsparse_operation_conjugate_transpose))
        return aoclsparse_status_not_implemented;

    if(descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_triangular
       && descr->type != aoclsparse_matrix_type_general
       && descr->type != aoclsparse_matrix_type_hermitian)
    {
        return aoclsparse_status_invalid_value;
    }
    if(A->m < 0 || A->nnz < 0 || A->n < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return for size 0 matrices, Do nothing
    if((A->m == 0) || (A->n == 0) || (A->nnz == 0))
        return aoclsparse_status_success;

    if(A->m != A->n) // Matrix not square
    {
        return aoclsparse_status_invalid_size;
    }
    if(A->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }
    // Unpack A and check
    if(!A->opt_csr_ready)
    {
        // user did not check the matrix, call optimize
        status = aoclsparse_csr_csc_optimize<T>(A);
        if(status != aoclsparse_status_success)
        {
            return status;
        }
    }
    if(A->symgs_info.sgs_ready == false)
    {
        /*
            currently optimize API allocates working buffers needed for SGS
            functionality. SGS Optimize functionality to be extended in future
        */
        status = aoclsparse_optimize_symgs(A);
        if(status != aoclsparse_status_success)
        {
            return status;
        }
    }

    const bool unit = descr->diag_type == aoclsparse_diag_type_unit;
    if(!A->opt_csr_full_diag && !unit) // not of full rank, linear system cannot be solved
    {
        return aoclsparse_status_invalid_value;
    }

    aoclsparse_int usekid
        = 0; // Defaults to 0 (reference Gauss Seidel, TRSV to use default AVX extension)
    if(kid >= 0)
    {
        switch(kid)
        {
        case 0:
            /* reference SYMGS implementation based on 2 trsv's and 2 spmv's
                    x0 = x*alpha;
                    (L + D)*x1 = b - U*x0;
                    (U + D)*x = b - L*x1;
                    where A = U + L + D
                */
            usekid = kid;
            break;
        default: // use kid suggested by CPU ID...
            break;
        }
    }

    switch(usekid)
    {
    default: // Reference implementation
        status = symgs_ref<T>(trans, alpha, A, descr, b, x, y, fuse_mv);
        break;
    }
    return status;
}
#endif // AOCLSPARSE_SYMGS_HPP
