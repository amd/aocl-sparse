/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_SM_HPP
#define AOCLSPARSE_SM_HPP

#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_gthr.hpp"
#include "aoclsparse_sctr.hpp"
#include "aoclsparse_trsv.hpp"
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

/* TRiangular Solver for Multiple RHS
 *  ==================================
 * Solves A*x = alpha*B or A^T*x = alpha*B or A^H*x = alpha*B with
 * A lower (L) or upper (U) triangular, B and X dense matrices.
 * Optimized version, requires A to have been previously "optimized". If A is not
 * optimized explicitly, it is optimized on the fly.
 * Calls multiple times TRSV over each column of B.
 * Note: for some dense layouts, temporary vectors for the columns of X and B are
 * allocated.
 */
template <typename T>
aoclsparse_status
    aoclsparse_trsv(const aoclsparse_operation transpose, /* matrix operation */
                    const T                    alpha, /* scalar for rescaling RHS */
                    aoclsparse_matrix          A, /* matrix data */
                    const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
                    const T                   *b, /* RHS */
                    T                         *x, /* solution */
                    const aoclsparse_int       kid /* Kernel ID request */);

template <typename T>
aoclsparse_status
    aoclsparse_trsm(const aoclsparse_operation transpose, /* matrix operation */
                    const T                    alpha, /* scalar for rescaling RHS */
                    aoclsparse_matrix          A, /* matrix data */
                    const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
                    aoclsparse_order     order, /*Layout of the right-hand-side dense-matrix B*/
                    const T             *B, /* RHS dense matrix mxn*/
                    aoclsparse_int       n, /*number of columns of the dense matrix B*/
                    aoclsparse_int       ldb, /*leading dimension of dense matrix B*/
                    T                   *X, /*solution matrix*/
                    aoclsparse_int       ldx, /*leading dimension of dense matrix X*/
                    const aoclsparse_int kid /* Kernel ID request */);

/*
 * TRSM dispatcher
 * ===============
 */
template <typename T>
aoclsparse_status
    aoclsparse_trsm(const aoclsparse_operation transpose, /* matrix operation */
                    const T                    alpha, /* scalar for rescaling RHS */
                    aoclsparse_matrix          A, /* matrix data */
                    const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
                    aoclsparse_order     order, /*Layout of the right-hand-side dense-matrix B*/
                    const T             *B, /* RHS dense matrix mxn*/
                    aoclsparse_int       n, /*number of columns of the dense matrix B*/
                    aoclsparse_int       ldb, /*leading dimension of dense matrix B*/
                    T                   *X, /*solution matrix*/
                    aoclsparse_int       ldx, /*leading dimension of dense matrix X*/
                    const aoclsparse_int kid /* Kernel ID request */)
{
    aoclsparse_status status = aoclsparse_status_success;
    // Read the environment variables to update global variable
    // This function updates the num_threads only once.
    aoclsparse_init_once();
    aoclsparse_context context;
    context.num_threads = sparse_global_context.num_threads;
#ifdef _OPENMP
    aoclsparse_int chunk;
#endif

    // Quick initial checks
    if(!A || !X || !B || !descr)
        return aoclsparse_status_invalid_pointer;

    // Only CSR input format supported
    if(A->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_not_implemented;
    }

    const aoclsparse_int m = A->m;

    if(m < 0 || A->nnz < 0 || n < 0)
        return aoclsparse_status_invalid_size;

    // Check for a quick exit. Quick return when no of columns in dense matrices is zero
    if(m == 0 || A->n == 0 || A->nnz == 0 || n == 0)
        return aoclsparse_status_success;

    if(m != A->n) // Matrix not square
    {
        return aoclsparse_status_invalid_size;
    }

    if(ldb < 0 || ldx < 0) //invalid leading dimension
    {
        return aoclsparse_status_invalid_size;
    }

    // Check for base index incompatibility
    // There is an issue that zero-based indexing is defined in two separate places and
    // can lead to ambiguity, we check that both are consistent.
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    // Check if descriptor's index-base is valid (and A's index-base must be the same)
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    if(transpose != aoclsparse_operation_none && transpose != aoclsparse_operation_transpose
       && transpose != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;

    if(descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_triangular)
        return aoclsparse_status_invalid_value;

    if(descr->fill_mode != aoclsparse_fill_mode_lower
       && descr->fill_mode != aoclsparse_fill_mode_upper)
        return aoclsparse_status_not_implemented;

    // Unpack A and check
    if(!A->opt_csr_ready)
    {
        // user did not check the matrix, call optimize
        aoclsparse_status status = aoclsparse_csr_optimize<T>(A);
        if(status != aoclsparse_status_success)
            return status;
    }
    // FIXME: remove these work arrays once strided-trsv is done
    std::vector<T> wcolb;
    std::vector<T> wcolx;

    try
    {
        wcolb.resize(m);
        wcolx.resize(m);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    //default reference implementation of gather with stride
    aoclsparse_int gkid = 0;
    //default reference implementation of scatter with stride
    aoclsparse_int skid = 0;
    switch(order)
    {
    case aoclsparse_order_row:
        for(int c = 0; c < n; c++)
        {
            status = aoclsparse_gthrs<T, false>(m, &B[c], &wcolb[0], ldb, gkid);
            status
                = aoclsparse_trsv<T>(transpose, alpha, A, descr, wcolb.data(), wcolx.data(), kid);
            //early exit in case there is error
            if(status != aoclsparse_status_success)
            {
                break;
            }
            status = aoclsparse_scatters<T>(m, &wcolx[0], ldx, &X[c], skid);
        }
        break;
    case aoclsparse_order_column:
#ifdef _OPENMP
        chunk = (n / context.num_threads) ? (n / context.num_threads) : 1;
#pragma omp parallel for num_threads(context.num_threads) schedule(dynamic, chunk)
#endif
        for(int c = 0; c < n; c++)
        {
            status = aoclsparse_trsv<T>(transpose, alpha, A, descr, &B[c * ldb], &X[c * ldx], kid);
        }
        break;
    default:
        status = aoclsparse_status_invalid_value;
        break;
    }
    return status;
}
#endif // AOCLSPARSE_SM_HPP
