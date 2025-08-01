/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse.hpp"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_csr_util.hpp"

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
#ifdef _OPENMP
    aoclsparse_int chunk;
#endif

    // Quick initial checks
    if(!A || !X || !B || !descr)
        return aoclsparse_status_invalid_pointer;
    if(A->mats.empty() || !A->mats[0])
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
    if(A->mats[0]->base != descr->base)
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

    aoclsparse::csr *A_opt_csr = nullptr;
    // call optimize
    status = aoclsparse_csr_csc_optimize<T>(A, &A_opt_csr);
    if(status != aoclsparse_status_success)
        return status;
    if(!A_opt_csr)
        return aoclsparse_status_internal_error;

    aoclsparse_int incb, incx, b_offset, x_offset;

    if(order == aoclsparse_order_row)
    {
        incb     = ldb;
        incx     = ldx;
        b_offset = 1;
        x_offset = 1;
    }
    else if(order == aoclsparse_order_column)
    {
        incb     = 1;
        incx     = 1;
        b_offset = ldb;
        x_offset = ldx;
    }
    else // Early return for invalid order
    {
        return aoclsparse_status_invalid_value;
    }

    using namespace aoclsparse;

#ifdef _OPENMP
    chunk = (n / context::get_context()->get_num_threads())
                ? (n / context::get_context()->get_num_threads())
                : 1;
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) \
    schedule(dynamic, chunk)
#endif
    for(aoclsparse_int ld = 0; ld < n; ++ld)
    {
        status = aoclsparse::trsv<T>(
            transpose, alpha, A, descr, &B[ld * b_offset], incb, &X[ld * x_offset], incx, kid);
    }

    return status;
}
#endif // AOCLSPARSE_SM_HPP
