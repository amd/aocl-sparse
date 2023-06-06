/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_csrmm.hpp"

#include <algorithm>
#include <cmath>

template <typename T>
aoclsparse_status aoclsparse_csrmm_template(aoclsparse_operation       trans,
                                            const T                   *alpha,
                                            const aoclsparse_matrix    csr,
                                            const aoclsparse_mat_descr descr,
                                            aoclsparse_order           order,
                                            const T                   *B,
                                            aoclsparse_int             n,
                                            aoclsparse_int             ldb,
                                            const T                   *beta,
                                            T                         *C,
                                            aoclsparse_int             ldc)
{
    // Check for valid handle and matrix descriptor
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(csr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(!((csr->val_type == aoclsparse_dmat && std::is_same_v<T, double>)
         || (csr->val_type == aoclsparse_smat && std::is_same_v<T, float>)))
    {
        return aoclsparse_status_wrong_type;
    }
    aoclsparse_int        m           = csr->m;
    aoclsparse_int        k           = csr->n;
    const T              *csr_val     = static_cast<T *>(csr->csr_mat.csr_val);
    const aoclsparse_int *csr_col_ind = csr->csr_mat.csr_col_ptr;
    const aoclsparse_int *csr_row_ptr = csr->csr_mat.csr_row_ptr;
    if(trans != aoclsparse_operation_none)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    if(descr->type != aoclsparse_matrix_type_general)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }
    // Check sizes
    if(m < 0 || n < 0 || k < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || k == 0)
    {
        return aoclsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //
    if(alpha == nullptr || beta == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if(*alpha == static_cast<T>(0) && *beta == static_cast<T>(1))
    {
        return aoclsparse_status_success;
    }

    //
    // Check the rest of pointer arguments
    //
    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || B == nullptr
       || C == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check leading dimension of B
    aoclsparse_int one = 1;
    if(ldb < std::max(one, order == aoclsparse_order_column ? k : n))
    {
        return aoclsparse_status_invalid_size;
    }

    // Check leading dimension of C
    if(ldc < std::max(one, order == aoclsparse_order_column ? m : n))
    {
        return aoclsparse_status_invalid_size;
    }

    if(order == aoclsparse_order_column)
        return aoclsparse_csrmm_col_major(
            alpha, csr_val, csr_col_ind, csr_row_ptr, m, k, B, n, ldb, beta, C, ldc);
    else
        return aoclsparse_csrmm_row_major(
            alpha, csr_val, csr_col_ind, csr_row_ptr, m, k, B, n, ldb, beta, C, ldc);
}

#define INSTANTIATE(TTYPE)                                                                        \
    template aoclsparse_status aoclsparse_csrmm_template<TTYPE>(aoclsparse_operation       trans, \
                                                                const TTYPE               *alpha, \
                                                                const aoclsparse_matrix    csr,   \
                                                                const aoclsparse_mat_descr descr, \
                                                                aoclsparse_order           order, \
                                                                const TTYPE               *B,     \
                                                                aoclsparse_int             n,     \
                                                                aoclsparse_int             ldb,   \
                                                                const TTYPE               *beta,  \
                                                                TTYPE                     *C,     \
                                                                aoclsparse_int             ldc);

INSTANTIATE(float);
INSTANTIATE(double);

/*
 * * ===========================================================================
 * *    C wrapper
 * * ===========================================================================
 * */

#define C_IMPL(NAME, TYPE)                                              \
    extern "C" aoclsparse_status NAME(aoclsparse_operation       trans, \
                                      const TYPE                *alpha, \
                                      const aoclsparse_matrix    csr,   \
                                      const aoclsparse_mat_descr descr, \
                                      aoclsparse_order           order, \
                                      const TYPE                *B,     \
                                      aoclsparse_int             n,     \
                                      aoclsparse_int             ldb,   \
                                      const TYPE                *beta,  \
                                      TYPE                      *C,     \
                                      aoclsparse_int             ldc)   \
    {                                                                   \
        return aoclsparse_csrmm_template(                               \
            trans, alpha, csr, descr, order, B, n, ldb, beta, C, ldc);  \
    }

C_IMPL(aoclsparse_scsrmm, float);
C_IMPL(aoclsparse_dcsrmm, double);

#undef C_IMPL
