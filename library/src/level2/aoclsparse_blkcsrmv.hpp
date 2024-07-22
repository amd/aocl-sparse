/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_BLKCSRMV_HPP
#define AOCLSPARSE_BLKCSRMV_HPP
#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_blkcsrmv_avx512.hpp"

// This routine performs sparse-matrix multiplication on matrices stored in blocked CSR format for double.
// Supports blocking factors of size 1x8, 2x8 and 4x8. Blocking size is chosen depending on the matrix characteristics.
// Blocked SpMV only supports single threaded usecases.
template <typename T>
std::enable_if_t<std::is_same_v<T, double>, aoclsparse_status>
    aoclsparse_blkcsrmv_t(aoclsparse_operation            trans,
                          [[maybe_unused]] const T       *alpha,
                          aoclsparse_int                  m,
                          aoclsparse_int                  n,
                          aoclsparse_int                  nnz,
                          const uint8_t                  *masks,
                          const T                        *blk_csr_val,
                          const aoclsparse_int           *blk_col_ind,
                          const aoclsparse_int           *blk_row_ptr,
                          const aoclsparse_mat_descr      descr,
                          const T                        *x,
                          [[maybe_unused]] const T       *beta,
                          T                              *y,
                          [[maybe_unused]] aoclsparse_int nRowsblk)
{
    using namespace aoclsparse;

    /*
        Check if the requested operation can execute
        This check needs to be done only once in a run
    */
    static bool can_exec
        = context::get_context()->supports<context_isa_t::AVX512F, context_isa_t::AVX512VL>();

    if(can_exec)
    {
        if(descr == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }

        // Check index base
        if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
        {
            return aoclsparse_status_invalid_value;
        }

        // Support General and symmetric matrices.
        // Return for any other matrix type
        if((descr->type != aoclsparse_matrix_type_general)
           && (descr->type != aoclsparse_matrix_type_symmetric))
        {
            // TODO
            return aoclsparse_status_not_implemented;
        }

        if(trans != aoclsparse_operation_none)
        {
            // TODO
            return aoclsparse_status_not_implemented;
        }

        // Check sizes
        if(m < 0)
        {
            return aoclsparse_status_invalid_size;
        }
        else if(n < 8)
        {
            return aoclsparse_status_invalid_size;
        }
        else if(nnz < 0)
        {
            return aoclsparse_status_invalid_size;
        }

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return aoclsparse_status_success;
        }

        // Check pointer arguments
        if(blk_csr_val == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(blk_row_ptr == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(blk_col_ind == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(masks == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(x == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }
        else if(y == nullptr)
        {
            return aoclsparse_status_invalid_pointer;
        }

#ifdef USE_AVX512
        if(nRowsblk == 1)
            return aoclsparse_blkcsrmv_1x8_vectorized_avx512(
                descr->base, *alpha, m, masks, blk_csr_val, blk_col_ind, blk_row_ptr, x, *beta, y);
        if(nRowsblk == 2)
            return aoclsparse_blkcsrmv_2x8_vectorized_avx512(
                descr->base, *alpha, m, masks, blk_csr_val, blk_col_ind, blk_row_ptr, x, *beta, y);
        if(nRowsblk == 4)
            return aoclsparse_blkcsrmv_4x8_vectorized_avx512(
                descr->base, *alpha, m, masks, blk_csr_val, blk_col_ind, blk_row_ptr, x, *beta, y);
        else
            return aoclsparse_status_invalid_size;
#endif
    }
    // else the API cannot be executed
    return aoclsparse_status_not_implemented;
}

#endif // AOCLSPARSE_BLKCSRMV_HPP
