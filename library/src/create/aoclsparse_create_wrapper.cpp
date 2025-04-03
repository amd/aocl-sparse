/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse_descr.h"
#include "aoclsparse_types.h"
#include "aoclsparse.hpp"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_mat_structures.hpp"

/********************************************************************************
 * \brief aoclsparse_create_csr sets the sparse matrix in the CSR format for
 * the appropriate data type (float, double, float complex, double complex).
 ********************************************************************************/
// Macro to generate aoclsparse_create_*csr functions for each type
#define DEFINE_CREATE_CSR_FUNC(PREFIX, SUF)                                                     \
    extern "C" aoclsparse_status aoclsparse_create_##PREFIX##csr(aoclsparse_matrix    *mat,     \
                                                                 aoclsparse_index_base base,    \
                                                                 aoclsparse_int        M,       \
                                                                 aoclsparse_int        N,       \
                                                                 aoclsparse_int        nnz,     \
                                                                 aoclsparse_int       *row_ptr, \
                                                                 aoclsparse_int       *col_idx, \
                                                                 SUF                  *val)     \
    {                                                                                           \
        /* In-case of complex types, the val pointer is cast to std::complex<T> internally */   \
        using internal_type = typename get_data_type<SUF>::type;                                \
        return aoclsparse::create_csr(                                                          \
            mat, base, M, N, nnz, row_ptr, col_idx, reinterpret_cast<internal_type *>(val));    \
    }

INSTANTIATE_FOR_ALL_TYPES_SUFFIX(DEFINE_CREATE_CSR_FUNC);

/********************************************************************************
 * \brief aoclsparse_create_bsr sets the sparse matrix in the BSR format for
 * the appropriate data type (float, double, float complex, double complex).
 ********************************************************************************/
// Macro to generate aoclsparse_create_*bsr functions for each type
#define DEFINE_CREATE_BSR_FUNC(PREFIX, SUF)                                                        \
    extern "C" aoclsparse_status aoclsparse_create_##PREFIX##bsr(aoclsparse_matrix     *mat,       \
                                                                 aoclsparse_index_base  base,      \
                                                                 const aoclsparse_order order,     \
                                                                 const aoclsparse_int   bM,        \
                                                                 const aoclsparse_int   bN,        \
                                                                 const aoclsparse_int   block_dim, \
                                                                 aoclsparse_int        *row_ptr,   \
                                                                 aoclsparse_int        *col_idx,   \
                                                                 SUF                   *val,       \
                                                                 bool fast_chck = false)           \
    {                                                                                              \
        /* In-case of complex types, the val pointer is cast to std::complex<T> internally */      \
        using internal_type = typename get_data_type<SUF>::type;                                   \
        return aoclsparse::create_bsr(mat,                                                         \
                                      base,                                                        \
                                      order,                                                       \
                                      bM,                                                          \
                                      bN,                                                          \
                                      block_dim,                                                   \
                                      row_ptr,                                                     \
                                      col_idx,                                                     \
                                      reinterpret_cast<internal_type *>(val),                      \
                                      fast_chck);                                                  \
    }

INSTANTIATE_FOR_ALL_TYPES_SUFFIX(DEFINE_CREATE_BSR_FUNC);