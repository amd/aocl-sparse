/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
 * \brief aoclsparse-types.h defines data types used by aoclsparse
 */
#ifndef AOCLSPARSE_TYPES_H_
#define AOCLSPARSE_TYPES_H_
#include <stddef.h>
#include <stdint.h>

/*! \ingroup types_module
 *  \brief Specifies whether int32 or int64 is used.
 */
#if defined(aoclsparse_ILP64)
typedef int64_t aoclsparse_int;
#else
typedef int32_t aoclsparse_int;
#endif

#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup types_module
 *  \brief List of aoclsparse status codes definition.
 *
 *  \details
 *  This is a list of the \ref aoclsparse_status types that are used by the aoclSPARSE
 *  library.
 */
typedef enum aoclsparse_status_
{
    aoclsparse_status_success         = 0, /**< success. */
    aoclsparse_status_invalid_handle  = 1, /**< handle not initialized, invalid or null. */
    aoclsparse_status_not_implemented = 2, /**< function is not implemented. */
    aoclsparse_status_invalid_pointer = 3, /**< invalid pointer parameter. */
    aoclsparse_status_invalid_size    = 4, /**< invalid size parameter. */
    aoclsparse_status_memory_error    = 5, /**< failed memory allocation, copy, dealloc. */
    aoclsparse_status_internal_error  = 6, /**< other internal library failure. */
    aoclsparse_status_invalid_value   = 7, /**< invalid value parameter. */
    aoclsparse_status_arch_mismatch   = 8, /**< device arch is not supported. */
    aoclsparse_status_zero_pivot      = 9 /**< encountered zero pivot. */
} aoclsparse_status;

/*! \ingroup types_module
 *  \brief Specify the matrix index base.
 *
 *  \details
 *  The \ref aoclsparse_index_base indicates the index base of the indices. For a
 *  given \ref aoclsparse_mat_descr, the \ref aoclsparse_index_base can be set using
 *  aoclsparse_set_mat_index_base(). The current \ref aoclsparse_index_base of a matrix
 *  can be obtained by aoclsparse_get_mat_index_base().
 */
typedef enum aoclsparse_index_base_
{
    aoclsparse_index_base_zero = 0, /**< zero based indexing. */
    aoclsparse_index_base_one  = 1 /**< one based indexing. */
} aoclsparse_index_base;

#ifdef __cplusplus
}
#endif

#endif // AOCLSPARSE_TYPES_H_

