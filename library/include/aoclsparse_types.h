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
 * \brief aoclsparse_types.h defines data types used by aoclsparse
 */
#ifndef AOCLSPARSE_TYPES_H_
#define AOCLSPARSE_TYPES_H_
#include <stddef.h>
#include <stdint.h>

/*! \ingroup types_module
 *  \brief Macro for function attribute
 *
 *  \details
 *  The macro specifies visibility attribute of public functions
 */
#if defined(_WIN32) || defined(_WIN64)
// Windows specific attribute for exporting function to dll
#define DLL_PUBLIC __declspec(dllexport)
#else
#define DLL_PUBLIC __attribute__((__visibility__("default")))
#endif

/*! \ingroup types_module
 *  \brief Specifies whether int32 or int64 is used.
 */
#if defined(aoclsparse_ILP64)
typedef int64_t aoclsparse_int;
#else
typedef int32_t aoclsparse_int;
#endif

typedef struct
{
    float x, y;
} aoclsparse_float_complex;

typedef struct
{
    double x, y;
} aoclsparse_double_complex;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The aoclSPARSE matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using aoclsparse_create_mat_descr() and the returned
 *  descriptor must be passed to all subsequent library calls that involve the matrix.
 *  It should be destroyed at the end using aoclsparse_destroy_mat_descr().
 */
typedef struct _aoclsparse_mat_descr* aoclsparse_mat_descr;

/*! \ingroup types_module
 *  \brief CSR matrix storage format.
 *
 *  \details
 *  The aoclSPARSE CSR matrix structure holds the CSR matrix. It must be initialized using
 *  aoclsparse_create_(d/s)csr() and the returned CSR matrix must be passed to all
 *  subsequent library calls that involve the matrix. It should be destroyed at the end
 *  using aoclsparse_destroy().
 */

typedef struct _aoclsparse_csr*         aoclsparse_csr;
typedef struct _aoclsparse_ell*         aoclsparse_ell;
typedef struct _aoclsparse_ell_csr_hyb* aoclsparse_ell_csr_hyb;
typedef struct _aoclsparse_matrix*      aoclsparse_matrix;

/* TBD To be deprecated structure and API
 */
typedef struct _aoclsparse_matrix* aoclsparse_mat_csr;
#define aoclsparse_destroy_mat_csr aoclsparse_destroy;
#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup types_module
 *  \brief Specify whether the matrix is to be transposed or not.
 *
 *  \details
 *  The \ref aoclsparse_operation indicates the operation performed with the given matrix.
 */
typedef enum aoclsparse_operation_
{
    aoclsparse_operation_none                = 111, /**< Operate with matrix. */
    aoclsparse_operation_transpose           = 112, /**< Operate with transpose. */
    aoclsparse_operation_conjugate_transpose = 113 /**< Operate with conj. transpose. */
} aoclsparse_operation;

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

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 *  \details
 *  The \ref aoclsparse_matrix_type indices the type of a matrix. For a given
 *  \ref aoclsparse_mat_descr, the \ref aoclsparse_matrix_type can be set using
 *  aoclsparse_set_mat_type(). The current \ref aoclsparse_matrix_type of a matrix can be
 *  obtained by aoclsparse_get_mat_type().
 */
typedef enum aoclsparse_matrix_type_
{
    aoclsparse_matrix_type_general    = 0, /**< general matrix type. */
    aoclsparse_matrix_type_symmetric  = 1, /**< symmetric matrix type. */
    aoclsparse_matrix_type_hermitian  = 2, /**< hermitian matrix type. */
    aoclsparse_matrix_type_triangular = 3 /**< triangular matrix type. */
} aoclsparse_matrix_type;

/*! \ingroup types_module
 *  \brief Specify the matrix data type.
 *
 *  \details
 *  The \ref aoclsparse_matrix_data_type indices the data-type of a matrix.
 */
typedef enum aoclsparse_matrix_data_type_
{
    aoclsparse_dmat = 0, /**< double precision data. */
    aoclsparse_smat = 1, /**< single precision data. */
    aoclsparse_cmat = 2, /**< single precision complex data. */
    aoclsparse_zmat = 3 /**< double precision complex data. */
} aoclsparse_matrix_data_type;

/*! \ingroup types_module
 *  \brief Specify the type of ILU factorization.
 *
 *  \details
 *  The \ref aoclsparse_ilu_type indicates the type of ILU factorization like ILU0, ILU(p) etc.
  */
typedef enum aoclsparse_ilu_type_
{
    aoclsparse_ilu0 = 0, /**< ILU0. */
    aoclsparse_ilup = 1, /**< ILU(p). */
} aoclsparse_ilu_type;

/*! \ingroup types_module
 *  \brief Specify the matrix storage format type.
 *
 *  \details
 *  The \ref aoclsparse_matrix_format_type indices the storage format of a sparse matrix.
 */
typedef enum aoclsparse_matrix_format_type_
{
    aoclsparse_csr_mat          = 0, /**< CSR format. */
    aoclsparse_ell_mat          = 1, /**< ELLPACK format. */
    aoclsparse_ellt_mat         = 2, /**< ELLPACK format stored as transpose format. */
    aoclsparse_ellt_csr_hyb_mat = 3, /**< ELLPACK transpose + CSR hybrid format. */
    aoclsparse_ell_csr_hyb_mat  = 4, /**< ELLPACK + CSR hybrid format. */
    aoclsparse_dia_mat          = 5, /**< diag format. */
    aoclsparse_csr_mat_br4      = 6 /**< Modified CSR format for AVX2 double. */
} aoclsparse_matrix_format_type;

/*! \ingroup types_module
 *  \brief Indicates if the diagonal entries are unity.
 *
 *  \details
 *  The \ref aoclsparse_diag_type indicates whether the diagonal entries of a matrix are
 *  unity or not. If \ref aoclsparse_diag_type_unit is specified, all present diagonal
 *  values will be ignored. For a given \ref aoclsparse_mat_descr, the
 *  \ref aoclsparse_diag_type can be set using aoclsparse_set_mat_diag_type(). The current
 *  \ref aoclsparse_diag_type of a matrix can be obtained by
 *  aoclsparse_get_mat_diag_type().
 */
typedef enum aoclsparse_diag_type_
{
    aoclsparse_diag_type_non_unit = 0, /**< diagonal entries are non-unity. */
    aoclsparse_diag_type_unit     = 1 /**< diagonal entries are unity */
} aoclsparse_diag_type;

/*! \ingroup types_module
 *  \brief Specify the matrix fill mode.
 *
 *  \details
 *  The \ref aoclsparse_fill_mode indicates whether the lower or the upper part is stored
 *  in a sparse triangular matrix. For a given \ref aoclsparse_mat_descr, the
 *  \ref aoclsparse_fill_mode can be set using aoclsparse_set_mat_fill_mode(). The current
 *  \ref aoclsparse_fill_mode of a matrix can be obtained by
 *  aoclsparse_get_mat_fill_mode().
 */
typedef enum aoclsparse_fill_mode_
{
    aoclsparse_fill_mode_lower = 0, /**< lower triangular part is stored. */
    aoclsparse_fill_mode_upper = 1 /**< upper triangular part is stored. */
} aoclsparse_fill_mode;

/*! \ingroup types_module
 *  \brief List of dense matrix ordering.
 *
 *  \details
 *  This is a list of supported \ref aoclsparse_order types that are used to describe the
 *  memory layout of a dense matrix
 */
typedef enum aoclsparse_order_
{
    aoclsparse_order_row    = 0, /**< Row major. */
    aoclsparse_order_column = 1 /**< Column major. */
} aoclsparse_order;

/*! \ingroup types_module
 *  \brief List of aoclsparse status codes definition.
 *
 *  \details
 *  List of \ref aoclsparse_status values returned by the functions in the
 *  library.
 */
typedef enum aoclsparse_status_
{
    aoclsparse_status_success             = 0, /**< success. */
    aoclsparse_status_not_implemented     = 1, /**< functionality is not implemented. */
    aoclsparse_status_invalid_pointer     = 2, /**< invalid pointer parameter. */
    aoclsparse_status_invalid_size        = 3, /**< invalid size parameter. */
    aoclsparse_status_internal_error      = 4, /**< internal library failure. */
    aoclsparse_status_invalid_value       = 5, /**< invalid parameter value. */
    aoclsparse_status_invalid_index_value = 6, /**< invalid index value. */
    aoclsparse_status_maxit               = 7, /**< function stopped after reaching number of iteration limit. */
    aoclsparse_status_user_stop           = 8, /**< user requested termination. */
    aoclsparse_status_wrong_type          = 9, /**< function called on the wrong type (double/float). */
    aoclsparse_status_memory_error        = 10, /**< memory allocation failure. */
    aoclsparse_status_numerical_error     = 11, /**< numerical error, e.g., matrix is not positive definite, divide-by-zero error */
    aoclsparse_status_invalid_operation   = 12, /**< cannot proceed with the request at this point. */
} aoclsparse_status;

/*! \ingroup types_module
 *  \brief List of request stages for sparse matrix * sparse matrix.
 *
 *  \details
 *  This is a list of the \ref aoclsparse_request types that are used by the
 *  aoclsparse_csr2m funtion.
 */
typedef enum aoclsparse_request_
{
    aoclsparse_stage_nnz_count
    = 0, /**< Only rowIndex array of the CSR matrix is computed internally. */
    aoclsparse_stage_finalize
    = 1, /**< Finalize computation. Has to be called only after csr2m call with aoclsparse_stage_nnz_count parameter. */
    aoclsparse_stage_full_computation = 2, /**< Perform the entire computation in a single step. */
} aoclsparse_request;

#ifdef __cplusplus
}
#endif

#endif // AOCLSPARSE_TYPES_H_
