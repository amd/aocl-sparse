/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.
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
 *  \brief Specifies the size in bits of integer type to be used.
 *  \details
 *  Typedef used to define the integer type this can be either
 *  32-bit or 64-bit interger type.
 *
 *  This is determined at compile-time and can be specified using the CMake
 *  option @rst :command:`-DBUILD_ILP64=On|Off` @endrst
 *  Setting to \b On will use use 64-bit integer data type.
 */
#if defined(aoclsparse_ILP64)
typedef int64_t aoclsparse_int;
#else
typedef int32_t aoclsparse_int;
#endif

/* C standard doesn't allow alignment on the typedef of the sructure
 * thus remove alignas in case of pure C
 */
#ifndef __cplusplus
#define alignas(s)
#endif

#if !defined(aoclsparse_float_complex)
/**
 * @brief
 * Default complex float type.
 * @details
 * User can redefine to accomodate custom complex float type definition.
 *
 * @note The library expects that complex numbers real and imaginary parts
 * are contiguous in memory.
 */
typedef struct alignas(2 * sizeof(float))
{
    float real; ///< Real part
    float imag; ///< Imaginary part
} aoclsparse_float_complex;
#endif

#if !defined(aoclsparse_double_complex)
/**
 * @brief
 * Default complex double type.
 * @details
 * User can redefine to accomodate custom complex double type definition.
 *
 * @note The library expects that complex numbers real and imaginary parts
 * are contiguous in memory.
 */
typedef struct alignas(2 * sizeof(double)) aoclsparse_double_complex_
{
    double real; ///< Real part
    double imag; ///< Imaginary part
} aoclsparse_double_complex;
#endif

#ifndef __cplusplus
#undef alignas
#endif

/*! \ingroup types_module
 *  \brief Matrix object descriptor.
 *
 *  \details
 *  This structure holds properties describing a matrix and how to access its data.
 *  It must be initialized using \ref aoclsparse_create_mat_descr and the returned
 *  descriptor object is passed to all subsequent library calls that involve the matrix.
 *  It is destroyed by using \ref aoclsparse_destroy_mat_descr.
 */
typedef struct _aoclsparse_mat_descr   *aoclsparse_mat_descr;
typedef struct _aoclsparse_csr         *aoclsparse_csr;
typedef struct _aoclsparse_ell         *aoclsparse_ell;
typedef struct _aoclsparse_ell_csr_hyb *aoclsparse_ell_csr_hyb;

/*! \ingroup types_module
 *  \brief Matrix object.
 *
 *  \details
 *  This structure holds the matrix data.
 *  It is initialized using e.g. \ref aoclsparse_create_scsr
 *  (or other variants, see table bellow). The returned
 *  matrix object needs be passed to all subsequent library calls that
 *  involve the matrix.
 *  It should be destroyed at the end using \ref aoclsparse_destroy.
 *
 *  @rst
 *  .. csv-table:: Initialization of matrix objects.
 *     :header: "Storage", :ref:`Precision<NamingConvention>` :code:`P`, "Initialization function"
 *     :escape: !
 *     :align: left
 *     :widths: auto
 *
 *      :ref:`Compressed Storage Rows (CSR)<storage_csr>`,    s!, d!, c!, z, :code:`aoclsparse_create_Pcsr`
 *      :ref:`Compressed Storage Columns (CSC)<storage_csc>`, s!, d!, c!, z, :code:`aoclsparse_create_Pcsc`
 *      :ref:`Coordinate storage (COO)<storage_coo>`,         s!, d!, c!, z, :code:`aoclsparse_create_Pcoo`
 *  @endrst
 */
typedef struct _aoclsparse_matrix *aoclsparse_matrix;

/** \deprecated */
typedef struct _aoclsparse_matrix *aoclsparse_mat_csr;
#define aoclsparse_destroy_mat_csr aoclsparse_destroy;
#ifdef __cplusplus
extern "C" {
#endif
/*! \ingroup types_module
 *  Indicate the operation type performed on a matrix.
 */
typedef enum aoclsparse_operation_
{
    aoclsparse_operation_none                = 111, /**< No operation is performed on the matrix. */
    aoclsparse_operation_transpose           = 112, /**< Operate with transpose. */
    aoclsparse_operation_conjugate_transpose = 113 /**< Operate with conjugate transpose. */
} aoclsparse_operation;

/*! \ingroup types_module
 *  \brief Specify the matrix index base.
 *
 *  \details
 *  Indicate the base used on the matrix indices, either 0-base (C, C++) or 1-base (Fortran).
 *  The base is set using aoclsparse_set_mat_index_base.
 *  The current of a matrix object can be obtained by calling \ref aoclsparse_get_mat_index_base.
 *
 *  \note The base-indexing information is stored in two distinc locations: the matrix object
 *  \ref aoclsparse_matrix and the matrix object descriptior \ref aoclsparse_mat_descr, these \b must
 *  coincide, either be both zero or both one. Any function accepting both objects will fail if these
 *  do not match.
 */
typedef enum aoclsparse_index_base_
{
    aoclsparse_index_base_zero = 0, /**< zero based indexing, C/C++ indexing.*/
    aoclsparse_index_base_one  = 1 /**< one based indexing, Fortran indexing. */
} aoclsparse_index_base;

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 *  \details
 *  Specifies the type of a matrix. A matrix object descriptor
 *  describes how to interpret the type of the matrix. The data in the matrix object need not
 *  match the type in the matrix object descriptor.
 *  It can be set using \ref aoclsparse_set_mat_type and retrieved using \ref aoclsparse_get_mat_type.
 */
typedef enum aoclsparse_matrix_type_
{
    // clang-format off
    aoclsparse_matrix_type_general = 0,   ///< general matrix, no special pattern.
    aoclsparse_matrix_type_symmetric = 1, ///< symmetric matrix, \f$ A=A^T\f$. It stores only a
                                          ///< single triangle specified using \ref aoclsparse_fill_mode.
    aoclsparse_matrix_type_hermitian = 2, ///< hermitian matrix, \f$ A=A^H\f$. Same storage comment
                                          ///< as for the symmetric case.
    aoclsparse_matrix_type_triangular = 3 ///< triangular matrix, \f$ A=\text{tril}(A) \f$ or
                                          ///< \f$ A=\text{triu}(A). \f$ Here too, \ref aoclsparse_fill_mode
                                          ///< specifies which triangle is available.
    // clang-format om
} aoclsparse_matrix_type;

/*! \ingroup types_module
 *  \brief  @rst Specify the matrix :ref:`data type<NamingConvention>`. @endrst
 */
typedef enum aoclsparse_matrix_data_type_
{
    aoclsparse_dmat = 0, /**< double precision data. */
    aoclsparse_smat = 1, /**< single precision data. */
    aoclsparse_cmat = 2, /**< single precision complex data. */
    aoclsparse_zmat = 3 /**< double precision complex data. */
} aoclsparse_matrix_data_type;

/*! \ingroup types_module
 *  \brief Specify the type of Incomplete LU (ILU) factorization.
 *
 *  \details
 *  Indicates the type of factorization to perform.
  */
typedef enum aoclsparse_ilu_type_
{
    aoclsparse_ilu0 = 0, /**< Incomplete LU with zero fill-in, ILU(0). */
    aoclsparse_ilup = 1, /**< Incomplete LU with thresholding, ILU(p). */
                         /**< Not implemented in this release. */
} aoclsparse_ilu_type;

/*! \ingroup types_module
 *  \brief Specify the matrix storage format type.
 */
typedef enum aoclsparse_matrix_format_type_
{
    // clang-format off
    aoclsparse_csr_mat  = 0, ///< @rst :ref:`CSR<storage_csr>` format. @endrst
    aoclsparse_ell_mat  = 1, ///< @rst :ref:`ELLPACK<storage_ell_all>` format. @endrst
    aoclsparse_ellt_mat = 2, ///< @rst :ref:`ELLPACK<storage_ell_all>` format stored
                             ///< as transpose format. @endrst
    aoclsparse_ellt_csr_hyb_mat = 3, ///< @rst :ref:`ELLPACK<storage_ell_all>` transpose + CSR
                                     ///< hybrid format. @endrst
    aoclsparse_ell_csr_hyb_mat = 4, ///< @rst :ref:`ELLPACK<storage_ell_all>` + CSR hybrid format.
                                    ///< @endrst
    aoclsparse_dia_mat     = 5, ///< @rst :ref:`DIAG<storage_dia>` format. @endrst
    aoclsparse_csr_mat_br4 = 6, ///< @rst :ref:`Optimized CSR<storage_csr_mat_br4>` format for
                                ///< AVX2 double precision data type. @endrst
    aoclsparse_csc_mat = 7, ///< @rst :ref:`CSC<storage_csc>` format. @endrst
    aoclsparse_coo_mat = 8 ///< @rst :ref:`COO<storage_coo>` format. @endrst
    // clang-format on
} aoclsparse_matrix_format_type;

/*! \ingroup types_module
 *  \brief Indicates how to interpret the diagonal entries of a matrix.
 *
 *  \details
 *  Used to indicate how to use the diagonal elements of a matrix.
 *  The purpose of this is to optimize certain operations inside the kernels.
 *  If the diagonal elements are not stored but should be interpreted has being all
 *  ones, then this can accelerate the operation by avoiding unnecessary memory accesses.
 *  For a given \ref aoclsparse_mat_descr, the
 *  diagonal type can be set using \ref aoclsparse_set_mat_diag_type and can
 *  be retrieved by calling \ref aoclsparse_get_mat_diag_type.
 */
typedef enum aoclsparse_diag_type_
{
    // clang-format off
    aoclsparse_diag_type_non_unit = 0, ///< diagonal entries are present and arbitrary.
    aoclsparse_diag_type_unit     = 1, ///< diagonal entries are to be considered all ones.
                                       ///< Kernels will not access the diagonal elements in the matrix data.
    aoclsparse_diag_type_zero = 2      ///< ignore diagonal entries: for specifying strict lower or upper
                                       ///< triangular matrices.
    //clang-format on
} aoclsparse_diag_type;

/*! \ingroup types_module
 *  \brief Specify the matrix fill mode.
 *
 *  \details
 *  Indicates if the lower or the upper part of a triangular or symmetric matrix is stored.
 *  The fill mode can be set using \ref aoclsparse_set_mat_fill_mode, and can be retrieved
 *  by calling \ref aoclsparse_get_mat_fill_mode.
 */
typedef enum aoclsparse_fill_mode_
{
    aoclsparse_fill_mode_lower = 0, /**< lower triangular part is stored. */
    aoclsparse_fill_mode_upper = 1 /**< upper triangular part is stored. */
} aoclsparse_fill_mode;

/*! \ingroup types_module
 *  \brief Specify the memory layout (order) used to store a dense matrix.
 */
typedef enum aoclsparse_order_
{
    aoclsparse_order_row    = 0, /**< Row major, (C/C++ storage). */
    aoclsparse_order_column = 1 /**< Column major, (Fortran storage). */
} aoclsparse_order;

/*! \ingroup types_module
 *  \brief
 *  Values returned by the library API to indicate success or failure.
 *
 *  \details
 *  This table provides a brief explanation on the reason why a
 *  function call failed. It is \b strongly encouraged during the development
 *  cycle of applications or services to check the exit status of any call.
 */
typedef enum aoclsparse_status_
{
    // clang-format off
    aoclsparse_status_success             = 0,  ///< success.
    aoclsparse_status_not_implemented     = 1,  ///< functionality is not implemented.
    aoclsparse_status_invalid_pointer     = 2,  ///< invalid pointer parameter.
    aoclsparse_status_invalid_size        = 3,  ///< invalid size parameter.
    aoclsparse_status_internal_error      = 4,  ///< internal library failure.
    aoclsparse_status_invalid_value       = 5,  ///< invalid parameter value.
    aoclsparse_status_invalid_index_value = 6,  ///< invalid index value.
    aoclsparse_status_maxit               = 7,  ///< function stopped after reaching number of iteration limit.
    aoclsparse_status_user_stop           = 8,  ///< user requested termination.
    aoclsparse_status_wrong_type          = 9,  ///< function called on the wrong type (double/float).
    aoclsparse_status_memory_error        = 10, ///< memory allocation failure.
    aoclsparse_status_numerical_error     = 11, ///< numerical error, e.g., matrix is not positive
                                                ///< definite, divide-by-zero error
    aoclsparse_status_invalid_operation   = 12  ///< cannot proceed with the request at this point.
    // clang-format on
} aoclsparse_status;

/*! \ingroup types_module
 *  \brief Request stages for API that perform sparse matrix products.
 *
 *  \details
 *  @rst
 *  This list  describes the possible request types used by matrix product
 *  kernels such as :cpp:func:`aoclsparse_csr2m<aoclsparse_dcsr2m>`.
 *  @endrst
 */
typedef enum aoclsparse_request_
{
    // clang-format off
    aoclsparse_stage_nnz_count = 0, ///< Perform only first stage of analysis and computation.
                                    ///< No result is returned but it is useful when optimizing
                                    ///< for multiple calls.
    aoclsparse_stage_finalize = 1,  ///< Perform computation. After this stage the product result
                                    ///< is returned. Needs to follow after a call with
                                    ///< \ref aoclsparse_stage_nnz_count request.
    aoclsparse_stage_full_computation = 2, ///< Indicates to perform the entire computation in a
                                           ///< single call.
    // clang-format on
} aoclsparse_request;

/*! \ingroup types_module
 *  \brief List of successive over-relaxation types.
 *
 *  \details
 *  This is a list of supported SOR types that are supported by \ref aoclsparse_sorv() function.
 */
typedef enum aoclsparse_sor_type_
{
    aoclsparse_sor_forward   = 0, /**< Forward sweep. */
    aoclsparse_sor_backward  = 1, /**< Backward sweep. */
    aoclsparse_sor_symmetric = 2 /**< Symmetric preconditioner. */
} aoclsparse_sor_type;

/*! \ingroup types_module
 *  \brief List of memory utilization policy.
 *
 *  \details
 *  This is a list of supported \ref aoclsparse_memory_usage() types that are used by optimization routine.
 */
typedef enum aoclsparse_memory_usage_
{
    aoclsparse_memory_usage_minimal = 0, /**< Allocate memory only for auxiliary structures.*/
    aoclsparse_memory_usage_unrestricted
    = 1 /**< Allocate memory upto matrix size for appropriate sparse format conversion. Default value. */
} aoclsparse_memory_usage;

#ifdef __cplusplus
}
#endif

#endif // AOCLSPARSE_TYPES_H_
