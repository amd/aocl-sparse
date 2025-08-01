/* ************************************************************************
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
/*! \file
 *  \brief aoclsparse_functions.h provides AMD CPU hardware optimized level
 *  1, 2, and 3 Sparse Linear Algebra Subprograms (Sparse BLAS)
 */
#ifndef AOCLSPARSE_FUNCTIONS_H_
#define AOCLSPARSE_FUNCTIONS_H_

#include "aoclsparse_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup level1_module
 *  \brief A variant of sparse vector-vector addition between compressed sparse vector and dense vector.
 *
 *  \details
 *  \P{aoclsparse_?axpyi} adds a scalar multiple of compressed sparse vector to a dense vector.
 *
 *  Let \f$y\in R^m\f$ (or \f$C^m\f$) be a dense vector, \f$x\f$ be a compressed sparse vector and \f$I_x\f$
 *  be the nonzero indices set for \p x of length at least \p nnz described by \p indx, then
 *
 *  \f[
 *     y_{I_{x_{i}}} = a\,x_i + y_{I_{x_{i}}}, \quad i\in\{1,\ldots,{\bf\mathsf{nnz}}\}.
 *  \f]
 *
 *  \note The contents of the vectors are not checked for NaNs.
 *
 *  @param[in]
 *  nnz     The number of elements in \f$x\f$ and \p indx.
 *  @param[in]
 *  a       Scalar value.
 *  @param[in]
 *  x       Sparse vector stored in compressed form of at least \p nnz elements.
 *  @param[in]
 *  indx    Nonzero indices set, \f$I_x\f$, of \p x described by this array of length at least \p nnz.
 *          The elements in this vector are only checked for
 *          non-negativity. The caller should make sure that all indices are less than the size of \p y.
 *          Array is assumed to be in zero base.
 *  @param[inout]
 *  y       Array of at least \f$\max(I_{x_i}, i \in \{ 1,\ldots,{\bf\mathsf{nnz}}\})\f$ elements.
 *
 *  \retval     aoclsparse_status_success               The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer       At least one of the pointers \p x, \p indx, \p y is invalid.
 *  \retval     aoclsparse_status_invalid_size          Indicates that provided \p nnz is less than zero.
 *  \retval     aoclsparse_status_invalid_index_value   At least one of the indices in indx is negative.
 *
 * @rst
 *   .. collapse:: Example (tests/examples/sample_axpyi.cpp)
 *
 *      .. only:: html
 *
 *         .. literalinclude:: ../tests/examples/sample_axpyi.cpp
 *            :language: C++
 *            :linenos:
 * @endrst
 * @{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_zaxpyi(
    const aoclsparse_int nnz, const void *a, const void *x, const aoclsparse_int *indx, void *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_caxpyi(
    const aoclsparse_int nnz, const void *a, const void *x, const aoclsparse_int *indx, void *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_daxpyi(const aoclsparse_int  nnz,
                                    const double          a,
                                    const double         *x,
                                    const aoclsparse_int *indx,
                                    double               *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_saxpyi(
    const aoclsparse_int nnz, const float a, const float *x, const aoclsparse_int *indx, float *y);
/**@}*/

/*! \ingroup level1_module
 *  \brief Sparse conjugate dot product for single and double data precision complex types.
 *
 *  \details
 *  aoclsparse_cdotci() (complex float) and aoclsparse_zdotci() (complex double) compute the dot product of the conjugate of a complex vector stored
 *  in a compressed format and a complex dense vector.  Let \f$x\f$ and \f$y\f$ be respectively a sparse and dense vectors in \f$C^m\f$ with
 *  \p indx (\f$I_x\f$) the nonzero indices array of \p x of length at least \p nnz that is used to index into the entries of dense
 *  vector \f$y\f$, then these functions return
 *
 *  \f[
 *    {\bf\mathsf{dot}} = \sum_{i=0}^{{\bf\mathsf{nnz}}-1} \overline{\,x_i\,} \cdot y_{I_{x_i}}.
 *  \f]
 *
 *  \note The contents of the vectors are not checked for NaNs.
 *
 *  @param[in]
 *  nnz       The number of elements (length) of vectors \p x and \p indx.
 *  @param[in]
 *  x       Array of at least \p nnz complex elements.
 *  @param[in]
 *  indx    Nonzero indices set, \f$I_x\f$, of \p x described by this array of length at least \p nnz.
 *          Each entry must contain a valid index into \p y and
 *          be unique. The entries of \p indx are not checked for validity.
 *  @param[in]
 *  y       Array of at least \f$\max(I_{x_i}, i \in \{ 1,\ldots,{\bf\mathsf{nnz}}\})\f$ complex elements.
 * @param[out]
 *  dot     The dot product of the conjugate of \f$x\f$ and \f$y\f$ when \p nnz \f$> 0\f$. If \p nnz \f$\le 0\f$, \p dot is set to 0.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer At least one of the pointers \p x, \p indx, \p y, \p dot is invalid.
 *  \retval     aoclsparse_status_invalid_size Indicates that the provided \p nnz is not positive.
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_dotp.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_dotp.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_cdotci(
    const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, const void *y, void *dot);

DLL_PUBLIC
aoclsparse_status aoclsparse_zdotci(
    const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, const void *y, void *dot);
/**@}*/

/*! \ingroup level1_module
 *  \brief Sparse dot product for single and double data precision complex types.
 *
 *  \details
 *  aoclsparse_cdotui() (complex float) and aoclsparse_zdotui() (complex double) compute the dot product of a complex vector stored
 *  in a compressed format and a complex dense vector.  Let \f$x\f$ and \f$y\f$ be respectively a sparse and dense vectors in \f$C^m\f$ with
 *  \p indx (\f$I_x\f$) the nonzero indices array of \p x of length at least \p nnz that is used to index into the entries of dense
 *  vector \f$y\f$, then these functions return
 *
 *  \f[
 *    {\bf\mathsf{dot}} = \sum_{i=0}^{{\bf\mathsf{nnz}}-1} x_{i} \cdot y_{I_{x_i}}.
 *  \f]
 *
 *  \note The contents of the vectors are not checked for NaNs.
 *
 *  @param[in]
 *  nnz       The number of elements (length) of vectors \f$x\f$ and \f$indx\f$.
 *  @param[in]
 *  x       Array of at least \p nnz complex elements.
 *  @param[in]
 *  indx    Nonzero indices set, \f$I_x\f$, of \p x described by this array of length at least \p nnz.
 *          Each entry must contain a valid index into \p y and
 *          be unique. The entries of \p indx are not checked for validity.
 *  @param[in]
 *  y       Array of at least \f$\max(I_{x_i}, i \in \{ 1,\ldots,{\bf\mathsf{nnz}}\})\f$ complex elements.
 * @param[out]
 *  dot     The dot product of \p x and \p y when \p nnz \f$> 0\f$. If \p nnz \f$\le 0\f$, \p dot is set to 0.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer At least one of the pointers \p x, \p indx, \p y, \p dot is invalid.
 *  \retval     aoclsparse_status_invalid_size Indicates that the provided \p nnz is not positive.
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_dotp.cpp)
 *
 *      .. only:: html
 *
 *         .. literalinclude:: ../tests/examples/sample_dotp.cpp
 *            :language: C++
 *            :linenos:
 * @endrst
 * @{
 */

DLL_PUBLIC
aoclsparse_status aoclsparse_zdotui(
    const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, const void *y, void *dot);

DLL_PUBLIC
aoclsparse_status aoclsparse_cdotui(
    const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, const void *y, void *dot);
/**@}*/

/*! \ingroup level1_module
 *  \brief Sparse dot product for single and double data precision real types.
 *
 *  \details
 *  aoclsparse_sdoti() and aoclsparse_ddoti() compute the dot product of a real vector stored
 *  in a compressed format and a real dense vector.  Let \f$x\f$ and \f$y\f$ be respectively a sparse and dense vectors in \f$R^m\f$ with
 *  \p indx (\f$I_x\f$) an indices array of length at least \p nnz that is used to index into the entries of dense vector \f$y\f$,
 *  then these functions return
 *
 *  \f[
 *    {\bf\mathsf{dot}} = \sum_{i=0}^{{\bf\mathsf{nnz}}-1} x_{i} \cdot y_{I_{x_i}}.
 *  \f]
 *
 *  \note The contents of the vectors are not checked for NaNs.
 *
 *  @param[in]
 *  nnz       The number of elements to access in vectors \p x and \p indx.
 *  @param[in]
 *  x       Array of at least \p nnz elements.
 *  @param[in]
 *  indx    Nonzero indices set, \f$I_x\f$, of \p x described by this array of length at least \p nnz.
 *          Each entry must contain a valid index into \p y and
 *          be unique. The entries of \p indx are not checked for validity.
 *  @param[in]
 *  y       Array of at least \f$\max(I_{x_i}, i \in \{ 1,\ldots,{\bf\mathsf{nnz}}\})\f$ elements.
 *
 *  \retval  dot Value of the dot product if \p nnz is positive, otherwise returns 0.
 * @{
 */
DLL_PUBLIC
double aoclsparse_ddoti(const aoclsparse_int  nnz,
                        const double         *x,
                        const aoclsparse_int *indx,
                        const double         *y);

DLL_PUBLIC
float aoclsparse_sdoti(const aoclsparse_int  nnz,
                       const float          *x,
                       const aoclsparse_int *indx,
                       const float          *y);
/**@}*/

/*! \ingroup level1_module
 *  \brief Sparse scatter for single and double precision real and complex types.
 *
 *  \details
 *
 *  \P{aoclsparse_?sctr} scatter the elements of a compressed sparse vector into a dense vector.
 *
 *  Let \f$y\in R^m\f$ (or \f$C^m\f$) be a dense vector, and \f$x\f$ be a compressed sparse vector with \f$I_x\f$
 *  be its nonzero indices set of length at least \p nnz and described by the array \p indx, then
 *
 *  \f[
 *     y_{I_{x_{i}}} = x_i, \quad i\in\{1,\ldots,{\bf\mathsf{nnz}}\}.
 *  \f]
 *
 *  \note The contents of the vectors are not checked for NaNs.
 *
 *  @param[in]
 *  nnz       The number of elements to use from \f$x\f$ and \f${\bf\mathsf{indx}}\f$.
 *  @param[in]
 *  x       Dense array of at least size \f${\bf\mathsf{nnz}}\f$. The first \f${\bf\mathsf{nnz}}\f$ elements are to be scattered.
 *  @param[in]
 *  indx    Nonzero index set for \f$x\f$ of size at least \f${\bf\mathsf{nnz}}\f$. The first \f${\bf\mathsf{nnz}}\f$
 *          indices are used for the scattering. The elements in this vector are
 *          only checked for non-negativity. The user should make sure that index is less than
 *          the size of \p y.
 *  @param[out]
 *  y       Array of at least \f$\max(I_{x_i}, i \in \{ 1,\ldots,{\bf\mathsf{nnz}}\})\f$ elements.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer At least one of the pointers \p x, \p indx, \p y is invalid.
 *  \retval     aoclsparse_status_invalid_size Indicates that provided \p nnz is less than zero.
 *  \retval     aoclsparse_status_invalid_index_value At least one of the indices in indx is negative.
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_sctr.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_sctr.cpp
 *          :language: C++
 *          :linenos:
 *
 * @endrst
 * @{
 */
DLL_PUBLIC
aoclsparse_status
    aoclsparse_zsctr(const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, void *y);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_csctr(const aoclsparse_int nnz, const void *x, const aoclsparse_int *indx, void *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_dsctr(const aoclsparse_int  nnz,
                                   const double         *x,
                                   const aoclsparse_int *indx,
                                   double               *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_ssctr(const aoclsparse_int  nnz,
                                   const float          *x,
                                   const aoclsparse_int *indx,
                                   float                *y);
/**@}*/

/*! \ingroup level1_module
 *  \brief Sparse scatter with stride for real/complex single and double data precisions.
 *
 *  \details
 *
 *  \P{aoclsparse_?sctrs} scatters the elements of a compressed sparse vector into a dense vector using a stride.
 *
 *  Let \f$y\f$ be a dense vector of length \f$n>0\f$, \f$x\f$ be a compressed sparse vector with \p nnz > 0 nonzeros, and
 *  \p stride be a striding distance, then
 *  \f[ y_{{\bf\mathsf{stride}} \times i} = x_i,\quad i\in\{1,\ldots,{\bf\mathsf{nnz}}\}.\f]
 *
 *  \note Contents of the vector \p x are accessed but not checked.
 *
 *  @param[in]
 *  nnz       Number of nonzero elements to access in \f$x\f$.
 *  @param[in]
 *  x       Array of at least \p nnz elements. The first \p nnz elements are to be scattered into \p y.
 *  @param[in]
 *  stride     (Positive) striding distance used to store elements in vector \p y.
 *  @param[out]
 *  y       Array of size at least \p stride \f$\times\f$ \p nnz.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer At least one of the pointers \p x, \p y is invalid.
 *  \retval     aoclsparse_status_invalid_size Indicates that one or more of the values provided in \p nnz or \p stride is not positive.
 *
 * @{
 */
DLL_PUBLIC
aoclsparse_status
    aoclsparse_zsctrs(const aoclsparse_int nnz, const void *x, aoclsparse_int stride, void *y);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_csctrs(const aoclsparse_int nnz, const void *x, aoclsparse_int stride, void *y);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_dsctrs(const aoclsparse_int nnz, const double *x, aoclsparse_int stride, double *y);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_ssctrs(const aoclsparse_int nnz, const float *x, aoclsparse_int stride, float *y);
/**@}*/

/*! \ingroup level1_module
 *  \brief Apply Givens rotation to single or double precision real vectors.
 *
 *  \details
 *  aoclsparse_sroti() and aoclsparse_droti() apply the Givens rotation
 *  on elements of two real vectors.
 *
 *  Let \f$y\in R^m\f$ be a vector in full storage form, \f$x\f$ be a vector in a compressed form and \f$I_x\f$
 *  its nonzero indices set of length at least \p nnz described by the array \p indx, then
 *
 *  \f[
 *     x_i = {\bf\mathsf{c}} * x_i + {\bf\mathsf{s}} * y_{I_{x_{i}}},
 *  \f]
 *  \f[
 *     y_{I_{x_{i}}} = {\bf\mathsf{c}} * y_{I_{x_{i}}} - {\bf\mathsf{s}} * x_i,
 *  \f]
 *
 *  for \f$i\in 1, \ldots, {\bf\mathsf{nnz}}\f$. The elements \p c, \p s are scalars.
 *
 *  \note The contents of the vectors are not checked for NaNs.
 *
 *  @param[in]
 *  nnz       The number of elements to use from \f$x\f$ and \f${\bf\mathsf{indx}}\f$.
 *  @param[in,out]
 *  x       Array \f$x\f$ of at least \f${\bf\mathsf{nnz}}\f$ elements in compressed form. The elements of the array are updated
 *          after applying the Givens rotation.
 *  @param[in]
 *  indx    Nonzero index set of \f$x\f$, \f$I_x\f$, with at least \f${\bf\mathsf{nnz}}\f$ elements. The first \f${\bf\mathsf{nnz}}\f$ elements are used to
 *          apply the Givens rotation. The elements in this vector are
 *          only checked for non-negativity. The caller should make sure that each entry is less than
 *          the size of \p y and are all distinct.
 *  @param[in,out]
 *  y       Dense array of at least \f$\max(I_{x_i}, \text{ for } i \in \{ 1,\ldots, {\bf\mathsf{nnz}}\})\f$  elements in full storage form.
 *          The elements of the array are updated after applying the Givens rotation.
 *  @param[in]
 *  c       A scalar.
 *  @param[in]
 *  s       A scalar.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer At least one of the pointers \p x, \p indx, \p y is invalid.
 *  \retval     aoclsparse_status_invalid_size Indicates that provided \p nnz is less than zero.
 *  \retval     aoclsparse_status_invalid_index_value At least one of the indices in indx is negative. With this error,
 *              the values of vectors x and y are undefined.
 * @rst
 * .. collapse:: Example (tests/examples/sample_roti.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_roti.cpp
 *          :language: C++
 *          :linenos:
 *
 * @endrst
 * @{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_droti(const aoclsparse_int  nnz,
                                   double               *x,
                                   const aoclsparse_int *indx,
                                   double               *y,
                                   const double          c,
                                   const double          s);

DLL_PUBLIC
aoclsparse_status aoclsparse_sroti(const aoclsparse_int  nnz,
                                   float                *x,
                                   const aoclsparse_int *indx,
                                   float                *y,
                                   const float           c,
                                   const float           s);
/**@}*/

/*! \ingroup level1_module
 *  \brief Gather elements from a dense vector and store them into a sparse vector.
 *
 *  \details
 *  The \P{aoclsparse_?gthr} is a group of functions that gather the elements
 *  indexed in \p indx from the dense vector \p y into the sparse vector \p x.
 *
 *  Let \f$y\in R^m\f$ (or \f$C^m\f$) be a dense vector, \f$x\f$ be a sparse vector
 *  from the same space and \f$I_x\f$ be a set of indices of size \f$0<\f$ \p nnz
 *  \f$\le m\f$ described by \p indx, then
 *  \f[
 *     x_i = y_{I_{x_i}}, i\in\{1,\ldots,{\bf\mathsf{nnz}}\}.
 *  \f]
 *
 *  For double precision complex vectors use aoclsparse_zgthr() and for single
 *  precision complex vectors use aoclsparse_cgthr().
 *
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *              If \p nnz is zero, then none of the entries of vectors
 *              \p x, \p y, and \p indx are touched.
 *  @param[in]
 *  y           pointer to dense vector \f$y\f$ of size at least \f$m\f$.
 *  @param[out]
 *  x           pointer to sparse vector \f$x\f$ with at least \p nnz
 *              non-zero elements.
 *  @param[in]
 *  indx        index vector of size \p nnz, containing the indices of
 *              the non-zero values of \f$x\f$. Indices should range from 0
 *              to \f$m-1\f$, need not be ordered. The elements in this vector
 *              are only checked for non-negativity. The user should make
 *              sure that no index is out-of-bound and that it does not contains any
 *              duplicates.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully
 *  \retval     aoclsparse_status_invalid_size \p nnz parameter value is negative
 *  \retval     aoclsparse_status_invalid_pointer at least one of the pointers \p y,
 *              \p x or \p indx is invalid
 *  \retval     aoclsparse_status_invalid_index_value at least one of the indices
 *              in \p indx is negative
 *  \note
 *  These functions assume that the indices stored in \p indx are less than \f$m\f$ without
 *  duplicate elements, and
 *  that \p x and \p indx are pointers to vectors of size at least \p nnz.
 *
 * @rst
 * .. collapse:: Example - Complex space (tests/examples/sample_zgthr.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_zgthr.cpp
 *          :language: C++
 *          :linenos:
 *
 * @endrst
 * @{
 */

DLL_PUBLIC
aoclsparse_status
    aoclsparse_zgthr(aoclsparse_int nnz, const void *y, void *x, const aoclsparse_int *indx);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_cgthr(aoclsparse_int nnz, const void *y, void *x, const aoclsparse_int *indx);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_dgthr(aoclsparse_int nnz, const double *y, double *x, const aoclsparse_int *indx);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_sgthr(aoclsparse_int nnz, const float *y, float *x, const aoclsparse_int *indx);
/**@}*/

/*! \ingroup level1_module
 *  \brief Gather and zero out elements from a dense vector and store them into a
 *  sparse vector.
 *
 *  \details
 *  The \P{aoclsparse_?gthrz} is a group of functions that gather the elements
 *
 *  indexed in \p indx from the dense vector \p y into the sparse vector \p x.
 *  The gathered elements in \f$y\f$ are replaced by zero.
 *
 *  Let \f$y\in R^m\f$ (or \f$C^m\f$) be a dense vector, \f$x\f$ be a sparse vector
 *  from the same space and \f$I_x\f$ be a set of indices of size \f$0<\f$ \p nnz
 *  \f$\le m\f$ described by \p indx, then
 *  \f[
 *     x_i = y_{I_{x_i}}, i\in\{1,\ldots,{\bf\mathsf{nnz}}\},
 * \text{ and after the assignment, }
 *     y_{I_{x_i}}=0, i\in\{1,\ldots,{\bf\mathsf{nnz}}\}.
 *  \f]
 *
 *  For double precision complex vectors use aoclsparse_zgthrz() and for single
 *  precision complex vectors use aoclsparse_cgthrz().
 *
 *  @param[in]
 *  nnz         number of non-zero entries of \f$x\f$.
 *              If \p nnz is zero, then none of the entries of vectors
 *              \p x, \p y, and \p indx are touched.
 *  @param[in]
 *  y           pointer to dense vector \f$y\f$ of size at least \f$m\f$.
 *  @param[out]
 *  x           pointer to sparse vector \f$x\f$ with at least \p nnz
 *              non-zero elements.
 *  @param[in]
 *  indx        index vector of size \p nnz, containing the indices of
 *              the non-zero values of \f$x\f$. Indices should range from 0
 *              to \f$m-1\f$, need not be ordered. The elements in this vector
 *              are only checked for non-negativity. The user should make
 *              sure that no index is out-of-bound and that it does not contains any
 *              duplicates.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully
 *  \retval     aoclsparse_status_invalid_size \p nnz parameter value is negative
 *  \retval     aoclsparse_status_invalid_pointer at least one of the pointers \p y,
 *              \p x or \p indx is invalid
 *  \retval     aoclsparse_status_invalid_index_value at least one of the indices
 *              in \p indx is negative
 *  \note
 *  These functions assume that the indices stored in \p indx are less than \f$m\f$ without
 *  duplicate elements, and
 *  that \p x and \p indx are pointers to vectors of size at least \p nnz.
 * @{
 */
DLL_PUBLIC
aoclsparse_status
    aoclsparse_zgthrz(aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_cgthrz(aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_dgthrz(aoclsparse_int nnz, double *y, double *x, const aoclsparse_int *indx);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_sgthrz(aoclsparse_int nnz, float *y, float *x, const aoclsparse_int *indx);
/**@}*/

/*! \ingroup level1_module
 *  \brief Gather elements from a dense vector using a stride and store them into a sparse vector.
 *
 *  \details
 *
 *  The \P{aoclsparse_?gthrs} is a group of functions that gather the elements
 *  from the dense vector \p y using a fixed stride distance and copies them into the
 *  sparse vector \p x.
 *
 *  Let \f$y\in R^m\f$ (or \f$C^m\f$) be a dense vector, \f$x\f$ be a sparse vector
 *  from the same space and \p stride be a (positive) striding distance, then
 *  \f$
 *     x_i = y_{\text{stride} \times i}, \quad i\in\{1,\ldots,{\bf\mathsf{nnz}}\}.
 *  \f$
 *
 *  @param[in]
 *  nnz         Number of non-zero entries of \f$x\f$.
 *              If \p nnz is zero, then none of the entries of vectors
 *              \p x and \p y are accessed. Note that \p nnz must be such that
 *              \p stride \f$\times\f$ \p nnz must be less or equal to \f$m\f$.
 *  @param[in]
 *  y           Pointer to dense vector \f$y\f$ of size at least \f$m\f$.
 *  @param[out]
 *  x           Pointer to sparse vector \f$x\f$ with at least \p nnz
 *              non-zero elements.
 *  @param[in]
 *  stride      Striding distance used to access elements in the dense vector \p y.
 *              It must be such that \p stride \f$\times\f$ \p nnz is less or equal
 *              to \f$m\f$.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size at least one of the parameters \p nnz or \p stride has a
 *              negative value.
 *  \retval     aoclsparse_status_invalid_pointer at least one of the pointers \p y,
 *              or \p x is invalid.
 * @{
 */
DLL_PUBLIC
aoclsparse_status
    aoclsparse_zgthrs(aoclsparse_int nnz, const void *y, void *x, aoclsparse_int stride);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_cgthrs(aoclsparse_int nnz, const void *y, void *x, aoclsparse_int stride);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_dgthrs(aoclsparse_int nnz, const double *y, double *x, aoclsparse_int stride);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_sgthrs(aoclsparse_int nnz, const float *y, float *x, aoclsparse_int stride);
/**@}*/

/*! \ingroup level2_module
 *  \brief Real single and double precision sparse matrix-vector multiplication using CSR storage format
 *
 *  \details
 *  \P{aoclsparse_?csrmv} multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y = \alpha \, op(A) \, x + \beta \, y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  n           number of columns of the sparse CSR matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m +1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix. Currently, only
 *              \ref aoclsparse_matrix_type_general and
 *              \ref aoclsparse_matrix_type_symmetric is supported.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) = A\f$) or \p m elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) = A\f$) or \p n elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n or \p nnz is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p x, \p beta or \p y pointer is
 *              invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \p trans is not \ref aoclsparse_operation_none and
 *              \p trans is not \ref aoclsparse_operation_transpose.
 *              \ref aoclsparse_matrix_type is not \ref aoclsparse_matrix_type_general, or
 *              \ref aoclsparse_matrix_type is not \ref aoclsparse_matrix_type_symmetric.
 *
 * @{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_dcsrmv(aoclsparse_operation       trans,
                                    const double              *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const double              *csr_val,
                                    const aoclsparse_int      *csr_col_ind,
                                    const aoclsparse_int      *csr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const double              *x,
                                    const double              *beta,
                                    double                    *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_scsrmv(aoclsparse_operation       trans,
                                    const float               *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const float               *csr_val,
                                    const aoclsparse_int      *csr_col_ind,
                                    const aoclsparse_int      *csr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const float               *x,
                                    const float               *beta,
                                    float                     *y);
/**@}*/

/*! \ingroup level2_module
 *  \brief Real single and double precision sparse matrix vector product using ELL storage format
 *
 *  \details
 *  \P{aoclsparse_?ellmv} multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in ELL storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y = \alpha \, op(A) \, x + \beta \, y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  Currently, only \p trans = \ref aoclsparse_operation_none is supported.
 *
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  m           number of rows of the sparse ELL matrix.
 *  @param[in]
 *  n           number of columns of the sparse ELL matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the sparse ELL matrix.
 *  @param[in]
 *  descr       descriptor of the sparse ELL matrix. Both, base-zero and
 *              base-one input arrays of ELL matrix are supported
 *  @param[in]
 *  ell_val     array that contains the elements of the sparse ELL matrix. Padded
 *              elements should be zero.
 *  @param[in]
 *  ell_col_ind array that contains the column indices of the sparse ELL matrix.
 *              Padded column indices should be -1.
 *  @param[in]
 *  ell_width   number of non-zero elements per row of the sparse ELL matrix.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) = A\f$) or \p m elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) = A\f$) or \p n elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p ell_val,
 *              \p ell_col_ind, \p x, \p beta or \p y pointer is invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \p trans is not \ref aoclsparse_operation_none, or
 *              \ref aoclsparse_matrix_type is not \ref aoclsparse_matrix_type_general.
 * @{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_dellmv(aoclsparse_operation       trans,
                                    const double              *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const double              *ell_val,
                                    const aoclsparse_int      *ell_col_ind,
                                    aoclsparse_int             ell_width,
                                    const aoclsparse_mat_descr descr,
                                    const double              *x,
                                    const double              *beta,
                                    double                    *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_sellmv(aoclsparse_operation       trans,
                                    const float               *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const float               *ell_val,
                                    const aoclsparse_int      *ell_col_ind,
                                    aoclsparse_int             ell_width,
                                    const aoclsparse_mat_descr descr,
                                    const float               *x,
                                    const float               *beta,
                                    float                     *y);
/**@}*/

#ifndef DOXYGEN_SHOULD_SKIP_THIS
DLL_PUBLIC
aoclsparse_status aoclsparse_selltmv(aoclsparse_operation       trans,
                                     const float               *alpha,
                                     aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     aoclsparse_int             nnz,
                                     const float               *ell_val,
                                     const aoclsparse_int      *ell_col_ind,
                                     aoclsparse_int             ell_width,
                                     const aoclsparse_mat_descr descr,
                                     const float               *x,
                                     const float               *beta,
                                     float                     *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_delltmv(aoclsparse_operation       trans,
                                     const double              *alpha,
                                     aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     aoclsparse_int             nnz,
                                     const double              *ell_val,
                                     const aoclsparse_int      *ell_col_ind,
                                     aoclsparse_int             ell_width,
                                     const aoclsparse_mat_descr descr,
                                     const double              *x,
                                     const double              *beta,
                                     double                    *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_sellthybmv(aoclsparse_operation       trans,
                                        const float               *alpha,
                                        aoclsparse_int             m,
                                        aoclsparse_int             n,
                                        aoclsparse_int             nnz,
                                        const float               *ell_val,
                                        const aoclsparse_int      *ell_col_ind,
                                        aoclsparse_int             ell_width,
                                        const aoclsparse_int       ell_m,
                                        const float               *csr_val,
                                        const aoclsparse_int      *csr_row_ind,
                                        const aoclsparse_int      *csr_col_ind,
                                        aoclsparse_int            *row_idx_map,
                                        aoclsparse_int            *csr_row_idx_map,
                                        const aoclsparse_mat_descr descr,
                                        const float               *x,
                                        const float               *beta,
                                        float                     *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_dellthybmv(aoclsparse_operation       trans,
                                        const double              *alpha,
                                        aoclsparse_int             m,
                                        aoclsparse_int             n,
                                        aoclsparse_int             nnz,
                                        const double              *ell_val,
                                        const aoclsparse_int      *ell_col_ind,
                                        aoclsparse_int             ell_width,
                                        const aoclsparse_int       ell_m,
                                        const double              *csr_val,
                                        const aoclsparse_int      *csr_row_ind,
                                        const aoclsparse_int      *csr_col_ind,
                                        aoclsparse_int            *row_idx_map,
                                        aoclsparse_int            *csr_row_idx_map,
                                        const aoclsparse_mat_descr descr,
                                        const double              *x,
                                        const double              *beta,
                                        double                    *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_dblkcsrmv(aoclsparse_operation       trans,
                                       const double              *alpha,
                                       aoclsparse_int             m,
                                       aoclsparse_int             n,
                                       aoclsparse_int             nnz,
                                       const uint8_t             *masks,
                                       const double              *csr_val,
                                       const aoclsparse_int      *csr_col_ind,
                                       const aoclsparse_int      *csr_row_ptr,
                                       const aoclsparse_mat_descr descr,
                                       const double              *x,
                                       const double              *beta,
                                       double                    *y,
                                       aoclsparse_int             nRowsblk);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/*! \ingroup level2_module
 *  \brief Real single and double precision sparse matrix vector product using DIA storage format
 *
 *  \details
 *  \P{aoclsparse_?diamv} multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in DIA storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y = \alpha \, op(A) \, x + \beta \, y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  Currently, only \p trans = \ref aoclsparse_operation_none is supported.
 *
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  m           number of rows of the  matrix.
 *  @param[in]
 *  n           number of columns of the  matrix.
 *  @param[in]
 *  nnz         number of non-zero entries of the  matrix.
 *  @param[in]
 *  descr       descriptor of the sparse DIA matrix.
 *  @param[in]
 *  dia_val     array that contains the elements of the  matrix. Padded
 *              elements should be zero.
 *  @param[in]
 *  dia_offset  array that contains the offsets of each diagonal of the matrix.
 *
 *  @param[in]
 *  dia_num_diag  number of diagonals in the matrix.
 *  @param[in]
 *  x           array of \p n elements (\f$op(A) = A\f$) or \p m elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of \p m elements (\f$op(A) = A\f$) or \p n elements
 *              (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n or \p ell_width is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p ell_val,
 *              \p ell_col_ind, \p x, \p beta or \p y pointer is invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \p trans is not \ref aoclsparse_operation_none, or
 *              \ref aoclsparse_matrix_type is not \ref aoclsparse_matrix_type_general.
 * @{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_ddiamv(aoclsparse_operation       trans,
                                    const double              *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const double              *dia_val,
                                    const aoclsparse_int      *dia_offset,
                                    aoclsparse_int             dia_num_diag,
                                    const aoclsparse_mat_descr descr,
                                    const double              *x,
                                    const double              *beta,
                                    double                    *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_sdiamv(aoclsparse_operation       trans,
                                    const float               *alpha,
                                    aoclsparse_int             m,
                                    aoclsparse_int             n,
                                    aoclsparse_int             nnz,
                                    const float               *dia_val,
                                    const aoclsparse_int      *dia_offset,
                                    aoclsparse_int             dia_num_diag,
                                    const aoclsparse_mat_descr descr,
                                    const float               *x,
                                    const float               *beta,
                                    float                     *y);
/**@}*/

/*! \ingroup level2_module
*  \brief Real single and double precision matrix vector product using BSR storage format
*
*  \details
*  \P{aoclsparse_?bsrmv} multiplies the scalar \f$\alpha\f$ with a sparse
*  \p mb times \p bsr_dim by \p nb times \p bsr_dim
*  matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y = \alpha \, op(A) \, x + \beta \, y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  Only \p trans = \ref aoclsparse_operation_none is supported.
*
*  @param[in]
*  trans       matrix operation type.
*  @param[in]
*  mb          number of block rows of the sparse BSR matrix.
*  @param[in]
*  nb          number of block columns of the sparse BSR matrix.
*  @param[in]
*  alpha       scalar \f$\alpha\f$.
*  @param[in]
*  descr       descriptor of the sparse BSR matrix. Both, base-zero and
 *              base-one input arrays of BSR matrix are supported.
*  @param[in]
*  bsr_val     array of \p nnzb blocks of the sparse BSR matrix.
*  @param[in]
*  bsr_row_ptr array of \p mb+1 elements that point to the start of every block row of
*              the sparse BSR matrix.
*  @param[in]
*  bsr_col_ind array of \p nnz containing the block column indices of the sparse
*              BSR matrix.
*  @param[in]
*  bsr_dim     block dimension of the sparse BSR matrix.
*  @param[in]
*  x           array of \p nb times \p bsr_dim elements (\f$op(A) = A\f$) or \p mb times \p bsr_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*  @param[in]
*  beta        scalar \f$\beta\f$.
*  @param[inout]
*  y           array of \p mb times \p bsr_dim elements (\f$op(A) = A\f$) or \p nb times \p bsr_dim
*              elements (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
*
*  \retval     aoclsparse_status_success the operation completed successfully.
*  \retval     aoclsparse_status_invalid_handle the library context was not initialized.
*  \retval     aoclsparse_status_invalid_size \p mb, \p nb, \p nnzb or \p bsr_dim is
*              invalid.
*  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p bsr_val,
*              \p bsr_row_ind, \p bsr_col_ind, \p x, \p beta or \p y pointer is invalid.
*  \retval     aoclsparse_status_arch_mismatch the device is not supported.
*  \retval     aoclsparse_status_not_implemented
*              \p trans is not \ref aoclsparse_operation_none, or
*              \ref aoclsparse_matrix_type is not \ref aoclsparse_matrix_type_general.
* @{
*/
DLL_PUBLIC
aoclsparse_status aoclsparse_dbsrmv(aoclsparse_operation       trans,
                                    const double              *alpha,
                                    aoclsparse_int             mb,
                                    aoclsparse_int             nb,
                                    aoclsparse_int             bsr_dim,
                                    const double              *bsr_val,
                                    const aoclsparse_int      *bsr_col_ind,
                                    const aoclsparse_int      *bsr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const double              *x,
                                    const double              *beta,
                                    double                    *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_sbsrmv(aoclsparse_operation       trans,
                                    const float               *alpha,
                                    aoclsparse_int             mb,
                                    aoclsparse_int             nb,
                                    aoclsparse_int             bsr_dim,
                                    const float               *bsr_val,
                                    const aoclsparse_int      *bsr_col_ind,
                                    const aoclsparse_int      *bsr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const float               *x,
                                    const float               *beta,
                                    float                     *y);
/**@}*/

/*! \ingroup level2_module
 *  \brief Compute sparse matrix-vector multiplication for real/complex single and double data precisions.
 *
 *  \details
 *  The \P{aoclsparse_?mv} perform sparse matrix-vector products of the form
 *  \f[
 *    y = \alpha \, op(A) \, x + \beta \, y,
 *  \f]
 *  where, \f$x\f$ and \f$y\f$ are dense vectors, \f$\alpha\f$ and \f$\beta\f$ are scalars,
 *  and \f$A\f$ is a sparse matrix structure. The matrix operation \f$op()\f$ is defined as:
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  @param[in]
 *  op          Matrix operation, \p op can be one of \ref aoclsparse_operation_none,
 *              \ref aoclsparse_operation_conjugate_transpose, or \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  alpha       Scalar \f$\alpha\f$.
 *  @param[in]
 *  A           The sparse matrix created using e.g. aoclsparse_create_scsr() or other variant.
 *              Matrix is considered of size \f$m\f$ by \f$n\f$.
 *  @param[in]
 *  descr       Descriptor of the matrix. These functions support the following \ref aoclsparse_matrix_type types:
 *              \ref aoclsparse_matrix_type_general, \ref aoclsparse_matrix_type_triangular,
 *              \ref aoclsparse_matrix_type_symmetric, and \ref aoclsparse_matrix_type_hermitian.
 *              Both base-zero and base-one are supported, however,
 *              the index base needs to match with the one defined in matrix \p A.
 *  @param[in]
 *  x           An array of \p n elements if \f$op(A) = A\f$; or of \p m elements
 *              if \f$op(A) = A^T\f$ or \f$op(A) = A^H\f$.
 *  @param[in]
 *  beta        Scalar \f$\beta\f$.
 *  @param[inout]
 *  y           An array of \p m elements if \f$op(A) = A\f$; or of \p n elements
 *              if \f$op(A) = A^T\f$ or \f$op(A) = A^H\f$.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size The value of \p m, \p n or \p nnz is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, internal
 *              structures related to the sparse matrix \p A, \p x, \p beta or \p y has
 *              an invalid pointer.
 *  \retval     aoclsparse_status_not_implemented The requested functionality is not implemented.
 *
 * @rst
 * .. collapse:: Example - C++ (tests/examples/sample_spmv.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_spmv.cpp
 *          :language: C++
 *          :linenos:
 *
 * \
 *
 * .. collapse:: Example - C (tests/examples/sample_spmv_c.c)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_spmv_c.c
 *          :language: C
 *          :linenos:
 *
 * @endrst
 * @{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_zmv(aoclsparse_operation             op,
                                 const aoclsparse_double_complex *alpha,
                                 aoclsparse_matrix                A,
                                 const aoclsparse_mat_descr       descr,
                                 const aoclsparse_double_complex *x,
                                 const aoclsparse_double_complex *beta,
                                 aoclsparse_double_complex       *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_cmv(aoclsparse_operation            op,
                                 const aoclsparse_float_complex *alpha,
                                 aoclsparse_matrix               A,
                                 const aoclsparse_mat_descr      descr,
                                 const aoclsparse_float_complex *x,
                                 const aoclsparse_float_complex *beta,
                                 aoclsparse_float_complex       *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_dmv(aoclsparse_operation       op,
                                 const double              *alpha,
                                 aoclsparse_matrix          A,
                                 const aoclsparse_mat_descr descr,
                                 const double              *x,
                                 const double              *beta,
                                 double                    *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_smv(aoclsparse_operation       op,
                                 const float               *alpha,
                                 aoclsparse_matrix          A,
                                 const aoclsparse_mat_descr descr,
                                 const float               *x,
                                 const float               *beta,
                                 float                     *y);
/**@}*/

/*! \ingroup level2_module
 *  \deprecated
 *  This API is superseded by aoclsparse_strsv() and aoclsparse_dtrsv().
 *  \brief Sparse triangular solve using CSR storage format for single and double
 *      data precisions.
 *
 *
 *  \P{aoclsparse_?csrsv} solves a sparse triangular linear system of a sparse
 *  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
 *  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
 *  \f[
 *    op(A) \, y = \alpha \, x,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  Only \p trans = \ref aoclsparse_operation_none is supported.
 *
 *  \note
 *  The input matrix has to be sparse upper or lower triangular matrix
 *  with unit or non-unit main diagonal. Matrix has to be sorted.
 *  No diagonal element can be omitted from a sparse storage
 *  if the solver is called with the non-unit indicator.
 *
 *  @param[in]
 *  trans       matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  m           number of rows of the sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements of the sparse CSR matrix.
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start
 *              of every row of the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the sparse
 *              CSR matrix.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix.
 *  @param[in]
 *  x           array of \p m elements, holding the right-hand side.
 *  @param[out]
 *  y           array of \p m elements, holding the solution.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m is invalid.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p alpha, \p csr_val,
 *              \p csr_row_ptr, \p csr_col_ind, \p x or \p y pointer is invalid.
 *  \retval     aoclsparse_status_internal_error an internal error occurred.
 *  \retval     aoclsparse_status_not_implemented
 *              \p trans = \ref aoclsparse_operation_conjugate_transpose or
 *              \p trans = \ref aoclsparse_operation_transpose or
 *              \ref aoclsparse_matrix_type is not \ref aoclsparse_matrix_type_general.
 *
 * \{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_dcsrsv(aoclsparse_operation       trans,
                                    const double              *alpha,
                                    aoclsparse_int             m,
                                    const double              *csr_val,
                                    const aoclsparse_int      *csr_col_ind,
                                    const aoclsparse_int      *csr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const double              *x,
                                    double                    *y);

DLL_PUBLIC
aoclsparse_status aoclsparse_scsrsv(aoclsparse_operation       trans,
                                    const float               *alpha,
                                    aoclsparse_int             m,
                                    const float               *csr_val,
                                    const aoclsparse_int      *csr_col_ind,
                                    const aoclsparse_int      *csr_row_ptr,
                                    const aoclsparse_mat_descr descr,
                                    const float               *x,
                                    float                     *y);
/**@}*/

/** \ingroup level2_module
 *  \brief Sparse triangular solver for real/complex single and double data precisions.
 *
 *  \details
 *  The function aoclsparse_strsv() and variants solve sparse lower (or upper) triangular
 *  linear system of equations. The system is defined by the sparse
 *  \f$m \times m\f$ matrix \f$A\f$, the dense solution \f$m\f$-vector
 *  \f$x\f$, and the right-hand side dense \f$m\f$-vector \f$b\f$. Vector \f$b\f$ is
 *  multiplied by \f$\alpha\f$. The solution \f$x\f$ is estimated by solving
 *  \f[
 *    op(L) \cdot x = \alpha \cdot b, \quad \text{ or } \quad
 *    op(U) \cdot x = \alpha \cdot b,
 *  \f]
 *  where
 *  \f$L = \text{tril}(A)\f$ is the lower triangle of matrix \f$A\f$, similarly,
 *  \f$U = \text{triu}(A)\f$ is the upper triangle of matrix \f$A\f$. The operator
 *  \f$op()\f$ is regarded as the matrix linear operation,
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  <b>Notes</b>
 *
 * 1. This routine supports sparse matrices in CSR and CSC formats.
 *
 * 2. If the matrix descriptor \p descr specifies that the matrix \f$A\f$ is to be regarded as
 *    having a unitary diagonal, then the main diagonal entries of matrix \f$A\f$ are not accessed and
 *    are all considered to be unitary.
 *
 * 3. The input matrix need not be (upper or lower) triangular matrix, in the \p descr, the \p fill_mode
 *    entity specifies which triangle to consider, namely, if \p fill_mode = \ref aoclsparse_fill_mode_lower,
 *    then
 *    \f[
 *       op(L) \cdot x = \alpha \cdot b,
 *    \f]
 *
 *    otherwise, if \p fill_mode = \ref aoclsparse_fill_mode_upper, then
 *
 *    \f[
 *       op(U) \cdot x = \alpha \cdot b,
 *    \f]
 *
 *    is solved.
 *
 * 4. To increase performance and if the matrix \f$A\f$ is to be used more than once to solve for different right-hand
 *    sides \p b, then it is encouraged to provide hints using  aoclsparse_set_sv_hint() and aoclsparse_optimize(),
 *    otherwise, the optimization for the matrix will be done by the solver on entry.
 *
 * 5. There is a \p kid (Kernel ID) variation of TRSV, namely with a suffix of \p _kid,
 *    aoclsparse_strsv_kid() (and variations) where it is possible to specify the TRSV kernel to use (if possible).
 *
 * @rst
 *
 * .. collapse:: Example - Real space (tests/examples/sample_dtrsv.cpp)
 *
 *      .. only:: html
 *
 *         .. literalinclude:: ../tests/examples/sample_dtrsv.cpp
 *            :language: C++
 *            :linenos:
 *
 * \
 *
 * .. collapse:: Example - Complex space (tests/examples/sample_ztrsv.cpp)
 *
 *      .. only:: html
 *
 *         .. literalinclude:: ../tests/examples/sample_ztrsv.cpp
 *            :language: C++
 *            :linenos:
 *
 * @endrst
 *
 *  @param[in]
 *  trans       matrix operation type, either \ref aoclsparse_operation_none, \ref aoclsparse_operation_transpose,
 *              or \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$, used to multiply right-hand side vector \f$b\f$.
 *  @param[inout]
 *  A           matrix containing data used to represent the \f$m \times m\f$ triangular linear system to solve.
 *  @param[in]
 *  descr       matrix descriptor. Supported matrix types are \ref aoclsparse_matrix_type_symmetric and
 *              \ref aoclsparse_matrix_type_triangular.
 *  @param[in]
 *  b           array of \p m elements, storing the right-hand side.
 *  @param[out]
 *  x           array of \p m elements, storing the solution if solver returns \ref aoclsparse_status_success.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully and  \f$x\f$ contains the solution
 *              to the linear system of equations.
 *  \retval     aoclsparse_status_invalid_size matrix \f$A\f$ or \f$op(A)\f$ is invalid.
 *  \retval     aoclsparse_status_invalid_pointer One or more of \p A,  \p descr, \p x, \p b are invalid pointers.
 *  \retval     aoclsparse_status_internal_error an internal error occurred.
 *  \retval     aoclsparse_status_not_implemented the requested operation is not yet implemented.
 *  \retval     other possible failure values from a call to \ref aoclsparse_optimize.
 *
 * \{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_ztrsv(aoclsparse_operation             trans,
                                   const aoclsparse_double_complex  alpha,
                                   aoclsparse_matrix                A,
                                   const aoclsparse_mat_descr       descr,
                                   const aoclsparse_double_complex *b,
                                   aoclsparse_double_complex       *x);

DLL_PUBLIC
aoclsparse_status aoclsparse_ctrsv(aoclsparse_operation            trans,
                                   const aoclsparse_float_complex  alpha,
                                   aoclsparse_matrix               A,
                                   const aoclsparse_mat_descr      descr,
                                   const aoclsparse_float_complex *b,
                                   aoclsparse_float_complex       *x);

DLL_PUBLIC
aoclsparse_status aoclsparse_dtrsv(aoclsparse_operation       trans,
                                   const double               alpha,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   const double              *b,
                                   double                    *x);

DLL_PUBLIC
aoclsparse_status aoclsparse_strsv(aoclsparse_operation       trans,
                                   const float                alpha,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   const float               *b,
                                   float                     *x);
/**\}*/

/** \ingroup level2_module
 *  \brief Sparse triangular solver for real/complex single and double data precisions (kernel flag variation).
 *
 *  \details
 * @rst For full details refer to :cpp:func:`aoclsparse_?trsv()<aoclsparse_strsv>`.
 *
 * This variation of TRSV, namely with a suffix of :code:`_kid`, allows to choose which
 * TRSV kernel to use (if possible). Currently the possible choices are:
 *
 * :code:`kid=0`
 *     Reference implementation (No explicit AVX instructions).
 *
 * :code:`kid=1`
 *     Alias to :code:`kid=2` (Kernel Template AVX 256-bit implementation)
 *
 * :code:`kid=2`
 *     Kernel Template version using :code:`AVX2` extensions.
 *
 * :code:`kid=3`
 *     Kernel Template version using :code:`AVX512F+` CPU extensions.
 *
 * Any other Kernel ID value will default to :code:`kid` = 0.
 * @endrst
 *
 *  @param[in]
 *  trans       matrix operation type, either \ref aoclsparse_operation_none, \ref aoclsparse_operation_transpose,
 *              or \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$, used to multiply right-hand side vector \f$b\f$.
 *  @param[inout]
 *  A           matrix containing data used to represent the \f$m \times m\f$ triangular linear system to solve.
 *  @param[in]
 *  descr       matrix descriptor. Supported matrix types are \ref aoclsparse_matrix_type_symmetric and
 *              \ref aoclsparse_matrix_type_triangular.
 *  @param[in]
 *  b           array of \p m elements, storing the right-hand side.
 *  @param[out]
 *  x           array of \p m elements, storing the solution if solver returns \ref aoclsparse_status_success.
 *  @param[in]
 *  kid         Kernel ID, hints a request on which TRSV kernel to use.
 * \{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_ztrsv_kid(aoclsparse_operation             trans,
                                       const aoclsparse_double_complex  alpha,
                                       aoclsparse_matrix                A,
                                       const aoclsparse_mat_descr       descr,
                                       const aoclsparse_double_complex *b,
                                       aoclsparse_double_complex       *x,
                                       aoclsparse_int                   kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_ctrsv_kid(aoclsparse_operation            trans,
                                       const aoclsparse_float_complex  alpha,
                                       aoclsparse_matrix               A,
                                       const aoclsparse_mat_descr      descr,
                                       const aoclsparse_float_complex *b,
                                       aoclsparse_float_complex       *x,
                                       aoclsparse_int                  kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_dtrsv_kid(aoclsparse_operation       trans,
                                       const double               alpha,
                                       aoclsparse_matrix          A,
                                       const aoclsparse_mat_descr descr,
                                       const double              *b,
                                       double                    *x,
                                       aoclsparse_int             kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_strsv_kid(aoclsparse_operation       trans,
                                       const float                alpha,
                                       aoclsparse_matrix          A,
                                       const aoclsparse_mat_descr descr,
                                       const float               *b,
                                       float                     *x,
                                       aoclsparse_int             kid);
/**\}*/

/** \ingroup level2_module
 * \brief This is a variation of TRSV, namely with a suffix of \p _strided, allows to set
 * the stride for the dense vectors \p b and \p x.
 * \details
 * @rst For full details refer to :cpp:func:`aoclsparse_?trsv()<aoclsparse_strsv>`.@endrst
 *
 *  @param[in]
 *  trans       matrix operation type, either \ref aoclsparse_operation_none, \ref aoclsparse_operation_transpose,
 *              or \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$, used to multiply right-hand side vector \f$b\f$.
 *  @param[inout]
 *  A           matrix containing data used to represent the \f$m \times m\f$ triangular linear system to solve.
 *  @param[in]
 *  descr       matrix descriptor. Supported matrix types are \ref aoclsparse_matrix_type_symmetric and
 *              \ref aoclsparse_matrix_type_triangular.
 *  @param[in]
 *  b           array of \p m elements, storing the right-hand side.
 *  @param[in]
 *  incb      a positive integer holding the stride value for \f$b\f$ vector.
 *  @param[out]
 *  x           array of \p m elements, storing the solution if solver returns \ref aoclsparse_status_success.
 *  @param[in]
 *  incx      a positive integer holding the stride value for \f$x\f$ vector.
 * \{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_strsv_strided(aoclsparse_operation       trans,
                                           const float                alpha,
                                           aoclsparse_matrix          A,
                                           const aoclsparse_mat_descr descr,
                                           const float               *b,
                                           const aoclsparse_int       incb,
                                           float                     *x,
                                           const aoclsparse_int       incx);

DLL_PUBLIC
aoclsparse_status aoclsparse_dtrsv_strided(aoclsparse_operation       trans,
                                           const double               alpha,
                                           aoclsparse_matrix          A,
                                           const aoclsparse_mat_descr descr,
                                           const double              *b,
                                           const aoclsparse_int       incb,
                                           double                    *x,
                                           const aoclsparse_int       incx);

DLL_PUBLIC
aoclsparse_status aoclsparse_ctrsv_strided(aoclsparse_operation            trans,
                                           const aoclsparse_float_complex  alpha,
                                           aoclsparse_matrix               A,
                                           const aoclsparse_mat_descr      descr,
                                           const aoclsparse_float_complex *b,
                                           const aoclsparse_int            incb,
                                           aoclsparse_float_complex       *x,
                                           const aoclsparse_int            incx);

DLL_PUBLIC
aoclsparse_status aoclsparse_ztrsv_strided(aoclsparse_operation             trans,
                                           const aoclsparse_double_complex  alpha,
                                           aoclsparse_matrix                A,
                                           const aoclsparse_mat_descr       descr,
                                           const aoclsparse_double_complex *b,
                                           const aoclsparse_int             incb,
                                           aoclsparse_double_complex       *x,
                                           const aoclsparse_int             incx);
/**@}*/

/*! \ingroup level2_module
 *  \brief Performs sparse matrix-vector multiplication followed by vector-vector multiplication
 *
 *  \details
 *  \P{aoclsparse_?dotmv} multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
 *  matrix, defined in a sparse storage format, and the dense vector \f$x\f$ and adds the
 *  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
 *  such that
 *  \f[
 *    y := \alpha \, op(A) \, x + \beta \, y,
 *  \quad \text{ with } \quad
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 * followed by dot product of dense vectors \f$x\f$ and \f$y\f$ such that
 * \f[
 *   {\bf\mathsf{d}} = \left\{
 *    \begin{array}{ll}
 *        \sum_{i=0}^{\min(m,n)-1} x_{i} \; y_{i}, & \text{real case} \\
 *        \sum_{i=0}^{\min(m,n)-1} \overline{x_i} \; y_{i}, & \text{complex case}
 *    \end{array}
 *    \right.
 * \f]
 *
 *  @param[in]
 *  op          matrix operation type.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  A           the sparse \f$m \times n\f$ matrix structure that is created using
 *              aoclsparse_create_scsr() or other variation.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix. Both base-zero and base-one
 *              are supported, however, the index base needs to match the one used
 *              when aoclsparse_matrix was created.
 *  @param[in]
 *  x           array of atleast \p n elements if \f$op(A) = A\f$ or at least \p m elements
 *              if \f$op(A) = A^T\f$ or \f$A^H\f$.
 *  @param[in]
 *  beta        scalar \f$\beta\f$.
 *  @param[inout]
 *  y           array of atleast \p m elements if \f$op(A) = A\f$ or at least \p n elements
 *              if \f$op(A) = A^T\f$ or \f$A^H\f$.
 *  @param[out]
 *  d           dot product of y and x.
 *
 *  \retval     aoclsparse_status_success         the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size    \p m, \p n or \p nnz is invalid.
 *  \retval     aoclsparse_status_invalid_value   (base index is neither \ref aoclsparse_index_base_zero nor
 *                  \ref aoclsparse_index_base_one, or matrix base index and descr base index values do not match.
 *  \retval     aoclsparse_status_invalid_pointer \p descr, \p internal structures
 *                  related to the sparse matrix  \p A, \p x, \p y or \p d are invalid pointer.
 *  \retval     aoclsparse_status_wrong_type      matrix data type is not supported.
 *  \retval     aoclsparse_status_not_implemented
 *                   \ref aoclsparse_matrix_type is \ref aoclsparse_matrix_type_hermitian or,
 *                   \ref aoclsparse_matrix_format_type is not \ref aoclsparse_csr_mat
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_dotmv.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_dotmv.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 * \{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_zdotmv(const aoclsparse_operation       op,
                                    const aoclsparse_double_complex  alpha,
                                    aoclsparse_matrix                A,
                                    const aoclsparse_mat_descr       descr,
                                    const aoclsparse_double_complex *x,
                                    const aoclsparse_double_complex  beta,
                                    aoclsparse_double_complex       *y,
                                    aoclsparse_double_complex       *d);

DLL_PUBLIC
aoclsparse_status aoclsparse_cdotmv(const aoclsparse_operation      op,
                                    const aoclsparse_float_complex  alpha,
                                    aoclsparse_matrix               A,
                                    const aoclsparse_mat_descr      descr,
                                    const aoclsparse_float_complex *x,
                                    const aoclsparse_float_complex  beta,
                                    aoclsparse_float_complex       *y,
                                    aoclsparse_float_complex       *d);

DLL_PUBLIC
aoclsparse_status aoclsparse_ddotmv(const aoclsparse_operation op,
                                    const double               alpha,
                                    aoclsparse_matrix          A,
                                    const aoclsparse_mat_descr descr,
                                    const double              *x,
                                    const double               beta,
                                    double                    *y,
                                    double                    *d);

DLL_PUBLIC
aoclsparse_status aoclsparse_sdotmv(const aoclsparse_operation op,
                                    const float                alpha,
                                    aoclsparse_matrix          A,
                                    const aoclsparse_mat_descr descr,
                                    const float               *x,
                                    const float                beta,
                                    float                     *y,
                                    float                     *d);
/**@}*/

/*! \ingroup level3_module
 *  \brief Solve sparse triangular linear system of equations with multiple right hand sides
 *         for real/complex single and double data precisions.
 *
 *  \details
 *  \P{aoclsparse_?trsm} solves
 *  a sparse triangular linear system of equations with multiple right hand sides, of the form
 *  \f[
 *  op(A)\; X = \alpha B,
 *  \f]
 *  where \f$A\f$ is a sparse matrix of size \f$m\f$, \f$op()\f$ is a linear operator, \f$X\f$ and
 *  \f$B\f$ are rectangular dense matrices of appropiate size, while \f$\alpha\f$
 *  is a scalar.
 *  The sparse matrix \f$A\f$ can be interpreted either as a lower triangular or
 *  upper triangular. This is indicated by \p fill_mode from the matrix descriptor \p descr
 *  where either upper or
 *  lower triangular portion of the matrix is only referenced. The matrix can also be of class symmetric in
 *  which case only the selected triangular part is used. Matrix \f$A\f$ must be of full rank,
 *  that is, the matrix must be invertible. The linear operator \f$op()\f$ can
 *  define the transposition or Hermitian transposition operations. By default, no transposition
 *  is performed. The right-hand-side matrix \f$B\f$ and the solution matrix \f$X\f$ are dense
 *  and must be of the correct size, that is \f$m\f$ by \f$n\f$, see \p ldb and \p ldx input parameters
 *  for further details.
 *
 *  Explicitly, this kernel solves
 *  \f[
 *  op(A)\; X = \alpha \; B\text{, with solution } X = \alpha \; (op(A)^{-1}) \; B,
 *  \f]
 *  where
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{trans}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *  If a linear operator is applied then, the possible problems solved are
 *  \f[
 *  A^T \; X = \alpha \; B\text{, with solution } X = \alpha \; A^{-T} \; B\text{, and } \;
 *  A^H \; X = \alpha \; B\text{, with solution } X = \alpha \; A^{-H} \; B.
 *  \f]
 *
 *  <b>Notes</b>
 *@rst
 *
 *  1. If the matrix descriptor :code:`descr` specifies that the matrix :math:`A` is to be regarded as
 *     having a unitary diagonal, then the main diagonal entries of matrix :math:`A` are not accessed and
 *     are considered to all be ones.
 *
 *  2. If the matrix :math:`A` is described as upper triangular, then only the upper triangular portion of the
 *     matrix is referenced. Conversely, if the matrix :math:`A` is described lower triangular, then only the
 *     lower triangular portion of the matrix is used.
 *
 *  3. This set of APIs allocates work array of size :math:`m` for each case where the matrices :math:`B` or :math:`X` are
 *     stored in row-major format (\ref aoclsparse_order_row).
 *
 *  4. A subset of kernels are parallel (on parallel builds) and can be expected potential acceleration in the solve.
 *     These kernels are available when both dense matrices :math:`X` and :math:`B` are stored in
 *     column-major format (\ref aoclsparse_order_column) and thread count is greater than 1 on a parallel build.
 *
 *  5. There is :cpp:func:`aoclsparse_trsm_kid<aoclsparse_strsm_kid>` (Kernel ID) variation of TRSM, namely with a
 *     suffix of :code:`_kid`, this solver allows to choose which
 *     underlying TRSV kernels to use (if possible). Currently, all the existing kernels avilable in
 *     :cpp:func:`aoclsparse_trsv_kid<aoclsparse_strsv_kid>` are supported.
 *
 *  6. This routine supports only sparse matrices in CSR format.
 *@endrst
 *
 * @rst
 * .. collapse:: Example - Real space (tests/examples/sample_dtrsm.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_dtrsm.cpp
 *          :language: C++
 *          :linenos:
 *
 * \
 *
 * .. collapse:: Example - Complex space (tests/examples/sample_ztrsm.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_ztrsm.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 *
 *  @param[in]
 *  trans       matrix operation to perform on \f$A\f$. Possible values are \ref aoclsparse_operation_none,
 *              \ref aoclsparse_operation_transpose, and \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  A           sparse matrix \f$A\f$ of size \f$m\f$.
 *  @param[in]
 *  descr       descriptor of the sparse matrix \f$A\f$.
 *  @param[in]
 *  order       storage order of dense matrices \f$B\f$ and \f$X\f$. Possible options are
 *              \ref aoclsparse_order_row and \ref aoclsparse_order_column.
 *  @param[in]
 *  B           dense matrix, potentially rectangular, of size \f$m \times n\f$.
 *  @param[in]
 *  n           \f$n,\f$ number of columns of the dense matrix \f$B\f$.
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$. Eventhough the matrix \f$B\f$ is considered of size
 *              \f$m \times n\f$, its memory layout may correspond to a larger matrix (\p ldb by \f$N>n\f$)
 *              in which only the
 *              submatrix \f$B\f$ is of interest.  In this case, this parameter provides means
 *              to access the correct elements of \f$B\f$ within the larger layout.
 *  <table>
 *  <tr><th>matrix layout   </th>                   <th>row count  </th>   <th>column count</th></tr>
 *  <tr><td>\ref aoclsparse_order_row</td><td>\f$m\f$</td><td> \p ldb with \p ldb \f$\ge n\f$</td></tr>
 *  <tr><td>\ref aoclsparse_order_column</td><td> \p ldb with \p ldb \f$\ge m\f$</td><td>\f$n\f$</td></tr>
 *  </table>
 *  @param[out]
 *  X           solution matrix \f$X,\f$ dense and potentially rectangular matrix of size \f$m \times n\f$.
 *  @param[in]
 *  ldx         leading dimension of \f$X\f$. Eventhough the matrix \f$X\f$ is considered of size
 *              \f$m \times n\f$, its memory layout may correspond to a larger matrix (\p ldx by \f$N>n\f$)
 *              in which only the
 *              submatrix \f$X\f$ is of interest. In this case, this parameter provides means
 *              to access the correct elements of \f$X\f$ within the larger layout.
 *  <table>
 *  <tr><th>matrix layout   </th>                   <th>row count  </th>   <th>column count</th></tr>
 *  <tr><td>\ref aoclsparse_order_row</td><td>\f$m\f$</td><td> \p ldx with \p ldx \f$\ge n\f$</td></tr>
 *  <tr><td>\ref aoclsparse_order_column</td><td> \p ldx with \p ldx \f$\ge m\f$</td><td>\f$n\f$</td></tr>
 *  </table>
 *
 *  \retval     aoclsparse_status_success indicates that the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size informs that either \p m, \p n, \p nnz, \p ldb or \p ldx
 *              is invalid.
 *  \retval     aoclsparse_status_invalid_pointer informs that either \p descr, \p alpha, \p A,
 *              \p B, or \p X pointer is invalid.
 *  \retval     aoclsparse_status_not_implemented this error occurs when the provided matrix
 *              \ref aoclsparse_matrix_type is \ref aoclsparse_matrix_type_general or \ref aoclsparse_matrix_type_hermitian
 *              or when matrix \p A is not in CSR format.
 * \{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_ztrsm(aoclsparse_operation             trans,
                                   const aoclsparse_double_complex  alpha,
                                   aoclsparse_matrix                A,
                                   const aoclsparse_mat_descr       descr,
                                   aoclsparse_order                 order,
                                   const aoclsparse_double_complex *B,
                                   aoclsparse_int                   n,
                                   aoclsparse_int                   ldb,
                                   aoclsparse_double_complex       *X,
                                   aoclsparse_int                   ldx);

DLL_PUBLIC
aoclsparse_status aoclsparse_ctrsm(aoclsparse_operation            trans,
                                   const aoclsparse_float_complex  alpha,
                                   aoclsparse_matrix               A,
                                   const aoclsparse_mat_descr      descr,
                                   aoclsparse_order                order,
                                   const aoclsparse_float_complex *B,
                                   aoclsparse_int                  n,
                                   aoclsparse_int                  ldb,
                                   aoclsparse_float_complex       *X,
                                   aoclsparse_int                  ldx);

DLL_PUBLIC
aoclsparse_status aoclsparse_dtrsm(const aoclsparse_operation trans,
                                   const double               alpha,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   aoclsparse_order           order,
                                   const double              *B,
                                   aoclsparse_int             n,
                                   aoclsparse_int             ldb,
                                   double                    *X,
                                   aoclsparse_int             ldx);

DLL_PUBLIC
aoclsparse_status aoclsparse_strsm(const aoclsparse_operation trans,
                                   const float                alpha,
                                   aoclsparse_matrix          A,
                                   const aoclsparse_mat_descr descr,
                                   aoclsparse_order           order,
                                   const float               *B,
                                   aoclsparse_int             n,
                                   aoclsparse_int             ldb,
                                   float                     *X,
                                   aoclsparse_int             ldx);
/**@}*/

/*! \ingroup level3_module
 *  \brief Solve sparse triangular linear system of equations with multiple right hand sides
 *         for real/complex single and double data precisions (kernel flag variation).
 *
 *  \details
 * @rst
 * For full details refer to :cpp:func:`aoclsparse_?trsm()<aoclsparse_strsm>`.
 *
 * This variation of TRSM, namely with a suffix of :code:`_kid`, allows to choose which
 * underlying TRSV kernels to use (if possible). Currently, all the existing kernels supported by
 * :cpp:func:`aoclsparse_?trsv_kid()<aoclsparse_strsv_kid>` are available here as well.
 * @endrst
 *
 *  @param[in]
 *  trans       matrix operation to perform on \f$A\f$. Possible values are \ref aoclsparse_operation_none,
 *              \ref aoclsparse_operation_transpose, and \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  A           sparse matrix \f$A\f$ of size \f$m\f$.
 *  @param[in]
 *  descr       descriptor of the sparse matrix \f$A\f$.
 *  @param[in]
 *  order       storage order of dense matrices \f$B\f$ and \f$X\f$. Possible options are
 *              \ref aoclsparse_order_row and \ref aoclsparse_order_column.
 *  @param[in]
 *  B           dense matrix, potentially rectangular, of size \f$m \times n\f$.
 *  @param[in]
 *  n           \f$n,\f$ number of columns of the dense matrix \f$B\f$.
 *  @param[in]
 *  ldb         leading dimension of \f$B\f$. Eventhough the matrix \f$B\f$ is considered of size
 *              \f$m \times n\f$, its memory layout may correspond to a larger matrix (\p ldb by \f$N>n\f$)
 *              in which only the
 *              submatrix \f$B\f$ is of interest.  In this case, this parameter provides means
 *              to access the correct elements of \f$B\f$ within the larger layout.
 *  <table>
 *  <tr><th>matrix layout   </th>                   <th>row count  </th>   <th>column count</th></tr>
 *  <tr><td>\ref aoclsparse_order_row</td><td>\f$m\f$</td><td> \p ldb with \p ldb \f$\ge n\f$</td></tr>
 *  <tr><td>\ref aoclsparse_order_column</td><td> \p ldb with \p ldb \f$\ge m\f$</td><td>\f$n\f$</td></tr>
 *  </table>
 *  @param[out]
 *  X           solution matrix \f$X,\f$ dense and potentially rectangular matrix of size \f$m \times n\f$.
 *  @param[in]
 *  ldx         leading dimension of \f$X\f$. Eventhough the matrix \f$X\f$ is considered of size
 *              \f$m \times n\f$, its memory layout may correspond to a larger matrix (\p ldx by \f$N>n\f$)
 *              in which only the
 *              submatrix \f$X\f$ is of interest. In this case, this parameter provides means
 *              to access the correct elements of \f$X\f$ within the larger layout.
 *  <table>
 *  <tr><th>matrix layout   </th>                   <th>row count  </th>   <th>column count</th></tr>
 *  <tr><td>\ref aoclsparse_order_row</td><td>\f$m\f$</td><td> \p ldx with \p ldx \f$\ge n\f$</td></tr>
 *  <tr><td>\ref aoclsparse_order_column</td><td> \p ldx with \p ldx \f$\ge m\f$</td><td>\f$n\f$</td></tr>
 *  </table>
 *
 *  @param[in]
 *  kid         kernel ID, hints which kernel to use.
 * \{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_ztrsm_kid(aoclsparse_operation             trans,
                                       const aoclsparse_double_complex  alpha,
                                       aoclsparse_matrix                A,
                                       const aoclsparse_mat_descr       descr,
                                       aoclsparse_order                 order,
                                       const aoclsparse_double_complex *B,
                                       aoclsparse_int                   n,
                                       aoclsparse_int                   ldb,
                                       aoclsparse_double_complex       *X,
                                       aoclsparse_int                   ldx,
                                       const aoclsparse_int             kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_ctrsm_kid(aoclsparse_operation            trans,
                                       const aoclsparse_float_complex  alpha,
                                       aoclsparse_matrix               A,
                                       const aoclsparse_mat_descr      descr,
                                       aoclsparse_order                order,
                                       const aoclsparse_float_complex *B,
                                       aoclsparse_int                  n,
                                       aoclsparse_int                  ldb,
                                       aoclsparse_float_complex       *X,
                                       aoclsparse_int                  ldx,
                                       const aoclsparse_int            kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_dtrsm_kid(const aoclsparse_operation trans,
                                       const double               alpha,
                                       aoclsparse_matrix          A,
                                       const aoclsparse_mat_descr descr,
                                       aoclsparse_order           order,
                                       const double              *B,
                                       aoclsparse_int             n,
                                       aoclsparse_int             ldb,
                                       double                    *X,
                                       aoclsparse_int             ldx,
                                       const aoclsparse_int       kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_strsm_kid(const aoclsparse_operation trans,
                                       const float                alpha,
                                       aoclsparse_matrix          A,
                                       const aoclsparse_mat_descr descr,
                                       aoclsparse_order           order,
                                       const float               *B,
                                       aoclsparse_int             n,
                                       aoclsparse_int             ldb,
                                       float                     *X,
                                       aoclsparse_int             ldx,
                                       const aoclsparse_int       kid);
/**@}*/

/*! \ingroup level3_module
 *  \brief Sparse matrix Sparse matrix multiplication for real and complex datatypes.
 *  \details
 *  \P{aoclsparse_?sp2m} multiplies two sparse matrices in CSR storage format. The
 *  result is stored in a newly allocated sparse matrix in CSR format, such that
 *  \f[
 *    C =  op(A) \, op(B),
 *  \f]
 *  with
 *  \f[
 *     op(A) = \left\{
 *     \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{opA}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{opA}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{opA}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *     \end{array}
 *     \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if } {\bf\mathsf{opB}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *        B^T, & \text{if } {\bf\mathsf{opB}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *        B^H, & \text{if } {\bf\mathsf{opB}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *  where \f$A\f$ is a  \f$m \times k\f$ matrix ,\f$B\f$ is a \f$k \times n\f$ matrix,
 *  resulting in  \f$m \times n\f$ matrix \f$C\f$, for \p opA and \p opB =
 *  \ref aoclsparse_operation_none. \f$A\f$ is a \f$k \times m\f$ matrix
 *  when \p opA = \ref aoclsparse_operation_transpose or \ref aoclsparse_operation_conjugate_transpose
 *  and \f$B\f$ is a \f$n \times k\f$ matrix when \p opB = \ref aoclsparse_operation_transpose
 *  or \ref aoclsparse_operation_conjugate_transpose
 *
 *  aoclsparse_sp2m can be run in single-stage or two-stage. The single-stage algorithm
 *  allocates and computes the entire output matrix in a single stage
 *  \ref aoclsparse_stage_full_computation. Whereas, in two-stage algorithm, the first
 *  stage \ref aoclsparse_stage_nnz_count allocates memory for the output matrix and
 *  computes the number of entries of the matrix. The second stage
 *  \ref aoclsparse_stage_finalize computes column indices of non-zero elements and
 *  values of the output matrix. The second stage has to be invoked only after the
 *  first stage. But, it can be also be invoked multiple times consecutively when the
 *  sparsity structure of input matrices remains unchanged, with only the values getting
 *  updated.
 *
 *  @param[in]
 *  opA     matrix \f$A\f$ operation type.
 *  @param[in]
 *  descrA      descriptor of the sparse CSR matrix \f$A\f$. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  A        sparse CSR matrix \f$A\f$ .
 *  @param[in]
 *  opB     matrix \f$B\f$ operation type.
 *  @param[in]
 *  descrB      descriptor of the sparse CSR matrix \f$B\f$. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  B        sparse CSR matrix \f$B\f$ .
 *  @param[in]
 *  request     Specifies full computation or two-stage algorithm
 *              \ref aoclsparse_stage_nnz_count , Only rowIndex array of the
 *              CSR matrix is computed internally. The output sparse CSR matrix
 *              can be extracted to measure the memory required for full operation.
 *              \ref aoclsparse_stage_finalize . Finalize computation of remaining
 *              output arrays ( column indices and values of output matrix entries) .
 *              Has to be called only after aoclsparse_sp2m call with
 *              aoclsparse_stage_nnz_count parameter.
 *              \ref aoclsparse_stage_full_computation . Perform the entire
 *              computation in a single step.
 *
 *  @param[out]
 *  *C        Pointer to sparse CSR matrix \f$C\f$ .
 *  	      Matrix \f$C\f$ arrays will always have zero-based indexing, irrespective of matrix \f$A\f$
 *  	      or matrix \f$B\f$ being one-based or zero-based indexing.
 *  	      The column indices of the output matrix in CSR format can appear unsorted.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer \p descrA, \p descrB, \p A, \p B, \p C is invalid.
 *  \retval     aoclsparse_status_invalid_size input size parameters contain an invalid value.
 *  \retval     aoclsparse_status_invalid_value input parameters contain an invalid value.
 *  \retval     aoclsparse_status_wrong_type A and B matrix datatypes dont match.
 *  \retval     aoclsparse_status_memory_error Memory allocation failure.
 *  \retval     aoclsparse_status_not_implemented
 *              \ref aoclsparse_matrix_type is not \ref aoclsparse_matrix_type_general or
 *              input matrices \p A or \p B is not in CSR format
 * @rst
 * .. collapse:: Example - Complex space (tests/examples/sample_zsp2m.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_zsp2m.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_sp2m(aoclsparse_operation       opA,
                                  const aoclsparse_mat_descr descrA,
                                  const aoclsparse_matrix    A,
                                  aoclsparse_operation       opB,
                                  const aoclsparse_mat_descr descrB,
                                  const aoclsparse_matrix    B,
                                  const aoclsparse_request   request,
                                  aoclsparse_matrix         *C);

/*! \ingroup level3_module
 *  \brief Sparse matrix Sparse matrix multiplication for real and complex datatypes.
 *  \details
 *  \P{aoclsparse_?spmm} multiplies two sparse matrices in CSR storage format. The
 *  result is stored in a newly allocated sparse matrix in CSR format, such that
 * @rst
 * .. math::
 *    C =  op(A) \cdot B,
 *    \text{ with }
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A,   & \text{ if } {\bf\mathsf{opA}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{ if } {\bf\mathsf{opA}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{ if } {\bf\mathsf{opA}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *@endrst
 *  where \f$A\f$ is a  \f$m \times k\f$ matrix ,\f$B\f$ is a \f$k \times n\f$ matrix,
 *  resulting in  \f$m \times n\f$ matrix \f$C\f$, for \p opA = \ref aoclsparse_operation_none.
 *  \f$A\f$ is a \f$k \times m\f$ matrix when opA = \ref aoclsparse_operation_transpose
 *  or \ref aoclsparse_operation_conjugate_transpose
 *
 *  @param[in]
 *  opA     matrix \f$A\f$ operation type.
 *
 *  @param[in]
 *  A        sparse CSR matrix \f$A\f$.
 *
 *  @param[in]
 *  B        sparse CSR matrix \f$B\f$.
 *
 *  @param[out]
 *  *C        Pointer to sparse CSR matrix \f$C\f$ .
 *  	      Matrix \f$C\f$ arrays will always have zero-based indexing, irrespective of matrix \f$A\f$
 *  	      or matrix \f$B\f$ being one-based or zero-based indexing.
 *  	      The column indices of the output matrix in CSR format can appear unsorted.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer \p A, \p B, \p C is invalid.
 *  \retval     aoclsparse_status_invalid_size input size parameters contain an invalid value.
 *  \retval     aoclsparse_status_invalid_value input parameters contain an invalid value.
 *  \retval     aoclsparse_status_wrong_type \p A and \p B matrix data types do not match.
 *  \retval     aoclsparse_status_memory_error Memory allocation failure.
 *  \retval     aoclsparse_status_not_implemented Input matrices \p A or \p B is not in CSR format
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_spmm(aoclsparse_operation    opA,
                                  const aoclsparse_matrix A,
                                  const aoclsparse_matrix B,
                                  aoclsparse_matrix      *C);

/*! \ingroup level3_module
 *  \brief Symmetric product of three sparse matrices for real and complex datatypes stored as a sparse matrix.
 *  \details
 *  \P{aoclsparse_sypr} multiplies three sparse matrices in CSR storage format. The result
 *  is returned in a newly allocated symmetric or Hermitian sparse matrix stored as an upper
 *  triangle in CSR format.
 *
 *  If \f$opA\f$ is \ref aoclsparse_operation_none,
 *  \f[
 *    C =  A \cdot B \cdot A^T,
 *  \f]
 *  or
 *  \f[
 *    C =  A \cdot B \cdot A^H,
 *  \f]
 *  for real or complex input matrices, respectively, where \f$A\f$ is
 *  a \f$m \times n\f$ general matrix ,\f$B\f$ is a \f$n \times n\f$
 *  symmetric (for real data types) or Hermitian (for complex data types)
 *  matrix, resulting in a symmetric or Hermitian \f$m \times m\f$ matrix \f$C\f$.
 *
 *  Otherwise,
 *  \f[
 *    C =  op(A) \cdot B \cdot A,
 *  \f]
 *  with
 *  \f[
 *     op(A) = \left\{
 *     \begin{array}{ll}
 *         A^T, & \text{if } {\bf\mathsf{opA}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{opA}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *     \end{array}
 *     \right.
 *  \f]
 *  where \f$A\f$ is a \f$m \times n\f$ matrix and \f$B\f$ is a \f$m \times m\f$ symmetric
 *  (or Hermitian) matrix, resulting in a \f$n \times n\f$ symmetric (or Hermitian)
 *  matrix \f$C\f$.
 *
 *  Depending on \p request, \p aoclsparse_sypr might compute the result in a single stage
 *  (\ref aoclsparse_stage_full_computation) or in two stages. Then the first stage
 *  (\ref aoclsparse_stage_nnz_count) allocates memory for the new output matrix \f$C\f$
 *  and computes its number of non-zeros and their structure which is followed by
 *  the second stage (\ref aoclsparse_stage_finalize) to compute the column indices
 *  and values of all elements. The second stage can be invoked multiple times (either
 *  after \ref aoclsparse_stage_full_computation or \ref aoclsparse_stage_nnz_count)
 *  to recompute the numerical values of \f$C\f$ on assumption that the sparsity
 *  structure of the input matrices remained unchanged and only the values of the
 *  non-zero elements were modified (e.g., by a call to aoclsparse_supdate_values()
 *  and variants).
 *
 *  \note \p aoclsparse_sypr supports only matrices in CSR format which have sorted column
 *  indices in each row. If the matrices are unsorted, you might want to call
 *  aoclsparse_order_mat().
 *  \note
 *  Currently, \p opA = \ref aoclsparse_operation_transpose is supported only for real data types.
 *
 *  @param[in]
 *  opA     matrix \f$A\f$ operation type.
 *  @param[in]
 *  A        sorted sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  B        sorted sparse CSR matrix \f$B\f$ to be interpreted as symmetric (or Hermitian).
 *  @param[in]
 *  descrB      descriptor of the sparse CSR matrix \f$B\f$. \ref aoclsparse_matrix_type
 *              must be
 *              \ref aoclsparse_matrix_type_symmetric for real matrices
 *              or
 *              \ref aoclsparse_matrix_type_hermitian for complex matrices.
 *              \ref aoclsparse_fill_mode might be either
 *              \ref aoclsparse_fill_mode_upper or \ref aoclsparse_fill_mode_lower
 *              to process the upper or lower triangular matrix part, respectively.
 *
 *  @param[in]
 *  request     Specifies if the computation takes place in one stage
 *              (\ref aoclsparse_stage_full_computation) or in two stages
 *              (\ref aoclsparse_stage_nnz_count followed by \ref aoclsparse_stage_finalize).
 *
 *  @param[inout]
 *  *C        Pointer to the new sparse CSR symmetric/Hermitian matrix \f$C\f$ .
 *            Only upper triangle of the result matrix is computed.
 *  	      Matrix \f$C\f$ will always have zero-based indexing, irrespective
 *  	      of the zero/one-based indexing of the input matrices \f$A\f$ and \f$B\f$.
 *  	      The column indices of the output matrix in CSR format might be unsorted.
 *  	      If \p request is \ref aoclsparse_stage_finalize, matrix \f$C\f$ must
 *  	      not be modified by the user since the last call to \p aoclsparse_sypr,
 *  	      in the other cases is \f$C\f$ treated as an output only. The matrix
 *  	      should be freed by aoclsparse_destroy() when no longer needed.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer \p descrB, \p A, \p B or \p C is invalid.
 *  \retval     aoclsparse_status_invalid_size Matrix dimensions do not match \p A or \p B is not square.
 *  \retval     aoclsparse_status_invalid_value Input parameters are invalid,
 *              for example, \p descrB does not match \p B indexing or \p B is not
 *              symmetric/Hermitian, \p C has been modified between stages
 *              or \p opA or \p request is not recognized.
 *  \retval     aoclsparse_status_wrong_type \p A and \p B matrix data types do not match.
 *  \retval     aoclsparse_status_not_implemented
 *              Input matrix \p A or \p B is not in CSR format.
 *  \retval     aoclsparse_status_unsorted_input Input matrices are not sorted.
 *  \retval     aoclsparse_status_memory_error Memory allocation failure.
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_zsypr.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_zsypr.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_sypr(aoclsparse_operation       opA,
                                  const aoclsparse_matrix    A,
                                  const aoclsparse_matrix    B,
                                  const aoclsparse_mat_descr descrB,
                                  aoclsparse_matrix         *C,
                                  const aoclsparse_request   request);

/*! \ingroup level3_module
 *  \brief Sparse matrix dense matrix multiplication using CSR storage format
 *
 *  \details
 *  \P{aoclsparse_?csrmm} multiplies a scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
 *  matrix \f$A\f$, defined in CSR storage format, and a dense \f$k \times n\f$
 *  matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
 *  is multiplied by a scalar \f$\beta\f$, such that
 *  \f[
 *    C = \alpha \, op(A) \, B + \beta \, C,
 *  \quad \text{ with } \quad
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{ if } {\bf\mathsf{trans}\_\mathsf{A}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *        A^T, & \text{ if } {\bf\mathsf{trans}\_\mathsf{A}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *        A^H, & \text{ if } {\bf\mathsf{trans}\_\mathsf{A}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  @param[in]
 *  op          Matrix \f$A\f$ operation type.
 *  @param[in]
 *  alpha       Scalar \f$\alpha\f$.
 *  @param[in]
 *  A           Sparse CSR matrix \f$A\f$ structure.
 *  @param[in]
 *  descr       descriptor of the sparse CSR matrix \f$A\f$. Currently, supports
 *              \ref aoclsparse_matrix_type_general, \ref aoclsparse_matrix_type_symmetric,
 *              and \ref aoclsparse_matrix_type_hermitian matrices. Both, base-zero and
 *              base-one input arrays of CSR matrix are supported.
 *  @param[in]
 *  order       \ref aoclsparse_order_row / \ref aoclsparse_order_column for dense matrix
 *  @param[in]
 *  B           Array of dimension \f$ldb \times n\f$ for \ref aoclsparse_order_column.
 *              For \ref aoclsparse_order_row, the dimension is
 *              \f$(number of columns in A) \times ldb\f$ if \f$op(A) = A\f$,
 *              or \f$(number of rows in A) \times ldb\f$ if \f$op(A) = A^T\f$.
 *  @param[in]
 *  n           Number of columns of the dense matrix \f$C\f$.
 *  @param[in]
 *  ldb         Leading dimension of \f$B\f$.
 *  @param[in]
 *  beta        Scalar \f$\beta\f$.
 *  @param[inout]
 *  C           Array of dimension \f$ldc \times n\f$ for \ref aoclsparse_order_column.
 *              For \ref aoclsparse_order_row, the dimension is
 *              \f$(number of rows in A) \times ldc\f$ if \f$op(A) = A\f$,
 *              or \f$(number of columns in A) \times ldc\f$ if \f$op(A) = A^T\f$.
 *  @param[in]
 *  ldc         Leading dimension of \f$C\f$.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size The value of \p m, \p n, \p k, \p nnz, \p ldb or \p ldc
 *              is invalid.
 *  \retval     aoclsparse_status_invalid_pointer The pointer \p descr, \p A, \p B, or \p C
 *              is invalid.
 *  \retval     aoclsparse_status_invalid_value The values of \p descr->base and \p A->mats[0]->base do not coincide.
 *  \retval     aoclsparse_status_not_implemented
 *              \ref aoclsparse_matrix_type is not one of these: \ref aoclsparse_matrix_type_general,
 *              \ref aoclsparse_matrix_type_symmetric, \ref aoclsparse_matrix_type_hermitian  or
 *              input matrix \p A is not in CSR format
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_csrmm.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_csrmm.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 * \{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_zcsrmm(aoclsparse_operation             op,
                                    const aoclsparse_double_complex  alpha,
                                    const aoclsparse_matrix          A,
                                    const aoclsparse_mat_descr       descr,
                                    aoclsparse_order                 order,
                                    const aoclsparse_double_complex *B,
                                    aoclsparse_int                   n,
                                    aoclsparse_int                   ldb,
                                    const aoclsparse_double_complex  beta,
                                    aoclsparse_double_complex       *C,
                                    aoclsparse_int                   ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_ccsrmm(aoclsparse_operation            op,
                                    const aoclsparse_float_complex  alpha,
                                    const aoclsparse_matrix         A,
                                    const aoclsparse_mat_descr      descr,
                                    aoclsparse_order                order,
                                    const aoclsparse_float_complex *B,
                                    aoclsparse_int                  n,
                                    aoclsparse_int                  ldb,
                                    const aoclsparse_float_complex  beta,
                                    aoclsparse_float_complex       *C,
                                    aoclsparse_int                  ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsrmm(aoclsparse_operation       op,
                                    const double               alpha,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_mat_descr descr,
                                    aoclsparse_order           order,
                                    const double              *B,
                                    aoclsparse_int             n,
                                    aoclsparse_int             ldb,
                                    const double               beta,
                                    double                    *C,
                                    aoclsparse_int             ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_scsrmm(aoclsparse_operation       op,
                                    const float                alpha,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_mat_descr descr,
                                    aoclsparse_order           order,
                                    const float               *B,
                                    aoclsparse_int             n,
                                    aoclsparse_int             ldb,
                                    const float                beta,
                                    float                     *C,
                                    aoclsparse_int             ldc);
/**@}*/

//-------------------------------------------------------------------------------------------
/*! \ingroup level3_module
 *  \brief Matrix multiplication of two sparse matrices stored in the CSR storage format. The output
 *         matrix is stored in a dense format.
 *  \details
 *  \P{aoclsparse_?spmmd} multiplies a sparse
 *  matrix \f$A\f$  and a sparse matrix \f$B\f$, both stored in the CSR storage format, and saves the result in a dense  matrix \f$C\f$, such that
 *  \f[
 *    C := op(A) \cdot B,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if op} = \text{aoclsparse\_operation\_none} \\
 *        A^T, & \text{if op} = \text{aoclsparse\_operation\_transpose} \\
 *        A^H, & \text{if op} = \text{aoclsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *
 *  @param[in]
 *  op     Operation to perform on matrix \f$A\f$.
 *  @param[in]
 *  A      Matrix structure containing sparse matrix \f$A\f$ of size \f$m \times k\f$.
 *  @param[in]
 *  B      Matrix structure containing sparse matrix \f$B\f$ of size \f$k \times n\f$ if \p op is \ref aoclsparse_operation_none otherwise of size \f$m \times n\f$.
 *  @param[in]
 *  layout Ordering of the dense output matrix: valid values are \ref aoclsparse_order_row and \ref aoclsparse_order_column.
 *  @param[inout]
 *  C      Dense output matrix \f$C\f$ of size \f$m \times n\f$ if \p op is \ref aoclsparse_operation_none, otherwise of size \f$k \times n\f$ containing the matrix-matrix product of \f$A\f$ and \f$B\f$.
 *  @param[in]
 *  ldc    Leading dimension of \f$C\f$, e.g., for C stored in \p aoclsparse_order_row, \p ldc
 *         must be at least \f$\max{(1, m)}\f$  when \f$op(A) = A\f$, or
 *         \f$\max{(1, k)}\f$ if \f$op(A) = A^T\f$ or \f$op(A) = A^H\f$.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n, \p k, \p nnz or \p ldc is not valid.
 *  \retval     aoclsparse_status_invalid_pointer \p A, \p B or \p C pointer is not valid.
 *  \retval     aoclsparse_status_wrong_type \ref aoclsparse_matrix_data_type does not match the precision type.
 *  \retval     aoclsparse_status_not_implemented
 *              \ref aoclsparse_matrix_format_type is not \ref aoclsparse_csr_mat.
 *
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_sspmmd(const aoclsparse_operation op,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_matrix    B,
                                    const aoclsparse_order     layout,
                                    float                     *C,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_dspmmd(const aoclsparse_operation op,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_matrix    B,
                                    const aoclsparse_order     layout,
                                    double                    *C,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_cspmmd(const aoclsparse_operation op,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_matrix    B,
                                    const aoclsparse_order     layout,
                                    aoclsparse_float_complex  *C,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_zspmmd(const aoclsparse_operation op,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_matrix    B,
                                    const aoclsparse_order     layout,
                                    aoclsparse_double_complex *C,
                                    const aoclsparse_int       ldc);

/**@}*/

//-------------------------------------------------------------------------------------------
/*! \ingroup level3_module
 *  \brief A variant of matrix multiplication of two sparse matrices stored in the CSR storage format. The output
 *         matrix is stored in a dense format. Supports operations on both sparse matrices.
 *  \details
 *  \P{aoclsparse_?sp2md} multiplies a sparse
 *  matrix \f$A\f$  and a sparse matrix \f$B\f$, both stored in the CSR storage format, and saves the result in a dense matrix \f$C\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if opA} = \text{aoclsparse\_operation\_none} \\
 *        A^T, & \text{if opA} = \text{aoclsparse\_operation\_transpose} \\
 *        A^H, & \text{if opA} = \text{aoclsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if opB} = \text{aoclsparse\_operation\_none} \\
 *        B^T, & \text{if opB} = \text{aoclsparse\_operation\_transpose} \\
 *        B^H, & \text{if opB} = \text{aoclsparse\_operation\_conjugate\_transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  @param[in]
 *  opA     Operation to perform on matrix \f$A\f$.
 *  @param[in]
 *  descrA  Descriptor of A. Only \ref aoclsparse_matrix_type_general is supported at present.
 *          As a consequence, all other parameters within the descriptor are ignored.
 *  @param[in]
 *  A     Matrix structure containing sparse matrix \f$A\f$ of size \f$m \times k\f$.
 *  @param[in]
 *  opB     Operation to perform on matrix \f$B\f$.
 *  @param[in]
 *  descrB  Descriptor of B. Only \ref aoclsparse_matrix_type_general is supported at present.
 *          As a consequence, all other parameters within the descriptor are ignored.
 *  @param[in]
 *  B      Matrix structure containing sparse matrix \f$B\f$ of size \f$k \times n\f$ if \p op is \ref aoclsparse_operation_none otherwise of size \f$m \times n\f$.
 *  @param[in]
 *  alpha  Value of \f$ \alpha\f$.
 *  @param[in]
 *  beta   Value of \f$ \beta\f$.
 *  @param[inout]
 *  C      Dense output matrix \f$C\f$.
 *  @param[in]
 *  layout Ordering of the dense output matrix: valid values are \ref aoclsparse_order_row and \ref aoclsparse_order_column.
 *  @param[in]
 *  ldc    Leading dimension of \f$C\f$, e.g., for C stored in \p aoclsparse_order_row, \p ldc
 *         must be at least \f$\max{(1, m)}\f$ (\f$op(A) = A\f$) or
 *         \f$\max{(1, k)}\f$ (\f$op(A) = A^T\f$ or \f$op(A) = A^H\f$).
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size \p m, \p n, \p k, \p nnz or \p ldc is not valid.
 *  \retval     aoclsparse_status_invalid_pointer \p A, \p B or \p C pointer is not valid.
 *  \retval     aoclsparse_status_wrong_type \ref aoclsparse_matrix_data_type does not match the precision type.
 *  \retval     aoclsparse_status_not_implemented
 *              \ref aoclsparse_matrix_format_type is not \ref aoclsparse_csr_mat.
 *  \retval     aoclsparse_status_internal_error An internal error occurred.
 *
 */

/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_ssp2md(const aoclsparse_operation opA,
                                    const aoclsparse_mat_descr descrA,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_operation opB,
                                    const aoclsparse_mat_descr descrB,
                                    const aoclsparse_matrix    B,
                                    const float                alpha,
                                    const float                beta,
                                    float                     *C,
                                    const aoclsparse_order     layout,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_dsp2md(const aoclsparse_operation opA,
                                    const aoclsparse_mat_descr descrA,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_operation opB,
                                    const aoclsparse_mat_descr descrB,
                                    const aoclsparse_matrix    B,
                                    const double               alpha,
                                    const double               beta,
                                    double                    *C,
                                    const aoclsparse_order     layout,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_csp2md(const aoclsparse_operation opA,
                                    const aoclsparse_mat_descr descrA,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_operation opB,
                                    const aoclsparse_mat_descr descrB,
                                    const aoclsparse_matrix    B,
                                    aoclsparse_float_complex   alpha,
                                    aoclsparse_float_complex   beta,
                                    aoclsparse_float_complex  *C,
                                    const aoclsparse_order     layout,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_zsp2md(const aoclsparse_operation opA,
                                    const aoclsparse_mat_descr descrA,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_operation opB,
                                    const aoclsparse_mat_descr descrB,
                                    const aoclsparse_matrix    B,
                                    aoclsparse_double_complex  alpha,
                                    aoclsparse_double_complex  beta,
                                    aoclsparse_double_complex *C,
                                    const aoclsparse_order     layout,
                                    const aoclsparse_int       ldc);

/**@}*/

/*! \ingroup level3_module
 *  \brief Sparse matrix Sparse matrix multiplication using CSR storage format
 *  for single and double precision datatypes.
 *  \details
 *  \P{aoclsparse_?csr2m} multiplies a sparse \f$m \times k\f$
 *  matrix \f$A\f$, defined in CSR storage format, and the sparse \f$k \times n\f$
 *  matrix \f$B\f$, defined in CSR storage format and stores the result to the sparse
 *  \f$m \times n\f$ matrix \f$C\f$, such that
 *  \f[
 *    C =  op(A) \cdot op(B),
 *  \f]
 *  with
 *  \f[
 *     op(A) = \left\{
 *     \begin{array}{ll}
 *         A,   & \text{if } {\bf\mathsf{trans}\_\mathsf{A}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{if } {\bf\mathsf{trans}\_\mathsf{A}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{if } {\bf\mathsf{trans}\_\mathsf{A}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *     \end{array}
 *     \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if } {\bf\mathsf{trans}\_\mathsf{B}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *        B^T, & \text{if } {\bf\mathsf{trans}\_\mathsf{B}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *        B^H, & \text{if } {\bf\mathsf{trans}\_\mathsf{B}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  @param[in]
 *  trans_A     matrix \f$A\f$ operation type.
 *  @param[in]
 *  descrA      descriptor of the sparse CSR matrix \f$A\f$. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  csrA        sparse CSR matrix \f$A\f$ structure.
 *  @param[in]
 *  trans_B     matrix \f$B\f$ operation type.
 *  @param[in]
 *  descrB      descriptor of the sparse CSR matrix \f$B\f$. Currently, only
 *              \ref aoclsparse_matrix_type_general is supported.
 *  @param[in]
 *  csrB        sparse CSR matrix \f$B\f$ structure.
 *  @param[in]
 *  request     Specifies full computation or two-stage algorithm
 *  		\ref aoclsparse_stage_nnz_count , Only rowIndex array of the
 *  		CSR matrix is computed internally. The output sparse CSR matrix
 *  		can be extracted to measure the memory required for full operation.
 *  		\ref aoclsparse_stage_finalize . Finalize computation of remaining
 *  		output arrays ( column indices and values of output matrix entries) .
 *  		Has to be called only after aoclsparse_dcsr2m() call with
 *  		\p aoclsparse_stage_nnz_count parameter.
 *  		\ref aoclsparse_stage_full_computation. Perform the entire
 *  		computation in a single step.
 *
 *  @param[out]
 *  *csrC        Pointer to sparse CSR matrix \f$C\f$ structure.
 *
 *  \retval     aoclsparse_status_success the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size input parameters contain an invalid value.
 *  \retval     aoclsparse_status_invalid_pointer \p descrA,  \p csr,
 *              \p descrB,  \p csrB, \p csrC is invalid.
 *  \retval     aoclsparse_status_not_implemented
 *              \ref aoclsparse_matrix_type is not \ref aoclsparse_matrix_type_general or
 *              input matrices \p A or \p B is not in CSR format
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_csr2m.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_csr2m.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 * @{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_scsr2m(aoclsparse_operation       trans_A,
                                    const aoclsparse_mat_descr descrA,
                                    const aoclsparse_matrix    csrA,
                                    aoclsparse_operation       trans_B,
                                    const aoclsparse_mat_descr descrB,
                                    const aoclsparse_matrix    csrB,
                                    const aoclsparse_request   request,
                                    aoclsparse_matrix         *csrC);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsr2m(aoclsparse_operation       trans_A,
                                    const aoclsparse_mat_descr descrA,
                                    const aoclsparse_matrix    csrA,
                                    aoclsparse_operation       trans_B,
                                    const aoclsparse_mat_descr descrB,
                                    const aoclsparse_matrix    csrB,
                                    const aoclsparse_request   request,
                                    aoclsparse_matrix         *csrC);

/**@}*/

/*! \ingroup level3_module
 *  \brief Addition of two sparse matrices
 *
 *  \details
 *  \P{aoclsparse_?add} adds two sparse matrices and returns a sparse matrix.
 *  Matrices can be either real or complex types but cannot be intermixed.
 *  It performs
 *  \f[
 *    C = \alpha \, op(A) + B
 *  \qquad\text{ with } \qquad
 *     op(A) = \left\{
 *     \begin{array}{ll}
 *         A,   & \text{ if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{none} \\
 *         A^T, & \text{ if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \\
 *         A^H, & \text{ if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose}
 *     \end{array}
 *     \right.
 *  \f]
 *  where \f$A\f$ is a \f$m \times n\f$ matrix and \f$B\f$ is a \f$m \times n\f$ matrix,
 *  if \p op = \ref aoclsparse_operation_none. Otherwise \f$A\f$ is \f$n \times m\f$
 *  and the result matrix \f$C\f$ has the same dimension as \f$B\f$.
 *
 *  \note Only matrices in CSR format are supported in this release.
 *
 *  @param[in]  op      matrix \f$A\f$ operation type.
 *  @param[in]  alpha   scalar with same precision as \f$A\f$ and \f$B\f$ matrix
 *  @param[in]  A       source sparse matrix \f$A\f$
 *  @param[in]  B       source sparse matrix \f$B\f$
 *  @param[out] *C      pointer to the sparse output matrix \f$C\f$
 *
 *  \retval     aoclsparse_status_success               The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer       \p A or \p B or \p C are invalid
 *  \retval     aoclsparse_status_invalid_size          The dimensions of \p A and \p B are not compatible.
 *  \retval     aoclsparse_status_internal_error        Internal Error Occured
 *  \retval     aoclsparse_status_memory_error          Memory allocation failure.
 *  \retval     aoclsparse_status_not_implemented       Matrices are not in CSR format.
 * @{
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_zadd(const aoclsparse_operation      op,
                                  const aoclsparse_matrix         A,
                                  const aoclsparse_double_complex alpha,
                                  const aoclsparse_matrix         B,
                                  aoclsparse_matrix              *C);

DLL_PUBLIC
aoclsparse_status aoclsparse_cadd(const aoclsparse_operation     op,
                                  const aoclsparse_matrix        A,
                                  const aoclsparse_float_complex alpha,
                                  const aoclsparse_matrix        B,
                                  aoclsparse_matrix             *C);

DLL_PUBLIC
aoclsparse_status aoclsparse_dadd(const aoclsparse_operation op,
                                  const aoclsparse_matrix    A,
                                  const double               alpha,
                                  const aoclsparse_matrix    B,
                                  aoclsparse_matrix         *C);

DLL_PUBLIC
aoclsparse_status aoclsparse_sadd(const aoclsparse_operation op,
                                  const aoclsparse_matrix    A,
                                  const float                alpha,
                                  const aoclsparse_matrix    B,
                                  aoclsparse_matrix         *C);
/**@}*/

/*! \ingroup level3_module
 *  \brief Performs symmetric triple product of a sparse matrix and a dense matrix and stores the output as a dense matrix.
 *
 *  \details
 *  \P{aoclsparse_?syprd} performs product of a scalar \f$\alpha\f$, with the
 *  symmetric triple product of a sparse\f$m \times k\f$ matrix \f$A\f$, defined in CSR format,
 *  with a \f$k \times k\f$ symmetric dense (or Hermitian) matrix \f$B\f$, and a \f$k \times m\f$ \f$op(A)\f$.
 *  Adds the resulting matrix to \f$m \times m\f$ symmetric dense (or Hermitian)  matrix \f$C\f$ that is multiplied
 *  by a scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot A \cdot B \cdot op(A)  + \beta \cdot C
 *  \f]
 *  if \f$op\f$ is \ref aoclsparse_operation_none.
 *
 *  Otherwise,
 *  \f[
 *    C := \alpha \cdot op(A) \cdot B \cdot A  + \beta \cdot C
 *  \f]
 *
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A^T, & \text{if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{transpose} \text{ (real matrices)}\\
 *         A^H, & \text{if } {\bf\mathsf{op}} = \text{aoclsparse}\_\text{operation}\_\text{conjugate}\_\text{transpose} \text{ (complex matrices)}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  <b>Notes</b>
 *
 * 1. This routine assumes the dense matrices (B and C) are stored in full although
 *    the computations happen on the upper triangular portion of the matrices.
 *
 * 2. \ref aoclsparse_operation_transpose is only supported for real matrices.
 *
 * 3. \ref aoclsparse_operation_conjugate_transpose is only supported for complex matrices.
 *
 * 4. Complex dense matrices are assumed to be Hermitian matrices.
 *
 *  @param[in]
 *  op          Matrix \f$A\f$ operation type.
 *  @param[in]
 *  A           Sparse CSR matrix \f$A\f$ structure.
 *  @param[in]
 *  B           Array of dimension \f$ldb \times ldb\f$.
 *              Only the upper triangular matrix is used for computation.
 *  @param[in]
 *  orderB      \ref aoclsparse_order_row or \ref aoclsparse_order_column for dense matrix B.
 *  @param[in]
 *  ldb         Leading dimension of \f$B\f$, must be at least \f$\max{(1, k)}\f$
 *              (\f$op(A) = A\f$) or \f$\max{(1, m)}\f$ (\f$op(A) = A^T\f$ or
 *              \f$op(A) = A^H\f$).
 *  @param[in]
 *  alpha       Scalar \f$\alpha\f$.
 *  @param[in]
 *  beta        Scalar \f$\beta\f$.
 *  @param[in, out]
 *  C           Array of dimension \f$ldc \times ldc\f$.
 *              Only upper triangular part of the matrix is processed.
 *  @param[in]
 *  orderC      \ref aoclsparse_order_row or \ref aoclsparse_order_column for dense matrix C.
 *  @param[in]
 *  ldc         Leading dimension of \f$C\f$, must be at least \f$\max{(1, m)}\f$
 *              (\f$op(A) = A\f$) or \f$\max{(1, k)}\f$ (\f$op(A) = A^T\f$ or
 *              \f$op(A) = A^H\f$).
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_invalid_operation The operation is invalid if the matrix B and C has a
 *              different layout ordering.
 *  \retval     aoclsparse_status_wrong_type The data type of the matrices are not matching
 *              or invalid.
 *  \retval     aoclsparse_status_invalid_size The value of \p m, \p k, \p nnz, \p ldb or \p ldc
 *              is invalid.
 *  \retval     aoclsparse_status_invalid_pointer The pointer \p A, \p B, or \p C
 *              is invalid.
 *  \retval     aoclsparse_status_not_implemented The values of \p orderB and \p orderC are different.
 *
*/
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_ssyprd(const aoclsparse_operation op,
                                    const aoclsparse_matrix    A,
                                    const float               *B,
                                    const aoclsparse_order     orderB,
                                    const aoclsparse_int       ldb,
                                    const float                alpha,
                                    const float                beta,
                                    float                     *C,
                                    const aoclsparse_order     orderC,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_dsyprd(const aoclsparse_operation op,
                                    const aoclsparse_matrix    A,
                                    const double              *B,
                                    const aoclsparse_order     orderB,
                                    const aoclsparse_int       ldb,
                                    const double               alpha,
                                    const double               beta,
                                    double                    *C,
                                    const aoclsparse_order     orderC,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_csyprd(const aoclsparse_operation      op,
                                    const aoclsparse_matrix         A,
                                    const aoclsparse_float_complex *B,
                                    const aoclsparse_order          orderB,
                                    const aoclsparse_int            ldb,
                                    const aoclsparse_float_complex  alpha,
                                    const aoclsparse_float_complex  beta,
                                    aoclsparse_float_complex       *C,
                                    const aoclsparse_order          orderC,
                                    const aoclsparse_int            ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_zsyprd(const aoclsparse_operation       op,
                                    const aoclsparse_matrix          A,
                                    const aoclsparse_double_complex *B,
                                    const aoclsparse_order           orderB,
                                    const aoclsparse_int             ldb,
                                    const aoclsparse_double_complex  alpha,
                                    const aoclsparse_double_complex  beta,
                                    aoclsparse_double_complex       *C,
                                    const aoclsparse_order           orderC,
                                    const aoclsparse_int             ldc);
/**@}*/

/*! \ingroup level3_module
 *  \brief Multiplication of a sparse matrix and its transpose (or conjugate transpose) stored as a sparse matrix.
 *  \details
 *  \P{aoclsparse_syrk} multiplies a sparse matrix with its transpose (or conjugate transpose) in CSR storage format.
 *  The result is stored in a newly allocated sparse matrix in CSR format, such that
  \f[
 *    C := A \cdot op(A)
 *  \f]
 *  if \f$opA\f$ is \ref aoclsparse_operation_none.
 *
 *  Otherwise,
 *  \f[
 *    C := op(A) \cdot A,
 *  \f]
 *
 * where
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A^T, & \text{transpose of } {\bf\mathsf{A}} \text{ for real matrices}\\
 *         A^H, & \text{conjugate transpose of } {\bf\mathsf{A}} \text{ for complex matrices}\\
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  where \f$A\f$ is a  \f$m \times n\f$ matrix, opA is one of \ref aoclsparse_operation_none,
 * \ref aoclsparse_operation_transpose (for real matrices) or \ref aoclsparse_operation_conjugate_transpose
 * (for complex matrices). The output matrix \f$C\f$ is a sparse symmetric (or Hermitian) matrix stored as an
 *  upper triangular matrix in CSR format.
 *
 *  \note \p aoclsparse_syrk assumes that the input CSR matrix has sorted column
 *  indices in each row. If not, call aoclsparse_order_mat() before calling
 *  \p aoclsparse_syrk.
 *
 *  \note \p aoclsparse_syrk currently does not support \ref aoclsparse_operation_transpose for complex \p A.
 *
 *  @param[in]
 *  opA     Matrix \f$A\f$ operation type.
 *  @param[in]
 *  A        Sorted sparse CSR matrix \f$A\f$.

 *
 *  @param[out]
 *  *C        Pointer to the new sparse CSR symmetric/Hermitian matrix \f$C\f$.
 *            Only upper triangle of the result matrix is computed.
 *  	      The column indices of the output matrix in CSR format might be unsorted.
 *  	      The matrix should be freed by aoclsparse_destroy() when no longer needed.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer \p A, \p C is invalid.
 *  \retval     aoclsparse_status_wrong_type A and its operation type do not match.
 *  \retval     aoclsparse_status_not_implemented The input matrix is not in the CSR format or
 *              \p opA is aoclsparse_operation_transpose and \p A has complex values.
 *  \retval     aoclsparse_status_invalid_value The value of opA is invalid.
 *  \retval     aoclsparse_status_unsorted_input Input matrices are not sorted.
 *  \retval     aoclsparse_status_memory_error Memory allocation failure.
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_dsyrk.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_dsyrk.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 *
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_syrk(const aoclsparse_operation opA,
                                  const aoclsparse_matrix    A,
                                  aoclsparse_matrix         *C);
/**@}*/

/*! \ingroup level3_module
 *  \brief Multiplication of a sparse matrix and its transpose (or conjugate transpose) for all data types.
 *  \details
 *  \P{aoclsparse_syrkd} multiplies a sparse matrix with its transpose (or conjugate transpose) in CSR storage format.
 *  The result is stored in a dense format, such that
  \f[
 *    C := \alpha \cdot A \cdot op(A) + \beta \cdot C
 *  \f]
 *  if \f$opA\f$ is \ref aoclsparse_operation_none.
 *
 *  Otherwise,
 *  \f[
 *    C := \alpha \cdot op(A) \cdot A + \beta \cdot C
 *  \f]
 *
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *         A^T, & \text{transpose of } {\bf\mathsf{A}} \text{ for real matrices}\\
 *         A^H, & \text{conjugate transpose of } {\bf\mathsf{A}} \text{ for complex matrices}\\
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  where \f$A\f$ is a  \f$m \times n\f$ sparse matrix, \p opA is one of \ref aoclsparse_operation_none,
 * \ref aoclsparse_operation_transpose (for real matrices) or \ref aoclsparse_operation_conjugate_transpose
 * (for complex matrices). The output matrix \f$C\f$ is a dense symmetric (or Hermitian) matrix stored as an
 *  upper triangular matrix.
 *
 *  \note \p aoclsparse_syrkd assumes that the input CSR matrix has sorted column
 *  indices in each row. If not, call aoclsparse_order_mat() before calling
 *  \p aoclsparse_syrkd.
 *
 *  \note For complex type, only the real parts of \f$\alpha\f$ and \f$\beta\f$ are taken
 *  into account to preserve Hermitian \f$C\f$.
 *
 *  \note \p aoclsparse_syrkd currently does not support \ref aoclsparse_operation_transpose for complex \p A.
 *
 *  @param[in]
 *  opA     Matrix \f$A\f$ operation type.
 *  @param[in]
 *  A        Sorted sparse CSR matrix \f$A\f$.
 *  @param[in]
 *  alpha       Scalar \f$\alpha\f$.
 *  @param[in]
 *  beta        Scalar \f$\beta\f$.
 *  @param[in, out]
 *  C           Output dense matrix. Only upper triangular part of the matrix is processed
 *              during the computation, the strictly lower triangle is not modified.
 *  @param[in]
 *  orderC      Storage format of the output dense matrix, \f$C\f$.
 *              It can be \ref aoclsparse_order_row or \ref aoclsparse_order_column.
 *  @param[in]
 *  ldc         Leading dimension of \f$C\f$.
 *
 *  \retval     aoclsparse_status_success The operation completed successfully.
 *  \retval     aoclsparse_status_invalid_pointer \p A, \p C is invalid.
 *  \retval     aoclsparse_status_wrong_type \p A and its operation type do not match.
 *  \retval     aoclsparse_status_not_implemented The input matrix is not in the CSR format or
 *              \p opA is aoclsparse_operation_transpose and \p A has complex values.
 *  \retval     aoclsparse_status_invalid_value The value of \p opA, \p orderC or \p ldc is invalid.
 *  \retval     aoclsparse_status_unsorted_input Input matrix is not sorted.
 *  \retval     aoclsparse_status_memory_error Memory allocation failure.
 *
 *  @rst
 * .. collapse:: Example (tests/examples/sample_dsyrkd.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_dsyrkd.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 */

/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_ssyrkd(const aoclsparse_operation opA,
                                    const aoclsparse_matrix    A,
                                    const float                alpha,
                                    const float                beta,
                                    float                     *C,
                                    const aoclsparse_order     orderC,
                                    const aoclsparse_int       ldc);
DLL_PUBLIC
aoclsparse_status aoclsparse_dsyrkd(const aoclsparse_operation opA,
                                    const aoclsparse_matrix    A,
                                    const double               alpha,
                                    const double               beta,
                                    double                    *C,
                                    const aoclsparse_order     orderC,
                                    const aoclsparse_int       ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_csyrkd(const aoclsparse_operation     opA,
                                    const aoclsparse_matrix        A,
                                    const aoclsparse_float_complex alpha,
                                    const aoclsparse_float_complex beta,
                                    aoclsparse_float_complex      *C,
                                    const aoclsparse_order         orderC,
                                    const aoclsparse_int           ldc);

DLL_PUBLIC
aoclsparse_status aoclsparse_zsyrkd(const aoclsparse_operation      opA,
                                    const aoclsparse_matrix         A,
                                    const aoclsparse_double_complex alpha,
                                    const aoclsparse_double_complex beta,
                                    aoclsparse_double_complex      *C,
                                    const aoclsparse_order          orderC,
                                    const aoclsparse_int            ldc);

/**@}*/

// ====================================================================================
// Undocumented public interfaces with KID. Used for testing and benchmarking
// different kernels in the API
// ====================================================================================

DLL_PUBLIC
aoclsparse_status aoclsparse_saxpyi_kid(const aoclsparse_int  nnz,
                                        const float           a,
                                        const float          *x,
                                        const aoclsparse_int *indx,
                                        float                *y,
                                        aoclsparse_int        kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_zaxpyi_kid(const aoclsparse_int  nnz,
                                        const void           *a,
                                        const void           *x,
                                        const aoclsparse_int *indx,
                                        void                 *y,
                                        aoclsparse_int        kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_caxpyi_kid(const aoclsparse_int  nnz,
                                        const void           *a,
                                        const void           *x,
                                        const aoclsparse_int *indx,
                                        void                 *y,
                                        aoclsparse_int        kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_daxpyi_kid(const aoclsparse_int  nnz,
                                        const double          a,
                                        const double         *x,
                                        const aoclsparse_int *indx,
                                        double               *y,
                                        aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_cdotci_kid(const aoclsparse_int  nnz,
                                                   const void           *x,
                                                   const aoclsparse_int *indx,
                                                   const void           *y,
                                                   void                 *dot,
                                                   aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_zdotci_kid(const aoclsparse_int  nnz,
                                                   const void           *x,
                                                   const aoclsparse_int *indx,
                                                   const void           *y,
                                                   void                 *dot,
                                                   aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_cdotui_kid(const aoclsparse_int  nnz,
                                                   const void           *x,
                                                   const aoclsparse_int *indx,
                                                   const void           *y,
                                                   void                 *dot,
                                                   aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_zdotui_kid(const aoclsparse_int  nnz,
                                                   const void           *x,
                                                   const aoclsparse_int *indx,
                                                   const void           *y,
                                                   void                 *dot,
                                                   aoclsparse_int        kid);

DLL_PUBLIC float aoclsparse_sdoti_kid(const aoclsparse_int  nnz,
                                      const float          *x,
                                      const aoclsparse_int *indx,
                                      const float          *y,
                                      aoclsparse_int        kid);

DLL_PUBLIC double aoclsparse_ddoti_kid(const aoclsparse_int  nnz,
                                       const double         *x,
                                       const aoclsparse_int *indx,
                                       const double         *y,
                                       aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_sgthr_kid(const aoclsparse_int  nnz,
                                                  const float          *y,
                                                  float                *x,
                                                  const aoclsparse_int *indx,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_dgthr_kid(const aoclsparse_int  nnz,
                                                  const double         *y,
                                                  double               *x,
                                                  const aoclsparse_int *indx,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_cgthr_kid(const aoclsparse_int  nnz,
                                                  const void           *y,
                                                  void                 *x,
                                                  const aoclsparse_int *indx,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_zgthr_kid(const aoclsparse_int  nnz,
                                                  const void           *y,
                                                  void                 *x,
                                                  const aoclsparse_int *indx,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_sgthrz_kid(
    const aoclsparse_int nnz, float *y, float *x, const aoclsparse_int *indx, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_dgthrz_kid(
    const aoclsparse_int nnz, double *y, double *x, const aoclsparse_int *indx, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_cgthrz_kid(
    const aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_zgthrz_kid(
    const aoclsparse_int nnz, void *y, void *x, const aoclsparse_int *indx, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_sgthrs_kid(
    const aoclsparse_int nnz, const float *y, float *x, aoclsparse_int stride, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_dgthrs_kid(const aoclsparse_int nnz,
                                                   const double        *y,
                                                   double              *x,
                                                   aoclsparse_int       stride,
                                                   aoclsparse_int       kid);

DLL_PUBLIC aoclsparse_status aoclsparse_cgthrs_kid(
    const aoclsparse_int nnz, const void *y, void *x, aoclsparse_int stride, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_zgthrs_kid(
    const aoclsparse_int nnz, const void *y, void *x, aoclsparse_int stride, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_sroti_kid(const aoclsparse_int  nnz,
                                                  float                *x,
                                                  const aoclsparse_int *indx,
                                                  float                *y,
                                                  const float           c,
                                                  const float           s,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_droti_kid(const aoclsparse_int  nnz,
                                                  double               *x,
                                                  const aoclsparse_int *indx,
                                                  double               *y,
                                                  const double          c,
                                                  const double          s,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_csctr_kid(const aoclsparse_int  nnz,
                                                  const void           *x,
                                                  const aoclsparse_int *indx,
                                                  void                 *y,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_zsctr_kid(const aoclsparse_int  nnz,
                                                  const void           *x,
                                                  const aoclsparse_int *indx,
                                                  void                 *y,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_ssctr_kid(const aoclsparse_int  nnz,
                                                  const float          *x,
                                                  const aoclsparse_int *indx,
                                                  float                *y,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_dsctr_kid(const aoclsparse_int  nnz,
                                                  const double         *x,
                                                  const aoclsparse_int *indx,
                                                  double               *y,
                                                  aoclsparse_int        kid);

DLL_PUBLIC aoclsparse_status aoclsparse_csctrs_kid(
    const aoclsparse_int nnz, const void *x, aoclsparse_int stride, void *y, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_zsctrs_kid(
    const aoclsparse_int nnz, const void *x, aoclsparse_int stride, void *y, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_ssctrs_kid(
    const aoclsparse_int nnz, const float *x, aoclsparse_int stride, float *y, aoclsparse_int kid);

DLL_PUBLIC aoclsparse_status aoclsparse_dsctrs_kid(const aoclsparse_int nnz,
                                                   const double        *x,
                                                   aoclsparse_int       stride,
                                                   double              *y,
                                                   aoclsparse_int       kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_zcsrmm_kid(aoclsparse_operation             op,
                                        const aoclsparse_double_complex  alpha,
                                        const aoclsparse_matrix          A,
                                        const aoclsparse_mat_descr       descr,
                                        aoclsparse_order                 order,
                                        const aoclsparse_double_complex *B,
                                        aoclsparse_int                   n,
                                        aoclsparse_int                   ldb,
                                        const aoclsparse_double_complex  beta,
                                        aoclsparse_double_complex       *C,
                                        aoclsparse_int                   ldc,
                                        const aoclsparse_int             kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_ccsrmm_kid(aoclsparse_operation            op,
                                        const aoclsparse_float_complex  alpha,
                                        const aoclsparse_matrix         A,
                                        const aoclsparse_mat_descr      descr,
                                        aoclsparse_order                order,
                                        const aoclsparse_float_complex *B,
                                        aoclsparse_int                  n,
                                        aoclsparse_int                  ldb,
                                        const aoclsparse_float_complex  beta,
                                        aoclsparse_float_complex       *C,
                                        aoclsparse_int                  ldc,
                                        const aoclsparse_int            kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_dcsrmm_kid(aoclsparse_operation       op,
                                        const double               alpha,
                                        const aoclsparse_matrix    A,
                                        const aoclsparse_mat_descr descr,
                                        aoclsparse_order           order,
                                        const double              *B,
                                        aoclsparse_int             n,
                                        aoclsparse_int             ldb,
                                        const double               beta,
                                        double                    *C,
                                        aoclsparse_int             ldc,
                                        const aoclsparse_int       kid);

DLL_PUBLIC
aoclsparse_status aoclsparse_scsrmm_kid(aoclsparse_operation       op,
                                        const float                alpha,
                                        const aoclsparse_matrix    A,
                                        const aoclsparse_mat_descr descr,
                                        aoclsparse_order           order,
                                        const float               *B,
                                        aoclsparse_int             n,
                                        aoclsparse_int             ldb,
                                        const float                beta,
                                        float                     *C,
                                        aoclsparse_int             ldc,
                                        const aoclsparse_int       kid);
#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_FUNCTIONS_H_
