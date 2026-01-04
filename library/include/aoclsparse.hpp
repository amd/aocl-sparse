/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#ifndef AOCLSPARSE_HPP
#define AOCLSPARSE_HPP
#pragma once
#include "aoclsparse.h"

#include <string>
namespace aoclsparse
{
    /** \ingroup level2_module
    *  \details
    *  \p aoclsparse::trsv is the C++ interface to <tt>aoclsparse_?trsv</tt> and it performs a sparse
    *  triangular solve using the provided input parameters.
    *  @tparam
    *  T           Data type supported for \p T are double, float, std::complex<double>
    *              or std::complex<float>
    *
    *
     *
     * @rst
     *
     * .. collapse:: Example - Complex space (tests/examples/sample_trsv_cpp.cpp)
     *
     *      .. only:: html
     *
     *         .. literalinclude:: ../tests/examples/sample_trsv_cpp.cpp
     *            :language: C++
     *            :linenos:
     *
     * @endrst
     *
     *@{
     */
    template <typename T>
    aoclsparse_status trsv(const aoclsparse_operation trans,
                           const T                    alpha,
                           aoclsparse_matrix          A,
                           const aoclsparse_mat_descr descr,
                           const T                   *b,
                           const aoclsparse_int       incb,
                           T                         *x,
                           const aoclsparse_int       incx,
                           aoclsparse_int             kid = -1);
    /**@}*/

    /*! \ingroup level2_module
    *  \brief C++ function to compute sparse matrix-vector multiplication for real/complex single and double data precisions.
    *
    *  \details
    *  \p aoclsparse::mv is the C++ interface to <tt>aoclsparse_?mv</tt> that computes
    *   sparse matrix-vector multiplication using the provided input parameters.
    *  @tparam
    *  T           Data type supported for \p T are double, float, std::complex<double> and std::complex<float>
    *
    * @rst
    * .. collapse:: Example - C++ (tests/examples/sample_mv_cpp.cpp)
    *
    *    .. only:: html
    *
    *       .. literalinclude:: ../tests/examples/sample_mv_cpp.cpp
    *          :language: C++
    *          :linenos:
    *
    * @endrst
    * @{
    */
    template <typename T>
    aoclsparse_status mv(aoclsparse_operation       op,
                         const T                   *alpha,
                         aoclsparse_matrix          A,
                         const aoclsparse_mat_descr descr,
                         const T                   *x,
                         const T                   *beta,
                         T                         *y);
    /**@}*/

    /*! \ingroup aux_module
    *  \details
    *  \p aoclsparse::create_csr is the C++ interface to <tt>aoclsparse_create_?csr</tt> that creates
    *  \ref aoclsparse_matrix and initializes it with provided input parameters.
    *  @tparam
    *  T           Data type supported for \p T are double, float, std::complex<double>
    *              or std::complex<float>
    */
    /**@{*/
    template <typename T>
    aoclsparse_status create_csr(aoclsparse_matrix    *mat,
                                 aoclsparse_index_base base,
                                 aoclsparse_int        M,
                                 aoclsparse_int        N,
                                 aoclsparse_int        nnz,
                                 aoclsparse_int       *row_ptr,
                                 aoclsparse_int       *col_idx,
                                 T                    *val,
                                 bool                  fast_chck = false);

    /*! \ingroup level3_module
    *  \details
    *  \p aoclsparse::sp2m is the C++ interface to \ref aoclsparse_sp2m that multiplies two
    *   sparse matrices in CSR storage format and the result is stored in a newly allocated sparse
    *   matrix in CSR format.
    *  @tparam
    *  T           Data type supported for \p T are double, float, std::complex<double>
    *              or std::complex<float>
    * @rst
    * .. collapse:: Example - Complex space (tests/examples/sample_csr2m_cpp.cpp)
    *
    *    .. only:: html
    *
    *       .. literalinclude:: ../tests/examples/sample_csr2m_cpp.cpp
    *          :language: C++
    *          :linenos:
    * @endrst
    */
    /**@{*/
    template <typename T>
    aoclsparse_status sp2m(aoclsparse_operation       opA,
                           const aoclsparse_mat_descr descrA,
                           const aoclsparse_matrix    A,
                           aoclsparse_operation       opB,
                           const aoclsparse_mat_descr descrB,
                           const aoclsparse_matrix    B,
                           aoclsparse_request         request,
                           aoclsparse_matrix         *C);

    /**@}*/

    /*! \ingroup aux_module
    *  \details
    *  \p aoclsparse::create_bsr is the C++ interface to <tt>aoclsparse_create_?bsr</tt> that creates \ref aoclsparse_matrix
    *   in BSR format and initializes it with the provided input parameters.
    *  @tparam
    *  T           Data type supported for \p T are double, float, std::complex<double>
    *              or std::complex<float>
    */
    /**@{*/
    template <typename T>
    aoclsparse_status create_bsr(aoclsparse_matrix          *mat,
                                 const aoclsparse_index_base base,
                                 const aoclsparse_order      order,
                                 const aoclsparse_int        bM,
                                 const aoclsparse_int        bN,
                                 const aoclsparse_int        block_dim,
                                 aoclsparse_int             *row_ptr,
                                 aoclsparse_int             *col_idx,
                                 T                          *val,
                                 bool                        fast_chck = false);
    /**@}*/

    // Test wrappers for unit tests
    // All test wrappers will be prefixed with "t_"
    namespace test
    {
        template <typename T>
        aoclsparse_int dispatcher(std::string    dispatch,
                                  aoclsparse_int kid   = -1,
                                  aoclsparse_int begin = 0,
                                  aoclsparse_int end   = 0);
    } // namespace test_wrapper
} // namespace aoclsparse

#endif
