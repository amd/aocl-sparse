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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_MTX_DISPATCH_HPP
#define AOCLSPARSE_MTX_DISPATCH_HPP
#include "aoclsparse.h"

#include <complex>
#include <utility>

namespace aoclsparse
{
    /* DOID = Descriptor + Operation ID, it flattens down all the supported combinations,
    one DOID defines quite well what a kernel needs to do, thus it can be used
    as index to a table of kernels */
    enum class doid
    {
        gn = 0, // general normal (non-transpose) full marix as it is given
        gt, // transposed
        gh, // conjugate transpose

        // symmetric matrix: transpose/non-transpose is the same, conjugate transpose not supported
        sl, // symmetric based on L triangle
        su, // symmetix based on U triangle

        // hermitian matrix
        hl, // hermitian based on L
        hu, // hermitian based on U

        // triangular matrix
        tln, // triangular based on L, normal
        tlt, // triangular based on L, transposed
        tlh, // triangular based on L, conjugate transposed
        tun, // triangular based on U, normal
        tut, // triangular based on U, transposed
        tuh, // triangular based on U, conjugate transposed

        len = tuh + 1

        // All other doid values are invalid
    };

    /* Given descriptor and operation, return DOID or not_implemented error for complex types*/
    template <typename T>
    inline aoclsparse::doid get_doid(const aoclsparse_mat_descr descr,
                                     const aoclsparse_operation op)
    {
        aoclsparse::doid d_id = doid::len;
        aoclsparse_int   op_v = op - 111;

        switch(descr->type)
        {
        case aoclsparse_matrix_type_general:
            // d_id [0,2]
            d_id = static_cast<aoclsparse::doid>(op_v);
            break;
        case aoclsparse_matrix_type_symmetric:
            // d_id [3,4]
            d_id = static_cast<aoclsparse::doid>(3 + (descr->fill_mode));
            break;
        case aoclsparse_matrix_type_hermitian:
            // d_id [5,6]
            d_id = static_cast<aoclsparse::doid>(5 + (descr->fill_mode));
            break;
        case aoclsparse_matrix_type_triangular:
            // d_id [7,12]
            d_id = static_cast<aoclsparse::doid>(7 + (3 * (descr->fill_mode)) + op_v);
            break;
        default:
            break;
        }

        return d_id;
    }
}

#endif
