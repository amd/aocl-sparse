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
#include "aoclsparse_descr.h"

#include <complex>
#include <utility>

namespace aoclsparse
{
    /* DOID = Descriptor + Operation ID, it flattens down all the supported combinations,
    one DOID defines quite well what a kernel needs to do, thus it can be used
    as index to a table of kernels
    To maintain "symmetry" (see trans_doid()) between transposed-non-transposed
    kernels, an additional operation "conjugate" has been added (all doid::*c). */
    enum class doid
    {
        gn = 0, // general normal (non-transpose) full marix as it is given
        gt, // transposed
        gh, // conjugate transpose
        gc, // general matrix conjugate

        // symmetric matrix: transpose/non-transpose is the same, conjugate transpose not supported
        sl, // symmetric based on L triangle
        su, // symmetix based on U triangle

        // hermitian matrix
        hl, // hermitian based on L
        hlc, // hermitian conjugate lower
        hu, // hermitian based on U
        huc, // hermitian conjugate upper

        // triangular matrix
        tln, // triangular based on L, normal
        tlt, // triangular based on L, transposed
        tlh, // triangular based on L, conjugate transposed
        tlc, // triangular lower matrix conjugate
        tun, // triangular based on U, normal
        tut, // triangular based on U, transposed
        tuh, // triangular based on U, conjugate transposed
        tuc, // traingular upper matrix conjugate

        len = tuc + 1 // number of valid DOIDs, also used
        // to indicate an invalid DOID
    };

    /* Given descriptor and operation, return DOID or not_implemented error for complex types*/
    template <typename T>
    inline aoclsparse::doid get_doid(const aoclsparse_mat_descr descr,
                                     const aoclsparse_operation op)
    {
        aoclsparse::doid       d_id  = doid::len;
        aoclsparse_operation   op_t  = op;
        aoclsparse_matrix_type mtx_t = descr->type;
        aoclsparse_fill_mode   fm    = descr->fill_mode;

        // For real types, simplify the operation and matrix type
        if constexpr(std::is_same_v<float, T> || std::is_same_v<double, T>)
        {
            if(op_t == aoclsparse_operation_conjugate_transpose)
                op_t = aoclsparse_operation_transpose;

            if(mtx_t == aoclsparse_matrix_type_hermitian)
                mtx_t = aoclsparse_matrix_type_symmetric;
        }

        aoclsparse_int op_v = op_t - 111;

        switch(mtx_t)
        {
        case aoclsparse_matrix_type_general:
            // d_id [0,3]
            d_id = static_cast<aoclsparse::doid>(op_v);
            break;
        case aoclsparse_matrix_type_symmetric:
            // d_id [4,5]
            d_id = static_cast<aoclsparse::doid>(4 + fm);
            break;
        case aoclsparse_matrix_type_hermitian:
            // d_id [6,9]
            d_id = static_cast<aoclsparse::doid>(6 + (2 * fm));
            break;
        case aoclsparse_matrix_type_triangular:
            // d_id [10,17]
            d_id = static_cast<aoclsparse::doid>(10 + (4 * fm) + op_v);
            break;
        default:
            break;
        }

        return d_id;
    }

    /* Return DOID matching the original operation assuming that the input
       is already transposed. This is useful for binding CSC and CSR kernels.

       The data reporesentation of CSC matrix is exactly the same as CSR transposed,
       thus using doid::gt kernel on CSC data is the same as doid::gn on CSR.
       Similarly gh<->gc, sl<->su, tln<->tut, etc. */
    inline aoclsparse::doid trans_doid(aoclsparse::doid d_id)
    {
        // Return early for invalid value
        aoclsparse_int dv = static_cast<aoclsparse_int>(d_id);
        if(dv < 0 || dv >= static_cast<aoclsparse_int>(doid::len))
            return doid::len;

        // clang-format off
        static constexpr aoclsparse::doid tr_v[] = {doid::gt,  doid::gn,  doid::gc,  doid::gh,
                                                    doid::su,  doid::sl,  doid::huc, doid::hu,
                                                    doid::hlc, doid::hl,  doid::tut, doid::tun,
                                                    doid::tuc, doid::tuh, doid::tlt, doid::tln,
                                                    doid::tlc, doid::tlh};
        // clang-format on
        return tr_v[dv];
    }
}

#endif
