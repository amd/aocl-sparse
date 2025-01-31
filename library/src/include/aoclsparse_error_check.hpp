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
#ifndef AOCLSPARSE_ERROR_CHECK_HPP
#define AOCLSPARSE_ERROR_CHECK_HPP
#include "aoclsparse.h"
#include "aoclsparse_descr.h"

namespace aoclsparse
{

    // Checks if the index base is a valid value
    inline bool is_valid_base(aoclsparse_index_base bs)
    {
        if((bs != aoclsparse_index_base_zero && bs != aoclsparse_index_base_one))
            return false;

        return true;
    }

    // Checks if the matrix type is a valid value
    inline bool is_valid_mtx_t(aoclsparse_matrix_type mtx_t)
    {
        if((mtx_t != aoclsparse_matrix_type_general && mtx_t != aoclsparse_matrix_type_symmetric
            && mtx_t != aoclsparse_matrix_type_triangular
            && mtx_t != aoclsparse_matrix_type_hermitian))
            return false;

        return true;
    }

    // Checks if the fill mode is a valid value
    inline bool is_valid_fill_mode(aoclsparse_fill_mode fm)
    {
        if(fm != aoclsparse_fill_mode_lower && fm != aoclsparse_fill_mode_upper)
            return false;

        return true;
    }

    // Checks if the diag type is a valid value
    inline bool is_valid_diag_type(aoclsparse_diag_type diag_t)
    {
        if(diag_t != aoclsparse_diag_type_non_unit && diag_t != aoclsparse_diag_type_unit
           && diag_t != aoclsparse_diag_type_zero)
            return false;

        return true;
    }

    // Checks if the value inside the matrix descriptor is a valid value
    inline bool is_valid_descr(aoclsparse_mat_descr desc)
    {
        if(is_valid_base(desc->base) && is_valid_mtx_t(desc->type)
           && is_valid_fill_mode(desc->fill_mode) && is_valid_diag_type(desc->diag_type))
        {
            return true;
        }

        return false;
    }

    // Checks if the operation is a valid value
    inline bool is_valid_op(aoclsparse_operation op)
    {
        if((op != aoclsparse_operation_none) && (op != aoclsparse_operation_transpose)
           && (op != aoclsparse_operation_conjugate_transpose))
        {
            return false;
        }

        return true;
    }
}
#endif