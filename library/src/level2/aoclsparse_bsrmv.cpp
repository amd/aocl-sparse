/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse.h"
#include "aoclsparse_bsrmv.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

extern "C" aoclsparse_status aoclsparse_sbsrmv(aoclsparse_operation       trans,
        const float*              alpha,
        aoclsparse_int             mb,
        aoclsparse_int             nb,
        aoclsparse_int             bsr_dim,
        const float*              bsr_val,
        const aoclsparse_int*      bsr_col_ind,
        const aoclsparse_int*      bsr_row_ptr,
        const aoclsparse_mat_descr descr,
        const float*             x,
        const float*             beta,
        float*                   y
        )
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    if(descr->type != aoclsparse_matrix_type_general)
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
    if(mb < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(nb < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(bsr_dim < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 )
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_col_ind == nullptr)
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

    if(bsr_dim == 2)
        return aoclsparse_bsrmv_2x2(*alpha,
                mb,
                nb,
                bsr_val,
                bsr_col_ind,
                bsr_row_ptr,
                x,
                *beta,
                y);
    else if(bsr_dim == 3)
        return aoclsparse_bsrmv_3x3(*alpha,
                mb,
                nb,
                bsr_val,
                bsr_col_ind,
                bsr_row_ptr,
                x,
                *beta,
                y);
    else if(bsr_dim == 4)
        return aoclsparse_bsrmv_4x4(*alpha,
                mb,
                nb,
                bsr_val,
                bsr_col_ind,
                bsr_row_ptr,
                x,
                *beta,
                y);
    else
        return aoclsparse_bsrmv_general(*alpha,
                mb,
                nb,
                bsr_dim,
                bsr_val,
                bsr_col_ind,
                bsr_row_ptr,
                x,
                *beta,
                y);
}

extern "C" aoclsparse_status aoclsparse_dbsrmv(aoclsparse_operation       trans,
        const double*              alpha,
        aoclsparse_int             mb,
        aoclsparse_int             nb,
        aoclsparse_int             bsr_dim,
        const double*              bsr_val,
        const aoclsparse_int*      bsr_col_ind,
        const aoclsparse_int*      bsr_row_ptr,
        const aoclsparse_mat_descr descr,
        const double*             x,
        const double*             beta,
        double*                   y
        )
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero)
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    if(descr->type != aoclsparse_matrix_type_general)
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
    if(mb < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(nb < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(bsr_dim < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(mb == 0 || nb == 0 )
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(bsr_val == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_col_ind == nullptr)
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
    if(bsr_dim == 2)
        return aoclsparse_bsrmv_2x2(*alpha,
                mb,
                nb,
                bsr_val,
                bsr_col_ind,
                bsr_row_ptr,
                x,
                *beta,
                y);
    else if(bsr_dim == 3)
        return aoclsparse_bsrmv_3x3(*alpha,
                mb,
                nb,
                bsr_val,
                bsr_col_ind,
                bsr_row_ptr,
                x,
                *beta,
                y);
    else if(bsr_dim == 4)
        return aoclsparse_bsrmv_4x4(*alpha,
                mb,
                nb,
                bsr_val,
                bsr_col_ind,
                bsr_row_ptr,
                x,
                *beta,
                y);
    else
        return aoclsparse_bsrmv_general(*alpha,
                mb,
                nb,
                bsr_dim,
                bsr_val,
                bsr_col_ind,
                bsr_row_ptr,
                x,
                *beta,
                y);
}

