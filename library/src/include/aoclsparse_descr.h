/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef AOCLSPARSE_DESCR_H
#define AOCLSPARSE_DESCR_H

#include "aoclsparse.h"

/********************************************************************************
 * \brief aoclsparse_mat_descr is a structure holding the aoclsparse matrix
 * descriptor. It must be initialized using aoclsparse_create_mat_descr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using aoclsparse_destroy_mat_descr().
 *******************************************************************************/
struct _aoclsparse_mat_descr
{
    // matrix type
    aoclsparse_matrix_type type = aoclsparse_matrix_type_general;
    // fill mode
    aoclsparse_fill_mode fill_mode = aoclsparse_fill_mode_lower;
    // diagonal type
    aoclsparse_diag_type diag_type = aoclsparse_diag_type_non_unit;
    // index base
    aoclsparse_index_base base = aoclsparse_index_base_zero;
};

#endif // AOCLSPARSE_DESCR_H
