/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef AOCLSPARSE_MAT_CSR_H
#define AOCLSPARSE_MAT_CSR_H

#include "aoclsparse.h"

/********************************************************************************
 * \brief aoclsparse_mat_csr is a structure holding the aoclsparse matrix
 * in csr format. It must be initialized using aoclsparse_create_mat_csr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using aoclsparse_destroy_mat_csr().
 *******************************************************************************/
struct _aoclsparse_mat_csr
{
    // num rows
    aoclsparse_int m = 0;
    // num cols
    aoclsparse_int n = 0;

    // CSR matrix part
    aoclsparse_int  csr_nnz     = 0;
    aoclsparse_int* csr_row_ptr = nullptr;
    aoclsparse_int* csr_col_ind = nullptr;
    void*          csr_val     = nullptr;
};

#endif // AOCLSPARSE_MATRIX_H
