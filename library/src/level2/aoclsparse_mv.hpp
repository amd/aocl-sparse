/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_OPTMV_HPP
#define AOCLSPARSE_OPTMV_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include <immintrin.h>

#include<iostream>

aoclsparse_status aoclsparse_mv_template(aoclsparse_operation       op,
                                    const float                alpha,
                                    aoclsparse_matrix A,
                                    const aoclsparse_mat_descr descr,
                                    const float*             x,
                                    const float              beta,
                                    float*                   y )
{
    // ToDo: float version need to be implemented
    return aoclsparse_status_not_implemented;
}

aoclsparse_status aoclsparse_mv_template(aoclsparse_operation       op,
                                    const double                alpha,
                                    aoclsparse_matrix A,
                                    const aoclsparse_mat_descr descr,
                                    const double*             x,
                                    const double              beta,
                                    double*                   y )
{
 
    if (A->mat_type == aoclsparse_csr_mat) {
        //Invoke SPMV API for CSR storage format(double precision)
        aoclsparse_dcsrmv(op,
            &alpha,
            A->m,
            A->n,
            A->nnz,
            (double *) A->csr_mat.csr_val,
            A->csr_mat.csr_col_ptr,
            A->csr_mat.csr_row_ptr,
            descr,
            x,
            &beta,
            y);
    } else if (A->mat_type == aoclsparse_ellt_csr_hyb_mat) {
        aoclsparse_dellthybmv(op,
            &alpha,
            A->m,
            A->n,
            A->nnz,
            (double*) A->ell_csr_hyb_mat.ell_val,
            A->ell_csr_hyb_mat.ell_col_ind,
            A->ell_csr_hyb_mat.ell_width,
            A->ell_csr_hyb_mat.ell_m,
            (double*) A->ell_csr_hyb_mat.csr_val,
            A->csr_mat.csr_row_ptr,
            A->csr_mat.csr_col_ptr,
            nullptr,
            A->ell_csr_hyb_mat.csr_row_id_map,
            descr,
            x,
            &beta,
            y );

    }

    return aoclsparse_status_success;
}


#endif // AOCLSPARSE_OPTMV_HPP

