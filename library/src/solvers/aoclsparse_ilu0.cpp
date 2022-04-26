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

#include "aoclsparse.h"
#include "aoclsparse_ilu0.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
extern "C" aoclsparse_status aoclsparse_silu0(aoclsparse_operation       trans,
                                   aoclsparse_int               m,
                                   aoclsparse_int               n,
                                   aoclsparse_int               nnz,
                                   aoclsparse_int*              lu_diag_ptr,
                                   aoclsparse_int*              col_idx_mapper,
                                   bool*                        is_ilu0_factorized,
                                   const aoclsparse_ilu_type*   ilu_fact_type,
                                   float*                       csr_val,
                                   const aoclsparse_int*        csr_row_ind,
                                   const aoclsparse_int*        csr_col_ind,
                                   const aoclsparse_mat_descr   descr,
                                   const float*                 diag,
                                   const float*                 approx_inv_diag,                                       
                                   float*                       x,
                                   const float*                 b )
{
    return aoclsparse_ilu0_template(m,
                            n,
                            nnz,
                            lu_diag_ptr,
                            col_idx_mapper,
                            is_ilu0_factorized,
                            ilu_fact_type,
                            csr_val,
                            csr_row_ind,
                            csr_col_ind,
                            diag,
                            approx_inv_diag,
                            x,
                            b);                           

}

extern "C" aoclsparse_status aoclsparse_dilu0(aoclsparse_operation          trans,
                                            aoclsparse_int                  m,
                                            aoclsparse_int                  n,
                                            aoclsparse_int                  nnz,
                                            aoclsparse_int*                 lu_diag_ptr,
                                            aoclsparse_int*                 col_idx_mapper,
                                            bool*                           is_ilu0_factorized,
                                            const aoclsparse_ilu_type*      ilu_fact_type,
                                            double*                         csr_val,
                                            const aoclsparse_int*           csr_row_ind,
                                            const aoclsparse_int*           csr_col_ind,
                                            const aoclsparse_mat_descr      descr,
                                            const double*                   diag,
                                            const double*                   approx_inv_diag,                                        
                                            double*                         x,
                                            const double*                   b )
{
    return aoclsparse_ilu0_template(m,
                            n,
                            nnz,
                            lu_diag_ptr,
                            col_idx_mapper,
                            is_ilu0_factorized,
                            ilu_fact_type,
                            csr_val,
                            csr_row_ind,
                            csr_col_ind,
                            diag,
                            approx_inv_diag,                            
                            x,
                            b);                           

}

