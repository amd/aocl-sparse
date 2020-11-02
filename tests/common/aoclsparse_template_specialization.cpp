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

#include "aoclsparse.hpp"

#include <aoclsparse.h>

template <>
aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                 const float*          alpha,
                                 aoclsparse_int             m,
                                 aoclsparse_int             n,
                                 aoclsparse_int             nnz,
                                 const float*             csr_val,
                                 const aoclsparse_int*      csr_col_ind,
                                 const aoclsparse_int*      csr_row_ptr,
                                 const aoclsparse_mat_descr descr,
                                 const float*             x,
                                 const float*             beta,
                                 float*                   y)
{
    return aoclsparse_scsrmv(trans,
                            alpha,
                            m,
                            n,
                            nnz,
                            csr_val,
                            csr_col_ind,
                            csr_row_ptr,
                            descr,
                            x,
                            beta,
                            y);
}

template <>
aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                 const double*          alpha,
                                 aoclsparse_int             m,
                                 aoclsparse_int             n,
                                 aoclsparse_int             nnz,
                                 const double*             csr_val,
                                 const aoclsparse_int*      csr_col_ind,
                                 const aoclsparse_int*      csr_row_ptr,
                                 const aoclsparse_mat_descr descr,
                                 const double*             x,
                                 const double*             beta,
                                 double*                   y)
{
    return aoclsparse_dcsrmv(trans,
                            alpha,
                            m,
                            n,
                            nnz,
                            csr_val,
                            csr_col_ind,
                            csr_row_ptr,
                            descr,
                            x,
                            beta,
                            y);
}

template <>
aoclsparse_status aoclsparse_ellmv(aoclsparse_operation       trans,
                                 const float*             alpha,
                                 aoclsparse_int             m,
                                 aoclsparse_int             n,
                                 aoclsparse_int             nnz,
                                 const float*             ell_val,
                                 const aoclsparse_int*      ell_col_ind,
                                 const aoclsparse_int      ell_width,
                                 const aoclsparse_mat_descr descr,
                                 const float*             x,
                                 const float*             beta,
                                 float*                   y)
{
    return aoclsparse_sellmv(trans,
                            alpha,
                            m,
                            n,
                            nnz,
                            ell_val,
                            ell_col_ind,
                            ell_width,
                            descr,
                            x,
                            beta,
                            y);

}

template <>
aoclsparse_status aoclsparse_ellmv(aoclsparse_operation       trans,
                                 const double*             alpha,
                                 aoclsparse_int             m,
                                 aoclsparse_int             n,
                                 aoclsparse_int             nnz,
                                 const double*             ell_val,
                                 const aoclsparse_int*      ell_col_ind,
                                 const aoclsparse_int      ell_width,
                                 const aoclsparse_mat_descr descr,
                                 const double*             x,
                                 const double*             beta,
                                 double*                   y)
{
    return aoclsparse_dellmv(trans,
                            alpha,
                            m,
                            n,
                            nnz,
                            ell_val,
                            ell_col_ind,
                            ell_width,
                            descr,
                            x,
                            beta,
                            y);

}

template <>
aoclsparse_status aoclsparse_diamv(aoclsparse_operation       trans,
                                const float*              alpha,
                                aoclsparse_int             m,
                                aoclsparse_int             n,
                                aoclsparse_int             nnz,
                                const float*              dia_val,
                                const aoclsparse_int*      dia_offset,
                                aoclsparse_int      dia_num_diag,
                                const aoclsparse_mat_descr descr,
                                const float*             x,
                                const float*            beta,
                                float*                   y )
{
    return aoclsparse_sdiamv(trans,
                            alpha,
                            m,
                            n,
                            nnz,
                            dia_val,
                            dia_offset,
                            dia_num_diag,
                            descr,
                            x,
                            beta,
                            y);
}

template <>
aoclsparse_status aoclsparse_diamv(aoclsparse_operation       trans,
                                const double*              alpha,
                                aoclsparse_int             m,
                                aoclsparse_int             n,
                                aoclsparse_int             nnz,
                                const double*              dia_val,
                                const aoclsparse_int*      dia_offset,
                                aoclsparse_int      dia_num_diag,
                                const aoclsparse_mat_descr descr,
                                const double*             x,
                                const double*            beta,
                                double*                   y )
{
    return aoclsparse_ddiamv(trans,
                            alpha,
                            m,
                            n,
                            nnz,
                            dia_val,
                            dia_offset,
                            dia_num_diag,
                            descr,
                            x,
                            beta,
                            y);

}

template <>
aoclsparse_status aoclsparse_bsrmv(aoclsparse_operation       trans,
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
                                float*                   y)
{
    return aoclsparse_sbsrmv(trans,
                            alpha,
                            mb,
                            nb,
                            bsr_dim,
                            bsr_val,
                            bsr_col_ind,
                            bsr_row_ptr,
                            descr,
                            x,
                            beta,
                            y);
}

template <>
aoclsparse_status aoclsparse_bsrmv(aoclsparse_operation       trans,
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
                                double*                   y)
{
    return aoclsparse_dbsrmv(trans,
                            alpha,
                            mb,
                            nb,
                            bsr_dim,
                            bsr_val,
                            bsr_col_ind,
                            bsr_row_ptr,
                            descr,
                            x,
                            beta,
                            y);

}

template <>
aoclsparse_status aoclsparse_csrsv(aoclsparse_operation       trans,
                                const float*              alpha,
                                aoclsparse_int             m,
                                const float*              csr_val,
                                const aoclsparse_int*      csr_col_ind,
                                const aoclsparse_int*      csr_row_ptr,
                                const aoclsparse_mat_descr descr,
                                const float*             x,
                                float*                  y )
{
    return aoclsparse_scsrsv(trans,
                            alpha,
                            m,
                            csr_val,
                            csr_col_ind,
                            csr_row_ptr,
                            descr,
                            x,
                            y);
}

template <>
aoclsparse_status aoclsparse_csrsv(aoclsparse_operation       trans,
                                  const double*              alpha,
                                  aoclsparse_int             m,
                                  const double*              csr_val,
                                  const aoclsparse_int*      csr_col_ind,
                                  const aoclsparse_int*      csr_row_ptr,
                                  const aoclsparse_mat_descr descr,
                                  const double*             x,
                                  double*                  y )
{
    return aoclsparse_dcsrsv(trans,
                            alpha,
                            m,
                            csr_val,
                            csr_col_ind,
                            csr_row_ptr,
                            descr,
                            x,
                            y);
}
