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
#ifndef AOCLSPARSE_SV_HPP
#define AOCLSPARSE_SV_HPP
#include "aoclsparse.h"

#include <complex>

namespace aoclsparse
{

    /* Core computation of TRSV, assumed A is optimized
     * solves L*x = alpha*b and conj(L)*x = alpha*b
     * The parameter CONJ specifies if the input csr matrix described by
     * <descr, csr_val, csr_col_ind and csr_row_ptr> needs conjugation.
     */
    template <typename T, bool CONJ = false>
    static inline aoclsparse_status ref_trsv_l(const T               alpha,
                                               aoclsparse_int        m,
                                               aoclsparse_index_base base,
                                               const T *__restrict__ a,
                                               const aoclsparse_int *__restrict__ icol,
                                               const aoclsparse_int *__restrict__ ilrow,
                                               const aoclsparse_int *__restrict__ idiag,
                                               const T *__restrict__ b,
                                               aoclsparse_int incb,
                                               T *__restrict__ x,
                                               aoclsparse_int incx,
                                               const bool     unit)
    {
        aoclsparse_int        i, idx;
        T                     xi;
        const aoclsparse_int *icol_fix = icol - base;
        const T              *a_fix    = a - base;
        T                    *x_fix    = x - base * incx;
        for(i = 0; i < m; i++)
        {
            xi = alpha * b[i * incb];
            for(idx = ilrow[i]; idx < idiag[i]; idx++)
            {
                if constexpr(CONJ && aoclsparse::is_dt_complex<T>())
                    xi -= aoclsparse::conj(a_fix[idx]) * x_fix[icol_fix[idx] * incx];
                else
                    xi -= a_fix[idx] * x_fix[icol_fix[idx] * incx];
            }
            if(!unit)
            {
                if constexpr(CONJ && aoclsparse::is_dt_complex<T>())
                    xi /= aoclsparse::conj(a_fix[idiag[i]]);
                else
                    xi /= a_fix[idiag[i]];
            }
            x[i * incx] = xi;
        }
        return aoclsparse_status_success;
    }

    /* Core computation of TRSV, assumed A is optimized
     * solves L^H*x = alpha*b and L'*x = alpha*b
     */
    template <typename T, bool CONJ = false>
    static inline aoclsparse_status ref_trsv_lth(const T               alpha,
                                                 aoclsparse_int        m,
                                                 aoclsparse_index_base base,
                                                 const T *__restrict__ a,
                                                 const aoclsparse_int *__restrict__ icol,
                                                 const aoclsparse_int *__restrict__ ilrow,
                                                 const aoclsparse_int *__restrict__ idiag,
                                                 const T *__restrict__ b,
                                                 aoclsparse_int incb,
                                                 T *__restrict__ x,
                                                 aoclsparse_int incx,
                                                 const bool     unit)
    {
        aoclsparse_int        i, idx;
        const aoclsparse_int *icol_fix = icol - base;
        const T              *a_fix    = a - base;
        T                    *x_fix    = x - base * incx;

        for(i = 0; i < m; i++)
            x[i * incx] = alpha * b[i * incb];
        for(i = m - 1; i >= 0; i--)
        {
            if(!unit)
            {
                if constexpr(aoclsparse::is_dt_complex<T>() && CONJ)
                    x[i * incx] /= std::conj(a_fix[idiag[i]]);
                else
                    x[i * incx] /= a_fix[idiag[i]];
            }
            // propagate value of x[i * incx] through the column (used to be the row but now is transposed)
            for(idx = ilrow[i]; idx < idiag[i]; idx++)
            {
                if constexpr(aoclsparse::is_dt_complex<T>() && CONJ)
                    x_fix[icol_fix[idx] * incx] -= std::conj(a_fix[idx]) * x[i * incx];
                else
                    x_fix[icol_fix[idx] * incx] -= a_fix[idx] * x[i * incx];
            }
        }
        return aoclsparse_status_success;
    }

    /* Core computation of TRSV, assumed A is optimized
     * solves U*x = alpha*b and conj(U)*x = alpha*b
     * The parameter CONJ specifies if the input csr matrix described by
     * <descr, csr_val, csr_col_ind and csr_row_ptr> needs conjugation.
     */
    template <typename T, bool CONJ = false>
    static inline aoclsparse_status ref_trsv_u(const T               alpha,
                                               aoclsparse_int        m,
                                               aoclsparse_index_base base,
                                               const T *__restrict__ a,
                                               const aoclsparse_int *__restrict__ icol,
                                               const aoclsparse_int *__restrict__ ilrow,
                                               const aoclsparse_int *__restrict__ iurow,
                                               const T *__restrict__ b,
                                               aoclsparse_int incb,
                                               T *__restrict__ x,
                                               aoclsparse_int incx,
                                               const bool     unit)
    {
        aoclsparse_int        i, idx, idxstart, idxend, idiag;
        T                     xi;
        const aoclsparse_int *icol_fix = icol - base;
        const T              *a_fix    = a - base;
        T                    *x_fix    = x - base * incx;

        for(i = m - 1; i >= 0; i--)
        {
            idxstart = iurow[i];
            // ilrow[i+1]-1 always points to last element of U at row i
            idxend = ilrow[i + 1] - 1;
            xi     = alpha * b[i * incb];
            for(idx = idxstart; idx <= idxend; idx++)
            {
                if constexpr(CONJ && aoclsparse::is_dt_complex<T>())
                    xi -= aoclsparse::conj(a_fix[idx]) * x_fix[icol_fix[idx] * incx];
                else
                    xi -= a_fix[idx] * x_fix[icol_fix[idx] * incx];
            }
            x[i * incx] = xi;
            if(!unit)
            {
                // urow[i]-1 always points to idiag[i]
                idiag = iurow[i] - 1;
                if constexpr(CONJ && aoclsparse::is_dt_complex<T>())
                    x[i * incx] /= aoclsparse::conj(a_fix[idiag]);
                else
                    x[i * incx] /= a_fix[idiag];
            }
        }
        return aoclsparse_status_success;
    }

    /* Core computation of TRSV, assumed A is optimized
     * solves U^H*x = alpha*b and U'*x = alpha*b
     */
    template <typename T, bool CONJ = false>
    static inline aoclsparse_status ref_trsv_uth(const T               alpha,
                                                 aoclsparse_int        m,
                                                 aoclsparse_index_base base,
                                                 const T *__restrict__ a,
                                                 const aoclsparse_int *__restrict__ icol,
                                                 const aoclsparse_int *__restrict__ ilrow,
                                                 const aoclsparse_int *__restrict__ iurow,
                                                 const T *__restrict__ b,
                                                 aoclsparse_int incb,
                                                 T *__restrict__ x,
                                                 aoclsparse_int incx,
                                                 const bool     unit)
    {
        aoclsparse_int        i, idx, idxstart, idxend, idiag;
        const aoclsparse_int *icol_fix = icol - base;
        const T              *a_fix    = a - base;
        T                    *x_fix    = x - base * incx;
        for(i = 0; i < m; i++)
            x[i * incx] = alpha * b[i * incb];
        for(i = 0; i < m; i++)
        {
            if(!unit)
            {
                // urow[i]-1 always points to idiag[i]
                idiag = iurow[i] - 1;
                if constexpr(aoclsparse::is_dt_complex<T>() && CONJ)
                    x[i * incx] /= std::conj(a_fix[idiag]);
                else
                    x[i * incx] /= a_fix[idiag];
            }
            idxstart = iurow[i];
            // ilrow[i+1]-1 always points to last element of U at row i
            idxend = ilrow[i + 1] - 1;
            for(idx = idxstart; idx <= idxend; idx++)
            {
                if constexpr(aoclsparse::is_dt_complex<T>() && CONJ)
                    x_fix[icol_fix[idx] * incx] -= std::conj(a_fix[idx]) * x[i * incx];
                else
                    x_fix[icol_fix[idx] * incx] -= a_fix[idx] * x[i * incx];
            }
        }
        return aoclsparse_status_success;
    }
}
#endif