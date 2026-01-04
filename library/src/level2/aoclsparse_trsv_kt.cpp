/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse.h"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_utils.hpp"

/*
 * Macro kernel templates for L, L^T, L^H, U, U^T and U^T kernels
 * ==============================================================
 */

using namespace kernel_templates;

/* ## KERNEL TEMPLATE TRIANGULAR SOLVE
 * Template to define clean CSR TRSV for lower triangular matrices, unit or not diagonal
 * What: solves `alpha` `A` `x` = `b` with lower triangular matrix of A
 * With: `A` matrix, `b` dense arrays
 * Returns: dense array `x`
 *
 * ## User inputs
 *
 * `aoclsparse_status kt_trsv_l<SZ,SUF,EXT>`:
 *
 * - `const T alpha,`
 * - `aoclsparse_int m,` (size of matrix `A`)
 * - `aoclsparse_index_base base,` (base for matrix `A`, either 0 or 1)
 * - `const T *a,` (matrix A values pointer)
 * - `const aoclsparse_int *icol,` (matrix csr column array pointer)
 * - `const aoclsparse_int *ilrow,` (matrix csr compressed row pointer)
 * - `const aoclsparse_int *idiag,` (csr pointer to diagonal elements)
 * - `const T *b,`
 * - `T *x,`
 * - `const bool unit` (`true` if to use unitary diagonal)
 *
 * ##  Template inputs
 *
 * - `SZ`  an enum (bsz) representing the length (in bits) of AVX vector, i.e., 256 or 512
 * - `SUF` suffix of working type, i.e., `double` or `float`
 * - `EXP` AVX capability, kt_avxext e.g. `AVX` or `AVX512F`, etc...
 * - `CONJ` bool for complex conjugation on the matrix entries
 */
template <bsz SZ, typename SUF, kt_avxext EXT, bool CONJ>
aoclsparse_status kt_trsv_l(const SUF             alpha,
                            aoclsparse_int        m,
                            aoclsparse_index_base base,
                            const SUF *__restrict__ a,
                            const aoclsparse_int *__restrict__ icol,
                            const aoclsparse_int *__restrict__ ilrow,
                            const aoclsparse_int *__restrict__ idiag,
                            const SUF *__restrict__ b,
                            aoclsparse_int incb,
                            SUF *__restrict__ x,
                            aoclsparse_int incx,
                            const bool     unit)
{
    aoclsparse_int       i, idx, idxend;
    SUF                  ad, xi;
    avxvector_t<SZ, SUF> avec, xvec, pvec;
    aoclsparse_int       idxcnt, idxrem;
    // get the vector length (type size)
    const aoclsparse_int  tsz      = tsz_v<SZ, SUF>;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *a_fix    = a - base;
    SUF                  *x_fix    = x - base * incx;

    aoclsparse_int iidx[tsz];

    for(i = 0; i < m; i++)
    {
        idxend = idiag[i];
        xi     = alpha * b[i * incb];
        idxcnt = idxend - ilrow[i];
        idxrem = idxcnt % tsz;
        pvec   = kt_setzero_p<SZ, SUF>();
        for(idx = ilrow[i]; idx < idxend - idxrem; idx += tsz)
        {
            for(aoclsparse_int jj = 0; jj < tsz; jj++)
                iidx[jj] = icol_fix[idx + jj] * incx;

            avec = kt_loadu_p<SZ, SUF>(&a_fix[idx]);
            if constexpr(CONJ)
                avec = kt_conj_p<SZ, SUF>(avec);
            xvec = kt_set_p<SZ, SUF>(x_fix, iidx);
            pvec = kt_fmadd_p<SZ, SUF>(avec, xvec, pvec);
        }
        if(idxcnt - tsz >= 0)
        {
            xi -= kt_hsum_p<SZ, SUF>(pvec);
        }
        // process remainder
        // use type container size -1 with zero paddding -> intrinsic
        // rest -> use a loop
        // FIXME: Take care of corner case where tsz-1 = 1
        switch(idxrem)
        {
        case(tsz - 1):
            idx = idxend - idxrem;

            for(aoclsparse_int jj = 0; jj < tsz - 1; jj++)
                iidx[jj] = icol_fix[idx + jj] * incx;

            avec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(a_fix, idx);
            if constexpr(CONJ)
                avec = kt_conj_p<SZ, SUF>(avec);
            xvec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(x_fix, iidx);
            xi -= kt_dot_p<SZ, SUF>(avec, xvec);
            break;
        default:
            if constexpr(aoclsparse::is_dt_complex<SUF>() && CONJ)
                for(idx = idxend - idxrem; idx < idxend; idx++)
                    xi -= std::conj(a_fix[idx]) * x_fix[icol_fix[idx] * incx];
            else
                for(idx = idxend - idxrem; idx < idxend; idx++)
                    xi -= a_fix[idx] * x_fix[icol_fix[idx] * incx];
        }
        x[i * incx] = xi;
        if(!unit)
        {
            ad = a_fix[idxend];
            if constexpr(aoclsparse::is_dt_complex<SUF>() && CONJ)
                ad = std::conj(ad);
            x[i * incx] /= ad;
        }
    }
    return aoclsparse_status_success;
}

/* ## KERNEL TEMPLATE TRIANGULAR SOLVE
 * Template to define clean CSR TRSV for lower triangular matrices, unit or not diagonal
 * What: solves `alpha` `A`^T `x` = `b` or `alpha` `A`^H `x` = `b`
 *       with lower triangular matrix of A
 * With: `A` matrix, `b` dense arrays
 * Returns: dense array `x`
 * Extra AVX vectors: YES
 *
 * ## User inputs
 *
 * `aoclsparse_status kt_trsv_lt<SZ,SUF,EXT,OP>`:
 *
 * - `const T alpha,`
 * - `aoclsparse_int m,` (size of matrix `A`)
 * - `aoclsparse_index_base base,` (base for matrix `A`, either 0 or 1)
 * - `const T *a,` (matrix A values pointer)
 * - `const aoclsparse_int *icol,` (matrix csr column array pointer)
 * - `const aoclsparse_int *ilrow,` (matrix csr compressed row pointer)
 * - `const aoclsparse_int *idiag,` (csr pointer to diagonal elements)
 * - `const T *b,`
 * - `T *x,`
 * - `const bool unit` (`true` if to use unitary diagonal)
 *
 * ##  Template inputs
 *
 * - `SZ`  an enum (bsz) representing the length (in bits) of AVX vector, i.e., 256 or 512
 * - `SUF` suffix of working type, i.e., `double` or `float`
 * - `EXP` AVX capability, kt_avxext e.g. `AVX` or `AVX512F`, etc...
 * - `CONJ` bool for complex conjugation on the matrix entries
 */
template <bsz SZ, typename SUF, kt_avxext EXT, bool CONJ>
aoclsparse_status kt_trsv_lt(const SUF             alpha,
                             aoclsparse_int        m,
                             aoclsparse_index_base base,
                             const SUF *__restrict__ a,
                             const aoclsparse_int *__restrict__ icol,
                             const aoclsparse_int *__restrict__ ilrow,
                             const aoclsparse_int *__restrict__ idiag,
                             const SUF *__restrict__ b,
                             aoclsparse_int incb,
                             SUF *__restrict__ x,
                             aoclsparse_int incx,
                             const bool     unit)
{
    aoclsparse_int       i, idx, idxend;
    aoclsparse_int       idxcnt, idxrem;
    SUF                  ad, xi, mxi;
    avxvector_t<SZ, SUF> avec, xvec, xivec;
    // get the vector length (considering the type size)
    const aoclsparse_int  tsz      = tsz_v<SZ, SUF>;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *a_fix    = a - base;
    SUF                  *x_fix    = x - base * incx;
    const SUF            *xvptr    = reinterpret_cast<SUF *>(&xvec);

    aoclsparse_int iidx[tsz];

    if(alpha != (SUF)0)
        for(i = 0; i < m; i++)
            x[i * incx] = alpha * b[i * incb];
    for(i = m - 1; i >= 0; i--)
    {
        idxend = idiag[i];
        if(!unit)
        {

            ad = a_fix[idiag[i]];
            if constexpr(aoclsparse::is_dt_complex<SUF>() && CONJ)
                ad = std::conj(ad);
            x[i * incx] /= ad;
        }
        xi     = x[i * incx];
        mxi    = -xi;
        idxcnt = idxend - ilrow[i];
        idxrem = idxcnt % tsz;
        for(idx = ilrow[i]; idx < idxend - idxrem; idx += tsz)
        {
            for(aoclsparse_int jj = 0; jj < tsz; jj++)
                iidx[jj] = icol_fix[idx + jj] * incx;

            xvec = kt_set_p<SZ, SUF>(x_fix, iidx);
            avec = kt_loadu_p<SZ, SUF>(&a_fix[idx]);
            if constexpr(CONJ)
                avec = kt_conj_p<SZ, SUF>(avec);
            xivec = kt_set1_p<SZ, SUF>(mxi);
            xvec  = kt_fmadd_p<SZ, SUF>(avec, xivec, xvec);
            kt_scatter_p<SZ, SUF>(xvec, x_fix, iidx);
        }
        // process remainder
        // use packet-size -1 with zero paddding -> intrinsic
        // rest -> use a loop
        switch(idxrem)
        {
        case(tsz - 1):
            idx = idxend - idxrem;

            for(aoclsparse_int jj = 0; jj < tsz - 1; jj++)
                iidx[jj] = icol_fix[idx + jj] * incx;

            xvec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(x_fix, iidx);
            avec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(a_fix, idx);
            if constexpr(CONJ)
                avec = kt_conj_p<SZ, SUF>(avec);
            xivec = kt_set1_p<SZ, SUF>(mxi);
            xvec  = kt_fmadd_p<SZ, SUF>(avec, xivec, xvec);
            for(aoclsparse_int k = 0; k < tsz - 1; k++)
                x_fix[icol_fix[idx + k] * incx] = xvptr[k];
            break;
        default:
            if constexpr(aoclsparse::is_dt_complex<SUF>() && CONJ)
                for(idx = idxend - idxrem; idx < idxend; idx++)
                    x_fix[icol_fix[idx] * incx] -= std::conj(a_fix[idx]) * xi;
            else
                for(idx = idxend - idxrem; idx < idxend; idx++)
                    x_fix[icol_fix[idx] * incx] -= a_fix[idx] * xi;
        }
    }
    return aoclsparse_status_success;
}

/* ## KERNEL TEMPLATE TRIANGULAR SOLVE
 * Template to define clean CSR TRSV for upper triangular matrices, unit or not diagonal
 * What: solves `alpha` `A` `x` = `b` with upper triangular matrix of A
 * With: `A` matrix, `b` dense arrays
 * Returns: dense array `x`
 *
 * ## User inputs
 *
 * `aoclsparse_status kt_trsv_u<SZ,SUF,EXT>`:
 *
 * - `const T alpha,`
 * - `aoclsparse_int m,` (size of matrix `A`)
 * - `aoclsparse_index_base base,` (base for matrix `A`, either 0 or 1)
 * - `const T *a,` (matrix A values pointer)
 * - `const aoclsparse_int *icol,` (matrix csr column array pointer)
 * - `const aoclsparse_int *ilrow,` (matrix csr compressed row pointer)
 * - `const aoclsparse_int *iurow,` (csr pointer to upper triangle elements)
 * - `const T *b,`
 * - `T *x,`
 * - `const bool unit` (`true` if to use unitary diagonal)
 *
 * ##  Template inputs
 *
 * - `SZ`  an enum (bsz) representing the length (in bits) of AVX vector, i.e., 256 or 512
 * - `SUF` suffix of working type, i.e., `double` or `float`
 * - `EXP` AVX capability, kt_avxext e.g. `AVX` or `AVX512F`, etc...
 * - `CONJ` bool for complex conjugation on the matrix entries
 */
template <bsz SZ, typename SUF, kt_avxext EXT, bool CONJ>
aoclsparse_status kt_trsv_u(const SUF             alpha,
                            aoclsparse_int        m,
                            aoclsparse_index_base base,
                            const SUF *__restrict__ a,
                            const aoclsparse_int *__restrict__ icol,
                            const aoclsparse_int *__restrict__ ilrow,
                            const aoclsparse_int *__restrict__ iurow,
                            const SUF *__restrict__ b,
                            aoclsparse_int incb,
                            SUF *__restrict__ x,
                            aoclsparse_int incx,
                            const bool     unit)
{
    aoclsparse_int       i, idiag, idx, idxstart, idxend;
    aoclsparse_int       idxcnt, idxrem;
    SUF                  ad, xi;
    avxvector_t<SZ, SUF> avec, xvec, pvec;
    // get the vector length (type size)
    const aoclsparse_int  tsz      = tsz_v<SZ, SUF>;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *a_fix    = a - base;
    SUF                  *x_fix    = x - base * incx;

    aoclsparse_int iidx[tsz];

    for(i = m - 1; i >= 0; i--)
    {
        idxstart = iurow[i];
        idxend   = ilrow[i + 1] - 1;
        xi       = alpha * b[i * incb];
        idxcnt   = idxend - idxstart + 1;
        idxrem   = idxcnt % tsz;
        pvec     = kt_setzero_p<SZ, SUF>();

        for(idx = idxstart; idx <= idxend - idxrem; idx += tsz)
        {
            for(aoclsparse_int jj = 0; jj < tsz; jj++)
                iidx[jj] = icol_fix[idx + jj] * incx;

            xvec = kt_set_p<SZ, SUF>(x_fix, iidx);
            avec = kt_loadu_p<SZ, SUF>(&a_fix[idx]);
            if constexpr(CONJ)
                avec = kt_conj_p<SZ, SUF>(avec);
            pvec = kt_fmadd_p<SZ, SUF>(avec, xvec, pvec);
        }
        if(idxcnt - tsz >= 0)
        {
            xi -= kt_hsum_p<SZ, SUF>(pvec);
        }
        // process remainder
        // use packet-size -1 with zero paddding -> intrinsic
        // rest -> use a loop
        switch(idxrem)
        {
        case(tsz - 1):
            idx = idxend - idxrem + 1;

            for(aoclsparse_int jj = 0; jj < tsz - 1; jj++)
                iidx[jj] = icol_fix[idx + jj] * incx;

            xvec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(x_fix, iidx);
            avec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(a_fix, idx);
            if constexpr(CONJ)
                avec = kt_conj_p<SZ, SUF>(avec);
            xi -= kt_dot_p<SZ, SUF>(avec, xvec);
            break;
        default:
            if constexpr(aoclsparse::is_dt_complex<SUF>() && CONJ)
                for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                    xi -= std::conj(a_fix[idx]) * x_fix[icol_fix[idx] * incx];
            else
                for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                    xi -= a_fix[idx] * x_fix[icol_fix[idx] * incx];
        }
        x[i * incx] = xi;
        if(!unit)
        {
            idiag = iurow[i] - 1;
            ad    = a_fix[idiag];
            if constexpr(aoclsparse::is_dt_complex<SUF>() && CONJ)
                ad = std::conj(ad);
            x[i * incx] /= ad;
        }
    }
    return aoclsparse_status_success;
}

/* ## KERNEL TEMPLATE TRIANGULAR SOLVE
 * Template to define clean CSR TRSV for upper triangular matrices, unit or not diagonal
 * What: solves `alpha` `A`^T `x` = `b` or `alpha` `A`^H `x` = `b`
 *       with upper triangular matrix of A
 * With: `A` matrix, `b` dense arrays
 * Returns: dense array `x`
 *
 * ## User inputs
 *
 * `aoclsparse_status kt_trsv_ut<SZ,SUF,EXT,OP>`:
 *
 * - `const T alpha,`
 * - `aoclsparse_int m,` (size of matrix `A`)
 * - `aoclsparse_index_base base,` (base for matrix `A`, either 0 or 1)
 * - `const T *a,` (matrix A values pointer)
 * - `const aoclsparse_int *icol,` (matrix csr column array pointer)
 * - `const aoclsparse_int *ilrow,` (matrix csr compressed row pointer)
 * - `const aoclsparse_int *iurow,` (csr pointer to upper triangular elements)
 * - `const T *b,`
 * - `T *x,`
 * - `const bool unit` (`true` if to use unitary diagonal)
 *
 * ##  Template inputs
 *
 * - `SZ`  an enum (bsz) representing the length (in bits) of AVX vector, i.e., 256 or 512
 * - `SUF` suffix of working type, i.e., `double` or `float`
 * - `EXP` AVX capability, kt_avxext e.g. `AVX` or `AVX512F`, etc...
 * - `CONJ` bool for complex conjugation on the matrix entries
 */
template <bsz SZ, typename SUF, kt_avxext EXT, bool CONJ>
aoclsparse_status kt_trsv_ut(const SUF             alpha,
                             aoclsparse_int        m,
                             aoclsparse_index_base base,
                             const SUF *__restrict__ a,
                             const aoclsparse_int *__restrict__ icol,
                             const aoclsparse_int *__restrict__ ilrow,
                             const aoclsparse_int *__restrict__ iurow,
                             const SUF *__restrict__ b,
                             aoclsparse_int incb,
                             SUF *__restrict__ x,
                             aoclsparse_int incx,
                             const bool     unit)
{
    aoclsparse_int       i, idx, idxstart, idxend, idiag;
    aoclsparse_int       idxcnt, idxrem;
    SUF                  xi, mxi, ad;
    avxvector_t<SZ, SUF> avec, xivec, xvec;
    // get the vector length (type size)
    const aoclsparse_int  tsz      = tsz_v<SZ, SUF>;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *a_fix    = a - base;
    SUF                  *x_fix    = x - base * incx;
    const SUF            *xvptr    = reinterpret_cast<SUF *>(&xvec);

    aoclsparse_int iidx[tsz];

    if(alpha != (SUF)0.0)
        for(i = 0; i < m; i++)
            x[i * incx] = alpha * b[i * incb];
    for(i = 0; i < m; i++)
    {
        idxstart = iurow[i];
        idxend   = ilrow[i + 1] - 1;
        if(!unit)
        {
            idiag = iurow[i] - 1;
            ad    = a_fix[idiag];
            if constexpr(aoclsparse::is_dt_complex<SUF>() && CONJ)
                ad = std::conj(ad);
            x[i * incx] = x[i * incx] / ad;
        }
        xi     = x[i * incx];
        mxi    = -xi;
        idxcnt = idxend - idxstart + 1;
        idxrem = idxcnt % tsz;
        for(idx = idxstart; idx <= idxend - idxrem; idx += tsz)
        {
            for(aoclsparse_int jj = 0; jj < tsz; jj++)
                iidx[jj] = icol_fix[idx + jj] * incx;

            xvec = kt_set_p<SZ, SUF>(x_fix, iidx);
            avec = kt_loadu_p<SZ, SUF>(&a_fix[idx]);
            if constexpr(CONJ)
                avec = kt_conj_p<SZ, SUF>(avec);
            xivec = kt_set1_p<SZ, SUF>(mxi);
            xvec  = kt_fmadd_p<SZ, SUF>(avec, xivec, xvec);
            kt_scatter_p<SZ, SUF>(xvec, x_fix, iidx);
        }
        // process remainder
        // use packet-size -1 with zero paddding -> intrinsic
        // rest -> use a loop
        switch(idxrem)
        {
        case(tsz - 1):
            idx = idxend - idxrem + 1;

            for(aoclsparse_int jj = 0; jj < tsz - 1; jj++)
                iidx[jj] = icol_fix[idx + jj] * incx;

            xvec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(x_fix, iidx);
            avec = kt_maskz_set_p<SZ, SUF, EXT, tsz - 1>(a_fix, idx);
            if constexpr(CONJ)
                avec = kt_conj_p<SZ, SUF>(avec);
            xivec = kt_set1_p<SZ, SUF>(mxi);
            xvec  = kt_fmadd_p<SZ, SUF>(avec, xivec, xvec);
            for(aoclsparse_int k = 0; k < tsz - 1; k++)
                x_fix[icol_fix[idx + k] * incx] = xvptr[k];
            break;
        default:
            if constexpr(aoclsparse::is_dt_complex<SUF>() && CONJ)
                for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                    x_fix[icol_fix[idx] * incx] -= std::conj(a_fix[idx]) * xi;
            else
                for(idx = idxend - idxrem + 1; idx <= idxend; idx++)
                    x_fix[icol_fix[idx] * incx] -= a_fix[idx] * xi;
        }
    }
    return aoclsparse_status_success;
}

// ---- Lower -----

#define TRSV_L_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, CONJ)      \
    template aoclsparse_status kt_trsv_l<BSZ, SUF, EXT, CONJ>( \
        const SUF             alpha,                           \
        aoclsparse_int        m,                               \
        aoclsparse_index_base base,                            \
        const SUF *__restrict__ a,                             \
        const aoclsparse_int *__restrict__ icol,               \
        const aoclsparse_int *__restrict__ ilrow,              \
        const aoclsparse_int *__restrict__ idiag,              \
        const SUF *__restrict__ b,                             \
        aoclsparse_int incb,                                   \
        SUF *__restrict__ x,                                   \
        aoclsparse_int incx,                                   \
        const bool     unit);

#define TRSV_L_TEMPLATE_DECLARATION_CONJ(BSZ, SUF, EXT) \
    TRSV_L_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, true)

#define TRSV_L_TEMPLATE_DECLARATION(BSZ, SUF, EXT) \
    TRSV_L_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, false)

// ---- Lower transpose -----

#define TRSV_LT_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, CONJ)      \
    template aoclsparse_status kt_trsv_lt<BSZ, SUF, EXT, CONJ>( \
        const SUF             alpha,                            \
        aoclsparse_int        m,                                \
        aoclsparse_index_base base,                             \
        const SUF *__restrict__ a,                              \
        const aoclsparse_int *__restrict__ icol,                \
        const aoclsparse_int *__restrict__ ilrow,               \
        const aoclsparse_int *__restrict__ idiag,               \
        const SUF *__restrict__ b,                              \
        aoclsparse_int incb,                                    \
        SUF *__restrict__ x,                                    \
        aoclsparse_int incx,                                    \
        const bool     unit);

#define TRSV_LT_TEMPLATE_DECLARATION_CONJ(BSZ, SUF, EXT) \
    TRSV_LT_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, true)

#define TRSV_LT_TEMPLATE_DECLARATION(BSZ, SUF, EXT) \
    TRSV_LT_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, false)

// ---- Upper -----

#define TRSV_U_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, CONJ)      \
    template aoclsparse_status kt_trsv_u<BSZ, SUF, EXT, CONJ>( \
        const SUF             alpha,                           \
        aoclsparse_int        m,                               \
        aoclsparse_index_base base,                            \
        const SUF *__restrict__ a,                             \
        const aoclsparse_int *__restrict__ icol,               \
        const aoclsparse_int *__restrict__ ilrow,              \
        const aoclsparse_int *__restrict__ iurow,              \
        const SUF *__restrict__ b,                             \
        aoclsparse_int incb,                                   \
        SUF *__restrict__ x,                                   \
        aoclsparse_int incx,                                   \
        const bool     unit);

#define TRSV_U_TEMPLATE_DECLARATION_CONJ(BSZ, SUF, EXT) \
    TRSV_U_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, true)

#define TRSV_U_TEMPLATE_DECLARATION(BSZ, SUF, EXT) \
    TRSV_U_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, false)

// ---- Upper transpose -----

#define TRSV_UT_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, CONJ)      \
    template aoclsparse_status kt_trsv_ut<BSZ, SUF, EXT, CONJ>( \
        const SUF             alpha,                            \
        aoclsparse_int        m,                                \
        aoclsparse_index_base base,                             \
        const SUF *__restrict__ a,                              \
        const aoclsparse_int *__restrict__ icol,                \
        const aoclsparse_int *__restrict__ ilrow,               \
        const aoclsparse_int *__restrict__ iurow,               \
        const SUF *__restrict__ b,                              \
        aoclsparse_int incb,                                    \
        SUF *__restrict__ x,                                    \
        aoclsparse_int incx,                                    \
        const bool     unit);

#define TRSV_UT_TEMPLATE_DECLARATION_CONJ(BSZ, SUF, EXT) \
    TRSV_UT_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, true)

#define TRSV_UT_TEMPLATE_DECLARATION(BSZ, SUF, EXT) \
    TRSV_UT_TEMPLATE_DECLARATION_(BSZ, SUF, EXT, false)

#define KT_INSTANTIATE_TRSV(BSZ, EXT)                                \
    KT_INSTANTIATE_EXT(TRSV_L_TEMPLATE_DECLARATION_CONJ, BSZ, EXT);  \
    KT_INSTANTIATE_EXT(TRSV_L_TEMPLATE_DECLARATION, BSZ, EXT);       \
    KT_INSTANTIATE_EXT(TRSV_LT_TEMPLATE_DECLARATION_CONJ, BSZ, EXT); \
    KT_INSTANTIATE_EXT(TRSV_LT_TEMPLATE_DECLARATION, BSZ, EXT);      \
    KT_INSTANTIATE_EXT(TRSV_U_TEMPLATE_DECLARATION_CONJ, BSZ, EXT);  \
    KT_INSTANTIATE_EXT(TRSV_U_TEMPLATE_DECLARATION, BSZ, EXT);       \
    KT_INSTANTIATE_EXT(TRSV_UT_TEMPLATE_DECLARATION_CONJ, BSZ, EXT); \
    KT_INSTANTIATE_EXT(TRSV_UT_TEMPLATE_DECLARATION, BSZ, EXT);

// This instantiates AVX2/AVX512 kernel based on the build flags
KT_INSTANTIATE_TRSV(get_bsz(), get_kt_ext());
#ifndef KT_AVX2_BUILD
// Instantiate AVX512VL kernels for AVX512 builds
KT_INSTANTIATE_TRSV(bsz::b256, kt_avxext::AVX512VL);
#endif
