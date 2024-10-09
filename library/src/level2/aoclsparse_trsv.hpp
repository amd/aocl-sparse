/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_dispatcher.hpp"
#include "aoclsparse_l2_kt.hpp"

#include <complex>
#include <type_traits>

/* Core computation of TRSV, assumed A is optimized
 * solves L*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_l_ref_core(const T               alpha,
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
            xi -= a_fix[idx] * x_fix[icol_fix[idx] * incx];
        if(!unit)
            xi /= a_fix[idiag[i]];
        x[i * incx] = xi;
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves L'*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_lt_ref_core(const T               alpha,
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
            x[i * incx] /= a_fix[idiag[i]];
        // propagate value of x[i * incx] through the column (used to be the row but now is transposed)
        for(idx = ilrow[i]; idx < idiag[i]; idx++)
            x_fix[icol_fix[idx] * incx] -= a_fix[idx] * x[i * incx];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves L^H*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_lh_ref_core(const T               alpha,
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
            x[i * incx] /= std::conj(a_fix[idiag[i]]);
        // propagate value of x[i * incx] through the column (used to be the row but now is transposed)
        for(idx = ilrow[i]; idx < idiag[i]; idx++)
            x_fix[icol_fix[idx] * incx] -= std::conj(a_fix[idx]) * x[i * incx];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_u_ref_core(const T               alpha,
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
            xi -= a_fix[idx] * x_fix[icol_fix[idx] * incx];
        x[i * incx] = xi;
        if(!unit)
        {
            // urow[i]-1 always points to idiag[i]
            idiag = iurow[i] - 1;
            x[i * incx] /= a_fix[idiag];
        }
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U'*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_ut_ref_core(const T               alpha,
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
            x[i * incx] /= a_fix[idiag];
        }
        idxstart = iurow[i];
        // ilrow[i+1]-1 always points to last element of U at row i
        idxend = ilrow[i + 1] - 1;
        for(idx = idxstart; idx <= idxend; idx++)
            x_fix[icol_fix[idx] * incx] -= a_fix[idx] * x[i * incx];
    }
    return aoclsparse_status_success;
}

/* Core computation of TRSV, assumed A is optimized
 * solves U^H*x = alpha*b
 */
template <typename T>
static inline aoclsparse_status trsv_uh_ref_core(const T               alpha,
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
            x[i * incx] /= std::conj(a_fix[idiag]);
        }
        idxstart = iurow[i];
        // ilrow[i+1]-1 always points to last element of U at row i
        idxend = ilrow[i + 1] - 1;
        for(idx = idxstart; idx <= idxend; idx++)
            x_fix[icol_fix[idx] * incx] -= std::conj(a_fix[idx]) * x[i * incx];
    }
    return aoclsparse_status_success;
}

/* TRiangular SolVer dispatcher
 * ============================
 * TRSV dispatcher and various templated and vectorized triangular solve kernels
 * Solves A*x = alpha*b or A'*x = alpha*b with A lower (L) or upper (U) triangular.
 * Optimized version, requires A to have been previously "optimized". If A is not
 * optimized previously by user, it is optimized on the fly.
 */
template <typename T>
aoclsparse_status
    aoclsparse_trsv(const aoclsparse_operation transpose, /* matrix operation */
                    const T                    alpha, /* scalar for rescaling RHS */
                    aoclsparse_matrix          A, /* matrix data */
                    const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
                    const T                   *b, /* RHS */
                    const aoclsparse_int       incb, /* Stride for B */
                    T                         *x, /* solution */
                    const aoclsparse_int       incx, /* Stride for X */
                    const aoclsparse_int       kid /* user request of Kernel ID (kid) to use */)
{
    // Quick initial checks
    if(!A || !x || !b || !descr)
        return aoclsparse_status_invalid_pointer;

    // Only CSR and TCSR input format supported
    if(A->input_format != aoclsparse_csr_mat && A->input_format != aoclsparse_tcsr_mat)
    {
        return aoclsparse_status_not_implemented;
    }

    const aoclsparse_int nnz = A->nnz;
    const aoclsparse_int m   = A->m;

    if(m <= 0 || nnz <= 0)
        return aoclsparse_status_invalid_size;

    if(m != A->n || incb <= 0 || incx <= 0) // Matrix not square or invalid strides
    {
        return aoclsparse_status_invalid_value;
    }

    // Check for base index incompatibility
    // There is an issue that zero-based indexing is defined in two separate places and
    // can lead to ambiguity, we check that both are consistent.
    if(A->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    // Check if descriptor's index-base is valid (and A's index-base must be the same)
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    if(transpose != aoclsparse_operation_none && transpose != aoclsparse_operation_transpose
       && transpose != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_not_implemented;

    if(descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_triangular)
    {
        return aoclsparse_status_invalid_value;
    }

    // Matrix is singular, system cannot be solved
    if(descr->diag_type == aoclsparse_diag_type_zero)
    {
        return aoclsparse_status_invalid_value;
    }

    if(descr->fill_mode != aoclsparse_fill_mode_lower
       && descr->fill_mode != aoclsparse_fill_mode_upper)
        return aoclsparse_status_not_implemented;

    // Unpack A and check
    if(!A->opt_csr_ready)
    {
        // user did not check the matrix, call optimize
        aoclsparse_status status;
        // Optimize TCSR matrix
        if(A->input_format == aoclsparse_tcsr_mat)
            status = aoclsparse_tcsr_optimize<T>(A);
        // Optimize CSR matrix
        else
            status = aoclsparse_csr_optimize<T>(A);
        if(status != aoclsparse_status_success)
            return status; // LCOV_EXCL_LINE
    }

    // From this point on A->opt_csr_ready is true

    // Make sure we have the right type before casting
    if(!((A->val_type == aoclsparse_dmat && std::is_same_v<T, double>)
         || (A->val_type == aoclsparse_smat && std::is_same_v<T, float>)
         || (A->val_type == aoclsparse_cmat && std::is_same_v<T, std::complex<float>>)
         || (A->val_type == aoclsparse_zmat && std::is_same_v<T, std::complex<double>>)))
        return aoclsparse_status_wrong_type;

    const bool unit = descr->diag_type == aoclsparse_diag_type_unit;
    if(!A->opt_csr_full_diag && !unit) // not of full rank, linear system cannot be solved
    {
        return aoclsparse_status_invalid_value;
    }

    T              *a;
    aoclsparse_int *icol, *ilrow, *idiag, *iurow;

    if(A->input_format == aoclsparse_tcsr_mat)
    {
        if(descr->fill_mode == aoclsparse_fill_mode_lower)
        {
            a     = (T *)((A->tcsr_mat).val_L);
            icol  = (A->tcsr_mat).col_idx_L;
            ilrow = (A->tcsr_mat).row_ptr_L;
            idiag = A->idiag;
            iurow = (A->tcsr_mat).row_ptr_L + 1;
        }
        else
        {
            a     = (T *)((A->tcsr_mat).val_U);
            icol  = (A->tcsr_mat).col_idx_U;
            ilrow = (A->tcsr_mat).row_ptr_U;
            idiag = (A->tcsr_mat).row_ptr_U;
            iurow = A->iurow;
        }
    }
    else
    {
        a    = (T *)((A->opt_csr_mat).csr_val);
        icol = (A->opt_csr_mat).csr_col_ptr;
        // beginning of the row
        ilrow = (A->opt_csr_mat).csr_row_ptr;
        // position of the diagonal element (includes zeros) always has min(m,n) elements
        idiag = A->idiag;
        // ending of the row
        iurow = A->iurow;
    }

    const bool            lower = descr->fill_mode == aoclsparse_fill_mode_lower;
    aoclsparse_index_base base  = A->internal_base_index;

    using namespace aoclsparse;

    /*
        Check if the AVX512 kernels can execute
        This check needs to be done only once in a run
    */
    static bool can_exec_avx512 = context::get_context()->supports<context_isa_t::AVX512F>();

    // CPU ID dispatcher sets recommended Kernel ID to use, this can be influenced by
    // the user-requested "kid" hint
    // TODO update when libcpuid is merged into aoclsparse
    aoclsparse_int usekid = 2; // Defaults to 2 (AVX2 256-bits)
    if(kid >= 0)
    {
        switch(kid)
        {
        case 0: // reference implementation (no explicit vectorization)
            usekid = kid;
            break;
        case 1:
        case 2: // AVX2 256b
            usekid = kid;
            break;
        case 3: // AVX-512F 512b
            if(can_exec_avx512)
                usekid = kid;
            // Requested kid not available on host,
            // stay with kid suggested by CPU ID...
            break;
        default: // use kid suggested by CPU ID...
            break;
        }
    }

    using namespace kernel_templates;

    /* Available kernel table
     * ======================
     * kid | kernel                 | description                             | type support
     * ----+------------------------+-----------------------------------------+------------------------------
     * 0   | trsv_l_ref_core        | reference vanilla for Lx=b              | float/double/cfloat/cdouble
     * 0   | trsv_lt_ref_core       | reference vanilla for L^T x=b           | float/double/cfloat/cdouble
     * 0   | trsv_lh_ref_core       | reference vanilla for L^H x=b           |              cfloat/cdouble
     * 0   | trsv_u_ref_core        | reference vanilla for Ux=b              | float/double/cfloat/cdouble
     * 0   | trsv_ut_ref_core       | reference vanilla for U^T x=b           | float/double/cfloat/cdouble
     * 0   | trsv_uh_ref_core       | reference vanilla for U^H x=b           |              cfloat/cdouble
     * - - - - - - - - - - - -+- - - - - - - - - - - - - - - - - - - - -+- - - - - - - - - - - - - - - - - -
     * 1, 2| kt_trsv_l<256,*,AVX>   | L solver AVX extensions on 256-bit      | float/double/cfloat/cdouble
     *     |                        | wide register implementation            |
     * 1, 2| kt_trsv_lt<256,*,AVX>  | L^T/H solver AVX extensions on 256-bit  | float/double/cfloat/cdouble
     *     |                        | wide register implementation            |
     * 1, 2| kt_trsv_u<256,*,AVX>   | U solver AVX extensions on 256-bit      | float/double/cfloat/cdouble
     *     |                        | wide register implementation            |
     * 1, 2| kt_trsv_ut<256,*,AVX>  | U^T/H solver AVX extensions on 256-bit  | float/double/cfloat/cdouble
     *     |                        | wide register implementation            |
     * - - - - - - - - - - - -+- - - - - - - - - - - - - - - - - - - - -+- - - - - - - - - - - - - - - - - -
     * 3   | kt_trsv_l_<512,*,*>    | L solver AVX512F extensions on 512-bit  | float/double/cfloat/cdouble
     *     |                        | wide register implementation            |
     * 3   | kt_trsv_lt_<512,*,*>   | L^T/H solver AVX512F extensions on 512- | float/double/cfloat/cdouble
     *     |                        | bit wide register implementation        |
     * 3   | kt_trsv_u_<512,*,*>    | U solver AVX512F extensions on 512-bit  | float/double/cfloat/cdouble
     *     |                        | wide register implementation            |
     * 3   | kt_trsv_ut_<512,*,*>   | U^T/H solver AVX512F extensions on 512- | float/double/cfloat/cdouble
     *     |                        | bit wide register implementation        |
     * -----------------------+-----------------------------------------------------------------------------
     */
    if(lower)
    {
        switch(transpose)
        {
        case aoclsparse_operation_none:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                return kt_trsv_l<bsz::b512, T, kt_avxext::AVX512F>(
                    alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
#endif
            case 2: // AVX2
            case 1:
                return kt_trsv_l<bsz::b256, T, kt_avxext::AVX2>(
                    alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
            default: // Reference implementation
                return trsv_l_ref_core(
                    alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
            }
            break;
        case aoclsparse_operation_transpose:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                return kt_trsv_lt<bsz::b512, T, kt_avxext::AVX512F>(
                    alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
#endif
            case 2: // AVX2
            case 1:
                return kt_trsv_lt<bsz::b256, T, kt_avxext::AVX2>(
                    alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
            default: // Reference implementation
                return trsv_lt_ref_core(
                    alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
            }
            break;
        case aoclsparse_operation_conjugate_transpose:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return kt_trsv_lt<bsz::b512, T, kt_avxext::AVX512F, trsv_op::herm>(
                        alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                else
                    return kt_trsv_lt<bsz::b512, T, kt_avxext::AVX512F>(
                        alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
#endif
            case 2: // AVX2
            case 1: // AVX2
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return kt_trsv_lt<bsz::b256, T, kt_avxext::AVX2, trsv_op::herm>(
                        alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                else
                    return kt_trsv_lt<bsz::b256, T, kt_avxext::AVX2>(
                        alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
            default: // Reference implementation
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return trsv_lh_ref_core<T>(
                        alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                else
                    return trsv_lt_ref_core(
                        alpha, m, base, a, icol, ilrow, idiag, b, incb, x, incx, unit);
                break;
            }
            break;
        }
    }
    else // upper
    {
        switch(transpose)
        {
        case aoclsparse_operation_none:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                return kt_trsv_u<bsz::b512, T, kt_avxext::AVX512F>(
                    alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
#endif
            case 2: // AVX2
            case 1:
                return kt_trsv_u<bsz::b256, T, kt_avxext::AVX2>(
                    alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
            default: // Reference implementation
                return trsv_u_ref_core(
                    alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
            }
            break;
        case aoclsparse_operation_transpose:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                return kt_trsv_ut<bsz::b512, T, kt_avxext::AVX512F>(
                    alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
#endif
            case 2: // AVX2
            case 1:
                return kt_trsv_ut<bsz::b256, T, kt_avxext::AVX2>(
                    alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
            default: // Reference implementation
                return trsv_ut_ref_core(
                    alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
            }
            break;
        case aoclsparse_operation_conjugate_transpose:
            switch(usekid)
            {
            case 3: // AVX-512F (Note: if not available then trickle down to next best)
#if USE_AVX512
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return kt_trsv_ut<bsz::b512, T, kt_avxext::AVX512F, trsv_op::herm>(
                        alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                else
                    return kt_trsv_ut<bsz::b512, T, kt_avxext::AVX512F>(
                        alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
#endif
            case 2: // AVX2
            case 1: // AVX2
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return kt_trsv_ut<bsz::b256, T, kt_avxext::AVX2, trsv_op::herm>(
                        alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                else
                    return kt_trsv_ut<bsz::b256, T, kt_avxext::AVX2>(
                        alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
            default: // Reference implementation
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                    return trsv_uh_ref_core<T>(
                        alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                else
                    return trsv_ut_ref_core(
                        alpha, m, base, a, icol, ilrow, iurow, b, incb, x, incx, unit);
                break;
            }
            break;
        }
    }
    // It should never be reached...
    return aoclsparse_status_internal_error;
}
#endif // AOCLSPARSE_SV_HPP
