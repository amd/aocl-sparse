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
 * ************************************************************************ */

#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse.hpp"
#include "aoclsparse_cntx_dispatcher.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_mtx_dispatcher.hpp"
#include "aoclsparse_trsv_kr.hpp"
#include "aoclsparse_utils.hpp"

#include <complex>
#include <type_traits>

/* Triangular Solver (TRSV) dispatcher
 * ===================================
 * TRSV dispatcher and various templated and vectorized triangular solve kernels
 * Solves A*x = alpha*b or A'*x = alpha*b with A lower (L) or upper (U) triangular.
 * Optimized version, requires A to have been previously "optimized". If A is not
 * optimized previously by user, it is optimized on the fly.
 */
template <typename T>
aoclsparse_status
    aoclsparse::trsv(const aoclsparse_operation transpose, /* matrix operation */
                     const T                    alpha, /* scalar for rescaling RHS */
                     aoclsparse_matrix          A, /* matrix data */
                     const aoclsparse_mat_descr descr, /* matrix type, fill_mode, diag type, base */
                     const T                   *b, /* RHS */
                     const aoclsparse_int       incb, /* Stride for B */
                     T                         *x, /* solution */
                     const aoclsparse_int       incx, /* Stride for X */
                     aoclsparse_int             kid /* user request of Kernel ID (kid) to use */)
{
    // Quick initial checks
    if(!A || !x || !b || !descr)
        return aoclsparse_status_invalid_pointer;

    // Only CSR, CSC and TCSR input format supported
    if(A->input_format != aoclsparse_csr_mat && A->input_format != aoclsparse_tcsr_mat
       && A->input_format != aoclsparse_csc_mat)
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
    if(!(A->opt_csr_ready || A->opt_csc_ready))
    {
        // user did not check the matrix, call optimize
        aoclsparse_status status;
        // Optimize TCSR matrix
        if(A->input_format == aoclsparse_tcsr_mat)
            status = aoclsparse_tcsr_optimize<T>(A);
        else
        {
            // Optimize CSR/CSC matrix
            status = aoclsparse_csr_csc_optimize<T>(A);
        }
        if(status != aoclsparse_status_success)
            return status; // LCOV_EXCL_LINE
    }

    // From this point on A->opt_*_ready is true

    // Make sure we have the right type before casting
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    const bool unit = descr->diag_type == aoclsparse_diag_type_unit;
    if(!A->opt_csr_full_diag && !unit) // not of full rank, linear system cannot be solved
    {
        return aoclsparse_status_invalid_value;
    }

    T               *a;
    aoclsparse_int  *icol, *ilrow, *idiag, *iurow;
    aoclsparse::doid doid;

    // When the matrix is symm it can be treated as tri.
    if(descr->type == aoclsparse_matrix_type_symmetric)
    {
        _aoclsparse_mat_descr descr_cpy;
        aoclsparse_copy_mat_descr(&descr_cpy, descr);

        descr_cpy.type = aoclsparse_matrix_type_triangular;

        doid = aoclsparse::get_doid<T>(&descr_cpy, transpose);
    }
    else
        doid = aoclsparse::get_doid<T>(descr, transpose);

    if(A->opt_csr_ready)
    {
        if(A->input_format == aoclsparse_tcsr_mat)
        {
            if(descr->fill_mode == aoclsparse_fill_mode_lower)
            {
                a     = (T *)((A->tcsr_mat).val_L);
                icol  = (A->tcsr_mat).col_idx_L;
                ilrow = (A->tcsr_mat).row_ptr_L;
                idiag = (A->tcsr_mat).idiag;
                iurow = (A->tcsr_mat).row_ptr_L + 1;
            }
            else
            {
                a     = (T *)((A->tcsr_mat).val_U);
                icol  = (A->tcsr_mat).col_idx_U;
                ilrow = (A->tcsr_mat).row_ptr_U;
                idiag = (A->tcsr_mat).row_ptr_U;
                iurow = (A->tcsr_mat).iurow;
            }
        }
        else
        {
            //CSR
            a    = (T *)((A->opt_csr_mat).csr_val);
            icol = (A->opt_csr_mat).csr_col_ptr;
            // beginning of the row
            ilrow = (A->opt_csr_mat).csr_row_ptr;

            // position of the diagonal element (includes zeros) always has min(m,n) elements
            idiag = (A->opt_csr_mat).idiag;
            // ending of the row
            iurow = (A->opt_csr_mat).iurow;
        }
    }
    else if(A->opt_csc_ready)
    {
        //CSC
        a    = (T *)((A->opt_csc_mat).val);
        icol = (A->opt_csc_mat).row_idx;
        // beginning of the col
        ilrow = (A->opt_csc_mat).col_ptr;

        // position of the diagonal element (includes zeros) always has min(m,n) elements
        idiag = (A->opt_csc_mat).idiag;
        // ending of the col
        iurow = (A->opt_csc_mat).iurow;

        doid = trans_doid(doid);
    }
    else
    {
        // It should never be reached,unless there is something wrong in logic
        return aoclsparse_status_internal_error;
    }
    using namespace aoclsparse;
    using namespace kernel_templates;

    aoclsparse_index_base base = A->internal_base_index;

    /* Kernels table
     * =============
     *
     * Table key
     * =========
     *
     * Datatypes (SUF)
     * ---------
     * s   - float
     * d   - double
     * c   - cfloat
     * z   - cdouble
     * ALL - s, d, c, z
     *
     * Kernel types
     * ------------
     * Ref         - Plain c++ code
     * KT AVX2     - KT-based kernel 256-bit vectors with AVX2 ISA
     * KT AVX512VL - KT-based kernel 256-bit vectors with AVX512 ISA
     * KT AVX512F  - KT-based kernel 512-bit vectors with AVX512 ISA
     *
     * tbl offset - Offset in the Oracle table
     *
     * Operation
     * ---------
     * ====================================================================
     * Equation - L * x = alpha * b       || DOID = 12
     * ----+------------------------------+--------------+-----+------------
     * kid | Kernel Name                  | Kernel Type  | SUF | tbl offset
     * ----+------------------------------+--------------+-----+------------
     *  0  | ref_trsv_l<SUF, false>       | Ref          | ALL |    0
     *  1  | kt_trsv_l<b256, SUF, false>  | KT AVX2      | ALL |    1
     *  2  | kt_trsv_l<b256, SUF, false>  | KT AVX512VL  | ALL |    2
     *  3  | kt_trsv_l<b512, SUF, false>  | KT AVX512    | ALL |    3
     * ----+------------------------------+--------------+-----+------------
     *
     * =====================================================================
     * Equation - L^T * x = alpha * b      || DOID = 13
     * ----+------------------------------+--------------+-----+------------
     * kid | Kernel Name                  | Kernel Type  | SUF | tbl offset
     * ----+------------------------------+--------------+-----+------------
     *  0  | ref_trsv_lth<SUF, false>     | Ref          | ALL |    4
     *  1  | kt_trsv_lth<b256, SUF, false>| KT AVX2      | ALL |    5
     *  2  | kt_trsv_lth<b256, SUF, false>| KT AVX512VL  | ALL |    6
     *  3  | kt_trsv_lth<b512, SUF, false>| KT AVX512    | ALL |    7
     * ----+------------------------------+--------------+-----+------------
     *
     * =====================================================================
     * Equation - L^H * x = alpha * b     || DOID = 14
     * ----+------------------------------+--------------+-----+------------
     * kid | Kernel Name                  | Kernel Type  | SUF | tbl offset
     * ----+------------------------------+--------------+-----+------------
     *  0  | ref_trsv_lth<SUF, true>      | Ref          | ALL |    8
     *  1  | kt_trsv_lth<b256, SUF, true> | KT AVX2      | ALL |    9
     *  2  | kt_trsv_lth<b256, SUF, true> | KT AVX512VL  | ALL |    10
     *  3  | kt_trsv_lth<b512, SUF, true> | KT AVX512    | ALL |    11
     * ----+------------------------------+--------------+-----+------------
     *
     * =====================================================================
     * Equation - conj(L) * x = alpha * b  || DOID = 15
     * ----+------------------------------+--------------+-----+------------
     * kid | Kernel Name                  | Kernel Type  | SUF | tbl offset
     * ----+------------------------------+--------------+-----+------------
     *  0  | ref_trsv_l<SUF, true>        | Ref          | ALL |    12
     *  1  | kt_trsv_l<b256, SUF, true>   | KT AVX2      | ALL |    13
     *  2  | kt_trsv_l<b256, SUF, true>   | KT AVX512VL  | ALL |    14
     *  3  | kt_trsv_l<b512, SUF, true>   | KT AVX512    | ALL |    15
     * ----+------------------------------+--------------+-----+------------
     *
     * =====================================================================
     * Equation - U * x = alpha * b        || DOID = 16
     * ----+------------------------------+--------------+-----+------------
     * kid | Kernel Name                  | Kernel Type  | SUF | tbl offset
     * ----+------------------------------+--------------+-----+------------
     *  0  | ref_trsv_u<SUF, false>       | Ref          | ALL |    16
     *  1  | kt_trsv_u<b256, SUF, false>  | KT AVX2      | ALL |    17
     *  2  | kt_trsv_u<b256, SUF, false>  | KT AVX512VL  | ALL |    18
     *  3  | kt_trsv_u<b512, SUF, false>  | KT AVX512    | ALL |    19
     * ----+------------------------------+--------------+-----+------------
     *
     * =====================================================================
     * Equation - U^T * x = alpha * b      || DOID = 17
     * ----+------------------------------+--------------+-----+------------
     * kid | Kernel Name                  | Kernel Type  | SUF | tbl offset
     * ----+------------------------------+--------------+-----+------------
     *  0  | ref_trsv_uth<SUF, false>     | Ref          | ALL |    20
     *  1  | kt_trsv_uth<b256, SUF, false>| KT AVX2      | ALL |    21
     *  2  | kt_trsv_uth<b256, SUF, false>| KT AVX512VL  | ALL |    22
     *  3  | kt_trsv_uth<b512, SUF, false>| KT AVX512    | ALL |    23
     * ----+------------------------------+--------------+-----+------------
     *
     * =====================================================================
     * Equation - U^H * x = alpha * b      || DOID = 18
     * ----+------------------------------+--------------+-----+------------
     * kid | Kernel Name                  | Kernel Type  | SUF | tbl offset
     * ----+------------------------------+--------------+-----+------------
     *  0  | ref_trsv_uth<SUF, true>      | Ref          | ALL |    24
     *  1  | kt_trsv_uth<b256, SUF, true> | KT AVX2      | ALL |    25
     *  2  | kt_trsv_uth<b256, SUF, true> | KT AVX512VL  | ALL |    26
     *  3  | kt_trsv_uth<b512, SUF, true> | KT AVX512    | ALL |    27
     * ----+------------------------------+--------------+-----+------------
     *
     * =====================================================================
     * Equation - conj(U) * x = alpha * b  || DOID = 19
     * ----+------------------------------+--------------+-----+------------
     * kid | Kernel Name                  | Kernel Type  | SUF | tbl offset
     * ----+------------------------------+--------------+-----+------------
     *  0  | ref_trsv_u<SUF, true>        | Ref          | ALL |    28
     *  1  | kt_trsv_u<b256, SUF, true>   | KT AVX2      | ALL |    29
     *  2  | kt_trsv_u<b256, SUF, true>   | KT AVX512VL  | ALL |    30
     *  3  | kt_trsv_u<b512, SUF, true>   | KT AVX512    | ALL |    31
     * ----+------------------------------+--------------+-----+------------
     */
    using K = decltype(&ref_trsv_l<T, false>);

    // clang-format off
     static constexpr Dispatch::Table<K> tbl[]{
     // Lower
                      {ref_trsv_l<T>,                                       context_isa_t::GENERIC,  0U | archs::ALL},
                      {kt_trsv_l<bsz::b256, T, kt_avxext::AVX2>,            context_isa_t::AVX2,     0U | archs::ALL},
     Dispatch::ORL<K>({kt_trsv_l<bsz::b256, T, kt_avxext::AVX512VL>,        context_isa_t::AVX512VL, 0U | archs::ALL}),
     Dispatch::ORL<K>({kt_trsv_l<bsz::b512, T, kt_avxext::AVX512F>,         context_isa_t::AVX512F,  0U | archs::ALL}),
     // Lower transpose
                      {ref_trsv_lth<T>,                                     context_isa_t::GENERIC,  0U | archs::ALL},
                      {kt_trsv_lt<bsz::b256, T, kt_avxext::AVX2>,           context_isa_t::AVX2,     0U | archs::ALL},
     Dispatch::ORL<K>({kt_trsv_lt<bsz::b256, T, kt_avxext::AVX512VL>,       context_isa_t::AVX512VL, 0U | archs::ALL}),
     Dispatch::ORL<K>({kt_trsv_lt<bsz::b512, T, kt_avxext::AVX512F>,        context_isa_t::AVX512F,  0U | archs::ALL}),
     // Lower Hermitian transpose
                      {ref_trsv_lth<T, true>,                               context_isa_t::GENERIC,  0U | archs::ALL},
                      {kt_trsv_lt<bsz::b256, T, kt_avxext::AVX2, true>,     context_isa_t::AVX2,     0U | archs::ALL},
     Dispatch::ORL<K>({kt_trsv_lt<bsz::b256, T, kt_avxext::AVX512VL, true>, context_isa_t::AVX512VL, 0U | archs::ALL}),
     Dispatch::ORL<K>({kt_trsv_lt<bsz::b512, T, kt_avxext::AVX512F, true>,  context_isa_t::AVX512F,  0U | archs::ALL}),
     // Lower conjugate
                      {ref_trsv_l<T, true>,                                 context_isa_t::GENERIC,  0U | archs::ALL},
                      {kt_trsv_l<bsz::b256, T, kt_avxext::AVX2, true>,      context_isa_t::AVX2,     0U | archs::ALL},
     Dispatch::ORL<K>({kt_trsv_l<bsz::b256, T, kt_avxext::AVX512VL, true>,  context_isa_t::AVX512VL, 0U | archs::ALL}),
     Dispatch::ORL<K>({kt_trsv_l<bsz::b512, T, kt_avxext::AVX512F, true>,   context_isa_t::AVX512F,  0U | archs::ALL}),
     // Upper
                      {ref_trsv_u<T>,                                       context_isa_t::GENERIC,  0U | archs::ALL},
                      {kt_trsv_u<bsz::b256, T, kt_avxext::AVX2>,            context_isa_t::AVX2,     0U | archs::ALL},
     Dispatch::ORL<K>({kt_trsv_u<bsz::b256, T, kt_avxext::AVX512VL>,        context_isa_t::AVX512VL, 0U | archs::ALL}),
     Dispatch::ORL<K>({kt_trsv_u<bsz::b512, T, kt_avxext::AVX512F>,         context_isa_t::AVX512F,  0U | archs::ALL}),
     // Upper transpose
                      {ref_trsv_uth<T>,                                     context_isa_t::GENERIC,  0U | archs::ALL},
                      {kt_trsv_ut<bsz::b256, T, kt_avxext::AVX2>,           context_isa_t::AVX2,     0U | archs::ALL},
     Dispatch::ORL<K>({kt_trsv_ut<bsz::b256, T, kt_avxext::AVX512VL>,       context_isa_t::AVX512VL, 0U | archs::ALL}),
     Dispatch::ORL<K>({kt_trsv_ut<bsz::b512, T, kt_avxext::AVX512F>,        context_isa_t::AVX512F,  0U | archs::ALL}),
     // Upper Hermitian transpose
                      {ref_trsv_uth<T, true>,                               context_isa_t::GENERIC,  0U | archs::ALL},
                      {kt_trsv_ut<bsz::b256, T, kt_avxext::AVX2, true>,     context_isa_t::AVX2,     0U | archs::ALL},
     Dispatch::ORL<K>({kt_trsv_ut<bsz::b256, T, kt_avxext::AVX512VL, true>, context_isa_t::AVX512VL, 0U | archs::ALL}),
     Dispatch::ORL<K>({kt_trsv_ut<bsz::b512, T, kt_avxext::AVX512F, true>,  context_isa_t::AVX512F,  0U | archs::ALL}),
     // Upper conjugate
                      {ref_trsv_u<T, true>,                                 context_isa_t::GENERIC,  0U | archs::ALL},
                      {kt_trsv_u<bsz::b256, T, kt_avxext::AVX2, true>,      context_isa_t::AVX2,     0U | archs::ALL},
     Dispatch::ORL<K>({kt_trsv_u<bsz::b256, T, kt_avxext::AVX512VL, true>,  context_isa_t::AVX512VL, 0U | archs::ALL}),
     Dispatch::ORL<K>({kt_trsv_u<bsz::b512, T, kt_avxext::AVX512F, true>,   context_isa_t::AVX512F,  0U | archs::ALL}),
     };
    // clang-format on

    // Index at which traingular doids start - 12
    // Number of kernels for any given doid  - 4 (ref, AVX2, AVX512VL, AVX512F)
    aoclsparse_int doid_offset = (static_cast<aoclsparse_int>(doid) - 12);
    aoclsparse_int tbl_offset  = doid_offset * 4;

    // Number of sets of TRSV kernels present - 8
    // Each set of kernel corresponds to a unique doid
    thread_local K kache[8];

    K kernel = Dispatch::Oracle<K>(tbl, kache[doid_offset], kid, tbl_offset, tbl_offset + 4);

    if(!kernel)
        return aoclsparse_status_invalid_kid;

    aoclsparse_int *ilend;

    // ilend needs to be updated based on the fill mode
    // to point to the correct end of diagonal value
    switch(doid)
    {
    case aoclsparse::doid::tln:
    case aoclsparse::doid::tlt:
    case aoclsparse::doid::tlh:
    case aoclsparse::doid::tlc:
        ilend = idiag;
        break;
    case aoclsparse::doid::tun:
    case aoclsparse::doid::tut:
    case aoclsparse::doid::tuh:
    case aoclsparse::doid::tuc:
        ilend = iurow;
    default:
        break;
    }

    // icol the column indices
    // ilrow pointer to the start of each row
    // ilend pointer to the end of each row
    return kernel(alpha, m, base, a, icol, ilrow, ilend, b, incb, x, incx, unit);
}

#define TRSV_DISPATCHER(SUF)                                     \
    template DLL_PUBLIC aoclsparse_status aoclsparse::trsv<SUF>( \
        const aoclsparse_operation transpose,                    \
        const SUF                  alpha,                        \
        aoclsparse_matrix          A,                            \
        const aoclsparse_mat_descr descr,                        \
        const SUF                 *b,                            \
        const aoclsparse_int       incb,                         \
        SUF                       *x,                            \
        const aoclsparse_int       incx,                         \
        aoclsparse_int             kid);

INSTANTIATE_DISPATCHER(TRSV_DISPATCHER);

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

extern "C" aoclsparse_status aoclsparse_strsv(aoclsparse_operation       trans,
                                              const float                alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              const float               *b,
                                              float                     *x)
{
    const aoclsparse_int kid  = -1; /* auto */
    const aoclsparse_int incb = 1, incx = 1;

    return aoclsparse::trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsv(aoclsparse_operation       trans,
                                              const double               alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              const double              *b,
                                              double                    *x)
{
    const aoclsparse_int kid  = -1; /* auto */
    const aoclsparse_int incb = 1, incx = 1;

    return aoclsparse::trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ctrsv(aoclsparse_operation            trans,
                                              const aoclsparse_float_complex  alpha,
                                              aoclsparse_matrix               A,
                                              const aoclsparse_mat_descr      descr,
                                              const aoclsparse_float_complex *b,
                                              aoclsparse_float_complex       *x)
{
    const aoclsparse_int       kid    = -1; /* auto */
    const std::complex<float> *alphap = reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    const aoclsparse_int       incb = 1, incx = 1;

    return aoclsparse::trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ztrsv(aoclsparse_operation             trans,
                                              const aoclsparse_double_complex  alpha,
                                              aoclsparse_matrix                A,
                                              const aoclsparse_mat_descr       descr,
                                              const aoclsparse_double_complex *b,
                                              aoclsparse_double_complex       *x)
{
    const aoclsparse_int        kid    = -1; /* auto */
    const std::complex<double> *alphap = reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    const aoclsparse_int        incb = 1, incx = 1;

    return aoclsparse::trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_strsv_strided(aoclsparse_operation       trans,
                                                      const float                alpha,
                                                      aoclsparse_matrix          A,
                                                      const aoclsparse_mat_descr descr,
                                                      const float               *b,
                                                      const aoclsparse_int       incb,
                                                      float                     *x,
                                                      const aoclsparse_int       incx)
{
    const aoclsparse_int kid = -1; /* auto */

    return aoclsparse::trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsv_strided(aoclsparse_operation       trans,
                                                      const double               alpha,
                                                      aoclsparse_matrix          A,
                                                      const aoclsparse_mat_descr descr,
                                                      const double              *b,
                                                      const aoclsparse_int       incb,
                                                      double                    *x,
                                                      const aoclsparse_int       incx)
{
    const aoclsparse_int kid = -1; /* auto */

    return aoclsparse::trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ctrsv_strided(aoclsparse_operation            trans,
                                                      const aoclsparse_float_complex  alpha,
                                                      aoclsparse_matrix               A,
                                                      const aoclsparse_mat_descr      descr,
                                                      const aoclsparse_float_complex *b,
                                                      const aoclsparse_int            incb,
                                                      aoclsparse_float_complex       *x,
                                                      const aoclsparse_int            incx)
{
    const aoclsparse_int       kid    = -1; /* auto */
    const std::complex<float> *alphap = reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);

    return aoclsparse::trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ztrsv_strided(aoclsparse_operation             trans,
                                                      const aoclsparse_double_complex  alpha,
                                                      aoclsparse_matrix                A,
                                                      const aoclsparse_mat_descr       descr,
                                                      const aoclsparse_double_complex *b,
                                                      const aoclsparse_int             incb,
                                                      aoclsparse_double_complex       *x,
                                                      const aoclsparse_int             incx)
{
    const aoclsparse_int        kid    = -1; /* auto */
    const std::complex<double> *alphap = reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);

    return aoclsparse::trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_strsv_kid(aoclsparse_operation       trans,
                                                  const float                alpha,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  const float               *b,
                                                  float                     *x,
                                                  const aoclsparse_int       kid)
{
    const aoclsparse_int incb = 1, incx = 1;

    return aoclsparse::trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_dtrsv_kid(aoclsparse_operation       trans,
                                                  const double               alpha,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  const double              *b,
                                                  double                    *x,
                                                  const aoclsparse_int       kid)
{
    const aoclsparse_int incb = 1, incx = 1;

    return aoclsparse::trsv(trans, alpha, A, descr, b, incb, x, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ctrsv_kid(aoclsparse_operation            trans,
                                                  const aoclsparse_float_complex  alpha,
                                                  aoclsparse_matrix               A,
                                                  const aoclsparse_mat_descr      descr,
                                                  const aoclsparse_float_complex *b,
                                                  aoclsparse_float_complex       *x,
                                                  const aoclsparse_int            kid)
{
    const std::complex<float> *alphap = reinterpret_cast<const std::complex<float> *>(&alpha);
    const std::complex<float> *bp     = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *xp     = reinterpret_cast<std::complex<float> *>(x);
    const aoclsparse_int       incb = 1, incx = 1;

    return aoclsparse::trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}

extern "C" aoclsparse_status aoclsparse_ztrsv_kid(aoclsparse_operation             trans,
                                                  const aoclsparse_double_complex  alpha,
                                                  aoclsparse_matrix                A,
                                                  const aoclsparse_mat_descr       descr,
                                                  const aoclsparse_double_complex *b,
                                                  aoclsparse_double_complex       *x,
                                                  const aoclsparse_int             kid)
{
    const std::complex<double> *alphap = reinterpret_cast<const std::complex<double> *>(&alpha);
    const std::complex<double> *bp     = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *xp     = reinterpret_cast<std::complex<double> *>(x);
    const aoclsparse_int        incb = 1, incx = 1;

    return aoclsparse::trsv(trans, *alphap, A, descr, bp, incb, xp, incx, kid);
}
