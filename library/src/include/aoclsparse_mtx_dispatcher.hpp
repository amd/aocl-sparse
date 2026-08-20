/* ************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_MTX_DISPATCH_HPP
#define AOCLSPARSE_MTX_DISPATCH_HPP
#include "aoclsparse_descr.h"

#include <complex>
#include <utility>

namespace aoclsparse
{
    /* DOID = Descriptor + Operation ID, it flattens down all the supported combinations,
    one DOID defines quite well what a kernel needs to do, thus it can be used
    as index to a table of kernels.

    Bit layout: [group:3~6][op:2] where bit 0 = conjugate, bit 1 = transpose
    (or 0=lower, 1=upper for symmetric/hermitian). This encoding enables arithmetic
    computation of conj_doid (XOR 1), trans_doid (XOR 2 for non-triangular),
    and same-group detection (group = value >> 2). */
    enum class doid
    {
        // Group 0: general
        gn = 0, // general normal (non-transpose) full matrix as it is given
        gc, // general conjugate
        gt, // general transposed
        gh, // general conjugate transpose

        // Group 1: symmetric (bit 1 = upper flag)
        sl, // symmetric based on lower triangle
        slc, // symmetric lower conjugate
        su, // symmetric upper
        suc, // symmetric upper conjugate

        // Group 2: hermitian (bit 1 = upper flag, hu/huc swapped so trans_doid = XOR 2)
        hl, // hermitian based on lower triangle
        hlc, // hermitian lower conjugate
        huc, // hermitian upper conjugate
        hu, // hermitian upper

        // Group 3: triangular lower (bit 0 = conjugate, bit 1 = transpose)
        tln, // triangular lower normal
        tlc, // triangular lower conjugate
        tlt, // triangular lower transposed
        tlh, // triangular lower conjugate transposed

        // Group 4: triangular upper (bit 0 = conjugate, bit 1 = transpose)
        tun, // triangular upper normal
        tuc, // triangular upper conjugate
        tut, // triangular upper transposed
        tuh, // triangular upper conjugate transposed

        len = tuh + 1 // number of valid DOIDs, also used to indicate an invalid DOID
    };

    /* Given descriptor and operation, return DOID or not_implemented error for complex types.
       Uses arithmetic on the bit-encoded enum: [group:3][op:2] where
       bit 0 = conjugate, bit 1 = transpose (or upper for symm/herm). */
    template <typename T>
    inline aoclsparse::doid get_doid(const aoclsparse_mat_descr descr,
                                     const aoclsparse_operation op)
    {
        aoclsparse::doid       d_id  = doid::len;
        aoclsparse_operation   op_t  = op;
        aoclsparse_matrix_type mtx_t = descr->type;
        aoclsparse_fill_mode   fm    = descr->fill_mode;

        // For real types, simplify the operation and matrix type
        if constexpr(std::is_same_v<float, T> || std::is_same_v<double, T>)
        {
            if(op_t == aoclsparse_operation_conjugate_transpose)
                op_t = aoclsparse_operation_transpose;

            if(mtx_t == aoclsparse_matrix_type_hermitian)
                mtx_t = aoclsparse_matrix_type_symmetric;
        }

        // Transpose of a symmetric matrix is the orginal matrix
        if(mtx_t == aoclsparse_matrix_type_symmetric && op_t == aoclsparse_operation_transpose)
            op_t = aoclsparse_operation_none;
        // Conjugate transpose of a Hermitian matrix is the orginal matrix
        else if(mtx_t == aoclsparse_matrix_type_hermitian
                && op_t == aoclsparse_operation_conjugate_transpose)
            op_t = aoclsparse_operation_none;

        // API op values: none=111, transpose=112, conj_transpose=113
        aoclsparse_int op_v = op_t - 111; // 0=none, 1=transpose, 2=conj_transpose
        if(op_v < 0 || op_v > 2)
            return doid::len; // invalid op

        // Map API op to bit-encoded op: 00=none, 01=conj, 10=trans, 11=conj-trans
        // clang-format off
        static constexpr aoclsparse_int op_to_bits[] = {0, 2, 3};
        // clang-format on

        switch(mtx_t)
        {
        case aoclsparse_matrix_type_general:
            // d_id [0,3]: group 0, op_bits directly
            d_id = static_cast<aoclsparse::doid>(op_to_bits[op_v]);
            break;
        case aoclsparse_matrix_type_symmetric:
            // d_id [4,7]: group 1, bit1=upper(fm), bit0=conj(op_v>>1)
            // After normalization op_v is 0(none) or 2(conj_trans), so op_v>>1 gives conj bit
            d_id = static_cast<aoclsparse::doid>(4 + 2 * fm + (op_v >> 1));
            break;
        case aoclsparse_matrix_type_hermitian:
            // d_id [8,11]: group 2, bit1=upper(fm), bit0=conj
            // After normalization op_v is 0(none) or 1(transpose).
            // Hermitian transpose = conjugate, so transpose maps to conj variant.
            // Due to encoding swap (huc before hu): conj_bit = op_v ^ fm
            d_id = static_cast<aoclsparse::doid>(8 + 2 * fm + (op_v ^ fm));
            break;
        case aoclsparse_matrix_type_triangular:
            // d_id [12,19]: group 3(L) or 4(U), op_bits from lookup
            d_id = static_cast<aoclsparse::doid>(12 + 4 * fm + op_to_bits[op_v]);
            break;
        default:
            break;
        }

        return d_id;
    }

    /* Return DOID matching the original operation assuming that the input
       is already transposed. This is useful for binding CSC and CSR kernels.

       The data representation of CSC matrix is exactly the same as CSR transposed,
       thus using doid::gt kernel on CSC data is the same as doid::gn on CSR.
       Similarly gh<->gc, sl<->su, slc<->suc, hl<->huc, tln<->tut, tlh<->tuc, etc.

       Arithmetic: for non-triangular DOIDs (groups 0-2), XOR with 2 flips the
       transpose/upper bit.  For triangular DOIDs (groups 3-4), transposition
       flips both the fill bit and the transpose bit within the 3-bit offset,
       i.e., (d - 12) XOR 6. */
    inline aoclsparse::doid trans_doid(aoclsparse::doid d_id)
    {
        aoclsparse_int dv = static_cast<aoclsparse_int>(d_id);
        if(dv < 0 || dv >= static_cast<aoclsparse_int>(doid::len))
            return doid::len;

        if(dv < 12)
            return static_cast<aoclsparse::doid>(dv ^ 2);
        return static_cast<aoclsparse::doid>(12 + ((dv - 12) ^ 6));
    }

    /* Inline predicates to classify a DOID into kernel groups.
       These are used by get_arch_score() to determine ISA availability for a
       given effective DOID. Each predicate maps one or more DOIDs to a kernel
       family that shares the same dispatcher path and ISA support profile.
       Using simple predicates instead of an enum avoids coupling DOID classification
       with DOID scoring, keeping them independently maintainable.

       With the [group:3][op:2] bit encoding, group = value >> 2.
       Symmetric = group 1, hermitian = group 2, triangular = groups 3-4. */
    inline bool isDOIDgenNT(aoclsparse::doid d) // general: no-transpose or transpose
    {
        return d == doid::gn || d == doid::gt;
    }
    inline bool isDOIDgenH(aoclsparse::doid d) // general: conjugate-transpose (complex only)
    {
        return d == doid::gh;
    }
    inline bool isDOIDsymm(aoclsparse::doid d) // symmetric: lower/upper, with/without conjugate
    {
        return (static_cast<int>(d) >> 2) == 1; // group 1: sl=4..suc=7
    }
    inline bool isDOIDherm(aoclsparse::doid d) // hermitian: lower/upper, with/without conjugate
    {
        return (static_cast<int>(d) >> 2) == 2; // group 2: hl=8..hu=11
    }
    inline bool isDOIDtriN(aoclsparse::doid d) // triangular: no-transpose
    {
        return d == doid::tln || d == doid::tun;
    }
    inline bool isDOIDtriT(aoclsparse::doid d) // triangular: transposed
    {
        return d == doid::tlt || d == doid::tut;
    }
    inline bool isDOIDtriH(aoclsparse::doid d) // triangular: conjugate-transposed (complex only)
    {
        return d == doid::tlh || d == doid::tuh;
    }

    /* Return DOID matching the conjugated operation.
        Conjugation flips the sign of imaginary parts:
        gn↔gc, gt↔gh, sl↔slc, su↔suc, hl↔hlc, hu↔huc,
        tln↔tlc, tlt↔tlh, tun↔tuc, tut↔tuh.

        Arithmetic: XOR with 1 flips the conjugate bit (bit 0) in the
        [group:3][op:2] encoding. */
    inline aoclsparse::doid conj_doid(aoclsparse::doid d_id)
    {
        aoclsparse_int dv = static_cast<aoclsparse_int>(d_id);
        if(dv < 0 || dv >= static_cast<aoclsparse_int>(doid::len))
            return doid::len;

        return static_cast<aoclsparse::doid>(dv ^ 1);
    }

    /* Compute the effective DOID -- given the input matrix stored in given DOID
       variant, what is the DOID the kernel must dispatch to ("effective DOID")
       to achieve the requested DOID, e.g., input GT + kernel GC = requested GH.

       SAME-GROUP:
         General, triangular: operation XOR (gn/gc/gt/gh; t*n/t*c/t*t/t*h),
                      since the data layout is identical and only
                      the algebraic operation differs.
         Symm/Herm:   XOR the conjugation bit but don't cross the given triangle,
                      e.g., sl can be turned into slc but not to su.

       CROSS-GROUP (general mat -> non-general req):
         The kernel must slice the relevant structure out of the general data
         and apply any necessary transform. The effective DOID encodes family,
         fill mode, and operation (e.g., sl, tut, huc).
         - Symmetric/hermitian: eff = req ^ mat_op
         - Triangular: eff = 12 + ((req-12) ^ tri_xor_mask[mat_op])
         Other inputs are not considered. For example, if the stored matrix is
         sl, we could create tl* by slicing out the lower triangle, but we
         might as well use the original gn input matrix as it would be the same
         effort.

       -----------------------------------------------------------------------
       COMPLETE EFFECTIVE DOID TABLE
       -----------------------------------------------------------------------
       Format: eff_doid.  "—" = incompatible (returns doid::len).

       Same-group: General
       req\mat |  gn    gc    gt    gh
       --------+-----------------------
       gn      |  gn    gc    gt    gh
       gc      |  gc    gn    gh    gt
       gt      |  gt    gh    gn    gc
       gh      |  gh    gt    gc    gn

       Same-group: Symmetric
       req\mat |  sl     slc    su     suc
       --------+-----------------------------
       sl      |  gn     gc      —      —
       slc     |  gc     gn      —      —
       su      |   —      —     gn     gc
       suc     |   —      —     gc     gn

       Same-group: Hermitian (analogous to Symmetric)

       Same-group: Triangular with the same triangle (L or U) (analogous to General)
       req\mat |  *n    *c    *t    *h
       --------+-----------------------
       *n      |  gn    gc    gt    gh
       *c      |  gc    gn    gh    gt
       *t      |  gt    gh    gn    gc
       *h      |  gh    gt    gc    gn

       Cross-group: General mat -> Symmetric req
       req\mat |  gn     gc      gt     gh
       --------+------------------------------
       sl      |  sl     slc     su     suc
       slc     |  slc    sl      suc    su
       su      |  su     suc     sl     slc
       suc     |  suc    su      slc    sl

       Cross-group: General mat -> Hermitian req
       req\mat |  gn     gc      gt     gh
       --------+------------------------------
       hl      |  hl     hlc     huc    hu
       hlc     |  hlc    hl      hu     huc
       huc     |  huc    hu      hl     hlc
       hu      |  hu     huc     hlc    hl

       Cross-group: General mat -> Triangular Lower req
       req\mat |  gn     gc      gt     gh
       --------+------------------------------
       tln     |  tln    tlc     tut    tuh
       tlc     |  tlc    tln     tuh    tut
       tlt     |  tlt    tlh     tun    tuc
       tlh     |  tlh    tlt     tuc    tun

       Cross-group: General mat -> Triangular Upper req
       req\mat |  gn     gc      gt     gh
       --------+------------------------------
       tun     |  tun    tuc     tlt    tlh
       tuc     |  tuc    tun     tlh    tlt
       tut     |  tut    tuh     tln    tlc
       tuh     |  tuh    tut     tlc    tln

       All other combinations return doid::len (incompatible).

       Note that trans_doid() is just special case of get_effective_doid(gt,*)
       and similarly, conj_doid() is get_effective_doid(gc,*).
       ----------------------------------------------------------------------- */
    inline aoclsparse::doid get_effective_doid(aoclsparse::doid mat_doid, aoclsparse::doid req_d_id)
    {
        int mat_v = static_cast<int>(mat_doid);
        int req_v = static_cast<int>(req_d_id);

        // mat_grp / req_grp: matrix family
        //   0 = general, 1 = symmetric, 2 = hermitian, 3 = triangular-lower, 4 = triangular-upper
        int mat_grp = mat_v >> 2;
        int req_grp = req_v >> 2;

        // mat_op: operation bits of the stored matrix
        //   0 = normal (no-op), 1 = conjugate, 2 = transpose, 3 = conjugate-transpose (or L/U for symm/herm)
        int mat_op = mat_v & 3;
        int req_op = req_v & 3;

        // ---- Same group ----
        if(mat_grp == req_grp)
        {
            // General or triangular: the effective DOID is the pure operation difference.
            // Symmetric/hermitian: need to filter out different triangles, i.e., accept
            // only exact (op_xor=0) and conjugate (op_xor=1) differences,
            // op_xor >= 2 crosses lower↔upper → incompatible.
            if(mat_grp == 0 || mat_grp >= 3 || (mat_op ^ req_op) <= 1)
                return static_cast<doid>(mat_op ^ req_op);
            return doid::len;
        }

        // ---- Cross-group: only general mat (group 0) -> non-general req ----
        if(mat_grp != 0)
            return doid::len;

        // Triangular cross-group
        if(req_grp >= 3)
        {
            // clang-format off
            static constexpr int tri_xor_mask[] = {0, 1, 6, 7};
            // clang-format on
            return static_cast<doid>(12 + ((req_v - 12) ^ tri_xor_mask[mat_op]));
        }

        // Symmetric/hermitian cross-group
        return static_cast<doid>(req_v ^ mat_op);
    }
}

#endif
