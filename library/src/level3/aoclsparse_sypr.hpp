/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef AOCLSPARSE_SYPR_HPP
#define AOCLSPARSE_SYPR_HPP

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_mat_structures.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <vector>
/* Add alpha-multiplication of a sparse row of W matrix to a dense vector
 * representing row 'i', but keeping track of the filled indices and their number.
 *
 * The sparse row is stored in arrays icolW[], valW[] from [iwstart, iwend)
 * (not including iwend), icolW are baseW.
 * The output is stored as follows:
 *   nnz[] dense flag array, nnz[j]==i means that there is j-th element
 *   val[] dense array storing the values, e.g., val[j]
 *   icolC[], nnzC store the list of elements and their number, e.g., icolC[*nnzC-1]=j
 *
 * It is possible to only count the number of nnz (REQUEST=aoclsparse_stage_nnz_count),
 * if full computation is done, icolC[] must be big enough.
 * By default all elements are build but it is possible to build only the upper
 * triangle (BUILD_ONLY_U=true).
 * If CONJ_W=true, conjugate W values during multiplication (used for CSC A path).
 */
template <typename T, aoclsparse_request REQUEST, bool BUILD_ONLY_U = false, bool CONJ_W = false>
void inline add_sprow(aoclsparse_int               i,
                      T                            alpha,
                      aoclsparse_int               iwstart,
                      aoclsparse_int               iwend,
                      const aoclsparse_int        *icolW,
                      const T                     *valW,
                      aoclsparse_index_base        baseW,
                      std::vector<aoclsparse_int> &nnz,
                      std::vector<T>              &val,
                      aoclsparse_int              *icolC,
                      aoclsparse_int              *nnzC,
                      int64_t                     *total_nnz)
{

    for(aoclsparse_int idxW = iwstart; idxW < iwend; ++idxW)
    {
        // mark all the nonzeroes in the flag array
        aoclsparse_int j = icolW[idxW] - baseW;

        if constexpr(BUILD_ONLY_U)
            if(j < i) // L triangle element, skip
                continue;

        if constexpr(REQUEST == aoclsparse_stage_nnz_count)
        {
            if(nnz[j] != i)
            {
                // newly created nonzero
                nnz[j] = i;
                (*total_nnz)++;
            }
        }
        else
        {
            // Get the W value, optionally conjugated for CSC A path
            T valW_eff;
            if constexpr(CONJ_W)
                valW_eff = aoclsparse::conj(valW[idxW]);
            else
                valW_eff = valW[idxW];

            if(nnz[j] != i)
            {
                // newly created nonzero
                nnz[j]            = i;
                icolC[*total_nnz] = j;
                (*total_nnz)++;
                val[j] = alpha * valW_eff;
            }
            else
            {
                // compute values
                val[j] += alpha * valW_eff;
            }
        }
    }
    *nnzC = static_cast<aoclsparse_int>(*total_nnz);
}

/* On Fly Transposition of m x n sorted CSR matrix
 * We can use this algorithm on assumption that CSR is sorted and we need
 * to go through the transposed matrix just once and in order of the columns.
 * It can also be used to transpose only part of the matrix, e.g., one triangle
 * to make matrix symmetric.
 * It works on the principle that we keep pointers to the first unused element
 * in each row and split them (in a linked list) to groups building the
 * individual columns.
 */
class oftrans
{
    // row start - index of the first element in each row to work with
    // e.g., initialize as if aoclsparse_int irstart[m] = {icrowA[0..m-1]};
    // in the middle of the algorithm, column j will be generated by
    // nonzeros from transposed j-th row: idx = irstart[j] ... irend[j]-e_offset-1
    std::vector<aoclsparse_int> irstart;

    // row end - index of the first element not to use in each row
    // (subject to r_offset), size m
    const aoclsparse_int *irend = NULL;

    // row end offset - access all elements up to 'irend[]-e_offset'
    aoclsparse_int e_offset = 0;

    // column indices of the matrix in CSR order
    // Row 'i' we want to transpose is icol[irstart[i]...irend[i]-e_offset-1]
    const aoclsparse_int *icol = NULL;

    // 0/1-base for icol indices
    aoclsparse_index_base base = aoclsparse_index_base_zero;

    // linked list with head implemented as arrays used to track what rows
    // have their first nonzero in what column
    // -1 serves as the terminator
    // col_head[j] is row number which has the first unused element in column j
    std::vector<aoclsparse_int> col_head;

    // row_next[i] is the row index which has the first
    // (unused) nonzero in the same column as row i,
    // or -1 if no such other one exists
    std::vector<aoclsparse_int> row_next;

public:
    // Initialize OnFlyTranspose for m x n matrix
    // from each row, indices irstart[i]-s_offset ... irend[i]-e_offset will be considered
    // their column indices are icol[]-base
    // s_offset/e_offset can be handy to correct base or working with a triangle
    // To return aoclsparse_status, have this instead of a constructor.
    aoclsparse_status init(aoclsparse_int        m,
                           aoclsparse_int        n,
                           const aoclsparse_int *irstartA,
                           aoclsparse_int        s_offsetA,
                           const aoclsparse_int *irendA,
                           aoclsparse_int        e_offsetA,
                           const aoclsparse_int *icolA,
                           aoclsparse_index_base baseA)
    {

        if(irstartA == nullptr || irendA == nullptr || icolA == nullptr)
            return aoclsparse_status_invalid_pointer;

        try
        {
            irstart.resize(m);
            col_head.resize(n, -1);
            row_next.resize(m, -1);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }

        irend    = irendA;
        e_offset = e_offsetA;
        icol     = icolA;
        base     = baseA;

        // Initialize the linked list
        // Look at the first nonzero in each row and assign it to the matching col_head
        aoclsparse_int idx, idxend, j;
        for(aoclsparse_int i = 0; i < m; i++)
        {
            // first element to use in the i-th row has index irstart[i] - s_offset
            idx        = irstartA[i] - s_offsetA;
            irstart[i] = idx;
            idxend     = irend[i] - e_offset;
            if(idx < idxend)
            { // row not empty
                j = icol[idx] - base;

                // push row i to the head of the linked list of j-th column
                row_next[i] = col_head[j];
                col_head[j] = i;
            }
        }
        return aoclsparse_status_success;
    }

    // Return first row with element in the column j
    // (assuming all columns <j were already treated and j is within the range)
    aoclsparse_int rfirst(aoclsparse_int j)
    {
        return col_head[j];
    }

    // Return index of the element in the row
    aoclsparse_int ridx(aoclsparse_int row)
    {
        return irstart[row];
    }

    // Return the next row building the same column and mark current row's element as used
    aoclsparse_int rnext(aoclsparse_int row)
    {
        aoclsparse_int idx, idxend, j, row_nextone;

        row_nextone = row_next[row];

        // look at the next nonzero in our row (if exists)
        idx    = ++irstart[row];
        idxend = irend[row] - e_offset;
        if(idx < idxend)
        {
            j = icol[idx] - base;

            // plug the row into j-th linked list
            row_next[row] = col_head[j];
            col_head[j]   = row;
        }

        return row_nextone;
    }
};

/* Compute C = (A+A^T)*B in the sense that A is symmetrized (or 'Hermitiezed'
 * for complex) matrix based on 'islower' triangle. A is sorted CSR m x k (k<=m),
 * B is m x n and the result is 0-based CSR m x n.
 * If REQUEST=aoclsparse_stage_nnz_count, only icrowC and nnzC is built,
 * irowC[k+1] needs to be already allocated.
 * Otherwise all C arrays need to be allocated to big enough size (not checked).
 * If CONJ_B=true, conjugate B values during multiplication (used for CSC A path).
 */
template <typename T, aoclsparse_request REQUEST, bool CONJ_B = false>
aoclsparse_status aoclsparse_sp2m_online_symab(aoclsparse_int        m,
                                               aoclsparse_int        k,
                                               aoclsparse_int        n,
                                               aoclsparse_index_base baseA,
                                               const aoclsparse_int *icrowA,
                                               const aoclsparse_int *idiagA,
                                               const aoclsparse_int *icolA,
                                               const T              *valA,
                                               aoclsparse_index_base baseB,
                                               const aoclsparse_int *icrowB,
                                               const aoclsparse_int *icolB,
                                               const T              *valB,
                                               const bool            islower,
                                               aoclsparse_int       *icrowC,
                                               aoclsparse_int       *icolC,
                                               T                    *valC,
                                               aoclsparse_int       *nnzC)
{

    if(icrowA == nullptr || idiagA == nullptr || icolA == nullptr || valA == nullptr
       || icrowB == nullptr || icolB == nullptr || valB == nullptr || icrowC == nullptr
       || nnzC == nullptr)
        return aoclsparse_status_invalid_pointer;
    if constexpr(REQUEST != aoclsparse_stage_nnz_count)
        if(icolC == nullptr || valC == nullptr)
            return aoclsparse_status_invalid_pointer;

    aoclsparse_status status;
    aoclsparse_int    idx, idxa, row, colA;

    // flag array of nonzeroes, init to -1
    // when building row i, if (nnz[j]==i) --> there is a nonzero in j-th column
    std::vector<aoclsparse_int> nnz;

    // array to hold values of one row of C
    std::vector<T> val;

    try
    {
        nnz.resize(n, -1);
        val.resize(n, 0);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    aoclsparse_int        s_offset, e_offset, s_offsetT;
    const aoclsparse_int *irstart, *irend;
    if(islower)
    {
        // for transpose:  icrowA[i] - baseA ... idiagA[i] - baseA
        // for normal:     icrowA[i] - baseA ... idiagA[i] + 1 - baseA
        irstart   = icrowA;
        s_offsetT = baseA;
        s_offset  = baseA;
        irend     = idiagA;
        e_offset  = baseA - 1;
    }
    else // fill in upper triangle
    {
        // for transpose: idiagA[i] + 1 - baseA ... icrowA[i + 1] - baseA
        // for normal:    idiagA[i] - baseA     ... icrowA[i + 1] - baseA
        irstart   = idiagA;
        s_offsetT = baseA - 1;
        s_offset  = baseA;
        irend     = icrowA + 1;
        e_offset  = baseA;
    }
    // On Fly Transpose
    oftrans oft;
    status = oft.init(m, k, irstart, s_offsetT, irend, baseA, icolA, baseA);
    if(status != aoclsparse_status_success)
        return status;

    *nnzC             = 0;
    icrowC[0]         = 0;
    int64_t total_nnz = 0;
    // Build i-th row of C, thus pass i-th row of A and symmetrize it
    for(aoclsparse_int i = 0; i < k; i++)
    {
        // Process the lower/upper half of symmetric matrix
        // including diagonal element
        // which is multiplied like usual sp2m routine.
        for(idxa = irstart[i] - s_offset; idxa < irend[i] - e_offset; ++idxa)
        {
            colA = icolA[idxa] - baseA;
            add_sprow<T, REQUEST, false, CONJ_B>(i,
                                                 valA[idxa],
                                                 icrowB[colA] - baseB,
                                                 icrowB[colA + 1] - baseB,
                                                 icolB,
                                                 valB,
                                                 baseB,
                                                 nnz,
                                                 val,
                                                 icolC,
                                                 nnzC,
                                                 &total_nnz);
        }

        // Multiply the other half of the symmetric or hermitian matrix,
        // which is created on the fly, using the linked list created
        // in the beginning
        row = oft.rfirst(i);
        while(row >= 0)
        {
            idxa    = oft.ridx(row);
            T val_A = aoclsparse::conj(valA[idxa]);

            add_sprow<T, REQUEST, false, CONJ_B>(i,
                                                 val_A,
                                                 icrowB[row] - baseB,
                                                 icrowB[row + 1] - baseB,
                                                 icolB,
                                                 valB,
                                                 baseB,
                                                 nnz,
                                                 val,
                                                 icolC,
                                                 nnzC,
                                                 &total_nnz);

            row = oft.rnext(row);
        }
        // i-th row of C is finished, copy out values valC[] <-- val[]
        icrowC[i + 1] = *nnzC;

        if constexpr(REQUEST != aoclsparse_stage_nnz_count)
        {
            for(idx = icrowC[i]; idx < icrowC[i + 1]; ++idx)
            {
                valC[idx]       = val[icolC[idx]];
                val[icolC[idx]] = 0.;
            }
        }
    }
    if constexpr(REQUEST == aoclsparse_stage_nnz_count)
    {
        // Check for overflow AFTER loop (no UB occurred since all arithmetic was 64-bit)
        if(total_nnz > aoclsparse_numeric::int_max)
            return aoclsparse_status_invalid_size;
    }
    return aoclsparse_status_success;
}

/* Computes C = A^T*B (or A^H*B for complex types) where A is sorted CSR m x k,
 * B is sorted CSR m x n and the result C of dimension k x n will have baseC.
 * If REQUEST=aoclsparse_stage_nnz_count, only icrowC and nnzC is built,
 * irowC[k+1] needs to be already allocated.
 * Otherwise all C arrays need to be allocated to big enough size (not checked).
 * BUILD_ONLY_U=true creates only upper triangle, otherwise full matrix.
 * If CONJ_A=true, conjugate A values during multiplication (default for A^H*B).
 * If CONJ_A=false, do not conjugate A values.
 * If CONJ_B=true, conjugate B values during multiplication.
 */
template <typename T,
          aoclsparse_request REQUEST,
          bool               BUILD_ONLY_U = false,
          bool               CONJ_A       = true,
          bool               CONJ_B       = false>
aoclsparse_status aoclsparse_sp2m_online_atb(aoclsparse_int        m,
                                             aoclsparse_int        k,
                                             aoclsparse_int        n,
                                             aoclsparse_index_base baseA,
                                             const aoclsparse_int *icrowA,
                                             const aoclsparse_int *icolA,
                                             const T              *valA,
                                             aoclsparse_index_base baseB,
                                             const aoclsparse_int *icrowB,
                                             const aoclsparse_int *icolB,
                                             const T              *valB,
                                             aoclsparse_index_base baseC,
                                             aoclsparse_int       *icrowC,
                                             aoclsparse_int       *icolC,
                                             T                    *valC,
                                             aoclsparse_int       *nnzC)
{

    if(icrowA == nullptr || icolA == nullptr || valA == nullptr || icrowB == nullptr
       || icolB == nullptr || valB == nullptr || icrowC == nullptr || nnzC == nullptr)
        return aoclsparse_status_invalid_pointer;
    if constexpr(REQUEST != aoclsparse_stage_nnz_count)
        if(icolC == nullptr || valC == nullptr)
            return aoclsparse_status_invalid_pointer;

    aoclsparse_int    idx, idxa, row;
    aoclsparse_status status;

    // flag array of nonzeroes, init to -1
    // when building row i, if (nnz[j]==i) --> there is a nonzero in j-th column
    std::vector<aoclsparse_int> nnz;

    // array to hold values of one row of C
    std::vector<T> val;

    try
    {
        nnz.resize(n, -1);
        val.resize(n, 0);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    // On Fly Transpose
    oftrans oft;
    status = oft.init(m, k, icrowA, baseA, icrowA + 1, baseA, icolA, baseA);
    if(status != aoclsparse_status_success)
        return status;

    *nnzC             = 0;
    icrowC[0]         = 0;
    int64_t total_nnz = 0;
    // Build i-th row of C, thus pass i-th column of A
    for(aoclsparse_int i = 0; i < k; i++)
    {
        row = oft.rfirst(i);
        while(row >= 0)
        {
            idxa = oft.ridx(row);
            T val_A;
            if constexpr(CONJ_A)
                val_A = aoclsparse::conj(valA[idxa]);
            else
                val_A = valA[idxa];

            add_sprow<T, REQUEST, BUILD_ONLY_U, CONJ_B>(i,
                                                        val_A,
                                                        icrowB[row] - baseB,
                                                        icrowB[row + 1] - baseB,
                                                        icolB,
                                                        valB,
                                                        baseB,
                                                        nnz,
                                                        val,
                                                        icolC,
                                                        nnzC,
                                                        &total_nnz);

            row = oft.rnext(row);
        }
        // i-th row of C is finished, copy out values valC[] <-- val[]
        // if REQUEST != aoclsparse_nnz_count, icrowC is already filled in
        // but the number is the same so we can overwrite and it will allow
        // us to run both stages together (if we have overestimate of nnzC)
        icrowC[i + 1] = *nnzC;
        if constexpr(REQUEST != aoclsparse_stage_nnz_count)
        {
            for(idx = icrowC[i]; idx < icrowC[i + 1]; ++idx)
            {
                valC[idx]       = val[icolC[idx]];
                val[icolC[idx]] = 0.;
            }
        }
    }

    if constexpr(REQUEST == aoclsparse_stage_nnz_count)
    {
        // Check for overflow AFTER loop (no UB occurred since all arithmetic was 64-bit)
        if(total_nnz > aoclsparse_numeric::int_max)
            return aoclsparse_status_invalid_size;
    }
    // correct base if needed, by default it is 0-based
    if(baseC == aoclsparse_index_base_one)
    {
        for(aoclsparse_int i = 0; i <= k; i++)
            icrowC[i]++;
        for(idx = 0; idx < *nnzC; idx++)
            icolC[idx]++;
    }
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_sypr_t(aoclsparse_operation       opA,
                                    const aoclsparse_matrix    A,
                                    const aoclsparse_matrix    B,
                                    const aoclsparse_mat_descr descrB,
                                    aoclsparse_matrix         *C,
                                    aoclsparse_request         request)
{
    aoclsparse_status           status = aoclsparse_status_success;
    aoclsparse_int             *icrowA = NULL;
    aoclsparse_int             *icolA  = NULL;
    T                          *valA   = NULL;
    aoclsparse_int             *icrowB = NULL;
    aoclsparse_int             *icolB  = NULL;
    T                          *valB   = NULL;
    std::vector<aoclsparse_int> icrowAt;
    std::vector<aoclsparse_int> icolAt;
    std::vector<T>              valAt;

    if((request != aoclsparse_stage_full_computation) && (request != aoclsparse_stage_nnz_count)
       && (request != aoclsparse_stage_finalize))
        return aoclsparse_status_invalid_value;

    if(opA != aoclsparse_operation_none && opA != aoclsparse_operation_transpose
       && opA != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;

    if(descrB == nullptr)
        return aoclsparse_status_invalid_pointer;

    if((A == nullptr) || (B == nullptr) || (C == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(request != aoclsparse_stage_finalize)
        *C = NULL; // unless it is second stage, we don't expect anything on input

    if((A->input_format != aoclsparse_csr_mat) || (B->input_format != aoclsparse_csr_mat))
    {
        return aoclsparse_status_not_implemented;
    }
    // Only CSR matrix format is supported
    aoclsparse::csr *A_csr = A->get_first_mtx_if_valid<aoclsparse::csr>();
    if(!A_csr)
        return aoclsparse_status_invalid_pointer;
    bool is_doid_gt = (A_csr->doid == aoclsparse::doid::gt);
    if(!is_doid_gt && A_csr->doid != aoclsparse::doid::gn)
        return aoclsparse_status_not_implemented;
    {
        aoclsparse::csr *B_csr_first = B->get_first_mtx_if_valid<aoclsparse::csr>();
        if(!B_csr_first)
            return aoclsparse_status_invalid_pointer;
        if(B_csr_first->doid != aoclsparse::doid::gn)
            return aoclsparse_status_not_implemented;
    }

    // C = op(A)·B·op(A)^H is Hermitian only when op is none or conj_trans.
    // op_transpose with complex types would give C = A^T·B·A which is NOT Hermitian — block it.
    if((A->val_type == aoclsparse_cmat || A->val_type == aoclsparse_zmat)
       && opA == aoclsparse_operation_transpose)
        return aoclsparse_status_not_implemented;

    // For CSC (doid::gt), internal storage is A^T. Flip op so the unchanged CSR
    // compute paths see the correct mathematical semantics. conj_flip=true signals
    // that conjugation moves from the pre-loop into kernel CONJ_B/CONJ_A template
    // parameters (complex CSC+op_none case only).
    //
    // Supported operations by format and type (C = op(A)·B·op(A)^H):
    //   CSR (doid::gn)  Real:    op_none → A·B·A^T,           op_t → A^T·B·A
    //   CSR (doid::gn)  Complex: op_none → A·B·A^H,           op_h → A^H·B·A
    //   CSC (doid::gt)  Real:    op_none → A^T·B·A,           op_t → A·B·A^T
    //   CSC (doid::gt)  Complex: op_none → A^T·B·conj(A),     op_h → conj(A)·B·A^T
    //
    // After the op-flip, eff_op drives the same two compute branches for both formats:
    //   eff_op=t/h:  CSR op_t/h  → A^T·B·A (real),           A^H·B·A (complex)
    //                CSC op_none → A^T·B·A (real), A^T·B·conj(A) (complex) [conj_flip=true]
    //   eff_op=none: CSR op_none → A·B·A^T (real),            A·B·A^H (complex)
    //                CSC op_t/h  → A·B·A^T (real),     conj(A)·B·A^T (complex)
    aoclsparse_operation eff_op    = opA;
    bool                 conj_flip = false;
    if(is_doid_gt)
    {
        if(opA == aoclsparse_operation_none)
        {
            // CSC op_none → use op_t path (A^T data used directly, no allocation)
            eff_op    = aoclsparse_operation_transpose;
            conj_flip = (A->val_type == aoclsparse_cmat || A->val_type == aoclsparse_zmat);
            //           ^^^^ true only for complex — flips CONJ_A/CONJ_B in kernels
        }
        else
        {
            // CSC op_t (real) / op_h (all) → use op_none path (csr2csc recovers A).
            // conj_flip stays false: csr2csc recovers A with plain values; atb<CONJ_A=true>
            // handles A^H in-kernel. Pre-loop conjugation is skipped via is_doid_gt guard below.
            eff_op = aoclsparse_operation_none;
        }
    }
    if(A->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }

    if(B->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }

    if(!B->is_descr_matching(descrB))
        return aoclsparse_status_invalid_value;
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        if(descrB->type != aoclsparse_matrix_type_symmetric)
        {
            return aoclsparse_status_invalid_value;
        }
    }
    if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>)
    {
        if(descrB->type != aoclsparse_matrix_type_hermitian)
        {
            return aoclsparse_status_invalid_value;
        }
    }
    if(descrB->diag_type != aoclsparse_diag_type_non_unit)
    {
        return aoclsparse_status_not_implemented;
    }

    // Check for size of symmetric matrix B
    if(B->m != B->n)
        return aoclsparse_status_invalid_size;

    // Size of Symmetric matrix B
    aoclsparse_int m = B->m;

    aoclsparse_int n;
    if((opA == aoclsparse_operation_transpose) || (opA == aoclsparse_operation_conjugate_transpose))
    {
        n = A->n;
        if(m != A->m)
            return aoclsparse_status_invalid_size;
    }
    else // (opA == aoclsparse_operation_none)
    {
        n = A->m;
        if(m != A->n)
            return aoclsparse_status_invalid_size;
    }

    aoclsparse::csr *B_opt_csr = nullptr;
    aoclsparse::csr *C_csr     = nullptr;
    if(A_csr->base != aoclsparse_index_base_zero && A_csr->base != aoclsparse_index_base_one)
        return aoclsparse_status_invalid_value;
    if(*C)
    {
        C_csr = (*C)->get_first_mtx_if_valid<aoclsparse::csr>();
    }

    // Basic check if 2nd stage was called without the first
    if(request == aoclsparse_stage_finalize
       && (*C == nullptr || C_csr == nullptr || C_csr->ptr == nullptr || C_csr->ind == nullptr
           || C_csr->val == nullptr || (*C)->m != n || (*C)->n != n))
        return aoclsparse_status_invalid_value;

    // Quick return for size 0 matrices, Do nothing
    // Return Valid Non-NULL pointers of C array.
    if((m == 0) || (n == 0) || (A->nnz == 0) || (B->nnz == 0))
    {
        if(*C == nullptr)
        {
            try
            {
                *C = new _aoclsparse_matrix;
                // Constructor handles row pointer initialization for empty matrix
                C_csr = new aoclsparse::csr(
                    n, n, 0, aoclsparse_csr_mat, aoclsparse_index_base_zero, get_data_type<T>());
                (*C)->mats.push_back(C_csr);
            }
            catch(std::bad_alloc &)
            {
                if(C_csr)
                {
                    delete C_csr;
                }
                if(*C)
                {
                    aoclsparse_destroy(C);
                }
                return aoclsparse_status_memory_error;
            }
            aoclsparse_init_mat(*C, n, n, 0, aoclsparse_csr_mat);
            (*C)->val_type = get_data_type<T>();
        }
        return aoclsparse_status_success;
    }

    // we need fully sorted rows if we apply on-fly transposition, thus B needs
    // to be sorted every time and A when we don't explicitly transpose
    if((A->sort != aoclsparse_fully_sorted && eff_op != aoclsparse_operation_none)
       || B->sort != aoclsparse_fully_sorted)
        return aoclsparse_status_unsorted_input;

    // If OP(A) == transpose/transpose_conjugate, C = At * B * A
    // Size of matrix B should be equal to #rows of matrix A(#cols of matrix At)
    // Size of resultant symmetric matrix C will be #columns of matrix A
    // Extract CSR arrays of matrix A as is
    // We will perform B * A first to generate intermediate product T matrix
    // And then perform At * T, without explicitly transposing A matrix.
    if((eff_op == aoclsparse_operation_transpose)
       || (eff_op == aoclsparse_operation_conjugate_transpose))
    {
        icrowA = A_csr->ptr;
        icolA  = A_csr->ind;
        valA   = (T *)A_csr->val;
    }
    // If OP(A) == none, C = A * B * At
    // Size of matrix B should be equal to #cols of matrix A(#rows of matrix At)
    // Size of resultant symmetric matrix C will be #rows of matrix A
    // Transpose matrix A and extract CSR arrays of matrix At
    // We will perform B * At first to generate intermediate product T matrix
    // And then perform A * T which is same as (At)t * T , without explicitly transposing At matrix.
    else // (opA == aoclsparse_operation_none)
    {
        try
        {
            icrowAt.resize(A_csr->n + 1);
            icolAt.resize(A->nnz);
            valAt.resize(A->nnz);
            icrowA = icrowAt.data();
            icolA  = icolAt.data();
            valA   = valAt.data();
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }

        status = aoclsparse_csr2csc_template(A_csr->m,
                                             A_csr->n,
                                             A->nnz,
                                             A_csr->base,
                                             A_csr->base,
                                             A_csr->ptr,
                                             A_csr->ind,
                                             (const T *)A_csr->val,
                                             icolA,
                                             icrowA,
                                             valA);
        if(status != aoclsparse_status_success)
            return aoclsparse_status_memory_error;

        // For CSR op_none: must conjugate — pre-loop produces A^H for correct Hermitian output.
        // For CSC op_h/op_t (is_doid_gt=true): csr2csc(A^T)=A with plain values;
        // atb<CONJ_A=true> handles A^H in-kernel — skip pre-loop.
        if(!is_doid_gt)
            for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
                valA[idx] = aoclsparse::conj(valA[idx]);
    }

    status = aoclsparse_csr_csc_optimize<T>(B, B_opt_csr);
    if(status != aoclsparse_status_success)
        return status;
    if(!B_opt_csr)
        return aoclsparse_status_internal_error;
    aoclsparse_index_base baseA = A_csr->base;
    aoclsparse_index_base baseB = B_opt_csr->base;
    // Retrieve CSR arrays of matrix B from optimised CSR.
    icrowB = B_opt_csr->ptr;
    icolB  = B_opt_csr->ind;
    valB   = (T *)B_opt_csr->val;

    const bool islower = (descrB->fill_mode == aoclsparse_fill_mode_lower);

    // Compressed row pointers of intermediate temporary matrix T = B*A
    std::vector<aoclsparse_int> icrowT;
    std::vector<T>              valT;
    std::vector<aoclsparse_int> icolT;
    aoclsparse_int              nnzT = 0;

    // Compressed row pointers of final matrix C = A'* T, where T = B*A
    aoclsparse_int *icrowC = NULL;
    T              *nullT  = NULL;
    aoclsparse_int  nnzC   = 0;

    try
    {
        icrowT.resize(m + 1);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    // Stage 1 — count nnz of T = sym(B)·A
    // T = sym(B)·A: CSC (doid::gt) Complex op_none → CONJ_B=true: B·conj(A^T); all others → CONJ_B=false: B·A
    if(conj_flip)
        status = aoclsparse_sp2m_online_symab<T, aoclsparse_stage_nnz_count, true>( // CONJ_B=true
            m,
            m,
            n,
            baseB,
            icrowB,
            B_opt_csr->idiag,
            icolB,
            valB,
            baseA,
            icrowA,
            icolA,
            valA,
            islower,
            icrowT.data(),
            icolT.data(),
            valT.data(),
            &nnzT);
    else
        status = aoclsparse_sp2m_online_symab<T,
                                              aoclsparse_stage_nnz_count,
                                              false>( // CONJ_B=false (default)
            m,
            m,
            n,
            baseB,
            icrowB,
            B_opt_csr->idiag,
            icolB,
            valB,
            baseA,
            icrowA,
            icolA,
            valA,
            islower,
            icrowT.data(),
            icolT.data(),
            valT.data(),
            &nnzT);
    if(status != aoclsparse_status_success)
        return status;

    // Resize the temporary arrays to exact size as returned from multiplication routine
    try
    {
        icolT.resize(nnzT);
        valT.resize(nnzT);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    // Stage 1 — compute values of T = sym(B)·A
    // T = sym(B)·A: CSC (doid::gt) Complex op_none → CONJ_B=true: B·conj(A^T); all others → CONJ_B=false: B·A
    if(conj_flip)
        status = aoclsparse_sp2m_online_symab<T, aoclsparse_stage_finalize, true>( // CONJ_B=true
            m,
            m,
            n,
            baseB,
            icrowB,
            B_opt_csr->idiag,
            icolB,
            valB,
            baseA,
            icrowA,
            icolA,
            valA,
            islower,
            icrowT.data(),
            icolT.data(),
            valT.data(),
            &nnzT);
    else
        status = aoclsparse_sp2m_online_symab<T,
                                              aoclsparse_stage_finalize,
                                              false>( // CONJ_B=false (default)
            m,
            m,
            n,
            baseB,
            icrowB,
            B_opt_csr->idiag,
            icolB,
            valB,
            baseA,
            icrowA,
            icolA,
            valA,
            islower,
            icrowT.data(),
            icolT.data(),
            valT.data(),
            &nnzT);

    if(status != aoclsparse_status_success)
        return status;

    // Next perform A * T or At * T to generate final product matrix C.
    if(request == aoclsparse_stage_full_computation || request == aoclsparse_stage_nnz_count)
    {
        try
        {
            *C    = new _aoclsparse_matrix;
            C_csr = new aoclsparse::csr(
                n, n, -1, aoclsparse_csr_mat, aoclsparse_index_base_zero, get_data_type<T>());
            icrowC = C_csr->ptr;
            (*C)->mats.push_back(C_csr);
        }
        catch(std::bad_alloc &)
        {
            if(C_csr)
            {
                delete C_csr;
            }
            aoclsparse_destroy(C);
            return aoclsparse_status_memory_error;
        }
        // Stage 2 — count nnz of C = op(A)^H·T
        // C = op(A)^H·T: CSC (doid::gt) Complex op_none → CONJ_A=false: A^T·T; all others → CONJ_A=true: A^H·T
        if(conj_flip)
            status = aoclsparse_sp2m_online_atb<T,
                                                aoclsparse_stage_nnz_count,
                                                true,
                                                false>( // CONJ_A=false
                m,
                n,
                n,
                baseA,
                icrowA,
                icolA,
                valA,
                aoclsparse_index_base_zero,
                icrowT.data(),
                icolT.data(),
                valT.data(),
                aoclsparse_index_base_zero,
                icrowC,
                NULL,
                nullT,
                &nnzC);
        else
            status = aoclsparse_sp2m_online_atb<T,
                                                aoclsparse_stage_nnz_count,
                                                true>( // CONJ_A=true (default)
                m,
                n,
                n,
                baseA,
                icrowA,
                icolA,
                valA,
                aoclsparse_index_base_zero,
                icrowT.data(),
                icolT.data(),
                valT.data(),
                aoclsparse_index_base_zero,
                icrowC,
                NULL,
                nullT,
                &nnzC);
        if(status != aoclsparse_status_success)
        {
            aoclsparse_destroy(C); // C is incomplete, so destroy it
            return status;
        }

        try
        {
            C_csr->ind = new aoclsparse_int[nnzC];
            C_csr->val = ::operator new(nnzC * sizeof(T));
        }
        catch(std::bad_alloc &)
        {
            aoclsparse_destroy(C); // C is incomplete, so destroy it
            return aoclsparse_status_memory_error;
        }
        aoclsparse_init_mat(*C, n, n, nnzC, aoclsparse_csr_mat);
        (*C)->val_type = get_data_type<T>();
    }

    if(request == aoclsparse_stage_full_computation || request == aoclsparse_stage_finalize)
    {
        // Stage 2 — compute values of C = op(A)^H·T
        // C = op(A)^H·T: CSC (doid::gt) Complex op_none → CONJ_A=false: A^T·T; all others → CONJ_A=true: A^H·T
        if(conj_flip)
            status = aoclsparse_sp2m_online_atb<T,
                                                aoclsparse_stage_finalize,
                                                true,
                                                false>( // CONJ_A=false
                m,
                n,
                n,
                baseA,
                icrowA,
                icolA,
                valA,
                aoclsparse_index_base_zero,
                icrowT.data(),
                icolT.data(),
                valT.data(),
                aoclsparse_index_base_zero,
                C_csr->ptr,
                C_csr->ind,
                (T *)(C_csr->val),
                &nnzC);
        else
            status = aoclsparse_sp2m_online_atb<T,
                                                aoclsparse_stage_finalize,
                                                true>( // CONJ_A=true (default)
                m,
                n,
                n,
                baseA,
                icrowA,
                icolA,
                valA,
                aoclsparse_index_base_zero,
                icrowT.data(),
                icolT.data(),
                valT.data(),
                aoclsparse_index_base_zero,
                C_csr->ptr,
                C_csr->ind,
                (T *)(C_csr->val),
                &nnzC);

        if(status != aoclsparse_status_success)
        {
            if(request == aoclsparse_stage_full_computation)
                aoclsparse_destroy(C); // C is incomplete, so destroy it
            return status;
        }
    }

    return aoclsparse_status_success;
}
#endif /* AOCLSPARSE_SYPR_HPP*/
