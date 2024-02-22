/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_csr_util.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
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
 */
template <typename T, aoclsparse_request REQUEST, bool BUILD_ONLY_U = false>
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
                      aoclsparse_int              *nnzC)
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
                (*nnzC)++;
            }
        }
        else
        {
            if(nnz[j] != i)
            {
                // newly created nonzero
                nnz[j]       = i;
                icolC[*nnzC] = j;
                (*nnzC)++;
                val[j] = alpha * valW[idxW];
            }
            else
            {
                // compute values
                val[j] += alpha * valW[idxW];
            }
        }
    }
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
    aoclsparse_int e_offset;

    // column indices of the matrix in CSR order
    // Row 'i' we want to transpose is icol[irstart[i]...irend[i]-e_offset-1]
    const aoclsparse_int *icol = NULL;

    // 0/1-base for icol indices
    aoclsparse_index_base base;

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
 */
template <typename T, aoclsparse_request REQUEST>
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

    *nnzC     = 0;
    icrowC[0] = 0;
    // Build i-th row of C, thus pass i-th row of A and symmetrize it
    for(aoclsparse_int i = 0; i < k; i++)
    {
        // Process the lower/upper half of symmetric matrix
        // including diagonal element
        // which is multiplied like usual sp2m routine.
        for(idxa = irstart[i] - s_offset; idxa < irend[i] - e_offset; ++idxa)
        {
            colA = icolA[idxa] - baseA;
            add_sprow<T, REQUEST, false>(i,
                                         valA[idxa],
                                         icrowB[colA] - baseB,
                                         icrowB[colA + 1] - baseB,
                                         icolB,
                                         valB,
                                         baseB,
                                         nnz,
                                         val,
                                         icolC,
                                         nnzC);
        }

        // Multiply the other half of the symmetric or hermitian matrix,
        // which is created on the fly, using the linked list created
        // in the beginning
        row = oft.rfirst(i);
        while(row >= 0)
        {
            idxa    = oft.ridx(row);
            T val_A = aoclsparse::conj(valA[idxa]);

            add_sprow<T, REQUEST, false>(i,
                                         val_A,
                                         icrowB[row] - baseB,
                                         icrowB[row + 1] - baseB,
                                         icolB,
                                         valB,
                                         baseB,
                                         nnz,
                                         val,
                                         icolC,
                                         nnzC);

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
    return aoclsparse_status_success;
}

/* Computes C = A^T*B (or A^H*B for complex types) where A is sorted CSR m x k,
 * B is sorted CSR m x n and the result C of dimension k x n will have baseC.
 * If REQUEST=aoclsparse_stage_nnz_count, only icrowC and nnzC is built,
 * irowC[k+1] needs to be already allocated.
 * Otherwise all C arrays need to be allocated to big enough size (not checked).
 * BUILD_ONLY_U=true creates only upper triangle, otherwise full matrix.
 */
template <typename T, aoclsparse_request REQUEST, bool BUILD_ONLY_U = false>
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

    *nnzC     = 0;
    icrowC[0] = 0;
    // Build i-th row of C, thus pass i-th column of A
    for(aoclsparse_int i = 0; i < k; i++)
    {
        row = oft.rfirst(i);
        while(row >= 0)
        {
            idxa    = oft.ridx(row);
            T val_A = aoclsparse::conj(valA[idxa]);

            add_sprow<T, REQUEST, BUILD_ONLY_U>(i,
                                                val_A,
                                                icrowB[row] - baseB,
                                                icrowB[row + 1] - baseB,
                                                icolB,
                                                valB,
                                                baseB,
                                                nnz,
                                                val,
                                                icolC,
                                                nnzC);

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

    if(A->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }

    if(B->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }
    // Check index base
    if(A->base != aoclsparse_index_base_zero && A->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    if(descrB->base != aoclsparse_index_base_zero && descrB->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    if(B->base != descrB->base)
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

    // Basic check if 2nd stage was called without the first
    if(request == aoclsparse_stage_finalize
       && (*C == nullptr || (*C)->csr_mat.csr_row_ptr == nullptr
           || (*C)->csr_mat.csr_col_ptr == nullptr || (*C)->csr_mat.csr_val == nullptr
           || (*C)->m != n || (*C)->n != n))
        return aoclsparse_status_invalid_value;

    // Quick return for size 0 matrices, Do nothing
    // Return Valid Non-NULL pointers of C array.
    if((m == 0) || (n == 0) || (A->nnz == 0) || (B->nnz == 0))
    {
        if(*C == nullptr)
        {
            try
            {
                *C                        = new _aoclsparse_matrix;
                (*C)->csr_mat.csr_row_ptr = new aoclsparse_int[n + 1]();
                (*C)->csr_mat.csr_col_ptr = new aoclsparse_int[0];
                (*C)->csr_mat.csr_val     = ::operator new(0);
            }
            catch(std::bad_alloc &)
            {
                /*Insufficient memory for output allocation */
                aoclsparse_destroy(C);
                return aoclsparse_status_memory_error;
            }
            aoclsparse_init_mat(*C, aoclsparse_index_base_zero, n, n, 0, aoclsparse_csr_mat);
            (*C)->val_type = get_data_type<T>();
        }
        return aoclsparse_status_success;
    }

    // we need fully sorted rows if we apply on-fly transposition, thus B needs
    // to be sorted every time and A when we don't explicitly transpose
    if((A->sort != aoclsparse_fully_sorted && opA != aoclsparse_operation_none)
       || B->sort != aoclsparse_fully_sorted)
        return aoclsparse_status_unsorted_input;

    // If OP(A) == transpose/transpose_conjugate, C = At * B * A
    // Size of matrix B should be equal to #rows of matrix A(#cols of matrix At)
    // Size of resultant symmetric matrix C will be #columns of matrix A
    // Extract CSR arrays of matrix A as is
    // We will perform B * A first to generate intermediate product T matrix
    // And then perform At * T, without explicitly transposing A matrix.
    if((opA == aoclsparse_operation_transpose) || (opA == aoclsparse_operation_conjugate_transpose))
    {
        icrowA = A->csr_mat.csr_row_ptr;
        icolA  = A->csr_mat.csr_col_ptr;
        valA   = (T *)A->csr_mat.csr_val;
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
            icrowAt.resize(A->n + 1);
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

        status = aoclsparse_csr2csc_template(A->m,
                                             A->n,
                                             A->nnz,
                                             A->base,
                                             A->base,
                                             A->csr_mat.csr_row_ptr,
                                             A->csr_mat.csr_col_ptr,
                                             (const T *)A->csr_mat.csr_val,
                                             icolA,
                                             icrowA,
                                             valA);
        if(status != aoclsparse_status_success)
            return aoclsparse_status_memory_error;

        // we need Hermition, so far we transposed so now conjugate
        for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
            valA[idx] = aoclsparse::conj(valA[idx]);
    }

    if(!B->opt_csr_ready)
    {
        status = aoclsparse_csr_optimize<T>(B);
        if(status != aoclsparse_status_success)
            return status;
    }
    aoclsparse_index_base baseA = A->base;
    aoclsparse_index_base baseB = B->internal_base_index;
    // Retrieve CSR arrays of matrix B from optimised CSR.
    icrowB = B->opt_csr_mat.csr_row_ptr;
    icolB  = B->opt_csr_mat.csr_col_ptr;
    valB   = (T *)B->opt_csr_mat.csr_val;

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

    // We will perform B * At  or B * A first to generate intermediate product T matrix
    status = aoclsparse_sp2m_online_symab<T, aoclsparse_stage_nnz_count>(m,
                                                                         m,
                                                                         n,
                                                                         baseB,
                                                                         icrowB,
                                                                         B->idiag,
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
    status = aoclsparse_sp2m_online_symab<T, aoclsparse_stage_finalize>(m,
                                                                        m,
                                                                        n,
                                                                        baseB,
                                                                        icrowB,
                                                                        B->idiag,
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
            *C                        = new _aoclsparse_matrix;
            (*C)->csr_mat.csr_row_ptr = new aoclsparse_int[n + 1];
            icrowC                    = (*C)->csr_mat.csr_row_ptr;
        }
        catch(std::bad_alloc &)
        {
            aoclsparse_destroy(C);
            return aoclsparse_status_memory_error;
        }
        status = aoclsparse_sp2m_online_atb<T, aoclsparse_stage_nnz_count, true>(
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
            (*C)->csr_mat.csr_col_ptr = new aoclsparse_int[nnzC];
            (*C)->csr_mat.csr_val     = ::operator new(nnzC * sizeof(T));
        }
        catch(std::bad_alloc &)
        {
            aoclsparse_destroy(C); // C is incomplete, so destroy it
            return aoclsparse_status_memory_error;
        }
        aoclsparse_init_mat(*C, aoclsparse_index_base_zero, n, n, nnzC, aoclsparse_csr_mat);
        (*C)->val_type = get_data_type<T>();
    }

    if(request == aoclsparse_stage_full_computation || request == aoclsparse_stage_finalize)
    {
        status = aoclsparse_sp2m_online_atb<T, aoclsparse_stage_finalize, true>(
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
            (*C)->csr_mat.csr_row_ptr,
            (*C)->csr_mat.csr_col_ptr,
            (T *)((*C)->csr_mat.csr_val),
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
