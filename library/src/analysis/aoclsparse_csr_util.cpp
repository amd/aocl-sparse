/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_csr_util.hpp"

#include <sstream>

/* Destroy the optimize_data linked list*/
void aoclsparse_optimize_destroy(aoclsparse_optimize_data *&opt)
{
    aoclsparse_optimize_data *optd_ptr = opt;
    aoclsparse_optimize_data *next     = nullptr;

    while(optd_ptr)
    {
        next = optd_ptr->next;
        delete optd_ptr;
        optd_ptr = next;
    }
    opt = nullptr;
}

/* Add a new hinted action at the start of an optimize_data linked list
 * Possible exit: memory allocation error
 */
aoclsparse_status aoclsparse_add_hint(aoclsparse_optimize_data *&list,
                                      aoclsparse_hinted_action   act,
                                      aoclsparse_mat_descr       desc,
                                      aoclsparse_operation       trans,
                                      aoclsparse_int             nop)
{
    aoclsparse_optimize_data *optd;
    try
    {
        optd = new aoclsparse_optimize_data;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    optd->act = act;

    // copy the descriptor info
    optd->type      = desc->type;
    optd->fill_mode = desc->fill_mode;

    optd->trans = trans;
    optd->nop   = nop;

    if(list)
        optd->next = list;
    else
        optd->next = nullptr;
    list = optd;

    return aoclsparse_status_success;
}

/* Check CSR matrix format (0-based) for correctness
 * shape is used to determine if the matrix is supposed to have a particular structure to enable individual checks
 * error_handler is an optional callback for debug purposes:
 * if defined, it gets the status and a string describing the probelm on its interface
 */
aoclsparse_status aoclsparse_csr_check_internal(aoclsparse_int        m,
                                                aoclsparse_int        n,
                                                aoclsparse_int        nnz,
                                                const aoclsparse_csr  csr_mat,
                                                aoclsparse_shape      shape,
                                                aoclsparse_index_base base,
                                                void (*error_handler)(aoclsparse_status status,
                                                                      std::string       message))
{

    std::ostringstream buffer;
    aoclsparse_status  status;

    if(n < 0 || m < 0 || nnz < 0)
    {
        status = aoclsparse_status_invalid_value;
        if(error_handler)
        {
            buffer << "Wrong n/m/nnz";
            error_handler(status, buffer.str());
        }
        return status;
    }

    if((csr_mat->csr_row_ptr[0] - base) != 0)
    {
        status = aoclsparse_status_invalid_value;
        if(error_handler)
        {
            buffer << "Wrong row_ptr[0]";
            error_handler(status, buffer.str());
        }
        return status;
    }
    if((csr_mat->csr_row_ptr[m] - base) != nnz)
    {
        status = aoclsparse_status_invalid_value;
        if(error_handler)
        {
            buffer << "Wrong row_ptr[m]!=nnz";
            error_handler(status, buffer.str());
        }
        return status;
    }

    for(aoclsparse_int i = 1; i <= m; i++)
    {
        if(csr_mat->csr_row_ptr[i - 1] > csr_mat->csr_row_ptr[i])
        {
            status = aoclsparse_status_invalid_value;
            if(error_handler)
            {
                buffer << "Wrong row_ptr - not nondecreasing";
                error_handler(status, buffer.str());
            }
            return status;
        }
    }

    aoclsparse_int idxstart, idxend, j, jmin = 0, jmax = n - 1;
    for(aoclsparse_int i = 0; i < m; i++)
    {
        idxend   = csr_mat->csr_row_ptr[i + 1] - base;
        idxstart = csr_mat->csr_row_ptr[i] - base;
        if(shape == shape_lower_triangle)
        {
            jmin = 0;
            jmax = i;
        }
        else if(shape == shape_upper_triangle)
        {
            jmin = i;
            jmax = n - 1;
        }
        int diag = -1;
        for(aoclsparse_int idx = idxstart; idx < idxend; idx++)
        {
            j = csr_mat->csr_col_ptr[idx] - base;
            if(j < jmin || j > jmax)
            {
                status = aoclsparse_status_invalid_value;
                if(error_handler)
                {
                    buffer << "Wrong col_ptr - out of bounds or triangle, @idx=" << idx
                           << ": j=" << j << ", i=" << i;
                    error_handler(status, buffer.str());
                }
                return status;
            }
            if(shape == shape_lower_triangle)
            {
                if(j == i)
                {
                    // diagonal element - check if it is unique and at the end of the row
                    if(diag >= 0)
                    {
                        status = aoclsparse_status_invalid_value;
                        if(error_handler)
                        {
                            buffer << "Wrong diag - duplicate diag for i=j=" << i;
                            error_handler(status, buffer.str());
                        }
                        return status;
                    }
                    diag = idx;
                    if(idx != idxend - 1)
                    {
                        status = aoclsparse_status_invalid_value;
                        if(error_handler)
                        {
                            buffer << "Wrong diag - not at the end of row i=" << i
                                   << " @idx=" << idx << " idxend-1=" << idxend - 1;
                            error_handler(status, buffer.str());
                        }
                        return status;
                    }
                }
            }
            if(shape == shape_upper_triangle)
            {
                if(j == i)
                {
                    // diagonal element - check if it is unique and at the start of the row
                    if(diag >= 0)
                    {
                        status = aoclsparse_status_invalid_value;
                        if(error_handler)
                        {
                            buffer << "Wrong diag - duplicate diag for i=j=" << i;
                            error_handler(status, buffer.str());
                        }
                        return status;
                    }
                    diag = idx;
                    if(idx != idxstart)
                    {
                        status = aoclsparse_status_invalid_value;
                        if(error_handler)
                        {
                            buffer << "Wrong diag - not at the start of row i=" << i
                                   << " @idx=" << idx << " idxstart=" << idxstart;
                            error_handler(status, buffer.str());
                        }
                        return status;
                    }
                }
            }
        }
    }

    return aoclsparse_status_success;
}

/* Given a valid CSR matrix, check if indices are grouped by lower triange,
 * diagonal, upper triangle within rows, order within groups is not checked,
 * and if group-ordered, check also if all diagonal
 * elements exist (fulldiag). If (!sorted), diagonal elements are not checked
 * and fulldiag=false is returned.
 *
 * Possible fails: invalid_size(m,n<0), invalid_pointer (input),
 *                 invalid_value (duplicate diagonal element)
 */
aoclsparse_status aoclsparse_csr_check_sort_diag(aoclsparse_int        m,
                                                 aoclsparse_int        n,
                                                 aoclsparse_index_base base,
                                                 const aoclsparse_csr  csr_mat,
                                                 bool                 &sorted,
                                                 bool                 &fulldiag)
{

    sorted   = false;
    fulldiag = false;

    if(m < 0 || n < 0)
        return aoclsparse_status_invalid_size;
    if(csr_mat->csr_row_ptr == nullptr || csr_mat->csr_col_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;

    // assume sorting & fulldiag unless proved otherwise
    sorted   = true;
    fulldiag = true;
    aoclsparse_int i, idx, idxend;
    bool           found, lower;

    for(i = 0; i < m; i++)
    {
        lower  = true; // assume the data starts with the lower triangular part
        found  = false;
        idxend = csr_mat->csr_row_ptr[i + 1] - base;
        for(idx = (csr_mat->csr_row_ptr[i] - base); idx < idxend; idx++)
        {
            aoclsparse_int j = csr_mat->csr_col_ptr[idx] - base;
            if(j == i)
            {
                if(found)
                {
                    // duplicate diagonal
                    return aoclsparse_status_invalid_value;
                }
                found  = true;
                sorted = lower;
                lower  = false;
            }
            else
            {
                if(lower)
                {
                    lower = j < i;
                }
                else
                {
                    sorted &= j > i;
                }
            }
            if(!sorted)
            {
                // early termination
                fulldiag = false;
                return aoclsparse_status_success;
            }
        }

        if(!found && i < n)
        {
            fulldiag = false;
        }
        if(!sorted)
        {
            // early termination
            fulldiag = false;
            return aoclsparse_status_success;
        }
    }

    return aoclsparse_status_success;
}

/* Given a square CSR matrix which is valid and sorted, generate index arrays
 * to position of diagonal and the first strictly upper triangle element,
 * if any of these is missing, the index points to where such an element
 * would be stored. The new index arrays are allocated here via new
 * so delete[] them when not needed. The output arrays idiag and iurow
 * are respecting the input base.
 *
 * This is used to access only L/D/U portion of the matrix
 * strictly L in row i: icrow[i] .. idiag[i]-1
 * diagonal in row i:   idiag[i] .. iurow[i]-1
 *   [if empty set, diag is not present]
 * strictly U in row i: iurow[i] .. icrow[i+1]-1
 *
 * On "fixed" matrix where diagonal is present, is should always be:
 *   idiag[i]+1 = iurow[i]
 *
 * For strict triangle:  start..end-1
 *   L: start=icrow, end=idiag
 *   U: start=iurow, end=&icrow[1]
 *
 * Possible fails: memory, size(m<0), invalid_pointer (input)
 * The matrix is not checked, sorting in row is assumed,
 * at most one diag element per row.
 */
aoclsparse_status aoclsparse_csr_indices(aoclsparse_int        m,
                                         aoclsparse_index_base base,
                                         const aoclsparse_int *icrow,
                                         const aoclsparse_int *icol,
                                         aoclsparse_int      **idiag,
                                         aoclsparse_int      **iurow)
{

    if(m < 0)
        return aoclsparse_status_invalid_size;
    if(icrow == nullptr || icol == nullptr || idiag == nullptr || iurow == nullptr)
        return aoclsparse_status_invalid_pointer;

    try
    {
        *idiag = new aoclsparse_int[m];
        *iurow = new aoclsparse_int[m];
    }
    catch(std::bad_alloc &)
    {
        delete[] *idiag;
        *idiag = nullptr;
        delete[] *iurow;
        *iurow = nullptr;
        return aoclsparse_status_memory_error;
    }

    aoclsparse_int i, idx, idxend;
    bool           found;

    for(i = 0; i < m; i++)
    {
        found  = false;
        idxend = icrow[i + 1] - base;
        for(idx = (icrow[i] - base); idx < idxend; idx++)
        {
            aoclsparse_int j = icol[idx] - base; // 0-based column index
            if(j >= i)
            {
                /*
                    the adjusted index to csr_val array, taking into account base-index( 0 or 1)
                    is performed here to update idiag and iurow arrays
                    If in case, user provides a sorted csr matrix with full diagonal,
                    then we do not copy csr_mat, but use the user's csr_mat pointers through opt_csr_mat.
                    In that case, row_ptr and col_ind arrays will be in 1-based mode and the execution kernels
                    will perform the correction. So, even idiag and iurow arrays also need to be in the
                    same base indexing mode, so that the kernels can perform an uniform base correction
                    for all arrays
                */
                aoclsparse_int adj_idx = idx + base;
                // first diag or U element
                // so 'idx' is where diag should be
                (*idiag)[i] = adj_idx;
                // if the current is diagonal, U should start just after
                (*iurow)[i] = j == i ? adj_idx + 1 : adj_idx;
                found       = true;
                break;
            }
        }
        if(!found)
        {
            // all elements were strictly L, diag and U should be here
            //TODO: add unit test to catch the issues, if the below idiag and iurow
            //      are not adjusted for base( ... + base)
            (*idiag)[i] = idxend + base;
            (*iurow)[i] = idxend + base;
        }
    }
    return aoclsparse_status_success;
}
