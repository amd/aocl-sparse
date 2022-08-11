/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include "aoclsparse_itsol_functions.hpp"
#include "aoclsparse.h"
#include "aoclsparse_itsol_data.hpp"
#include "aoclsparse_itsol_list_options.hpp"
#include "aoclsparse_itsol_options.hpp"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_solvers.h"
#include <cmath>
#include <iostream>

void aoclsparse_itsol_handle_prn_options(aoclsparse_itsol_handle handle)
{
    if(handle == nullptr)
        return;

    if(handle->type == aoclsparse_dmat)
    {
        if(handle->itsol_d)
            handle->itsol_d->opts.PrintOptions();
    }
    else
    {
        if(handle->itsol_s)
            handle->itsol_s->opts.PrintOptions();
    }
}

aoclsparse_status aoclsparse_itsol_option_set(aoclsparse_itsol_handle& handle,
                                              const char*              option,
                                              const char*              value)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(handle->type == aoclsparse_dmat)
    {
        if(handle->itsol_d == nullptr)
            return aoclsparse_status_internal_error;
        return handle_parse_option<double>(handle->itsol_d->opts, option, value);
    }
    else
    {
        if(handle->itsol_s == nullptr)
            return aoclsparse_status_internal_error;
        return handle_parse_option<float>(handle->itsol_s->opts, option, value);
    }
}

/* Deallocate the internal iterative solver memory */
void aoclsparse_itsol_destroy(aoclsparse_itsol_handle* handle)
{
    if(handle)
        if(*handle)
        {
            aoclsparse_itsol_data_free((*handle)->itsol_d);
            aoclsparse_itsol_data_free((*handle)->itsol_s);

            free(*handle);
            *handle = nullptr;
        }
}

/* Initialize the iterative solvers data structure' for double precision
 * Possible error: - Allocation, wrong pointer, internal error
 */
aoclsparse_status aoclsparse_itsol_d_init(aoclsparse_itsol_handle* handle)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;

    *handle = (aoclsparse_itsol_handle)malloc(sizeof(_aoclsparse_itsol_handle));
    if(!*handle)
        return aoclsparse_status_memory_error;

    (*handle)->type    = aoclsparse_dmat;
    (*handle)->itsol_s = nullptr;
    (*handle)->itsol_d = nullptr;

    aoclsparse_status status = aoclsparse_itsol_data_init(&(*handle)->itsol_d);
    if(status != aoclsparse_status_success)
    {
        aoclsparse_itsol_destroy(handle);
        return status;
    }

    return aoclsparse_status_success;
}

/* Initialize the iterative solvers data structure' for single precision
 * Possible error: - Allocation, wrong pointer, internal error
 */
aoclsparse_status aoclsparse_itsol_s_init(aoclsparse_itsol_handle* handle)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;

    *handle = (aoclsparse_itsol_handle)malloc(sizeof(_aoclsparse_itsol_handle));
    if(!*handle)
        return aoclsparse_status_memory_error;

    (*handle)->type    = aoclsparse_smat;
    (*handle)->itsol_s = nullptr;
    (*handle)->itsol_d = nullptr;

    aoclsparse_status status = aoclsparse_itsol_data_init(&(*handle)->itsol_s);
    if(status != aoclsparse_status_success)
    {
        aoclsparse_itsol_destroy(handle);
        return status;
    }

    return aoclsparse_status_success;
}

/* Initialize the iterative solver input data (double):
 * - n: dimension of the problem
 * - b: right hand side of the system 
 * Some checks on user data are done; 
 * NO check for NaNs 
 * Possible errors: 
 * - wrong type, handle was initialized for single precision
 * - invalid value: constraint n >= 0
 * - invalid pointer: b needs to be allocated by the user
 * - internal allocation: raised by itsol_rci_input (itsol->b error)
 */
aoclsparse_status
    aoclsparse_itsol_d_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const double* b)
{

    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_dmat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_rci_input(handle->itsol_d, n, b);
}

/* Equivalent to aoclsparse_itsol_d_rci_input for single precision */
aoclsparse_status
    aoclsparse_itsol_s_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const float* b)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_smat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_rci_input(handle->itsol_s, n, b);
}

/* 
 * Generic RCI entry point for all iterative solvers
 */
aoclsparse_status aoclsparse_itsol_d_rci_solve(aoclsparse_itsol_handle   handle,
                                               aoclsparse_itsol_rci_job* ircomm,
                                               double**                  u,
                                               double**                  v,
                                               double*                   x,
                                               double                    rinfo[100])
{
    /* Main entry point for the iterative solvers; checks input briefly and calls the instantiated solver
     * Possible exit codes:
     * - wrong type, handle was initialized for single precision
     * - 
     */
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_dmat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_rci_solve(handle->itsol_d, ircomm, u, v, x, rinfo);
}

aoclsparse_status aoclsparse_itsol_s_rci_solve(aoclsparse_itsol_handle   handle,
                                               aoclsparse_itsol_rci_job* ircomm,
                                               float**                   u,
                                               float**                   v,
                                               float*                    x,
                                               float                     rinfo[100])
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_smat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_rci_solve(handle->itsol_s, ircomm, u, v, x, rinfo);
}

/* 
 * Generic (direct/forward) interface for all iterative solvers
 */
aoclsparse_status aoclsparse_itsol_d_solve(
    aoclsparse_itsol_handle    handle,
    aoclsparse_int             n,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    const double*              b,
    double*                    x,
    double                     rinfo[100],
    aoclsparse_int precond(aoclsparse_int flag, const double* u, double* v, void* udata),
    aoclsparse_int monit(const double* x, const double* r, double rinfo[100], void* udata),
    void*          udata)
{
    if(!handle)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_dmat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_solve(
        handle->itsol_d, n, mat, descr, b, x, rinfo, precond, monit, udata);
}

aoclsparse_status aoclsparse_itsol_s_solve(
    aoclsparse_itsol_handle    handle,
    aoclsparse_int             n,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    const float*               b,
    float*                     x,
    float                      rinfo[100],
    aoclsparse_int             precond(aoclsparse_int flag, const float* u, float* v, void* udata),
    aoclsparse_int             monit(const float* x, const float* r, float rinfo[100], void* udata),
    void*                      udata)
{
    if(!handle)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_smat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_solve(
        handle->itsol_s, n, mat, descr, b, x, rinfo, precond, monit, udata);
}

template <>
inline aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                          const float*               alpha,
                                          aoclsparse_int             m,
                                          aoclsparse_int             n,
                                          aoclsparse_int             nnz,
                                          const float*               csr_val,
                                          const aoclsparse_int*      csr_col_ind,
                                          const aoclsparse_int*      csr_row_ptr,
                                          const aoclsparse_mat_descr descr,
                                          const float*               x,
                                          const float*               beta,
                                          float*                     y)
{
    return aoclsparse_scsrmv(
        trans, alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, beta, y);
}

template <>
inline aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                          const double*              alpha,
                                          aoclsparse_int             m,
                                          aoclsparse_int             n,
                                          aoclsparse_int             nnz,
                                          const double*              csr_val,
                                          const aoclsparse_int*      csr_col_ind,
                                          const aoclsparse_int*      csr_row_ptr,
                                          const aoclsparse_mat_descr descr,
                                          const double*              x,
                                          const double*              beta,
                                          double*                    y)
{
    return aoclsparse_dcsrmv(
        trans, alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, beta, y);
}
