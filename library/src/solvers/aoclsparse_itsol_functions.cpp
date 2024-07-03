/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_solvers.h"
#include "aoclsparse_itsol_data.hpp"
#include "aoclsparse_itsol_functions.hpp"
#include "aoclsparse_itsol_list_options.hpp"
#include "aoclsparse_itsol_options.hpp"

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
    else if(handle->type == aoclsparse_smat)
    {
        if(handle->itsol_s)
            handle->itsol_s->opts.PrintOptions();
    }
    else if(handle->type == aoclsparse_cmat)
    {
        if(handle->itsol_c)
            handle->itsol_c->opts.PrintOptions();
    }
    else if(handle->type == aoclsparse_zmat)
    {
        if(handle->itsol_z)
            handle->itsol_z->opts.PrintOptions();
    }
}

aoclsparse_status aoclsparse_itsol_option_set(aoclsparse_itsol_handle handle,
                                              const char             *option,
                                              const char             *value)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;

    switch(handle->type)
    {
    case aoclsparse_dmat:
        if(handle->itsol_d == nullptr)
            return aoclsparse_status_internal_error;
        return handle_parse_option<double>(handle->itsol_d->opts, option, value);
        break;
    case aoclsparse_smat:
        if(handle->itsol_s == nullptr)
            return aoclsparse_status_internal_error;
        return handle_parse_option<float>(handle->itsol_s->opts, option, value);
        break;
    case aoclsparse_zmat:
        if(handle->itsol_z == nullptr)
            return aoclsparse_status_internal_error;
        return handle_parse_option<aoclsparse_double_complex>(handle->itsol_z->opts, option, value);
        break;
    case aoclsparse_cmat:
        if(handle->itsol_c == nullptr)
            return aoclsparse_status_internal_error;
        return handle_parse_option<aoclsparse_float_complex>(handle->itsol_c->opts, option, value);
        break;
    default:
        return aoclsparse_status_invalid_value;
        break;
    }
}

/* Deallocate the internal iterative solver memory */
void aoclsparse_itsol_destroy(aoclsparse_itsol_handle *handle)
{
    if(handle)
        if(*handle)
        {
            aoclsparse_itsol_data_free((*handle)->itsol_d);
            aoclsparse_itsol_data_free((*handle)->itsol_s);
            aoclsparse_itsol_data_free((*handle)->itsol_z);
            aoclsparse_itsol_data_free((*handle)->itsol_c);

            delete *handle;
            *handle = nullptr;
        }
}

/* Initialize the iterative solvers data structure' for double precision
 * Possible error: - Allocation, wrong pointer, internal error
 */
aoclsparse_status aoclsparse_itsol_d_init(aoclsparse_itsol_handle *handle)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;

    try
    {
        *handle = new _aoclsparse_itsol_handle;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    (*handle)->type          = aoclsparse_dmat;
    (*handle)->itsol_s       = nullptr;
    (*handle)->itsol_d       = nullptr;
    (*handle)->itsol_c       = nullptr;
    (*handle)->itsol_z       = nullptr;
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
aoclsparse_status aoclsparse_itsol_s_init(aoclsparse_itsol_handle *handle)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;

    try
    {
        *handle = new _aoclsparse_itsol_handle;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    (*handle)->type    = aoclsparse_smat;
    (*handle)->itsol_s = nullptr;
    (*handle)->itsol_d = nullptr;
    (*handle)->itsol_c = nullptr;
    (*handle)->itsol_z = nullptr;

    aoclsparse_status status = aoclsparse_itsol_data_init(&(*handle)->itsol_s);
    if(status != aoclsparse_status_success)
    {
        aoclsparse_itsol_destroy(handle);
        return status;
    }

    return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_itsol_z_init(aoclsparse_itsol_handle *handle)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;

    try
    {
        *handle = new _aoclsparse_itsol_handle;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    (*handle)->type    = aoclsparse_zmat;
    (*handle)->itsol_s = nullptr;
    (*handle)->itsol_d = nullptr;

    (*handle)->itsol_c = nullptr;
    (*handle)->itsol_z = nullptr;

    aoclsparse_status status = aoclsparse_itsol_data_init(&(*handle)->itsol_z);
    if(status != aoclsparse_status_success)
    {
        aoclsparse_itsol_destroy(handle);
        return status;
    }

    return aoclsparse_status_success;
}
aoclsparse_status aoclsparse_itsol_c_init(aoclsparse_itsol_handle *handle)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;

    try
    {
        *handle = new _aoclsparse_itsol_handle;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    (*handle)->type    = aoclsparse_cmat;
    (*handle)->itsol_s = nullptr;
    (*handle)->itsol_d = nullptr;

    (*handle)->itsol_c = nullptr;
    (*handle)->itsol_z = nullptr;

    aoclsparse_status status = aoclsparse_itsol_data_init(&(*handle)->itsol_c);
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
    aoclsparse_itsol_d_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const double *b)
{

    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_dmat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_rci_input(handle->itsol_d, n, b);
}

/* Equivalent to aoclsparse_itsol_d_rci_input for single precision */
aoclsparse_status
    aoclsparse_itsol_s_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const float *b)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_smat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_rci_input(handle->itsol_s, n, b);
}
aoclsparse_status aoclsparse_itsol_z_rci_input(aoclsparse_itsol_handle          handle,
                                               aoclsparse_int                   n,
                                               const aoclsparse_double_complex *b)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_zmat)
        return aoclsparse_status_wrong_type;

    const std::complex<double> *pb = reinterpret_cast<const std::complex<double> *>(b);
    auto                       *z_itsol
        = reinterpret_cast<aoclsparse_itsol_data<std::complex<double>> *>(handle->itsol_z);

    return aoclsparse_itsol_rci_input(z_itsol, n, pb);
}
aoclsparse_status aoclsparse_itsol_c_rci_input(aoclsparse_itsol_handle         handle,
                                               aoclsparse_int                  n,
                                               const aoclsparse_float_complex *b)
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_cmat)
        return aoclsparse_status_wrong_type;

    const std::complex<float> *pb = reinterpret_cast<const std::complex<float> *>(b);
    auto *c_itsol = reinterpret_cast<aoclsparse_itsol_data<std::complex<float>> *>(handle->itsol_c);
    return aoclsparse_itsol_rci_input(c_itsol, n, pb);
}
/*
 * Generic RCI entry point for all iterative solvers
 */
aoclsparse_status aoclsparse_itsol_d_rci_solve(aoclsparse_itsol_handle   handle,
                                               aoclsparse_itsol_rci_job *ircomm,
                                               double                  **u,
                                               double                  **v,
                                               double                   *x,
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
                                               aoclsparse_itsol_rci_job *ircomm,
                                               float                   **u,
                                               float                   **v,
                                               float                    *x,
                                               float                     rinfo[100])
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_smat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_rci_solve(handle->itsol_s, ircomm, u, v, x, rinfo);
}
aoclsparse_status aoclsparse_itsol_z_rci_solve(aoclsparse_itsol_handle     handle,
                                               aoclsparse_itsol_rci_job   *ircomm,
                                               aoclsparse_double_complex **u,
                                               aoclsparse_double_complex **v,
                                               aoclsparse_double_complex  *x,
                                               double                      rinfo[100])
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_zmat)
        return aoclsparse_status_wrong_type;

    std::complex<double> **pu = reinterpret_cast<std::complex<double> **>(u);
    std::complex<double> **pv = reinterpret_cast<std::complex<double> **>(v);
    std::complex<double>  *px = reinterpret_cast<std::complex<double> *>(x);
    auto                  *z_itsol
        = reinterpret_cast<aoclsparse_itsol_data<std::complex<double>> *>(handle->itsol_z);

    return aoclsparse_itsol_rci_solve(z_itsol, ircomm, pu, pv, px, rinfo);
}
aoclsparse_status aoclsparse_itsol_c_rci_solve(aoclsparse_itsol_handle    handle,
                                               aoclsparse_itsol_rci_job  *ircomm,
                                               aoclsparse_float_complex **u,
                                               aoclsparse_float_complex **v,
                                               aoclsparse_float_complex  *x,
                                               float                      rinfo[100])
{
    if(handle == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_cmat)
        return aoclsparse_status_wrong_type;

    std::complex<float> **pu = reinterpret_cast<std::complex<float> **>(u);
    std::complex<float> **pv = reinterpret_cast<std::complex<float> **>(v);
    std::complex<float>  *px = reinterpret_cast<std::complex<float> *>(x);
    auto *c_itsol = reinterpret_cast<aoclsparse_itsol_data<std::complex<float>> *>(handle->itsol_c);

    return aoclsparse_itsol_rci_solve(c_itsol, ircomm, pu, pv, px, rinfo);
}
/*
 * Generic (direct/forward) interface for all iterative solvers
 */
aoclsparse_status aoclsparse_itsol_d_solve(
    aoclsparse_itsol_handle    handle,
    aoclsparse_int             n,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    const double              *b,
    double                    *x,
    double                     rinfo[100],
    aoclsparse_int             precond(
        aoclsparse_int flag, aoclsparse_int n, const double *u, double *v, void *udata),
    aoclsparse_int monit(
        aoclsparse_int n, const double *x, const double *r, double rinfo[100], void *udata),
    void *udata)
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
    const float               *b,
    float                     *x,
    float                      rinfo[100],
    aoclsparse_int             precond(
        aoclsparse_int flag, aoclsparse_int n, const float *u, float *v, void *udata),
    aoclsparse_int monit(
        aoclsparse_int n, const float *x, const float *r, float rinfo[100], void *udata),
    void *udata)
{
    if(!handle)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_smat)
        return aoclsparse_status_wrong_type;
    return aoclsparse_itsol_solve(
        handle->itsol_s, n, mat, descr, b, x, rinfo, precond, monit, udata);
}
aoclsparse_status
    aoclsparse_itsol_z_solve(aoclsparse_itsol_handle          handle,
                             aoclsparse_int                   n,
                             aoclsparse_matrix                mat,
                             const aoclsparse_mat_descr       descr,
                             const aoclsparse_double_complex *b,
                             aoclsparse_double_complex       *x,
                             double                           rinfo[100],
                             aoclsparse_int                   precond(aoclsparse_int                   flag,
                                                    aoclsparse_int                   n,
                                                    const aoclsparse_double_complex *u,
                                                    aoclsparse_double_complex       *v,
                                                    void                            *udata),
                             aoclsparse_int                   monit(aoclsparse_int                   n,
                                                  const aoclsparse_double_complex *x,
                                                  const aoclsparse_double_complex *r,
                                                  double                           rinfo[100],
                                                  void                            *udata),
                             void                            *udata)
{
    if(!handle)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_zmat)
        return aoclsparse_status_wrong_type;

    const std::complex<double> *pb = reinterpret_cast<const std::complex<double> *>(b);
    std::complex<double>       *px = reinterpret_cast<std::complex<double> *>(x);
    auto                       *z_itsol
        = reinterpret_cast<aoclsparse_itsol_data<std::complex<double>> *>(handle->itsol_z);
    aoclsparse_int (*zprecond_wrapper)(aoclsparse_int              flag,
                                       aoclsparse_int              n,
                                       const std::complex<double> *u,
                                       std::complex<double>       *v,
                                       void                       *udata)
        = reinterpret_cast<aoclsparse_int (*)(aoclsparse_int              flag,
                                              aoclsparse_int              n,
                                              const std::complex<double> *u,
                                              std::complex<double>       *v,
                                              void                       *udata)>(precond);
    aoclsparse_int (*zmonit_wrapper)(aoclsparse_int              n,
                                     const std::complex<double> *x,
                                     const std::complex<double> *r,
                                     double                      rinfo[100],
                                     void                       *udata)
        = reinterpret_cast<aoclsparse_int (*)(aoclsparse_int              n,
                                              const std::complex<double> *x,
                                              const std::complex<double> *r,
                                              double                      rinfo[100],
                                              void                       *udata)>(monit);

    return aoclsparse_itsol_solve(
        z_itsol, n, mat, descr, pb, px, rinfo, zprecond_wrapper, zmonit_wrapper, udata);
}
aoclsparse_status aoclsparse_itsol_c_solve(aoclsparse_itsol_handle         handle,
                                           aoclsparse_int                  n,
                                           aoclsparse_matrix               mat,
                                           const aoclsparse_mat_descr      descr,
                                           const aoclsparse_float_complex *b,
                                           aoclsparse_float_complex       *x,
                                           float                           rinfo[100],
                                           aoclsparse_int precond(aoclsparse_int flag,
                                                                  aoclsparse_int n,
                                                                  const aoclsparse_float_complex *u,
                                                                  aoclsparse_float_complex       *v,
                                                                  void *udata),
                                           aoclsparse_int monit(aoclsparse_int                  n,
                                                                const aoclsparse_float_complex *x,
                                                                const aoclsparse_float_complex *r,
                                                                float rinfo[100],
                                                                void *udata),
                                           void          *udata)
{
    if(!handle)
        return aoclsparse_status_invalid_pointer;
    if(handle->type != aoclsparse_cmat)
        return aoclsparse_status_wrong_type;

    const std::complex<float> *pb = reinterpret_cast<const std::complex<float> *>(b);
    std::complex<float>       *px = reinterpret_cast<std::complex<float> *>(x);
    auto *c_itsol = reinterpret_cast<aoclsparse_itsol_data<std::complex<float>> *>(handle->itsol_c);
    aoclsparse_int (*cprecond_wrapper)(aoclsparse_int             flag,
                                       aoclsparse_int             n,
                                       const std::complex<float> *u,
                                       std::complex<float>       *v,
                                       void                      *udata)
        = reinterpret_cast<aoclsparse_int (*)(aoclsparse_int             flag,
                                              aoclsparse_int             n,
                                              const std::complex<float> *u,
                                              std::complex<float>       *v,
                                              void                      *udata)>(precond);
    aoclsparse_int (*cmonit_wrapper)(aoclsparse_int             n,
                                     const std::complex<float> *x,
                                     const std::complex<float> *r,
                                     float                      rinfo[100],
                                     void                      *udata)
        = reinterpret_cast<aoclsparse_int (*)(aoclsparse_int             n,
                                              const std::complex<float> *x,
                                              const std::complex<float> *r,
                                              float                      rinfo[100],
                                              void                      *udata)>(monit);

    return aoclsparse_itsol_solve(
        c_itsol, n, mat, descr, pb, px, rinfo, cprecond_wrapper, cmonit_wrapper, udata);
}
template <>
inline aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                          const float               *alpha,
                                          aoclsparse_int             m,
                                          aoclsparse_int             n,
                                          aoclsparse_int             nnz,
                                          const float               *csr_val,
                                          const aoclsparse_int      *csr_col_ind,
                                          const aoclsparse_int      *csr_row_ptr,
                                          const aoclsparse_mat_descr descr,
                                          const float               *x,
                                          const float               *beta,
                                          float                     *y)
{
    return aoclsparse_scsrmv(
        trans, alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, beta, y);
}

template <>
inline aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                          const double              *alpha,
                                          aoclsparse_int             m,
                                          aoclsparse_int             n,
                                          aoclsparse_int             nnz,
                                          const double              *csr_val,
                                          const aoclsparse_int      *csr_col_ind,
                                          const aoclsparse_int      *csr_row_ptr,
                                          const aoclsparse_mat_descr descr,
                                          const double              *x,
                                          const double              *beta,
                                          double                    *y)
{
    return aoclsparse_dcsrmv(
        trans, alpha, m, n, nnz, csr_val, csr_col_ind, csr_row_ptr, descr, x, beta, y);
}
