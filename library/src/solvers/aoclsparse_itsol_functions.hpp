/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
#ifndef AOCLSPARSE_ITSOL_FUNCTIONS_HPP_
#define AOCLSPARSE_ITSOL_FUNCTIONS_HPP_

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_solvers.h"
#include "aoclsparse.hpp"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_ilu.hpp"
#include "aoclsparse_itsol_data.hpp"
#include "aoclsparse_itsol_list_options.hpp"
#include "aoclsparse_itsol_options.hpp"
#include "aoclsparse_lapack.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_utils.hpp"

// Define all entries for rinfo[] array
// All solvers using rinfo need to stick to these entries
#define RINFO_RES_NORM 0 // Residual 2-norm
#define RINFO_RHS_NORM 1 // 2-norm of the right hand side
#define RINFO_ITER 30 // Number of iterations

template <typename T>
aoclsparse_status aoclsparse_cg_data_init(const aoclsparse_int n, cg_data<T> **cg)
{
    /* Initializes Internal CG data including 4 vectors of working memory
     * Possible exits:
     * - Allocation
     */
    if(cg == nullptr)
        return aoclsparse_status_internal_error;

    try
    {
        *cg = new cg_data<T>;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    (*cg)->p = (*cg)->q = (*cg)->r = (*cg)->z = nullptr;
    try
    {
        (*cg)->p = new T[n];
        (*cg)->q = new T[n];
        (*cg)->r = new T[n];
        (*cg)->z = new T[n];
    }
    catch(std::bad_alloc &)
    {
        aoclsparse_cg_data_free(*cg);
        *cg = nullptr;
        return aoclsparse_status_memory_error;
    }
    (*cg)->task  = task_start;
    (*cg)->niter = 0;

    return aoclsparse_status_success;
}
/* Initializes Internal GMRES data including vectors of working memory
*   Possible exits:
* - Allocation errors
*/
template <typename T>
aoclsparse_status aoclsparse_gmres_data_init(const aoclsparse_int                   n,
                                             gmres_data<T>                        **gmres,
                                             aoclsparse_options::OptionRegistry<T> &opts)
{
    aoclsparse_int m;
    if(gmres == nullptr)
        return aoclsparse_status_internal_error;

    try
    {
        *gmres = new gmres_data<T>;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    //extract and populate restart iterations from user input for GMRES allocations of working buffers
    opts.GetOption("gmres restart iterations", m);
    (*gmres)->restart_iters = m;

    (*gmres)->v = (*gmres)->z = (*gmres)->h = (*gmres)->g = (*gmres)->s = nullptr;
    (*gmres)->c                                                         = nullptr;
    try
    {
        // Allocate and initialise arrays to 0
        (*gmres)->v = new T[(m + 1) * n]();
        (*gmres)->z = new T[(m + 1) * n]();
        (*gmres)->h = new T[m * m]();
        (*gmres)->g = new T[m + 1]();
        (*gmres)->c = new tolerance_t<T>[m]();
        (*gmres)->s = new T[m]();
    }
    catch(std::bad_alloc &)
    {
        aoclsparse_gmres_data_free(*gmres);
        *gmres = nullptr;
        return aoclsparse_status_memory_error;
    }

    (*gmres)->task  = task_gmres_start;
    (*gmres)->niter = 0;
    (*gmres)->j     = 0;
    return aoclsparse_status_success;
}
template <typename T>
void aoclsparse_cg_data_free(cg_data<T> *cg)
{
    if(!cg)
        // Nothing to do
        return;
    if(cg->p)
        delete[] cg->p;
    if(cg->q)
        delete[] cg->q;
    if(cg->r)
        delete[] cg->r;
    if(cg->z)
        delete[] cg->z;
    delete cg;
}
/*
    Deallocate GMRES's memory of working buffers
*/
template <typename T>
void aoclsparse_gmres_data_free(gmres_data<T> *gmres)
{
    if(!gmres)
    {
        // Nothing to do
        return;
    }
    if(gmres->v)
    {
        delete[] gmres->v;
        gmres->v = nullptr;
    }
    if(gmres->z)
    {
        delete[] gmres->z;
        gmres->z = nullptr;
    }
    if(gmres->h)
    {
        delete[] gmres->h;
        gmres->h = nullptr;
    }
    if(gmres->g)
    {
        delete[] gmres->g;
        gmres->g = nullptr;
    }
    if(gmres->c)
    {
        delete[] gmres->c;
        gmres->c = nullptr;
    }
    if(gmres->s)
    {
        delete[] gmres->s;
        gmres->s = nullptr;
    }
    delete gmres;
}
template <typename T>
aoclsparse_status aoclsparse_cg_data_options(cg_data<T>                            *cg,
                                             aoclsparse_options::OptionRegistry<T> &opts)
{
    aoclsparse_int flag;
    flag = opts.GetKey("cg preconditioner", cg->precond);
    flag += opts.GetOption("cg rel tolerance", cg->rtol);
    flag += opts.GetOption("cg abs tolerance", cg->atol);
    flag += opts.GetOption("cg iteration limit", cg->maxit);

    if(flag)
        return aoclsparse_status_internal_error;

    return aoclsparse_status_success;
}
template <typename T>
aoclsparse_status aoclsparse_gmres_data_options(gmres_data<T>                         *gmres,
                                                aoclsparse_options::OptionRegistry<T> &opts)
{
    aoclsparse_int flag;
    flag = opts.GetKey("gmres preconditioner", gmres->precond);
    flag += opts.GetOption("gmres rel tolerance", gmres->rtol);
    flag += opts.GetOption("gmres abs tolerance", gmres->atol);
    flag += opts.GetOption("gmres iteration limit", gmres->maxit);

    if(flag)
        return aoclsparse_status_internal_error;

    return aoclsparse_status_success;
}
/* aoclsparse_itsol_data_free - deallocate/reset aoclsparse_itsol_data,
 * this is used as final deallocation or when new fight hand side comes
 * in (potentially of a different size) and we need to clear all solver data.
 * In that case use keep_itsol=true. */
template <typename T>
void aoclsparse_itsol_data_free(aoclsparse_itsol_data<T> *itsol, bool keep_itsol = false)
{
    if(itsol)
    {

        itsol->n = 0;
        if(itsol->b)
            delete[] itsol->b;
        itsol->b       = nullptr;
        itsol->solving = false;
        itsol->solver  = 0;

        // deallocate all solver specific data
        aoclsparse_cg_data_free(itsol->cg);
        itsol->cg = nullptr;
        //GMRES data deallocation
        aoclsparse_gmres_data_free(itsol->gmres);
        itsol->gmres = nullptr;
        if(!keep_itsol)
            delete itsol;
    }
}

/* Initialize itsol_data structure, including the option settings */
template <typename T>
aoclsparse_status aoclsparse_itsol_data_init(aoclsparse_itsol_data<T> **itsol)
{
    if(itsol == nullptr)
        return aoclsparse_status_internal_error;

    try
    {
        *itsol = new aoclsparse_itsol_data<T>;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    (*itsol)->n       = 0;
    (*itsol)->b       = nullptr;
    (*itsol)->solving = false;
    (*itsol)->solver  = 0;
    (*itsol)->cg      = nullptr;
    (*itsol)->gmres   = nullptr;
    if(register_options((*itsol)->opts))
    {
        aoclsparse_itsol_data_free(*itsol);
        *itsol = nullptr;
        return aoclsparse_status_internal_error;
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status
    aoclsparse_itsol_rci_input(aoclsparse_itsol_data<T> *itsol, aoclsparse_int n, const T *b)
{
    /* Templated initialization of iterative solvers input
     * Possible exit: Allocation error
     */
    if(itsol == nullptr)
        return aoclsparse_status_internal_error;
    if(n < 0)
        return aoclsparse_status_invalid_value;
    if(!b)
        return aoclsparse_status_invalid_pointer;

    // deallocate any of the previous solver data & RHS
    // as the size might be different from last call
    // but keep itsol structure itself
    aoclsparse_itsol_data_free(itsol, true);

    try
    {
        itsol->b = new T[n];
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    itsol->n = n;

    // copy b into the handle
    for(aoclsparse_int i = 0; i < n; i++)
        itsol->b[i] = b[i];

    itsol->solving = false;

    return aoclsparse_status_success;
}

/* Internal routine to initialize all chosen solvers/preconditioners before
 * the solving (i.e., looping via RCI) starts, particularly set itsol->solver
 * based on the options */
template <typename T>
aoclsparse_status aoclsparse_itsol_solver_init(aoclsparse_itsol_data<T> *itsol)
{
    aoclsparse_status status;
    if(itsol == nullptr)
        return aoclsparse_status_internal_error;

    // skip solver's initialization on next RCI calls till we are finish (rci_stop)
    //itsol->solving = true;

    // find the chosen solver from options
    itsol->opts.GetKey("iterative method", itsol->solver);
    switch(itsol->solver)
    {
    case solver_cg:
        if(!itsol->cg)
        {
            // initalize CG data
            status = aoclsparse_cg_data_init(itsol->n, &itsol->cg);
            if(status != aoclsparse_status_success)
                // Allocation
                return status;
        }
        itsol->cg->task = task_start;
        status          = aoclsparse_cg_data_options(itsol->cg, itsol->opts);
        if(status != aoclsparse_status_success)
            return status;
        break;
    case solver_gmres:
        if(!itsol->gmres)
        {
            // initalize GMRES data
            status = aoclsparse_gmres_data_init(itsol->n, &itsol->gmres, itsol->opts);
            if(status != aoclsparse_status_success)
                // Allocation
                return status;
        }
        itsol->gmres->task = task_gmres_start;
        status             = aoclsparse_gmres_data_options(itsol->gmres, itsol->opts);
        if(status != aoclsparse_status_success)
            return status;
        break;
    }

    return aoclsparse_status_success;
}

/* Compute one step of a symmetric Gauss-Seidel preconditionner
 * This implementation solves the M^1*z = r preconditioner step in 3 stages:
 *  If A = L + D + U, where L is the strictly lower triangle,
 *  D is the diagonal and U is the strctly upper triangle
 *      1. solve (L+D)y = r using triangle solve
 *      2. compute y := Dy
 *      3. solve (U+D)z = y
 */
template <typename T>
aoclsparse_status
    aoclsparse_itsol_symgs(aoclsparse_matrix A, aoclsparse_mat_descr descr, T *r, T *y, T *z)
{
    // For internal use in CG.
    // Add additional input checks if exposing it to the user
    _aoclsparse_mat_descr descr_cpy;
    T                     alpha = 1.0;
    aoclsparse_int        avxversion, i;
    aoclsparse_status     status;
    T                    *aval = static_cast<T *>(A->opt_csr_mat.val);
    aoclsparse_operation  trans;

    if(descr->type != aoclsparse_matrix_type_general
       && descr->type != aoclsparse_matrix_type_symmetric)
        return aoclsparse_status_invalid_value;
    // Singular matrix, SymGS is not defined
    if(descr->diag_type == aoclsparse_diag_type_zero)
    {
        return aoclsparse_status_invalid_value;
    }

    aoclsparse_copy_mat_descr(&descr_cpy, descr);
    aoclsparse_set_mat_type(&descr_cpy, aoclsparse_matrix_type_triangular);

    // Use default AVX extension
    avxversion = -1;

    // (L+D)y := r
    if(descr->type == aoclsparse_matrix_type_general
       || descr->fill_mode == aoclsparse_fill_mode_lower)
    {
        // Use the lower triangle directly
        aoclsparse_set_mat_fill_mode(&descr_cpy, aoclsparse_fill_mode_lower);
        trans = aoclsparse_operation_none;
    }
    else
    {
        // symmetric with upper triangle given. use transpose trsv
        aoclsparse_set_mat_fill_mode(&descr_cpy, aoclsparse_fill_mode_upper);
        trans = aoclsparse_operation_transpose;
    }
    status = aoclsparse::trsv(trans, alpha, A, &descr_cpy, r, 1, y, 1, avxversion);
    if(status != aoclsparse_status_success)
        return status;

    // y := Dy
    if(descr->diag_type == aoclsparse_diag_type_non_unit)
    {
        for(i = 0; i < A->m; i++)
            y[i] *= aval[A->opt_csr_mat.idiag[i]];
    }

    // (U+D)z = y
    if(descr->type == aoclsparse_matrix_type_general
       || descr->fill_mode == aoclsparse_fill_mode_upper)
    {
        // Use the upper triangle directly
        aoclsparse_set_mat_fill_mode(&descr_cpy, aoclsparse_fill_mode_upper);
        trans = aoclsparse_operation_none;
    }
    else
    {
        // symmetric with lower triangle given. use transpose trsv
        aoclsparse_set_mat_fill_mode(&descr_cpy, aoclsparse_fill_mode_lower);
        trans = aoclsparse_operation_transpose;
    }
    status = aoclsparse::trsv(trans, alpha, A, &descr_cpy, y, 1, z, 1, avxversion);
    if(status != aoclsparse_status_success)
        return status;

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_itsol_rci_solve(aoclsparse_itsol_data<T> *itsol,
                                             aoclsparse_itsol_rci_job *ircomm,
                                             T                       **u,
                                             T                       **v,
                                             T                        *x,
                                             tolerance_t<T>            rinfo[100])
{
    /* Generic iterative solver interface, call the correct solver based on user defined option
     * Possible exits:
     * - Internal error: itsol->solver is unknown
     * From CG solver:
     * - maximum iteration
     * - User requested termination
     * - handle was not initialized
     */
    aoclsparse_status status;
    if(ircomm == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(itsol == nullptr)
    {
        *ircomm = aoclsparse_rci_stop;
        return aoclsparse_status_internal_error;
    }
    if(u == nullptr || v == nullptr || x == nullptr || rinfo == nullptr)
    {
        *ircomm = aoclsparse_rci_stop;
        return aoclsparse_status_invalid_pointer;
    }
    // Not solving yet ==> initialize solver's data structure
    // and read in the options
    if(!itsol->solving)
    {
        // Initialize solvers (& preconditioners if needed) based on the options
        status = aoclsparse_itsol_solver_init(itsol);
        if(status != aoclsparse_status_success)
        {
            *ircomm = aoclsparse_rci_stop;
            return status;
        }
        // skip solver's initialization on next RCI calls till we are finish (rci_stop)
        itsol->solving = true;
        // lock options so that they are unchangable while we are solving
        itsol->opts.Lock();
    }

    switch(itsol->solver)
    {
    case solver_cg:
        status = aoclsparse_cg_rci_solve(itsol, ircomm, u, v, x, rinfo);
        break;

    case solver_gmres:
        status = aoclsparse_gmres_rci_solve(itsol, ircomm, u, v, x, rinfo);
        break;

    default:
        // undefined solver
        status = aoclsparse_status_internal_error;
        break;
    }

    // On any error, we stop solving (reset the flag) to be able to
    // reinitialize on the next call
    if(status != aoclsparse_status_success)
        *ircomm = aoclsparse_rci_stop;
    if(*ircomm == aoclsparse_rci_stop)
    {
        itsol->solving = false;
        itsol->opts.UnLock();
    }

    return status;
}

template <typename T>
aoclsparse_status aoclsparse_itsol_solve(
    aoclsparse_itsol_data<T>  *itsol,
    aoclsparse_int             n,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    const T                   *b,
    T                         *x,
    tolerance_t<T>             rinfo[100],
    aoclsparse_int precond(aoclsparse_int flag, aoclsparse_int n, const T *u, T *v, void *udata),
    aoclsparse_int monit(
        aoclsparse_int n, const T *x, const T *r, tolerance_t<T> rinfo[100], void *udata),
    void *udata)
{
    aoclsparse_status status;
    if(itsol == nullptr)
        return aoclsparse_status_internal_error;
    if(x == nullptr || rinfo == nullptr)
        return aoclsparse_status_invalid_pointer;

    for(aoclsparse_int i = 0; i < 100; i++)
        rinfo[i] = 0.0;

    // check and initialize n & b
    status = aoclsparse_itsol_rci_input(itsol, n, b);
    if(status != aoclsparse_status_success)
        return status;

    // Initialize solvers (& preconditioners if needed) based on the options
    status = aoclsparse_itsol_solver_init(itsol);
    if(status != aoclsparse_status_success)
        return status;

    if(!mat->opt_csr_mat.is_optimized)
    {
        // CG needs opt_csr to run
        status = aoclsparse_csr_csc_optimize<T>(mat);
        if(status != aoclsparse_status_success)
            return status;
    }
    // indicate that initialization has been done
    // (not really needed unless the user start mixing forward iface and RCI)
    itsol->solving = true;
    // lock options so that they are unchangable while we are solving
    itsol->opts.Lock();

    switch(itsol->solver)
    {
    case solver_cg:
        status = aoclsparse_cg_solve(itsol, mat, descr, x, rinfo, precond, monit, udata);
        break;

    case solver_gmres:
        status = aoclsparse_gmres_solve(itsol, mat, descr, x, rinfo, precond, monit, udata);
        break;

    default:
        // undefined solver
        status = aoclsparse_status_internal_error;
        break;
    }

    // solving finished
    itsol->solving = false;
    itsol->opts.UnLock();

    return status;
}

/* CG solver in reverse communication interface
 * Possible exits:
 * - maximum number of iteration reached
 * - user requested termination
 * - Allocation
 */
template <typename T>
aoclsparse_status aoclsparse_cg_rci_solve(aoclsparse_itsol_data<T> *itsol,
                                          aoclsparse_itsol_rci_job *ircomm,
                                          T                       **u,
                                          T                       **v,
                                          T                        *x,
                                          tolerance_t<T>            rinfo[100])
{
    // all pointers were already checked by the caller --> safe to use them
    aoclsparse_status exit_status = aoclsparse_status_success;
    aoclsparse_int    i, n;
    cg_data<T>       *cg;
    T                 pq, rz_new;
    bool              loop;
    tolerance_t<T>    v1 = 1.0;
    cg                   = itsol->cg;
    n                    = itsol->n;

    // Check for user's request to stop (but ignore on the first input)
    if(cg->task != task_start && *ircomm == aoclsparse_rci_interrupt)
    {
        // Exit cleanly
        // TODO print exit message
        *ircomm = aoclsparse_rci_stop;
        return aoclsparse_status_user_stop;
    }
    do
    {
        loop = false;

        switch(cg->task)
        {
        case task_start:
            // reset statistics
            for(i = 0; i < 100; i++)
                rinfo[i] = 0.0;
            cg->niter = 0;

            // Set up memory to compute r = Ax - b
            // Set r = -b; p = x
            // Request q = A*p
            // store the norm of b
            for(i = 0; i < n; i++)
            {
                cg->r[i] = -itsol->b[i];
                cg->p[i] = x[i];
            }
            cg->bnorm2 = blis::cblas_nrm2(n, itsol->b, 1);
            if(cg->bnorm2 != cg->bnorm2) // test for NaN
                return aoclsparse_status_invalid_value; // vector b is rubbish
            rinfo[RINFO_RHS_NORM] = cg->bnorm2;
            cg->brtol             = cg->rtol * cg->bnorm2;

            // TO REPLACE with
            //bli_dcopyv(BLIS_NO_CONJUGATE, handle->n, handle->b, 1, cgd->r, 1);
            //scal = -1.0;
            //bli_dscalv(BLIS_NO_CONJUGATE, handle->n, &scal, cgd->r, 1);
            //bli_dcopyv(BLIS_NO_CONJUGATE, handle->n, x, 1, cgd->p, 1);
            *ircomm  = aoclsparse_rci_mv;
            cg->task = task_init_res;
            *u       = cg->p;
            *v       = cg->q;
            break;

        case task_init_res:
            // Have: q = Ax, r = -b
            // Compute initial residual r = r + q (=Ax-b)
            // and its norm
            for(i = 0; i < n; i++)
            {
                cg->r[i] = cg->r[i] + cg->q[i];
            }
            cg->rnorm2 = blis::cblas_nrm2(n, cg->r, 1);
            //bli_dnormfv(handle->n, cgd->r, 1, &cgd->rnorm2);
            if(cg->rnorm2 != cg->rnorm2) // test for NaN
            {
                exit_status = aoclsparse_status_numerical_error;
                break;
            }
            rinfo[RINFO_RES_NORM] = cg->rnorm2;

            // initialize p = 0 (so that initil p=-z from p:=-z + beta*p)
            for(i = 0; i < n; i++)
                cg->p[i] = aoclsparse_numeric::zero<T>();
            // rz doesn't matter in the first iteration as long as it is not 0
            if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
            {
                cg->rz = v1;
            }
            else
            {
                cg->rz = {(tolerance_t<T>)v1, (tolerance_t<T>)v1};
            }

            // continue without break to check convergence
            cg->task = task_check_conv;
            __attribute__((fallthrough));

        case task_check_conv:
            // check convergence via internal stopping criteria and
            // pass the results (including the full residual 'r') to the user;
            // if not converged go to monitoring step for him/her to implement
            // bespoke stop criteria
            *u = cg->r;
            *v = nullptr;

            // If we ever do printing, here would be the place
            // call printer from handle.printer (can be user defined or default printer_pcg(&stream, &rinfo))
            // if (handle.iopts[PRINT_LEVEL]>1 && handle.iter % handle.iopts[PRINT_FREQ] == 0) call handle.printer(&stream, &rinfo)

            // Absolute tolerance ||Ax_k - b|| < atol
            if((0.0 < cg->atol) && (cg->rnorm2 <= cg->atol))
            {
                *ircomm = aoclsparse_rci_stop;
                break;
            }
            // Relative tolerance ||Ax_k - b|| / ||b|| < rtol
            if((0.0 < cg->rtol) && (cg->rnorm2 <= cg->brtol))
            {
                *ircomm = aoclsparse_rci_stop;
                break;
            }
            if(cg->maxit > 0 && cg->niter > cg->maxit)
            {
                *ircomm     = aoclsparse_rci_stop;
                exit_status = aoclsparse_status_maxit;
                break;
            }
            // monitoring step: user defined criterion, break;
            // then continue starting new iteration
            cg->task = task_start_iter;
            *ircomm  = aoclsparse_rci_stopping_criterion;
            break;

        case task_start_iter:
            // start new iteration, request preconditioner z=inv(M)*r
            cg->niter++;
            rinfo[RINFO_ITER] = (tolerance_t<T>)cg->niter;

            cg->task = task_compute_beta;
            if(!cg->precond)
            {
                // unpreconditioned method, don't break and go directly to task_compute_beta
                for(i = 0; i < n; i++)
                    cg->z[i] = cg->r[i];
                //bli_dcopyv(BLIS_NO_CONJUGATE, handle->n, cgd->r, 1, cgd->z, 1);
            }
            else
            {
                *ircomm = aoclsparse_rci_precond;
                *u      = cg->r;
                *v      = cg->z;
                break;
            }
            __attribute__((fallthrough));

        case task_compute_beta:
            // Compute beta and new search direction p
            // Have: Mz = r
            // Compute beta = r^t.z / rz
            //            p = beta*p - z
            // request the matrix vector product for A*p

            rz_new = aoclsparse_numeric::zero<T>();
            for(i = 0; i < n; i++)
                rz_new += cg->r[i] * cg->z[i];
            //bli_ddotv(
            //    BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, handle->n, cgd->r, 1, cgd->z, 1, &cgd->beta);
            if(aoclsparse_is_negative_or_nearzero(cg->rz))
                return aoclsparse_status_numerical_error;

            cg->beta = rz_new / cg->rz;
            cg->rz   = rz_new;
            for(i = 0; i < n; i++)
                cg->p[i] = cg->beta * cg->p[i] - cg->z[i];
            //scal = -1.0;
            //bli_dscalv(BLIS_NO_CONJUGATE, handle->n, &cgd->beta, cgd->p, 1);
            //bli_daxpyv(BLIS_NO_CONJUGATE, handle->n, &scal, cgd->z, 1, cgd->p, 1);
            *ircomm  = aoclsparse_rci_mv;
            cg->task = task_take_step;
            *u       = cg->p;
            *v       = cg->q;
            break;

        case task_take_step:
            // Have: q = Ap
            // Compute pq = p^t.q
            //         alpha = rz/pq
            // x_{k+1} = x_k + alpha p_k
            // r_{k+1} = r_k + alpha Ap_k
            // Then check convergence
            pq = aoclsparse_numeric::zero<T>();
            for(i = 0; i < n; i++)
                pq += cg->p[i] * cg->q[i];
            //bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, handle->n, cgd->p, 1, cgd->q, 1, &pq);
            if(aoclsparse_is_negative_or_nearzero(pq)) // system matrix A is not positive definite
                return aoclsparse_status_numerical_error;
            cg->alpha = cg->rz / pq;
            for(i = 0; i < n; i++)
            {
                x[i] += cg->alpha * cg->p[i];
                cg->r[i] += cg->alpha * cg->q[i];
            }
            //bli_daxpyv(BLIS_NO_CONJUGATE, handle->n, &cgd->alpha, cgd->p, 1, x, 1);
            //bli_daxpyv(BLIS_NO_CONJUGATE, handle->n, &cgd->alpha, cgd->q, 1, cgd->r, 1);

            // compute norm of the residual in rinfo[0]
            cg->rnorm2 = blis::cblas_nrm2(n, cg->r, 1);
            //bli_dnormfv(handle->n, cgd->r, 1, &cgd->rnorm2);
            if(cg->rnorm2 != cg->rnorm2) // test for NaN
            {
                exit_status = aoclsparse_status_numerical_error;
                break;
            }
            rinfo[RINFO_RES_NORM] = cg->rnorm2;
            // jump to check convergence without going to the user
            loop     = true;
            cg->task = task_check_conv;
            break;

        default:
            // this shouldn't happen
            *ircomm = aoclsparse_rci_stop;
            return aoclsparse_status_internal_error;
            break;
        }
    } while(loop);

    return exit_status;
}

/*
    Performs a single pass backward triangular solve to compute solution for
    yk = Hk_inv.||r0||e1
    m[i/p]:  number of restart iterations
    nn[i/p]: dimension of matrix r
    r[i/p]: Upper Hessenberg Matrix
    g[i/p]: residual vector
    y[o/p]: correction vector for x, x = x0 + zk and zk = vk.yk
*/
template <typename T>
aoclsparse_status
    aoclsparse_backward_solve(aoclsparse_int m, aoclsparse_int nn, const T *r, const T *g, T *y)
{
    aoclsparse_status exit_status = aoclsparse_status_success;
    if(r == NULL || g == NULL || y == NULL)
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_int i, j;
    for(j = m - 1; j >= 0; j--)
    {
        T diag;
        y[j] = g[j];
        for(i = j + 1; i < m; i++)
        {
            y[j] -= r[j * nn + i] * y[i];
        }
        diag = r[j * nn + j];
        if(aoclsparse_is_nearzero(diag))
        {
            exit_status = aoclsparse_status_numerical_error;
            break;
        }
        y[j] /= diag;
    }
    return exit_status;
}
/* GMRES solver in reverse communication interface
 * Possible exits:
 * - maximum number of iteration reached
 * - user requested termination
 * - Allocation errors
 */
template <typename T>
aoclsparse_status aoclsparse_gmres_rci_solve(aoclsparse_itsol_data<T> *itsol,
                                             aoclsparse_itsol_rci_job *ircomm,
                                             T                       **io1,
                                             T                       **io2,
                                             T                        *x,
                                             tolerance_t<T>            rinfo[100])
{
    aoclsparse_status exit_status = aoclsparse_status_success;
    bool is_residnorm_below_abs_tolerance, is_rhsnorm_below_rel_tolerance, is_max_iters_reached;
    bool is_restart_cycled_ended, is_residual_vector_orthogonal;
    T   *v = NULL, *h = NULL, *g = NULL, *s = NULL, *z = NULL;
    aoclsparse_int n = 0, m = 0, j = 0, k = 0;
    gmres_data<T> *gmres;
    T              temp_hh, hv, rr, r1, r2, g0;
    bool           loop = false;
    tolerance_t<T> hh = 0.0, *c = NULL, temp_g0 = 0.0;

    gmres = itsol->gmres;
    n     = itsol->n;
    m     = gmres->restart_iters;
    v     = gmres->v;
    z     = gmres->z;
    h     = gmres->h;
    g     = gmres->g;
    c     = gmres->c;
    s     = gmres->s;

    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        hv = 0.0;
        rr = 0.0;
        r1 = 0.0;
        r2 = 0.0;
    }
    else
    {
        hv = aoclsparse_numeric::zero<T>();
        rr = aoclsparse_numeric::zero<T>();
        r1 = aoclsparse_numeric::zero<T>();
        r2 = aoclsparse_numeric::zero<T>();
    }
    // Check for user's request to stop (but ignore on the first input)
    if(gmres->task != task_gmres_start && *ircomm == aoclsparse_rci_interrupt)
    {
        // Exit cleanly
        // TODO print exit message
        *ircomm = aoclsparse_rci_stop;
        return aoclsparse_status_user_stop;
    }
    /*
        Any GMRES precondition(ILU0) specific tasks can be performed here
        like allocating any memory. Check for the following condition,
        if((itsol->gmres)->precond == 2 //ILU) {...}
    */
    do
    {
        loop = false;

        switch(gmres->task)
        {
        case task_gmres_start:
            // Set up memory to compute A.x0
            // Request v = A*x0
            *io1        = x;
            *io2        = v;
            *ircomm     = aoclsparse_rci_mv;
            gmres->task = task_gmres_init_res;
            break;

        case task_gmres_init_res:
            j = gmres->j;
            /* 1. start: setup initial residual data
            compute r0 = b - A.x0
            compute v1 = r0/||r0||

            spmv: v = A . x0
            axpby: v = b - v (i.e., v = b - A.x0)
            g[0] = || v || (Norm2)
            v1 = v/g[0] = v/||v||
            */
            //Compute b norm for relative tolerance checks
            gmres->bnorm2 = blis::cblas_nrm2(n, itsol->b, 1);

            if(std::isnan(gmres->bnorm2)) // test for NaN
            {
                return aoclsparse_status_invalid_value; // vector b is rubbish
            }

            rinfo[RINFO_RHS_NORM] = gmres->bnorm2;
            gmres->brtol          = gmres->rtol * gmres->bnorm2;
            rinfo[RINFO_RHS_NORM] = gmres->brtol;

            //exit if both the tolerances are zeroes
            if((aoclsparse_is_nearzero(gmres->atol)) && (aoclsparse_is_nearzero(gmres->brtol)))
            {
                exit_status = aoclsparse_status_invalid_value;
                *ircomm     = aoclsparse_rci_stop;
                break;
            }

            //step 1.1
            //v = b - v
            if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
            {
                blis::cblas_axpby(n, 1.0, itsol->b, 1, -1.0, v, 1);
            }
            else
            {
                blis::cblas_axpby(n, {1.0, 0.0}, itsol->b, 1, {-1.0, 0.0}, v, 1);
            }
            //step 1.2
            temp_g0 = blis::cblas_nrm2(n, v, 1);
            if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
            {
                g[0] = temp_g0;
            }
            else
            {
                g[0] = {(tolerance_t<T>)temp_g0, (tolerance_t<T>)0.0};
            }
            //step 1.3
            gmres->rnorm2         = temp_g0;
            rinfo[RINFO_RES_NORM] = gmres->rnorm2;
            // Absolute tolerance ||Ax_k - b|| < atol
            if((0 < gmres->rnorm2) && (gmres->rnorm2 <= gmres->atol))
            {
                exit_status       = aoclsparse_status_success;
                *ircomm           = aoclsparse_rci_stop;
                rinfo[RINFO_ITER] = (tolerance_t<T>)gmres->niter;
                break;
            }
            // Relative tolerance ||Ax_k - b|| / ||b|| < rtol
            if((0.0 < gmres->rnorm2) && (gmres->rnorm2 <= gmres->brtol))
            {
                exit_status       = aoclsparse_status_success;
                *ircomm           = aoclsparse_rci_stop;
                rinfo[RINFO_ITER] = (tolerance_t<T>)gmres->niter;
                break;
            }
            blis::cblas_scal(n, (1.0 / gmres->rnorm2), v, 1);

            if(!gmres->precond)
            {
                // Do not perform preconditionner step, don't return to user space
                //and go directly to task_init_precond
                gmres->task = task_gmres_init_precond;
                // jump to check convergence without going to the user
                loop = true;
            }
            else
            {
                *ircomm     = aoclsparse_rci_precond;
                gmres->task = task_gmres_init_precond;
                *io1        = v + j * n; //const input
                *io2        = z + j * n; //r-w output
            }
            break;

        case task_gmres_init_precond:
            // SPMV inside restart loop (j loop) after triangular solve from Precond
            j = gmres->j;
            if(!gmres->precond)
            {
                *io1 = v + j * n;
            }
            else
            {
                *io1 = z + j * n;
            }
            *io2 = v + (j + 1) * n;

            *ircomm     = aoclsparse_rci_mv;
            gmres->task = task_gmres_start_iter;
            break;

        case task_gmres_start_iter:
            j = gmres->j;
            //Form hessenberg Matrix and perform plane Rotations
            //step 2.2
            for(int i = 0; i <= j; i++)
            {
                if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
                {
                    h[i * m + j] = blis::cblas_dot(n, v + (j + 1) * n, 1, v + i * n, 1);
                }
                else
                {
                    h[i * m + j] = blis::cblas_dotc(n, v + (j + 1) * n, 1, v + i * n, 1);
                }
            }
            //step 2.3
            for(k = 0; k < n; k++)
            {
                if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
                {
                    hv = blis::cblas_dot(j + 1, h + j, m, v + k, n);
                }
                else
                {
                    hv = blis::cblas_dotc(j + 1, h + j, m, v + k, n);
                }
                v[(j + 1) * n + k] -= hv;
            }
            hh                            = blis::cblas_nrm2(n, v + (j + 1) * n, 1);
            is_residual_vector_orthogonal = (hh < gmres->atol) || (hh < gmres->brtol);
            if(!is_residual_vector_orthogonal)
            {
                blis::cblas_scal(n, (1.0 / hh), v + (j + 1) * n, 1);
            }
            else
            {
                // this condition means residual vector is already orthogonal
                // to the previous Krylov subspace vectors
                // the current approximation of the solution vector (x) is
                // appropriate as it is the best approximation we have.
                j += 1; //although convergence is reached while checking, we count the iteration so far
                gmres->niter += j; //update iterations
                rinfo[RINFO_ITER] = (tolerance_t<T>)gmres->niter;
                //Since hh is zero, it means that the current residual vector is already
                //orthogonal to the previous j columns of Q.
                rinfo[RINFO_RES_NORM] = hh;
                exit_status           = aoclsparse_status_success;
                *ircomm               = aoclsparse_rci_stop;
                break;
            }

            /* plane rotations */
            for(int i = 0; i < j; i++)
            {
                r1                 = h[i * m + j];
                r2                 = h[(i + 1) * m + j];
                h[i * m + j]       = c[i] * r1 - s[i] * r2;
                h[(i + 1) * m + j] = s[i] * r1 + c[i] * r2;
            }
            rr = h[j * m + j];
            hh = -hh;

            if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
            {
                temp_hh = hh;
                libflame::lartg<T>(&rr, &temp_hh, &c[j], &s[j], &h[j * m + j]);
            }
            else
            {
                temp_hh = {(tolerance_t<T>)hh, (tolerance_t<T>)0.0};
                if constexpr(std::is_same_v<T, std::complex<float>>)
                {
                    scomplex *pf = reinterpret_cast<scomplex *>(&rr);
                    scomplex *pg = reinterpret_cast<scomplex *>(&temp_hh);
                    scomplex *ps = reinterpret_cast<scomplex *>(&s[j]);
                    scomplex *pr = reinterpret_cast<scomplex *>(&h[j * m + j]);
                    libflame::lartg<scomplex, float>(pf, pg, &c[j], ps, pr);
                }
                else if constexpr(std::is_same_v<T, std::complex<double>>)
                {
                    dcomplex *pf = reinterpret_cast<dcomplex *>(&rr);
                    dcomplex *pg = reinterpret_cast<dcomplex *>(&temp_hh);
                    dcomplex *ps = reinterpret_cast<dcomplex *>(&s[j]);
                    dcomplex *pr = reinterpret_cast<dcomplex *>(&h[j * m + j]);
                    libflame::lartg<dcomplex, double>(pf, pg, &c[j], ps, pr);
                }
            }
            g0            = g[j];
            g[j]          = c[j] * g0;
            g[j + 1]      = s[j] * g0;
            gmres->rnorm2 = std::abs(g[j /*m*/]); /* residual */

            rinfo[RINFO_ITER]     = (tolerance_t<T>)gmres->niter;
            rinfo[RINFO_RES_NORM] = gmres->rnorm2;

            //control should come here only if (j < restart_iters) and inner loop is still iterating
            gmres->j++; //increment inner loop index (restart cycle loop)
            j = gmres->j;

            is_restart_cycled_ended = (j >= m);

            //check if restart loop has ended
            //if yes then restart the whole process again using the newly updated x
            if(is_restart_cycled_ended)
            {
                gmres->task = task_gmres_x_update;
                loop        = true;
                break;
            }

            if(!gmres->precond)
            {
                // Do not perform preconditionner step, don't return to user space
                //and go directly to task_init_precond
                gmres->task = task_gmres_end_iter;
                loop        = true;
            }
            else
            {
                *ircomm     = aoclsparse_rci_precond;
                gmres->task = task_gmres_end_iter;
                *io1        = v + j * n; //const input
                *io2        = z + j * n; //r-w output
            }
            break;
        case task_gmres_end_iter:
            // SPMV inside restart loop (j loop) after triangular solve from Precond
            j = gmres->j;
            if(!gmres->precond)
            {
                *io1 = v + j * n;
            }
            else
            {
                *io1 = z + j * n;
            }
            *io2 = v + (j + 1) * n;

            *ircomm     = aoclsparse_rci_mv;
            gmres->task = task_gmres_start_iter;
            break;

        case task_gmres_x_update:
            /* 3. form the approximate solution */
            // M . x = b, solve for x using backward pass of triangular solve
            // h . s = g, solve for s from Solver point of view
            j           = gmres->j;
            exit_status = aoclsparse_backward_solve<T>(j,
                                                       m,
                                                       h, //hessenberg matrix h, known
                                                       g, //g, rhs known
                                                       s); //s = ?, unknown

            if(exit_status != aoclsparse_status_success)
            {
                break;
            }
            //Step 3.2
            /*
                M_inv . Vm . ym
                Since (M_inv . Vm) is already calculated above in the j loop and is stored in
                z(m+1, n) matrix. Use the same to perform linear combination for x update
            */
            if(!gmres->precond)
            {
                for(int i = 0; i < n; i++)
                {
                    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
                    {
                        x[i] += blis::cblas_dot(m, v + i, n, s, 1);
                    }
                    else
                    {
                        x[i] += blis::cblas_dotc(m, v + i, n, s, 1);
                    }
                }
            }
            else
            {
                for(int i = 0; i < n; i++)
                {
                    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
                    {
                        x[i] += blis::cblas_dot(m, z + i, n, s, 1);
                    }
                    else
                    {
                        x[i] += blis::cblas_dotc(m, z + i, n, s, 1);
                    }
                }
            }
            gmres->rnorm2 = std::abs(g[j /*m*/]); /* residual */
            gmres->niter += j; //update iterations
            rinfo[RINFO_ITER]     = (tolerance_t<T>)gmres->niter;
            rinfo[RINFO_RES_NORM] = gmres->rnorm2;

            // check convergence via internal stopping criteria
            // Absolute tolerance ||Ax_k - b|| < atol
            is_residnorm_below_abs_tolerance
                = (0.0 < gmres->atol) && (gmres->rnorm2 <= gmres->atol);

            is_rhsnorm_below_rel_tolerance
                = (0.0 < gmres->rnorm2) && (gmres->rnorm2 <= gmres->brtol);

            is_max_iters_reached = ((gmres->maxit > 0) && (gmres->niter >= gmres->maxit));
            //check if restart loop has ended
            //if yes then restart the whole process again using the newly updated x
            if(j >= m)
            {
                gmres->j = 0;
            }
            // Absolute tolerance ||Ax_k - b|| < atol
            if(is_residnorm_below_abs_tolerance)
            {
                *ircomm     = aoclsparse_rci_stopping_criterion;
                gmres->task = task_gmres_convergence_check;
                break;
            }
            // Relative tolerance ||Ax_k - b|| / ||b|| < rtol
            if(is_rhsnorm_below_rel_tolerance)
            {
                *ircomm     = aoclsparse_rci_stopping_criterion;
                gmres->task = task_gmres_convergence_check;
                break;
            }
            if(is_max_iters_reached)
            {
                *ircomm     = aoclsparse_rci_stopping_criterion;
                gmres->task = task_gmres_convergence_check;
                break;
            }

            *ircomm     = aoclsparse_rci_stopping_criterion;
            gmres->task = task_gmres_restart_cycle;
            break;
        case task_gmres_restart_cycle:
            // Request v = A*x_est
            *io1 = x; //this x should contain the new estimate of x after restart loop
            *io2 = v; //v+0 buffer which will have spmv output using A and x_est

            *ircomm     = aoclsparse_rci_mv;
            gmres->task = task_gmres_init_res;
            break;
        case task_gmres_convergence_check:
            is_residnorm_below_abs_tolerance
                = (0.0 < gmres->atol) && (gmres->rnorm2 <= gmres->atol);

            is_rhsnorm_below_rel_tolerance
                = (0.0 < gmres->rnorm2) && (gmres->rnorm2 <= gmres->brtol);

            is_max_iters_reached = ((gmres->maxit > 0) && (gmres->niter >= gmres->maxit));

            // Absolute tolerance ||Ax_k - b|| < atol
            if(is_residnorm_below_abs_tolerance)
            {
                exit_status = aoclsparse_status_success;
                *ircomm     = aoclsparse_rci_stop;
                break;
            }
            // Relative tolerance ||Ax_k - b|| / ||b|| < rtol
            if(is_rhsnorm_below_rel_tolerance)
            {
                exit_status = aoclsparse_status_success;
                *ircomm     = aoclsparse_rci_stop;
                break;
            }
            if(is_max_iters_reached)
            {
                exit_status = aoclsparse_status_maxit;
                *ircomm     = aoclsparse_rci_stop;
                break;
            }
            break;
        default:
            // this shouldn't happen
            *ircomm = aoclsparse_rci_stop;
            return aoclsparse_status_internal_error;
            break;
        }
    } while(loop);
    return exit_status;
}

template <typename T>
aoclsparse_status aoclsparse_cg_solve(
    aoclsparse_itsol_data<T>  *itsol,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    T                         *x,
    tolerance_t<T>             rinfo[100],
    aoclsparse_int precond(aoclsparse_int flag, aoclsparse_int n, const T *u, T *v, void *udata),
    aoclsparse_int monit(
        aoclsparse_int n, const T *x, const T *r, tolerance_t<T> rinfo[100], void *udata),
    void *udata)
{
    /* CG solver in direct communication interface
     * Possible exits:
     * - maximum number of iteration reached
     * - user requested termination (via monit or precond)
     * - invalid size (system dimensions are not consistent with the input matrix)
     * - invalid value (only symmetric matrices with lower fill mode are allowed)
     * - Allocation
     * - internal error: MV failed
     */
    aoclsparse_int           flag;
    aoclsparse_int           n      = itsol->n;
    aoclsparse_itsol_rci_job ircomm = aoclsparse_rci_start;
    T                       *u      = nullptr;
    T                       *v      = nullptr;
    T                        alpha, beta;
    std::vector<T>           y;
    aoclsparse_status        exit_status = aoclsparse_status_success;
    aoclsparse_status        status;
    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        alpha = 1.0;
        beta  = 0.0;
    }
    else
    {
        alpha = {(tolerance_t<T>)1.0, (tolerance_t<T>)0.0};
        beta  = {(tolerance_t<T>)0.0, (tolerance_t<T>)0.0};
    }
    if(mat->m != n || mat->n != n)
        return aoclsparse_status_invalid_size;
    if(descr->type != aoclsparse_matrix_type_symmetric)
        return aoclsparse_status_invalid_value;
    if(descr->fill_mode != aoclsparse_fill_mode_lower)
        // TODO: support fill upper
        // symmetric matrix-vector product only work for lower triangular matrices...
        return aoclsparse_status_invalid_value;

    if(itsol->cg->precond == 1 && precond == nullptr)
        return aoclsparse_status_invalid_pointer;
    if((itsol->cg)->precond == 3) // Add other preconds here...
    {
        // Symmetric Gauss-Seidel requested, allocate some memory
        if((!(mat->opt_csr_full_diag) && descr->diag_type != aoclsparse_diag_type_unit)
           || descr->diag_type == aoclsparse_diag_type_zero)
            // Gauss-Seidel needs a full diagonal to perform the triangle solve
            return aoclsparse_status_invalid_value;
        try
        {
            y.resize(n, 0.0);
        }
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
        }
    }

    // Call CG solver
    while(ircomm != aoclsparse_rci_stop)
    {
        exit_status = aoclsparse_itsol_rci_solve(itsol, &ircomm, &u, &v, x, rinfo);
        if(exit_status && ircomm != aoclsparse_rci_stop)
            // shouldn't happen, exits should cleanly return the stop signal
            return exit_status;
        switch(ircomm)
        {
        case aoclsparse_rci_mv:
            // Compute v = Au
            status = aoclsparse::mv<T>(aoclsparse_operation_none, &alpha, mat, descr, u, &beta, v);
            if(status != aoclsparse_status_success)
                // Shouldn't happen, invalid pointer/value/not implemented should be checked before
                return aoclsparse_status_internal_error;
            break;

        case aoclsparse_rci_precond:
            switch((itsol->cg)->precond)
            {
            case 1:
                // User defined preconditioner, call precond(...)
                // precond pointer was already verified
                flag = precond(0, n, u, v, udata);
                // if the user indicates that preconditioner could not be applied
                // what to do: recovery or terminate?
                // skip precond step?
                if(flag != 0)
                    ircomm = aoclsparse_rci_interrupt;
                break;
            case 2:
                // Jacobi - Not yet implemented
                break;
            case 3:
                // Symmetric Gauss-Seidel
                status = aoclsparse_itsol_symgs(mat, descr, u, y.data(), v);
                if(status != aoclsparse_status_success)
                    // symgs step failed. shouldn't happen, internal error?
                    return aoclsparse_status_internal_error;
                break;
            default: // None
                for(aoclsparse_int i = 0; i < n; i++)
                    v[i] = u[i];
                break;
            }

            break;

        case aoclsparse_rci_stopping_criterion:
            if(monit)
            {
                flag = monit(n, u, v, rinfo, udata);
                if(flag != 0)
                    ircomm = aoclsparse_rci_interrupt;
            }
            break;

        default:
            break;
        }
    }

    return exit_status;
}
/* GMRES solver in direct communication interface
     * Possible exits:
     * - maximum number of iteration reached
     * - user requested termination (via monit or precond)
     * - invalid size (system dimensions are not consistent with the input matrix)
     * - invalid value (only symmetric matrices with lower fill mode are allowed)
     * - Allocation errors
     * - internal error: MV failed
     */
template <typename T>
aoclsparse_status aoclsparse_gmres_solve(
    aoclsparse_itsol_data<T>  *itsol,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    T                         *x,
    tolerance_t<T>             rinfo[100],
    aoclsparse_int             precond(
        aoclsparse_int flag, aoclsparse_int n, const T *io1, T *io2, void *udata),
    aoclsparse_int monit(
        aoclsparse_int n, const T *x, const T *r, tolerance_t<T> rinfo[100], void *udata),
    void *udata)
{
    aoclsparse_int           flag;
    aoclsparse_int           n      = itsol->n;
    aoclsparse_itsol_rci_job ircomm = aoclsparse_rci_start;
    T                       *io1    = nullptr;
    T                       *io2    = nullptr;
    T                        alpha, beta;
    T                       *precond_data = NULL;
    aoclsparse_status        exit_status  = aoclsparse_status_success;
    aoclsparse_status        status;

    if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        alpha = 1.0;
        beta  = 0.0;
    }
    else
    {
        alpha = {(tolerance_t<T>)1.0, (tolerance_t<T>)0.0};
        beta  = {(tolerance_t<T>)0.0, (tolerance_t<T>)0.0};
    }
    if(mat->m != n || mat->n != n)
        return aoclsparse_status_invalid_size;

    if(itsol->gmres->precond == 1 && precond == nullptr)
    {
        //user requested his/her own precond but valid function pointer not provided
        return aoclsparse_status_invalid_pointer;
    }
    /*
        Any GMRES precondition(ILU0) specific tasks can be performed here
        like allocating any memory. Check for the following condition,
        if((itsol->gmres)->precond == 2 //ILU) {...}
    */
    // Call GMRES solver
    while(ircomm != aoclsparse_rci_stop)
    {
        exit_status = aoclsparse_itsol_rci_solve(itsol, &ircomm, &io1, &io2, x, rinfo);
        if(exit_status && ircomm != aoclsparse_rci_stop)
            // shouldn't happen, exits should cleanly return the stop signal
            return exit_status;
        switch(ircomm)
        {
        case aoclsparse_rci_mv:
            // Compute v = Au
            status
                = aoclsparse::mv<T>(aoclsparse_operation_none, &alpha, mat, descr, io1, &beta, io2);
            if(status != aoclsparse_status_success)
                // Shouldn't happen, invalid pointer/value/not implemented should be checked before
                return aoclsparse_status_internal_error;
            break;

        case aoclsparse_rci_precond:
            switch((itsol->gmres)->precond)
            {
            case 1: // User defined preconditioner, call precond(...)
                // precond pointer was already verified
                flag = precond(0, n, io1, io2, udata);
                // if the user indicates that preconditioner could not be applied
                // what to do: recovery or terminate?
                // skip precond step?
                if(flag != 0)
                    ircomm = aoclsparse_rci_interrupt;
                break;
            case 2:
                //Run ILU Preconditioner only once in the beginning
                //Run Triangular Solve using ILU0 factorization
                aoclsparse_ilu_template(aoclsparse_operation_none,
                                        mat, //precond martix M
                                        descr,
                                        &precond_data,
                                        io2, //x = ?, io1 = z+j*n,
                                        (const T *)io1); //rhs, io2 = v+j*n
                break;
            default: // None
                for(aoclsparse_int i = 0; i < n; i++)
                    io2[i] = io1[i];
                break;
            }

            break;

        case aoclsparse_rci_stopping_criterion:
            if(monit)
            {
                flag = monit(n, io1, io2, rinfo, udata);
                if(flag != 0)
                    ircomm = aoclsparse_rci_interrupt;
            }
            break;

        default:
            break;
        }
    }

    return exit_status;
}
// Option setter helper
template <typename T>
aoclsparse_status handle_parse_option(aoclsparse_options::OptionRegistry<T> &opts,
                                      const char                            *option,
                                      const char                            *value)
{
    const aoclsparse_int byuser = 1;
    aoclsparse_int       flag;
    aoclsparse_int       iquery, iset;
    aoclsparse_int       otype = 0;
    tolerance_t<T>       rquery, rset;
    bool                 bquery, bset;
    std::string          squery, sset, name;

    if(!option)
        return aoclsparse_status_invalid_pointer;
    if(!value)
        return aoclsparse_status_invalid_pointer;

    name = option;

    // Find type of option and convert value before registering
    flag = opts.GetOption(name, iquery);
    if(!flag)
        otype = 1;

    if(!otype)
    {
        flag = opts.GetOption(name, rquery);
        if(!flag)
            otype = 2;
    }
    if(!otype)
    {
        flag = opts.GetOption(name, squery, iquery);
        if(!flag)
            otype = 3;
    }
    if(!otype)
    {
        flag = opts.GetOption(name, bquery);
        if(!flag)
            otype = 4;
    }

    if(!otype)
    {
        // Option not found in registry
        return aoclsparse_status_invalid_value;
    }

    switch(otype)
    {
    case 1: // integer
        iset = std::stoi(value);
        flag = opts.SetOption(name, iset, byuser);
        break;
    case 2: // real
        if constexpr(std::is_same_v<T, double> || std::is_same_v<T, double>
                     || std::is_same_v<T, aoclsparse_double_complex>)
        {
            rset = std::stod(value);
        }
        else
        {
            rset = std::stof(value);
        }

        flag = opts.SetOption(name, rset, byuser);
        break;
    case 3: // string
        sset = value;
        flag = opts.SetOption(name, sset, byuser);
        break;
    case 4: // boolean
        bset = (std::stoi(value)) != 0;
        flag = opts.SetOption(name, bset, byuser);
        break;
    default:
        return aoclsparse_status_internal_error;
        break;
    }

    switch(flag)
    {
    case 0:
        return aoclsparse_status_success;
        break;
    case 1: // option value is out-of-range // provide new status code?
    case 2: // option value is incorrect // provide new status code?
    case 3: // Option not found
        return aoclsparse_status_invalid_value;
        break;
    case 4: // Options are locked, cannot set
        return aoclsparse_status_invalid_value;
        break;
    default:
        return aoclsparse_status_internal_error;
        break;
    }
    return aoclsparse_status_internal_error; // never reached
}

template <typename T>
aoclsparse_status aoclsparse_csrmv(aoclsparse_operation       trans,
                                   const T                   *alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const T                   *csr_val,
                                   const aoclsparse_int      *csr_col_ind,
                                   const aoclsparse_int      *csr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const T                   *x,
                                   const T                   *beta,
                                   T                         *y);

#endif // ITSOL_FUNCTIONS_HPP
