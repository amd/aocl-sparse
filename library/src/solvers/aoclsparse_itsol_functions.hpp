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
#ifndef AOCLSPARSE_ITSOL_FUNCTIONS_HPP_
#define AOCLSPARSE_ITSOL_FUNCTIONS_HPP_

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_itsol_data.hpp"
#include "aoclsparse_itsol_list_options.hpp"
#include "aoclsparse_itsol_options.hpp"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_mv.hpp"
#include "aoclsparse_solvers.h"
#include "aoclsparse_trsv.hpp"
#include "aoclsparse_descr.h"
#include <cmath>
#include <iostream>

// Define all entries for rinfo[] array
// All solvers using rinfo need to stick to these entries
#define RINFO_RES_NORM 0 // Residual 2-norm
#define RINFO_RHS_NORM 1 // 2-norm of the right hand side
#define RINFO_ITER 30 // Number of iterations

template <typename T>
aoclsparse_status aoclsparse_cg_data_init(const aoclsparse_int n, cg_data<T>** cg)
{
    /* Initializes Internal CG data including 4 vectors of working memory
     * Possible exits:
     * - Allocation
     */
    if(cg == nullptr)
        return aoclsparse_status_internal_error;

    *cg = (cg_data<T>*)malloc(sizeof(cg_data<T>));
    if(*cg == nullptr)
        return aoclsparse_status_memory_error;

    // allocate +1 size to avoid special case when n=0 and malloc might return NULL
    (*cg)->p = (*cg)->q = (*cg)->r = (*cg)->z = nullptr;
    (*cg)->p                                  = (T*)malloc((n + 1) * sizeof(T));
    (*cg)->q                                  = (T*)malloc((n + 1) * sizeof(T));
    (*cg)->r                                  = (T*)malloc((n + 1) * sizeof(T));
    (*cg)->z                                  = (T*)malloc((n + 1) * sizeof(T));
    (*cg)->task                               = task_start;
    (*cg)->niter                              = 0;
    if((*cg)->p == nullptr || (*cg)->q == nullptr || (*cg)->r == nullptr || (*cg)->z == nullptr)
    {
        aoclsparse_cg_data_free(*cg);
        *cg = nullptr;
        return aoclsparse_status_memory_error;
    }

    return aoclsparse_status_success;
}

template <typename T>
void aoclsparse_cg_data_free(cg_data<T>* cg)
{
    if(!cg)
        // Nothing to do
        return;
    if(cg->p)
        free(cg->p);
    if(cg->q)
        free(cg->q);
    if(cg->r)
        free(cg->r);
    if(cg->z)
        free(cg->z);
    free(cg);
}

template <typename T>
aoclsparse_status aoclsparse_cg_data_options(cg_data<T>*                            cg,
                                             aoclsparse_options::OptionRegistry<T>& opts)
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

/* aoclsparse_itsol_data_free - deallocate/reset aoclsparse_itsol_data,
 * this is used as final deallocation or when new fight hand side comes
 * in (potentially of a different size) and we need to clear all solver data.
 * In that case use keep_itsol=true. */
template <typename T>
void aoclsparse_itsol_data_free(aoclsparse_itsol_data<T>* itsol, bool keep_itsol = false)
{
    if(itsol)
    {

        itsol->n = 0;
        if(itsol->b)
            free(itsol->b);
        itsol->b       = nullptr;
        itsol->solving = false;
        itsol->solver  = 0;

        // deallocate all solver specific data
        aoclsparse_cg_data_free(itsol->cg);
        itsol->cg = nullptr;
        // TODO GMRES data deallocation

        if(!keep_itsol)
            delete itsol;
    }
}

/* Initialize itsol_data structure, including the option settings */
template <typename T>
aoclsparse_status aoclsparse_itsol_data_init(aoclsparse_itsol_data<T>** itsol)
{
    if(itsol == nullptr)
        return aoclsparse_status_internal_error;

    try
    {
        *itsol = new aoclsparse_itsol_data<T>;
    }
    catch(std::bad_alloc&)
    {
        return aoclsparse_status_memory_error;
    }

    (*itsol)->n       = 0;
    (*itsol)->b       = nullptr;
    (*itsol)->solving = false;
    (*itsol)->solver  = 0;
    (*itsol)->cg      = nullptr;
    //TODO (*itsol)->gmres = nullptr;

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
    aoclsparse_itsol_rci_input(aoclsparse_itsol_data<T>* itsol, aoclsparse_int n, const T* b)
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

    // allocate +1 size to avoid special case when n=0 and malloc might return NULL
    itsol->b = (T*)malloc((n + 1) * sizeof(T));
    if(!itsol->b)
        return aoclsparse_status_memory_error;
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
aoclsparse_status aoclsparse_itsol_solver_init(aoclsparse_itsol_data<T>* itsol)
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
        // TODO add initialization og GMRES data structure
        return aoclsparse_status_not_implemented;
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
    aoclsparse_itsol_symgs(aoclsparse_matrix A, aoclsparse_mat_descr descr, T* r, T* y, T* z)
{
    // For internal use in CG.
    // Add additional input checks if exposing it to the user
    _aoclsparse_mat_descr descr_cpy;
    T                     alpha      = 1.0;
    aoclsparse_int        avxversion, i;
    aoclsparse_status     status;
    T*                    aval = static_cast<T*>(A->opt_csr_mat.csr_val);
    aoclsparse_operation  trans;

    if(descr->type != aoclsparse_matrix_type_general
       && descr->type != aoclsparse_matrix_type_symmetric)
        return aoclsparse_status_invalid_value;
    aoclsparse_copy_mat_descr(&descr_cpy, descr);
    aoclsparse_set_mat_type(&descr_cpy, aoclsparse_matrix_type_triangular);

    // triangle solve avx is not yet implemented for single precision.
    // TODO remove when switches to different avx implementations are written.
    if (A->val_type == aoclsparse_dmat)
        avxversion = 1;
    else
        avxversion = 0;

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
    status = aoclsparse_trsv(trans, alpha, A, &descr_cpy, r, y, avxversion);
    if(status != aoclsparse_status_success)
        return status;

    // y := Dy
    if(descr->diag_type == aoclsparse_diag_type_non_unit)
    {
        for(i = 0; i < A->m; i++)
            y[i] *= aval[A->idiag[i]];
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
    status = aoclsparse_trsv(trans, alpha, A, &descr_cpy, y, z, avxversion);
    if(status != aoclsparse_status_success)
        return status;

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_itsol_rci_solve(aoclsparse_itsol_data<T>* itsol,
                                             aoclsparse_itsol_rci_job* ircomm,
                                             T**                       u,
                                             T**                       v,
                                             T*                        x,
                                             T                         rinfo[100])
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
        // TODO add call to GMRES here
        status = aoclsparse_status_not_implemented;
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
    aoclsparse_itsol_data<T>*  itsol,
    aoclsparse_int             n,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    const T*                   b,
    T*                         x,
    T                          rinfo[100],
    aoclsparse_int             precond(aoclsparse_int flag, const T* u, T* v, void* udata),
    aoclsparse_int             monit(const T* x, const T* r, T rinfo[100], void* udata),
    void*                      udata)
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

    if(!mat->opt_csr_ready)
    {
        // CG needs opt_csr to run
        status = aoclsparse_csr_optimize<T>(mat);
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
        // TODO add call to GMRES here
        status = aoclsparse_status_not_implemented;
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
aoclsparse_status aoclsparse_cg_rci_solve(aoclsparse_itsol_data<T>* itsol,
                                          aoclsparse_itsol_rci_job* ircomm,
                                          T**                       u,
                                          T**                       v,
                                          T*                        x,
                                          T                         rinfo[100])
{
    // all pointers were already checked by the caller --> safe to use them

    aoclsparse_status exit_status = aoclsparse_status_success, status;
    aoclsparse_int    i, n;
    cg_data<T>*       cg;
    T                 pq, rz_new;
    bool              loop;

    cg = itsol->cg;
    n  = itsol->n;

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
            cg->bnorm2 = 0.0;
            for(i = 0; i < n; i++)
            {
                cg->r[i] = -itsol->b[i];
                cg->bnorm2 += itsol->b[i] * itsol->b[i];
                cg->p[i] = x[i];
            }
            cg->bnorm2 = sqrt(cg->bnorm2);
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
            cg->rnorm2 = 0.0;
            for(i = 0; i < n; i++)
            {
                cg->r[i] += cg->q[i];
                cg->rnorm2 += cg->r[i] * cg->r[i];
            }
            cg->rnorm2 = sqrt(cg->rnorm2);
            //bli_dnormfv(handle->n, cgd->r, 1, &cgd->rnorm2);
            if(cg->rnorm2 != cg->rnorm2) // test for NaN
            {
                exit_status = aoclsparse_status_numerical_error;
                break;
            }
            rinfo[RINFO_RES_NORM] = cg->rnorm2;

            // initialize p = 0 (so that initil p=-z from p:=-z + beta*p)
            for(i = 0; i < n; i++)
                cg->p[i] = 0.;
            // rz doesn't matter in the first iteration as long as it is not 0
            cg->rz = 1.;

            // continue without break to check convergence
            cg->task = task_check_conv;

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
            rinfo[RINFO_ITER] = (T)cg->niter;

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

        case task_compute_beta:
            // Compute beta and new search direction p
            // Have: Mz = r
            // Compute beta = r^t.z / rz
            //            p = beta*p - z
            // request the matrix vector product for A*p

            rz_new = 0.;
            for(i = 0; i < n; i++)
                rz_new += cg->r[i] * cg->z[i];
            //bli_ddotv(
            //    BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, handle->n, cgd->r, 1, cgd->z, 1, &cgd->beta);
            if(cg->rz <= 0.) // preconditioner M is not positive definite
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
            pq = 0.;
            for(i = 0; i < n; i++)
                pq += cg->p[i] * cg->q[i];
            //bli_ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, handle->n, cgd->p, 1, cgd->q, 1, &pq);
            if(pq <= 0.) // system matrix A is not positive definite
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
            cg->rnorm2 = 0.0;
            for(i = 0; i < n; i++)
                cg->rnorm2 += cg->r[i] * cg->r[i];
            cg->rnorm2 = sqrt(cg->rnorm2);
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

template <typename T>
aoclsparse_status
    aoclsparse_cg_solve(aoclsparse_itsol_data<T>*  itsol,
                        aoclsparse_matrix          mat,
                        const aoclsparse_mat_descr descr,
                        T*                         x,
                        T                          rinfo[100],
                        aoclsparse_int precond(aoclsparse_int flag, const T* u, T* v, void* udata),
                        aoclsparse_int monit(const T* x, const T* r, T rinfo[100], void* udata),
                        void*          udata)
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
    aoclsparse_int           nnz, flag;
    aoclsparse_int           n      = itsol->n;
    aoclsparse_itsol_rci_job ircomm = aoclsparse_rci_start;
    T*                       u      = nullptr;
    T*                       v      = nullptr;
    T                        alpha = 1.0, beta = 0., timing = 0.;
    T*                       y;
    aoclsparse_operation     trans       = aoclsparse_operation_none;
    aoclsparse_status        exit_status = aoclsparse_status_success;
    aoclsparse_status        status;

    if(mat->m != n || mat->n != n)
        return aoclsparse_status_invalid_size;
    if(descr->type != aoclsparse_matrix_type_symmetric)
        return aoclsparse_status_invalid_value;
    if(descr->fill_mode != aoclsparse_fill_mode_lower)
        // symmetric matrix-vector product only work for lower triangular matrices...
        return aoclsparse_status_invalid_value;

    nnz = mat->nnz;

    if(itsol->cg->precond == 1 && precond == nullptr)
        return aoclsparse_status_invalid_pointer;
    if((itsol->cg)->precond == 3) // Add other preconds here...
    {
        // Symmetric Gauss-Seidel requested, allocate some memory
        if(!mat->opt_csr_full_diag && !descr->diag_type == aoclsparse_diag_type_unit)
            // Gauss-Seidel needs a full diagonal to perform the triangle solve
            return aoclsparse_status_invalid_value;
        y = (T*)malloc(n * sizeof(T));
        if(y == nullptr)
            return aoclsparse_status_memory_error;
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = 0.0;
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
            beta   = 0.0;
            alpha  = 1.0;
            status = aoclsparse_mv(aoclsparse_operation_none, alpha, mat, descr, u, beta, v);
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
                flag = precond(0, u, v, udata);
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
                status = aoclsparse_itsol_symgs(mat, descr, u, y, v);
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
                flag = monit(u, v, rinfo, udata);
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
aoclsparse_status handle_parse_option(aoclsparse_options::OptionRegistry<T>& opts,
                                      const char*                            option,
                                      const char*                            value)
{
    const aoclsparse_int byuser = 1;
    aoclsparse_int       flag;
    aoclsparse_int       iquery, iset;
    aoclsparse_int       otype = 0;
    T                    rquery, rset;
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
        rset = std::stod(value); // TODO FIX float / double
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
                                   const T*                   alpha,
                                   aoclsparse_int             m,
                                   aoclsparse_int             n,
                                   aoclsparse_int             nnz,
                                   const T*                   csr_val,
                                   const aoclsparse_int*      csr_col_ind,
                                   const aoclsparse_int*      csr_row_ptr,
                                   const aoclsparse_mat_descr descr,
                                   const T*                   x,
                                   const T*                   beta,
                                   T*                         y);

#endif // ITSOL_FUNCTIONS_HPP
