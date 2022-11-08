/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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

#ifndef AOCLSPARSE_ITSOL_DATA_HPP_
#define AOCLSPARSE_ITSOL_DATA_HPP_

#include "aoclsparse.h"
#include "aoclsparse_itsol_options.hpp"

/*
 * Adding a new solver into ITSOL framework
 * ========================================
 *
 * Adding a new solver to ITSOL is done updating a few key parts of the
 * framework and adding specific functions inside aoclsparse_itsol_functions.hpp:
 * (e.g. if the new solver is gmres)
 *
 * 1. a solver data initializer: aoclsparse_gmres_data_init
 * 2. an RCI solver: aoclsparse_gmres_rci_solve,
 * 3. an Direct (or forward) solver: aoclsparse_gmres_solve, and
 * 4. a solver data cleanup function: aoclsparse_gmres_data_free.
 *
 * Updates required
 *
 * * In aoclsparse_itsol_data.hpp extend `itsol` with a new pointer *gmres
 *   to a bespoke structure containg gmres solver data.
 *
 * * In aoclsparse_itsol_data.hpp add new task enum: e.g. `gmres_rc_task` with
 *   the list of internal tasks to execute.
 *
 * * In aoclsparse_itsol_functions.cpp update the global initializers
 *   `aoclsparse_itsol_d_init` and `aoclsparse_itsol_s_init`.
 *
 * * In aoclsparse_itsol_functions.cpp deallocate solver data in
 *   `aoclsparse_itsol_data_free`, e.g.
 *        aoclsparse_gmres_data_free(itsol->gmres);
 *        itsol->gmres = nullptr;
 *
 * * In aoclsparse_itsol_functions.cpp for the RCI case, add calls to GMRES
 *   data structure initializer and GMRES RCI solver (two locations)
 *
 * * In aoclsparse_itsol_functions.cpp for the direct case, add calls to GMRES
 *   data structure initializer and GMRES direct solver (two locations)
 *
 * * In aoclsparse_list_options.hpp, add new options (on the very least)
 *   1. "gmres abs tolerance" (real) with default value scaled machine epsilon
 *   2. "gmres iteration limit" (int)
 *   3. "gmres rci stop" (string)
 *   4. "gmres precondtioner" (string)
 *
 * * Update option "iterative solver" and add an entry for you new solver.
 *   Solvers are selected using option "iterative solver" with "CG" or "GMRES".
 *
 * * Load options into the new solver data writing an option loader (e.g. aoclsparse_cg_data_options),
 *        and call the loader just after calling aoclsparse_cg_data_init. See aoclsparse_cg_data_options
 *        for further details.
 *
 * * Solver info and stats, that is all printable and user-useful information
 *   must be stored in rinfo[] array, e.g. last residual 2-norm,
 *   iteration number, etc. Each entry has an alloted position define by RINFO_* and are specified in
 *   aosparse_itsol_functions.hpp. Any new entries should be defined there.
 */

enum cg_rc_task
{
    task_start = 0,
    task_init_res,
    task_check_conv,
    task_start_iter,
    task_compute_beta,
    task_take_step
};

enum gmres_rc_task
{
    task_gmres_start = 0,
    task_gmres_init_res,
    task_gmres_init_precond,
    task_gmres_start_iter,
    task_gmres_end_iter,
    task_gmres_x_update,
    task_gmres_restart_cycle,
    task_gmres_convergence_check
};
/*
 * Strucure holding the working memory for the CG algorithm
*/
template <typename T>
struct cg_data
{
    /* Working vectors and values */
    T *r, *z, *p, *q;
    T  alpha, rz, beta, rnorm2, bnorm2, brtol;
    /* CG algorithm state */
    cg_rc_task     task;
    aoclsparse_int niter;

    /* CG algorithm options */
    T              rtol, atol;
    aoclsparse_int maxit, precond;
};

/*
 * Structure holding the working memory for the GMRES algorithm
*/
template <typename T>
struct gmres_data
{
    /* Working vectors and values */
    T *v = 0, *h = 0, *g = 0;
    T *c = 0, *s = 0, *z = 0;
    T  rnorm2, bnorm2, brtol;
    /* GMRES algorithm state */
    gmres_rc_task  task;
    aoclsparse_int niter;
    aoclsparse_int j;
    /* Restart iteration count for the GMRES problem: Needed for working buffer allocation */
    aoclsparse_int restart_iters;

    /* GMRES algorithm options */
    T              rtol, atol;
    aoclsparse_int maxit, precond;
};

enum aoclsparse_itsol_solver
{
    solver_cg = 1,
    solver_gmres,
};

template <typename T>
struct aoclsparse_itsol_data
{
    /* dimension of the problem */
    aoclsparse_int n;
    /* right hand side*/
    T *b;
    /* option settings for all solvers */
    aoclsparse_options::OptionRegistry<T> opts;

    /* flag to indicate that we are solving a system right now*/
    bool solving;

    /* which solver is being used */
    aoclsparse_int solver;

    /* Conjugate Gradient Method (CGM) solver's data */
    cg_data<T> *cg;
    /* GMRES solver's data */
    gmres_data<T> *gmres;
};

struct _aoclsparse_itsol_handle
{
    /* data type, only double and float are supported */
    aoclsparse_matrix_data_type type;
    /* Pointer to the templated itsolve_data, only the one matching type
       will be used. */
    aoclsparse_itsol_data<float>  *itsol_s;
    aoclsparse_itsol_data<double> *itsol_d;
};

#endif
