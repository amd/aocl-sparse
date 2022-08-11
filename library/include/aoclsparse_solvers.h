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
/*! \file
 * \brief aoclsparse_solvers.h provides iterative sparse linear system solvers.
 * 
 * \details TODO chapter intro
 */
#ifndef AOCLSPARSE_SOLVERS_H_
#define AOCLSPARSE_SOLVERS_H_

#include "aoclsparse.h"

typedef struct _aoclsparse_itsol_handle* aoclsparse_itsol_handle;

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup solvers_module
 *  \brief Used by the iterative solver reverse communication interface \ref aoclsparse_itsol_d_rci_solve() 
 * to communicate to the user which operation is required. 
 */
typedef enum aoclsparse_itsol_rci_job_
{
    aoclsparse_rci_interrupt = -1,     /**< if set by the user, signals the solver to terminate. This is never set by the solver. */
    aoclsparse_rci_stop = 0,           /**< the solver finished. */
    aoclsparse_rci_start,              /**< initial value for \ref ircomm. */
    aoclsparse_rci_mv,                 /**< request for user to perform a matrix vector product before calling the solver again. */
    aoclsparse_rci_precond,            /**< request for user to perform a preconditioning step. */
    aoclsparse_rci_stopping_criterion, /**< monitoring step. Can be used to check for custom stopping criterion. No user action is required. */
} aoclsparse_itsol_rci_job;

/*! \ingroup solvers_module
 * \brief Print options.
 * 
 * \details
 * This function prints to the standard output a list of available options and their current value. For available option,
 * see Options in \ref aoclsparse_itsol_option_set.
 *
 * @param[in]
 * handle  the pointer to the iterative solvers' data structure.
 *
 * \retval TODO aoclsparse_status_success the operation completed successfully.
 * \retval TODO aoclsparse_status_memory_error internal memory allocation error.
 * \retval TODO aoclsparse_status_invalid_pointer the pointer to the problem handle was invalid.
 * \retval TODO aoclsparse_status_internal_error an unexpected error occured.
 */
DLL_PUBLIC
void aoclsparse_itsol_handle_prn_options(aoclsparse_itsol_handle handle);

/*! \ingroup solvers_module
 * \brief Option Setter.
 * 
 * \details
 * This function sets the a value to a given option. Options can be printed using \ref aoclsparse_itsol_handle_prn_options.
 * Available options are listed in \ref anchor_itsol_options.
 * @param[in]
 * handle  the pointer to the iterative solvers' data structure.
 * @param[in]
 * option  the name of the option to set.
 * @param[in]
 * value    the value to set the option to.
 *
 * \subsubsection anchor_itsol_options Options
 * The iterative solver framework as defined the following options. 
 * 
 * | **Option name** |  Type  | Default value|
 * |:----------|:---------:|--------:|
 * | **cg iteration limit** |  integer  | \f$i = 500\f$|
 * |Set CG iteration limit|||
 * | Valid values: \f$1 \le  i\f$.|||
 * | |||
 * | **cg rel tolerance** |  real  | \f$r = 1.08735e-06\f$|
 * |Set relative convergence tolerance for cg method|||
 * |Valid values: \f$0 \le  r\f$.|||
 * | |||
 * | **cg abs tolerance** |  real  | \f$r = 0\f$|
 * |Set absolute convergence tolerance for cg method|||
 * Valid values: \f$0 \le  r\f$.|||
 * | |||
 * | **iterative method** |  string  | s = `cg`|
 * |Choose solver to use|||
 * Valid values:   `cg`,  `gm res`,  `gmres`, and  `pcg`.|||
 * | |||
 * | **cg preconditioner** |  string  | s = `none`|
 * |Choose preconditioner to use with cg method|||
 * Valid values:   `gs`,  `none`,  `sgs`,  `symgs`, and  `user`.|||
 * 
 * \note It is worth noting that only some options apply to each specific
 * solver, e.g. options understood by \ref aoclsparse_itsol_d_solve are the ones which name begins with "cg".
 * 
 * \retval TODO aoclsparse_status_success the operation completed successfully.
 * \retval TODO aoclsparse_status_memory_error internal memory allocation error.
 * \retval TODO aoclsparse_status_invalid_pointer the pointer to the problem handle was invalid.
 * \retval TODO aoclsparse_status_internal_error an unexpected error occured.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_option_set(aoclsparse_itsol_handle& handle,
                                              const char*              option,
                                              const char*              value);

/*! \ingroup solvers_module
 * \brief Initializes the data structure \ref aoclsparse_itsol_handle for the suite of iterative solvers in the library.
 * 
 * \details
 * \ref aoclsparse_itsol_(s/d)_init intialize a structure referred to as \ref handle used by iterative solvers in the library.
 * These solvers share a common interface and are generally aimed at large sparse linear systems.\n 
 * The suite presents two separate interfaces to all the iterative solvers, a direct one, aoclsparse_itsol_d_rci_solve(), 
 * and a reverse communication one aoclsparse_itsol_s_rci_solve(). While the underlying algorithms are exactly the same, 
 * the difference lies in how data is communicated to the solvers: 
 * - The direct communication interface assumes the matrix is stored in the AOCL sparse matrix format which is passed directly to the solver.
 * - The reverse communication interface makes no assumption on the matrix storage. Thus when the solver needs some operation such as a a matrixvector product,
 * it stops, asks the user perform the operation and provide the results before continuing.  
 * 
 * The expected workflow is as follows:
 * - Call \ref aoclsparse_itsol_(s/d)_init to initialize \ref aoclsparse_itsol_handle.
 * - Choose the solver and adjust its behaviour by setting optional parameters with aoclsparse_itsol_option_set()
 * - if the reverse communication interface is desired, define the system's input with aoclsparse_itsol_d_rci_input()
 * - Solve the system with \ref aoclsparse_itsol_(s/d)(_rci)_solve
 * - Free the memory with \ref aoclsparse_itsolve_destroy()
 *
 * \note
 * \ref s or \ref d denote functions dedicated to single and double precision resectively. Once a working precision is chosen by calling the corresponding initialization function,
 * the other functions of the suite need to stay with the same working precision.
 * 
 * @param[inout]
 * handle  the pointer to the iterative solvers' data structure.
 *
 * \retval aoclsparse_status_success the operation completed successfully.
 * \retval aoclsparse_status_memory_error internal memory allocation error.
 * \retval aoclsparse_status_invalid_pointer the pointer to the problem handle was invalid.
 * \retval aoclsparse_status_internal_error an unexpected error occured.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_d_init(aoclsparse_itsol_handle* handle);

DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_s_init(aoclsparse_itsol_handle* handle);
/**@}*/

/*! \ingroup solvers_module
 * \brief Frees the memory from the iterative solvers \ref handle initialized by aoclsparse_itsol_d_init() and nullifies the pointer.
 *
 * \details Deallocating the memory is advisable to avoid memory leaks once the \ref handle is no longer useful.
 * Please note that passing a \ref handle that has not been initialized by aoclsparse_itsol_d_init() or aoclsparse_itsol_s_init()
 * may have unpredictable results.
 * 
 * @param[inout] handle  the pointer to the iterative solvers data tructure to deallocate
 */
DLL_PUBLIC
void aoclsparse_itsol_destroy(aoclsparse_itsol_handle* handle);

/*! \ingroup solvers_module
 * \brief Initialize the linear system data for the reverse communication iterative solvers.
 *
 * \details This function needs to be called before the reverse communication interface of the iterative solver is called 
 * to provide the system dimension \p n and its right hand side \p b . It is not needed if the direct communication interface
 * is called instead.
 * 
 * @param[inout] handle iterative solvers' data structure. needs to be initialized by \ref aoclsparse_itsol_(s/d)_init.
 * @param[in] n the number of column of the linear system matrix.
 * @param[in] b the right hand side of the linear system. Must be a vector of size n. 
 * 
 * \retval aoclsparse_status_success Initialization completed uccessfully.
 * \retval aoclsparse_status_invalid_pointer One of the pointers \p handle or \p b are invalid.
 * \retval aoclsparse_status_wrong_type \p handle was initialized with a different floating point precision than requested here.
 * \retval aoclsparse_status_invalid_value \p n was set to a negative value.
 * \retval aoclsparse_status_memory_error Internal memory allocation error.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status
    aoclsparse_itsol_d_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const double* b);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_itsol_s_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const float* b);
/**@}*/

/*! \ingroup solvers_module
 * \brief Reverse communication interface to the iterative solvers of the aoclsparse suite.
 *
 * \details TODO
 * 
 * @param [inout] handle iterative solvers' data structure. needs to be initialized by \ref aoclsparse_itsol_(s/d)_init.
 * @param [inout] ircomm pointer to the reverse communication instructions defined in ::aoclsparse_itsol_rci_job.
 * @param [inout] u pointer to a vector of data. The solver will typically point to the data on which the operation defined by \p ircomm 
 * needs to be applied.
 * @param [inout] v pointer to a vector of data. The solver will typically ask that the result of the operation defined by \p ircomm 
 * is stored in \p v .
 * @param [inout] x on input must contain that starting point \f x_0 \f of the linear system. \p x will contain the solution on output when the solver converges.
 * during intermediate stops or in case the solver ends prematurely, x will contain the best estimate of the solution.
 * @param [out] rinfo contains measures and statistics on the solver progress such as the norm of the residuals or the number of iterations so far. 
 * These can be used to monitor progress and define a custom stopping criterion when the solver stops with \p ircomm = ::aoclsparse_rci_stopping_criterion
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_d_rci_solve(aoclsparse_itsol_handle   handle,
                                               aoclsparse_itsol_rci_job* ircomm,
                                               double**                  u,
                                               double**                  v,
                                               double*                   x,
                                               double                    rinfo[100]);

DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_s_rci_solve(aoclsparse_itsol_handle   handle,
                                               aoclsparse_itsol_rci_job* ircomm,
                                               float**                   u,
                                               float**                   v,
                                               float*                    x,
                                               float                     rinfo[100]);
/**@}*/

/*! \ingroup solvers_module
 * \brief  TODO
 * 
 * \details TODO 
 */
/**@{*/
DLL_PUBLIC
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
    void*          udata);

DLL_PUBLIC
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
    void*                      udata);
/**@}*/

#ifdef __cplusplus
}
#endif

#endif // AOCLSPARSE_SOLVERS_H_
