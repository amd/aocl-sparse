/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. Portions of this
 * file consist of AI-generated content.
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
 * \brief  Iterative sparse linear system solvers.
 *
 * \details
 * \section anchor_head Iterative Solver Suite (itsol)
 *
 * \subsection anchor_intro Introduction
 *
 * AOCL Sparse Iterative Solver Suite (itsol) is an iterative framework for solving large-scale sparse linear systems of equations of the form
 * \f[ Ax=b,\f]
 * where \f$A\f$ is a sparse full-rank square matrix of size \f$n\f$ by \f$n\f$, \f$b\f$ is a dense \f$n\f$-vector, and \f$x\f$ is the vector of unknowns also of size \f$n\f$.
 * The framework solves the previous problem using either the Conjugate Gradient method or GMRES. It supports a variety of preconditioners (_accelerators_) such as
 * Symmetric Gauss-Seidel or Incomplete LU factorization, ILU(0).
 *
 * Iterative solvers at each step (iteration) find a better approximation to the solution of the linear system of equations in the sense that it reduces an error metric.
 * In contrast, direct solvers only provide a solution once the full algorithm as been executed. A great advantage of iterative solvers is that they can be
 * interrupted once an approximate solution is deemed acceptable.
 *
 * \subsection anchor_FRCI Forward and Reverse Communication Interfaces
 *
 * The suite presents two separate interfaces to all the iterative solvers, a direct one, \ref aoclsparse_itsol_d_rci_solve (\ref aoclsparse_itsol_s_rci_solve),
 * and a reverse communication (RCI) one \ref aoclsparse_itsol_d_rci_solve ( \ref aoclsparse_itsol_s_rci_solve). While the underlying algorithms are exactly the same,
 * the difference lies in how data is communicated to the solvers.
 *
 * The direct communication interface expects to have explicit access to the coefficient matrix \f$A\f$. On the other hand, the reverse communication interface makes
 * no assumption on the matrix storage. Thus when the solver requires some matrix operation such as a
 * matrix-vector product, it returns control to the user and asks the user perform the operation and provide the results by calling again the RCI solver.
 *
 * \subsection workflow Recommended Workflow
 * For solving a linear system of equations, the following workflow is recommended:
 * - Call \ref aoclsparse_itsol_s_init or \ref aoclsparse_itsol_d_init to initialize a handle of type \ref aoclsparse_itsol_handle.
 * - Choose the solver and adjust its behaviour by setting optional parameters with \ref aoclsparse_itsol_option_set, see also \ref anchor_itsol_options.
 * - If the reverse communication interface is desired, define the system's input with \ref aoclsparse_itsol_d_rci_input.
 * - Solve the system with either using direct interface \ref aoclsparse_itsol_s_solve (or \ref aoclsparse_itsol_d_solve) or
 *   reverse communication interface \ref aoclsparse_itsol_s_rci_solve (or \ref aoclsparse_itsol_d_rci_solve)
 * - Free the memory with \ref aoclsparse_itsol_destroy.
 *
 * \subsection anchor_rinfo Information Array
 * The array \c rinfo[100] is used by the solvers (e.g. \ref aoclsparse_itsol_d_solve or \ref aoclsparse_itsol_s_rci_solve) to report
 * back useful convergence metrics and other solver statistics.
 * The user callback \c monit is also equipped with this array and can be used
 * to view or monitor the state of the solver.
 * The solver will populate the following entries with the most recent iteration data
 * | Index | Description                                                            |
 * |------:|:-----------------------------------------------------------------------|
 * |     0 | Absolute residual norm, \f$ r_{\mbox{abs}} = \| Ax-b\|_2 \f$.          |
 * |     1 | Norm of the right-hand side vector \f$b\f$, \f$\|b\|_2\f$.
 * |  2-29 | Reserved for future use.                                               |
 * |    30 | Iteration counter.                                                     |
 * | 31-99 | Reserved for future use.                                               |
 *
 * \subsection examples Examples
 * Each iterative solver in the suite is provided with an illustrative example on its usage. The source file for the examples can be found under the
 * \c tests/examples/ folder.
 * | Solver | Precision | Filename | Description |
 * |:-------|:---------:|:---------|:------------|
 * | itsol forward communication interface | double | \c sample_itsol_d_cg.cpp | Solves a linear system of equations using the Conjugate Gradient method. |
 * || single | \c sample_itsol_s_cg.cpp ||
 * | itsol reverse communication interface | double | \c sample_itsol_d_cg_rci.cpp | Solves a linear system of equations using the Conjugate Gradient method. |
 * || single | \c sample_itsol_s_cg_rci.cpp  ||
 * \subsection ref References
 * -# Yousef Saad, _Iterative Methods for Sparse Linear Systems_. 2nd ed. 2003. pp xxi + 547.
 * -# Conjugate gradients, method of. Encyclopedia of Mathematics. URL: <a href="https://encyclopediaofmath.org/index.php?title=Conjugate_gradients,_method_of&oldid=46470">Conjugate Gradients method</a>.
 * -# Acceleration methods. Encyclopedia of Mathematics. URL: <a href="https://encyclopediaofmath.org/index.php?title=Acceleration_methods&oldid=52131">Acceleration methods</a>.
 */
#ifndef AOCLSPARSE_SOLVERS_H_
#define AOCLSPARSE_SOLVERS_H_

#include "aoclsparse.h"
/**
 * @brief Optimization handle
 *
 * @details
 * This type of handle is a container box for storing problem data and optional parameter values.
 * it must be initialized using \ref aoclsparse_itsol_s_init, and should be destroyed after using it with \ref aoclsparse_itsol_destroy.
 * For double precision data types use \ref aoclsparse_itsol_d_init.
 *
 * For more details, refer to Solver chapter introduction \ref anchor_head.
 *
 */
typedef struct _aoclsparse_itsol_handle *aoclsparse_itsol_handle;

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup solvers_module
 *  \brief Values of \p ircomm used by the iterative solver reverse communication interface (RCI) \ref aoclsparse_itsol_d_rci_solve and
 * \ref aoclsparse_itsol_s_rci_solve to communicate back to the user which operation is required.
 */
typedef enum aoclsparse_itsol_rci_job_
{
    // clang-format off
    aoclsparse_rci_interrupt = -1,     ///< if set by the user, signals the solver to terminate.
                                       ///< This is never set by the solver. Terminate.
    aoclsparse_rci_stop = 0,           ///< found a solution within specified tolerance (see options "cg rel tolerance",
                                       ///< "cg abs tolerance", "gmres rel tolerance", and "gmres abs tolerance" in
                                       ///< \ref anchor_itsol_options). Terminate, vector \p x contains the solution.
    aoclsparse_rci_start,              ///< initial value of the \p ircomm flag, no action required. Call solver.
    aoclsparse_rci_mv,                 ///< perform the matrix-vector product \f$ v = Au\f$. Return control to solver.
    aoclsparse_rci_precond,            ///< perform a preconditioning step on the vector \f$u\f$ and store in \f$v\f$.
                                       ///< If the preconditioner \f$M\f$ has explicit matrix form, then applying the
                                       ///< preconditioner would result in the operations \f$ v=Mu \f$ or
                                       ///< \f$v=M^{-1}u\f$. The latter would be performed by solving the linear system
                                       ///< of equations \f$Mv=u\f$. Return control to solver.
    aoclsparse_rci_stopping_criterion, ///< perform a monitoring step and check for custom stopping criteria. If using a
                                       ///< positive tolerance value for the convergence options (see
                                       ///< \ref aoclsparse_rci_stop), then this step can be ignored and control can be
                                       ///< returned to solver.
    // clang-format on
} aoclsparse_itsol_rci_job;

/*! \ingroup solvers_module
 * \brief Print options stored in a problem handle.
 *
 * \details
 * This function prints to the standard output a list of available options stored in a problem handle
 * and their current value. For available options, see Options in \ref aoclsparse_itsol_option_set.
 *
 * @param[in]
 * handle pointer to the iterative solvers' data structure.
 */
DLL_PUBLIC
void aoclsparse_itsol_handle_prn_options(aoclsparse_itsol_handle handle);

/*! \ingroup solvers_module
 * \brief Option Setter.
 *
 * \details
 * This function sets the value to a given option inside the provided problem handle.
 * Handle options can be printed using \ref aoclsparse_itsol_handle_prn_options.
 * Available options are listed in \ref anchor_itsol_options.
 * @param[inout]
 * handle  pointer to the iterative solvers' data structure.
 * @param[in]
 * option  string specifying the name of the option to set.
 * @param[in]
 * value   string providing the value to set the option to.
 *
 * \section anchor_itsol_options Options
 * The iterative solver framework has the following options.
 *
 * @rst
 * .. csv-table::
 *    :header: "Option name", "Type", "Default", "Description", "Constraints"
 *
 *    "**cg iteration limit**", "integer", ":math:`i = 500`", "Set CG iteration limit", ":math:`1 \le i`."
 *    "**gmres iteration limit**", "integer", ":math:`i = 150`", "Set GMRES iteration limit", ":math:`1 \le i`."
 *    "**gmres restart iterations**", "integer", ":math:`i = 20`", "Set GMRES restart iterations", ":math:`1 \le i`."
 *    "**cg rel tolerance**", "real", ":math:`r = 1.08735e-06`", "Set relative convergence tolerance for cg method", ":math:`0 \le r`."
 *    "**cg abs tolerance**", "real", ":math:`r = 0`", "Set absolute convergence tolerance for cg method", ":math:`0 \le r`."
 *    "**gmres rel tolerance**", "real", ":math:`r = 1.08735e-06`", "Set relative convergence tolerance for gmres method", ":math:`0 \le r`."
 *    "**gmres abs tolerance**", "real", ":math:`r = 1e-06`", "Set absolute convergence tolerance for gmres method", ":math:`0 \le r`."
 *    "**iterative method**", "string", ":math:`s = cg`", "Choose solver to use", ":math:`s =` `cg`, `gm res`, `gmres`, or `pcg`."
 *    "**cg preconditioner**", "string", ":math:`s = none`", "Choose preconditioner to use with cg method", ":math:`s =` `gs`, `none`, `sgs`, `symgs`, or `user`."
 *    "**gmres preconditioner**", "string", ":math:`s = none`", "Choose preconditioner to use with gmres method", ":math:`s =` `ilu0`, `none`, or `user`."
 *
 * @endrst
 *
 * \note It is worth noting that only some options apply to each specific
 * solver, e.g. name of options that begin with "cg" affect the behaviour of the CG solver.
 *
 * \retval aoclsparse_status_success the operation completed successfully.
 * \retval aoclsparse_status_invalid_value either the option name was not found or the provided option
 *         value is out of the valid range.
 * \retval aoclsparse_status_invalid_pointer the pointer to the problem handle is invalid.
 * \retval aoclsparse_status_internal_error an unexpected error occurred.
 */
DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_option_set(aoclsparse_itsol_handle handle,
                                              const char             *option,
                                              const char             *value);

/*! \ingroup solvers_module
 * \brief Initialize a problem \p handle ( \c aoclsparse_itsol_handle) for the iterative solvers suite of the library.
 *
 * \details
 * \ref aoclsparse_itsol_s_init and aoclsparse_itsol_d_init initialize a data structure referred to as
 * problem \p handle. This \p handle is used by iterative solvers (itsol) suite to setup options, define which
 * solver to use, etc.
 *
 * @param[inout] handle the pointer to the problem handle data structure.
 *
 * \retval aoclsparse_status_success the operation completed successfully.
 * \retval aoclsparse_status_memory_error internal memory allocation error.
 * \retval aoclsparse_status_invalid_pointer the pointer to the problem handle is invalid.
 * \retval aoclsparse_status_internal_error an unexpected error occurred.
 *
 * \note Once the \p handle is no longer needed, it can be destroyed and the memory released by calling
 * \ref aoclsparse_itsol_destroy.
 *@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_d_init(aoclsparse_itsol_handle *handle);

DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_s_init(aoclsparse_itsol_handle *handle);
/**@}*/

/*! \ingroup solvers_module
 * \brief Free the memory reserved in a problem \p handle previously initialized by \ref aoclsparse_itsol_s_init or
 * \ref aoclsparse_itsol_d_init.
 *
 * \details Once the problem handle is no longer needed, calling this function to deallocate the memory is advisable
 * to avoid memory leaks.
 *
 * \note Passing a \p handle that has not been initialized by \ref aoclsparse_itsol_s_init or \ref aoclsparse_itsol_d_init
 * may have unpredictable results.
 *
 * @param[inout] handle pointer to a problem handle.
 */
DLL_PUBLIC
void aoclsparse_itsol_destroy(aoclsparse_itsol_handle *handle);

/*! \ingroup solvers_module
 * \brief Store partial data of the linear system of equations into the problem \p handle.
 *
 * \details This function needs to be called before the reverse communication interface iterative solver is called.
 * It registers the linear system's dimension \p n, and stores the right-hand side vector \p b.
 *
 * \note
 * This function does not need to be called if the forward communication interface is used.
 *
 * @param[inout] handle problem \p handle. Needs to be initialized by calling \ref aoclsparse_itsol_s_init or
 *                      \ref aoclsparse_itsol_d_init.
 * @param[in] n the number of columns of the (square) linear system matrix.
 * @param[in] b the right hand side of the linear system. Must be a vector of size \p n.
 *
 * \retval aoclsparse_status_success initialization completed successfully.
 * \retval aoclsparse_status_invalid_pointer one or more of the pointers \p handle, and \p b are invalid.
 * \retval aoclsparse_status_wrong_type \p handle was initialized with a different floating point precision than requested here, e.g.
 *         \ref aoclsparse_itsol_d_init (double precision)
 *         was used to initialize \p handle but \ref aoclsparse_itsol_s_rci_input (single precision) is being called instead of the correct double
 *         precision one, \ref aoclsparse_itsol_d_rci_input.
 * \retval aoclsparse_status_invalid_value \p n was set to a negative value.
 * \retval aoclsparse_status_memory_error internal memory allocation error.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status
    aoclsparse_itsol_d_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const double *b);

DLL_PUBLIC
aoclsparse_status
    aoclsparse_itsol_s_rci_input(aoclsparse_itsol_handle handle, aoclsparse_int n, const float *b);
/**@}*/

/*! \ingroup solvers_module
 * \brief Reverse Communication Interface (RCI) to the iterative solvers (itsol) suite.
 *
 * \details This function solves the linear system of equations
 * \f[ Ax=b, \f]
 * where the matrix of coefficients \f$A\f$ is not required to be provided explicitly. The right hand-side is the dense vector \p b and
 * the vector of unknowns is \p x. If \f$A\f$ is symmetric and positive definite then set the option "iterative method" to "cg"
 * to solve the problem using the <a href="https://encyclopediaofmath.org/index.php?title=Conjugate_gradients,_method_of&oldid=46470">Conjugate Gradient
 * method</a>, alternatively set the option to "gmres" to solve
 * using <a href="https://mathworld.wolfram.com/GeneralizedMinimalResidualMethod.html">GMRes</a>. See the \ref anchor_itsol_options
 * for a list of available options to modify the behaviour of each solver.
 *
 * The reverse communication interface (RCI), also know as _matrix-free_ interface does not require the user to explicitly provide the matrix \f$A\f$.
 * During the solve process whenever the algorithm
 * requires a matrix operation (matrix-vector or transposed matrix-vector products), it returns control to the user with a flag \p ircomm indicating what
 * operation is requested. Once the user performs the requested task it must call this function again to resume the solve.
 *
 * The expected workflow is as follows:
 * -# Call \ref aoclsparse_itsol_s_init or \ref aoclsparse_itsol_d_init to initialize the problem \p handle ( \ref aoclsparse_itsol_handle)
 * -# Choose the solver and adjust its behaviour by setting optional parameters with \ref aoclsparse_itsol_option_set, see also \ref anchor_itsol_options.
 * -# Define the problem size and right-hand side vector \f$b\f$ with \ref aoclsparse_itsol_d_rci_input.
 * -# Solve the system with either \ref aoclsparse_itsol_s_rci_solve or \ref aoclsparse_itsol_d_rci_solve.
 * -# If there is another linear system of equations to solve with the same matrix but a different right-hand side \f$b\f$, then repeat from step 3.
 * -# If solver terminated successfully then vector \p x contains the solution.
 * -# Free the memory with \ref aoclsparse_itsol_destroy.
 *
 * These reverse communication interfaces complement the _forward communication_ interfaces \ref aoclsparse_itsol_d_rci_solve and
 * \ref aoclsparse_itsol_s_rci_solve.
 *
 * @param [inout] handle problem \p handle. Needs to be previously initialized by \ref aoclsparse_itsol_s_init or
 *                \ref aoclsparse_itsol_d_init and then populated using either \ref aoclsparse_itsol_s_rci_input or \ref aoclsparse_itsol_d_rci_input, as appropriate.
 * @param [inout] ircomm pointer to the reverse communication instruction flag and defined in \ref aoclsparse_itsol_rci_job.
 * @param [inout] u pointer to a generic vector of data. The solver will point to the data on which the operation defined by \p ircomm
 * needs to be applied.
 * @param [inout] v pointer to a generic vector of data. The solver will ask that the result of the operation defined by \p ircomm
 * be stored in \p v.
 * @param [inout] x dense vector of unknowns. On input, it should contain the initial guess from which to start the iterative
 *                process. If there is no good initial estimate guess then any arbitrary but finite
 *                values can be used. On output, it contains an estimate to the solution of the linear system of equations up
 *                to the requested tolerance, e.g. see "cg rel tolerance" or "cg abs tolerance" in \ref anchor_itsol_options.
 * @param [out] rinfo vector containing information and stats related to the iterative solve, see
 *                \ref anchor_rinfo. This parameter can be used to monitor progress and define a custom stopping criterion when
 *               the solver returns control to user with \p ircomm = \ref aoclsparse_rci_stopping_criterion.
 *
 * \note
 * This function returns control back to the user under certain circumstances. The table in \ref aoclsparse_itsol_rci_job
 * indicates what actions are required to be performed by the user.
 *
 * @rst
 * .. collapse:: Example - CG / floating point double precision (tests/examples/sample_itsol_d_cg_rci.cpp)
 *
 *    .. only:: html
 *
 *        .. literalinclude:: ../tests/examples/sample_itsol_d_cg_rci.cpp
 *            :language: C++
 *            :linenos:
 *
 * \
 *
 * .. collapse:: Example - GMRES / floating point double precision (tests/examples/sample_itsol_d_gmres.cpp)
 *
 *    .. only:: html
 *
 *        .. literalinclude:: ../tests/examples/sample_itsol_d_gmres.cpp
 *            :language: C++
 *            :linenos:
 *
 * \
 *
 * .. collapse:: Example - CG floating point single precision (tests/examples/sample_itsol_s_cg_rci.cpp)
 *
 *    .. only:: html
 *
 *        .. literalinclude:: ../tests/examples/sample_itsol_s_cg_rci.cpp
 *            :language: C++
 *            :linenos:
 *
 * \
 *
 * .. collapse:: Example - GMRES / floating point single precision (tests/examples/sample_itsol_s_gmres.cpp)
 *
 *    .. only:: html
 *
 *        .. literalinclude:: ../tests/examples/sample_itsol_s_gmres.cpp
 *            :language: C++
 *            :linenos:
 *
 * @endrst
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_d_rci_solve(aoclsparse_itsol_handle   handle,
                                               aoclsparse_itsol_rci_job *ircomm,
                                               double                  **u,
                                               double                  **v,
                                               double                   *x,
                                               double                    rinfo[100]);

DLL_PUBLIC
aoclsparse_status aoclsparse_itsol_s_rci_solve(aoclsparse_itsol_handle   handle,
                                               aoclsparse_itsol_rci_job *ircomm,
                                               float                   **u,
                                               float                   **v,
                                               float                    *x,
                                               float                     rinfo[100]);
/**@}*/

/*! \ingroup solvers_module
 * \brief Forward communication interface to the iterative solvers suite of the library.
 *
 * \details This function solves the linear system of equations
 * \f[ Ax=b, \f]
 * where the matrix of coefficients \f$A\f$ is defined by \p mat. The right hand-side is the dense vector \p b and
 * the vector of unknowns is \p x. If \f$A\f$ is symmetric and positive definite then set the option "iterative method" to "cg"
 * to solve the problem using the <a href="https://encyclopediaofmath.org/index.php?title=Conjugate_gradients,_method_of&oldid=46470">Conjugate Gradient
 * method</a>, alternatively set the option to "gmres" to solve
 * using <a href="https://mathworld.wolfram.com/GeneralizedMinimalResidualMethod.html">GMRes</a>. See the \ref anchor_itsol_options
 * for a list of available options to modify the behaviour of each solver.
 *
 * The expected workflow is as follows:
 * -# Call \ref aoclsparse_itsol_s_init or \ref aoclsparse_itsol_d_init to initialize the problem \p handle ( \ref aoclsparse_itsol_handle).
 * -# Choose the solver and adjust its behaviour by setting optional parameters with \ref aoclsparse_itsol_option_set, see also \ref anchor_itsol_options.
 * -# Solve the system by calling \ref aoclsparse_itsol_s_solve or \ref aoclsparse_itsol_d_solve.
 * -# If there is another linear system of equations to solve with the same matrix but a different right-hand side \f$b\f$, then repeat from step 3.
 * -# If solver terminated successfully then vector \p x contains the solution.
 * -# Free the memory with \ref aoclsparse_itsol_destroy.
 *
 * This interface requires to explicitly provide the matrix \f$A\f$ and its descriptor \p descr, this kind of interface is also
 * known as _forward communication_ which contrasts with *reverse communication* in which case the
 * matrix \f$A\f$ and its descriptor \p descr need not be explicitly available. For more details on the latter, see \ref aoclsparse_itsol_d_rci_solve or
 * \ref aoclsparse_itsol_s_rci_solve.
 *
 * @param [inout] handle a valid problem handle, previously initialized by calling \ref aoclsparse_itsol_s_init or \ref aoclsparse_itsol_d_init.
 * @param [in] n the size of the square matrix \p mat.
 * @param [inout] mat coefficient matrix \f$A\f$.
 * @param [inout] descr matrix descriptor for \p mat.
 * @param [in] b right-hand side dense vector \f$b\f$.
 * @param [inout] x dense vector of unknowns. On input, it should contain the initial guess from which to start the iterative
 *                process. If there is no good initial estimate guess then any arbitrary but finite
 *                values can be used. On output, it contains an estimate to the solution of the linear system of equations up
 *                to the requested tolerance, e.g. see "cg rel tolerance" or "cg abs tolerance" in \ref anchor_itsol_options.
 * @param [out]   rinfo vector containing information and stats related to the iterative solve, see
 *                \ref anchor_rinfo.
 * @param [in]    precond (optional, can be nullptr) function pointer to a user routine that applies the preconditioning step
 * \f[ v = Mu \text{or } v = M^{-1}u,\f]
 * where \f$v\f$ is the resulting vector of applying a preconditioning step on the vector \f$u\f$ and \f$M\f$ refers to the
 * user specified preconditioner in matrix form and need not be explicitly available. The void pointer udata, is a convenience pointer that can be used by the user
 * to point to user data and is not used by the itsol framework. If the user requests to use a predefined preconditioner already
 * available in the suite (refer to e.g. "cg preconditioner" or "gmres preconditioner" in \ref anchor_itsol_options), then this parameter need not be provided.
 * @param [in]    monit (optional, can be nullptr) function pointer to a user monitoring routine. If provided, then at each
 *                iteration, the routine is called and can be used to define a custom stopping criteria or to oversee the
 *                convergence process. In general, this function need not be provided. If provided then the solver provides
 *                \p n the problem size,
 *                \p x the current iterate, \p r the current residual vector (\f$r = Ax-b\f$),
 *                \p rinfo the current solver's stats, see \ref anchor_rinfo, and \p udata a convenience pointer that can be used by the user
 *                to point to arbitrary user data and is not used by the itsol framework.
 * @param [inout] udata (optional, can be nullptr) user convenience pointer, it can be used by the user to pass a pointer to user data.
 *                It is not modified by the solver.
 *
 * @rst
 * .. collapse:: Example - CG / floating point double precision (tests/examples/sample_itsol_d_cg.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_itsol_d_cg.cpp
 *           :language: C++
 *           :linenos:
 *
 * \
 *
 * .. collapse:: Example - GMRES / floating point double precision (tests/examples/sample_itsol_d_gmres.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_itsol_d_gmres.cpp
 *           :language: C++
 *           :linenos:
 *
 * \
 *
 * .. collapse:: Example - CG / floating point single precision (tests/examples/sample_itsol_s_cg.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_itsol_s_cg.cpp
 *           :language: C++
 *           :linenos:
 *
 * \
 *
 * .. collapse:: Example - GMRES / floating point single precision (tests/examples/sample_itsol_s_gmres.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_itsol_s_gmres.cpp
 *           :language: C++
 *           :linenos:
 *
 * @endrst
 */
/**@{*/
DLL_PUBLIC
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
    void *udata);

DLL_PUBLIC
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
    void *udata);
/**@}*/

/*! \ingroup solvers_module
 * \brief Performs successive over-relaxation preconditioner operation for single and
 * double precision datatypes to solve a linear system of equations \f$Ax=b\f$.
 *
 * \details
 * \P{aoclsparse_?sorv} performs successive over-relaxation preconditioner on a linear
 * system of equations represented using a sparse matrix \f$A\f$ in CSR storage format.
 * This is an iterative technique that solves the left hand side of this expression for \p x,
 * using an initial guess for \p x
 * @rst
 * .. math::
 *    (D + \omega \, L) \, x^1 = \omega \, b - (\omega \, U + (\omega -1) \, D) \, x^0
 * @endrst
 * where \f$A = L + D + U\f$, \f$x^0\f$ is an input vector \p x and \f$x^1\f$ is an output stored in vector \p x.
 *
 * Initially
 * \f[
 *    x^0 = \left\{
 *    \begin{array}{ll}
 *      alpha * x^0, & \text{ if } alpha \neq 0 \\
 *      0, & \text{ if } alpha = 0
 *    \end{array}
 *    \right.
 *  \f]
 * The convergence is guaranteed for strictly diagonally dominant and positive definite matrices from
 * any starting point, \f$x^0\f$. API returns the vector x after single iteration. Caller can invoke this
 * function in a loop until their desired convergence is reached.
 *
 * NOTE:
 *
 * 1. Input CSR matrix should have non-zero full diagonals with each diagonal occurring only once in a row.
 *
 * 2. API supports forward sweep on general matrix for single and double precision datatypes.
 *
 * @param [in]
 * sor_type Selects the type of operation performed by the preconditioner. Only \ref aoclsparse_sor_forward
 *          is supported at present.
 * @param [in]
 * descr    Descriptor of A. Only \ref aoclsparse_matrix_type_general is supported at present.
 *          As a consequence, all other parameters within the descriptor are ignored.
 * @param [in]
 * A        Matrix structure containing a square sparse matrix \f$A\f$ of size \f$m \times m\f$.
 * @param [in]
 * omega    Relaxation factor. For better convergence, 0 < \f$\omega\f$ < 2. If \f$\omega\f$ = 1,
 *          the preconditioner is equivalent to the Gauss-Seidel method.
 * @param [in]
 * alpha    Scalar value used to normalize or set to zero the vector \p x that holds an initial guess.
 * @param [inout]
 * x        A vector of \f$m\f$ elements that holds an initial guess as well as the solution vector.
 * @param [in]
 * b        A vector of \f$m\f$ elements that holds the right-hand side of the equation being solved.
 *
 * \retval  aoclsparse_status_success            Completed successfully.
 * \retval  aoclsparse_status_invalid_pointer    One or more of the pointers \p A, \p descr, \p x
 *                                               or \p b are invalid.
 * \retval  aoclsparse_status_wrong_type         Data type of \p A does not match the function.
 * \retval  aoclsparse_status_not_implemented    Expecting general matrix in CSR format for single
 *                                               or double precision datatypes with \ref aoclsparse_sor_forward.
 * \retval  aoclsparse_status_invalid_size       Matrix is not square.
 * \retval  aoclsparse_status_invalid_value      \p M or \p N is set to a negative value; or \p A,
 *                                               \p descr or \p sor_type has invalid value; or presence of zero-valued or
 *                                               repeated diagonal elements.
 *
 * @rst
 * .. collapse:: Example (tests/examples/sample_dsorv.cpp)
 *
 *    .. only:: html
 *
 *       .. literalinclude:: ../tests/examples/sample_dsorv.cpp
 *          :language: C++
 *          :linenos:
 * @endrst
 */
/**@{*/

DLL_PUBLIC
aoclsparse_status aoclsparse_ssorv(aoclsparse_sor_type        sor_type,
                                   const aoclsparse_mat_descr descr,
                                   const aoclsparse_matrix    A,
                                   float                      omega,
                                   float                      alpha,
                                   float                     *x,
                                   const float               *b);
DLL_PUBLIC
aoclsparse_status aoclsparse_dsorv(aoclsparse_sor_type        sor_type,
                                   const aoclsparse_mat_descr descr,
                                   const aoclsparse_matrix    A,
                                   double                     omega,
                                   double                     alpha,
                                   double                    *x,
                                   const double              *b);
DLL_PUBLIC
aoclsparse_status aoclsparse_csorv(aoclsparse_sor_type             sor_type,
                                   const aoclsparse_mat_descr      descr,
                                   const aoclsparse_matrix         A,
                                   aoclsparse_float_complex        omega,
                                   aoclsparse_float_complex        alpha,
                                   aoclsparse_float_complex       *x,
                                   const aoclsparse_float_complex *b);
DLL_PUBLIC
aoclsparse_status aoclsparse_zsorv(aoclsparse_sor_type              sor_type,
                                   const aoclsparse_mat_descr       descr,
                                   const aoclsparse_matrix          A,
                                   aoclsparse_double_complex        omega,
                                   aoclsparse_double_complex        alpha,
                                   aoclsparse_double_complex       *x,
                                   const aoclsparse_double_complex *b);
/**@}*/
/*! \ingroup solvers_module
 *  \brief Symmetric Gauss Seidel(SYMGS) Preconditioner for real/complex single and double data precisions.
 *
 *  \details
 *  \P{aoclsparse_?symgs} performs an iteration of Gauss Seidel preconditioning. Krylov methods such
 *  as CG (Conjugate Gradient) and GMRES (Generalized Minimal Residual) are used to solve large sparse
 *  linear systems of the form
 *  \f[
 *  op(A)\; x = \alpha b,
 *  \f]
 *  where \f$A\f$ is a sparse matrix of size \f$m\f$, \f$op()\f$ is a linear operator, \f$b\f$ is a dense right-hand
 *  side vector and \f$x\f$ is the unknown dense vector, while \f$\alpha\f$ is a scalar.
 *  This Gauss Seidel(GS) relaxation is typically used either as a preconditioner for a Krylov solver directly,
 *  or as a smoother in a V –cycle of a multigrid preconditioner to accelerate the convergence rate.
 *  The Symmetric Gauss Seidel algorithm performs a forward sweep followed by a backward sweep to maintain
 *  symmetry of the matrix operation.
 *
 *  To solve a linear system \f$Ax = b\f$, Gauss Seidel(GS) iteration is based on the matrix splitting
 *  \f[ A = L + D + U = -E + D - F \f]
 *  where \f$-E\f$ or \f$L\f$ is strictly lower triangle,
 *  \f$D\f$ is diagonal and
 *  \f$-F\f$ or \f$D\f$ is strictly upper triangle.
 *  Gauss-Seidel is best derived as element-wise (refer Yousef Saad's book Iterative Methods for
 *  Sparse Linear Systems, Second Edition, Chapter 4.1, p. 125 onwards):
 *  \f[
 *  x_i = \frac{1}{a_{ii}} \; \left (b_i - \sum_{j=1}^{i-1} a_{ij} \; x_j - \sum_{j=i+1}^n a_{ij} \; x_j\right)
 *  \f]
 *  where the first sum is lower triangle i.e., \f$-Ex\f$ and the second sum is upper triangle i.e., \f$-Fx\f$.
 *  If we iterate through the rows \p i=1 to \p n and keep overwriting/reusing the new \f$x_{i}\f$, we get
 *  forward GS, expressed in matrix form as,
 *  \f[
 *  (D-E) \; x_{k+1} = F \; x_k + b
 *  \f]
 *  Iterating through the rows in reverse order from \p i=n to \p 1, the upper triangle keeps using the new
 *  \f$x_{k+1}\f$ elements and we get backward GS, expressed in matrix form as,
 *  \f[
 *  (D-F) \; x_{k+1} = E \; x_k + b
 *  \f]
 *  The above two equations can be expressed in terms of \p L, \p D and \p U as follows,
 *  \f[
 *  (L + D)\;x_1 = b - U\;x_0
 *  \f]
 *  \f[
 *  (U + D)\;x = b - L\;x_1
 *  \f]
 *  So, Symmetric Gauss Seidel (SYMGS) can be computed using two \P{aoclsparse_?mv}
 *  and two \P{aoclsparse_?trsv} operations.
 *
 *  The sparse matrix \f$A\f$ can be either a symmetric or a Hermitian matrix, whose fill
 *  is indicated by \p fill_mode from the matrix descriptor \p descr
 *  where either upper or lower triangular portion of the matrix is used.
 *  Matrix \f$A\f$ must be of full rank,  that is, the matrix must be invertible. The linear operator \f$op()\f$ can
 *  define the transposition or conjugate transposition operations. By default, no transposition
 *  is performed. The right-hand-side vector \f$b\f$ and the solution vector \f$x\f$ are dense
 *  and must be of the correct size, that is \f$m\f$. If used as fixed point iterative method,
 *  the convergence is guaranteed for  strictly diagonally dominant and symmetric positive definite matrices from
 *  any starting point, \p x0. However, the API can be applied to wider types of input or as a preconditioning step.
 *  Refer Yousef Saad's Iterative Methods for Sparse Linear Systems 2nd Edition, Theorem 4.9
 *  and related literature for mathematical theory.
 *
 *  \note
 *
 *  1. If the matrix descriptor \p descr specifies that the matrix \f$A\f$ is to be regarded as
 *     having a unitary diagonal, then the main diagonal entries of matrix \f$A\f$ are not accessed and
 *     are considered to all be ones.
 *
 *  2. If the matrix \f$A\f$ is described as upper triangular, then only the upper triangular portion of the
 *     matrix is referenced. Conversely, if the matrix \f$A\f$ is described lower triangular, then only the
 *     lower triangular portion of the matrix is used.
 *
 *  3. This set of APIs allocates couple of work array buffers of size \f$m\f$ for to store intermediate results
 *
 *  4. If the input matrix is of triangular type, the SGS is computed using a single \P{aoclsparse_?trsv} operation
 *     and a quick return is made without going through the 3-step reference(described above)
 *
 *  @param[in]
 *  trans       matrix operation to perform on \f$A\f$. Possible values are \ref aoclsparse_operation_none,
 *              \ref aoclsparse_operation_transpose, and \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  A           sparse matrix \f$A\f$ of size \f$m\f$.
 *  @param[in]
 *  descr       descriptor of the sparse matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  b           dense vector, of size \f$m\f$.
 *  @param[out]
 *  x           solution vector \f$x,\f$ dense vector of size \f$m\f$.
 *
 *  \retval     aoclsparse_status_success indicates that the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size informs that either \p m, \p n or \p nnz
 *              is invalid. The error code also informs if the given sparse matrix \f$A\f$ is not square.
 *  \retval     aoclsparse_status_invalid_value informs that either \p base, \p trans, matrix type \p descr->type or
 *              fill mode \p descr->fill_mode is invalid. If the sparse matrix \f$A\f$ is not of full rank, the error code is returned to indicate that the
 *              linear system cannot be solved.
 *  \retval     aoclsparse_status_invalid_pointer informs that either \p descr, \p A,
 *              \p b, or \p x pointer is invalid.
 *  \retval     aoclsparse_status_not_implemented this error occurs when the provided matrix's
 *              \ref aoclsparse_fill_mode is \ref aoclsparse_diag_type_unit or the input format is not \ref aoclsparse_csr_mat,
 *              or when \ref aoclsparse_matrix_type is \ref aoclsparse_matrix_type_general and \p trans is
 *              \ref aoclsparse_operation_conjugate_transpose.
 * @rst
 *
 * .. collapse:: Example - Real space (tests/examples/sample_dsymgs.cpp)
 *
 *      .. only:: html
 *
 *         .. literalinclude:: ../tests/examples/sample_dsymgs.cpp
 *            :language: C++
 *            :linenos:
 *
 * \
 *
 * .. collapse:: Example - Complex space (tests/examples/sample_zsymgs.cpp)
 *
 *      .. only:: html
 *
 *         .. literalinclude:: ../tests/examples/sample_zsymgs.cpp
 *            :language: C++
 *            :linenos:
 *
 * @endrst
 */

/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_zsymgs(aoclsparse_operation             trans,
                                    aoclsparse_matrix                A,
                                    const aoclsparse_mat_descr       descr,
                                    const aoclsparse_double_complex  alpha,
                                    const aoclsparse_double_complex *b,
                                    aoclsparse_double_complex       *x);
DLL_PUBLIC
aoclsparse_status aoclsparse_csymgs(aoclsparse_operation            trans,
                                    aoclsparse_matrix               A,
                                    const aoclsparse_mat_descr      descr,
                                    const aoclsparse_float_complex  alpha,
                                    const aoclsparse_float_complex *b,
                                    aoclsparse_float_complex       *x);
DLL_PUBLIC
aoclsparse_status aoclsparse_dsymgs(aoclsparse_operation       trans,
                                    aoclsparse_matrix          A,
                                    const aoclsparse_mat_descr descr,
                                    const double               alpha,
                                    const double              *b,
                                    double                    *x);
DLL_PUBLIC
aoclsparse_status aoclsparse_ssymgs(aoclsparse_operation       trans,
                                    aoclsparse_matrix          A,
                                    const aoclsparse_mat_descr descr,
                                    const float                alpha,
                                    const float               *b,
                                    float                     *x);
/**@}*/

/*! \ingroup solvers_module
 *  \brief Symmetric Gauss Seidel Preconditioner for real/complex single and double data precisions. (kernel flag variation).
 *
 *  \details
 * @rst For full details refer to :cpp:func:`aoclsparse_?symgs()<aoclsparse_ssymgs>`.
 *
 * This variation of SYMGS, namely with a suffix of `_kid`, allows to choose which
 * SYMGS kernel to use (if possible). Currently the possible choices are:
 *
 * :code:`kid=0`
 *     Reference implementation (No explicit AVX instructions).
 *
 * Any other Kernel ID value will default to :code:`kid` = 0.
 * @endrst
 *
 *  @param[in]
 *  trans       matrix operation to perform on \f$A\f$. Possible values are \ref aoclsparse_operation_none,
 *              \ref aoclsparse_operation_transpose, and \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  A           sparse matrix \f$A\f$ of size \f$m\f$.
 *  @param[in]
 *  descr       descriptor of the sparse matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  b           dense vector, of size \f$m\f$.
 *  @param[out]
 *  x           solution vector \f$x,\f$ dense vector of size \f$m\f$.
 *  @param[in]
 *  kid         Kernel ID, hints a request on which SYMGS kernel to use.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_zsymgs_kid(aoclsparse_operation             trans,
                                        aoclsparse_matrix                A,
                                        const aoclsparse_mat_descr       descr,
                                        const aoclsparse_double_complex  alpha,
                                        const aoclsparse_double_complex *b,
                                        aoclsparse_double_complex       *x,
                                        const aoclsparse_int             kid);
DLL_PUBLIC
aoclsparse_status aoclsparse_csymgs_kid(aoclsparse_operation            trans,
                                        aoclsparse_matrix               A,
                                        const aoclsparse_mat_descr      descr,
                                        const aoclsparse_float_complex  alpha,
                                        const aoclsparse_float_complex *b,
                                        aoclsparse_float_complex       *x,
                                        const aoclsparse_int            kid);
DLL_PUBLIC
aoclsparse_status aoclsparse_dsymgs_kid(aoclsparse_operation       trans,
                                        aoclsparse_matrix          A,
                                        const aoclsparse_mat_descr descr,
                                        const double               alpha,
                                        const double              *b,
                                        double                    *x,
                                        const aoclsparse_int       kid);
DLL_PUBLIC
aoclsparse_status aoclsparse_ssymgs_kid(aoclsparse_operation       trans,
                                        aoclsparse_matrix          A,
                                        const aoclsparse_mat_descr descr,
                                        const float                alpha,
                                        const float               *b,
                                        float                     *x,
                                        const aoclsparse_int       kid);
/**@}*/

/*! \ingroup solver_module
 *  \brief Symmetric Gauss Seidel Preconditioner followed by SPMV for single and double precision datatypes.
 *  \details
 *  @rst For full details refer to :cpp:func:`aoclsparse_?symgs()<aoclsparse_ssymgs>`.
 *
 *  This variation of SYMGS, namely with a suffix of `_mv`, performs matrix-vector multiplication between
 *  the sparse matrix \f$A\f$ and the Gauss Seidel solution vector \f$x\f$.
 *  @endrst
 *  @param[in]
 *  trans       matrix operation to perform on \f$A\f$. Possible values are \ref aoclsparse_operation_none,
 *              \ref aoclsparse_operation_transpose, and \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  A           sparse matrix \f$A\f$ of size \f$m\f$.
 *  @param[in]
 *  descr       descriptor of the sparse matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  b           dense vector, of size \f$m\f$.
 *  @param[out]
 *  x           solution vector \f$x,\f$ dense vector of size \f$m\f$.
 *  @param[out]
 *  y           sparse-product vector \f$y,\f$ dense vector of size \f$m\f$.
 *
 *  \retval     aoclsparse_status_success indicates that the operation completed successfully.
 *  \retval     aoclsparse_status_invalid_size informs that either \p m, \p n or \p nnz
 *              is invalid. The error code also informs if the given sparse matrix \f$A\f$ is not square.
 *  \retval     aoclsparse_status_invalid_value informs that either \p base, \p trans, matrix type \p descr->type or
 *              fill mode \p descr->fill_mode is invalid. If the sparse matrix \f$A\f$ is not of full rank, the error code is returned to indicate that the
 *              linear system cannot be solved.
 *  \retval     aoclsparse_status_invalid_pointer informs that either \p descr, \p A,
 *              \p b, \p x or \p y pointer is invalid.
 *  \retval     aoclsparse_status_not_implemented this error occurs when the provided matrix's
 *              \ref aoclsparse_fill_mode is \ref aoclsparse_diag_type_unit or the input format is not \ref aoclsparse_csr_mat,
 *              or when \ref aoclsparse_matrix_type is \ref aoclsparse_matrix_type_general and \p trans is
 *              \ref aoclsparse_operation_conjugate_transpose.
 * @rst
 *
 * .. collapse:: Example - Real space (tests/examples/sample_dsymgs_mv.cpp)
 *
 *      .. only:: html
 *
 *         .. literalinclude:: ../tests/examples/sample_dsymgs_mv.cpp
 *            :language: C++
 *            :linenos:
 *
 * \
 *
 * .. collapse:: Example - Complex space (tests/examples/sample_zsymgs_mv.cpp)
 *
 *      .. only:: html
 *
 *         .. literalinclude:: ../tests/examples/sample_zsymgs_mv.cpp
 *            :language: C++
 *            :linenos:
 *
 * @endrst
 */

/**@{*/

DLL_PUBLIC
aoclsparse_status aoclsparse_zsymgs_mv(aoclsparse_operation             trans,
                                       aoclsparse_matrix                A,
                                       const aoclsparse_mat_descr       descr,
                                       const aoclsparse_double_complex  alpha,
                                       const aoclsparse_double_complex *b,
                                       aoclsparse_double_complex       *x,
                                       aoclsparse_double_complex       *y);
DLL_PUBLIC
aoclsparse_status aoclsparse_csymgs_mv(aoclsparse_operation            trans,
                                       aoclsparse_matrix               A,
                                       const aoclsparse_mat_descr      descr,
                                       const aoclsparse_float_complex  alpha,
                                       const aoclsparse_float_complex *b,
                                       aoclsparse_float_complex       *x,
                                       aoclsparse_float_complex       *y);
DLL_PUBLIC
aoclsparse_status aoclsparse_dsymgs_mv(aoclsparse_operation       trans,
                                       aoclsparse_matrix          A,
                                       const aoclsparse_mat_descr descr,
                                       const double               alpha,
                                       const double              *b,
                                       double                    *x,
                                       double                    *y);
DLL_PUBLIC
aoclsparse_status aoclsparse_ssymgs_mv(aoclsparse_operation       trans,
                                       aoclsparse_matrix          A,
                                       const aoclsparse_mat_descr descr,
                                       const float                alpha,
                                       const float               *b,
                                       float                     *x,
                                       float                     *y);
/**@}*/

/*! \ingroup solver_module
 *  \brief Symmetric Gauss Seidel Preconditioner followed by SPMV
 *  for single and double precision datatypes. (kernel flag variation).
 *
 *  \details
 *  @rst For full details refer to :cpp:func:`aoclsparse_?symgs()<aoclsparse_ssymgs>`.
 *
 * This variation of Fused SYMGS, namely with a suffix of `_kid`, allows to choose which
 * Fused SYMGS kernel to use (if possible). Currently the possible choices are:
 *
 * :code:`kid=0`
 *     Reference implementation (No explicit AVX instructions).
 *
 * Any other Kernel ID value will default to :code:`kid` = 0.
 * @endrst
 *
 *  @param[in]
 *  trans       matrix operation to perform on \f$A\f$. Possible values are \ref aoclsparse_operation_none,
 *              \ref aoclsparse_operation_transpose, and \ref aoclsparse_operation_conjugate_transpose.
 *  @param[in]
 *  A           sparse matrix \f$A\f$ of size \f$m\f$.
 *  @param[in]
 *  descr       descriptor of the sparse matrix \f$A\f$.
 *  @param[in]
 *  alpha       scalar \f$\alpha\f$.
 *  @param[in]
 *  b           dense vector, of size \f$m\f$.
 *  @param[out]
 *  x           solution vector \f$x,\f$ dense vector of size \f$m\f$.
 *  @param[out]
 *  y           sparse-product vector \f$y,\f$ dense vector of size \f$m\f$.
 *  @param[in]
 *  kid         Kernel ID, hints a request on which Fused SYMGS kernel to use.
 */
/**@{*/
DLL_PUBLIC
aoclsparse_status aoclsparse_zsymgs_mv_kid(aoclsparse_operation             trans,
                                           aoclsparse_matrix                A,
                                           const aoclsparse_mat_descr       descr,
                                           const aoclsparse_double_complex  alpha,
                                           const aoclsparse_double_complex *b,
                                           aoclsparse_double_complex       *x,
                                           aoclsparse_double_complex       *y,
                                           const aoclsparse_int             kid);
DLL_PUBLIC
aoclsparse_status aoclsparse_csymgs_mv_kid(aoclsparse_operation            trans,
                                           aoclsparse_matrix               A,
                                           const aoclsparse_mat_descr      descr,
                                           const aoclsparse_float_complex  alpha,
                                           const aoclsparse_float_complex *b,
                                           aoclsparse_float_complex       *x,
                                           aoclsparse_float_complex       *y,
                                           const aoclsparse_int            kid);
DLL_PUBLIC
aoclsparse_status aoclsparse_dsymgs_mv_kid(aoclsparse_operation       trans,
                                           aoclsparse_matrix          A,
                                           const aoclsparse_mat_descr descr,
                                           const double               alpha,
                                           const double              *b,
                                           double                    *x,
                                           double                    *y,
                                           const aoclsparse_int       kid);
DLL_PUBLIC
aoclsparse_status aoclsparse_ssymgs_mv_kid(aoclsparse_operation       trans,
                                           aoclsparse_matrix          A,
                                           const aoclsparse_mat_descr descr,
                                           const float                alpha,
                                           const float               *b,
                                           float                     *x,
                                           float                     *y,
                                           const aoclsparse_int       kid);
/**@}*/
#ifdef __cplusplus
}
#endif

#endif // AOCLSPARSE_SOLVERS_H_
