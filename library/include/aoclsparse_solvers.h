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
 * \{
 */
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
#ifdef __cplusplus
}
#endif

#endif // AOCLSPARSE_SOLVERS_H_
