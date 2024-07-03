..
   Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
..
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
..
   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
..
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

Iterative Linear System Solvers
*******************************

.. _itsol_intro:

Introduction of Iterative Solver Suite (itsol)
==============================================

AOCL-Sparse Iterative Solver Suite (itsol) is an iterative framework for solving large-scale sparse linear systems of equations of the form

.. math::

   Ax=b,

where :math:`A` is a sparse full-rank square matrix of size :math:`n` by :math:`n`, :math:`b` is a dense :math:`n`-vector, and :math:`x` is the vector of unknowns also of size :math:`n`.
The framework solves the previous problem using either the Conjugate Gradient method or GMRES. It supports a variety of preconditioners (*accelerators*) such as
Symmetric Gauss-Seidel or Incomplete LU factorization, ILU(0).

Iterative solvers at each step (iteration) find a better approximation to the solution of the linear system of equations in the sense that it reduces an error metric.
In contrast, direct solvers only provide a solution once the full algorithm as been executed. A great advantage of iterative solvers is that they can be
interrupted once an approximate solution is deemed acceptable.

Forward and Reverse Communication Interfaces
--------------------------------------------

The suite presents two separate interfaces to all the iterative solvers, a direct one, :cpp:func:`aoclsparse_itsol_d_solve`,
(:cpp:func:`aoclsparse_itsol_s_solve`, :cpp:func:`aoclsparse_itsol_c_solve`, :cpp:func:`aoclsparse_itsol_z_solve`)
and a reverse communication (RCI) one :cpp:func:`aoclsparse_itsol_d_rci_solve` (:cpp:func:`aoclsparse_itsol_s_rci_solve`
:cpp:func:`aoclsparse_itsol_c_rci_solve`, :cpp:func:`aoclsparse_itsol_z_rci_solve`) While the underlying algorithms are exactly the same,
the difference lies in how data is communicated to the solvers.

The direct communication interface expects to have explicit access to the coefficient matrix :math:`A`. On the other hand, the reverse communication interface makes
no assumption on the matrix storage. Thus when the solver requires some matrix operation such as a
matrix-vector product, it returns control to the user and asks the user perform the operation and provide the results by calling again the RCI solver.

Recommended Workflow
--------------------

For solving a linear system of equations, the following workflow is recommended:

- Call :cpp:func:`aoclsparse_itsol_s_init` or :cpp:func:`aoclsparse_itsol_d_init` or :cpp:func:`aoclsparse_itsol_c_init` or :cpp:func:`aoclsparse_itsol_z_init` to initialize aoclsparse_itsol_handle.
- Choose the solver and adjust its behaviour by setting optional parameters with :cpp:func:`aoclsparse_itsol_option_set`, see there all options available.
- If the reverse communication interface is desired, define the system's input with
  :cpp:func:`aoclsparse_itsol_s_rci_input` (or :cpp:func:`aoclsparse_itsol_d_rci_input` or :cpp:func:`aoclsparse_itsol_c_rci_input` or :cpp:func:`aoclsparse_itsol_z_rci_input`).
- Solve the system with either using direct interface :cpp:func:`aoclsparse_itsol_s_solve` (or :cpp:func:`aoclsparse_itsol_d_solve` or :cpp:func:`aoclsparse_itsol_c_solve` or :cpp:func:`aoclsparse_itsol_z_solve`) or
  reverse communication interface :cpp:func:`aoclsparse_itsol_s_rci_solve` (or :cpp:func:`aoclsparse_itsol_d_rci_solve` or :cpp:func:`aoclsparse_itsol_c_rci_solve` or :cpp:func:`aoclsparse_itsol_z_rci_solve`)
- Free the memory with :cpp:func:`aoclsparse_itsol_destroy`.

Information Array
-----------------

The array ``rinfo[100]`` is used by the solvers (e.g. :cpp:func:`aoclsparse_itsol_s_solve` or :cpp:func:`aoclsparse_itsol_d_rci_solve` or :cpp:func:`aoclsparse_itsol_c_rci_solve`  or :cpp:func:`aoclsparse_itsol_z_rci_solve`) to report
back useful convergence metrics and other solver statistics.
The user callback ``monit`` is also equipped with this array and can be used
to view or monitor the state of the solver.
The solver will populate the following entries with the most recent iteration data

.. csv-table::
   :header: "Index", "Description"
   :widths: 10, 40

   "0", "Absolute residual norm, :math:`r_{\text{abs}} = \| Ax-b\|_2`."
   "1", "Norm of the right-hand side vector :math:`b`, :math:`\|b\|_2`."
   "2-29", "Reserved for future use."
   "30", "Iteration counter."
   "31-99", "Reserved for future use."

References
----------

.. bibliography::
   :all:
   :list: bullet



API documentation
=================

.. .. doxygentypedef:: aoclsparse_itsol_rci_job
.. .. doxygenenum:: aoclsparse_itsol_rci_job_

aoclsparse_itsol_rci_job
------------------------

.. doxygenenum:: aoclsparse_itsol_rci_job

aoclsparse_itsol\_?_init()
---------------------------

.. doxygenfunction:: aoclsparse_itsol_s_init
    :outline:
.. doxygenfunction:: aoclsparse_itsol_d_init
    :outline:
.. doxygenfunction:: aoclsparse_itsol_c_init
    :outline:
.. doxygenfunction:: aoclsparse_itsol_z_init

aoclsparse_itsol_destroy()
--------------------------

.. doxygenfunction:: aoclsparse_itsol_destroy

aoclsparse_itsol\_?_solve()
----------------------------

.. doxygenfunction:: aoclsparse_itsol_s_solve
    :outline:
.. doxygenfunction:: aoclsparse_itsol_d_solve
    :outline:
.. doxygenfunction:: aoclsparse_itsol_c_solve
    :outline:
.. doxygenfunction:: aoclsparse_itsol_z_solve

aoclsparse_itsol_option_set()
-----------------------------

.. doxygenfunction:: aoclsparse_itsol_option_set


aoclsparse_itsol_handle_prn_options()
-------------------------------------

.. doxygenfunction:: aoclsparse_itsol_handle_prn_options

aoclsparse_itsol\_?_rci_input()
--------------------------------

.. doxygenfunction:: aoclsparse_itsol_s_rci_input
    :outline:
.. doxygenfunction:: aoclsparse_itsol_d_rci_input
    :outline:
.. doxygenfunction:: aoclsparse_itsol_c_rci_input
    :outline:
.. doxygenfunction:: aoclsparse_itsol_z_rci_input

aoclsparse_itsol\_?_rci_solve()
--------------------------------

.. doxygenfunction:: aoclsparse_itsol_s_rci_solve
    :outline:
.. doxygenfunction:: aoclsparse_itsol_d_rci_solve
    :outline:
.. doxygenfunction:: aoclsparse_itsol_c_rci_solve
    :outline:
.. doxygenfunction:: aoclsparse_itsol_z_rci_solve

aoclsparse\_?symgs()
--------------------------------

.. doxygenfunction:: aoclsparse_ssymgs
    :outline:
.. doxygenfunction:: aoclsparse_dsymgs
    :outline:
.. doxygenfunction:: aoclsparse_csymgs
    :outline:
.. doxygenfunction:: aoclsparse_zsymgs

.. doxygenfunction:: aoclsparse_ssymgs_mv
    :outline:
.. doxygenfunction:: aoclsparse_dsymgs_mv
    :outline:
.. doxygenfunction:: aoclsparse_csymgs_mv
    :outline:
.. doxygenfunction:: aoclsparse_zsymgs_mv

aoclsparse\_?sorv()
-------------------
.. doxygenfunction:: aoclsparse_ssorv
   :outline:
.. doxygenfunction:: aoclsparse_dsorv

aoclsparse_ilu\_?smoother()
---------------------------

.. doxygenfunction:: aoclsparse_silu_smoother
    :outline:
.. doxygenfunction:: aoclsparse_dilu_smoother
    :outline:
.. doxygenfunction:: aoclsparse_cilu_smoother
    :outline:
.. doxygenfunction:: aoclsparse_zilu_smoother