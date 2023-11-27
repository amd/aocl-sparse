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

AOCL-Sparse Types
*****************

Numerical types
===============

.. doxygentypedef:: aoclsparse_int
.. .. doxygentypedef:: aoclsparse_float_complex
.. .. doxygenstruct:: aoclsparse_float_complex_
.. doxygenstruct:: aoclsparse_float_complex
    :members:
.. .. doxygentypedef:: aoclsparse_double_complex
.. .. doxygenstruct:: aoclsparse_double_complex_
.. doxygenstruct:: aoclsparse_double_complex
    :members:

Matrix object and descriptor
============================

.. doxygentypedef:: aoclsparse_matrix
.. doxygentypedef:: aoclsparse_mat_descr

Enums
=====

Function return status
----------------------
.. .. doxygentypedef:: aoclsparse_status
.. .. doxygenenum:: aoclsparse_status_
.. doxygenenum:: aoclsparse_status

Associated with :cpp:type:`aoclsparse_matrix`
------------------------------------------------

.. .. doxygentypedef:: aoclsparse_matrix_data_type
.. .. doxygenenum:: aoclsparse_matrix_data_type_
.. doxygenenum:: aoclsparse_matrix_data_type

.. .. only:: internal

    .. .. doxygenenum:: aoclsparse_matrix_format_type
    .. .. doxygentypedef:: aoclsparse_matrix_format_type
    .. .. doxygenenum:: aoclsparse_matrix_format_type_

See also:

* :cpp:type:`aoclsparse_index_base`

Associated with matrix descriptor (:cpp:type:`aoclsparse_mat_descr`)
-----------------------------------------------------------------------

.. .. doxygentypedef:: aoclsparse_matrix_type
.. .. doxygenenum:: aoclsparse_matrix_type_
.. doxygenenum:: aoclsparse_matrix_type

.. .. doxygentypedef:: aoclsparse_index_base
.. .. doxygenenum:: aoclsparse_index_base_
.. doxygenenum:: aoclsparse_index_base

.. .. doxygentypedef:: aoclsparse_diag_type
.. .. doxygenenum:: aoclsparse_diag_type_
.. doxygenenum:: aoclsparse_diag_type

.. .. doxygentypedef:: aoclsparse_fill_mode
.. .. doxygenenum:: aoclsparse_fill_mode_
.. doxygenenum:: aoclsparse_fill_mode

.. .. doxygentypedef:: aoclsparse_order
.. .. doxygenenum:: aoclsparse_order_
.. doxygenenum:: aoclsparse_order

Miscellaneous
-------------

.. .. doxygentypedef:: aoclsparse_operation
.. .. doxygenenum:: aoclsparse_operation_
.. doxygenenum:: aoclsparse_operation


.. doxygentypedef:: aoclsparse_itsol_handle

.. .. doxygentypedef:: aoclsparse_ilu_type
.. .. doxygenenum:: aoclsparse_ilu_type_
.. doxygenenum:: aoclsparse_ilu_type
..
.. .. doxygentypedef:: aoclsparse_request
.. .. doxygenenum:: aoclsparse_request_
.. doxygenenum:: aoclsparse_request

.. .. doxygentypedef:: aoclsparse_sor_type
.. .. doxygenenum:: aoclsparse_sor_type_
.. doxygenenum:: aoclsparse_sor_type

.. .. doxygentypedef:: aoclsparse_memory_usage
.. .. doxygenenum:: aoclsparse_memory_usage_
.. doxygenenum:: aoclsparse_memory_usage
