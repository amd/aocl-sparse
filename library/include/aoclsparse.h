/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
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
#ifndef AOCLSPARSE_H_
#define AOCLSPARSE_H_

/** \mainpage AOCL-Sparse Introduction
*AOCL-Sparse is a library that contains basic linear algebra subroutines for sparse matrices and vectors
*optimized for AMD EPYC family of processors. It is designed to be used with C and C++.
*The current functionality of AOCL-Sparse is organized in the following categories:
* <ol><li>Sparse Level 3 functions describe the operations between a matrix in sparse format and a
*          matrix in dense/sparse format.</li>
* <li>Sparse Level 2 functions describe the operations between a matrix in sparse format and a
*   vector in dense format.</li>
* <li>Sprase Level 1 functions perform vector operations such as dot product, vector additions on
*   sparse vectors, i.e., vectors stored in a compressed form. </li>
* <li>Iterative sparse solvers that solve a linear system of equations.</li>
* <li>Analysis and execute functionalities for performing optimized Sparse Matrix-Dense Vector
* multiplication and Sparse Solver.</li>
* <li>Sparse Format Conversion functions describe operations on a matrix in sparse format to
* obtain a different matrix format.</li>
*<li>Sparse Auxiliary Functions describe auxiliary functions.</li></ol>

*
*/

#include "aoclsparse_analysis.h"
#include "aoclsparse_auxiliary.h"
#include "aoclsparse_convert.h"
#include "aoclsparse_functions.h"
#include "aoclsparse_solvers.h"
#include "aoclsparse_types.h"
#include "aoclsparse_version.h"

#endif // AOCLSPARSE_H_
