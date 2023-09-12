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

/** \mainpage The AMD AOCL-Sparse Library
 *
 *  # Introduction
 *
 * The AMD Optimized CPU Library AOCL-Sparse is a library that contains Basic Linear Algebra Subroutines for sparse matrices and vectors
 * (Sparse BLAS) and is optimized for AMD EPYC and RYZEN family of CPU processors. It implements numerical algorithms in C++ while providing a 
 * public-facing C interface.
 *
 * Functionality of AMD AOCL-Sparse library is organized in the following categories
 * <ol>
 * <li> <b>Sparse Level 1</b> functions perform vector operations such as dot product, vector additions on
 *   sparse vectors, and other similar operations. </li>
 *
 * <li><b>Sparse Level 2</b> functions describe the operations between a matrix in sparse format and a
 *   vector in dense format.</li>
 *
 * <li><b>Sparse Level 3</b> functions describe the operations between a matrix in sparse format and one or more
 *          dense/sparse matrices.</li>
 *
 * <li><b>Iterative sparse solvers</b> that solve sparse linear system of equations.</li>
 *
 * <li><b>Analysis and execute functionalities</b> for performing optimized operations.</li>
 *
 * <li>Sparse format conversion functions for translating matrices in a variety of
 * sparse storage formats.</li>
 *
 *<li>Sparse auxiliary functions used to perform miscelaneous tasks adjacent to the ones described above.</li></ol>
 */

#include "aoclsparse_analysis.h"
#include "aoclsparse_auxiliary.h"
#include "aoclsparse_convert.h"
#include "aoclsparse_functions.h"
#include "aoclsparse_solvers.h"
#include "aoclsparse_types.h"
#include "aoclsparse_version.h"

#endif // AOCLSPARSE_H_
