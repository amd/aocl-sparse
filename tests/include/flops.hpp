/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
 *  \brief flops.hpp provides floating point counts of Sparse Linear Algebra Subprograms
 *  of Level 1, 2 and 3.
 */

#pragma once
#ifndef FLOPS_HPP
#define FLOPS_HPP

#include <aoclsparse.h>

/*
 *===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double spmv_gflop_count(aoclsparse_int M, aoclsparse_int nnz, bool beta = false)
{
    return (2.0 * nnz + (beta ? M : 0)) / 1e9;
}

#endif // FLOPS_HPP
