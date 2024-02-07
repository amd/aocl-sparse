/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.All rights reserved.
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
 *  \brief aoclsparse_gbyte.hpp provides data transfer counts of Sparse
 * Linear Algebra Subprograms of Level 1, 2 and 3.
 */

#pragma once
#ifndef AOCLSPARSE_GBYTE_HPP
#define AOCLSPARSE_GBYTE_HPP

#include "aoclsparse.h"

/*
 * ===========================================================================
 *    level 2 SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double
    csrmv_gbyte_count(aoclsparse_int M, aoclsparse_int N, aoclsparse_int nnz, bool beta = false)
{
    return ((M + 1 + nnz) * sizeof(aoclsparse_int) + (M + N + nnz + (beta ? M : 0)) * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double
    ellmv_gbyte_count(aoclsparse_int M, aoclsparse_int N, aoclsparse_int nnz, bool beta = false)
{
    return (nnz * sizeof(aoclsparse_int) + (M + N + nnz + (beta ? M : 0)) * sizeof(T)) / 1e9;
}

template <typename T>
constexpr double bsrmv_gbyte_count(aoclsparse_int mb,
                                   aoclsparse_int nb,
                                   aoclsparse_int nnzb,
                                   aoclsparse_int bsr_dim,
                                   bool           beta = false)
{
    return ((mb + 1 + nnzb) * sizeof(aoclsparse_int)
            + ((mb + nb) * bsr_dim + nnzb * bsr_dim * bsr_dim + (beta ? mb * bsr_dim : 0))
                  * sizeof(T))
           / 1e9;
}

template <typename T>
constexpr double csrsv_gbyte_count(aoclsparse_int M, aoclsparse_int nnz)
{
    return ((M + 1 + nnz) * sizeof(aoclsparse_int) + (M + M + nnz) * sizeof(T)) / 1e9;
}

/*
 * ===========================================================================
 *    level 3 SPARSE
 * ===========================================================================
 */
template <typename T>
constexpr double csrmm_gbyte_count(aoclsparse_int M,
                                   aoclsparse_int nnz_A,
                                   aoclsparse_int nnz_B,
                                   aoclsparse_int nnz_C,
                                   bool           beta = false)
{
    return (((M + 1) * sizeof(aoclsparse_int)) + (nnz_A * sizeof(aoclsparse_int))
            + ((nnz_A + nnz_B + nnz_C + (beta ? nnz_C : 0)) * sizeof(T)))
           / 1e9;
}

template <typename T>
constexpr double csr2m_gbyte_count(aoclsparse_int                  M,
                                   [[maybe_unused]] aoclsparse_int N,
                                   aoclsparse_int                  K,
                                   aoclsparse_int                  nnz_A,
                                   aoclsparse_int                  nnz_B,
                                   aoclsparse_int                  nnz_C)
{
    double size_A = ((M + 1.0) * sizeof(aoclsparse_int)) + (nnz_A * sizeof(aoclsparse_int))
                    + (nnz_A * sizeof(T));
    double size_B = ((K + 1.0) * sizeof(aoclsparse_int)) + (nnz_B * sizeof(aoclsparse_int))
                    + (nnz_B * sizeof(T));
    double size_C = ((M + 1.0) * sizeof(aoclsparse_int)) + (nnz_C * sizeof(aoclsparse_int))
                    + (nnz_C * sizeof(T));
    return (size_A + size_B + size_C) / 1e9;
}

template <typename T>
constexpr double csr_add_gbyte_count(aoclsparse_int M, aoclsparse_int nnz_A, aoclsparse_int nnz_B)
{
    return ((3 * (M + 1) * sizeof(aoclsparse_int))
            + (((nnz_A + nnz_B) * sizeof(aoclsparse_int)) + ((nnz_A + nnz_B) * sizeof(T))))
           / 1e9;
}

/*
 * ===========================================================================
 *    SPARSE Solvers
 * ===========================================================================
 */
template <typename T>
constexpr double csrilu0_gbyte_count(aoclsparse_int M, aoclsparse_int nnz)
{
    return ((M + 1 + nnz) * sizeof(aoclsparse_int) + 2.0 * nnz * sizeof(T)) / 1e9;
}
#endif // AOCLSPARSE_GBYTE_HPP
