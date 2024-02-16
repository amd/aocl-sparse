/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */

/*
 * Reusable vectorized kernel template library (KT)
 * ================================================
 *
 * SYNOPSYS
 * --------
 *
 * Micro kernels are classified on two levels:
 *  - Level 0 micro kernels, these expand directly to IMM intrinsic instructions, while
 *  - Level 1 micro kernels, make use of one or more level 0 micro kernels.
 *
 * For a simple implementation of BLAS Level 2 SPMV solver instantiated 8 times, see
 * ktlvl2_test.cpp
 * For BLAS Level 2 TRSV solver using this template see library/src/level2/aoclsparse_trsv.hpp
 */

#ifndef KERNEL_TEMPLATES_T_HPP
#define KERNEL_TEMPLATES_T_HPP

// L0 micro kernels that are generic
#include "kt_l0.hpp"
// L0 micro kernels AVX 2 specific
#include "kt_l0_avx2.hpp"

// L1 micro kernels: these only depend on L0 micro kernels.
#include "kt_l0_avx2.hpp"

#ifdef __AVX512F__
// L0 microkernels (AVX512)
#include "kt_l0_avx512.hpp"
#endif

// L1 micro kernels: these only depend on L0 micro kernels.
#include "kt_l1.hpp"

#endif // KERNEL_TEMPLATES_T_HPP
