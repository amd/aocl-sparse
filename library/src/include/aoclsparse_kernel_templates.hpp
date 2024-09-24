/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_KERNEL_TEMPLATES_T_HPP
#define AOCLSPARSE_KERNEL_TEMPLATES_T_HPP

#include "aoclsparse.h"

// Set the kt_int
using kt_int_t = aoclsparse_int;

#include "kernel-templates/kernel_templates.hpp"

// Helper functions and macros that assist in instantiation of KT-based functions
// ------------------------------------------------------------------------------

// Instantiates a kernel template function defined by "FUNC"
#define KT_INSTANTIATE(FUNC, BSZ)   \
    FUNC(BSZ, float);               \
    FUNC(BSZ, double);              \
    FUNC(BSZ, std::complex<float>); \
    FUNC(BSZ, std::complex<double>);

// Generates extern declaration for a given Kernel templates function
// defined by "FUNC"
#ifdef USE_AVX512
#define EXTERN_DECLARE(FUNC)                                        \
    extern FUNC(kernel_templates::bsz::b256, float);                \
    extern FUNC(kernel_templates::bsz::b256, double);               \
    extern FUNC(kernel_templates::bsz::b256, std::complex<float>);  \
    extern FUNC(kernel_templates::bsz::b256, std::complex<double>); \
    extern FUNC(kernel_templates::bsz::b512, float);                \
    extern FUNC(kernel_templates::bsz::b512, double);               \
    extern FUNC(kernel_templates::bsz::b512, std::complex<float>);  \
    extern FUNC(kernel_templates::bsz::b512, std::complex<double>);
#else
#define EXTERN_DECLARE(FUNC)                                       \
    extern FUNC(kernel_templates::bsz::b256, float);               \
    extern FUNC(kernel_templates::bsz::b256, double);              \
    extern FUNC(kernel_templates::bsz::b256, std::complex<float>); \
    extern FUNC(kernel_templates::bsz::b256, std::complex<double>);
#endif

//--------------------------------------------------------------------------------

#endif // AOCLSPARSE_KERNEL_TEMPLATES_T_HPP
