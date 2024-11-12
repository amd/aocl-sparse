/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#ifndef AOCLSPARSE_LAPACK_HPP
#define AOCLSPARSE_LAPACK_HPP

#include "aoclsparse_types.h"

#include <cmath>
#include <limits>

// clang-format off
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment" // for libflame header
#pragma GCC diagnostic ignored "-Wunused-parameter" // for blis header
#include "cblas.hh"
#include "libflame_interface.hh"
#pragma GCC diagnostic pop
// clang-format on

/* Check that the size of integers in the used libraries is OK. */
static_assert(
    sizeof(f77_int) == sizeof(aoclsparse_int),
    "Error: Incompatible size of ints in blis. Using wrong header or compilation of the library?");
static_assert(
    sizeof(integer) == sizeof(aoclsparse_int),
    "Error: Incompatible size of ints in flame. Using wrong header or compilation of the library?");

#endif //AOCLSPARSE_LAPACK_HPP
