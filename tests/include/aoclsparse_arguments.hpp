/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.
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
 *  \brief aoclsparse_arguments.hpp provides a class to parse command arguments in clients
 */

#pragma once
#ifndef AOCLSPARSE_ARGUMENTS_HPP
#define AOCLSPARSE_ARGUMENTS_HPP

#include "aoclsparse.h"
#include "aoclsparse_datatype2string.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

struct Arguments
{
    aoclsparse_int M;
    aoclsparse_int N;
    aoclsparse_int K;
    aoclsparse_int nnz;
    aoclsparse_int blk;
    aoclsparse_int block_dim;

    double alpha;
    double beta;

    aoclsparse_operation   transA;
    aoclsparse_operation   transB;
    aoclsparse_matrix_type mattypeA;
    aoclsparse_index_base  baseA;
    aoclsparse_diag_type   diag;
    aoclsparse_fill_mode   uplo;
    aoclsparse_order       order;
    aoclsparse_int         stage;

    aoclsparse_matrix_init matrix;

    aoclsparse_int unit_check;
    aoclsparse_int timing;
    aoclsparse_int iters;

    std::string filename;
    char        function[64];

private:
    // Function to read Structures data from stream
    friend std::istream &operator>>(std::istream &str, Arguments &arg)
    {
        str.read(reinterpret_cast<char *>(&arg), sizeof(arg));
        return str;
    }

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream &str, const T &x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream &str, double x)
    {
        if(std::isnan(x))
            str << ".nan";
        else if(std::isinf(x))
            str << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char *end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream &str, char c)
    {
        char s[]{c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream &str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream &str, const char *s)
    {
        str << std::quoted(s);
    }
};

/* Common data used for API benchmarking
 *
 * This structure is used across all APIs, it comprises of the input data
 * (e.g., input matrix A and vectors x and y for SPMV alpha*A*x+beta*y)
 * and space for the outputs (results) which then might be checked against
 * the reference results. It works in conjunction with Arguments
 * which specifies exactly what operation (e.g., transposed) and descriptor
 * is used. The sizes are already adjusted to match the operations.
 */
template <typename T>
struct testdata
{
    // problem sizes, e.g., for non-transposed SPMV the dimension of A is m x n.
    aoclsparse_int m;
    aoclsparse_int n;
    aoclsparse_int k;

    // A matrix - Used for all APIs
    aoclsparse_int              nnzA;
    std::vector<aoclsparse_int> csr_row_ptrA;
    std::vector<aoclsparse_int> csr_col_indA;
    std::vector<T>              csr_valA;

    // B matrix - Used for spadd, 2m, sypr
    aoclsparse_int              nnzB;
    std::vector<aoclsparse_int> csr_row_ptrB;
    std::vector<aoclsparse_int> csr_col_indB;
    std::vector<T>              csr_valB;

    // multipliers
    T alpha;
    T beta;

    // vectors compatible with (including sizes): op(A) . x = y
    std::vector<T> x; // input
    std::vector<T> y_in; // input y
    std::vector<T> y; // result of this computation
};

/* Type for any test function to be added to the testqueue, either internal (AOCL)
 * or for external benchmarking.
 * Arguments:
 * - arg [input] - to be used to understand the parameters of the tests
 *     (descriptor, transpose, etc.)
 * - td [input/output] - generated data (such as matrices and input vectors),
 *     these shouldn't be changed by the individual test, and workspace and
 *     output data. The last result of the computation should be preserved
 *     to be check against reference results.
 * - timings[arg.iter] - cpu clock measurements for each computation over
 *     the arg.iter iterations
 * Returns: 0 on success, an error otherwise
 */
template <typename T>
using testfunc = int (*)(const Arguments &arg, testdata<T> &td, double timings[]);

/* structure to join test name and test function into one element to build a test queue */
template <typename T>
struct testsetting
{
    const char *name; // name of the test for printing purposes
    testfunc<T> tf; // test function which runs the computation over several iterations
};

#endif // AOCLSPARSE_ARGUMENTS_HPP
