/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.All rights reserved.
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
 *  \brief aoclsparse_test.cpp is the main test application which reads the
 *  command line arguments and verifies the specific aoclsparse API's
 *  and calculates flops and bandwdith data
 *  Example to invoke test bench for CSR-SPMV with standard test input:
 *  ./aoclsparse-bench --function=csrmv --precision=d
 *      --alpha=1 --beta=0 --iters=1000 --mtx=<matrix_market_file_name>
 *  Example to invoke test bench for CSR-SPMV on randomly generated matrix:
 *  ./aoclsparse-bench --function=csrmv --precision=d --sizem=1000
 *      --sizen=1000 --sizennz=4000 --verify=1
 */

#include "aoclsparse.h"

#include <complex>
// Level2
#include "testing_blkcsrmv.hpp"
#include "testing_bsrmv.hpp"
#include "testing_csrmv.hpp"
#include "testing_csrsv.hpp"
#include "testing_diamv.hpp"
#include "testing_ellmv.hpp"
#include "testing_optmv.hpp"
#include "testing_sycsrmv.hpp"

// Level3
#include "testing_csr2m.hpp"
#include "testing_csrmm.hpp"

//Solvers
#include "testing_ilu.hpp"

// Testing/validating the loading of complex data from an mtx file
// and converting it into a CSR representation
#include "testing_complex_mtx_load.hpp"

int main(int argc, char *argv[])
{
    Arguments arg;
    // default values
    arg.unit_check = 0;
    arg.iters      = 10;
    arg.M          = 128;
    arg.N          = 128;
    arg.K          = 128;
    arg.nnz        = 0;
    arg.blk        = 4;
    arg.block_dim  = 2;
    arg.alpha      = 1.0;
    arg.beta       = 0.0;
    arg.stage      = 0;
    char precision = 'd';
    char transA    = 'N';
    char transB    = 'N';
    char mattypeA  = 'G';
    int  baseA     = 0;
    char diag      = 'N';
    char uplo      = 'L';
    int  order     = 1;
    strcpy(arg.function, "csrmv");
    // Initialize command line
    aoclsparse_command_line_args args(argc, argv);

    if(args.aoclsparse_check_cmdline_flag("help") || (argc == 1))
    {
        printf(
            "aoclsparse test command line options:"
            "\n\t"
            "%s "
            "\n\t"
            " --help \t  produces this help message"
            "\n\t"
            "--sizem=<Number of rows> \t  m is only  applicable to SPARSE-2 & SPARSE-3: the number "
            " of rows (default: 128)"
            "\n\t"
            "--sizen=<Number of columns> \t  SPARSE-1:  the length of the dense vector. SPARSE-2 & "
            "SPARSE-3: the number of columns (default: 128)"
            "\n\t"
            "--sizek=<Number of columns> \t  SPARSE-2 & SPARSE-3: the number of columns (default: "
            "128)"
            "\n\t"
            "--sizennz=<Number of non-zeroes> \t  Number of the non-zeroes in sparse matrix/vector"
            "\n\t"
            "--sizeblk=<Blocking factor> \t  Specifies the size of blocking for blkcsr (default: 4)"
            "\n\t"
            "--blockdim=<block dimension> \t  block dimension for bsrmv (default: 2)"
            "\n\t"
            "--mtx=<matrix market (.mtx)> \t  Read from matrix market (.mtx) format. This will "
            "override parameters -sizem, -sizen, and -sizennz."
            "\n\t"
            "--alpha=<scalar alpha> \t Specifies the scalar alpha (default: 1.0)"
            "\n\t"
            "--beta=<scalar beta> \t Specifies the scalar beta (default: 0.0)"
            "\n\t"
            "--transposeA=<N/T> \t N = no transpose, T = transpose (default: N)"
            "\n\t"
            "--transposeB=<N/T> \t N = no transpose, T = transpose (default: N)"
            "\n\t"
            "--matrixtypeA=<G/S/T> \t G = general (use whole matrix), S = symmetric (use one "
            "triangle based on uplo & symmetrize), T = triangular (use one triangle based on uplo) "
            "(default: G)"
            "\n\t"
            "--indexbaseA=<0/1> \t 0 = zero-based indexing, 1 = one-based indexing (default: 0)"
            "\n\t"
            "--diag=<N/U> \t N = non-unit diagonal, U = unit diagonal (default = N)"
            "\n\t"
            "--uplo=<L/U> \t L = lower fill, U = upper fill (default = L)"
            "\n\t"
            "--function=<function to test> \t SPARSE function to test. Options:  Level2: csrmv "
            "optmv blkcsrmv ellmv diamv bsrmv csrsv Level3: csrmm csr2m ilu (default: csrmv)"
            "\n\t"
            "--precision=<s/d> \t Options: s,d (default: d)"
            "\n\t"
            "--verify=<0/1> \t Validate results ? 0 = No, 1 = Yes (default: No)"
            "\n\t"
            "--iters=<num of iterations> \t Iterations to run inside timing loop (default: 10)"
            "\n\t"
            "--order=<0/1> \t Indicates whether a dense matrix is laid out in column-major "
            "storage: 1, or row-major storage 0 (default: 1)"
            "\n\t"
            "--stage=<0/1> \t Indicates whether csr2m routine performs in single stage: 0 "
            "or double stage: 1 (default: 0)"
            "\n",
            argv[0]);

        return 0;
    }
    args.aoclsparse_get_cmdline_argument("sizem", arg.M);
    args.aoclsparse_get_cmdline_argument("sizen", arg.N);
    args.aoclsparse_get_cmdline_argument("sizek", arg.K);
    args.aoclsparse_get_cmdline_argument("sizennz", arg.nnz);
    args.aoclsparse_get_cmdline_argument("sizeblk", arg.blk);
    args.aoclsparse_get_cmdline_argument("blockdim", arg.block_dim);
    args.aoclsparse_get_cmdline_argument("mtx", arg.filename);
    args.aoclsparse_get_cmdline_argument("alpha", arg.alpha);
    args.aoclsparse_get_cmdline_argument("beta", arg.beta);
    args.aoclsparse_get_cmdline_argument("transposeA", transA);
    args.aoclsparse_get_cmdline_argument("transposeB", transB);
    args.aoclsparse_get_cmdline_argument("matrixtypeA", mattypeA);
    args.aoclsparse_get_cmdline_argument("indexbaseA", baseA);
    args.aoclsparse_get_cmdline_argument("diag", diag);
    args.aoclsparse_get_cmdline_argument("uplo", uplo);
    args.aoclsparse_get_cmdline_argument("function", arg.function);
    args.aoclsparse_get_cmdline_argument("precision", precision);
    args.aoclsparse_get_cmdline_argument("verify", arg.unit_check);
    args.aoclsparse_get_cmdline_argument("iters", arg.iters);
    args.aoclsparse_get_cmdline_argument("order", order);
    args.aoclsparse_get_cmdline_argument("stage", arg.stage);

    if(precision != 's' && precision != 'd' && precision != 'c' && precision != 'z')
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }

    arg.transA = aoclsparse_operation_none;
    if(transA == 'N')
    {
        arg.transA = aoclsparse_operation_none;
    }
    else if(transA == 'T')
    {
        arg.transA = aoclsparse_operation_transpose;
    }

    arg.transB = aoclsparse_operation_none;
    if(transB == 'N')
    {
        arg.transB = aoclsparse_operation_none;
    }
    else if(transB == 'T')
    {
        arg.transB = aoclsparse_operation_transpose;
    }
    if(mattypeA == 'G')
    {
        arg.mattypeA = aoclsparse_matrix_type_general;
    }
    else if(mattypeA == 'S')
    {
        arg.mattypeA = aoclsparse_matrix_type_symmetric;
    }
    else if(mattypeA == 'T')
    {
        arg.mattypeA = aoclsparse_matrix_type_triangular;
    }
    else if(mattypeA == 'H')
    {
        arg.mattypeA = aoclsparse_matrix_type_hermitian;
    }
    else
    {
        std::cerr << "Invalid value for --matrixtypeA" << std::endl;
        return -1;
    }

    arg.baseA = (baseA == 0) ? aoclsparse_index_base_zero : aoclsparse_index_base_one;
    arg.diag  = (diag == 'N') ? aoclsparse_diag_type_non_unit : aoclsparse_diag_type_unit;
    arg.uplo  = (uplo == 'L') ? aoclsparse_fill_mode_lower : aoclsparse_fill_mode_upper;
    arg.order = (order == 1) ? aoclsparse_order_column : aoclsparse_order_row;

    if(arg.filename != "")
    {
        arg.matrix = aoclsparse_matrix_file_mtx;
    }
    else
    {
        arg.matrix = aoclsparse_matrix_random;
    }

    /* ============================================================================================
     */
    if(arg.M < 0 || arg.N < 0)
    {
        std::cerr << "Invalid dimension" << std::endl;
        return -1;
    }
    if(strcmp(arg.function, "csrmv") == 0)
    {
        if(precision == 's')
            testing_csrmv<float>(arg);
        else if(precision == 'd')
            testing_csrmv<double>(arg);
        else if(precision == 'c')
            testing_csrmv<aoclsparse_float_complex>(arg);
        else if(precision == 'z')
            testing_csrmv<aoclsparse_double_complex>(arg);
    }
    else if(strcmp(arg.function, "load") == 0)
    {
        if(precision == 'c')
            // FIXME: need to support std:complex<float/double>
            // FIXME: need to support aoclsparse_double_complex
            testing_complex_mtx_load<aoclsparse_float_complex>(arg);
    }
    else if(strcmp(arg.function, "blkcsrmv") == 0)
    {
        // FIXME: don't run these tests if CPU does not
        // have AVX-512 flags runnable.
        if(precision == 'd' && aoclsparse_get_vec_extn_context())
            testing_blkcsrmv<double>(arg);
    }
    else if(strcmp(arg.function, "ellmv") == 0)
    {
        if(precision == 's')
            testing_ellmv<float>(arg);
        else if(precision == 'd')
            testing_ellmv<double>(arg);
    }
    else if(strcmp(arg.function, "optmv") == 0)
    {
        if(precision == 's')
            testing_optmv<float>(arg);
        else if(precision == 'd')
            testing_optmv<double>(arg);
    }
    else if(strcmp(arg.function, "diamv") == 0)
    {
        if(precision == 's')
            testing_diamv<float>(arg);
        else if(precision == 'd')
            testing_diamv<double>(arg);
    }
    else if(strcmp(arg.function, "bsrmv") == 0)
    {
        if(precision == 's')
            testing_bsrmv<float>(arg);
        else if(precision == 'd')
            testing_bsrmv<double>(arg);
    }
    else if(strcmp(arg.function, "csrsv") == 0)
    {
        if(precision == 'd')
            testing_csrsv<double>(arg);
        else if(precision == 's')
            testing_csrsv<float>(arg);
    }
    else if(strcmp(arg.function, "csrmm") == 0)
    {
        if(precision == 'd')
            testing_csrmm<double>(arg);
        else if(precision == 's')
            testing_csrmm<float>(arg);
    }
    else if(strcmp(arg.function, "csr2m") == 0)
    {
        if(precision == 'd')
            testing_csr2m<double>(arg);
        else if(precision == 's')
            testing_csr2m<float>(arg);
    }
    else if(strcmp(arg.function, "ilu") == 0)
    {
        if(precision == 'd')
            testing_ilu<double>(arg);
        else if(precision == 's')
            testing_ilu<float>(arg);
    }
    else
    {
        std::cerr << "Invalid value for --function" << std::endl;
        return -1;
    }
    return 0;
}
