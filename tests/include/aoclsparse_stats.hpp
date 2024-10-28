/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.All rights reserved.
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
 *  \brief aoclsparse_utility.hpp provides common utilities
 */

#pragma once
#ifndef AOCLSPARSE_STATS_HPP
#define AOCLSPARSE_STATS_HPP

#include "aoclsparse.h"
#include "aoclsparse_reference.hpp"

#include <algorithm>
#include <cfloat>
#include <sstream>
#include <string>
#include <vector>

// Hold all computed statistics of one testing_* run
struct data_stats
{
    int    n; // sample size
    double mean;
    double stdev; // std. deviation
    double min;
    double q1; // first quartile (25th percentile)
    double median; // the middle value
    double q3; // third quartile (75th percentile)
    double max;
};

// Compute statistics on data[n] array, as a side product, data[] gets sorted
int compute_stats(double *data, int n, data_stats &dstats);

// generate a representative problem name
// Note: 'td' is needed only for updated problem sizes
template <typename T>
std::string gen_problem_name(const Arguments &arg, const testdata<T> &td)
{

    std::string ret = arg.function;
    if(arg.transA == aoclsparse_operation_transpose)
        ret += "T";
    else if(arg.transA == aoclsparse_operation_conjugate_transpose)
        ret += "H";
    // TODO add other modifiers (arg.mattypeA, .uplo, .diag)
    // add type name
    const char *type_names[] = {"-d-", "-s-", "-c-", "-z-"};
    ret += type_names[get_data_type<T>()];
    // add 0/1-base
    const char *base_names[] = {"B0:", "B1:"};
    ret += base_names[arg.baseA];

    if(arg.matrix == aoclsparse_matrix_random)
    {
        ret += "rnd" + std::to_string(td.m) + "x" + std::to_string(td.n) + "x"
               + std::to_string(td.nnzA);
    }
    else if(arg.matrix == aoclsparse_matrix_file_mtx)
    {
        // strip slashes and .mtx from the filename
        size_t start = arg.filename.rfind("/");
        start        = start == std::string::npos ? 0 : start + 1;
        size_t stop  = arg.filename.rfind(".mtx");
        size_t len   = stop == std::string::npos ? std::string::npos : stop - start;
        ret          = ret + arg.filename.substr(start, len);
        ret += "x" + std::to_string(td.m) + "x" + std::to_string(td.n) + "x"
               + std::to_string(td.nnzA);
    }
    else if(arg.matrix == aoclsparse_matrix_random_diag_dom)
    {
        ret += "rnd_diag_dom" + std::to_string(td.m) + "x" + std::to_string(td.n) + "x"
               + std::to_string(td.nnzA);
    }
    else if(arg.matrix == aoclsparse_matrix_herm_random_diag_dom)
    {
        ret += "rnd_herm_diag_dom" + std::to_string(td.m) + "x" + std::to_string(td.n) + "x"
               + std::to_string(td.nnzA);
    }
    else
        ret += "ERR-unknown_source";

    return ret;

    // TODO extend for Level3 for the other matrix and Level1 to kick out the matrix
    /* to use
  arg.function  (csrmv, dotmv, blkcsrmv
            "--function=<function to test> \t SPARSE function to test. (default: csrmv) Options:  "
            "\n\t\tLevel-1: gthr gthrz sctr axpyi roti doti dotui dotci"
            "\n\t\tLevel-2: csrmv optmv blkcsrmv(only precision=d) ellmv diamv bsrmv trsv dotmv"
            "\n\t\tLevel-3: csrmm csr2m sp2md"
  */
}

struct twosample_test_result
{
    double alpha; // 0 < alpha < 1 -> significance = 1 - alpha
    double diff; // a - b
    double ci[2]; // confidence interval for a-b
    double v; // test's statistic
    double pvalue; // test's p-value
    size_t dof; // degrees of freedom (na + nb - 2)
    bool   normal; // used z-test or t-test?
    bool   result; // true => Ho not rejected
    bool   t_welch; // was the t-Student replaced with unequal variance Welch's t-test?
    double speedup; // if A!=B, this gives the speed-up based on the means
};

twosample_test_result
    twosample_test(const data_stats &a, const data_stats &b, double alpha = 0.95, int ztest = -1);

void twosample_print(const twosample_test_result &cmp,
                     const char                  *nameA,
                     const data_stats            &a,
                     const char                  *nameB,
                     const data_stats            &b,
                     bool                         debug  = false,
                     bool                         onehdr = true);

void print_results(const char                  *test_name,
                   const char                  *prob_name,
                   int                          verify,
                   const data_stats            &tstats,
                   const twosample_test_result *cmp,
                   bool                         onehdr);

#endif // AOCLSPARSE_STATS_HPP
