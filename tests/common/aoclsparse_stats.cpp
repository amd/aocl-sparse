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
 * ************************************************************************ */

#include "aoclsparse.h"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_stats.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/statistics/t_test.hpp>
#include <boost/math/statistics/z_test.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

// Compute statistics on data[n] array, as a side product, data[] gets sorted
int compute_stats(double *data, int n, data_stats &dstats)
{
    if(n <= 0)
        return -1;

    dstats.n = n;
    if(n == 1)
    {
        dstats.mean   = data[0];
        dstats.stdev  = 0.;
        dstats.min    = data[0];
        dstats.q1     = data[0];
        dstats.median = data[0];
        dstats.q3     = data[0];
        dstats.max    = data[0];
        return 0;
    }

    // For quantiles we need sorting, we wouldn't need to sort the array fully
    // and search for k-th smallest element instead but for the current
    // sizes it doesn't matter.
    std::sort(data, data + n);

    double h_median = (n + 1) * 0.5 - 1.0;
    double h_upper  = (n + 1) * 0.75 - 1.0;
    double h_lower  = (n + 1) * 0.25 - 1.0;

    int h_median_floor = std::clamp((int)std::floor(h_median), 0, n - 1);
    int h_median_ceil  = std::clamp((int)std::ceil(h_median), 0, n - 1);
    int h_upper_floor  = std::clamp((int)std::floor(h_upper), 0, n - 1);
    int h_upper_ceil   = std::clamp((int)std::ceil(h_upper), 0, n - 1);
    int h_lower_floor  = std::clamp((int)std::floor(h_lower), 0, n - 1);
    int h_lower_ceil   = std::clamp((int)std::ceil(h_lower), 0, n - 1);

    dstats.min = data[0];
    dstats.q1  = data[h_lower_floor]
                + (h_lower - h_lower_floor) * (data[h_lower_ceil] - data[h_lower_floor]);
    dstats.median = data[h_median_floor]
                    + (h_median - h_median_floor) * (data[h_median_ceil] - data[h_median_floor]);
    dstats.q3 = data[h_upper_floor]
                + (h_upper - h_upper_floor) * (data[h_upper_ceil] - data[h_upper_floor]);
    dstats.max = data[n - 1];

    // compute average and standard deviation
    double dsum = 0., dev = 0.;
    for(int i = 0; i < n; i++)
        dsum += data[i];
    dstats.mean = dsum / n;
    for(int i = 0; i < n; i++)
        dev += (data[i] - dstats.mean) * (data[i] - dstats.mean);
    dstats.stdev = sqrt(dev / n);

    return 0;
}

twosample_test_result
    twosample_test(const data_stats &a, const data_stats &b, double alpha, int ztest)
{
    twosample_test_result result;
    using namespace boost::math;

    double alph{alpha};
    if(alpha <= 0 || alpha >= 1)
        alph = 0.95;

    const double diff = a.mean - b.mean;
    const size_t dof  = a.n + b.n - 2;

    // Copy input
    result.alpha = alph;
    result.diff  = diff;
    result.dof   = dof;

    const double avar         = a.stdev * a.stdev;
    const double bvar         = b.stdev * b.stdev;
    const double significance = 1 - alpha;
    const double Sp           = ((a.n - 1) * avar + (b.n - 1) * bvar) / dof;
    const double SE           = 1. / a.n + 1. / b.n;

    result.normal = ztest > 0;
    if(ztest < 0)
        result.normal = (a.n >= 30 && b.n >= 30);

    std::pair<double, double> ret;
    double                    t;
    result.t_welch = false;

    if(result.normal)
    {
        normal zdist;
        t = quantile(complement(zdist, significance / 2.));
        // perform two-sample z-test
        ret = statistics::detail::two_sample_z_test_impl<decltype(ret)>(
            a.mean, avar, double(a.n), b.mean, bvar, double(b.n));
        // P-value from ret is for some reason wrong. Use standard formula to get it
        ret.second = 2. * (1. - cdf(zdist, std::abs(ret.first))); // p-value
    }
    else
    {
        students_t tdist(dof);
        t = quantile(complement(tdist, significance / 2.));
        // perform two-sample t-test
        if(a.stdev > 2 * b.stdev || b.stdev > 2 * a.stdev)
        {
            // Variance are not similar, change and use Welch's t-test
            ret = statistics::detail::welchs_t_test_impl<decltype(ret)>(
                a.mean, avar, double(a.n), b.mean, bvar, double(b.n));
            result.t_welch = true;
        }
        else
        {
            ret = statistics::detail::two_sample_t_test_impl<decltype(ret)>(
                a.mean, avar, double(a.n), b.mean, bvar, double(b.n));
        }
    }

    result.v      = ret.first; // test_statistic;
    result.pvalue = ret.second;

    // Test result
    result.result = result.pvalue >= significance; // if true then A=B

    // build confidence interval
    double ci_width = t * sqrt(Sp) * sqrt(SE);
    result.ci[0]    = diff - ci_width; // lower
    result.ci[1]    = diff + ci_width; // upper

    // potential speed-up
    result.speedup = b.mean / a.mean;

    return result;
}

void twosample_print(const twosample_test_result &cmp,
                     const char                  *nameA,
                     const data_stats            &a,
                     const char                  *nameB,
                     const data_stats            &b,
                     bool                         debug,
                     bool                         onehdr)
{
    using namespace std::literals;
    if(debug)
    {
        std::string ttchar;
        if(cmp.normal)
            ttchar = "z"s;
        else
            ttchar = (cmp.t_welch ? "w"s : "t"s);

        std::cout << "two-sample mean test" << std::endl;
        std::cout << "A: " << nameA << std::endl;
        std::cout << "B: " << nameB << std::endl;
        std::cout << "H0: mean(A) = mean(B)" << std::endl;
        std::cout << "H1: mean(A) /= mean(B)" << std::endl << std::endl;
        std::cout << "| stats      |   Grp A(n=" << a.n << ") |   Grp B(n=" << b.n << ") |      "
                  << (cmp.normal ? "   Normal" : "t-Student") << "|" << std::endl;
        std::cout << "|:-----------|--------------:|--------------:|--------------:|" << std::endl;
        std::cout << std::fixed;
        std::cout << "| mean       |" << std::setw(15) << a.mean << "|" << std::setw(15) << b.mean
                  << "|               |" << std::endl;
        std::cout << "| stdev      |" << std::setw(15) << a.stdev << "|" << std::setw(15) << b.stdev
                  << "|               |" << std::endl;
        std::cout << "| Err(A-B)   |               |               |" << std::setw(15) << cmp.diff
                  << "|" << std::endl;
        std::cout << "| " << ttchar + "-stats    |               |               |"s
                  << std::setw(15) << cmp.v << "|" << std::endl;
        std::cout << "| P-value    |               |               |" << std::setw(15) << cmp.pvalue
                  << "|" << std::endl;
        std::cout << "| CI(" << std::setw(4) << std::setprecision(2) << cmp.alpha
                  << ") L |               |               |" << std::setprecision(6)
                  << std::setw(15) << cmp.ci[0] << "|" << std::endl;
        std::cout << "| CI(" << std::setw(4) << std::setprecision(2) << cmp.alpha
                  << ") U |               |               |" << std::setprecision(6)
                  << std::setw(15) << cmp.ci[1] << "|" << std::endl;
        return;
    }

    const unsigned int d{1 + 8};

    if(onehdr)
    {
        // clang-format off
        std::cout
        << std::setw(16)  << "A:         name"
        << std::setw(9)   << "n"
        << std::setw(d)   << "mean"
        << std::setw(d)   << "stdev" << "|"
        << std::setw(16)  << "B:         name"
        << std::setw(9)   << "n"
        << std::setw(d)   << "mean"
        << std::setw(d)   << "stdev" << "|"
        << std::setw(d)   << "(A-B)"
        << std::setw(d)   << "ci(low"
        << std::setw(d)   << "upper)"
        << std::setw(d)   << "signce"
        << std::setw(d)   << "p-val"
        << std::setw(1+5) << "A==B"
        << std::endl;
        // clang-format on
    }
    // clang-format off
    std::cout << std::fixed << std::setprecision(0)
    << std::setw(16)  << nameA
    << std::setw(9)   << a.n     << std::scientific << std::setprecision(1)
    << std::setw(d)   << a.mean
    << std::setw(d)   << a.stdev << "|"
    << std::setw(16)  << nameB  << std::fixed << std::setprecision(0)
    << std::setw(9)   << b.n     << std::scientific << std:: setprecision(1)
    << std::setw(d)   << b.mean
    << std::setw(d)   << b.stdev << "|"
    << std::setw(d)   << cmp.diff
    << std::setw(d)   << cmp.ci[0]
    << std::setw(d)   << cmp.ci[1] << std::fixed << std::setprecision(4)
    << std::setw(d)   << 1. - cmp.alpha << std::setprecision(5)
    << std::setw(d)   << cmp.pvalue
    << std::setw(1+5) << (cmp.result ? "PASS" : "FAIL")
    << std::endl;
    // clang-format on
}

void print_results(const char                  *test_name,
                   const char                  *prob_name,
                   int                          verify,
                   const data_stats            &tstats,
                   const twosample_test_result *cmp,
                   bool                         onehdr)
{
    const char *ver_names[] = {"??", "OK", "F "};
    const char *sep         = ",";

    if(onehdr)
    {
        // clang-format off
        std::cout
        << std::setw(2)  << " "
        << std::setw(40)  << "name" << sep
        << std::right
        << std::setw(3)  << "ver" << sep
        << std::setw(10) << "mean" << sep
        << std::setw(10) << "stdev"  << sep
        << std::setw(5)  << "n"  << sep
        << std::setw(10) << "min"  << sep
        << std::setw(10) << "q1" << sep
        << std::setw(10) << "median" << sep
        << std::setw(10) << "q3" << sep
        << std::setw(10) << "max" << sep;
        if (true || cmp) {
          std::cout
          << std::setw(8)   << "(A-B)" <<sep
          << std::setw(8)   << "ci(low" <<sep
          << std::setw(8)   << "upper)" << sep
          << std::setw(5)   << "sgnc" <<sep
          << std::setw(7)   << "p-val" << sep
          << std::setw(1+5) << "A==B" << sep
          << std::setw(7) << "A/B" << sep;
        }
        std::cout << std::endl;
        // clang-format on
    }
    std::string full_name = std::string(test_name) + ":" + prob_name;
    // clang-format off
    std::cout << std::fixed << std::setprecision(0) << std::left
    << "* "
    << std::setw(40)  << full_name << sep
    << std::right
    << std::setw(3)  << ver_names[verify] << sep
    << std::scientific << std::setprecision(2)
    << std::setw(10) << tstats.mean << sep
    << std::setw(10) << tstats.stdev  << sep
    << std::setw(5)  << tstats.n  << sep
    << std::setw(10) << tstats.min  << sep
    << std::setw(10) << tstats.q1 << sep
    << std::setw(10) << tstats.median << sep
    << std::setw(10) << tstats.q3 << sep
    << std::setw(10) << tstats.max << sep;
    if (cmp) {
      std::cout << std::scientific << std::setprecision(1)
      << std::setw(8)   << cmp->diff <<sep
      << std::setw(8)   << cmp->ci[0] <<sep
      << std::setw(8)   << cmp->ci[1] << sep
      <<std::fixed << std::setprecision(2)
      << std::setw(5)   << 1. - cmp->alpha <<sep
      << std::setw(7)   << cmp->pvalue << sep
      << std::setw(1+5) << (cmp->result ? "same" : "DIFF") << sep
      << std::setw(7) <<std::fixed << std::setprecision(1)
      << ((false && cmp->result) ? 1. : cmp->speedup) << sep;
    }
    std::cout << std::endl;
}
