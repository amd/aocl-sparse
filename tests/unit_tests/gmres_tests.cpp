/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "gtest/gtest.h"
#include "gmres_ut_functions.hpp"

namespace
{

    using namespace std;

    // GMRES Data Driven Test
    // Structure to store all parameters needed for one test
    typedef struct
    {
        const char *
            testname; // unique identifier of the test, use only 0-9a-zA-Z (particularly no undescores!)
        aoclsparse_status
            status_exp; // expected return status (for positive tests typically aoclsparse_status_success)
        matrix_id               mid; // matrix to use
        PrecondType<float>      precondf; // preconditioner used in float tests
        PrecondType<double>     precondd; // preconditioner used in double tests
        MonitType<float>        monitf; // monitoring function for float
        MonitType<double>       monitd; // monitoring function for double
        std::vector<itsol_opts> opts; // pairs of option & value if any, for none use {}
    } GmresDDParamType;
    // List of all desired tests
    const GmresDDParamType GMRESTestValues[] = {
        {"GmresNoPrecondNullptr",
         aoclsparse_status_success,
         sample_gmres_mat_01,
         nullptr,
         nullptr,
         nullptr,
         nullptr,
         {{"iterative method", "GMRES"}, {"gmres restart iterations", "7"}}},
        {"GmresNoPrecond",
         aoclsparse_status_success,
         sample_gmres_mat_01,
         precond_dummy<float>,
         precond_dummy<double>,
         monit_print<float>,
         monit_print<double>,
         {{"iterative method", "GMRES"}, {"gmres restart iterations", "7"}}},
        {"GmresIlu0Precond",
         aoclsparse_status_success,
         sample_gmres_mat_01,
         precond_dummy<float>,
         precond_dummy<double>,
         monit_dummy<float>,
         monit_dummy<double>,
         {{"iterative method", "GMRES"},
          {"gmres restart iterations", "7"},
          {"gmres preconditioner", "ILU0"},
          {"gmres iteration limit", "50"}}},
        {"GmresUserPrecond",
         aoclsparse_status_success,
         sample_gmres_mat_01,
         precond_identity<float>,
         precond_identity<double>,
         monit_dummy<float>,
         monit_dummy<double>,
         {{"iterative method", "GMRES"},
          {"gmres restart iterations", "7"},
          {"gmres preconditioner", "User"},
          {"gmres iteration limit", "50"}}},
        {"GmresMonitStop",
         aoclsparse_status_user_stop,
         sample_gmres_mat_01,
         nullptr,
         nullptr,
         monit_tolstop<float>,
         monit_tolstop<double>,
         {{"iterative method", "GMRES"},
          {"gmres restart iterations", "7"},
          {"gmres rel tolerance", "0"},
          {"gmres iteration limit", "50"}}},
        {"GmresMaxIterReached",
         aoclsparse_status_maxit,
         sample_gmres_mat_01,
         nullptr,
         nullptr,
         nullptr,
         nullptr,
         {{"iterative method", "GMRES"},
          {"gmres restart iterations", "2"},
          {"gmres iteration limit", "2"}}},
        {"GmresUsrMonitStopIt2",
         aoclsparse_status_user_stop,
         sample_gmres_mat_01,
         nullptr,
         nullptr,
         monit_stopit2<float>,
         monit_stopit2<double>,
         {{"iterative method", "GMRES"},
          {"gmres Iteration Limit", "20"},
          {"gmres restart iterations", "2"}}},
        {"GmresUserPrecondStop",
         aoclsparse_status_user_stop,
         sample_gmres_mat_01,
         precond_dummy<float>,
         precond_dummy<double>,
         nullptr,
         nullptr,
         {{"iterative method", "GMRES"},
          {"gmres restart iterations", "7"},
          {"gmres preconditioner", "User"}}},
        {"GmresNormalTestCase",
         aoclsparse_status_success,
         sample_gmres_mat_02,
         nullptr,
         nullptr,
         nullptr,
         nullptr,
         {{"iterative method", "GMRES"},
          {"gmres preconditioner", "ILU0"},
          {"gmres rel tolerance", "0"},
          {"gmres iteration limit", "50"},
          {"gmres restart iterations", "7"}}},
    };
    // Teach GTest how to print CGDDParamType
    // in this case use only user's unique testname
    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const GmresDDParamType &param, ::std::ostream *os)
    {
        *os << param.testname;
    }
    // Gmres Data Driven (parametrized) Tests
    class GmresDDTest : public testing::TestWithParam<GmresDDParamType>
    {
    };
    // Alternative to PrintTo to generate test name for ctest, if used PrintTo() on param is added as suffix
    // currently not used
    std::string print_GmresDDTest_name(const testing::TestParamInfo<GmresDDTest::ParamType> &info)
    {
        return std::string(info.param.testname);
    }

    // tests with double type
    TEST_P(GmresDDTest, Double)
    {
        // Inside a test, access the test parameter with the GetParam() method
        // of the TestWithParam<T> class:
        const GmresDDParamType &param = GetParam();
        test_gmres<double>(param.mid, param.opts, param.precondd, param.monitd, param.status_exp);
    }
    //// tests with float type
    TEST_P(GmresDDTest, Float)
    {
        // Inside a test, access the test parameter with the GetParam() method
        // of the TestWithParam<T> class:
        const GmresDDParamType &param = GetParam();
        test_gmres<float>(param.mid, param.opts, param.precondf, param.monitf, param.status_exp);
    }
    INSTANTIATE_TEST_SUITE_P(GmresSuite, GmresDDTest, testing::ValuesIn(GMRESTestValues));
} // namespace
