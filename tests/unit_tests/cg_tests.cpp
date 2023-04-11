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
#include "cg_ut_functions.hpp"

namespace
{

    // CG Data Driven Test
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
    } CgDDParamType;

    // List of all desired positive tests
    const CgDDParamType CGTestPosValues[] = {
        {"SmallMatNoPrecondNullptr",
         aoclsparse_status_success,
         sample_cg_mat,
         nullptr,
         nullptr,
         nullptr,
         nullptr,
         {{"CG Abs Tolerance", "1.0e-4"}, {"CG Rel Tolerance", "0"}}},
        {"SmallMatNoPrecond",
         aoclsparse_status_success,
         sample_cg_mat,
         precond_dummy<float>,
         precond_dummy<double>,
         monit_print<float>,
         monit_print<double>,
         {{"CG Abs Tolerance", "1.0e-4"}, {"CG Rel Tolerance", "0"}}},
        {"SmallMatSymgsPrecond",
         aoclsparse_status_success,
         sample_cg_mat,
         precond_dummy<float>,
         precond_dummy<double>,
         monit_dummy<float>,
         monit_dummy<double>,
         {{"CG Preconditioner", "SGS"},
          {"CG Iteration Limit", "6"},
          {"CG Abs Tolerance", "1.0e-4"},
          {"CG Rel Tolerance", "0"}}},
        {"SmallMatUsrIPrecond",
         aoclsparse_status_success,
         sample_cg_mat,
         precond_identity<float>,
         precond_identity<double>,
         monit_dummy<float>,
         monit_dummy<double>,
         {{"CG Preconditioner", "User"},
          {"CG Iteration Limit", "8"},
          {"CG Abs Tolerance", "0"},
          {"CG Rel Tolerance", "1.0e-5"}}},
        {"SmallMatUsrMonitStop",
         aoclsparse_status_user_stop,
         sample_cg_mat,
         nullptr,
         nullptr,
         monit_tolstop<float>,
         monit_tolstop<double>,
         {{"Iterative Method", "CG"},
          {"CG Abs Tolerance", "1.0e-12"},
          {"CG Rel Tolerance", "0"},
          {"CG Iteration Limit", "10"}}}
        // turn off the default tolerance check and use monit to check convergence which triggers stop --> acurate solution should be obtained but different return status
    };

    // List of all desired negative tests
    const CgDDParamType CGTestNegValues[] = {{"MaxItReached",
                                              aoclsparse_status_maxit,
                                              sample_cg_mat,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              {{"CG Iteration Limit", "2"}}},
                                             {"UsrMonitStopIt2",
                                              aoclsparse_status_user_stop,
                                              sample_cg_mat,
                                              nullptr,
                                              nullptr,
                                              monit_stopit2<float>,
                                              monit_stopit2<double>,
                                              {{"CG Iteration Limit", "20"}}},
                                             {"InvalidMatrix",
                                              aoclsparse_status_invalid_value,
                                              invalid_mat,
                                              precond_dummy<float>,
                                              precond_dummy<double>,
                                              monit_dummy<float>,
                                              monit_dummy<double>,
                                              {}},
                                             {"NonsymmetrixMatrix",
                                              aoclsparse_status_invalid_value,
                                              N5_full_sorted,
                                              precond_dummy<float>,
                                              precond_dummy<double>,
                                              monit_dummy<float>,
                                              monit_dummy<double>,
                                              {}},
                                             {"PrecondStop",
                                              aoclsparse_status_user_stop,
                                              sample_cg_mat,
                                              precond_dummy<float>,
                                              precond_dummy<double>,
                                              nullptr,
                                              nullptr,
                                              {{"CG Preconditioner", "User"}}}};

    // Teach GTest how to print CGDDParamType
    // in this case use only user's unique testname
    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const CgDDParamType &param, ::std::ostream *os)
    {
        *os << param.testname;
    }

    // Cg Data Driven (parametrized) Tests
    class CgDDTestPos : public testing::TestWithParam<CgDDParamType>
    {
    };

    // Alternative to PrintTo to generate test name for ctest, if used PrintTo() on param is added as suffix
    // currently not used
    std::string print_CgDDTest_name(const testing::TestParamInfo<CgDDTestPos::ParamType> &info)
        __attribute__((unused));
    std::string print_CgDDTest_name(const testing::TestParamInfo<CgDDTestPos::ParamType> &info)
    {
        return std::string(info.param.testname);
    }

    // Positive (aoclsparse_success) tests with double type
    TEST_P(CgDDTestPos, Double)
    {
        // Inside a test, access the test parameter with the GetParam() method
        // of the TestWithParam<T> class:
        const CgDDParamType &param = GetParam();
        test_cg_positive<double>(
            param.mid, param.opts, param.precondd, param.monitd, param.status_exp);
    }

    // Positive (aoclsparse_success) tests with float type
    TEST_P(CgDDTestPos, Float)
    {
        const CgDDParamType &param = GetParam();
        test_cg_positive<float>(
            param.mid, param.opts, param.precondf, param.monitf, param.status_exp);
    }

    INSTANTIATE_TEST_SUITE_P(CgSuitePos, CgDDTestPos, testing::ValuesIn(CGTestPosValues));

    class CgDDTestErr : public testing::TestWithParam<CgDDParamType>
    {
    };

    // Error tests with double type
    TEST_P(CgDDTestErr, Double)
    {
        const CgDDParamType &param = GetParam();
        test_cg_error<double>(
            param.status_exp, param.mid, param.opts, param.precondd, param.monitd);
    }

    // Error tests with float type
    TEST_P(CgDDTestErr, Float)
    {
        const CgDDParamType &param = GetParam();
        test_cg_error<float>(param.status_exp, param.mid, param.opts, param.precondf, param.monitf);
    }

    INSTANTIATE_TEST_SUITE_P(CgSuiteErr, CgDDTestErr, testing::ValuesIn(CGTestNegValues));

    // Individual tests which don't fall into the parametrized pattern

    TEST(CgTest, DoubleCallDouble)
    {
        test_cg_double_call<double>();
    }

    TEST(CgTest, DoubleCallFloat)
    {
        test_cg_double_call<float>();
    }

} // namespace
