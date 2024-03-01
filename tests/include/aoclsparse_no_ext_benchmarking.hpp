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
 * ************************************************************************ */

#pragma once
#ifndef AOCLSPARSE_NO_EXT_BENCHMARKING_HPP
#define AOCLSPARSE_NO_EXT_BENCHMARKING_HPP

#include "aoclsparse_arguments.hpp"

#include <vector>

/* Here are functions which would normally register extra test cases
 * to be run in addition to AOCL standard ones,
 * such as benchmarking against external library.
 * However, these all are dummy and don't extend the tests.
 */

template <typename T>
void register_tests_csrmv([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
    //testqueue.push_back({"new_test_name",&testing_csrmv_new<T>});
}
template <typename T>
void register_tests_dotmv([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_trsv([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
    //testqueue.push_back({"new_test_name",&testing_trsv_new<T>});
}
template <typename T>
void register_tests_gthr([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_gthrz([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_sctr([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_axpyi([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_roti([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_doti([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_dotui([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_dotci([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_symgs([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_symgs_mv([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
}
template <typename T>
void register_tests_trsm([[maybe_unused]] std::vector<testsetting<T>> &testqueue)
{
    //testqueue.push_back({"new_test_name",&testing_trsm_new<T>});
}
#endif
