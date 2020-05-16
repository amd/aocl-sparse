/* ************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef AOCLSPARSE_TEST_HPP
#define AOCLSPARSE_TEST_HPP

#include "aoclsparse_arguments.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <aoclsparse.h>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>


inline const char* aoclsparse_status_to_string(aoclsparse_status status)
{
    switch(status)
    {
    case aoclsparse_status_success:
        return "aoclsparse_status_success";
    case aoclsparse_status_invalid_handle:
        return "aoclsparse_status_invalid_handle";
    case aoclsparse_status_not_implemented:
        return "aoclsparse_status_not_implemented";
    case aoclsparse_status_invalid_pointer:
        return "aoclsparse_status_invalid_pointer";
    case aoclsparse_status_invalid_size:
        return "aoclsparse_status_invalid_size";
    case aoclsparse_status_memory_error:
        return "aoclsparse_status_memory_error";
    case aoclsparse_status_internal_error:
        return "aoclsparse_status_internal_error";
    default:
        return "<undefined aoclsparse_status value>";
    }
}

inline void aoclsparse_expect_status(aoclsparse_status status, aoclsparse_status expect)
{
    if(status != expect)
    {
        std::cerr << "aoclSPARSE status error: Expected " << aoclsparse_status_to_string(expect)
                  << ", received " << aoclsparse_status_to_string(status) << std::endl;
        if(expect == aoclsparse_status_success)
            exit(EXIT_FAILURE);
    }
}


#define EXPECT_AOCLSPARSE_STATUS aoclsparse_expect_status

#define CHECK_AOCLSPARSE_ERROR2(STATUS) EXPECT_AOCLSPARSE_STATUS(STATUS, aoclsparse_status_success)
#define CHECK_AOCLSPARSE_ERROR(STATUS) CHECK_AOCLSPARSE_ERROR2(STATUS)

// ----------------------------------------------------------------------------
// Error case which returns false when converted to bool. A void specialization
// of the FILTER class template above, should be derived from this class, in
// order to indicate that the type combination is invalid.
// ----------------------------------------------------------------------------
struct aoclsparse_test_invalid
{
    // Return false to indicate the type combination is invalid, for filtering
    explicit operator bool()
    {
        return false;
    }

    // If this specialization is actually called, print fatal error message
    void operator()(const Arguments&)
    {
        static constexpr char msg[] = "Internal error: Test called with invalid types\n";

        fputs(msg, stderr);
        exit(EXIT_FAILURE);
    }
};

#endif // AOCLSPARSE_TEST_HPP
