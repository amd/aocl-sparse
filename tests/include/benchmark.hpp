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

/*! \file
 *  \brief benchmark.hpp provides a class to identify benchmark drivers (vendor libs)
 */

#pragma once
#ifndef AOCLSPARSE_BENCHMARK_HPP
#define AOCLSPARSE_BENCHMARK_HPP

#include "aoclsparse.h"

#include <iostream>
#include <string>
#include <vector>

namespace BenchmarkInfo
{
    // struct that holds the info of a benchmark driver
    struct DriverInfo
    {
        std::string name;
        std::string desc;
        std::string version;
        std::string compiler;
        std::string extra;

        // print method
        void print(void)
        {
            std::cout << "     vendor info: ";
            if(name != "")
                std::cout << name << ' ';
            if(version != "")
                std::cout << version;
            std::cout << '\n';
            if(compiler != "")
                std::cout << "     compiler: " << compiler << '\n';
            if(desc != "")
                std::cout << "     desc: " << desc << '\n';
            if(extra != "")
                std::cout << "     extra info: " << extra << "\n";
        };
    };

    class Registry
    {
        std::vector<DriverInfo> infolist;

    public:
        // constructor
        Registry(DriverInfo i)
        {
            infolist.push_back(i);
        }

        void add(struct DriverInfo i)
        {
            infolist.push_back(i);
        }
        void print(void)
        {
            for(auto i : infolist)
            {
                i.print();
                std::cout << std::endl;
            }
        }
    };

    // Define default driver (AOCL-Sparse)
    const struct DriverInfo aoclsparse_bench
    {
        // clang-format off
        .name = std::string(AOCLSPARSE_VERSION_STRING),
        .desc = "AMD (R) AOCL-Sparse Library",
        .version = HASH,
        .compiler = CID,
        .extra = "id=aocl"
        // clang-format on
    };
}

// Global registry
// initialize with default benchmark driver
BenchmarkInfo::Registry driver_info(BenchmarkInfo::aoclsparse_bench);

#endif // AOCLSPARSE_BENCHMARK_HPP
