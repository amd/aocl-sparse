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

/*! \file
 *  \brief aoclsparse_arguments.hpp provides a class to parse command arguments in both,
 *  clients and gtest. If class structure is changed, aoclsparse_common.yaml must also be
 *  changed.
 */

#pragma once
#ifndef AOCLSPARSE_ARGUMENTS_HPP
#define AOCLSPARSE_ARGUMENTS_HPP

#include "aoclsparse_datatype2string.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <aoclsparse.h>

struct Arguments
{
    aoclsparse_int M;
    aoclsparse_int N;
    aoclsparse_int K;
    aoclsparse_int nnz;

    aoclsparse_datatype compute_type;

    double alpha;
    double beta;

    aoclsparse_matrix_init matrix;

    aoclsparse_int unit_check;
    aoclsparse_int timing;
    aoclsparse_int iters;


    uint32_t algo;

    char filename[64];
    char function[64];
    char name[64];
    char category[32];

    // Validate input format.
    // aoclsparse_gentest.py is expected to conform to this format.
    // aoclsparse_gentest.py uses aoclsparse_common.yaml to generate this format.
    static void validate(std::istream& ifs)
    {
        auto error = [](auto name) {
            std::cerr << "Arguments field " << name << " does not match format.\n\n"
                      << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that aoclsparse_arguments.hpp and aoclsparse_common.yaml\n"
                         "define exactly the same Arguments, that aoclsparse_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            abort();
        };

        char      header[10]{}, trailer[10]{};
        Arguments arg{};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));

        if(strcmp(header, "aoclSPARSE"))
            error("header");
        else if(strcmp(trailer, "AOCLsparse"))
            error("trailer");

        auto check_func = [&, sig = (unsigned char)0](const auto& elem, auto name) mutable {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(unsigned char i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const unsigned char*>(&elem)[i] ^ sig ^ i)
                    error(name);
            sig += 89;
        };

#define AOCLSPARSE_FORMAT_CHECK(x) check_func(arg.x, #x)

        // Order is important
        AOCLSPARSE_FORMAT_CHECK(M);
        AOCLSPARSE_FORMAT_CHECK(N);
        AOCLSPARSE_FORMAT_CHECK(K);
        AOCLSPARSE_FORMAT_CHECK(nnz);
        AOCLSPARSE_FORMAT_CHECK(compute_type);
        AOCLSPARSE_FORMAT_CHECK(alpha);
        AOCLSPARSE_FORMAT_CHECK(beta);
        AOCLSPARSE_FORMAT_CHECK(matrix);
        AOCLSPARSE_FORMAT_CHECK(unit_check);
        AOCLSPARSE_FORMAT_CHECK(timing);
        AOCLSPARSE_FORMAT_CHECK(iters);
        AOCLSPARSE_FORMAT_CHECK(algo);
        AOCLSPARSE_FORMAT_CHECK(filename);
        AOCLSPARSE_FORMAT_CHECK(function);
        AOCLSPARSE_FORMAT_CHECK(name);
        AOCLSPARSE_FORMAT_CHECK(category);
    }

private:


    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg)
    {
        str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return str;
    }

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream& str, double x)
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
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[]{c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream& str, const char* s)
    {
        str << std::quoted(s);
    }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

        print("function", arg.function);
        print("compute_type", aoclsparse_datatype2string(arg.compute_type));
        print("M", arg.M);
        print("N", arg.N);
        print("nnz", arg.nnz);
        print("alpha", arg.alpha);
        print("beta", arg.beta);
        print("matrix", aoclsparse_matrix2string(arg.matrix));
        print("file", arg.filename);
        print("algo", arg.algo);
        print("name", arg.name);
        print("category", arg.category);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);
        return str << " }\n";
    }
};

static_assert(std::is_standard_layout<Arguments>{},
              "Arguments is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<Arguments>{},
              "Arguments is not a trivial type, and thus is "
              "incompatible with C.");

#endif // AOCLSPARSE_ARGUMENTS_HPP
