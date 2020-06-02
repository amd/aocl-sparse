/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <aoclsparse.h>

// Level2
#include "testing_ellmv.hpp"
#include "testing_csrmv.hpp"
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    Arguments arg;
    arg.unit_check = 0;
    arg.timing = 1;

    std::string   function;
    std::string   mtxfile;
    char          precision = 'd';
    aoclsparse_int dir;
    po::options_description desc("aoclsparse client command line options");
    desc.add_options()("help,h", "produces this help message")
        ("sizem,m",
         po::value<aoclsparse_int>(&arg.M)->default_value(128),
         "Specific matrix size testing: sizem is only applicable to SPARSE-2 "
         "& SPARSE-3: the number of rows.")

        ("sizen,n",
         po::value<aoclsparse_int>(&arg.N)->default_value(128),
         "Specific matrix/vector size testing: SPARSE-1: the length of the "
         "dense vector. SPARSE-2 & SPARSE-3: the number of columns")

        ("sizennz,z",
         po::value<aoclsparse_int>(&arg.nnz)->default_value(0),
         "Specific vector size testing, LEVEL-1: the number of non-zero elements "
         "of the sparse vector.")

        ("mtx",
         po::value<std::string>(&mtxfile)->default_value(""), "read from matrix "
         "market (.mtx) format. This will override parameters -m, -n, and -z.")

        ("alpha", 
          po::value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta", 
          po::value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         po::value<std::string>(&function)->default_value("csrmv"),
         "SPARSE function to test. Options:\n"
         "  Level2: csrmv ellmv")

        ("precision,r",
         po::value<char>(&precision)->default_value('d'), "Options: s,d")

        ("verify,v",
         po::value<aoclsparse_int>(&arg.unit_check)->default_value(0),
         "Validate results ? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         po::value<aoclsparse_int>(&arg.iters)->default_value(10),
         "Iterations to run inside timing loop");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help") || (argc == 1))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    if(precision != 's' && precision != 'd' && precision != 'c' && precision != 'z')
    {
        std::cerr << "Invalid value for --precision" << std::endl;
        return -1;
    }
    
    if(mtxfile != "")
    {
        strcpy(arg.filename, mtxfile.c_str());
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
    if(function == "csrmv")
    {
        arg.algo = 1;
        if(precision == 's')
            testing_csrmv<float>(arg);
        else if(precision == 'd')
            testing_csrmv<double>(arg);
    }
    else if(function == "ellmv")
    {
        arg.algo = 1;
        if(precision == 's')
            testing_ellmv<float>(arg);
        else if(precision == 'd')
            testing_ellmv<double>(arg);
    }
    else
    {
        std::cerr << "Invalid value for --function" << std::endl;
        return -1;
    }
    return 0;
}
