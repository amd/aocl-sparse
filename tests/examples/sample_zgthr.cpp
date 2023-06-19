/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <cfloat>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    std::cout << "-------------------------------" << std::endl
              << "---- Gather sample program ----" << std::endl
              << "-------------------------------" << std::endl
              << std::endl;

    // Input data

    // Number of non-zeros of the sparse vector
    const aoclsparse_int nnz = 3;

    // Sparse index vector (does not need to be ordered)
    std::vector<aoclsparse_int> indx = {0, 5, 3};

    // Sparse value vector
    std::vector<std::complex<double>> x(3);

    // Dense vector
    std::vector<std::complex<double>> y{
        {1, 1}, {1, 2}, {2, 3}, {1, 0}, {4, 1}, {1.2, -1}, {7, -1}, {0, 2}, {-2, 3}};

    // Expected sparse value vector
    std::vector<std::complex<double>> x_exp{{1, 1}, {1.2, -1}, {1, 0}};

    // Expected dense vector
    std::vector<std::complex<double>> y_exp{
        {0, 0}, {1, 2}, {2, 3}, {0, 0}, {4, 1}, {0, 0}, {7, -1}, {0, 2}, {-2, 3}};

    aoclsparse_int ny = y.size();

    aoclsparse_status status;

    // Call the gather function
    status = aoclsparse_zgthr(nnz, y.data(), x.data(), indx.data());
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_zgthr, status = " << status << "."
                  << std::endl;
        return 1;
    }

    // Check and print the result
    std::cout << "Gather from vector y: " << std::endl
              << std::setw(11) << "x"
              << "   " << std::setw(11) << "expected" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(1);
    bool oki, ok = true;
    for(aoclsparse_int i = 0; i < nnz; i++)
    {
        oki = x[i] == x_exp[i];
        ok &= oki;
        std::cout << std::setw(11) << x[i] << std::setw(3) << "" << std::setw(11) << x_exp[i]
                  << std::setw(2) << (oki ? "" : " !") << std::endl;
    }

    // Call the gather with zero function
    status = aoclsparse_zgthrz(nnz, y.data(), x.data(), indx.data());
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_zgthr, status = " << status << "."
                  << std::endl;
        return 3;
    }

    // Check and print the result
    std::cout << std::endl
              << "Gather from vector y: " << std::endl
              << std::setw(11) << "y" << std::setw(3) << "" << std::setw(11) << "expected"
              << std::setw(3) << "" << std::setw(11) << "x" << std::setw(3) << "" << std::setw(11)
              << "expected" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(1);
    bool oky;
    for(aoclsparse_int i = 0; i < ny; i++)
    {
        oky = y[i] == y_exp[i];
        ok &= oky;
        std::cout << std::setw(11) << y[i] << std::setw(3) << "" << std::setw(11) << y_exp[i]
                  << std::setw(2) << (oky ? "" : " !");
        if(i < nnz)
        {
            oki = x[i] == x_exp[i];
            ok &= oki;
            std::cout << " " << std::setw(11) << x[i] << std::setw(3) << "" << std::setw(11)
                      << x_exp[i] << std::setw(2) << (oki ? "" : " !");
        }
        std::cout << std::endl;
    }

    return (ok ? 0 : 6);
}
