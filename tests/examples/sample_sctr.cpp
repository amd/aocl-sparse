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

// Sample program to illustrate the usage of scatter of a compressed sparse vector to the full storage form (complex double precision)
// Using other precisions are also similar to below example
int main(void)
{
    std::cout << "--------------------------------" << std::endl
              << "---- Scatter sample program ----" << std::endl
              << "--------------------------------" << std::endl
              << std::endl;

    // Number of non-zeros of the sparse vector
    const aoclsparse_int nnz = 3;

    // Sparse index vector (does not need to be ordered)
    std::vector<aoclsparse_int> indx = {0, 3, 6};
    // Sparse value vector in compressed form
    std::vector<std::complex<double>> x{{1.01, -1.13}, {2.4, -2.0}, {-0.3, 1.3}};

    // Output vector
    std::vector<std::complex<double>> y{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};

    // Expected output vector
    std::vector<std::complex<double>> y_exp{
        {1.01, -1.13}, {0, 0}, {0, 0}, {2.4, -2.0}, {0, 0}, {0, 0}, {-0.3, 1.3}};

    aoclsparse_int ny = y.size();

    aoclsparse_status status;

    std::cout << "Invoking aoclsparse_zsctr...\n";
    //Invoke complex scatter
    status = aoclsparse_zsctr(nnz, x.data(), indx.data(), y.data());
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_zsctr, status = " << status << "."
                  << std::endl;
        return 3;
    }

    // Check and print the result
    std::cout << "Scatter to vector y: " << std::endl
              << std::setw(10) << "y"
              << "   " << std::setw(17) << "expected y" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(4);
    bool oki, ok = true;
    for(aoclsparse_int i = 0; i < ny; i++)
    {
        oki = y[i] == y_exp[i];
        ok &= oki;
        std::cout << std::setw(11) << y[i] << std::setw(3) << "" << std::setw(11) << y_exp[i]
                  << std::setw(2) << (oki ? "" : " ! Error") << std::endl;
    }
    return (ok ? 0 : 6);
}
