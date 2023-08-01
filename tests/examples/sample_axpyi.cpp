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

/* Sample program to illustrate the usage of axpyi API which performs vector-vector
 * addition operation of the form y = ax + y, where
 * a: scalar value
 * x: compressed sparse vector
 * y: dense vector
 * Below example is for complex double precision. Using other precisions are also similar.
 */
int main(void)
{
    std::cout << "--------------------------------" << std::endl
              << "----- axpyi sample program -----" << std::endl
              << "--------------------------------" << std::endl
              << std::endl;

    // Number of non-zeros of the sparse vector
    const aoclsparse_int nnz = 3;
    // Scalar value
    const std::complex<double> a = {10, 20};

    // Sparse index vector (does not need to be ordered)
    std::vector<aoclsparse_int> indx = {0, 6, 3};
    // Sparse value vector in compressed form
    std::vector<std::complex<double>> x{{2, 3}, {4, 1}, {5, 1}};

    // Output vector
    std::vector<std::complex<double>> y{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {65, -999}};

    // Expected output vector
    std::vector<std::complex<double>> y_exp{
        {-40, 70}, {0, 0}, {0, 0}, {30, 110}, {0, 0}, {0, 0}, {85, -909}};

    aoclsparse_int ny = y.size();

    aoclsparse_status status;

    std::cout << "Invoking aoclsparse_zsctr...\n";
    //Invoke complex axpyi
    status = aoclsparse_zaxpyi(nnz, &a, x.data(), indx.data(), y.data());
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_zsctr, status = " << status << "."
                  << std::endl;
        return 3;
    }

    // Check and print the result
    std::cout << "vector-vector addition of y: " << std::endl
              << std::setw(10) << "y"
              << "   " << std::setw(20) << "expected y" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(4);
    bool oki, ok = true;
    for(aoclsparse_int i = 0; i < ny; i++)
    {
        oki = y[i] == y_exp[i];
        ok &= oki;
        std::cout << std::setw(15) << y[i] << std::setw(5) << "" << std::setw(15) << y_exp[i]
                  << std::setw(5) << (oki ? "" : " ! Error") << std::endl;
    }
    return (ok ? 0 : 6);
}
