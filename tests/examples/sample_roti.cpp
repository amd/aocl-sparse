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
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// Sample program to illustrate the usage of Givens rotation on a compressed sparse vector and a full storage vector(double precision)
int main(void)
{
    std::cout << "-----------------------------" << std::endl
              << "---- roti sample program ----" << std::endl
              << "-----------------------------" << std::endl
              << std::endl;

    // Input data

    // Number of non-zeros of the sparse vector
    const aoclsparse_int nnz = 5;

    // Sparse index vector (does not need to be ordered)
    std::vector<aoclsparse_int> indx{0, 3, 6, 7, 9};

    // Sparse value vector in compressed form
    std::vector<double> x{-0.75, 4, -9.5, 46, 1.25};

    // Output vector
    std::vector<double> y{-0.75, 0, 0, 4, 0, 0, -9.5, 46, 0, 1.25};

    // Expected output vectors
    std::vector<double> y_exp{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<double> x_exp{-3, 16, -38, 184, 5};

    const double c = 2;
    const double s = 2;

    aoclsparse_int ny = y.size();
    aoclsparse_int nx = x.size();

    aoclsparse_status status;
    std::cout << "Invoking aoclsparse_droti...\n";
    //Invoke aoclsparse_droti to apply Givens rotation
    status = aoclsparse_droti(nnz, x.data(), indx.data(), y.data(), c, s);

    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_droti, status = " << status << "."
                  << std::endl;
        return 3;
    }

    // Check and print the result
    std::cout << "The vector y after Givens rotation: " << std::endl
              << std::setw(11) << "y"
              << "   " << std::setw(11) << "expected" << std::endl;
    std::cout << std::fixed;

    std::cout.precision(2);

    bool oki, ok_y = true, ok_x = true;

    //Initializing precision tolerance range for double
    const double tol = std::sqrt(std::numeric_limits<double>::epsilon());

    for(aoclsparse_int i = 0; i < ny; i++)
    {
        oki = std::abs(y[i] - y_exp[i]) <= tol;
        ok_y &= oki;
        std::cout << std::setw(11) << y[i] << std::setw(3) << "" << std::setw(11) << y_exp[i]
                  << std::setw(2) << (oki ? "" : " !") << std::endl;
    }
    std::cout << std::endl;
    std::cout << "The vector x after Givens rotation: " << std::endl
              << std::setw(11) << "x"
              << "   " << std::setw(11) << "expected" << std::endl;
    std::cout << std::fixed;
    for(aoclsparse_int i = 0; i < nx; i++)
    {
        oki = std::abs(x[i] - x_exp[i]) <= tol;
        ok_x &= oki;
        std::cout << std::setw(11) << x[i] << std::setw(3) << "" << std::setw(11) << x_exp[i]
                  << std::setw(2) << (oki ? "" : " !") << std::endl;
    }
    return ((ok_y && ok_x) ? 0 : 6);
}
