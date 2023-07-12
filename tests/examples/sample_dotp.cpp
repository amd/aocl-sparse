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

#include <complex>
#include <iostream>
#include <vector>

/* One can use std::complex<double> instead of aoclsparse_double_complex
#define STD_CMPLX  (or pass it as a compilation flag -DSTD_CMPLX)
#ifdef STD_CMPLX
#define aoclsparse_double_complex std::complex<double>
#endif
*/

#include "aoclsparse.h"

int main(void)
{
    aoclsparse_status                      status;
    const aoclsparse_int                   nnz  = 3;
    std::vector<aoclsparse_int>            indx = {0, 1, 2};
    std::vector<aoclsparse_double_complex> x{{1, 1}, {2, 2}, {3, 3}};
    std::vector<aoclsparse_double_complex> y{
        {1, 1}, {1, 2}, {2, 3}, {1, 0}, {4, 1}, {1.2, -1}, {7, -1}, {0, 2}, {-2, 3}};
    aoclsparse_double_complex dot_exp{-5, 23};
    aoclsparse_double_complex dotc_exp{23, 5};
    aoclsparse_double_complex dotp;
    bool                      ok;

    std::cout << "Invoking aoclsparse_zdotui...\n";

    //Invoke complex double dot product
    status = aoclsparse_zdotui(nnz, x.data(), indx.data(), y.data(), &dotp);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_cdotui, status = " << status << "."
                  << std::endl;
        return 1;
    }

#ifdef STD_CMPLX
    std::cout << "Dot product: " << dotp << ", Expected dot product: " << dot_exp << std::endl;
    ok = (dotp == dot_exp);
#else
    std::cout << "Dot product: (" << dotp.real << ", " << dotp.imag << "), Expected dot product: ("
              << dot_exp.real << ", " << dot_exp.imag << ")" << std::endl;
    ok = ((dotp.real == dot_exp.real) && (dotp.imag == dot_exp.imag));
#endif

    std::cout << "Invoking aoclsparse_zdotci...\n";

    //Invoke complex double dot product
    status = aoclsparse_zdotci(nnz, x.data(), indx.data(), y.data(), &dotp);
    if(status != aoclsparse_status_success)
    {
        std::cerr << "Error returned from aoclsparse_cdotci, status = " << status << "."
                  << std::endl;
        return 2;
    }
#ifdef STD_CMPLX
    std::cout << "Conjugated dot product: " << dotp
              << ", Expected conjugated dot product: " << dotc_exp << std::endl;
    ok &= (dotp == dotc_exp);
    return ok ? 0 : 3;

#else
    std::cout << "Dot product: (" << dotp.real << ", " << dotp.imag << "), Expected dot product: ("
              << dotc_exp.real << ", " << dotc_exp.imag << ")" << std::endl;
    ok &= ((dotp.real == dotc_exp.real) && (dotp.imag == dotc_exp.imag));
    return ok ? 0 : 4;
#endif
}
