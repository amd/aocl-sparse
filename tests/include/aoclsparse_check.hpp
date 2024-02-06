/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.
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
#ifndef AOCLSPARSE_CHECK_HPP
#define AOCLSPARSE_CHECK_HPP

#include "aoclsparse_utils.hpp"

#include <aoclsparse.h>
#include <limits>

#define MAX_TOL_MULTIPLIER 4

template <typename T>
int near_check_general(aoclsparse_int M, aoclsparse_int N, aoclsparse_int lda, T *refOut, T *actOut)
{
    int            tolm      = 1;
    tolerance_t<T> ref_rpart = 0, ref_ipart = 0, actual_rpart = 0, actual_ipart = 0;
    tolerance_t<T> abserr, compare_val_real, compare_val_imag;
    bool           does_ipart_exists = false;

    abserr = expected_precision<tolerance_t<T>>(10);
    for(aoclsparse_int j = 0; j < N; ++j)
    {
        for(aoclsparse_int i = 0; i < M; ++i)
        {
            int k;
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref_rpart = std::real(refOut[i + j * lda]);
                ref_ipart = std::imag(refOut[i + j * lda]);

                actual_rpart      = std::real(actOut[i + j * lda]);
                actual_ipart      = std::imag(actOut[i + j * lda]);
                does_ipart_exists = true;
            }
            else if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                              || std::is_same_v<T, aoclsparse_float_complex>)
            {
                ref_rpart = refOut[i + j * lda].real;
                ref_ipart = refOut[i + j * lda].imag;

                actual_rpart      = actOut[i + j * lda].real;
                actual_ipart      = actOut[i + j * lda].imag;
                does_ipart_exists = true;
            }
            else
            {
                ref_rpart         = refOut[i + j * lda];
                actual_rpart      = actOut[i + j * lda];
                does_ipart_exists = false;
            }

            compare_val_real = std::max(std::abs(ref_rpart * 1e-6f), abserr);
            if(does_ipart_exists)
            {
                compare_val_imag = std::max(std::abs(actual_ipart * 1e-6f), abserr);
            }
            for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
            {
                bool real_check = false, imag_check = false;
                real_check = std::abs(ref_rpart - actual_rpart) <= compare_val_real * k;
                if(does_ipart_exists)
                {
                    imag_check = std::abs(ref_ipart - actual_ipart) <= compare_val_imag * k;
                }
                if(real_check || (does_ipart_exists && imag_check))
                {
                    break;
                }
            }

            if(k > MAX_TOL_MULTIPLIER)
            {
                std::cerr.precision(12);
                std::cerr << "ASSERT_NEAR(" << ref_rpart << ", " << actual_rpart
                          << ") failed: " << std::abs(ref_rpart - actual_rpart)
                          << " real part exceeds permissive range [" << compare_val_real << ","
                          << compare_val_real * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                if(does_ipart_exists)
                {
                    std::cerr << "ASSERT_NEAR(" << ref_ipart << ", " << actual_ipart
                              << ") failed: " << std::abs(ref_ipart - actual_ipart)
                              << " imaginary part exceeds permissive range [" << compare_val_imag
                              << "," << compare_val_imag * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                }
                return 1;
            }
            tolm = std::max(tolm, k);
        }
    }
    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
    return 0;
}

inline void unit_check_general(aoclsparse_int  M,
                               aoclsparse_int  N,
                               aoclsparse_int  lda,
                               aoclsparse_int *refOut,
                               aoclsparse_int *actOut)
{
    for(aoclsparse_int j = 0; j < N; ++j)
    {
        for(aoclsparse_int i = 0; i < M; ++i)
        {
            if(refOut[i + j * lda] != actOut[i + j * lda])
            {
                std::cerr.precision(12);
                std::cerr << "ASSERT_EQ(" << refOut[i + j * lda] << ", " << actOut[i + j * lda]
                          << ") failed. " << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

#endif // AOCLSPARSE_CHECK_HPP
