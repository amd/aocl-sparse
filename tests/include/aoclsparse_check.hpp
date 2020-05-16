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
#ifndef AOCLSPARSE_CHECK_HPP
#define AOCLSPARSE_CHECK_HPP

#include <aoclsparse.h>


template <typename T>
void near_check_general(aoclsparse_int M, aoclsparse_int N, aoclsparse_int lda, T* refOut, T* actOut);

template <>
inline void near_check_general(
    aoclsparse_int M, aoclsparse_int N, aoclsparse_int lda, float* refOut, float* actOut)
{
    for(aoclsparse_int j = 0; j < N; ++j)
    {
        for(aoclsparse_int i = 0; i < M; ++i)
        {
            float compare_val = std::max(std::abs(refOut[i + j * lda] * 1e-3f),
                                         10 * std::numeric_limits<float>::epsilon());
            if(std::abs(refOut[i + j * lda] - actOut[i + j * lda]) >= compare_val)
            {
                std::cerr.precision(12);
                std::cerr << "ASSERT_NEAR(" << refOut[i + j * lda] << ", " << actOut[i + j * lda]
                          << ") failed: " << std::abs(refOut[i + j * lda] - actOut[i + j * lda])
                          << " exceeds compare_val " << compare_val << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

template <>
inline void near_check_general(
    aoclsparse_int M, aoclsparse_int N, aoclsparse_int lda, double* refOut, double* actOut)
{
    for(aoclsparse_int j = 0; j < N; ++j)
    {
        for(aoclsparse_int i = 0; i < M; ++i)
        {
            double compare_val = std::max(std::abs(refOut[i + j * lda] * 1e-10),
                                          10 * std::numeric_limits<double>::epsilon());
            if(std::abs(refOut[i + j * lda] - actOut[i + j * lda]) >= compare_val)
            {
                std::cerr.precision(16);
                std::cerr << "ASSERT_NEAR(" << refOut[i + j * lda] << ", " << actOut[i + j * lda]
                          << ") failed: " << std::abs(refOut[i + j * lda] - actOut[i + j * lda])
                          << " exceeds compare_val " << compare_val << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}
#if 0
template <>
inline void near_check_general(aoclsparse_int            M,
                               aoclsparse_int            N,
                               aoclsparse_int            lda,
                               aoclsparse_float_complex* refOut,
                               aoclsparse_float_complex* actOut)
{
    for(aoclsparse_int j = 0; j < N; ++j)
    {
        for(aoclsparse_int i = 0; i < M; ++i)
        {
            aoclsparse_float_complex compare_val
                = aoclsparse_float_complex(std::max(std::abs(std::real(refOut[i + j * lda]) * 1e-3f),
                                                   10 * std::numeric_limits<float>::epsilon()),
                                          std::max(std::abs(std::imag(refOut[i + j * lda]) * 1e-3f),
                                                   10 * std::numeric_limits<float>::epsilon()));
            if(std::abs(std::real(refOut[i + j * lda]) - std::real(actOut[i + j * lda]))
                   >= std::real(compare_val)
               || std::abs(std::imag(refOut[i + j * lda]) - std::imag(actOut[i + j * lda]))
                      >= std::imag(compare_val))
            {
                std::cerr.precision(16);
                std::cerr << "ASSERT_NEAR(" << refOut[i + j * lda] << ", " << actOut[i + j * lda]
                          << ") failed: " << std::abs(refOut[i + j * lda] - actOut[i + j * lda])
                          << " exceeds compare_val " << compare_val << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

template <>
inline void near_check_general(aoclsparse_int             M,
                               aoclsparse_int             N,
                               aoclsparse_int             lda,
                               aoclsparse_double_complex* refOut,
                               aoclsparse_double_complex* actOut)
{
    for(aoclsparse_int j = 0; j < N; ++j)
    {
        for(aoclsparse_int i = 0; i < M; ++i)
        {
            aoclsparse_double_complex compare_val
                = aoclsparse_double_complex(std::max(std::abs(std::real(refOut[i + j * lda]) * 1e-10),
                                                    10 * std::numeric_limits<double>::epsilon()),
                                           std::max(std::abs(std::imag(refOut[i + j * lda]) * 1e-10),
                                                    10 * std::numeric_limits<double>::epsilon()));
            if(std::abs(std::real(refOut[i + j * lda]) - std::real(actOut[i + j * lda]))
                   >= std::real(compare_val)
               || std::abs(std::imag(refOut[i + j * lda]) - std::imag(actOut[i + j * lda]))
                      >= std::imag(compare_val))
            {
                std::cerr.precision(16);
                std::cerr << "ASSERT_NEAR(" << refOut[i + j * lda] << ", " << actOut[i + j * lda]
                          << ") failed: " << std::abs(refOut[i + j * lda] - actOut[i + j * lda])
                          << " exceeds compare_val " << compare_val << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

#endif
#endif // AOCLSPARSE_CHECK_HPP
