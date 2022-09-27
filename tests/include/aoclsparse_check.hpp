/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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

#include <limits>

#define MAX_TOL_MULTIPLIER 4

template <typename T>
void near_check_general(
    aoclsparse_int M, aoclsparse_int N, aoclsparse_int lda, T *refOut, T *actOut);

template <>
inline void near_check_general(
    aoclsparse_int M, aoclsparse_int N, aoclsparse_int lda, float *refOut, float *actOut)
{
    int tolm = 1;
    for(aoclsparse_int j = 0; j < N; ++j)
    {
        for(aoclsparse_int i = 0; i < M; ++i)
        {
            float compare_val = std::max(std::abs(refOut[i + j * lda] * 1e-3f),
                                         10 * std::numeric_limits<float>::epsilon());
            int   k;
            for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
            {
                if(std::abs(refOut[i + j * lda] - actOut[i + j * lda]) <= compare_val * k)
                {
                    break;
                }
            }

            if(k > MAX_TOL_MULTIPLIER)
            {
                std::cerr.precision(12);
                std::cerr << "ASSERT_NEAR(" << refOut[i + j * lda] << ", " << actOut[i + j * lda]
                          << ") failed: " << std::abs(refOut[i + j * lda] - actOut[i + j * lda])
                          << " exceeds permissive range [" << compare_val << ","
                          << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                exit(EXIT_FAILURE);
            }
            tolm = std::max(tolm, k);
        }
    }
    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
}

template <>
inline void near_check_general(
    aoclsparse_int M, aoclsparse_int N, aoclsparse_int lda, double *refOut, double *actOut)
{
    int tolm = 1;
    for(aoclsparse_int j = 0; j < N; ++j)
    {
        for(aoclsparse_int i = 0; i < M; ++i)
        {
            double compare_val = std::max(std::abs(refOut[i + j * lda] * 1e-06),
                                          10 * std::numeric_limits<double>::epsilon());

            int k;
            for(k = 1; k <= MAX_TOL_MULTIPLIER; ++k)
            {
                if(std::abs(refOut[i + j * lda] - actOut[i + j * lda]) <= compare_val * k)
                {
                    break;
                }
            }

            if(k > MAX_TOL_MULTIPLIER)
            {
                std::cerr.precision(12);
                std::cerr << "ASSERT_NEAR(" << refOut[i + j * lda] << ", " << actOut[i + j * lda]
                          << ") failed: " << std::abs(refOut[i + j * lda] - actOut[i + j * lda])
                          << " exceeds permissive range [" << compare_val << ","
                          << compare_val * MAX_TOL_MULTIPLIER << " ]" << std::endl;
                exit(EXIT_FAILURE);
            }
            tolm = std::max(tolm, k);
        }
    }
    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
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
