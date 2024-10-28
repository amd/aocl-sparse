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

#include "aoclsparse_reference.hpp"

#include <aoclsparse.h>
#include <limits>

#define MAX_TOL_MULTIPLIER 4

template <typename T>
int near_check_general(
    aoclsparse_int M, aoclsparse_int N, aoclsparse_int lda, const T *refOut, const T *actOut)
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

            compare_val_real = (std::max)(std::abs(ref_rpart * 1e-6f), abserr);
            if(does_ipart_exists)
            {
                compare_val_imag = (std::max)(std::abs(actual_ipart * 1e-6f), abserr);
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
            tolm = (std::max)(tolm, k);
        }
    }
    if(tolm > 1)
    {
        std::cerr << "WARNING near_check has been permissive with a tolerance multiplier equal to "
                  << tolm << std::endl;
    }
    return 0;
}

inline int unit_check_general(aoclsparse_int        M,
                              aoclsparse_int        N,
                              aoclsparse_int        lda,
                              const aoclsparse_int *refOut,
                              const aoclsparse_int *actOut)
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
                return 1;
            }
        }
    }
    return 0;
}

/* Given two sparse CSR matrices (assumed sorted or in the same order),
 * the first is R (reference), the second C (computed),
 * compare if they are equal up to near_check_general() above.
 * Returns 0 if no difference is found (i.e., are the same).
 */
template <typename T>
int csrmat_check(aoclsparse_int                     mR,
                 aoclsparse_int                     nR,
                 aoclsparse_int                     nnzR,
                 aoclsparse_index_base              baseR,
                 const std::vector<aoclsparse_int> &row_ptrR,
                 const std::vector<aoclsparse_int> &col_indR,
                 const std::vector<T>              &valR,
                 aoclsparse_int                     mC,
                 aoclsparse_int                     nC,
                 aoclsparse_int                     nnzC,
                 aoclsparse_index_base              baseC,
                 const std::vector<aoclsparse_int> &row_ptrC,
                 const std::vector<aoclsparse_int> &col_indC,
                 const std::vector<T>              &valC)
{
    // dimensions/nnz check
    if(mR != mC || nR != nC || nnzR != nnzC || baseR != baseC)
    {
        std::cout << "Computed matrix is not matching the reference result (m x n x nnz, base):\n"
                  << "ref:  " << mR << " x " << nR << " x " << nnzR << ", " << baseR << std::endl
                  << "comp: " << mC << " x " << nC << " x " << nnzC << ", " << baseC << std::endl;
        return 1;
    }

    // Do arrays match the expected number of elements?
    size_t m = mR, nnz = nnzR;
    if(row_ptrR.size() < m + 1 || col_indR.size() < nnz || valR.size() < nnz)
    {
        std::cout << "Sizes of arrays of R matrix don't match the expected dimension." << std::endl
                  << "row_ptr size = " << row_ptrR.size() << " vs. expected " << m + 1 << std::endl
                  << "col_ind size = " << col_indR.size() << " vs. expected " << nnz << std::endl
                  << "val size = " << valR.size() << " vs. expected " << nnz << std::endl;
        return 2;
    }
    if(row_ptrC.size() < m + 1 || col_indC.size() < nnz || valC.size() < nnz)
    {
        std::cout << "Sizes of arrays of C matrix don't match the expected dimension." << std::endl
                  << "row_ptr size = " << row_ptrC.size() << " vs. expected " << m + 1 << std::endl
                  << "col_ind size = " << col_indC.size() << " vs. expected " << nnz << std::endl
                  << "val size = " << valC.size() << " vs. expected " << nnz << std::endl;
        return 3;
    }

    // matrix content
    if(unit_check_general(mR + 1, 1, 0, row_ptrC.data(), row_ptrR.data())
       || unit_check_general(nnzR, 1, 0, col_indC.data(), col_indR.data())
       || near_check_general(nnzR, 1, 0, valC.data(), valR.data()))
    {
        std::cout << "Array content of C and R matrices doesn't match." << std::endl;
        return 4;
    }

    return 0;
}

#endif // AOCLSPARSE_CHECK_HPP
