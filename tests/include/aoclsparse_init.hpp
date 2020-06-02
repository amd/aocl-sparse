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

#pragma once
#ifndef AOCLSPARSE_INIT_HPP
#define AOCLSPARSE_INIT_HPP

#include "aoclsparse_datatype2string.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_random.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <aoclsparse.h>
#include <vector>
#include <complex>

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
inline void aoclsparse_init(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator_normal<T>();
}
// Initializes sparse index vector with nnz entries ranging from start to end
inline void
    aoclsparse_init_index(std::vector<aoclsparse_int>& x, size_t nnz, size_t start, size_t end)
{
    std::vector<bool> check(end - start, false);

    aoclsparse_int num = 0;

    while(num < nnz)
    {
        aoclsparse_int val = random_generator<aoclsparse_int>(start, end - 1);
        if(!check[val - start])
        {
            x[num++]           = val;
            check[val - start] = true;
        }
    }

    std::sort(x.begin(), x.end());
}

// Initialize matrix so adjacent entries have alternating sign.
// In gemm if either A or B are initialized with alernating
// sign the reduction sum will be summing positive
// and negative numbers, so it should not get too large.
// This helps reduce floating point inaccuracies for 16bit
// arithmetic where the exponent has only 5 bits, and the
// mantissa 10 bits.
template <typename T>
inline void aoclsparse_init_alternating_sign(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
            {
                auto value                        = random_generator<T>();
                A[i + j * lda + i_batch * stride] = (i ^ j) & 1 ? value : -value;
            }
}

inline void coo_to_csr(aoclsparse_int                     M,
                            aoclsparse_int                     nnz,
                            const std::vector<aoclsparse_int>& coo_row_ind,
                            std::vector<aoclsparse_int>&       csr_row_ptr,
                            aoclsparse_index_base              base)
{
    // Resize and initialize csr_row_ptr with zeros
    csr_row_ptr.resize(M + 1, 0);

    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr[coo_row_ind[i] + 1 - base];
    }

    csr_row_ptr[0] = base;
    for(aoclsparse_int i = 0; i < M; ++i)
    {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }
}

inline void csr_to_coo(aoclsparse_int                     M,
                            aoclsparse_int                     nnz,
                            const std::vector<aoclsparse_int>& csr_row_ptr,
                            std::vector<aoclsparse_int>&       coo_row_ind,
                            aoclsparse_index_base              base)
{
    // Resize coo_row_ind
    coo_row_ind.resize(nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(aoclsparse_int i = 0; i < M; ++i)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;

        for(aoclsparse_int j = row_begin; j < row_end; ++j)
        {
            coo_row_ind[j] = i + base;
        }
    }
}
/* ==================================================================================== */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

template <typename T>
inline void aoclsparse_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(aoclsparse_nan_rng());
}

template <typename T>
inline void aoclsparse_init_nan(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(aoclsparse_nan_rng());
}

/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
inline void aoclsparse_init_coo_matrix(std::vector<aoclsparse_int>& row_ind,
                                      std::vector<aoclsparse_int>& col_ind,
                                      std::vector<T>&             val,
                                      size_t                      M,
                                      size_t                      N,
                                      size_t                      nnz,
                                      aoclsparse_index_base        base,
                                      bool                        full_rank = false)
{
    // If M > N, full rank is not possible
    if(full_rank && M > N)
    {
        std::cerr << "ERROR: M > N, cannot generate matrix with full rank" << std::endl;
        full_rank = false;
    }

    // If nnz < M, full rank is not possible
    if(full_rank && nnz < M)
    {
        std::cerr << "ERROR: nnz < M, cannot generate matrix with full rank" << std::endl;
        full_rank = false;
    }

    if(row_ind.size() != nnz)
    {
        row_ind.resize(nnz);
    }
    if(col_ind.size() != nnz)
    {
        col_ind.resize(nnz);
    }
    if(val.size() != nnz)
    {
        val.resize(nnz);
    }

    // Add diagonal entry, if full rank is flagged
    size_t i = 0;

    if(full_rank)
    {
        for(; i < M; ++i)
        {
            row_ind[i] = i;
        }
    }

    // Uniform distributed row indices
    for(; i < nnz; ++i)
    {
        row_ind[i] = random_generator<aoclsparse_int>(0, M - 1);
    }

    // Sort row indices
    std::sort(row_ind.begin(), row_ind.end());

    // Sample column indices
    std::vector<bool> check(nnz, false);

    i = 0;
    while(i < nnz)
    {
        size_t begin = i;
        while(row_ind[i] == row_ind[begin])
        {
            ++i;
            if(i >= nnz)
            {
                break;
            }
        }

        // Sample i disjunct column indices
        size_t idx = begin;

        if(full_rank)
        {
            check[row_ind[idx]] = true;
            col_ind[idx++]      = row_ind[begin];
        }

        while(idx < i)
        {
            // Normal distribution around the diagonal
            aoclsparse_int rng = (i - begin) * random_generator_normal<double>();

            if(M <= N)
            {
                rng += row_ind[begin];
            }

            // Repeat if running out of bounds
            if(rng < 0 || rng > N - 1)
            {
                continue;
            }

            // Check for disjunct column index in current row
            if(!check[rng])
            {
                check[rng]     = true;
                col_ind[idx++] = rng;
            }
        }

        // Reset disjunct check array
        for(size_t j = begin; j < i; ++j)
        {
            check[col_ind[j]] = false;
        }

        // Partially sort column indices
        std::sort(&col_ind[begin], &col_ind[i]);
    }

    // Correct index base accordingly
    if(base == aoclsparse_index_base_one)
    {
        for(aoclsparse_int i = 0; i < nnz; ++i)
        {
            ++row_ind[i];
            ++col_ind[i];
        }
    }

    // Sample random off-diagonal values
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        if(row_ind[i] == col_ind[i])
        {
            // Sample diagonal values
            val[i] = random_generator<T>();
        }
        else
        {
            // Samples off-diagonal values
            val[i] = random_generator<T>();
        }
    }
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
static inline void
    read_mtx_value(std::istringstream& is, aoclsparse_int& row, aoclsparse_int& col, float& val)
{
    is >> row >> col >> val;
}

static inline void
    read_mtx_value(std::istringstream& is, aoclsparse_int& row, aoclsparse_int& col, double& val)
{
    is >> row >> col >> val;
}

static inline void read_mtx_value(std::istringstream&      is,
                                  aoclsparse_int&           row,
                                  aoclsparse_int&           col,
                                  aoclsparse_float_complex& val)
{
    float real;
    float imag;

    is >> row >> col >> real >> imag;

    val = {real, imag};
}

static inline void read_mtx_value(std::istringstream&       is,
                                  aoclsparse_int&            row,
                                  aoclsparse_int&            col,
                                  aoclsparse_double_complex& val)
{
    double real;
    double imag;

    is >> row >> col >> real >> imag;

    val = {real, imag};
}

template <typename T>
inline void aoclsparse_init_coo_mtx(const char*                 filename,
                                   std::vector<aoclsparse_int>& coo_row_ind,
                                   std::vector<aoclsparse_int>& coo_col_ind,
                                   std::vector<T>&             coo_val,
                                   aoclsparse_int&              M,
                                   aoclsparse_int&              N,
                                   aoclsparse_int&              nnz,
                                   aoclsparse_index_base        base)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "Reading matrix " << filename << " ... ";
    }

    FILE* f = fopen(filename, "r");
    if(!f)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    char line[1024];

    // Check for banner
    if(!fgets(line, 1024, f))
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if(sscanf(line, "%s %s %s %s %s", banner, array, coord, data, type) != 5)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    // Convert to lower case
    for(char* p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    // Check array type
    if(strcmp(array, "matrix") != 0)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    // Check coord
    if(strcmp(coord, "coordinate") != 0)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    // Check data
    if(strcmp(data, "real") != 0 && strcmp(data, "integer") != 0 && strcmp(data, "pattern") != 0
       && strcmp(data, "complex") != 0)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    // Check type
    if(strcmp(type, "general") != 0 && strcmp(type, "symmetric") != 0)
    {
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
    }

    // Symmetric flag
    aoclsparse_int symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if(line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    aoclsparse_int snnz;

    int inrow;
    int incol;
    int innz;

    sscanf(line, "%d %d %d", &inrow, &incol, &innz);

    M    = static_cast<aoclsparse_int>(inrow);
    N    = static_cast<aoclsparse_int>(incol);
    snnz = static_cast<aoclsparse_int>(innz);

//    nnz = symm ? (snnz - M) * 2 + M : snnz;
    nnz = symm ? snnz  * 2 : snnz;

    std::vector<aoclsparse_int> unsorted_row(nnz);
    std::vector<aoclsparse_int> unsorted_col(nnz);
    std::vector<T>             unsorted_val(nnz);

    // Read entries
    aoclsparse_int idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= nnz)
        {
            CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
        }

        aoclsparse_int irow;
        aoclsparse_int icol;
        T             ival;

        std::istringstream ss(line);

        if(!strcmp(data, "pattern"))
        {
            ss >> irow >> icol;
            ival = static_cast<T>(1);
        }
        else
        {
            read_mtx_value(ss, irow, icol, ival);
        }

        if(base == aoclsparse_index_base_zero)
        {
            --irow;
            --icol;
        }

        unsorted_row[idx] = irow;
        unsorted_col[idx] = icol;
        unsorted_val[idx] = ival;

        ++idx;

        if(symm && irow != icol)
        {
            if(idx >= nnz)
            {
                CHECK_AOCLSPARSE_ERROR(aoclsparse_status_internal_error);
            }

            unsorted_row[idx] = icol;
            unsorted_col[idx] = irow;
            unsorted_val[idx] = ival;
            ++idx;
        }
    }
    fclose(f);
    nnz = idx;
    coo_row_ind.resize(nnz);
    coo_col_ind.resize(nnz);
    coo_val.resize(nnz);

    // Sort by row and column index
    std::vector<aoclsparse_int> perm(nnz);
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&](const aoclsparse_int& a, const aoclsparse_int& b) {
        if(unsorted_row[a] < unsorted_row[b])
        {
            return true;
        }
        else if(unsorted_row[a] == unsorted_row[b])
        {
            return (unsorted_col[a] < unsorted_col[b]);
        }
        else
        {
            return false;
        }
    });

    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        coo_row_ind[i] = unsorted_row[perm[i]];
        coo_col_ind[i] = unsorted_col[perm[i]];
        coo_val[i]     = unsorted_val[perm[i]];
    }

    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        std::cout << "done." << std::endl;
    }
}

/* ==================================================================================== */
/*! \brief  Read matrix from mtx file in CSR format */
template <typename T>
inline void aoclsparse_init_csr_mtx(const char*                 filename,
                                   std::vector<aoclsparse_int>& csr_row_ptr,
                                   std::vector<aoclsparse_int>& csr_col_ind,
                                   std::vector<T>&             csr_val,
                                   aoclsparse_int&              M,
                                   aoclsparse_int&              N,
                                   aoclsparse_int&              nnz,
                                   aoclsparse_index_base        base)
{
    std::vector<aoclsparse_int> coo_row_ind;

    // Read COO matrix
    aoclsparse_init_coo_mtx(filename, coo_row_ind, csr_col_ind, csr_val, M, N, nnz, base);

    // Convert to CSR
    csr_row_ptr.resize(M + 1);
    coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr, base);
}

/* ==================================================================================== */
/*! \brief  Read matrix from binary file in aoclALUTION format */
static inline void read_csr_values(std::ifstream& in, aoclsparse_int nnz, float* csr_val, bool mod)
{
    // Temporary array to convert from double to float
    std::vector<double> tmp(nnz);

    // Read in double values
    in.read((char*)tmp.data(), sizeof(double) * nnz);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        if(mod)
        {
            csr_val[i] = std::abs(static_cast<float>(tmp[i]));
        }
        else
        {
            csr_val[i] = static_cast<float>(tmp[i]);
        }
    }
}

static inline void read_csr_values(std::ifstream& in, aoclsparse_int nnz, double* csr_val, bool mod)
{
    in.read((char*)csr_val, sizeof(double) * nnz);

    if(mod)
    {
        for(aoclsparse_int i = 0; i < nnz; ++i)
        {
            csr_val[i] = std::abs(csr_val[i]);
        }
    }
}
/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in CSR format */
template <typename T>
inline void aoclsparse_init_csr_random(std::vector<aoclsparse_int>& row_ptr,
                                      std::vector<aoclsparse_int>& col_ind,
                                      std::vector<T>&             val,
                                      aoclsparse_int               M,
                                      aoclsparse_int               N,
                                      aoclsparse_int&              nnz,
                                      aoclsparse_index_base        base,
                                      bool                        full_rank = false)
{
    // Compute non-zero entries of the matrix
    if(!nnz)
        nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;

    // Sample random matrix
    std::vector<aoclsparse_int> row_ind(nnz);

    // Sample COO matrix
    aoclsparse_init_coo_matrix(row_ind, col_ind, val, M, N, nnz, base, full_rank);

    // Convert to CSR
    coo_to_csr(M, nnz, row_ind, row_ptr, base);
}
/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
inline void aoclsparse_init_coo_random(std::vector<aoclsparse_int>& row_ind,
                                      std::vector<aoclsparse_int>& col_ind,
                                      std::vector<T>&             val,
                                      aoclsparse_int               M,
                                      aoclsparse_int               N,
                                      aoclsparse_int&              nnz,
                                      aoclsparse_index_base        base,
                                      bool                        full_rank = false)
{
    // Compute non-zero entries of the matrix
    if(!nnz)
        nnz = M * ((M > 1000 || N > 1000) ? 2.0 / std::max(M, N) : 0.02) * N;

    // Sample random matrix
    aoclsparse_init_coo_matrix(row_ind, col_ind, val, M, N, nnz, base, full_rank);
}
/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in CSR format */
template <typename T>
inline void aoclsparse_init_csr_matrix(std::vector<aoclsparse_int>& csr_row_ptr,
                                      std::vector<aoclsparse_int>& csr_col_ind,
                                      std::vector<T>&             csr_val,
                                      aoclsparse_int&              M,
                                      aoclsparse_int&              N,
                                      aoclsparse_int&              nnz,
                                      aoclsparse_index_base        base,
                                      aoclsparse_matrix_init       matrix,
                                      const char*                 filename,
                                      bool                        toint     = false,
                                      bool                        full_rank = false)
{
    // Differentiate the different matrix generators
    if(matrix == aoclsparse_matrix_random)
    {
        aoclsparse_init_csr_random(csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base, full_rank);
    }
    else if(matrix == aoclsparse_matrix_file_mtx)
    {
        aoclsparse_init_csr_mtx(filename, csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base);
    }
}
/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in COO format */
template <typename T>
inline void aoclsparse_init_coo_matrix(std::vector<aoclsparse_int>& coo_row_ind,
                                      std::vector<aoclsparse_int>& coo_col_ind,
                                      std::vector<T>&             coo_val,
                                      aoclsparse_int&              M,
                                      aoclsparse_int&              N,
                                      aoclsparse_int&              nnz,
                                      aoclsparse_index_base        base,
                                      aoclsparse_matrix_init       matrix,
                                      const char*                 filename,
                                      bool                        toint     = false,
                                      bool                        full_rank = false)
{
    // Differentiate the different matrix generators
    if(matrix == aoclsparse_matrix_random)
    {
        aoclsparse_init_coo_random(coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, full_rank);
    }
    else if(matrix == aoclsparse_matrix_file_mtx)
    {
        aoclsparse_init_coo_mtx(filename, coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base);
    }
}
#endif // AOCLSPARSE_INIT_HPP
