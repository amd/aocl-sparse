/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse.h"
#include "aoclsparse_datatype2string.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_test.hpp"

#include <algorithm>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

/* ==================================================================================== */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
inline void aoclsparse_init(
    std::vector<T> &A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator_normal<T>();
}

// Initializes sparse index vector with nnz entries ranging from start to end
inline void
    aoclsparse_init_index(std::vector<aoclsparse_int> &x, size_t nnz, size_t start, size_t end)
{
    std::vector<bool> check(end - start, false);

    size_t num = 0;

    // produce at most full vector
    if(nnz > end - start)
        nnz = end - start;

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

// Decide the size of a sparse vector, if nnz is negative, it provides sparsity in pct
inline aoclsparse_int aoclsparse_init_spvec_size(aoclsparse_int nnz, aoclsparse_int n)
{
    if(nnz >= 0)
        return (std::min)(nnz, n);
    else if(nnz <= -100)
        return n;
    else
        return (aoclsparse_int)(0.01 * (-nnz) * n);
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
    std::vector<T> &A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
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
                       const std::vector<aoclsparse_int> &coo_row_ind,
                       std::vector<aoclsparse_int>       &csr_row_ptr,
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
                       const std::vector<aoclsparse_int> &csr_row_ptr,
                       std::vector<aoclsparse_int>       &coo_row_ind,
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
inline void aoclsparse_init_nan(T *A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(aoclsparse_nan_rng());
}

template <typename T>
inline void aoclsparse_init_nan(
    std::vector<T> &A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(aoclsparse_nan_rng());
}
/* ==================================================================================== */
/*! \brief  Generate a random sparse matrix in COO format in the given base and sizes.
 * If nnz<0, it is interpretted as the desired sparsity percentage (nnz=-3 -> 3% sparsity).
 * if square and full_diag, the resulting matrix will have full diagonal
 * and will be diagonally dominant. If suggested nnz was too small,
 * it will get increased.
 * If is_herm, the diagonal entries will have imag=0. */
template <typename T>
aoclsparse_status aoclsparse_generate_coo_matrix(std::vector<aoclsparse_int> &row_ind,
                                                 std::vector<aoclsparse_int> &col_ind,
                                                 std::vector<T>              &val,
                                                 aoclsparse_int               M,
                                                 aoclsparse_int               N,
                                                 aoclsparse_int              &nnz,
                                                 aoclsparse_index_base        base,
                                                 bool                         full_diag = false,
                                                 bool                         is_herm   = false)
{
    if(base != aoclsparse_index_base_zero && base != aoclsparse_index_base_one)
        return aoclsparse_status_invalid_value;

    if(M < 0 || N < 0)
        return aoclsparse_status_invalid_size;

    if(nnz <= -100)
        return aoclsparse_status_invalid_value;
    else if(nnz < 0)
        nnz = (-nnz / 100.0) * M * N;

    if((size_t)nnz > (size_t)M * (size_t)N)
        return aoclsparse_status_invalid_value;

    // Increase nnz if too small to be diagonally dominant (if requested)
    full_diag = M == N && full_diag;
    if(full_diag)
        if(nnz < M)
            nnz = M;

    //allocate coo arrays (row_ind[],  col_ind[], val[]) of sufficient space i.e., nnz
    if(row_ind.size() != (size_t)nnz)
        row_ind.resize(nnz);
    if(col_ind.size() != (size_t)nnz)
        col_ind.resize(nnz);
    if(val.size() != (size_t)nnz)
        val.resize(nnz);

    // reserve size not to return NULL pointers with nnz=0
    if(nnz == 0)
    {
        row_ind.reserve(1);
        col_ind.reserve(1);
        val.reserve(1);
    }

    // Add diagonal entry, if full rank is flagged
    aoclsparse_int i = 0, row_idx = 0;

    // Uniform distributed row indices
    aoclsparse_int rand_row_ind;
    //array to store no of nnz entries in each row
    std::vector<aoclsparse_int> nnzs_row_wise(M, 0);
    /*
        if full-diagonal is requested, make sure
            1. row_ind[] which stores row indices (COO) first covers indices for diagonal entries
            2. update nnzs_row_wise[] to indicate diagonals were considered
    */
    if(full_diag)
    {
        //Generate diagonal indices first
        for(; i < M; ++i)
        {
            row_ind[i] = i;
            nnzs_row_wise[i]++;
        }
    }
    /*
        if full-diagonal is requested, then use the pre-filled arrays of
        row_ind[] and nnzs_row_wise[] to pick out indices for the remaning nnz
        entries
    */
    for(; i < nnz; ++i)
    {
        //start with same loop index variable 'i' so that nnz counting starts from where it left off
        do
        {
            //Generate a random row index between 0 to M-1
            rand_row_ind = random_generator<aoclsparse_int>(0, M - 1);
        } while(nnzs_row_wise[rand_row_ind]
                >= N); //condition to avoid any row storing more than N nnz entries
        row_ind[i] = rand_row_ind;
        //accumulate nnz for that specific row
        nnzs_row_wise[rand_row_ind]++;
    }
    // Sort row indices
    std::sort(row_ind.begin(), row_ind.end());

    // Sample column indices
    std::vector<bool> check(N, false);
    /*
        ri_row_start_idx: indexes to first entry of current row index in row_ind[]
        ri_row_end_idx: indexes to first entry of next row index in row_ind[]
        difference gives no of nnz entries for the current row
    */
    aoclsparse_int ri_row_start_idx, ri_row_end_idx, col_idx;
    ri_row_end_idx = 0;
    while(ri_row_end_idx < nnz)
    {
        ri_row_start_idx = ri_row_end_idx;
        //count how many row indices of same value exist, which should ideally
        //give us the no of nnz to be filled in that specific row
        while(row_ind[ri_row_end_idx] == row_ind[ri_row_start_idx])
        {
            ++ri_row_end_idx;
            if(ri_row_end_idx >= nnz)
            {
                break;
            }
        }

        // Sample (ri_row_end_idx - ri_row_start_idx) column indices
        col_idx = ri_row_start_idx;
        row_idx = row_ind[ri_row_start_idx];
        while(col_idx < ri_row_end_idx)
        {
            //fill the diagonals first
            if(full_diag && !check[row_idx])
            {
                check[row_idx]   = true;
                col_ind[col_idx] = row_idx;
                col_idx++;
                continue;
            }
            //uniform distribution from 0 to n-1, sample the whole row
            aoclsparse_int rng = random_generator<aoclsparse_int>(0, N - 1);
            if(full_diag)
            {
                //if full-diagonal, sample entries other than diagonal index since it is already marked
                if((col_idx != row_idx) && !check[rng])
                {
                    check[rng]         = true;
                    col_ind[col_idx++] = rng;
                }
            }
            else
            {
                //for non-full-diagonal cases, sample all the nnzs allocated for this row
                if(!check[rng])
                {
                    check[rng]         = true;
                    col_ind[col_idx++] = rng;
                }
            }
        }

        //reset check[] array that indicates all the column index entries for a row
        std::fill(check.begin(), check.end(), false);
        // Partially sort column indices
        std::sort((&col_ind[0] + ri_row_start_idx), (&col_ind[0] + ri_row_end_idx));
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
    T max_val_entry = random_generator_normal<T>();

    // Sample random off-diagonal values
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        val[i] = random_generator_normal<T>();
        // Sample nnz values, and collect the max nnz entry along the way
        if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                     || std::is_same_v<T, aoclsparse_double_complex>)
        {
            max_val_entry.real = (std::max)(std::abs(val[i].real), max_val_entry.real);
            if(!is_herm)
            {
                max_val_entry.imag = (std::max)(std::abs(val[i].imag), max_val_entry.imag);
            }
            else
            {
                max_val_entry.imag = 0.0;
            }
        }
        else
        {
            max_val_entry = (std::max)(std::abs(val[i]), max_val_entry);
        }
    }
    if(full_diag)
    {
        // choosing a large number to scale the diagonal element, so as to
        // generate a diagonally dominant matrix with low condition number
        const double scale = 4. * 10.;
        for(aoclsparse_int i = 0; i < nnz; ++i)
        {
            //scale the diagonal element
            if(row_ind[i] == col_ind[i])
            {
                // rare case of val[i] == zero, in this case replace with 1.
                if(val[i] == aoclsparse_numeric::zero<T>())
                {
                    if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                                 || std::is_same_v<T, aoclsparse_double_complex>)
                    {
                        val[i].real = 1.0;
                        if(!is_herm)
                        {
                            val[i].imag = 1.0;
                        }
                        else
                        {
                            val[i].imag = 0.0;
                        }
                    }
                    else
                    {
                        val[i] = 1.0;
                    }
                }
                else
                {
                    //adjust diagonal element to greater than max(0, 10 x abs(L(0..k-1))
                    if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                                 || std::is_same_v<T, aoclsparse_double_complex>)
                    {
                        val[i].real = scale * max_val_entry.real;
                        if(!is_herm)
                        {
                            val[i].imag = scale * max_val_entry.imag;
                        }
                        else
                        {
                            val[i].imag = 0.0;
                        }
                    }
                    else
                    {
                        val[i] = scale * max_val_entry;
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
static inline void
    read_mtx_value(std::istringstream &is, aoclsparse_int &row, aoclsparse_int &col, float &val)
{
    is >> row >> col >> val;
}

static inline void
    read_mtx_value(std::istringstream &is, aoclsparse_int &row, aoclsparse_int &col, double &val)
{
    is >> row >> col >> val;
}

template <typename T>
static inline void
    read_mtx_value(std::istringstream &is, aoclsparse_int &row, aoclsparse_int &col, T &val)
{
    if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                 || std::is_same_v<T, std::complex<float>>)
    {
        // initialize with some value other than 1/0
        float real = 1.3;
        float imag = 2.7;

        is >> row >> col >> real >> imag;
        val = {real, imag};
    }

    if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                 || std::is_same_v<T, std::complex<double>>)
    {
        // initialize with some value other than 1/0
        double real = 1.3;
        double imag = 2.7;

        is >> row >> col >> real >> imag;

        val = {real, imag};
    }
}

static inline void val_init(float *val)
{
    *val = 1;
}

static inline void val_init(double *val)
{
    *val = 1;
}

template <typename T>
static inline void val_init(T *val, float set = 1)
{
    *val = {set, set};
}

/* ==================================================================================== */
/*! \brief  Read matrix from mtx file in COO format*/
template <typename T>
inline aoclsparse_status aoclsparse_readmtx_coo(const char                  *filename,
                                                std::vector<aoclsparse_int> &coo_row_ind,
                                                std::vector<aoclsparse_int> &coo_col_ind,
                                                std::vector<T>              &coo_val,
                                                aoclsparse_int              &M,
                                                aoclsparse_int              &N,
                                                aoclsparse_int              &nnz,
                                                aoclsparse_index_base        base,
                                                bool                        &issymm,
                                                bool                         general)
{
    aoclsparse_status status = aoclsparse_status_success;
    const char       *env    = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        //std::cout << "Reading matrix " << filename << " ... ";
    }

    FILE *f = fopen(filename, "r");
    if(NULL == f)
    {
        return aoclsparse_status_internal_error;
    }

    char line[1024];

    // Check for banner
    if(!fgets(line, 1024, f))
    {
        return aoclsparse_status_internal_error;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if(sscanf(line, "%s %s %s %s %s", banner, array, coord, data, type) != 5)
    {
        return aoclsparse_status_internal_error;
    }

    // Convert to lower case
    for(char *p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char *p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char *p = data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char *p = type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        return aoclsparse_status_internal_error;
    }

    // Check array type
    if(strcmp(array, "matrix") != 0)
    {
        return aoclsparse_status_internal_error;
    }

    // Check coord
    if(strcmp(coord, "coordinate") != 0)
    {
        return aoclsparse_status_internal_error;
    }

    // Check data
    if(strcmp(data, "real") != 0 && strcmp(data, "integer") != 0 && strcmp(data, "pattern") != 0
       && strcmp(data, "complex") != 0)
    {
        return aoclsparse_status_internal_error;
    }

    // Check type
    if(strcmp(type, "general") != 0 && strcmp(type, "symmetric") != 0
       && strcmp(type, "hermitian") != 0)
    {
        return aoclsparse_status_internal_error;
    }

    // Symmetric flag
    aoclsparse_int symm = !strcmp(type, "symmetric");
    // Hermition flag
    bool hermitian = !strcmp(type, "hermitian");
    issymm         = (symm ? true : false) || hermitian;
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

    int inrow = 0;
    int incol = 0;
    int innz  = 0;

    sscanf(line, "%d %d %d", &inrow, &incol, &innz);

    M    = static_cast<aoclsparse_int>(inrow);
    N    = static_cast<aoclsparse_int>(incol);
    snnz = static_cast<aoclsparse_int>(innz);

    nnz = (symm || hermitian) ? snnz * 2 : snnz;

    std::vector<aoclsparse_int> unsorted_row(nnz);
    std::vector<aoclsparse_int> unsorted_col(nnz);
    std::vector<T>              unsorted_val(nnz);

    // Read entries
    aoclsparse_int idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= nnz)
        {
            return aoclsparse_status_internal_error;
        }

        aoclsparse_int irow = 0;
        aoclsparse_int icol = 0;
        T              ival = aoclsparse_numeric::zero<T>();

        std::istringstream ss(line);

        if(!strcmp(data, "pattern"))
        {
            ss >> irow >> icol;
            val_init(&ival);
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
        if(general)
        {
            if(symm && irow != icol)
            {
                if(idx >= nnz)
                {
                    return aoclsparse_status_internal_error;
                }

                unsorted_row[idx] = icol;
                unsorted_col[idx] = irow;
                unsorted_val[idx] = ival;
                ++idx;
            }
            else if(hermitian && irow != icol)
            {
                if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                             || std::is_same_v<T, aoclsparse_double_complex>)
                {
                    unsorted_row[idx] = icol;
                    unsorted_col[idx] = irow;
                    unsorted_val[idx] = {ival.real, -(ival.imag)};
                    ++idx;
                }
                if constexpr(std::is_same_v<T, std::complex<float>>
                             || std::is_same_v<T, std::complex<double>>)
                {
                    unsorted_row[idx] = icol;
                    unsorted_col[idx] = irow;
                    unsorted_val[idx] = {ival.real(), -(ival.imag())};
                    ++idx;
                }
            }
        }
    }
    fclose(f);
    nnz = idx;

    std::vector<aoclsparse_int> perm(nnz);
    coo_row_ind.resize(nnz);
    coo_col_ind.resize(nnz);
    coo_val.resize(nnz);

    // Sort by row and column index
    for(aoclsparse_int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&](const aoclsparse_int &a, const aoclsparse_int &b) {
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
        //std::cout << "done." << std::endl;
    }
    return status;
}

/* ==================================================================================== */
/*! \brief  Read matrix from mtx file in COO format and then convert to CSR format */
template <typename T>
inline aoclsparse_status aoclsparse_readmtx_csr(const char                  *filename,
                                                std::vector<aoclsparse_int> &csr_row_ptr,
                                                std::vector<aoclsparse_int> &csr_col_ind,
                                                std::vector<T>              &csr_val,
                                                aoclsparse_int              &M,
                                                aoclsparse_int              &N,
                                                aoclsparse_int              &nnz,
                                                aoclsparse_index_base        base,
                                                bool                        &issymm,
                                                bool                         general)
{
    aoclsparse_status           status = aoclsparse_status_success;
    std::vector<aoclsparse_int> coo_row_ind;

    // Read COO matrix
    status = aoclsparse_readmtx_coo(
        filename, coo_row_ind, csr_col_ind, csr_val, M, N, nnz, base, issymm, general);
    if(status != aoclsparse_status_success)
    {
        return status;
    }
    // Convert to CSR
    coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr, base);
    return status;
}

/* ==================================================================================== */
/*! \brief  Using Fisher-Yates algorithm, the below function shuffles arrays between a range of
    indices
*/
template <typename T>
inline void aoclsparse_shuffle_core(aoclsparse_int  start_idx,
                                    aoclsparse_int  end_idx,
                                    aoclsparse_int *col_arr,
                                    T              *val_arr,
                                    aoclsparse_int *row_arr
                                    = nullptr) //optional: csr does not need it
{
    aoclsparse_int nnzs_in_triang = end_idx - start_idx;
    if(nnzs_in_triang < 2)
    {
        //quick exit
        return;
    }
    else if(nnzs_in_triang == 2)
    {
        //swap the 2 elements
        std::swap(col_arr[0], col_arr[1]);
        std::swap(val_arr[0], val_arr[1]);
        if(row_arr)
        {
            std::swap(row_arr[0], row_arr[1]);
        }
    }
    for(aoclsparse_int j = end_idx - 1; j > (start_idx + 1); j--)
    {
        //random index between start_idx and end_idx
        size_t k = random_generator<size_t>(start_idx, j);
        std::swap(col_arr[k], col_arr[j]);
        std::swap(val_arr[k], val_arr[j]);
        if(row_arr)
        {
            std::swap(row_arr[k], row_arr[j]);
        }
    }
}
/* ==================================================================================== */
/*! \brief  Shuffle the column index and value arrays in CSR and the triplet arrays in COO
            using Fisher-Yates Shuffle. Refer Fisher-Yates algorithm
*/
template <typename T>
inline void aoclsparse_full_shuffle(aoclsparse_matrix_format_type mtype,
                                    std::vector<aoclsparse_int>  &row_ptr,
                                    std::vector<aoclsparse_int>  &col_ind,
                                    std::vector<T>               &val,
                                    aoclsparse_int                m,
                                    aoclsparse_int                nnz,
                                    aoclsparse_index_base         base)
{
    if(mtype == aoclsparse_csr_mat || mtype == aoclsparse_csc_mat)
    {
        for(aoclsparse_int i = 0; i < m; i++)
        {
            aoclsparse_int rowstart, rowend, nnz_in_row;
            rowstart   = row_ptr[i] - base;
            rowend     = row_ptr[i + 1] - base;
            nnz_in_row = rowend - rowstart;
            if(nnz_in_row)
            {
                aoclsparse_shuffle_core(0, nnz_in_row, &col_ind[rowstart], &val[rowstart]);
            }
        }
    }
    else if(mtype == aoclsparse_coo_mat)
    {
        aoclsparse_shuffle_core(0, nnz, &col_ind[0], &val[0], &row_ptr[0]);
    }
}
/* ==================================================================================== */
/*! \brief  Shuffle the index and value arrays in CSR/CSC using Fisher-Yates Shuffle.
            Refer Fisher-Yates algorithm. Either lower or upper triangle is shuffled
            with leading diagonal untouched, so the function does partial shuffling
*/
template <typename T>
inline void aoclsparse_partial_shuffle(std::vector<aoclsparse_int> &row_ptr,
                                       std::vector<aoclsparse_int> &col_ind,
                                       std::vector<T>              &val,
                                       aoclsparse_int               m,
                                       aoclsparse_index_base        base)
{
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int rowstart, rowend, diag_index = 0;
        aoclsparse_int nnz_in_L, nnz_in_row;
        rowstart   = row_ptr[i] - base;
        rowend     = row_ptr[i + 1] - base;
        nnz_in_row = rowend - rowstart;
        for(aoclsparse_int j = rowstart; j < rowend; j++)
        {
            aoclsparse_int col_idx = col_ind[j] - base;
            //is this diagonal
            if(col_idx == i)
            {
                diag_index = j;
                break;
            }
        }
        nnz_in_L = diag_index - rowstart;
        //shuffle strictly lower traingle section
        aoclsparse_shuffle_core(0, nnz_in_L, &col_ind[rowstart], &val[rowstart], nullptr);
        //shuffle strictly upper traingle section
        aoclsparse_shuffle_core(
            diag_index + 1 - rowstart, nnz_in_row, &col_ind[rowstart], &val[rowstart], nullptr);
    }
}
/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in COO format. Also handle whether to generate a
            matrix with full diagonals based on input matrix type. The level of sorting
            (full, unsorting) also is controlled using a input parameter 'sort'. Partial
            Sorting is not supported for COO format */
template <typename T>
aoclsparse_status aoclsparse_init_coo_matrix(std::vector<aoclsparse_int> &coo_row_ind,
                                             std::vector<aoclsparse_int> &coo_col_ind,
                                             std::vector<T>              &coo_val,
                                             aoclsparse_int              &M,
                                             aoclsparse_int              &N,
                                             aoclsparse_int              &nnz,
                                             aoclsparse_index_base        base,
                                             aoclsparse_matrix_init       matrix,
                                             const char                  *filename,
                                             bool                        &issymm,
                                             bool                         general,
                                             aoclsparse_matrix_sort sort = aoclsparse_unsorted)
{
    aoclsparse_status status = aoclsparse_status_success;

    //check matrix differentiator
    if(matrix != aoclsparse_matrix_file_mtx && matrix != aoclsparse_matrix_random
       && matrix != aoclsparse_matrix_random_diag_dom
       && matrix != aoclsparse_matrix_herm_random_diag_dom)
    {
        return aoclsparse_status_invalid_value;
    }
    //check sort option
    if(sort != aoclsparse_unsorted && sort != aoclsparse_partially_sorted
       && sort != aoclsparse_fully_sorted)
    {
        return aoclsparse_status_invalid_value;
    }
    //partial sorting in coo not supported
    if(sort == aoclsparse_partially_sorted)
    {
        return aoclsparse_status_not_implemented;
    }

    //deal with file based mtx input first, and get matrix dimensions
    if(matrix == aoclsparse_matrix_file_mtx)
    {
        //process a matrix market file
        status = aoclsparse_readmtx_coo(
            filename, coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, issymm, general);
    }
    else if(matrix == aoclsparse_matrix_random)
    {
        //generate a random matrix whose diagonal may or may not contain full-diagonal
        status = aoclsparse_generate_coo_matrix(
            coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, false, false);
    }
    else if(matrix == aoclsparse_matrix_random_diag_dom)
    {
        /*
        generates random matrix with dominant diagonals and since only the triangular portion is
        considered during an operation on a trinagular/symmetric matrix (L+D or D+U), we don't need to
        generate explicit symmetric matrices.
        */
        status = aoclsparse_generate_coo_matrix(
            coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, true, false);
    }
    else if(matrix == aoclsparse_matrix_herm_random_diag_dom)
    {
        /*
        generates random complex matrix with real diagonals and since only the triangular portion is
        considered during an operation on a hermitian matrix (L+D or D+U), we don't need to
        generate explicit hermitian matrices.
        */
        status = aoclsparse_generate_coo_matrix(
            coo_row_ind, coo_col_ind, coo_val, M, N, nnz, base, true, true);
    }

    if(status != aoclsparse_status_success)
        return status;

    //default random generation in COO, is a sorted operation. So do nothing.
    if(sort == aoclsparse_unsorted)
    {
        //shuffle the coo col_ind and csr_val arrays
        aoclsparse_full_shuffle(
            aoclsparse_coo_mat, coo_row_ind, coo_col_ind, coo_val, M, nnz, base);
    }

    return status;
}
/* ==================================================================================== */
/*! \brief  Initialize a sparse matrix in CSR format. Also handle whether to generate a
            matrix with full diagonals based on input matrix type. The level of sorting
            (full, partial, unsorting) also is controlled using a input parameter 'sort'
 */
template <typename T>
aoclsparse_status aoclsparse_init_csr_matrix(std::vector<aoclsparse_int> &csr_row_ptr,
                                             std::vector<aoclsparse_int> &csr_col_ind,
                                             std::vector<T>              &csr_val,
                                             aoclsparse_int              &M,
                                             aoclsparse_int              &N,
                                             aoclsparse_int              &nnz,
                                             aoclsparse_index_base        base,
                                             aoclsparse_matrix_init       matrix,
                                             const char                  *filename,
                                             bool                        &issymm,
                                             bool                         general,
                                             aoclsparse_matrix_sort sort = aoclsparse_unsorted)
{
    aoclsparse_status           status = aoclsparse_status_success;
    std::vector<aoclsparse_int> coo_row_ind;

    //check sort option
    if(sort != aoclsparse_unsorted && sort != aoclsparse_partially_sorted
       && sort != aoclsparse_fully_sorted)
        return aoclsparse_status_invalid_value;

    // generate COO matrix of the same parameters which is fully sorted
    // (that's default) so that we can apply coo_csr()
    status = aoclsparse_init_coo_matrix(coo_row_ind,
                                        csr_col_ind,
                                        csr_val,
                                        M,
                                        N,
                                        nnz,
                                        base,
                                        matrix,
                                        filename,
                                        issymm,
                                        general,
                                        aoclsparse_fully_sorted);
    if(status != aoclsparse_status_success)
        return status;

    // convert to CSR (already in that order, just need the correct csr_row_ptr
    coo_to_csr(M, nnz, coo_row_ind, csr_row_ptr, base);

    //default random generation in COO, is a sorted operation. So do nothing.
    if(sort == aoclsparse_unsorted)
    {
        //shuffle the csr col_ind and csr_val arrays
        aoclsparse_full_shuffle(
            aoclsparse_csr_mat, csr_row_ptr, csr_col_ind, csr_val, M, nnz, base);
    }
    else if(sort == aoclsparse_partially_sorted)
    {
        aoclsparse_partial_shuffle(csr_row_ptr, csr_col_ind, csr_val, M, base);
    }
    return status;
}
/* ==================================================================================== */
/*! \brief Create a random aoclsparse_matrix in the given format (csr, csc, coo) and return
 *  the underlying vectors as 4 arrays - row and column indices and values and compressed
 *  pointer array (for CSR or CSC, otherwise empty). Invalid values (e.g., M, N, NNZ<0) will
 *  get corrected to 0. */
template <typename T>
aoclsparse_status aoclsparse_init_matrix_random(aoclsparse_index_base         base,
                                                aoclsparse_int               &M,
                                                aoclsparse_int               &N,
                                                aoclsparse_int               &NNZ,
                                                aoclsparse_matrix_format_type format_type,
                                                std::vector<aoclsparse_int>  &coo_row,
                                                std::vector<aoclsparse_int>  &coo_col,
                                                std::vector<T>               &coo_val,
                                                std::vector<aoclsparse_int>  &ptr,
                                                aoclsparse_matrix            &mat,
                                                bool                          full_diag = false,
                                                bool                          is_herm   = false)
{
    aoclsparse_status status = aoclsparse_status_success;
    mat                      = nullptr;

    status = aoclsparse_generate_coo_matrix(
        coo_row, coo_col, coo_val, M, N, NNZ, base, full_diag, is_herm);
    if(status != aoclsparse_status_success)
        return status;

    switch(format_type)
    {
    case aoclsparse_coo_mat:
        // note, coo indices sorted already by the generator in CSR order
        ptr.clear();
        return aoclsparse_create_coo(
            &mat, base, M, N, NNZ, coo_row.data(), coo_col.data(), coo_val.data());

    case aoclsparse_csr_mat:
        coo_to_csr(M, NNZ, coo_row, ptr, base);
        return aoclsparse_create_csr(
            &mat, base, M, N, NNZ, ptr.data(), coo_col.data(), coo_val.data());

    case aoclsparse_csc_mat:
    {
        // find permutation to match CSC order
        std::vector<aoclsparse_int> perm(NNZ);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [coo_col, coo_row](auto a, auto b) {
            if(coo_col[a] == coo_col[b])
                return coo_row[a] < coo_row[b];
            return coo_col[a] < coo_col[b];
        });
        std::vector<aoclsparse_int> tmp_row = coo_row;
        std::vector<T>              tmp_val = coo_val;
        for(aoclsparse_int i = 0; i < NNZ; i++)
        {
            coo_row[i] = tmp_row[perm[i]];
            coo_val[i] = tmp_val[perm[i]];
        }
        // sorting coo_col has the same effect as shuffling with perm[]
        std::sort(coo_col.begin(), coo_col.end());

        coo_to_csr(N, NNZ, coo_col, ptr, base);
        return aoclsparse_create_csc(
            &mat, base, M, N, NNZ, ptr.data(), coo_row.data(), coo_val.data());
    }
    default:
        return aoclsparse_status_not_implemented;
    }

    return status;
}
#endif // AOCLSPARSE_INIT_HPP
