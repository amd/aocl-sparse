/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <typeinfo>
#include <vector>

// Utilities to compare complex real scalars and vectors =============================================

#define EXPECT_COMPLEX_FLOAT_EQ_VEC(n, x, y)                                     \
    for(auto i = 0; i < n; i++)                                                  \
    {                                                                            \
        EXPECT_FLOAT_EQ(std::real(x[i]), std::real(y[i]))                        \
            << " Real parts of " #x " and " #y " differ at index i = " << i      \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i]); \
        EXPECT_FLOAT_EQ(std::imag(x[i]), std::imag(y[i]))                        \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i]); \
    }

#define EXPECT_COMPLEX_DOUBLE_EQ_VEC(n, x, y)                                    \
    for(auto i = 0; i < n; i++)                                                  \
    {                                                                            \
        EXPECT_DOUBLE_EQ(std::real(x[i]), std::real(y[i]))                       \
            << " Real parts of " #x " and " #y " differ  at index i = " << i     \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i]); \
        EXPECT_DOUBLE_EQ(std::imag(x[i]), std::imag(y[i]))                       \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i]); \
    }

#define EXPECT_COMPLEX_FLOAT_EQ(x, y)                                      \
    do                                                                     \
    {                                                                      \
        EXPECT_FLOAT_EQ(std::real(x), std::real(y))                        \
            << " Real parts of " #x " and " #y " differ."                  \
            << " values are: " << std::real(x) << " and " << std::real(y); \
        EXPECT_FLOAT_EQ(std::imag(x), std::imag(y))                        \
            << " Imaginary parts of " #x " and " #y " differ."             \
            << " values are: " << std::imag(x) << " and " << std::imag(y); \
    } while(0)

#define EXPECT_COMPLEX_DOUBLE_EQ(x, y)                                     \
    do                                                                     \
    {                                                                      \
        EXPECT_DOUBLE_EQ(std::real(x), std::real(y))                       \
            << " Real parts of " #x " and " #y " differ."                  \
            << " values are: " << std::real(x) << " and " << std::real(y); \
        EXPECT_DOUBLE_EQ(std::imag(x), std::imag(y))                       \
            << " Imaginary parts of " #x " and " #y " differ."             \
            << " values are: " << std::imag(x) << " and " << std::imag(y); \
    } while(0)

// Utilities to compare real scalars and vectors =============================================

#define EXPECT_EQ_VEC(n, x, y)                                                              \
    for(auto i = 0; i < n; i++)                                                             \
    {                                                                                       \
        EXPECT_EQ((x)[i], (y)[i]) << " vectors " #x " and " #y " differ at index i = " << i \
                                  << " values are: " << (x)[i] << " and " << (y)[i]         \
                                  << "respectively.";                                       \
    }

#define EXPECT_FLOAT_EQ_VEC(n, x, y)                                                               \
    for(auto i = 0; i < n; i++)                                                                    \
    {                                                                                              \
        EXPECT_FLOAT_EQ((x)[i], (y)[i]) << " vectors " #x " and " #y " differ at index i = " << i  \
                                        << " by abs err: " << abs((x)[i] - (y)[i])                 \
                                        << " rel err: " << abs(((x)[i] - (y)[i]) / (x)[i]) << "."; \
    }

#define EXPECT_DOUBLE_EQ_VEC(n, x, y)                                  \
    for(auto i = 0; i < n; i++)                                        \
    {                                                                  \
        EXPECT_DOUBLE_EQ((x)[i], (y)[i])                               \
            << " vectors " #x " and " #y " differ at index i = " << i  \
            << " by abs err: " << abs((x)[i] - (y)[i])                 \
            << " rel err: " << abs(((x)[i] - (y)[i]) / (x)[i]) << "."; \
    }

#define EXPECT_EQ_VEC_ERR(n, x, y)                                                                 \
    for(size_t i = 0; i < n; i++)                                                                  \
    {                                                                                              \
        EXPECT_EQ((x)[i], ((y)[i])) << " vectors " #x " and " #y " differ at index i = " << i      \
                                    << " by abs err: " << abs((x)[i] - ((y)[i]))                   \
                                    << " rel err: " << abs(((x)[i] - *((y) + i)) / (x)[i]) << "."; \
    }

#define EXPECT_ARR_NEAR(n, x, y, abs_error)    \
    for(int j = 0; j < (n); j++)               \
    EXPECT_NEAR(((x)[j]), ((y)[j]), abs_error) \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

// Define precision to which we expect the results to match ==================================

template <typename T>
struct safeguard
{
};
// Add safeguarding scaling (may differ for each data type)
template <>
struct safeguard<double>
{
    static constexpr double value = 1.0;
};
template <>
struct safeguard<float>
{
    static constexpr float value = 2.0f;
};

template <typename T>
constexpr T expected_precision(T scale = (T)1.0) noexcept
{
    const T macheps      = std::numeric_limits<T>::epsilon();
    const T safe_macheps = (T)2.0 * macheps;
    return scale * safeguard<T>::value * sqrt(safe_macheps);
}

// Convenience templated interfaces ==========================================================

template <typename T>
aoclsparse_status itsol_solve(
    aoclsparse_itsol_handle    handle,
    aoclsparse_int             n,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    const T                   *b,
    T                         *x,
    T                          rinfo[100],
    aoclsparse_int precond(aoclsparse_int flag, aoclsparse_int n, const T *u, T *v, void *udata),
    aoclsparse_int monit(aoclsparse_int n, const T *x, const T *r, T rinfo[100], void *udata),
    void          *udata);

template <typename T>
aoclsparse_status itsol_rci_solve(aoclsparse_itsol_handle   handle,
                                  aoclsparse_itsol_rci_job *ircomm,
                                  T                       **u,
                                  T                       **v,
                                  T                        *x,
                                  T                         rinfo[100]);

template <typename T>
aoclsparse_status itsol_init(aoclsparse_itsol_handle *handle);

template <typename T>
aoclsparse_status create_aoclsparse_matrix(aoclsparse_matrix           &A,
                                           aoclsparse_int               m,
                                           aoclsparse_int               n,
                                           aoclsparse_int               nnz,
                                           std::vector<aoclsparse_int> &csr_row_ptr,
                                           std::vector<aoclsparse_int> &csr_col_ind,
                                           std::vector<T>              &csr_val);

// Problem DATABASES =========================================================================
// DB for matrices: returns a matrix
// Use create_matrix(...)
enum matrix_id
{
    /* Used mainly by hint_tests*/
    N5_full_sorted,
    N5_full_unsorted,
    N59_partial_sort,
    N5_1_hole,
    N5_empty_rows,
    N10_random,
    M5_rect_N7,
    M5_rect_N7_2holes,
    M7_rect_N5,
    M7_rect_N5_2holes,
    /****************************/
    /* CG tests matrices */
    sample_cg_mat, // matrix from the CG example
    /* GMRES tests matrices */
    sample_gmres_mat_01,
    sample_gmres_mat_02,
    sample_gmres_mat_03,
    invalid_mat,
};

// DB for linear system equations: returns matrix, rhs, expected sol, tolerance, ....
// Use create_linear_system(...)
enum linear_system_id
{
    /* Used by TRSV */
    // diagonal matrix
    D7_Lx_aB, // 0
    D7_LL_Ix_aB, // 1
    D7_LTx_aB, // 2
    D7_LL_ITx_aB, // 3
    D7_Ux_aB, // 4
    D7_UU_Ix_aB, // 5
    D7_UTx_aB, // 6
    D7_UU_ITx_aB, // 7
    // small matrix 7x7
    S7_Lx_aB, // 8
    S7_LL_Ix_aB, // 9
    S7_LTx_aB, // 10
    S7_LL_ITx_aB, // 11
    S7_Ux_aB, // 12
    S7_UU_Ix_aB, // 13
    S7_UTx_aB, // 14
    S7_UU_ITx_aB, // 15
    // matrix 25x25
    N25_Lx_aB, // 16
    N25_LL_Ix_aB, // 17
    N25_LTx_aB, // 18
    N25_LL_ITx_aB, // 19
    N25_Ux_aB, // 20
    N25_UU_Ix_aB, // 21
    N25_UTx_aB, // 22
    N25_UU_ITx_aB, // 23
    // Matrix used for hinting
    D7Lx_aB_hint, // 24
    A_nullptr, // 25
    D1_descr_nullptr, // 26
    D1_neg_num_hint, // 27
    D1_mattype_gen_hint // 28
};

template <typename T>
aoclsparse_status create_matrix(matrix_id                    mid,
                                aoclsparse_int              &m,
                                aoclsparse_int              &n,
                                aoclsparse_int              &nnz,
                                std::vector<aoclsparse_int> &csr_row_ptr,
                                std::vector<aoclsparse_int> &csr_col_ind,
                                std::vector<T>              &csr_val,
                                aoclsparse_matrix           &A,
                                aoclsparse_mat_descr        &descr,
                                aoclsparse_int               verbose)
{
    aoclsparse_status ret = aoclsparse_status_success;

    // default descriptor
    aoclsparse_create_mat_descr(&descr);

    switch(mid)
    {
    case N5_full_sorted:
        // Small sorted matrix with full diagonal
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  4  0  0
        //  0  5  0  6  7
        //  0  0  0  0  8
        n = m       = 5;
        nnz         = 8;
        csr_row_ptr = {0, 2, 3, 4, 7, 8};
        csr_col_ind = {0, 3, 1, 2, 1, 3, 4, 4};
        csr_val     = {1, 2, 3, 4, 5, 6, 7, 8};
        break;

    case N5_full_unsorted:
        // same as N5 full sorted with rows 0 and 3 shuffled
        n = m       = 5;
        nnz         = 8;
        csr_row_ptr = {0, 2, 3, 4, 7, 8};
        csr_col_ind = {3, 0, 1, 2, 3, 1, 4, 4};
        csr_val     = {2, 1, 3, 4, 6, 5, 7, 8};
        break;

    case N59_partial_sort:
        // Small partially sorted matrix with full diagonal
        // Already clean no need to copy
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  4  0  0
        //  0  5  9  6  7 -> stored as (9,5 | 6 | 7) so 9 and 5 are swaped
        //  0  0  0  0  8
        n = m       = 5;
        nnz         = 9;
        csr_row_ptr = {0, 2, 3, 4, 8, 9};
        csr_col_ind = {0, 3, 1, 2, 2, 1, 3, 4, 4};
        csr_val     = {1, 2, 3, 4, 9, 5, 6, 7, 8};
        break;

    case N5_1_hole:
        // same as N5 full unsorted with row 3 diag element removed
        n = m       = 5;
        nnz         = 7;
        csr_row_ptr = {0, 2, 3, 4, 6, 7};
        csr_col_ind = {3, 0, 1, 2, 1, 4, 4};
        csr_val     = {2, 1, 3, 4, 5, 7, 8};
        break;

    case N5_empty_rows:
        // removed even more diag elements, creating empty rows
        n = m       = 5;
        nnz         = 5;
        csr_row_ptr = {0, 2, 2, 3, 5, 5};
        csr_col_ind = {3, 0, 2, 1, 4};
        csr_val     = {2, 1, 4, 5, 7};
        break;

    case N10_random:
        // randomly generated matrix with missing elements and unsorted rows
        n = m       = 10;
        nnz         = 42;
        csr_row_ptr = {0, 3, 6, 8, 18, 24, 27, 31, 36, 38, 42};
        csr_col_ind = {9, 4, 6, 3, 8, 6, 0, 6, 4, 6, 7, 1, 2, 9, 3, 8, 5, 0, 6, 2, 1,
                       5, 3, 8, 3, 8, 5, 1, 4, 8, 5, 9, 1, 4, 8, 5, 4, 6, 6, 2, 3, 7};
        csr_val
            = {5.91, 5.95, 7.95, 0.83, 5.48, 6.75, 0.01, 4.78, 9.20, 3.40, 2.26, 3.01, 8.34, 6.82,
               7.40, 1.12, 3.31, 4.96, 2.66, 1.77, 5.28, 8.95, 3.09, 2.37, 4.48, 2.92, 1.46, 6.17,
               8.77, 9.96, 7.19, 9.61, 6.48, 4.95, 6.76, 8.87, 5.07, 3.58, 2.09, 8.66, 6.77, 3.69};
        break;

    case M5_rect_N7:
        // same as N5_full_sorted with 2 added columns
        //  1  0  0  2  0  1  0
        //  0  3  0  0  0  2  0
        //  0  0  4  0  0  0  0
        //  0  5  0  6  7  0  3
        //  0  0  0  0  8  4  5
        n           = 7;
        m           = 5;
        nnz         = 13;
        csr_row_ptr = {0, 3, 5, 6, 10, 13};
        csr_col_ind = {0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6};
        csr_val     = {1, 2, 1, 3, 2, 4, 5, 6, 7, 3, 8, 4, 5};
        break;

    case M5_rect_N7_2holes:
        // same as M5_rect_N7, with missing diag elements in rows 2 an 4
        n           = 7;
        m           = 5;
        nnz         = 11;
        csr_row_ptr = {0, 3, 5, 5, 9, 11};
        csr_col_ind = {0, 3, 5, 1, 5, 1, 3, 4, 6, 5, 6};
        csr_val     = {1, 2, 1, 3, 2, 5, 6, 7, 3, 4, 5};
        break;

    case M7_rect_N5:
        // ame as N5_full_sorted with 2 added rows
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  4  0  0
        //  0  5  0  6  7
        //  0  0  0  0  8
        //  0  1  2  0  0
        //  3  0  0  4  0
        n           = 5;
        m           = 7;
        nnz         = 12;
        csr_row_ptr = {0, 2, 3, 4, 7, 8, 10, 12};
        csr_col_ind = {0, 3, 1, 2, 1, 3, 4, 4, 1, 2, 0, 3};
        csr_val     = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4};
        break;

    case M7_rect_N5_2holes:
        // ame as N5_full_sorted with 2 added rows
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  0  0  0
        //  0  5  0  6  7
        //  0  0  0  0  0
        //  0  1  2  0  0
        //  3  0  0  4  0
        n = m       = 5;
        m           = 7;
        nnz         = 10;
        csr_row_ptr = {0, 2, 3, 3, 6, 6, 8, 10};
        csr_col_ind = {0, 3, 1, 3, 1, 4, 1, 2, 0, 3};
        csr_val     = {1, 2, 3, 6, 5, 7, 1, 2, 3, 4};
        break;

    case sample_cg_mat:
        // matrix from the CG sample examples
        // symmetric, lower triangle filled
        n = m       = 8;
        nnz         = 18;
        csr_row_ptr = {0, 1, 2, 5, 6, 8, 11, 15, 18};
        csr_col_ind = {0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
        csr_val     = {19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        break;

    case sample_gmres_mat_01:
        // matrix from the GMRES sample examples
        //symmetry = 0
        //"cage4.mtx"
        n           = 9;
        m           = 9;
        nnz         = 49;
        csr_row_ptr = {0, 5, 10, 15, 20, 26, 32, 38, 44, 49};
        csr_col_ind = {0, 1, 3, 4, 7, 0, 1, 2, 4, 5, 1, 2, 3, 5, 6, 0, 2, 3, 6, 7, 0, 1, 4, 5, 6,
                       8, 1, 2, 4, 5, 7, 8, 2, 3, 4, 6, 7, 8, 0, 3, 5, 6, 7, 8, 4, 5, 6, 7, 8};
        csr_val     = {0.75, 0.14, 0.11, 0.14, 0.11, 0.08, 0.69, 0.11, 0.08, 0.11, 0.09, 0.67, 0.08,
                   0.09, 0.08, 0.09, 0.14, 0.73, 0.14, 0.09, 0.04, 0.04, 0.54, 0.14, 0.11, 0.25,
                   0.05, 0.05, 0.08, 0.45, 0.08, 0.15, 0.04, 0.04, 0.09, 0.47, 0.09, 0.18, 0.05,
                   0.05, 0.14, 0.11, 0.55, 0.25, 0.08, 0.08, 0.09, 0.08, 0.17};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
        break;
    case sample_gmres_mat_02:
        // matrix from the GMRES sample examples
        //symmetry = 1
        //"Trefethen_20b.mtx"
        n   = 19;
        m   = 19;
        nnz = 147;
        csr_row_ptr
            = {0, 6, 13, 21, 28, 36, 44, 52, 60, 69, 78, 87, 95, 103, 111, 119, 126, 134, 141, 147};
        csr_col_ind
            = {0,  1,  2,  4,  8,  16, 0,  1,  2,  3,  5,  9,  17, 0,  1,  2,  3,  4,  6,  10, 18,
               1,  2,  3,  4,  5,  7,  11, 0,  2,  3,  4,  5,  6,  8,  12, 1,  3,  4,  5,  6,  7,
               9,  13, 2,  4,  5,  6,  7,  8,  10, 14, 3,  5,  6,  7,  8,  9,  11, 15, 0,  4,  6,
               7,  8,  9,  10, 12, 16, 1,  5,  7,  8,  9,  10, 11, 13, 17, 2,  6,  8,  9,  10, 11,
               12, 14, 18, 3,  7,  9,  10, 11, 12, 13, 15, 4,  8,  10, 11, 12, 13, 14, 16, 5,  9,
               11, 12, 13, 14, 15, 17, 6,  10, 12, 13, 14, 15, 16, 18, 7,  11, 13, 14, 15, 16, 17,
               0,  8,  12, 14, 15, 16, 17, 18, 1,  9,  13, 15, 16, 17, 18, 2,  10, 14, 16, 17, 18};
        csr_val = {3.00, 1.00,  1.00, 1.00,  1.00,  1.00, 1.00, 5.00,  1.00, 1.00, 1.00,  1.00,
                   1.00, 1.00,  1.00, 7.00,  1.00,  1.00, 1.00, 1.00,  1.00, 1.00, 1.00,  11.00,
                   1.00, 1.00,  1.00, 1.00,  1.00,  1.00, 1.00, 13.00, 1.00, 1.00, 1.00,  1.00,
                   1.00, 1.00,  1.00, 17.00, 1.00,  1.00, 1.00, 1.00,  1.00, 1.00, 1.00,  19.00,
                   1.00, 1.00,  1.00, 1.00,  1.00,  1.00, 1.00, 23.00, 1.00, 1.00, 1.00,  1.00,
                   1.00, 1.00,  1.00, 1.00,  29.00, 1.00, 1.00, 1.00,  1.00, 1.00, 1.00,  1.00,
                   1.00, 31.00, 1.00, 1.00,  1.00,  1.00, 1.00, 1.00,  1.00, 1.00, 37.00, 1.00,
                   1.00, 1.00,  1.00, 1.00,  1.00,  1.00, 1.00, 41.00, 1.00, 1.00, 1.00,  1.00,
                   1.00, 1.00,  1.00, 43.00, 1.00,  1.00, 1.00, 1.00,  1.00, 1.00, 1.00,  47.00,
                   1.00, 1.00,  1.00, 1.00,  1.00,  1.00, 1.00, 53.00, 1.00, 1.00, 1.00,  1.00,
                   1.00, 1.00,  1.00, 59.00, 1.00,  1.00, 1.00, 1.00,  1.00, 1.00, 1.00,  61.00,
                   1.00, 1.00,  1.00, 1.00,  1.00,  1.00, 1.00, 67.00, 1.00, 1.00, 1.00,  1.00,
                   1.00, 1.00,  71.00};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
        break;
    case sample_gmres_mat_03:
        // this special matrix data tests GMRES where
        // HH = 0.0, which means residual vector is already being orthogonal
        // to the previous Krylov subspace vectors, in the very first iteration and thus
        // the initial x0 is the best solution
        //"bcsstm05.mtx"
        n   = 153;
        m   = 153;
        nnz = 153;
        csr_row_ptr
            = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
               16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
               32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
               48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
               64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
               80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
               96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
               112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
               128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
               144, 145, 146, 147, 148, 149, 150, 151, 152, 153};
        csr_col_ind
            = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
               17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
               34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
               51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,
               68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
               85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101,
               102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
               119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
               136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152};
        csr_val
            = {0.16, 0.16, 0.16, 0.08, 0.08, 0.08, 0.15, 0.15, 0.15, 0.17, 0.17, 0.17, 0.08, 0.08,
               0.08, 0.15, 0.15, 0.15, 0.17, 0.17, 0.17, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,
               0.14, 0.14, 0.14, 0.14, 0.14, 0.09, 0.09, 0.09, 0.20, 0.20, 0.20, 0.22, 0.22, 0.22,
               0.09, 0.09, 0.09, 0.20, 0.20, 0.20, 0.22, 0.22, 0.22, 0.16, 0.16, 0.16, 0.16, 0.16,
               0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.07, 0.07, 0.07, 0.28, 0.28, 0.28, 0.30,
               0.30, 0.30, 0.07, 0.07, 0.07, 0.28, 0.28, 0.28, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
               0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.31, 0.31, 0.31, 0.31, 0.31,
               0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34,
               0.34, 0.34, 0.34, 0.34, 0.34, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40,
               0.44, 0.44, 0.44, 0.58, 0.58, 0.58, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63,
               0.63, 0.93, 0.93, 0.93, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
        break;
    case invalid_mat:
        // matrix from the CG sample examples
        // symmetric, lower triangle filled
        n = m       = 8;
        nnz         = 18;
        csr_row_ptr = {0, 1, 2, 5, 6, 8, 11, 15, 17};
        csr_col_ind = {0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
        csr_val     = {19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        break;

    default:
        if(verbose)
            std::cout << "Non recognized matrix id: " << mid << std::endl;
        return aoclsparse_status_invalid_value;
    }

    ret = create_aoclsparse_matrix<T>(A, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val);
    if(ret != aoclsparse_status_success && verbose)
        std::cout << "Unexpected error in matrix creation" << std::endl;
    return ret;
}

template <typename T>
aoclsparse_status create_linear_system(linear_system_id                id,
                                       std::string                    &title,
                                       aoclsparse_operation           &trans,
                                       aoclsparse_matrix              &A,
                                       aoclsparse_mat_descr           &descr,
                                       T                              &alpha,
                                       std::vector<T>                 &b,
                                       std::vector<T>                 &x,
                                       std::vector<T>                 &xref,
                                       T                              &xtol,
                                       std::vector<aoclsparse_int>    &icrowa,
                                       std::vector<aoclsparse_int>    &icola,
                                       std::vector<T>                 &aval,
                                       std::array<aoclsparse_int, 10> &iparm,
                                       std::array<T, 10>              &dparm,
                                       aoclsparse_status              &exp_status)
{
    aoclsparse_status     status = aoclsparse_status_success;
    aoclsparse_int        n, nnz;
    aoclsparse_diag_type  diag;
    aoclsparse_fill_mode  fill_mode;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    alpha                      = (T)1.0;
    xtol       = (T)0.0; // By default not used, set only for ill conditioned problems
    exp_status = aoclsparse_status_success;
    std::fill(iparm.begin(), iparm.end(), 0);
    std::fill(dparm.begin(), dparm.end(), 0.0);
    switch(id)
    {
    case D7_Lx_aB:
    case D7_LL_Ix_aB:
    case D7_LTx_aB:
    case D7_LL_ITx_aB:
    case D7_Ux_aB:
    case D7_UU_Ix_aB:
    case D7_UTx_aB:
    case D7_UU_ITx_aB:
    case D7Lx_aB_hint:
        switch(id)
        {
        case D7Lx_aB_hint:
            title     = "Lx=b  SV hint";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_Lx_aB: // diag test set
            /*
                      Solve a   Dx = b
                      0   1  2   3   4   5    6      #cols  #row start #row end  #idiag  #iurow
            A  =  [  -2   0  0   0   0   0    0;      1      0          1         0       1
                      0  -4  0   0   0   0    0;      1      1          2         1       2
                      0   0  3   0   0   0    0;      1      2          3         2       3
                      0   0  0   5   0   0    0;      1      3          4         3       4
                      0   0  0   0  -7   0    0;      1      4          5         4       5
                      0   0  0   0   0   9    0;      1      5          6         5       6
                      0   0  0   0   0   0    4];     1      6          7         6       7
                                                                    nnz=7
            b = [1.0  -2.0  8.0  5.0  -1.0 11.0 3.0]'
            Dx = b ==> x* = [-1/2, 1/2, 8/3, 1, 1/7, 11/9, 3/4]
            */
            title     = "diag: Lx = alpha*b" + title;
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_Ix_aB:
            title     = "diag: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_LTx_aB:
            title     = "diag: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_ITx_aB:
            title     = "diag: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_Ux_aB:
            title     = "diag: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_Ix_aB:
            title     = "diag: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_UTx_aB:
            title     = "diag: U'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_ITx_aB:
            title     = "diag: [triu(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        alpha = (T)-9.845233;
        n     = 7;
        nnz   = 7;
        b.resize(n);
        b = {(T)1.0, -(T)2.0, (T)8.0, (T)5.0, (T)-1.0, (T)11.0, (T)3.0};
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0.0);
        xref.resize(n);
        icrowa.resize(n + 1);
        icrowa = {0, 1, 2, 3, 4, 5, 6, 7};
        icola.resize(nnz);
        icola = {0, 1, 2, 3, 4, 5, 6};
        aval.resize(nnz);
        aval = {(T)-2.0, (T)-4.0, (T)3.0, (T)5.0, (T)-7.0, (T)9.0, (T)4.0};
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        if(diag == aoclsparse_diag_type_unit)
            xref = {(T)1.0, (T)-2.0, (T)8.0, (T)5.0, (T)-1.0, (T)11.0, (T)3.0};
        else
            xref = {(T)-0.5, (T)0.5, (T)(8. / 3.), (T)1.0, (T)(1. / 7.), (T)(11. / 9.), (T)0.75};
        // xref *= alpha;
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](T &d) { return alpha * d; });
        break;

    case S7_Lx_aB: // small m test set
    case S7_LL_Ix_aB:
    case S7_LTx_aB:
    case S7_LL_ITx_aB:
    case S7_Ux_aB:
    case S7_UU_Ix_aB:
    case S7_UTx_aB:
    case S7_UU_ITx_aB:
        switch(id)
        {
        case S7_Lx_aB: // small m test set
            /*
             * Solve a   Ax = b  nnz=34
             *        0  1  2  3  4  5  6   #cols #row start #row end  #idiag  #iurow
             * A  = [-2  1  0  0  3  7 -1;   5     0          4         0       1
             *        2 -4  1  2  0  4  0;   5     5          9         6       7
             *        0  6 -2  9  1  0  9;   6    10         14        11      12
             *       -9  0  1 -2  1  1  1;   6    15         20        17      18
             *        0  8  2  1 -2  2  0;   5    21         25        24      25
             *        8  0  4  3  0  7  0;   4    26         29        29      30
             *        0  0  3  6  9  0  2];  4    30         33        33      34
             *
             * b = [1.0 -2.0 0.0 2.0 -1.0 0.0 3.0]'
             */
            title     = "small m: Lx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)-5.0e-01,
                    (T)2.5e-01,
                    (T)7.5e-01,
                    (T)1.625,
                    (T)3.0625,
                    (T)-5.535714285714286e-01,
                    (T)-1.828125e+01};
            break;
        case S7_LL_Ix_aB:
            title     = "small m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)1.0, (T)-4.0, (T)24.0, (T)-13.0, (T)-4.0, (T)-65.0, (T)45.0};
            break;
        case S7_LTx_aB:
            title     = "small m: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)2.03125, (T)34.59375, (T)1.30625e+01, (T)7.125, (T)7.25, (T)0.0, (T)1.5};
            break;
        case S7_LL_ITx_aB:
            title     = "small m: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)85.0, (T)12.0, (T)35.0, (T)12.0, (T)-28.0, (T)0.0, (T)3.0};
            break;
        case S7_Ux_aB:
            title     = "small m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)0.625, (T)2.25, (T)7.0, (T)0.0, (T)0.5, (T)0.0, (T)1.5};
            break;
        case S7_UU_Ix_aB:
            title     = "small m: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)-17.0, (T)24.0, (T)-26.0, (T)0.0, (T)-1.0, (T)0.0, (T)3.0};
            break;
        case S7_UTx_aB:
            title     = "small m: U'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)-5.0e-1,
                    (T)3.75e-1,
                    (T)1.875e-1,
                    (T)2.1875e-1,
                    (T)-4.6875e-2,
                    (T)2.678571428571428e-1,
                    (T)2.96875e-1};
            break;
        case S7_UU_ITx_aB:
            title     = "small m: [triu(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)1.0, (T)-3.0, (T)3.0, (T)-19.0, (T)12.0, (T)0.0, (T)-4.0};
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        alpha = (T)1.3334;
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](T &d) { return alpha * d; });
        n   = 7;
        nnz = 34;
        b.resize(n);
        b = {(T)1.0, (T)-2.0, (T)0.0, (T)2.0, (T)-1.0, (T)0.0, (T)3.0};
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0.0);
        icrowa.resize(n + 1);
        icrowa = {0, 5, 10, 15, 21, 26, 30, 34};
        icola.resize(nnz);
        icola = {0, 1, 4, 5, 6, 0, 1, 2, 3, 5, 1, 2, 3, 4, 6, 0, 2,
                 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 2, 3, 5, 2, 3, 4, 6};
        aval.resize(nnz);
        aval = {(T)-2.0, (T)1.0, (T)3.0,  (T)7.0, -(T)1.0, (T)2.0, (T)-4.0, (T)1.0, (T)2.0,
                (T)4.0,  (T)6.0, (T)-2.0, (T)9.0, (T)1.0,  (T)9.0, -(T)9.0, (T)1.0, (T)-2.0,
                (T)1.0,  (T)1.0, (T)1.0,  (T)8.0, (T)2.0,  (T)1.0, (T)-2.0, (T)2.0, (T)8.0,
                (T)4.0,  (T)3.0, (T)7.0,  (T)3.0, (T)6.0,  (T)9.0, (T)2.0};
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        break;

    case N25_Lx_aB: // large m test set
    case N25_LL_Ix_aB:
    case N25_LTx_aB:
    case N25_LL_ITx_aB:
    case N25_Ux_aB:
    case N25_UU_Ix_aB:
    case N25_UTx_aB:
    case N25_UU_ITx_aB:
        b.resize(25);
        alpha = (T)2.0;
        switch(id)
        {
        case N25_Lx_aB: // large m test set
            title     = "large m: Lx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIL(A) \ x = b
            b = {(T)3.1361606, (T)3.186481,  (T)3.2156856, (T)3.3757991, (T)3.4331077,
                 (T)3.4378236, (T)3.4479576, (T)3.7813114, (T)3.6439652, (T)3.8630548,
                 (T)3.6942673, (T)4.0985743, (T)3.5603244, (T)4.0383875, (T)4.1604922,
                 (T)3.9736567, (T)4.1527048, (T)3.9825762, (T)4.5399151, (T)4.2421458,
                 (T)4.997713,  (T)4.5352989, (T)4.7214001, (T)5.0616227, (T)4.8763871};
            break;
        case N25_LL_Ix_aB:
            title     = "large m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            // [TRIL(A,-1)+I] \ x = b
            b = {(T)1.5,       (T)1.5961924, (T)1.627927,  (T)1.7501192, (T)1.8427591,
                 (T)1.8526063, (T)1.8360447, (T)2.2656105, (T)2.0948749, (T)2.2219455,
                 (T)2.0955758, (T)2.5173541, (T)2.0557853, (T)2.4322676, (T)2.5737137,
                 (T)2.4639234, (T)2.5900785, (T)2.3540759, (T)2.9555803, (T)2.6225588,
                 (T)3.4008447, (T)2.9154333, (T)3.1143417, (T)3.4302724, (T)3.3440599};
            break;
        case N25_LTx_aB:
            title     = "large m: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIL(A)^T \ x = b
            b = {(T)4.7999156, (T)4.4620748, (T)4.7557297, (T)4.6435574, (T)4.861375,
                 (T)4.9847636, (T)4.712793,  (T)4.0188244, (T)4.0008485, (T)4.2367363,
                 (T)4.0991821, (T)3.9070598, (T)3.7680278, (T)3.940451,  (T)3.5216086,
                 (T)3.712464,  (T)3.7282857, (T)3.7685651, (T)3.6239308, (T)3.481001,
                 (T)3.2874493, (T)3.4135264, (T)3.2454106, (T)3.1509047, (T)3.0323272};
            break;
        case N25_LL_ITx_aB:
            title     = "large m: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            // [TRIL(A,-1)+I]^T \ x = b
            b = {(T)3.163755,  (T)2.8717862, (T)3.1679711, (T)3.0178774, (T)3.2710265,
                 (T)3.3995463, (T)3.1008801, (T)2.5031235, (T)2.4517582, (T)2.595627,
                 (T)2.5004906, (T)2.3258396, (T)2.2634888, (T)2.3343311, (T)1.9348302,
                 (T)2.2027308, (T)2.1656595, (T)2.1400648, (T)2.039596,  (T)1.861414,
                 (T)1.690581,  (T)1.7936608, (T)1.6383522, (T)1.5195545, (T)1.5};
            break;
        case N25_Ux_aB:
            title     = "large m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIU(A) \ x = b
            b = {(T)4.8346637, (T)4.951536,  (T)4.6879249, (T)4.9721482, (T)4.7600031,
                 (T)5.0265474, (T)4.2555835, (T)4.481163,  (T)4.1084789, (T)4.373474,
                 (T)4.3306394, (T)3.6651722, (T)4.0562577, (T)3.9586845, (T)4.1683247,
                 (T)3.8959224, (T)3.7310783, (T)3.8032438, (T)3.5453961, (T)3.442597,
                 (T)3.4147358, (T)3.3051947, (T)3.2346656, (T)3.1676077, (T)3.0323272};
            break;
        case N25_UU_Ix_aB:
            title     = "large m: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            // [TRIU(A,+1)+I] \ x = b
            b = {(T)3.1985031, (T)3.3612475, (T)3.1001663, (T)3.3464683, (T)3.1696546,
                 (T)3.44133,   (T)2.6436706, (T)2.9654621, (T)2.5593886, (T)2.7323646,
                 (T)2.7319479, (T)2.0839519, (T)2.5517187, (T)2.3525646, (T)2.5815463,
                 (T)2.3861891, (T)2.168452,  (T)2.1747436, (T)1.9610614, (T)1.82301,
                 (T)1.8178675, (T)1.6853291, (T)1.6276072, (T)1.5362574, (T)1.5};
            break;
        case N25_UTx_aB:
            title     = "large m: U'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIU(A)^T \ x = b
            b = {(T)3.1361606, (T)3.1561547, (T)3.2234122, (T)3.3076284, (T)3.3438639,
                 (T)3.4040321, (T)3.5559743, (T)3.6570364, (T)3.6873402, (T)3.7903524,
                 (T)3.9234929, (T)3.7647271, (T)3.769945,  (T)4.2276095, (T)3.9546192,
                 (T)4.6078492, (T)4.4359968, (T)4.6314379, (T)4.6641532, (T)5.0164253,
                 (T)4.8013899, (T)4.8843545, (T)4.7333244, (T)4.7972265, (T)4.7288635};
            break;
        case N25_UU_ITx_aB:
            title     = "large m: [triu(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            // [TRIU(A,+1)+I]^T \ x = b
            b = {(T)1.5,       (T)1.5658661, (T)1.6356537, (T)1.6819485, (T)1.7535154,
                 (T)1.8188148, (T)1.9440614, (T)2.1413355, (T)2.1382498, (T)2.149243,
                 (T)2.3248014, (T)2.1835068, (T)2.265406,  (T)2.6214895, (T)2.3678407,
                 (T)3.0981159, (T)2.8733705, (T)3.0029376, (T)3.0798184, (T)3.3968382,
                 (T)3.2045216, (T)3.2644888, (T)3.1262659, (T)3.1658763, (T)3.1965363};
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        // === START PART 1 Content autogenerated = make_trsvmat_b.m START ===
        n   = 25;
        nnz = 565;
        // === END PART 1 Content autogenerated = make_trsvmat_b.m END ===
        xref.resize(n);
        std::fill(xref.begin(), xref.end(), (T)3.0);
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0.0);
        icrowa.resize(n + 1);
        icola.resize(nnz);
        aval.resize(nnz);
        // === START PART 2 Content autogenerated = make_trsvmat_b.m START ===
        icrowa = {0,   24,  48,  70,  94,  118, 141, 163, 186, 208, 230, 254, 277,
                  296, 318, 342, 365, 386, 406, 429, 453, 476, 499, 521, 541, 565};
        icola = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                 18, 20, 21, 22, 23, 24, 0,  1,  2,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                 16, 17, 18, 19, 21, 22, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 0,  1,  2,  4,  5,  6,  7,  9,
                 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  6,
                 7,  8,  9,  10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 0,  1,  2,  3,  4,
                 5,  6,  7,  8,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 0,  2,  3,
                 4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 1,  2,
                 3,  4,  5,  6,  7,  8,  9,  10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,
                 1,  2,  3,  4,  5,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                 19, 20, 22, 23, 0,  2,  3,  5,  7,  8,  9,  12, 13, 15, 16, 17, 18, 19, 20, 21, 22,
                 23, 24, 1,  2,  3,  4,  5,  6,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 14, 15,
                 16, 17, 18, 19, 20, 21, 22, 23, 0,  2,  3,  4,  5,  6,  7,  10, 11, 12, 13, 15, 16,
                 17, 18, 19, 20, 21, 22, 23, 24, 0,  2,  3,  4,  5,  6,  7,  9,  12, 13, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                 15, 16, 17, 18, 19, 20, 21, 23, 24, 0,  1,  2,  3,  4,  5,  6,  8,  9,  10, 11, 12,
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,
                 9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 0,  2,  3,  4,  5,  6,  7,
                 8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 1,  3,  4,  5,  6,
                 7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  4,
                 5,  6,  7,  8,  9,  11, 12, 13, 15, 16, 17, 18, 19, 21, 23, 24, 0,  1,  2,  3,  4,
                 5,  6,  7,  8,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        aval  = {(T)2.09077374, (T)0.04391073, (T)0.04037834, (T)0.02612067, (T)0.04732957,
                (T)0.06528987, (T)0.03409377, (T)0.01417883, (T)0.02313281, (T)0.06966413,
                (T)0.05219750, (T)0.07436849, (T)0.06642035, (T)0.02096635, (T)0.08536398,
                (T)0.09015785, (T)0.03194168, (T)0.02949518, (T)0.04793079, (T)0.09548889,
                (T)0.03258194, (T)0.05993188, (T)0.05477309, (T)0.02661871, (T)0.06412829,
                (T)2.06019238, (T)0.05005746, (T)0.09517831, (T)0.01766596, (T)0.01743365,
                (T)0.07471561, (T)0.05321011, (T)0.04903184, (T)0.03846841, (T)0.03050719,
                (T)0.06036992, (T)0.01180341, (T)0.05769229, (T)0.02145119, (T)0.08678571,
                (T)0.02775075, (T)0.02499010, (T)0.09620239, (T)0.07705960, (T)0.08973039,
                (T)0.08498417, (T)0.07964857, (T)0.09609462, (T)0.05793688, (T)0.02734780,
                (T)2.05850569, (T)0.05767273, (T)0.01846547, (T)0.03907892, (T)0.08975531,
                (T)0.07780116, (T)0.07509255, (T)0.08296988, (T)0.08129907, (T)0.01645204,
                (T)0.09245670, (T)0.05893480, (T)0.07302568, (T)0.04358678, (T)0.04469032,
                (T)0.01785212, (T)0.05831480, (T)0.08339813, (T)0.03895424, (T)0.01697685,
                (T)0.05985614, (T)0.05025507, (T)0.05663490, (T)2.08378665, (T)0.04634200,
                (T)0.08670018, (T)0.04252953, (T)0.09163347, (T)0.08422832, (T)0.04215457,
                (T)0.02198648, (T)0.03188594, (T)0.07733250, (T)0.09235560, (T)0.06113743,
                (T)0.02803702, (T)0.06523582, (T)0.09887345, (T)0.08603304, (T)0.08129097,
                (T)0.05959284, (T)0.03273917, (T)0.04441168, (T)0.05647884, (T)0.07812804,
                (T)0.01411877, (T)0.07381025, (T)0.06244903, (T)2.06023235, (T)0.02465400,
                (T)0.03243171, (T)0.09138908, (T)0.06844232, (T)0.08755165, (T)0.09311017,
                (T)0.05925158, (T)0.01355292, (T)0.04593215, (T)0.02631154, (T)0.04425167,
                (T)0.08204778, (T)0.04307994, (T)0.06897342, (T)0.07885457, (T)0.04817361,
                (T)0.09909779, (T)0.04087351, (T)0.06512364, (T)0.06767424, (T)0.01066261,
                (T)0.09730985, (T)0.05942414, (T)2.05681158, (T)0.07319138, (T)0.02105568,
                (T)0.08172491, (T)0.02461240, (T)0.08609462, (T)0.07860446, (T)0.09776448,
                (T)0.04337716, (T)0.03509689, (T)0.08234870, (T)0.08752213, (T)0.08772559,
                (T)0.06570205, (T)0.08171715, (T)0.09923191, (T)0.08666116, (T)0.08146845,
                (T)0.08032089, (T)0.02881627, (T)0.02701135, (T)0.01128081, (T)0.08680126,
                (T)0.07012012, (T)2.07460859, (T)0.06633452, (T)0.03027692, (T)0.02249302,
                (T)0.05064716, (T)0.01106338, (T)0.03090903, (T)0.06230078, (T)0.06284175,
                (T)0.06389900, (T)0.03378182, (T)0.01704478, (T)0.07341236, (T)0.08219659,
                (T)0.06047072, (T)0.02855574, (T)0.06621948, (T)0.02638756, (T)0.09985003,
                (T)0.02579213, (T)0.07990810, (T)0.08825378, (T)0.09353681, (T)0.09667862,
                (T)2.01046725, (T)0.09258652, (T)0.03366170, (T)0.09935408, (T)0.05188379,
                (T)0.05025819, (T)0.09453112, (T)0.08619486, (T)0.02343119, (T)0.06375560,
                (T)0.05912321, (T)0.07732350, (T)0.02599300, (T)0.09900412, (T)0.05207746,
                (T)0.06779640, (T)0.03214791, (T)0.09326843, (T)0.04104519, (T)0.03272975,
                (T)0.09588411, (T)0.08077796, (T)0.02072990, (T)2.03272689, (T)0.01567945,
                (T)0.06295875, (T)0.02236562, (T)0.05878420, (T)0.06035869, (T)0.04029226,
                (T)0.07097701, (T)0.04526654, (T)0.07471676, (T)0.03485489, (T)0.07382112,
                (T)0.09348647, (T)0.03006073, (T)0.02263655, (T)0.03863734, (T)0.07284162,
                (T)0.04073123, (T)0.06912000, (T)0.09856783, (T)0.03914256, (T)0.02984625,
                (T)0.09241015, (T)2.09407292, (T)0.09721636, (T)0.01704300, (T)0.02144765,
                (T)0.09614223, (T)0.08135449, (T)0.06566343, (T)0.06439262, (T)0.06485830,
                (T)0.08053572, (T)0.07320127, (T)0.06242242, (T)0.07668488, (T)0.02061405,
                (T)0.05227499, (T)0.06471903, (T)0.02878251, (T)0.08404055, (T)0.02715183,
                (T)0.06552423, (T)0.03407719, (T)0.01450639, (T)0.02597382, (T)2.06579435,
                (T)0.01505039, (T)0.05299549, (T)0.04055540, (T)0.02457671, (T)0.09994050,
                (T)0.03579789, (T)0.08634282, (T)0.04782615, (T)0.08928474, (T)0.06572785,
                (T)0.02939611, (T)0.09114714, (T)0.06608592, (T)0.07657146, (T)0.03432247,
                (T)0.05031438, (T)0.04599930, (T)0.06582127, (T)0.07785931, (T)0.08743179,
                (T)0.08422025, (T)0.05810171, (T)0.01650092, (T)0.06907023, (T)0.08859443,
                (T)2.05414682, (T)0.04638696, (T)0.03901442, (T)0.04011880, (T)0.08030885,
                (T)0.01991078, (T)0.01079420, (T)0.06244956, (T)0.03102600, (T)0.02601979,
                (T)0.01023678, (T)0.02303515, (T)0.06240540, (T)0.04337962, (T)0.02656144,
                (T)0.09580308, (T)0.06526411, (T)0.03846964, (T)0.03864026, (T)2.00302603,
                (T)0.07394240, (T)0.09671779, (T)0.06829638, (T)0.06696414, (T)0.05400376,
                (T)0.05167050, (T)0.02791243, (T)0.05112525, (T)0.09797018, (T)0.04914760,
                (T)0.06339536, (T)0.02245613, (T)0.08762153, (T)0.05376251, (T)0.09231551,
                (T)0.09586983, (T)0.09980409, (T)0.04270442, (T)0.05135901, (T)0.03838703,
                (T)0.03723164, (T)2.07074661, (T)0.06311469, (T)0.09566913, (T)0.08867069,
                (T)0.03418988, (T)0.08102404, (T)0.06174401, (T)0.06244787, (T)0.03298625,
                (T)0.01326038, (T)0.01686786, (T)0.01840157, (T)0.07552113, (T)0.01014024,
                (T)0.07753066, (T)0.09602185, (T)0.05143221, (T)0.03194798, (T)0.04948350,
                (T)0.03669790, (T)0.04593919, (T)0.05033461, (T)0.04084196, (T)0.06832395,
                (T)0.03328524, (T)0.04830874, (T)2.05785230, (T)0.05209454, (T)0.06547174,
                (T)0.09930253, (T)0.08180502, (T)0.04806767, (T)0.09880571, (T)0.08533096,
                (T)0.09440344, (T)0.09574924, (T)0.05572878, (T)0.01621505, (T)0.04467451,
                (T)0.05303066, (T)0.07997375, (T)0.07903903, (T)0.04705687, (T)0.05499467,
                (T)0.03815641, (T)0.08747533, (T)0.02270457, (T)0.01331255, (T)0.03627146,
                (T)0.01398194, (T)2.00648886, (T)0.06955278, (T)0.09085033, (T)0.09980214,
                (T)0.09032994, (T)0.02443237, (T)0.05803462, (T)0.09991658, (T)0.05787400,
                (T)0.02675780, (T)0.03672458, (T)0.03683969, (T)0.04098114, (T)0.06360458,
                (T)0.08399171, (T)0.04366732, (T)0.09898645, (T)0.07298343, (T)0.06818030,
                (T)0.08365233, (T)0.07034969, (T)2.04175084, (T)0.07813689, (T)0.07605426,
                (T)0.07559373, (T)0.07293096, (T)0.01176305, (T)0.05468230, (T)0.05705068,
                (T)0.01942281, (T)0.09834169, (T)0.09004424, (T)0.02683593, (T)0.02496351,
                (T)0.04043255, (T)0.01943640, (T)0.02023616, (T)0.03208487, (T)0.05890288,
                (T)0.01583810, (T)0.06163846, (T)0.08062913, (T)2.08566685, (T)0.05751474,
                (T)0.07280763, (T)0.07758564, (T)0.08910963, (T)0.03726282, (T)0.07522263,
                (T)0.04032595, (T)0.04052489, (T)0.08704129, (T)0.06155397, (T)0.09613535,
                (T)0.01891930, (T)0.06817169, (T)0.02830823, (T)0.06412720, (T)0.07913742,
                (T)0.04833279, (T)0.06416750, (T)0.08726795, (T)0.07058048, (T)0.08228388,
                (T)0.01339266, (T)0.01080381, (T)0.04963847, (T)2.05622318, (T)0.08404672,
                (T)0.03825738, (T)0.09928940, (T)0.03663767, (T)0.04914308, (T)0.03289106,
                (T)0.02544694, (T)0.02206353, (T)0.02544478, (T)0.01149985, (T)0.06236985,
                (T)0.03831232, (T)0.05577997, (T)0.08759868, (T)0.02755643, (T)0.01486695,
                (T)0.04765429, (T)0.07202891, (T)0.04789970, (T)0.01119167, (T)0.09350937,
                (T)0.03945597, (T)0.03280223, (T)2.07972469, (T)0.02702375, (T)0.01413034,
                (T)0.04274179, (T)0.04638098, (T)0.08506312, (T)0.09080870, (T)0.09499486,
                (T)0.04179676, (T)0.07881016, (T)0.05837755, (T)0.09923804, (T)0.07092092,
                (T)0.01111529, (T)0.08432953, (T)0.08376871, (T)0.06574820, (T)0.04492859,
                (T)0.04223124, (T)0.03956827, (T)0.05093601, (T)0.04387681, (T)0.09013035,
                (T)0.09609362, (T)0.07955621, (T)2.06457886, (T)0.05584714, (T)0.09394850,
                (T)0.06211601, (T)0.02193982, (T)0.02432012, (T)0.01141585, (T)0.09460019,
                (T)0.05765404, (T)0.08199949, (T)0.06098509, (T)0.01229933, (T)0.01006737,
                (T)0.05185637, (T)0.08294399, (T)0.09769179, (T)0.05505617, (T)0.03296707,
                (T)0.03839105, (T)0.01375664, (T)0.09555001, (T)0.06840983, (T)0.03171797,
                (T)2.07991043, (T)0.03151611, (T)0.06928449, (T)0.02275212, (T)0.09865699,
                (T)0.05611091, (T)0.09084405, (T)0.03271674, (T)0.06539162, (T)0.04406103,
                (T)0.05284084, (T)0.06977812, (T)0.09016333, (T)0.04432702, (T)0.05283902,
                (T)0.01388691, (T)0.07854750, (T)0.03217368, (T)0.01613700, (T)0.07977081,
                (T)0.06167762, (T)0.03463541, (T)0.06166919, (T)2.07137227, (T)0.03003964,
                (T)0.05503182, (T)0.03024827, (T)0.09156941, (T)0.01391510, (T)0.09025232,
                (T)0.09441320, (T)0.09724469, (T)0.08210779, (T)0.04851376, (T)0.04512955,
                (T)0.09452913, (T)0.05561656, (T)0.09276520, (T)0.09517233, (T)0.09930465,
                (T)0.08250400, (T)0.06024601, (T)0.06766611, (T)0.04565020, (T)2.08756682,
                (T)0.02417163, (T)0.07232965, (T)0.08508686, (T)0.06263633, (T)0.08628786,
                (T)0.02464998, (T)0.05341119, (T)0.04462073, (T)0.07855638, (T)0.02792770,
                (T)0.09382215, (T)0.01294497, (T)0.02412206, (T)0.02292053, (T)0.05929903,
                (T)0.04886780, (T)0.06971889, (T)0.05329406, (T)0.02240817, (T)0.03204271,
                (T)0.06070061, (T)0.08845448, (T)0.09223481, (T)0.01303630, (T)2.02155148};
        // === END PART 2 Content autogenerated = make_trsvmat_b.m END ===
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        break;

    case A_nullptr:
        title      = "Invalid matrix A (ptr NULL)";
        A          = nullptr;
        exp_status = aoclsparse_status_invalid_pointer;
        break;

    case D1_descr_nullptr:
        title = "eye(1) with null descriptor";
        n = nnz = 1;
        icrowa.resize(2);
        icola.resize(1);
        aval.resize(1);
        icrowa[0] = 0;
        icrowa[1] = 1;
        icola[0]  = 0;
        aval[0]   = 1.0;
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr      = nullptr;
        exp_status = aoclsparse_status_invalid_pointer;
        break;

    case D1_neg_num_hint:
        title = "eye(1) with valid descriptor but negative expected_no_of_calls";
        n = nnz = 1;
        icrowa.resize(2);
        icola.resize(1);
        aval.resize(1);
        icrowa[0] = 0;
        icrowa[1] = 1;
        icola[0]  = 0;
        aval[0]   = 1.0;
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        iparm[0]   = -10;
        exp_status = aoclsparse_status_invalid_value;
        break;

    case D1_mattype_gen_hint:
        title = "eye(1) with matrix type set to general";
        n = nnz = 1;
        icrowa.resize(2);
        icola.resize(1);
        aval.resize(1);
        icrowa[0] = 0;
        icrowa[1] = 1;
        icola[0]  = 0;
        aval[0]   = 1.0;
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type = aoclsparse_matrix_type_general;
        exp_status  = aoclsparse_status_success;
        break;

    default:
        // no data with id found
        return aoclsparse_status_internal_error;
        break;
    }
    return status;
}
