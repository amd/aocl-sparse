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
#include "aoclsparse_reference.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

// Utilities to compare complex real scalars and vectors =============================================

#define EXPECT_COMPLEX_EQ_VEC(n, x, y)                                           \
    for(size_t i = 0; i < (size_t)n; i++)                                        \
    {                                                                            \
        EXPECT_EQ(std::real(x[i]), std::real(y[i]))                              \
            << " Real parts of " #x " and " #y " differ at index i = " << i      \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i]); \
        EXPECT_EQ(std::imag(x[i]), std::imag(y[i]))                              \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i]); \
    }

#define EXPECT_COMPLEX_FLOAT_EQ_VEC(n, x, y)                                     \
    for(size_t i = 0; i < (size_t)n; i++)                                        \
    {                                                                            \
        EXPECT_FLOAT_EQ(std::real(x[i]), std::real(y[i]))                        \
            << " Real parts of " #x " and " #y " differ at index i = " << i      \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i]); \
        EXPECT_FLOAT_EQ(std::imag(x[i]), std::imag(y[i]))                        \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i]); \
    }

#define EXPECT_COMPLEX_DOUBLE_EQ_VEC(n, x, y)                                    \
    for(size_t i = 0; i < (size_t)n; i++)                                        \
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
    for(size_t i = 0; i < (size_t)n; i++)                                                   \
    {                                                                                       \
        EXPECT_EQ((x)[i], (y)[i]) << " vectors " #x " and " #y " differ at index i = " << i \
                                  << " values are: " << (x)[i] << " and " << (y)[i]         \
                                  << " respectively.";                                      \
    }

#define EXPECT_FLOAT_EQ_VEC(n, x, y)                                                               \
    for(size_t i = 0; i < (size_t)n; i++)                                                          \
    {                                                                                              \
        EXPECT_FLOAT_EQ((x)[i], (y)[i]) << " vectors " #x " and " #y " differ at index i = " << i  \
                                        << " by abs err: " << abs((x)[i] - (y)[i])                 \
                                        << " rel err: " << abs(((x)[i] - (y)[i]) / (x)[i]) << "."; \
    }

#define EXPECT_DOUBLE_EQ_VEC(n, x, y)                                  \
    for(size_t i = 0; i < (size_t)n; i++)                              \
    {                                                                  \
        EXPECT_DOUBLE_EQ((x)[i], (y)[i])                               \
            << " vectors " #x " and " #y " differ at index i = " << i  \
            << " by abs err: " << abs((x)[i] - (y)[i])                 \
            << " rel err: " << abs(((x)[i] - (y)[i]) / (x)[i]) << "."; \
    }

#define EXPECT_EQ_VEC_ERR(n, x, y)                                                                 \
    for(size_t i = 0; i < (size_t)n; i++)                                                          \
    {                                                                                              \
        EXPECT_EQ((x)[i], ((y)[i])) << " vectors " #x " and " #y " differ at index i = " << i      \
                                    << " by abs err: " << abs((x)[i] - ((y)[i]))                   \
                                    << " rel err: " << abs(((x)[i] - *((y) + i)) / (x)[i]) << "."; \
    }

#define EXPECT_ARR_NEAR(n, x, y, abs_error)    \
    for(size_t j = 0; j < (size_t)(n); j++)    \
    EXPECT_NEAR(((x)[j]), ((y)[j]), abs_error) \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#define EXPECT_COMPLEX_ARR_NEAR(n, x, y, abs_error)                              \
    for(size_t i = 0; i < (size_t)n; i++)                                        \
    {                                                                            \
        EXPECT_NEAR(std::real(x[i]), std::real(y[i]), abs_error)                 \
            << " Real parts of " #x " and " #y " differ at index i = " << i      \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i])  \
            << " by abs err: " << abs(std::real(x[i]) - std::real(y[i]));        \
        EXPECT_NEAR(std::imag(x[i]), std::imag(y[i]), abs_error)                 \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i])  \
            << " by abs err: " << abs(std::real(x[i]) - std::real(y[i]));        \
    }

#define EXPECT_MAT_NEAR(m, n, ld, x, y, abs_error)                                      \
    for(size_t c = 0; c < (size_t)m; c++)                                               \
    {                                                                                   \
        for(size_t j = 0; j < (size_t)n; j++)                                           \
        {                                                                               \
            size_t offset = c * ld + j;                                                 \
            EXPECT_NEAR(((x)[offset]), ((y)[offset]), abs_error)                        \
                << "Vectors " #x " and " #y " different at index j =" << offset << "."; \
        }                                                                               \
    }
#define EXPECT_COMPLEX_MAT_NEAR(m, n, ld, x, y, abs_error)                                    \
    for(size_t c = 0; c < (size_t)m; c++)                                                     \
    {                                                                                         \
        for(size_t j = 0; j < (size_t)n; j++)                                                 \
        {                                                                                     \
            size_t offset = c * ld + j;                                                       \
            EXPECT_NEAR(std::real(x[offset]), std::real(y[offset]), abs_error)                \
                << " Real parts of " #x " and " #y " differ at index j = " << offset          \
                << " values are: " << std::real(x[offset]) << " and " << std::real(y[offset]) \
                << " by abs err: " << abs(std::real(x[offset]) - std::real(y[offset]));       \
            EXPECT_NEAR(std::imag(x[offset]), std::imag(y[offset]), abs_error)                \
                << " Imaginary parts of " #x " and " #y " differ at index j = " << (offset)   \
                << " values are: " << std::imag(x[offset]) << " and " << std::imag(y[offset]) \
                << " by abs err: " << abs(std::real(x[offset]) - std::real(y[offset]));       \
        }                                                                                     \
    }

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
                                           const aoclsparse_mat_descr   descr,
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

// DB for linear system equations: returns matrix, rhs, expected sol, tolerance,
// Use create_linear_system(...)
// Warning: Don't change the order of these enums, some test rely on it.
// Append towards the end any new problem id.
enum linear_system_id
{
    /* Used by TRSV */
    // diagonal matrix 7x7
    D7_Lx_aB,
    D7_LL_Ix_aB,
    D7_LTx_aB,
    D7_LL_ITx_aB,
    D7_LHx_aB,
    D7_LL_IHx_aB,
    D7_Ux_aB,
    D7_UU_Ix_aB,
    D7_UTx_aB,
    D7_UU_ITx_aB,
    D7_UHx_aB,
    D7_UU_IHx_aB,
    // small matrix 7x7
    S7_Lx_aB,
    S7_LL_Ix_aB,
    S7_LTx_aB,
    S7_LL_ITx_aB,
    S7_LHx_aB,
    S7_LL_IHx_aB,
    S7_Ux_aB,
    S7_UU_Ix_aB,
    S7_UTx_aB,
    S7_UU_ITx_aB,
    S7_UHx_aB,
    S7_UU_IHx_aB,
    // matrix 25x25
    N25_Lx_aB,
    N25_LL_Ix_aB,
    N25_LTx_aB,
    N25_LL_ITx_aB,
    N25_LHx_aB,
    N25_LL_IHx_aB,
    N25_Ux_aB,
    N25_UU_Ix_aB,
    N25_UTx_aB,
    N25_UU_ITx_aB,
    N25_UHx_aB,
    N25_UU_IHx_aB,
    // matrix used for hinting
    D7Lx_aB_hint,
    A_nullptr,
    D1_descr_nullptr,
    D1_neg_num_hint,
    D1_mattype_gen_hint,
    // matrices used for TRSM
    SM_S7_XB910,
    SM_S7_XB1716,
    SM_S1X1_XB1X1,
    SM_S1X1_XB1X3
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
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
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
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        break;
    case invalid_mat:
        // matrix from the CG sample examples
        // symmetric, lower triangle filled
        // making matrix invalid after memory is allocated for it
        n = m       = 8;
        nnz         = 18;
        csr_row_ptr = {0, 1, 2, 5, 6, 8, 11, 15, 18};
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

    ret = create_aoclsparse_matrix<T>(A, descr, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val);
    if(ret != aoclsparse_status_success && verbose)
        std::cout << "Unexpected error in matrix creation" << std::endl;

    if(mid == invalid_mat)
    {
        csr_row_ptr[m] = 17;
    }
    return ret;
}

template <typename T>
inline T bc(T v)
{
    if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
        return T(std::real(v), std::real(v) / 10);
    else
        return v;
};

template <typename T>
inline T bcd(T v, T w)
{
    if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
        return T(std::real(v), std::real(w));
    else
        return v;
};

template <typename T>
aoclsparse_status create_linear_system(linear_system_id                id,
                                       std::string                    &title,
                                       aoclsparse_operation           &trans,
                                       aoclsparse_matrix              &A,
                                       aoclsparse_mat_descr           &descr,
                                       aoclsparse_index_base           base,
                                       T                              &alpha,
                                       std::vector<T>                 &b,
                                       std::vector<T>                 &x,
                                       std::vector<T>                 &xref,
                                       T                              &xtol,
                                       std::vector<aoclsparse_int>    &icrowa,
                                       std::vector<aoclsparse_int>    &icola,
                                       std::vector<T>                 &aval,
                                       std::array<aoclsparse_int, 10> &iparm, //in-out parameter
                                       std::array<T, 10>              &dparm,
                                       aoclsparse_status              &exp_status)
{
    aoclsparse_status    status = aoclsparse_status_success;
    aoclsparse_int       n, nnz, big_m, big_n, k;
    aoclsparse_diag_type diag;
    aoclsparse_fill_mode fill_mode;
    aoclsparse_order     order;
    T                    beta;
    std::vector<T>       wcolxref, wcolb;
    const bool           cplx
        = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

    alpha      = (T)1;
    xtol       = (T)0; // By default not used, set only for ill conditioned problems
    exp_status = aoclsparse_status_success;
    std::fill(dparm.begin(), dparm.end(), 0);
    switch(id)
    {
    case D7_Lx_aB:
    case D7_LL_Ix_aB:
    case D7_LTx_aB:
    case D7_LL_ITx_aB:
    case D7_LHx_aB:
    case D7_LL_IHx_aB:
    case D7_Ux_aB:
    case D7_UU_Ix_aB:
    case D7_UTx_aB:
    case D7_UU_ITx_aB:
    case D7_UHx_aB:
    case D7_UU_IHx_aB:
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

            Note
            ====
            For complex types, we add 1/10 scaled entries to the nz real part, so e.g.
            b  = [1+.1i, -2-.2i, 8+.8i, ...], this applies to A as well.

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
            title     = "diag: L^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_ITx_aB:
            title     = "diag: [tril(L,-1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_LHx_aB:
            title     = "diag: L^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_IHx_aB:
            title     = "diag: [tril(L,-1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
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
            title     = "diag: U^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_ITx_aB:
            title     = "diag: [triu(U,1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_UHx_aB:
            title     = "diag: U^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_IHx_aB:
            title     = "diag: [triu(U,1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        alpha = bc((T)-9.845233);
        n     = 7;
        nnz   = 7;
        b.resize(n);
        b = {bc((T)1), bc(-(T)2), bc((T)8), bc((T)5), bc(-(T)1), bc((T)11), bc((T)3)};
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0);
        xref.resize(n);
        icrowa.resize(n + 1);
        icrowa = {0, 1, 2, 3, 4, 5, 6, 7};
        icola.resize(nnz);
        icola = {0, 1, 2, 3, 4, 5, 6};

        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)-2), bc((T)-4), bc((T)3), bc((T)5), bc((T)-7), bc((T)9), bc((T)4)};
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);

        if(diag == aoclsparse_diag_type_unit)
        {
            xref = {bc((T)1), bc(-(T)2), bc((T)8), bc((T)5), bc(-(T)1), bc((T)11), bc((T)3)};
        }
        else
        {
            xref = {-(T)0.5, (T)0.5, (T)(8. / 3.), (T)1.0, (T)(1. / 7.), (T)(11. / 9.), (T)0.75};
            if constexpr(cplx)
                if(trans == aoclsparse_operation_conjugate_transpose)
                    xref = {(T){-99. / 202, -10. / 101},
                            (T){99. / 202, 10. / 101},
                            (T){264. / 101, 160. / 303},
                            (T){99. / 101, 20. / 101},
                            (T){99. / 707, 20. / 707},
                            (T){121. / 101, 220. / 909},
                            (T){297. / 404, 15. / 101}};
        }
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](T &d) { return alpha * d; });
        break;

    case S7_Lx_aB: // small m test set
    case S7_LL_Ix_aB:
    case S7_LTx_aB:
    case S7_LL_ITx_aB:
    case S7_LHx_aB:
    case S7_LL_IHx_aB:
    case S7_Ux_aB:
    case S7_UU_Ix_aB:
    case S7_UTx_aB:
    case S7_UU_ITx_aB:
    case S7_UHx_aB:
    case S7_UU_IHx_aB:
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
            xref = {
                -(T)0.5, (T)0.25, (T)0.75, (T)1.625, (T)3.0625, -(T)0.553571428571, -(T)18.28125};
            break;
        case S7_LL_Ix_aB:
            title     = "small m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)1, (T)-4, (T)24, (T)-13, (T)-4, (T)-65, (T)45};
            else
                xref = {(T){1., 0.1},
                        (T){-3.98, -0.6},
                        (T){23.52, 5.988},
                        (T){-12.0112, -6.34},
                        (T){-4.1052, -1.25488},
                        (T){-65.4732, -12.33664},
                        (T){38.317008, 35.51532}};
            break;
        case S7_LTx_aB:
            title     = "small m: L^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)2.03125, (T)34.59375, (T)13.0625, (T)7.125, (T)7.25, (T)0, (T)1.5};
            break;
        case S7_LL_ITx_aB:
            title     = "small m: [tril(L,-1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)85, (T)12, (T)35, (T)12, (T)-28, (T)0, (T)3};
            else
                xref = {{70.980624, 65.90608},
                        {13.2184, -7.18438},
                        {34.5773, 8.737},
                        {11.36, 4.873},
                        {-27.73, -5.5},
                        {0, 0},
                        {3., 0.3}};
            break;

        case S7_LHx_aB:
            title     = "small m: L^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)2.03125, (T)34.59375, (T)13.0625, (T)7.125, (T)7.25, (T)0, (T)1.5};
            else
                xref = {{1.99102722, 0.40222772},
                        {33.9087252, 6.85024752},
                        {12.8038366, 2.58663366},
                        {6.98391089, 1.41089108},
                        {7.10643564, 1.43564356},
                        {0, 0},
                        {1.47029702, 0.29702970}};
            break;
        case S7_LL_IHx_aB:
            title     = "small m: [tril(L,-1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)85, (T)12, (T)35, (T)12, (T)-28, (T)0, (T)3};
            else
                xref = {{82.625776, -50.54544},
                        {11.534, +9.65962},
                        {35.6227, -1.717},
                        {12.1, -2.527},
                        {-28.27, -0.1},
                        {0, 0},
                        {3., 0.3}};
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
            if constexpr(!cplx)
                xref = {(T)-17, (T)24, (T)-26, (T)0, (T)-1, (T)0, (T)3};
            else
                xref = {{-16.1158, -7.29182},
                        {23.6782, 6.224},
                        {-26.1, -3.418},
                        {0.02, -0.2},
                        {-1.0, -0.1},
                        {0, 0},
                        {3.0, 0.3}};
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
                    (T)2.6785714e-1,
                    (T)2.96875e-1};
            break;
        case S7_UU_ITx_aB:
            title     = "small m: [triu(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)1, (T)-3, (T)3, (T)-19, (T)12, (T)0, (T)-4};
            else
                xref = {{1.0, 0.1},
                        {-2.99, -0.4},
                        {2.95, 0.699},
                        {-18.0209, -7.348},
                        {10.436, 7.45609},
                        {2.775318, -6.45329},
                        {-4.6448, 0.70409}};
            break;

        case S7_UHx_aB:
            title     = "small m: U^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)-5.0e-1,
                        (T)3.75e-1,
                        (T)1.875e-1,
                        (T)2.1875e-1,
                        (T)-4.6875e-2,
                        (T)2.6785714e-1,
                        (T)2.96875e-1};
            else
                xref = {
                    {-4.9009900e-1, -9.9009900e-2},
                    {3.6757425e-1, +7.4257425e-2},
                    {1.8378712e-1, +3.7128712e-2},
                    {2.1441831e-1, +4.3316831e-2},
                    {-4.5946782e-2, -9.2821782e-3},
                    {2.6255304e-1, +5.3041018e-2},
                    {2.9099628e-1, +5.8787128e-2},
                };
            break;
        case S7_UU_IHx_aB:
            title     = "small m: [triu(U,1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)1, (T)-3, (T)3, (T)-19, (T)12, (T)0, (T)-4};
            else
                xref = {{1., 0.1},
                        {-3.01, -0.2},
                        {3.03, -0.101},
                        {-19.1191, 3.634},
                        {11.7058, -5.24191},
                        {1.442482, 6.87507},
                        {-4.4134, -1.60991}};
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        alpha = bc((T)1.33340000);
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](T &d) { return alpha * d; });
        n   = 7;
        nnz = 34;
        b.resize(n);
        b = {bc((T)1), bc(-(T)2), bc((T)0), bc((T)2), -bc((T)1), bc((T)0), bc((T)3)};
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0);
        icrowa.resize(n + 1);
        icrowa = {0, 5, 10, 15, 21, 26, 30, 34};
        icola.resize(nnz);
        icola = {0, 1, 4, 5, 6, 0, 1, 2, 3, 5, 1, 2, 3, 4, 6, 0, 2,
                 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 2, 3, 5, 2, 3, 4, 6};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)-2), bc((T)1),  bc((T)3), bc((T)7),  bc(-(T)1), bc((T)2), bc((T)-4),
                bc((T)1),  bc((T)2),  bc((T)4), bc((T)6),  bc((T)-2), bc((T)9), bc((T)1),
                bc((T)9),  bc(-(T)9), bc((T)1), bc((T)-2), bc((T)1),  bc((T)1), bc((T)1),
                bc((T)8),  bc((T)2),  bc((T)1), bc((T)-2), bc((T)2),  bc((T)8), bc((T)4),
                bc((T)3),  bc((T)7),  bc((T)3), bc((T)6),  bc((T)9),  bc((T)2)};
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        break;

    case N25_Lx_aB: // large m test set
    case N25_LL_Ix_aB:
    case N25_LTx_aB:
    case N25_LHx_aB:
    case N25_LL_ITx_aB:
    case N25_LL_IHx_aB:
    case N25_Ux_aB:
    case N25_UU_Ix_aB:
    case N25_UTx_aB:
    case N25_UHx_aB:
    case N25_UU_ITx_aB:
    case N25_UU_IHx_aB:
        b.resize(25);
        alpha = (T)2.0;
        switch(id)
        {
        case N25_Lx_aB: // large m test set
            title     = "large m: Lx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIL(A) \ x = alpha b
            if constexpr(!cplx)
                b = {(T)3.1361606, (T)3.186481,  (T)3.2156856, (T)3.3757991, (T)3.4331077,
                     (T)3.4378236, (T)3.4479576, (T)3.7813114, (T)3.6439652, (T)3.8630548,
                     (T)3.6942673, (T)4.0985743, (T)3.5603244, (T)4.0383875, (T)4.1604922,
                     (T)3.9736567, (T)4.1527048, (T)3.9825762, (T)4.5399151, (T)4.2421458,
                     (T)4.997713,  (T)4.5352989, (T)4.7214001, (T)5.0616227, (T)4.8763871};
            else
                b = {{3.104799, 0.62723212},  {3.1546162, 0.6372962},  {3.1835287, 0.64313711},
                     {3.3420411, 0.67515983}, {3.3987766, 0.68662153}, {3.4034454, 0.68756473},
                     {3.413478, 0.68959152},  {3.7434983, 0.75626228}, {3.6075256, 0.72879304},
                     {3.8244243, 0.77261097}, {3.6573247, 0.73885347}, {4.0575886, 0.81971486},
                     {3.5247211, 0.71206487}, {3.9980036, 0.80767749}, {4.1188873, 0.83209844},
                     {3.9339201, 0.79473133}, {4.1111777, 0.83054096}, {3.9427504, 0.79651523},
                     {4.4945159, 0.90798302}, {4.1997243, 0.84842916}, {4.9477359, 0.9995426},
                     {4.4899459, 0.90705979}, {4.6741861, 0.94428002}, {5.0110064, 1.0123245},
                     {4.8276232, 0.97527742}};
            break;
        case N25_LL_Ix_aB:
            title     = "large m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            // [TRIL(A,-1)+I] \ x = alpha b
            if constexpr(!cplx)
                b = {(T)1.5,       (T)1.5961924, (T)1.627927,  (T)1.7501192, (T)1.8427591,
                     (T)1.8526063, (T)1.8360447, (T)2.2656105, (T)2.0948749, (T)2.2219455,
                     (T)2.0955758, (T)2.5173541, (T)2.0557853, (T)2.4322676, (T)2.5737137,
                     (T)2.4639234, (T)2.5900785, (T)2.3540759, (T)2.9555803, (T)2.6225588,
                     (T)3.4008447, (T)2.9154333, (T)3.1143417, (T)3.4302724, (T)3.3440599};
            else
                b = {{1.5, 0.15},
                     {1.5952305, 0.16923849},
                     {1.6266477, 0.1755854},
                     {1.747618, 0.20002383},
                     {1.8393315, 0.21855183},
                     {1.8490802, 0.22052125},
                     {1.8326843, 0.21720894},
                     {2.2579544, 0.30312211},
                     {2.0889261, 0.26897498},
                     {2.214726, 0.29438909},
                     {2.0896201, 0.26911516},
                     {2.5071805, 0.35347082},
                     {2.0502275, 0.26115707},
                     {2.4229449, 0.33645351},
                     {2.5629766, 0.36474275},
                     {2.4542841, 0.34278467},
                     {2.5791777, 0.36801571},
                     {2.3455351, 0.32081518},
                     {2.9410245, 0.44111606},
                     {2.6113332, 0.37451175},
                     {3.3818363, 0.53016895},
                     {2.901279, 0.43308666},
                     {3.0981983, 0.47286834},
                     {3.4109697, 0.53605448},
                     {3.3256193, 0.51881197}};
            break;
        case N25_LTx_aB:
            title     = "large m: L^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIL(A)^T \ x = alpha b
            if constexpr(!cplx)
                b = {(T)4.7999156, (T)4.4620748, (T)4.7557297, (T)4.6435574, (T)4.861375,
                     (T)4.9847636, (T)4.712793,  (T)4.0188244, (T)4.0008485, (T)4.2367363,
                     (T)4.0991821, (T)3.9070598, (T)3.7680278, (T)3.940451,  (T)3.5216086,
                     (T)3.712464,  (T)3.7282857, (T)3.7685651, (T)3.6239308, (T)3.481001,
                     (T)3.2874493, (T)3.4135264, (T)3.2454106, (T)3.1509047, (T)3.0323272};
            else
                b = {{4.7519164, 0.95998312}, {4.417454, 0.89241496},  {4.7081724, 0.95114593},
                     {4.5971218, 0.92871148}, {4.8127613, 0.97227501}, {4.934916, 0.99695273},
                     {4.6656651, 0.9425586},  {3.9786362, 0.80376488}, {3.96084, 0.8001697},
                     {4.194369, 0.84734727},  {4.0581903, 0.81983642}, {3.8679892, 0.78141196},
                     {3.7303476, 0.75360557}, {3.9010465, 0.7880902},  {3.4863926, 0.70432173},
                     {3.6753394, 0.74249281}, {3.6910029, 0.74565715}, {3.7308794, 0.75371301},
                     {3.5876915, 0.72478616}, {3.446191, 0.6962002},   {3.2545748, 0.65748986},
                     {3.3793912, 0.68270529}, {3.2129565, 0.64908212}, {3.1193956, 0.63018094},
                     {3.0020039, 0.60646544}};
            break;
        case N25_LHx_aB:
            title     = "large m: L^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            if constexpr(!cplx) // TRIL(A)^T \ x = alpha b
                b = {(T)4.7999156, (T)4.4620748, (T)4.7557297, (T)4.6435574, (T)4.861375,
                     (T)4.9847636, (T)4.712793,  (T)4.0188244, (T)4.0008485, (T)4.2367363,
                     (T)4.0991821, (T)3.9070598, (T)3.7680278, (T)3.940451,  (T)3.5216086,
                     (T)3.712464,  (T)3.7282857, (T)3.7685651, (T)3.6239308, (T)3.481001,
                     (T)3.2874493, (T)3.4135264, (T)3.2454106, (T)3.1509047, (T)3.0323272};
            else // TRIL(A)^H \ x = alpha b
                b = {(T)4.84791474, (T)4.50669554, (T)4.80328696, (T)4.68999295, (T)4.90998879,
                     (T)5.03461128, (T)4.75992095, (T)4.05901265, (T)4.04085700, (T)4.27910371,
                     (T)4.14017391, (T)3.94613041, (T)3.80570812, (T)3.97985550, (T)3.55682474,
                     (T)3.74958869, (T)3.76556859, (T)3.80625072, (T)3.66017008, (T)3.51581102,
                     (T)3.32032377, (T)3.44766171, (T)3.27786473, (T)3.18241373, (T)3.06265049};
            break;
        case N25_LL_ITx_aB:
            title     = "large m: [tril(L,-1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            // [TRIL(A,-1)+I]^T \ x = alpha b
            if constexpr(!cplx)
                b = {(T)3.163755,  (T)2.8717862, (T)3.1679711, (T)3.0178774, (T)3.2710265,
                     (T)3.3995463, (T)3.1008801, (T)2.5031235, (T)2.4517582, (T)2.595627,
                     (T)2.5004906, (T)2.3258396, (T)2.2634888, (T)2.3343311, (T)1.9348302,
                     (T)2.2027308, (T)2.1656595, (T)2.1400648, (T)2.039596,  (T)1.861414,
                     (T)1.690581,  (T)1.7936608, (T)1.6383522, (T)1.5195545, (T)1.5};
            else
                b = {{3.1471174, 0.48275099},
                     {2.8580684, 0.42435725},
                     {3.1512914, 0.48359423},
                     {3.0026986, 0.45357548},
                     {3.2533162, 0.5042053},
                     {3.3805508, 0.52990925},
                     {3.0848713, 0.47017603},
                     {2.4930923, 0.35062471},
                     {2.4422406, 0.34035164},
                     {2.5846707, 0.36912539},
                     {2.4904857, 0.35009811},
                     {2.3175812, 0.31516792},
                     {2.2558539, 0.30269776},
                     {2.3259878, 0.31686622},
                     {1.9304819, 0.23696604},
                     {2.1957034, 0.29054615},
                     {2.1590029, 0.28313189},
                     {2.1336641, 0.27801296},
                     {2.0342, 0.2579192},
                     {1.8577998, 0.2222828},
                     {1.6886752, 0.1881162},
                     {1.7907242, 0.20873216},
                     {1.6369687, 0.17767044},
                     {1.5193589, 0.15391089},
                     {1.5, 0.15}};
            break;
        case N25_LL_IHx_aB:
            title     = "large m: [tril(L,-1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            if constexpr(!cplx) // [TRIL(A,-1)+I]^T \ x = alpha b
                b = {(T)3.163755,  (T)2.8717862, (T)3.1679711, (T)3.0178774, (T)3.2710265,
                     (T)3.3995463, (T)3.1008801, (T)2.5031235, (T)2.4517582, (T)2.595627,
                     (T)2.5004906, (T)2.3258396, (T)2.2634888, (T)2.3343311, (T)1.9348302,
                     (T)2.2027308, (T)2.1656595, (T)2.1400648, (T)2.039596,  (T)1.861414,
                     (T)1.690581,  (T)1.7936608, (T)1.6383522, (T)1.5195545, (T)1.5};
            else // [TRIL(A,-1)+I]^H \ x = alpha b
                b = {{3.1803925, 0.15}, {2.8855041, 0.15}, {3.1846508, 0.15}, {3.0330562, 0.15},
                     {3.2887368, 0.15}, {3.4185417, 0.15}, {3.1168889, 0.15}, {2.5131548, 0.15},
                     {2.4612758, 0.15}, {2.6065832, 0.15}, {2.5104955, 0.15}, {2.334098, 0.15},
                     {2.2711237, 0.15}, {2.3426744, 0.15}, {1.9391785, 0.15}, {2.2097581, 0.15},
                     {2.1723161, 0.15}, {2.1464654, 0.15}, {2.044992, 0.15},  {1.8650281, 0.15},
                     {1.6924868, 0.15}, {1.7965974, 0.15}, {1.6397357, 0.15}, {1.51975, 0.15},
                     {1.5, 0.15}};
            break;
        case N25_Ux_aB:
            title     = "large m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIU(A) \ x = alpha b
            if constexpr(!cplx)
                b = {(T)4.8346637, (T)4.951536,  (T)4.6879249, (T)4.9721482, (T)4.7600031,
                     (T)5.0265474, (T)4.2555835, (T)4.481163,  (T)4.1084789, (T)4.373474,
                     (T)4.3306394, (T)3.6651722, (T)4.0562577, (T)3.9586845, (T)4.1683247,
                     (T)3.8959224, (T)3.7310783, (T)3.8032438, (T)3.5453961, (T)3.442597,
                     (T)3.4147358, (T)3.3051947, (T)3.2346656, (T)3.1676077, (T)3.0323272};
            else
                b = {{4.7863171, 0.96693274}, {4.9020207, 0.99030721}, {4.6410456, 0.93758497},
                     {4.9224268, 0.99442965}, {4.7124031, 0.95200062}, {4.9762819, 1.0053095},
                     {4.2130276, 0.85111669}, {4.4363514, 0.8962326},  {4.0673941, 0.82169578},
                     {4.3297393, 0.8746948},  {4.287333, 0.86612788},  {3.6285204, 0.73303443},
                     {4.0156952, 0.81125155}, {3.9190976, 0.79173689}, {4.1266415, 0.83366495},
                     {3.8569632, 0.77918449}, {3.6937675, 0.74621566}, {3.7652114, 0.76064877},
                     {3.5099422, 0.70907923}, {3.408171, 0.6885194},   {3.3805884, 0.68294715},
                     {3.2721428, 0.66103895}, {3.2023189, 0.64693312}, {3.1359316, 0.63352154},
                     {3.0020039, 0.60646544}};
            break;
        case N25_UU_Ix_aB:
            title     = "large m: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            // [TRIU(A,+1)+I] \ x = alpha b
            if constexpr(!cplx)
                b = {(T)3.1985031, (T)3.3612475, (T)3.1001663, (T)3.3464683, (T)3.1696546,
                     (T)3.44133,   (T)2.6436706, (T)2.9654621, (T)2.5593886, (T)2.7323646,
                     (T)2.7319479, (T)2.0839519, (T)2.5517187, (T)2.3525646, (T)2.5815463,
                     (T)2.3861891, (T)2.168452,  (T)2.1747436, (T)1.9610614, (T)1.82301,
                     (T)1.8178675, (T)1.6853291, (T)1.6276072, (T)1.5362574, (T)1.5};
            else
                b = {{3.1815181, 0.48970062},
                     {3.342635, 0.52224949},
                     {3.0841647, 0.47003326},
                     {3.3280036, 0.51929366},
                     {3.152958, 0.48393092},
                     {3.4219167, 0.538266},
                     {2.6322339, 0.37873412},
                     {2.9508075, 0.44309242},
                     {2.5487947, 0.36187771},
                     {2.720041, 0.39647293},
                     {2.7196284, 0.39638957},
                     {2.0781124, 0.26679039},
                     {2.5412015, 0.36034374},
                     {2.3440389, 0.32051291},
                     {2.5707308, 0.36630926},
                     {2.3773272, 0.32723783},
                     {2.1617675, 0.2836904},
                     {2.1679961, 0.28494871},
                     {1.9564508, 0.24221228},
                     {1.8197799, 0.21460199},
                     {1.8146888, 0.2135735},
                     {1.6834758, 0.18706582},
                     {1.6263311, 0.17552144},
                     {1.5358949, 0.15725149},
                     {1.5, 0.15}};
            break;
        case N25_UTx_aB:
            title     = "large m: U^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIU(A)^T \ x = alpha b
            if constexpr(!cplx)
                b = {(T)3.1361606, (T)3.1561547, (T)3.2234122, (T)3.3076284, (T)3.3438639,
                     (T)3.4040321, (T)3.5559743, (T)3.6570364, (T)3.6873402, (T)3.7903524,
                     (T)3.9234929, (T)3.7647271, (T)3.769945,  (T)4.2276095, (T)3.9546192,
                     (T)4.6078492, (T)4.4359968, (T)4.6314379, (T)4.6641532, (T)5.0164253,
                     (T)4.8013899, (T)4.8843545, (T)4.7333244, (T)4.7972265, (T)4.7288635};
            else
                b = {{3.104799, 0.62723212},  {3.1245931, 0.63123093}, {3.1911781, 0.64468245},
                     {3.2745522, 0.66152569}, {3.3104253, 0.66877278}, {3.3699918, 0.68080643},
                     {3.5204145, 0.71119485}, {3.620466, 0.73140728},  {3.6504668, 0.73746803},
                     {3.7524489, 0.75807048}, {3.884258, 0.78469858},  {3.7270798, 0.75294541},
                     {3.7322456, 0.753989},   {4.1853334, 0.84552189}, {3.915073, 0.79092383},
                     {4.5617707, 0.92156984}, {4.3916368, 0.88719936}, {4.5851235, 0.92628757},
                     {4.6175117, 0.93283064}, {4.966261, 1.0032851},   {4.753376, 0.96027797},
                     {4.8355109, 0.9768709},  {4.6859911, 0.94666487}, {4.7492542, 0.9594453},
                     {4.6815749, 0.9457727}};
            break;
        case N25_UHx_aB:
            title     = "large m: U^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            if constexpr(!cplx) // TRIU(A)^T \ x = alpha b
                b = {(T)3.1361606, (T)3.1561547, (T)3.2234122, (T)3.3076284, (T)3.3438639,
                     (T)3.4040321, (T)3.5559743, (T)3.6570364, (T)3.6873402, (T)3.7903524,
                     (T)3.9234929, (T)3.7647271, (T)3.769945,  (T)4.2276095, (T)3.9546192,
                     (T)4.6078492, (T)4.4359968, (T)4.6314379, (T)4.6641532, (T)5.0164253,
                     (T)4.8013899, (T)4.8843545, (T)4.7333244, (T)4.7972265, (T)4.7288635};
            else // TRIU(A)^H \ x = alpha b
                b = {(T)3.16752222, (T)3.18771621, (T)3.25564636, (T)3.34070473, (T)3.37730255,
                     (T)3.43807245, (T)3.59153401, (T)3.69360674, (T)3.72421357, (T)3.82825594,
                     (T)3.96272784, (T)3.80237433, (T)3.80764446, (T)4.26988556, (T)3.99416536,
                     (T)4.65392767, (T)4.48035677, (T)4.67775224, (T)4.71079475, (T)5.06658952,
                     (T)4.84940375, (T)4.93319802, (T)4.78065760, (T)4.84519875, (T)4.77615216};
            break;
        case N25_UU_ITx_aB:
            title     = "large m: [triu(U,1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            // [TRIU(A,+1)+I]^T \ x = alpha b
            if constexpr(!cplx)
                b = {(T)1.5,       (T)1.5658661, (T)1.6356537, (T)1.6819485, (T)1.7535154,
                     (T)1.8188148, (T)1.9440614, (T)2.1413355, (T)2.1382498, (T)2.149243,
                     (T)2.3248014, (T)2.1835068, (T)2.265406,  (T)2.6214895, (T)2.3678407,
                     (T)3.0981159, (T)2.8733705, (T)3.0029376, (T)3.0798184, (T)3.3968382,
                     (T)3.2045216, (T)3.2644888, (T)3.1262659, (T)3.1658763, (T)3.1965363};
            else
                b = {{1.5, 0.15},
                     {1.5652074, 0.16317322},
                     {1.6342972, 0.17713074},
                     {1.680129, 0.18638969},
                     {1.7509802, 0.20070308},
                     {1.8156266, 0.21376295},
                     {1.9396208, 0.23881228},
                     {2.1349221, 0.2782671},
                     {2.1318673, 0.27764997},
                     {2.1427506, 0.27984861},
                     {2.3165534, 0.31496028},
                     {2.1766718, 0.28670137},
                     {2.2577519, 0.30308119},
                     {2.6102747, 0.37429791},
                     {2.3591623, 0.32356814},
                     {3.0821347, 0.46962318},
                     {2.8596368, 0.42467411},
                     {2.9879082, 0.45058752},
                     {3.0640203, 0.46596369},
                     {3.3778698, 0.52936765},
                     {3.1874763, 0.49090431},
                     {3.2468439, 0.50289777},
                     {3.1100033, 0.47525319},
                     {3.1492175, 0.48317525},
                     {3.1795709, 0.48930726}};
            break;
        case N25_UU_IHx_aB:
            title     = "large m: [triu(U,1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            if constexpr(!cplx) // [TRIU(A,+1)+I]^T \ x = alpha b
                b = {(T)1.5,       (T)1.5658661, (T)1.6356537, (T)1.6819485, (T)1.7535154,
                     (T)1.8188148, (T)1.9440614, (T)2.1413355, (T)2.1382498, (T)2.149243,
                     (T)2.3248014, (T)2.1835068, (T)2.265406,  (T)2.6214895, (T)2.3678407,
                     (T)3.0981159, (T)2.8733705, (T)3.0029376, (T)3.0798184, (T)3.3968382,
                     (T)3.2045216, (T)3.2644888, (T)3.1262659, (T)3.1658763, (T)3.1965363};
            else // [TRIU(A,+1)+I]^H \ x = alpha b
                b = {{1.5, 0.15},       {1.5665248, 0.15}, {1.6370102, 0.15}, {1.683768, 0.15},
                     {1.7560505, 0.15}, {1.8220029, 0.15}, {1.948502, 0.15},  {2.1477489, 0.15},
                     {2.1446323, 0.15}, {2.1557355, 0.15}, {2.3330494, 0.15}, {2.1903419, 0.15},
                     {2.27306, 0.15},   {2.6327044, 0.15}, {2.3765191, 0.15}, {3.114097, 0.15},
                     {2.8871042, 0.15}, {3.017967, 0.15},  {3.0956166, 0.15}, {3.4158066, 0.15},
                     {3.2215668, 0.15}, {3.2821337, 0.15}, {3.1425286, 0.15}, {3.182535, 0.15},
                     {3.2135017, 0.15}};
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
        //update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        // add scaled complex part to xref, and aval;
        transform(aval.begin(), aval.end(), aval.begin(), [](T &v) { return bc(v); });
        transform(xref.begin(), xref.end(), xref.begin(), [](T &v) { return bc(v); });

        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
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
        //update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

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
        //update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

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
        //update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type = aoclsparse_matrix_type_general;
        aoclsparse_set_mat_index_base(descr, base);
        exp_status = aoclsparse_status_success;
        break;
    case SM_S7_XB910:
    case SM_S7_XB1716:
        /*
         * Small sparse 7x7 matrix A used to solve multiple RHS vectors
         * Targeted for testing TRSM triangle(A) X = alpha B with L or U
         * "X" and "B" are purposefully larger to test strides and layouts.
         * The actual sub-matrices X and B are a window of the larger XBIG
         * and BIGB matrices, and
         * OUT PARAM: iparm[0] = indicates where X (window starts in XBIG)
         * OUT PARAM: iparm[1] = ldx
         * OUT PARAM: iparm[2] = indicates where B (window starts in BIGB)
         * OUT PARAM: iparm[3] = ldb
         * IN PARAM:  iparm[4] = indicates whether X/B dense matrices are row major(0) or column major(1)
         * IN PARAM:  iparm[5] = indicates the no of columns in X/B dense matrices
         */
        fill_mode = aoclsparse_fill_mode_lower;
        trans     = aoclsparse_operation_none;
        diag      = aoclsparse_diag_type_non_unit;
        n         = 7;
        nnz       = 34;
        alpha     = (T)1.0;
        icrowa.resize(n + 1);
        icrowa = {0, 5, 10, 15, 21, 26, 30, 34};
        icola.resize(nnz);
        icola = {0, 1, 4, 5, 6, 0, 1, 2, 3, 5, 1, 2, 3, 4, 6, 0, 2,
                 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 2, 3, 5, 2, 3, 4, 6};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        aval.resize(nnz);
        aval = {bc((T)-2), bc((T)1),  bc((T)3), bc((T)7),  bc(-(T)1), bc((T)2), bc((T)-4),
                bc((T)1),  bc((T)2),  bc((T)4), bc((T)6),  bc((T)-2), bc((T)9), bc((T)1),
                bc((T)9),  bc(-(T)9), bc((T)1), bc((T)-2), bc((T)1),  bc((T)1), bc((T)1),
                bc((T)8),  bc((T)2),  bc((T)1), bc((T)-2), bc((T)2),  bc((T)8), bc((T)4),
                bc((T)3),  bc((T)7),  bc((T)3), bc((T)6),  bc((T)9),  bc((T)2)};
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        switch(id)
        {
        case SM_S7_XB910:
            // A Sparse 7x7, X,B dense 9x10
            title = "A[7,7] X(7,k) = alpha * B(7,k), XBIG,BIGB[9,10]";
            //allocate dense matrices of size = (n+2) * (n+2) to cover all k = (2, (m-2), m, (m+2)) scenarios
            xref.resize((n + 2) * (n + 3));
            // clang-format off
                xref.assign({bc((T)1.1),bc((T)1.2),bc((T)1.3),bc((T)1.4),bc((T)1.5),bc((T)1.6),bc((T)1.7),bc((T)1.8),bc((T)1.9),bc((T)1.11),
                             bc((T)2.1),bc((T)2.2),bc((T)2.3),bc((T)2.4),bc((T)2.5),bc((T)2.6),bc((T)2.7),bc((T)2.8),bc((T)2.9),bc((T)2.11),
                             bc((T)3.1),bc((T)3.2),bc((T)3.3),bc((T)3.4),bc((T)3.5),bc((T)3.6),bc((T)3.7),bc((T)3.8),bc((T)3.9),bc((T)3.11),
                             bc((T)4.1),bc((T)4.2),bc((T)4.3),bc((T)4.4),bc((T)4.5),bc((T)4.6),bc((T)4.7),bc((T)4.8),bc((T)4.9),bc((T)4.11),
                             bc((T)5.1),bc((T)5.2),bc((T)5.3),bc((T)5.4),bc((T)5.5),bc((T)5.6),bc((T)5.7),bc((T)5.8),bc((T)5.9),bc((T)5.11),
                             bc((T)6.1),bc((T)6.2),bc((T)6.3),bc((T)6.4),bc((T)6.5),bc((T)6.6),bc((T)6.7),bc((T)6.8),bc((T)6.9),bc((T)6.11),
                             bc((T)7.1),bc((T)7.2),bc((T)7.3),bc((T)7.4),bc((T)7.5),bc((T)7.6),bc((T)7.7),bc((T)7.8),bc((T)7.9),bc((T)7.11),
                             bc((T)8.1),bc((T)8.2),bc((T)8.3),bc((T)8.4),bc((T)8.5),bc((T)8.6),bc((T)8.7),bc((T)8.8),bc((T)8.9),bc((T)8.11),
                             bc((T)9.1),bc((T)9.2),bc((T)9.3),bc((T)9.4),bc((T)9.5),bc((T)9.6),bc((T)9.7),bc((T)9.8),bc((T)9.9),bc((T)9.11)});
            // clang-format on
            /*
             * starting offset to reach the window of X, and B
             */
            iparm[0] = 0; // starting_offset X
            iparm[1] = n + 3; //ldx = n+3 = 10
            iparm[2] = 0; // starting_offset B
            iparm[3] = n + 3; // ldb = n+3 as well
            x.resize((n + 2) * (n + 3));
            std::fill(x.begin(), x.end(), (T)0);
            b.resize((n + 2) * (n + 3));
            break;
        case SM_S7_XB1716:
            // A Sparse 7x7, X,B dense 17x16
            title = "A[7,7] X(7,k) = alpha * B(7,k), XBIG,BIGB[17,16]";
            big_m = 17;
            big_n = 16;
            //allocate dense matrices of size = 16 * 17 to access a window of 9x9 and cover all k (= 2, (m-2), m, (m+2)) scenarios
            xref.resize(big_m * big_n);
            // clang-format off
                xref.assign({-1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1, bc((T)1.1),bc((T)1.2),bc((T)1.3),bc((T)1.4),bc((T)1.5),bc((T)1.6),bc((T)1.7),bc((T)1.8),bc((T)1.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)2.1),bc((T)2.2),bc((T)2.3),bc((T)2.4),bc((T)2.5),bc((T)2.6),bc((T)2.7),bc((T)2.8),bc((T)2.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)3.1),bc((T)3.2),bc((T)3.3),bc((T)3.4),bc((T)3.5),bc((T)3.6),bc((T)3.7),bc((T)3.8),bc((T)3.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)4.1),bc((T)4.2),bc((T)4.3),bc((T)4.4),bc((T)4.5),bc((T)4.6),bc((T)4.7),bc((T)4.8),bc((T)4.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)5.1),bc((T)5.2),bc((T)5.3),bc((T)5.4),bc((T)5.5),bc((T)5.6),bc((T)5.7),bc((T)5.8),bc((T)5.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)6.1),bc((T)6.2),bc((T)6.3),bc((T)6.4),bc((T)6.5),bc((T)6.6),bc((T)6.7),bc((T)6.8),bc((T)6.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)7.1),bc((T)7.2),bc((T)7.3),bc((T)7.4),bc((T)7.5),bc((T)7.6),bc((T)7.7),bc((T)7.8),bc((T)7.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)8.1),bc((T)8.2),bc((T)8.3),bc((T)8.4),bc((T)8.5),bc((T)8.6),bc((T)8.7),bc((T)8.8),bc((T)8.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)9.1),bc((T)9.2),bc((T)9.3),bc((T)9.4),bc((T)9.5),bc((T)9.6),bc((T)9.7),bc((T)9.8),bc((T)9.9), -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1});
            // clang-format on
            /*
             * Starting offset to reach window of XRef = (4 * big_n + stride_left(4) = 68)
             */
            iparm[0] = (4 * big_n) + 4; // starting_offset_x
            iparm[1] = big_n; // leading dimension ldx
            iparm[2] = (3 * big_n) + 2; // starting_offset_b
            iparm[3] = big_n - 5;
            x.resize(big_m * big_n);
            std::fill(x.begin(), x.end(), (T)0);
            b.resize(big_m * big_n);
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        beta  = (T)0;
        order = (aoclsparse_order)iparm[4]; //matrix layout
        k     = iparm[5]; //no of columns in dense matrix, X/B
        wcolxref.resize(n);
        wcolb.resize(n);
        //Build B[]
        std::fill(b.begin(), b.end(), (T)-1);
        if(order == aoclsparse_order_column) //col major order
        {
            for(int col = 0; col < k; col++)
            {
                //Generate 'b' for known xref[]
                ref_csrmvtrg(alpha,
                             n,
                             n,
                             &aval[0],
                             &icola[0],
                             &icrowa[0],
                             fill_mode,
                             diag,
                             base,
                             &xref[iparm[0] + (col * iparm[1])],
                             beta,
                             &b[iparm[2] + (col * iparm[3])]);
            }
        }
        else if(order == aoclsparse_order_row) //row major order
        {
            for(int col = 0; col < k; col++)
            {
                aoclsparse_gthrs(n, &xref[iparm[0] + col], &wcolxref[0], iparm[1]);
                //Generate 'b' for known xref[]
                ref_csrmvtrg(alpha,
                             n,
                             n,
                             &aval[0],
                             &icola[0],
                             &icrowa[0],
                             fill_mode,
                             diag,
                             base,
                             &wcolxref[0],
                             beta,
                             &wcolb[0]);
                aoclsparse_sctrs<T>(n, &wcolb[0], iparm[3], &b[iparm[2] + col], 0);
            }
        }
        break;
    case SM_S1X1_XB1X1:
    case SM_S1X1_XB1X3:
        fill_mode = aoclsparse_fill_mode_lower;
        trans     = aoclsparse_operation_none;
        diag      = aoclsparse_diag_type_non_unit;
        n         = 1;
        nnz       = 1;
        alpha     = (T)1.0;
        icrowa.resize(n + 1);
        icrowa = {0, 1};
        icola.resize(nnz);
        icola = {0};
        // update row pointer and column index arrays to reflect values as per base-index
        icrowa[0] = icrowa[0] + base;
        icrowa[1] = icrowa[1] + base;
        icola[0]  = icola[0] + base;
        aval.resize(nnz);
        aval = {bc((T)2)};
        if(aoclsparse_create_csr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        switch(id)
        {
        case SM_S1X1_XB1X1:
            title = "very small m, A(1x1) . X(1x1) = alpha * B(1x1)";
            xref.resize(1);
            xref.assign({bc((T)1.1)});
            iparm[0] = 0; // starting_offset_x
            iparm[1] = 1; // leading dimension ldx
            iparm[2] = iparm[0]; // starting_offset_b
            iparm[3] = iparm[1]; // ldb
            x.resize(1);
            x[0] = (T)0;
            b.resize(1);
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                b[0] = bcd((T)2.178, (T)0.44);
            }
            else
            {
                b[0] = (T)2.2;
            }
            break;
        case SM_S1X1_XB1X3:
            title = "very small m with different k, A(1x1) . X(1x3) = alpha * B(1x3)";
            xref.resize(3);
            xref.assign({bc((T)1.1), bc((T)2.1), bc((T)3.1)});
            iparm[0] = 0; // starting_offset_x
            iparm[1] = 1; // leading dimension ldx
            iparm[2] = iparm[0]; // starting_offset_b
            iparm[3] = iparm[1]; // ldb
            x.resize(3);
            x[0] = (T)0;
            x[1] = (T)0;
            x[2] = (T)0;
            b.resize(3);
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                b[0] = bcd((T)2.178, (T)0.44);
                b[1] = bcd((T)4.158, (T)0.84);
                b[2] = bcd((T)6.138, (T)1.24);
            }
            else
            {
                b[0] = (T)2.2;
                b[1] = (T)4.2;
                b[2] = (T)6.2;
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    default:
        // no data with id found
        return aoclsparse_status_internal_error;
        break;
    }
    return status;
}
