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
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_descr.h"
#include "gtest/gtest.h"

#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>

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

#define EXPECT_ARR_NEAR(n, x, y, abs_error) \
    for(int j = 0; j < (n); j++)            \
    EXPECT_NEAR((x[j]), (y[j]), abs_error)  \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

// Define precision to which we expect the results to match ==================================

template <typename T> struct safeguard {};
// Add safeguarding scaling (may differ for each data type)
template <> struct safeguard<double> { static constexpr double value = 1.0; };
template <> struct safeguard<float> { static constexpr float value = 2.0f; };

template <typename T> constexpr
T expected_precision(T scale = (T)1.0) noexcept {
    const T macheps = std::numeric_limits<T>::epsilon();
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
    N15_LTx_aB, // 18
    N15_LL_ITx_aB, // 19
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
    xtol       = (T)0.0; // By default not used, set only foe ill conditioned problems
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
            // title = " (sv hint)";
            // __attribute__((fallthrough));
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
            // title     = "diag: Lx = alpha*b" + title;
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_Ix_aB:
            // title     = "diag: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_LTx_aB:
            // title     = "diag: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_ITx_aB:
            // title     = "diag: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_Ux_aB:
            // title     = "diag: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_Ix_aB:
            // title     = "diag: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_UTx_aB:
            // title     = "diag: U'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_ITx_aB:
            // title     = "diag: [triu(U,1) + I]'x = alpha*b";
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
            // title     = "small m: Lx = alpha*b";
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
            // title     = "small m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)1.0, (T)-4.0, (T)24.0, (T)-13.0, (T)-4.0, (T)-65.0, (T)45.0};
            break;
        case S7_LTx_aB:
            // title     = "small m: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)2.03125, (T)34.59375, (T)1.30625e+01, (T)7.125, (T)7.25, (T)0.0, (T)1.5};
            break;
        case S7_LL_ITx_aB:
            // title     = "small m: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)85.0, (T)12.0, (T)35.0, (T)12.0, (T)-28.0, (T)0.0, (T)3.0};
            break;
        case S7_Ux_aB:
            // title     = "small m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)0.625, (T)2.25, (T)7.0, (T)0.0, (T)0.5, (T)0.0, (T)1.5};
            break;
        case S7_UU_Ix_aB:
            // title     = "small m: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)-17.0, (T)24.0, (T)-26.0, (T)0.0, (T)-1.0, (T)0.0, (T)3.0};
            break;
        case S7_UTx_aB:
            // title     = "small m: U'x = alpha*b";
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
            // title     = "small m: [triu(U,1) + I]'x = alpha*b";
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
    case N15_LTx_aB:
    case N15_LL_ITx_aB:
    case N25_Ux_aB:
    case N25_UU_Ix_aB:
    case N25_UTx_aB:
    case N25_UU_ITx_aB:
        xref.resize(25);
        alpha = (T)2.0;
        switch(id)
        {
        case N25_Lx_aB: // large m test set
            // title     = "large m: Lx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIL(A) \ x = b
            xref = {(T)2.8694405, (T)2.8234737, (T)2.796182,  (T)2.6522445, (T)2.6045138,
                    (T)2.6003519, (T)2.6008353, (T)2.3046048, (T)2.43224,   (T)2.2662551,
                    (T)2.3948101, (T)2.076541,  (T)2.5212212, (T)2.1300172, (T)2.0418797,
                    (T)2.1804297, (T)2.0745273, (T)2.1919066, (T)1.7710128, (T)2.0428122,
                    (T)1.4405187, (T)1.8362006, (T)1.7075999, (T)1.443063,  (T)1.5821274};
            break;
        case N25_LL_Ix_aB:
            // title     = "large m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            // [TRIL(A,-1)+I] \ x = b
            xref = {(T)6,          (T)5.616,     (T)5.500368,  (T)5.045679,  (T)4.7335167,
                    (T)4.7174108,  (T)4.8435437, (T)3.4059263, (T)4.016664,  (T)3.7183309,
                    (T)4.0277302,  (T)2.8618217, (T)4.2833461, (T)3.125947,  (T)2.7968869,
                    (T)3.0935508,  (T)3.0170354, (T)3.4556542, (T)1.9581833, (T)3.0960567,
                    (T)0.84321483, (T)2.4842384, (T)2.0155569, (T)1.1194741, (T)1.4286963};
            break;
        case N15_LTx_aB:
            // title     = "large m: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIL(A)^T \ x = b
            xref = {(T)1.6955307, (T)1.848481,  (T)1.7244022, (T)1.763537,  (T)1.5746618,
                    (T)1.4425858, (T)1.6341808, (T)2.1487072, (T)2.1772862, (T)1.9986702,
                    (T)2.0781942, (T)2.228435,  (T)2.3520746, (T)2.1954149, (T)2.5435454,
                    (T)2.371718,  (T)2.355061,  (T)2.3217005, (T)2.4406206, (T)2.5581752,
                    (T)2.7305435, (T)2.6135037, (T)2.7653322, (T)2.8550883, (T)2.9673591};
            break;
        case N15_LL_ITx_aB:
            // title     = "large m: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            // [TRIL(A,-1)+I]^T \ x = b
            xref = {(T)2.1495739,  (T)2.2359359, (T)2.2242253, (T)2.253681,  (T)1.4580321,
                    (T)0.90891908, (T)1.5083596, (T)3.0297651, (T)3.2326871, (T)2.8348607,
                    (T)2.9481561,  (T)3.3924716, (T)3.7099236, (T)3.3144706, (T)4.588616,
                    (T)3.6591345,  (T)3.7313403, (T)3.7752722, (T)4.0853448, (T)4.6445086,
                    (T)5.287742,   (T)4.861812,  (T)5.448,     (T)5.922,     (T)6};
            break;
        case N25_Ux_aB:
            // title     = "large m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIU(A) \ x = b
            xref = {(T)1.6820205, (T)1.5182902, (T)1.7779084, (T)1.5514784, (T)1.6709507,
                    (T)1.3941567, (T)2.0238063, (T)1.8176034, (T)2.1365065, (T)1.9042648,
                    (T)1.9035674, (T)2.4770427, (T)2.1084998, (T)2.2295349, (T)1.9961781,
                    (T)2.2324927, (T)2.3696066, (T)2.2939014, (T)2.5115299, (T)2.5909073,
                    (T)2.616432,  (T)2.7148834, (T)2.7772147, (T)2.8394556, (T)2.9673591};
            break;
        case N25_UU_Ix_aB:
            // title     = "large m: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            // [TRIU(A,+1)+I] \ x = b
            xref = {(T)2.1403777,  (T)1.1981332, (T)2.3704862, (T)1.5934843, (T)1.8240657,
                    (T)0.59309717, (T)3.0075404, (T)2.015312,  (T)3.2820111, (T)2.5256976,
                    (T)2.3395546,  (T)4.5188939, (T)2.8162071, (T)3.6641606, (T)2.4795398,
                    (T)3.2170294,  (T)3.8526006, (T)3.6686584, (T)4.3818473, (T)4.7803937,
                    (T)4.8157353,  (T)5.2821178, (T)5.49432,   (T)5.856,     (T)6};
            break;
        case N25_UTx_aB:
            // title     = "large m: U'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIU(A)^T \ x = b
            xref = {(T)2.8694405, (T)2.8513323, (T)2.7890509, (T)2.7133004, (T)2.6831241,
                    (T)2.6308583, (T)2.4989068, (T)2.4098153, (T)2.3941426, (T)2.3030199,
                    (T)2.213122,  (T)2.3404009, (T)2.3446515, (T)1.9607205, (T)2.2188761,
                    (T)1.6943958, (T)1.8582681, (T)1.7539545, (T)1.7230434, (T)1.5025302,
                    (T)1.6208129, (T)1.5838626, (T)1.7347633, (T)1.6944515, (T)1.7187472};
            break;
        case N25_UU_ITx_aB:
            // title     = "large m: [triu(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            // [TRIU(A,+1)+I]^T \ x = b
            xref = {(T)6,         (T)5.736,     (T)5.4732,    (T)5.29908,   (T)5.0535487,
                    (T)4.8266117, (T)4.4104285, (T)3.7795686, (T)3.8814496, (T)3.7293536,
                    (T)3.3889675, (T)3.8192886, (T)3.6467563, (T)2.4584425, (T)3.5099871,
                    (T)1.5469311, (T)2.2524797, (T)2.1773924, (T)1.9585843, (T)1.3946507,
                    (T)1.5683126, (T)1.5643115, (T)2.1974827, (T)2.0979256, (T)1.9465109};
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        // === START PART 1 Content autogenerated = make_trsvmat.m START ===
        n   = 25;
        nnz = 565;
        // === END PART 1 Content autogenerated = make_trsvmat.m END ===
        b.resize(n);
        std::fill(b.begin(), b.end(), (T)3.0);
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0.0);
        icrowa.resize(n + 1);
        icola.resize(nnz);
        aval.resize(nnz);
        // === START PART 2 Content autogenerated = make_trsvmat.m START ===
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
        aval  = {(T)2.091, (T)0.044, (T)0.040, (T)0.026, (T)0.047, (T)0.065, (T)0.034, (T)0.014,
                 (T)0.023, (T)0.070, (T)0.052, (T)0.074, (T)0.066, (T)0.021, (T)0.085, (T)0.090,
                 (T)0.032, (T)0.029, (T)0.048, (T)0.095, (T)0.033, (T)0.060, (T)0.055, (T)0.027,
                 (T)0.064, (T)2.060, (T)0.050, (T)0.095, (T)0.018, (T)0.017, (T)0.075, (T)0.053,
                 (T)0.049, (T)0.038, (T)0.031, (T)0.060, (T)0.012, (T)0.058, (T)0.021, (T)0.087,
                 (T)0.028, (T)0.025, (T)0.096, (T)0.077, (T)0.090, (T)0.085, (T)0.080, (T)0.096,
                 (T)0.058, (T)0.027, (T)2.059, (T)0.058, (T)0.018, (T)0.039, (T)0.090, (T)0.078,
                 (T)0.075, (T)0.083, (T)0.081, (T)0.016, (T)0.092, (T)0.059, (T)0.073, (T)0.044,
                 (T)0.045, (T)0.018, (T)0.058, (T)0.083, (T)0.039, (T)0.017, (T)0.060, (T)0.050,
                 (T)0.057, (T)2.084, (T)0.046, (T)0.087, (T)0.043, (T)0.092, (T)0.084, (T)0.042,
                 (T)0.022, (T)0.032, (T)0.077, (T)0.092, (T)0.061, (T)0.028, (T)0.065, (T)0.099,
                 (T)0.086, (T)0.081, (T)0.060, (T)0.033, (T)0.044, (T)0.056, (T)0.078, (T)0.014,
                 (T)0.074, (T)0.062, (T)2.060, (T)0.025, (T)0.032, (T)0.091, (T)0.068, (T)0.088,
                 (T)0.093, (T)0.059, (T)0.014, (T)0.046, (T)0.026, (T)0.044, (T)0.082, (T)0.043,
                 (T)0.069, (T)0.079, (T)0.048, (T)0.099, (T)0.041, (T)0.065, (T)0.068, (T)0.011,
                 (T)0.097, (T)0.059, (T)2.057, (T)0.073, (T)0.021, (T)0.082, (T)0.025, (T)0.086,
                 (T)0.079, (T)0.098, (T)0.043, (T)0.035, (T)0.082, (T)0.088, (T)0.088, (T)0.066,
                 (T)0.082, (T)0.099, (T)0.087, (T)0.081, (T)0.080, (T)0.029, (T)0.027, (T)0.011,
                 (T)0.087, (T)0.070, (T)2.075, (T)0.066, (T)0.030, (T)0.022, (T)0.051, (T)0.011,
                 (T)0.031, (T)0.062, (T)0.063, (T)0.064, (T)0.034, (T)0.017, (T)0.073, (T)0.082,
                 (T)0.060, (T)0.029, (T)0.066, (T)0.026, (T)0.100, (T)0.026, (T)0.080, (T)0.088,
                 (T)0.094, (T)0.097, (T)2.010, (T)0.093, (T)0.034, (T)0.099, (T)0.052, (T)0.050,
                 (T)0.095, (T)0.086, (T)0.023, (T)0.064, (T)0.059, (T)0.077, (T)0.026, (T)0.099,
                 (T)0.052, (T)0.068, (T)0.032, (T)0.093, (T)0.041, (T)0.033, (T)0.096, (T)0.081,
                 (T)0.021, (T)2.033, (T)0.016, (T)0.063, (T)0.022, (T)0.059, (T)0.060, (T)0.040,
                 (T)0.071, (T)0.045, (T)0.075, (T)0.035, (T)0.074, (T)0.093, (T)0.030, (T)0.023,
                 (T)0.039, (T)0.073, (T)0.041, (T)0.069, (T)0.099, (T)0.039, (T)0.030, (T)0.092,
                 (T)2.094, (T)0.097, (T)0.017, (T)0.021, (T)0.096, (T)0.081, (T)0.066, (T)0.064,
                 (T)0.065, (T)0.081, (T)0.073, (T)0.062, (T)0.077, (T)0.021, (T)0.052, (T)0.065,
                 (T)0.029, (T)0.084, (T)0.027, (T)0.066, (T)0.034, (T)0.015, (T)0.026, (T)2.066,
                 (T)0.015, (T)0.053, (T)0.041, (T)0.025, (T)0.100, (T)0.036, (T)0.086, (T)0.048,
                 (T)0.089, (T)0.066, (T)0.029, (T)0.091, (T)0.066, (T)0.077, (T)0.034, (T)0.050,
                 (T)0.046, (T)0.066, (T)0.078, (T)0.087, (T)0.084, (T)0.058, (T)0.017, (T)0.069,
                 (T)0.089, (T)2.054, (T)0.046, (T)0.039, (T)0.040, (T)0.080, (T)0.020, (T)0.011,
                 (T)0.062, (T)0.031, (T)0.026, (T)0.010, (T)0.023, (T)0.062, (T)0.043, (T)0.027,
                 (T)0.096, (T)0.065, (T)0.038, (T)0.039, (T)2.003, (T)0.074, (T)0.097, (T)0.068,
                 (T)0.067, (T)0.054, (T)0.052, (T)0.028, (T)0.051, (T)0.098, (T)0.049, (T)0.063,
                 (T)0.022, (T)0.088, (T)0.054, (T)0.092, (T)0.096, (T)0.100, (T)0.043, (T)0.051,
                 (T)0.038, (T)0.037, (T)2.071, (T)0.063, (T)0.096, (T)0.089, (T)0.034, (T)0.081,
                 (T)0.062, (T)0.062, (T)0.033, (T)0.013, (T)0.017, (T)0.018, (T)0.076, (T)0.010,
                 (T)0.078, (T)0.096, (T)0.051, (T)0.032, (T)0.049, (T)0.037, (T)0.046, (T)0.050,
                 (T)0.041, (T)0.068, (T)0.033, (T)0.048, (T)2.058, (T)0.052, (T)0.065, (T)0.099,
                 (T)0.082, (T)0.048, (T)0.099, (T)0.085, (T)0.094, (T)0.096, (T)0.056, (T)0.016,
                 (T)0.045, (T)0.053, (T)0.080, (T)0.079, (T)0.047, (T)0.055, (T)0.038, (T)0.087,
                 (T)0.023, (T)0.013, (T)0.036, (T)0.014, (T)2.006, (T)0.070, (T)0.091, (T)0.100,
                 (T)0.090, (T)0.024, (T)0.058, (T)0.100, (T)0.058, (T)0.027, (T)0.037, (T)0.037,
                 (T)0.041, (T)0.064, (T)0.084, (T)0.044, (T)0.099, (T)0.073, (T)0.068, (T)0.084,
                 (T)0.070, (T)2.042, (T)0.078, (T)0.076, (T)0.076, (T)0.073, (T)0.012, (T)0.055,
                 (T)0.057, (T)0.019, (T)0.098, (T)0.090, (T)0.027, (T)0.025, (T)0.040, (T)0.019,
                 (T)0.020, (T)0.032, (T)0.059, (T)0.016, (T)0.062, (T)0.081, (T)2.086, (T)0.058,
                 (T)0.073, (T)0.078, (T)0.089, (T)0.037, (T)0.075, (T)0.040, (T)0.041, (T)0.087,
                 (T)0.062, (T)0.096, (T)0.019, (T)0.068, (T)0.028, (T)0.064, (T)0.079, (T)0.048,
                 (T)0.064, (T)0.087, (T)0.071, (T)0.082, (T)0.013, (T)0.011, (T)0.050, (T)2.056,
                 (T)0.084, (T)0.038, (T)0.099, (T)0.037, (T)0.049, (T)0.033, (T)0.025, (T)0.022,
                 (T)0.025, (T)0.011, (T)0.062, (T)0.038, (T)0.056, (T)0.088, (T)0.028, (T)0.015,
                 (T)0.048, (T)0.072, (T)0.048, (T)0.011, (T)0.094, (T)0.039, (T)0.033, (T)2.080,
                 (T)0.027, (T)0.014, (T)0.043, (T)0.046, (T)0.085, (T)0.091, (T)0.095, (T)0.042,
                 (T)0.079, (T)0.058, (T)0.099, (T)0.071, (T)0.011, (T)0.084, (T)0.084, (T)0.066,
                 (T)0.045, (T)0.042, (T)0.040, (T)0.051, (T)0.044, (T)0.090, (T)0.096, (T)0.080,
                 (T)2.065, (T)0.056, (T)0.094, (T)0.062, (T)0.022, (T)0.024, (T)0.011, (T)0.095,
                 (T)0.058, (T)0.082, (T)0.061, (T)0.012, (T)0.010, (T)0.052, (T)0.083, (T)0.098,
                 (T)0.055, (T)0.033, (T)0.038, (T)0.014, (T)0.096, (T)0.068, (T)0.032, (T)2.080,
                 (T)0.032, (T)0.069, (T)0.023, (T)0.099, (T)0.056, (T)0.091, (T)0.033, (T)0.065,
                 (T)0.044, (T)0.053, (T)0.070, (T)0.090, (T)0.044, (T)0.053, (T)0.014, (T)0.079,
                 (T)0.032, (T)0.016, (T)0.080, (T)0.062, (T)0.035, (T)0.062, (T)2.071, (T)0.030,
                 (T)0.055, (T)0.030, (T)0.092, (T)0.014, (T)0.090, (T)0.094, (T)0.097, (T)0.082,
                 (T)0.049, (T)0.045, (T)0.095, (T)0.056, (T)0.093, (T)0.095, (T)0.099, (T)0.083,
                 (T)0.060, (T)0.068, (T)0.046, (T)2.088, (T)0.024, (T)0.072, (T)0.085, (T)0.063,
                 (T)0.086, (T)0.025, (T)0.053, (T)0.045, (T)0.079, (T)0.028, (T)0.094, (T)0.013,
                 (T)0.024, (T)0.023, (T)0.059, (T)0.049, (T)0.070, (T)0.053, (T)0.022, (T)0.032,
                 (T)0.061, (T)0.088, (T)0.092, (T)0.013, (T)2.022};
        // === END PART 2 Content autogenerated = make_trsvmat.m END ===
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
        // title = "Invalid matrix A (ptr NULL)";
        A          = nullptr;
        exp_status = aoclsparse_status_invalid_pointer;
        break;

    case D1_descr_nullptr:
        // title = "eye(1) with null descriptor";
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
        // title = "eye(1) with valid descriptor but negative expected_no_of_calls";
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
        // title = "eye(1) with matrix type set to general";
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
