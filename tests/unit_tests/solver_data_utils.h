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

#include <iostream>
#include <string>
#include <vector>

enum matrix_id
{
    /* Used mainly by hint_tests*/
    N5_full_sorted,
    N5_full_unsorted,
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

template <typename T>
aoclsparse_status create_aoclsparse_matrix(aoclsparse_matrix           &A,
                                           aoclsparse_int               m,
                                           aoclsparse_int               n,
                                           aoclsparse_int               nnz,
                                           std::vector<aoclsparse_int> &csr_row_ptr,
                                           std::vector<aoclsparse_int> &csr_col_ind,
                                           std::vector<T>              &csr_val);

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
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_status     ret  = aoclsparse_status_success;

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
void comp_exact_vec(
    std::string id, aoclsparse_int n, T *v1, T *v2, bool &pass, aoclsparse_int verbose)
{
    pass = true;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        if(v1[i] != v2[i])
        {
            pass = false;
            if(verbose)
                std::cout << id << " index " << i << ": v1[i] = " << v1[i] << ", v2[i] = " << v2[i]
                          << std::endl;
            break;
        }
    }
}

template <typename T>
void comp_tol_vec(
    std::string id, T tol, aoclsparse_int n, T *v1, T *v2, bool &pass, aoclsparse_int verbose)
{
    pass = true;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        printf("exp[%02d] = %0.4f, x[%d] = %0.4f, abs_diff = %0.4f, tol = %e\n",
               i,
               v1[i],
               i,
               v2[i],
               abs(v1[i] - v2[i]),
               tol);
        fflush(stdout);
        if(abs(v1[i] - v2[i]) > tol)
        {
            pass = false;
            //if(verbose)
            //    std::cout << id << " index " << i << ": v1[i] = " << v1[i] << ", v2[i] = " << v2[i]
            //              << std::endl;
            //break;
        }
    }
}

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
