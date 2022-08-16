/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#include <iostream>
#include <vector>

#define VERBOSE 1
#define TOL_COMP 1.0e-10

typedef struct
{
    aoclsparse_int              nnz;
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<aoclsparse_int> idiag, iurow;
    std::vector<double>         aval;
} sol_opt_csr;

enum matrix_id
{
    N5_full_sorted,
    N5_full_unsorted,
    N5_1_hole,
    N5_empty_rows,
    N10_random,
    M5_rect_N7,
    M5_rect_N7_2holes,
    M7_rect_N5,
    M7_rect_N5_2holes,
};

aoclsparse_status hint_optimize(aoclsparse_int        mv_hints,
                                aoclsparse_int        trsv_hints,
                                aoclsparse_int        lu_hints,
                                aoclsparse_mat_descr& descr,
                                aoclsparse_operation  trans,
                                aoclsparse_matrix&    A)
{
    aoclsparse_status status;
    if(mv_hints > 0)
    {
        status = aoclsparse_set_mv_hint(A, trans, descr, mv_hints);
        if(status != aoclsparse_status_success)
            std::cout << "Something went wrong in hinting MV: " << status << std::endl;
    }
    if(trsv_hints > 0)
    {
        // Only works with symmetric matrices
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        status = aoclsparse_set_sv_hint(A, trans, descr, trsv_hints);
        if(status != aoclsparse_status_success)
            std::cout << "Something went wrong in hinting TRSV: " << status << std::endl;
    }
    if(lu_hints > 0)
    {
        status = aoclsparse_set_lu_smoother_hint(A, trans, descr, lu_hints);
        if(status != aoclsparse_status_success)
            std::cout << "Something went wrong in hinting LU: " << status << std::endl;
    }

    status = aoclsparse_optimize(A);
    if(status != aoclsparse_status_success)
        std::cout << "Something went wrong in Optimize: " << status << std::endl;

    return status;
}

aoclsparse_status create_matrix(matrix_id                    mid,
                                std::vector<aoclsparse_int>& icrow,
                                std::vector<aoclsparse_int>& icol,
                                std::vector<double>&         aval,
                                aoclsparse_matrix&           A)
{
    int                   n, m, nnz;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_status     ret  = aoclsparse_status_success;

    switch(mid)
    {
    case N5_full_sorted:
        // Small sorted matrix with full diagonal
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  4  0  0
        //  0  5  0  6  7
        //  0  0  0  0  8
        n = m = 5;
        nnz   = 8;
        icrow = {0, 2, 3, 4, 7, 8};
        icol  = {0, 3, 1, 2, 1, 3, 4, 4};
        aval  = {1, 2, 3, 4, 5, 6, 7, 8};
        break;

    case N5_full_unsorted:
        // same as N5 full sorted with rows 0 and 3 shuffled
        n = m = 5;
        nnz   = 8;
        icrow = {0, 2, 3, 4, 7, 8};
        icol  = {3, 0, 1, 2, 3, 1, 4, 4};
        aval  = {2, 1, 3, 4, 6, 5, 7, 8};
        break;

    case N5_1_hole:
        // same as N5 full unsorted with row 3 diag element removed
        n = m = 5;
        nnz   = 7;
        icrow = {0, 2, 3, 4, 6, 7};
        icol  = {3, 0, 1, 2, 1, 4, 4};
        aval  = {2, 1, 3, 4, 5, 7, 8};
        break;

    case N5_empty_rows:
        // removed even more diag elements, creating empty rows
        n = m = 5;
        nnz   = 5;
        icrow = {0, 2, 2, 3, 5, 5};
        icol  = {3, 0, 2, 1, 4};
        aval  = {2, 1, 4, 5, 7};
        break;

    case N10_random:
        // randomly generated matrix with missing elements and unsorted rows
        n = m = 10;
        nnz   = 42;
        icrow = {0, 3, 6, 8, 18, 24, 27, 31, 36, 38, 42};
        icol  = {9, 4, 6, 3, 8, 6, 0, 6, 4, 6, 7, 1, 2, 9, 3, 8, 5, 0, 6, 2, 1,
                 5, 3, 8, 3, 8, 5, 1, 4, 8, 5, 9, 1, 4, 8, 5, 4, 6, 6, 2, 3, 7};
        aval  = {5.91, 5.95, 7.95, 0.83, 5.48, 6.75, 0.01, 4.78, 9.20, 3.40, 2.26, 3.01, 8.34, 6.82,
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
        n     = 7;
        m     = 5;
        nnz   = 13;
        icrow = {0, 3, 5, 6, 10, 13};
        icol  = {0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6};
        aval  = {1, 2, 1, 3, 2, 4, 5, 6, 7, 3, 8, 4, 5};
        break;

    case M5_rect_N7_2holes:
        // same as M5_rect_N7, with missing diag elements in rows 2 an 4
        n     = 7;
        m     = 5;
        nnz   = 11;
        icrow = {0, 3, 5, 5, 9, 11};
        icol  = {0, 3, 5, 1, 5, 1, 3, 4, 6, 5, 6};
        aval  = {1, 2, 1, 3, 2, 5, 6, 7, 3, 4, 5};
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
        n     = 5;
        m     = 7;
        nnz   = 12;
        icrow = {0, 2, 3, 4, 7, 8, 10, 12};
        icol  = {0, 3, 1, 2, 1, 3, 4, 4, 1, 2, 0, 3};
        aval  = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4};
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
        n = m = 5;
        m     = 7;
        nnz   = 10;
        icrow = {0, 2, 3, 3, 6, 6, 8, 10};
        icol  = {0, 3, 1, 3, 1, 4, 1, 2, 0, 3};
        aval  = {1, 2, 3, 6, 5, 7, 1, 2, 3, 4};
        break;

    default:
        if(VERBOSE)
            std::cout << "Non recognized matrix id: " << mid << std::endl;
        return aoclsparse_status_invalid_value;
    }

    ret = aoclsparse_create_dcsr(A, base, m, n, nnz, &icrow[0], &icol[0], &aval[0]);
    if(ret != aoclsparse_status_success && VERBOSE)
        std::cout << "Unexpected error in matrix creation" << std::endl;

    return ret;
}

void expected_sorted_csr(sol_opt_csr& sol, matrix_id mid)
{
    switch(mid)
    {
    case N5_full_sorted:
    case N5_full_unsorted:
        sol.icrow = {0, 2, 3, 4, 7, 8};
        sol.icol  = {0, 3, 1, 2, 1, 3, 4, 4};
        sol.aval  = {1, 2, 3, 4, 5, 6, 7, 8};
        sol.idiag = {0, 2, 3, 5, 7};
        sol.iurow = {1, 3, 4, 6, 8};
        break;
    case N5_1_hole:
        sol.icrow = {0, 2, 3, 4, 7, 8};
        sol.icol  = {0, 3, 1, 2, 1, 3, 4, 4};
        sol.aval  = {1, 2, 3, 4, 5, 0, 7, 8};
        sol.idiag = {0, 2, 3, 5, 7};
        sol.iurow = {1, 3, 4, 6, 8};
        break;
    case N5_empty_rows:
        sol.icrow = {0, 2, 3, 4, 7, 8};
        sol.icol  = {0, 3, 1, 2, 1, 3, 4, 4};
        sol.aval  = {1, 2, 0, 4, 5, 0, 7, 0};
        sol.idiag = {0, 2, 3, 5, 7};
        sol.iurow = {1, 3, 4, 6, 8};
        break;
    case N10_random:
        sol.icrow = {0, 4, 8, 11, 21, 28, 31, 36, 42, 45, 50};
        sol.icol  = {0, 4, 6, 9, 1, 3, 6, 8, 0, 2, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4,
                     5, 6, 8, 3, 5, 8, 1, 4, 5, 6, 8, 1, 4, 5, 7, 8, 9, 4, 6, 8, 2, 3, 6, 7, 9};
        sol.aval  = {0,    5.95, 7.95, 5.91, 0,    0.83, 6.75, 5.48, 0.01, 0,    4.78, 4.96, 3.01,
                     8.34, 7.4,  9.2,  3.31, 3.4,  2.26, 1.12, 6.82, 5.28, 1.77, 3.09, 0,    8.95,
                     2.66, 2.37, 4.48, 1.46, 2.92, 6.17, 8.77, 7.19, 0,    9.96, 6.48, 4.95, 8.87,
                     0,    6.76, 9.61, 5.07, 3.58, 0,    8.66, 6.77, 2.09, 3.69, 0};
        sol.idiag = {0, 4, 9, 14, 24, 29, 34, 39, 44, 49};
        sol.iurow = {1, 5, 10, 15, 25, 30, 35, 40, 45, 50};
        break;
    case M5_rect_N7:
        sol.icrow = {0, 3, 5, 6, 10, 13};
        sol.icol  = {0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6};
        sol.aval  = {1, 2, 1, 3, 2, 4, 5, 6, 7, 3, 8, 4, 5};
        sol.idiag = {0, 3, 5, 7, 10};
        sol.iurow = {1, 4, 6, 8, 11};
        break;
    case M5_rect_N7_2holes:
        sol.icrow = {0, 3, 5, 6, 10, 13};
        sol.icol  = {0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6};
        sol.aval  = {1, 2, 1, 3, 2, 0, 5, 6, 7, 3, 0, 4, 5};
        sol.idiag = {0, 3, 5, 7, 10};
        sol.iurow = {1, 4, 6, 8, 11};
        break;
    case M7_rect_N5:
        sol.icrow = {0, 2, 3, 4, 7, 8, 10, 12};
        sol.icol  = {0, 3, 1, 2, 1, 3, 4, 4, 1, 2, 0, 3};
        sol.aval  = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4};
        sol.idiag = {0, 2, 3, 5, 7};
        sol.iurow = {1, 3, 4, 6, 8};
        break;
    case M7_rect_N5_2holes:
        sol.icrow = {0, 2, 3, 4, 7, 8, 10, 12};
        sol.icol  = {0, 3, 1, 2, 1, 3, 4, 4, 1, 2, 0, 3};
        sol.aval  = {1, 2, 3, 0, 5, 6, 7, 0, 1, 2, 3, 4};
        sol.idiag = {0, 2, 3, 5, 7};
        sol.iurow = {1, 3, 4, 6, 8};
        break;

    default:
        if(VERBOSE)
            std::cout << "unknown problem id" << std::endl;
        break;
    }
}

template <typename T>
void comp_exact_vec(std::string id, aoclsparse_int n, T* v1, T* v2, bool& pass)
{
    pass = true;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        if(v1[i] != v2[i])
        {
            pass = false;
            if(VERBOSE)
                std::cout << id << " index " << i << ": v1[i] = " << v1[i] << ", v2[i] = " << v2[i]
                          << std::endl;
            break;
        }
    }
}

template <typename T>
void comp_tol_vec(std::string id, T tol, aoclsparse_int n, T* v1, T* v2, bool& pass)
{
    pass = true;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        if(abs(v1[i] - v2[i]) > tol)
        {
            pass = false;
            if(VERBOSE)
                std::cout << id << " index " << i << ": v1[i] = " << v1[i] << ", v2[i] = " << v2[i]
                          << std::endl;
            break;
        }
    }
}

void check_opt_csr(aoclsparse_matrix& A, sol_opt_csr& sol, bool& pass)
{
    bool           pass_vec;
    aoclsparse_int dim = std::min(A->n, A->m);
    pass               = true;
    comp_exact_vec("icrow comp", A->m + 1, A->opt_csr_mat.csr_row_ptr, &sol.icrow[0], pass_vec);
    pass = pass && pass_vec;
    comp_exact_vec("icol comp",
                   A->opt_csr_mat.csr_row_ptr[A->m],
                   A->opt_csr_mat.csr_col_ptr,
                   &sol.icol[0],
                   pass_vec);
    pass = pass && pass_vec;
    comp_tol_vec("val comp",
                 TOL_COMP,
                 A->opt_csr_mat.csr_row_ptr[A->m],
                 (double*)A->opt_csr_mat.csr_val,
                 &sol.aval[0],
                 pass_vec);
    pass = pass && pass_vec;
    comp_exact_vec("idiag comp", dim, A->idiag, &sol.idiag[0], pass_vec);
    pass = pass && pass_vec;
    comp_exact_vec("iurow comp", dim, A->iurow, &sol.iurow[0], pass_vec);
    pass = pass && pass_vec;
}

aoclsparse_int test_opt_csr(aoclsparse_int itest, bool& pass)
{
    std::string                 testid;
    aoclsparse_status           ret;
    aoclsparse_matrix           A;
    std::vector<aoclsparse_int> icrow, icol;
    std::vector<double>         aval;
    sol_opt_csr                 sol;
    aoclsparse_operation        trans = aoclsparse_operation_none;
    aoclsparse_mat_descr        descr;
    bool                        pass_comp;

    aoclsparse_create_mat_descr(&descr);
    pass = true;

    switch(itest)
    {
    case 0:
        testid = "using original memory";
        ret    = create_matrix(N5_full_sorted, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, N5_full_sorted);
        break;

    case 1:
        testid = "full diag unsorted matrix";
        ret    = create_matrix(N5_full_unsorted, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, N5_full_unsorted);
        break;

    case 2:
        testid = "unsorted matrix, 1 diag element missing";
        ret    = create_matrix(N5_1_hole, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, N5_1_hole);
        break;

    case 3:
        testid = "with empty rows";
        ret    = create_matrix(N5_empty_rows, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, N5_empty_rows);
        break;

    case 4:
        testid = "bigger random matrix";
        ret    = create_matrix(N10_random, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, N10_random);
        break;

    case 5:
        testid = "rectangular matrix N > M";
        ret    = create_matrix(M5_rect_N7, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, M5_rect_N7);
        break;

    case 6:
        testid = "rectangular matrix N > M, missing diag elements";
        ret    = create_matrix(M5_rect_N7_2holes, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, M5_rect_N7_2holes);
        break;

    case 7:
        testid = "rectangular matrix M > N";
        ret    = create_matrix(M7_rect_N5, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, M7_rect_N5);
        break;

    case 8:
        testid = "rectangular matrix M > N, missing diag elements";
        ret    = create_matrix(M7_rect_N5_2holes, icrow, icol, aval, A);
        if(ret != aoclsparse_status_success)
            pass = false;
        expected_sorted_csr(sol, M7_rect_N5_2holes);
        break;

    default:
        // no more tests to perform return
        return 0;
    }

    std::cout << "Testing " << testid << "... " << std::endl;

    // Hint several actions to force creation of opt_csr
    hint_optimize(100, 100, 0, descr, trans, A);
    check_opt_csr(A, sol, pass_comp);
    pass = pass && pass_comp;
    if(pass)
        std::cout << "OK" << std::endl;
    else
        std::cout << "FAILED" << std::endl;

    return 1;
}

int main()
{
    aoclsparse_int itest = 0, more_opt_csr_tests = 1;
    bool           pass, overall_pass            = true;

    std::cout << "Testing CSR optimize functionality" << std::endl
              << "-----------------------------------" << std::endl;
    while(more_opt_csr_tests)
    {
        more_opt_csr_tests = test_opt_csr(itest, pass);
        overall_pass       = overall_pass && pass;
        itest++;
    }
    std::cout << "-----------------------------------" << std::endl << "CSR optimized tests: ";
    if(overall_pass)
        std::cout << "OK" << std::endl;
    else
        std::cout << "FAILED" << std::endl;

    return 0;
}