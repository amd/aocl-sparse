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
#include "common_data_utils.h"
#include "gtest/gtest.h"

#include <string>

// Unit-Tests for hinting system functionality

namespace
{

    typedef struct
    {
        aoclsparse_int              nnz;
        std::vector<aoclsparse_int> icrow, icol;
        std::vector<aoclsparse_int> idiag, iurow;
        std::vector<double>         aval;
        bool                        opt_csr_is_users;
    } sol_opt_csr;

    void hint_optimize(aoclsparse_int        mv_hints,
                       aoclsparse_int        trsv_hints,
                       aoclsparse_int        lu_hints,
                       aoclsparse_mat_descr &descr,
                       aoclsparse_operation  trans,
                       aoclsparse_matrix    &A)
    {
        if(mv_hints > 0)
        {
            ASSERT_EQ(aoclsparse_set_mv_hint(A, trans, descr, mv_hints), aoclsparse_status_success);
        }
        if(trsv_hints > 0)
        {
            // Only works with symmetric matrices
            aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
            ASSERT_EQ(aoclsparse_set_sv_hint(A, trans, descr, trsv_hints),
                      aoclsparse_status_success);
        }
        if(lu_hints > 0)
        {
            // Currently there is nothing to test for LU hints
            ASSERT_EQ(aoclsparse_set_lu_smoother_hint(A, trans, descr, lu_hints),
                      aoclsparse_status_success);
        }

        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

        return;
    }

    void expected_sorted_csr(sol_opt_csr &sol, matrix_id mid)
    {
        switch(mid)
        {
        case N5_full_sorted:
            sol.icrow            = {0, 2, 3, 4, 7, 8};
            sol.icol             = {0, 3, 1, 2, 1, 3, 4, 4};
            sol.aval             = {1, 2, 3, 4, 5, 6, 7, 8};
            sol.idiag            = {0, 2, 3, 5, 7};
            sol.iurow            = {1, 3, 4, 6, 8};
            sol.opt_csr_is_users = true;
            break;
        case N5_full_unsorted:
            sol.icrow            = {0, 2, 3, 4, 7, 8};
            sol.icol             = {0, 3, 1, 2, 1, 3, 4, 4};
            sol.aval             = {1, 2, 3, 4, 5, 6, 7, 8};
            sol.idiag            = {0, 2, 3, 5, 7};
            sol.iurow            = {1, 3, 4, 6, 8};
            sol.opt_csr_is_users = false;
            break;
        case N59_partial_sort:
            sol.icrow            = {0, 2, 3, 4, 8, 9};
            sol.icol             = {0, 3, 1, 2, 2, 1, 3, 4, 4};
            sol.aval             = {1, 2, 3, 4, 9, 5, 6, 7, 8};
            sol.idiag            = {0, 2, 3, 5, 7};
            sol.iurow            = {1, 3, 4, 6, 8};
            sol.opt_csr_is_users = true;
            break;
        case N5_1_hole:
            sol.icrow            = {0, 2, 3, 4, 7, 8};
            sol.icol             = {0, 3, 1, 2, 1, 3, 4, 4};
            sol.aval             = {1, 2, 3, 4, 5, 0, 7, 8};
            sol.idiag            = {0, 2, 3, 5, 7};
            sol.iurow            = {1, 3, 4, 6, 8};
            sol.opt_csr_is_users = false;
            break;
        case N5_empty_rows:
            sol.icrow            = {0, 2, 3, 4, 7, 8};
            sol.icol             = {0, 3, 1, 2, 1, 3, 4, 4};
            sol.aval             = {1, 2, 0, 4, 5, 0, 7, 0};
            sol.idiag            = {0, 2, 3, 5, 7};
            sol.iurow            = {1, 3, 4, 6, 8};
            sol.opt_csr_is_users = false;
            break;
        case N10_random:
            sol.icrow = {0, 4, 8, 11, 21, 28, 31, 36, 42, 45, 50};
            sol.icol  = {0, 4, 6, 9, 1, 3, 6, 8, 0, 2, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4,
                         5, 6, 8, 3, 5, 8, 1, 4, 5, 6, 8, 1, 4, 5, 7, 8, 9, 4, 6, 8, 2, 3, 6, 7, 9};
            sol.aval
                = {0,    5.95, 7.95, 5.91, 0,    0.83, 6.75, 5.48, 0.01, 0,    4.78, 4.96, 3.01,
                   8.34, 7.4,  9.2,  3.31, 3.4,  2.26, 1.12, 6.82, 5.28, 1.77, 3.09, 0,    8.95,
                   2.66, 2.37, 4.48, 1.46, 2.92, 6.17, 8.77, 7.19, 0,    9.96, 6.48, 4.95, 8.87,
                   0,    6.76, 9.61, 5.07, 3.58, 0,    8.66, 6.77, 2.09, 3.69, 0};
            sol.idiag            = {0, 4, 9, 14, 24, 29, 34, 39, 44, 49};
            sol.iurow            = {1, 5, 10, 15, 25, 30, 35, 40, 45, 50};
            sol.opt_csr_is_users = false;
            break;
        case M5_rect_N7:
            sol.icrow            = {0, 3, 5, 6, 10, 13};
            sol.icol             = {0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6};
            sol.aval             = {1, 2, 1, 3, 2, 4, 5, 6, 7, 3, 8, 4, 5};
            sol.idiag            = {0, 3, 5, 7, 10};
            sol.iurow            = {1, 4, 6, 8, 11};
            sol.opt_csr_is_users = true;
            break;
        case M5_rect_N7_2holes:
            sol.icrow            = {0, 3, 5, 6, 10, 13};
            sol.icol             = {0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6};
            sol.aval             = {1, 2, 1, 3, 2, 0, 5, 6, 7, 3, 0, 4, 5};
            sol.idiag            = {0, 3, 5, 7, 10};
            sol.iurow            = {1, 4, 6, 8, 11};
            sol.opt_csr_is_users = false;
            break;
        case M7_rect_N5:
            sol.icrow            = {0, 2, 3, 4, 7, 8, 10, 12};
            sol.icol             = {0, 3, 1, 2, 1, 3, 4, 4, 1, 2, 0, 3};
            sol.aval             = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4};
            sol.idiag            = {0, 2, 3, 5, 7};
            sol.iurow            = {1, 3, 4, 6, 8};
            sol.opt_csr_is_users = true;
            break;
        case M7_rect_N5_2holes:
            sol.icrow            = {0, 2, 3, 4, 7, 8, 10, 12};
            sol.icol             = {0, 3, 1, 2, 1, 3, 4, 4, 1, 2, 0, 3};
            sol.aval             = {1, 2, 3, 0, 5, 6, 7, 0, 1, 2, 3, 4};
            sol.idiag            = {0, 2, 3, 5, 7};
            sol.iurow            = {1, 3, 4, 6, 8};
            sol.opt_csr_is_users = false;
            break;

        default: // LCOV_EXCL_LINE
            FAIL() << "ERROR: Unknown matrix ID: mid=" << mid << "."; // LCOV_EXCL_LINE
            break;
        }
    }

    void check_opt_csr(aoclsparse_matrix &A, sol_opt_csr &sol)
    {
        aoclsparse_int dim = std::min(A->n, A->m);
        ASSERT_EQ(A->opt_csr_is_users, sol.opt_csr_is_users);
        EXPECT_EQ_VEC(A->m + 1, A->opt_csr_mat.csr_row_ptr, sol.icrow);
        EXPECT_EQ_VEC(A->opt_csr_mat.csr_row_ptr[A->m], A->opt_csr_mat.csr_col_ptr, sol.icol);
        EXPECT_DOUBLE_EQ_VEC(
            A->opt_csr_mat.csr_row_ptr[A->m], (double *)A->opt_csr_mat.csr_val, sol.aval);
        EXPECT_EQ_VEC(dim, A->idiag, sol.idiag);
        EXPECT_EQ_VEC(dim, A->iurow, sol.iurow);
    }

    typedef struct
    {
        std::string testname;
        matrix_id   mid;
    } hlist_t;

    class Pos : public testing::TestWithParam<hlist_t>
    {
    };

    void PrintTo(const hlist_t &param, ::std::ostream *os)
    {
        *os << param.testname;
    }

    /*
    * Positive Hinting tests
    * ======================
    *
    * The following test verify that "clean CSR" (sorted CSR with explicit
    * pointers to L, D and U) is correctly generated by comparing it to explicit
    * hand-written results.
    *
    * At the moment that's the case for any TRSV hint: Either copy & sort the
    * unsorted data or take directly user's data.
    *
    */

    TEST_P(Pos, CleanCSR)
    {
        matrix_id                   mid = GetParam().mid;
        aoclsparse_matrix           A;
        std::vector<aoclsparse_int> icrow, icol;
        std::vector<double>         aval;
        sol_opt_csr                 sol;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        aoclsparse_mat_descr        descr;
        aoclsparse_int              m, n, nnz;
        ASSERT_EQ(create_matrix(mid, m, n, nnz, icrow, icol, aval, A, descr, 1),
                  aoclsparse_status_success);
        expected_sorted_csr(sol, mid);
        hint_optimize(100, 100, 100, descr, trans, A);
        check_opt_csr(A, sol);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    };

    hlist_t hlist[] = {{"UsingOriginalMemory", N5_full_sorted},
                       {"FullDiagUnsortedMatrix", N5_full_unsorted},
                       {"nsrtdMat1DiagElmntMissng", N5_1_hole},
                       {"WithEmptyRows", N5_empty_rows},
                       {"BiggerRandMatrix", N10_random},
                       {"RectangularMatrixNgtM", M5_rect_N7},
                       {"RectMatNgtMwMissngDiagElms", M5_rect_N7_2holes},
                       {"RectMatrixMgtN", M7_rect_N5},
                       {"RectMat_MgtNwMissngDiagElms", M7_rect_N5_2holes},
                       {"PartialOrderingNoCopy", N59_partial_sort}};

    INSTANTIATE_TEST_CASE_P(HintSuite, Pos, ::testing::ValuesIn(hlist));

} // namespace
