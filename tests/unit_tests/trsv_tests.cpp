/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * Unit-tests for TRiangular SolVers (aoclsparse_trsv)
 */
#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#define VERBOSE 1

namespace
{
    // If kid > 0 then force to only test the specified KID, otherwise
    // cycle through all of them
    template <typename T>
    void trsv_driver(linear_system_id  id,
                     aoclsparse_int    kid,
                     aoclsparse_int    base_index,
                     aoclsparse_status trsv_status = aoclsparse_status_success)
    {
        aoclsparse_status    status;
        std::string          title;
        T                    alpha;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        std::vector<T>       b;
        std::vector<T>       x;
        std::vector<T>       xref;
        T                    xtol;
        aoclsparse_operation trans;
        // permanent storage of matrix data
        std::vector<T>                 aval;
        std::vector<aoclsparse_int>    icola;
        std::vector<aoclsparse_int>    icrowa;
        std::array<aoclsparse_int, 10> iparm{0};
        std::array<T, 10>              dparm;
        aoclsparse_status              exp_status;
        aoclsparse_index_base          base = (aoclsparse_index_base)base_index;

        decltype(std::real(xtol)) tol;

        status = create_linear_system<T>(id,
                                         title,
                                         trans,
                                         A,
                                         descr,
                                         base,
                                         alpha,
                                         b,
                                         x,
                                         xref,
                                         xtol,
                                         icrowa,
                                         icola,
                                         aval,
                                         iparm,
                                         dparm,
                                         exp_status);
        ASSERT_EQ(status, aoclsparse_status_success)
            << "Error: could not find linear system id " << id << "!";
        tol = std::real(xtol); // get the tolerance.
#if(VERBOSE > 0)
        std::string oplabel{""};
        switch(trans)
        {
        case aoclsparse_operation_none:
            oplabel = "None";
            break;
        case aoclsparse_operation_transpose:
            oplabel = "Transpose";
            break;
        case aoclsparse_operation_conjugate_transpose:
            oplabel = "Hermitian Transpose";
            break;
        }
        std::string dtype = "unknown";
        if(typeid(T) == typeid(double))
        {
            dtype = "double";
        }
        else if(typeid(T) == typeid(float))
        {
            dtype = "float";
        }
        if(typeid(T) == typeid(std::complex<double>))
        {
            dtype = "cdouble";
        }
        else if(typeid(T) == typeid(std::complex<float>))
        {
            dtype = "cfloat";
        }
        const bool     unit    = descr->diag_type == aoclsparse_diag_type_unit;
        aoclsparse_int kidlabs = (std::max<aoclsparse_int>)((std::min<aoclsparse_int>)(kid, 4), 0);
        const std::string avxlabs[5] = {"NONE (reference)",
                                        "AVX2 alias (KT 256b)",
                                        "AVX2 (KT 256b)",
                                        "AVX-512 (KT 512b)",
                                        "unknown"};
        std::cout << "Problem id: " << id << " \"" << title << "\"" << std::endl;
        std::cout << "Configuration: <" << dtype << "> unit=" << (unit ? "Unit" : "Non-unit")
                  << " op=" << oplabel << "   Kernel id=" << kid << " <" << avxlabs[kidlabs] << ">"
                  << std::endl;
#endif
        const aoclsparse_int n = A->n;
        status                 = aoclsparse_trsv_kid<T>(trans, alpha, A, descr, &b[0], &x[0], kid);
        ASSERT_EQ(status, trsv_status)
            << "Test failed with unexpected return from aoclsparse_trsv for type " << dtype;
        if(status == aoclsparse_status_success)
        {
            if(tol <= 0.0)
                tol = 10;

            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                EXPECT_COMPLEX_ARR_NEAR(n, x, xref, expected_precision<decltype(tol)>(tol));
            }
            else
            {
                EXPECT_ARR_NEAR(n, x, xref, expected_precision<decltype(tol)>(tol));
            }
        }
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

    typedef struct
    {
        linear_system_id id;
        std::string      testname;
        aoclsparse_int   kid;
        aoclsparse_int   base;
        aoclsparse_int   op;
    } trsv_list_t;

#define BASEZERO 0
#define BASEONE 1
#define NONE 0
#define TRAN 1
#define HERM 2

    // clang-format off
#define ADD_TEST(ID, KID, BASE, OP) { ID, #ID "/kid[" #KID "]" "/" #BASE, KID, BASE, OP }
    // Kernel kid=3 fallbacks to kid=2 on non-Zen4 and should not fail
    // Kernel kid=1 is an alias to kid=2, so they both are interparsed
#define ADD_TEST_BATCH(PRE)                                                                  \
    ADD_TEST(PRE##_Lx_aB,    0,BASEZERO,NONE), ADD_TEST(PRE##_Lx_aB,    1,BASEZERO,NONE), ADD_TEST(PRE##_Lx_aB,    3,BASEZERO,NONE),\
    ADD_TEST(PRE##_Lx_aB,    0,BASEONE, NONE), ADD_TEST(PRE##_Lx_aB,    2,BASEONE, NONE), ADD_TEST(PRE##_Lx_aB,    3,BASEONE, NONE),\
    ADD_TEST(PRE##_LL_Ix_aB, 0,BASEZERO,NONE), ADD_TEST(PRE##_LL_Ix_aB, 2,BASEZERO,NONE), ADD_TEST(PRE##_LL_Ix_aB, 3,BASEZERO,NONE),\
    ADD_TEST(PRE##_LL_Ix_aB, 0,BASEONE, NONE), ADD_TEST(PRE##_LL_Ix_aB, 1,BASEONE, NONE), ADD_TEST(PRE##_LL_Ix_aB, 3,BASEONE, NONE),\
    ADD_TEST(PRE##_LTx_aB,   0,BASEZERO,TRAN), ADD_TEST(PRE##_LTx_aB,   1,BASEZERO,TRAN), ADD_TEST(PRE##_LTx_aB,   3,BASEZERO,TRAN),\
    ADD_TEST(PRE##_LTx_aB,   0,BASEONE, TRAN), ADD_TEST(PRE##_LTx_aB,   2,BASEONE, TRAN), ADD_TEST(PRE##_LTx_aB,   3,BASEONE, TRAN),\
    ADD_TEST(PRE##_LL_ITx_aB,0,BASEZERO,TRAN), ADD_TEST(PRE##_LL_ITx_aB,1,BASEZERO,TRAN), ADD_TEST(PRE##_LL_ITx_aB,3,BASEZERO,TRAN),\
    ADD_TEST(PRE##_LL_ITx_aB,0,BASEONE, TRAN), ADD_TEST(PRE##_LL_ITx_aB,2,BASEONE, TRAN), ADD_TEST(PRE##_LL_ITx_aB,3,BASEONE, TRAN),\
    ADD_TEST(PRE##_LHx_aB,   0,BASEZERO,HERM), ADD_TEST(PRE##_LHx_aB,   1,BASEZERO,HERM), ADD_TEST(PRE##_LHx_aB,   3,BASEZERO,HERM),\
    ADD_TEST(PRE##_LHx_aB,   0,BASEONE, HERM), ADD_TEST(PRE##_LHx_aB,   2,BASEONE, HERM), ADD_TEST(PRE##_LHx_aB,   3,BASEONE, HERM),\
    ADD_TEST(PRE##_LL_IHx_aB,0,BASEZERO,HERM), ADD_TEST(PRE##_LL_IHx_aB,2,BASEZERO,HERM), ADD_TEST(PRE##_LL_IHx_aB,3,BASEZERO,HERM),\
    ADD_TEST(PRE##_LL_IHx_aB,0,BASEONE, HERM), ADD_TEST(PRE##_LL_IHx_aB,1,BASEONE, HERM), ADD_TEST(PRE##_LL_IHx_aB,3,BASEONE, HERM),\
    ADD_TEST(PRE##_Ux_aB,    0,BASEZERO,NONE), ADD_TEST(PRE##_Ux_aB,    2,BASEZERO,NONE), ADD_TEST(PRE##_Ux_aB,    3,BASEZERO,NONE),\
    ADD_TEST(PRE##_Ux_aB,    0,BASEONE, NONE), ADD_TEST(PRE##_Ux_aB,    1,BASEONE, NONE), ADD_TEST(PRE##_Ux_aB,    3,BASEONE, NONE),\
    ADD_TEST(PRE##_UU_Ix_aB, 0,BASEZERO,NONE), ADD_TEST(PRE##_UU_Ix_aB, 1,BASEZERO,NONE), ADD_TEST(PRE##_UU_Ix_aB, 3,BASEZERO,NONE),\
    ADD_TEST(PRE##_UU_Ix_aB, 0,BASEONE, NONE), ADD_TEST(PRE##_UU_Ix_aB, 2,BASEONE, NONE), ADD_TEST(PRE##_UU_Ix_aB, 3,BASEONE, NONE),\
    ADD_TEST(PRE##_UTx_aB,   0,BASEZERO,TRAN), ADD_TEST(PRE##_UTx_aB,   2,BASEZERO,TRAN), ADD_TEST(PRE##_UTx_aB,   3,BASEZERO,TRAN),\
    ADD_TEST(PRE##_UTx_aB,   0,BASEONE, TRAN), ADD_TEST(PRE##_UTx_aB,   1,BASEONE, TRAN), ADD_TEST(PRE##_UTx_aB,   3,BASEONE, TRAN),\
    ADD_TEST(PRE##_UU_ITx_aB,0,BASEZERO,TRAN), ADD_TEST(PRE##_UU_ITx_aB,2,BASEZERO,TRAN), ADD_TEST(PRE##_UU_ITx_aB,3,BASEZERO,TRAN),\
    ADD_TEST(PRE##_UU_ITx_aB,0,BASEONE, TRAN), ADD_TEST(PRE##_UU_ITx_aB,1,BASEONE, TRAN), ADD_TEST(PRE##_UU_ITx_aB,3,BASEONE, TRAN),\
    ADD_TEST(PRE##_UHx_aB,   0,BASEZERO,HERM), ADD_TEST(PRE##_UHx_aB,   2,BASEZERO,HERM), ADD_TEST(PRE##_UHx_aB,   3,BASEZERO,HERM),\
    ADD_TEST(PRE##_UHx_aB,   0,BASEONE, HERM), ADD_TEST(PRE##_UHx_aB,   1,BASEONE, HERM), ADD_TEST(PRE##_UHx_aB,   3,BASEONE, HERM),\
    ADD_TEST(PRE##_UU_IHx_aB,0,BASEZERO,HERM), ADD_TEST(PRE##_UU_IHx_aB,2,BASEZERO,HERM), ADD_TEST(PRE##_UU_IHx_aB,3,BASEZERO,HERM),\
    ADD_TEST(PRE##_UU_IHx_aB,0,BASEONE, HERM), ADD_TEST(PRE##_UU_IHx_aB,1,BASEONE, HERM), ADD_TEST(PRE##_UU_IHx_aB,3,BASEONE, HERM)
    // clang-format on

    trsv_list_t trsv_list[] = {ADD_TEST_BATCH(D7),
                               ADD_TEST_BATCH(S7),
                               ADD_TEST_BATCH(N25),
                               ADD_TEST(D7_Lx_aB, 999, BASEZERO, NONE),
                               ADD_TEST(D7_Lx_aB, -1, BASEZERO, NONE)};
    void        PrintTo(const trsv_list_t &param, ::std::ostream *os)
    {
        *os << param.testname;
    }

    class PosDouble : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosDouble, Solver)
    {
        const linear_system_id  id          = GetParam().id;
        const aoclsparse_int    kid         = GetParam().kid;
        const aoclsparse_int    base        = GetParam().base;
        const aoclsparse_status trsv_status = aoclsparse_status_success;
#if(VERBOSE > 0)
        std::cout << "Pos/Double/Solver test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsv_driver<double>(id, kid, base, trsv_status);
    }
    INSTANTIATE_TEST_SUITE_P(TrsvSuite, PosDouble, ::testing::ValuesIn(trsv_list));

    class PosFloat : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosFloat, Solver)
    {
        const linear_system_id  id          = GetParam().id;
        const aoclsparse_int    kid         = GetParam().kid;
        const aoclsparse_int    base        = GetParam().base;
        const aoclsparse_status trsv_status = aoclsparse_status_success;
#if(VERBOSE > 0)
        std::cout << "Pos/Float/Solver test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsv_driver<float>(id, kid, base, trsv_status);
    }
    INSTANTIATE_TEST_SUITE_P(TrsvSuite, PosFloat, ::testing::ValuesIn(trsv_list));

    class PosCplxDouble : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosCplxDouble, Solver)
    {
        const linear_system_id  id          = GetParam().id;
        const aoclsparse_int    kid         = GetParam().kid;
        const aoclsparse_int    base        = GetParam().base;
        const aoclsparse_status trsv_status = aoclsparse_status_success;
#if(VERBOSE > 0)
        std::cout << "Pos/CplxDouble/Solver test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        trsv_driver<std::complex<double>>(id, kid, base, trsv_status);
    }
    INSTANTIATE_TEST_SUITE_P(TrsvSuite, PosCplxDouble, ::testing::ValuesIn(trsv_list));

    class PosCplxFloat : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosCplxFloat, Solver)
    {
        const linear_system_id  id          = GetParam().id;
        const aoclsparse_int    kid         = GetParam().kid;
        const aoclsparse_int    base        = GetParam().base;
        const aoclsparse_status trsv_status = aoclsparse_status_success;
#if(VERBOSE > 0)
        std::cout << "Pos/CplxFloat/Solver test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        trsv_driver<std::complex<float>>(id, kid, base, trsv_status);
    }
    INSTANTIATE_TEST_SUITE_P(TrsvSuite, PosCplxFloat, ::testing::ValuesIn(trsv_list));

    TEST(TrsvSuite, BaseZeroDouble)
    {
        double               alpha = 1.0;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        aoclsparse_int       kid   = 0;
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int n = 4, m = 4, nnz = 10;
        aoclsparse_int csr_row_ptr[5]  = {0, 2, 5, 8, 10};
        aoclsparse_int csr_col_ind[10] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
        double         csr_val[10]     = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
        double         b[4]            = {1, 4, 4, 4};
        double         x_gold[4]       = {1.00, 1.00, 1.00, 1.00};
        double         x[4]            = {0};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_create_dcsr(
                      &A, aoclsparse_index_base_zero, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(n, x, x_gold, expected_precision<double>(10.0));
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };
    TEST(TrsvSuite, BaseOneDouble)
    {
        double               alpha = 1.0;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        aoclsparse_int       kid   = 0;
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int n = 4, m = 4, nnz = 10;
        aoclsparse_int csr_row_ptr[5]  = {1, 3, 6, 9, 11};
        aoclsparse_int csr_col_ind[10] = {1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
        double         csr_val[10]     = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
        double         b[4]            = {1, 4, 4, 4};
        double         x_gold[4]       = {1.00, 1.00, 1.00, 1.00};
        double         x[4]            = {0};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_create_dcsr(
                      &A, aoclsparse_index_base_one, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_success);

        EXPECT_ARR_NEAR(n, x, x_gold, expected_precision<double>(10.0));
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };
    TEST(TrsvSuite, BaseOneFloat)
    {
        float                alpha = 1.0;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        aoclsparse_int       kid   = 0;
        aoclsparse_operation trans = aoclsparse_operation_none;

        aoclsparse_int n = 4, m = 4, nnz = 10;
        aoclsparse_int csr_row_ptr[5]  = {1, 3, 6, 9, 11};
        aoclsparse_int csr_col_ind[10] = {1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
        float          csr_val[10]     = {1, 2, 3, 1, 2, 3, 1, 2, 3, 1};
        float          b[4]            = {1, 4, 4, 4};
        float          x_gold[4]       = {1.00, 1.00, 1.00, 1.00};
        float          x[4]            = {0};

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_create_scsr(
                      &A, aoclsparse_index_base_one, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_strsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(n, x, x_gold, expected_precision<float>(10.0));
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };
    TEST(TrsvSuite, TcsrBaseZeroDouble)
    {
        double               alpha = 1.0;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        aoclsparse_int       kid   = 0;
        aoclsparse_operation trans = aoclsparse_operation_none;

        // Input parameters
        aoclsparse_int              m, n, nnz;
        std::vector<aoclsparse_int> row_ptr_L, row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L, col_idx_U;
        std::vector<double>         val_L, val_U;
        init_tcsr_matrix<double>(m,
                                 n,
                                 nnz,
                                 row_ptr_L,
                                 row_ptr_U,
                                 col_idx_L,
                                 col_idx_U,
                                 val_L,
                                 val_U,
                                 aoclsparse_fully_sorted,
                                 aoclsparse_index_base_zero);

        std::vector<double> b, x_gold_L, x_gold_U, x;
        b.assign({1, 1, 1, 1});
        x_gold_L.assign({-1.00, 0.25, -1.00, 0.66666666666666663});
        x_gold_U.assign({-1.0, 0.25, -0.16666666666666666, 0.1111111111111111});
        x.assign({0, 0, 0, 0});

        // Create mat descr
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        // Create TCSR Matrix
        ASSERT_EQ(aoclsparse_create_dtcsr(&A,
                                          aoclsparse_index_base_zero,
                                          m,
                                          n,
                                          nnz,
                                          row_ptr_L.data(),
                                          row_ptr_U.data(),
                                          col_idx_L.data(),
                                          col_idx_U.data(),
                                          val_L.data(),
                                          val_U.data()),
                  aoclsparse_status_success);

        // Solve for (L+D)x=b
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b.data(), x.data(), kid),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(n, x, x_gold_L, expected_precision<double>(10.0));

        // Solve for (D+U)x = b
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b.data(), x.data(), kid),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(n, x, x_gold_U, expected_precision<double>(10.0));

        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };
    TEST(TrsvSuite, TcsrBaseOneDouble)
    {
        double               alpha = 1.0;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        aoclsparse_int       kid   = 0;
        aoclsparse_operation trans = aoclsparse_operation_none;

        // Input parameters
        aoclsparse_int              m = 4, n = 4, nnz = 9;
        std::vector<aoclsparse_int> row_ptr_L, row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L, col_idx_U;
        std::vector<double>         val_L, val_U;
        init_tcsr_matrix<double>(m,
                                 n,
                                 nnz,
                                 row_ptr_L,
                                 row_ptr_U,
                                 col_idx_L,
                                 col_idx_U,
                                 val_L,
                                 val_U,
                                 aoclsparse_fully_sorted,
                                 aoclsparse_index_base_one);

        std::vector<double> b, x_gold_L, x_gold_U, x;
        b.assign({1.0, 2.0, 3.0, 4.0});
        x_gold_L.assign({-1.00, 0.5, -1.3333333333333333, 0.77777777777777779});
        x_gold_U.assign({-0.66666666666666674, 0.5, -0.5, 0.44444444444444442});
        x.assign({0, 0, 0, 0});

        // Create mat descr
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        // Create TCSR Matrix
        ASSERT_EQ(aoclsparse_create_dtcsr(&A,
                                          aoclsparse_index_base_one,
                                          m,
                                          n,
                                          nnz,
                                          row_ptr_L.data(),
                                          row_ptr_U.data(),
                                          col_idx_L.data(),
                                          col_idx_U.data(),
                                          val_L.data(),
                                          val_U.data()),
                  aoclsparse_status_success);

        // Solve for (L+D)x=b
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b.data(), x.data(), kid),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(n, x, x_gold_L, expected_precision<double>(10.0));

        // Solve for (D+U)x = b
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b.data(), x.data(), kid),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(n, x, x_gold_U, expected_precision<double>(10.0));

        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };
    TEST(TrsvSuite, TcsrBaseOnefloat)
    {
        float                alpha = 1.0;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        aoclsparse_int       kid   = 0;
        aoclsparse_operation trans = aoclsparse_operation_none;

        // Input parameters
        aoclsparse_int              m = 4, n = 4, nnz = 9;
        std::vector<aoclsparse_int> row_ptr_L, row_ptr_U;
        std::vector<aoclsparse_int> col_idx_L, col_idx_U;
        std::vector<float>          val_L, val_U;
        init_tcsr_matrix<float>(m,
                                n,
                                nnz,
                                row_ptr_L,
                                row_ptr_U,
                                col_idx_L,
                                col_idx_U,
                                val_L,
                                val_U,
                                aoclsparse_fully_sorted,
                                aoclsparse_index_base_one);

        std::vector<float> b, x_gold_L, x_gold_U, x;
        b.assign({1.0, 2.0, 3.0, 4.0});
        x_gold_L.assign({-1.00, 0.5, -1.3333333333333333, 0.77777777777777779});
        x_gold_U.assign({-0.66666666666666674, 0.5, -0.5, 0.44444444444444442});
        x.assign({0, 0, 0, 0});

        // Create mat descr
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);

        // Create TCSR Matrix
        ASSERT_EQ(aoclsparse_create_stcsr(&A,
                                          aoclsparse_index_base_one,
                                          m,
                                          n,
                                          nnz,
                                          row_ptr_L.data(),
                                          row_ptr_U.data(),
                                          col_idx_L.data(),
                                          col_idx_U.data(),
                                          val_L.data(),
                                          val_U.data()),
                  aoclsparse_status_success);

        // Solve for (L+D)x=b
        ASSERT_EQ(aoclsparse_strsv_kid(trans, alpha, A, descr, b.data(), x.data(), kid),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(n, x, x_gold_L, expected_precision<float>(10.0));

        // Solve for (D+U)x = b
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        ASSERT_EQ(aoclsparse_strsv_kid(trans, alpha, A, descr, b.data(), x.data(), kid),
                  aoclsparse_status_success);
        EXPECT_ARR_NEAR(n, x, x_gold_U, expected_precision<float>(10.0));

        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };
    TEST(TrsvSuite, WrongTypeDouble)
    {
        float                       alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        float                       b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        // Force it to lower triangular
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        // Create matrix data for type double
        ASSERT_EQ(create_matrix<double>(N5_full_sorted, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_sv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_strsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_wrong_type);

        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    };

    TEST(TrsvSuite, WrongTypeFloat)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<float>          aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        // Force it to lower triangular
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        // Create matrix data for type double
        ASSERT_EQ(create_matrix<float>(N5_full_sorted, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_sv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_wrong_type);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, NullPtrA)
    {
        double               alpha = 0.0;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        double               b[1], x[1];
        aoclsparse_int       kid   = 0;
        aoclsparse_operation trans = aoclsparse_operation_none;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_pointer);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    };

    TEST(TrsvSuite, NullPtrDesc)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, nullptr, b, x, kid),
                  aoclsparse_status_invalid_pointer);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, NullPtrX)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, nullptr, kid),
                  aoclsparse_status_invalid_pointer);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, NullPtrB)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, nullptr, x, kid),
                  aoclsparse_status_invalid_pointer);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, InvalidRowSize)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        A->m = -1;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_size);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, InvalidNNZ)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        A->nnz = -1;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_size);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, GeneralMatrix)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        descr->type = aoclsparse_matrix_type_general;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_value);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, NotSquare)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        A->n = 0;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_value);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, GeneralIncompleteRank)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        // Force it to lower triangular
        descr->fill_mode = aoclsparse_fill_mode_lower;
        descr->type      = aoclsparse_matrix_type_triangular;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_value);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };

    template <typename T>
    void trsv_hint_driver(linear_system_id id, aoclsparse_int base_index)
    {
        std::string          title;
        T                    alpha;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        std::vector<T>       b;
        std::vector<T>       x;
        std::vector<T>       xref;
        T                    xtol;
        aoclsparse_operation trans;
        // permanent storage of matrix data
        std::vector<T>                 aval;
        std::vector<aoclsparse_int>    icola;
        std::vector<aoclsparse_int>    icrowa;
        std::array<aoclsparse_int, 10> iparm{0};
        std::array<T, 10>              dparm;
        aoclsparse_status              status, exp_status;
        aoclsparse_index_base          base = (aoclsparse_index_base)base_index;

        // number of calls need to be >0
        iparm[0] = 2;
        status   = create_linear_system<T>(id,
                                         title,
                                         trans,
                                         A,
                                         descr,
                                         base,
                                         alpha,
                                         b,
                                         x,
                                         xref,
                                         xtol,
                                         icrowa,
                                         icola,
                                         aval,
                                         iparm,
                                         dparm,
                                         exp_status);
        ASSERT_EQ(status, aoclsparse_status_success) << "could not find linear system id " << id;
        aoclsparse_int ncalls = iparm[0];
        status                = aoclsparse_set_sv_hint(A, trans, descr, ncalls);
        ASSERT_EQ(status, exp_status)
            << "failed with unexpected return from aoclsparse_set_sv_hint";
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
#define ADD_TEST_HINT(ID, BASE)          \
    {                                    \
        ID, #ID "/" #BASE, 0, BASE, NONE \
    }
    trsv_list_t trsv_hint_list[] = {ADD_TEST_HINT(D7Lx_aB_hint, BASEZERO),
                                    ADD_TEST_HINT(A_nullptr, BASEZERO),
                                    ADD_TEST_HINT(D1_descr_nullptr, BASEZERO),
                                    ADD_TEST_HINT(D1_neg_num_hint, BASEZERO),
                                    ADD_TEST_HINT(D1_mattype_gen_hint, BASEZERO),
                                    ADD_TEST_HINT(D7Lx_aB_hint, BASEONE),
                                    ADD_TEST_HINT(A_nullptr, BASEONE),
                                    ADD_TEST_HINT(D1_descr_nullptr, BASEONE),
                                    ADD_TEST_HINT(D1_neg_num_hint, BASEONE),
                                    ADD_TEST_HINT(D1_mattype_gen_hint, BASEONE)};

    class PosHDouble : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosHDouble, Hint)
    {
        const linear_system_id id   = GetParam().id;
        const aoclsparse_int   base = GetParam().base;
#if(VERBOSE > 0)
        std::cout << "PosH/Double test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsv_hint_driver<double>(id, base);
    }
    INSTANTIATE_TEST_SUITE_P(TrsvSuite, PosHDouble, ::testing::ValuesIn(trsv_hint_list));

    class PosHFloat : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosHFloat, Hint)
    {
        const linear_system_id id   = GetParam().id;
        const aoclsparse_int   base = GetParam().base;
#if(VERBOSE > 0)
        std::cout << "PosH/Float test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsv_hint_driver<float>(id, base);
    }
    INSTANTIATE_TEST_SUITE_P(TrsvSuite, PosHFloat, ::testing::ValuesIn(trsv_hint_list));

    // Test for stride of x or stride of b are invalid
    TEST(TrsvSuite, wrongStride)
    {
        float                       sa{1.0f};
        double                      da{1.0};
        aoclsparse_float_complex    ca{1.0f, 1.0f};
        aoclsparse_double_complex   za{1.0, 1.0};
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        float                       sb[1], sx[1];
        double                      db[1], dx[1];
        aoclsparse_float_complex    cb[1], cx[1];
        aoclsparse_double_complex   zb[1], zx[1];
        aoclsparse_int              m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular),
                  aoclsparse_status_success);
        // Force it to lower triangular
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        // Create matrix data for type double
        ASSERT_EQ(create_matrix<double>(N5_full_sorted, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_set_sv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);

        // Tests are Ok since incx and incb are check BEFORE data type in A checked
        ASSERT_EQ(aoclsparse_strsv_strided(trans, sa, A, descr, sb, 1, sx, -1),
                  aoclsparse_status_invalid_value);
        ASSERT_EQ(aoclsparse_dtrsv_strided(trans, da, A, descr, db, -1, dx, 0),
                  aoclsparse_status_invalid_value);
        ASSERT_EQ(aoclsparse_ctrsv_strided(trans, ca, A, descr, cb, -1, cx, 1),
                  aoclsparse_status_invalid_value);
        // Coverage of public interface
        ASSERT_EQ(aoclsparse_ztrsv_strided(trans, za, A, descr, zb, -1, zx, -1),
                  aoclsparse_status_invalid_value);

        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }
}
// namespace
