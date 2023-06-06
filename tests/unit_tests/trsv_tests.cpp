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
    template <typename T>
    aoclsparse_status aoclsparse_trsv_kid(const aoclsparse_operation trans,
                                          const T                    alpha,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          const T                   *b,
                                          T                         *x,
                                          const aoclsparse_int       kid);

    template <>
    aoclsparse_status aoclsparse_trsv_kid<double>(const aoclsparse_operation trans,
                                                  const double               alpha,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  const double              *b,
                                                  double                    *x,
                                                  const aoclsparse_int       kid)
    {
        if(kid >= 0)
            return aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid);
        else
            return aoclsparse_dtrsv(trans, alpha, A, descr, b, x);
    }

    template <>
    aoclsparse_status aoclsparse_trsv_kid<float>(const aoclsparse_operation trans,
                                                 const float                alpha,
                                                 aoclsparse_matrix          A,
                                                 const aoclsparse_mat_descr descr,
                                                 const float               *b,
                                                 float                     *x,
                                                 const aoclsparse_int       kid)
    {
        if(kid >= 0)
            return aoclsparse_strsv_kid(trans, alpha, A, descr, b, x, kid);
        else
            return aoclsparse_strsv(trans, alpha, A, descr, b, x);
    }

    // If kid > 0 then force to only test the specified KID, otherwise
    // cycle through all of them
    template <typename T>
    void trsv_driver(linear_system_id  id,
                     aoclsparse_int    kid,
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
        std::array<aoclsparse_int, 10> iparm;
        std::array<T, 10>              dparm;
        aoclsparse_status              exp_status;

        status = create_linear_system<T>(id,
                                         title,
                                         trans,
                                         A,
                                         descr,
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
#if(VERBOSE > 0)
        std::string dtype = "unknown";
        if(typeid(T) == typeid(double))
        {
            dtype = "double";
        }
        else if(typeid(T) == typeid(float))
        {
            dtype = "float";
        }
        const bool        unit    = descr->diag_type == aoclsparse_diag_type_unit;
        aoclsparse_int    kidlabs = std::max<aoclsparse_int>(std::min<aoclsparse_int>(kid, 4), 0);
        const std::string avxlabs[5] = {"NONE (reference)",
                                        "AVX2 (reference 256b)",
                                        "AVX2 (KT 256b)",
                                        "AVX-512 (KT 512b)",
                                        "unknown"};
        std::cout << "Problem id: " << id << " \"" << title << "\"" << std::endl;
        std::cout << "Configuration: <" << dtype << "> unit=" << (unit ? "Unit" : "Non-unit")
                  << " trans=" << (trans == aoclsparse_operation_transpose ? "Yes" : "No")
                  << "   Kernel id=" << kid << " <" << avxlabs[kidlabs] << ">" << std::endl;
#endif
        const aoclsparse_int n = A->n;
        status                 = aoclsparse_trsv_kid<T>(trans, alpha, A, descr, &b[0], &x[0], kid);
        ASSERT_EQ(status, trsv_status)
            << "Test failed with unexpected return from aoclsparse_dtrsv";
        if(status == aoclsparse_status_success)
        {
            if(xtol > 0.0)
                EXPECT_ARR_NEAR(n, x, xref, expected_precision<T>(xtol));
            else
                EXPECT_ARR_NEAR(n, x, xref, expected_precision<T>(10.0));
        }
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

    typedef struct
    {
        linear_system_id id;
        std::string      testname;
        aoclsparse_int   kid;
    } trsv_list_t;

#define ADD_TEST(ID, KID)             \
    {                                 \
        ID, #ID "/kid[" #KID "]", KID \
    }

// Kernel kid=3 fallbacks to kid=2 on non-Zen4 and should not fail
#define ADD_TEST_BATCH(PRE)                                                                       \
    ADD_TEST(PRE##_Lx_aB, 0), ADD_TEST(PRE##_Lx_aB, 1), ADD_TEST(PRE##_Lx_aB, 2),                 \
        ADD_TEST(PRE##_Lx_aB, 3),                                                                 \
                                                                                                  \
        ADD_TEST(PRE##_LL_Ix_aB, 0), ADD_TEST(PRE##_LL_Ix_aB, 1), ADD_TEST(PRE##_LL_Ix_aB, 2),    \
        ADD_TEST(PRE##_LL_Ix_aB, 3),                                                              \
                                                                                                  \
        ADD_TEST(PRE##_LTx_aB, 0), ADD_TEST(PRE##_LTx_aB, 1), ADD_TEST(PRE##_LTx_aB, 2),          \
        ADD_TEST(PRE##_LTx_aB, 3),                                                                \
                                                                                                  \
        ADD_TEST(PRE##_LL_ITx_aB, 0), ADD_TEST(PRE##_LL_ITx_aB, 1), ADD_TEST(PRE##_LL_ITx_aB, 2), \
        ADD_TEST(PRE##_LL_ITx_aB, 3),                                                             \
                                                                                                  \
        ADD_TEST(PRE##_Ux_aB, 0), ADD_TEST(PRE##_Ux_aB, 1), ADD_TEST(PRE##_Ux_aB, 2),             \
        ADD_TEST(PRE##_Ux_aB, 3),                                                                 \
                                                                                                  \
        ADD_TEST(PRE##_UU_Ix_aB, 0), ADD_TEST(PRE##_UU_Ix_aB, 1), ADD_TEST(PRE##_UU_Ix_aB, 2),    \
        ADD_TEST(PRE##_UU_Ix_aB, 3),                                                              \
                                                                                                  \
        ADD_TEST(PRE##_UTx_aB, 0), ADD_TEST(PRE##_UTx_aB, 1), ADD_TEST(PRE##_UTx_aB, 2),          \
        ADD_TEST(PRE##_UTx_aB, 3),                                                                \
                                                                                                  \
        ADD_TEST(PRE##_UU_ITx_aB, 0), ADD_TEST(PRE##_UU_ITx_aB, 1), ADD_TEST(PRE##_UU_ITx_aB, 2), \
        ADD_TEST(PRE##_UU_ITx_aB, 3)

    trsv_list_t trsv_list[] = {ADD_TEST_BATCH(D7),
                               ADD_TEST_BATCH(S7),
                               ADD_TEST_BATCH(N25),
                               ADD_TEST(D7_Lx_aB, 999),
                               ADD_TEST(D7_Lx_aB, -1)};

    void PrintTo(const trsv_list_t &param, ::std::ostream *os)
    {
        *os << param.testname;
    }

    class PosDouble : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosDouble, Solver)
    {
        const linear_system_id id  = GetParam().id;
        const aoclsparse_int   kid = GetParam().kid;
#if(VERBOSE > 0)
        std::cout << "Pos/Double/Solver test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsv_driver<double>(id, kid, aoclsparse_status_success);
    }
    INSTANTIATE_TEST_SUITE_P(TrsvSuite, PosDouble, ::testing::ValuesIn(trsv_list));

    class PosFloat : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosFloat, Solver)
    {
        const linear_system_id  id  = GetParam().id;
        const aoclsparse_int    kid = GetParam().kid;
        const aoclsparse_status trsv_status
            = kid == 1 ? aoclsparse_status_not_implemented : aoclsparse_status_success;
#if(VERBOSE > 0)
        std::cout << "Pos/Float/Solver test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsv_driver<float>(id, kid, trsv_status);
    }
    INSTANTIATE_TEST_SUITE_P(TrsvSuite, PosFloat, ::testing::ValuesIn(trsv_list));

    TEST(TrsvSuite, BaseOne)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        // Change to base-1
        A->base = descr->base = aoclsparse_index_base_one;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_not_implemented);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        // Create matrix data for type double
        ASSERT_EQ(create_matrix<double>(N5_full_sorted, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        // Force it to lower triangular
        descr->fill_mode = aoclsparse_fill_mode_lower;
        descr->type      = aoclsparse_matrix_type_triangular;
        ASSERT_EQ(aoclsparse_set_sv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_strsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_wrong_type);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        // Create matrix data for type double
        ASSERT_EQ(create_matrix<float>(N5_full_sorted, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        // Force it to lower triangular
        descr->fill_mode = aoclsparse_fill_mode_lower;
        descr->type      = aoclsparse_matrix_type_triangular;
        ASSERT_EQ(aoclsparse_set_sv_hint(A, aoclsparse_operation_none, descr, 1),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_optimize(A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_wrong_type);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, nullptr, b, x, kid),
                  aoclsparse_status_invalid_pointer);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, nullptr, kid),
                  aoclsparse_status_invalid_pointer);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, nullptr, x, kid),
                  aoclsparse_status_invalid_pointer);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        A->m = -1;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_size);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        A->nnz = -1;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_size);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    };

    TEST(TrsvSuite, NotImpletemedOpCT)
    {
        double                      alpha = 0.0;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;
        double                      b[1], x[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_conjugate_transpose;
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_not_implemented);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        descr->type = aoclsparse_matrix_type_general;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_value);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        A->n = 0;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_value);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
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
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);
        // Force it to lower triangular
        descr->fill_mode = aoclsparse_fill_mode_lower;
        descr->type      = aoclsparse_matrix_type_triangular;
        ASSERT_EQ(aoclsparse_dtrsv_kid(trans, alpha, A, descr, b, x, kid),
                  aoclsparse_status_invalid_value);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
    };

    template <typename T>
    void trsv_hint_driver(linear_system_id id)
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
        std::array<aoclsparse_int, 10> iparm;
        std::array<T, 10>              dparm;
        aoclsparse_status              status, exp_status;

        status = create_linear_system<T>(id,
                                         title,
                                         trans,
                                         A,
                                         descr,
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
        ASSERT_EQ(aoclsparse_destroy(A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

#define ADD_TEST_HINT(ID) \
    {                     \
        ID, #ID, 0        \
    }
    trsv_list_t trsv_hint_list[] = {ADD_TEST_HINT(D7Lx_aB_hint),
                                    ADD_TEST_HINT(A_nullptr),
                                    ADD_TEST_HINT(D1_descr_nullptr),
                                    ADD_TEST_HINT(D1_neg_num_hint),
                                    ADD_TEST_HINT(D1_mattype_gen_hint)};

    class PosHDouble : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosHDouble, Hint)
    {
        const linear_system_id id = GetParam().id;
#if(VERBOSE > 0)
        std::cout << "PosH/Double test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsv_hint_driver<double>(id);
    }
    INSTANTIATE_TEST_CASE_P(TrsvSuite, PosHDouble, ::testing::ValuesIn(trsv_hint_list));

    class PosHFloat : public testing::TestWithParam<trsv_list_t>
    {
    };
    TEST_P(PosHFloat, Hint)
    {
        const linear_system_id id = GetParam().id;
#if(VERBOSE > 0)
        std::cout << "PosH/Float test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsv_hint_driver<float>(id);
    }
    INSTANTIATE_TEST_CASE_P(TrsvSuite, PosHFloat, ::testing::ValuesIn(trsv_hint_list));
}
// namespace
