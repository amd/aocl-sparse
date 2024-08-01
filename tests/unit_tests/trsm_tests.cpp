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
 * Unit-tests for TRiangular Solver for Multiple RHS (aoclsparse_trsm)
 */
#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
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
    /*  Accepts 1-D xref/b inputs from TRSV problem set and builds 2-D inputs XRef, B
    *   needed for TRSM
    */
    template <typename T>
    void build_trsm_problem(aoclsparse_order order,
                            std::vector<T>  &xref,
                            std::vector<T>  &b,
                            std::vector<T>  &xref_scaled,
                            std::vector<T>  &b_scaled,
                            std::vector<T>  &XRef,
                            std::vector<T>  &B,
                            aoclsparse_int   m,
                            aoclsparse_int   col,
                            aoclsparse_int   ld)
    {
        T scale = 1 >> col;
        //scale each column by a factor depending on column number 'c', such that each column vector of X/B
        //is different
        std::transform(
            xref.begin(), xref.end(), xref_scaled.begin(), [&scale](auto &d) { return d * scale; });
        std::transform(
            b.begin(), b.end(), b_scaled.begin(), [&scale](auto &d) { return d * scale; });
        if(order == aoclsparse_order_column) //col major order
        {
            std::copy(xref_scaled.begin(), xref_scaled.end(), &XRef[col * ld]);
            std::copy(b_scaled.begin(), b_scaled.end(), &B[col * ld]);
        }
        else if(order == aoclsparse_order_row) //row major order
        {
            aoclsparse_sctrs<T>(m, &xref_scaled[0], ld, &XRef[col], 0);
            aoclsparse_sctrs<T>(m, &b_scaled[0], ld, &B[col], 0);
        }
    }
    /*  Unit test driver function for TRSM API. Combinations of real/complex
    *   data types, zero-base/1-base indexing, column-major/row-major,
    *   reference/avx2/avx512 kernel templates, different K's (that defines
    *   no of columns in X/B dense matrices of TRSM) are tested. Inputs range
    *   from TRSV's 1-D inputs that are built into 2-D X/B inputs and the native
    *   2-D xref inputs (2-D xref which is used to compute 'b' using reference spmv
    *   kernel and then build inputs needed as per column major/row major cases)
    *
    */
    template <typename T>
    void trsm_driver(linear_system_id id,
                     aoclsparse_int   kid,
                     aoclsparse_int   base_index,
                     aoclsparse_int   layout,
                     aoclsparse_int   k)
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
        aoclsparse_order               order = (aoclsparse_order)layout;
        aoclsparse_index_base          base  = (aoclsparse_index_base)base_index;

        decltype(std::real(xtol)) tol;

        /*
         * Below parameters are inputs to create_linear_system(), which are needed to build
         * the RHS matrix B[Mxk]
         * iparm[4] = indicates whether X/B dense matrices are row major(0) or column major(1)
         * iparm[5] = indicates the no of columns in X/B dense matrices
         */
        iparm[4] = layout;
        iparm[5] = k;
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
        ASSERT_EQ(status, aoclsparse_status_success)
            << "Error: could not find linear system id " << id << "!";
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
                                        "AVX2 (alias KT 256b)",
                                        "AVX2 (KT 256b)",
                                        "AVX-512 (KT 512b)",
                                        "unknown"};
        std::cout << "Problem id: " << id << " \"" << title << "\"" << std::endl;
        std::cout << "Configuration: <" << dtype << "> unit=" << (unit ? "Unit" : "Non-unit")
                  << " op=" << oplabel << "   Kernel id=" << kid << " <" << avxlabs[kidlabs] << ">"
                  << std::endl;
#endif
        const aoclsparse_int m = A->m, n = A->n;
        aoclsparse_int       ldb, ldx; // leading dimensions of dense matrices B and X
        int                  mm, nn, starting_offset_x, starting_offset_b;
        std::vector<T>       B, X, XRef;
        std::vector<T>       xref_scaled, b_scaled;
        std::vector<T>       wcolxref, wcolb;

        tol = std::real(xtol); // get the tolerance.
        if(tol <= 0)
            tol = 10;
        /*
         *  * For trsv problems, X and B are vectors, so they are "extended" to
         *    m times k rectangular matrices before calling TRSM
         *
         *  * For trsm problems,
         *      1. compute "B(m x k)" dense matrix using reference xref(m x k)
         *      2. Use this B matrix, to call TRSM
         *      3. validate "X" (output of TRSM) against xref(m x k)
         */
        if(id < SM_S7_XB910)
        {
            B.resize(m * k);
            X.resize(m * k);
            XRef.resize(m * k);
            xref_scaled.resize(n);
            b_scaled.resize(m);
            // These problems don't use offsets.
            starting_offset_x = 0;
            starting_offset_b = 0;
            // build inputs for TRSM using trsv problems
            if(order == aoclsparse_order_column) // col major order
            {
                ldb = m;
                ldx = m;
                for(int col = 0; col < k; col++)
                {
                    build_trsm_problem(order, xref, b, xref_scaled, b_scaled, XRef, B, m, col, ldx);
                }
                mm = k;
                nn = ldx;
            }
            else if(order == aoclsparse_order_row) // row major order
            {
                ldb = k;
                ldx = k;
                for(int col = 0; col < k; col++)
                {
                    build_trsm_problem(order, xref, b, xref_scaled, b_scaled, XRef, B, m, col, ldx);
                }
                mm = m;
                nn = k;
            }
        }
        else // TRSM Problems
        {
            wcolxref.resize(n);
            wcolb.resize(m);
            // for strided window problems, starting offset and leading-dimension are needed
            // to know where the input starts
            starting_offset_x = iparm[0];
            ldx               = iparm[1]; // leading dimension
            starting_offset_b = iparm[2];
            ldb               = iparm[3];
            if(order == aoclsparse_order_column) //col major order
            {
                /*
                 *  In case of column major layout, leading dimension ld = m and k = <user_given_column_#>
                 *  dimensions of (X,XRef,B): ld x k
                 *  k, which indicates the no of columns in (X,XRef,B) can vary between 1 to infinity
                 *  This decides the total no of columns, which means that many RHS's to be solved for.
                 *  k comes as an input from trsm_driver() call.
                 *  4 scenarios are tested,
                 *      1. k = 2    (edge case with only 2 columns, a strip rectangular dense matrix)
                 *      2. k = m - 2 = 5 (where k < m)
                 *      3. k = m = 7
                 *      4. k = m + 2 = 9 (where k > m)
                 */
                mm = k;
                nn = m;
            }
            else if(order == aoclsparse_order_row) //row major order
            {
                /*
                    In case of row major layout, leading dimension ld = k and m, where k -> <user_given_column_#>
                    dimensions of (X,XRef,B): m x ld = m x k
                    k, which indicates the no of columns in (X,XRef,B) can vary between 1 to infinity
                    This decides the total no of columns, which means that many RHS's to be solved for.
                    k comes as an input from trsm_driver() call.
                    4 scenarios are tested,
                        1. k = 2    (edge case with only 2 columns, a strip rectangular dense matrix)
                        2. k = m - 2 = 5 (where k < m)
                        3. k = m = 7
                        4. k = m + 2 = 9 (where k > m)
                */
                mm = m;
                nn = k;
            }
            //assign b/x/xref matrices for easier reference later
            B    = std::move(b);
            X    = std::move(x);
            XRef = std::move(xref);
        }
        status = aoclsparse_trsm_kid<T>(trans,
                                        alpha,
                                        A,
                                        descr,
                                        order,
                                        &B[starting_offset_b + 0],
                                        k,
                                        ldb,
                                        &X[starting_offset_x + 0],
                                        ldx,
                                        kid);
        ASSERT_EQ(status, aoclsparse_status_success)
            << "Test failed with unexpected return from aoclsparse_trsm_kid";
        if(status == aoclsparse_status_success)
        {
            auto XRef_start = XRef.cbegin() + starting_offset_x;
            auto X_start    = X.cbegin() + starting_offset_x;
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {

                EXPECT_COMPLEX_MAT_NEAR(
                    mm, nn, ldx, X_start, XRef_start, expected_precision<decltype(tol)>(tol));
            }
            else
            {
                EXPECT_MAT_NEAR(
                    mm, nn, ldx, X_start, XRef_start, expected_precision<decltype(tol)>(tol));
            }
        }
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

#define KD_REF 0
#define KD_AVX2 1
#define KT_AVX2 2
#define KT_AVX512 3
#define KID_M1 -1
#define KID_999 999

#define BASEZERO 0
#define BASEONE 1

#define ROWMAJOR 0
#define COLMAJOR 1
    typedef struct
    {
        linear_system_id id;
        std::string      testname;
        aoclsparse_int   kid;
        aoclsparse_int   base;
        aoclsparse_int   order;
        aoclsparse_int   k;
    } trsm_list_t;

#undef ADD_TEST
#define ADD_TEST(ID, KID, BASE, ORDER, K) \
    ID, #ID "/" #ORDER "/" #KID "/" #BASE "/K=" #K, KID, BASE, ORDER, K

#define ADD_TEST_BATCH(PRE, K)                                       \
    ADD_TEST(PRE##_Lx_aB, KD_REF, BASEONE, ROWMAJOR, K),             \
        ADD_TEST(PRE##_LTx_aB, KD_AVX2, BASEZERO, ROWMAJOR, K),      \
        ADD_TEST(PRE##_LL_ITx_aB, KT_AVX2, BASEONE, ROWMAJOR, K),    \
        ADD_TEST(PRE##_LHx_aB, KT_AVX512, BASEZERO, ROWMAJOR, K),    \
        ADD_TEST(PRE##_LL_IHx_aB, KD_REF, BASEONE, ROWMAJOR, K),     \
        ADD_TEST(PRE##_Ux_aB, KD_AVX2, BASEZERO, ROWMAJOR, K),       \
        ADD_TEST(PRE##_UTx_aB, KT_AVX2, BASEONE, ROWMAJOR, K),       \
        ADD_TEST(PRE##_UU_ITx_aB, KT_AVX512, BASEZERO, ROWMAJOR, K), \
        ADD_TEST(PRE##_UHx_aB, KD_REF, BASEONE, ROWMAJOR, K),        \
        ADD_TEST(PRE##_Lx_aB, KD_AVX2, BASEZERO, COLMAJOR, K),       \
        ADD_TEST(PRE##_LL_Ix_aB, KT_AVX2, BASEONE, COLMAJOR, K),     \
        ADD_TEST(PRE##_LTx_aB, KT_AVX512, BASEZERO, COLMAJOR, K),    \
        ADD_TEST(PRE##_LL_ITx_aB, KD_REF, BASEONE, COLMAJOR, K),     \
        ADD_TEST(PRE##_LHx_aB, KD_AVX2, BASEZERO, COLMAJOR, K),      \
        ADD_TEST(PRE##_Ux_aB, KT_AVX2, BASEONE, COLMAJOR, K),        \
        ADD_TEST(PRE##_UTx_aB, KT_AVX512, BASEZERO, COLMAJOR, K),    \
        ADD_TEST(PRE##_UU_ITx_aB, KD_REF, BASEONE, COLMAJOR, K),     \
        ADD_TEST(PRE##_UHx_aB, KD_AVX2, BASEZERO, COLMAJOR, K)

#define ADD_SM_TEST_BATCH(PRE, K)                                                               \
    ADD_TEST(PRE, KD_REF, BASEZERO, COLMAJOR, K), ADD_TEST(PRE, KD_AVX2, BASEONE, COLMAJOR, K), \
        ADD_TEST(PRE, KT_AVX2, BASEZERO, COLMAJOR, K),                                          \
        ADD_TEST(PRE, KT_AVX512, BASEONE, COLMAJOR, K),                                         \
        ADD_TEST(PRE, KD_REF, BASEZERO, ROWMAJOR, K),                                           \
        ADD_TEST(PRE, KD_AVX2, BASEONE, ROWMAJOR, K),                                           \
        ADD_TEST(PRE, KT_AVX2, BASEZERO, ROWMAJOR, K),                                          \
        ADD_TEST(PRE, KT_AVX512, BASEONE, ROWMAJOR, K)

    trsm_list_t trsm_list[] = {ADD_TEST_BATCH(D7, 5),
                               ADD_TEST_BATCH(D7, 7),
                               ADD_TEST_BATCH(D7, 9),
                               ADD_TEST_BATCH(S7, 5),
                               ADD_TEST_BATCH(S7, 7),
                               ADD_TEST_BATCH(S7, 9),
                               ADD_TEST_BATCH(N25, 23),
                               ADD_TEST_BATCH(N25, 25),
                               ADD_TEST_BATCH(N25, 27),
                               ADD_TEST(D7_Lx_aB, KID_999, BASEZERO, ROWMAJOR, 7),
                               ADD_TEST(D7_Lx_aB, KID_M1, BASEZERO, ROWMAJOR, 7),
                               ADD_SM_TEST_BATCH(SM_S7_XB910, 5),
                               ADD_SM_TEST_BATCH(SM_S7_XB910, 7),
                               ADD_SM_TEST_BATCH(SM_S7_XB910, 9),
                               ADD_SM_TEST_BATCH(SM_S7_XB1716, 5),
                               ADD_SM_TEST_BATCH(SM_S7_XB1716, 7),
                               ADD_SM_TEST_BATCH(SM_S7_XB1716, 9),
                               ADD_TEST(SM_S1X1_XB1X1, KD_REF, BASEZERO, COLMAJOR, 1),
                               ADD_TEST(SM_S1X1_XB1X1, KD_AVX2, BASEONE, ROWMAJOR, 1),
                               ADD_TEST(SM_S1X1_XB1X3, KT_AVX2, BASEZERO, COLMAJOR, 3),
                               ADD_TEST(SM_S1X1_XB1X3, KT_AVX512, BASEONE, ROWMAJOR, 3)};
    void        PrintTo(const trsm_list_t &param, ::std::ostream *os)
    {
        *os << param.testname;
    }
    class PosDouble : public testing::TestWithParam<trsm_list_t>
    {
    };
    TEST_P(PosDouble, Solver)
    {
        const linear_system_id id    = GetParam().id;
        const aoclsparse_int   kid   = GetParam().kid;
        const aoclsparse_int   base  = GetParam().base;
        const aoclsparse_int   order = GetParam().order;
        const aoclsparse_int   k     = GetParam().k;
#if(VERBOSE > 0)
        std::cout << "Pos/Double/Solver test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsm_driver<double>(id, kid, base, order, k);
    }
    INSTANTIATE_TEST_SUITE_P(TrsmSuite, PosDouble, ::testing::ValuesIn(trsm_list));

    class PosFloat : public testing::TestWithParam<trsm_list_t>
    {
    };
    TEST_P(PosFloat, Solver)
    {
        const linear_system_id id    = GetParam().id;
        const aoclsparse_int   kid   = GetParam().kid;
        const aoclsparse_int   base  = GetParam().base;
        const aoclsparse_int   order = GetParam().order;
        const aoclsparse_int   k     = GetParam().k;
#if(VERBOSE > 0)
        std::cout << "Pos/Float/Solver test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        trsm_driver<float>(id, kid, base, order, k);
    }
    INSTANTIATE_TEST_SUITE_P(TrsmSuite, PosFloat, ::testing::ValuesIn(trsm_list));

    class PosCplxDouble : public testing::TestWithParam<trsm_list_t>
    {
    };
    TEST_P(PosCplxDouble, Solver)
    {
        const linear_system_id id    = GetParam().id;
        const aoclsparse_int   kid   = GetParam().kid;
        const aoclsparse_int   base  = GetParam().base;
        const aoclsparse_int   order = GetParam().order;
        const aoclsparse_int   k     = GetParam().k;
#if(VERBOSE > 0)
        std::cout << "Pos/CplxDouble/Solver test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        trsm_driver<std::complex<double>>(id, kid, base, order, k);
    }
    INSTANTIATE_TEST_SUITE_P(TrsmSuite, PosCplxDouble, ::testing::ValuesIn(trsm_list));

    class PosCplxFloat : public testing::TestWithParam<trsm_list_t>
    {
    };
    TEST_P(PosCplxFloat, Solver)
    {
        const linear_system_id id    = GetParam().id;
        const aoclsparse_int   kid   = GetParam().kid;
        const aoclsparse_int   base  = GetParam().base;
        const aoclsparse_int   order = GetParam().order;
        const aoclsparse_int   k     = GetParam().k;
#if(VERBOSE > 0)
        std::cout << "Pos/CplxFloat/Solver test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        trsm_driver<std::complex<float>>(id, kid, base, order, k);
    }
    INSTANTIATE_TEST_SUITE_P(TrsmSuite, PosCplxFloat, ::testing::ValuesIn(trsm_list));

    void test_trsm_invalid(void)
    {
        using T                       = float;
        aoclsparse_status      status = aoclsparse_status_success;
        aoclsparse_operation   trans  = aoclsparse_operation_none;
        const aoclsparse_int   M = 7, N = 7, NNZ = 34;
        float                  alpha{0};
        aoclsparse_matrix_type mattype;
        aoclsparse_fill_mode   fill;
        aoclsparse_diag_type   diag;
        aoclsparse_index_base  base, invalid_base;
        aoclsparse_order       order        = aoclsparse_order_column, invalid_order;
        aoclsparse_int         kid          = KD_REF;
        aoclsparse_int         invalid_fill = 2;
        aoclsparse_status      exp_status   = aoclsparse_status_success;
        float                  X[M * N]{0};
        /* Dense 1-D vector of size 'M x N'*/
        float B[M * N]{0};

        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        mattype = aoclsparse_matrix_type_triangular;
        fill    = aoclsparse_fill_mode_lower;
        diag    = aoclsparse_diag_type_non_unit;
        base    = aoclsparse_index_base_zero;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descr, diag), aoclsparse_status_success);

        aoclsparse_int    csr_row_ptr[M + 1] = {0, 5, 10, 15, 21, 26, 30, 34};
        aoclsparse_int    csr_col_ind[NNZ]   = {0, 1, 4, 5, 6, 0, 1, 2, 3, 5, 1, 2, 3, 4, 6, 0, 2,
                                                3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 2, 3, 5, 2, 3, 4, 6};
        float             csr_val[NNZ]       = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr<T>(&A, base, M, N, NNZ, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        aoclsparse_int ldb, ldx;
        ldb = M;
        ldx = M;

        //Check nullptrs (A)
        exp_status = aoclsparse_status_invalid_pointer;
        status     = aoclsparse_trsm_kid<T>(
            trans, alpha, nullptr, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (A) from aoclsparse_trsm_kid";

        //Check nullptrs (descr)
        status = aoclsparse_trsm_kid<T>(
            trans, alpha, A, nullptr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (descr) from aoclsparse_trsm_kid";

        //Check nullptrs (B)
        status = aoclsparse_trsm_kid<T>(
            trans, alpha, A, descr, order, nullptr, N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (B) from aoclsparse_trsm_kid";

        //Check nullptrs (X)
        status = aoclsparse_trsm_kid<T>(
            trans, alpha, A, descr, order, &B[0], N, ldb, nullptr, ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (X) from aoclsparse_trsm_kid";

        //Check for invalid matrix type
        exp_status = aoclsparse_status_invalid_value;
        mattype    = aoclsparse_matrix_type_general;
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate matrix type from aoclsparse_trsm_kid";

        mattype = aoclsparse_matrix_type_hermitian;
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate matrix type from aoclsparse_trsm_kid";

        mattype = aoclsparse_matrix_type_triangular;
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);

        //Check for invalid fill-mode
        exp_status       = aoclsparse_status_not_implemented;
        descr->fill_mode = (aoclsparse_fill_mode)invalid_fill;
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate fill-mode from aoclsparse_trsm_kid";

        //Check for invalid dense matrix layout
        exp_status    = aoclsparse_status_invalid_value;
        invalid_order = (aoclsparse_order)2;
        //restore fill-mode options
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        status = aoclsparse_trsm_kid<T>(
            trans, alpha, A, descr, invalid_order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate matrix order from aoclsparse_trsm_kid";

        //Check for incompatible base
        exp_status  = aoclsparse_status_invalid_value;
        descr->base = aoclsparse_index_base_one;
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status) << "Test failed to validate compatible base (between A and "
                                         "descriptor) from aoclsparse_trsm_kid";

        exp_status   = aoclsparse_status_invalid_value;
        invalid_base = (aoclsparse_index_base)2;
        descr->base  = invalid_base;
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate descriptor base from aoclsparse_trsm_kid";

        //Restore base options
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        //Check for valid leading dimension
        exp_status = aoclsparse_status_invalid_size;
        status     = aoclsparse_trsm_kid<T>(
            trans, alpha, A, descr, order, &B[0], N, -5 /*ldb < 0*/, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate leading dimensions from aoclsparse_trsm_kid";
        status = aoclsparse_trsm_kid<T>(
            trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], -5 /*ldx < 0*/, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate leading dimensions from aoclsparse_trsm_kid";

        //Check for quick returns m =0, nnz = 0
        exp_status = aoclsparse_status_success;
        A->m       = 0;
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_trsm_kid";
        A->m = M;
        A->n = 0;
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_trsm_kid";
        A->n   = N;
        A->nnz = 0;
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_trsm_kid";
        A->nnz = NNZ;
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], 0, ldb, &X[0], ldx, kid);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_trsm_kid";
        //Tamper with A so it is a rectangular matrix
        ++(A->n);
        exp_status = aoclsparse_status_invalid_size;
        status
            = aoclsparse_trsm_kid<T>(trans, alpha, A, descr, order, &B[0], N, ldb, &X[0], ldx, kid);
        --(A->n);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate correct dimensions from aoclsparse_trsm_kid";
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

    TEST(TrsmNegative, InvalidChecks)
    {
        test_trsm_invalid();
    }
}
// namespace
