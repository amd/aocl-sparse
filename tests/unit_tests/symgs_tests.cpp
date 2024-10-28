/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * Unit-tests for Symmetric Gauss Seidel Preconditioner (aoclsparse_symgs)
 */
#include "aoclsparse.h"
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
    template <typename T>
    aoclsparse_status aoclsparse_symgs_kid(aoclsparse_operation       trans,
                                           aoclsparse_matrix          A,
                                           const aoclsparse_mat_descr descr,
                                           const T                    alpha,
                                           const T                   *b,
                                           T                         *x,
                                           T                         *y,
                                           const aoclsparse_int       kid,
                                           const bool                 fuse_mv);
    template <>
    aoclsparse_status aoclsparse_symgs_kid<float>(aoclsparse_operation       trans,
                                                  aoclsparse_matrix          A,
                                                  const aoclsparse_mat_descr descr,
                                                  const float                alpha,
                                                  const float               *b,
                                                  float                     *x,
                                                  float                     *y,
                                                  const aoclsparse_int       kid,
                                                  const bool                 fuse_mv)
    {
        if(!fuse_mv)
        {
            if(kid >= 0)
                return aoclsparse_ssymgs_kid(trans, A, descr, alpha, b, x, kid);
            else
                return aoclsparse_ssymgs(trans, A, descr, alpha, b, x);
        }
        else
        {
            if(kid >= 0)
                return aoclsparse_ssymgs_mv_kid(trans, A, descr, alpha, b, x, y, kid);
            else
                return aoclsparse_ssymgs_mv(trans, A, descr, alpha, b, x, y);
        }
    }
    template <>
    aoclsparse_status aoclsparse_symgs_kid<double>(aoclsparse_operation       trans,
                                                   aoclsparse_matrix          A,
                                                   const aoclsparse_mat_descr descr,
                                                   const double               alpha,
                                                   const double              *b,
                                                   double                    *x,
                                                   double                    *y,
                                                   const aoclsparse_int       kid,
                                                   const bool                 fuse_mv)
    {
        if(!fuse_mv)
        {
            if(kid >= 0)
                return aoclsparse_dsymgs_kid(trans, A, descr, alpha, b, x, kid);
            else
                return aoclsparse_dsymgs(trans, A, descr, alpha, b, x);
        }
        else
        {
            if(kid >= 0)
                return aoclsparse_dsymgs_mv_kid(trans, A, descr, alpha, b, x, y, kid);
            else
                return aoclsparse_dsymgs_mv(trans, A, descr, alpha, b, x, y);
        }
    }
    template <>
    aoclsparse_status aoclsparse_symgs_kid<std::complex<float>>(aoclsparse_operation       trans,
                                                                aoclsparse_matrix          A,
                                                                const aoclsparse_mat_descr descr,
                                                                const std::complex<float>  alpha,
                                                                const std::complex<float> *b,
                                                                std::complex<float>       *x,
                                                                std::complex<float>       *y,
                                                                const aoclsparse_int       kid,
                                                                const bool                 fuse_mv)
    {
        const aoclsparse_float_complex *palpha
            = reinterpret_cast<const aoclsparse_float_complex *>(&alpha);
        const aoclsparse_float_complex *pb = reinterpret_cast<const aoclsparse_float_complex *>(b);
        aoclsparse_float_complex       *px = reinterpret_cast<aoclsparse_float_complex *>(x);
        aoclsparse_float_complex       *py = reinterpret_cast<aoclsparse_float_complex *>(y);
        if(!fuse_mv)
        {
            if(kid >= 0)
                return aoclsparse_csymgs_kid(trans, A, descr, *palpha, pb, px, kid);
            else
                return aoclsparse_csymgs(trans, A, descr, *palpha, pb, px);
        }
        else
        {
            if(kid >= 0)
                return aoclsparse_csymgs_mv_kid(trans, A, descr, *palpha, pb, px, py, kid);
            else
                return aoclsparse_csymgs_mv(trans, A, descr, *palpha, pb, px, py);
        }
    }
    template <>
    aoclsparse_status aoclsparse_symgs_kid<std::complex<double>>(aoclsparse_operation        trans,
                                                                 aoclsparse_matrix           A,
                                                                 const aoclsparse_mat_descr  descr,
                                                                 const std::complex<double>  alpha,
                                                                 const std::complex<double> *b,
                                                                 std::complex<double>       *x,
                                                                 std::complex<double>       *y,
                                                                 const aoclsparse_int        kid,
                                                                 const bool fuse_mv)
    {
        const aoclsparse_double_complex *palpha
            = reinterpret_cast<const aoclsparse_double_complex *>(&alpha);
        const aoclsparse_double_complex *pb
            = reinterpret_cast<const aoclsparse_double_complex *>(b);
        aoclsparse_double_complex *px = reinterpret_cast<aoclsparse_double_complex *>(x);
        aoclsparse_double_complex *py = reinterpret_cast<aoclsparse_double_complex *>(y);
        if(!fuse_mv)
        {
            if(kid >= 0)
                return aoclsparse_zsymgs_kid(trans, A, descr, *palpha, pb, px, kid);
            else
                return aoclsparse_zsymgs(trans, A, descr, *palpha, pb, px);
        }
        else
        {
            if(kid >= 0)
                return aoclsparse_zsymgs_mv_kid(trans, A, descr, *palpha, pb, px, py, kid);
            else
                return aoclsparse_zsymgs_mv(trans, A, descr, *palpha, pb, px, py);
        }
    }
    // driver to invoke symgs and mv-fused-symgs kernels based on the macro (ADD_TEST) and
    // linear system database that feeds various input data params
    template <typename T>
    void symgs_driver(linear_system_id  id, // identifier from linear system database
                      aoclsparse_int    fuse_mv, //flag to indicate symgs or symgs_mv kernel,
                      aoclsparse_int    kid, // kernel id
                      aoclsparse_int    base_index, //zero base or one-base
                      aoclsparse_int    iters, //no of iterations to run the kernel
                      aoclsparse_int    mtype, //matrix type
                      aoclsparse_int    fmode, //fill mode
                      aoclsparse_int    transp, // flag to indicate transposition operation
                      aoclsparse_status symgs_status = aoclsparse_status_success) //expected status
    {
        aoclsparse_status    status;
        std::string          title;
        T                    alpha;
        aoclsparse_matrix    A     = nullptr;
        aoclsparse_mat_descr descr = nullptr;
        std::vector<T>       b;
        std::vector<T>       x;
        std::vector<T>       xref, symgs_x;
        T                    xtol;
        aoclsparse_operation trans;
        // permanent storage of matrix data
        std::vector<T>                 aval;
        std::vector<aoclsparse_int>    icola;
        std::vector<aoclsparse_int>    icrowa;
        std::array<aoclsparse_int, 10> iparm;
        std::array<T, 10>              dparm;
        aoclsparse_status              exp_status;
        aoclsparse_index_base          base = (aoclsparse_index_base)base_index;

        decltype(std::real(xtol)) tol;

        /*
         * Below parameters are inputs to create_linear_system(), which are needed to populate correct reference vectors
         * iparm[6] = matrix type
         * iparm[7] = fill mode
         * iparm[8] = tranpose mode
         */
        iparm[6] = mtype;
        iparm[7] = fmode;
        iparm[8] = transp;
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
        std::string kerlabel{""};
        if(fuse_mv == 0)
            kerlabel = "SYMGS";
        else if(fuse_mv == 1)
            kerlabel = "SYMGS_MV";
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
            oplabel = "Conjugate Transpose";
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
                                        "AVX2 (reference 256b)",
                                        "AVX2 (KT 256b)",
                                        "AVX-512 (KT 512b)",
                                        "unknown"};
        std::cout << "Problem id: " << id << " \"" << kerlabel << " \"" << title << "\""
                  << std::endl;
        std::cout << "Configuration: <" << dtype << "> unit=" << (unit ? "Unit" : "Non-unit")
                  << " op=" << oplabel << "   Kernel id=" << kid << " <" << avxlabs[kidlabs] << ">"
                  << std::endl;
#endif
        const aoclsparse_int m = A->m;
        std::vector<T>       y;
        y.resize(m);
        tol = std::real(xtol); // get the tolerance.
        if(tol <= 0)
            tol = 10;

        for(aoclsparse_int i = 0; i < iters; i++)
        {
            status = aoclsparse_symgs_kid<T>(
                trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, fuse_mv);
            ASSERT_EQ(status, symgs_status)
                << "Test failed with unexpected return from aoclsparse_symgs_kid";
        }

        if(status == aoclsparse_status_success)
        {
            if(fuse_mv == 0)
            {
                //validate symgs output
                symgs_x = std::move(x);
            }
            else if(fuse_mv == 1)
            {
                //validate symgs_mv output
                symgs_x = std::move(y);
            }
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                EXPECT_COMPLEX_ARR_NEAR(m, symgs_x, xref, expected_precision<decltype(tol)>(tol));
            }
            else
            {
                EXPECT_ARR_NEAR(m, symgs_x, xref, expected_precision<decltype(tol)>(tol));
            }
        }
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

#undef KD_REF
#undef KD_AVX2
#undef KT_AVX2
#undef KT_AVX512
#undef KID_M1
#undef KID_999
#undef BASEZERO
#undef BASEONE

#define KD_REF 0
#define KD_AVX2 1
#define KT_AVX2 2
#define KT_AVX512 3
#define KID_M1 -1
#define KID_999 999

#define BASEZERO 0
#define BASEONE 1
#define LOWER 0
#define UPPER 1

#define GENERAL 0
#define SYMMETRIC 1
#define HERMIT 2
#define TRIANG 3

#define NONTRANS 0
#define TRANS 1

#define DISABLED 0
#define ENABLED 1

    typedef struct
    {
        linear_system_id id;
        std::string      testname;
        aoclsparse_int   fuse_mv;
        aoclsparse_int   kid;
        aoclsparse_int   base;
        aoclsparse_int   iters;
        aoclsparse_int   matrix_type;
        aoclsparse_int   fill_mode;
        aoclsparse_int   transp;
    } symgs_list_t;

#undef ADD_TEST
#define ADD_TEST(ID, FUSE_MV, KID, BASE, ITERS, MTYPE, FMODE, TRANSP)                           \
    {                                                                                           \
        ID, #ID "/" #KID "/" #BASE "/ITERS=" #ITERS "/" #MTYPE "/" #FMODE "/" #TRANSP, FUSE_MV, \
            KID, BASE, ITERS, MTYPE, FMODE, TRANSP                                              \
    }

    symgs_list_t symgs_list[]
        = {ADD_TEST(GS_S7, DISABLED, KD_REF, BASEZERO, 1, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_S7, DISABLED, KID_M1, BASEZERO, 1, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_S7, DISABLED, KD_REF, BASEZERO, 1, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_S7, DISABLED, KID_M1, BASEZERO, 1, SYMMETRIC, UPPER, TRANS),
           ADD_TEST(GS_MV_S7, ENABLED, KD_REF, BASEONE, 1, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_MV_S7, ENABLED, KID_M1, BASEONE, 1, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_MV_S7, ENABLED, KD_REF, BASEONE, 1, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_MV_S7, ENABLED, KID_M1, BASEONE, 1, SYMMETRIC, UPPER, TRANS),

           ADD_TEST(GS_TRIDIAG_M5, DISABLED, KD_REF, BASEZERO, 1, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_TRIDIAG_M5, DISABLED, KID_M1, BASEZERO, 1, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_TRIDIAG_M5, DISABLED, KD_REF, BASEZERO, 1, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_TRIDIAG_M5, DISABLED, KID_M1, BASEZERO, 1, SYMMETRIC, UPPER, TRANS),
           ADD_TEST(GS_MV_TRIDIAG_M5, ENABLED, KD_REF, BASEONE, 1, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_MV_TRIDIAG_M5, ENABLED, KID_M1, BASEONE, 1, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_MV_TRIDIAG_M5, ENABLED, KD_REF, BASEONE, 1, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_MV_TRIDIAG_M5, ENABLED, KID_M1, BASEONE, 1, SYMMETRIC, UPPER, TRANS),

           ADD_TEST(GS_BLOCK_TRDIAG_S9, DISABLED, KD_REF, BASEZERO, 1, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_BLOCK_TRDIAG_S9, DISABLED, KID_M1, BASEZERO, 1, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_BLOCK_TRDIAG_S9, DISABLED, KD_REF, BASEZERO, 1, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_BLOCK_TRDIAG_S9, DISABLED, KID_M1, BASEZERO, 1, SYMMETRIC, UPPER, TRANS),
           ADD_TEST(GS_MV_BLOCK_TRDIAG_S9, ENABLED, KD_REF, BASEONE, 1, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_MV_BLOCK_TRDIAG_S9, ENABLED, KID_M1, BASEONE, 1, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_MV_BLOCK_TRDIAG_S9, ENABLED, KD_REF, BASEONE, 1, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_MV_BLOCK_TRDIAG_S9, ENABLED, KID_M1, BASEONE, 1, SYMMETRIC, UPPER, TRANS),

           ADD_TEST(GS_CONVERGE_S4, DISABLED, KD_REF, BASEZERO, 8, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_CONVERGE_S4, DISABLED, KID_M1, BASEZERO, 8, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_CONVERGE_S4, DISABLED, KD_REF, BASEZERO, 8, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_CONVERGE_S4, DISABLED, KID_M1, BASEZERO, 8, SYMMETRIC, UPPER, TRANS),
           ADD_TEST(GS_MV_CONVERGE_S4, ENABLED, KD_REF, BASEONE, 8, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_MV_CONVERGE_S4, ENABLED, KID_M1, BASEONE, 8, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_MV_CONVERGE_S4, ENABLED, KD_REF, BASEONE, 8, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_MV_CONVERGE_S4, ENABLED, KID_M1, BASEONE, 8, SYMMETRIC, UPPER, TRANS),

           ADD_TEST(GS_NONSYM_S4, DISABLED, KD_REF, BASEZERO, 1, GENERAL, LOWER, NONTRANS),
           ADD_TEST(GS_NONSYM_S4, DISABLED, KID_M1, BASEZERO, 1, GENERAL, LOWER, TRANS),
           ADD_TEST(GS_NONSYM_S4, DISABLED, KD_REF, BASEZERO, 1, GENERAL, UPPER, NONTRANS),
           ADD_TEST(GS_NONSYM_S4, DISABLED, KID_M1, BASEZERO, 1, GENERAL, UPPER, TRANS),
           ADD_TEST(GS_MV_NONSYM_S4, ENABLED, KD_REF, BASEONE, 1, GENERAL, LOWER, NONTRANS),
           ADD_TEST(GS_MV_NONSYM_S4, ENABLED, KID_M1, BASEONE, 1, GENERAL, LOWER, TRANS),
           ADD_TEST(GS_MV_NONSYM_S4, ENABLED, KD_REF, BASEONE, 1, GENERAL, UPPER, NONTRANS),
           ADD_TEST(GS_MV_NONSYM_S4, ENABLED, KID_M1, BASEONE, 1, GENERAL, UPPER, TRANS),

           ADD_TEST(GS_HR4, DISABLED, KD_REF, BASEZERO, 1, HERMIT, LOWER, NONTRANS),
           ADD_TEST(GS_HR4, DISABLED, KID_M1, BASEZERO, 1, HERMIT, LOWER, TRANS),
           ADD_TEST(GS_HR4, DISABLED, KD_REF, BASEZERO, 1, HERMIT, UPPER, NONTRANS),
           ADD_TEST(GS_HR4, DISABLED, KID_M1, BASEZERO, 1, HERMIT, UPPER, TRANS),
           ADD_TEST(GS_MV_HR4, ENABLED, KD_REF, BASEONE, 1, HERMIT, LOWER, NONTRANS),
           ADD_TEST(GS_MV_HR4, ENABLED, KID_M1, BASEONE, 1, HERMIT, LOWER, TRANS),
           ADD_TEST(GS_MV_HR4, ENABLED, KD_REF, BASEONE, 1, HERMIT, UPPER, NONTRANS),
           ADD_TEST(GS_MV_HR4, ENABLED, KID_M1, BASEONE, 1, HERMIT, UPPER, TRANS),

           ADD_TEST(GS_TRIANGLE_S5, DISABLED, KD_REF, BASEZERO, 1, TRIANG, LOWER, NONTRANS),
           ADD_TEST(GS_TRIANGLE_S5, DISABLED, KID_M1, BASEZERO, 1, TRIANG, LOWER, TRANS),
           ADD_TEST(GS_TRIANGLE_S5, DISABLED, KD_REF, BASEZERO, 1, TRIANG, UPPER, NONTRANS),
           ADD_TEST(GS_TRIANGLE_S5, DISABLED, KID_M1, BASEZERO, 1, TRIANG, UPPER, TRANS),
           ADD_TEST(GS_MV_TRIANGLE_S5, ENABLED, KD_REF, BASEONE, 1, TRIANG, LOWER, NONTRANS),
           ADD_TEST(GS_MV_TRIANGLE_S5, ENABLED, KID_M1, BASEONE, 1, TRIANG, LOWER, TRANS),
           ADD_TEST(GS_MV_TRIANGLE_S5, ENABLED, KD_REF, BASEONE, 1, TRIANG, UPPER, NONTRANS),
           ADD_TEST(GS_MV_TRIANGLE_S5, ENABLED, KID_M1, BASEONE, 1, TRIANG, UPPER, TRANS),

           ADD_TEST(GS_SYMM_ALPHA2_S9, DISABLED, KD_REF, BASEZERO, 1, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_SYMM_ALPHA2_S9, DISABLED, KID_M1, BASEZERO, 1, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_SYMM_ALPHA2_S9, DISABLED, KD_REF, BASEZERO, 1, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_SYMM_ALPHA2_S9, DISABLED, KID_M1, BASEZERO, 1, SYMMETRIC, UPPER, TRANS),
           ADD_TEST(GS_MV_SYMM_ALPHA2_S9, ENABLED, KD_REF, BASEONE, 1, SYMMETRIC, LOWER, NONTRANS),
           ADD_TEST(GS_MV_SYMM_ALPHA2_S9, ENABLED, KID_M1, BASEONE, 1, SYMMETRIC, LOWER, TRANS),
           ADD_TEST(GS_MV_SYMM_ALPHA2_S9, ENABLED, KD_REF, BASEONE, 1, SYMMETRIC, UPPER, NONTRANS),
           ADD_TEST(GS_MV_SYMM_ALPHA2_S9, ENABLED, KID_M1, BASEONE, 1, SYMMETRIC, UPPER, TRANS)};
    void PrintTo(const symgs_list_t &param, ::std::ostream *os)
    {
        *os << param.testname;
    }
    class PosDouble : public testing::TestWithParam<symgs_list_t>
    {
    };
    TEST_P(PosDouble, Solver)
    {
        const linear_system_id id      = GetParam().id;
        const aoclsparse_int   fuse_mv = GetParam().fuse_mv;
        const aoclsparse_int   kid     = GetParam().kid;
        const aoclsparse_int   base    = GetParam().base;
        const aoclsparse_int   iters   = GetParam().iters;

        aoclsparse_int          mtype        = GetParam().matrix_type;
        const aoclsparse_int    fmode        = GetParam().fill_mode;
        aoclsparse_int          transp       = GetParam().transp;
        const aoclsparse_status symgs_status = aoclsparse_status_success;

        //database populates real only data when double or float variants are called, so
        //force matrix type to symmetric since the data that is fed is symmetric
        if(mtype == aoclsparse_matrix_type_hermitian)
        {
            mtype = SYMMETRIC;
        }

        if(transp == NONTRANS)
        {
            transp = (aoclsparse_int)aoclsparse_operation_none;
        }
        else if(transp == TRANS)
        {
            transp = (aoclsparse_int)aoclsparse_operation_transpose;
        }
#if(VERBOSE > 0)
        std::cout << "Pos/Double/Solver test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        symgs_driver<double>(id, fuse_mv, kid, base, iters, mtype, fmode, transp, symgs_status);
    }
    INSTANTIATE_TEST_SUITE_P(SymgsSuite, PosDouble, ::testing::ValuesIn(symgs_list));

    class PosFloat : public testing::TestWithParam<symgs_list_t>
    {
    };
    TEST_P(PosFloat, Solver)
    {
        const linear_system_id id      = GetParam().id;
        const aoclsparse_int   fuse_mv = GetParam().fuse_mv;
        const aoclsparse_int   kid     = GetParam().kid;
        const aoclsparse_int   base    = GetParam().base;
        const aoclsparse_int   iters   = GetParam().iters;

        aoclsparse_int          mtype        = GetParam().matrix_type;
        const aoclsparse_int    fmode        = GetParam().fill_mode;
        aoclsparse_int          transp       = GetParam().transp;
        const aoclsparse_status symgs_status = aoclsparse_status_success;

        //database populates real only data when Double or Float variants are called, so
        //force matrix type to symmetric since the data thatis fed is symmetric
        if(mtype == aoclsparse_matrix_type_hermitian)
        {
            mtype = SYMMETRIC;
        }

        if(transp == NONTRANS)
        {
            transp = (aoclsparse_int)aoclsparse_operation_none;
        }
        else if(transp == TRANS)
        {
            transp = (aoclsparse_int)aoclsparse_operation_transpose;
        }
#if(VERBOSE > 0)
        std::cout << "Pos/Float/Solver test name: \"" << GetParam().testname << "\"" << std::endl;
#endif
        symgs_driver<float>(id, fuse_mv, kid, base, iters, mtype, fmode, transp, symgs_status);
    }
    INSTANTIATE_TEST_SUITE_P(SymgsSuite, PosFloat, ::testing::ValuesIn(symgs_list));

    class PosCplxDouble : public testing::TestWithParam<symgs_list_t>
    {
    };
    TEST_P(PosCplxDouble, Solver)
    {
        const linear_system_id  id           = GetParam().id;
        const aoclsparse_int    fuse_mv      = GetParam().fuse_mv;
        const aoclsparse_int    kid          = GetParam().kid;
        const aoclsparse_int    base         = GetParam().base;
        const aoclsparse_int    iters        = GetParam().iters;
        const aoclsparse_int    mtype        = GetParam().matrix_type;
        const aoclsparse_int    fmode        = GetParam().fill_mode;
        aoclsparse_int          transp       = GetParam().transp;
        const aoclsparse_status symgs_status = aoclsparse_status_success;

        if(transp == NONTRANS)
        {
            transp = (aoclsparse_int)aoclsparse_operation_none;
        }
        else if(transp == TRANS)
        {
            //check for complex-conjugate tranpose, if matrix type is hermitian, else simple transpose
            if(mtype == aoclsparse_matrix_type_hermitian)
            {
                transp = (aoclsparse_int)aoclsparse_operation_conjugate_transpose;
            }
            else
            {
                transp = (aoclsparse_int)aoclsparse_operation_transpose;
            }
        }
#if(VERBOSE > 0)
        std::cout << "Pos/CplxDouble/Solver test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        symgs_driver<std::complex<double>>(
            id, fuse_mv, kid, base, iters, mtype, fmode, transp, symgs_status);
    }
    INSTANTIATE_TEST_SUITE_P(SymgsSuite, PosCplxDouble, ::testing::ValuesIn(symgs_list));

    class PosCplxFloat : public testing::TestWithParam<symgs_list_t>
    {
    };
    TEST_P(PosCplxFloat, Solver)
    {
        const linear_system_id id      = GetParam().id;
        const aoclsparse_int   fuse_mv = GetParam().fuse_mv;
        const aoclsparse_int   kid     = GetParam().kid;
        const aoclsparse_int   base    = GetParam().base;
        const aoclsparse_int   iters   = GetParam().iters;

        const aoclsparse_int    mtype        = GetParam().matrix_type;
        const aoclsparse_int    fmode        = GetParam().fill_mode;
        aoclsparse_int          transp       = GetParam().transp;
        const aoclsparse_status symgs_status = aoclsparse_status_success;

        if(transp == NONTRANS)
        {
            transp = (aoclsparse_int)aoclsparse_operation_none;
        }
        else if(transp == TRANS)
        {
            //check for complex-conjugate tranpose, if matrix type is hermitian, else simple transpose
            if(mtype == aoclsparse_matrix_type_hermitian)
            {
                transp = (aoclsparse_int)aoclsparse_operation_conjugate_transpose;
            }
            else
            {
                transp = (aoclsparse_int)aoclsparse_operation_transpose;
            }
        }
#if(VERBOSE > 0)
        std::cout << "Pos/CplxFloat/Solver test name: \"" << GetParam().testname << "\""
                  << std::endl;
#endif
        symgs_driver<std::complex<float>>(
            id, fuse_mv, kid, base, iters, mtype, fmode, transp, symgs_status);
    }
    INSTANTIATE_TEST_SUITE_P(SymgsSuite, PosCplxFloat, ::testing::ValuesIn(symgs_list));

    void test_symgs_invalid(void)
    {
        using T                       = float;
        aoclsparse_status      status = aoclsparse_status_success;
        aoclsparse_operation   trans  = aoclsparse_operation_none;
        const aoclsparse_int   m = 7, n = 7, nnz = 34;
        float                  alpha{0};
        aoclsparse_matrix_type mattype;
        aoclsparse_fill_mode   fill;
        aoclsparse_diag_type   diag;
        aoclsparse_index_base  base, invalid_base;
        aoclsparse_int         kid          = KD_REF;
        aoclsparse_int         invalid_fill = 2;
        aoclsparse_status      exp_status   = aoclsparse_status_success;
        float                  x[m]{0};
        float                  y[m]{0};
        /* Dense 1-D vector of size 'M x N'*/
        float b[m]{0};

        aoclsparse_mat_descr descr;
        aoclsparse_create_mat_descr(&descr);
        mattype = aoclsparse_matrix_type_symmetric;
        fill    = aoclsparse_fill_mode_lower;
        diag    = aoclsparse_diag_type_non_unit;
        base    = aoclsparse_index_base_zero;
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_diag_type(descr, diag), aoclsparse_status_success);

        aoclsparse_int    csr_row_ptr[m + 1] = {0, 5, 10, 15, 21, 26, 30, 34};
        aoclsparse_int    csr_col_ind[nnz]   = {0, 1, 4, 5, 6, 0, 1, 2, 3, 5, 1, 2, 3, 4, 6, 0, 2,
                                                3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 2, 3, 5, 2, 3, 4, 6};
        float             csr_val[nnz]       = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aoclsparse_matrix A;
        ASSERT_EQ(aoclsparse_create_csr<T>(&A, base, m, n, nnz, csr_row_ptr, csr_col_ind, csr_val),
                  aoclsparse_status_success);

        //Check nullptrs (A)
        exp_status = aoclsparse_status_invalid_pointer;
        status     = aoclsparse_symgs_kid<T>(
            trans, nullptr, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (A) from aoclsparse_symgs_kid";
        status
            = aoclsparse_symgs_kid<T>(trans, nullptr, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (A) from aoclsparse_symgs_mv_kid";

        //Check nullptrs (descr)
        status = aoclsparse_symgs_kid<T>(trans, A, nullptr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (A) from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, nullptr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (A) from aoclsparse_symgs_mv_kid";

        //Check nullptrs (b)
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, nullptr, &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (b) from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, nullptr, &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (b) from aoclsparse_symgs_mv_kid";

        //Check nullptrs (x)
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], nullptr, &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (x) from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], nullptr, &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to check nullptr (x) from aoclsparse_symgs_mv_kid";

        //Check for invalid matrix type
        exp_status  = aoclsparse_status_invalid_value;
        descr->type = (aoclsparse_matrix_type)4;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate invalid matrix type from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate invalid matrix type from aoclsparse_symgs_mv_kid";

        ASSERT_EQ(aoclsparse_set_mat_type(descr, mattype), aoclsparse_status_success);

        //Check for invalid fill-mode
        exp_status       = aoclsparse_status_invalid_value;
        descr->fill_mode = (aoclsparse_fill_mode)invalid_fill;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate fill-mode from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate fill-mode from aoclsparse_symgs_mv_kid";
        //restore fill-mode options
        ASSERT_EQ(aoclsparse_set_mat_fill_mode(descr, fill), aoclsparse_status_success);

        //Check for invalid transpose
        exp_status = aoclsparse_status_invalid_value;
        status     = aoclsparse_symgs_kid<T>(
            (aoclsparse_operation)114, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate invalid transpose from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(
            (aoclsparse_operation)114, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate invalid transpose from aoclsparse_symgs_mv_kid";

        //Check for invalid storage format
        exp_status      = aoclsparse_status_not_implemented;
        A->input_format = aoclsparse_ell_mat;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate invalid storage format from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate invalid storage format from aoclsparse_symgs_mv_kid";
        //restore storage format
        A->input_format = aoclsparse_csr_mat;

        //Check for unit diagonal type requesting kernel to ignore and not-process diagonal elements
        exp_status       = aoclsparse_status_not_implemented;
        descr->diag_type = aoclsparse_diag_type_unit;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate unit-diagonal type from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate unit-diagonal type from aoclsparse_symgs_mv_kid";
        //restore diagonal type
        descr->diag_type = aoclsparse_diag_type_non_unit;

        //Check if general matrix is proccessed for conjugate transpose operation
        exp_status  = aoclsparse_status_not_implemented;
        trans       = aoclsparse_operation_conjugate_transpose;
        descr->type = aoclsparse_matrix_type_general;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status) << "Test failed to validate conjugate transpose operation on "
                                         "general matrix from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status) << "Test failed to validate conjugate transpose operation on "
                                         "general matrix from aoclsparse_symgs_mv_kid";
        //restore matrix type and transpose oepration
        descr->type = aoclsparse_matrix_type_symmetric;
        trans       = aoclsparse_operation_none;

        //Check for incompatible base
        exp_status  = aoclsparse_status_invalid_value;
        descr->base = aoclsparse_index_base_one;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status) << "Test failed to validate compatible base (between A and "
                                         "descriptor) from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status) << "Test failed to validate compatible base (between A and "
                                         "descriptor) from aoclsparse_symgs_mv_kid";

        exp_status   = aoclsparse_status_invalid_value;
        invalid_base = (aoclsparse_index_base)2;
        descr->base  = invalid_base;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate descriptor base from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate descriptor base from aoclsparse_symgs_mv_kid";

        //Restore base options
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, base), aoclsparse_status_success);

        // check for wrong data type
        if(std::is_same_v<T, double>)
        {
            A->val_type = aoclsparse_smat;
        }
        else if(std::is_same_v<T, float>)
        {
            A->val_type = aoclsparse_cmat;
        }
        else if(std::is_same_v<T, aoclsparse_float_complex>)
        {
            A->val_type = aoclsparse_zmat;
        }
        else if(std::is_same_v<T, aoclsparse_double_complex>)
        {
            A->val_type = aoclsparse_dmat;
        }
        exp_status = aoclsparse_status_wrong_type;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate data type from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate data type from aoclsparse_symgs_mv_kid";

        //Restore data type
        if(std::is_same_v<T, double>)
        {
            A->val_type = aoclsparse_dmat;
        }
        else if(std::is_same_v<T, float>)
        {
            A->val_type = aoclsparse_smat;
        }
        else if(std::is_same_v<T, aoclsparse_float_complex>)
        {
            A->val_type = aoclsparse_cmat;
        }
        else if(std::is_same_v<T, aoclsparse_double_complex>)
        {
            A->val_type = aoclsparse_zmat;
        }

        //Check for quick returns m =0, nnz = 0
        exp_status = aoclsparse_status_success;
        A->m       = 0;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_symgs_mv_kid";
        A->m   = m;
        A->n   = 0;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_symgs_mv_kid";
        A->n   = n;
        A->nnz = 0;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate zero-size dimensions from aoclsparse_symgs_mv_kid";
        A->nnz = nnz;

        //Check for negative m, n, nnz
        A->n       = -7;
        exp_status = aoclsparse_status_invalid_size;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate negative dimensions(n) from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate negative dimensions(n) from aoclsparse_symgs_mv_kid";

        A->n       = n;
        A->m       = -7;
        exp_status = aoclsparse_status_invalid_size;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate negative dimensions(m) from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate negative dimensions(m) from aoclsparse_symgs_mv_kid";

        A->n       = n;
        A->m       = m;
        A->nnz     = -34;
        exp_status = aoclsparse_status_invalid_size;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate negative dimensions(nnz) from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate negative dimensions(nnz) from aoclsparse_symgs_mv_kid";
        A->nnz = nnz;

        //Tamper with A so it is a rectangular matrix
        ++(A->n);
        exp_status = aoclsparse_status_invalid_size;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate correct dimensions from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate correct dimensions from aoclsparse_symgs_mv_kid";
        --(A->n);

        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
    }

    TEST(SymgsAndSymgsMvNegative, InvalidChecks)
    {
        test_symgs_invalid();
    }
    TEST(SymgsAndSymgsMvNegative, GeneralIncompleteRank)
    {
        using T                            = double;
        aoclsparse_status           status = aoclsparse_status_success, exp_status;
        double                      alpha  = 0.0;
        aoclsparse_matrix           A      = nullptr;
        aoclsparse_mat_descr        descr  = nullptr;
        double                      b[1], x[1], y[1];
        aoclsparse_int              kid = 0, m, n, nnz;
        std::vector<double>         aval;
        std::vector<aoclsparse_int> acol, acrow;
        aoclsparse_operation        trans = aoclsparse_operation_none;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        ASSERT_EQ(create_matrix(N5_1_hole, m, n, nnz, acrow, acol, aval, A, descr, 0),
                  aoclsparse_status_success);

        descr->fill_mode = aoclsparse_fill_mode_lower;
        descr->type      = aoclsparse_matrix_type_symmetric;
        exp_status       = aoclsparse_status_invalid_value;
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, false);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate incomplete rank matrix from aoclsparse_symgs_kid";
        status = aoclsparse_symgs_kid<T>(trans, A, descr, alpha, &b[0], &x[0], &y[0], kid, true);
        ASSERT_EQ(status, exp_status)
            << "Test failed to validate incomplete rank matrix from aoclsparse_symgs_mv_kid";

        ASSERT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    };
} // namespace
