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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <vector>

#define NONE 0 // fallback - reference implementation (no AVX)
#define COREAVX 1 // reference implementation AVX2 256b
#define AVX2 2 // AVX2 256b (Milan)
#define AVX512 3 // AVX-512F 512b (Genoa)

// Number of test for TRSV
#define NTEST_TRSV 23 /// <------------------------------------- REMOVE
// Number of test for Hinting for TRSV
#define NTEST_TRSV_HINT (NTEST_TRSV + 5)

namespace {
using namespace std; /// <-------------------- REMOVE

// aoclsparse_?_trsv_kid wrappers
template <typename T>
aoclsparse_status aoclsparse_trsv_kid_wrapper();

template <typename T>
aoclsparse_status aoclsparse_trsv_kid_wrapper(const aoclsparse_operation transpose,
                                              const T                    alpha,
                                              aoclsparse_matrix          A,
                                              const aoclsparse_mat_descr descr,
                                              const T                   *b,
                                              T                         *x,
                                              const aoclsparse_int       kid);

template <>
aoclsparse_status aoclsparse_trsv_kid_wrapper<double>(const aoclsparse_operation transpose,
                                                      const double               alpha,
                                                      aoclsparse_matrix          A,
                                                      const aoclsparse_mat_descr descr,
                                                      const double              *b,
                                                      double                    *x,
                                                      const aoclsparse_int       kid)
{
    return aoclsparse_dtrsv_kid(transpose, alpha, A, descr, b, x, kid);
}

template <>
aoclsparse_status aoclsparse_trsv_kid_wrapper<float>(const aoclsparse_operation transpose,
                                                     const float                alpha,
                                                     aoclsparse_matrix          A,
                                                     const aoclsparse_mat_descr descr,
                                                     const float               *b,
                                                     float                     *x,
                                                     const aoclsparse_int       kid)
{
    return aoclsparse_strsv_kid(transpose, alpha, A, descr, b, x, kid);
}

template <typename T>
void test_aoclsparse_trsv(const aoclsparse_int        testid,
                          const string                testdesc,
                          const T                     alpha,
                          aoclsparse_matrix          &A,
                          const aoclsparse_mat_descr &descr,
                          const T                    *b,
                          T                          *x,
                          const aoclsparse_operation  trans,
                          const aoclsparse_int        kid,
                          const T                    *xref,
                          const T                     tol,
                          const aoclsparse_int        verbose)
{
    aoclsparse_status ret;
    aoclsparse_int    n        = A->n;
    aoclsparse_int    isdouble = std::is_same_v<T, double>;
    if(verbose)
    {
        bool         unit = descr->diag_type == aoclsparse_diag_type_unit;
        const string avxlabs[4]
            = {"NONE (reference)", "AVX2 (reference 256b)", "AVX2 (KT 256b)", "AVX-512 (KT 512b)"};
        const string typelabs[2] = {"Float ", "Double"};
        cout << endl << "TEST #" << testid << " " << testdesc << endl;
        cout << "Configuration: unit=" << (unit ? "UNIT" : "NON-UNIT")
             << " trans=" << (trans == aoclsparse_operation_transpose ? "TRANSPOSE" : "NO")
             << " Kernel ID=\"" << avxlabs[kid] << "\" Type: " << typelabs[isdouble] << endl;
    }
    ret = aoclsparse_trsv_kid_wrapper<T>(trans, alpha, A, descr, b, x, kid);

    if(ret != aoclsparse_status_success)
    {
        cout << "Test failed with unexpected return from aoclsparse_?trsv, status = " << ret
             << endl;
    }
    bool pass = (ret == aoclsparse_status_success);
    if(pass)
    {
        T err = 0.0;
        for(int i = 0; i < n; i++)
            err += (x[i] - xref[i]) * (x[i] - xref[i]);
        err  = sqrt(err);
        pass = err <= tol;
        if(!pass)
        {
            cout << "Test failed with tolerance = " << tol << endl;
            cout << "||x-xref|| = " << err << endl;
        }
    }
    if(verbose || !pass)
    {
        cout << "TEST #" << testid << " : " << (pass ? "PASS" : "FAILED") << endl;
    }
    return pass;
}

bool test_aoclsparse_set_sv_hint(const aoclsparse_int        testid,
                                 const string                testdesc,
                                 aoclsparse_matrix          &A,
                                 const aoclsparse_mat_descr &descr,
                                 const aoclsparse_operation  trans,
                                 const aoclsparse_int        ncalls,
                                 const aoclsparse_status     exp_status,
                                 const aoclsparse_int        verbose)
{
    aoclsparse_status ret;
    if(verbose)
    {
        cout << endl << "TEST #" << testid << " " << testdesc << endl;
    }
    ret       = aoclsparse_set_sv_hint(A, trans, descr, ncalls);
    bool pass = (ret == exp_status);
    if(!pass)
    {
        cout << "Test failed with unexpected return from aoclsparse_set_sv_hint, status = " << ret
             << endl;
        cout << "Expected status = " << exp_status << endl;
    }
    if(verbose || !pass)
    {
        cout << "TEST #" << testid << " : " << (pass ? "PASS" : "FAILED") << endl;
    }
    return pass;
}



// If kid > 0 then force to only test the specified KID, otherwise
// cycle through all of them
template <typename T>
bool trsv_kid_driver(aoclsparse_int &testid, aoclsparse_int kid = -1)
{
    bool                 ok      = true;
    aoclsparse_int       verbose = 1;
    aoclsparse_int       KID; // (AVX2=2) and AVX512=3
    string               title = "unknown";
    T                    alpha;
    aoclsparse_matrix    A;
    aoclsparse_mat_descr descr;
    vector<T>            b;
    vector<T>            x;
    vector<T>            xref;
    T                    xtol;
    aoclsparse_operation trans;
    // permanent storage of matrix data
    vector<T>                 aval;
    vector<aoclsparse_int>    icola;
    vector<aoclsparse_int>    icrowa;
    array<aoclsparse_int, 10> iparm;
    array<T, 10>              dparm;
    aoclsparse_status         exp_status;
    bool                      isdouble(std::is_same_v<T, double>);

    if(testid <= NTEST_TRSV)
    {
        // fail on getting data or gone beyond last testid
        ok = get_data<T>(testid,
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
        if(ok)
        {
            // Set default to AVX2
            KID = AVX2;
#if USE_AVX512
            if(global_context.is_avx512)
                KID = AVX512;
#endif

            // Call test as many times as available kernels
            for(aoclsparse_int ikid = 0; ikid <= KID; ikid++)
            {
                if(kid >= 0)
                    ikid = kid;
                // AVX Reference does not implement float type
                if(!(!isdouble && ikid == 1))
                {
                    ok &= test_aoclsparse_trsv<T>(testid,
                                                  title,
                                                  alpha,
                                                  A,
                                                  descr,
                                                  &b[0],
                                                  &x[0],
                                                  trans,
                                                  ikid,
                                                  &xref[0],
                                                  xtol,
                                                  verbose);
                }
                if(kid >= 0)
                {
                    testid = -1;
                    return ok;
                }
            }
            ++testid;
        }
    }
    else
    {
        testid = -1;
    }
    return ok;
}

template <typename T>
bool trsv_hint_driver(aoclsparse_int &testid)
{
    bool                 ok      = true;
    aoclsparse_int       verbose = 1;
    string               title;
    T                    alpha;
    aoclsparse_matrix    A;
    aoclsparse_mat_descr descr;
    vector<T>            b;
    vector<T>            x;
    vector<T>            xref;
    T                    xtol;
    aoclsparse_operation trans;
    // permanent storage of matrix data
    vector<T>                 aval;
    vector<aoclsparse_int>    icola;
    vector<aoclsparse_int>    icrowa;
    array<aoclsparse_int, 10> iparm;
    array<T, 10>              dparm;
    aoclsparse_status         exp_status;

    if(NTEST_TRSV < testid && testid <= NTEST_TRSV_HINT)
    {
        // fail on getting data or gone beyond last testid
        ok = get_data<T>(testid,
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
        if(ok)
        {
            ok &= test_aoclsparse_set_sv_hint(
                testid, title, A, descr, trans, iparm[0], exp_status, verbose);
            ++testid;
        }
        else
        {
            testid = -1;
        }
    }
    else
    {
        testid = -1;
    }
    return ok;
}

int main(void)
{
    bool           ok     = true;
    aoclsparse_int testid = 0, itest = -1;
    // set this to the id you want to test/debug
    // will run only this test with the given kid >= 0,
    // if kid<0 then all kernels are tested.
    aoclsparse_int debug_testid = -1;
    aoclsparse_int kid          = -1;

    // Read the environment variables to update global variable
    // This function updates the num_threads only once.
    aoclsparse_init_once();

    // Run Tests -> Call driver
    cout << endl;
    cout << "================================================" << endl;
    cout << "   AOCLSPARSE TRSV TESTS" << endl;
    cout << "================================================" << endl;
    if(0 <= debug_testid && debug_testid <= NTEST_TRSV)
        testid = debug_testid;
    do
    {
        itest = testid;
        ok &= trsv_kid_driver<double>(testid, kid);
        testid = itest;
        ok &= trsv_kid_driver<float>(testid, kid);
    } while(testid >= 0 && debug_testid < 0);

    cout << endl;
    cout << "================================================" << endl;
    cout << "   AOCLSPARSE SV_HINT TESTS" << endl;
    cout << "================================================" << endl;
    testid = NTEST_TRSV + 1;
    if(NTEST_TRSV <= debug_testid && debug_testid <= NTEST_TRSV_HINT)
        testid = debug_testid;
    do
    {
        itest = testid;
        ok &= trsv_hint_driver<double>(testid);
        testid = itest;
        ok &= trsv_hint_driver<float>(testid);
    } while(testid >= 0 && debug_testid < 0);

    cout << endl;
    cout << "================================================" << endl;
    cout << "   OVERALL TESTS : " << (ok ? "ALL PASSSED" : "FAILED") << endl;
    cout << "================================================" << endl;

    return (ok ? 0 : 1);
}

} // namespace