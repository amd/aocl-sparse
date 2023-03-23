/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/* TRSV tests
 * These tests exersize the available kernels the triangular solver aoclsparse_trsv
 * Currently the test suite is only checking for functionality
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
#define NTEST_TRSV 23
// Number of test for Hinting for TRSV
#define NTEST_TRSV_HINT (NTEST_TRSV + 5)

template <typename T>
struct XTOL
{
    constexpr operator T() const noexcept
    {
        return std::sqrt((T)2.0 * std::numeric_limits<T>::epsilon());
    }
};

// ?trsv_kid wrapper
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

using namespace std;

template <typename T>
bool test_aoclsparse_trsv(const aoclsparse_int        testid,
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

template <typename T>
aoclsparse_status aoclsparse_create_csr(aoclsparse_matrix    &mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        csr_nnz,
                                        aoclsparse_int       *csr_row_ptr,
                                        aoclsparse_int       *csr_col_ptr,
                                        T                    *csr_val);
template <>
aoclsparse_status aoclsparse_create_csr<double>(aoclsparse_matrix    &mat,
                                                aoclsparse_index_base base,
                                                aoclsparse_int        M,
                                                aoclsparse_int        N,
                                                aoclsparse_int        csr_nnz,
                                                aoclsparse_int       *csr_row_ptr,
                                                aoclsparse_int       *csr_col_ptr,
                                                double               *csr_val)
{
    return aoclsparse_create_dcsr(mat, base, M, N, csr_nnz, csr_row_ptr, csr_col_ptr, csr_val);
}

template <>
aoclsparse_status aoclsparse_create_csr<float>(aoclsparse_matrix    &mat,
                                               aoclsparse_index_base base,
                                               aoclsparse_int        M,
                                               aoclsparse_int        N,
                                               aoclsparse_int        csr_nnz,
                                               aoclsparse_int       *csr_row_ptr,
                                               aoclsparse_int       *csr_col_ptr,
                                               float                *csr_val)
{
    return aoclsparse_create_scsr(mat, base, M, N, csr_nnz, csr_row_ptr, csr_col_ptr, csr_val);
}

template <typename T>
bool get_data(const aoclsparse_int       id,
              string                    &title,
              aoclsparse_operation      &trans,
              aoclsparse_matrix         &A,
              aoclsparse_mat_descr      &descr,
              T                         &alpha,
              vector<T>                 &b,
              vector<T>                 &x,
              vector<T>                 &xref,
              T                         &xtol,
              vector<aoclsparse_int>    &icrowa,
              vector<aoclsparse_int>    &icola,
              vector<T>                 &aval,
              array<aoclsparse_int, 10> &iparm,
              array<T, 10>              &dparm,
              aoclsparse_status         &exp_status)
{
    aoclsparse_int        n, nnz;
    aoclsparse_diag_type  diag;
    aoclsparse_fill_mode  fill_mode;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    alpha                      = (T)1.0;
    xtol                       = (T)10.0 * XTOL<T>();
    exp_status                 = aoclsparse_status_success;
    std::fill(iparm.begin(), iparm.end(), 0);
    std::fill(dparm.begin(), dparm.end(), 0.0);
    title               = "";
    bool const isdouble = std::is_same_v<T, double>;
    switch(id)
    {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 24:
        switch(id)
        {
        case 24:
            title = " (sv hint)";
            __attribute__((fallthrough));
        case 0: // diag test set
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
            title     = "diag: Lx = alpha*b" + title;
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case 1:
            title     = "diag: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            break;
        case 2:
            title     = "diag: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case 3:
            title     = "diag: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        case 4:
            title     = "diag: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case 5:
            title     = "diag: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            break;
        case 6:
            title     = "diag: U'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case 7:
            title     = "diag: [triu(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        default:
            return false;
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
            return false;

        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return false;

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

    case 8: // small m test set
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
        switch(id - 8)
        {
        case 0: // small m test set
            /*
            Solve a   Ax = b
                      0   1  2   3   4   5    6      #cols  #row start #row end  #idiag  #iurow
            A  =  [  -2   1  0   0   3   7   -1;      5      0          4         0       1
                    2  -4  1   2   0   4    0;      5      5          9         6       7
                    0   6  -2  9   1   0    9;      6     10         14        11      12
                    -9   0   1 -2   1   1    1;      6     15         20        17      18
                    0   8   2  1  -2   2    0;      5     21         25        24      25
                    8   0   4  3   0   7    0;      4     26         29        29      30
                    0   0   3  6   9   0    2];     4     30         33        33      34
                                                                nnz=34
            b = [1.0  -2.0  0.0  2.0  -1.0 0.0 3.0]'
            */
            title     = "small m: Lx = alpha*b";
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
        case 1:
            title     = "small m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)1.0, (T)-4.0, (T)24.0, (T)-13.0, (T)-4.0, (T)-65.0, (T)45.0};
            break;
        case 2:
            title     = "small m: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)2.03125, (T)34.59375, (T)1.30625e+01, (T)7.125, (T)7.25, (T)0.0, (T)1.5};
            break;
        case 3:
            title     = "small m: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)85.0, (T)12.0, (T)35.0, (T)12.0, (T)-28.0, (T)0.0, (T)3.0};
            break;
        case 4:
            title     = "small m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)0.625, (T)2.25, (T)7.0, (T)0.0, (T)0.5, (T)0.0, (T)1.5};
            break;
        case 5:
            title     = "small m: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)-17.0, (T)24.0, (T)-26.0, (T)0.0, (T)-1.0, (T)0.0, (T)3.0};
            break;
        case 6:
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
                    (T)2.678571428571428e-1,
                    (T)2.96875e-1};
            break;
        case 7:
            title     = "small m: [triu(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {(T)1.0, (T)-3.0, (T)3.0, (T)-19.0, (T)12.0, (T)0.0, (T)-4.0};
            break;
        default:
            return false;
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
            return false;

        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return false;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        break;

    case 16: // large m test set
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
        xref.resize(25);
        alpha = (T)2.0;
        switch(id - 16)
        {
        case 0: // large m test set
            title     = "large m: Lx = alpha*b";
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
        case 1:
            title     = "large m: [tril(L,-1) + I]x = alpha*b";
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
        case 2:
            title     = "large m: L'x = alpha*b";
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
        case 3:
            title     = "large m: [tril(L,-1) + I]'x = alpha*b";
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
        case 4:
            title     = "large m: Ux = alpha*b";
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
        case 5:
            title     = "large m: [triu(U,1) + I]x = alpha*b";
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
        case 6:
            title     = "large m: U'x = alpha*b";
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
        case 7:
            title     = "large m: [triu(U,1) + I]'x = alpha*b";
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
            return false;
            break;
        }
        // transform(xref.begin(), xref.end(), xref.begin(), [alpha](T &d) { return alpha * d; });
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
            return false;

        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return false;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        break;

    case 25:
        title      = "Invalid matrix A (ptr NULL)";
        A          = nullptr;
        exp_status = aoclsparse_status_invalid_pointer;
        break;

    case 26:
        title = "eye(1) with null descriptor";
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
            return false;
        descr      = nullptr;
        exp_status = aoclsparse_status_invalid_pointer;
        break;

    case 27:
        title = "eye(1) with valid descriptor but negative expected_no_of_calls";
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
            return false;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return false;
        iparm[0]   = -10;
        exp_status = aoclsparse_status_invalid_value;
        break;

    case 28:
        title = "eye(1) with matrix type set to general";
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
            return false;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return false;
        descr->type = aoclsparse_matrix_type_general;
        exp_status  = aoclsparse_status_success;
        break;

    default:
        // no data with id found
        return false;
        break;
    }
    title = title + " [" + (isdouble ? "D" : "S") + "]";
    return true;
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
    aoclsparse_matrix    A = nullptr;
    aoclsparse_mat_descr descr = nullptr;
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
    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(A);
    return ok;
}

template <typename T>
bool trsv_hint_driver(aoclsparse_int &testid)
{
    bool                 ok      = true;
    aoclsparse_int       verbose = 1;
    string               title;
    T                    alpha;
    aoclsparse_matrix    A = nullptr;
    aoclsparse_mat_descr descr = nullptr;
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
    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(A);
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
