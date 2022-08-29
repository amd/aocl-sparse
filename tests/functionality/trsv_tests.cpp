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
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <vector>

#define NONE 0 // fallback
#define AVX 1
#define AVX2 1 // Milan
#define AVX512 2 // Genoa

using namespace std;

bool test_aoclsparse_trsv(const aoclsparse_int       testid,
                          const string               testdesc,
                          const double               alpha,
                          aoclsparse_matrix          &A,
                          const aoclsparse_mat_descr &descr,
                          const double*              b,
                          double*                    x,
                          const aoclsparse_operation trans,
                          const aoclsparse_int       avxversion,
                          const double*              xref,
                          const double               tol,
                          const aoclsparse_int       verbose)
{
    aoclsparse_status ret;
    aoclsparse_int    n = A->n;
    if(verbose)
    {
        bool         unit       = descr->diag_type == aoclsparse_diag_type_unit;
        const string avxlabs[3] = {"NONE (reference)", "AVX2", "AVX512"};
        cout << endl << "TEST #" << testid << " " << testdesc << endl;
        cout << "Configuration: unit=" << (unit ? "UNIT" : "NON-UNIT")
             << " trans=" << (trans == aoclsparse_operation_transpose ? "TRANSPOSE" : "NO")
             << " avxversion=" << avxlabs[avxversion] << endl;
    }
    //ret = aoclsparse_trsv(trans, alpha, A, descr, b, x, avxversion);
    ret = aoclsparse_dtrsv(trans, alpha, A, descr, b, x);
    if(ret != aoclsparse_status_success)
    {
        cout << "Test failed with unexpected return from aoclsparse_trsv, status = " << ret << endl;
    }
    bool pass = ret == aoclsparse_status_success;
    if(pass)
    {
        double err = 0.0;
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
    if(verbose)
    {
        cout << "TEST #" << testid << " : " << (pass ? "PASS" : "FAILED") << endl;
    }
    return pass;
}

bool get_data(const aoclsparse_int    id,
              string&                 title,
              aoclsparse_operation&   trans,
              aoclsparse_matrix&      A,
              aoclsparse_mat_descr&   descr,
              double&                 alpha,
              vector<double>&         b,
              vector<double>&         x,
              vector<double>&         xref,
              double&                 xtol,
              vector<aoclsparse_int>& icrowa,
              vector<aoclsparse_int>& icola,
              vector<double>&         aval)
{
    aoclsparse_int        n, nnz;
    aoclsparse_diag_type  diag;
    aoclsparse_fill_mode  fill_mode;
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    alpha                      = 1.0;
    xtol                       = 1.0e-8;
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
        switch(id)
        {
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
            title     = "diag: Lx = alpha*b";
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
            title     = "diag: [tril(U,1) + I]x = alpha*b";
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
            title     = "diag: [tril(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        default:
            return false;
            break;
        }
        alpha = -9.845233;
        n     = 7;
        nnz   = 7;
        b.resize(n);
        b = {1.0, -2.0, 8.0, 5.0, -1.0, 11.0, 3.0};
        x.resize(n);
        std::fill(x.begin(), x.end(), 0.0);
        xref.resize(n);
        icrowa.resize(n + 1);
        icrowa = {0, 1, 2, 3, 4, 5, 6, 7};
        icola.resize(nnz);
        icola = {0, 1, 2, 3, 4, 5, 6};
        aval.resize(nnz);
        aval = {-2.0, -4.0, 3.0, 5.0, -7.0, 9.0, 4.0};
        if(aoclsparse_create_dcsr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return false;

        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return false;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        if(diag == aoclsparse_diag_type_unit)
            xref = {1.0, -2.0, 8.0, 5.0, -1.0, 11.0, 3.0};
        else
            xref = {-0.5, 0.5, 8. / 3., 1.0, 1. / 7., 11. / 9., 0.75};
        // xref *= alpha;
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](double& d) { return alpha * d; });
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
            xref = {
                -5.0e-01, 2.5e-01, 7.5e-01, 1.625, 3.0625, -5.535714285714286e-01, -1.828125e+01};
            break;
        case 1:
            title     = "small m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {1.0, -4.0, 24.0, -13.0, -4.0, -65.0, 45.0};
            break;
        case 2:
            title     = "small m: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {2.03125, 34.59375, 1.30625e+01, 7.125, 7.25, 0, 1.5};
            break;
        case 3:
            title     = "small m: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {85.0, 12.0, 35.0, 12.0, -28.0, 0.0, 3.0};
            break;
        case 4:
            title     = "small m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {0.625, 2.25, 7.0, 0.0, 0.5, 0.0, 1.5};
            break;
        case 5:
            title     = "small m: [tril(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {-17.0, 24.0, -26.0, 0.0, -1.0, 0.0, 3.0};
            break;
        case 6:
            title     = "small m: U'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {-5.0e-1,
                    3.75e-1,
                    1.875e-1,
                    2.1875e-1,
                    -4.6875e-2,
                    2.678571428571428e-1,
                    2.96875e-1};
            break;
        case 7:
            title     = "small m: [tril(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            xref = {1.0, -3.0, 3.0, -19.0, 12.0, 0.0, -4.0};
            break;
        default:
            return false;
            break;
        }
        alpha = 1.3334;
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](double& d) { return alpha * d; });
        n   = 7;
        nnz = 34;
        b.resize(n);
        b = {1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 3.0};
        x.resize(n);
        std::fill(x.begin(), x.end(), 0.0);
        icrowa.resize(n + 1);
        icrowa = {0, 5, 10, 15, 21, 26, 30, 34};
        icola.resize(nnz);
        icola = {0, 1, 4, 5, 6, 0, 1, 2, 3, 5, 1, 2, 3, 4, 6, 0, 2,
                 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 2, 3, 5, 2, 3, 4, 6};
        aval.resize(nnz);
        aval = {-2.0, 1.0, 3.0, 7.0,  -1.0, 2.0,  -4.0, 1.0, 2.0, 4.0, 6.0, -2.0,
                9.0,  1.0, 9.0, -9.0, 1.0,  -2.0, 1.0,  1.0, 1.0, 8.0, 2.0, 1.0,
                -2.0, 2.0, 8.0, 4.0,  3.0,  7.0,  3.0,  6.0, 9.0, 2.0};
        if(aoclsparse_create_dcsr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return false;

        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return false;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        break;

    case 16: // small m test set
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
        switch(id - 16)
        {
        case 0: // large m test set

            title     = "large m: Lx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(25);
            xref = {22.50000000,     -29.59158416,     0.60344828,      7.36902949,
                    0.35313901,      6.96098318,       -3.22325028,     4.36270955,
                    -41.25004375,    418.11611572,     641.78764582,    -1.76990937,
                    -1520.45079758,  -67.79837748,     -3253.64596006,  536.33671084,
                    40956.42016049,  -240315.33122282, 108048.96797262, -142.25127350,
                    41.03239512,     -249085.87884931, 256238.41460554, -547689.07187637,
                    1832933.45174056};
            break;
        case 1:
            title     = "large m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(25);
            xref = {3.15000000,           -22.39650000,        3.15000000,
                    21.06720000,          3.15000000,          -141.06890700,
                    97.12111500,          777.99517677,        -5514.52102232,
                    25551.34154355,       39691.17075985,      288.48038121,
                    -333746.10858156,     -28449.33410860,     -41808.32818536,
                    1167397.49342707,     2740030.82395464,    -20826876.87800828,
                    79462783.32178056,    303906.03635735,     126695.45699191,
                    -541221946.86130440,  4451957552.04631233, -38065059811.08203888,
                    160061850136.86447144};
            break;
        case 2:
            title     = "large m: L'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(25);
            xref = {
                1109946.57447404, 8320.22836232,   49608.59974037, 15392.72484029,  18342.60735427,
                -17342.60041707,  -10303.87031452, 21888.38859196, -14398.43244835, 10521.91066968,
                5945.38745655,    254.50168851,    -770.82846866,  -138.69537858,   93.50972554,
                9.93719186,       341.77562392,    -10.73709417,   2.57693312,      -0.70901009,
                6.12185399,       -4.10547785,     1.12600628,     -0.84855313,     1.27016129};
            break;
        case 3:
            title     = "large m: [tril(L,-1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(25);
            xref = {6935464282.41652679,
                    3042481822.58313370,
                    36182521907.10009766,
                    41854296751.45114136,
                    44682619196.94413757,
                    -4649203128.03015041,
                    -2576033902.18770599,
                    554009582.16372883,
                    -70119029.18721765,
                    5392625.93657788,
                    6142855.13668140,
                    4801989.23532077,
                    -1216547.16999674,
                    -1359885.61384011,
                    154714.66071554,
                    76733.55435697,
                    133560.80130906,
                    -18140.65366431,
                    4639.38339413,
                    -17.19900000,
                    7328.16068494,
                    -801.44613360,
                    90.57132000,
                    -9.67050000,
                    3.15000000};
            break;
        case 4:
            title     = "large m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(25);
            xref = {8422.64682503, -385.32336118, 359.63019091, 203.73603215, 17.99742920,
                    -390.82672167, 116.45563268,  -29.40255322, 108.87470697, 1.60549241,
                    -58.35652677,  -12.83135054,  53.18109227,  -0.85679764,  -48.43414031,
                    -0.81695457,   10.50000000,   10.77389595,  -0.68813786,  0.10688594,
                    -1.77747358,   0.78791787,    -0.82852204,  1.32352941,   1.27016129};
            break;
        case 5:
            title     = "large m: [tril(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(25);
            xref = {548827936.82895064, -89971900.29543361, 19736045.24593790, 2838202.78574788,
                    627279.35216463,    -3813589.00964278,  478983.46059670,   -182634.07952505,
                    51577.85584669,     -104.47400880,      -9276.75779773,    -12131.57642582,
                    -10633.27457233,    -1783.44530412,     8293.84481121,     -1551.75222699,
                    3.15000000,         2589.26855670,      -79.33023000,      -2.77200000,
                    -345.54030840,      59.12172000,        -23.37300000,      3.15000000,
                    3.15000000};
            break;
        case 6:
            title     = "large m: U'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(25);
            xref = {22.50000000,    -23.91089109,  11.80593490,   -2.69407573,    24.98966380,
                    -6.66851489,    39.45390422,   -15.62441784,  20.02412821,    118.44503254,
                    -662.20402523,  42.58098933,   -15.41034592,  -54.26686484,   -70.82489049,
                    85.02988016,    1869.52057717, 1863.41258056, -1373.90871739, 369.61754662,
                    -2169.14708417, -475.98770556, 1091.99336450, 4642.25731391,  3047.76386992};
            break;
        case 7:
            title     = "large m: [tril(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(25);
            xref = {3.15000000,        -17.57700000,      70.10451000,        29.51046000,
                    128.16548640,      -426.53947442,     3854.16377714,      -9476.91018578,
                    40199.53283129,    27848.49828622,    -367024.74933116,   70054.38281425,
                    -56623.14116001,   -143863.79672409,  34409.37920797,     213495.19529202,
                    -5838.00204054,    320880.31544025,   -2965469.35362804,  24755926.05339951,
                    -4463690.91167807, 28170687.84895727, -63940055.40049978, 537417949.93606544,
                    -40536128.32094885};
            break;
        default:
            return false;
            break;
        }
        xtol  = 5.e-4;
        alpha = 1.17;
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](double& d) { return alpha * d; });
        n   = 25;
        nnz = 335;
        b.resize(n);
        std::fill(b.begin(), b.end(), 3.15);
        x.resize(n);
        std::fill(x.begin(), x.end(), 0.0);
        icrowa.resize(n + 1);
        icrowa = {0,   12,  31,  37,  56,  61,  82,  96,  113, 135, 142, 153, 160,
                  178, 186, 202, 223, 226, 243, 255, 260, 268, 288, 302, 319, 335};
        icola.resize(nnz);
        icola = {0,  1,  2,  3,  8,  9,  10, 12, 14, 18, 19, 20, 0,  1,  2,  3,  4,  5,  6,  8,  9,
                 10, 11, 13, 14, 15, 16, 19, 21, 22, 24, 2,  5,  6,  14, 20, 24, 1,  3,  4,  5,  6,
                 8,  9,  10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 4,  5,  9,  10, 21, 0,  1,
                 2,  3,  4,  5,  6,  7,  8,  9,  11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 24, 0,  1,
                 4,  6,  7,  8,  10, 13, 14, 18, 20, 21, 22, 24, 1,  3,  5,  6,  7,  8,  9,  11, 12,
                 13, 14, 15, 16, 18, 19, 20, 23, 0,  1,  2,  4,  5,  6,  7,  8,  10, 11, 12, 13, 14,
                 15, 16, 17, 19, 20, 21, 22, 23, 24, 0,  4,  8,  9,  19, 21, 23, 1,  4,  6,  8,  10,
                 13, 17, 18, 19, 22, 23, 4,  5,  11, 13, 17, 21, 22, 0,  3,  6,  7,  9,  10, 11, 12,
                 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 0,  1,  3,  7,  9,  11, 13, 17, 1,  2,  3,
                 5,  6,  9,  10, 11, 13, 14, 15, 16, 18, 19, 20, 22, 0,  1,  2,  3,  4,  6,  7,  9,
                 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 0,  12, 16, 6,  7,  9,  10, 11,
                 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 0,  3,  4,  8,  10, 11, 17, 18, 19,
                 21, 22, 24, 4,  9,  12, 19, 24, 3,  4,  11, 13, 20, 21, 23, 24, 0,  1,  2,  3,  4,
                 5,  6,  7,  8,  9,  10, 13, 14, 15, 16, 18, 20, 21, 22, 24, 0,  1,  2,  5,  7,  10,
                 16, 17, 18, 20, 21, 22, 23, 24, 0,  1,  2,  3,  7,  8,  9,  11, 12, 14, 15, 16, 18,
                 20, 21, 22, 23, 1,  2,  4,  5,  6,  10, 11, 13, 14, 15, 18, 19, 20, 21, 23, 24};
        aval.resize(nnz);
        aval = {0.14, 6.58, 1.79, 2.68, 4.93, 3.97, 2.81, 4.48, 9.04, 4.19, 7.41, 4.39, 8.11,  6.06,
                4.13, 1.98, 9.06, 1.57, 5.62, 9.31, 5.35, 5.95, 2.48, 1.12, 8.51, 5.59, 0.79,  4.99,
                8.87, 4.16, 6.11, 5.22, 5.70, 4.14, 2.50, 6.72, 3.54, 0.80, 3.64, 1.16, 1.26,  2.64,
                2.39, 2.84, 8.36, 7.78, 1.54, 6.88, 9.00, 2.06, 4.25, 9.20, 8.84, 9.55, 6.51,  0.88,
                8.92, 0.16, 4.54, 1.83, 5.90, 1.85, 4.89, 7.65, 9.61, 6.78, 4.08, 9.66, 4.34,  2.95,
                8.44, 7.95, 6.32, 1.09, 8.93, 3.66, 7.64, 2.31, 5.49, 8.33, 2.84, 8.58, 8.68,  6.11,
                4.93, 4.06, 2.94, 2.31, 8.35, 5.03, 3.04, 2.13, 0.11, 6.18, 5.78, 4.48, 3.60,  5.12,
                9.61, 5.70, 5.37, 5.04, 2.62, 8.92, 2.40, 3.52, 8.32, 2.72, 0.29, 8.02, 4.87,  3.45,
                8.48, 9.40, 0.82, 8.80, 7.57, 9.78, 8.93, 7.67, 6.42, 8.32, 0.44, 2.04, 10.00, 7.78,
                2.49, 2.06, 4.99, 3.57, 9.64, 5.98, 7.31, 7.16, 5.91, 3.30, 9.14, 4.64, 0.28,  5.15,
                2.04, 0.41, 3.06, 2.93, 8.73, 7.34, 0.66, 0.17, 3.81, 0.92, 8.01, 8.79, 5.91,  0.33,
                2.03, 6.27, 5.39, 8.41, 1.06, 3.89, 4.41, 1.40, 8.93, 8.34, 4.25, 5.47, 2.24,  3.55,
                9.87, 4.95, 7.34, 6.71, 3.14, 4.34, 3.35, 1.97, 2.77, 2.93, 2.88, 6.82, 9.08,  4.42,
                0.95, 2.40, 5.00, 0.69, 7.63, 4.35, 8.58, 7.69, 6.89, 5.28, 3.11, 4.18, 7.64,  1.09,
                3.82, 6.96, 0.70, 1.69, 6.56, 2.48, 5.11, 7.97, 3.40, 9.08, 9.33, 8.59, 0.56,  4.15,
                0.47, 7.19, 3.62, 3.07, 7.02, 0.31, 1.82, 7.04, 2.67, 7.81, 2.03, 0.99, 2.90,  8.85,
                8.21, 0.30, 7.78, 4.43, 3.38, 2.28, 3.26, 8.69, 4.64, 8.45, 3.92, 7.10, 1.06,  6.31,
                7.24, 7.50, 8.83, 0.86, 0.42, 9.37, 1.21, 8.51, 2.71, 8.00, 5.17, 3.83, 8.47,  8.35,
                2.50, 2.69, 6.57, 0.80, 7.96, 1.52, 7.13, 1.88, 5.56, 0.65, 3.21, 4.49, 6.63,  5.47,
                7.97, 0.06, 6.32, 0.81, 6.16, 4.11, 7.21, 3.56, 6.44, 3.10, 4.74, 5.26, 2.73,  8.05,
                1.70, 7.02, 6.21, 6.48, 9.26, 3.84, 2.64, 1.82, 3.70, 2.76, 6.11, 2.15, 0.09,  9.45,
                2.46, 4.14, 6.13, 1.71, 8.98, 9.61, 7.82, 0.60, 1.32, 3.90, 5.29, 8.06, 6.07,  8.54,
                7.83, 0.46, 9.40, 7.65, 0.47, 0.50, 0.08, 7.29, 4.04, 9.04, 2.38, 2.15, 4.49,  1.14,
                5.26, 9.17, 3.60, 5.55, 5.58, 4.27, 3.70, 0.86, 6.46, 3.81, 9.63, 4.07, 2.48};
        if(aoclsparse_create_dcsr(A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return false;

        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return false;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        break;

    default:
        // no data with id found
        return false;
        break;
    }
    return true;
}

int main(void)
{
    aoclsparse_int       testid = 0;
    bool                 okload, ok = true;
    aoclsparse_int       verbose    = 1;
    aoclsparse_int       avxversion = NONE; // NONE=0 AVX=1 (AVX2=1) and AVX512=2
    string               title      = "unknown";
    double               alpha;
    aoclsparse_matrix    A;
    aoclsparse_mat_descr descr;
    vector<double>       b;
    vector<double>       x;
    vector<double>       xref;
    double               xtol;
    aoclsparse_operation trans;
    // permanent storage of matrix data
    vector<double>         aval;
    vector<aoclsparse_int> icola;
    vector<aoclsparse_int> icrowa;

    cout << "================================================" << endl;
    cout << "   AOCLSPARSE TRSV TESTS" << endl;
    cout << "================================================" << endl;

    do
    {
        // fail on getting data or gone beyond last testid
        okload = get_data(
            testid, title, trans, A, descr, alpha, b, x, xref, xtol, icrowa, icola, aval);
        if(okload) // TODO PASS_BY-REFERENCE!!!
            // Call test as many times as available kernels
            // only run on AVX2 until calling non-public APIs is possible
            // TODO change when testing framework is adopted
            for(int avxversion = AVX2; avxversion <= AVX2; avxversion++)
            {
                ok &= test_aoclsparse_trsv(testid,
                                           title,
                                           alpha,
                                           A,
                                           descr,
                                           &b[0],
                                           &x[0],
                                           trans,
                                           avxversion,
                                           &xref[0],
                                           xtol,
                                           verbose);
            }
        ++testid;
    } while(okload && ok);

    cout << "================================================" << endl;
    cout << "   OVERALL TESTS : " << (ok ? "ALL PASSSED" : "FAILED") << endl;
    cout << "================================================" << endl;

    return (ok ? 0 : 1);
}
