/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_descr.h"
#include "gtest/gtest.h"
#include "aoclsparse.hpp"
#include "aoclsparse_mat_structures.hpp" // FIXME: library internal header used for testing
#include "aoclsparse_reference.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

//aocl utils
#include "Au/Cpuid/X86Cpu.hh"
// Utilities to compare complex real scalars and vectors =============================================

#define EXPECT_COMPLEX_EQ_VEC(n, x, y)                                           \
    for(size_t i = 0; i < (size_t)n; i++)                                        \
    {                                                                            \
        EXPECT_EQ(std::real(x[i]), std::real(y[i]))                              \
            << " Real parts of " #x " and " #y " differ at index i = " << i      \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i]); \
        EXPECT_EQ(std::imag(x[i]), std::imag(y[i]))                              \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i]); \
    }

#define EXPECT_COMPLEX_FLOAT_EQ_VEC(n, x, y)                                     \
    for(size_t i = 0; i < (size_t)n; i++)                                        \
    {                                                                            \
        EXPECT_FLOAT_EQ(std::real(x[i]), std::real(y[i]))                        \
            << " Real parts of " #x " and " #y " differ at index i = " << i      \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i]); \
        EXPECT_FLOAT_EQ(std::imag(x[i]), std::imag(y[i]))                        \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i]); \
    }

#define EXPECT_COMPLEX_DOUBLE_EQ_VEC(n, x, y)                                    \
    for(size_t i = 0; i < (size_t)n; i++)                                        \
    {                                                                            \
        EXPECT_DOUBLE_EQ(std::real(x[i]), std::real(y[i]))                       \
            << " Real parts of " #x " and " #y " differ  at index i = " << i     \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i]); \
        EXPECT_DOUBLE_EQ(std::imag(x[i]), std::imag(y[i]))                       \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i]); \
    }

#define EXPECT_COMPLEX_FLOAT_EQ(x, y)                                      \
    do                                                                     \
    {                                                                      \
        EXPECT_FLOAT_EQ(std::real(x), std::real(y))                        \
            << " Real parts of " #x " and " #y " differ."                  \
            << " values are: " << std::real(x) << " and " << std::real(y); \
        EXPECT_FLOAT_EQ(std::imag(x), std::imag(y))                        \
            << " Imaginary parts of " #x " and " #y " differ."             \
            << " values are: " << std::imag(x) << " and " << std::imag(y); \
    } while(0)

#define EXPECT_COMPLEX_DOUBLE_EQ(x, y)                                     \
    do                                                                     \
    {                                                                      \
        EXPECT_DOUBLE_EQ(std::real(x), std::real(y))                       \
            << " Real parts of " #x " and " #y " differ."                  \
            << " values are: " << std::real(x) << " and " << std::real(y); \
        EXPECT_DOUBLE_EQ(std::imag(x), std::imag(y))                       \
            << " Imaginary parts of " #x " and " #y " differ."             \
            << " values are: " << std::imag(x) << " and " << std::imag(y); \
    } while(0)

// Template function to compare a single value irrespective of type =================================
template <typename T>
void expect_eq(T res, T ref)
{
    if constexpr(std::is_same_v<T, double>)
        EXPECT_DOUBLE_EQ(res, ref);
    else if constexpr(std::is_same_v<T, float>)
        EXPECT_FLOAT_EQ(res, ref);
    else if constexpr(std::is_same_v<T, std::complex<double>>)
        EXPECT_COMPLEX_DOUBLE_EQ(res, ref);
    else if constexpr(std::is_same_v<T, std::complex<float>>)
        EXPECT_COMPLEX_FLOAT_EQ(res, ref);
    else
    {
        std::string err = "expect_eq does not support type: " + std::string(typeid(T).name());
        FAIL() << err;
    }
}

// Utilities to compare real scalars and vectors =============================================

#define EXPECT_EQ_VEC(n, x, y)                                                              \
    for(size_t i = 0; i < (size_t)n; i++)                                                   \
    {                                                                                       \
        EXPECT_EQ((x)[i], (y)[i]) << " vectors " #x " and " #y " differ at index i = " << i \
                                  << " values are: " << (x)[i] << " and " << (y)[i]         \
                                  << " respectively.";                                      \
    }

#define EXPECT_FLOAT_EQ_VEC(n, x, y)                                                               \
    for(size_t i = 0; i < (size_t)n; i++)                                                          \
    {                                                                                              \
        EXPECT_FLOAT_EQ((x)[i], (y)[i]) << " vectors " #x " and " #y " differ at index i = " << i  \
                                        << " by abs err: " << abs((x)[i] - (y)[i])                 \
                                        << " rel err: " << abs(((x)[i] - (y)[i]) / (x)[i]) << "."; \
    }

#define EXPECT_DOUBLE_EQ_VEC(n, x, y)                                  \
    for(size_t i = 0; i < (size_t)n; i++)                              \
    {                                                                  \
        EXPECT_DOUBLE_EQ((x)[i], (y)[i])                               \
            << " vectors " #x " and " #y " differ at index i = " << i  \
            << " by abs err: " << abs((x)[i] - (y)[i])                 \
            << " rel err: " << abs(((x)[i] - (y)[i]) / (x)[i]) << "."; \
    }

// Template function to compare a vector irrespective of type =================================
template <typename T>
void expect_eq_vec(aoclsparse_int n, T *res, T *ref)
{
    if constexpr(std::is_same_v<T, double>)
    {
        EXPECT_DOUBLE_EQ_VEC(n, res, ref);
    }
    else if constexpr(std::is_same_v<T, float>)
    {
        EXPECT_FLOAT_EQ_VEC(n, res, ref);
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>)
    {
        EXPECT_COMPLEX_DOUBLE_EQ_VEC(n, res, ref);
    }
    else if constexpr(std::is_same_v<T, std::complex<float>>)
    {
        EXPECT_COMPLEX_FLOAT_EQ_VEC(n, res, ref);
    }
    else
    {
        std::string err = "expect_eq_vec does not support type: " + std::string(typeid(T).name());
        FAIL() << err;
    }
}

#define EXPECT_ARR_NEAR(n, x, y, abs_error)    \
    for(size_t j = 0; j < (size_t)(n); j++)    \
    EXPECT_NEAR(((x)[j]), ((y)[j]), abs_error) \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

#define EXPECT_COMPLEX_ARR_NEAR(n, x, y, abs_error)                              \
    for(size_t i = 0; i < (size_t)n; i++)                                        \
    {                                                                            \
        EXPECT_NEAR(std::real(x[i]), std::real(y[i]), abs_error)                 \
            << " Real parts of " #x " and " #y " differ at index i = " << i      \
            << " values are: " << std::real(x[i]) << " and " << std::real(y[i])  \
            << " by abs err: " << abs(std::real(x[i]) - std::real(y[i]));        \
        EXPECT_NEAR(std::imag(x[i]), std::imag(y[i]), abs_error)                 \
            << " Imaginary parts of " #x " and " #y " differ at index i = " << i \
            << " values are: " << std::imag(x[i]) << " and " << std::imag(y[i])  \
            << " by abs err: " << abs(std::real(x[i]) - std::real(y[i]));        \
    }

// Template function to compare a matrix irrespective of type =================================
template <typename T>
void expect_arr_near(aoclsparse_int n, T *x, T *y, tolerance_t<T> abs_error)
{
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        EXPECT_ARR_NEAR(n, x, y, abs_error);
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>
                      || std::is_same_v<T, std::complex<float>>)
    {
        EXPECT_COMPLEX_ARR_NEAR(n, x, y, abs_error);
    }
    else
    {
        std::string err = "expect_arr_near does not support type: " + std::string(typeid(T).name());
        FAIL() << err;
    }
}

#define EXPECT_MAT_NEAR(m, n, ld, x, y, abs_error)                                      \
    for(size_t c = 0; c < (size_t)m; c++)                                               \
    {                                                                                   \
        for(size_t j = 0; j < (size_t)n; j++)                                           \
        {                                                                               \
            size_t offset = c * ld + j;                                                 \
            EXPECT_NEAR(((x)[offset]), ((y)[offset]), abs_error)                        \
                << "Vectors " #x " and " #y " different at index j =" << offset << "."; \
        }                                                                               \
    }

#define EXPECT_TRIMAT_NEAR(m, n, ld, x, y, abs_error)                                   \
    for(size_t c = 0; c < (size_t)m; c++)                                               \
    {                                                                                   \
        for(size_t j = c; j < (size_t)n; j++)                                           \
        {                                                                               \
            size_t offset = c * ld + j;                                                 \
            EXPECT_NEAR(((x)[offset]), ((y)[offset]), abs_error)                        \
                << "Vectors " #x " and " #y " different at index j =" << offset << "."; \
        }                                                                               \
    }

// Template function to compare a matrix irrespective of type =================================
template <typename T>
void expect_mat_near(
    aoclsparse_int m, aoclsparse_int n, aoclsparse_int ld, T *x, T *y, tolerance_t<T> abs_error)
{
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        EXPECT_MAT_NEAR(m, n, ld, x, y, abs_error);
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>
                      || std::is_same_v<T, std::complex<float>>)
    {
        EXPECT_COMPLEX_MAT_NEAR(m, n, ld, x, y, abs_error);
    }
    else
    {
        std::string err = "expect_mat_near does not support type: " + std::string(typeid(T).name());
        FAIL() << err;
    }
}

#define EXPECT_COMPLEX_MAT_NEAR(m, n, ld, x, y, abs_error)                                    \
    for(size_t c = 0; c < (size_t)m; c++)                                                     \
    {                                                                                         \
        for(size_t j = 0; j < (size_t)n; j++)                                                 \
        {                                                                                     \
            size_t offset = c * ld + j;                                                       \
            EXPECT_NEAR(std::real(x[offset]), std::real(y[offset]), abs_error)                \
                << " Real parts of " #x " and " #y " differ at index j = " << offset          \
                << " values are: " << std::real(x[offset]) << " and " << std::real(y[offset]) \
                << " by abs err: " << abs(std::real(x[offset]) - std::real(y[offset]));       \
            EXPECT_NEAR(std::imag(x[offset]), std::imag(y[offset]), abs_error)                \
                << " Imaginary parts of " #x " and " #y " differ at index j = " << (offset)   \
                << " values are: " << std::imag(x[offset]) << " and " << std::imag(y[offset]) \
                << " by abs err: " << abs(std::real(x[offset]) - std::real(y[offset]));       \
        }                                                                                     \
    }

#define EXPECT_COMPLEX_TRIMAT_NEAR(m, n, ld, x, y, abs_error)                                 \
    for(size_t c = 0; c < (size_t)m; c++)                                                     \
    {                                                                                         \
        for(size_t j = c; j < (size_t)n; j++)                                                 \
        {                                                                                     \
            size_t offset = c * ld + j;                                                       \
            EXPECT_NEAR(std::real(x[offset]), std::real(y[offset]), abs_error)                \
                << " Real parts of " #x " and " #y " differ at index j = " << offset          \
                << " values are: " << std::real(x[offset]) << " and " << std::real(y[offset]) \
                << " by abs err: " << abs(std::real(x[offset]) - std::real(y[offset]));       \
            EXPECT_NEAR(std::imag(x[offset]), std::imag(y[offset]), abs_error)                \
                << " Imaginary parts of " #x " and " #y " differ at index j = " << (offset)   \
                << " values are: " << std::imag(x[offset]) << " and " << std::imag(y[offset]) \
                << " by abs err: " << abs(std::real(x[offset]) - std::real(y[offset]));       \
        }                                                                                     \
    }

/*
    Relative error tolerance for comparing floating point numbers.
    The first argument 'x' is the reference value.
    The minimum error tolerance is chosen between the absolute error tolerance
    and relative tolerance.
*/
#define EXPECT_REL(x, y, rel_error, abs_error) \
    EXPECT_NEAR((x), (y), std::min(rel_error *std::abs(x), abs_error))

// Template function to compare a triangular-matrix irrespective of type =================================
template <typename T>
void expect_trimat_near(
    aoclsparse_int m, aoclsparse_int n, aoclsparse_int ld, T *x, T *y, tolerance_t<T> abs_error)
{
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        EXPECT_TRIMAT_NEAR(m, n, ld, x, y, abs_error);
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>
                      || std::is_same_v<T, std::complex<float>>)
    {
        EXPECT_COMPLEX_TRIMAT_NEAR(m, n, ld, x, y, abs_error);
    }
    else
    {
        std::string err
            = "expect_trimat_near does not support type: " + std::string(typeid(T).name());
        FAIL() << err;
    }
}
enum special_value
{
    ET_ZERO = 0, //zero
    ET_NAN, //Nan
    ET_INF, //Infinity
    ET_NUM, //number
    ET_POVRFLOW, //positive overflow
    ET_NOVRFLOW, //negative overflow
    ET_PUNDRFLOW, //positive underflow
    ET_NUNDRFLOW, //negative overflow
    ET_CPLX_NAN_INF //for complex cases when result is of the form (NaN + Inf . i)
};

/*
    check whether inputs are extreme(NaN, Inf) or non-extreme and this applies only for real data
    The minimum error tolerance for comparing floating point numbers is
    chosen between the absolute error tolerance and relative tolerance.

    enable_if_t<> resolves to either double or float at compile time, based on the provided template parameter
*/
template <typename T>
std::enable_if_t<(std::is_same_v<T, double> || std::is_same_v<T, float>), bool>
    is_matching(T ref, T computed, T rel_error, T abs_error)
{
    tolerance_t<T> tol = (std::min)(rel_error * std::abs(ref), abs_error);
    if(std::isnan(ref))
    {
        return (std::isnan(ref) && std::isnan(computed));
    }
    else if(std::isinf(ref))
    {
        return (std::isinf(ref) && std::isinf(computed));
    }
    else
        return (std::abs(ref - computed) <= tol);
}

/*
    overloaded function, addresses complex data
    enable_if_t<> resolves to either complex-double or complex-float at compile time,
    based on the provided template parameter
*/
template <typename T>
std::enable_if_t<
    ((std::is_same_v<T, std::complex<double>>) || (std::is_same_v<T, std::complex<float>>)),
    bool>
    is_matching(T ref, T computed, tolerance_t<T> rel_error, tolerance_t<T> abs_error)
{
    return (is_matching(std::real(ref), std::real(computed), rel_error, abs_error))
           && (is_matching(std::imag(ref), std::imag(computed), rel_error, abs_error));
}
template <typename T>
std::enable_if_t<((std::is_same_v<T, aoclsparse_double_complex>)
                  || (std::is_same_v<T, aoclsparse_float_complex>)),
                 bool>
    is_matching(T ref, T computed, tolerance_t<T> rel_error, tolerance_t<T> abs_error)
{
    return (is_matching(ref.real, computed.real, rel_error, abs_error))
           && (is_matching(ref.imag, computed.imag, rel_error, abs_error));
}

#define EXPECT_MATCH(T, ref, computed, rel_error, abs_error) \
    EXPECT_PRED4(is_matching<T>, ref, computed, rel_error, abs_error)
#define EXPECT_ARR_MATCH(T, n, x, y, rel_error, abs_error)    \
    for(size_t j = 0; j < (size_t)(n); j++)                   \
    EXPECT_MATCH(T, ((x)[j]), ((y)[j]), rel_error, abs_error) \
        << "Vectors " #x " and " #y " different at index j=" << j << "."

// Convenience templated interfaces ==========================================================

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

// Problem DATABASES =========================================================================
// DB for matrices: returns a matrix
// Use create_matrix(...)
enum matrix_id
{
    /* Used mainly by hint_tests*/
    N5_full_sorted,
    N5_full_unsorted,
    N59_partial_sort,
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
    sample_gmres_mat_03,
    S5_sym_fullmatrix,
    S5_sym_lowerfill,
    S5_sym_upperfill,
    S4_herm_rsym,
    invalid_mat,
};

// DB for linear system equations: returns matrix, rhs, expected sol, tolerance,
// Use create_linear_system(...)
// Warning: Don't change the order of these enums, some test rely on it.
// Append towards the end any new problem id.
enum linear_system_id
{
    /* Used by TRSV */
    // diagonal matrix 7x7
    D7_Lx_aB,
    D7_LL_Ix_aB,
    D7_LTx_aB,
    D7_LL_ITx_aB,
    D7_LHx_aB,
    D7_LL_IHx_aB,
    D7_Ux_aB,
    D7_UU_Ix_aB,
    D7_UTx_aB,
    D7_UU_ITx_aB,
    D7_UHx_aB,
    D7_UU_IHx_aB,
    // small matrix 7x7
    S7_Lx_aB,
    S7_LL_Ix_aB,
    S7_LTx_aB,
    S7_LL_ITx_aB,
    S7_LHx_aB,
    S7_LL_IHx_aB,
    S7_Ux_aB,
    S7_UU_Ix_aB,
    S7_UTx_aB,
    S7_UU_ITx_aB,
    S7_UHx_aB,
    S7_UU_IHx_aB,
    // matrix 25x25
    N25_Lx_aB,
    N25_LL_Ix_aB,
    N25_LTx_aB,
    N25_LL_ITx_aB,
    N25_LHx_aB,
    N25_LL_IHx_aB,
    N25_Ux_aB,
    N25_UU_Ix_aB,
    N25_UTx_aB,
    N25_UU_ITx_aB,
    N25_UHx_aB,
    N25_UU_IHx_aB,
    // matrix used for hinting
    D7Lx_aB_hint,
    A_nullptr,
    D1_descr_nullptr,
    D1_neg_num_hint,
    D1_mattype_gen_hint,
    // matrices used for TRSM
    SM_S7_XB910,
    SM_S7_XB1716,
    SM_S1X1_XB1X1,
    SM_S1X1_XB1X3,
    // SYMGS Problems
    GS_S7,
    GS_MV_S7,
    GS_TRIDIAG_M5,
    GS_MV_TRIDIAG_M5,
    GS_BLOCK_TRDIAG_S9,
    GS_MV_BLOCK_TRDIAG_S9,
    GS_CONVERGE_S4,
    GS_MV_CONVERGE_S4,
    GS_NONSYM_S4,
    GS_MV_NONSYM_S4,
    GS_HR4,
    GS_MV_HR4,
    GS_TRIANGLE_S5,
    GS_MV_TRIANGLE_S5,
    GS_SYMM_ALPHA2_S9,
    GS_MV_SYMM_ALPHA2_S9,
    // Extreme Value Cases
    EXT_G5,
    EXT_G5_B0,
    EXT_S5,
    EXT_S5_B0,
    EXT_NSYMM_5,
    EXT_H5,
    EXT_H5_B0,
    H5_Lx_aB,
    H5_LTx_aB,
    H5_LHx_aB,
    H5_Ux_aB,
    H5_UTx_aB,
    H5_UHx_aB
};

template <typename T>
inline T bc(T v)
{
    if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
        return T(std::real(v), std::real(v) / 10);
    else
        return v;
};

template <typename T>
inline T bcd(T v, T w)
{
    if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
        return T(std::real(v), std::real(w));
    else
        return v;
};

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
    aoclsparse_status     ret  = aoclsparse_status_success;
    aoclsparse_index_base base = descr->base;

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
        csr_val = {bc((T)1), bc((T)2), bc((T)3), bc((T)4), bc((T)5), bc((T)6), bc((T)7), bc((T)8)};
        break;

    case N5_full_unsorted:
        // same as N5 full sorted with rows 0 and 3 shuffled
        n = m       = 5;
        nnz         = 8;
        csr_row_ptr = {0, 2, 3, 4, 7, 8};
        csr_col_ind = {3, 0, 1, 2, 3, 1, 4, 4};
        csr_val = {bc((T)2), bc((T)1), bc((T)3), bc((T)4), bc((T)6), bc((T)5), bc((T)7), bc((T)8)};
        break;

    case N59_partial_sort:
        // Small partially sorted matrix with full diagonal
        // Already clean no need to copy
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  4  0  0
        //  0  5  9  6  7 -> stored as (9,5 | 6 | 7) so 9 and 5 are swaped
        //  0  0  0  0  8
        n = m       = 5;
        nnz         = 9;
        csr_row_ptr = {0, 2, 3, 4, 8, 9};
        csr_col_ind = {0, 3, 1, 2, 2, 1, 3, 4, 4};
        csr_val     = {1, 2, 3, 4, 9, 5, 6, 7, 8};
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
        csr_val     = {bc((T)1),
                       bc((T)2),
                       bc((T)3),
                       bc((T)6),
                       bc((T)5),
                       bc((T)7),
                       bc((T)1),
                       bc((T)2),
                       bc((T)3),
                       bc((T)4)};
        break;

    case sample_cg_mat:
        // matrix from the CG sample examples
        // symmetric, lower triangle filled
        n = m       = 8;
        nnz         = 18;
        csr_row_ptr = {0, 1, 2, 5, 6, 8, 11, 15, 18};
        csr_col_ind = {0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
        csr_val     = {bc((T)19),
                       bc((T)10),
                       bc((T)1),
                       bc((T)8),
                       bc((T)11),
                       bc((T)13),
                       bc((T)2),
                       bc((T)11),
                       bc((T)2),
                       bc((T)1),
                       bc((T)9),
                       bc((T)7),
                       bc((T)9),
                       bc((T)5),
                       bc((T)12),
                       bc((T)5),
                       bc((T)5),
                       bc((T)9)};
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
        csr_val     = {bc((T)0.75), bc((T)0.14), bc((T)0.11), bc((T)0.14), bc((T)0.11), bc((T)0.08),
                       bc((T)0.69), bc((T)0.11), bc((T)0.08), bc((T)0.11), bc((T)0.09), bc((T)0.67),
                       bc((T)0.08), bc((T)0.09), bc((T)0.08), bc((T)0.09), bc((T)0.14), bc((T)0.73),
                       bc((T)0.14), bc((T)0.09), bc((T)0.04), bc((T)0.04), bc((T)0.54), bc((T)0.14),
                       bc((T)0.11), bc((T)0.25), bc((T)0.05), bc((T)0.05), bc((T)0.08), bc((T)0.45),
                       bc((T)0.08), bc((T)0.15), bc((T)0.04), bc((T)0.04), bc((T)0.09), bc((T)0.47),
                       bc((T)0.09), bc((T)0.18), bc((T)0.05), bc((T)0.05), bc((T)0.14), bc((T)0.11),
                       bc((T)0.55), bc((T)0.25), bc((T)0.08), bc((T)0.08), bc((T)0.09), bc((T)0.08),
                       bc((T)0.17)};
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
        csr_val = {bc((T)3.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)5.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)7.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)11.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)13.00), bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)17.00), bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)19.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)23.00), bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)29.00), bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)31.00), bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)37.00), bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)41.00), bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)43.00), bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)47.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)53.00), bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)59.00), bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)61.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)67.00), bc((T)1.00), bc((T)1.00),  bc((T)1.00),  bc((T)1.00),
                   bc((T)1.00), bc((T)1.00),  bc((T)71.00)};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        break;
    case sample_gmres_mat_03:
        // this special matrix data tests GMRES where
        // HH = 0.0, which means residual vector is already being orthogonal
        // to the previous Krylov subspace vectors, in the very first iteration and thus
        // the initial x0 is the best solution
        //"bcsstm05.mtx"
        n   = 153;
        m   = 153;
        nnz = 153;
        csr_row_ptr
            = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
               16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
               32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
               48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
               64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
               80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
               96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
               112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
               128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
               144, 145, 146, 147, 148, 149, 150, 151, 152, 153};
        csr_col_ind
            = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
               17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
               34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
               51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,
               68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
               85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101,
               102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
               119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
               136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152};
        csr_val
            = {0.16, 0.16, 0.16, 0.08, 0.08, 0.08, 0.15, 0.15, 0.15, 0.17, 0.17, 0.17, 0.08, 0.08,
               0.08, 0.15, 0.15, 0.15, 0.17, 0.17, 0.17, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14,
               0.14, 0.14, 0.14, 0.14, 0.14, 0.09, 0.09, 0.09, 0.20, 0.20, 0.20, 0.22, 0.22, 0.22,
               0.09, 0.09, 0.09, 0.20, 0.20, 0.20, 0.22, 0.22, 0.22, 0.16, 0.16, 0.16, 0.16, 0.16,
               0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.07, 0.07, 0.07, 0.28, 0.28, 0.28, 0.30,
               0.30, 0.30, 0.07, 0.07, 0.07, 0.28, 0.28, 0.28, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
               0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.31, 0.31, 0.31, 0.31, 0.31,
               0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34,
               0.34, 0.34, 0.34, 0.34, 0.34, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40,
               0.44, 0.44, 0.44, 0.58, 0.58, 0.58, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63,
               0.63, 0.93, 0.93, 0.93, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        break;
    case S5_sym_fullmatrix:
        n           = 5;
        m           = 5;
        nnz         = 13;
        csr_row_ptr = {0, 3, 6, 7, 10, 13};
        csr_col_ind = {0, 1, 3, 0, 1, 4, 2, 0, 3, 4, 1, 3, 4};
        //one-base test case
        transform(csr_row_ptr.begin(),
                  csr_row_ptr.end(),
                  csr_row_ptr.begin(),
                  [base](aoclsparse_int &d) { return d + base; });
        transform(csr_col_ind.begin(),
                  csr_col_ind.end(),
                  csr_col_ind.begin(),
                  [base](aoclsparse_int &d) { return d + base; });

        csr_val = {bc((T)0.10000000000000001),
                   bc((T)0.4081472627313027),
                   bc((T)0.6498582641797952),
                   bc((T)0.4081472627313027),
                   bc((T)0.10000000000000001),
                   bc((T)-0.86888798575545645),
                   bc((T)1.3257997547570228),
                   bc((T)0.6498582641797952),
                   bc((T)-0.45905550575269161),
                   bc((T)-0.26103653234662899),
                   bc((T)-0.86888798575545645),
                   bc((T)-0.26103653234662899),
                   bc((T)0.10000000000000001)};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
        break;
    case S5_sym_lowerfill:
        n           = 5;
        m           = 5;
        nnz         = 9;
        csr_row_ptr = {0, 1, 3, 4, 6, 9};
        csr_col_ind = {0, 0, 1, 2, 0, 3, 1, 3, 4};
        csr_val     = {bc((T)0.10000000000000001),
                       bc((T)0.4081472627313027),
                       bc((T)0.10000000000000001),
                       bc((T)1.3257997547570228),
                       bc((T)0.6498582641797952),
                       bc((T)-0.45905550575269161),
                       bc((T)-0.86888798575545645),
                       bc((T)-0.26103653234662899),
                       bc((T)0.10000000000000001)};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        break;
    case S5_sym_upperfill:
        n           = 5;
        m           = 5;
        nnz         = 9;
        csr_row_ptr = {0, 3, 5, 6, 8, 9};
        csr_col_ind = {0, 1, 3, 1, 4, 2, 3, 4, 4};
        csr_val     = {bc((T)0.10000000000000001),
                       bc((T)0.4081472627313027),
                       bc((T)0.6498582641797952),
                       bc((T)0.10000000000000001),
                       bc((T)-0.86888798575545645),
                       bc((T)1.3257997547570228),
                       bc((T)-0.45905550575269161),
                       bc((T)-0.26103653234662899),
                       bc((T)0.10000000000000001)};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        break;
    case S4_herm_rsym:
        /* Small Hermitian matrix that is symmetric for real components as well
        -154 +   0i     1 -   1i     1 +   2i     0 -   1i;
           1 +   1i   160 +   0i    -2 +   0i     3 -   2i;
           1 -   2i    -2 +   0i   142 +   0i     4 +   0i;
           0 +   1i     3 +   2i     4 +   0i   178 +   0i;
        */

        n = m = 4;
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            //real upper hermitian
            nnz         = 10;
            csr_row_ptr = {0, 4, 7, 9, 10};
            csr_col_ind = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
            csr_val     = {bcd((T)-154, (T)0),
                           bcd((T)1, (T)-1),
                           bcd((T)1, (T)2),
                           bcd((T)0, (T)-1),
                           bcd((T)160, (T)0),
                           bcd((T)-2, (T)0),
                           bcd((T)3, (T)-2),
                           bcd((T)142, (T)0),
                           bcd((T)4, (T)0),
                           bcd((T)178, (T)0)};
            aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_hermitian);
            aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        }
        else
        {
            //real lower symmetric
            nnz         = 9;
            csr_row_ptr = {0, 1, 3, 6, 9};
            csr_col_ind = {0, 0, 1, 0, 1, 2, 1, 2, 3};
            csr_val     = {-154, 1, 160, 1, -2, 142, 3, 4, 178};
            aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
            aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        }
        break;
    case invalid_mat:
        // matrix from the CG sample examples
        // symmetric, lower triangle filled
        // making matrix invalid after memory is allocated for it
        n = m       = 8;
        nnz         = 18;
        csr_row_ptr = {0, 1, 2, 5, 6, 8, 11, 15, 18};
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

    ret = aoclsparse_create_csr<T>(
        &A, base, m, n, nnz, &csr_row_ptr[0], &csr_col_ind[0], &csr_val[0]);
    if(ret != aoclsparse_status_success && verbose)
        std::cout << "Unexpected error in matrix creation" << std::endl;

    if(mid == invalid_mat)
    {
        csr_row_ptr[m] = 17;
    }
    return ret;
}

/*
    compute right exponents to calculate operand values for extreme value testing.
    This is used in overflow and underflow testing
*/
template <typename T>
aoclsparse_status compute_exponents(aoclsparse_int &op1_exp,
                                    aoclsparse_int &op2_exp,
                                    bool            is_underflow = false,
                                    bool            is_negative  = false,
                                    aoclsparse_int  ou_range     = 0)
{
    /*
        FLT_MAX (maximum 32 bit value for double data type) = 3.402823e+38
            exponent = 38
        DBL_MAX (maximum 64 bit value for double data type) = 1.797693e+308
            exponent = 308
    */
    const aoclsparse_int exponents[2] = {38, 308};
    aoclsparse_int       distance, exp, ouf_sn_rng_id = 0, expected_exp_range = 0;

    //minimum distance from MIN for underflow to trigger
    distance = 1 + 8 * sizeof(tolerance_t<T>) / sizeof(float);

    if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, float>)
    {
        //use exponents[0] = 38
        exp = exponents[0];
    }
    else if constexpr(std::is_same_v<T, std::complex<double>> || std::is_same_v<T, double>)
    {
        //use exponents[1] = 308
        exp = exponents[1];
    }
    /*
        [1 BIT: overflow=0/underflow=1][1 BIT: sign=0/1][1 BIT: within-bound=0/out-of-range=1]
    */
    ouf_sn_rng_id = (((aoclsparse_int)is_underflow & 0x1) << 2)
                    | (((aoclsparse_int)is_negative & 0x1) << 1) | (ou_range & 0x1);
    /*
        EG:
        FLT_MAX = 3.402823e+38
        op1 * op2 = 10 ^ (19 + 20) = 10 ^ 39
            where 39 > 38 and thus product should cross FLT_MAX boundary and enter Inf
    */
    switch(ouf_sn_rng_id)
    {
    //0 0 0
    case 0:
        //positive, overflow, edge-within-bound
        expected_exp_range = exp + 2;
        op1_exp            = expected_exp_range / 2;
        op2_exp            = expected_exp_range - op1_exp;
        break;
    //0 0 1
    case 1:
        //positive, overflow, outof-bound
        expected_exp_range = exp - 11;
        op1_exp            = expected_exp_range / 2;
        op2_exp            = expected_exp_range - op1_exp;
        break;
    //0 1 0
    case 2:
        //negative, overflow, edge-within-bound
        expected_exp_range = exp + 2;
        op1_exp            = expected_exp_range / 2;
        op2_exp            = expected_exp_range - op1_exp;
        break;
    //0 1 1
    case 3:
        //negative, overflow, outof-bound
        expected_exp_range = exp - 11;
        op1_exp            = expected_exp_range / 2;
        op2_exp            = expected_exp_range - op1_exp;
        break;
    //1 0 0
    case 4:
        //positive, underflow, edge-within-bound
        expected_exp_range = exp + 1;
        op1_exp            = expected_exp_range / 2;
        op2_exp            = expected_exp_range - op1_exp;
        break;
    //1 0 1
    case 5:
        //positive, underflow, outof-bound
        expected_exp_range = exp - distance - 11;
        op1_exp            = expected_exp_range / 2;
        op2_exp            = expected_exp_range - op1_exp;
        break;
    //1 1 0
    case 6:
        //negative, underflow, edge-within-bound
        expected_exp_range = exp + 1;
        op1_exp            = expected_exp_range / 2;
        op2_exp            = expected_exp_range - op1_exp;
        break;
    //1 1 1
    case 7:
        //negative, underflow, outof-bound
        expected_exp_range = exp - distance - 11;
        op1_exp            = expected_exp_range / 2;
        op2_exp            = expected_exp_range - op1_exp;
        break;
    default:
        std::cerr << "Invalid overflow/underflow test case" << std::endl;
        return aoclsparse_status_internal_error;
    }
    return aoclsparse_status_success;
}
/*
    Based on whether csrv_val(operand 1) or x[](operand 2), the value and
    enums are decided for the test
    In case of overflo/underflow tests, operand values are computed using the
    exponents to come up with a test that is
        1. within boundary and thus the product is valid floating point
        2. well outside boundary. product is either infinity or zero
*/
template <typename T>
aoclsparse_status lookup_special_value(T              &operand,
                                       aoclsparse_int &id,
                                       aoclsparse_int  which_operand = 0,
                                       aoclsparse_int  ou_range      = 0)
{
    tolerance_t<T> temp;
    aoclsparse_int op1_exponent = 0, op2_exponent = 0; //initialize
    switch(id)
    {
    case ET_ZERO:
        operand = aoclsparse_numeric::zero<T>();
        break;
    case ET_NAN:
        operand = aoclsparse_numeric::quiet_NaN<T>();
        break;
    case ET_INF:
        operand = aoclsparse_numeric::infinity<T>();
        break;
    case ET_NUM:
        //do nothing. Since it is a number, no special value like
        // (Inf, NaN, Zero) to assign
        break;
    case ET_POVRFLOW:
        temp = aoclsparse_numeric::maximum<tolerance_t<T>>();
        compute_exponents<T>(op1_exponent, op2_exponent, false, false, ou_range);
        if(which_operand == 1)
        {
            temp = temp / pow(10, op1_exponent);
        }
        else if(which_operand == 2)
        {
            temp = temp / pow(10, op2_exponent);
        }
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            operand = {static_cast<tolerance_t<T>>(temp), static_cast<tolerance_t<T>>(temp)};
        }
        else
        {
            operand = temp;
        }
        break;
    case ET_NOVRFLOW:
        temp = -aoclsparse_numeric::maximum<tolerance_t<T>>();
        compute_exponents<T>(op1_exponent, op2_exponent, false, true, ou_range);
        if(which_operand == 1)
        {
            temp = temp / pow(10, op1_exponent);
        }
        else if(which_operand == 2)
        {
            temp = temp / pow(10, op2_exponent);
        }
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            operand = {static_cast<tolerance_t<T>>(temp), static_cast<tolerance_t<T>>(temp)};
        }
        else
        {
            operand = temp;
        }
        break;
    case ET_PUNDRFLOW:
        temp = aoclsparse_numeric::minimum<tolerance_t<T>>();
        compute_exponents<T>(op1_exponent, op2_exponent, true, false, ou_range);
        if(which_operand == 1)
        {
            temp = temp * pow(10, op1_exponent);
        }
        else if(which_operand == 2)
        {
            temp = temp * pow(10, op2_exponent);
        }
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            operand = {static_cast<tolerance_t<T>>(temp), static_cast<tolerance_t<T>>(temp)};
        }
        else
        {
            operand = temp;
        }
        break;
    case ET_NUNDRFLOW:
        temp = -aoclsparse_numeric::minimum<tolerance_t<T>>();
        compute_exponents<T>(op1_exponent, op2_exponent, true, true, ou_range);
        if(which_operand == 1)
        {
            temp = temp * pow(10, op1_exponent);
        }
        else if(which_operand == 2)
        {
            temp = -temp * pow(10, op2_exponent);
        }
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            operand = {static_cast<tolerance_t<T>>(temp), static_cast<tolerance_t<T>>(temp)};
        }
        else
        {
            operand = temp;
        }
        break;
    default:
        std::cerr << "Invalid Special case id" << std::endl;
        return aoclsparse_status_internal_error;
    }
    return aoclsparse_status_success;
}
/*
    based on input value enums, the operands are assigned to csr_val[] and x[]buffers
    overflow and underflow (both positive and negative cases) are assigned with
    appropriate operands for multiplication
*/
template <typename T>
aoclsparse_status assign_special_values(T              &op1_csr_val,
                                        T              &op2_x_vec,
                                        aoclsparse_int &spl_op_csrval, //input
                                        aoclsparse_int &spl_op_x, //input
                                        aoclsparse_int &ou_range) //range
{
    /*
        assign special values to operands for NaN*Num=NaN, Inf*Num=Inf, Inf*Zero=NaN
        and for Real Positive/negative overflow/underflow cases
    */
    if((spl_op_csrval == ET_NAN && spl_op_x == ET_NUM) //NaN*Num=NaN
       || (spl_op_csrval == ET_INF && spl_op_x == ET_NUM) //Inf*Num=Inf
       || (spl_op_csrval == ET_INF && spl_op_x == ET_ZERO) //Inf*Zero=NaN
       || (spl_op_csrval == ET_POVRFLOW && spl_op_x == ET_POVRFLOW)
       || (spl_op_csrval == ET_NOVRFLOW && spl_op_x == ET_NOVRFLOW)
       || (spl_op_csrval == ET_PUNDRFLOW && spl_op_x == ET_PUNDRFLOW)
       || (spl_op_csrval == ET_NUNDRFLOW && spl_op_x == ET_NUNDRFLOW))
    {
        if(lookup_special_value(op1_csr_val, spl_op_csrval, 1, ou_range)
           != aoclsparse_status_success)
        {
            return aoclsparse_status_internal_error;
        }
        if(lookup_special_value(op2_x_vec, spl_op_x, 2, ou_range) != aoclsparse_status_success)
        {
            return aoclsparse_status_internal_error;
        }
    }
    return aoclsparse_status_success;
}
/*
 * Database to create linear systems
 * Inputs:
 * matrix ID
 * iparm: some problems are parametrized and use iparm to define the shape of the problem
 * In general
 * INPUT (Expected)
 * iparm[*] = 0
 * OUTPUT
 * iparm[0] = a number to pass to optimize hints
 *
 * Custom TRSM problems A X = alpha B:
 * INPUT/OUTPUT
 * iparm[0] = starting offset X
 * iparm[1] = ldx
 * iparm[2] = starting offset B
 * iparm[3] = ldb
 * Custom SYMGS/SYMGS-MV problems to solve A.x = b:
 * INPUTS
 * iparm[6] = matrix type
 * iparm[7] = fill mode
 * iparm[8] = transpose mode;
 * Custom Extreme Value Testing options (SPMV)
 * INPUT: iparm[4] = type of special value in operand 1 in csr_val
 * INPUT: iparm[5] = type of special value in x[]
 * INPUT: iparm[6] = matrix type
 * INPUT: iparm[7] = fill mode
 * INPUT: iparm[8] = transpose mode;
 * INPUT: iparm[9] = overflow/underflow range
 * OUTPUT: dparm[0] = beta
 */
template <typename T>
aoclsparse_status
    create_linear_system(linear_system_id                id,
                         std::string                    &title,
                         aoclsparse_operation           &trans,
                         aoclsparse_matrix              &A,
                         aoclsparse_mat_descr           &descr,
                         aoclsparse_index_base           base,
                         T                              &alpha,
                         std::vector<T>                 &b,
                         std::vector<T>                 &x,
                         std::vector<T>                 &xref,
                         T                              &xtol,
                         std::vector<aoclsparse_int>    &icrowa,
                         std::vector<aoclsparse_int>    &icola,
                         std::vector<T>                 &aval,
                         std::array<aoclsparse_int, 10> &iparm, //in-out parameter def: all zeros.
                         std::array<T, 10>              &dparm,
                         aoclsparse_status              &exp_status)
{
    aoclsparse_status      status = aoclsparse_status_success;
    aoclsparse_int         n, nnz, big_m, big_n, k;
    aoclsparse_diag_type   diag;
    aoclsparse_matrix_type mattype;
    aoclsparse_fill_mode   fill_mode;
    aoclsparse_order       order;
    T                      beta;
    std::vector<T>         wcolxref, wcolb;
    aoclsparse_int         csr_value_offset = 0, x_offset = 0;
    const bool             cplx
        = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

    std::string msg;
    // Set defaults.
    title = "N/A";
    trans = aoclsparse_operation_none;
    A     = nullptr;
    descr = nullptr;
    alpha = (T)1;
    b.resize(0);
    x.resize(0);
    xref.resize(0);
    xtol = (T)0; // By default not used, set only for ill conditioned problems
    icrowa.resize(0);
    icola.resize(0);
    aval.resize(0);
    std::fill(dparm.begin(), dparm.end(), 0);
    // iparm is expected to be correctly initialized by the caller
    exp_status = aoclsparse_status_success;

    switch(id)
    {
    case D7_Lx_aB:
    case D7_LL_Ix_aB:
    case D7_LTx_aB:
    case D7_LL_ITx_aB:
    case D7_LHx_aB:
    case D7_LL_IHx_aB:
    case D7_Ux_aB:
    case D7_UU_Ix_aB:
    case D7_UTx_aB:
    case D7_UU_ITx_aB:
    case D7_UHx_aB:
    case D7_UU_IHx_aB:
    case D7Lx_aB_hint:
        switch(id)
        {
        case D7Lx_aB_hint:
            title     = "Lx=b  SV hint";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_Lx_aB: // diag test set
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

            Note
            ====
            For complex types, we add 1/10 scaled entries to the nz real part, so e.g.
            b  = [1+.1i, -2-.2i, 8+.8i, ...], this applies to A as well.

            */
            title     = "diag: Lx = alpha*b" + title;
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_Ix_aB:
            title     = "diag: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_LTx_aB:
            title     = "diag: L^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_ITx_aB:
            title     = "diag: [tril(L,-1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_LHx_aB:
            title     = "diag: L^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_LL_IHx_aB:
            title     = "diag: [tril(L,-1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_Ux_aB:
            title     = "diag: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_Ix_aB:
            title     = "diag: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_UTx_aB:
            title     = "diag: U^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_ITx_aB:
            title     = "diag: [triu(U,1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        case D7_UHx_aB:
            title     = "diag: U^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            break;
        case D7_UU_IHx_aB:
            title     = "diag: [triu(U,1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        alpha = bc((T)-9.845233);
        n     = 7;
        nnz   = 7;
        b.resize(n);
        b = {bc((T)1), bc(-(T)2), bc((T)8), bc((T)5), bc(-(T)1), bc((T)11), bc((T)3)};
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0);
        xref.resize(n);
        icrowa.resize(n + 1);
        icrowa = {0, 1, 2, 3, 4, 5, 6, 7};
        icola.resize(nnz);
        icola = {0, 1, 2, 3, 4, 5, 6};

        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)-2), bc((T)-4), bc((T)3), bc((T)5), bc((T)-7), bc((T)9), bc((T)4)};
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);

        if(diag == aoclsparse_diag_type_unit)
        {
            xref = {bc((T)1), bc(-(T)2), bc((T)8), bc((T)5), bc(-(T)1), bc((T)11), bc((T)3)};
        }
        else
        {
            xref = {-(T)0.5, (T)0.5, (T)(8. / 3.), (T)1.0, (T)(1. / 7.), (T)(11. / 9.), (T)0.75};
            if constexpr(cplx)
                if(trans == aoclsparse_operation_conjugate_transpose)
                    xref = {(T){-99. / 202, -10. / 101},
                            (T){99. / 202, 10. / 101},
                            (T){264. / 101, 160. / 303},
                            (T){99. / 101, 20. / 101},
                            (T){99. / 707, 20. / 707},
                            (T){121. / 101, 220. / 909},
                            (T){297. / 404, 15. / 101}};
        }
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](T &d) { return alpha * d; });
        break;

    case S7_Lx_aB: // small m test set
    case S7_LL_Ix_aB:
    case S7_LTx_aB:
    case S7_LL_ITx_aB:
    case S7_LHx_aB:
    case S7_LL_IHx_aB:
    case S7_Ux_aB:
    case S7_UU_Ix_aB:
    case S7_UTx_aB:
    case S7_UU_ITx_aB:
    case S7_UHx_aB:
    case S7_UU_IHx_aB:
        switch(id)
        {
        case S7_Lx_aB: // small m test set
            /*
             * Solve a   Ax = b  nnz=34
             *        0  1  2  3  4  5  6   #cols #row start #row end  #idiag  #iurow
             * A  = [-2  1  0  0  3  7 -1;   5     0          4         0       1
             *        2 -4  1  2  0  4  0;   5     5          9         6       7
             *        0  6 -2  9  1  0  9;   6    10         14        11      12
             *       -9  0  1 -2  1  1  1;   6    15         20        17      18
             *        0  8  2  1 -2  2  0;   5    21         25        24      25
             *        8  0  4  3  0  7  0;   4    26         29        29      30
             *        0  0  3  6  9  0  2];  4    30         33        33      34
             *
             * b = [1.0 -2.0 0.0 2.0 -1.0 0.0 3.0]'
             */
            title     = "small m: Lx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {
                -(T)0.5, (T)0.25, (T)0.75, (T)1.625, (T)3.0625, -(T)0.553571428571, -(T)18.28125};
            break;
        case S7_LL_Ix_aB:
            title     = "small m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)1, (T)-4, (T)24, (T)-13, (T)-4, (T)-65, (T)45};
            else
                xref = {(T){1., 0.1},
                        (T){-3.98, -0.6},
                        (T){23.52, 5.988},
                        (T){-12.0112, -6.34},
                        (T){-4.1052, -1.25488},
                        (T){-65.4732, -12.33664},
                        (T){38.317008, 35.51532}};
            break;
        case S7_LTx_aB:
            title     = "small m: L^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)2.03125, (T)34.59375, (T)13.0625, (T)7.125, (T)7.25, (T)0, (T)1.5};
            break;
        case S7_LL_ITx_aB:
            title     = "small m: [tril(L,-1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)85, (T)12, (T)35, (T)12, (T)-28, (T)0, (T)3};
            else
                xref = {{70.980624, 65.90608},
                        {13.2184, -7.18438},
                        {34.5773, 8.737},
                        {11.36, 4.873},
                        {-27.73, -5.5},
                        {0, 0},
                        {3., 0.3}};
            break;

        case S7_LHx_aB:
            title     = "small m: L^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)2.03125, (T)34.59375, (T)13.0625, (T)7.125, (T)7.25, (T)0, (T)1.5};
            else
                xref = {{1.99102722, 0.40222772},
                        {33.9087252, 6.85024752},
                        {12.8038366, 2.58663366},
                        {6.98391089, 1.41089108},
                        {7.10643564, 1.43564356},
                        {0, 0},
                        {1.47029702, 0.29702970}};
            break;
        case S7_LL_IHx_aB:
            title     = "small m: [tril(L,-1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)85, (T)12, (T)35, (T)12, (T)-28, (T)0, (T)3};
            else
                xref = {{82.625776, -50.54544},
                        {11.534, +9.65962},
                        {35.6227, -1.717},
                        {12.1, -2.527},
                        {-28.27, -0.1},
                        {0, 0},
                        {3., 0.3}};
            break;

        case S7_Ux_aB:
            title     = "small m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            xref = {(T)0.625, (T)2.25, (T)7.0, (T)0.0, (T)0.5, (T)0.0, (T)1.5};
            break;
        case S7_UU_Ix_aB:
            title     = "small m: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)-17, (T)24, (T)-26, (T)0, (T)-1, (T)0, (T)3};
            else
                xref = {{-16.1158, -7.29182},
                        {23.6782, 6.224},
                        {-26.1, -3.418},
                        {0.02, -0.2},
                        {-1.0, -0.1},
                        {0, 0},
                        {3.0, 0.3}};
            break;
        case S7_UTx_aB:
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
                    (T)2.6785714e-1,
                    (T)2.96875e-1};
            break;
        case S7_UU_ITx_aB:
            title     = "small m: [triu(U,1) + I]'x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)1, (T)-3, (T)3, (T)-19, (T)12, (T)0, (T)-4};
            else
                xref = {{1.0, 0.1},
                        {-2.99, -0.4},
                        {2.95, 0.699},
                        {-18.0209, -7.348},
                        {10.436, 7.45609},
                        {2.775318, -6.45329},
                        {-4.6448, 0.70409}};
            break;

        case S7_UHx_aB:
            title     = "small m: U^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)-5.0e-1,
                        (T)3.75e-1,
                        (T)1.875e-1,
                        (T)2.1875e-1,
                        (T)-4.6875e-2,
                        (T)2.6785714e-1,
                        (T)2.96875e-1};
            else
                xref = {
                    {-4.9009900e-1, -9.9009900e-2},
                    {3.6757425e-1, +7.4257425e-2},
                    {1.8378712e-1, +3.7128712e-2},
                    {2.1441831e-1, +4.3316831e-2},
                    {-4.5946782e-2, -9.2821782e-3},
                    {2.6255304e-1, +5.3041018e-2},
                    {2.9099628e-1, +5.8787128e-2},
                };
            break;
        case S7_UU_IHx_aB:
            title     = "small m: [triu(U,1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            xref.resize(7);
            if constexpr(!cplx)
                xref = {(T)1, (T)-3, (T)3, (T)-19, (T)12, (T)0, (T)-4};
            else
                xref = {{1., 0.1},
                        {-3.01, -0.2},
                        {3.03, -0.101},
                        {-19.1191, 3.634},
                        {11.7058, -5.24191},
                        {1.442482, 6.87507},
                        {-4.4134, -1.60991}};
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        alpha = bc((T)1.33340000);
        transform(xref.begin(), xref.end(), xref.begin(), [alpha](T &d) { return alpha * d; });
        n   = 7;
        nnz = 34;
        b.resize(n);
        b = {bc((T)1), bc(-(T)2), bc((T)0), bc((T)2), -bc((T)1), bc((T)0), bc((T)3)};
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0);
        icrowa.resize(n + 1);
        icrowa = {0, 5, 10, 15, 21, 26, 30, 34};
        icola.resize(nnz);
        icola = {0, 1, 4, 5, 6, 0, 1, 2, 3, 5, 1, 2, 3, 4, 6, 0, 2,
                 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 2, 3, 5, 2, 3, 4, 6};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)-2), bc((T)1),  bc((T)3), bc((T)7),  bc(-(T)1), bc((T)2), bc((T)-4),
                bc((T)1),  bc((T)2),  bc((T)4), bc((T)6),  bc((T)-2), bc((T)9), bc((T)1),
                bc((T)9),  bc(-(T)9), bc((T)1), bc((T)-2), bc((T)1),  bc((T)1), bc((T)1),
                bc((T)8),  bc((T)2),  bc((T)1), bc((T)-2), bc((T)2),  bc((T)8), bc((T)4),
                bc((T)3),  bc((T)7),  bc((T)3), bc((T)6),  bc((T)9),  bc((T)2)};
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        break;

    case N25_Lx_aB: // large m test set
    case N25_LL_Ix_aB:
    case N25_LTx_aB:
    case N25_LHx_aB:
    case N25_LL_ITx_aB:
    case N25_LL_IHx_aB:
    case N25_Ux_aB:
    case N25_UU_Ix_aB:
    case N25_UTx_aB:
    case N25_UHx_aB:
    case N25_UU_ITx_aB:
    case N25_UU_IHx_aB:
        b.resize(25);
        alpha = (T)2.0;
        switch(id)
        {
        case N25_Lx_aB: // large m test set
            title     = "large m: Lx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIL(A) \ x = alpha b
            if constexpr(!cplx)
                b = {(T)3.1361606, (T)3.186481,  (T)3.2156856, (T)3.3757991, (T)3.4331077,
                     (T)3.4378236, (T)3.4479576, (T)3.7813114, (T)3.6439652, (T)3.8630548,
                     (T)3.6942673, (T)4.0985743, (T)3.5603244, (T)4.0383875, (T)4.1604922,
                     (T)3.9736567, (T)4.1527048, (T)3.9825762, (T)4.5399151, (T)4.2421458,
                     (T)4.997713,  (T)4.5352989, (T)4.7214001, (T)5.0616227, (T)4.8763871};
            else
                b = {{3.104799, 0.62723212},  {3.1546162, 0.6372962},  {3.1835287, 0.64313711},
                     {3.3420411, 0.67515983}, {3.3987766, 0.68662153}, {3.4034454, 0.68756473},
                     {3.413478, 0.68959152},  {3.7434983, 0.75626228}, {3.6075256, 0.72879304},
                     {3.8244243, 0.77261097}, {3.6573247, 0.73885347}, {4.0575886, 0.81971486},
                     {3.5247211, 0.71206487}, {3.9980036, 0.80767749}, {4.1188873, 0.83209844},
                     {3.9339201, 0.79473133}, {4.1111777, 0.83054096}, {3.9427504, 0.79651523},
                     {4.4945159, 0.90798302}, {4.1997243, 0.84842916}, {4.9477359, 0.9995426},
                     {4.4899459, 0.90705979}, {4.6741861, 0.94428002}, {5.0110064, 1.0123245},
                     {4.8276232, 0.97527742}};
            break;
        case N25_LL_Ix_aB:
            title     = "large m: [tril(L,-1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            // [TRIL(A,-1)+I] \ x = alpha b
            if constexpr(!cplx)
                b = {(T)1.5,       (T)1.5961924, (T)1.627927,  (T)1.7501192, (T)1.8427591,
                     (T)1.8526063, (T)1.8360447, (T)2.2656105, (T)2.0948749, (T)2.2219455,
                     (T)2.0955758, (T)2.5173541, (T)2.0557853, (T)2.4322676, (T)2.5737137,
                     (T)2.4639234, (T)2.5900785, (T)2.3540759, (T)2.9555803, (T)2.6225588,
                     (T)3.4008447, (T)2.9154333, (T)3.1143417, (T)3.4302724, (T)3.3440599};
            else
                b = {{1.5, 0.15},
                     {1.5952305, 0.16923849},
                     {1.6266477, 0.1755854},
                     {1.747618, 0.20002383},
                     {1.8393315, 0.21855183},
                     {1.8490802, 0.22052125},
                     {1.8326843, 0.21720894},
                     {2.2579544, 0.30312211},
                     {2.0889261, 0.26897498},
                     {2.214726, 0.29438909},
                     {2.0896201, 0.26911516},
                     {2.5071805, 0.35347082},
                     {2.0502275, 0.26115707},
                     {2.4229449, 0.33645351},
                     {2.5629766, 0.36474275},
                     {2.4542841, 0.34278467},
                     {2.5791777, 0.36801571},
                     {2.3455351, 0.32081518},
                     {2.9410245, 0.44111606},
                     {2.6113332, 0.37451175},
                     {3.3818363, 0.53016895},
                     {2.901279, 0.43308666},
                     {3.0981983, 0.47286834},
                     {3.4109697, 0.53605448},
                     {3.3256193, 0.51881197}};
            break;
        case N25_LTx_aB:
            title     = "large m: L^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIL(A)^T \ x = alpha b
            if constexpr(!cplx)
                b = {(T)4.7999156, (T)4.4620748, (T)4.7557297, (T)4.6435574, (T)4.861375,
                     (T)4.9847636, (T)4.712793,  (T)4.0188244, (T)4.0008485, (T)4.2367363,
                     (T)4.0991821, (T)3.9070598, (T)3.7680278, (T)3.940451,  (T)3.5216086,
                     (T)3.712464,  (T)3.7282857, (T)3.7685651, (T)3.6239308, (T)3.481001,
                     (T)3.2874493, (T)3.4135264, (T)3.2454106, (T)3.1509047, (T)3.0323272};
            else
                b = {{4.7519164, 0.95998312}, {4.417454, 0.89241496},  {4.7081724, 0.95114593},
                     {4.5971218, 0.92871148}, {4.8127613, 0.97227501}, {4.934916, 0.99695273},
                     {4.6656651, 0.9425586},  {3.9786362, 0.80376488}, {3.96084, 0.8001697},
                     {4.194369, 0.84734727},  {4.0581903, 0.81983642}, {3.8679892, 0.78141196},
                     {3.7303476, 0.75360557}, {3.9010465, 0.7880902},  {3.4863926, 0.70432173},
                     {3.6753394, 0.74249281}, {3.6910029, 0.74565715}, {3.7308794, 0.75371301},
                     {3.5876915, 0.72478616}, {3.446191, 0.6962002},   {3.2545748, 0.65748986},
                     {3.3793912, 0.68270529}, {3.2129565, 0.64908212}, {3.1193956, 0.63018094},
                     {3.0020039, 0.60646544}};
            break;
        case N25_LHx_aB:
            title     = "large m: L^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            if constexpr(!cplx) // TRIL(A)^T \ x = alpha b
                b = {(T)4.7999156, (T)4.4620748, (T)4.7557297, (T)4.6435574, (T)4.861375,
                     (T)4.9847636, (T)4.712793,  (T)4.0188244, (T)4.0008485, (T)4.2367363,
                     (T)4.0991821, (T)3.9070598, (T)3.7680278, (T)3.940451,  (T)3.5216086,
                     (T)3.712464,  (T)3.7282857, (T)3.7685651, (T)3.6239308, (T)3.481001,
                     (T)3.2874493, (T)3.4135264, (T)3.2454106, (T)3.1509047, (T)3.0323272};
            else // TRIL(A)^H \ x = alpha b
                b = {(T)4.84791474, (T)4.50669554, (T)4.80328696, (T)4.68999295, (T)4.90998879,
                     (T)5.03461128, (T)4.75992095, (T)4.05901265, (T)4.04085700, (T)4.27910371,
                     (T)4.14017391, (T)3.94613041, (T)3.80570812, (T)3.97985550, (T)3.55682474,
                     (T)3.74958869, (T)3.76556859, (T)3.80625072, (T)3.66017008, (T)3.51581102,
                     (T)3.32032377, (T)3.44766171, (T)3.27786473, (T)3.18241373, (T)3.06265049};
            break;
        case N25_LL_ITx_aB:
            title     = "large m: [tril(L,-1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            // [TRIL(A,-1)+I]^T \ x = alpha b
            if constexpr(!cplx)
                b = {(T)3.163755,  (T)2.8717862, (T)3.1679711, (T)3.0178774, (T)3.2710265,
                     (T)3.3995463, (T)3.1008801, (T)2.5031235, (T)2.4517582, (T)2.595627,
                     (T)2.5004906, (T)2.3258396, (T)2.2634888, (T)2.3343311, (T)1.9348302,
                     (T)2.2027308, (T)2.1656595, (T)2.1400648, (T)2.039596,  (T)1.861414,
                     (T)1.690581,  (T)1.7936608, (T)1.6383522, (T)1.5195545, (T)1.5};
            else
                b = {{3.1471174, 0.48275099},
                     {2.8580684, 0.42435725},
                     {3.1512914, 0.48359423},
                     {3.0026986, 0.45357548},
                     {3.2533162, 0.5042053},
                     {3.3805508, 0.52990925},
                     {3.0848713, 0.47017603},
                     {2.4930923, 0.35062471},
                     {2.4422406, 0.34035164},
                     {2.5846707, 0.36912539},
                     {2.4904857, 0.35009811},
                     {2.3175812, 0.31516792},
                     {2.2558539, 0.30269776},
                     {2.3259878, 0.31686622},
                     {1.9304819, 0.23696604},
                     {2.1957034, 0.29054615},
                     {2.1590029, 0.28313189},
                     {2.1336641, 0.27801296},
                     {2.0342, 0.2579192},
                     {1.8577998, 0.2222828},
                     {1.6886752, 0.1881162},
                     {1.7907242, 0.20873216},
                     {1.6369687, 0.17767044},
                     {1.5193589, 0.15391089},
                     {1.5, 0.15}};
            break;
        case N25_LL_IHx_aB:
            title     = "large m: [tril(L,-1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_lower;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            if constexpr(!cplx) // [TRIL(A,-1)+I]^T \ x = alpha b
                b = {(T)3.163755,  (T)2.8717862, (T)3.1679711, (T)3.0178774, (T)3.2710265,
                     (T)3.3995463, (T)3.1008801, (T)2.5031235, (T)2.4517582, (T)2.595627,
                     (T)2.5004906, (T)2.3258396, (T)2.2634888, (T)2.3343311, (T)1.9348302,
                     (T)2.2027308, (T)2.1656595, (T)2.1400648, (T)2.039596,  (T)1.861414,
                     (T)1.690581,  (T)1.7936608, (T)1.6383522, (T)1.5195545, (T)1.5};
            else // [TRIL(A,-1)+I]^H \ x = alpha b
                b = {{3.1803925, 0.15}, {2.8855041, 0.15}, {3.1846508, 0.15}, {3.0330562, 0.15},
                     {3.2887368, 0.15}, {3.4185417, 0.15}, {3.1168889, 0.15}, {2.5131548, 0.15},
                     {2.4612758, 0.15}, {2.6065832, 0.15}, {2.5104955, 0.15}, {2.334098, 0.15},
                     {2.2711237, 0.15}, {2.3426744, 0.15}, {1.9391785, 0.15}, {2.2097581, 0.15},
                     {2.1723161, 0.15}, {2.1464654, 0.15}, {2.044992, 0.15},  {1.8650281, 0.15},
                     {1.6924868, 0.15}, {1.7965974, 0.15}, {1.6397357, 0.15}, {1.51975, 0.15},
                     {1.5, 0.15}};
            break;
        case N25_Ux_aB:
            title     = "large m: Ux = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIU(A) \ x = alpha b
            if constexpr(!cplx)
                b = {(T)4.8346637, (T)4.951536,  (T)4.6879249, (T)4.9721482, (T)4.7600031,
                     (T)5.0265474, (T)4.2555835, (T)4.481163,  (T)4.1084789, (T)4.373474,
                     (T)4.3306394, (T)3.6651722, (T)4.0562577, (T)3.9586845, (T)4.1683247,
                     (T)3.8959224, (T)3.7310783, (T)3.8032438, (T)3.5453961, (T)3.442597,
                     (T)3.4147358, (T)3.3051947, (T)3.2346656, (T)3.1676077, (T)3.0323272};
            else
                b = {{4.7863171, 0.96693274}, {4.9020207, 0.99030721}, {4.6410456, 0.93758497},
                     {4.9224268, 0.99442965}, {4.7124031, 0.95200062}, {4.9762819, 1.0053095},
                     {4.2130276, 0.85111669}, {4.4363514, 0.8962326},  {4.0673941, 0.82169578},
                     {4.3297393, 0.8746948},  {4.287333, 0.86612788},  {3.6285204, 0.73303443},
                     {4.0156952, 0.81125155}, {3.9190976, 0.79173689}, {4.1266415, 0.83366495},
                     {3.8569632, 0.77918449}, {3.6937675, 0.74621566}, {3.7652114, 0.76064877},
                     {3.5099422, 0.70907923}, {3.408171, 0.6885194},   {3.3805884, 0.68294715},
                     {3.2721428, 0.66103895}, {3.2023189, 0.64693312}, {3.1359316, 0.63352154},
                     {3.0020039, 0.60646544}};
            break;
        case N25_UU_Ix_aB:
            title     = "large m: [triu(U,1) + I]x = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_none;
            diag      = aoclsparse_diag_type_unit;
            // [TRIU(A,+1)+I] \ x = alpha b
            if constexpr(!cplx)
                b = {(T)3.1985031, (T)3.3612475, (T)3.1001663, (T)3.3464683, (T)3.1696546,
                     (T)3.44133,   (T)2.6436706, (T)2.9654621, (T)2.5593886, (T)2.7323646,
                     (T)2.7319479, (T)2.0839519, (T)2.5517187, (T)2.3525646, (T)2.5815463,
                     (T)2.3861891, (T)2.168452,  (T)2.1747436, (T)1.9610614, (T)1.82301,
                     (T)1.8178675, (T)1.6853291, (T)1.6276072, (T)1.5362574, (T)1.5};
            else
                b = {{3.1815181, 0.48970062},
                     {3.342635, 0.52224949},
                     {3.0841647, 0.47003326},
                     {3.3280036, 0.51929366},
                     {3.152958, 0.48393092},
                     {3.4219167, 0.538266},
                     {2.6322339, 0.37873412},
                     {2.9508075, 0.44309242},
                     {2.5487947, 0.36187771},
                     {2.720041, 0.39647293},
                     {2.7196284, 0.39638957},
                     {2.0781124, 0.26679039},
                     {2.5412015, 0.36034374},
                     {2.3440389, 0.32051291},
                     {2.5707308, 0.36630926},
                     {2.3773272, 0.32723783},
                     {2.1617675, 0.2836904},
                     {2.1679961, 0.28494871},
                     {1.9564508, 0.24221228},
                     {1.8197799, 0.21460199},
                     {1.8146888, 0.2135735},
                     {1.6834758, 0.18706582},
                     {1.6263311, 0.17552144},
                     {1.5358949, 0.15725149},
                     {1.5, 0.15}};
            break;
        case N25_UTx_aB:
            title     = "large m: U^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            // TRIU(A)^T \ x = alpha b
            if constexpr(!cplx)
                b = {(T)3.1361606, (T)3.1561547, (T)3.2234122, (T)3.3076284, (T)3.3438639,
                     (T)3.4040321, (T)3.5559743, (T)3.6570364, (T)3.6873402, (T)3.7903524,
                     (T)3.9234929, (T)3.7647271, (T)3.769945,  (T)4.2276095, (T)3.9546192,
                     (T)4.6078492, (T)4.4359968, (T)4.6314379, (T)4.6641532, (T)5.0164253,
                     (T)4.8013899, (T)4.8843545, (T)4.7333244, (T)4.7972265, (T)4.7288635};
            else
                b = {{3.104799, 0.62723212},  {3.1245931, 0.63123093}, {3.1911781, 0.64468245},
                     {3.2745522, 0.66152569}, {3.3104253, 0.66877278}, {3.3699918, 0.68080643},
                     {3.5204145, 0.71119485}, {3.620466, 0.73140728},  {3.6504668, 0.73746803},
                     {3.7524489, 0.75807048}, {3.884258, 0.78469858},  {3.7270798, 0.75294541},
                     {3.7322456, 0.753989},   {4.1853334, 0.84552189}, {3.915073, 0.79092383},
                     {4.5617707, 0.92156984}, {4.3916368, 0.88719936}, {4.5851235, 0.92628757},
                     {4.6175117, 0.93283064}, {4.966261, 1.0032851},   {4.753376, 0.96027797},
                     {4.8355109, 0.9768709},  {4.6859911, 0.94666487}, {4.7492542, 0.9594453},
                     {4.6815749, 0.9457727}};
            break;
        case N25_UHx_aB:
            title     = "large m: U^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_non_unit;
            if constexpr(!cplx) // TRIU(A)^T \ x = alpha b
                b = {(T)3.1361606, (T)3.1561547, (T)3.2234122, (T)3.3076284, (T)3.3438639,
                     (T)3.4040321, (T)3.5559743, (T)3.6570364, (T)3.6873402, (T)3.7903524,
                     (T)3.9234929, (T)3.7647271, (T)3.769945,  (T)4.2276095, (T)3.9546192,
                     (T)4.6078492, (T)4.4359968, (T)4.6314379, (T)4.6641532, (T)5.0164253,
                     (T)4.8013899, (T)4.8843545, (T)4.7333244, (T)4.7972265, (T)4.7288635};
            else // TRIU(A)^H \ x = alpha b
                b = {(T)3.16752222, (T)3.18771621, (T)3.25564636, (T)3.34070473, (T)3.37730255,
                     (T)3.43807245, (T)3.59153401, (T)3.69360674, (T)3.72421357, (T)3.82825594,
                     (T)3.96272784, (T)3.80237433, (T)3.80764446, (T)4.26988556, (T)3.99416536,
                     (T)4.65392767, (T)4.48035677, (T)4.67775224, (T)4.71079475, (T)5.06658952,
                     (T)4.84940375, (T)4.93319802, (T)4.78065760, (T)4.84519875, (T)4.77615216};
            break;
        case N25_UU_ITx_aB:
            title     = "large m: [triu(U,1) + I]^Tx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_transpose;
            diag      = aoclsparse_diag_type_unit;
            // [TRIU(A,+1)+I]^T \ x = alpha b
            if constexpr(!cplx)
                b = {(T)1.5,       (T)1.5658661, (T)1.6356537, (T)1.6819485, (T)1.7535154,
                     (T)1.8188148, (T)1.9440614, (T)2.1413355, (T)2.1382498, (T)2.149243,
                     (T)2.3248014, (T)2.1835068, (T)2.265406,  (T)2.6214895, (T)2.3678407,
                     (T)3.0981159, (T)2.8733705, (T)3.0029376, (T)3.0798184, (T)3.3968382,
                     (T)3.2045216, (T)3.2644888, (T)3.1262659, (T)3.1658763, (T)3.1965363};
            else
                b = {{1.5, 0.15},
                     {1.5652074, 0.16317322},
                     {1.6342972, 0.17713074},
                     {1.680129, 0.18638969},
                     {1.7509802, 0.20070308},
                     {1.8156266, 0.21376295},
                     {1.9396208, 0.23881228},
                     {2.1349221, 0.2782671},
                     {2.1318673, 0.27764997},
                     {2.1427506, 0.27984861},
                     {2.3165534, 0.31496028},
                     {2.1766718, 0.28670137},
                     {2.2577519, 0.30308119},
                     {2.6102747, 0.37429791},
                     {2.3591623, 0.32356814},
                     {3.0821347, 0.46962318},
                     {2.8596368, 0.42467411},
                     {2.9879082, 0.45058752},
                     {3.0640203, 0.46596369},
                     {3.3778698, 0.52936765},
                     {3.1874763, 0.49090431},
                     {3.2468439, 0.50289777},
                     {3.1100033, 0.47525319},
                     {3.1492175, 0.48317525},
                     {3.1795709, 0.48930726}};
            break;
        case N25_UU_IHx_aB:
            title     = "large m: [triu(U,1) + I]^Hx = alpha*b";
            fill_mode = aoclsparse_fill_mode_upper;
            trans     = aoclsparse_operation_conjugate_transpose;
            diag      = aoclsparse_diag_type_unit;
            if constexpr(!cplx) // [TRIU(A,+1)+I]^T \ x = alpha b
                b = {(T)1.5,       (T)1.5658661, (T)1.6356537, (T)1.6819485, (T)1.7535154,
                     (T)1.8188148, (T)1.9440614, (T)2.1413355, (T)2.1382498, (T)2.149243,
                     (T)2.3248014, (T)2.1835068, (T)2.265406,  (T)2.6214895, (T)2.3678407,
                     (T)3.0981159, (T)2.8733705, (T)3.0029376, (T)3.0798184, (T)3.3968382,
                     (T)3.2045216, (T)3.2644888, (T)3.1262659, (T)3.1658763, (T)3.1965363};
            else // [TRIU(A,+1)+I]^H \ x = alpha b
                b = {{1.5, 0.15},       {1.5665248, 0.15}, {1.6370102, 0.15}, {1.683768, 0.15},
                     {1.7560505, 0.15}, {1.8220029, 0.15}, {1.948502, 0.15},  {2.1477489, 0.15},
                     {2.1446323, 0.15}, {2.1557355, 0.15}, {2.3330494, 0.15}, {2.1903419, 0.15},
                     {2.27306, 0.15},   {2.6327044, 0.15}, {2.3765191, 0.15}, {3.114097, 0.15},
                     {2.8871042, 0.15}, {3.017967, 0.15},  {3.0956166, 0.15}, {3.4158066, 0.15},
                     {3.2215668, 0.15}, {3.2821337, 0.15}, {3.1425286, 0.15}, {3.182535, 0.15},
                     {3.2135017, 0.15}};
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        // === START PART 1 Content autogenerated = make_trsvmat_b.m START ===
        n   = 25;
        nnz = 565;
        // === END PART 1 Content autogenerated = make_trsvmat_b.m END ===
        xref.resize(n);
        std::fill(xref.begin(), xref.end(), (T)3.0);
        x.resize(n);
        std::fill(x.begin(), x.end(), (T)0.0);
        icrowa.resize(n + 1);
        icola.resize(nnz);
        aval.resize(nnz);
        // === START PART 2 Content autogenerated = make_trsvmat_b.m START ===
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
        aval  = {(T)2.09077374, (T)0.04391073, (T)0.04037834, (T)0.02612067, (T)0.04732957,
                 (T)0.06528987, (T)0.03409377, (T)0.01417883, (T)0.02313281, (T)0.06966413,
                 (T)0.05219750, (T)0.07436849, (T)0.06642035, (T)0.02096635, (T)0.08536398,
                 (T)0.09015785, (T)0.03194168, (T)0.02949518, (T)0.04793079, (T)0.09548889,
                 (T)0.03258194, (T)0.05993188, (T)0.05477309, (T)0.02661871, (T)0.06412829,
                 (T)2.06019238, (T)0.05005746, (T)0.09517831, (T)0.01766596, (T)0.01743365,
                 (T)0.07471561, (T)0.05321011, (T)0.04903184, (T)0.03846841, (T)0.03050719,
                 (T)0.06036992, (T)0.01180341, (T)0.05769229, (T)0.02145119, (T)0.08678571,
                 (T)0.02775075, (T)0.02499010, (T)0.09620239, (T)0.07705960, (T)0.08973039,
                 (T)0.08498417, (T)0.07964857, (T)0.09609462, (T)0.05793688, (T)0.02734780,
                 (T)2.05850569, (T)0.05767273, (T)0.01846547, (T)0.03907892, (T)0.08975531,
                 (T)0.07780116, (T)0.07509255, (T)0.08296988, (T)0.08129907, (T)0.01645204,
                 (T)0.09245670, (T)0.05893480, (T)0.07302568, (T)0.04358678, (T)0.04469032,
                 (T)0.01785212, (T)0.05831480, (T)0.08339813, (T)0.03895424, (T)0.01697685,
                 (T)0.05985614, (T)0.05025507, (T)0.05663490, (T)2.08378665, (T)0.04634200,
                 (T)0.08670018, (T)0.04252953, (T)0.09163347, (T)0.08422832, (T)0.04215457,
                 (T)0.02198648, (T)0.03188594, (T)0.07733250, (T)0.09235560, (T)0.06113743,
                 (T)0.02803702, (T)0.06523582, (T)0.09887345, (T)0.08603304, (T)0.08129097,
                 (T)0.05959284, (T)0.03273917, (T)0.04441168, (T)0.05647884, (T)0.07812804,
                 (T)0.01411877, (T)0.07381025, (T)0.06244903, (T)2.06023235, (T)0.02465400,
                 (T)0.03243171, (T)0.09138908, (T)0.06844232, (T)0.08755165, (T)0.09311017,
                 (T)0.05925158, (T)0.01355292, (T)0.04593215, (T)0.02631154, (T)0.04425167,
                 (T)0.08204778, (T)0.04307994, (T)0.06897342, (T)0.07885457, (T)0.04817361,
                 (T)0.09909779, (T)0.04087351, (T)0.06512364, (T)0.06767424, (T)0.01066261,
                 (T)0.09730985, (T)0.05942414, (T)2.05681158, (T)0.07319138, (T)0.02105568,
                 (T)0.08172491, (T)0.02461240, (T)0.08609462, (T)0.07860446, (T)0.09776448,
                 (T)0.04337716, (T)0.03509689, (T)0.08234870, (T)0.08752213, (T)0.08772559,
                 (T)0.06570205, (T)0.08171715, (T)0.09923191, (T)0.08666116, (T)0.08146845,
                 (T)0.08032089, (T)0.02881627, (T)0.02701135, (T)0.01128081, (T)0.08680126,
                 (T)0.07012012, (T)2.07460859, (T)0.06633452, (T)0.03027692, (T)0.02249302,
                 (T)0.05064716, (T)0.01106338, (T)0.03090903, (T)0.06230078, (T)0.06284175,
                 (T)0.06389900, (T)0.03378182, (T)0.01704478, (T)0.07341236, (T)0.08219659,
                 (T)0.06047072, (T)0.02855574, (T)0.06621948, (T)0.02638756, (T)0.09985003,
                 (T)0.02579213, (T)0.07990810, (T)0.08825378, (T)0.09353681, (T)0.09667862,
                 (T)2.01046725, (T)0.09258652, (T)0.03366170, (T)0.09935408, (T)0.05188379,
                 (T)0.05025819, (T)0.09453112, (T)0.08619486, (T)0.02343119, (T)0.06375560,
                 (T)0.05912321, (T)0.07732350, (T)0.02599300, (T)0.09900412, (T)0.05207746,
                 (T)0.06779640, (T)0.03214791, (T)0.09326843, (T)0.04104519, (T)0.03272975,
                 (T)0.09588411, (T)0.08077796, (T)0.02072990, (T)2.03272689, (T)0.01567945,
                 (T)0.06295875, (T)0.02236562, (T)0.05878420, (T)0.06035869, (T)0.04029226,
                 (T)0.07097701, (T)0.04526654, (T)0.07471676, (T)0.03485489, (T)0.07382112,
                 (T)0.09348647, (T)0.03006073, (T)0.02263655, (T)0.03863734, (T)0.07284162,
                 (T)0.04073123, (T)0.06912000, (T)0.09856783, (T)0.03914256, (T)0.02984625,
                 (T)0.09241015, (T)2.09407292, (T)0.09721636, (T)0.01704300, (T)0.02144765,
                 (T)0.09614223, (T)0.08135449, (T)0.06566343, (T)0.06439262, (T)0.06485830,
                 (T)0.08053572, (T)0.07320127, (T)0.06242242, (T)0.07668488, (T)0.02061405,
                 (T)0.05227499, (T)0.06471903, (T)0.02878251, (T)0.08404055, (T)0.02715183,
                 (T)0.06552423, (T)0.03407719, (T)0.01450639, (T)0.02597382, (T)2.06579435,
                 (T)0.01505039, (T)0.05299549, (T)0.04055540, (T)0.02457671, (T)0.09994050,
                 (T)0.03579789, (T)0.08634282, (T)0.04782615, (T)0.08928474, (T)0.06572785,
                 (T)0.02939611, (T)0.09114714, (T)0.06608592, (T)0.07657146, (T)0.03432247,
                 (T)0.05031438, (T)0.04599930, (T)0.06582127, (T)0.07785931, (T)0.08743179,
                 (T)0.08422025, (T)0.05810171, (T)0.01650092, (T)0.06907023, (T)0.08859443,
                 (T)2.05414682, (T)0.04638696, (T)0.03901442, (T)0.04011880, (T)0.08030885,
                 (T)0.01991078, (T)0.01079420, (T)0.06244956, (T)0.03102600, (T)0.02601979,
                 (T)0.01023678, (T)0.02303515, (T)0.06240540, (T)0.04337962, (T)0.02656144,
                 (T)0.09580308, (T)0.06526411, (T)0.03846964, (T)0.03864026, (T)2.00302603,
                 (T)0.07394240, (T)0.09671779, (T)0.06829638, (T)0.06696414, (T)0.05400376,
                 (T)0.05167050, (T)0.02791243, (T)0.05112525, (T)0.09797018, (T)0.04914760,
                 (T)0.06339536, (T)0.02245613, (T)0.08762153, (T)0.05376251, (T)0.09231551,
                 (T)0.09586983, (T)0.09980409, (T)0.04270442, (T)0.05135901, (T)0.03838703,
                 (T)0.03723164, (T)2.07074661, (T)0.06311469, (T)0.09566913, (T)0.08867069,
                 (T)0.03418988, (T)0.08102404, (T)0.06174401, (T)0.06244787, (T)0.03298625,
                 (T)0.01326038, (T)0.01686786, (T)0.01840157, (T)0.07552113, (T)0.01014024,
                 (T)0.07753066, (T)0.09602185, (T)0.05143221, (T)0.03194798, (T)0.04948350,
                 (T)0.03669790, (T)0.04593919, (T)0.05033461, (T)0.04084196, (T)0.06832395,
                 (T)0.03328524, (T)0.04830874, (T)2.05785230, (T)0.05209454, (T)0.06547174,
                 (T)0.09930253, (T)0.08180502, (T)0.04806767, (T)0.09880571, (T)0.08533096,
                 (T)0.09440344, (T)0.09574924, (T)0.05572878, (T)0.01621505, (T)0.04467451,
                 (T)0.05303066, (T)0.07997375, (T)0.07903903, (T)0.04705687, (T)0.05499467,
                 (T)0.03815641, (T)0.08747533, (T)0.02270457, (T)0.01331255, (T)0.03627146,
                 (T)0.01398194, (T)2.00648886, (T)0.06955278, (T)0.09085033, (T)0.09980214,
                 (T)0.09032994, (T)0.02443237, (T)0.05803462, (T)0.09991658, (T)0.05787400,
                 (T)0.02675780, (T)0.03672458, (T)0.03683969, (T)0.04098114, (T)0.06360458,
                 (T)0.08399171, (T)0.04366732, (T)0.09898645, (T)0.07298343, (T)0.06818030,
                 (T)0.08365233, (T)0.07034969, (T)2.04175084, (T)0.07813689, (T)0.07605426,
                 (T)0.07559373, (T)0.07293096, (T)0.01176305, (T)0.05468230, (T)0.05705068,
                 (T)0.01942281, (T)0.09834169, (T)0.09004424, (T)0.02683593, (T)0.02496351,
                 (T)0.04043255, (T)0.01943640, (T)0.02023616, (T)0.03208487, (T)0.05890288,
                 (T)0.01583810, (T)0.06163846, (T)0.08062913, (T)2.08566685, (T)0.05751474,
                 (T)0.07280763, (T)0.07758564, (T)0.08910963, (T)0.03726282, (T)0.07522263,
                 (T)0.04032595, (T)0.04052489, (T)0.08704129, (T)0.06155397, (T)0.09613535,
                 (T)0.01891930, (T)0.06817169, (T)0.02830823, (T)0.06412720, (T)0.07913742,
                 (T)0.04833279, (T)0.06416750, (T)0.08726795, (T)0.07058048, (T)0.08228388,
                 (T)0.01339266, (T)0.01080381, (T)0.04963847, (T)2.05622318, (T)0.08404672,
                 (T)0.03825738, (T)0.09928940, (T)0.03663767, (T)0.04914308, (T)0.03289106,
                 (T)0.02544694, (T)0.02206353, (T)0.02544478, (T)0.01149985, (T)0.06236985,
                 (T)0.03831232, (T)0.05577997, (T)0.08759868, (T)0.02755643, (T)0.01486695,
                 (T)0.04765429, (T)0.07202891, (T)0.04789970, (T)0.01119167, (T)0.09350937,
                 (T)0.03945597, (T)0.03280223, (T)2.07972469, (T)0.02702375, (T)0.01413034,
                 (T)0.04274179, (T)0.04638098, (T)0.08506312, (T)0.09080870, (T)0.09499486,
                 (T)0.04179676, (T)0.07881016, (T)0.05837755, (T)0.09923804, (T)0.07092092,
                 (T)0.01111529, (T)0.08432953, (T)0.08376871, (T)0.06574820, (T)0.04492859,
                 (T)0.04223124, (T)0.03956827, (T)0.05093601, (T)0.04387681, (T)0.09013035,
                 (T)0.09609362, (T)0.07955621, (T)2.06457886, (T)0.05584714, (T)0.09394850,
                 (T)0.06211601, (T)0.02193982, (T)0.02432012, (T)0.01141585, (T)0.09460019,
                 (T)0.05765404, (T)0.08199949, (T)0.06098509, (T)0.01229933, (T)0.01006737,
                 (T)0.05185637, (T)0.08294399, (T)0.09769179, (T)0.05505617, (T)0.03296707,
                 (T)0.03839105, (T)0.01375664, (T)0.09555001, (T)0.06840983, (T)0.03171797,
                 (T)2.07991043, (T)0.03151611, (T)0.06928449, (T)0.02275212, (T)0.09865699,
                 (T)0.05611091, (T)0.09084405, (T)0.03271674, (T)0.06539162, (T)0.04406103,
                 (T)0.05284084, (T)0.06977812, (T)0.09016333, (T)0.04432702, (T)0.05283902,
                 (T)0.01388691, (T)0.07854750, (T)0.03217368, (T)0.01613700, (T)0.07977081,
                 (T)0.06167762, (T)0.03463541, (T)0.06166919, (T)2.07137227, (T)0.03003964,
                 (T)0.05503182, (T)0.03024827, (T)0.09156941, (T)0.01391510, (T)0.09025232,
                 (T)0.09441320, (T)0.09724469, (T)0.08210779, (T)0.04851376, (T)0.04512955,
                 (T)0.09452913, (T)0.05561656, (T)0.09276520, (T)0.09517233, (T)0.09930465,
                 (T)0.08250400, (T)0.06024601, (T)0.06766611, (T)0.04565020, (T)2.08756682,
                 (T)0.02417163, (T)0.07232965, (T)0.08508686, (T)0.06263633, (T)0.08628786,
                 (T)0.02464998, (T)0.05341119, (T)0.04462073, (T)0.07855638, (T)0.02792770,
                 (T)0.09382215, (T)0.01294497, (T)0.02412206, (T)0.02292053, (T)0.05929903,
                 (T)0.04886780, (T)0.06971889, (T)0.05329406, (T)0.02240817, (T)0.03204271,
                 (T)0.06070061, (T)0.08845448, (T)0.09223481, (T)0.01303630, (T)2.02155148};
        // === END PART 2 Content autogenerated = make_trsvmat_b.m END ===
        //update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        // add scaled complex part to xref, and aval;
        transform(aval.begin(), aval.end(), aval.begin(), [](T &v) { return bc(v); });
        transform(xref.begin(), xref.end(), xref.begin(), [](T &v) { return bc(v); });

        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        break;

    case A_nullptr:
        title      = "Invalid matrix A (ptr NULL)";
        A          = nullptr;
        exp_status = aoclsparse_status_invalid_pointer;
        break;

    case D1_descr_nullptr:
        title = "eye(1) with null descriptor";
        n = nnz = 1;
        icrowa.resize(2);
        icola.resize(1);
        aval.resize(1);
        icrowa[0] = 0;
        icrowa[1] = 1;
        icola[0]  = 0;
        aval[0]   = 1.0;
        //update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr      = nullptr;
        exp_status = aoclsparse_status_invalid_pointer;
        break;

    case D1_neg_num_hint:
        title = "eye(1) with valid descriptor but negative expected_no_of_calls";
        n = nnz = 1;
        icrowa.resize(2);
        icola.resize(1);
        aval.resize(1);
        icrowa[0] = 0;
        icrowa[1] = 1;
        icola[0]  = 0;
        aval[0]   = 1.0;
        //update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        iparm[0]   = -10;
        exp_status = aoclsparse_status_invalid_value;
        break;

    case D1_mattype_gen_hint:
        title = "eye(1) with matrix type set to general";
        n = nnz = 1;
        icrowa.resize(2);
        icola.resize(1);
        aval.resize(1);
        icrowa[0] = 0;
        icrowa[1] = 1;
        icola[0]  = 0;
        aval[0]   = 1.0;
        //update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type = aoclsparse_matrix_type_general;
        aoclsparse_set_mat_index_base(descr, base);
        exp_status = aoclsparse_status_success;
        break;
    case SM_S7_XB910:
    case SM_S7_XB1716:
        /*
         * Small sparse 7x7 matrix A used to solve multiple RHS vectors
         * Targeted for testing TRSM triangle(A) X = alpha B with L or U
         * "X" and "B" are purposefully larger to test strides and layouts.
         * The actual sub-matrices X and B are a window of the larger XBIG
         * and BIGB matrices, and
         * OUT PARAM: iparm[0] = indicates where X (window starts in XBIG)
         * OUT PARAM: iparm[1] = ldx
         * OUT PARAM: iparm[2] = indicates where B (window starts in BIGB)
         * OUT PARAM: iparm[3] = ldb
         * IN PARAM:  iparm[4] = indicates whether X/B dense matrices are row major(0) or column major(1)
         * IN PARAM:  iparm[5] = indicates the no of columns in X/B dense matrices
         */
        fill_mode = aoclsparse_fill_mode_lower;
        trans     = aoclsparse_operation_none;
        diag      = aoclsparse_diag_type_non_unit;
        n         = 7;
        nnz       = 34;
        alpha     = (T)1.0;
        icrowa.resize(n + 1);
        icrowa = {0, 5, 10, 15, 21, 26, 30, 34};
        icola.resize(nnz);
        icola = {0, 1, 4, 5, 6, 0, 1, 2, 3, 5, 1, 2, 3, 4, 6, 0, 2,
                 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 2, 3, 5, 2, 3, 4, 6};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        aval.resize(nnz);
        aval = {bc((T)-2), bc((T)1),  bc((T)3), bc((T)7),  bc(-(T)1), bc((T)2), bc((T)-4),
                bc((T)1),  bc((T)2),  bc((T)4), bc((T)6),  bc((T)-2), bc((T)9), bc((T)1),
                bc((T)9),  bc(-(T)9), bc((T)1), bc((T)-2), bc((T)1),  bc((T)1), bc((T)1),
                bc((T)8),  bc((T)2),  bc((T)1), bc((T)-2), bc((T)2),  bc((T)8), bc((T)4),
                bc((T)3),  bc((T)7),  bc((T)3), bc((T)6),  bc((T)9),  bc((T)2)};
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        switch(id)
        {
        case SM_S7_XB910:
            // A Sparse 7x7, X,B dense 9x10
            title = "A[7,7] X(7,k) = alpha * B(7,k), XBIG,BIGB[9,10]";
            //allocate dense matrices of size = (n+2) * (n+2) to cover all k = (2, (m-2), m, (m+2)) scenarios
            xref.resize((n + 2) * (n + 3));
            // clang-format off
                xref.assign({bc((T)1.1),bc((T)1.2),bc((T)1.3),bc((T)1.4),bc((T)1.5),bc((T)1.6),bc((T)1.7),bc((T)1.8),bc((T)1.9),bc((T)1.11),
                             bc((T)2.1),bc((T)2.2),bc((T)2.3),bc((T)2.4),bc((T)2.5),bc((T)2.6),bc((T)2.7),bc((T)2.8),bc((T)2.9),bc((T)2.11),
                             bc((T)3.1),bc((T)3.2),bc((T)3.3),bc((T)3.4),bc((T)3.5),bc((T)3.6),bc((T)3.7),bc((T)3.8),bc((T)3.9),bc((T)3.11),
                             bc((T)4.1),bc((T)4.2),bc((T)4.3),bc((T)4.4),bc((T)4.5),bc((T)4.6),bc((T)4.7),bc((T)4.8),bc((T)4.9),bc((T)4.11),
                             bc((T)5.1),bc((T)5.2),bc((T)5.3),bc((T)5.4),bc((T)5.5),bc((T)5.6),bc((T)5.7),bc((T)5.8),bc((T)5.9),bc((T)5.11),
                             bc((T)6.1),bc((T)6.2),bc((T)6.3),bc((T)6.4),bc((T)6.5),bc((T)6.6),bc((T)6.7),bc((T)6.8),bc((T)6.9),bc((T)6.11),
                             bc((T)7.1),bc((T)7.2),bc((T)7.3),bc((T)7.4),bc((T)7.5),bc((T)7.6),bc((T)7.7),bc((T)7.8),bc((T)7.9),bc((T)7.11),
                             bc((T)8.1),bc((T)8.2),bc((T)8.3),bc((T)8.4),bc((T)8.5),bc((T)8.6),bc((T)8.7),bc((T)8.8),bc((T)8.9),bc((T)8.11),
                             bc((T)9.1),bc((T)9.2),bc((T)9.3),bc((T)9.4),bc((T)9.5),bc((T)9.6),bc((T)9.7),bc((T)9.8),bc((T)9.9),bc((T)9.11)});
            // clang-format on
            /*
             * starting offset to reach the window of X, and B
             */
            iparm[0] = 0; // starting_offset X
            iparm[1] = n + 3; //ldx = n+3 = 10
            iparm[2] = 0; // starting_offset B
            iparm[3] = n + 3; // ldb = n+3 as well
            x.resize((n + 2) * (n + 3));
            std::fill(x.begin(), x.end(), (T)0);
            b.resize((n + 2) * (n + 3));
            break;
        case SM_S7_XB1716:
            // A Sparse 7x7, X,B dense 17x16
            title = "A[7,7] X(7,k) = alpha * B(7,k), XBIG,BIGB[17,16]";
            big_m = 17;
            big_n = 16;
            //allocate dense matrices of size = 16 * 17 to access a window of 9x9 and cover all k (= 2, (m-2), m, (m+2)) scenarios
            xref.resize(big_m * big_n);
            // clang-format off
                xref.assign({-1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1, bc((T)1.1),bc((T)1.2),bc((T)1.3),bc((T)1.4),bc((T)1.5),bc((T)1.6),bc((T)1.7),bc((T)1.8),bc((T)1.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)2.1),bc((T)2.2),bc((T)2.3),bc((T)2.4),bc((T)2.5),bc((T)2.6),bc((T)2.7),bc((T)2.8),bc((T)2.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)3.1),bc((T)3.2),bc((T)3.3),bc((T)3.4),bc((T)3.5),bc((T)3.6),bc((T)3.7),bc((T)3.8),bc((T)3.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)4.1),bc((T)4.2),bc((T)4.3),bc((T)4.4),bc((T)4.5),bc((T)4.6),bc((T)4.7),bc((T)4.8),bc((T)4.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)5.1),bc((T)5.2),bc((T)5.3),bc((T)5.4),bc((T)5.5),bc((T)5.6),bc((T)5.7),bc((T)5.8),bc((T)5.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)6.1),bc((T)6.2),bc((T)6.3),bc((T)6.4),bc((T)6.5),bc((T)6.6),bc((T)6.7),bc((T)6.8),bc((T)6.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)7.1),bc((T)7.2),bc((T)7.3),bc((T)7.4),bc((T)7.5),bc((T)7.6),bc((T)7.7),bc((T)7.8),bc((T)7.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)8.1),bc((T)8.2),bc((T)8.3),bc((T)8.4),bc((T)8.5),bc((T)8.6),bc((T)8.7),bc((T)8.8),bc((T)8.9), -1, -1, -1,
                             -1, -1, -1, -1, bc((T)9.1),bc((T)9.2),bc((T)9.3),bc((T)9.4),bc((T)9.5),bc((T)9.6),bc((T)9.7),bc((T)9.8),bc((T)9.9), -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1,
                             -1, -1, -1, -1,         -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1,        -1, -1, -1, -1});
            // clang-format on
            /*
             * Starting offset to reach window of XRef = (4 * big_n + stride_left(4) = 68)
             */
            iparm[0] = (4 * big_n) + 4; // starting_offset_x
            iparm[1] = big_n; // leading dimension ldx
            iparm[2] = (3 * big_n) + 2; // starting_offset_b
            iparm[3] = big_n - 5;
            x.resize(big_m * big_n);
            std::fill(x.begin(), x.end(), (T)0);
            b.resize(big_m * big_n);
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        beta  = (T)0;
        order = (aoclsparse_order)iparm[4]; //matrix layout
        k     = iparm[5]; //no of columns in dense matrix, X/B
        wcolxref.resize(n);
        wcolb.resize(n);
        //Build B[]
        std::fill(b.begin(), b.end(), (T)-1);
        if(order == aoclsparse_order_column) //col major order
        {
            for(int col = 0; col < k; col++)
            {
                //Generate 'b' for known xref[]
                ref_csrmvtrg(aoclsparse_operation_none,
                             alpha,
                             n,
                             n,
                             &aval[0],
                             &icola[0],
                             &icrowa[0],
                             fill_mode,
                             diag,
                             base,
                             &xref[iparm[0] + (col * iparm[1])],
                             beta,
                             &b[iparm[2] + (col * iparm[3])]);
            }
        }
        else if(order == aoclsparse_order_row) //row major order
        {
            for(int col = 0; col < k; col++)
            {
                aoclsparse_gthrs(n, &xref[iparm[0] + col], &wcolxref[0], iparm[1]);
                //Generate 'b' for known xref[]
                ref_csrmvtrg(aoclsparse_operation_none,
                             alpha,
                             n,
                             n,
                             &aval[0],
                             &icola[0],
                             &icrowa[0],
                             fill_mode,
                             diag,
                             base,
                             &wcolxref[0],
                             beta,
                             &wcolb[0]);
                aoclsparse_sctrs<T>(n, &wcolb[0], iparm[3], &b[iparm[2] + col], 0);
            }
        }
        break;
    case SM_S1X1_XB1X1:
    case SM_S1X1_XB1X3:
        fill_mode = aoclsparse_fill_mode_lower;
        trans     = aoclsparse_operation_none;
        diag      = aoclsparse_diag_type_non_unit;
        n         = 1;
        nnz       = 1;
        alpha     = (T)1.0;
        icrowa.resize(n + 1);
        icrowa = {0, 1};
        icola.resize(nnz);
        icola = {0};
        // update row pointer and column index arrays to reflect values as per base-index
        icrowa[0] = icrowa[0] + base;
        icrowa[1] = icrowa[1] + base;
        icola[0]  = icola[0] + base;
        aval.resize(nnz);
        aval = {bc((T)2)};
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = aoclsparse_matrix_type_triangular;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        switch(id)
        {
        case SM_S1X1_XB1X1:
            title = "very small m, A(1x1) . X(1x1) = alpha * B(1x1)";
            xref.resize(1);
            xref.assign({bc((T)1.1)});
            iparm[0] = 0; // starting_offset_x
            iparm[1] = 1; // leading dimension ldx
            iparm[2] = iparm[0]; // starting_offset_b
            iparm[3] = iparm[1]; // ldb
            x.resize(1);
            x[0] = (T)0;
            b.resize(1);
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                b[0] = bcd((T)2.178, (T)0.44);
            }
            else
            {
                b[0] = (T)2.2;
            }
            break;
        case SM_S1X1_XB1X3:
            title = "very small m with different k, A(1x1) . X(1x3) = alpha * B(1x3)";
            xref.resize(3);
            xref.assign({bc((T)1.1), bc((T)2.1), bc((T)3.1)});
            iparm[0] = 0; // starting_offset_x
            iparm[1] = 1; // leading dimension ldx
            iparm[2] = iparm[0]; // starting_offset_b
            iparm[3] = iparm[1]; // ldb
            x.resize(3);
            x[0] = (T)0;
            x[1] = (T)0;
            x[2] = (T)0;
            b.resize(3);
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                b[0] = bcd((T)2.178, (T)0.44);
                b[1] = bcd((T)4.158, (T)0.84);
                b[2] = bcd((T)6.138, (T)1.24);
            }
            else
            {
                b[0] = (T)2.2;
                b[1] = (T)4.2;
                b[2] = (T)6.2;
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    case GS_S7:
    case GS_MV_S7:
        /*
            small m = 7
            A = [ 7.00000  -0.54907   0.00000   0.00000   2.11036  -0.55651   0.00000;
                -0.54907   7.49175   0.00000   0.00000   0.00000   0.29361   0.00000;
                0.00000    0.00000  7.00000   0.00000   0.00000   0.00000   0.00000;
                0.00000    0.00000  0.00000   5.79194   0.37467   0.00000   0.00000;
                2.11036    0.00000  0.00000   0.37467   7.00000   0.00000   0.00000;
                -0.55651   0.29361   0.00000   0.00000   0.00000   7.00000   0.00000;
                0.00000    0.00000  0.00000   0.00000   0.00000   0.00000   7.00000;
                ]
        */
        title     = "small symmetric m, A.x = b, solve for x using Gauss Seidel Preconditioner";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode
        diag      = aoclsparse_diag_type_non_unit;

        alpha = (T)1;
        n     = 7;
        nnz   = 17;

        x.resize(n);
        x = {bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0)};
        icrowa.resize(n + 1);
        icrowa = {0, 4, 7, 8, 10, 13, 16, 17};
        icola.resize(nnz);
        icola = {0, 1, 4, 5, 0, 1, 5, 2, 3, 4, 0, 3, 4, 0, 1, 5, 6};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)7),
                bc((T)-0.54907),
                bc((T)2.11036),
                bc((T)-0.55651),
                bc((T)-0.54907),
                bc((T)7.49175),
                bc((T)0.29361),
                bc((T)7),
                bc((T)5.79194),
                bc((T)0.37467),
                bc((T)2.11036),
                bc((T)0.37467),
                bc((T)7),
                bc((T)-0.55651),
                bc((T)0.29361),
                bc((T)7),
                bc((T)7)};

        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = aoclsparse_matrix_type_symmetric;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        b.resize(n);
        wcolxref.resize(n);
        xref.resize(n);
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            b = {bcd((T)12.983454, (T)2.6229200000000006),
                 bcd((T)16.034129099999998, (T)3.2392179999999997),
                 bcd((T)20.789999999999999, (T)4.1999999999999993),
                 bcd((T)24.790698899999999, (T)5.008222),
                 bcd((T)38.2229496, (T)7.7218080000000002),
                 bcd((T)41.610402899999997, (T)8.4061419999999991),
                 bcd((T)48.509999999999998, (T)9.7999999999999989)};

            wcolxref = {bcd((T)1.0783550185695689, (T)0.10783550185695701),
                        bcd((T)2.0516354291649486, (T)0.20516354291649494),
                        bcd((T)3.000000, (T)0.29999999999999988),
                        bcd((T)4.0151319990701806, (T)0.40151319990701828),
                        bcd((T)4.7660777999451653, (T)0.47660777999451653),
                        bcd((T)6.0475707728300723, (T)0.60475707728300709),
                        bcd((T)7.000000, (T)0.69999999999999996)};
        }
        else
        {
            b        = {bc((T)13.114599999999999),
                        bc((T)16.196089999999998),
                        bc((T)21),
                        bc((T)25.04111),
                        bc((T)38.60904),
                        bc((T)42.030709999999999),
                        bc((T)49)};
            wcolxref = {bc((T)1.0783550185695687),
                        bc((T)2.051635429164949),
                        bc((T)3),
                        bc((T)4.0151319990701806),
                        bc((T)4.7660777999451653),
                        bc((T)6.0475707728300723),
                        bc((T)7)};
        }
        switch(id)
        {
        case GS_S7:
            xref = std::move(wcolxref);
            break;
        case GS_MV_S7:
            //xref below contains the spmv output between "A" and computed x (symgs output)
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                xref = {bcd((T)12.983454000000002, (T)2.6229200000000006),
                        bcd((T)16.388335845101036, (T)3.3107749182022301),
                        bcd((T)20.789999999999999, (T)4.1999999999999993),
                        bcd((T)24.790698899999995, (T)5.008222),
                        bcd((T)36.771185288669294, (T)7.4285222805392515),
                        bcd((T)41.911908129415643, (T)8.4670521473566929),
                        bcd((T)48.509999999999998, (T)9.7999999999999989)};
            }
            else
            {
                xref = {bc((T)13.114599999999998),
                        bc((T)16.553874591011152),
                        bc((T)21),
                        bc((T)25.041109999999996),
                        bc((T)37.142611402696254),
                        bc((T)42.335260736783475),
                        bc((T)49)};
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;

    case GS_TRIDIAG_M5:
    case GS_MV_TRIDIAG_M5:
        /*
            small m = 5
            A = [   239 3 0 0 0;
                    3 239 7 0 0;
                    0 7 239 8 0;
                    0 0 8 239 11;
                    0 0 0 11 239;
                ]
        */
        title     = "small symmetric tridiagonal m, A.x = b, solve for x using Gauss Seidel "
                    "Preconditioner";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode

        diag  = aoclsparse_diag_type_non_unit;
        xtol  = (T)0;
        alpha = (T)1;
        n     = 5;
        nnz   = 13;
        x.resize(n);
        x = {bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0)};
        icrowa.resize(n + 1);
        icrowa = {0, 2, 5, 8, 11, 13};
        icola.resize(nnz);
        icola = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)239),
                bc((T)3),
                bc((T)3),
                bc((T)239),
                bc((T)7),
                bc((T)7),
                bc((T)239),
                bc((T)8),
                bc((T)8),
                bc((T)239),
                bc((T)11),
                bc((T)11),
                bc((T)239)};
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        wcolxref.resize(n);
        xref.resize(n);
        b.resize(n);
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            b = {bcd((T)242.55000000000001, (T)49),
                 bcd((T)496.98000000000002, (T)100.40000000000001),
                 bcd((T)755.37, (T)152.59999999999999),
                 bcd((T)1024.6500000000001, (T)207),
                 bcd((T)1226.6099999999999, (T)247.80000000000001)};

            wcolxref = {bcd((T)1.0000013846416498, (T)0.10000013846416499),
                        bcd((T)1.9998896902152357, (T)0.19998896902152347),
                        bcd((T)2.9983867335418628, (T)0.29983867335418624),
                        bcd((T)3.9970789705447745, (T)0.39970789705447735),
                        bcd((T)4.9916788264986858, (T)0.49916788264986872)};
        }
        else
        {
            b        = {bc((T)245), bc((T)502), bc((T)763), bc((T)1035), bc((T)1239)};
            wcolxref = {bc((T)1.0000013846416498),
                        bc((T)1.9998896902152352),
                        bc((T)2.9983867335418632),
                        bc((T)3.997078970544774),
                        bc((T)4.9916788264986858)};
        }
        switch(id)
        {
        case GS_TRIDIAG_M5:
            xref = std::move(wcolxref);
            break;
        case GS_MV_TRIDIAG_M5:
            //xref below contains the spmv output between "A" and computed x (symgs output)
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                xref = {bcd((T)242.55000000000001, (T)49),
                        bcd((T)496.94272377765776, (T)100.39246945003183),
                        bcd((T)754.96438602324633, (T)152.51805778247399),
                        bcd((T)1023.8554605708213, (T)206.83948698400428),
                        bcd((T)1224.6093171270866, (T)247.39582164183571)};
            }
            else
            {
                xref = {bc((T)245),
                        bc((T)501.96234725015921),
                        bc((T)762.59028891237006),
                        bc((T)1034.1974349200214),
                        bc((T)1236.9791082091785)};
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    case GS_BLOCK_TRDIAG_S9:
    case GS_MV_BLOCK_TRDIAG_S9:
        /*
            Block Tridiagonal Matrix
            A = [
                      4  -1   0  -1   0   0   0   0   0;
                    - 1   4  -1   0  -1   0   0   0   0;
                      0  -1   4   0   0  -1   0   0   0;
                    - 1   0   0   4  -1   0  -1   0   0;
                      0  -1   0  -1   4  -1   0  -1   0;
                      0   0  -1   0  -1   4   0   0  -1;
                      0   0   0  -1   0   0   4  -1   0;
                      0   0   0   0  -1   0  -1   4  -1;
                      0   0   0   0   0  -1   0  -1   4;
                ]
        */
        title     = "medium symmetric block tridiagonal m, A.x = b, solve for x using Gauss Seidel "
                    "Preconditioner";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode
        diag      = aoclsparse_diag_type_non_unit;

        alpha = (T)1;
        n     = 9;
        nnz   = 33;
        x.resize(n);
        x = {bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0)};
        icrowa.resize(n + 1);
        icrowa = {0, 3, 7, 10, 14, 19, 23, 26, 30, 33};
        icola.resize(nnz);
        icola = {0, 1, 3, 0, 1, 2, 4, 1, 2, 5, 0, 3, 4, 6, 1, 3, 4,
                 5, 7, 2, 4, 5, 8, 3, 6, 7, 4, 6, 7, 8, 5, 7, 8};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)4),  bc((T)-1), bc((T)-1), bc((T)-1), bc((T)4),  bc((T)-1), bc((T)-1),
                bc((T)-1), bc((T)4),  bc((T)-1), bc((T)-1), bc((T)4),  bc((T)-1), bc((T)-1),
                bc((T)-1), bc((T)-1), bc((T)4),  bc((T)-1), bc((T)-1), bc((T)-1), bc((T)-1),
                bc((T)4),  bc((T)-1), bc((T)-1), bc((T)4),  bc((T)-1), bc((T)-1), bc((T)-1),
                bc((T)4),  bc((T)-1), bc((T)-1), bc((T)-1), bc((T)4)};
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        wcolxref.resize(n);
        xref.resize(n);
        b.resize(n);
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            b = {bcd((T)-1.98, (T)-0.40000000000000002),
                 bcd((T)-0.99000000000000021, (T)-0.19999999999999996),
                 bcd((T)3.96, (T)0.79999999999999982),
                 bcd((T)2.9699999999999989, (T)0.60000000000000009),
                 bcd((T)0, (T)-4.4408920985006262e-16),
                 bcd((T)6.9299999999999979, (T)1.3999999999999997),
                 bcd((T)15.84, (T)3.2000000000000002),
                 bcd((T)10.890000000000001, (T)2.2000000000000002),
                 bcd((T)21.780000000000001, (T)4.4000000000000004)};

            wcolxref = {bcd((T)0.47747802734374972, (T)0.047747802734374963),
                        bcd((T)0.99206542968749967, (T)0.099206542968749992),
                        bcd((T)2.08837890625, (T)0.20883789062499997),
                        bcd((T)2.9178466796874991, (T)0.29178466796874997),
                        bcd((T)2.8798828124999991, (T)0.28798828124999987),
                        bcd((T)4.1035156249999991, (T)0.41035156249999993),
                        bcd((T)5.79150390625, (T)0.57915039062499996),
                        bcd((T)5.916015625, (T)0.59160156249999996),
                        bcd((T)7.2265625, (T)0.72265625)};
        }
        else
        {
            b        = {bc((T)-2),
                        bc((T)-1),
                        bc((T)4),
                        bc((T)3),
                        bc((T)0),
                        bc((T)7),
                        bc((T)16),
                        bc((T)11),
                        bc((T)22)};
            wcolxref = {bc((T)0.47747802734375),
                        bc((T)0.9920654296875),
                        bc((T)2.08837890625),
                        bc((T)2.9178466796875),
                        bc((T)2.8798828125),
                        bc((T)4.103515625),
                        bc((T)5.79150390625),
                        bc((T)5.916015625),
                        bc((T)7.2265625)};
        }
        switch(id)
        {
        case GS_BLOCK_TRDIAG_S9:
            xref = std::move(wcolxref);
            break;
        case GS_MV_BLOCK_TRDIAG_S9:
            //xref below contains the spmv output between "A" and computed x (symgs output)
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                xref = {bcd((T)-1.98, (T)-0.40000000000000013),
                        bcd((T)-1.4627032470703127, (T)-0.29549560546874981),
                        bcd((T)3.225355224609376, (T)0.65158691406250013),
                        bcd((T)2.4972967529296834, (T)0.50450439453125018),
                        bcd((T)-2.3858129882812511, (T)-0.48198242187500051),
                        bcd((T)4.1770458984374965, (T)0.84384765624999991),
                        bcd((T)14.188831787109375, (T)2.8664306640625004),
                        bcd((T)7.6884521484375004, (T)1.55322265625),
                        bcd((T)18.697851562500002, (T)3.77734375)};
            }
            else
            {
                xref = {bc((T)-2),
                        bc((T)-1.47747802734375),
                        bc((T)3.2579345703125),
                        bc((T)2.52252197265625),
                        bc((T)-2.409912109375),
                        bc((T)4.21923828125),
                        bc((T)14.3321533203125),
                        bc((T)7.76611328125),
                        bc((T)18.88671875)};
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    case GS_CONVERGE_S4:
    case GS_MV_CONVERGE_S4:
        /*
            Convergence Test using SYMGS Iterations
            A = [
                    10   -1    2    0;
                    -1   11   -1    3;
                     2   -1   10   -1;
                     0    3   -1    8;
                ]
            This is the only input in linear system database to SYMGS/SYMGS_MV which runs for
            8 iterations and reaches convergence. Convergence of the precondtioner is verified for
            a simple input by setting alpha = 1, thus feeding updated x as initial-x
            after every iteration.
        */
        title     = "small symmetric m and mulitple iterations, A.x = b, solve for x using Gauss "
                    "Seidel Preconditioner";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode
        diag      = aoclsparse_diag_type_non_unit;

        alpha = (T)1;
        n     = 4;
        nnz   = 14;
        x.resize(n);
        x = {bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0)};
        icrowa.resize(n + 1);
        icrowa = {0, 3, 7, 11, 14};
        icola.resize(nnz);
        icola = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)10),
                bc((T)-1),
                bc((T)2),
                bc((T)-1),
                bc((T)11),
                bc((T)-1),
                bc((T)3),
                bc((T)2),
                bc((T)-1),
                bc((T)10),
                bc((T)-1),
                bc((T)3),
                bc((T)-1),
                bc((T)8)};
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        wcolxref.resize(n);
        xref.resize(n);
        b.resize(n);
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            b = {bcd((T)13.859999999999999, (T)2.7999999999999998),
                 bcd((T)29.699999999999999, (T)6),
                 bcd((T)25.739999999999998, (T)5.2000000000000002),
                 bcd((T)34.649999999999999, (T)7)};

            wcolxref = {bcd((T)0.99999999992700483, (T)0.099999999992700442),
                        bcd((T)2.0000000217847944, (T)0.20000000217847949),
                        bcd((T)3.0000000112573724, (T)0.30000000112573733),
                        bcd((T)3.9999999237056727, (T)0.39999999237056733)};
        }
        else
        {
            b        = {bc((T)14), bc((T)30), bc((T)26), bc((T)35)};
            wcolxref = {bc((T)0.99999999992700483),
                        bc((T)2.0000000217847949),
                        bc((T)3.0000000112573728),
                        bc((T)3.9999999237056727)};
        }
        switch(id)
        {
        case GS_CONVERGE_S4:
            xref = std::move(wcolxref);
            break;
        case GS_MV_CONVERGE_S4:
            //xref below contains the spmv output between "A" and computed x (symgs output)
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                xref = {bcd((T)13.859999999999999, (T)2.7999999999999994),
                        bcd((T)29.699999999569727, (T)5.9999999999130766),
                        bcd((T)25.740000165267894, (T)5.2000000333874539),
                        bcd((T)34.649999449304964, (T)6.99999988874848)};
            }
            else
            {
                xref = {bc((T)13.999999999999998),
                        bc((T)29.999999999565382),
                        bc((T)26.000000166937269),
                        bc((T)34.999999443742396)};
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    case GS_NONSYM_S4:
    case GS_MV_NONSYM_S4:
        /*
            Non-Symmetric and SYMGS test
            A = [
                    2 1 2 1;
                    6 -6 6 12;
                    4 3 3 -3;
                    2 2 -1 1;
                 ]
        */
        title     = "small non-symmetric m, A.x = b, solve for x using Gauss Seidel Preconditioner";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode
        diag      = aoclsparse_diag_type_non_unit;

        alpha = (T)1;
        n     = 4;
        nnz   = 16;
        x.resize(n);
        x = {bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0)};
        icrowa.resize(n + 1);
        icrowa = {0, 4, 8, 12, 16};
        icola.resize(nnz);
        icola = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        aval.resize(nnz);
        aval = {bc((T)2),
                bc((T)1),
                bc((T)2),
                bc((T)1),
                bc((T)6),
                bc((T)-6),
                bc((T)6),
                bc((T)12),
                bc((T)4),
                bc((T)3),
                bc((T)3),
                bc((T)-3),
                bc((T)2),
                bc((T)2),
                bc((T)-1),
                bc((T)1)};
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        wcolxref.resize(n);
        xref.resize(n);
        b.resize(n);
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            b = {bcd((T)6, (T)0.59999999999999998),
                 bcd((T)36, (T)3.6000000000000001),
                 bcd((T)-1, (T)-0.10000000000000001),
                 bcd((T)10, (T)1)};
            if(trans == aoclsparse_operation_none)
            {
                wcolxref = {bcd((T)-35, (T)-1.5500000000000007),
                            bcd((T)35.333333333333329, (T)1.3666666666666676),
                            bcd((T)13.666666666666668, (T)0.63333333333333353),
                            bcd((T)13.333333333333332, (T)0.46666666666666701)};
            }
            else if(trans == aoclsparse_operation_transpose)
            {
                wcolxref = {bcd((T)-406.16666666666669, (T)-8.1833333333333371),
                            bcd((T)60.5, (T)1.2166666666666686),
                            bcd((T)53.333333333333336, (T)1.1666666666666679),
                            bcd((T)121, (T)2.2000000000000011)};
            }
        }
        else
        {
            b = {bc((T)6), bc((T)36), bc((T)-1), bc((T)10)};
            if(trans == aoclsparse_operation_none)
            {
                wcolxref = {bc((T)-35),
                            bc((T)35.333333333333336),
                            bc((T)13.666666666666666),
                            bc((T)13.333333333333332)};
            }
            else if(trans == aoclsparse_operation_transpose)
            {
                wcolxref = {
                    bc((T)-406.16666666666669), bc((T)60.5), bc((T)53.333333333333336), bc((T)121)};
            }
        }
        switch(id)
        {
        case GS_NONSYM_S4:
            xref = std::move(wcolxref);
            break;
        case GS_MV_NONSYM_S4:
            //xref below contains the spmv output between "A" and computed x (symgs output)
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                if(trans == aoclsparse_operation_none)
                {
                    xref = {bcd((T)6.0000000000000036, (T)0.60000000000000087),
                            bcd((T)-179.19, (T)-26.100000000000009),
                            bcd((T)-32.839999999999989, (T)-4.8999999999999995),
                            bcd((T)0.38666666666666899, (T)-0.49999999999999889)};
                }
                else if(trans == aoclsparse_operation_transpose)
                {
                    xref = {bcd((T)5.9999999999999707, (T)0.59999999999999631),
                            bcd((T)-366.40833333333342, (T)-44.300000000000011),
                            bcd((T)-409.55666666666673, (T)-48.800000000000011),
                            bcd((T)280.32166666666666, (T)33.20000000000001)};
                }
            }
            else
            {
                if(trans == aoclsparse_operation_none)
                {
                    xref = {bc((T)6), bc((T)-180), bc((T)-33), bc((T)0.3333333333333286)};
                }
                else if(trans == aoclsparse_operation_transpose)
                {
                    xref = {bc((T)6),
                            bc((T)-367.16666666666669),
                            bc((T)-410.33333333333337),
                            bc((T)280.83333333333326)};
                }
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    case GS_HR4:
    case GS_MV_HR4:
        /*
           Hermitian and SYMGS test for complex and conjugate complex transpose cases
            A = [
                    -154    1       1       0;
                    1       160     -2      3;
                    1       -2      142     4;
                    0       3       4       178;
                 ]
            Imag = [
                0       -1      2       -1;
                1       0       0       -2;
                -2      0       0       0;
                1       2       0       0;
            ];
            A
            -154 +   0i     1 -   1i     1 +   2i     0 -   1i;
                1 +   1i   160 +   0i    -2 +   0i     3 -   2i;
                1 -   2i    -2 +   0i   142 +   0i     4 +   0i;
                0 +   1i     3 +   2i     4 +   0i   178 +   0i;
            A(complex conjugate transpose, A')
            -154 -   0i     1 -   1i     1 +   2i     0 -   1i;
                1 +   1i   160 -   0i    -2 -   0i     3 -   2i;
                1 -   2i    -2 -   0i   142 -   0i     4 -   0i;
                0 +   1i     3 +   2i     4 -   0i   178 -   0i;
            A(simple transpose, A.')
            -154 +   0i     1 +   1i     1 -   2i     0 +   1i;
                1 -   1i   160 +   0i    -2 +   0i     3 +   2i;
                1 +   2i    -2 +   0i   142 +   0i     4 +   0i;
                0 -   1i     3 -   2i     4 +   0i   178 +   0i;
        */
        title     = "small hermitian m, A.x = b, solve for x using Gauss Seidel Preconditioner";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode
        diag      = aoclsparse_diag_type_non_unit;

        alpha = (T)1;
        n     = 4;
        x.resize(n);
        x = {bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0)};
        icrowa.resize(n + 1);
        b.resize(n);
        wcolxref.resize(n);
        xref.resize(n);
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            nnz = 16;
            icola.resize(nnz);
            aval.resize(nnz);
            icrowa = {0, 4, 8, 12, 16};
            icola  = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
            //complex, since it is hermitian, data is same for non-transpose and conj-complex-transpose cases
            if(trans == aoclsparse_operation_none
               || trans == aoclsparse_operation_conjugate_transpose)
            {
                aval     = {bcd((T)-154, (T)0),
                            bcd((T)1, (T)-1),
                            bcd((T)1, (T)2),
                            bcd((T)0, (T)-1),
                            bcd((T)1, (T)1),
                            bcd((T)160, (T)0),
                            bcd((T)-2, (T)0),
                            bcd((T)3, (T)-2),
                            bcd((T)1, (T)-2),
                            bcd((T)-2, (T)0),
                            bcd((T)142, (T)0),
                            bcd((T)4, (T)0),
                            bcd((T)0, (T)1),
                            bcd((T)3, (T)2),
                            bcd((T)4, (T)0),
                            bcd((T)178, (T)0)};
                b        = {bcd((T)-149, (T)-14.9),
                            bcd((T)327.69999999999999, (T)25.700000000000003),
                            bcd((T)439.19999999999999, (T)41.899999999999999),
                            bcd((T)729.5, (T)78)};
                wcolxref = {bcd((T)1.0000169486383628, (T)0.10002305870917698),
                            bcd((T)2.0001717516140047, (T)0.2000861062380952),
                            bcd((T)3.0007408507317193, (T)0.2992537289548915),
                            bcd((T)3.9970987438291647, (T)0.40011883963381922)};
            }
            else if(trans == aoclsparse_operation_transpose)
            {
                aval     = {bcd((T)-154, (T)0),
                            bcd((T)1, (T)1),
                            bcd((T)1, (T)-2),
                            bcd((T)0, (T)1),
                            bcd((T)1, (T)-1),
                            bcd((T)160, (T)0),
                            bcd((T)-2, (T)0),
                            bcd((T)3, (T)2),
                            bcd((T)1, (T)2),
                            bcd((T)-2, (T)0),
                            bcd((T)142, (T)0),
                            bcd((T)4, (T)0),
                            bcd((T)0, (T)-1),
                            bcd((T)3, (T)-2),
                            bcd((T)4, (T)0),
                            bcd((T)178, (T)0)};
                b        = {bcd((T)-149, (T)-14.9),
                            bcd((T)326.30000000000001, (T)39.700000000000003),
                            bcd((T)438.80000000000001, (T)45.899999999999999),
                            bcd((T)730.5, (T)68)};
                wcolxref = {bcd((T)1.0000211791027867, (T)0.099980754064937963),
                            bcd((T)2.0001854013321623, (T)0.19994960905652157),
                            bcd((T)3.000578403975624, (T)0.3008781965158428),
                            bcd((T)3.9971797270471652, (T)0.39930900745381376)};
            }
        }
        else
        {
            nnz = 14;
            icola.resize(nnz);
            aval.resize(nnz);
            //real, since it is symmetric data is same for non-transpose and transpose cases
            icrowa   = {0, 3, 7, 11, 14};
            icola    = {0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3};
            aval     = {bc((T)-154),
                        bc((T)1),
                        bc((T)1),
                        bc((T)1),
                        bc((T)160),
                        bc((T)-2),
                        bc((T)3),
                        bc((T)1),
                        bc((T)-2),
                        bc((T)142),
                        bc((T)4),
                        bc((T)3),
                        bc((T)4),
                        bc((T)178)};
            b        = {bc((T)-149), bc((T)327), bc((T)439), bc((T)730)};
            wcolxref = {bc((T)1.0000053468333496),
                        bc((T)2.0001756154131849),
                        bc((T)3.000647796922669),
                        bc((T)3.9975592157387636)};
        }
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);

        switch(id)
        {
        case GS_HR4:
            xref = std::move(wcolxref);
            break;
        case GS_MV_HR4:
            //xref below contains the spmv output between "A" and computed x (symgs output)
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                if(trans == aoclsparse_operation_none
                   || trans == aoclsparse_operation_conjugate_transpose)
                {
                    xref = {bcd((T)-149, (T)-14.9),
                            bcd((T)327.71752635746162, (T)25.721468578776118),
                            bcd((T)439.29331534204954, (T)41.794321819086136),
                            bcd((T)728.98685978817491, (T)78.018787141220045)};
                }
                else if(trans == aoclsparse_operation_transpose)
                {
                    xref = {bcd((T)-149, (T)-14.900000000000002),
                            bcd((T)326.32143050459626, (T)39.682427107429682),
                            bcd((T)438.87054114103586, (T)46.022063829222404),
                            bcd((T)730.0007412064723, (T)67.879972958244679)};
                }
            }
            else
            {
                xref = {bc((T)-149),
                        bc((T)327.01948586631386),
                        bc((T)439.08187814198101),
                        bc((T)729.56865843543017)};
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    case GS_TRIANGLE_S5:
    case GS_MV_TRIANGLE_S5:
        /*
            Convergence Test using SYMGS Iterations
            A = [
                  211.00000     0.00000     0.00000     0.00000     0.00000;
                    0.00000   271.00000     0.00000     0.00000     0.00000;
                    2.50000     0.00000   311.00000     0.00000     0.00000;
                    1.00000     0.00000     1.20000   287.00000     0.00000;
                    0.50000     2.00000     3.00000     0.00000   251.00000;
                ]
        */
        title     = "small triangle m and single iterations, A.x = b, solve for x using Gauss "
                    "Seidel Preconditioner";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode
        diag      = aoclsparse_diag_type_non_unit;

        alpha = bc((T)2.0);
        n     = 5;
        nnz   = 11;
        x.resize(n);
        x = {bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0), bc((T)1.0)};
        icrowa.resize(n + 1);
        icola.resize(nnz);
        aval.resize(nnz);
        if(fill_mode == aoclsparse_fill_mode_lower)
        {
            //If Only Lower triangle is given
            icrowa = {0, 1, 2, 4, 7, 11};
            icola  = {0, 1, 0, 2, 0, 2, 3, 0, 1, 2, 4};
            aval   = {bc((T)211),
                      bc((T)271),
                      bc((T)2.5),
                      bc((T)311),
                      bc((T)1),
                      bc((T)1.2),
                      bc((T)287),
                      bc((T)0.5),
                      bc((T)2),
                      bc((T)3),
                      bc((T)251)};
        }
        else if(fill_mode == aoclsparse_fill_mode_upper)
        {
            //If Only Upper triangle is given
            icrowa = {0, 4, 6, 9, 10, 11};
            icola  = {0, 2, 3, 4, 1, 4, 2, 3, 4, 3, 4};
            aval   = {bc((T)211),
                      bc((T)2.5),
                      bc((T)1),
                      bc((T)0.5),
                      bc((T)271),
                      bc((T)2),
                      bc((T)311),
                      bc((T)1.2),
                      bc((T)3),
                      bc((T)287),
                      bc((T)251)};
        }

        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        wcolxref.resize(n);
        xref.resize(n);
        b.resize(n);
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            if((trans == aoclsparse_operation_transpose && fill_mode == aoclsparse_fill_mode_lower)
               || (trans == aoclsparse_operation_none && fill_mode == aoclsparse_fill_mode_upper))
            {
                b = {bcd((T)436.50000000000006, (T)134.55000000000001),
                     bcd((T)1070.8800000000001, (T)330.09600000000006),
                     bcd((T)1848.4319999999998, (T)569.77440000000001),
                     bcd((T)2227.1199999999999, (T)686.50400000000002),
                     bcd((T)2434.7000000000003, (T)750.49000000000001)};
            }
            else if((trans == aoclsparse_operation_none && fill_mode == aoclsparse_fill_mode_lower)
                    || (trans == aoclsparse_operation_transpose
                        && fill_mode == aoclsparse_fill_mode_upper))
            {
                b = {bcd((T)409.33999999999997, (T)126.178),
                     bcd((T)1051.48, (T)324.11600000000004),
                     bcd((T)1814.8699999999999, (T)559.42900000000009),
                     bcd((T)2236.0440000000003, (T)689.25480000000005),
                     bcd((T)2460.8900000000003, (T)758.56299999999999)};
            }
            wcolxref = {bcd((T)1.9799999999999998, (T)0.40000000000000002),
                        bcd((T)3.9600000000000004, (T)0.80000000000000004),
                        bcd((T)5.9399999999999995, (T)1.2000000000000002),
                        bcd((T)7.9200000000000008, (T)1.5999999999999999),
                        bcd((T)9.9000000000000021, (T)1.9999999999999998)};
        }
        else
        {
            if((trans == aoclsparse_operation_transpose && fill_mode == aoclsparse_fill_mode_lower)
               || (trans == aoclsparse_operation_none && fill_mode == aoclsparse_fill_mode_upper))
            {
                b = {bc((T)450), bc((T)1104), bc((T)1905.5999999999999), bc((T)2296), bc((T)2510)};
            }
            else if((trans == aoclsparse_operation_none && fill_mode == aoclsparse_fill_mode_lower)
                    || (trans == aoclsparse_operation_transpose
                        && fill_mode == aoclsparse_fill_mode_upper))
            {
                b = {bc((T)422), bc((T)1084), bc((T)1871), bc((T)2305.1999999999998), bc((T)2537)};
            }
            wcolxref = {bc((T)2), bc((T)4), bc((T)6), bc((T)8), bc((T)10)};
        }
        switch(id)
        {
        case GS_TRIANGLE_S5:
            xref = std::move(wcolxref);
            break;
        case GS_MV_TRIANGLE_S5:
            //xref below contains the spmv output between "A" and computed x (symgs output)
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                if((trans == aoclsparse_operation_transpose
                    && fill_mode == aoclsparse_fill_mode_lower)
                   || (trans == aoclsparse_operation_none
                       && fill_mode == aoclsparse_fill_mode_upper))
                {
                    xref = {bcd((T)436.50000000000006, (T)134.55000000000001),
                            bcd((T)1070.8800000000001, (T)330.09600000000006),
                            bcd((T)1848.4319999999998, (T)569.77440000000001),
                            bcd((T)2227.1199999999994, (T)686.50400000000002),
                            bcd((T)2434.7000000000007, (T)750.49000000000001)};
                }
                else if((trans == aoclsparse_operation_none
                         && fill_mode == aoclsparse_fill_mode_lower)
                        || (trans == aoclsparse_operation_transpose
                            && fill_mode == aoclsparse_fill_mode_upper))
                {
                    xref = {bcd((T)409.33999999999997, (T)126.178),
                            bcd((T)1051.48, (T)324.11600000000004),
                            bcd((T)1814.8699999999997, (T)559.42900000000009),
                            bcd((T)2236.0440000000003, (T)689.25479999999993),
                            bcd((T)2460.8900000000008, (T)758.56299999999999)};
                }
            }
            else
            {
                if((trans == aoclsparse_operation_transpose
                    && fill_mode == aoclsparse_fill_mode_lower)
                   || (trans == aoclsparse_operation_none
                       && fill_mode == aoclsparse_fill_mode_upper))
                {
                    xref = {bc((T)450),
                            bc((T)1104),
                            bc((T)1905.5999999999999),
                            bc((T)2296),
                            bc((T)2510)};
                }
                else if((trans == aoclsparse_operation_none
                         && fill_mode == aoclsparse_fill_mode_lower)
                        || (trans == aoclsparse_operation_transpose
                            && fill_mode == aoclsparse_fill_mode_upper))
                {
                    xref = {bc((T)422),
                            bc((T)1084),
                            bc((T)1871),
                            bc((T)2305.1999999999998),
                            bc((T)2537)};
                }
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    case GS_SYMM_ALPHA2_S9:
    case GS_MV_SYMM_ALPHA2_S9:
        /*
            Symmetric Matrix
            A = [
                    162 0 0 0 0 0 0 3 0;
                    0 162 0 7 9 0 0 11 0;
                    0 0 162 0 0 0 0 0 0;
                    0 7 0 162 0 0 0 0 0;
                    0 9 0 0 162 13 0 0 3;
                    0 0 0 0 13 162 0 0 5;
                    0 0 0 0 0 0 162 0 0;
                    3 11 0 0 0 0 0 162 17;
                    0 0 0 0 3 5 0 17 162;
                ]
        */
        title     = "medium symmetric m, A.x = b, solve for x using Gauss Seidel Preconditioner";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode
        diag      = aoclsparse_diag_type_non_unit;

        alpha = bc((T)2.0);
        n     = 9;
        nnz   = 25;
        x.resize(n);
        x = {bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0),
             bc((T)1.0)};
        icrowa.resize(n + 1);
        icrowa = {0, 2, 6, 7, 9, 13, 16, 17, 21, 25};
        icola.resize(nnz);
        icola = {0, 7, 1, 3, 4, 7, 2, 1, 3, 1, 4, 5, 8, 4, 5, 8, 6, 0, 1, 7, 8, 4, 5, 7, 8};

        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        aval.resize(nnz);
        aval = {bc((T)162.000000), bc((T)3.000000),  bc((T)162.000000), bc((T)7.000000),
                bc((T)9.000000),   bc((T)11.000000), bc((T)162.000000), bc((T)7.000000),
                bc((T)162.000000), bc((T)9.000000),  bc((T)162.000000), bc((T)13.000000),
                bc((T)3.000000),   bc((T)13.000000), bc((T)162.000000), bc((T)5.000000),
                bc((T)162.000000), bc((T)3.000000),  bc((T)11.000000),  bc((T)162.000000),
                bc((T)17.000000),  bc((T)3.000000),  bc((T)5.000000),   bc((T)17.000000),
                bc((T)162.000000)};
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        wcolxref.resize(n);
        xref.resize(n);
        b.resize(n);

        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            b = {bcd((T)360.83999999999997, (T)111.22800000000001),
                 bcd((T)940.90000000000009, (T)290.02999999999997),
                 bcd((T)942.83999999999992, (T)290.62799999999999),
                 bcd((T)1284.28, (T)395.87599999999998),
                 bcd((T)1810.0200000000002, (T)557.93399999999997),
                 bcd((T)2099.0799999999999, (T)647.03599999999994),
                 bcd((T)2199.9599999999996, (T)678.13199999999995),
                 bcd((T)2859.5599999999995, (T)881.45200000000011),
                 bcd((T)3179.6599999999999, (T)980.12200000000007)};

            wcolxref = {bcd((T)1.9817716906880167, (T)0.40035791731071063),
                        bcd((T)3.9740800399396567, (T)0.80284445251306147),
                        bcd((T)5.9399999999999986, (T)1.2),
                        bcd((T)7.8492318244170098, (T)1.585703398872123),
                        bcd((T)9.8185327808376517, (T)1.9835419759267974),
                        bcd((T)11.805977999201147, (T)2.3850460604446759),
                        bcd((T)13.859999999999996, (T)2.7999999999999998),
                        bcd((T)15.744328702847072, (T)3.1806724652216332),
                        bcd((T)17.626658504895179, (T)3.5609411121000374)};
        }
        else
        {
            b        = {bc((T)372),
                        bc((T)970),
                        bc((T)972),
                        bc((T)1324),
                        bc((T)1866),
                        bc((T)2164),
                        bc((T)2268),
                        bc((T)2948),
                        bc((T)3278)};
            wcolxref = {bc((T)2.0017895865535529),
                        bc((T)4.0142222625653083),
                        bc((T)6),
                        bc((T)7.9285169943606162),
                        bc((T)9.9177098796339926),
                        bc((T)11.925230302223381),
                        bc((T)14),
                        bc((T)15.903362326108159),
                        bc((T)17.804705560500182)};
        }
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        switch(id)
        {
        case GS_SYMM_ALPHA2_S9:
            xref = std::move(wcolxref);
            break;
        case GS_MV_SYMM_ALPHA2_S9:
            //xref below contains the spmv output between "A" and computed x (symgs output)
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                xref = {bcd((T)360.83999999999992, (T)111.22800000000001),
                        bcd((T)940.90000000000009, (T)290.02999999999997),
                        bcd((T)942.83999999999969, (T)290.62799999999999),
                        bcd((T)1273.1437296566455, (T)392.44327336838865),
                        bcd((T)1795.7019381299729, (T)553.52049433078525),
                        bcd((T)2085.3458332419928, (T)642.80247849418117),
                        bcd((T)2199.9599999999991, (T)678.13199999999983),
                        bcd((T)2841.3109098557261, (T)875.82676499676541),
                        bcd((T)3146.7757792500402, (T)969.98552370697143)};
            }
            else
            {
                xref = {bc((T)372),
                        bc((T)970),
                        bc((T)972),
                        bc((T)1312.5193089243769),
                        bc((T)1851.2391114741988),
                        bc((T)2149.8410651979307),
                        bc((T)2268),
                        bc((T)2929.1865050059037),
                        bc((T)3244.0987414948872)};
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        break;
    case EXT_G5:
    case EXT_G5_B0:
    case EXT_S5:
    case EXT_S5_B0:
        /*
            Symmetric Matrix
            A = [
                    211     0       2.5     1       0.5;
                    0       271     0       0       2;
                    2.5     0       311     1.2     3;
                    1       0       1.2     287     0;
                    0.5     2       3       0       251;
                ]
        */
        title   = "small symmetric m with extreme values, compute A.x = b";
        mattype = (aoclsparse_matrix_type)iparm[6]; // matrix type
        // fill mode
        if(iparm[7] == 0 || iparm[7] == 2)
        {
            fill_mode = aoclsparse_fill_mode_lower;
        }
        else if(iparm[7] == 1 || iparm[7] == 3)
        {
            fill_mode = aoclsparse_fill_mode_upper;
        }
        trans = (aoclsparse_operation)iparm[8]; //transpose mode
        //iparm[4]; //A_operand
        //iparm[5]; //x_operand

        if(iparm[7] == 2 || iparm[7] == 3)
        {
            //strictly lower or upper triangle
            diag = aoclsparse_diag_type_zero;
        }
        else
        {
            diag = aoclsparse_diag_type_non_unit;
        }

        alpha    = bc((T)2.0);
        dparm[0] = bc((T)2.0);
        n        = 5;
        nnz      = 17;

        icrowa.resize(n + 1);
        icrowa = {0, 4, 6, 10, 13, 17};
        icola.resize(nnz);
        icola = {0, 2, 3, 4, 1, 4, 0, 2, 3, 4, 0, 2, 3, 0, 1, 2, 4};

        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        aval.resize(nnz);
        aval = {bc((T)211),
                bc((T)2.5),
                bc((T)1),
                bc((T)0.5),
                bc((T)271),
                bc((T)2),
                bc((T)2.5),
                bc((T)311),
                bc((T)1.2),
                bc((T)3),
                bc((T)1),
                bc((T)1.2),
                bc((T)287),
                bc((T)0.5),
                bc((T)2),
                bc((T)3),
                bc((T)251)};
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);

        b.resize(n); //reference solution vector in matrix-vector multiplication
        xref.resize(n); //allocate solution vector for aocl-spmv operation
        //assign intial values
        xref = {bc((T)10.0), bc((T)10.0), bc((T)10.0), bc((T)10.0), bc((T)10.0)};
        b    = xref; //get the initial values into reference rhs vector
        x.resize(n); //input dense vector in aocl matrix-vector multiplication
        x = {bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0)};

        /*          _                                       _
            x-->   |_  	1	    2	    3	    4	    5   _|	    b

                    _                                       _
                   |   211	    0	    2.5	    1	    0.5  |	    225
                   |   0	    271	    0	    0	    2	 |	    552
            A-->   |   2.5	    0	    311	    1.2	    3	 |	    955.3
                   |   1	    0	    1.2	    287	    0	 |	    1152.6
                   |_  0.5	    2	    3	    0	    251 _|	    1268.5

        */
        switch(id)
        {
        case EXT_G5_B0:
            //test beta=zero case for general matrices
            dparm[0] = aoclsparse_numeric::zero<T>();
            [[fallthrough]];
        case EXT_G5:
            //matrix type = General
            if(trans == aoclsparse_operation_none)
            {
                csr_value_offset = 9; //access nnz element from row #3 in csr matrix A(2,4) = 3
                x_offset         = 4; //access 5th element from x[]   x[4] = 5
            }
            else if(trans == aoclsparse_operation_transpose)
            {
                csr_value_offset = 15; //access nnz element from row #5 in csr matrix A(4,2) = 3
                x_offset         = 4;
            }
            break;
        case EXT_S5_B0:
            //test beta=zero case for symmetric matrices
            dparm[0] = aoclsparse_numeric::zero<T>();
            [[fallthrough]];
        case EXT_S5:
            //nnz offset for the element from corresponding traingle(lower/upper)
            if((fill_mode == aoclsparse_fill_mode_lower && trans == aoclsparse_operation_transpose)
               || (fill_mode == aoclsparse_fill_mode_lower && trans == aoclsparse_operation_none))
            {
                csr_value_offset = 15;
                x_offset         = 4;
            }
            else if((fill_mode == aoclsparse_fill_mode_upper
                     && trans == aoclsparse_operation_transpose)
                    || (fill_mode == aoclsparse_fill_mode_upper
                        && trans == aoclsparse_operation_none))
            {
                csr_value_offset = 9;
                x_offset         = 4;
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        if(assign_special_values(aval[csr_value_offset] /*nnz in 3rd row*/,
                                 x[x_offset] /*5th entry in input vector x*/,
                                 iparm[4], //op1 csr_val[]
                                 iparm[5], //op1 x[]
                                 iparm[9]) // range of overflow/underflow input
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        if((id == EXT_S5 || id == EXT_S5_B0) && (mattype == aoclsparse_matrix_type_symmetric)
           && (iparm[5] == ET_ZERO))
        {
            x[2] = x[x_offset];
        }
        //Generate 'b' for known xref[]
        ref_csrmv(trans,
                  alpha,
                  n,
                  n,
                  &aval[0],
                  &icola[0],
                  &icrowa[0],
                  mattype,
                  fill_mode,
                  diag,
                  base,
                  &x[0],
                  dparm[0], //beta
                  &b[0]);
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        break;
    case EXT_H5:
    case EXT_H5_B0:
        /*
            Symmetric Matrix
            A = [
                    99      2i      0       0       0;
                    -2i     -178    3-i     0       0;
                    0       3+i     561     1-5i    9+i;
                    0       0       1+5i    711     4;
                    0       0       9-i       4     363;
                ]
            nnz_matrix =
            [
                    0	1
                    2	3	4
                        5	6	7	8
                            9	10	11
                            12	13	14
            ]
        */
        title   = "small hermitian m with extreme values, compute A.x = b";
        mattype = (aoclsparse_matrix_type)iparm[6]; // matrix type
        // fill mode
        if(iparm[7] == 0 || iparm[7] == 2)
        {
            fill_mode = aoclsparse_fill_mode_lower;
        }
        else if(iparm[7] == 1 || iparm[7] == 3)
        {
            fill_mode = aoclsparse_fill_mode_upper;
        }
        trans = (aoclsparse_operation)iparm[8]; //transpose mode
        //iparm[4]; //A_operand
        //iparm[5]; //x_operand

        if(iparm[7] == 2 || iparm[7] == 3)
        {
            //strictly lower or upper triangle
            diag = aoclsparse_diag_type_zero;
        }
        else
        {
            diag = aoclsparse_diag_type_non_unit;
        }

        alpha    = bc((T)2.0);
        dparm[0] = bc((T)2.0);
        n        = 5;
        nnz      = 15;

        icrowa.resize(n + 1);
        icrowa = {0, 2, 5, 9, 12, 15};
        icola.resize(nnz);
        icola = {0, 1, 0, 1, 2, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4};

        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        aval.resize(nnz);
        aval = {bcd((T)99, (T)0),
                bcd((T)0, (T)2),
                bcd((T)0, (T)-2),
                bcd((T)-178, (T)0),
                bcd((T)3, (T)-1),
                bcd((T)3, (T)1),
                bcd((T)561, (T)0),
                bcd((T)1, (T)-5),
                bcd((T)9, (T)1),
                bcd((T)1, (T)5),
                bcd((T)711, (T)0),
                bcd((T)4, (T)0),
                bcd((T)9, (T)-1),
                bcd((T)4, (T)0),
                bcd((T)363, (T)0)};
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);

        b.resize(n); //reference solution vector in matrix-vector multiplication
        xref.resize(n); //allocate solution vector for aocl-spmv operation
        //assign intial values
        xref = {bc((T)10.0), bc((T)10.0), bc((T)10.0), bc((T)10.0), bc((T)10.0)};
        b    = xref; //get the initial values into reference rhs vector
        x.resize(n); //input dense vector in aocl matrix-vector multiplication
        x = {bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0)};

        switch(id)
        {
        case EXT_H5_B0:
            //test beta=zero case for symmetric matrices
            dparm[0] = aoclsparse_numeric::zero<T>();
            [[fallthrough]];
        case EXT_H5:
            //nnz offset for the element from corresponding triangle(lower/upper)
            if((fill_mode == aoclsparse_fill_mode_lower
                && trans == aoclsparse_operation_conjugate_transpose)
               || (fill_mode == aoclsparse_fill_mode_lower && trans == aoclsparse_operation_none)
               || (fill_mode == aoclsparse_fill_mode_lower
                   && trans == aoclsparse_operation_transpose))
            {
                csr_value_offset = 12;
                x_offset         = 4;
            }
            else if((fill_mode == aoclsparse_fill_mode_upper
                     && trans == aoclsparse_operation_conjugate_transpose)
                    || (fill_mode == aoclsparse_fill_mode_upper
                        && trans == aoclsparse_operation_none)
                    || (fill_mode == aoclsparse_fill_mode_upper
                        && trans == aoclsparse_operation_transpose))
            {
                csr_value_offset = 8;
                x_offset         = 4;
            }
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        if(assign_special_values(aval[csr_value_offset] /*nnz in 3rd row*/,
                                 x[x_offset] /*5th entry in input vector x*/,
                                 iparm[4], //op1 csr_val[]
                                 iparm[5], //op1 x[]
                                 iparm[9]) // range of overflow/underflow input
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        if((id == EXT_H5 || id == EXT_H5_B0) && (mattype == aoclsparse_matrix_type_symmetric)
           && (iparm[5] == ET_ZERO))
        {
            x[2] = x[x_offset];
        }
        //Generate 'b' for known xref[]
        ref_csrmv(trans,
                  alpha,
                  n,
                  n,
                  &aval[0],
                  &icola[0],
                  &icrowa[0],
                  mattype,
                  fill_mode,
                  diag,
                  base,
                  &x[0],
                  dparm[0], //beta
                  &b[0]);
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        break;
    case EXT_NSYMM_5:
        /*
            Non-Symmetric General Matrix
            A = [
                    211     0       2.5     1       0.5;
                    0       271     0       0       2;
                    0       0       0       0     3;
                    1       0       1.2     287     0;
                    0.5     2       3       0       251;
                ]
        */
        title     = "small non-symmetric of size 5 to test underflow, compute A.x = b";
        mattype   = (aoclsparse_matrix_type)iparm[6]; // matrix type
        fill_mode = (aoclsparse_fill_mode)iparm[7]; // fill mode
        trans     = (aoclsparse_operation)iparm[8]; //transpose mode
        //iparm[4]; //A_operand
        //iparm[5]; //x_operand
        diag = aoclsparse_diag_type_non_unit;

        alpha    = (T)1.0;
        dparm[0] = (T)1.0;
        n        = 5;
        nnz      = 14;

        icrowa.resize(n + 1);
        icrowa = {0, 4, 6, 7, 10, 14};
        icola.resize(nnz);
        icola = {0, 2, 3, 4, 1, 4, 4, 0, 2, 3, 0, 1, 2, 4};

        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        aval.resize(nnz);
        aval = {bc((T)211),
                bc((T)2.5),
                bc((T)1),
                bc((T)0.5),
                bc((T)271),
                bc((T)2),
                bc((T)3),
                bc((T)1),
                bc((T)1.2),
                bc((T)287),
                bc((T)0.5),
                bc((T)2),
                bc((T)3),
                bc((T)251)};
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;

        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);

        b.resize(n); //reference solution vector in matrix-vector multiplication
        xref.resize(n); //allocate solution vector for aocl-spmv operation
        //assign intial values
        std::fill(xref.begin(), xref.end(), bc((T)0.0));
        b = xref; //get the initial values into reference rhs vector
        x.resize(n); //input dense vector in aocl matrix-vector multiplication
        x = {bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0)};

        /*          _                                       _
            x-->   |_  	1	    2	    3	    4	    5   _|	    b

                    _                                       _
                   |   211	    0	    2.5	    1	    0.5  |	    225
                   |   0	    271	    0	    0	    2	 |	    552
            A-->   |   0	    0	    0	    0	    3	 |	    15 (expect special result here)
                   |   1	    0	    1.2	    287	    0	 |	    1152.6
                   |_  0.5	    2	    3	    0	    251 _|	    1268.5

        */
        //matrix type = General
        if(trans == aoclsparse_operation_none)
        {
            csr_value_offset = 6; //access nnz element from row #3 in csr matrix A(2,4) = 3
            x_offset         = 4; //access 5th element from x[]   x[4] = 5
        }
        //initialize to default
        if(assign_special_values(aval[csr_value_offset] /*nnz in 3rd row*/,
                                 x[x_offset] /*5th entry in input vector x*/,
                                 iparm[4], //op1 csr_val[]
                                 iparm[5], //op1 x[]
                                 iparm[9]) // range of overflow/underflow input
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        //Generate 'b' for known xref[]
        ref_csrmv(trans,
                  alpha,
                  n,
                  n,
                  &aval[0],
                  &icola[0],
                  &icrowa[0],
                  mattype,
                  fill_mode,
                  diag,
                  base,
                  &x[0],
                  dparm[0], //beta
                  &b[0]);
        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        break;
    case H5_Lx_aB:
    case H5_LTx_aB:
    case H5_LHx_aB:
    case H5_Ux_aB:
    case H5_UTx_aB:
    case H5_UHx_aB:
        /*
            Solve a linear system:Ax = alpha.b, where n=5, nnz=11 and A is Hermitian if data
            is complex or symmetric if data is real
            xref = [1.0 2.0 3.0 4.0 5.0]';
            b = A.xref
            x = A \ b
        */
        title   = "small hermitian matrix of size 5 to test trsv";
        diag    = aoclsparse_diag_type_non_unit;
        mattype = aoclsparse_matrix_type_triangular;

        alpha = (T)1.0;
        n     = 5;
        nnz   = 11;

        icrowa.resize(n + 1);
        icola.resize(nnz);
        aval.resize(nnz);

        switch(id)
        {
        case H5_Lx_aB:
        case H5_LTx_aB:
        case H5_LHx_aB:
            /*
                    Solve H5_L.x = alpha.b, where
                    H5_L(Hermitian Lower matrix) =
                    4           0           0           0           0;
                    2 - 2i      5 + 1i      0           0           0;
                    0           1 - 2i      3 + 3i      0           0;
                    0           0           2 - 0.5i    4 + 4i      0;
                    1 - 2i      3 + 3i      0 + 2i      0           2 + 2i;
                    validate TRSV operations for CSR and corresponding CSC inputs for following cases
                        1. Lcsr * x = aB and Ucsc-T * x = aB
                        2. Ucsr * x = aB and Lcsc-T * x = aB
                        3. Lcsr-CT * x = aB and Ucsc-C * x = aB
                        4. Ucsr-CT * x = aB and Lcsc-C * x = aB
                    Input chosen is a Hermitian matrix.
                */
            fill_mode = aoclsparse_fill_mode_lower;
            icrowa    = {0, 1, 3, 5, 7, 11};
            icola     = {0, 0, 1, 1, 2, 2, 3, 0, 1, 2, 4};
            aval      = {bcd((T)4, (T)-0),
                         bcd((T)2, (T)2),
                         bcd((T)5, (T)-1),
                         bcd((T)1, (T)2),
                         bcd((T)3, (T)-3),
                         bcd((T)2, (T)0.5),
                         bcd((T)4, (T)-4),
                         bcd((T)1, (T)2),
                         bcd((T)3, (T)-3),
                         bcd((T)0, (T)-2),
                         bcd((T)2, (T)-2)};
            break;
        case H5_Ux_aB:
        case H5_UTx_aB:
        case H5_UHx_aB:
            /*
                    Solve H5_U.x = alpha.b, where
                    H5_L(Hermitian Upper matrix) =
                    4           2 + 2i      0           0           1 + 2i;
                    0           5 + 1i      1 + 2i      0           3 - 3i;
                    0           0           3 + 3i      2 + 0.5i    0 - 2i;
                    0           0           0           4 + 4i      0     ;
                    0           0           0           0           2 + 2i;
                */
            fill_mode = aoclsparse_fill_mode_upper;
            icrowa    = {0, 1, 3, 5, 7, 11};
            icola     = {0, 0, 1, 1, 2, 2, 3, 0, 1, 2, 4};
            aval      = {bcd((T)4, (T)-0),
                         bcd((T)2, (T)2),
                         bcd((T)5, (T)-1),
                         bcd((T)1, (T)2),
                         bcd((T)3, (T)-3),
                         bcd((T)2, (T)0.5),
                         bcd((T)4, (T)-4),
                         bcd((T)1, (T)2),
                         bcd((T)3, (T)-3),
                         bcd((T)0, (T)-2),
                         bcd((T)2, (T)-2)};
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        // update row pointer and column index arrays to reflect values as per base-index
        // icrowa = icrowa + base;
        transform(icrowa.begin(), icrowa.end(), icrowa.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });
        // icola = icola + base;
        transform(icola.begin(), icola.end(), icola.begin(), [base](aoclsparse_int &d) {
            return d + base;
        });

        b.resize(n); //reference solution vector in matrix-vector multiplication
        xref.resize(n); //allocate solution vector for aocl-trsv operation
        //assign intial values
        xref = {bc((T)1.0), bc((T)2.0), bc((T)3.0), bc((T)4.0), bc((T)5.0)};
        x.resize(n); // uknown vector, trsv needs to compute

        if(aoclsparse_create_csr(&A, base, n, n, nnz, &icrowa[0], &icola[0], &aval[0])
           != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        if(aoclsparse_create_mat_descr(&descr) != aoclsparse_status_success)
            return aoclsparse_status_internal_error;
        descr->type      = mattype;
        descr->fill_mode = fill_mode;
        descr->diag_type = diag;
        aoclsparse_set_mat_index_base(descr, base);
        switch(id)
        {
        case H5_Lx_aB:
        case H5_Ux_aB:
            trans = aoclsparse_operation_none;
            break;
        case H5_LTx_aB:
        case H5_UTx_aB:
            trans = aoclsparse_operation_transpose;
            break;
        case H5_LHx_aB:
        case H5_UHx_aB:
            trans = aoclsparse_operation_conjugate_transpose;
            break;
        default:
            return aoclsparse_status_internal_error;
            break;
        }
        dparm[0] = 0.; //unused
        ref_csrmvtrg(trans,
                     alpha,
                     n,
                     n,
                     &aval[0],
                     &icola[0],
                     &icrowa[0],
                     fill_mode,
                     diag,
                     base,
                     &xref[0],
                     dparm[0],
                     &b[0]);
        break;
    default:
        // no data with id found
        return aoclsparse_status_internal_error;
        break;
    }
    return status;
}

// Class object to capture all the information from
// 'aoclsparse_debug_get' function
class debug_info
{
public:
    char           *global_isa;
    char           *tl_isa;
    aoclsparse_int *sparse_nt;
    bool           *is_isa_updated;
    char           *arch;

    debug_info()
    {
        global_isa     = new char[20];
        tl_isa         = new char[20];
        sparse_nt      = new aoclsparse_int;
        is_isa_updated = new bool;
        arch           = new char[20];
    }

    ~debug_info()
    {
        delete[] global_isa;
        delete[] tl_isa;
        delete sparse_nt;
        delete is_isa_updated;
        delete[] arch;
    }
};

// Returns 'true' if blkcsrmv tests can be executed in the given build
bool can_exec_blkcsrmv();

// Initialize TCSR matrix
template <typename T>
void init_tcsr_matrix(aoclsparse_int              &m,
                      aoclsparse_int              &n,
                      aoclsparse_int              &nnz,
                      std::vector<aoclsparse_int> &row_ptr_L,
                      std::vector<aoclsparse_int> &row_ptr_U,
                      std::vector<aoclsparse_int> &col_idx_L,
                      std::vector<aoclsparse_int> &col_idx_U,
                      std::vector<T>              &val_L,
                      std::vector<T>              &val_U,
                      aoclsparse_matrix_sort       sort,
                      aoclsparse_index_base        b)
{
    //Initialize matrix
    // 1 0 2 3
    // 0 4 0 0
    // 5 0 6 0
    // 7 8 0 9
    m = 4, n = 4, nnz = 9;
    // row ptr
    row_ptr_L.assign({b, 1 + b, 2 + b, 4 + b, 7 + b});
    row_ptr_U.assign({b, 3 + b, 4 + b, 5 + b, 6 + b});
    // col idx
    if(sort == aoclsparse_fully_sorted || sort == aoclsparse_unknown_sort)
    {
        col_idx_L.assign({b, 1 + b, b, 2 + b, b, 1 + b, 3 + b});
        col_idx_U.assign({b, 2 + b, 3 + b, 1 + b, 2 + b, 3 + b});
    }
    else if(sort == aoclsparse_partially_sorted)
    {
        col_idx_L.assign({b, 1 + b, b, 2 + b, 1 + b, b, 3 + b});
        col_idx_U.assign({b, 3 + b, 2 + b, 1 + b, 2 + b, 3 + b});
    }
    else if(sort == aoclsparse_unsorted)
    {
        col_idx_L.assign({b, 1 + b, b, 2 + b, 3 + b, 1 + b, b});
        col_idx_U.assign({2 + b, b, 3 + b, 1 + b, 2 + b, 3 + b});
    }
    // val idx
    if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
    {
        if(sort == aoclsparse_partially_sorted)
        {
            val_L.assign({-1.0, 4.0, 5.0, -6.0, 8.0, 7.0, 9.0});
            val_U.assign({-1.0, 3.0, 2.0, 4.0, -6.0, 9.0});
        }
        else
        {
            val_L.assign({-1.0, 4.0, 5.0, -6.0, 7.0, 8.0, 9.0});
            val_U.assign({-1.0, 2.0, 3.0, 4.0, -6.0, 9.0});
        }
    }
    else if constexpr(std::is_same_v<T, std::complex<double>>
                      || std::is_same_v<T, std::complex<float>>
                      || std::is_same_v<T, aoclsparse_double_complex>
                      || std::is_same_v<T, aoclsparse_float_complex>)
    {
        if(sort == aoclsparse_partially_sorted)
        {
            val_L.assign({{-1, 3}, {4, 2}, {5, 6}, {-6, 3}, {8, -8}, {7, 1}, {9, 0}});
            val_U.assign({{-1, 3}, {3, 3}, {2, -1}, {4, 2}, {-6, 3}, {9, 0}});
        }
        else
        {
            val_L.assign({{-1, 3}, {4, 2}, {5, 6}, {-6, 3}, {7, 1}, {8, -8}, {9, 0}});
            val_U.assign({{-1, 3}, {2, -1}, {3, 3}, {4, 2}, {-6, 3}, {9, 0}});
        }
    }
}
/* Returns 'true' if AVX512 tests can be executed in the given build
 * i.e. the build is AVX512-enabled running on AVX512 hardware.
 */
bool can_exec_avx512_tests();
