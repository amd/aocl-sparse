/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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

#ifndef AOCLSPARSE_AUXILIARY_HPP
#define AOCLSPARSE_AUXILIARY_HPP

#include "aoclsparse_mat_structures.h"

#include <cmath>
#include <limits>

// Ignore compiler warning from BLIS
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wunused-function"
// The fix order of BLIS/Flame headers
// clang-format off
#include "blis.h"
#include "cblas.hh"
#include "FLAME.h"
// clang-format on
// Restore
#pragma GCC diagnostic pop

/* Check that the size of integers in the used libraries is OK. */
static_assert(
    sizeof(f77_int) == sizeof(aoclsparse_int),
    "Error: Incompatible size of ints in blis. Using wrong header or compilation of the library?");
static_assert(
    sizeof(integer) == sizeof(aoclsparse_int),
    "Error: Incompatible size of ints in flame. Using wrong header or compilation of the library?");

void aoclsparse_init_mat(aoclsparse_matrix             A,
                         aoclsparse_index_base         base,
                         aoclsparse_int                M,
                         aoclsparse_int                N,
                         aoclsparse_int                nnz,
                         aoclsparse_matrix_format_type matrix_type);

template <typename T>
aoclsparse_status aoclsparse_create_csr_t(aoclsparse_matrix    &mat,
                                          aoclsparse_index_base base,
                                          aoclsparse_int        M,
                                          aoclsparse_int        N,
                                          aoclsparse_int        nnz,
                                          aoclsparse_int       *col_ptr,
                                          aoclsparse_int       *row_idx,
                                          T                    *val);

template <typename T>
aoclsparse_status aoclsparse_create_csc(aoclsparse_matrix    &mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *col_ptr,
                                        aoclsparse_int       *row_idx,
                                        T                    *val);

template <typename T>
aoclsparse_status aoclsparse_create_coo(aoclsparse_matrix          &mat,
                                        const aoclsparse_index_base base,
                                        const aoclsparse_int        M,
                                        const aoclsparse_int        N,
                                        const aoclsparse_int        nnz,
                                        aoclsparse_int             *row_ptr,
                                        aoclsparse_int             *col_ptr,
                                        T                          *val);

/********************************************************************************
 * \brief generates a plane rotation with cosine and sine. Slower and more accurate
 * version of BLAS's DROTG performs the Givens Rotation. The mathematical formulas
 * used for C and S are
        hv = sqrt(rr^2 + hh^2)
        c = rr/hv
        s = hh/hv
        h_mj_j = hv
 *
 *******************************************************************************/
inline void aoclsparse_givens_rotation(double &rr, double &hh, double &c, double &s, double &h_mj_j)
{
    dlartg_(&rr, &hh, &c, &s, &h_mj_j);
}
inline void aoclsparse_givens_rotation(float &rr, float &hh, float &c, float &s, float &h_mj_j)
{
    slartg_(&rr, &hh, &c, &s, &h_mj_j);
}

/*
    Perform a comparison test to determine if the value is near zero
*/
template <typename T>
bool aoclsparse_zerocheck(const T &value)
{
    bool        is_value_zero = false;
    constexpr T macheps       = std::numeric_limits<T>::epsilon();
    constexpr T safe_macheps  = (T)2.0 * macheps;
    is_value_zero             = std::fabs(value) <= safe_macheps;
    return is_value_zero;
}

extern const size_t data_size[];

/*
    Return aoclsparse_matrix_data_type based on the input type (s/d/c/z)
*/
template <typename T>
struct get_data_type
{
};
template <>
struct get_data_type<float>
{
    constexpr operator aoclsparse_matrix_data_type() const
    {
        return aoclsparse_smat;
    }
};
template <>
struct get_data_type<double>
{
    constexpr operator aoclsparse_matrix_data_type() const
    {
        return aoclsparse_dmat;
    }
};
template <>
struct get_data_type<aoclsparse_float_complex>
{
    constexpr operator aoclsparse_matrix_data_type() const
    {
        return aoclsparse_cmat;
    }
};
template <>
struct get_data_type<aoclsparse_double_complex>
{
    constexpr operator aoclsparse_matrix_data_type() const
    {
        return aoclsparse_zmat;
    }
};
#endif
