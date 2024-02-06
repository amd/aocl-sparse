/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#ifndef AOCLSPARSE_UTILS_HPP
#define AOCLSPARSE_UTILS_HPP

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

// Common defines
#if defined(_WIN32) || defined(_WIN64)
// Windows equivalent of gcc c99 type qualifier __restrict__
#define __restrict__ __restrict
#endif

extern const size_t data_size[];

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

/* Conjugate functionality the returns both complex and real types */
/* The standard std::conj return only complex types */
namespace aoclsparse
{
    template <typename T>
    constexpr T conj(const T a)
    {
        if constexpr(std::is_same_v<T, std::complex<double>>
                     || std::is_same_v<T, std::complex<float>>)
        {
            return std::conj(a);
        }
        return a;
    }
}

namespace aoclsparse_numeric
{
    /* Provide a "zero" for all floating point data types */
    /* Default definition handles real/std types */
    template <typename T>
    struct zero
    {
        static constexpr T value{0};
        constexpr operator T() const noexcept
        {
            return value;
        }
    };
    template <>
    struct zero<aoclsparse_float_complex>
    {
        static constexpr aoclsparse_float_complex value{0, 0};
        constexpr operator aoclsparse_float_complex() const noexcept
        {
            return value;
        }
    };
    template <>
    struct zero<aoclsparse_double_complex>
    {
        static constexpr aoclsparse_double_complex value{0, 0};
        constexpr operator aoclsparse_double_complex() const noexcept
        {
            return value;
        }
    };
}

/* Convenience operator for comparing with zero<T>
 * These enable the comparisons
 * T a // for T \in { float, double, std::complex<float|double> and aoclsparse_*_complex }
 * a == zero<T> and a != zero<T>, as well as the swapped variants
 * zero<T> == a and zero<T> != a.
 *
 * Warning: these operators should not be used for tolerance-based comparisons, i.e., for a
 * set tolerance (eps>0) to query if variable (a) can be considered zero or not, use a tolerance
 * approach such as |a| < eps.
 */
template <typename T>
constexpr bool operator==(const T &lhs, [[maybe_unused]] const aoclsparse_numeric::zero<T> &_)
{
    if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                 || std::is_same_v<T, aoclsparse_double_complex>)
        return lhs.real == 0 && lhs.imag == 0;
    else
        return lhs == (T)0;
}

template <typename T>
constexpr bool operator==(const aoclsparse_numeric::zero<T> &lhs, const T &rhs)
{
    return rhs == lhs;
}

template <typename T>
constexpr bool operator!=(const T &lhs, [[maybe_unused]] const aoclsparse_numeric::zero<T> &_)
{
    if constexpr(std::is_same_v<T, aoclsparse_float_complex>
                 || std::is_same_v<T, aoclsparse_double_complex>)
        return lhs.real != 0 || lhs.imag != 0;
    else
        return lhs != (T)0;
}

template <typename T>
constexpr bool operator!=(const aoclsparse_numeric::zero<T> &lhs, const T &rhs)
{
    return rhs != lhs;
}

/*
    Return aoclsparse_matrix_data_type based on the input type
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
template <>
struct get_data_type<std::complex<float>>
{
    constexpr operator aoclsparse_matrix_data_type() const
    {
        return aoclsparse_cmat;
    }
};
template <>
struct get_data_type<std::complex<double>>
{
    constexpr operator aoclsparse_matrix_data_type() const
    {
        return aoclsparse_zmat;
    }
};

/* Common indexing for gather- and scatter-type APIs
 * It is useful also for other API that need to handle
 * arrays in both indexed and strided modes.
 */
namespace Index
{
    enum class type
    {
        strided,
        indexed
    };

    template <type I>
    using index_t =
        typename std::conditional<I == type::strided, aoclsparse_int, const aoclsparse_int *>::type;

}

/* real data type for tolerance in case the templated type is of complex data
 */
template <typename T>
struct tolerance
{
    using type = T;
};
template <>
struct tolerance<std::complex<float>>
{
    using type = float;
};
template <>
struct tolerance<std::complex<double>>
{
    using type = double;
};
template <>
struct tolerance<aoclsparse_float_complex>
{
    using type = float;
};
template <>
struct tolerance<aoclsparse_double_complex>
{
    using type = double;
};

template <typename T>
using tolerance_t = typename tolerance<T>::type;

// Define precision to which we expect the results to match ==================================
template <typename T>
struct safeguard
{
};
// Add safeguarding scaling (may differ for each data type)
template <>
struct safeguard<double>
{
    static constexpr double value = 1.0;
};
template <>
struct safeguard<float>
{
    static constexpr float value = 2.0f;
};

template <typename T>
constexpr T expected_precision(T scale = (T)1.0) noexcept
{
    const T macheps      = std::numeric_limits<T>::epsilon();
    const T safe_macheps = (T)2.0 * macheps;
    return scale * safeguard<T>::value * sqrt(safe_macheps);
}
#endif
