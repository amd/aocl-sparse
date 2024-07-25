/* ************************************************************************
 * Copyright (c) 2020-2024 Advanced Micro Devices, Inc.All rights reserved.
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
/*! \file
 *  \brief aoclsparse_utility.hpp provides common utilities
 */

#pragma once
#ifndef AOCLSPARSE_UTILITY_HPP
#define AOCLSPARSE_UTILITY_HPP

#include "aoclsparse.h"
#include "aoclsparse.hpp"

#include <cfloat>
#include <sstream>
#include <string>
#include <vector>

/* ==================================================================================== */
/*! \brief  local matrix descriptor which is automatically created and destroyed  */
class aoclsparse_local_mat_descr
{
    aoclsparse_mat_descr descr;

public:
    aoclsparse_local_mat_descr()
    {
        aoclsparse_create_mat_descr(&descr);
    }
    ~aoclsparse_local_mat_descr()
    {
        aoclsparse_destroy_mat_descr(descr);
    }

    // Allow aoclsparse_local_mat_descr to be used anywhere aoclsparse_mat_descr is expected
    operator aoclsparse_mat_descr &()
    {
        return descr;
    }
    operator const aoclsparse_mat_descr &() const
    {
        return descr;
    }
};

/* ============================================================================================ */
// Return path of this executable
std::string aoclsparse_exepath();

/* ==================================================================================== */

/*! \brief  CPU Timer(in second): return wall time
*/
double aoclsparse_clock(void);
double aoclsparse_clock_min_diff(double time_min, double time_start);
double aoclsparse_clock_diff(double time_start);

double aoclsparse_clock_helper(void);

/**
 * Utility for parsing command line arguments
 */
struct aoclsparse_command_line_args
{

    std::vector<std::string> keys;
    std::vector<std::string> values;
    std::vector<std::string> args;

    /**
     * Constructor
     */
    aoclsparse_command_line_args(aoclsparse_int argc, char **argv)
        : keys(4)
        , values(4)
    {

        for(aoclsparse_int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];

            if((arg[0] != '-') || (arg[1] != '-'))
            {
                args.push_back(arg);
                continue;
            }

            std::string::size_type pos;
            std::string            key, val;
            if((pos = arg.find('=')) == std::string::npos)
            {
                key = std::string(arg, 2, arg.length() - 2);
                val = "";
            }
            else
            {
                key = std::string(arg, 2, pos - 2);
                val = std::string(arg, pos + 1, arg.length() - 1);
            }

            keys.push_back(key);
            values.push_back(val);
        }
    }

    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool aoclsparse_check_cmdline_flag(const char *arg_name)
    {

        for(aoclsparse_int i = 0; i < aoclsparse_int(keys.size()); ++i)
        {
            if(keys[i] == std::string(arg_name))
                return true;
        }
        return false;
    }

    /**
     * Returns number of naked (non-flag and non-key-value) commandline parameters
     */
    template <typename T>
    aoclsparse_int aoclsparse_num_naked_args()
    {
        return args.size();
    }

    /**
     * Returns the commandline parameter for a given index (not including flags)
     */
    template <typename T>
    void aoclsparse_get_cmdline_argument(aoclsparse_int index, T &val)
    {

        if((size_t)index < args.size())
        {
            std::istringstream str_stream(args[index]);
            str_stream >> val;
        }
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void aoclsparse_get_cmdline_argument(const char *arg_name, T &val)
    {

        for(aoclsparse_int i = 0; i < aoclsparse_int(keys.size()); ++i)
        {
            if(keys[i] == std::string(arg_name))
            {
                std::istringstream str_stream(values[i]);
                str_stream >> val;
            }
        }
    }

    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void aoclsparse_get_cmdline_arguments(const char *arg_name, std::vector<T> &vals)
    {

        if(aoclsparse_check_cmdline_flag(arg_name))
        {
            // Clear any default values
            vals.clear();

            // Recover from multi-value string
            for(aoclsparse_int i = 0; i < (aoclsparse_int)keys.size(); ++i)
            {
                if(keys[i] == std::string(arg_name))
                {
                    std::string            val_string(values[i]);
                    std::istringstream     str_stream(val_string);
                    std::string::size_type old_pos = 0;
                    std::string::size_type new_pos = 0;

                    // Iterate comma-separated values
                    T val;
                    while((new_pos = val_string.find(',', old_pos)) != std::string::npos)
                    {
                        if(new_pos != old_pos)
                        {
                            str_stream.width(new_pos - old_pos);
                            str_stream >> val;
                            vals.push_back(val);
                        }

                        // skip over comma
                        str_stream.ignore(1);
                        old_pos = new_pos + 1;
                    }

                    // Read last value
                    str_stream >> val;
                    vals.push_back(val);
                }
            }
        }
    }

    /**
     * The number of pairs parsed
     */
    aoclsparse_int aoclsparse_parsed_argc()
    {
        return (aoclsparse_int)keys.size();
    }
};

template <typename T>
struct internal_t_map
{
    using type = T;
};

template <>
struct internal_t_map<aoclsparse_float_complex>
{
    using type = std::complex<float>;
};

template <>
struct internal_t_map<aoclsparse_double_complex>
{
    using type = std::complex<double>;
};

// aoclsparse_*_complex to std::complex<*> map
// For any other type, it returns the same type.
template <typename T>
using internal_t = typename internal_t_map<T>::type;

/* Export/copy out AOCL Sparse CSR matrix into 3-array format,
 * sort elements in rows.
 * Returns aoclsparse_status code so might be check with
 * NEW_CHECK_AOCLSPARSE_ERROR().
 */
template <typename T>
aoclsparse_status aocl_csr_sorted_export(const aoclsparse_matrix      mat,
                                         aoclsparse_index_base       &base,
                                         aoclsparse_int              &m,
                                         aoclsparse_int              &n,
                                         aoclsparse_int              &nnz,
                                         std::vector<aoclsparse_int> &row_ptr,
                                         std::vector<aoclsparse_int> &col_ind,
                                         std::vector<T>              &val)
{

    aoclsparse_status status        = aoclsparse_status_success;
    aoclsparse_int   *csr_row_ptr_C = NULL;
    aoclsparse_int   *csr_col_ind_C = NULL;
    T                *csr_val_C     = NULL;

    status = aoclsparse_export_csr(
        mat, &base, &m, &n, &nnz, &csr_row_ptr_C, &csr_col_ind_C, &csr_val_C);
    if(status != aoclsparse_status_success)
        return status;

    row_ptr = std::vector<aoclsparse_int>(csr_row_ptr_C, csr_row_ptr_C + m + 1);
    col_ind = std::vector<aoclsparse_int>(nnz);
    val     = std::vector<T>(nnz);

    // getting sorted col and val in each row
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int start = csr_row_ptr_C[i] - base, end = csr_row_ptr_C[i + 1] - base;
        std::vector<aoclsparse_int> idxs(end - start);
        std::iota(idxs.begin(), idxs.end(), start);
        std::sort(idxs.begin(), idxs.end(), [csr_col_ind_C](auto a, auto b) {
            return csr_col_ind_C[a] < csr_col_ind_C[b];
        });
        for(auto idx : idxs)
        {
            col_ind[start] = csr_col_ind_C[idx];
            val[start]     = csr_val_C[idx];
            start++;
        }
    }
    return status;
}

#endif // AOCLSPARSE_UTILITY_HPP
