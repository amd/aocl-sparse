/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.All rights reserved.
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
        using namespace std;

        for(aoclsparse_int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if((arg[0] != '-') || (arg[1] != '-'))
            {
                args.push_back(arg);
                continue;
            }

            string::size_type pos;
            string            key, val;
            if((pos = arg.find('=')) == string::npos)
            {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            }
            else
            {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
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
        using namespace std;

        for(aoclsparse_int i = 0; i < aoclsparse_int(keys.size()); ++i)
        {
            if(keys[i] == string(arg_name))
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
        using namespace std;
        if(index < args.size())
        {
            istringstream str_stream(args[index]);
            str_stream >> val;
        }
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void aoclsparse_get_cmdline_argument(const char *arg_name, T &val)
    {
        using namespace std;

        for(aoclsparse_int i = 0; i < aoclsparse_int(keys.size()); ++i)
        {
            if(keys[i] == string(arg_name))
            {
                istringstream str_stream(values[i]);
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
        using namespace std;

        if(aoclsparse_check_cmdline_flag(arg_name))
        {
            // Clear any default values
            vals.clear();

            // Recover from multi-value string
            for(aoclsparse_int i = 0; i < keys.size(); ++i)
            {
                if(keys[i] == string(arg_name))
                {
                    string            val_string(values[i]);
                    istringstream     str_stream(val_string);
                    string::size_type old_pos = 0;
                    string::size_type new_pos = 0;

                    // Iterate comma-separated values
                    T val;
                    while((new_pos = val_string.find(',', old_pos)) != string::npos)
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

#endif // AOCLSPARSE_UTILITY_HPP
