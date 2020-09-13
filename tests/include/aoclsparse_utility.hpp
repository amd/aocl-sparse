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
#include <string>


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
    operator aoclsparse_mat_descr&()
    {
        return descr;
    }
    operator const aoclsparse_mat_descr&() const
    {
        return descr;
    }
};

/* ============================================================================================ */
// Return path of this executable
std::string aoclsparse_exepath();

/* ==================================================================================== */

/*! \brief  CPU Timer(in microsecond): return wall time
 */
double get_time_us(void);


#endif // AOCLSPARSE_UTILITY_HPP
