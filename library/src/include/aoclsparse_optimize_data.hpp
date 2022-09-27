/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#ifndef AOCLSPARSE_OPTIMIZE_DATA
#define AOCLSPARSE_OPTIMIZE_DATA

#include "aoclsparse.h"

enum aoclsparse_hinted_action
{
    aoclsparse_action_none = 0,
    aoclsparse_action_mv,
    aoclsparse_action_sv,
    aoclsparse_action_mm,
    aoclsparse_action_2m,
    aoclsparse_action_ilu0,
};

/* Linked list of all the hint information that was passed through the 
 * functions aoclsparse_*_hint()
 */
struct aoclsparse_optimize_data
{
    aoclsparse_hinted_action act   = aoclsparse_action_none;
    aoclsparse_operation     trans = aoclsparse_operation_none;
    aoclsparse_matrix_type   type;
    aoclsparse_fill_mode     fill_mode;
    // number of operations estimated
    aoclsparse_int nop = 0;
    // store if the matrix has already been optimized for this specific operation
    bool action_optimized = false;
    // next hint requested
    aoclsparse_optimize_data *next = nullptr;
};

/* Add a new optimize hint to the list */
aoclsparse_status aoclsparse_add_hint(aoclsparse_optimize_data *&list,
                                      aoclsparse_hinted_action   op,
                                      aoclsparse_mat_descr       desc,
                                      aoclsparse_operation       trans,
                                      aoclsparse_int             nop);

/* Deallocate the aoclsparse_optimize_data linked list*/
void aoclsparse_optimize_destroy(aoclsparse_optimize_data *&opt);

#endif // AOCLSPARSE_OPTIMIZE_DATA