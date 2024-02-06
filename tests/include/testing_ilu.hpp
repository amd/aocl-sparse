/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#ifndef TESTING_ILU_HPP
#define TESTING_ILU_HPP

#include "aoclsparse_analysis.h"
#include "aoclsparse_convert.h"
#include "aoclsparse.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

#include <fstream>
#include <string>

using namespace std;

template <typename T>
inline void ref_csrilu0(aoclsparse_int                     M,
                        aoclsparse_int                     base,
                        const std::vector<aoclsparse_int> &csr_row_ptr,
                        const std::vector<aoclsparse_int> &csr_col_ind,
                        std::vector<T>                    &csr_val,
                        aoclsparse_int                    *struct_pivot,
                        aoclsparse_int                    *numeric_pivot)
{
    // Initialize pivot
    *struct_pivot  = -1;
    *numeric_pivot = -1;

    // pointer of upper part of each row
    std::vector<aoclsparse_int> diag_offset(M);
    std::vector<aoclsparse_int> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(aoclsparse_int ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        aoclsparse_int row_begin = csr_row_ptr[ai] - base;
        aoclsparse_int row_end   = csr_row_ptr[ai + 1] - base;
        aoclsparse_int j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = j;
        }

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if(csr_col_ind[j] - base < ai)
            {

                aoclsparse_int col_j  = csr_col_ind[j] - base;
                aoclsparse_int diag_j = diag_offset[col_j];

                if(csr_val[diag_j] != static_cast<T>(0))
                {
                    // multiplication factor
                    csr_val[j] = csr_val[j] / csr_val[diag_j];

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(aoclsparse_int k = diag_j + 1; k < csr_row_ptr[col_j + 1] - base; ++k)
                    {
                        // if nnz at this position do linear combination
                        if(nnz_entries[csr_col_ind[k] - base] != 0)
                        {
                            aoclsparse_int idx = nnz_entries[csr_col_ind[k] - base];
                            csr_val[idx]       = std::fma(-csr_val[j], csr_val[k], csr_val[idx]);
                        }
                    }
                }
                else
                {
                    // Numerical zero diagonal
                    *numeric_pivot = col_j + base;
                    return;
                }
            }
            else if(csr_col_ind[j] - base == ai)
            {
                has_diag = true;
                break;
            }
            else
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            *struct_pivot  = ai + base;
            *numeric_pivot = ai + base;
            return;
        }

        // set diagonal pointer to diagonal element
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - base] = 0;
        }
    }
}

template <typename T>
void testing_ilu(const Arguments &arg)
{
    aoclsparse_status      ret;
    aoclsparse_int         M     = arg.M;
    aoclsparse_int         N     = arg.N;
    aoclsparse_int         nnz   = arg.nnz;
    aoclsparse_matrix_init mat   = arg.matrix;
    aoclsparse_operation   trans = arg.transA;
    aoclsparse_index_base  base  = arg.baseA;
    bool                   issymm;
    std::string            filename        = arg.filename;
    T                     *approx_inv_diag = NULL;

    // Create matrix descriptor
    aoclsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_AOCLSPARSE_ERROR(aoclsparse_set_mat_index_base(descr, base));

    // Allocate memory for matrix
    std::vector<aoclsparse_int> csr_row_ptr;
    std::vector<aoclsparse_int> csr_col_ind;
    std::vector<T>              csr_val;

    std::vector<T> x(N);
    std::vector<T> b(N);
    aoclsparse_seedrand();

    aoclsparse_init_csr_matrix(
        csr_row_ptr, csr_col_ind, csr_val, M, N, nnz, base, mat, filename.c_str(), issymm, true);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);

    aoclsparse_matrix A;

    CHECK_AOCLSPARSE_ERROR(aoclsparse_create_csr<T>(
        &A, base, M, N, nnz, csr_row_ptr.data(), csr_col_ind.data(), csr_val.data()));

    // Initialize data
    T *precond_csr_val = NULL;

    aoclsparse_int h_analysis_pivot_gold;
    aoclsparse_int h_solve_pivot_gold;
    std::vector<T> csr_val_gold(nnz);
    std::vector<T> ref_copy_csr_val(nnz);
    //save the csr values for in-place operation of ilu0
    ref_copy_csr_val = csr_val;

    if(arg.unit_check)
    {
        csr_val_gold = csr_val;
        //compute ILU0 using reference c code and store the LU factors as part of csr_val_gold
        ref_csrilu0<T>(M,
                       base,
                       csr_row_ptr,
                       csr_col_ind,
                       csr_val_gold,
                       &h_analysis_pivot_gold,
                       &h_solve_pivot_gold);
    }

    //Basic routine type checks
    ret = aoclsparse_set_lu_smoother_hint(A, trans, descr, 0);
    if(aoclsparse_status_success != ret)
    {
        std::cerr << "aoclSPARSE status error: Expected "
                  << aoclsparse_status_to_string(aoclsparse_status_success) << ", received "
                  << aoclsparse_status_to_string(ret) << std::endl;
        aoclsparse_destroy(&A);
        return;
    }

    // Optimize the matrix, "A"
    CHECK_AOCLSPARSE_ERROR(aoclsparse_optimize(A));

    CHECK_AOCLSPARSE_ERROR(aoclsparse_ilu_smoother(
        trans, A, descr, &precond_csr_val, approx_inv_diag, x.data(), b.data()));

    if(arg.unit_check)
    {
        if(near_check_general<T>(1, nnz, 1, csr_val_gold.data(), precond_csr_val))
        {
            aoclsparse_destroy(&A);
            return;
        }
    }

    std::cout.precision(2);
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::left);

    std::cout << std::setw(20) << "input" << std::setw(12) << "M" << std::setw(12) << "nnz"
              << std::setw(12) << "verified" << std::endl;

    std::cout << std::setw(20) << filename.c_str() << std::setw(12) << M << std::setw(12) << nnz
              << std::setw(12) << (arg.unit_check ? "yes" : "no") << std::endl;

    aoclsparse_destroy(&A);
    return;
}

#endif // TESTING_ILU_HPP
