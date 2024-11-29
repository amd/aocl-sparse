/* ************************************************************************
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_CSRMV_HPP
#define AOCLSPARSE_CSRMV_HPP

#include "aoclsparse_csrmv_avx512.hpp"
#include "aoclsparse_csrmv_kernels.hpp"
#include "aoclsparse_l2_kt.hpp"

template <typename T>
aoclsparse_status aoclsparse_csrmv_t(aoclsparse_operation       trans,
                                     const T                   *alpha,
                                     aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     aoclsparse_int             nnz,
                                     const T                   *csr_val,
                                     const aoclsparse_int      *csr_col_ind,
                                     const aoclsparse_int      *csr_row_ptr,
                                     const aoclsparse_mat_descr descr,
                                     const T                   *x,
                                     const T                   *beta,
                                     T                         *y)
{
    using namespace aoclsparse;

    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Check index base
    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }

    // Support General and symmetric matrices.
    // Return for any other matrix type
    if((descr->type != aoclsparse_matrix_type_general)
       && (descr->type != aoclsparse_matrix_type_symmetric))
    {
        // TODO
        return aoclsparse_status_not_implemented;
    }

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return aoclsparse_status_success;
    }

    // Check pointer arguments
    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr || x == nullptr
       || y == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    if constexpr(std::is_same_v<T, float>)
    {
        switch(trans)
        {
        case aoclsparse_operation_none:
            if(descr->type == aoclsparse_matrix_type_symmetric)
            {
                return aoclsparse_csrmv_symm(
                    descr->base, *alpha, m, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
            }
            else
            {
                return aoclsparse_csrmv_vectorized(
                    descr->base, *alpha, m, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
            }
            break;

        case aoclsparse_operation_transpose:
            if(descr->type == aoclsparse_matrix_type_symmetric)
            {
                //when a matrix is symmetric, then matrix is equal to its transpose, and thus the matrix product
                //would also be same
                return aoclsparse_csrmv_symm(
                    descr->base, *alpha, m, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
            }
            else
            {
#ifdef USE_AVX512
                if(context::get_context()->supports<context_isa_t::AVX512F>())
                {
                    return aoclsparse::csrmvt_kt<kernel_templates::bsz::b512, T>(
                        descr->base, *alpha, m, n, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
                }
                else
#endif
                    return aoclsparse::csrmvt_kt<kernel_templates::bsz::b256, T>(
                        descr->base, *alpha, m, n, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
            }
            break;

        case aoclsparse_operation_conjugate_transpose:
            //TODO
            return aoclsparse_status_not_implemented;
            break;

        default:
            return aoclsparse_status_invalid_value;
            break;
        }
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        using namespace aoclsparse;

        switch(trans)
        {
        case aoclsparse_operation_none:
            if(descr->type == aoclsparse_matrix_type_symmetric)
            {
                return aoclsparse_csrmv_symm(
                    descr->base, *alpha, m, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
            }
            else
            {
                using K = decltype(&aoclsparse_csrmv_general<T>);

                K kernel;

                // Sparse matrices with Mean nnz = nnz/m <10 have very few non-zeroes in most of the rows
                // and few unevenly long rows . Loop unrolling and vectorization doesnt optimise performance
                // for this category of matrices . Hence , we invoke the generic dcsrmv kernel without
                // vectorization and innerloop unrolling . For the other category of sparse matrices
                // (Mean nnz > 10) , we continue to invoke the vectorised version of csrmv , since
                // it improves performance.
                if(nnz <= (10 * m))
                    kernel = aoclsparse_csrmv_general;
                else
                {
#ifdef USE_AVX512
                    if(context::get_context()->get_isa_hint() == context_isa_t::AVX512F)
                        kernel = aoclsparse_csrmv_vectorized_avx512;
                    else
#endif
                        kernel = aoclsparse_csrmv_vectorized_avx2;
                }

                // Invoke the kernel
                return kernel(
                    descr->base, *alpha, m, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
            }
            break;

        case aoclsparse_operation_transpose:
            if(descr->type == aoclsparse_matrix_type_symmetric)
            {
                //when a matrix is symmetric, then matrix is equal to its transpose, and thus the matrix product
                //would also be same
                return aoclsparse_csrmv_symm(
                    descr->base, *alpha, m, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
            }
            else
            {
#ifdef USE_AVX512
                if(context::get_context()->supports<context_isa_t::AVX512F>())
                {
                    return aoclsparse::csrmvt_kt<kernel_templates::bsz::b512, T>(
                        descr->base, *alpha, m, n, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
                }
                else
#endif
                    return aoclsparse::csrmvt_kt<kernel_templates::bsz::b256, T>(
                        descr->base, *alpha, m, n, csr_val, csr_col_ind, csr_row_ptr, x, *beta, y);
            }
            break;

        case aoclsparse_operation_conjugate_transpose:
            //TODO
            return aoclsparse_status_not_implemented;
            break;

        default:
            return aoclsparse_status_invalid_value;
            break;
        }
    }
}

#endif // AOCLSPARSE_CSRMV_HPP
