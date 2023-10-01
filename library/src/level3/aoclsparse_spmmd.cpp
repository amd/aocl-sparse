/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */

#include "aoclsparse.h"
#include "aoclsparse_sp2md.hpp"

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */

/* 
 * Computes the product of two sparse matrices stored in compressed sparse row (CSR) format
 * and stores the result in a dense format. Supports s/d/c/z data types.
 */
extern "C" aoclsparse_status aoclsparse_sspmmd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_matrix    B,
                                               const aoclsparse_order     layout,
                                               float                     *C,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    if((nullptr == A) || (nullptr == B))
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_status    status;
    aoclsparse_mat_descr descrA;
    status = aoclsparse_create_mat_descr(&descrA);
    if(status != aoclsparse_status_success)
        return status;
    descrA->type = aoclsparse_matrix_type_general;
    aoclsparse_mat_descr descrB;
    status = aoclsparse_create_mat_descr(&descrB);
    if(status != aoclsparse_status_success)
        return status;
    descrB->type = aoclsparse_matrix_type_general;

    status = aoclsparse_set_mat_index_base(descrA, A->base);
    if(status != aoclsparse_status_success)
        return status;
    status = aoclsparse_set_mat_index_base(descrB, B->base);
    if(status != aoclsparse_status_success)
        return status;

    aoclsparse_operation op_B  = aoclsparse_operation_none;
    float                alpha = 1.0;
    float                beta  = 0.0;

    status = aoclsparse_sp2md_t(op, descrA, A, op_B, descrB, B, alpha, beta, C, layout, ldc, kid);
    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy_mat_descr(descrB);
    return status;
}

extern "C" aoclsparse_status aoclsparse_dspmmd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_matrix    B,
                                               const aoclsparse_order     layout,
                                               double                    *C,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    if((nullptr == A) || (nullptr == B))
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_status    status;
    aoclsparse_mat_descr descrA;
    status = aoclsparse_create_mat_descr(&descrA);
    if(status != aoclsparse_status_success)
        return status;
    descrA->type = aoclsparse_matrix_type_general;
    aoclsparse_mat_descr descrB;
    status = aoclsparse_create_mat_descr(&descrB);
    if(status != aoclsparse_status_success)
        return status;
    descrB->type = aoclsparse_matrix_type_general;

    status = aoclsparse_set_mat_index_base(descrA, A->base);
    if(status != aoclsparse_status_success)
        return status;
    status = aoclsparse_set_mat_index_base(descrB, B->base);
    if(status != aoclsparse_status_success)
        return status;

    aoclsparse_operation op_B  = aoclsparse_operation_none;
    double               alpha = 1.0;
    double               beta  = 0.0;

    status = aoclsparse_sp2md_t(op, descrA, A, op_B, descrB, B, alpha, beta, C, layout, ldc, kid);
    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy_mat_descr(descrB);
    return status;
}

extern "C" aoclsparse_status aoclsparse_cspmmd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_matrix    B,
                                               const aoclsparse_order     layout,
                                               aoclsparse_float_complex  *C,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    if((nullptr == A) || (nullptr == B))
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_status    status;
    aoclsparse_mat_descr descrA;
    status = aoclsparse_create_mat_descr(&descrA);
    if(status != aoclsparse_status_success)
        return status;
    descrA->type = aoclsparse_matrix_type_general;
    aoclsparse_mat_descr descrB;
    status = aoclsparse_create_mat_descr(&descrB);
    if(status != aoclsparse_status_success)
        return status;
    descrB->type = aoclsparse_matrix_type_general;

    status = aoclsparse_set_mat_index_base(descrA, A->base);
    if(status != aoclsparse_status_success)
        return status;
    status = aoclsparse_set_mat_index_base(descrB, B->base);
    if(status != aoclsparse_status_success)
        return status;

    aoclsparse_operation op_B  = aoclsparse_operation_none;
    std::complex<float>  alpha = 1.0;
    std::complex<float>  beta  = 0.0;

    status = aoclsparse_sp2md_t(
        op, descrA, A, op_B, descrB, B, alpha, beta, (std::complex<float> *)C, layout, ldc, kid);
    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy_mat_descr(descrB);
    return status;
}

extern "C" aoclsparse_status aoclsparse_zspmmd(const aoclsparse_operation op,
                                               const aoclsparse_matrix    A,
                                               const aoclsparse_matrix    B,
                                               const aoclsparse_order     layout,
                                               aoclsparse_double_complex *C,
                                               const aoclsparse_int       ldc)
{
    const aoclsparse_int kid = -1; /* auto */
    if((nullptr == A) || (nullptr == B))
    {
        return aoclsparse_status_invalid_pointer;
    }
    aoclsparse_status    status;
    aoclsparse_mat_descr descrA;
    status = aoclsparse_create_mat_descr(&descrA);
    if(status != aoclsparse_status_success)
        return status;
    descrA->type = aoclsparse_matrix_type_general;
    aoclsparse_mat_descr descrB;
    status = aoclsparse_create_mat_descr(&descrB);
    if(status != aoclsparse_status_success)
        return status;
    descrB->type = aoclsparse_matrix_type_general;

    status = aoclsparse_set_mat_index_base(descrA, A->base);
    if(status != aoclsparse_status_success)
        return status;
    status = aoclsparse_set_mat_index_base(descrB, B->base);
    if(status != aoclsparse_status_success)
        return status;

    aoclsparse_operation op_B  = aoclsparse_operation_none;
    std::complex<double> alpha = 1.0;
    std::complex<double> beta  = 0.0;

    status = aoclsparse_sp2md_t(
        op, descrA, A, op_B, descrB, B, alpha, beta, (std::complex<double> *)C, layout, ldc, kid);
    aoclsparse_destroy_mat_descr(descrA);
    aoclsparse_destroy_mat_descr(descrB);
    return status;
}
