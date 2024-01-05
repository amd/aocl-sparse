/* ************************************************************************
 * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse_sypr.hpp"

aoclsparse_status aoclsparse_sypr(aoclsparse_operation       opA,
                                  const aoclsparse_matrix    A,
                                  const aoclsparse_matrix    B,
                                  const aoclsparse_mat_descr descrB,
                                  aoclsparse_matrix         *C,
                                  const aoclsparse_request   request)
{
    if((A == nullptr) || (B == nullptr) || (C == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    if((A->val_type == aoclsparse_smat) && (B->val_type == aoclsparse_smat))
        return aoclsparse_sypr_t<float>(opA, A, B, descrB, C, request);
    else if((A->val_type == aoclsparse_dmat) && (B->val_type == aoclsparse_dmat))
        return aoclsparse_sypr_t<double>(opA, A, B, descrB, C, request);
    else if((A->val_type == aoclsparse_cmat) && (B->val_type == aoclsparse_cmat))
        return aoclsparse_sypr_t<std::complex<float>>(opA, A, B, descrB, C, request);
    else if((A->val_type == aoclsparse_zmat) && (B->val_type == aoclsparse_zmat))
        return aoclsparse_sypr_t<std::complex<double>>(opA, A, B, descrB, C, request);
    else
        return aoclsparse_status_wrong_type;
}
