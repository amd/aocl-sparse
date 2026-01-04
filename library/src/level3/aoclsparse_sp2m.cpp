/* ************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse.hpp"
#include "aoclsparse_mat_structures.hpp"

aoclsparse_status aoclsparse_sp2m(aoclsparse_operation       opA,
                                  const aoclsparse_mat_descr descrA,
                                  const aoclsparse_matrix    A,
                                  aoclsparse_operation       opB,
                                  const aoclsparse_mat_descr descrB,
                                  const aoclsparse_matrix    B,
                                  aoclsparse_request         request,
                                  aoclsparse_matrix         *C)
{
    if((A == nullptr) || (B == nullptr) || (C == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    if((A->val_type == aoclsparse_smat) && (B->val_type == aoclsparse_smat))
        return aoclsparse::sp2m<float>(opA, descrA, A, opB, descrB, B, request, C);
    else if((A->val_type == aoclsparse_dmat) && (B->val_type == aoclsparse_dmat))
        return aoclsparse::sp2m<double>(opA, descrA, A, opB, descrB, B, request, C);
    else if((A->val_type == aoclsparse_cmat) && (B->val_type == aoclsparse_cmat))
        return aoclsparse::sp2m<std::complex<float>>(opA, descrA, A, opB, descrB, B, request, C);
    else if((A->val_type == aoclsparse_zmat) && (B->val_type == aoclsparse_zmat))
        return aoclsparse::sp2m<std::complex<double>>(opA, descrA, A, opB, descrB, B, request, C);
    else
        return aoclsparse_status_wrong_type;
}
