/* ************************************************************************
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "aoclsparse_csr2m.hpp"

aoclsparse_status aoclsparse_dcsr2m(aoclsparse_operation       transA,
                                    const aoclsparse_mat_descr descrA,
                                    const aoclsparse_matrix    csrA,
                                    aoclsparse_operation       transB,
                                    const aoclsparse_mat_descr descrB,
                                    const aoclsparse_matrix    csrB,
                                    aoclsparse_request         request,
                                    aoclsparse_matrix         *csrC)
{
    return aoclsparse_csr2m_t<double>(transA, descrA, csrA, transB, descrB, csrB, request, csrC);
}

aoclsparse_status aoclsparse_scsr2m(aoclsparse_operation       transA,
                                    const aoclsparse_mat_descr descrA,
                                    const aoclsparse_matrix    csrA,
                                    aoclsparse_operation       transB,
                                    const aoclsparse_mat_descr descrB,
                                    const aoclsparse_matrix    csrB,
                                    aoclsparse_request         request,
                                    aoclsparse_matrix         *csrC)
{
    return aoclsparse_csr2m_t<float>(transA, descrA, csrA, transB, descrB, csrB, request, csrC);
}
