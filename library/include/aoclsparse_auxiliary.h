/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
 *  \brief aoclsparse_auxiliary.h provides auxilary functions in aoclsparse
 */
#ifndef AOCLSPARSE_AUXILIARY_H_
#define AOCLSPARSE_AUXILIARY_H_

#include "aoclsparse_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Get aoclsparse version
 *
 *  \details
 *  \p aoclsparse_get_version gets the aoclsparse library version number.
 *  - patch = version % 100
 *  - minor = version / 100 % 1000
 *  - major = version / 100000
 *
 *  @param[out]
 *  version the version number of the aoclsparse library.
 *
 *  \retval aoclsparse_status_success the operation completed successfully.
 *  \retval aoclsparse_status_invalid_pointer \p version is invalid.
 */
__attribute__((__visibility__("default"))) aoclsparse_status aoclsparse_get_version(aoclsparse_int* version);

#ifdef __cplusplus
}
#endif
#endif // AOCLSPARSE_AUXILIARY_H_

