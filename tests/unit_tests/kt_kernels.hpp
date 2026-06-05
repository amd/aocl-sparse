/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

// Re-includable macros and list of ukernels to test

#ifndef KT_TEST_DO3
#error "KT_TEST_DO3 must be defined before including this header"
#endif

#ifndef KT_TEST_DO3_SKIP
// If skip is not defined, then there is no need for a skip unit and we
// want to repeat whatever do3 does
#define KT_TEST_DO3_SKIP(FUNC, SZ, SUF) KT_TEST_DO3(FUNC, SZ, SUF)
#endif

#ifndef KT_TEST_DO4
#error "KT_TEST_DO4 must be defined before including this header"
#endif

#ifndef KT_TEST_DO4_SKIP
// If skip is not defined, then there is no need for a skip unit and we
// want to repeat whatever do4 does
#define KT_TEST_DO4_SKIP(FUNC, SZ, SUF, IDX) KT_TEST_DO4(FUNC, SZ, SUF, IDX)
#endif

#ifdef __AVX512FP16__
#define KT_FP16_E
#else
#define KT_FP16_E _SKIP
#endif

#define PASTE_(A, B) A##B
#define PASTE(A, B) PASTE_(A, B)

// -----------------------------------------------------------------------------
// 1/3 Instantiation and driver definition macros
// -----------------------------------------------------------------------------

#define KT_TEST_DO_ALL_INDEX_TYPES(FUNC, SZ, SUF) \
    KT_TEST_DO4(FUNC, SZ, SUF, int32_t);          \
    KT_TEST_DO4(FUNC, SZ, SUF, int64_t);          \
    KT_TEST_DO4(FUNC, SZ, SUF, uint32_t);         \
    KT_TEST_DO4(FUNC, SZ, SUF, uint64_t);

#define KT_TEST_DO_ALL_INDEX_TYPES_SKIP(FUNC, SZ, SUF) \
    KT_TEST_DO4_SKIP(FUNC, SZ, SUF, int32_t);          \
    KT_TEST_DO4_SKIP(FUNC, SZ, SUF, int64_t);          \
    KT_TEST_DO4_SKIP(FUNC, SZ, SUF, uint32_t);         \
    KT_TEST_DO4_SKIP(FUNC, SZ, SUF, uint64_t);

#if defined(KT_AVX2_BUILD) || defined(KT_TEST_ADD_ONLY_AVX2)
// Test instantiation macros for all data types during AVX2 build
// b128 float double cfloat cdouble int32_t int64_t
// b256 float double cfloat cdouble int32_t int64_t
#define KT_TEST_DO_REAL_COMPLEX(func) \
    KT_TEST_DO3(func, b128, float);   \
    KT_TEST_DO3(func, b128, double);  \
    KT_TEST_DO3(func, b128, cfloat);  \
    KT_TEST_DO3(func, b128, cdouble); \
    KT_TEST_DO3(func, b256, float);   \
    KT_TEST_DO3(func, b256, double);  \
    KT_TEST_DO3(func, b256, cfloat);  \
    KT_TEST_DO3(func, b256, cdouble);

#define KT_TEST_DO_REAL(func)        \
    KT_TEST_DO3(func, b128, float);  \
    KT_TEST_DO3(func, b128, double); \
    KT_TEST_DO3(func, b256, float);  \
    KT_TEST_DO3(func, b256, double);

#define KT_TEST_DO_INTEGER(func)      \
    KT_TEST_DO3(func, b128, int32_t); \
    KT_TEST_DO3(func, b128, int64_t); \
    KT_TEST_DO3(func, b256, int32_t); \
    KT_TEST_DO3(func, b256, int64_t);

#define KT_TEST_DO_INDEX(func)                       \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b128, float);   \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b128, double);  \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b128, cfloat);  \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b128, cdouble); \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b256, float);   \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b256, double);  \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b256, cfloat);  \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b256, cdouble);

#else
// Test instantiation macros for all data types during AVX512 build
// b128 _Float16
// b256 _Float16
// b512 _Float16 float double cfloat cdouble int32_t int64_t
#define KT_TEST_DO_REAL_COMPLEX(func)                    \
    PASTE(KT_TEST_DO3, KT_FP16_E)(func, b128, _Float16); \
    PASTE(KT_TEST_DO3, KT_FP16_E)(func, b256, _Float16); \
    PASTE(KT_TEST_DO3, KT_FP16_E)(func, b512, _Float16); \
    KT_TEST_DO3(func, b512, float);                      \
    KT_TEST_DO3(func, b512, double);                     \
    KT_TEST_DO3(func, b512, cfloat);                     \
    KT_TEST_DO3(func, b512, cdouble);

#define KT_TEST_DO_REAL(func)                            \
    PASTE(KT_TEST_DO3, KT_FP16_E)(func, b128, _Float16); \
    PASTE(KT_TEST_DO3, KT_FP16_E)(func, b256, _Float16); \
    PASTE(KT_TEST_DO3, KT_FP16_E)(func, b512, _Float16); \
    KT_TEST_DO3(func, b512, float);                      \
    KT_TEST_DO3(func, b512, double);

#define KT_TEST_DO_INTEGER(func)      \
    KT_TEST_DO3(func, b512, int32_t); \
    KT_TEST_DO3(func, b512, int64_t);

#define KT_TEST_DO_INDEX(func)                                          \
    PASTE(KT_TEST_DO_ALL_INDEX_TYPES, KT_FP16_E)(func, b128, _Float16); \
    PASTE(KT_TEST_DO_ALL_INDEX_TYPES, KT_FP16_E)(func, b256, _Float16); \
    PASTE(KT_TEST_DO_ALL_INDEX_TYPES, KT_FP16_E)(func, b512, _Float16); \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b512, float);                      \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b512, double);                     \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b512, cfloat);                     \
    KT_TEST_DO_ALL_INDEX_TYPES(func, b512, cdouble);

#endif
// -----------------------------------------------------------------------------
// ADD NEW Unit-test drivers here
// -----------------------------------------------------------------------------
namespace TestsKT
{
    // Test that support _Float16, float, double, [cfp16], cfloat, cdouble
    // Also test that support int32_t and int64_t
    // Any new ukernel test driver is to be added here and in kt_kernels.cpp
    KT_TEST_DO_REAL_COMPLEX(kt_loadu_p_test);
    KT_TEST_DO_INTEGER(kt_loadu_p_test);

    KT_TEST_DO_REAL_COMPLEX(kt_load_p_test);
    KT_TEST_DO_INTEGER(kt_load_p_test);

    KT_TEST_DO_REAL_COMPLEX(kt_setzero_p_test);
    KT_TEST_DO_INTEGER(kt_setzero_p_test);

    KT_TEST_DO_REAL_COMPLEX(kt_set1_p_test);
    KT_TEST_DO_INTEGER(kt_set1_p_test);

    KT_TEST_DO_REAL_COMPLEX(kt_add_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_sub_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_mul_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_hsum_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_conj_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_dot_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_cdot_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_storeu_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_fmadd_B_test);
    KT_TEST_DO_REAL_COMPLEX(kt_hsum_B_test);
    KT_TEST_DO_REAL_COMPLEX(kt_div_p_test);
    KT_TEST_DO_REAL_COMPLEX(kt_pow2_p_test);

    // Test that only support real types
    KT_TEST_DO_REAL(kt_max_p_test);

    // Test that use indirect memory access
    KT_TEST_DO_INDEX(kt_scatter_p_test);
    KT_TEST_DO_INDEX(kt_set_p_test);
    KT_TEST_DO_INDEX(kt_fmadd_p_test);
    KT_TEST_DO_INDEX(kt_fmsub_p_test);
}
// -----------------------------------------------------------------------------

#undef KT_FP16_E
#undef PASTE_
#undef PASTE
#undef KT_TEST_DO_REAL_COMPLEX
#undef KT_TEST_DO_REAL
#undef KT_TEST_DO_INTEGER
#undef KT_TEST_DO_INDEX

#ifndef KT_TEST_ADD_ONLY_AVX2
// Don't undef macros, they are reused when #including again this file
// without KT_TEST_ADD_ONLY_AVX2 defined, to add the AVX512 tests
#undef KT_TEST_DO3
#undef KT_TEST_DO3_SKIP
#undef KT_TEST_DO4
#undef KT_TEST_DO4_SKIP
#endif