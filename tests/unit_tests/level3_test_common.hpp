/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Shared test helpers for Level-3 sparse kernel tests.
// Must be included AFTER the consumer file's own includes (aoclsparse.h,
// common_data_utils.h, gtest, aoclsparse_init.hpp, aoclsparse_interface.hpp)
// since common_data_utils.h has no include guard.

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wtype-limits"
#include "blis.hh"
#include "cblas.hh"
#pragma GCC diagnostic pop

#include <algorithm>
#include <complex>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

// CSR arrays always populated; CSC arrays filled on demand when use_csr=false.
template <typename T>
struct sparse_mat_data
{
    std::vector<T>              csr_val;
    std::vector<aoclsparse_int> csr_col_ind, csr_row_ptr;
    std::vector<T>              csc_val;
    std::vector<aoclsparse_int> csc_col_ptr, csc_row_ind;
};

// Generate a random sparse matrix and create an AOCL handle.
// use_csr=true (default): CSR handle; csc_* not touched.
// use_csr=false: CSC handle via aoclsparse_csr2csc; csr_* still populated for reference.
// pdescr non-null: descriptor created with base set.
template <typename T>
void gen_mat(aoclsparse_int        m,
             aoclsparse_int        n,
             aoclsparse_int        nnz,
             aoclsparse_index_base base,
             sparse_mat_data<T>   &src,
             aoclsparse_matrix    &mat,
             aoclsparse_mat_descr *pdescr  = nullptr,
             bool                  use_csr = true)
{
    std::vector<aoclsparse_int> coo_row;
    aoclsparse_matrix           mat_csr = nullptr;
    ASSERT_EQ(aoclsparse_init_matrix_random(base,
                                            m,
                                            n,
                                            nnz,
                                            aoclsparse_csr_mat,
                                            coo_row,
                                            src.csr_col_ind,
                                            src.csr_val,
                                            src.csr_row_ptr,
                                            mat_csr),
              aoclsparse_status_success);
    if(!use_csr)
    {
        aoclsparse_mat_descr descr_conv;
        ASSERT_EQ(aoclsparse_create_mat_descr(&descr_conv), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr_conv, base), aoclsparse_status_success);
        src.csc_col_ptr.resize(n + 1);
        // At least 1 element so .data() is non-null (create_csc validates even when nnz=0).
        src.csc_row_ind.resize((std::max)(nnz, (aoclsparse_int)1));
        src.csc_val.resize((std::max)(nnz, (aoclsparse_int)1));
        if(nnz > 0)
        {
            ASSERT_EQ(aoclsparse_csr2csc(m,
                                         n,
                                         nnz,
                                         descr_conv,
                                         base,
                                         src.csr_row_ptr.data(),
                                         src.csr_col_ind.data(),
                                         src.csr_val.data(),
                                         src.csc_row_ind.data(),
                                         src.csc_col_ptr.data(),
                                         src.csc_val.data()),
                      aoclsparse_status_success);
        }
        else
        {
            // Empty matrix: col_ptr is all base-index values.
            aoclsparse_int b = (base == aoclsparse_index_base_one) ? 1 : 0;
            std::fill(src.csc_col_ptr.begin(), src.csc_col_ptr.end(), b);
        }
        aoclsparse_destroy_mat_descr(descr_conv);
        ASSERT_EQ(aoclsparse_create_csc<T>(&mat,
                                           base,
                                           m,
                                           n,
                                           nnz,
                                           src.csc_col_ptr.data(),
                                           src.csc_row_ind.data(),
                                           src.csc_val.data()),
                  aoclsparse_status_success);
        aoclsparse_destroy(&mat_csr);
    }
    else
    {
        mat = mat_csr;
    }
    if(pdescr != nullptr)
    {
        ASSERT_EQ(aoclsparse_create_mat_descr(pdescr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(*pdescr, base), aoclsparse_status_success);
    }
}

// Source arrays for two sparse matrices (CSR layout always; CSC arrays populated on demand).
template <typename T>
struct spmm_mats
{
    std::vector<T>              val_a, val_b;
    std::vector<aoclsparse_int> col_ind_a, row_ptr_a;
    std::vector<aoclsparse_int> col_ind_b, row_ptr_b;
    // optional CSC arrays (filled by gen_AB when use_csr_a / use_csr_b is false)
    std::vector<aoclsparse_int> csc_col_ptr_a, csc_row_ind_a;
    std::vector<T>              csc_val_a;
    std::vector<aoclsparse_int> csc_col_ptr_b, csc_row_ind_b;
    std::vector<T>              csc_val_b;
};

// Build a SCOPED_TRACE label.
template <typename T>
std::string
    make_test_name(aoclsparse_int m_a, aoclsparse_int n_a, aoclsparse_int m_b, aoclsparse_int n_b)
{
    std::ostringstream s;
    s << typeid(T).name() << " A=" << m_a << "x" << n_a << " B=" << m_b << "x" << n_b;
    return s.str();
}

// Generate random sparse A and B; delegates to gen_mat() × 2.
// spmm_mats<T> fields are bridged via std::move for backward compat.
template <typename T>
void gen_AB(aoclsparse_int        m_a,
            aoclsparse_int        n_a,
            aoclsparse_int        m_b,
            aoclsparse_int        n_b,
            aoclsparse_int        nnz_a,
            aoclsparse_int        nnz_b,
            aoclsparse_index_base b_a,
            aoclsparse_index_base b_b,
            spmm_mats<T>         &src,
            aoclsparse_matrix    &A,
            aoclsparse_matrix    &B,
            aoclsparse_mat_descr *pDescrA   = nullptr,
            aoclsparse_mat_descr *pDescrB   = nullptr,
            bool                  use_csr_a = true,
            bool                  use_csr_b = true)
{
    sparse_mat_data<T> src_a, src_b;
    gen_mat(m_a, n_a, nnz_a, b_a, src_a, A, pDescrA, use_csr_a);
    gen_mat(m_b, n_b, nnz_b, b_b, src_b, B, pDescrB, use_csr_b);
    // Bridge into spmm_mats fields for backward compat.
    src.row_ptr_a     = std::move(src_a.csr_row_ptr);
    src.col_ind_a     = std::move(src_a.csr_col_ind);
    src.val_a         = std::move(src_a.csr_val);
    src.csc_col_ptr_a = std::move(src_a.csc_col_ptr);
    src.csc_row_ind_a = std::move(src_a.csc_row_ind);
    src.csc_val_a     = std::move(src_a.csc_val);
    src.row_ptr_b     = std::move(src_b.csr_row_ptr);
    src.col_ind_b     = std::move(src_b.csr_col_ind);
    src.val_b         = std::move(src_b.csr_val);
    src.csc_col_ptr_b = std::move(src_b.csc_col_ptr);
    src.csc_row_ind_b = std::move(src_b.csc_row_ind);
    src.csc_val_b     = std::move(src_b.csc_val);
}

// Export sparse C to dense (row-major). Caller receives m_c, n_c, and the dense vector.
template <typename T>
void export_and_dense(aoclsparse_matrix C,
                      aoclsparse_int   &m_c,
                      aoclsparse_int   &n_c,
                      std::vector<T>   &dense_c)
{
    aoclsparse_int        nnz_c;
    aoclsparse_int       *row_ptr_c = nullptr;
    aoclsparse_int       *col_ind_c = nullptr;
    T                    *val_c     = nullptr;
    aoclsparse_index_base base_c;
    ASSERT_EQ(aoclsparse_export_csr(C, &base_c, &m_c, &n_c, &nnz_c, &row_ptr_c, &col_ind_c, &val_c),
              aoclsparse_status_success);
    aoclsparse_mat_descr descrC;
    ASSERT_EQ(aoclsparse_create_mat_descr(&descrC), aoclsparse_status_success);
    ASSERT_EQ(aoclsparse_set_mat_index_base(descrC, base_c), aoclsparse_status_success);
    dense_c.resize(m_c * n_c, T{});
    aoclsparse_csr2dense(
        m_c, n_c, descrC, val_c, row_ptr_c, col_ind_c, dense_c.data(), n_c, aoclsparse_order_row);
    aoclsparse_destroy_mat_descr(descrC);
}

// Compute BLIS gemm reference for op(A) * op(B) and return the dense result vector.
// lda = n_a (stored cols of A), ldb = n_b (stored cols of B), ldc = n_c.
// op_b defaults to none for spmm; set explicitly for sp2m.
// NOTE: comparison (EXPECT_ARR_NEAR / EXPECT_COMPLEX_ARR_NEAR) is deliberately left
// to the calling test function so that assertion failures point directly to the test
// that triggered them, aiding debuggability.
template <typename T>
std::vector<T> gemm_ref(aoclsparse_int        m_c,
                        aoclsparse_int        n_c,
                        aoclsparse_int        m_a,
                        aoclsparse_int        n_a,
                        aoclsparse_int        n_b,
                        const std::vector<T> &dense_a,
                        const std::vector<T> &dense_b,
                        aoclsparse_operation  op_a,
                        aoclsparse_operation  op_b = aoclsparse_operation_none)
{
    const bool           op_a_t = (op_a == aoclsparse_operation_transpose
                         || op_a == aoclsparse_operation_conjugate_transpose);
    const aoclsparse_int k      = op_a_t ? m_a : n_a;

    std::vector<T> dense_c_exp(m_c * n_c, T{});

    if constexpr(std::is_same_v<T, aoclsparse_float_complex>)
    {
        T alpha = {1, 0};
        T beta  = {0, 0};
        blis::gemm(CblasRowMajor,
                   (CBLAS_TRANSPOSE)op_a,
                   (CBLAS_TRANSPOSE)op_b,
                   (int64_t)m_c,
                   (int64_t)n_c,
                   (int64_t)k,
                   *reinterpret_cast<const std::complex<float> *>(&alpha),
                   (std::complex<float> const *)dense_a.data(),
                   (int64_t)n_a,
                   (std::complex<float> const *)dense_b.data(),
                   (int64_t)n_b,
                   *reinterpret_cast<const std::complex<float> *>(&beta),
                   (std::complex<float> *)dense_c_exp.data(),
                   (int64_t)n_c);
    }
    else if constexpr(std::is_same_v<T, aoclsparse_double_complex>)
    {
        T alpha = {1, 0};
        T beta  = {0, 0};
        blis::gemm(CblasRowMajor,
                   (CBLAS_TRANSPOSE)op_a,
                   (CBLAS_TRANSPOSE)op_b,
                   (int64_t)m_c,
                   (int64_t)n_c,
                   (int64_t)k,
                   *reinterpret_cast<const std::complex<double> *>(&alpha),
                   (std::complex<double> const *)dense_a.data(),
                   (int64_t)n_a,
                   (std::complex<double> const *)dense_b.data(),
                   (int64_t)n_b,
                   *reinterpret_cast<const std::complex<double> *>(&beta),
                   (std::complex<double> *)dense_c_exp.data(),
                   (int64_t)n_c);
    }
    else
    {
        T alpha = 1;
        T beta  = 0;
        blis::gemm(CblasRowMajor,
                   (CBLAS_TRANSPOSE)op_a,
                   (CBLAS_TRANSPOSE)op_b,
                   (int64_t)m_c,
                   (int64_t)n_c,
                   (int64_t)k,
                   (T)alpha,
                   (T const *)dense_a.data(),
                   (int64_t)n_a,
                   (T const *)dense_b.data(),
                   (int64_t)n_b,
                   (T)beta,
                   (T *)dense_c_exp.data(),
                   (int64_t)n_c);
    }
    return dense_c_exp;
}
