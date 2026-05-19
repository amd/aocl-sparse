/* ************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file lp64_overflow_tests.cpp
 * @brief TDD tests for LP64 integer overflow detection in offset calculations
 *
 * These tests verify that:
 * 1. API validation rejects overflow-prone parameters early
 *
 * Test Strategy:
 * - API validation tests: Prove early rejection
 *
 * NOTE: These tests only apply to LP64 mode where aoclsparse_int = int32_t.
 * In ILP64 mode (aoclsparse_int = int64_t), the test values ((10-1) * 600M = 5.4B)
 * do not overflow 64-bit integers, so the tests are skipped via compile-time guard.
 */

#include "aoclsparse.h"
#include "gtest/gtest.h"

#include <climits>
#include <cstdint>
#include <vector>

// ============================================================
// LP64 Mode Only: Skip tests in ILP64 (64-bit integers don't overflow at these values)
// ============================================================

#if !defined(aoclsparse_ILP64)

namespace
{
    // ============================================================
    // Constants for overflow testing
    // ============================================================

    // For API tests: huge ld with small dimensions
    // The overflow check uses (dim-1)*ld, so with smallest dimension 10:
    // (10-1) * 600,000,000 = 5,400,000,000 > INT32_MAX (2,147,483,647)
    constexpr aoclsparse_int SMALL_DIM = 10;
    constexpr aoclsparse_int HUGE_LD   = 600000000;

    // ============================================================
    // These tests verify that APIs reject overflow-prone parameters
    // ============================================================

    /**
     * Base helper: Holds common CSR matrix data and handles resource cleanup.
     * Derived classes set up specific sparsity patterns and descriptor properties.
     */
    class SmallCSRMatrixBase
    {
    public:
        aoclsparse_int              m   = 5;
        aoclsparse_int              n   = 5;
        aoclsparse_int              nnz = 0;
        std::vector<aoclsparse_int> row_ptr;
        std::vector<aoclsparse_int> col_ind;
        std::vector<double>         val;
        aoclsparse_matrix           A     = nullptr;
        aoclsparse_mat_descr        descr = nullptr;

        virtual ~SmallCSRMatrixBase()
        {
            if(A)
                aoclsparse_destroy(&A);
            if(descr)
                aoclsparse_destroy_mat_descr(descr);
        }

    protected:
        void init()
        {
            val.resize(nnz, 1.0);
            aoclsparse_status status = aoclsparse_create_dcsr(&A,
                                                              aoclsparse_index_base_zero,
                                                              m,
                                                              n,
                                                              nnz,
                                                              row_ptr.data(),
                                                              col_ind.data(),
                                                              val.data());
            ASSERT_EQ(status, aoclsparse_status_success);
            status = aoclsparse_create_mat_descr(&descr);
            ASSERT_EQ(status, aoclsparse_status_success);
            aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero);
        }
    };

    /**
     * Helper: Create a small valid CSR matrix for API testing
     * Returns a 5x5 sparse matrix with 10 non-zeros
     */
    class SmallCSRMatrix : public SmallCSRMatrixBase
    {
    public:
        SmallCSRMatrix()
        {
            nnz = 10;
            // Simple 5x5 matrix with 2 elements per row
            row_ptr = {0, 2, 4, 6, 8, 10};
            col_ind = {0, 1, 1, 2, 2, 3, 3, 4, 0, 4};
            init();
        }
    };

    /**
     * Helper: Create a small valid lower-triangular CSR matrix for TRSV/TRSM testing
     * Returns a 5x5 lower-triangular matrix with unit diagonal
     */
    class SmallCSRTriangMatrix : public SmallCSRMatrixBase
    {
    public:
        SmallCSRTriangMatrix()
        {
            nnz = 11;
            // Lower triangular 5x5 matrix with unit diagonal
            // Row 0: [1, 0, 0, 0, 0]       -> 1 element
            // Row 1: [1, 1, 0, 0, 0]       -> 2 elements
            // Row 2: [1, 0, 1, 0, 0]       -> 2 elements
            // Row 3: [0, 1, 1, 1, 0]       -> 3 elements
            // Row 4: [1, 0, 1, 0, 1]       -> 3 elements
            row_ptr = {0, 1, 3, 5, 8, 11};
            col_ind = {0, 0, 1, 0, 2, 1, 2, 3, 0, 2, 4};
            init();
            aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
            aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        }
    };

    /**
     * Helper: Create a small valid upper-triangular CSR matrix for TRSV/TRSM testing
     * Returns a 5x5 upper-triangular matrix with unit diagonal
     */
    class SmallCSRTriangMatrixUpper : public SmallCSRMatrixBase
    {
    public:
        SmallCSRTriangMatrixUpper()
        {
            nnz = 11;
            // Upper triangular 5x5 matrix with unit diagonal
            // Row 0: [1, 1, 1, 0, 0]       -> 3 elements
            // Row 1: [0, 1, 0, 1, 0]       -> 2 elements
            // Row 2: [0, 0, 1, 1, 1]       -> 3 elements
            // Row 3: [0, 0, 0, 1, 1]       -> 2 elements
            // Row 4: [0, 0, 0, 0, 1]       -> 1 element
            row_ptr = {0, 3, 5, 8, 10, 11};
            col_ind = {0, 1, 2, 1, 3, 2, 3, 4, 3, 4, 4};
            init();
            aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_triangular);
            aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_upper);
        }
    };

    TEST(LP64OverflowAPI, CSRMM_RowMajor_LargeLD_RejectsEarly)
    {
        SmallCSRMatrix csr;

        // Small dimensions, huge leading dimension
        aoclsparse_int n   = SMALL_DIM;
        aoclsparse_int ldb = HUGE_LD;
        aoclsparse_int ldc = HUGE_LD;

        // Tiny buffers - just need non-null pointers
        // Validation should reject BEFORE any memory access
        std::vector<double> B(16, 1.0);
        std::vector<double> C(16, 0.0);

        aoclsparse_status status = aoclsparse_dcsrmm(aoclsparse_operation_none,
                                                     1.0, // alpha
                                                     csr.A,
                                                     csr.descr,
                                                     aoclsparse_order_row,
                                                     B.data(),
                                                     n,
                                                     ldb,
                                                     0.0, // beta
                                                     C.data(),
                                                     ldc);

        EXPECT_EQ(status, aoclsparse_status_invalid_size)
            << "Expected aoclsparse_status_invalid_size for overflow-prone ld";
    }

    TEST(LP64OverflowAPI, CSRMM_ColMajor_LargeLD_RejectsEarly)
    {
        SmallCSRMatrix csr;

        aoclsparse_int n   = SMALL_DIM;
        aoclsparse_int ldb = HUGE_LD;
        aoclsparse_int ldc = HUGE_LD;

        std::vector<double> B(16, 1.0);
        std::vector<double> C(16, 0.0);

        aoclsparse_status status = aoclsparse_dcsrmm(aoclsparse_operation_none,
                                                     1.0,
                                                     csr.A,
                                                     csr.descr,
                                                     aoclsparse_order_column,
                                                     B.data(),
                                                     n,
                                                     ldb,
                                                     0.0,
                                                     C.data(),
                                                     ldc);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

    TEST(LP64OverflowAPI, TRSM_ColMajor_LargeLD_RejectsEarly)
    {
        SmallCSRTriangMatrix tri;

        aoclsparse_int n   = SMALL_DIM;
        aoclsparse_int ldb = HUGE_LD;
        aoclsparse_int ldx = HUGE_LD;

        std::vector<double> B(16, 1.0);
        std::vector<double> X(16, 0.0);

        aoclsparse_status status = aoclsparse_dtrsm(aoclsparse_operation_none,
                                                    1.0, // alpha
                                                    tri.A,
                                                    tri.descr,
                                                    aoclsparse_order_column,
                                                    B.data(),
                                                    n,
                                                    ldb,
                                                    X.data(),
                                                    ldx);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

    TEST(LP64OverflowAPI, TRSM_RowMajor_LargeLD_RejectsEarly)
    {
        SmallCSRTriangMatrixUpper tri;

        aoclsparse_int n   = SMALL_DIM;
        aoclsparse_int ldb = HUGE_LD;
        aoclsparse_int ldx = HUGE_LD;

        std::vector<double> B(16, 1.0);
        std::vector<double> X(16, 0.0);

        aoclsparse_status status = aoclsparse_dtrsm(aoclsparse_operation_none,
                                                    1.0, // alpha
                                                    tri.A,
                                                    tri.descr,
                                                    aoclsparse_order_row,
                                                    B.data(),
                                                    n,
                                                    ldb,
                                                    X.data(),
                                                    ldx);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

    TEST(LP64OverflowAPI, TRSV_Strided_LargeInc_RejectsEarly)
    {
        SmallCSRTriangMatrix tri;

        // Large stride values that cause (m-1)*inc to overflow
        aoclsparse_int incb = HUGE_LD;
        aoclsparse_int incx = HUGE_LD;

        std::vector<double> b(16, 1.0);
        std::vector<double> x(16, 0.0);

        aoclsparse_status status = aoclsparse_dtrsv_strided(aoclsparse_operation_none,
                                                            1.0, // alpha
                                                            tri.A,
                                                            tri.descr,
                                                            b.data(),
                                                            incb,
                                                            x.data(),
                                                            incx);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

    TEST(LP64OverflowAPI, SP2MD_RowMajor_LargeLD_RejectsEarly)
    {
        // Reuse SmallCSRMatrix for both A and B (goal is to test overflow check)
        SmallCSRMatrix csrA;
        SmallCSRMatrix csrB;

        aoclsparse_int ldc = HUGE_LD;

        std::vector<double> C(16, 0.0);

        aoclsparse_status status = aoclsparse_dsp2md(aoclsparse_operation_none,
                                                     csrA.descr,
                                                     csrA.A,
                                                     aoclsparse_operation_none,
                                                     csrB.descr,
                                                     csrB.A,
                                                     1.0, // alpha
                                                     0.0, // beta
                                                     C.data(),
                                                     aoclsparse_order_row,
                                                     ldc);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

    TEST(LP64OverflowAPI, SYRKD_RowMajor_LargeLD_RejectsEarly)
    {
        SmallCSRMatrix csr;

        aoclsparse_int ldc = HUGE_LD;

        std::vector<double> C(16, 0.0);

        // SYRKD computes C = A^T * A (or A^H * A) with dense symmetric output
        aoclsparse_status status = aoclsparse_dsyrkd(aoclsparse_operation_transpose,
                                                     csr.A,
                                                     1.0, // alpha
                                                     0.0, // beta
                                                     C.data(),
                                                     aoclsparse_order_row,
                                                     ldc);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

    TEST(LP64OverflowAPI, SYPRD_RowMajor_LargeLD_RejectsEarly)
    {
        SmallCSRMatrix csr;

        aoclsparse_int ldb = HUGE_LD;
        aoclsparse_int ldc = HUGE_LD;

        std::vector<double> B(16, 1.0);
        std::vector<double> C(16, 0.0);

        // SYPRD computes C = A^T * B * A with dense symmetric output
        aoclsparse_status status = aoclsparse_dsyprd(aoclsparse_operation_none,
                                                     csr.A,
                                                     B.data(),
                                                     aoclsparse_order_row,
                                                     ldb,
                                                     1.0, // alpha
                                                     0.0, // beta
                                                     C.data(),
                                                     aoclsparse_order_row,
                                                     ldc);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

    TEST(LP64OverflowAPI, CSR2Dense_RowMajor_LargeLD_RejectsEarly)
    {
        SmallCSRMatrix csr;

        aoclsparse_int ld = HUGE_LD;

        // Tiny buffer
        std::vector<double> dense(16, 0.0);

        aoclsparse_status status = aoclsparse_dcsr2dense(csr.m,
                                                         csr.n,
                                                         csr.descr,
                                                         csr.val.data(),
                                                         csr.row_ptr.data(),
                                                         csr.col_ind.data(),
                                                         dense.data(),
                                                         ld,
                                                         aoclsparse_order_row);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

    TEST(LP64OverflowAPI, CSR2Dense_ColMajor_LargeLD_RejectsEarly)
    {
        SmallCSRMatrix csr;

        aoclsparse_int ld = HUGE_LD;

        std::vector<double> dense(16, 0.0);

        aoclsparse_status status = aoclsparse_dcsr2dense(csr.m,
                                                         csr.n,
                                                         csr.descr,
                                                         csr.val.data(),
                                                         csr.row_ptr.data(),
                                                         csr.col_ind.data(),
                                                         dense.data(),
                                                         ld,
                                                         aoclsparse_order_column);

        EXPECT_EQ(status, aoclsparse_status_invalid_size);
    }

} // namespace

#endif // !aoclsparse_ILP64