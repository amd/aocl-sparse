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
#include "aoclsparse.h"
#include "common_data_utils.h"
#include "gtest/gtest.h"
#include "aoclsparse_interface.hpp"
#define VERBOSE 1

namespace
{

    // Several tests in one when nullptr is passed instead
    // of valid data
    template <typename T>
    void test_ilu_nullptr()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A               = nullptr;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        aoclsparse_mat_descr        descr;
        T                           x[5] = {1.0};
        T                           b[5] = {1.0};
        m                                = 5;
        n                                = 5;
        nnz                              = 8;
        csr_row_ptr.assign({0, 2, 3, 4, 7, 8});
        csr_col_ind.assign({0, 3, 1, 2, 1, 3, 4, 4});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_create_csr<T>(&A,
                                           aoclsparse_index_base_zero,
                                           m,
                                           n,
                                           nnz,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_success);
        // In turns pass nullptr in every single pointer argument
        /*
            trans -> can be checked for invalid values
            A -> can be checked for nullptr
            descr -> can be checked for nullptr
            precond_csr_val -> is a output argument, will be passed as nullptr and expect LU factors in it
            approx_inv_diag ->  unused argument
            x -> can be checked for nullptr
            b -> can be checked for nullptr
        */
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, nullptr, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(
            aoclsparse_ilu_smoother<T>(trans, A, nullptr, &precond_csr_val, approx_inv_diag, x, b),
            aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A, descr, &precond_csr_val, approx_inv_diag, nullptr, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A, descr, &precond_csr_val, approx_inv_diag, x, nullptr),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }

    // tests with wrong scalar data n, m, nnz
    template <typename T>
    void test_ilu_wrong_size()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A_n_wrong       = nullptr;
        aoclsparse_matrix           A_m_wrong       = nullptr;
        aoclsparse_matrix           A_nnz_wrong     = nullptr;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        aoclsparse_int              wrong = -1;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        aoclsparse_mat_descr        descr;
        T                           x[5] = {1.0};
        T                           b[5] = {1.0};

        m   = 5;
        n   = 5;
        nnz = 8;
        csr_row_ptr.assign({0, 2, 3, 4, 7, 8});
        csr_col_ind.assign({0, 3, 1, 2, 1, 3, 4, 4});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_create_csr<T>(&A_n_wrong,
                                           aoclsparse_index_base_zero,
                                           m,
                                           wrong,
                                           nnz,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_invalid_size);
        ASSERT_EQ(aoclsparse_create_csr<T>(&A_m_wrong,
                                           aoclsparse_index_base_zero,
                                           wrong,
                                           n,
                                           nnz,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_invalid_size);
        ASSERT_EQ(aoclsparse_create_csr<T>(&A_nnz_wrong,
                                           aoclsparse_index_base_zero,
                                           m,
                                           n,
                                           wrong,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_invalid_size);

        // aoclsparse_matrix "A" which contains members m,n and nnz are validated during matrix creation.
        // the below call should return aoclsparse_status_invalid_pointer, since matrix "A" is nullptr
        // which never got created due to wrong m,n,nnz values

        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_n_wrong, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_m_wrong, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_pointer);
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_nnz_wrong, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_pointer);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A_n_wrong), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A_m_wrong), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A_nnz_wrong), aoclsparse_status_success);
    }
    // zero matrix size is valid - just do nothing, zero nnz is an error
    template <typename T>
    void test_ilu_do_nothing()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A_mn_zero       = nullptr;
        aoclsparse_matrix           A_m_zero        = nullptr;
        aoclsparse_matrix           A_nnz_zero      = nullptr;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n;
        aoclsparse_int              zero = 0;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        aoclsparse_mat_descr        descr;
        T                           x[5] = {1.0};
        T                           b[5] = {1.0};

        m = 5;
        n = 5;
        csr_row_ptr.assign({0, 0, 0, 0, 0, 0});
        csr_col_ind.assign({0, 3, 1, 2, 1, 3, 4, 4});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);
        /*
            pass zero arguments for m, n and nnz to test the creation API.
        */
        ASSERT_EQ(aoclsparse_create_csr<T>(&A_mn_zero,
                                           aoclsparse_index_base_zero,
                                           zero,
                                           zero,
                                           zero,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(&A_m_zero,
                                           aoclsparse_index_base_zero,
                                           zero,
                                           n,
                                           zero,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_create_csr<T>(&A_nnz_zero,
                                           aoclsparse_index_base_zero,
                                           m,
                                           n,
                                           zero,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_success);

        /*
            to check if the ILU API exits gracefully with success
            when the values zero are passed for m, n and nnz
        */
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_mn_zero, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_success);
        // non-square matrix is an error even with zero size
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_m_zero, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_invalid_size);
        // empty matrix has a missing diagonal, so far reported as numerical error
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                      trans, A_nnz_zero, descr, &precond_csr_val, approx_inv_diag, x, b),
                  aoclsparse_status_numerical_error);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A_mn_zero), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A_m_zero), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A_nnz_zero), aoclsparse_status_success);
    }
    // test one-base and zero-based indexing support
    template <typename T>
    void test_ilu_baseOneIndexing()
    {
        aoclsparse_operation        trans              = aoclsparse_operation_none;
        int                         invalid_index_base = 2;
        T                          *approx_inv_diag    = NULL;
        T                          *precond_csr_val    = NULL;
        aoclsparse_int              m, n, nnz;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        //std::vector<T>              ilu0_precond_gold(nnz);
        aoclsparse_mat_descr descr;
        T                    x[5] = {1.0};
        T                    b[5] = {1.0};
        m                         = 5;
        n                         = 5;
        nnz                       = 8;
        T ilu0_precond_gold[8]    = {1.00, 2.00, 3.00, 4.00, 1.6666666666666667, 6.00, 7.00, 8.00};

        csr_row_ptr.assign({1, 3, 4, 5, 8, 9});
        csr_col_ind.assign({1, 4, 2, 3, 2, 4, 5, 5});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_one),
                  aoclsparse_status_success);
        /*
            check if the One-Based Indexing is supported
        */
        aoclsparse_matrix A = nullptr;
        ASSERT_EQ(aoclsparse_create_csr<T>(&A,
                                           aoclsparse_index_base_one,
                                           m,
                                           n,
                                           nnz,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_success);

        EXPECT_EQ(
            aoclsparse_ilu_smoother<T>(trans, A, descr, &precond_csr_val, approx_inv_diag, x, b),
            aoclsparse_status_success);

        EXPECT_ARR_NEAR(nnz, precond_csr_val, ilu0_precond_gold, expected_precision<T>(1.0));

        descr->base = (aoclsparse_index_base)invalid_index_base;
        EXPECT_EQ(
            aoclsparse_ilu_smoother<T>(trans, A, descr, &precond_csr_val, approx_inv_diag, x, b),
            aoclsparse_status_invalid_value);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }
    // test not-implemented/supported scenarios
    template <typename T>
    void test_ilu_unsupported()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A = nullptr;
        std::vector<T>              approx_inv_diag;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        aoclsparse_mat_descr        descr;
        T                           x[5] = {1.0};
        T                           b[5] = {1.0};
        m                                = 5;
        n                                = 5;
        nnz                              = 8;
        csr_row_ptr.assign({0, 2, 3, 4, 7, 8});
        csr_col_ind.assign({0, 3, 1, 2, 1, 3, 4, 4});
        csr_val.assign({1, 2, 3, 4, 5, 6, 7, 8});

        ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
        ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                  aoclsparse_status_success);

        ASSERT_EQ(aoclsparse_create_csr<T>(&A,
                                           aoclsparse_index_base_zero,
                                           m,
                                           n,
                                           nnz,
                                           &csr_row_ptr[0],
                                           &csr_col_ind[0],
                                           &csr_val[0]),
                  aoclsparse_status_success);
        /*
            check if the transpose operation is supported
        */
        trans = aoclsparse_operation_transpose;
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(trans, A, descr, &precond_csr_val, nullptr, x, b),
                  aoclsparse_status_not_implemented);

        /*
            check if the conjugate transpose operation is supported
        */
        trans = aoclsparse_operation_conjugate_transpose;
        EXPECT_EQ(aoclsparse_ilu_smoother<T>(trans, A, descr, &precond_csr_val, nullptr, x, b),
                  aoclsparse_status_not_implemented);

        EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
        EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
    }
    /*
        ILU0 reference results as generated by octave are loaded for validation.

        ILU0 Octave code for validation:
        opts.type = "nofill";
        [L_ilu,U_ilu] = (ilu (A, opts));

        # Merge the L and U factors into a single matrix
        CTF = U_ilu + L_ilu - eye(size(A));

        # Get the non-zero indices of the merged matrix in raster scan order
        [r, c, v] = find(CTF');
        # Concatenate the non-zero values into a vector
        CTF_data = [v'];
    */
    template <typename T>
    aoclsparse_status load_ILU_reference_results(matrix_id mid, std::vector<T> &ref)
    {
        switch(mid)
        {
        case N5_full_sorted:
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref = {bcd((T)1, (T)0.10000000000000001),
                       bcd((T)2, (T)0.20000000000000001),
                       bcd((T)3, (T)0.29999999999999999),
                       bcd((T)4, (T)0.40000000000000002),
                       bcd((T)1.6666666666666667, (T)1.8320511957510834e-17),
                       bcd((T)6, (T)0.59999999999999998),
                       bcd((T)7, (T)0.69999999999999996),
                       bcd((T)8, (T)0.80000000000000004)};
            }
            else
            {
                ref = {1, 2, 3, 4, 1.6666666666666667, 6, 7, 8};
            }
            break;
#if 0
        case sample_cg_mat:
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref = {bcd((T)19, (T)1.8999999999999999),
                       bcd((T)10, (T)1),
                       bcd((T)0.052631578947368418, (T)7.2317810358595388e-19),
                       bcd((T)0.80000000000000004, (T)0),
                       bcd((T)11, (T)1.1000000000000001),
                       bcd((T)13, (T)1.3),
                       bcd((T)0.20000000000000001, (T)0),
                       bcd((T)11, (T)1.1000000000000001),
                       bcd((T)0.10526315789473684, (T)1.4463562071719078e-18),
                       bcd((T)0.090909090909090912, (T)0),
                       bcd((T)9, (T)0.90000000000000002),
                       bcd((T)0.36842105263157893, (T)0),
                       bcd((T)0.69230769230769229, (T)0),
                       bcd((T)0.45454545454545453, (T)0),
                       bcd((T)12, (T)1.2),
                       bcd((T)0.45454545454545453, (T)0),
                       bcd((T)0.55555555555555558, (T)0),
                       bcd((T)9, (T)0.90000000000000002)};
            }
            else
            {
                ref = {19,
                       10,
                       0.052631578947368418,
                       0.80000000000000004,
                       11,
                       13,
                       0.20000000000000001,
                       11,
                       0.10526315789473684,
                       0.090909090909090912,
                       9,
                       0.36842105263157893,
                       0.69230769230769229,
                       0.45454545454545453,
                       12,
                       0.45454545454545453,
                       0.55555555555555558,
                       9};
            }
            break;
#endif
        case sample_gmres_mat_01:
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref = {bcd((T)0.75, (T)0.074999999999999997),
                       bcd((T)0.14000000000000001, (T)0.014000000000000002),
                       bcd((T)0.11, (T)0.010999999999999999),
                       bcd((T)0.14000000000000001, (T)0.014000000000000002),
                       bcd((T)0.11, (T)0.010999999999999999),
                       bcd((T)0.10666666666666667, (T)0),
                       bcd((T)0.67506666666666648, (T)0.067506666666666659),
                       bcd((T)0.11, (T)0.010999999999999999),
                       bcd((T)0.065066666666666662, (T)0.0065066666666666667),
                       bcd((T)0.11, (T)0.010999999999999999),
                       bcd((T)0.1333201659095398, (T)0),
                       bcd((T)0.65533478174995063, (T)0.065533478174995066),
                       bcd((T)0.080000000000000002, (T)0.0080000000000000002),
                       bcd((T)0.075334781749950619, (T)0.0075334781749950615),
                       bcd((T)0.080000000000000002, (T)0.0080000000000000002),
                       bcd((T)0.12, (T)0),
                       bcd((T)0.21363126740527316, (T)0),
                       bcd((T)0.69970949860757825, (T)0.069970949860757803),
                       bcd((T)0.12290949860757816, (T)0.012290949860757816),
                       bcd((T)0.076799999999999993, (T)0.0076799999999999993),
                       bcd((T)0.053333333333333337, (T)0),
                       bcd((T)0.048192771084337345, (T)0),
                       bcd((T)0.52939759036144585, (T)0.052939759036144583),
                       bcd((T)0.13469879518072292, (T)0.013469879518072291),
                       bcd((T)0.11, (T)0.010999999999999999),
                       bcd((T)0.25, (T)0.025000000000000001),
                       bcd((T)0.074066758838633229, (T)-1.2721321321474231e-18),
                       bcd((T)0.06386454327357774, (T)0),
                       bcd((T)0.1420118343195266, (T)0),
                       bcd((T)0.41791261211443054, (T)0.04179126121144304),
                       bcd((T)0.080000000000000002, (T)0.0080000000000000002),
                       bcd((T)0.11449704142011835, (T)0.011449704142011834),
                       bcd((T)0.061037504972935176, (T)0),
                       bcd((T)0.05018797039635451, (T)1.2273293412727432e-18),
                       bcd((T)0.17000455166135636, (T)0),
                       bcd((T)0.44024792064186813, (T)0.044024792064186809),
                       bcd((T)0.086145563873559969, (T)0.0086145563873559969),
                       bcd((T)0.13749886208466089, (T)0.013749886208466089),
                       bcd((T)0.06666666666666668, (T)0),
                       bcd((T)0.060977686813703864, (T)1.2273293412727432e-18),
                       bcd((T)0.33499826504797142, (T)4.1098257057299615e-18),
                       bcd((T)0.23283531456100504, (T)-3.9013199506144358e-18),
                       bcd((T)0.49112598965300114, (T)0.049112598965300097),
                       bcd((T)0.17962909896587237, (T)0.017962909896587238),
                       bcd((T)0.15111515703231676, (T)0),
                       bcd((T)0.14272115433996413, (T)0),
                       bcd((T)0.16667275252422142, (T)-1.9506599753072179e-18),
                       bcd((T)0.11040790051155086, (T)3.4971637262164698e-18),
                       bcd((T)0.073130275321728666, (T)0.0073130275321728541)};
            }
            else
            {
                ref = {0.75,
                       0.14000000000000001,
                       0.11,
                       0.14000000000000001,
                       0.11,
                       0.10666666666666667,
                       0.67506666666666648,
                       0.11,
                       0.065066666666666662,
                       0.11,
                       0.1333201659095398,
                       0.65533478174995063,
                       0.080000000000000002,
                       0.075334781749950619,
                       0.080000000000000002,
                       0.12,
                       0.21363126740527313,
                       0.69970949860757825,
                       0.12290949860757816,
                       0.076799999999999993,
                       0.053333333333333337,
                       0.048192771084337352,
                       0.52939759036144585,
                       0.13469879518072289,
                       0.11,
                       0.25,
                       0.074066758838633229,
                       0.06386454327357774,
                       0.14201183431952663,
                       0.41791261211443054,
                       0.080000000000000002,
                       0.11449704142011834,
                       0.061037504972935176,
                       0.05018797039635451,
                       0.17000455166135636,
                       0.44024792064186813,
                       0.086145563873559969,
                       0.13749886208466089,
                       0.066666666666666666,
                       0.060977686813703885,
                       0.33499826504797137,
                       0.23283531456100504,
                       0.49112598965300114,
                       0.17962909896587237,
                       0.15111515703231679,
                       0.14272115433996413,
                       0.1666727525242214,
                       0.11040790051155087,
                       0.073130275321728444};
            }
            break;
#if 0
        case sample_gmres_mat_02:
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref = {bcd((T)3, (T)0.29999999999999999),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)0.33333333333333337, (T)4.5801279893777085e-18),
                       bcd((T)4.666666666666667, (T)0.46666666666666667),
                       bcd((T)0.66666666666666663, (T)0.066666666666666652),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)0.33333333333333337, (T)4.5801279893777085e-18),
                       bcd((T)0.14285714285714285, (T)0),
                       bcd((T)6.5714285714285721, (T)0.65714285714285714),
                       bcd((T)0.85714285714285721, (T)0.085714285714285715),
                       bcd((T)0.66666666666666663, (T)0.066666666666666652),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)0.21428571428571427, (T)2.9443679931713836e-18),
                       bcd((T)0.13043478260869565, (T)0),
                       bcd((T)10.673913043478262, (T)1.0673913043478263),
                       bcd((T)0.91304347826086962, (T)0.091304347826086957),
                       bcd((T)0.7857142857142857, (T)0.07857142857142857),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)0.33333333333333337, (T)4.5801279893777085e-18),
                       bcd((T)0.10144927536231883, (T)0),
                       bcd((T)0.085539714867617106, (T)-1.2872864817395596e-18),
                       bcd((T)12.520932337632948, (T)1.2520932337632948),
                       bcd((T)0.9327902240325866, (T)0.093279022403258666),
                       bcd((T)0.89855072463768115, (T)0.089855072463768115),
                       bcd((T)0.66666666666666663, (T)0.066666666666666652),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)0.21428571428571427, (T)2.9443679931713836e-18),
                       bcd((T)0.073610707011929011, (T)0),
                       bcd((T)0.074498463762877276, (T)1.0973930373247837e-18),
                       bcd((T)16.658385862930025, (T)1.6658385862930027),
                       bcd((T)0.93305935140147256, (T)0.093305935140147261),
                       bcd((T)0.92638929298807104, (T)0.092638929298807099),
                       bcd((T)0.7857142857142857, (T)0.07857142857142857),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)0.15217391304347824, (T)2.0909279951506929e-18),
                       bcd((T)0.071763883104534854, (T)1.0973930373247837e-18),
                       bcd((T)0.056011390243865901, (T)-8.2483285482716888e-19),
                       bcd((T)18.731080646338096, (T)1.873108064633809),
                       bcd((T)0.94811164779270607, (T)0.094811164779270624),
                       bcd((T)0.95215741126364339, (T)0.095215741126364356),
                       bcd((T)0.84782608695652173, (T)0.084782608695652184),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)0.09368635437881874, (T)0),
                       bcd((T)0.055610987799818525, (T)-1.6496657096543378e-18),
                       bcd((T)0.05061702876059427, (T)2.9342426585130002e-18),
                       bcd((T)22.806805627406362, (T)2.2806805627406361),
                       bcd((T)0.95180462092945517, (T)0.09518046209294552),
                       bcd((T)0.9563056524429997, (T)0.095630565244299984),
                       bcd((T)0.90631364562118122, (T)0.090631364562118136),
                       bcd((T)1, (T)0.10000000000000001),
                       bcd((T)0.33333333333333337, (T)4.5801279893777085e-18),
                       bcd((T)0.053244171335622634, (T)-1.0973930373247837e-18),
                       bcd((T)0.050833020755254137, (T)2.9342426585130002e-18),
                       bcd((T)0.041733359615504222, (T)6.0246858734226472e-19),
                       bcd((T)28.543047510464941, (T)2.8543047510464934),
                       bcd((T)0.96009015230425687, (T)0.096009015230425696),
                       bcd((T)0.95690243892489324, (T)0.095690243892489327),
                       bcd((T)0.94675582866437735, (T)0.094675582866437749),
                       bcd((T)0.66666666666666663, (T)0.066666666666666652),
                       bcd((T)0.21428571428571427, (T)2.9443679931713836e-18),
                       bcd((T)0.047166291631096077, (T)-8.2483285482716888e-19),
                       bcd((T)0.041930714369478879, (T)1.2049371746845294e-18),
                       bcd((T)0.033636567782478456, (T)9.6278324611941938e-19),
                       bcd((T)30.676262339927785, (T)3.0676262339927796),
                       bcd((T)0.96781308625188389, (T)0.096781308625188397),
                       bcd((T)0.96199762139629719, (T)0.096199762139629719),
                       bcd((T)0.95283370836890391, (T)0.095283370836890396),
                       bcd((T)0.7857142857142857, (T)0.07857142857142857),
                       bcd((T)0.15217391304347824, (T)2.0909279951506929e-18),
                       bcd((T)0.045263063192366899, (T)2.2006819938847504e-18),
                       bcd((T)0.033524886877410592, (T)9.6278324611941938e-19),
                       bcd((T)0.031549250541914688, (T)-8.9583169004581353e-19),
                       bcd((T)36.74683705765289, (T)3.6746837057652888),
                       bcd((T)0.96964969602184226, (T)0.096964969602184231),
                       bcd((T)0.96826011794349764, (T)0.09682601179434977),
                       bcd((T)0.95473693680763305, (T)0.095473693680763319),
                       bcd((T)0.84782608695652173, (T)0.084782608695652184),
                       bcd((T)0.09368635437881874, (T)0),
                       bcd((T)0.039738736780046344, (T)1.2049371746845294e-18),
                       bcd((T)0.031359675136960044, (T)-8.9583169004581353e-19),
                       bcd((T)0.026387296803274205, (T)3.7392018111859657e-19),
                       bcd((T)40.814543519004026, (T)4.0814543519004021),
                       bcd((T)0.97445023288505161, (T)0.09744502328850517),
                       bcd((T)0.97011944444600628, (T)0.097011944444600637),
                       bcd((T)0.96026126321995364, (T)0.096026126321995364),
                       bcd((T)0.079866257003433944, (T)0),
                       bcd((T)0.033169402402363012, (T)1.4441748691791292e-18),
                       bcd((T)0.026349481900289101, (T)3.7392018111859657e-19),
                       bcd((T)0.023875073659254061, (T)3.3665411354497546e-19),
                       bcd((T)42.839952194398883, (T)4.2839952194398876),
                       bcd((T)0.97683832680557692, (T)0.097683832680557706),
                       bcd((T)0.97484317636404982, (T)0.097484317636404985),
                       bcd((T)0.97788706506509138, (T)0.097788706506509143),
                       bcd((T)0.0600298257123041, (T)-8.2483285482716888e-19),
                       bcd((T)0.031060945359327859, (T)-8.9583169004581353e-19),
                       bcd((T)0.023768964707256868, (T)3.3665411354497546e-19),
                       bcd((T)0.022802040543203356, (T)9.6221283621761e-19),
                       bcd((T)46.865041616566685, (T)4.6865041616566696),
                       bcd((T)0.97777158636928174, (T)0.09777715863692818),
                       bcd((T)0.97717558392477899, (T)0.097717558392477907),
                       bcd((T)0.97559497150338526, (T)0.097559497150338528),
                       bcd((T)0.053387202739714798, (T)2.2006819938847504e-18),
                       bcd((T)0.025981472509041419, (T)7.4784036223719314e-19),
                       bcd((T)0.022755468352075002, (T)6.414752241450734e-19),
                       bcd((T)0.020863559545493751, (T)-2.9319047832181871e-19),
                       bcd((T)52.879224517021164, (T)5.2879224517021157),
                       bcd((T)0.9796126390183828, (T)0.097961263901838277),
                       bcd((T)0.97774772183900782, (T)0.097774772183900779),
                       bcd((T)0.977972229829291, (T)0.097797222982929102),
                       bcd((T)0.043846561256185967, (T)6.0246858734226472e-19),
                       bcd((T)0.023527428716012412, (T)3.3665411354497546e-19),
                       bcd((T)0.02085084212491875, (T)-2.9319047832181871e-19),
                       bcd((T)0.018525472867005788, (T)0),
                       bcd((T)58.895038239131601, (T)5.8895038239131612),
                       bcd((T)0.98188676110829476, (T)0.098188676110829473),
                       bcd((T)0.97965802327131835, (T)0.09796580232713184),
                       bcd((T)0.33333333333333337, (T)4.5801279893777085e-18),
                       bcd((T)0.023356534246115168, (T)0),
                       bcd((T)0.022826520922050546, (T)6.414752241450734e-19),
                       bcd((T)0.018490205383482564, (T)0),
                       bcd((T)0.016671807854535023, (T)-4.6660582551429975e-19),
                       bcd((T)60.594325300679671, (T)6.0594325300679666),
                       bcd((T)0.98366732967286696, (T)0.098366732967286707),
                       bcd((T)0.98191709261111404, (T)0.098191709261111398),
                       bcd((T)0.21428571428571427, (T)2.9443679931713836e-18),
                       bcd((T)0.025613103611114029, (T)-8.9583169004581353e-19),
                       bcd((T)0.020817115228134456, (T)-2.9319047832181871e-19),
                       bcd((T)0.016633965314592575, (T)-2.3330291275714987e-19),
                       bcd((T)0.016233654303298817, (T)2.2676024363586736e-19),
                       bcd((T)66.713016518409162, (T)6.6713016518409169),
                       bcd((T)0.98405989736405097, (T)0.098405989736405103),
                       bcd((T)0.15217391304347824, (T)2.0909279951506929e-18),
                       bcd((T)0.02307208333676037, (T)3.7392018111859657e-19),
                       bcd((T)0.018494451058269471, (T)2.598446572095676e-19),
                       bcd((T)0.016204769798799958, (T)0),
                       bcd((T)0.014750643108642884, (T)0),
                       bcd((T)70.779750656491444, (T)7.0779750656491434)};
            }
            else
            {
                ref = {3,
                       1,
                       1,
                       1,
                       1,
                       1,
                       0.33333333333333331,
                       4.666666666666667,
                       0.66666666666666674,
                       1,
                       1,
                       1,
                       1,
                       0.33333333333333331,
                       0.14285714285714288,
                       6.5714285714285721,
                       0.8571428571428571,
                       0.66666666666666674,
                       1,
                       1,
                       1,
                       0.21428571428571427,
                       0.13043478260869562,
                       10.673913043478262,
                       0.91304347826086962,
                       0.7857142857142857,
                       1,
                       1,
                       0.33333333333333331,
                       0.10144927536231885,
                       0.085539714867617106,
                       12.520932337632948,
                       0.9327902240325866,
                       0.89855072463768115,
                       0.66666666666666674,
                       1,
                       0.21428571428571427,
                       0.073610707011928997,
                       0.07449846376287729,
                       16.658385862930025,
                       0.93305935140147256,
                       0.92638929298807104,
                       0.7857142857142857,
                       1,
                       0.15217391304347824,
                       0.071763883104534854,
                       0.056011390243865908,
                       18.731080646338096,
                       0.94811164779270607,
                       0.95215741126364339,
                       0.84782608695652173,
                       1,
                       0.093686354378818726,
                       0.055610987799818525,
                       0.05061702876059427,
                       22.806805627406362,
                       0.95180462092945517,
                       0.9563056524429997,
                       0.90631364562118133,
                       1,
                       0.33333333333333331,
                       0.053244171335622641,
                       0.05083302075525413,
                       0.041733359615504222,
                       28.543047510464941,
                       0.96009015230425687,
                       0.95690243892489324,
                       0.94675582866437735,
                       0.66666666666666674,
                       0.21428571428571427,
                       0.047166291631096077,
                       0.041930714369478879,
                       0.033636567782478456,
                       30.676262339927785,
                       0.96781308625188389,
                       0.96199762139629719,
                       0.95283370836890391,
                       0.7857142857142857,
                       0.15217391304347824,
                       0.045263063192366892,
                       0.033524886877410592,
                       0.031549250541914688,
                       36.74683705765289,
                       0.96964969602184226,
                       0.96826011794349764,
                       0.95473693680763305,
                       0.84782608695652173,
                       0.093686354378818726,
                       0.039738736780046351,
                       0.031359675136960044,
                       0.026387296803274205,
                       40.814543519004026,
                       0.97445023288505161,
                       0.97011944444600628,
                       0.96026126321995364,
                       0.079866257003433944,
                       0.033169402402363012,
                       0.026349481900289101,
                       0.023875073659254061,
                       42.839952194398883,
                       0.97683832680557692,
                       0.97484317636404982,
                       0.97788706506509138,
                       0.0600298257123041,
                       0.031060945359327859,
                       0.023768964707256868,
                       0.022802040543203356,
                       46.865041616566685,
                       0.97777158636928174,
                       0.97717558392477899,
                       0.97559497150338526,
                       0.053387202739714798,
                       0.025981472509041419,
                       0.022755468352075002,
                       0.020863559545493751,
                       52.879224517021164,
                       0.9796126390183828,
                       0.97774772183900782,
                       0.977972229829291,
                       0.043846561256185974,
                       0.023527428716012412,
                       0.02085084212491875,
                       0.018525472867005788,
                       58.895038239131601,
                       0.98188676110829476,
                       0.97965802327131835,
                       0.33333333333333331,
                       0.023356534246115172,
                       0.022826520922050546,
                       0.018490205383482561,
                       0.016671807854535023,
                       60.594325300679671,
                       0.98366732967286696,
                       0.98191709261111404,
                       0.21428571428571427,
                       0.025613103611114033,
                       0.020817115228134453,
                       0.016633965314592575,
                       0.016233654303298817,
                       66.713016518409162,
                       0.98405989736405097,
                       0.15217391304347824,
                       0.02307208333676037,
                       0.018494451058269471,
                       0.016204769798799958,
                       0.014750643108642884,
                       70.779750656491444};
            }
            break;
#endif
        case S5_sym_fullmatrix:
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref = {bcd((T)0.10000000000000009, (T)0.01),
                       bcd((T)0.4081472627313027, (T)0.040814726273130271),
                       bcd((T)0.6498582641797952, (T)0.064985826417979523),
                       bcd((T)4.0814726273130262, (T)6.8701919840665627e-17),
                       bcd((T)-1.5658418807505499, (T)-0.15658418807505503),
                       bcd((T)-0.86888798575545645, (T)-0.086888798575545642),
                       bcd((T)1.3257997547570231, (T)0.13257997547570227),
                       bcd((T)6.4985826417979515, (T)1.3740383968133125e-16),
                       bcd((T)-4.6822131409804557, (T)-0.46822131409804579),
                       bcd((T)-0.26103653234662899, (T)-0.026103653234662898),
                       bcd((T)0.554901485543978, (T)-1.7550155142800232e-17),
                       bcd((T)0.055750672702603185, (T)-2.9345917313913413e-18),
                       bcd((T)0.59670019634529714, (T)0.059670019634529683)};
            }
            else
            {
                ref = {(T)0.10000000000000009,
                       (T)0.4081472627313027,
                       (T)0.6498582641797952,
                       (T)4.0814726273130271,
                       (T)-1.5658418807505503,
                       (T)-0.86888798575545645,
                       (T)1.3257997547570231,
                       (T)6.4985826417979515,
                       (T)-4.6822131409804557,
                       (T)-0.26103653234662899,
                       (T)0.55490148554397778,
                       (T)0.055750672702603178,
                       (T)0.5967001963452967};
            }
            break;
#if 0
        case S5_sym_lowerfill:
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref = {bcd((T)0.10000000000000009, (T)0.01),
                       bcd((T)4.0814726273130262, (T)6.8701919840665627e-17),
                       bcd((T)0.10000000000000009, (T)0.01),
                       bcd((T)1.3257997547570231, (T)0.13257997547570227),
                       bcd((T)6.4985826417979515, (T)1.3740383968133125e-16),
                       bcd((T)-0.45905550575269161, (T)-0.045905550575269162),
                       bcd((T)-8.6888798575545643, (T)0),
                       bcd((T)0.5686382781067395, (T)-7.4829643670233664e-18),
                       bcd((T)0.10000000000000009, (T)0.01)};
            }
            else
            {
                ref = {0.10000000000000009,
                       4.0814726273130271,
                       0.10000000000000009,
                       1.3257997547570231,
                       6.4985826417979515,
                       -0.45905550575269161,
                       -8.6888798575545643,
                       0.5686382781067395,
                       0.10000000000000009};
            }
            break;
        case S5_sym_upperfill:
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref = {bcd((T)0.10000000000000009, (T)0.01),
                       bcd((T)0.4081472627313027, (T)0.040814726273130271),
                       bcd((T)0.6498582641797952, (T)0.064985826417979523),
                       bcd((T)0.10000000000000009, (T)0.01),
                       bcd((T)-0.86888798575545645, (T)-0.086888798575545642),
                       bcd((T)1.3257997547570231, (T)0.13257997547570227),
                       bcd((T)-0.45905550575269161, (T)-0.045905550575269162),
                       bcd((T)-0.26103653234662899, (T)-0.026103653234662898),
                       bcd((T)0.10000000000000009, (T)0.01)};
            }
            else
            {
                ref = {0.10000000000000009,
                       0.4081472627313027,
                       0.6498582641797952,
                       0.10000000000000009,
                       -0.86888798575545645,
                       1.3257997547570231,
                       -0.45905550575269161,
                       -0.26103653234662899,
                       0.10000000000000009};
            }
            break;
        case S4_herm_rsym:
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                ref = {bcd((T)-154, (T)0),
                       bcd((T)1, (T)-1),
                       bcd((T)1, (T)2),
                       bcd((T)0, (T)-1),
                       bcd((T)160, (T)0),
                       bcd((T)-2, (T)0),
                       bcd((T)3, (T)-2),
                       bcd((T)142, (T)0),
                       bcd((T)4, (T)0),
                       bcd((T)178, (T)0)};
            }
            else
            {
                //real lower symmetric
                ref = {-154,
                       -0.0064935064935064939,
                       160,
                       -0.0064935064935064939,
                       -0.012500000000000001,
                       142,
                       0.018749999999999999,
                       0.028169014084507043,
                       178};
            }
            break;
#endif
        default:
            // no data with id found
            return aoclsparse_status_internal_error;
            break;
        }
        return aoclsparse_status_success;
    }

    // test by passing predefined matrices
    template <typename T>
    void test_ilu_predefined_matrices()
    {
        aoclsparse_operation trans = aoclsparse_operation_none;
        // Create aocl sparse matrix
        aoclsparse_matrix           A;
        T                          *approx_inv_diag = NULL;
        T                          *precond_csr_val = NULL;
        aoclsparse_int              m, n, nnz;
        std::vector<aoclsparse_int> csr_row_ptr;
        std::vector<aoclsparse_int> csr_col_ind;
        std::vector<T>              csr_val;
        T                           init_x = 1.0;
        aoclsparse_mat_descr        descr;
        std::vector<T>              b, x;
        std::vector<T>              ilu_factors, ilu_factors_gold;
        tolerance_t<T>              tol;

        /*
            use the 8 matrices from data utils to test positive scenarios
        */
        enum matrix_id mids[] = {N5_full_sorted,
                                  /*sample_cg_mat,*/
                                  sample_gmres_mat_01,
                                  /*sample_gmres_mat_02,*/
                                  S5_sym_fullmatrix/*,
                                  S5_sym_lowerfill,
                                  S5_sym_upperfill,
                                  S4_herm_rsym*/};
        for(unsigned int idx = 0; idx < sizeof(mids) / sizeof(enum matrix_id); idx++)
        {
            A     = nullptr;
            descr = nullptr;

            ASSERT_EQ(aoclsparse_create_mat_descr(&descr), aoclsparse_status_success);
            ASSERT_EQ(aoclsparse_set_mat_index_base(descr, aoclsparse_index_base_zero),
                      aoclsparse_status_success);
            ASSERT_EQ(
                create_matrix(
                    mids[idx], m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, A, descr, VERBOSE),
                aoclsparse_status_success);
            try
            {
                b.resize(n);
            }
            catch(std::bad_alloc &)
            {
                std::cout << "Memory allocation error, b\n";
            }
            try
            {
                x.resize(n, init_x);
            }
            catch(std::bad_alloc &)
            {
                std::cout << "Memory allocation error, x\n";
            }
            try
            {
                ilu_factors.resize(nnz);
            }
            catch(std::bad_alloc &)
            {
                std::cout << "Memory allocation error, ilu_factors\n";
            }

            try
            {
                ilu_factors_gold.resize(nnz);
            }
            catch(std::bad_alloc &)
            {
                std::cout << "Memory allocation error, ilu_factors_gold\n";
            }

            load_ILU_reference_results(mids[idx], ilu_factors_gold);

            EXPECT_EQ(aoclsparse_ilu_smoother<T>(
                          trans, A, descr, &precond_csr_val, approx_inv_diag, &x[0], &b[0]),
                      aoclsparse_status_success);
            //so that comparison below is easier in vectors
            std::copy(precond_csr_val, precond_csr_val + nnz, ilu_factors.begin());

            tol = 10;
            if constexpr(std::is_same_v<T, std::complex<double>>
                         || std::is_same_v<T, std::complex<float>>)
            {
                EXPECT_COMPLEX_ARR_NEAR(
                    nnz, ilu_factors, ilu_factors_gold, expected_precision<decltype(tol)>(tol));
            }
            else
            {
                EXPECT_ARR_NEAR(
                    nnz, ilu_factors, ilu_factors_gold, expected_precision<decltype(tol)>(tol));
            }
            EXPECT_EQ(aoclsparse_destroy_mat_descr(descr), aoclsparse_status_success);
            EXPECT_EQ(aoclsparse_destroy(&A), aoclsparse_status_success);
        }
    }
    //TODO add:
    // * invalid array data (but we don't test these right now, e.g., col_ind out of bounds)
    // * nnz not matching row_ptr

    TEST(ilu, NullArgDouble)
    {
        test_ilu_nullptr<double>();
    }
    TEST(ilu, NullArgFloat)
    {
        test_ilu_nullptr<float>();
    }

    TEST(ilu, WrongSizeDouble)
    {
        test_ilu_wrong_size<double>();
    }
    TEST(ilu, WrongSizeFloat)
    {
        test_ilu_wrong_size<float>();
    }

    TEST(ilu, DoNothingDouble)
    {
        test_ilu_do_nothing<double>();
    }
    TEST(ilu, DoNothingFloat)
    {
        test_ilu_do_nothing<float>();
    }
    TEST(ilu, BaseOneDouble)
    {
        test_ilu_baseOneIndexing<double>();
    }
    TEST(ilu, BaseOneFloat)
    {
        test_ilu_baseOneIndexing<float>();
    }
    TEST(ilu, UnsupportedDouble)
    {
        test_ilu_unsupported<double>();
    }
    TEST(ilu, UnsupportedFloat)
    {
        test_ilu_unsupported<float>();
    }

    TEST(ilu, PredefinedMatricesDouble)
    {
        test_ilu_predefined_matrices<double>();
    }
    TEST(ilu, PredefinedMatricesFloat)
    {
        test_ilu_predefined_matrices<float>();
    }
    TEST(ilu, PredefinedMatricesCplxDouble)
    {
        test_ilu_predefined_matrices<std::complex<double>>();
    }
    TEST(ilu, PredefinedMatricesCplxFloat)
    {
        test_ilu_predefined_matrices<std::complex<float>>();
    }
} // namespace
