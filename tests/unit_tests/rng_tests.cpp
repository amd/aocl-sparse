/* ************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse.hpp"
#include "aoclsparse_init.hpp"

namespace
{
#define CSR 0
#define COO 1

#define FULL_SORT 1
#define PARTIAL_SORT 2
#define UNSORTED 3

#define SUCCESS 0
#define NOT_IMPLEMENTED 1
#define INVALID_SIZE 3
#define INVALID_VALUE 5

    //check if an array defined by the range (start_idx, end_idx) is sorted or not
    bool check_sorting_core(aoclsparse_int  start_idx,
                            aoclsparse_int  end_idx,
                            aoclsparse_int *col_arr)
    {
        bool psort = true; //assume fully sorted and in ascending order
        for(aoclsparse_int j = start_idx + 1; j < end_idx; j++)
        {
            if(col_arr[j] < col_arr[j - 1])
            {
                psort = false;
                break;
            }
        }
        return psort;
    }

    /* check if csr and csc buffers are sorted or not
        1. CSR: check if column indices are sorted or not
        2. CSC: check if row indices are sorted or not
    */
    bool check_sorting(aoclsparse_int       *csr_row_ptr,
                       aoclsparse_int       *csr_col_ind,
                       aoclsparse_int       &m,
                       aoclsparse_index_base base,
                       aoclsparse_int        isort)
    {
        bool psort = true; //assume ascending order
        for(aoclsparse_int i = 0; i < m; i++)
        {
            aoclsparse_int row_start, row_end, diag_index;
            row_start = csr_row_ptr[i] - base;
            row_end   = csr_row_ptr[i + 1] - base;

            //quick check for rows with nnz <= 1, because there is nothing to check wrt sorting
            if((row_end - row_start) <= 1)
            {
                continue;
            }
            //find diagonal index for row #i
            for(aoclsparse_int j = row_start; j < row_end; j++)
            {
                aoclsparse_int col_idx = csr_col_ind[j] - base;
                //is this diagonal?
                if(col_idx == i)
                {
                    diag_index = j;
                    break;
                }
            }
            if(isort == PARTIAL_SORT)
            {
                //check if lower and upper regions of the row are sorted or not
                //if atleast one triangle is unsorted, we consider partial sorting
                //diagonal is has been checked to be in the correct positio (i,i) for row #i
                //no need to check the second triangle/row, if the first one is already unsorted.
                psort = check_sorting_core(row_start, diag_index, csr_col_ind);
                if(psort)
                {
                    psort &= check_sorting_core(diag_index + 1, row_end, csr_col_ind);
                }
            }
            else
            {
                //check for fully sorted and unsorted cases
                psort = check_sorting_core(row_start, row_end, csr_col_ind);
            }
            //quick exit since a matrix is considered partially sorted or unsorted if
            //atleast a single row fails the sort check
            if(!psort)
            {
                //unsorted
                break;
            }
        }
        return psort;
    }

    // check if the given csr structure is with full diagonal
    template <typename T>
    bool check_csr_for_fulldiag(aoclsparse_int       *csr_row_ptr,
                                aoclsparse_int       *csr_col_ind,
                                T                    *csr_val,
                                aoclsparse_int       &m,
                                aoclsparse_index_base base)
    {
        bool fulldiag = true; //assume full diagonal

        for(aoclsparse_int i = 0; i < m; i++)
        {
            bool           diagonal;
            aoclsparse_int row_start, row_end;
            row_start = csr_row_ptr[i] - base;
            row_end   = csr_row_ptr[i + 1] - base;
            diagonal  = false; //assume the current row has no diagonal unless detected
            for(aoclsparse_int j = row_start; j < row_end; j++)
            {
                aoclsparse_int col_idx = csr_col_ind[j] - base;
                //check if there exists a non-zero diagonal element
                if((i == col_idx) && (csr_val[j] != aoclsparse_numeric::zero<T>()))
                {
                    //detected diagonal in row #i
                    diagonal = true;
                    break;
                }
            }
            //quick exit since a matrix is considered not of full-rank if
            //at least a single row fails the diagonal check
            if(!diagonal)
            {
                //no diagonal present in row #i
                fulldiag = false;
                break;
            }
        }
        return fulldiag;
    }

    /* RNG matrix test driver

    * Checks randomly generated matrix (in CSR, COO)
    * Generates a random matrix with the given requirements
    * then checks that these requirements are fulfilled: full/partial sorting or unsorting,
            full diagonal, diagonal dominance
    * Optionally, it can sort/check mtx files
        1. fully unsorted (s or c) matrix -> returns partial / full sorting
        2. partially sorted ,matrix -> returns partial / full sorting
        3. fully sorted matrix -> returns unchanged
    */
    template <typename T>
    void test_random_generation(aoclsparse_int ibase,
                                bool           is_random,
                                aoclsparse_int iformat,
                                aoclsparse_int m,
                                aoclsparse_int n,
                                aoclsparse_int nnz,
                                aoclsparse_int imatrix,
                                aoclsparse_int isort,
                                bool           fulldiag_exp_status,
                                bool           sortexp_status)
    {
        bool                        fulldiag, sorted;
        aoclsparse_status           status;
        bool                        is_symm = false;
        std::vector<T>              csr_val, csc_val, coo_val;
        std::vector<aoclsparse_int> csr_col_ind, csc_row_ind, csr_row_ptr, csc_col_ptr;
        std::vector<aoclsparse_int> coo_row_ind, coo_col_ind;
        aoclsparse_matrix           src_mat         = nullptr, dest_mat;
        aoclsparse_int             *coo2csr_row_ptr = nullptr;
        aoclsparse_int             *coo2csr_col_ind = nullptr;
        T                          *coo2csr_val     = nullptr;
        aoclsparse_int              coo2csr_n, coo2csr_m, coo2csr_nnz;
        aoclsparse_index_base       coo2csr_base;

        aoclsparse_index_base  base     = (aoclsparse_index_base)ibase;
        aoclsparse_matrix_init matrix   = (aoclsparse_matrix_init)imatrix;
        aoclsparse_matrix_sort sort     = (aoclsparse_matrix_sort)isort;
        char                  *filename = nullptr, mtx_input[64];

        if(!is_random)
        {
            if constexpr(std::is_same_v<T, aoclsparse_double_complex>
                         || std::is_same_v<T, aoclsparse_float_complex>)
            {
                strcpy(mtx_input, "data/tinyc.mtx");
            }
            else if constexpr(std::is_same_v<T, double> || std::is_same_v<T, float>)
            {
                strcpy(mtx_input, "data/tinyr.mtx");
            }

            filename = mtx_input;
        }
        aoclsparse_seedrand();
        if(iformat == CSR)
        {
            //allocate for csr buffers
            csr_row_ptr.resize(m + 1);
            csr_col_ind.resize(nnz);
            csr_val.resize(nnz);
            status = aoclsparse_init_csr_matrix(csr_row_ptr,
                                                csr_col_ind,
                                                csr_val,
                                                m,
                                                n,
                                                nnz,
                                                base,
                                                matrix,
                                                filename,
                                                is_symm,
                                                true,
                                                sort);
            EXPECT_EQ(status, aoclsparse_status_success)
                << "Test failed to validate aoclsparse_init_csr_matrix";

            //if generation is successful, validate diagonal and the order of indices
            if(status == aoclsparse_status_success)
            {
                //csr arrays validation
                fulldiag = check_csr_for_fulldiag(
                    &csr_row_ptr[0], &csr_col_ind[0], &csr_val[0], m, base);
                EXPECT_EQ(fulldiag, fulldiag_exp_status)
                    << "Test failed to validate full diagonal functionality in initialization "
                       "of csr matrix";
                sorted = check_sorting(&csr_row_ptr[0], &csr_col_ind[0], m, base, isort);
                EXPECT_EQ(sorted, sortexp_status) << "Test failed to validate sort functionality "
                                                     "in initialization of csr matrix";
            }
        }
        else if(iformat == COO)
        {
            //allocate for coo buffers
            coo_row_ind.resize(nnz);
            coo_col_ind.resize(nnz);
            coo_val.resize(nnz);
            status = aoclsparse_init_coo_matrix(coo_row_ind,
                                                coo_col_ind,
                                                coo_val,
                                                m,
                                                n,
                                                nnz,
                                                base,
                                                matrix,
                                                filename,
                                                is_symm,
                                                true,
                                                sort);
            EXPECT_EQ(status, aoclsparse_status_success)
                << "Test failed to validate aoclsparse_init_coo_matrix";

            //if generation is successful, convert to csr to validate sorting and full diagonal
            if(status == aoclsparse_status_success)
            {
                EXPECT_EQ(
                    aoclsparse_create_coo(
                        &src_mat, base, m, n, nnz, &coo_row_ind[0], &coo_col_ind[0], &coo_val[0]),
                    aoclsparse_status_success);

                EXPECT_EQ(aoclsparse_convert_csr(src_mat, aoclsparse_operation_none, &dest_mat),
                          aoclsparse_status_success);
                EXPECT_EQ(aoclsparse_export_csr(dest_mat,
                                                &coo2csr_base,
                                                &coo2csr_m,
                                                &coo2csr_n,
                                                &coo2csr_nnz,
                                                &coo2csr_row_ptr,
                                                &coo2csr_col_ind,
                                                &coo2csr_val),
                          aoclsparse_status_success);

                //coo arrays validation
                EXPECT_EQ(coo2csr_base, base);
                EXPECT_EQ(coo2csr_nnz, nnz);
                fulldiag = check_csr_for_fulldiag(
                    coo2csr_row_ptr, coo2csr_col_ind, coo2csr_val, coo2csr_m, coo2csr_base);
                EXPECT_EQ(fulldiag, fulldiag_exp_status)
                    << "Test failed to validate full diagonal functionality in initialization "
                       "of coo matrix";
                sorted = check_sorting(
                    coo2csr_row_ptr, coo2csr_col_ind, coo2csr_m, coo2csr_base, isort);
                EXPECT_EQ(sorted, sortexp_status) << "Test failed to validate sort functionality "
                                                     "in initialization of coo matrix";

                EXPECT_EQ(aoclsparse_destroy(&dest_mat), aoclsparse_status_success);
                EXPECT_EQ(aoclsparse_destroy(&src_mat), aoclsparse_status_success);
            }
        }
    }

#define TRUE true
#define FALSE false

#define BS0 0
#define BS1 1

#define RANDOM 0
#define MTX 1
#define DIAG_DOM 3

    typedef struct
    {
        std::string    testname;
        bool           is_random;
        aoclsparse_int base;
        aoclsparse_int format;
        aoclsparse_int m;
        aoclsparse_int n;
        aoclsparse_int nnz;
        aoclsparse_int
            matrix; //0 - random-default, 1-mtx file as input, 3-random-diagonally dominant
        aoclsparse_int sort; //0 - unsorted, 1 - partially sorted, 2 - full sorted
        bool           fulldiag_exp_status;
        bool           sort_exp_status;
    } rng_params;

#undef ADD_TEST
#define ADD_TEST(BASE, IS_RANDOM, FORMAT, M, N, NNZ, MAT, SORT, FD_STATUS, SRT_STATUS)        \
    {                                                                                         \
        "Random-" #IS_RANDOM "/" #BASE "/" #FORMAT "/" #M "x" #N "x" #NNZ "/" #MAT "/" #SORT, \
            IS_RANDOM, BASE, FORMAT, M, N, NNZ, MAT, SORT, FD_STATUS, SRT_STATUS              \
    }

    rng_params rng_generator_list[] = {
        ADD_TEST(BS0, TRUE, CSR, 10, 10, 35, DIAG_DOM, UNSORTED, TRUE, FALSE),
        ADD_TEST(BS0, TRUE, CSR, 10, 10, 35, DIAG_DOM, FULL_SORT, TRUE, TRUE),
        ADD_TEST(BS1, TRUE, CSR, 30, 30, 378, DIAG_DOM, PARTIAL_SORT, TRUE, FALSE),
        ADD_TEST(BS0, TRUE, CSR, 10, 10, 35, RANDOM, UNSORTED, FALSE, FALSE),
        ADD_TEST(BS1, TRUE, COO, 27, 27, 279, DIAG_DOM, FULL_SORT, TRUE, TRUE),
        ADD_TEST(BS1, TRUE, COO, 27, 27, 279, DIAG_DOM, UNSORTED, TRUE, FALSE),
        ADD_TEST(BS0, TRUE, COO, 10, 10, 35, RANDOM, UNSORTED, FALSE, FALSE),
        //MTX INPUTS: matrix is of full diagonal(LFAT5.mtx)
        ADD_TEST(BS0, FALSE, CSR, 14, 14, 30, MTX, UNSORTED, TRUE, FALSE),
        ADD_TEST(BS0, FALSE, COO, 14, 14, 30, MTX, UNSORTED, TRUE, FALSE),
        ADD_TEST(BS1, FALSE, CSR, 14, 14, 30, MTX, PARTIAL_SORT, TRUE, FALSE),
        ADD_TEST(BS0, FALSE, CSR, 14, 14, 30, MTX, FULL_SORT, TRUE, TRUE),
        ADD_TEST(BS1, FALSE, COO, 14, 14, 30, MTX, FULL_SORT, TRUE, TRUE),
        //edge case csr: computed nnz greater than matrix dimension
        ADD_TEST(BS0, TRUE, CSR, 1001, 1001, 0, DIAG_DOM, FULL_SORT, TRUE, TRUE),
        //edge case coo: computed nnz greater than matrix dimension
        ADD_TEST(BS1, TRUE, COO, 1001, 1001, 0, DIAG_DOM, FULL_SORT, TRUE, TRUE),
    };

    // It is used to when testing::PrintToString(GetParam()) to generate test name for ctest
    void PrintTo(const rng_params &param, std::ostream *os)
    {
        *os << param.testname;
    }

    class PosDouble : public testing::TestWithParam<rng_params>
    {
    };
    // tests with double type
    TEST_P(PosDouble, Rangen)
    {
        const rng_params &param = GetParam();
        test_random_generation<double>(param.base,
                                       param.is_random,
                                       param.format,
                                       param.m,
                                       param.n,
                                       param.nnz,
                                       param.matrix,
                                       param.sort,
                                       param.fulldiag_exp_status,
                                       param.sort_exp_status);
    }
    INSTANTIATE_TEST_SUITE_P(RanGenSuite, PosDouble, ::testing::ValuesIn(rng_generator_list));

    class PosCplxDouble : public testing::TestWithParam<rng_params>
    {
    };
    TEST_P(PosCplxDouble, Rangen)
    {
        const rng_params &param = GetParam();
        test_random_generation<aoclsparse_double_complex>(param.base,
                                                          param.is_random,
                                                          param.format,
                                                          param.m,
                                                          param.n,
                                                          param.nnz,
                                                          param.matrix,
                                                          param.sort,
                                                          param.fulldiag_exp_status,
                                                          param.sort_exp_status);
    }
    INSTANTIATE_TEST_SUITE_P(RanGenSuite, PosCplxDouble, ::testing::ValuesIn(rng_generator_list));

    void test_random_generation_invalid(void)
    {
        using T = float;
        aoclsparse_status           status, exp_status;
        bool                        is_symm = false;
        std::vector<T>              val;
        std::vector<aoclsparse_int> row_array, col_array;
        aoclsparse_index_base       base   = aoclsparse_index_base_zero;
        aoclsparse_matrix_init      matrix = aoclsparse_matrix_random_diag_dom;
        aoclsparse_matrix_sort      sort   = aoclsparse_unsorted;
        aoclsparse_int              m      = 10;
        aoclsparse_int              n      = 10;
        aoclsparse_int              nnz = 35, wrong, zero = 0;

        //allocate for csr buffers
        row_array.resize(m + 1);
        col_array.resize(nnz);
        val.resize(nnz);

        //m=0,n=0 quick exit check
        exp_status = aoclsparse_status_invalid_size;
        status     = aoclsparse_init_csr_matrix(
            row_array, col_array, val, zero, n, nnz, base, matrix, nullptr, is_symm, true, sort);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate zero dimensions in aoclsparse_init_csr_matrix";
        status = aoclsparse_init_coo_matrix(
            row_array, col_array, val, m, zero, nnz, base, matrix, nullptr, is_symm, true, sort);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate zero dimensions in aoclsparse_init_coo_matrix";

        //negative dimensions quick exit check
        exp_status = aoclsparse_status_invalid_size;
        wrong      = -7;
        status     = aoclsparse_init_csr_matrix(
            row_array, col_array, val, wrong, n, nnz, base, matrix, nullptr, is_symm, true, sort);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate negative dimensions in aoclsparse_init_csr_matrix";
        status = aoclsparse_init_coo_matrix(
            row_array, col_array, val, m, wrong, nnz, base, matrix, nullptr, is_symm, true, sort);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate negative dimensions in aoclsparse_init_coo_matrix";

        //nnz > m*n check
        wrong      = m * n + 1;
        exp_status = aoclsparse_status_invalid_size;
        status     = aoclsparse_init_csr_matrix(
            row_array, col_array, val, m, n, wrong, base, matrix, nullptr, is_symm, true, sort);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate nnz if greater than m*n in aoclsparse_init_csr_matrix";
        status = aoclsparse_init_coo_matrix(
            row_array, col_array, val, m, n, wrong, base, matrix, nullptr, is_symm, true, sort);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate nnz if greater than m*n in aoclsparse_init_coo_matrix";

        //invalid matrix differentiator
        exp_status = aoclsparse_status_invalid_value;
        status     = aoclsparse_init_csr_matrix(row_array,
                                            col_array,
                                            val,
                                            m,
                                            n,
                                            nnz,
                                            base,
                                            (aoclsparse_matrix_init)4,
                                            nullptr,
                                            is_symm,
                                            true,
                                            sort);
        EXPECT_EQ(status, exp_status) << "Test failed to validate invalid matrix differentiator in "
                                         "aoclsparse_init_csr_matrix";
        status = aoclsparse_init_coo_matrix(row_array,
                                            col_array,
                                            val,
                                            m,
                                            n,
                                            nnz,
                                            base,
                                            (aoclsparse_matrix_init)4,
                                            nullptr,
                                            is_symm,
                                            true,
                                            sort);
        EXPECT_EQ(status, exp_status) << "Test failed to validate invalid matrix differentiator in "
                                         "aoclsparse_init_coo_matrix";

        //invalid sort option
        exp_status = aoclsparse_status_invalid_value;
        status     = aoclsparse_init_csr_matrix(row_array,
                                            col_array,
                                            val,
                                            m,
                                            n,
                                            nnz,
                                            base,
                                            matrix,
                                            nullptr,
                                            is_symm,
                                            true,
                                            (aoclsparse_matrix_sort)4);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate invalid sort option in aoclsparse_init_csr_matrix";
        status = aoclsparse_init_coo_matrix(row_array,
                                            col_array,
                                            val,
                                            m,
                                            n,
                                            nnz,
                                            base,
                                            matrix,
                                            nullptr,
                                            is_symm,
                                            true,
                                            (aoclsparse_matrix_sort)4);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate invalid sort option in aoclsparse_init_coo_matrix";

        //zero nnz, expect nnz computed to be 2% of m*n and a random non-full-diagonal output
        exp_status = aoclsparse_status_success;
        status     = aoclsparse_init_csr_matrix(
            row_array, col_array, val, m, n, zero, base, matrix, nullptr, is_symm, true, sort);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate zero nnz in aoclsparse_init_csr_matrix";
        status = aoclsparse_init_coo_matrix(
            row_array, col_array, val, m, n, zero, base, matrix, nullptr, is_symm, true, sort);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate zero nnz in aoclsparse_init_coo_matrix";

        //partial sorting in coo
        exp_status = aoclsparse_status_not_implemented;
        status     = aoclsparse_init_coo_matrix(row_array,
                                            col_array,
                                            val,
                                            m,
                                            n,
                                            nnz,
                                            base,
                                            matrix,
                                            nullptr,
                                            is_symm,
                                            true,
                                            aoclsparse_partially_sorted);
        EXPECT_EQ(status, exp_status)
            << "Test failed to validate partial sorting in aoclsparse_init_coo_matrix";
    }
    TEST(RanGenNegative, InvalidChecks)
    {
        test_random_generation_invalid();
    }
} // namespace
