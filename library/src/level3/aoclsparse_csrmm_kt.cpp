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
 * ************************************************************************
 */
#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l3_kt.hpp"

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status aoclsparse::csrmm_col_kt(const SUF                  alpha,
                                           const aoclsparse_mat_descr descr,
                                           const SUF *__restrict__ csr_val,
                                           const aoclsparse_int *__restrict__ csr_col_ind,
                                           const aoclsparse_int *__restrict__ csr_row_ptr,
                                           aoclsparse_int m,
                                           const SUF     *B,
                                           aoclsparse_int n,
                                           aoclsparse_int ldb,
                                           SUF            beta,
                                           SUF           *C,
                                           aoclsparse_int ldc)
{
    using namespace kernel_templates;

    aoclsparse_int       idxend, nnz, rem, mul;
    SUF                  cij, cijp1, cijp2, cijp3, cdot0, cdot1, cdot2, cdot3;
    const SUF           *bcol0, *bcol1, *bcol2, *bcol3;
    avxvector_t<SZ, SUF> avec, bvec0, bvec1, bvec2, bvec3, cvec0, cvec1, cvec2, cvec3;
    const aoclsparse_int psz = tsz_v<SZ, SUF>;
    /* Block size for matrix B
     * blocks are of size 8x4
     */
    const aoclsparse_int bblk = 4;
    aoclsparse_int       blkr = bblk;

    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const SUF            *csr_val_fix     = csr_val - base;
    const SUF            *B_fix           = B - base;

    for(aoclsparse_int j = 0; j < n; j += bblk)
    {
        // Process last column of B
        if(j == n - 1)
        {
            blkr  = 1;
            bcol0 = B_fix + j * ldb; // last column of B
            bcol1 = bcol0;
            bcol2 = bcol0;
            bcol3 = bcol0;
        }
        else if(j == n - 2)
        {
            blkr  = 2;
            bcol0 = B_fix + j * ldb; // second to last column of B
            bcol1 = bcol0 + ldb; // last column of B
            bcol2 = bcol0;
            bcol3 = bcol0;
        }
        else if(j == n - 3)
        {
            blkr  = 3;
            bcol0 = B_fix + j * ldb; // third to last column of B
            bcol1 = bcol0 + ldb; // second to last column of B
            bcol2 = bcol1 + ldb; // last column of B
            bcol3 = bcol0;
        }
        else
        {
            // blkr = 4
            bcol0 = B_fix + j * ldb; // j-th column of B
            bcol1 = bcol0 + ldb; // (j+1)-th column of B
            bcol2 = bcol1 + ldb; // (j+2)-th column of B
            bcol3 = bcol2 + ldb; // (j+3)-th column of B
        }

        for(aoclsparse_int i = 0; i < m; i++)
        {
            cij    = 0.0f;
            cijp1  = 0.0f;
            cijp2  = 0.0f;
            cijp3  = 0.0f;
            idxend = csr_row_ptr[i + 1];
            nnz    = idxend - csr_row_ptr[i];
            mul    = nnz / psz;
            rem    = nnz - psz * mul;
            if(mul)
            {
                cvec0 = kt_setzero_p<SZ, SUF>();
                cvec1 = kt_setzero_p<SZ, SUF>();
                cvec2 = kt_setzero_p<SZ, SUF>();
                cvec3 = kt_setzero_p<SZ, SUF>();
                for(aoclsparse_int idx = csr_row_ptr[i]; idx < idxend - rem; idx += psz)
                {
                    // Sequential 4x8 matrix-vector multiplication
                    avec  = kt_loadu_p<SZ, SUF>(&csr_val_fix[idx]);
                    bvec0 = kt_set_p<SZ, SUF>(bcol0, &csr_col_ind_fix[idx]);
                    bvec1 = kt_set_p<SZ, SUF>(bcol1, &csr_col_ind_fix[idx]);
                    bvec2 = kt_set_p<SZ, SUF>(bcol2, &csr_col_ind_fix[idx]);
                    bvec3 = kt_set_p<SZ, SUF>(bcol3, &csr_col_ind_fix[idx]);
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec, bvec0, cvec0);
                    cvec1 = kt_fmadd_p<SZ, SUF>(avec, bvec1, cvec1);
                    cvec2 = kt_fmadd_p<SZ, SUF>(avec, bvec2, cvec2);
                    cvec3 = kt_fmadd_p<SZ, SUF>(avec, bvec3, cvec3);
                }
                cdot0 = kt_hsum_p<SZ, SUF>(cvec0);
                cdot1 = kt_hsum_p<SZ, SUF>(cvec1);
                cdot2 = kt_hsum_p<SZ, SUF>(cvec2);
                cdot3 = kt_hsum_p<SZ, SUF>(cvec3);
                cij += cdot0;
                cijp1 += cdot1;
                cijp2 += cdot2;
                cijp3 += cdot3;
            }
            if(rem)
            {
                for(aoclsparse_int idx = idxend - rem; idx < idxend; idx++)
                {
                    cij += csr_val_fix[idx] * bcol0[csr_col_ind_fix[idx]];
                    cijp1 += csr_val_fix[idx] * bcol1[csr_col_ind_fix[idx]];
                    cijp2 += csr_val_fix[idx] * bcol2[csr_col_ind_fix[idx]];
                    cijp3 += csr_val_fix[idx] * bcol3[csr_col_ind_fix[idx]];
                }
            }
            switch(blkr)
            {
            case 4:
                cijp3 *= alpha;
                C[(j + 3) * ldc + i] = beta * C[(j + 3) * ldc + i] + cijp3;
                [[fallthrough]];
            case 3:
                cijp2 *= alpha;
                cijp2 += beta * C[(j + 2) * ldc + i];
                C[(j + 2) * ldc + i] = cijp2;
                [[fallthrough]];
            case 2:
                cijp1 *= alpha;
                cijp1 += beta * C[(j + 1) * ldc + i];
                C[(j + 1) * ldc + i] = cijp1;
                [[fallthrough]];
            case 1:
                cij *= alpha;
                cij += beta * C[j * ldc + i];
                C[j * ldc + i] = cij;
            }
        }
    }
    return aoclsparse_status_success;
}

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status aoclsparse::csrmm_row_kt(const SUF                  alpha,
                                           const aoclsparse_mat_descr descr,
                                           const SUF *__restrict__ csr_val,
                                           const aoclsparse_int *__restrict__ csr_col_ind,
                                           const aoclsparse_int *__restrict__ csr_row_ptr,
                                           aoclsparse_int m,
                                           const SUF     *B,
                                           aoclsparse_int n,
                                           aoclsparse_int ldb,
                                           SUF            beta,
                                           SUF           *C,
                                           aoclsparse_int ldc)
{
    using namespace kernel_templates;
    const aoclsparse_int psz = tsz_v<SZ, SUF>;

    const aoclsparse_int bblk = 4;
    aoclsparse_int       blkr = bblk;

    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const SUF            *csr_val_fix     = csr_val - base;
    const SUF            *B_fix           = B - (base * ldb);

    aoclsparse_int       rem, mul;
    avxvector_t<SZ, SUF> avec, avec1, avec2, avec3, bvec0, bvec1, bvec2, bvec3, cvec0;
    aoclsparse_int       idx0, idx1, idx2, idx3;
    aoclsparse_int       start = 0;
    aoclsparse_int       end   = m;
    for(aoclsparse_int i = start; i < end; i++)
    {
        aoclsparse_int idx_C = i * ldc;
        for(aoclsparse_int j = 0; j < n; j++)
        {
            C[idx_C + j] = C[idx_C + j] * beta;
        }
        for(aoclsparse_int idx = csr_row_ptr[i]; idx < csr_row_ptr[i + 1]; idx += bblk)
        {
            if(idx == csr_row_ptr[i + 1] - 1)
            {
                blkr = 1;
                idx0 = idx; // Last non-zero of the row
                idx1 = idx0;
                idx2 = idx0;
                idx3 = idx0;
            }
            else if(idx == csr_row_ptr[i + 1] - 2)
            {
                blkr = 2;
                idx0 = idx; // Last non-zero of the row
                idx1 = idx0 + 1; // Second last non-zero of the row
                idx2 = idx0;
                idx3 = idx0;
            }
            else if(idx == csr_row_ptr[i + 1] - 3)
            {
                blkr = 3;
                idx0 = idx; //Last non-zero of the row
                idx1 = idx0 + 1; // Second last non-zero of the row
                idx2 = idx1 + 1; // Third last non-zero of the row
                idx3 = idx0;
            }
            else
            {
                // Four non-zeros of the row
                blkr = 4;
                idx0 = idx;
                idx1 = idx0 + 1;
                idx2 = idx1 + 1;
                idx3 = idx2 + 1;
            }
            aoclsparse_int idx_B  = csr_col_ind_fix[idx0] * ldb;
            aoclsparse_int idx_B1 = csr_col_ind_fix[idx1] * ldb;
            aoclsparse_int idx_B2 = csr_col_ind_fix[idx2] * ldb;
            aoclsparse_int idx_B3 = csr_col_ind_fix[idx3] * ldb;

            // Broadcast four non-zeros into the four vectors
            avec  = kt_set1_p<SZ, SUF>(alpha * csr_val_fix[idx0]);
            avec1 = kt_set1_p<SZ, SUF>(alpha * csr_val_fix[idx1]);
            avec2 = kt_set1_p<SZ, SUF>(alpha * csr_val_fix[idx2]);
            avec3 = kt_set1_p<SZ, SUF>(alpha * csr_val_fix[idx3]);
            mul   = n / psz;
            rem   = n - psz * mul;
            for(aoclsparse_int j = 0; j < n - rem; j += psz)
            {
                cvec0 = kt_loadu_p<SZ, SUF>(&C[idx_C + j]);

                // Load psz elements from rows of B corresponding to the non-zeros in A
                bvec0 = kt_loadu_p<SZ, SUF>(&B_fix[j + idx_B]);
                bvec1 = kt_loadu_p<SZ, SUF>(&B_fix[j + idx_B1]);
                bvec2 = kt_loadu_p<SZ, SUF>(&B_fix[j + idx_B2]);
                bvec3 = kt_loadu_p<SZ, SUF>(&B_fix[j + idx_B3]);
                switch(blkr)
                {
                case 4:
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec, bvec0, cvec0);
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec1, bvec1, cvec0);
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec2, bvec2, cvec0);
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec3, bvec3, cvec0);
                    break;
                case 3:
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec, bvec0, cvec0);
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec1, bvec1, cvec0);
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec2, bvec2, cvec0);
                    break;
                case 2:
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec, bvec0, cvec0);
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec1, bvec1, cvec0);
                    break;
                case 1:
                    cvec0 = kt_fmadd_p<SZ, SUF>(avec, bvec0, cvec0);
                    break;
                }
                kt_storeu_p<SZ, SUF>(&C[idx_C + j], cvec0);
            }

            if(rem)
            {
                for(aoclsparse_int j = n - rem; j < n; j++)
                {
                    switch(blkr)
                    {
                    case 4:
                        C[idx_C + j] += csr_val_fix[idx0] * B_fix[idx_B + j] * alpha;
                        C[idx_C + j] += csr_val_fix[idx1] * B_fix[idx_B1 + j] * alpha;
                        C[idx_C + j] += csr_val_fix[idx2] * B_fix[idx_B2 + j] * alpha;
                        C[idx_C + j] += csr_val_fix[idx3] * B_fix[idx_B3 + j] * alpha;
                        break;
                    case 3:
                        C[idx_C + j] += csr_val_fix[idx0] * B_fix[idx_B + j] * alpha;
                        C[idx_C + j] += csr_val_fix[idx1] * B_fix[idx_B1 + j] * alpha;
                        C[idx_C + j] += csr_val_fix[idx2] * B_fix[idx_B2 + j] * alpha;
                        break;
                    case 2:
                        C[idx_C + j] += csr_val_fix[idx0] * B_fix[idx_B + j] * alpha;
                        C[idx_C + j] += csr_val_fix[idx1] * B_fix[idx_B1 + j] * alpha;
                        break;
                    case 1:
                        C[idx_C + j] += csr_val_fix[idx0] * B_fix[idx_B + j] * alpha;
                        break;
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}

#define CSRMM_COL_TEMPLATE_DECLARATION(BSZ, SUF)                   \
    template aoclsparse_status aoclsparse::csrmm_col_kt<BSZ, SUF>( \
        const SUF                  alpha,                          \
        const aoclsparse_mat_descr descr,                          \
        const SUF *__restrict__ csr_val,                           \
        const aoclsparse_int *__restrict__ csr_col_ind,            \
        const aoclsparse_int *__restrict__ csr_row_ptr,            \
        aoclsparse_int m,                                          \
        const SUF     *B,                                          \
        aoclsparse_int n,                                          \
        aoclsparse_int ldb,                                        \
        SUF            beta,                                       \
        SUF           *C,                                          \
        aoclsparse_int ldc);

#define CSRMM_ROW_TEMPLATE_DECLARATION(BSZ, SUF)                   \
    template aoclsparse_status aoclsparse::csrmm_row_kt<BSZ, SUF>( \
        const SUF                  alpha,                          \
        const aoclsparse_mat_descr descr,                          \
        const SUF *__restrict__ csr_val,                           \
        const aoclsparse_int *__restrict__ csr_col_ind,            \
        const aoclsparse_int *__restrict__ csr_row_ptr,            \
        aoclsparse_int m,                                          \
        const SUF     *B,                                          \
        aoclsparse_int n,                                          \
        aoclsparse_int ldb,                                        \
        SUF            beta,                                       \
        SUF           *C,                                          \
        aoclsparse_int ldc);

KT_INSTANTIATE(CSRMM_COL_TEMPLATE_DECLARATION, kernel_templates::get_bsz());
KT_INSTANTIATE(CSRMM_ROW_TEMPLATE_DECLARATION, kernel_templates::get_bsz());
