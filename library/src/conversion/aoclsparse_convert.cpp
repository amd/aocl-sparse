/* ************************************************************************
 * Copyright (c) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
aoclsparse_int aoclsparse_opt_blksize(aoclsparse_int        m,
                                      aoclsparse_int        nnz,
                                      aoclsparse_index_base base,
                                      const aoclsparse_int *csr_row_ptr,
                                      const aoclsparse_int *csr_col_ind,
                                      aoclsparse_int       *total_blks)
{
    // Check sizes
    if((m <= 0) || (nnz <= 0))
    {
        return 0;
    }

    // Check pointer arguments
    if((csr_row_ptr == nullptr) || (csr_col_ind == nullptr) || (total_blks == nullptr))
    {
        return 0;
    }

    //Initialize block width
    const aoclsparse_int blk_width = 8;
    aoclsparse_int       optimal_blk;
    aoclsparse_int       nBlk_factor[3] = {1, 2, 4};
    aoclsparse_int       total_nBlk[3];
    double               perBlk[3];
    double               blkUtil[3];

    //Variables to track number of blocks increasing as we change block size
    double pc_blks_inc[2];
    double pc_diff_blks    = 0;
    double pc_diff_blkutil = 0;
    double nnzpr           = nnz / m;

    for(int i = 0; i < 3; i++)
    {
        aoclsparse_int total_num_blks = 0;

        for(int iRow = 0; iRow < m; iRow += nBlk_factor[i])
        {
            int num_cur_blks = 0;
            int iVal[nBlk_factor[i]];
            memset(iVal, 0, nBlk_factor[i] * sizeof(int));

            //Store the indexes from the value array for each row of the block
            for(int iSubRow = 0; (iSubRow < nBlk_factor[i]) && (iRow + iSubRow < m); iSubRow++)
                iVal[iSubRow] = csr_row_ptr[iRow + iSubRow] - base;

            //Continue this loop until all blocks that can be made on the matrix
            while(true)
            {
                bool blockComplete = true;
                int  iCol          = INT_MAX;

                //For each block, traverse each subrow to find the column index of the first nnz
                for(int iSubRow = 0; (iSubRow < nBlk_factor[i]) && (iRow + iSubRow < m); iSubRow++)
                {
                    if(iVal[iSubRow] < (csr_row_ptr[iRow + iSubRow + 1] - base))
                    {
                        blockComplete = false;
                        iCol          = (std::min<int>)(iCol, (csr_col_ind[iVal[iSubRow]] - base));
                    }
                }
                if(blockComplete)
                    break;

                for(int iSubRow = 0; (iSubRow < nBlk_factor[i]) && (iRow + iSubRow < m); iSubRow++)
                    while(iVal[iSubRow] < (csr_row_ptr[iRow + iSubRow + 1] - base)
                          && (csr_col_ind[iVal[iSubRow]] - base) < iCol + blk_width)
                        iVal[iSubRow] += 1;
                //Count the number of blocks in the current row block
                num_cur_blks += 1;
            }
            //Update the total number of blocks
            total_num_blks += num_cur_blks;
        }

        total_nBlk[i] = total_num_blks;
        perBlk[i]     = double(nnz) / double(total_num_blks);
        blkUtil[i]    = (perBlk[i] / ((double)nBlk_factor[i] * 8)) * 100;

        if((nnzpr < 30 && blkUtil[0] < 40) || (nnzpr > 30 && blkUtil[0] < 50))
            return 0;

        if(i != 0)
        {
            pc_blks_inc[i - 1] = (double(perBlk[i] - perBlk[i - 1]) / double(perBlk[i - 1])) * 100;
        }
    }
    pc_diff_blks    = abs(pc_blks_inc[0] - pc_blks_inc[1]);
    pc_diff_blkutil = abs(blkUtil[1] - blkUtil[2]);

    //The percentages and cutoffs are derived based on empirical evaluation. Potential issue: overfitting
    if((blkUtil[2] > 24) && (pc_diff_blks < 12.5 || pc_diff_blkutil < 12.5)
       && (pc_blks_inc[1] > 51))
    {
        *total_blks = total_nBlk[2];
        optimal_blk = 4;
        return optimal_blk;
    }
    else if(blkUtil[1] > 28)
    {
        *total_blks = total_nBlk[1];
        optimal_blk = 2;
        return optimal_blk;
    }

    return 0;
}

aoclsparse_status aoclsparse_csr2blkcsr(aoclsparse_int        m,
                                        aoclsparse_int        n,
                                        aoclsparse_int        nnz,
                                        const aoclsparse_int *csr_row_ptr,
                                        const aoclsparse_int *csr_col_ind,
                                        const double         *csr_val,
                                        aoclsparse_int       *blk_row_ptr,
                                        aoclsparse_int       *blk_col_ind,
                                        double               *blk_csr_val,
                                        uint8_t              *masks,
                                        aoclsparse_int        nRowsblk,
                                        aoclsparse_index_base base)
{
    // Check sizes
    if((m < 0) || (n < 8) || (nnz < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Check pointer arguments
    if((csr_row_ptr == nullptr) || (csr_col_ind == nullptr) || (csr_val == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }

    if((blk_row_ptr == nullptr) || (blk_col_ind == nullptr) || (blk_csr_val == nullptr)
       || (masks == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Validate nRowsblk. Only 1x8, 2x8 and 4x8 blocks are
    //supported as part of blkcsrmv implementation
    if((nRowsblk != 1) && (nRowsblk != 2) && (nRowsblk != 4))
    {
        return aoclsparse_status_invalid_size;
    }

    //Initialize block width
    const aoclsparse_int        blk_width = 8;
    std::vector<aoclsparse_int> blk_row_ptr_local;
    std::vector<aoclsparse_int> blk_col_ind_local;
    std::vector<double>         blk_csr_val_local;
    std::vector<uint8_t>        masks_local;
    aoclsparse_int              total_num_blks = 0;

    try
    {
        blk_row_ptr_local.resize(m + 1, 0);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    blk_row_ptr_local.assign(csr_row_ptr, csr_row_ptr + m + 1);

    /*
      TODO: can the below logic be optimized?
      Some comments from Jan as part of "Support to one-base indexing" commit:
      We could build the arrays easily with the base we want.
      Second question is why to actually preserve 1-base in this case given that it is purely
      internal format. Yet another question is why even the original routine uses *_local arrays
      given that proper size is known and the destination arrays have been allocated.
      We should be also more careful in this routine with catching memory allocations and
      given the fact that nRowsblk is not constant, many local arrays will trigger memory
      allocation (which is probably unnecessary).
    */
    for(int iRow = 0; iRow < m; iRow += nRowsblk)
    {
        int num_cur_blks = 0;
        int iVal[nRowsblk];
        memset(iVal, 0, nRowsblk * sizeof(int));

        //Store the indexes from the value array for each row of the block
        for(int iSubRow = 0; (iSubRow < nRowsblk) && (iRow + iSubRow < m); iSubRow++)
            iVal[iSubRow] = csr_row_ptr[iRow + iSubRow] - base;
        //Continue this loop until all blocks that can be made on the matrix
        while(true)
        {
            bool blockComplete = true;
            int  iCol          = INT_MAX;

            //For each block, traverse each subrow to find the column index of the first nnz
            for(int iSubRow = 0; (iSubRow < nRowsblk) && (iRow + iSubRow < m); iSubRow++)
            {
                if(iVal[iSubRow] < (blk_row_ptr_local[iRow + iSubRow + 1] - base))
                {
                    blockComplete = false;
                    iCol          = (std::min<int>)(iCol, (csr_col_ind[iVal[iSubRow]] - base));
                }
            }
            if(blockComplete)
                break;

            uint8_t mask[nRowsblk];
            memset(mask, 0u, nRowsblk * sizeof(uint8_t));
            uint8_t newmask[nRowsblk];
            memset(newmask, 0u, nRowsblk * sizeof(uint8_t));

            for(int iSubRow = 0; (iSubRow < nRowsblk) && (iRow + iSubRow < m); iSubRow++)
            {
                while(iVal[iSubRow] < (blk_row_ptr_local[iRow + iSubRow + 1] - base)
                      && (csr_col_ind[iVal[iSubRow]] - base) < iCol + blk_width)
                {
                    //Reorder and store subrows(in block) sequentially to for requred blocks from the original CSR val array
                    blk_csr_val_local.insert(blk_csr_val_local.end(), csr_val[iVal[iSubRow]]);

                    //Calculate the mask for each subrow in the block
                    mask[iSubRow] |= (1u << ((csr_col_ind[iVal[iSubRow]] - base) - iCol));

                    //Calculate mask for with modified column index (resized block to prevent out of bound reads)
                    //Applicable only for the last block of the rows
                    if(iCol + blk_width > n)
                        newmask[iSubRow] = mask[iSubRow] << (blk_width - (n - iCol));
                    iVal[iSubRow] += 1;
                }
            }

            //Store masks and column indexes for each block in a vector
            if(iCol + blk_width > n)
            {
                blk_col_ind_local.insert(blk_col_ind_local.end(), (n - blk_width));
                masks_local.insert(masks_local.end(), &newmask[0], &newmask[nRowsblk]);
            }
            else
            {
                blk_col_ind_local.insert(blk_col_ind_local.end(), iCol);
                masks_local.insert(masks_local.end(), &mask[0], &mask[nRowsblk]);
            }

            //Count the number of blocks in the current row block
            num_cur_blks += 1;
        }

        //Update the row pointers for all the blocked subrows
        blk_row_ptr_local[iRow] = total_num_blks;
        total_num_blks += num_cur_blks;

        //This loop is needed to update row pointers when the number of rows in a block > 1
        for(int iSubRow = 1; (iSubRow < nRowsblk) && (iRow + iSubRow < m); iSubRow++)
            blk_row_ptr_local[iRow + iSubRow] = total_num_blks;
    }

    //Storing total number of blocks as last element in a row pointer vector
    blk_row_ptr_local[m] = total_num_blks;

    //Return after copying new values to output vector, while reverting the original base-index
    //since 1-based buffers were converted to 0-based. Maintain original base-index
    std::transform(blk_row_ptr_local.begin(),
                   blk_row_ptr_local.end(),
                   blk_row_ptr,
                   [base](aoclsparse_int &d) { return d + base; });
    std::transform(blk_col_ind_local.begin(),
                   blk_col_ind_local.end(),
                   blk_col_ind,
                   [base](aoclsparse_int &d) { return d + base; });
    copy(blk_csr_val_local.begin(), blk_csr_val_local.end(), blk_csr_val);
    copy(masks_local.begin(), masks_local.end(), masks);

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csr2ell_width(aoclsparse_int                  m,
                                           [[maybe_unused]] aoclsparse_int nnz,
                                           const aoclsparse_int           *csr_row_ptr,
                                           aoclsparse_int                 *ell_width)
{
    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Check ell_width pointer
    if(ell_width == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Determine ELL width
    *ell_width = 0;

    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        *ell_width             = (std::max)(row_nnz, *ell_width);
    }

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csr2ellthyb_width(aoclsparse_int        m,
                                               aoclsparse_int        nnz,
                                               const aoclsparse_int *csr_row_ptr,
                                               aoclsparse_int       *ell_m,
                                               aoclsparse_int       *ell_width)
{
    // Check sizes
    if(m < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    if(m == 0)
    {
        *ell_width = 0;
        *ell_m     = 0;
        return aoclsparse_status_success;
    }

    // Check ell_width pointer
    if(ell_width == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Determine ELL width
    *ell_width = 0;

    aoclsparse_int mx_nnz_lt_nnza = 0, mn_nnz_gt_nnza = nnz, cmn = 0, cmx = 0, nnza = nnz / m;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int nnzi = csr_row_ptr[i + 1] - csr_row_ptr[i];
        if((nnzi > mx_nnz_lt_nnza) && (nnzi <= nnza))
        {
            mx_nnz_lt_nnza = nnzi;
        }
        if((nnzi < mn_nnz_gt_nnza) && (nnzi > nnza))
        {
            mn_nnz_gt_nnza = nnzi;
        }

        if((nnzi <= nnza))
            cmx++;
        else
            cmn++;
    }
    if(cmx >= cmn)
        *ell_width = mx_nnz_lt_nnza;
    else
        *ell_width = mn_nnz_gt_nnza;

    *ell_m = 0;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        if(row_nnz <= *ell_width)
            (*ell_m)++;
    }

    return aoclsparse_status_success;
}

extern "C" aoclsparse_status aoclsparse_scsr2ell(aoclsparse_int             m,
                                                 const aoclsparse_mat_descr descr,
                                                 const aoclsparse_int      *csr_row_ptr,
                                                 const aoclsparse_int      *csr_col_ind,
                                                 const float               *csr_val,
                                                 aoclsparse_int            *ell_col_ind,
                                                 float                     *ell_val,
                                                 aoclsparse_int             ell_width)
{
    return aoclsparse_csr2ell_template(
        m, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ell(aoclsparse_int             m,
                                                 const aoclsparse_mat_descr descr,
                                                 const aoclsparse_int      *csr_row_ptr,
                                                 const aoclsparse_int      *csr_col_ind,
                                                 const double              *csr_val,
                                                 aoclsparse_int            *ell_col_ind,
                                                 double                    *ell_val,
                                                 aoclsparse_int             ell_width)
{
    return aoclsparse_csr2ell_template(
        m, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

extern "C" aoclsparse_status aoclsparse_scsr2ellt(aoclsparse_int             m,
                                                  const aoclsparse_mat_descr descr,
                                                  const aoclsparse_int      *csr_row_ptr,
                                                  const aoclsparse_int      *csr_col_ind,
                                                  const float               *csr_val,
                                                  aoclsparse_int            *ell_col_ind,
                                                  float                     *ell_val,
                                                  aoclsparse_int             ell_width)
{
    return aoclsparse_csr2ellt_template(
        m, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ellt(aoclsparse_int             m,
                                                  const aoclsparse_mat_descr descr,
                                                  const aoclsparse_int      *csr_row_ptr,
                                                  const aoclsparse_int      *csr_col_ind,
                                                  const double              *csr_val,
                                                  aoclsparse_int            *ell_col_ind,
                                                  double                    *ell_val,
                                                  aoclsparse_int             ell_width)
{
    return aoclsparse_csr2ellt_template(
        m, descr, csr_row_ptr, csr_col_ind, csr_val, ell_col_ind, ell_val, ell_width);
}

extern "C" aoclsparse_status aoclsparse_scsr2ellthyb(aoclsparse_int                   m,
                                                     aoclsparse_index_base            base,
                                                     aoclsparse_int                  *ell_m,
                                                     const aoclsparse_int            *csr_row_ptr,
                                                     const aoclsparse_int            *csr_col_ind,
                                                     const float                     *csr_val,
                                                     [[maybe_unused]] aoclsparse_int *row_idx_map,
                                                     aoclsparse_int *csr_row_idx_map,
                                                     aoclsparse_int *ell_col_ind,
                                                     float          *ell_val,
                                                     aoclsparse_int  ell_width)
{
    return aoclsparse_csr2ellthybrid_template(m,
                                              base,
                                              ell_m,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              csr_val,
                                              csr_row_idx_map,
                                              ell_col_ind,
                                              ell_val,
                                              ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ellthyb(aoclsparse_int                   m,
                                                     aoclsparse_index_base            base,
                                                     aoclsparse_int                  *ell_m,
                                                     const aoclsparse_int            *csr_row_ptr,
                                                     const aoclsparse_int            *csr_col_ind,
                                                     const double                    *csr_val,
                                                     [[maybe_unused]] aoclsparse_int *row_idx_map,
                                                     aoclsparse_int *csr_row_idx_map,
                                                     aoclsparse_int *ell_col_ind,
                                                     double         *ell_val,
                                                     aoclsparse_int  ell_width)
{
    return aoclsparse_csr2ellthybrid_template(m,
                                              base,
                                              ell_m,
                                              csr_row_ptr,
                                              csr_col_ind,
                                              csr_val,
                                              csr_row_idx_map,
                                              ell_col_ind,
                                              ell_val,
                                              ell_width);
}

extern "C" aoclsparse_status aoclsparse_csr2dia_ndiag(aoclsparse_int             m,
                                                      aoclsparse_int             n,
                                                      const aoclsparse_mat_descr descr,
                                                      aoclsparse_int             nnz,
                                                      const aoclsparse_int      *csr_row_ptr,
                                                      const aoclsparse_int      *csr_col_ind,
                                                      aoclsparse_int            *dia_num_diag)
{

    // Check sizes
    if((m < 0) || (n < 0) || (nnz < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Check dia_num_diag pointer
    if(dia_num_diag == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check pointer arguments
    if((csr_row_ptr == nullptr) || (csr_col_ind == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Determine number of populated diagonals
    *dia_num_diag = 0;

    std::vector<aoclsparse_int> diag_idx;
    try
    {
        diag_idx.resize(m + n, 0);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    aoclsparse_index_base base = descr->base;

    // Loop over rows and increment ndiag counter if diag offset has not been visited yet
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        for(aoclsparse_int j = (csr_row_ptr[i] - base); j < (csr_row_ptr[i + 1] - base); ++j)
        {
            // Diagonal offset the current entry belongs to
            aoclsparse_int offset = csr_col_ind[j] - base - i + m;
            if(diag_idx[offset] == 0)
            {
                diag_idx[offset] = 1;
                ++*dia_num_diag;
            }
        }
    }

    return aoclsparse_status_success;
}

extern "C" aoclsparse_status aoclsparse_scsr2dia(aoclsparse_int             m,
                                                 aoclsparse_int             n,
                                                 const aoclsparse_mat_descr descr,
                                                 const aoclsparse_int      *csr_row_ptr,
                                                 const aoclsparse_int      *csr_col_ind,
                                                 const float               *csr_val,
                                                 aoclsparse_int             dia_num_diag,
                                                 aoclsparse_int            *dia_offset,
                                                 float                     *dia_val)
{
    return aoclsparse_csr2dia_template(
        m, n, descr, csr_row_ptr, csr_col_ind, csr_val, dia_num_diag, dia_offset, dia_val);
}

extern "C" aoclsparse_status aoclsparse_dcsr2dia(aoclsparse_int             m,
                                                 aoclsparse_int             n,
                                                 const aoclsparse_mat_descr descr,
                                                 const aoclsparse_int      *csr_row_ptr,
                                                 const aoclsparse_int      *csr_col_ind,
                                                 const double              *csr_val,
                                                 aoclsparse_int             dia_num_diag,
                                                 aoclsparse_int            *dia_offset,
                                                 double                    *dia_val)
{
    return aoclsparse_csr2dia_template(
        m, n, descr, csr_row_ptr, csr_col_ind, csr_val, dia_num_diag, dia_offset, dia_val);
}

extern "C" aoclsparse_status aoclsparse_csr2bsr_nnz(aoclsparse_int             m,
                                                    aoclsparse_int             n,
                                                    const aoclsparse_mat_descr descr,
                                                    const aoclsparse_int      *csr_row_ptr,
                                                    const aoclsparse_int      *csr_col_ind,
                                                    aoclsparse_int             block_dim,
                                                    aoclsparse_int            *bsr_row_ptr,
                                                    aoclsparse_int            *bsr_nnz)
{
    // Check sizes
    if(m < 0 || n < 0 || block_dim <= 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Check pointer arguments
    if(csr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(csr_col_ind == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_row_ptr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(bsr_nnz == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || block_dim == 0)
    {
        *bsr_nnz = 0;
        return aoclsparse_status_success;
    }

    if((descr->base != aoclsparse_index_base_zero) && (descr->base != aoclsparse_index_base_one))
    {
        return aoclsparse_status_invalid_value;
    }

    aoclsparse_index_base       base = descr->base;
    aoclsparse_int              mb   = (m + block_dim - 1) / block_dim;
    aoclsparse_int              nb   = (n + block_dim - 1) / block_dim;
    std::vector<bool>           blockcol;
    std::vector<aoclsparse_int> erase;
    try
    {
        blockcol.resize(nb, false);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    try
    {
        erase.resize(nb);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    // Loop over blocked rows
    for(aoclsparse_int bcsr_i = 0; bcsr_i < mb; ++bcsr_i)
    {
        // CSR row index
        aoclsparse_int csr_i = bcsr_i * block_dim;

        // number of blocks required in the blocked row
        aoclsparse_int nblocks = 0;

        // Loop over rows inside the current block
        for(aoclsparse_int i = 0; i < block_dim; ++i)
        {
            // Do not exceed CSR rows
            if(i >= m - csr_i)
            {
                break;
            }

            aoclsparse_int csr_row_begin = csr_row_ptr[csr_i + i] - base;
            aoclsparse_int csr_row_end   = csr_row_ptr[csr_i + i + 1] - base;

            // Loop over CSR columns for each of the rows in the block
            for(aoclsparse_int csr_j = csr_row_begin; csr_j < csr_row_end; ++csr_j)
            {
                // Block column index
                aoclsparse_int bcsr_j = (csr_col_ind[csr_j] - base) / block_dim;
                // Increment block counter for current blocked row if this column
                // creates a new block
                if(blockcol[bcsr_j] == false)
                {
                    blockcol[bcsr_j] = true;
                    erase[nblocks++] = bcsr_j;
                }
            }
        }

        // Store number of blocks of the current blocked row
        bsr_row_ptr[bcsr_i + 1] = nblocks;

        // Clear block buffer
        for(aoclsparse_int i = 0; i < nblocks; ++i)
        {
            blockcol[erase[i]] = false;
        }
    }

    // Exclusive sum to obtain BCSR row pointers while preserving the base-index
    bsr_row_ptr[0] = base;
    for(aoclsparse_int i = 0; i < mb; ++i)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    // Extract BCSR nnz
    *bsr_nnz = bsr_row_ptr[mb] - base;

    return aoclsparse_status_success;
}

extern "C" aoclsparse_status aoclsparse_scsr2bsr(aoclsparse_int             m,
                                                 aoclsparse_int             n,
                                                 const aoclsparse_mat_descr descr,
                                                 const aoclsparse_order     block_order,
                                                 const float               *csr_val,
                                                 const aoclsparse_int      *csr_row_ptr,
                                                 const aoclsparse_int      *csr_col_ind,
                                                 aoclsparse_int             block_dim,
                                                 float                     *bsr_val,
                                                 aoclsparse_int            *bsr_row_ptr,
                                                 aoclsparse_int            *bsr_col_ind)
{
    return aoclsparse_csr2bsr_template(m,
                                       n,
                                       descr,
                                       block_order,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       block_dim,
                                       bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind);
}

extern "C" aoclsparse_status aoclsparse_dcsr2bsr(aoclsparse_int             m,
                                                 aoclsparse_int             n,
                                                 const aoclsparse_mat_descr descr,
                                                 const aoclsparse_order     block_order,
                                                 const double              *csr_val,
                                                 const aoclsparse_int      *csr_row_ptr,
                                                 const aoclsparse_int      *csr_col_ind,
                                                 aoclsparse_int             block_dim,
                                                 double                    *bsr_val,
                                                 aoclsparse_int            *bsr_row_ptr,
                                                 aoclsparse_int            *bsr_col_ind)
{
    return aoclsparse_csr2bsr_template(m,
                                       n,
                                       descr,
                                       block_order,
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       block_dim,
                                       bsr_val,
                                       bsr_row_ptr,
                                       bsr_col_ind);
}

extern "C" aoclsparse_status aoclsparse_ccsr2bsr(aoclsparse_int                  m,
                                                 aoclsparse_int                  n,
                                                 const aoclsparse_mat_descr      descr,
                                                 const aoclsparse_order          block_order,
                                                 const aoclsparse_float_complex *csr_val,
                                                 const aoclsparse_int           *csr_row_ptr,
                                                 const aoclsparse_int           *csr_col_ind,
                                                 aoclsparse_int                  block_dim,
                                                 aoclsparse_float_complex       *bsr_val,
                                                 aoclsparse_int                 *bsr_row_ptr,
                                                 aoclsparse_int                 *bsr_col_ind)
{
    return aoclsparse_csr2bsr_template(m,
                                       n,
                                       descr,
                                       block_order,
                                       reinterpret_cast<const std::complex<float> *>(csr_val),
                                       csr_row_ptr,
                                       csr_col_ind,
                                       block_dim,
                                       reinterpret_cast<std::complex<float> *>(bsr_val),
                                       bsr_row_ptr,
                                       bsr_col_ind);
}

extern "C" aoclsparse_status aoclsparse_zcsr2bsr(aoclsparse_int                   m,
                                                 aoclsparse_int                   n,
                                                 const aoclsparse_mat_descr       descr,
                                                 const aoclsparse_order           block_order,
                                                 const aoclsparse_double_complex *csr_val,
                                                 const aoclsparse_int            *csr_row_ptr,
                                                 const aoclsparse_int            *csr_col_ind,
                                                 aoclsparse_int                   block_dim,
                                                 aoclsparse_double_complex       *bsr_val,
                                                 aoclsparse_int                  *bsr_row_ptr,
                                                 aoclsparse_int                  *bsr_col_ind)
{
    return aoclsparse_csr2bsr_template(m,
                                       n,
                                       descr,
                                       block_order,
                                       reinterpret_cast<const std::complex<double> *>(csr_val),
                                       csr_row_ptr,
                                       csr_col_ind,
                                       block_dim,
                                       reinterpret_cast<std::complex<double> *>(bsr_val),
                                       bsr_row_ptr,
                                       bsr_col_ind);
}

extern "C" aoclsparse_status aoclsparse_scsr2csc(aoclsparse_int             m,
                                                 aoclsparse_int             n,
                                                 aoclsparse_int             nnz,
                                                 const aoclsparse_mat_descr descr,
                                                 aoclsparse_index_base      baseCSC,
                                                 const aoclsparse_int      *csr_row_ptr,
                                                 const aoclsparse_int      *csr_col_ind,
                                                 const float               *csr_val,
                                                 aoclsparse_int            *csc_row_ind,
                                                 aoclsparse_int            *csc_col_ptr,
                                                 float                     *csc_val)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_csr2csc_template(m,
                                       n,
                                       nnz,
                                       descr->base,
                                       baseCSC,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       csc_row_ind,
                                       csc_col_ptr,
                                       csc_val);
}

extern "C" aoclsparse_status aoclsparse_dcsr2csc(aoclsparse_int             m,
                                                 aoclsparse_int             n,
                                                 aoclsparse_int             nnz,
                                                 const aoclsparse_mat_descr descr,
                                                 aoclsparse_index_base      baseCSC,
                                                 const aoclsparse_int      *csr_row_ptr,
                                                 const aoclsparse_int      *csr_col_ind,
                                                 const double              *csr_val,
                                                 aoclsparse_int            *csc_row_ind,
                                                 aoclsparse_int            *csc_col_ptr,
                                                 double                    *csc_val)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_csr2csc_template(m,
                                       n,
                                       nnz,
                                       descr->base,
                                       baseCSC,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       csc_row_ind,
                                       csc_col_ptr,
                                       csc_val);
}

extern "C" aoclsparse_status aoclsparse_ccsr2csc(aoclsparse_int                  m,
                                                 aoclsparse_int                  n,
                                                 aoclsparse_int                  nnz,
                                                 const aoclsparse_mat_descr      descr,
                                                 aoclsparse_index_base           baseCSC,
                                                 const aoclsparse_int           *csr_row_ptr,
                                                 const aoclsparse_int           *csr_col_ind,
                                                 const aoclsparse_float_complex *csr_val,
                                                 aoclsparse_int                 *csc_row_ind,
                                                 aoclsparse_int                 *csc_col_ptr,
                                                 aoclsparse_float_complex       *csc_val)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_csr2csc_template(m,
                                       n,
                                       nnz,
                                       descr->base,
                                       baseCSC,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       csc_row_ind,
                                       csc_col_ptr,
                                       csc_val);
}

extern "C" aoclsparse_status aoclsparse_zcsr2csc(aoclsparse_int                   m,
                                                 aoclsparse_int                   n,
                                                 aoclsparse_int                   nnz,
                                                 const aoclsparse_mat_descr       descr,
                                                 aoclsparse_index_base            baseCSC,
                                                 const aoclsparse_int            *csr_row_ptr,
                                                 const aoclsparse_int            *csr_col_ind,
                                                 const aoclsparse_double_complex *csr_val,
                                                 aoclsparse_int                  *csc_row_ind,
                                                 aoclsparse_int                  *csc_col_ptr,
                                                 aoclsparse_double_complex       *csc_val)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    return aoclsparse_csr2csc_template(m,
                                       n,
                                       nnz,
                                       descr->base,
                                       baseCSC,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       csr_val,
                                       csc_row_ind,
                                       csc_col_ptr,
                                       csc_val);
}

extern "C" aoclsparse_status aoclsparse_scsr2dense(aoclsparse_int             m,
                                                   aoclsparse_int             n,
                                                   const aoclsparse_mat_descr descr,
                                                   const float               *csr_val,
                                                   const aoclsparse_int      *csr_row_ptr,
                                                   const aoclsparse_int      *csr_col_ind,
                                                   float                     *A,
                                                   aoclsparse_int             ld,
                                                   aoclsparse_order           order)
{
    return aoclsparse_csr2dense_template(
        m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld, order);
}

extern "C" aoclsparse_status aoclsparse_dcsr2dense(aoclsparse_int             m,
                                                   aoclsparse_int             n,
                                                   const aoclsparse_mat_descr descr,
                                                   const double              *csr_val,
                                                   const aoclsparse_int      *csr_row_ptr,
                                                   const aoclsparse_int      *csr_col_ind,
                                                   double                    *A,
                                                   aoclsparse_int             ld,
                                                   aoclsparse_order           order)
{
    return aoclsparse_csr2dense_template(
        m, n, descr, csr_val, csr_row_ptr, csr_col_ind, A, ld, order);
}

extern "C" aoclsparse_status aoclsparse_ccsr2dense(aoclsparse_int                  m,
                                                   aoclsparse_int                  n,
                                                   const aoclsparse_mat_descr      descr,
                                                   const aoclsparse_float_complex *csr_val,
                                                   const aoclsparse_int           *csr_row_ptr,
                                                   const aoclsparse_int           *csr_col_ind,
                                                   aoclsparse_float_complex       *A,
                                                   aoclsparse_int                  ld,
                                                   aoclsparse_order                order)
{
    return aoclsparse_csr2dense_template<std::complex<float>>(m,
                                                              n,
                                                              descr,
                                                              (const std::complex<float> *)csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              (std::complex<float> *)A,
                                                              ld,
                                                              order);
}

extern "C" aoclsparse_status aoclsparse_zcsr2dense(aoclsparse_int                   m,
                                                   aoclsparse_int                   n,
                                                   const aoclsparse_mat_descr       descr,
                                                   const aoclsparse_double_complex *csr_val,
                                                   const aoclsparse_int            *csr_row_ptr,
                                                   const aoclsparse_int            *csr_col_ind,
                                                   aoclsparse_double_complex       *A,
                                                   aoclsparse_int                   ld,
                                                   aoclsparse_order                 order)
{
    return aoclsparse_csr2dense_template<std::complex<double>>(
        m,
        n,
        descr,
        (const std::complex<double> *)csr_val,
        csr_row_ptr,
        csr_col_ind,
        (std::complex<double> *)A,
        ld,
        order);
}

template <typename T>
aoclsparse_status aoclsparse_convert_csr_t(const aoclsparse_matrix    src_mat,
                                           const aoclsparse_operation op,
                                           aoclsparse_matrix         *dest_mat)
{
    if(src_mat == nullptr || dest_mat == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    *dest_mat = nullptr;
    if(src_mat->mats.empty() || !src_mat->mats[0])
        return aoclsparse_status_invalid_pointer;

    aoclsparse_status status = aoclsparse_status_success;
    T                *src_val, *dest_val;
    aoclsparse_int    m_dest, n_dest;

    if(op == aoclsparse_operation_none)
    {
        m_dest = src_mat->m;
        n_dest = src_mat->n;
    }
    else
    {
        m_dest = src_mat->n;
        n_dest = src_mat->m;
    }

    aoclsparse::csr *dest_csr = nullptr;
    try
    {
        *dest_mat = new _aoclsparse_matrix;
        dest_csr  = new aoclsparse::csr(m_dest,
                                       n_dest,
                                       src_mat->nnz,
                                       aoclsparse_csr_mat,
                                       src_mat->mats[0]->base,
                                       get_data_type<T>());
        (*dest_mat)->mats.push_back(dest_csr);
    }
    catch(std::bad_alloc &)
    {
        if(dest_csr)
            delete dest_csr;
        aoclsparse_destroy(dest_mat);
        return aoclsparse_status_memory_error;
    }

    switch(src_mat->input_format)
    {
    case aoclsparse_coo_mat:
    {
        aoclsparse::coo *coo_mat = dynamic_cast<aoclsparse::coo *>(src_mat->mats[0]);
        if(!coo_mat)
        {
            status = aoclsparse_status_not_implemented;
            break;
        }
        src_val  = reinterpret_cast<T *>(coo_mat->val);
        dest_val = reinterpret_cast<T *>(dest_csr->val);
        if(op == aoclsparse_operation_none)
        {
            status = aoclsparse_coo2csr_template(src_mat->m,
                                                 src_mat->n,
                                                 src_mat->nnz,
                                                 coo_mat->base,
                                                 coo_mat->row_ind,
                                                 coo_mat->col_ind,
                                                 src_val,
                                                 dest_csr->ptr,
                                                 dest_csr->ind,
                                                 dest_val);
        }
        else
        {
            status = aoclsparse_coo2csr_template(src_mat->n,
                                                 src_mat->m,
                                                 src_mat->nnz,
                                                 coo_mat->base,
                                                 coo_mat->col_ind,
                                                 coo_mat->row_ind,
                                                 src_val,
                                                 dest_csr->ptr,
                                                 dest_csr->ind,
                                                 dest_val);
        }
        break;
    }
    case aoclsparse_csr_mat:
    {
        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(src_mat->mats[0]);
        if(!csr_mat)
        {
            status = aoclsparse_status_not_implemented;
            break;
        }
        src_val     = reinterpret_cast<T *>(csr_mat->val);
        dest_val    = reinterpret_cast<T *>(dest_csr->val);
        bool is_csc = (csr_mat->doid == aoclsparse::doid::gt);
        // Determine if a direct memory copy is possible (no transpose for CSR, or transpose for CSC)
        bool direct_copy = (!is_csc && (op == aoclsparse_operation_none))
                           || (is_csc && (op != aoclsparse_operation_none));
        if(direct_copy)
        {
            memcpy(dest_csr->ptr, csr_mat->ptr, (csr_mat->m + 1) * sizeof(aoclsparse_int));
            memcpy(dest_csr->ind, csr_mat->ind, csr_mat->nnz * sizeof(aoclsparse_int));
            memcpy(dest_val, src_val, csr_mat->nnz * sizeof(T));
        }
        else
        {
            status = aoclsparse_csr2csc_template(csr_mat->m,
                                                 csr_mat->n,
                                                 csr_mat->nnz,
                                                 csr_mat->base,
                                                 csr_mat->base,
                                                 csr_mat->ptr,
                                                 csr_mat->ind,
                                                 src_val,
                                                 dest_csr->ind,
                                                 dest_csr->ptr,
                                                 dest_val);
        }
        break;
    }
    default:
        status = aoclsparse_status_not_implemented;
        break;
    }
    if(status != aoclsparse_status_success)
    {
        aoclsparse_destroy(dest_mat);
        return status;
    }

    if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
    {
        if(op == aoclsparse_operation_conjugate_transpose)
        {
            // transpose is done, now conjugate
            for(aoclsparse_int i = 0; i < src_mat->nnz; i++)
                dest_val[i] = std::conj(dest_val[i]);
        }
    }

    // creation of destination matrix depending on type of operation
    aoclsparse_init_mat(*dest_mat, m_dest, n_dest, src_mat->nnz, aoclsparse_csr_mat);
    (*dest_mat)->val_type = get_data_type<T>();
    return status;
}

aoclsparse_status aoclsparse_convert_csr(const aoclsparse_matrix    src_mat,
                                         const aoclsparse_operation op,
                                         aoclsparse_matrix         *dest_mat)
{

    if(src_mat == nullptr || dest_mat == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    *dest_mat = nullptr;

    aoclsparse_status status;

    switch(src_mat->val_type)
    {
    case aoclsparse_smat:
    {

        if((status = aoclsparse_convert_csr_t<float>(src_mat, op, dest_mat))
           != aoclsparse_status_success)
            return status;
    }
    break;
    case aoclsparse_dmat:
    {
        if((status = aoclsparse_convert_csr_t<double>(src_mat, op, dest_mat))
           != aoclsparse_status_success)
            return status;
    }
    break;
    case aoclsparse_cmat:
    {
        if((status = aoclsparse_convert_csr_t<std::complex<float>>(src_mat, op, dest_mat))
           != aoclsparse_status_success)
            return status;
    }
    break;
    case aoclsparse_zmat:
    {
        if((status = aoclsparse_convert_csr_t<std::complex<double>>(src_mat, op, dest_mat))
           != aoclsparse_status_success)
            return status;
    }
    break;
    default:
        return aoclsparse_status_invalid_value;
    }
    return aoclsparse_status_success;
}

namespace aoclsparse
{
    template <typename T>
    aoclsparse_status convert_bsr_t(const aoclsparse_matrix src_mat,
                                    aoclsparse_int          block_dim,
                                    aoclsparse_order        block_order,
                                    aoclsparse_operation    op,
                                    aoclsparse_matrix      *dest_mat)
    {
        if(src_mat == nullptr || dest_mat == nullptr)
            return aoclsparse_status_invalid_pointer;

        if(src_mat->mats.empty() || !src_mat->mats[0])
            return aoclsparse_status_invalid_pointer;

        aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(src_mat->mats[0]);
        if(!csr_mat)
            return aoclsparse_status_invalid_pointer;

        aoclsparse_matrix     transposed_mat = nullptr;
        aoclsparse_status     status         = aoclsparse_status_success;
        aoclsparse_int        m; //Number of rows in the source matrix
        aoclsparse_int        n; //Number of columns in the source matrix
        aoclsparse_int        mb; //Number of rows in the BSR matrix
        aoclsparse_int        nb; //Number of columns in the BSR matrix
        aoclsparse_int        bsr_nnz = 0; // Total number of non-zero blocks in BSR matrix
        aoclsparse_index_base base    = csr_mat->base;
        aoclsparse_mat_descr  descr;
        aoclsparse_int       *ind = nullptr, *ptr = nullptr;
        void                 *val = nullptr;

        aoclsparse_int *bsr_row_ptr = nullptr, *bsr_col_ind = nullptr;
        T              *bsr_val = nullptr;
        T               zero    = aoclsparse_numeric::zero<T>();
        aoclsparse_create_mat_descr(&descr);
        aoclsparse_set_mat_index_base(descr, base);

        switch(src_mat->input_format)
        {
        case aoclsparse_csr_mat:
            if(op == aoclsparse_operation_none)
            {
                // op = none, use the input matrix as is
                m   = src_mat->m;
                n   = src_mat->n;
                mb  = (m + block_dim - 1) / block_dim;
                nb  = (n + block_dim - 1) / block_dim;
                ptr = csr_mat->ptr;
                ind = csr_mat->ind;
                val = csr_mat->val;
            }
            else if(op == aoclsparse_operation_transpose
                    || op == aoclsparse_operation_conjugate_transpose)
            {
                // Use aoclsparse_convert_csr to handle transposition
                status = aoclsparse_convert_csr(src_mat, op, &transposed_mat);

                if(status != aoclsparse_status_success)
                {
                    aoclsparse_destroy_mat_descr(descr);
                    return status;
                }

                // Get the transposed CSR matrix
                if(transposed_mat->mats.empty() || !transposed_mat->mats[0])
                {
                    aoclsparse_destroy_mat_descr(descr);
                    return aoclsparse_status_invalid_pointer;
                }
                aoclsparse::csr *transposed_csr
                    = dynamic_cast<aoclsparse::csr *>(transposed_mat->mats[0]);
                if(!transposed_csr)
                {
                    aoclsparse_destroy(&transposed_mat);
                    aoclsparse_destroy_mat_descr(descr);
                    return aoclsparse_status_invalid_pointer;
                }

                // Set dimensions for transposed matrix
                m  = transposed_csr->m;
                n  = transposed_csr->n;
                mb = (m + block_dim - 1) / block_dim;
                nb = (n + block_dim - 1) / block_dim;

                // Use the transposed matrix data
                ptr = transposed_csr->ptr;
                ind = transposed_csr->ind;
                val = transposed_csr->val;
            }
            // Unsupported operation
            else
            {
                aoclsparse_destroy_mat_descr(descr);
                return aoclsparse_status_not_implemented;
            }

            // Allocate the memory for BSR matrix components
            try
            {
                bsr_row_ptr = new aoclsparse_int[mb + 1];
            }
            catch(std::bad_alloc &)
            {
                if(op != aoclsparse_operation_none)
                    aoclsparse_destroy(&transposed_mat);
                aoclsparse_destroy_mat_descr(descr);
                return aoclsparse_status_memory_error;
            }
            // Compute bsr_nnz count and populate BSR row pointer
            // ptr, ind and val points to source matrix or transposed matrix based on the op
            status
                = aoclsparse_csr2bsr_nnz(m, n, descr, ptr, ind, block_dim, bsr_row_ptr, &bsr_nnz);
            if(status != aoclsparse_status_success)
            {
                delete[] bsr_row_ptr;
                // Clean up the temporary matrix used in transpose conversions
                if(op != aoclsparse_operation_none)
                    aoclsparse_destroy(&transposed_mat);
                aoclsparse_destroy_mat_descr(descr);
                return status;
            }
            // Allocate memory for BSR column indices and values
            try
            {
                bsr_col_ind = new aoclsparse_int[bsr_nnz];
                bsr_val     = reinterpret_cast<T *>(
                    ::operator new(sizeof(T) * bsr_nnz * block_dim * block_dim));
            }
            catch(std::bad_alloc &)
            {
                delete[] bsr_row_ptr;
                delete[] bsr_col_ind;
                ::operator delete(bsr_val);
                // Clean up the temporary matrix used in transpose conversions
                if(op != aoclsparse_operation_none)
                    aoclsparse_destroy(&transposed_mat);
                aoclsparse_destroy_mat_descr(descr);
                return aoclsparse_status_memory_error;
            }

            // Initialize BSR values to zero
            // This ensures that all blocks are padded to zero, as some positions in blocks may not be filled during conversion
            for(aoclsparse_int i = 0; i < bsr_nnz * block_dim * block_dim; i++)
                bsr_val[i] = zero;

            // Call csr2bsr_template to fill bsr_col_ind and bsr_val
            status = aoclsparse_csr2bsr_template(m,
                                                 n,
                                                 descr,
                                                 block_order,
                                                 reinterpret_cast<const T *>(val),
                                                 ptr,
                                                 ind,
                                                 block_dim,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind);

            // Clean up the temporary matrix used in transpose conversions
            if(op != aoclsparse_operation_none)
                aoclsparse_destroy(&transposed_mat);
            break;
        default:
            // Unsupported input format
            status = aoclsparse_status_not_implemented;
            break;
        }

        // Check if the BSR conversion was successful
        if(status != aoclsparse_status_success)
        {
            delete[] bsr_row_ptr;
            delete[] bsr_col_ind;
            ::operator delete(bsr_val);
            aoclsparse_destroy_mat_descr(descr);
            return status;
        }
        aoclsparse::bsr *dest_bsr = nullptr;
        // Invoking constructor to create destination matrix by setting is_internal=true to represent internal allocation.
        // Whereas the public API aoclsparse_create_bsr will always have is_internal set to false.
        try
        {
            *dest_mat = new _aoclsparse_matrix;
            dest_bsr  = new aoclsparse::bsr(mb,
                                           nb,
                                           bsr_nnz,
                                           aoclsparse_bsr_mat,
                                           base,
                                           src_mat->val_type,
                                           block_dim,
                                           block_order,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_val,
                                           true);
            (*dest_mat)->mats.push_back(dest_bsr);
        }
        catch(std::bad_alloc &)
        {
            delete[] bsr_row_ptr;
            delete[] bsr_col_ind;
            ::operator delete(bsr_val);
            if(dest_bsr)
                delete dest_bsr;
            aoclsparse_destroy_mat_descr(descr);
            return aoclsparse_status_memory_error;
        }

        // Init base class properties of the destination matrix with actual dimensions
        aoclsparse_init_mat(*dest_mat,
                            mb * block_dim,
                            nb * block_dim,
                            bsr_nnz * block_dim * block_dim,
                            aoclsparse_bsr_mat);
        (*dest_mat)->val_type = get_data_type<T>();

        aoclsparse_destroy_mat_descr(descr);
        return status;
    }
} // namespace aoclsparse

aoclsparse_status aoclsparse_convert_bsr(const aoclsparse_matrix src_mat,
                                         aoclsparse_int          block_dim,
                                         aoclsparse_order        block_order,
                                         aoclsparse_operation    op,
                                         aoclsparse_matrix      *dest_mat)
{
    if(src_mat == nullptr || dest_mat == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(block_dim <= 0)
        return aoclsparse_status_invalid_value;

    if((block_order != aoclsparse_order_row && block_order != aoclsparse_order_column))
        return aoclsparse_status_invalid_value;

    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_not_implemented;

    *dest_mat = nullptr;
    aoclsparse_status status;

    if(src_mat->input_format != aoclsparse_csr_mat)
        return aoclsparse_status_not_implemented;

    switch(src_mat->val_type)
    {
    case aoclsparse_smat:
        status = aoclsparse::convert_bsr_t<float>(src_mat, block_dim, block_order, op, dest_mat);
        break;
    case aoclsparse_dmat:
        status = aoclsparse::convert_bsr_t<double>(src_mat, block_dim, block_order, op, dest_mat);
        break;
    case aoclsparse_cmat:
        status = aoclsparse::convert_bsr_t<std::complex<float>>(
            src_mat, block_dim, block_order, op, dest_mat);
        break;
    case aoclsparse_zmat:
        status = aoclsparse::convert_bsr_t<std::complex<double>>(
            src_mat, block_dim, block_order, op, dest_mat);
        break;
    default:
        return aoclsparse_status_invalid_value;
    }
    return status;
}
