/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <limits.h>

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
    if((m < 0) || (nnz < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Check pointer arguments
    if((csr_row_ptr == nullptr) || (csr_col_ind == nullptr) || (total_blks == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
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
                        iCol          = std::min<int>(iCol, (csr_col_ind[iVal[iSubRow]] - base));
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
                    iCol          = std::min<int>(iCol, (csr_col_ind[iVal[iSubRow]] - base));
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
    if(m < 0 || n < 0 || block_dim < 0)
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
                                       csr_val,
                                       csr_row_ptr,
                                       csr_col_ind,
                                       block_dim,
                                       bsr_val,
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
    return aoclsparse_csr2csc_template(m,
                                       n,
                                       nnz,
                                       descr,
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
    return aoclsparse_csr2csc_template(m,
                                       n,
                                       nnz,
                                       descr,
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

template <typename T>
aoclsparse_status aoclsparse_convert_csr_t(const aoclsparse_matrix    src_mat,
                                           const aoclsparse_operation op,
                                           aoclsparse_matrix         *dest_mat)
{
    if(src_mat == nullptr || dest_mat == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    aoclsparse_status status  = aoclsparse_status_success;
    aoclsparse_int   *row_ptr = nullptr, *col_ind = nullptr;
    T                *val        = nullptr, *src_val;
    void             *temp_alloc = nullptr;
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

    try
    {
        row_ptr    = new aoclsparse_int[m_dest + 1];
        col_ind    = new aoclsparse_int[src_mat->nnz];
        temp_alloc = ::operator new(src_mat->nnz * sizeof(T));
        val        = reinterpret_cast<T *>(temp_alloc);
        *dest_mat  = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        delete[] row_ptr;
        delete[] col_ind;
        ::operator delete(val);
        delete *dest_mat;
        *dest_mat = nullptr;
        return aoclsparse_status_memory_error;
    }

    switch(src_mat->input_format)
    {
    case aoclsparse_coo_mat:

        src_val = reinterpret_cast<T *>(src_mat->coo_mat.val);
        if(op == aoclsparse_operation_none)
        {
            status = aoclsparse_coo2csr_template(src_mat->m,
                                                 src_mat->n,
                                                 src_mat->nnz,
                                                 src_mat->base,
                                                 src_mat->coo_mat.row_ind,
                                                 src_mat->coo_mat.col_ind,
                                                 src_val,
                                                 row_ptr,
                                                 col_ind,
                                                 val);
        }
        else
        {
            status = aoclsparse_coo2csr_template(src_mat->n,
                                                 src_mat->m,
                                                 src_mat->nnz,
                                                 src_mat->base,
                                                 src_mat->coo_mat.col_ind,
                                                 src_mat->coo_mat.row_ind,
                                                 src_val,
                                                 row_ptr,
                                                 col_ind,
                                                 val);
        }
        break;

    case aoclsparse_csr_mat:

        src_val = reinterpret_cast<T *>(src_mat->csr_mat.csr_val);
        if(op == aoclsparse_operation_none)
        {
            memcpy(
                row_ptr, src_mat->csr_mat.csr_row_ptr, (src_mat->m + 1) * sizeof(aoclsparse_int));
            memcpy(col_ind, src_mat->csr_mat.csr_col_ptr, src_mat->nnz * sizeof(aoclsparse_int));
            memcpy(val, src_val, src_mat->nnz * sizeof(T));
        }
        else
        {
            _aoclsparse_mat_descr descr;
            descr.base = src_mat->base;
            status     = aoclsparse_csr2csc_template(src_mat->m,
                                                 src_mat->n,
                                                 src_mat->nnz,
                                                 &descr,
                                                 src_mat->base,
                                                 src_mat->csr_mat.csr_row_ptr,
                                                 src_mat->csr_mat.csr_col_ptr,
                                                 src_val,
                                                 col_ind,
                                                 row_ptr,
                                                 val);
        }
        break;

    case aoclsparse_csc_mat:

        src_val = reinterpret_cast<T *>(src_mat->csc_mat.val);
        if(op == aoclsparse_operation_none)
        {
            _aoclsparse_mat_descr descr;
            descr.base = src_mat->base;
            status     = aoclsparse_csr2csc_template(src_mat->n,
                                                 src_mat->m,
                                                 src_mat->nnz,
                                                 &descr,
                                                 src_mat->base,
                                                 src_mat->csc_mat.col_ptr,
                                                 src_mat->csc_mat.row_idx,
                                                 src_val,
                                                 col_ind,
                                                 row_ptr,
                                                 val);
        }
        else
        {
            memcpy(row_ptr, src_mat->csc_mat.col_ptr, (src_mat->n + 1) * sizeof(aoclsparse_int));
            memcpy(col_ind, src_mat->csc_mat.row_idx, src_mat->nnz * sizeof(aoclsparse_int));
            memcpy(val, src_val, src_mat->nnz * sizeof(T));
        }
        break;

    default:
        status = aoclsparse_status_not_implemented;
        break;
    }
    if(status != aoclsparse_status_success)
    {
        delete[] row_ptr;
        delete[] col_ind;
        ::operator delete(val);
        delete *dest_mat;
        *dest_mat = nullptr;
        return status;
    }

    if constexpr(std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
    {
        if(op == aoclsparse_operation_conjugate_transpose)
        {
            // transpose is done, now conjugate
            for(aoclsparse_int i = 0; i < src_mat->nnz; i++)
                val[i] = std::conj(val[i]);
        }
    }

    aoclsparse_init_mat(*dest_mat, src_mat->base, m_dest, n_dest, src_mat->nnz, aoclsparse_csr_mat);
    (*dest_mat)->csr_mat.csr_row_ptr = row_ptr;
    (*dest_mat)->csr_mat.csr_col_ptr = col_ind;
    (*dest_mat)->csr_mat.csr_val     = val;
    (*dest_mat)->val_type            = get_data_type<T>();
    (*dest_mat)->csr_mat_is_users    = false;
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
