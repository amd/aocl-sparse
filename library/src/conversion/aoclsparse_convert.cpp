/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_convert.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
aoclsparse_status aoclsparse_csr2ell_width(
        aoclsparse_int       m,
        aoclsparse_int       nnz,
        const aoclsparse_int *csr_row_ptr,
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

    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        *ell_width             = std::max(row_nnz, *ell_width);
    }

    // Compute ELL non-zeros
    aoclsparse_int ell_nnz = *ell_width * m;

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_csr2ellthyb_width(
        aoclsparse_int       m,
        aoclsparse_int       nnz,
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

    aoclsparse_int mx_nnz_lt_nnza = 0, mn_nnz_gt_nnza = nnz, cmn = 0, cmx = 0, nnza = nnz/m;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int nnzi = csr_row_ptr[i + 1] - csr_row_ptr[i];
        if ((nnzi > mx_nnz_lt_nnza) && (nnzi <= nnza)) {
            mx_nnz_lt_nnza = nnzi;
        }
        if ((nnzi < mn_nnz_gt_nnza) && (nnzi > nnza)) {
            mn_nnz_gt_nnza = nnzi;
        }

        if ((nnzi <= nnza))
            cmx++;
        else
            cmn++;
    }
    if (cmx >= cmn)
       *ell_width = mx_nnz_lt_nnza;
    else
       *ell_width = mn_nnz_gt_nnza;
    
    *ell_m = 0;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
        if (row_nnz <= *ell_width)
           (*ell_m)++;
    }

    return aoclsparse_status_success;
}


extern "C" aoclsparse_status aoclsparse_scsr2ell(
        aoclsparse_int       m,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const float          *csr_val,
        aoclsparse_int       *ell_col_ind,
        float                *ell_val,
        aoclsparse_int       ell_width)
{
    return aoclsparse_csr2ell_template(m,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            ell_col_ind,
            ell_val,
            ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ell(
        aoclsparse_int       m,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const double         *csr_val,
        aoclsparse_int       *ell_col_ind,
        double               *ell_val,
        aoclsparse_int       ell_width)
{
    return aoclsparse_csr2ell_template(m,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            ell_col_ind,
            ell_val,
            ell_width);
}


extern "C" aoclsparse_status aoclsparse_scsr2ellt(
        aoclsparse_int       m,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const float          *csr_val,
        aoclsparse_int       *ell_col_ind,
        float                *ell_val,
        aoclsparse_int       ell_width)
{
    return aoclsparse_csr2ellt_template(m,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            ell_col_ind,
            ell_val,
            ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ellt(
        aoclsparse_int       m,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const double         *csr_val,
        aoclsparse_int       *ell_col_ind,
        double               *ell_val,
        aoclsparse_int       ell_width)
{
    return aoclsparse_csr2ellt_template(m,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            ell_col_ind,
            ell_val,
            ell_width);
}

extern "C" aoclsparse_status aoclsparse_scsr2ellthyb(
        aoclsparse_int       m,
	aoclsparse_int       *ell_m,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const float          *csr_val,
        aoclsparse_int       *row_idx_map,
        aoclsparse_int       *csr_row_idx_map,
        aoclsparse_int       *ell_col_ind,
        float                *ell_val,
        aoclsparse_int       ell_width)
{
    return aoclsparse_csr2ellthybrid_template(m,
            ell_m,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            row_idx_map,
	    csr_row_idx_map, 
            ell_col_ind,
            ell_val,
            ell_width);
}

extern "C" aoclsparse_status aoclsparse_dcsr2ellthyb(
        aoclsparse_int       m,
	aoclsparse_int       *ell_m,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const double         *csr_val,
        aoclsparse_int       *row_idx_map,
        aoclsparse_int       *csr_row_idx_map,	
        aoclsparse_int       *ell_col_ind,
        double               *ell_val,
        aoclsparse_int       ell_width)
{
    return aoclsparse_csr2ellthybrid_template(m,
	    ell_m,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            row_idx_map,
            csr_row_idx_map,	    
            ell_col_ind,
            ell_val,
            ell_width);
}

extern "C" aoclsparse_status aoclsparse_csr2dia_ndiag(
        aoclsparse_int       m,
        aoclsparse_int       n,
        aoclsparse_int       nnz,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        aoclsparse_int       *dia_num_diag)
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

    std::vector<aoclsparse_int> diag_idx(m + n, 0);

    // Loop over rows and increment ndiag counter if diag offset has not been visited yet
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        for(aoclsparse_int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
        {
            // Diagonal offset the current entry belongs to
            aoclsparse_int offset = csr_col_ind[j] - i + m;
            if(diag_idx[offset] == 0)
            {
                diag_idx[offset] = 1;
                ++*dia_num_diag;
            }
        }
    }
    aoclsparse_int size = (m > n) ? m : n;
    aoclsparse_int nnz_dia = size * *dia_num_diag;

    return aoclsparse_status_success;
}

extern "C" aoclsparse_status aoclsparse_scsr2dia(
        aoclsparse_int       m,
        aoclsparse_int       n,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const float          *csr_val,
        aoclsparse_int       dia_num_diag,
        aoclsparse_int       *dia_offset,
        float                *dia_val)
{
    return aoclsparse_csr2dia_template(m,
            n,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            dia_num_diag,
            dia_offset,
            dia_val);
}

extern "C" aoclsparse_status aoclsparse_dcsr2dia(
        aoclsparse_int       m,
        aoclsparse_int       n,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const double         *csr_val,
        aoclsparse_int       dia_num_diag,
        aoclsparse_int       *dia_offset,
        double               *dia_val)
{
    return aoclsparse_csr2dia_template(m,
            n,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            dia_num_diag,
            dia_offset,
            dia_val);
}

extern "C" aoclsparse_status aoclsparse_csr2bsr_nnz(
        aoclsparse_int        m,
        aoclsparse_int        n,
        const aoclsparse_int  *csr_row_ptr,
        const aoclsparse_int  *csr_col_ind,
        aoclsparse_int        block_dim,
        aoclsparse_int        *bsr_row_ptr,
        aoclsparse_int        *bsr_nnz)
{
    // Check sizes
    if(m < 0 || n < 0 || block_dim < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || block_dim == 0)
    {
        *bsr_nnz = 0;
        return aoclsparse_status_success;
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
    aoclsparse_int mb = (m + block_dim - 1) / block_dim;
    aoclsparse_int nb = (n + block_dim - 1) / block_dim;
    std::vector<bool>      blockcol(nb, false);
    std::vector<aoclsparse_int> erase(nb);
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

            aoclsparse_int csr_row_begin = csr_row_ptr[csr_i + i];
            aoclsparse_int csr_row_end   = csr_row_ptr[csr_i + i + 1];

            // Loop over CSR columns for each of the rows in the block
            for(aoclsparse_int csr_j = csr_row_begin; csr_j < csr_row_end; ++csr_j)
            {
                // Block column index
                aoclsparse_int bcsr_j = csr_col_ind[csr_j] / block_dim;

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

    // Exclusive sum to obtain BCSR row pointers
    bsr_row_ptr[0] = 0;
    for(aoclsparse_int i = 0; i < mb; ++i)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    // Extract BCSR nnz
    *bsr_nnz = bsr_row_ptr[mb];
    return aoclsparse_status_success;
}

extern "C" aoclsparse_status aoclsparse_scsr2bsr(
        aoclsparse_int       m,
        aoclsparse_int       n,
        const float          *csr_val,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        aoclsparse_int       block_dim,
        float                *bsr_val,
        aoclsparse_int       *bsr_row_ptr,
        aoclsparse_int       *bsr_col_ind)
{
    return aoclsparse_csr2bsr_template(m,
            n,
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            block_dim,
            bsr_val,
            bsr_row_ptr,
            bsr_col_ind);
}

extern "C" aoclsparse_status aoclsparse_dcsr2bsr(
        aoclsparse_int       m,
        aoclsparse_int       n,
        const double         *csr_val,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        aoclsparse_int       block_dim,
        double               *bsr_val,
        aoclsparse_int       *bsr_row_ptr,
        aoclsparse_int       *bsr_col_ind)
{
    return aoclsparse_csr2bsr_template(m,
            n,
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            block_dim,
            bsr_val,
            bsr_row_ptr,
            bsr_col_ind);
}

extern "C" aoclsparse_status aoclsparse_scsr2csc(
        aoclsparse_int       m,
        aoclsparse_int       n,
        aoclsparse_int       nnz,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const float          *csr_val,
        aoclsparse_int       *csc_row_ind,
        aoclsparse_int       *csc_col_ptr,
        float                *csc_val)
{
    return aoclsparse_csr2csc_template(m,
            n,
            nnz,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            csc_row_ind,
            csc_col_ptr,
            csc_val);
}

extern "C" aoclsparse_status aoclsparse_dcsr2csc(
        aoclsparse_int       m,
        aoclsparse_int       n,
        aoclsparse_int       nnz,
        const aoclsparse_int *csr_row_ptr,
        const aoclsparse_int *csr_col_ind,
        const double         *csr_val,
        aoclsparse_int       *csc_row_ind,
        aoclsparse_int       *csc_col_ptr,
        double               *csc_val)
{
    return aoclsparse_csr2csc_template(m,
            n,
            nnz,
            csr_row_ptr,
            csr_col_ind,
            csr_val,
            csc_row_ind,
            csc_col_ptr,
            csc_val);
}

extern "C" aoclsparse_status aoclsparse_scsr2dense(
            aoclsparse_int             m,
            aoclsparse_int             n,
            const aoclsparse_mat_descr descr,
            const float*               csr_val,
            const aoclsparse_int*      csr_row_ptr,
            const aoclsparse_int*      csr_col_ind,
            float*                     A,
            aoclsparse_int             ld,
            aoclsparse_order           order)
{
    return aoclsparse_csr2dense_template(m,
            n,
	    descr,
	    csr_val,
	    csr_row_ptr,
	    csr_col_ind,
	    A,
	    ld,
	    order);
}

extern "C" aoclsparse_status aoclsparse_dcsr2dense(
            aoclsparse_int             m,
            aoclsparse_int             n,
            const aoclsparse_mat_descr descr,
            const double*              csr_val,
            const aoclsparse_int*      csr_row_ptr,
            const aoclsparse_int*      csr_col_ind,
            double*                    A,
            aoclsparse_int             ld,
            aoclsparse_order           order)
{
    return aoclsparse_csr2dense_template(m,
            n,
	    descr,
	    csr_val,
	    csr_row_ptr,
	    csr_col_ind,
	    A,
	    ld,
	    order);
}

aoclsparse_status aoclsparse_optimize(aoclsparse_matrix A)
{
    // Validations

    // check if already optimized
    if (A->optimized) {
        return aoclsparse_status_success;
    } 

    // Check sizes
    if((A->m < 0) || (A->n < 0) ||  (A->nnz < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Check CSR matrix is populated, it not return an error. ToDo: need to handle CSC / COO cases later
    if((A->csr_mat.csr_row_ptr == nullptr) || (A->csr_mat.csr_col_ptr == nullptr) || (A->csr_mat.csr_val == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }


    // collect the required for decision making
    aoclsparse_int *row_ptr = A->csr_mat.csr_row_ptr;
    aoclsparse_int m = A->m;
    // 1: ELL width
    aoclsparse_int ell_width = 0, nnz = A->nnz;
    double nnza = (double)nnz/m;
    aoclsparse_int mx_nnz_lt_nnza = 0, mn_nnz_gt_nnza = nnz, cmn = 0, cmx = 0;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int nnzi = row_ptr[i + 1] - row_ptr[i];
        if ((nnzi > mx_nnz_lt_nnza) && (nnzi <= nnza)) {
            mx_nnz_lt_nnza = nnzi;
        }
        if ((nnzi < mn_nnz_gt_nnza) && (nnzi > nnza)) {
            mn_nnz_gt_nnza = nnzi;
        }

        if ((nnzi <= nnza))
            cmx++;
        else
            cmn++;
    }
    if (cmx >= cmn)
       ell_width = mx_nnz_lt_nnza;
    else
       ell_width = mn_nnz_gt_nnza;

    // 2: csr_rows_with_nnz_lt_10, ell_csr_nnz (hybrid fillin), ...
    aoclsparse_int ell_m = 0, ell_csr_nnz = 0, ell_csr_g_ew_l_10 = 0,  ell_csr_g_ew_g_10 = 0, csr_lt_10 = 0, rem = 0;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_nnz = row_ptr[i + 1] - row_ptr[i];
        if (row_nnz <= ell_width)
           (ell_m)++;
        else {
            ell_csr_nnz += row_nnz;
            if (row_nnz <= 10)
                    ell_csr_g_ew_l_10++;
            else
                    ell_csr_g_ew_g_10++;
        }
        if (row_nnz <= 10)
                csr_lt_10++;
        rem += row_nnz%4;
    }

    aoclsparse_int ell_nnz = ell_width*m + ell_csr_nnz;

    // 3: Fill-in for csr_br4 implementation
    aoclsparse_int i,j, tnnz = 0;
    aoclsparse_int row_nnz;
    for (i = 0 ; i < m; i+=4){

        aoclsparse_int m1, m2;
        if ((m - i) < 4)
            break;
        m1 = std::max((row_ptr[i+1]-row_ptr[i]),
                        (row_ptr[i+2]-row_ptr[i+1]) );
        m2 = std::max((row_ptr[i+3]-row_ptr[i+2]),
                        (row_ptr[i+4]-row_ptr[i+3]) );
        row_nnz = std::max(m1,m2);
        tnnz += 4*row_nnz;
    }
    for (j = i; j < m; ++j) {
        row_nnz = row_ptr[j+1]-row_ptr[j];
        tnnz += row_nnz;
    }
    double prctg_rows_lt_10 = (double)(csr_lt_10/m)*100;
    double fill_ratio = ((double)(tnnz-ell_nnz)/tnnz)*100;

    if (ell_width <= 30){ // ToDo: why this cutoff?
      if (fill_ratio < -0.1) {
          A->mat_type = aoclsparse_csr_mat_br4;   // CSR-BT-AVX2
      } else if ((fill_ratio > -0.1) && (fill_ratio < 0.1)){
          A->mat_type = aoclsparse_ellt_csr_hyb_mat;   // ELL-CSR-HYB 
      } else {
          if ( ell_csr_g_ew_l_10 < ell_csr_g_ew_g_10) {
              A->mat_type = aoclsparse_ellt_csr_hyb_mat; // ELL-CSR-HYB 
          } else {
             A->mat_type = aoclsparse_csr_mat_br4;  // CSR-BT-AVX2
          }
      }
    } else {
      if (prctg_rows_lt_10 < 10.0) {
         A->mat_type = aoclsparse_csr_mat;     // CSR 
      } else {
         A->mat_type = aoclsparse_csr_mat_br4; // CSR-BT-AVX2
      }
    }

    if (A->mat_type == aoclsparse_ellt_csr_hyb_mat) {

        aoclsparse_int *ell_col_ind; 
        double *ell_dval;
        float *ell_sval;
        aoclsparse_int *csr_row_idx_map;
        aoclsparse_int *row_idx_map;
        aoclsparse_int              ell_width;
        aoclsparse_int              ell_m;

        // get the ell_width
        aoclsparse_csr2ellthyb_width(A->m, A->nnz, A->csr_mat.csr_row_ptr, &ell_m, &ell_width);

        ell_col_ind = (aoclsparse_int *) malloc(sizeof(aoclsparse_int)*ell_width * A->m);

        if (A->val_type == aoclsparse_dmat) {
            ell_dval = (double *) malloc(sizeof(double)*ell_width * A->m);
        } else if (A->val_type == aoclsparse_smat) {
            ell_sval = (float *) malloc(sizeof(float)*ell_width * A->m);
        }
        csr_row_idx_map = (aoclsparse_int*) malloc(sizeof(aoclsparse_int)*(A->m - ell_m));

        // convert to hybrid ELLT-CSR format
        if (A->val_type == aoclsparse_dmat) {
            aoclsparse_dcsr2ellthyb(A->m, &ell_m, A->csr_mat.csr_row_ptr, A->csr_mat.csr_col_ptr, 
            (double *)A->csr_mat.csr_val, row_idx_map, csr_row_idx_map, ell_col_ind, ell_dval, ell_width);
            A->ell_csr_hyb_mat.ell_val = (double*)ell_dval;
        } else if (A->val_type == aoclsparse_smat) {
            aoclsparse_scsr2ellthyb(A->m, &ell_m, A->csr_mat.csr_row_ptr, A->csr_mat.csr_col_ptr, 
            (float *) A->csr_mat.csr_val, row_idx_map, csr_row_idx_map, ell_col_ind, ell_sval, ell_width);
            A->ell_csr_hyb_mat.ell_val = (float*)ell_sval;
        }

        // set appropriate members of "A"
        A->ell_csr_hyb_mat.ell_width = ell_width;
        A->ell_csr_hyb_mat.ell_m = ell_m;
        A->ell_csr_hyb_mat.ell_col_ind = ell_col_ind;
        A->ell_csr_hyb_mat.csr_row_id_map = csr_row_idx_map;
        A->ell_csr_hyb_mat.csr_col_ptr = A->csr_mat.csr_col_ptr;
        A->ell_csr_hyb_mat.csr_val = A->csr_mat.csr_val;

    } else if (A->mat_type == aoclsparse_csr_mat_br4) {  // vectorized csr blocked format for AVX2
        aoclsparse_int *col_ptr;
        aoclsparse_int *row_ptr;
        aoclsparse_int *trow_ptr;   // ToDo: need to replace row_ptr with trow_ptr
        void *csr_val;
        aoclsparse_int row_nnz;

        // populate row_nnz
        aoclsparse_int i;
        aoclsparse_int j;
        aoclsparse_int tnnz = 0;
        row_ptr =(aoclsparse_int *) malloc(sizeof(aoclsparse_int) * (A->m));
        trow_ptr =(aoclsparse_int *) malloc(sizeof(aoclsparse_int) * (A->m + 1));
        trow_ptr[0] = A->base;
        for (i = 0 ; i < A->m; i+=4){

            aoclsparse_int m1, m2;
            if ((A->m - i) < 4)
               break;
            m1 = std::max((A->csr_mat.csr_row_ptr[i+1]-A->csr_mat.csr_row_ptr[i]),
                           (A->csr_mat.csr_row_ptr[i+2]-A->csr_mat.csr_row_ptr[i+1]) );
            m2 = std::max((A->csr_mat.csr_row_ptr[i+3]-A->csr_mat.csr_row_ptr[i+2]),
                           (A->csr_mat.csr_row_ptr[i+4]-A->csr_mat.csr_row_ptr[i+3]) );
            row_nnz = std::max(m1,m2);
            row_ptr[i] = row_ptr[i+1] = row_ptr[i+2] = row_ptr[i+3] = row_nnz;
            trow_ptr[i+1] = trow_ptr[i] + row_nnz;
            trow_ptr[i+2] = trow_ptr[i+1] + row_nnz;
            trow_ptr[i+3] = trow_ptr[i+2] + row_nnz;
            trow_ptr[i+4] = trow_ptr[i+3] + row_nnz;
            tnnz += 4*row_nnz;
        }
        for (j = i; j < A->m; ++j) {
            row_nnz = A->csr_mat.csr_row_ptr[j+1]-A->csr_mat.csr_row_ptr[j];
            tnnz += row_nnz;
            row_ptr[j] = row_nnz;
            trow_ptr[j+1] = trow_ptr[j] + row_nnz;
        }

        // create the new csr matrix and convert to the csr-avx2 format
        col_ptr =(aoclsparse_int *) malloc(sizeof(aoclsparse_int) * tnnz);
        if (A->val_type == aoclsparse_dmat) {
            csr_val = (double *) malloc(sizeof(double)*tnnz);
        } else if (A->val_type == aoclsparse_smat) {
            csr_val = (float *) malloc(sizeof(float)*tnnz);
        }
        aoclsparse_int tc = 0; // count of nonzeros
        for (i = 0; i < A->m; ++i) {
            aoclsparse_int nz = A->csr_mat.csr_row_ptr[i+1]-A->csr_mat.csr_row_ptr[i];
            aoclsparse_int ridx = A->csr_mat.csr_row_ptr[i];
            for (j = 0; j < nz; ++j) {
                ((double *)csr_val)[tc] = ((double *)A->csr_mat.csr_val)[ridx+j];
                col_ptr[tc] = A->csr_mat.csr_col_ptr[ridx+j];
                tc++;
            }
            if (nz < row_ptr[i]) { // ToDo -- can remove the if condition
                for (j = nz; j < row_ptr[i]; ++j) {
                    col_ptr[tc] = A->csr_mat.csr_col_ptr[ridx+nz-1];
                    ((double *)csr_val)[tc] = static_cast<double>(0);
                    tc++;
                }
            }
        }
        tc = 0;
        aoclsparse_int *cptr = (aoclsparse_int *) col_ptr;
        double *vptr = (double *) csr_val;
        for(i = 0; i < A->m; i+=4) {
            cptr = col_ptr + tc;
            vptr = (double*)csr_val + tc;
            aoclsparse_int nnz = row_ptr[i];
            if ((A->m - i) < 4)
               break;
            // transponse the chunk into an auxiliary buffer
            double *bufval = (double *)malloc(sizeof(double)*nnz*4);
            aoclsparse_int *bufidx = (aoclsparse_int *)malloc(sizeof(aoclsparse_int)*nnz*4);
            aoclsparse_int ii, jj;
            for (ii = 0; ii < nnz; ++ii) {
                for (jj = 0; jj < 4;  ++jj) {
                    bufval[jj+ii*4] = vptr[ii + jj*nnz];
                    bufidx[jj+ii*4] = cptr[ii + jj*nnz];
                }
            }
            memcpy(vptr, bufval, sizeof(double)*nnz*4);
            memcpy(cptr, bufidx, sizeof(aoclsparse_int)*nnz*4);
            free (bufval);
            free(bufidx);
            tc += nnz*4;
        }

        // set appropriate members of "A"
//        A->csr_mat_br4.csr_row_ptr = row_ptr;
        free(row_ptr);
        A->csr_mat_br4.csr_row_ptr = trow_ptr;   // ToDo: replace row_ptr with this
        A->csr_mat_br4.csr_col_ptr = col_ptr;
        A->csr_mat_br4.csr_val = csr_val;
    }
    
    A->optimized = true;

    return aoclsparse_status_success;
}
