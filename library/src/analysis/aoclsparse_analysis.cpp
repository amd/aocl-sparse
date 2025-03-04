/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "aoclsparse_context.h"
#include "aoclsparse_analysis.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
aoclsparse_status aoclsparse_optimize_mv(aoclsparse_matrix A)
{
    // Return without optimizations if
    // 1) the matrix type is not double precision real
    // 2) matrix dimensions are <= 1
    // 3) matrix is not stored in the csr format
    // 4) ToDo: add more exceptions
    if(A->val_type != aoclsparse_dmat || A->mat_type != aoclsparse_csr_mat || A->m <= 1
       || A->n <= 1)
    {
        A->optimized = true;
        return aoclsparse_status_success;
    }

    // early return if already optimized
    // ToDo: could have side effect of not optimizing A if
    //       another  *_optimize_* API updates this variable
    if(A->optimized)
    {
        return aoclsparse_status_success;
    }

    // collect the required for decision making
    aoclsparse_int *row_ptr = A->csr_mat.csr_row_ptr;
    aoclsparse_int  m       = A->m;
    // 1: ELL width
    aoclsparse_int ell_width = 0, nnz = A->nnz;
    double         nnza           = (double)nnz / m;
    aoclsparse_int mx_nnz_lt_nnza = 0, mn_nnz_gt_nnza = nnz, cmn = 0, cmx = 0;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int nnzi = row_ptr[i + 1] - row_ptr[i];
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
        ell_width = mx_nnz_lt_nnza;
    else
        ell_width = mn_nnz_gt_nnza;
    // 2: csr_rows_with_nnz_lt_10, ell_csr_nnz (hybrid fillin), ...
    aoclsparse_int ell_m = 0, ell_csr_nnz = 0, ell_csr_g_ew_l_10 = 0, ell_csr_g_ew_g_10 = 0;
    for(aoclsparse_int i = 0; i < m; ++i)
    {
        aoclsparse_int row_nnz = row_ptr[i + 1] - row_ptr[i];
        if(row_nnz <= ell_width)
            (ell_m)++;
        else
        {
            ell_csr_nnz += row_nnz;
            if(row_nnz <= 10)
                ell_csr_g_ew_l_10++;
            else
                ell_csr_g_ew_g_10++;
        }
    }
    aoclsparse_int ell_nnz = ell_width * m + ell_csr_nnz;
    // 3: Fill-in for csr_br4 implementation
    aoclsparse_int i, j, tnnz = 0;
    aoclsparse_int row_nnz;
    for(i = 0; i < m; i += 4)
    {
        aoclsparse_int m1, m2;
        if((m - i) < 4)
            break;
        m1      = (std::max)((row_ptr[i + 1] - row_ptr[i]), (row_ptr[i + 2] - row_ptr[i + 1]));
        m2      = (std::max)((row_ptr[i + 3] - row_ptr[i + 2]), (row_ptr[i + 4] - row_ptr[i + 3]));
        row_nnz = (std::max)(m1, m2);
        tnnz += 4 * row_nnz;
    }
    for(j = i; j < m; ++j)
    {
        row_nnz = row_ptr[j + 1] - row_ptr[j];
        tnnz += row_nnz;
    }
    double fill_ratio = 0;
    if(tnnz != 0)
    {
        fill_ratio = ((double)(tnnz - ell_nnz) / tnnz) * 100;
    }

    using namespace aoclsparse;

    /*
        Check if the requested operation can execute
        This check needs to be done only once in a run
    */
    static bool can_exec = context::get_context()->supports<context_isa_t::AVX512F>()
                           && aoclsparse_is_avx512_build();

    // conversion of blkcsr assumes sorted indices in rows so don't test suitability
    // of this format unless the matrix is sorted
    if(can_exec && A->sort == aoclsparse_fully_sorted)
    {
        if(nnza >= 10)
        {
            aoclsparse_int total_blks = 0;
            aoclsparse_int nRowsblk   = aoclsparse_opt_blksize(
                A->m, A->nnz, A->base, A->csr_mat.csr_row_ptr, A->csr_mat.csr_col_ptr, &total_blks);
            if(nRowsblk != 0)
            {
                const aoclsparse_int blk_width  = 8;
                aoclsparse::blk_csr *mat_blkcsr = nullptr;
                mat_blkcsr                      = &(A->blk_csr_mat);

                try
                {
                    mat_blkcsr->blk_row_ptr = new aoclsparse_int[m + 1];
                    mat_blkcsr->blk_col_ptr = new aoclsparse_int[nnz];
                    mat_blkcsr->blk_val
                        = ::operator new((nnz + nRowsblk * blk_width) * sizeof(double));
                    mat_blkcsr->masks = new uint8_t[total_blks * nRowsblk];
                }
                catch(std::bad_alloc &)
                {
                    delete[] mat_blkcsr->blk_row_ptr;
                    delete[] mat_blkcsr->blk_col_ptr;
                    ::operator delete(mat_blkcsr->blk_val);
                    delete[] mat_blkcsr->masks;
                    return aoclsparse_status_memory_error;
                }
                aoclsparse_csr2blkcsr(A->m,
                                      A->n,
                                      A->nnz,
                                      A->csr_mat.csr_row_ptr,
                                      A->csr_mat.csr_col_ptr,
                                      (double *)A->csr_mat.csr_val,
                                      mat_blkcsr->blk_row_ptr,
                                      mat_blkcsr->blk_col_ptr,
                                      (double *)mat_blkcsr->blk_val,
                                      mat_blkcsr->masks,
                                      nRowsblk,
                                      A->base);
                A->blk_csr_mat.nRowsblk = nRowsblk;
                A->blk_optimized        = true;
                A->mat_type             = aoclsparse_csr_mat;
            }
        }
    }
    if(!A->blk_optimized && ell_width <= 30)
    { // ToDo: why this cutoff?
        if(fill_ratio < -0.1)
        {
            A->mat_type = aoclsparse_csr_mat_br4; // CSR-BT-AVX2
        }
        else if((fill_ratio > -0.1) && (fill_ratio < 0.1))
        {
            A->mat_type = aoclsparse_ellt_csr_hyb_mat; // ELL-CSR-HYB
        }
        else
        {
            if(ell_csr_g_ew_l_10 < ell_csr_g_ew_g_10)
            {
                A->mat_type = aoclsparse_ellt_csr_hyb_mat; // ELL-CSR-HYB
            }
            else
            {
                A->mat_type = aoclsparse_csr_mat_br4; // CSR-BT-AVX2
            }
        }
    }
    if(A->mat_type == aoclsparse_ellt_csr_hyb_mat)
    {
        aoclsparse_int           ell_width;
        aoclsparse_int           ell_m;
        aoclsparse::ell_csr_hyb *ell_csr_hyb_mat = &(A->ell_csr_hyb_mat);
        // get the ell_width
        aoclsparse_csr2ellthyb_width(A->m, A->nnz, A->csr_mat.csr_row_ptr, &ell_m, &ell_width);
        if(ell_width == 0)
        {
            A->mat_type = aoclsparse_csr_mat;
            return aoclsparse_status_success;
        }
        try
        {
            ell_csr_hyb_mat->ell_col_ind    = new aoclsparse_int[ell_width * A->m];
            ell_csr_hyb_mat->csr_row_id_map = new aoclsparse_int[A->m - ell_m];
            ell_csr_hyb_mat->ell_val = ::operator new(data_size[A->val_type] * ell_width * A->m);
        }
        catch(std::bad_alloc &)
        {
            delete[] ell_csr_hyb_mat->ell_col_ind;
            delete[] ell_csr_hyb_mat->csr_row_id_map;
            ::operator delete(ell_csr_hyb_mat->ell_val);
            return aoclsparse_status_memory_error;
        }
        // convert to hybrid ELLT-CSR format
        if(A->val_type == aoclsparse_dmat)
        {
            aoclsparse_dcsr2ellthyb(A->m,
                                    A->base,
                                    &ell_m,
                                    A->csr_mat.csr_row_ptr,
                                    A->csr_mat.csr_col_ptr,
                                    (double *)A->csr_mat.csr_val,
                                    NULL,
                                    ell_csr_hyb_mat->csr_row_id_map,
                                    ell_csr_hyb_mat->ell_col_ind,
                                    (double *)ell_csr_hyb_mat->ell_val,
                                    ell_width);
        }
        else if(A->val_type == aoclsparse_smat)
        {
            aoclsparse_scsr2ellthyb(A->m,
                                    A->base,
                                    &ell_m,
                                    A->csr_mat.csr_row_ptr,
                                    A->csr_mat.csr_col_ptr,
                                    (float *)A->csr_mat.csr_val,
                                    NULL,
                                    ell_csr_hyb_mat->csr_row_id_map,
                                    ell_csr_hyb_mat->ell_col_ind,
                                    (float *)ell_csr_hyb_mat->ell_val,
                                    ell_width);
        }
        // set appropriate members of "A"
        ell_csr_hyb_mat->ell_width   = ell_width;
        ell_csr_hyb_mat->ell_m       = ell_m;
        ell_csr_hyb_mat->csr_col_ptr = A->csr_mat.csr_col_ptr;
        ell_csr_hyb_mat->csr_val     = A->csr_mat.csr_val;
    }
    else if((A->mat_type == aoclsparse_csr_mat_br4) && (A->val_type == aoclsparse_dmat))
    { // vectorized csr blocked format for AVX2
        try
        {
            std::vector<aoclsparse_int> row_ptr(A->m);

            aoclsparse_int   row_nnz;
            aoclsparse::csr *csr_mat_br4 = &(A->csr_mat_br4);
            // populate row_nnz
            aoclsparse_int i;
            aoclsparse_int j;
            aoclsparse_int tnnz = 0;
            try
            {
                csr_mat_br4->csr_row_ptr = new aoclsparse_int[A->m + 1];
            }
            catch(std::bad_alloc &)
            {
                return aoclsparse_status_memory_error;
            }
            csr_mat_br4->csr_row_ptr[0] = A->base;
            for(i = 0; i < A->m; i += 4)
            {
                aoclsparse_int m1, m2;
                if((A->m - i) < 4)
                    break;
                m1 = (std::max)((A->csr_mat.csr_row_ptr[i + 1] - A->csr_mat.csr_row_ptr[i]),
                                (A->csr_mat.csr_row_ptr[i + 2] - A->csr_mat.csr_row_ptr[i + 1]));
                m2 = (std::max)((A->csr_mat.csr_row_ptr[i + 3] - A->csr_mat.csr_row_ptr[i + 2]),
                                (A->csr_mat.csr_row_ptr[i + 4] - A->csr_mat.csr_row_ptr[i + 3]));
                row_nnz    = (std::max)(m1, m2);
                row_ptr[i] = row_ptr[i + 1] = row_ptr[i + 2] = row_ptr[i + 3] = row_nnz;
                csr_mat_br4->csr_row_ptr[i + 1] = csr_mat_br4->csr_row_ptr[i] + row_nnz;
                csr_mat_br4->csr_row_ptr[i + 2] = csr_mat_br4->csr_row_ptr[i + 1] + row_nnz;
                csr_mat_br4->csr_row_ptr[i + 3] = csr_mat_br4->csr_row_ptr[i + 2] + row_nnz;
                csr_mat_br4->csr_row_ptr[i + 4] = csr_mat_br4->csr_row_ptr[i + 3] + row_nnz;
                tnnz += 4 * row_nnz;
            }
            for(j = i; j < A->m; ++j)
            {
                row_nnz = A->csr_mat.csr_row_ptr[j + 1] - A->csr_mat.csr_row_ptr[j];
                tnnz += row_nnz;
                row_ptr[j]                      = row_nnz;
                csr_mat_br4->csr_row_ptr[j + 1] = csr_mat_br4->csr_row_ptr[j] + row_nnz;
            }
            // create the new csr matrix and convert to the csr-avx2 format
            try
            {
                csr_mat_br4->csr_col_ptr = new aoclsparse_int[tnnz];
                csr_mat_br4->csr_val     = ::operator new(sizeof(double) * tnnz);
            }
            catch(std::bad_alloc &)
            {
                delete[] csr_mat_br4->csr_col_ptr;
                ::operator delete(csr_mat_br4->csr_val);
                return aoclsparse_status_memory_error;
            }
            aoclsparse_int tc = 0; // count of nonzeros
            // keeps track of the last column index accessed during the conversion
            aoclsparse_int last_col_idx = 0;
            for(i = 0; i < A->m; ++i)
            {
                aoclsparse_int nz   = A->csr_mat.csr_row_ptr[i + 1] - A->csr_mat.csr_row_ptr[i];
                aoclsparse_int ridx = A->csr_mat.csr_row_ptr[i];
                for(j = 0; j < nz; ++j)
                {
                    ((double *)csr_mat_br4->csr_val)[tc]
                        = ((double *)A->csr_mat.csr_val)[ridx - A->base + j];
                    csr_mat_br4->csr_col_ptr[tc] = A->csr_mat.csr_col_ptr[ridx - A->base + j];
                    last_col_idx                 = csr_mat_br4->csr_col_ptr[tc];
                    tc++;
                }
                if(nz < row_ptr[i])
                { // ToDo -- can remove the if condition
                    for(j = nz; j < row_ptr[i]; ++j)
                    {
                        csr_mat_br4->csr_col_ptr[tc]         = last_col_idx;
                        ((double *)csr_mat_br4->csr_val)[tc] = static_cast<double>(0);
                        tc++;
                    }
                }
            }
            tc                   = 0;
            aoclsparse_int *cptr = (aoclsparse_int *)csr_mat_br4->csr_col_ptr;
            double         *vptr = (double *)csr_mat_br4->csr_val;
            for(i = 0; i < A->m; i += 4)
            {
                cptr               = csr_mat_br4->csr_col_ptr + tc;
                vptr               = (double *)csr_mat_br4->csr_val + tc;
                aoclsparse_int nnz = row_ptr[i];
                if((A->m - i) < 4)
                    break;
                // transponse the chunk into an auxiliary buffer
                std::vector<double>         bufval(nnz * 4);
                std::vector<aoclsparse_int> bufidx(nnz * 4);
                aoclsparse_int              ii, jj;
                for(ii = 0; ii < nnz; ++ii)
                {
                    for(jj = 0; jj < 4; ++jj)
                    {
                        bufval[jj + ii * 4] = vptr[ii + jj * nnz];
                        bufidx[jj + ii * 4] = cptr[ii + jj * nnz];
                    }
                }
                memcpy(vptr, bufval.data(), sizeof(double) * nnz * 4);
                memcpy(cptr, bufidx.data(), sizeof(aoclsparse_int) * nnz * 4);
                tc += nnz * 4;
            }
            // set appropriate members of "A"
            //        A->csr_mat_br4.csr_row_ptr = row_ptr;
        }
        catch(std::bad_alloc &)
        {
            // row_ptr, bufval, bufidx memory allocation fail
            return aoclsparse_status_memory_error;
        }
    }
    A->optimized = true;
    return aoclsparse_status_success;
}
/*
    SYMGS optimize API allocates working buffers
*/
aoclsparse_status aoclsparse_optimize_symgs(aoclsparse_matrix A)
{
    aoclsparse_status ret = aoclsparse_status_success;
    void             *r, *q;
    //If already allocated, then no need to reallocate. So return. Need to happen only once in the beginning
    if(A->symgs_info.sgs_ready == true)
    {
        return ret;
    }

    try
    {
        r = ::operator new(data_size[A->val_type] * A->m);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    try
    {
        q = ::operator new(data_size[A->val_type] * A->m);
    }
    catch(std::bad_alloc &)
    {
        ::operator delete(r);
        return aoclsparse_status_memory_error;
    }
    A->symgs_info.r = r;
    A->symgs_info.q = q;
    //turn this flag on to indicate necessary allocations for SGS have been done
    A->symgs_info.sgs_ready = true;
    return ret;
}
/*
    ILU optimize API allocates working buffers and also
    memory for the precondtioned csr value buffer
*/
aoclsparse_status aoclsparse_optimize_ilu(aoclsparse_matrix A)
{
    aoclsparse_status ret     = aoclsparse_status_success;
    void             *ilu_val = nullptr;
    //If already allocated, then no need to reallocate. So return. Need to happen only once in the beginning
    if(A->ilu_info.ilu_ready == true)
    {
        return ret;
    }
    A->ilu_info.ilu_fact_type = aoclsparse_ilu0;
    try
    {
        A->ilu_info.lu_diag_ptr = new aoclsparse_int[A->m];
        ilu_val                 = ::operator new(data_size[A->val_type] * A->nnz);
    }
    catch(std::bad_alloc &)
    {
        delete[] A->ilu_info.lu_diag_ptr;
        ::operator delete(ilu_val);
        return aoclsparse_status_memory_error;
    }
    memcpy(ilu_val, A->csr_mat.csr_val, (data_size[A->val_type] * A->nnz));
    A->ilu_info.precond_csr_val = ilu_val;
    //turn this flag on to indicate necessary allocations for ILU have been done
    A->ilu_info.ilu_ready = true;
    return ret;
}
aoclsparse_status aoclsparse_optimize(aoclsparse_matrix A)
{
    aoclsparse_status         ret = aoclsparse_status_success;
    aoclsparse_optimize_data *optd;
    bool                      optimized;
    // Validations
    if(!A)
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check sizes
    if((A->m < 0) || (A->n < 0) || (A->nnz < 0))
    {
        return aoclsparse_status_invalid_size;
    }

    // Investigate potential optimized copies
    // if memory usage is unrestricted
    if(A->mem_policy == aoclsparse_memory_usage_unrestricted)
    {
        switch(A->val_type)
        {
        case aoclsparse_dmat:
            ret = aoclsparse_matrix_transform<double>(A);
            break;
        case aoclsparse_smat:
            ret = aoclsparse_matrix_transform<float>(A);
            break;
        case aoclsparse_cmat:
            ret = aoclsparse_matrix_transform<std::complex<float>>(A);
            break;
        case aoclsparse_zmat:
            ret = aoclsparse_matrix_transform<std::complex<double>>(A);
            break;
        }
        if(ret != aoclsparse_status_success)
            return ret;
    }

    // Optimize TCSR matrix
    // Check if the matrix is valid
    // Creates idiag ptr for lower and iurow ptr for upper triangualr matrix
    if(A->input_format == aoclsparse_tcsr_mat)
    {
        if(A->opt_csr_ready) // check if already optimized
            return aoclsparse_status_success;
        else
        {
            switch(A->val_type)
            {
            case aoclsparse_dmat:
                ret = aoclsparse_tcsr_optimize<double>(A);
                break;
            case aoclsparse_smat:
                ret = aoclsparse_tcsr_optimize<float>(A);
                break;
            case aoclsparse_cmat:
                ret = aoclsparse_tcsr_optimize<std::complex<float>>(A);
                break;
            case aoclsparse_zmat:
                ret = aoclsparse_tcsr_optimize<std::complex<double>>(A);
                break;
            }
        }
        return ret;
    }
    // Check CSR matrix is populated, it not return an error. ToDo: need to handle CSC / COO cases later
    if((A->csr_mat.csr_row_ptr == nullptr) || (A->csr_mat.csr_col_ptr == nullptr)
       || (A->csr_mat.csr_val == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    // ToDo: any other validations
    // Go through the linked list of hinted actions to decide which action to take
    optimized               = true;
    optd                    = A->optim_data;
    aoclsparse_int mv_count = 0, ilu_count = 0, sgs_count = 0;
    aoclsparse_int other_count = 0, sum = 0;
    while(optd)
    {
        optimized              = optimized && optd->action_optimized;
        optd->action_optimized = true;
        // Increment the actions counter that are implemented
        if((optd->act == aoclsparse_action_mv || optd->act == aoclsparse_action_dotmv)
           && optd->trans == aoclsparse_operation_none
           && optd->type == aoclsparse_matrix_type_general && A->val_type == aoclsparse_dmat
           && optd->nop > 0)
            mv_count++;
        else if(optd->act == aoclsparse_action_ilu0 && optd->nop > 0)
            ilu_count++;
        else if(optd->act == aoclsparse_action_symgs && optd->nop > 0)
            sgs_count++;
        else
            other_count++;
        sum++;
        optd = optd->next;
    }
    // all actions in the list were already optimized for
    if(optimized)
        return aoclsparse_status_success;
    // If 'other' hints have been passed, simply optimize the matrix by creating a clean CSR / clean CSC
    if(other_count || sum == 0)
    {
        if(A->opt_csr_ready || A->opt_csc_ready)
            ret = aoclsparse_status_success;
        else
        {
            switch(A->val_type)
            {
            case aoclsparse_dmat:
                ret = aoclsparse_csr_csc_optimize<double>(A);
                break;
            case aoclsparse_smat:
                ret = aoclsparse_csr_csc_optimize<float>(A);
                break;
            case aoclsparse_cmat:
                ret = aoclsparse_csr_csc_optimize<std::complex<float>>(A);
                break;
            case aoclsparse_zmat:
                ret = aoclsparse_csr_csc_optimize<std::complex<double>>(A);
                break;
            }
        }
    }
    else if(mv_count - sum >= 0)
    {
        // Only MV hints with nontransposed and double precision matrix has been passed
        // Optimize for MV
        ret = aoclsparse_optimize_mv(A);
    }
    else if(ilu_count)
    {
        // Only ilu hints have been passed
        ret = aoclsparse_optimize_ilu(A);
    }
    else if(sgs_count)
    {
        // Symgs Optimize/work-array allocations
        ret = aoclsparse_optimize_symgs(A);
    }
    return ret;
}

aoclsparse_status aoclsparse_set_hint(aoclsparse_matrix          mat,
                                      aoclsparse_hinted_action   act,
                                      aoclsparse_operation       trans,
                                      const aoclsparse_mat_descr descr,
                                      aoclsparse_int             expected_no_of_calls,
                                      aoclsparse_int             kid = -1)
{
    // Check matrix and descriptor
    if((mat == nullptr) || (descr == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    // Check index base
    if((descr->base != aoclsparse_index_base_zero) && (descr->base != aoclsparse_index_base_one))
    {
        return aoclsparse_status_invalid_value;
    }
    // Check for base index incompatibility
    // There is an issue that zero-based indexing is defined in two separate places and
    // can lead to ambiguity, we check that both are consistent.
    if(mat->base != descr->base)
    {
        return aoclsparse_status_invalid_value;
    }
    if((trans != aoclsparse_operation_none) && (trans != aoclsparse_operation_transpose)
       && (trans != aoclsparse_operation_conjugate_transpose))
    {
        return aoclsparse_status_invalid_value;
    }
    if((descr->fill_mode != aoclsparse_fill_mode_lower)
       && (descr->fill_mode != aoclsparse_fill_mode_upper))
    {
        return aoclsparse_status_invalid_value;
    }
    if((descr->diag_type != aoclsparse_diag_type_non_unit)
       && (descr->diag_type != aoclsparse_diag_type_unit)
       && (descr->diag_type != aoclsparse_diag_type_zero))
    {
        return aoclsparse_status_invalid_value;
    }
    if((descr->type != aoclsparse_matrix_type_general)
       && (descr->type != aoclsparse_matrix_type_symmetric)
       && (descr->type != aoclsparse_matrix_type_triangular)
       && (descr->type != aoclsparse_matrix_type_hermitian))
    {
        return aoclsparse_status_invalid_value;
    }
    if(expected_no_of_calls < 0 || (expected_no_of_calls == 0 && kid == -1))
    {
        return aoclsparse_status_invalid_value;
    }

    // Check if action is valid or not
    if((act <= aoclsparse_action_none) || (act >= aoclsparse_action_max))
    {
        return aoclsparse_status_invalid_operation;
    }
    // Add the hint at the start of the linked list
    return aoclsparse_add_hint(
        mat->optim_data, act, descr, trans, mat->val_type, expected_no_of_calls, kid);
}

aoclsparse_status aoclsparse_set_mv_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls)
{
    return aoclsparse_set_hint(mat, aoclsparse_action_mv, trans, descr, expected_no_of_calls);
}

aoclsparse_status aoclsparse_set_mv_hint_kid(aoclsparse_matrix          mat,
                                             aoclsparse_operation       trans,
                                             const aoclsparse_mat_descr descr,
                                             aoclsparse_int             expected_no_of_calls,
                                             aoclsparse_int             kid)
{
    return aoclsparse_set_hint(mat, aoclsparse_action_mv, trans, descr, expected_no_of_calls, kid);
}

aoclsparse_status aoclsparse_set_dotmv_hint(aoclsparse_matrix          mat,
                                            aoclsparse_operation       trans,
                                            const aoclsparse_mat_descr descr,
                                            aoclsparse_int             expected_no_of_calls)
{
    return aoclsparse_set_hint(mat, aoclsparse_action_dotmv, trans, descr, expected_no_of_calls);
}
aoclsparse_status aoclsparse_set_sv_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls)
{
    return aoclsparse_set_hint(mat, aoclsparse_action_sv, trans, descr, expected_no_of_calls);
}
aoclsparse_status aoclsparse_set_mm_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls)
{
    return aoclsparse_set_hint(mat, aoclsparse_action_mm, trans, descr, expected_no_of_calls);
}
aoclsparse_status aoclsparse_set_2m_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         aoclsparse_int             expected_no_of_calls)
{
    return aoclsparse_set_hint(mat, aoclsparse_action_2m, trans, descr, expected_no_of_calls);
}
aoclsparse_status aoclsparse_set_lu_smoother_hint(aoclsparse_matrix          mat,
                                                  aoclsparse_operation       trans,
                                                  const aoclsparse_mat_descr descr,
                                                  aoclsparse_int             expected_no_of_calls)
{
    return aoclsparse_set_hint(mat, aoclsparse_action_ilu0, trans, descr, expected_no_of_calls);
}
aoclsparse_status aoclsparse_set_sm_hint(aoclsparse_matrix          mat,
                                         aoclsparse_operation       trans,
                                         const aoclsparse_mat_descr descr,
                                         const aoclsparse_order     order,
                                         aoclsparse_int             expected_no_of_calls)
{
    aoclsparse_hinted_action act;
    if((order != aoclsparse_order_row) && (order != aoclsparse_order_column))
    {
        return aoclsparse_status_invalid_value;
    }
    if(order == aoclsparse_order_row)
    {
        act = aoclsparse_action_sm_row;
    }
    else
    {
        act = aoclsparse_action_sm_col;
    }
    return aoclsparse_set_hint(mat, act, trans, descr, expected_no_of_calls);
}
aoclsparse_status aoclsparse_set_symgs_hint(aoclsparse_matrix          mat,
                                            aoclsparse_operation       trans,
                                            const aoclsparse_mat_descr descr,
                                            aoclsparse_int             expected_no_of_calls)
{
    return aoclsparse_set_hint(mat, aoclsparse_action_symgs, trans, descr, expected_no_of_calls);
}
aoclsparse_status aoclsparse_set_sorv_hint(aoclsparse_matrix          mat,
                                           const aoclsparse_mat_descr descr,
                                           const aoclsparse_sor_type  type,
                                           const aoclsparse_int       expected_no_of_calls)
{
    aoclsparse_hinted_action act;
    if((type != aoclsparse_sor_forward) && (type != aoclsparse_sor_backward)
       && (type != aoclsparse_sor_symmetric))
    {
        return aoclsparse_status_invalid_value;
    }
    if(type == aoclsparse_sor_forward)
    {
        act = aoclsparse_action_sorv_forward;
    }
    else if(type == aoclsparse_sor_backward)
    {
        act = aoclsparse_action_sorv_backward;
    }
    else
    {
        act = aoclsparse_action_sorv_symm;
    }
    return aoclsparse_set_hint(mat, act, aoclsparse_operation_none, descr, expected_no_of_calls);
}

aoclsparse_status aoclsparse_set_memory_hint(aoclsparse_matrix             mat,
                                             const aoclsparse_memory_usage policy)
{
    if(mat == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if((policy != aoclsparse_memory_usage_minimal)
       && (policy != aoclsparse_memory_usage_unrestricted))
    {
        return aoclsparse_status_invalid_value;
    }
    mat->mem_policy = policy;
    return aoclsparse_status_success;
}
