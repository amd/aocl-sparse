/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
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
#include "aoclsparse_descr.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_types.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_optimize_data.hpp"

#include <cstring>
#include <string>

#define STRINGIFY(x) _STRINGIFY(x)
#define _STRINGIFY(x) #x

static const char aoclsparse_version[]
    = "AOCL-Sparse " STRINGIFY(AOCLSPARSE_VERSION_MAJOR) "." STRINGIFY(AOCLSPARSE_VERSION_MINOR) "." STRINGIFY(
        AOCLSPARSE_VERSION_PATCH) " Build " STRINGIFY(AOCL_SPARSE_BUILD_DATE);

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief Get aoclsparse version
 *******************************************************************************/
const char *aoclsparse_get_version()
{
    return aoclsparse_version;
}

/********************************************************************************
 * \brief aoclsparse_create_mat_descr_t is a structure holding the aoclsparse matrix
 * descriptor. It must be initialized using aoclsparse_create_mat_descr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using aoclsparse_destroy_mat_descr().
 *******************************************************************************/
aoclsparse_status aoclsparse_create_mat_descr(aoclsparse_mat_descr *descr)
{
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            *descr = new _aoclsparse_mat_descr;
        }
        catch(const aoclsparse_status &status)
        {
            return status;
        }
        return aoclsparse_status_success;
    }
}

/********************************************************************************
 * \brief copy matrix descriptor
 *******************************************************************************/
aoclsparse_status aoclsparse_copy_mat_descr(aoclsparse_mat_descr       dest,
                                            const aoclsparse_mat_descr src)
{
    if(dest == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    else if(src == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }

    dest->type      = src->type;
    dest->fill_mode = src->fill_mode;
    dest->diag_type = src->diag_type;
    dest->base      = src->base;

    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy_mat_descr(aoclsparse_mat_descr descr)
{
    // Destruct
    try
    {
        delete descr;
    }
    catch(const aoclsparse_status &status)
    {
        return status;
    }
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief Set the index base of the matrix descriptor.
 *******************************************************************************/
aoclsparse_status aoclsparse_set_mat_index_base(aoclsparse_mat_descr  descr,
                                                aoclsparse_index_base base)
{
    // Check if descriptor is valid
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(base != aoclsparse_index_base_zero && base != aoclsparse_index_base_one)
    {
        return aoclsparse_status_invalid_value;
    }
    descr->base = base;
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief Returns the index base of the matrix descriptor.
 *******************************************************************************/
aoclsparse_index_base aoclsparse_get_mat_index_base(const aoclsparse_mat_descr descr)
{
    // If descriptor is invalid, default index base is returned
    if(descr == nullptr)
    {
        return aoclsparse_index_base_zero;
    }
    return descr->base;
}

/********************************************************************************
 * \brief Set the matrix type of the matrix descriptor.
 *******************************************************************************/
aoclsparse_status aoclsparse_set_mat_type(aoclsparse_mat_descr descr, aoclsparse_matrix_type type)
{
    // Check if descriptor is valid
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(type != aoclsparse_matrix_type_general && type != aoclsparse_matrix_type_symmetric
       && type != aoclsparse_matrix_type_hermitian && type != aoclsparse_matrix_type_triangular)
    {
        return aoclsparse_status_invalid_value;
    }
    descr->type = type;
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief Returns the matrix type of the matrix descriptor.
 *******************************************************************************/
aoclsparse_matrix_type aoclsparse_get_mat_type(const aoclsparse_mat_descr descr)
{
    // If descriptor is invalid, default matrix type is returned
    if(descr == nullptr)
    {
        return aoclsparse_matrix_type_general;
    }
    return descr->type;
}

aoclsparse_status aoclsparse_set_mat_fill_mode(aoclsparse_mat_descr descr,
                                               aoclsparse_fill_mode fill_mode)
{
    // Check if descriptor is valid
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(fill_mode != aoclsparse_fill_mode_lower && fill_mode != aoclsparse_fill_mode_upper)
    {
        return aoclsparse_status_invalid_value;
    }
    descr->fill_mode = fill_mode;
    return aoclsparse_status_success;
}

aoclsparse_fill_mode aoclsparse_get_mat_fill_mode(const aoclsparse_mat_descr descr)
{
    // If descriptor is invalid, default fill mode is returned
    if(descr == nullptr)
    {
        return aoclsparse_fill_mode_lower;
    }
    return descr->fill_mode;
}

aoclsparse_status aoclsparse_set_mat_diag_type(aoclsparse_mat_descr descr,
                                               aoclsparse_diag_type diag_type)
{
    // Check if descriptor is valid
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(diag_type != aoclsparse_diag_type_unit && diag_type != aoclsparse_diag_type_non_unit)
    {
        return aoclsparse_status_invalid_value;
    }
    descr->diag_type = diag_type;
    return aoclsparse_status_success;
}

aoclsparse_diag_type aoclsparse_get_mat_diag_type(const aoclsparse_mat_descr descr)
{
    // If descriptor is invalid, default diagonal type is returned
    if(descr == nullptr)
    {
        return aoclsparse_diag_type_non_unit;
    }
    return descr->diag_type;
}

/********************************************************************************
 * \brief aoclsparse_create_?csr sets the sparse matrix in the CSR format for
 * the appropriate data type (float, double, float complex, double complex).
 ********************************************************************************/
aoclsparse_status aoclsparse_create_scsr(aoclsparse_matrix    &mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *row_ptr,
                                         aoclsparse_int       *col_idx,
                                         float                *val)
{
    return aoclsparse_create_csr_t(mat, base, M, N, nnz, row_ptr, col_idx, val);
}

aoclsparse_status aoclsparse_create_dcsr(aoclsparse_matrix    &mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *row_ptr,
                                         aoclsparse_int       *col_idx,
                                         double               *val)
{
    return aoclsparse_create_csr_t(mat, base, M, N, nnz, row_ptr, col_idx, val);
}

aoclsparse_status aoclsparse_create_ccsr(aoclsparse_matrix        &mat,
                                         aoclsparse_index_base     base,
                                         aoclsparse_int            M,
                                         aoclsparse_int            N,
                                         aoclsparse_int            nnz,
                                         aoclsparse_int           *row_ptr,
                                         aoclsparse_int           *col_idx,
                                         aoclsparse_float_complex *val)
{
    return aoclsparse_create_csr_t(mat, base, M, N, nnz, row_ptr, col_idx, val);
}

aoclsparse_status aoclsparse_create_zcsr(aoclsparse_matrix         &mat,
                                         aoclsparse_index_base      base,
                                         aoclsparse_int             M,
                                         aoclsparse_int             N,
                                         aoclsparse_int             nnz,
                                         aoclsparse_int            *row_ptr,
                                         aoclsparse_int            *col_idx,
                                         aoclsparse_double_complex *val)
{
    return aoclsparse_create_csr_t(mat, base, M, N, nnz, row_ptr, col_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_create_ell sets the sparse matrix in the ell format.
 * This function can be called after the matrix "mat" is initialized.
 ********************************************************************************/
aoclsparse_status aoclsparse_create_ell(aoclsparse_matrix mat,
                                        aoclsparse_int    ell_width,
                                        aoclsparse_int   *ell_col_ind,
                                        void             *ell_val)
{
    mat->ell_mat.ell_width   = ell_width;
    mat->ell_mat.ell_col_ind = ell_col_ind;
    mat->ell_mat.ell_val     = ell_val;
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief aoclsparse_create_ell_csr_hyb sets the sparse matrix in the hybrid format.
 * This function can be called after the matrix "mat" is initialized (which also
 * initializes the csr format by default)
 ********************************************************************************/
aoclsparse_status aoclsparse_create_ell_csr_hyb(aoclsparse_matrix mat,
                                                aoclsparse_int    ell_width,
                                                aoclsparse_int    ell_m,
                                                aoclsparse_int   *ell_col_ind,
                                                aoclsparse_int   *csr_row_id_map,
                                                void             *ell_val)
{
    mat->ell_csr_hyb_mat.ell_width      = ell_width;
    mat->ell_csr_hyb_mat.ell_m          = ell_m;
    mat->ell_csr_hyb_mat.ell_col_ind    = ell_col_ind;
    mat->ell_csr_hyb_mat.ell_val        = ell_val;
    mat->ell_csr_hyb_mat.csr_row_id_map = csr_row_id_map;
    mat->ell_csr_hyb_mat.csr_col_ptr    = mat->csr_mat.csr_col_ptr;
    mat->ell_csr_hyb_mat.csr_val        = mat->csr_mat.csr_val;

    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_create_scoo(aoclsparse_matrix          &mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         float                      *val)
{
    return aoclsparse_create_coo(mat, base, M, N, nnz, row_ind, col_ind, val);
}

aoclsparse_status aoclsparse_create_dcoo(aoclsparse_matrix          &mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         double                     *val)
{

    return aoclsparse_create_coo(mat, base, M, N, nnz, row_ind, col_ind, val);
}

aoclsparse_status aoclsparse_create_ccoo(aoclsparse_matrix          &mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         aoclsparse_float_complex   *val)
{
    return aoclsparse_create_coo(mat, base, M, N, nnz, row_ind, col_ind, val);
}

aoclsparse_status aoclsparse_create_zcoo(aoclsparse_matrix          &mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         aoclsparse_double_complex  *val)
{

    return aoclsparse_create_coo(mat, base, M, N, nnz, row_ind, col_ind, val);
}

/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the aoclsparse csr matrix.
 * Use this routine to export the contents of this straucture
 ********************************************************************************/
aoclsparse_status aoclsparse_export_mat_csr(aoclsparse_matrix     &csr,
                                            aoclsparse_index_base *base,
                                            aoclsparse_int        *M,
                                            aoclsparse_int        *N,
                                            aoclsparse_int        *csr_nnz,
                                            aoclsparse_int       **csr_row_ptr,
                                            aoclsparse_int       **csr_col_ind,
                                            void                 **csr_val)
{
    if(csr->input_format != aoclsparse_csr_mat)
    {
        return aoclsparse_status_invalid_value;
    }
    *M           = csr->m;
    *N           = csr->n;
    *csr_nnz     = csr->nnz;
    *csr_row_ptr = csr->csr_mat.csr_row_ptr;
    *csr_col_ind = csr->csr_mat.csr_col_ptr;
    *csr_val     = csr->csr_mat.csr_val;
    *base        = aoclsparse_index_base_zero;
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the sparse matrix A in CSR,
 * Ellpack, Diagonal and other hybrid formats. The working buffers allocated in SPMV's
 * optimize phase needs to be deallocated
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy_mv(aoclsparse_matrix A)
{

    if(A->mat_type == aoclsparse_ellt_csr_hyb_mat) // // ELL-CSR-HYB
    {
        aoclsparse_ell_csr_hyb ell_csr_hyb_mat = &(A->ell_csr_hyb_mat);

        if(ell_csr_hyb_mat->ell_col_ind != NULL)
        {
            free(ell_csr_hyb_mat->ell_col_ind);
            ell_csr_hyb_mat->ell_col_ind = NULL;
        }
        if(ell_csr_hyb_mat->ell_val != NULL)
        {
            free(ell_csr_hyb_mat->ell_val);
            ell_csr_hyb_mat->ell_val = NULL;
        }
        if(ell_csr_hyb_mat->csr_row_id_map != NULL)
        {
            free(ell_csr_hyb_mat->csr_row_id_map);
            ell_csr_hyb_mat->csr_row_id_map = NULL;
        }
    }
    else if(A->mat_type == aoclsparse_csr_mat_br4) // vectorized csr blocked format for AVX2
    {
        aoclsparse_csr csr_mat_br4 = &(A->csr_mat_br4);

        if(csr_mat_br4->csr_row_ptr != NULL)
        {
            free(csr_mat_br4->csr_row_ptr);
            csr_mat_br4->csr_row_ptr = NULL;
        }
        if(csr_mat_br4->csr_col_ptr != NULL)
        {
            free(csr_mat_br4->csr_col_ptr);
            csr_mat_br4->csr_col_ptr = NULL;
        }
        if(csr_mat_br4->csr_val != NULL)
        {
            free(csr_mat_br4->csr_val);
            csr_mat_br4->csr_val = NULL;
        }
    }

    return aoclsparse_status_success;
}
/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the sparse matrix A.
 * The working buffers allocated in CSR2M's nnz_count, finalize and final computation phases
 * needs to be deallocated.
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy_2m(aoclsparse_matrix A)
{
    if(!A->csr_mat_is_users)
    {
        if(A->csr_mat.csr_row_ptr != NULL)
        {
            free(A->csr_mat.csr_row_ptr);
            A->csr_mat.csr_row_ptr = NULL;
        }
        if(A->csr_mat.csr_col_ptr != NULL)
        {
            free(A->csr_mat.csr_col_ptr);
            A->csr_mat.csr_col_ptr = NULL;
        }
        if(A->csr_mat.csr_val != NULL)
        {
            free(A->csr_mat.csr_val);
            A->csr_mat.csr_val = NULL;
        }
    }
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief _aoclsparse_ilu is a structure holding the ILU related information
 * sparse matrix A. It must be deallocated using aoclsparse_destroy_ilu() which
 * looks for working buffers allocated in ILU's optimize phase
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy_ilu(_aoclsparse_ilu *ilu_info)
{
    if(ilu_info != NULL)
    {
        if(ilu_info->col_idx_mapper != NULL)
        {
            free(ilu_info->col_idx_mapper);
            ilu_info->col_idx_mapper = NULL;
        }
        if(ilu_info->lu_diag_ptr != NULL)
        {
            free(ilu_info->lu_diag_ptr);
            ilu_info->lu_diag_ptr = NULL;
        }
        if(ilu_info->precond_csr_val != NULL)
        {
            free(ilu_info->precond_csr_val);
            ilu_info->precond_csr_val = NULL;
        }
        ilu_info = NULL;
    }
    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_destroy_opt_csr(aoclsparse_matrix A)
{
    if(!A->opt_csr_is_users)
    {
        if(A->opt_csr_mat.csr_col_ptr)
            free(A->opt_csr_mat.csr_col_ptr);
        if(A->opt_csr_mat.csr_row_ptr)
            free(A->opt_csr_mat.csr_row_ptr);
        if(A->opt_csr_mat.csr_val)
            free(A->opt_csr_mat.csr_val);
    }
    if(A->idiag)
        free(A->idiag);
    if(A->iurow)
        free(A->iurow);

    if(A->blk_optimized)
    {
        if(A->csr_mat.blk_row_ptr)
            free(A->csr_mat.blk_row_ptr);
        if(A->csr_mat.blk_col_ptr)
            free(A->csr_mat.blk_col_ptr);
        if(A->csr_mat.blk_val)
            free(A->csr_mat.blk_val);
        if(A->csr_mat.masks)
            free(A->csr_mat.masks);
    }
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the sparse matrix A.
 * It must be deallocated using aoclsparse_destroy() by carefully looking for all the
 * allocations as part of different Sparse routines and their corresponding
 * Hint/Optimize functions.
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy(aoclsparse_matrix &A)
{
    aoclsparse_status ret = aoclsparse_status_success;

    if(A)
    {
        aoclsparse_optimize_destroy(A->optim_data);
        aoclsparse_destroy_opt_csr(A);
        aoclsparse_destroy_mv(A);
        aoclsparse_destroy_2m(A);
        aoclsparse_destroy_ilu(&(A->ilu_info));
        delete A;
        A = NULL;
    }
    return ret;
}

// This function returns 1 if the architecture supports AVX512, else 0
// ToDo: return type can be an enum covering various vector extensions like SSE, AVX2, AVX512
aoclsparse_int aoclsparse_get_vec_extn_context(void)
{
    aoclsparse_init_once();
    aoclsparse_context context;
    context.is_avx512 = sparse_global_context.is_avx512;
    if(context.is_avx512)
        return 1;
    return 0;
}

/********************************************************************************
 * \brief aoclsparse_create_scsc sets the sparse matrix in the CSC format for
 * float data type
 ********************************************************************************/
aoclsparse_status aoclsparse_create_scsc(aoclsparse_matrix    &mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *col_ptr,
                                         aoclsparse_int       *row_idx,
                                         float                *val)
{
    return aoclsparse_create_csc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_create_dcsc sets the sparse matrix in the CSC format for
 * double data type
 ********************************************************************************/
aoclsparse_status aoclsparse_create_dcsc(aoclsparse_matrix    &mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *col_ptr,
                                         aoclsparse_int       *row_idx,
                                         double               *val)
{
    return aoclsparse_create_csc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_create_dcsc sets the sparse matrix in the CSC format for
 * complex float data type
 ********************************************************************************/
aoclsparse_status aoclsparse_create_ccsc(aoclsparse_matrix        &mat,
                                         aoclsparse_index_base     base,
                                         aoclsparse_int            M,
                                         aoclsparse_int            N,
                                         aoclsparse_int            nnz,
                                         aoclsparse_int           *col_ptr,
                                         aoclsparse_int           *row_idx,
                                         aoclsparse_float_complex *val)
{
    return aoclsparse_create_csc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_create_dcsc sets the sparse matrix in the CSC format for
 * complex double data type
 ********************************************************************************/
aoclsparse_status aoclsparse_create_zcsc(aoclsparse_matrix         &mat,
                                         aoclsparse_index_base      base,
                                         aoclsparse_int             M,
                                         aoclsparse_int             N,
                                         aoclsparse_int             nnz,
                                         aoclsparse_int            *col_ptr,
                                         aoclsparse_int            *row_idx,
                                         aoclsparse_double_complex *val)
{
    return aoclsparse_create_csc(mat, base, M, N, nnz, col_ptr, row_idx, val);
}
#ifdef __cplusplus
}
#endif

void aoclsparse_init_mat(aoclsparse_matrix             A,
                         aoclsparse_index_base         base,
                         aoclsparse_int                M,
                         aoclsparse_int                N,
                         aoclsparse_int                nnz,
                         aoclsparse_matrix_format_type matrix_type)
{
    // Default values
    if(!A)
        return;

    A->m            = M;
    A->n            = N;
    A->nnz          = nnz;
    A->base         = base;
    A->input_format = matrix_type;
}

/********************************************************************************
 * \brief aoclsparse_create_csr_t sets the sparse matrix in the CSR format
 * for any data type
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_create_csr_t(aoclsparse_matrix    &mat,
                                          aoclsparse_index_base base,
                                          aoclsparse_int        M,
                                          aoclsparse_int        N,
                                          aoclsparse_int        nnz,
                                          aoclsparse_int       *row_ptr,
                                          aoclsparse_int       *col_idx,
                                          T                    *val)
{
    // Validate the input parameters
    if(M < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(N < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    if(row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(col_idx == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(val == nullptr)
        return aoclsparse_status_invalid_pointer;

    // check if the column indicies are within bounds
    for(aoclsparse_int i = 0; i < M; i++)
    {
        if(N == 0)
        {
            break;
        }
        for(aoclsparse_int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            if((col_idx[j] >= (N + base)) || (col_idx[j] < base))
                return aoclsparse_status_invalid_index_value;
        }
    }
    try
    {
        mat = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(mat, base, M, N, nnz, aoclsparse_csr_mat);
    mat->val_type            = get_data_type<T>();
    mat->mat_type            = aoclsparse_csr_mat;
    mat->csr_mat.csr_row_ptr = row_ptr;
    mat->csr_mat.csr_col_ptr = col_idx;
    mat->csr_mat.csr_val     = val;
    mat->csr_mat_is_users    = true;

    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief aoclsparse_create_csc sets the sparse matrix in the CSC format
 * for any data type
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_create_csc(aoclsparse_matrix    &mat,
                                        aoclsparse_index_base base,
                                        aoclsparse_int        M,
                                        aoclsparse_int        N,
                                        aoclsparse_int        nnz,
                                        aoclsparse_int       *col_ptr,
                                        aoclsparse_int       *row_idx,
                                        T                    *val)
{
    // Validate the input parameters
    if(M < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(N < 0)
    {
        return aoclsparse_status_invalid_size;
    }
    else if(nnz < 0)
    {
        return aoclsparse_status_invalid_size;
    }

    if(row_idx == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(col_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(val == nullptr)
        return aoclsparse_status_invalid_pointer;

    // check if the row indicies are within bounds
    for(aoclsparse_int i = 0; i < N; i++)
    {
        if(M == 0)
        {
            break;
        }
        for(aoclsparse_int j = col_ptr[i]; j < col_ptr[i + 1]; j++)
        {
            if((row_idx[j] >= (M + base)) || (row_idx[j] < base))
                return aoclsparse_status_invalid_index_value;
        }
    }
    try
    {
        mat = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(mat, base, M, N, nnz, aoclsparse_csc_mat);
    mat->val_type         = get_data_type<T>();
    mat->csr_mat_is_users = false;
    mat->csc_mat.col_ptr  = col_ptr;
    mat->csc_mat.row_idx  = row_idx;
    mat->csc_mat.val      = val;

    return aoclsparse_status_success;
}
template <typename T>
aoclsparse_status aoclsparse_create_coo(aoclsparse_matrix          &mat,
                                        const aoclsparse_index_base base,
                                        const aoclsparse_int        M,
                                        const aoclsparse_int        N,
                                        const aoclsparse_int        nnz,
                                        aoclsparse_int             *row_ind,
                                        aoclsparse_int             *col_ind,
                                        T                          *val)
{
    if(M < 0)
        return aoclsparse_status_invalid_size;

    else if(N < 0)
        return aoclsparse_status_invalid_size;

    else if(nnz < 0)
        return aoclsparse_status_invalid_size;

    if(row_ind == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(col_ind == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(val == nullptr)
        return aoclsparse_status_invalid_pointer;

    // check if codinates given is correct or not
    for(int i = 0; i < nnz; i++)
    {
        if(row_ind[i] < base || row_ind[i] >= (M + base))
            return aoclsparse_status_invalid_index_value;
        if(col_ind[i] < base || col_ind[i] >= (N + base))
            return aoclsparse_status_invalid_index_value;
    }

    try
    {
        mat = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(mat, base, M, N, nnz, aoclsparse_coo_mat);
    mat->val_type        = get_data_type<T>();
    mat->coo_mat.row_ind = row_ind;
    mat->coo_mat.col_ind = col_ind;
    mat->coo_mat.val     = val;

    return aoclsparse_status_success;
}
