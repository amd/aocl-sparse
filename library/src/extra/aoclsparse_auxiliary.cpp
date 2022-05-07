/* ************************************************************************
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

#include "aoclsparse_descr.h"
#include "aoclsparse.h"
#include "aoclsparse_mat_structures.h"
#include "aoclsparse_types.h"
#include <string>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief Get aoclsparse version
 *******************************************************************************/
char * aoclsparse_get_version()
{
    std::string ver = "AOCL-Sparse " + std::to_string(AOCLSPARSE_VERSION_MAJOR) + "."
	+ std::to_string(AOCLSPARSE_VERSION_MINOR) + "."
	+ std::to_string(AOCLSPARSE_VERSION_PATCH) ;

    char* version = strcpy(new char[ver.length() + 1], ver.c_str());
    return version;
}

/********************************************************************************
 * \brief aoclsparse_create_mat_descr_t is a structure holding the aoclsparse matrix
 * descriptor. It must be initialized using aoclsparse_create_mat_descr()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using aoclsparse_destroy_mat_descr().
 *******************************************************************************/
aoclsparse_status aoclsparse_create_mat_descr(aoclsparse_mat_descr* descr)
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
        catch(const aoclsparse_status& status)
        {
            return status;
        }
        return aoclsparse_status_success;
    }
}

/********************************************************************************
 * \brief copy matrix descriptor
 *******************************************************************************/
aoclsparse_status aoclsparse_copy_mat_descr(aoclsparse_mat_descr dest, const aoclsparse_mat_descr src)
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
    catch(const aoclsparse_status& status)
    {
        return status;
    }
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief Set the index base of the matrix descriptor.
 *******************************************************************************/
aoclsparse_status aoclsparse_set_mat_index_base(aoclsparse_mat_descr descr, aoclsparse_index_base base)
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
 * \brief aoclsparse_create_scsr sets the sparse matrix in the csr format.
 ********************************************************************************/
aoclsparse_status aoclsparse_create_scsr(aoclsparse_matrix &mat,
                    aoclsparse_index_base   base,
                    aoclsparse_int          M,
                    aoclsparse_int          N,
                    aoclsparse_int          csr_nnz,
                    aoclsparse_int*         csr_row_ptr,
                    aoclsparse_int*         csr_col_ptr,
                    float*                   csr_val)
{
    mat = new _aoclsparse_matrix;
    mat->m = M;
    mat->n = N;
    mat->nnz = csr_nnz;
    mat->base = base;
    mat->optimized = false;
    mat->val_type = aoclsparse_smat;
    mat->csr_mat.csr_row_ptr = csr_row_ptr;
    mat->csr_mat.csr_col_ptr = csr_col_ptr;
    mat->csr_mat.csr_val = csr_val;

    mat->hint_id = aoclsparse_none;   //initialize to 0 during hint operation

    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief aoclsparse_create_dcsr sets the sparse matrix in the csr format.
 ********************************************************************************/
aoclsparse_status aoclsparse_create_dcsr(aoclsparse_matrix &mat,
                    aoclsparse_index_base   base,
                    aoclsparse_int          M,
                    aoclsparse_int          N,
                    aoclsparse_int          csr_nnz,
                    aoclsparse_int*         csr_row_ptr,
                    aoclsparse_int*         csr_col_ptr,
                    double*                   csr_val)
{
    mat = new _aoclsparse_matrix;
    mat->m = M;
    mat->n = N;
    mat->nnz = csr_nnz;
    mat->base = base;
    mat->optimized = false;
    mat->val_type = aoclsparse_dmat;
    mat->csr_mat.csr_row_ptr = csr_row_ptr;
    mat->csr_mat.csr_col_ptr = csr_col_ptr;
    mat->csr_mat.csr_val = csr_val;

    mat->hint_id = aoclsparse_none;   //initialize to 0 during hint operation

    return aoclsparse_status_success;
}


/********************************************************************************
 * \brief aoclsparse_create_ell sets the sparse matrix in the ell format.
 * This function can be called after the matrix "mat" is initialized.
 ********************************************************************************/
aoclsparse_status aoclsparse_create_ell(aoclsparse_matrix mat,
		            aoclsparse_int          ell_width,
                    aoclsparse_int*         ell_col_ind,
                    void*                   ell_val)
{
    mat->ell_mat.ell_width = ell_width;
    mat->ell_mat.ell_col_ind = ell_col_ind;
    mat->ell_mat.ell_val = ell_val;
    return aoclsparse_status_success;
}


/********************************************************************************
 * \brief aoclsparse_create_ell_csr_hyb sets the sparse matrix in the hybrid format.
 * This function can be called after the matrix "mat" is initialized (which also
 * initializes the csr format by default)
 ********************************************************************************/
aoclsparse_status aoclsparse_create_ell_csr_hyb(aoclsparse_matrix mat,
                    aoclsparse_int          ell_width,
                    aoclsparse_int          ell_m,
                    aoclsparse_int*         ell_col_ind,
                    aoclsparse_int*         csr_row_id_map,
                    void*                   ell_val)
{
    mat->ell_csr_hyb_mat.ell_width = ell_width;
    mat->ell_csr_hyb_mat.ell_m = ell_m;
    mat->ell_csr_hyb_mat.ell_col_ind = ell_col_ind;
    mat->ell_csr_hyb_mat.ell_val = ell_val;
    mat->ell_csr_hyb_mat.csr_row_id_map = csr_row_id_map;
    mat->ell_csr_hyb_mat.csr_col_ptr = mat->csr_mat.csr_col_ptr;
    mat->ell_csr_hyb_mat.csr_val = mat->csr_mat.csr_val;

    return aoclsparse_status_success;
}


/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the aoclsparse csr matrix.
 * Use this routine to export the contents of this straucture
 ********************************************************************************/
aoclsparse_status aoclsparse_export_mat_csr(aoclsparse_matrix &csr,
	aoclsparse_index_base   *base,
	aoclsparse_int          *M,
	aoclsparse_int          *N,
	aoclsparse_int          *csr_nnz,
	aoclsparse_int*         *csr_row_ptr,
	aoclsparse_int*         *csr_col_ind,
	void*                   *csr_val)
{
    *M = csr->m ;
    *N = csr->n ;
    *csr_nnz = csr->nnz ;
    *csr_row_ptr = csr->csr_mat.csr_row_ptr ;
    *csr_col_ind = csr->csr_mat.csr_col_ptr ;
    *csr_val = csr->csr_mat.csr_val ;
    *base = aoclsparse_index_base_zero;
    return aoclsparse_status_success;
}


/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the sparse matrix A in CSR,
 * Ellpack, Diagonal and other hybrid formats. The working buffers allocated in SPMV's
 * optimize phase needs to be deallocated
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy_mv(aoclsparse_matrix A)
{

    if (A->mat_type == aoclsparse_ellt_csr_hyb_mat)     // // ELL-CSR-HYB
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
    else if (A->mat_type == aoclsparse_csr_mat_br4) // vectorized csr blocked format for AVX2
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
        ilu_info = NULL;
    }
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the sparse matrix A.
 * It must be deallocated using aoclsparse_destroy() by carefully looking for all the
 * allocations as part of different Sparse routines and their corresponding
 * Hint/Optimize functions.
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy(aoclsparse_matrix A)
{
    aoclsparse_status ret = aoclsparse_status_success;
    aoclsparse_int mv_hint, trsv_hint, mm_hint, twom_hint, ilu_hint;

    mv_hint = A->hint_id & aoclsparse_spmv;
    trsv_hint = (A->hint_id & aoclsparse_trsv) >> 1;
    mm_hint = (A->hint_id & aoclsparse_mm) >> 2;
    twom_hint = (A->hint_id & aoclsparse_2m) >> 3;
    ilu_hint = (A->hint_id & aoclsparse_ilu) >> 4;

    if(mv_hint)
    {
        //delete SPMV data structures allocated in hint/optimize
        ret = aoclsparse_destroy_mv(A);
    }
    if(mm_hint)
    {
        //delete Dense - Sparse Matrix Mult data structures allocated in hint/optimize
        //To Do
    }
    if(trsv_hint)
    {
        //delete triangular solve data structures allocated in hint/optimize
        //To Do
    }
    if(twom_hint)
    {
        //delete Sparse - Sparse Matrix Mult data structures allocated in hint/optimize
        ret = aoclsparse_destroy_2m(A);
    }
    if(ilu_hint)
    {
        //delete ILU data structures allocated in hint/optimize
        ret = aoclsparse_destroy_ilu(&(A->ilu_info));
    }

    if(A != NULL)
    {
        delete A;
        A = NULL;
    }
    return ret;

}

#ifdef __cplusplus
}
#endif
