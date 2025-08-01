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

#include "aoclsparse_context.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_types.h"
#include "aoclsparse.hpp"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_mat_structures.hpp"

#include <cstring>
#include <map>
#include <string>

static const char *aoclsparse_version = AOCLSPARSE_VERSION_STRING;

/*
   Get the size of the data type based on the matrix data type
   Enum values of aoclsparse_matrix_data_type forms the array indices
*/
const size_t data_size[] = {sizeof(double),
                            sizeof(float),
                            sizeof(aoclsparse_float_complex),
                            sizeof(aoclsparse_double_complex)};
#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief Set into the thread local hint the ISA path preference from
 * the environmental variable AOCL_ENABLE_INSTRUCTIONS
 *******************************************************************************/
aoclsparse_status aoclsparse_enable_instructions(const char isa_preference[])
{
    using namespace std::string_literals;
    using namespace aoclsparse;

    std::string isa{isa_preference}, next_isa;

    if(isa != "")
    {
        transform(isa.begin(), isa.end(), isa.begin(), ::toupper);

        if(isa == "ENV"s)
        {
            // Special case to re-read the enviromental variable and set
            // the ISA path preference via AOCL_ENABLE_INSTRUCTIONS
            isa = env_get_var("AOCL_ENABLE_INSTRUCTIONS", "");
            if(isa != "")
                transform(isa.begin(), isa.end(), isa.begin(), ::toupper);
        }

        if(isa == ""s)
        {
            // inform no ISA preference has been set
            tl_isa_hint.set_isa_hint(context_isa_t::UNSET);
            return aoclsparse_status_success;
        }
        if(isa == "AVX512"s)
        {
            if(context::get_context()->supports<context_isa_t::AVX512F>())
            {
                tl_isa_hint.set_isa_hint(context_isa_t::AVX512F);
                return aoclsparse_status_success;
            }
            next_isa = "AVX2";
        }
        if(isa == "AVX2"s || next_isa == "AVX2"s)
        {
            if(context::get_context()->supports<context_isa_t::AVX2>())
            {
                tl_isa_hint.set_isa_hint(context_isa_t::AVX2);
                return aoclsparse_status_success;
            }
            next_isa = "GENERIC";
        }
        if(isa == "GENERIC"s || next_isa == "GENERIC")
        {
            tl_isa_hint.set_isa_hint(context_isa_t::GENERIC);
            return aoclsparse_status_success;
        }
        else
        {
            return aoclsparse_status_invalid_value;
        }
    }

    tl_isa_hint.set_isa_hint(context_isa_t::UNSET);
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief Gets the ISA path preference, thread local ISA hint, the number of
 * threads and the architecture from library internal objects.
 *******************************************************************************/
aoclsparse_status aoclsparse_debug_get(char            isa_preference[],
                                       aoclsparse_int *num_threads,
                                       char            tl_isa_preference[],
                                       bool           *is_isa_updated,
                                       char            arch[])
{
    using namespace aoclsparse;

    std::map<context_isa_t, std::string> context_map;

    context_map[context_isa_t::AVX2]    = "AVX2";
    context_map[context_isa_t::AVX512F] = "AVX512";
    context_map[context_isa_t::GENERIC] = "GENERIC";

    std::map<archs, std::string> arch_map;

    arch_map[archs::ZEN]     = "ZEN";
    arch_map[archs::ZEN2]    = "ZEN2";
    arch_map[archs::ZEN3]    = "ZEN3";
    arch_map[archs::ZEN4]    = "ZEN4";
    arch_map[archs::ZEN5]    = "ZEN5";
    arch_map[archs::UNKNOWN] = "UNKNOWN";

    context_isa_t global_isa, tl_isa;
    archs         architecture;

    global_isa   = context::get_context()->get_isa_hint();
    tl_isa       = tl_isa_hint.get_isa_hint();
    architecture = context::get_context()->get_archs();

    std::string val;

    // Resize the string for longer arch and context names
    val.resize(20);

    val = context_map[global_isa];

    val.copy(isa_preference, val.length());
    isa_preference[val.length()] = '\0';

    val = context_map[tl_isa];

    val.copy(tl_isa_preference, val.length());
    tl_isa_preference[val.length()] = '\0';

    val = arch_map[architecture];

    val.copy(arch, val.length());
    arch[val.length()] = '\0';

    *num_threads = context::get_context()->get_num_threads();

    *is_isa_updated = tl_isa_hint.is_isa_updated();

    return aoclsparse_status_success;
}

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
        catch(std::bad_alloc &)
        {
            return aoclsparse_status_memory_error;
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
    if(descr != NULL)
    {
        delete descr;
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
    if(diag_type != aoclsparse_diag_type_unit && diag_type != aoclsparse_diag_type_non_unit
       && diag_type != aoclsparse_diag_type_zero)
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
 * \brief aoclsparse_create_tcsr sets the sparse matrix in the TCSR format for
 * the appropriate data type (float, double, float complex, double complex).
 ********************************************************************************/
aoclsparse_status aoclsparse_create_stcsr(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr_L,
                                          aoclsparse_int             *row_ptr_U,
                                          aoclsparse_int             *col_idx_L,
                                          aoclsparse_int             *col_idx_U,
                                          float                      *val_L,
                                          float                      *val_U)
{
    return aoclsparse_create_tcsr_t(
        mat, base, M, N, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);
}

aoclsparse_status aoclsparse_create_dtcsr(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr_L,
                                          aoclsparse_int             *row_ptr_U,
                                          aoclsparse_int             *col_idx_L,
                                          aoclsparse_int             *col_idx_U,
                                          double                     *val_L,
                                          double                     *val_U)
{
    return aoclsparse_create_tcsr_t(
        mat, base, M, N, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);
}

aoclsparse_status aoclsparse_create_ctcsr(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr_L,
                                          aoclsparse_int             *row_ptr_U,
                                          aoclsparse_int             *col_idx_L,
                                          aoclsparse_int             *col_idx_U,
                                          aoclsparse_float_complex   *val_L,
                                          aoclsparse_float_complex   *val_U)
{
    return aoclsparse_create_tcsr_t(
        mat, base, M, N, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);
}

aoclsparse_status aoclsparse_create_ztcsr(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr_L,
                                          aoclsparse_int             *row_ptr_U,
                                          aoclsparse_int             *col_idx_L,
                                          aoclsparse_int             *col_idx_U,
                                          aoclsparse_double_complex  *val_L,
                                          aoclsparse_double_complex  *val_U)
{
    return aoclsparse_create_tcsr_t(
        mat, base, M, N, nnz, row_ptr_L, row_ptr_U, col_idx_L, col_idx_U, val_L, val_U);
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
    if(!mat || mat->mats.empty() || !ell_col_ind || !csr_row_id_map || !ell_val)
        return aoclsparse_status_invalid_pointer;

    aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(mat->mats[0]);
    if(!csr_mat)
        return aoclsparse_status_not_implemented;

    aoclsparse::ell_csr_hyb *ell_csr_hyb_mat = nullptr;
    try
    {
        ell_csr_hyb_mat = new aoclsparse::ell_csr_hyb(mat->m,
                                                      mat->n,
                                                      mat->nnz,
                                                      csr_mat->base,
                                                      mat->val_type,
                                                      ell_width,
                                                      ell_m,
                                                      ell_col_ind,
                                                      ell_val,
                                                      csr_row_id_map,
                                                      csr_mat->ind,
                                                      csr_mat->val);
        mat->mats.push_back(ell_csr_hyb_mat);
    }
    catch(const std::bad_alloc &)
    {
        if(ell_csr_hyb_mat)
            delete ell_csr_hyb_mat;
        return aoclsparse_status_memory_error;
    }
    return aoclsparse_status_success;
}

aoclsparse_status aoclsparse_create_scoo(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         float                      *val)
{
    return aoclsparse_create_coo_t(mat, base, M, N, nnz, row_ind, col_ind, val);
}

aoclsparse_status aoclsparse_create_dcoo(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         double                     *val)
{

    return aoclsparse_create_coo_t(mat, base, M, N, nnz, row_ind, col_ind, val);
}

aoclsparse_status aoclsparse_create_ccoo(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         aoclsparse_float_complex   *val)
{
    return aoclsparse_create_coo_t(mat, base, M, N, nnz, row_ind, col_ind, val);
}

aoclsparse_status aoclsparse_create_zcoo(aoclsparse_matrix          *mat,
                                         const aoclsparse_index_base base,
                                         const aoclsparse_int        M,
                                         const aoclsparse_int        N,
                                         const aoclsparse_int        nnz,
                                         aoclsparse_int             *row_ind,
                                         aoclsparse_int             *col_ind,
                                         aoclsparse_double_complex  *val)
{

    return aoclsparse_create_coo_t(mat, base, M, N, nnz, row_ind, col_ind, val);
}

/********************************************************************************
 * \brief aoclsparse_?_update_values updates the value in sparse matrix for all coordinates
 * with the appropriate data type (float, double, float complex,
 * double complex).
 ********************************************************************************/
aoclsparse_status aoclsparse_supdate_values(aoclsparse_matrix A, aoclsparse_int len, float *val)
{
    return aoclsparse_update_values_t(A, len, val);
}

aoclsparse_status aoclsparse_dupdate_values(aoclsparse_matrix A, aoclsparse_int len, double *val)
{
    return aoclsparse_update_values_t(A, len, val);
}
aoclsparse_status aoclsparse_cupdate_values(aoclsparse_matrix         A,
                                            aoclsparse_int            len,
                                            aoclsparse_float_complex *val)
{
    return aoclsparse_update_values_t(A, len, val);
}

aoclsparse_status aoclsparse_zupdate_values(aoclsparse_matrix          A,
                                            aoclsparse_int             len,
                                            aoclsparse_double_complex *val)
{
    return aoclsparse_update_values_t(A, len, val);
}

aoclsparse_status aoclsparse_export_scsr(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **row_ptr,
                                         aoclsparse_int        **col_ind,
                                         float                 **val)
{
    return aoclsparse_export_csr_t(mat, base, m, n, nnz, row_ptr, col_ind, val);
}

aoclsparse_status aoclsparse_export_dcsr(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **row_ptr,
                                         aoclsparse_int        **col_ind,
                                         double                **val)
{
    return aoclsparse_export_csr_t(mat, base, m, n, nnz, row_ptr, col_ind, val);
}

aoclsparse_status aoclsparse_export_ccsr(const aoclsparse_matrix    mat,
                                         aoclsparse_index_base     *base,
                                         aoclsparse_int            *m,
                                         aoclsparse_int            *n,
                                         aoclsparse_int            *nnz,
                                         aoclsparse_int           **row_ptr,
                                         aoclsparse_int           **col_ind,
                                         aoclsparse_float_complex **val)
{
    return aoclsparse_export_csr_t(mat, base, m, n, nnz, row_ptr, col_ind, val);
}

aoclsparse_status aoclsparse_export_zcsr(const aoclsparse_matrix     mat,
                                         aoclsparse_index_base      *base,
                                         aoclsparse_int             *m,
                                         aoclsparse_int             *n,
                                         aoclsparse_int             *nnz,
                                         aoclsparse_int            **row_ptr,
                                         aoclsparse_int            **col_ind,
                                         aoclsparse_double_complex **val)
{
    return aoclsparse_export_csr_t(mat, base, m, n, nnz, row_ptr, col_ind, val);
}

aoclsparse_status aoclsparse_export_scoo(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **row_ptr,
                                         aoclsparse_int        **col_ptr,
                                         float                 **val)
{
    return aoclsparse_export_coo_t(mat, base, m, n, nnz, row_ptr, col_ptr, val);
}

aoclsparse_status aoclsparse_export_dcoo(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **row_ptr,
                                         aoclsparse_int        **col_ptr,
                                         double                **val)
{
    return aoclsparse_export_coo_t(mat, base, m, n, nnz, row_ptr, col_ptr, val);
}

aoclsparse_status aoclsparse_export_ccoo(const aoclsparse_matrix    mat,
                                         aoclsparse_index_base     *base,
                                         aoclsparse_int            *m,
                                         aoclsparse_int            *n,
                                         aoclsparse_int            *nnz,
                                         aoclsparse_int           **row_ptr,
                                         aoclsparse_int           **col_ptr,
                                         aoclsparse_float_complex **val)
{
    return aoclsparse_export_coo_t(mat, base, m, n, nnz, row_ptr, col_ptr, val);
}

aoclsparse_status aoclsparse_export_zcoo(const aoclsparse_matrix     mat,
                                         aoclsparse_index_base      *base,
                                         aoclsparse_int             *m,
                                         aoclsparse_int             *n,
                                         aoclsparse_int             *nnz,
                                         aoclsparse_int            **row_ptr,
                                         aoclsparse_int            **col_ptr,
                                         aoclsparse_double_complex **val)
{
    return aoclsparse_export_coo_t(mat, base, m, n, nnz, row_ptr, col_ptr, val);
}

/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the sparse matrix A.
 * It must be deallocated using aoclsparse_destroy() by carefully looking for all the
 * allocations as part of different Sparse routines and their corresponding
 * Hint/Optimize functions.
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy(aoclsparse_matrix *A)
{
    aoclsparse_status ret = aoclsparse_status_success;

    if(A && *A)
    {
        aoclsparse_optimize_destroy((*A)->optim_data);
        aoclsparse_destroy_ilu(&((*A)->ilu_info));
        aoclsparse_destroy_symgs(&((*A)->symgs_info));
        aoclsparse_destroy_mats(*A);
        delete *A;
        *A = NULL;
    }
    return ret;
}

/********************************************************************************
 * \brief \P{aoclsparse_?_set_value} sets the value in sparse matrix for a particular
 * coordinate with the appropriate data type (float, double, float complex,
 * double complex).
 ********************************************************************************/
aoclsparse_status aoclsparse_sset_value(aoclsparse_matrix A,
                                        aoclsparse_int    row_idx,
                                        aoclsparse_int    col_idx,
                                        float             val)
{
    return aoclsparse_set_value_t(A, row_idx, col_idx, val);
}

aoclsparse_status aoclsparse_dset_value(aoclsparse_matrix A,
                                        aoclsparse_int    row_idx,
                                        aoclsparse_int    col_idx,
                                        double            val)
{
    return aoclsparse_set_value_t(A, row_idx, col_idx, val);
}
aoclsparse_status aoclsparse_cset_value(aoclsparse_matrix        A,
                                        aoclsparse_int           row_idx,
                                        aoclsparse_int           col_idx,
                                        aoclsparse_float_complex val)
{
    return aoclsparse_set_value_t(A, row_idx, col_idx, val);
}

aoclsparse_status aoclsparse_zset_value(aoclsparse_matrix         A,
                                        aoclsparse_int            row_idx,
                                        aoclsparse_int            col_idx,
                                        aoclsparse_double_complex val)
{
    return aoclsparse_set_value_t(A, row_idx, col_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_create_scsc sets the sparse matrix in the CSC format for
 * float data type
 ********************************************************************************/
aoclsparse_status aoclsparse_create_scsc(aoclsparse_matrix    *mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *col_ptr,
                                         aoclsparse_int       *row_idx,
                                         float                *val)
{
    return aoclsparse_create_csc_t(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_create_dcsc sets the sparse matrix in the CSC format for
 * double data type
 ********************************************************************************/
aoclsparse_status aoclsparse_create_dcsc(aoclsparse_matrix    *mat,
                                         aoclsparse_index_base base,
                                         aoclsparse_int        M,
                                         aoclsparse_int        N,
                                         aoclsparse_int        nnz,
                                         aoclsparse_int       *col_ptr,
                                         aoclsparse_int       *row_idx,
                                         double               *val)
{
    return aoclsparse_create_csc_t(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_create_dcsc sets the sparse matrix in the CSC format for
 * complex float data type
 ********************************************************************************/
aoclsparse_status aoclsparse_create_ccsc(aoclsparse_matrix        *mat,
                                         aoclsparse_index_base     base,
                                         aoclsparse_int            M,
                                         aoclsparse_int            N,
                                         aoclsparse_int            nnz,
                                         aoclsparse_int           *col_ptr,
                                         aoclsparse_int           *row_idx,
                                         aoclsparse_float_complex *val)
{
    return aoclsparse_create_csc_t(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_create_dcsc sets the sparse matrix in the CSC format for
 * complex double data type
 ********************************************************************************/
aoclsparse_status aoclsparse_create_zcsc(aoclsparse_matrix         *mat,
                                         aoclsparse_index_base      base,
                                         aoclsparse_int             M,
                                         aoclsparse_int             N,
                                         aoclsparse_int             nnz,
                                         aoclsparse_int            *col_ptr,
                                         aoclsparse_int            *row_idx,
                                         aoclsparse_double_complex *val)
{
    return aoclsparse_create_csc_t(mat, base, M, N, nnz, col_ptr, row_idx, val);
}

/********************************************************************************
 * \brief aoclsparse_copy creates deep copy for sparse matrix
 ********************************************************************************/
aoclsparse_status aoclsparse_copy(const aoclsparse_matrix                     src,
                                  [[maybe_unused]] const aoclsparse_mat_descr descr,
                                  aoclsparse_matrix                          *dest)
{
    aoclsparse_status status = aoclsparse_status_success;

    if(src == nullptr || dest == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }
    if(src->m < 0 || src->n < 0 || src->nnz < 0)
        return aoclsparse_status_invalid_size;
    /* descr is not used in this release so doesn't need to be checked
    if(descr == nullptr)
    {
        return aoclsparse_status_invalid_pointer;
    }*/
    if(src == *dest)
    {
        return aoclsparse_status_invalid_pointer;
    }
    try
    {
        *dest = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(*dest, src->m, src->n, src->nnz, src->input_format);
    (*dest)->val_type = src->val_type;

    if(src->val_type == aoclsparse_smat)
    {
        status = aoclsparse_copy_mat<float>(src, *dest);
    }
    else if(src->val_type == aoclsparse_dmat)
    {
        status = aoclsparse_copy_mat<double>(src, *dest);
    }
    else if(src->val_type == aoclsparse_cmat)
    {
        status = aoclsparse_copy_mat<aoclsparse_float_complex>(src, *dest);
    }
    else if(src->val_type == aoclsparse_zmat)
    {
        status = aoclsparse_copy_mat<aoclsparse_double_complex>(src, *dest);
    }
    else
    {
        status = aoclsparse_status_wrong_type;
    }
    if(status != aoclsparse_status_success)
    {
        // free matrix memory
        aoclsparse_destroy(dest);
        *dest = nullptr;
    }
    return status;
}

/********************************************************************************
 * \brief aoclsparse_order perform ordering of index array in the given matrix
 * of CSR/CSC format.
 ********************************************************************************/
aoclsparse_status aoclsparse_order_mat(aoclsparse_matrix mat)
{
    aoclsparse_status status = aoclsparse_status_success;
    if(!mat)
        return aoclsparse_status_invalid_pointer;

    if((mat->m < 0) || (mat->n < 0) || (mat->nnz < 0))
        return aoclsparse_status_invalid_value;

    // Ordering is implemented only for CSR and CSC matrix
    if((mat->input_format != aoclsparse_csr_mat) && (mat->input_format != aoclsparse_csc_mat))
        return aoclsparse_status_not_implemented;

    // empty matrix --> nothing to sort
    if((mat->m == 0) || (mat->n == 0) || (mat->nnz == 0))
        return aoclsparse_status_success;

    if(mat->val_type == aoclsparse_smat)
    {
        status = aoclsparse_sort_mat<float>(mat);
    }
    else if(mat->val_type == aoclsparse_dmat)
    {
        status = aoclsparse_sort_mat<double>(mat);
    }
    else if(mat->val_type == aoclsparse_cmat)
    {
        status = aoclsparse_sort_mat<aoclsparse_float_complex>(mat);
    }
    else if(mat->val_type == aoclsparse_zmat)
    {
        status = aoclsparse_sort_mat<aoclsparse_double_complex>(mat);
    }
    else
    {
        status = aoclsparse_status_wrong_type;
    }
    return status;
}

aoclsparse_status aoclsparse_export_scsc(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **col_ptr,
                                         aoclsparse_int        **row_ind,
                                         float                 **val)
{
    return aoclsparse_export_csc_t(mat, base, m, n, nnz, col_ptr, row_ind, val);
}

aoclsparse_status aoclsparse_export_dcsc(const aoclsparse_matrix mat,
                                         aoclsparse_index_base  *base,
                                         aoclsparse_int         *m,
                                         aoclsparse_int         *n,
                                         aoclsparse_int         *nnz,
                                         aoclsparse_int        **col_ptr,
                                         aoclsparse_int        **row_ind,
                                         double                **val)
{
    return aoclsparse_export_csc_t(mat, base, m, n, nnz, col_ptr, row_ind, val);
}

aoclsparse_status aoclsparse_export_ccsc(const aoclsparse_matrix    mat,
                                         aoclsparse_index_base     *base,
                                         aoclsparse_int            *m,
                                         aoclsparse_int            *n,
                                         aoclsparse_int            *nnz,
                                         aoclsparse_int           **col_ptr,
                                         aoclsparse_int           **row_ind,
                                         aoclsparse_float_complex **val)
{
    return aoclsparse_export_csc_t(mat, base, m, n, nnz, col_ptr, row_ind, val);
}

aoclsparse_status aoclsparse_export_zcsc(const aoclsparse_matrix     mat,
                                         aoclsparse_index_base      *base,
                                         aoclsparse_int             *m,
                                         aoclsparse_int             *n,
                                         aoclsparse_int             *nnz,
                                         aoclsparse_int            **col_ptr,
                                         aoclsparse_int            **row_ind,
                                         aoclsparse_double_complex **val)
{
    return aoclsparse_export_csc_t(mat, base, m, n, nnz, col_ptr, row_ind, val);
}

aoclsparse_int aoclsparse_debug_dispatcher(const char                  dispatcher[],
                                           aoclsparse_matrix_data_type dt,
                                           aoclsparse_int              kid)
{
    std::string dispatch = dispatcher;

    if(dt == aoclsparse_dmat)
    {
        return aoclsparse::test::dispatcher<double>(dispatch, kid);
    }
    else if(dt == aoclsparse_smat)
    {
        return aoclsparse::test::dispatcher<float>(dispatch, kid);
    }
    else if(dt == aoclsparse_zmat)
    {
        return aoclsparse::test::dispatcher<std::complex<double>>(dispatch, kid);
    }
    else if(dt == aoclsparse_cmat)
    {
        return aoclsparse::test::dispatcher<std::complex<float>>(dispatch, kid);
    }

    return -1000;
}

#ifdef __cplusplus
}
#endif

/********************************************************************************
 * \brief _aoclsparse_ilu is a structure holding the ILU related information
 * sparse matrix A. It must be deallocated using aoclsparse_destroy_ilu() which
 * looks for working buffers allocated in ILU's optimize phase
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy_ilu(_aoclsparse_ilu *ilu_info)
{
    if(ilu_info != NULL)
    {
        if(ilu_info->lu_diag_ptr != NULL)
        {
            delete[] ilu_info->lu_diag_ptr;
            ilu_info->lu_diag_ptr = NULL;
        }
        if(ilu_info->precond_csr_val != NULL)
        {
            ::operator delete(ilu_info->precond_csr_val);
            ilu_info->precond_csr_val = NULL;
        }
    }
    return aoclsparse_status_success;
}
/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the sparse matrix A.
 * The working buffers of SYMGS needs to be deallocated.
 *******************************************************************************/
aoclsparse_status aoclsparse_destroy_symgs(_aoclsparse_symgs *sgs_info)
{
    if(sgs_info != NULL)
    {
        if(sgs_info->r != NULL)
        {
            ::operator delete(sgs_info->r);
            sgs_info->r = NULL;
        }
        if(sgs_info->q != NULL)
        {
            ::operator delete(sgs_info->q);
            sgs_info->q = NULL;
        }
    }
    return aoclsparse_status_success;
}

// Deallocate all the matrix representations in the mats vector of the given matrix
aoclsparse_status aoclsparse_destroy_mats(aoclsparse_matrix A)
{
    for(auto &mat : A->mats)
    {
        if(mat != nullptr)
        {
            delete mat;
        }
    }
    return aoclsparse_status_success;
}

// TODO: Can be removed, information stored in the base_mtx.
void aoclsparse_init_mat(aoclsparse_matrix             A,
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
    A->input_format = matrix_type;
}

/********************************************************************************
 * \brief aoclsparse_create_csc_t sets the sparse matrix in the CSC format
 * \brief aoclsparse_create_csc sets the sparse matrix in the CSC format
 * for any data type
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_create_csc_t(aoclsparse_matrix    *mat,
                                          aoclsparse_index_base base,
                                          aoclsparse_int        M,
                                          aoclsparse_int        N,
                                          aoclsparse_int        nnz,
                                          aoclsparse_int       *col_ptr,
                                          aoclsparse_int       *row_idx,
                                          T                    *val)
{
    aoclsparse_status status;
    if(!mat)
        return aoclsparse_status_invalid_pointer;
    *mat = nullptr;
    // Validate the input parameters
    aoclsparse_matrix_sort mat_sort;
    bool                   mat_fulldiag;
    if((status = aoclsparse_mat_check_internal(
            N, M, nnz, col_ptr, row_idx, val, shape_general, base, mat_sort, mat_fulldiag, nullptr))
       != aoclsparse_status_success)
    {
        return status;
    }
    aoclsparse::csr *csc_mat = nullptr;
    try
    {
        *mat    = new _aoclsparse_matrix;
        csc_mat = new aoclsparse::csr(
            M, N, nnz, aoclsparse_csc_mat, base, get_data_type<T>(), col_ptr, row_idx, val);
        (*mat)->mats.push_back(csc_mat);
    }
    catch(std::bad_alloc &)
    {
        if(csc_mat)
            delete csc_mat;
        if(*mat)
        {
            delete *mat;
            *mat = nullptr;
        }
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(*mat, M, N, nnz, aoclsparse_csc_mat);
    (*mat)->val_type = get_data_type<T>();
    // Assign the temporary CSCmatrix to the matrix structure
    (*mat)->sort     = mat_sort;
    (*mat)->fulldiag = mat_fulldiag;
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_create_coo_t(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ind,
                                          aoclsparse_int             *col_ind,
                                          T                          *val)
{
    if(!mat)
        return aoclsparse_status_invalid_pointer;
    *mat = nullptr;
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

    // check if coordinates given are within bounds or not
    for(int i = 0; i < nnz; i++)
    {
        if(row_ind[i] < base || row_ind[i] >= (M + base))
            return aoclsparse_status_invalid_index_value;
        if(col_ind[i] < base || col_ind[i] >= (N + base))
            return aoclsparse_status_invalid_index_value;
    }
    aoclsparse::coo *coo_mat = nullptr;
    try
    {
        *mat    = new _aoclsparse_matrix;
        coo_mat = new aoclsparse::coo(M, N, nnz, base, get_data_type<T>(), row_ind, col_ind, val);
        (*mat)->mats.push_back(coo_mat);
    }
    catch(std::bad_alloc &)
    {
        if(coo_mat)
            delete coo_mat;
        if(*mat)
        {
            delete *mat;
            *mat = nullptr;
        }
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(*mat, M, N, nnz, aoclsparse_coo_mat);
    (*mat)->val_type = get_data_type<T>();
    (*mat)->mat_type = aoclsparse_coo_mat;

    return aoclsparse_status_success;
}

/* Copy a CSR/CSC matrix
 * Possible exit: invalid size, invalid pointer, memory alloc
 */
template <typename T>
aoclsparse_status aoclsparse_copy_csr(const aoclsparse::csr *src, aoclsparse::csr *&dest)
{
    if(!src)
        return aoclsparse_status_invalid_pointer;
    // Get the pointer size based on the matrix type
    aoclsparse_int ptr_size = (src->mat_type == aoclsparse_csc_mat) ? (src->n) : (src->m);
    // Check for invalid sizes
    if((ptr_size < 0) || (src->nnz < 0))
        return aoclsparse_status_invalid_size;
    if((src->ind == nullptr) || (src->ptr == nullptr) || (src->val == nullptr))
        return aoclsparse_status_invalid_pointer;

    dest = nullptr;
    try
    {
        dest = new aoclsparse::csr(
            src->m, src->n, src->nnz, src->mat_type, src->base, src->val_type);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    memcpy(dest->ptr, src->ptr, ((ptr_size + 1) * sizeof(aoclsparse_int)));
    memcpy(dest->ind, src->ind, (src->nnz * sizeof(aoclsparse_int)));
    memcpy(dest->val, src->val, (src->nnz * sizeof(T)));
    return aoclsparse_status_success;
}

/* Copy a coo matrix
 * Possible exit: invalid size, invalid pointer, memory alloc
 */
template <typename T>
aoclsparse_status aoclsparse_copy_coo(const aoclsparse::coo *src, aoclsparse::coo *&dest)
{
    if(!src)
        return aoclsparse_status_invalid_pointer;
    if(src->nnz < 0)
        return aoclsparse_status_invalid_size;
    if((src->row_ind == nullptr) || (src->col_ind == nullptr) || (src->val == nullptr))
        return aoclsparse_status_invalid_pointer;
    dest = nullptr;
    try
    {
        dest = new aoclsparse::coo(src->m, src->n, src->nnz, src->base, src->val_type);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    // copy the matrix
    memcpy(dest->row_ind, src->row_ind, (src->nnz * sizeof(aoclsparse_int)));
    memcpy(dest->col_ind, src->col_ind, (src->nnz * sizeof(aoclsparse_int)));
    memcpy(dest->val, src->val, (src->nnz * sizeof(T)));
    return aoclsparse_status_success;
}

// Copy the internal matrix representation from src to dest for the supported formats.
template <typename T>
aoclsparse_status aoclsparse_copy_mat(const aoclsparse_matrix src, aoclsparse_matrix dest)
{
    if(!src || src->mats.empty() || !src->mats[0])
        return aoclsparse_status_invalid_pointer;

    aoclsparse_status     status   = aoclsparse_status_success;
    aoclsparse::base_mtx *dest_mat = nullptr;

    switch(src->mats[0]->mat_type)
    {
    case aoclsparse_csr_mat:
    case aoclsparse_csc_mat:
        status = aoclsparse_copy_csr<T>(dynamic_cast<aoclsparse::csr *>(src->mats[0]),
                                        reinterpret_cast<aoclsparse::csr *&>(dest_mat));
        break;
    case aoclsparse_coo_mat:
        status = aoclsparse_copy_coo<T>(dynamic_cast<aoclsparse::coo *>(src->mats[0]),
                                        reinterpret_cast<aoclsparse::coo *&>(dest_mat));
        break;
    default:
        return aoclsparse_status_invalid_value;
    }

    if(status == aoclsparse_status_success && dest_mat)
    {
        try
        {
            dest->mats.push_back(dest_mat);
        }
        catch(const std::bad_alloc &)
        {
            delete dest_mat;
            return aoclsparse_status_memory_error;
        }
    }
    else if(dest_mat)
    {
        delete dest_mat;
    }

    return status;
}

template <typename T>
aoclsparse_status aoclsparse_sort_mat(aoclsparse_matrix mat)
{
    std::vector<aoclsparse_int> temp_idx;
    std::vector<T>              temp_val;
    aoclsparse_status           status = aoclsparse_status_success;

    if(!mat || mat->mats.empty())
        return aoclsparse_status_invalid_pointer;

    aoclsparse::csr *src_mat = dynamic_cast<aoclsparse::csr *>(mat->mats[0]);
    if(!src_mat)
        return aoclsparse_status_not_implemented;
    if(!src_mat->ptr || !src_mat->ind || !src_mat->val)
        return aoclsparse_status_invalid_pointer;

    // copy the matrix
    try
    {
        temp_idx.assign(src_mat->ind, src_mat->ind + mat->nnz);
        temp_val.assign(static_cast<T *>(src_mat->val), static_cast<T *>(src_mat->val) + mat->nnz);
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    status = aoclsparse_sort_idx_val<T>(mat->m,
                                        mat->n,
                                        mat->nnz,
                                        src_mat->base,
                                        src_mat->ptr,
                                        temp_idx.data(),
                                        temp_val.data(),
                                        src_mat->base,
                                        src_mat->ind,
                                        static_cast<T *>(src_mat->val));

    return status;
}
/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the aoclsparse csr matrix.
 * Use this routine to export the contents of this straucture
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_export_csr_t(const aoclsparse_matrix mat,
                                          aoclsparse_index_base  *base,
                                          aoclsparse_int         *m,
                                          aoclsparse_int         *n,
                                          aoclsparse_int         *nnz,
                                          aoclsparse_int        **row_ptr,
                                          aoclsparse_int        **col_ind,
                                          T                     **val)
{
    if((mat == nullptr) || mat->mats.empty() || (base == nullptr) || (m == nullptr)
       || (n == nullptr) || (nnz == nullptr) || (row_ptr == nullptr) || (col_ind == nullptr)
       || (val == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    // check if data type of matrix is same as requested
    if(mat->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }

    aoclsparse::csr *csr_mat = nullptr;
    for(auto *mat_rep : mat->mats)
    {
        aoclsparse::csr *temp_mat = dynamic_cast<aoclsparse::csr *>(mat_rep);
        if(temp_mat && temp_mat->mat_type == aoclsparse_csr_mat && temp_mat->ptr && temp_mat->ind
           && temp_mat->val)
        {
            csr_mat = temp_mat; // Valid CSR
            if(temp_mat->is_optimized)
                break; // Found the optimized csr matrix
        }
    }
    if(!csr_mat)
        return aoclsparse_status_invalid_value;

    *row_ptr = csr_mat->ptr;
    *col_ind = csr_mat->ind;
    *val     = static_cast<T *>(csr_mat->val);
    *nnz     = csr_mat->ptr[mat->m] - csr_mat->base;
    *base    = csr_mat->base;

    *m = mat->m;
    *n = mat->n;
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the aoclsparse csc matrix.
 * Use this routine to export the csc contents of this straucture
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_export_csc_t(const aoclsparse_matrix mat,
                                          aoclsparse_index_base  *base,
                                          aoclsparse_int         *m,
                                          aoclsparse_int         *n,
                                          aoclsparse_int         *nnz,
                                          aoclsparse_int        **col_ptr,
                                          aoclsparse_int        **row_idx,
                                          T                     **val)
{
    // Input validation
    if((mat == nullptr) || mat->mats.empty() || (base == nullptr) || (m == nullptr)
       || (n == nullptr) || (nnz == nullptr) || (col_ptr == nullptr) || (row_idx == nullptr)
       || (val == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    // check if data type of matrix is same as requested
    if(mat->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }
    // Find the valid csc matrix
    aoclsparse::csr *csc_mat = nullptr;
    for(auto *mat_rep : mat->mats)
    {
        auto *temp_mat = dynamic_cast<aoclsparse::csr *>(mat_rep);
        if(temp_mat && temp_mat->mat_type == aoclsparse_csc_mat && temp_mat->ptr && temp_mat->ind
           && temp_mat->val)
        {
            csc_mat = temp_mat; // Valid CSC
            if(temp_mat->is_optimized)
                break; // Found the optimized CSC matrix
        }
    }
    if(!csc_mat)
        return aoclsparse_status_invalid_value;

    *col_ptr = csc_mat->ptr;
    *row_idx = csc_mat->ind;
    *val     = static_cast<T *>(csc_mat->val);
    *m       = mat->m;
    *n       = mat->n;
    *nnz     = mat->nnz;
    *base    = csc_mat->base;
    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief assign matrix properties such as fill_mode, matrix type, transpose and
 * diagonal type to descriptor and transpose parameters provided. These output
 * params (descr_dest, trans_dest) would be used to access either strict triangles
 * or triangles with diagonal in Gauss Seidel process.
 ********************************************************************************/
void set_symgs_matrix_properties(aoclsparse_mat_descr  descr_dest,
                                 aoclsparse_operation *trans_dest,
                                 aoclsparse_fill_mode &fmode,
                                 aoclsparse_diag_type &dtype,
                                 aoclsparse_operation &trans)
{
    aoclsparse_set_mat_fill_mode(descr_dest, fmode);
    aoclsparse_set_mat_diag_type(descr_dest, dtype);
    *trans_dest = trans;
    return;
}

// Returns '1' if the build is AVX512 enabled. Else return '0'
aoclsparse_int aoclsparse_is_avx512_build()
{
#ifdef USE_AVX512
    return 1;
#else
    return 0;
#endif
}

template <typename T>
aoclsparse_int aoclsparse::test::dispatcher(std::string    t_name,
                                            aoclsparse_int kid,
                                            aoclsparse_int begin,
                                            aoclsparse_int end)
{
    using namespace dispatcher_instantiations;

    if(t_name.compare("dispatch_only_ref") == 0)
        return dispatch_only_ref<T>(kid);
    else if(t_name.compare("dispatch_l1") == 0)
        return dispatch_l1<T>(kid);
    else if(t_name.compare("dispatch_multi") == 0)
        return dispatch_multi<T>(kid);
    else if(t_name.compare("dispatch_noexact") == 0)
        return dispatch_noexact<T>(kid);
    else if(t_name.compare("dispatch") == 0)
        return dispatch<T>(kid);
    else if(t_name.compare("dispatch_isa") == 0)
        return dispatch_isa<T>(kid);
    else if(t_name.compare("dispatch_AVX512VL") == 0)
        return dispatch_AVX512VL<T>(kid);
    else if(t_name.compare("dispatch_range") == 0)
        return dispatch<T, true>(begin, end, kid);

    return -1000; // Invalid dispatcher name
}

// Instantiate the dispatcher test wrapper for all data types
#define AOCLSPARSE_DISPATCHER(SUF)                                        \
    template DLL_PUBLIC aoclsparse_int aoclsparse::test::dispatcher<SUF>( \
        std::string dispatch, aoclsparse_int kid, aoclsparse_int begin, aoclsparse_int end);

INSTANTIATE_FOR_ALL_TYPES(AOCLSPARSE_DISPATCHER);