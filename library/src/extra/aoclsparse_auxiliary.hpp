/* ************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#ifndef AOCLSPARSE_AUXILIARY_HPP
#define AOCLSPARSE_AUXILIARY_HPP

#include "aoclsparse_mat_structures.h"
#include "aoclsparse_analysis.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_dispatcher.hpp"
#include "aoclsparse_utils.hpp"

#include <cmath>
#include <limits>

aoclsparse_status aoclsparse_destroy_mv(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_2m(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_ilu(_aoclsparse_ilu *ilu_info);
aoclsparse_status aoclsparse_destroy_symgs(_aoclsparse_symgs *sgs_info);
aoclsparse_status aoclsparse_destroy_opt_csr(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_csc(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_coo(aoclsparse_matrix A);
aoclsparse_status aoclsparse_destroy_tcsr(aoclsparse_matrix A);
void              set_symgs_matrix_properties(aoclsparse_mat_descr  descr_dest,
                                              aoclsparse_operation *trans_dest,
                                              aoclsparse_fill_mode &fmode,
                                              aoclsparse_diag_type &dtype,
                                              aoclsparse_operation &trans);

void aoclsparse_init_mat(aoclsparse_matrix             A,
                         aoclsparse_index_base         base,
                         aoclsparse_int                M,
                         aoclsparse_int                N,
                         aoclsparse_int                nnz,
                         aoclsparse_matrix_format_type matrix_type);

/********************************************************************************
 * \brief aoclsparse_create_csr_t sets the sparse matrix in the CSR format
 * for any data type
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_create_csr_t(aoclsparse_matrix    *mat,
                                          aoclsparse_index_base base,
                                          aoclsparse_int        M,
                                          aoclsparse_int        N,
                                          aoclsparse_int        nnz,
                                          aoclsparse_int       *row_ptr,
                                          aoclsparse_int       *col_idx,
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
            M, N, nnz, row_ptr, col_idx, val, shape_general, base, mat_sort, mat_fulldiag, nullptr))
       != aoclsparse_status_success)
    {
        return status;
    }
    try
    {
        *mat = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }
    aoclsparse_init_mat(*mat, base, M, N, nnz, aoclsparse_csr_mat);
    (*mat)->val_type            = get_data_type<T>();
    (*mat)->mat_type            = aoclsparse_csr_mat;
    (*mat)->csr_mat.csr_row_ptr = row_ptr;
    (*mat)->csr_mat.csr_col_ptr = col_idx;
    (*mat)->csr_mat.csr_val     = val;
    (*mat)->csr_mat_is_users    = true;
    (*mat)->sort                = mat_sort;
    (*mat)->fulldiag            = mat_fulldiag;

    return aoclsparse_status_success;
}

/********************************************************************************
 * \brief aoclsparse_create_tcsr_t sets the sparse matrix in the TCSR format
 * \brief aoclsparse_create_tcsr sets the sparse matrix in the TCSR format
 * for any data type
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_create_tcsr_t(aoclsparse_matrix          *mat,
                                           const aoclsparse_index_base base,
                                           const aoclsparse_int        M,
                                           const aoclsparse_int        N,
                                           const aoclsparse_int        nnz,
                                           aoclsparse_int             *row_ptr_L,
                                           aoclsparse_int             *row_ptr_U,
                                           aoclsparse_int             *col_idx_L,
                                           aoclsparse_int             *col_idx_U,
                                           T                          *val_L,
                                           T                          *val_U)
{
    aoclsparse_status status;
    // null pointer
    if(!mat)
        return aoclsparse_status_invalid_pointer;
    *mat = nullptr;
    if(row_ptr_L == nullptr || row_ptr_U == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(col_idx_L == nullptr || col_idx_U == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(val_L == nullptr || val_U == nullptr)
        return aoclsparse_status_invalid_pointer;

    // index base check
    if(base != aoclsparse_index_base_one && base != aoclsparse_index_base_zero)
        return aoclsparse_status_invalid_value;

    // invalid sizes
    if((M < 0) || (N < 0) || (nnz < 0))
        return aoclsparse_status_invalid_size;

    // supports only a square matrix with full diagonals
    // TODO: If zero diagonals are supported, remove this check
    if(M != N)
        return aoclsparse_status_invalid_size;

    aoclsparse_int lnnz = row_ptr_L[M] - base; // nnz elements in the lower part
    aoclsparse_int unnz = row_ptr_U[M] - base; // nnz elements in the upper part

    // check nnz
    if(nnz != (lnnz + unnz - M))
        return aoclsparse_status_invalid_size;

    aoclsparse_matrix_sort sort_L, sort_U;
    bool                   fulldiag_L, fulldiag_U;

    // Support for full diagonals - to be implemented
    // TCSR Matrix expects full diagonals and fully/partially sorted matrix
    // check lower triangular part
    if((status = aoclsparse_mat_check_internal(M,
                                               N,
                                               lnnz,
                                               row_ptr_L,
                                               col_idx_L,
                                               val_L,
                                               shape_lower_triangle,
                                               base,
                                               sort_L,
                                               fulldiag_L,
                                               nullptr))
       != aoclsparse_status_success)
        return status;
    if(sort_L == aoclsparse_unsorted)
        return aoclsparse_status_unsorted_input;
    if(!fulldiag_L)
        return aoclsparse_status_invalid_value;

    // check upper triangular part
    if((status = aoclsparse_mat_check_internal(M,
                                               N,
                                               unnz,
                                               row_ptr_U,
                                               col_idx_U,
                                               val_U,
                                               shape_upper_triangle,
                                               base,
                                               sort_U,
                                               fulldiag_U,
                                               nullptr))
       != aoclsparse_status_success)
        return status;
    if(sort_U == aoclsparse_unsorted)
        return aoclsparse_status_unsorted_input;
    if(!fulldiag_U)
        return aoclsparse_status_invalid_value;

    // create matrix
    try
    {
        *mat = new _aoclsparse_matrix;
    }
    catch(std::bad_alloc &)
    {
        return aoclsparse_status_memory_error;
    }

    aoclsparse_init_mat(*mat, base, M, N, nnz, aoclsparse_tcsr_mat);
    (*mat)->val_type           = get_data_type<T>();
    (*mat)->tcsr_mat.col_idx_L = col_idx_L;
    (*mat)->tcsr_mat.col_idx_U = col_idx_U;
    (*mat)->tcsr_mat.row_ptr_L = row_ptr_L;
    (*mat)->tcsr_mat.row_ptr_U = row_ptr_U;
    (*mat)->tcsr_mat.val_L     = val_L;
    (*mat)->tcsr_mat.val_U     = val_U;
    (*mat)->tcsr_mat_is_users  = true;
    (*mat)->fulldiag           = true;
    (*mat)->mat_type = aoclsparse_tcsr_mat; // Used to identify the matrix type in the mv dispatcher
    if((sort_L == aoclsparse_partially_sorted || sort_U == aoclsparse_partially_sorted))
        (*mat)->sort = aoclsparse_partially_sorted;
    else
        (*mat)->sort = aoclsparse_fully_sorted;
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_create_csc_t(aoclsparse_matrix    *mat,
                                          aoclsparse_index_base base,
                                          aoclsparse_int        M,
                                          aoclsparse_int        N,
                                          aoclsparse_int        nnz,
                                          aoclsparse_int       *col_ptr,
                                          aoclsparse_int       *row_idx,
                                          T                    *val);

template <typename T>
aoclsparse_status aoclsparse_create_coo_t(aoclsparse_matrix          *mat,
                                          const aoclsparse_index_base base,
                                          const aoclsparse_int        M,
                                          const aoclsparse_int        N,
                                          const aoclsparse_int        nnz,
                                          aoclsparse_int             *row_ptr,
                                          aoclsparse_int             *col_ptr,
                                          T                          *val);

template <typename T>
aoclsparse_status aoclsparse_update_values_t(aoclsparse_matrix A, aoclsparse_int len, T *val)
{
    if(A == nullptr || val == nullptr)
        return aoclsparse_status_invalid_pointer;
    if(len != A->nnz)
        return aoclsparse_status_invalid_size;

    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    T *A_val = nullptr;

    switch(A->input_format)
    {
    case aoclsparse_csr_mat:
        A_val = reinterpret_cast<T *>(A->csr_mat.csr_val);
        memcpy(A_val, val, len * sizeof(T));
        break;
    case aoclsparse_csc_mat:
        A_val = reinterpret_cast<T *>(A->csc_mat.val);
        memcpy(A_val, val, len * sizeof(T));
        break;
    case aoclsparse_coo_mat:
        A_val = reinterpret_cast<T *>(A->coo_mat.val);
        memcpy(A_val, val, len * sizeof(T));
        break;
    default:
        return aoclsparse_status_not_implemented;
    }

    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_copy_csc(aoclsparse_int                n,
                                      aoclsparse_int                nnz,
                                      const struct _aoclsparse_csc *src,
                                      struct _aoclsparse_csc       *dest);

template <typename T>
aoclsparse_status aoclsparse_copy_coo(aoclsparse_int                nnz,
                                      const struct _aoclsparse_coo *src,
                                      struct _aoclsparse_coo       *dest);

template <typename T>
aoclsparse_status aoclsparse_copy_mat(const aoclsparse_matrix src, aoclsparse_matrix dest);

template <typename T>
aoclsparse_status aoclsparse_sort_mat(aoclsparse_matrix mat);

template <typename T>
aoclsparse_status aoclsparse_export_csr_t(const aoclsparse_matrix csr,
                                          aoclsparse_index_base  *base,
                                          aoclsparse_int         *m,
                                          aoclsparse_int         *n,
                                          aoclsparse_int         *nnz,
                                          aoclsparse_int        **row_ptr,
                                          aoclsparse_int        **col_ind,
                                          T                     **val);

template <typename T>
aoclsparse_status aoclsparse_export_csc_t(const aoclsparse_matrix mat,
                                          aoclsparse_index_base  *base,
                                          aoclsparse_int         *m,
                                          aoclsparse_int         *n,
                                          aoclsparse_int         *nnz,
                                          aoclsparse_int        **col_ptr,
                                          aoclsparse_int        **row_idx,
                                          T                     **val);

/********************************************************************************
 * \brief aoclsparse_matrix is a structure holding the aoclsparse coo matrix.
 * Use this routine to export the contents of this structure
 ********************************************************************************/
template <typename T>
aoclsparse_status aoclsparse_export_coo_t(const aoclsparse_matrix mat,
                                          aoclsparse_index_base  *base,
                                          aoclsparse_int         *m,
                                          aoclsparse_int         *n,
                                          aoclsparse_int         *nnz,
                                          aoclsparse_int        **row_ptr,
                                          aoclsparse_int        **col_ptr,
                                          T                     **val)
{
    if((mat == nullptr) || (base == nullptr) || (m == nullptr) || (n == nullptr) || (nnz == nullptr)
       || (row_ptr == nullptr) || (col_ptr == nullptr) || (val == nullptr))
    {
        return aoclsparse_status_invalid_pointer;
    }
    // check if data type of matrix is same as requested
    if(mat->val_type != get_data_type<T>())
    {
        return aoclsparse_status_wrong_type;
    }
    if((mat->coo_mat.row_ind != nullptr) && (mat->coo_mat.col_ind != nullptr)
       && (mat->coo_mat.val != nullptr))
    {
        *row_ptr = mat->coo_mat.row_ind;
        *col_ptr = mat->coo_mat.col_ind;
        *val     = static_cast<T *>(mat->coo_mat.val);
    }
    else
    {
        return aoclsparse_status_invalid_value;
    }

    *m    = mat->m;
    *n    = mat->n;
    *nnz  = mat->nnz;
    *base = mat->base;
    return aoclsparse_status_success;
}

template <typename T>
aoclsparse_status aoclsparse_set_coo_value(aoclsparse_matrix A,
                                           aoclsparse_int    row_idx,
                                           aoclsparse_int    col_idx,
                                           T                 val)
{
    if(A == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->coo_mat.row_ind == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->coo_mat.col_ind == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->coo_mat.val == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->input_format != aoclsparse_coo_mat)
        return aoclsparse_status_internal_error;

    T *temp_val = reinterpret_cast<T *>(A->coo_mat.val);

    for(aoclsparse_int i = 0; i < A->nnz; i++)
    {
        if(A->coo_mat.row_ind[i] == row_idx && A->coo_mat.col_ind[i] == col_idx)
        {
            temp_val[i] = val;
            return aoclsparse_status_success;
        }
    }
    return aoclsparse_status_invalid_index_value;
}

template <typename T>
aoclsparse_status aoclsparse_set_csr_value(aoclsparse_index_base base,
                                           const aoclsparse_int *row_ptr,
                                           const aoclsparse_int *col_ptr,
                                           T                    *val_ptr,
                                           aoclsparse_int        row_idx,
                                           aoclsparse_int        col_idx,
                                           T                     val)
{
    if(base != aoclsparse_index_base_one && base != aoclsparse_index_base_zero)
        return aoclsparse_status_invalid_value;

    if(row_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(col_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(val_ptr == nullptr)
        return aoclsparse_status_invalid_pointer;

    aoclsparse_int row   = row_idx - base;
    aoclsparse_int begin = row_ptr[row] - base;
    aoclsparse_int end   = row_ptr[row + 1] - base;

    for(aoclsparse_int i = begin; i < end; i++)
    {
        if(col_ptr[i] == col_idx)
        {
            val_ptr[i] = val;
            return aoclsparse_status_success;
        }
    }
    return aoclsparse_status_invalid_index_value;
}

template <typename T>
aoclsparse_status aoclsparse_set_value_t(aoclsparse_matrix A,
                                         aoclsparse_int    row_idx,
                                         aoclsparse_int    col_idx,
                                         T                 val)
{
    if(A == nullptr)
        return aoclsparse_status_invalid_pointer;

    // check if coordinate given by user is within matrix bounds
    if((A->m + A->base <= row_idx || row_idx < A->base)
       || (A->n + A->base <= col_idx || col_idx < A->base))
        return aoclsparse_status_invalid_value;

    // if matrix type is same as T
    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    aoclsparse_status status;
    T                *val_ptr = nullptr;

    // different method to set value for different types
    switch(A->input_format)
    {
    case aoclsparse_csr_mat:
        val_ptr = reinterpret_cast<T *>(A->csr_mat.csr_val);
        status  = aoclsparse_set_csr_value(A->base,
                                          A->csr_mat.csr_row_ptr,
                                          A->csr_mat.csr_col_ptr,
                                          val_ptr,
                                          row_idx,
                                          col_idx,
                                          val);
        break;
    case aoclsparse_csc_mat:
        val_ptr = reinterpret_cast<T *>(A->csc_mat.val);
        status  = aoclsparse_set_csr_value(
            A->base, A->csc_mat.col_ptr, A->csc_mat.row_idx, val_ptr, col_idx, row_idx, val);
        break;
    case aoclsparse_coo_mat:
        status = aoclsparse_set_coo_value(A, row_idx, col_idx, val);
        break;
    default:
        return aoclsparse_status_not_implemented;
    }
    if(status != aoclsparse_status_success)
        return status;

    // destroy the previously optimized data
    return aoclsparse_destroy_opt_csr(A);
}

// This is library-side instantiations of "internal" dispatchers
// used to UT the L1 dispacher/oracle framework
namespace dispatcher_instantiations
{
    enum ext
    {
        RESERVE_1,
        RESERVE_2,
        RESERVE_3
    };

    // -------------------------------------------------------------------------
    // KERNELS
    // -------------------------------------------------------------------------

    // boilerplate reference kernel
    template <aoclsparse_int K, typename T>
    aoclsparse_int kernel_ref()
    {
        T           t{0};
        std::string id{typeid(t).name()};
        std::cout << "K=" + std::to_string(K) + " " + std::string(__func__) + "<" + id + ">"
                  << std::endl;
        return K;
    }

    // boilerplate KT kernel
    template <aoclsparse_int K, aoclsparse_int SZ, typename SUF, ext M>
    aoclsparse_int kernel_kt()
    {
        SUF         t{0};
        std::string id{typeid(t).name()};
        std::cout << "K=" + std::to_string(K) + " " + std::string(__func__) + "<" + id + ","
                         + std::to_string(SZ) + "," + std::to_string(M) + ">"
                  << std::endl;
        return K;
    }

    // -------------------------------------------------------------------------
    // DISPATCHERS
    // -------------------------------------------------------------------------

    // dispatcher to single kernel
    template <typename T>
    aoclsparse_int dispatch_only_ref(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name                 cpu flag requirements    suggested architecture
        {kernel_ref<0, T>,      context_isa_t::GENERIC, 0|archs::ALL}
        };
        // clang-format on

        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved0>(tbl, kid);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();

        return 0 + okid;
    }

    /* Boilerplate dispatcher common for all Level 1 Sparse BLAS API entry points */
    template <typename T>
    aoclsparse_int dispatch_l1(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;
        using namespace Dispatch;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name                               cpu flag requirements   suggested architecture
        {kernel_ref<0, T>,                    context_isa_t::GENERIC, 0|archs::ALL },
        {kernel_kt<1, 256, T, ext::RESERVE_1>,context_isa_t::AVX2,    0|archs::ZENS},            // Zen 1+ AVX2
        {kernel_kt<2, 256, T, ext::RESERVE_1>,context_isa_t::AVX2,    0|archs::ZEN4|archs::ZEN5},// Zen 4+ AVX2
 ORL<K>({kernel_kt<3, 512, T, ext::RESERVE_2>,context_isa_t::AVX512F, 0|archs::ZEN4|archs::ZEN5})// Zen 4+ AVX512F
        };
        // clang-format on
        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved1>(tbl, kid);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();

        return 10 + okid;
    }

    template <typename T>
    aoclsparse_int dispatch_multi(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        if constexpr(std::is_same_v<T, float>)
        {
            static constexpr Dispatch::Table<K> tbl[]{
                {kernel_ref<0, T>, context_isa_t::GENERIC, 0 | archs::ALL},
                {kernel_ref<1, T>, context_isa_t::GENERIC, 0 | archs::ALL},
                {kernel_ref<2, T>, context_isa_t::GENERIC, 0 | archs::ALL}};
            auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved2>(tbl, kid);

            if(kernel == nullptr)
                return aoclsparse_status_invalid_kid;

            return kernel();
        }
        else
        {
            static constexpr Dispatch::Table<K> tbl[]{
                {kernel_ref<7, T>, context_isa_t::GENERIC, 0 | archs::ALL}};
            auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved3>(tbl, kid);

            if(kernel == nullptr)
                return aoclsparse_status_invalid_kid;

            return kernel();
        }
    }

    // Table does not provide exact kernel match
    template <typename T>
    aoclsparse_int dispatch_noexact(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name            cpu flag requirements   suggested architecture
        {kernel_ref<0, T>, context_isa_t::GENERIC, 0|archs::ALL},
        {kernel_ref<1, T>, context_isa_t::GENERIC, 0|archs::ZEN123},
        {kernel_ref<2, T>, context_isa_t::GENERIC, 0|archs::ZENS}
        };
        // clang-format on
        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved4>(tbl, kid);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();
        return 0 + okid;
    }

    // high complexity dispatcher to test corner cases
    template <typename T>
    aoclsparse_int dispatch(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);
        using namespace aoclsparse;
        using namespace Dispatch;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name                              cpu flag requirements    suggested architecture
        {kernel_ref<0, T>,                     context_isa_t::GENERIC,  0|archs::ALL},             // 0 All machines
        {kernel_ref<1, T>,                     context_isa_t::AVX2,     0|archs::ZEN123},          // 1 Naples/Rome/Milan: Zen AVX2
        {kernel_kt<2, 256, T, ext::RESERVE_1>, context_isa_t::AVX2,     0|archs::ZENS},            // 2 AMD platform AVX2
        {kernel_kt<3, 256, T, ext::RESERVE_1>, context_isa_t::AVX2,     0|archs::ZEN3},            // 3 Milan: Zen3 AVX2
        {kernel_kt<4, 256, T, ext::RESERVE_1>, context_isa_t::AVX2,     0|archs::ZEN4|archs::ZEN5},// 4 Bergamo/Genoa/Siena: Zen4 AVX2
 ORL<K>({kernel_kt<5, 512, T, ext::RESERVE_3>, context_isa_t::AVX512DQ, 0|archs::ZENS}),           // 5 AMD platform AVX512F
 ORL<K>({kernel_kt<6, 512, T, ext::RESERVE_3>, context_isa_t::AVX512VL, 0|archs::ZEN4|archs::ZEN5})// 6 Bergamo/Genoa/Siena: Zen4 AVX512F
        };
        // clang-format on

        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved5>(tbl, kid);

        if(!kernel)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();

        return 1000 + okid;
    }

    // Table does not provide exactly one kernel per ISA_HINT
    template <typename T>
    aoclsparse_int dispatch_isa(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);

        using namespace aoclsparse;

        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name            cpu flag requirements    suggested architecture
        {kernel_ref<0, T>, context_isa_t::GENERIC, 0|archs::ALL},
        {kernel_ref<1, T>, context_isa_t::AVX2, 0|archs::ALL}
        };
        // clang-format on
        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved4>(tbl, kid);

        if(kernel == nullptr)
            return aoclsparse_status_invalid_kid;

        auto okid = kernel();

        return 0 + okid;
    }

    // Table to test for 256b kernels using AVX512VL flags
    template <typename T>
    aoclsparse_int dispatch_AVX512VL(aoclsparse_int kid = -1)
    {
        using K = decltype(&kernel_ref<0, T>);
        using namespace aoclsparse;
        using namespace Dispatch;
        // clang-format off
        static constexpr Dispatch::Table<K> tbl[] {
        // name                                cpu flag requirements    suggested architecture
        {kernel_kt<0, 256, T, ext::RESERVE_3>, context_isa_t::AVX2,     0|archs::ALL},
 ORL<K>({kernel_kt<1, 256, T, ext::RESERVE_3>, context_isa_t::AVX512VL, 0|archs::ZEN5|archs::ZEN4}),
 ORL<K>({kernel_kt<2, 512, T, ext::RESERVE_3>, context_isa_t::AVX512VL, 0|archs::ALL})
        };
        // clang-format on
        auto kernel = Dispatch::Oracle<K, Dispatch::api::reserved4>(tbl, kid);
        if(kernel == nullptr)
            return aoclsparse_status_invalid_kid;
        auto okid = kernel();
        return 4000 + okid;
    }

}

#endif
