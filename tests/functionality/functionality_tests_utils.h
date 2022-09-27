#include "aoclsparse.h"

#include <iostream>
#include <string>
#include <vector>

enum matrix_id
{
    /* Used mainly by hint_tests*/
    N5_full_sorted,
    N5_full_unsorted,
    N5_1_hole,
    N5_empty_rows,
    N10_random,
    M5_rect_N7,
    M5_rect_N7_2holes,
    M7_rect_N5,
    M7_rect_N5_2holes,
    /****************************/
    /* CG tests matrices */
    sample_cg_mat, // matrice from the CG examples
    invalid_mat,
};

template <typename T>
aoclsparse_status create_aoclsparse_matrix(aoclsparse_matrix           &A,
                                           aoclsparse_int               m,
                                           aoclsparse_int               n,
                                           aoclsparse_int               nnz,
                                           std::vector<aoclsparse_int> &icrow,
                                           std::vector<aoclsparse_int> &icol,
                                           std::vector<T>              &aval);

template <typename T>
aoclsparse_status create_matrix(matrix_id                    mid,
                                aoclsparse_int              &m,
                                aoclsparse_int              &n,
                                aoclsparse_int              &nnz,
                                std::vector<aoclsparse_int> &icrow,
                                std::vector<aoclsparse_int> &icol,
                                std::vector<T>              &aval,
                                aoclsparse_matrix           &A,
                                aoclsparse_mat_descr        &descr,
                                aoclsparse_int               verbose)
{
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_status     ret  = aoclsparse_status_success;

    // default descriptor
    aoclsparse_create_mat_descr(&descr);

    switch(mid)
    {
    case N5_full_sorted:
        // Small sorted matrix with full diagonal
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  4  0  0
        //  0  5  0  6  7
        //  0  0  0  0  8
        n = m = 5;
        nnz   = 8;
        icrow = {0, 2, 3, 4, 7, 8};
        icol  = {0, 3, 1, 2, 1, 3, 4, 4};
        aval  = {1, 2, 3, 4, 5, 6, 7, 8};
        break;

    case N5_full_unsorted:
        // same as N5 full sorted with rows 0 and 3 shuffled
        n = m = 5;
        nnz   = 8;
        icrow = {0, 2, 3, 4, 7, 8};
        icol  = {3, 0, 1, 2, 3, 1, 4, 4};
        aval  = {2, 1, 3, 4, 6, 5, 7, 8};
        break;

    case N5_1_hole:
        // same as N5 full unsorted with row 3 diag element removed
        n = m = 5;
        nnz   = 7;
        icrow = {0, 2, 3, 4, 6, 7};
        icol  = {3, 0, 1, 2, 1, 4, 4};
        aval  = {2, 1, 3, 4, 5, 7, 8};
        break;

    case N5_empty_rows:
        // removed even more diag elements, creating empty rows
        n = m = 5;
        nnz   = 5;
        icrow = {0, 2, 2, 3, 5, 5};
        icol  = {3, 0, 2, 1, 4};
        aval  = {2, 1, 4, 5, 7};
        break;

    case N10_random:
        // randomly generated matrix with missing elements and unsorted rows
        n = m = 10;
        nnz   = 42;
        icrow = {0, 3, 6, 8, 18, 24, 27, 31, 36, 38, 42};
        icol  = {9, 4, 6, 3, 8, 6, 0, 6, 4, 6, 7, 1, 2, 9, 3, 8, 5, 0, 6, 2, 1,
                 5, 3, 8, 3, 8, 5, 1, 4, 8, 5, 9, 1, 4, 8, 5, 4, 6, 6, 2, 3, 7};
        aval  = {5.91, 5.95, 7.95, 0.83, 5.48, 6.75, 0.01, 4.78, 9.20, 3.40, 2.26, 3.01, 8.34, 6.82,
                 7.40, 1.12, 3.31, 4.96, 2.66, 1.77, 5.28, 8.95, 3.09, 2.37, 4.48, 2.92, 1.46, 6.17,
                 8.77, 9.96, 7.19, 9.61, 6.48, 4.95, 6.76, 8.87, 5.07, 3.58, 2.09, 8.66, 6.77, 3.69};
        break;

    case M5_rect_N7:
        // same as N5_full_sorted with 2 added columns
        //  1  0  0  2  0  1  0
        //  0  3  0  0  0  2  0
        //  0  0  4  0  0  0  0
        //  0  5  0  6  7  0  3
        //  0  0  0  0  8  4  5
        n     = 7;
        m     = 5;
        nnz   = 13;
        icrow = {0, 3, 5, 6, 10, 13};
        icol  = {0, 3, 5, 1, 5, 2, 1, 3, 4, 6, 4, 5, 6};
        aval  = {1, 2, 1, 3, 2, 4, 5, 6, 7, 3, 8, 4, 5};
        break;

    case M5_rect_N7_2holes:
        // same as M5_rect_N7, with missing diag elements in rows 2 an 4
        n     = 7;
        m     = 5;
        nnz   = 11;
        icrow = {0, 3, 5, 5, 9, 11};
        icol  = {0, 3, 5, 1, 5, 1, 3, 4, 6, 5, 6};
        aval  = {1, 2, 1, 3, 2, 5, 6, 7, 3, 4, 5};
        break;

    case M7_rect_N5:
        // ame as N5_full_sorted with 2 added rows
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  4  0  0
        //  0  5  0  6  7
        //  0  0  0  0  8
        //  0  1  2  0  0
        //  3  0  0  4  0
        n     = 5;
        m     = 7;
        nnz   = 12;
        icrow = {0, 2, 3, 4, 7, 8, 10, 12};
        icol  = {0, 3, 1, 2, 1, 3, 4, 4, 1, 2, 0, 3};
        aval  = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4};
        break;

    case M7_rect_N5_2holes:
        // ame as N5_full_sorted with 2 added rows
        //  1  0  0  2  0
        //  0  3  0  0  0
        //  0  0  0  0  0
        //  0  5  0  6  7
        //  0  0  0  0  0
        //  0  1  2  0  0
        //  3  0  0  4  0
        n = m = 5;
        m     = 7;
        nnz   = 10;
        icrow = {0, 2, 3, 3, 6, 6, 8, 10};
        icol  = {0, 3, 1, 3, 1, 4, 1, 2, 0, 3};
        aval  = {1, 2, 3, 6, 5, 7, 1, 2, 3, 4};
        break;

    case sample_cg_mat:
        // matrix from the CG sample examples
        // symmetric, lower triangle filled
        n = m = 8;
        nnz   = 18;
        icrow = {0, 1, 2, 5, 6, 8, 11, 15, 18};
        icol  = {0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
        aval  = {19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        break;

    case invalid_mat:
        // matrix from the CG sample examples
        // symmetric, lower triangle filled
        n = m = 8;
        nnz   = 18;
        icrow = {0, 1, 2, 5, 6, 8, 11, 15, 17};
        icol  = {0, 1, 0, 1, 2, 3, 1, 4, 0, 4, 5, 0, 3, 4, 6, 2, 5, 7};
        aval  = {19, 10, 1, 8, 11, 13, 2, 11, 2, 1, 9, 7, 9, 5, 12, 5, 5, 9};
        aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_symmetric);
        aoclsparse_set_mat_fill_mode(descr, aoclsparse_fill_mode_lower);
        break;

    default:
        if(verbose)
            std::cout << "Non recognized matrix id: " << mid << std::endl;
        return aoclsparse_status_invalid_value;
    }

    ret = create_aoclsparse_matrix<T>(A, m, n, nnz, icrow, icol, aval);
    if(ret != aoclsparse_status_success && verbose)
        std::cout << "Unexpected error in matrix creation" << std::endl;

    return ret;
}

template <typename T>
void comp_exact_vec(
    std::string id, aoclsparse_int n, T *v1, T *v2, bool &pass, aoclsparse_int verbose)
{
    pass = true;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        if(v1[i] != v2[i])
        {
            pass = false;
            if(verbose)
                std::cout << id << " index " << i << ": v1[i] = " << v1[i] << ", v2[i] = " << v2[i]
                          << std::endl;
            break;
        }
    }
}

template <typename T>
void comp_tol_vec(
    std::string id, T tol, aoclsparse_int n, T *v1, T *v2, bool &pass, aoclsparse_int verbose)
{
    pass = true;
    for(aoclsparse_int i = 0; i < n; i++)
    {
        if(abs(v1[i] - v2[i]) > tol)
        {
            pass = false;
            if(verbose)
                std::cout << id << " index " << i << ": v1[i] = " << v1[i] << ", v2[i] = " << v2[i]
                          << std::endl;
            break;
        }
    }
}

template <typename T>
aoclsparse_status
    itsol_solve(aoclsparse_itsol_handle    handle,
                aoclsparse_int             n,
                aoclsparse_matrix          mat,
                const aoclsparse_mat_descr descr,
                const T                   *b,
                T                         *x,
                T                          rinfo[100],
                aoclsparse_int precond(aoclsparse_int flag, const T *u, T *v, void *udata),
                aoclsparse_int monit(const T *x, const T *r, T rinfo[100], void *udata),
                void          *udata);

template <typename T>
aoclsparse_status itsol_rci_solve(aoclsparse_itsol_handle   handle,
                                  aoclsparse_itsol_rci_job *ircomm,
                                  T                       **u,
                                  T                       **v,
                                  T                        *x,
                                  T                         rinfo[100]);

template <typename T>
aoclsparse_status itsol_init(aoclsparse_itsol_handle *handle);
