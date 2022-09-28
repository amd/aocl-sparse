#include "aoclsparse.h"
#include "functionality_tests_utils.h"

#include <iostream>
#include <vector>

template <>
aoclsparse_status create_aoclsparse_matrix(aoclsparse_matrix           &A,
                                           aoclsparse_int               m,
                                           aoclsparse_int               n,
                                           aoclsparse_int               nnz,
                                           std::vector<aoclsparse_int> &icrow,
                                           std::vector<aoclsparse_int> &icol,
                                           std::vector<double>         &aval)
{
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    return aoclsparse_create_dcsr(A, base, m, n, nnz, &icrow[0], &icol[0], &aval[0]);
}

template <>
aoclsparse_status create_aoclsparse_matrix(aoclsparse_matrix           &A,
                                           aoclsparse_int               m,
                                           aoclsparse_int               n,
                                           aoclsparse_int               nnz,
                                           std::vector<aoclsparse_int> &icrow,
                                           std::vector<aoclsparse_int> &icol,
                                           std::vector<float>          &aval)
{
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    return aoclsparse_create_scsr(A, base, m, n, nnz, &icrow[0], &icol[0], &aval[0]);
}

template <>
aoclsparse_status itsol_solve(
    aoclsparse_itsol_handle    handle,
    aoclsparse_int             n,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    const double              *b,
    double                    *x,
    double                     rinfo[100],
    aoclsparse_int             precond(
        aoclsparse_int flag, aoclsparse_int n, const double *u, double *v, void *udata),
    aoclsparse_int monit(
        aoclsparse_int n, const double *x, const double *r, double rinfo[100], void *udata),
    void *udata)
{
    return aoclsparse_itsol_d_solve(handle, n, mat, descr, b, x, rinfo, precond, monit, udata);
}

template <>
aoclsparse_status itsol_solve(
    aoclsparse_itsol_handle    handle,
    aoclsparse_int             n,
    aoclsparse_matrix          mat,
    const aoclsparse_mat_descr descr,
    const float               *b,
    float                     *x,
    float                      rinfo[100],
    aoclsparse_int             precond(
        aoclsparse_int flag, aoclsparse_int n, const float *u, float *v, void *udata),
    aoclsparse_int monit(
        aoclsparse_int n, const float *x, const float *r, float rinfo[100], void *udata),
    void *udata)
{
    return aoclsparse_itsol_s_solve(handle, n, mat, descr, b, x, rinfo, precond, monit, udata);
}

template <>
aoclsparse_status itsol_rci_solve(aoclsparse_itsol_handle   handle,
                                  aoclsparse_itsol_rci_job *ircomm,
                                  double                  **u,
                                  double                  **v,
                                  double                   *x,
                                  double                    rinfo[100])
{
    return aoclsparse_itsol_d_rci_solve(handle, ircomm, u, v, x, rinfo);
}

template <>
aoclsparse_status itsol_rci_solve(aoclsparse_itsol_handle   handle,
                                  aoclsparse_itsol_rci_job *ircomm,
                                  float                   **u,
                                  float                   **v,
                                  float                    *x,
                                  float                     rinfo[100])
{
    return aoclsparse_itsol_s_rci_solve(handle, ircomm, u, v, x, rinfo);
}

template <>
aoclsparse_status itsol_init<double>(aoclsparse_itsol_handle *handle)
{
    return aoclsparse_itsol_d_init(handle);
}

template <>
aoclsparse_status itsol_init<float>(aoclsparse_itsol_handle *handle)
{
    return aoclsparse_itsol_s_init(handle);
}
