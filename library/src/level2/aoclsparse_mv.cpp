/* ************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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
#include "aoclsparse_mv.hpp"

/* Template specializations */
template <>
aoclsparse_status aoclsparse_dcsr_mat_br4(aoclsparse_operation       op,
                                          const double               alpha,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          const double              *x,
                                          const double               beta,
                                          double                    *y)
{
    // Read the environment variables to update global variable
    // This function updates the num_threads only once.
    aoclsparse_init_once();

    aoclsparse_context context;
    context.num_threads = global_context.num_threads;

    __m256d res, vvals, vx, vy, va, vb;

    va  = _mm256_set1_pd(alpha);
    vb  = _mm256_set1_pd(beta);
    res = _mm256_setzero_pd();

    aoclsparse_int                 *tcptr = (aoclsparse_int *)A->csr_mat_br4.csr_col_ptr;
    aoclsparse_int                 *rptr  = (aoclsparse_int *)A->csr_mat_br4.csr_row_ptr;
    aoclsparse_int                 *cptr;
    double                         *tvptr = (double *)A->csr_mat_br4.csr_val;
    double                         *vptr;
    aoclsparse_int                  blk        = 4;
    [[maybe_unused]] aoclsparse_int chunk_size = (A->m) / (blk * context.num_threads);

#ifdef _OPENMP
#pragma omp parallel for num_threads(context.num_threads) \
    schedule(dynamic, chunk_size) private(res, vvals, vx, vy, vptr, cptr)
#endif
    for(aoclsparse_int i = 0; i < (A->m) / blk; i++)
    {

        aoclsparse_int r = rptr[i * blk];
        vptr             = (double *)(tvptr + r);
        cptr             = tcptr + r;

        res = _mm256_setzero_pd();
        // aoclsparse_int nnz = rptr[i*blk];
        aoclsparse_int nnz = rptr[i * blk + 1] - r;
        for(aoclsparse_int j = 0; j < nnz; ++j)
        {
            aoclsparse_int off = j * blk;
            vvals              = _mm256_loadu_pd((double const *)(vptr + off));

            vx = _mm256_set_pd(
                x[*(cptr + off + 3)], x[*(cptr + off + 2)], x[*(cptr + off + 1)], x[*(cptr + off)]);

            res = _mm256_fmadd_pd(vvals, vx, res);
        }
        /*
	   tc += blk*nnz;
	   vptr += blk*nnz;
	   cptr += blk*nnz;
	   */

        if(alpha != static_cast<double>(1))
        {
            res = _mm256_mul_pd(va, res);
        }

        if(beta != static_cast<double>(0))
        {
            vy  = _mm256_loadu_pd(&y[i * blk]);
            res = _mm256_fmadd_pd(vb, vy, res);
        }
        _mm256_storeu_pd(&y[i * blk], res);
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(context.num_threads)
#endif
    for(aoclsparse_int k = ((A->m) / blk) * blk; k < A->m; ++k)
    {
        double result = 0;
        /*
	   aoclsparse_int nnz = A->csr_mat_br4.csr_row_ptr[k];
	   for(j = 0; j < nnz; ++j)
	   {
	   result += ((double *)A->csr_mat_br4.csr_val)[tc] * x[A->csr_mat_br4.csr_col_ptr[tc]];
	   tc++;;
	   }
	   */
        for(aoclsparse_int j = A->csr_mat_br4.csr_row_ptr[k]; j < A->csr_mat_br4.csr_row_ptr[k + 1];
            ++j)
        {
            result += ((double *)A->csr_mat_br4.csr_val)[j] * x[A->csr_mat_br4.csr_col_ptr[j]];
        }

        if(alpha != static_cast<double>(1))
        {
            result = alpha * result;
        }

        if(beta != static_cast<double>(0))
        {
            result += beta * y[k];
        }
        y[k] = result;
    }

    return aoclsparse_status_success;
}

template <>
aoclsparse_status aoclsparse_dcsr_mat_br4(aoclsparse_operation       op,
                                          const float                alpha,
                                          aoclsparse_matrix          A,
                                          const aoclsparse_mat_descr descr,
                                          const float               *x,
                                          const float                beta,
                                          float                     *y)
{
    return aoclsparse_status_not_implemented;
}

template <>
aoclsparse_status aoclsparse_mv_general(aoclsparse_operation       op,
                                        const float                alpha,
                                        aoclsparse_matrix          A,
                                        const aoclsparse_mat_descr descr,
                                        const float               *x,
                                        const float                beta,
                                        float                     *y)
{
    // ToDo: optimized float versions need to be implemented
    if(A->mat_type == aoclsparse_csr_mat)
    {
        return (aoclsparse_scsrmv(op,
                                  &alpha,
                                  A->m,
                                  A->n,
                                  A->nnz,
                                  (float *)A->csr_mat.csr_val,
                                  A->csr_mat.csr_col_ptr,
                                  A->csr_mat.csr_row_ptr,
                                  descr,
                                  x,
                                  &beta,
                                  y));
    }
    else
    {
        return aoclsparse_status_not_implemented;
    }
}

template <>
aoclsparse_status aoclsparse_mv_general(aoclsparse_operation       op,
                                        const double               alpha,
                                        aoclsparse_matrix          A,
                                        const aoclsparse_mat_descr descr,
                                        const double              *x,
                                        const double               beta,
                                        double                    *y)
{
    if(A->mat_type == aoclsparse_csr_mat)
    {
        //Invoke SPMV API for CSR storage format(double precision)
#if USE_AVX512
        if(A->blk_optimized)
            return aoclsparse_dblkcsrmv(op,
                                        &alpha,
                                        A->m,
                                        A->n,
                                        A->nnz,
                                        A->csr_mat.masks,
                                        (double *)A->csr_mat.blk_val,
                                        A->csr_mat.blk_col_ptr,
                                        A->csr_mat.blk_row_ptr,
                                        descr,
                                        x,
                                        &beta,
                                        y,
                                        A->csr_mat.nRowsblk);
        else
            return (aoclsparse_dcsrmv(op,
                                      &alpha,
                                      A->m,
                                      A->n,
                                      A->nnz,
                                      (double *)A->csr_mat.csr_val,
                                      A->csr_mat.csr_col_ptr,
                                      A->csr_mat.csr_row_ptr,
                                      descr,
                                      x,
                                      &beta,
                                      y));

#else
        return (aoclsparse_dcsrmv(op,
                                  &alpha,
                                  A->m,
                                  A->n,
                                  A->nnz,
                                  (double *)A->csr_mat.csr_val,
                                  A->csr_mat.csr_col_ptr,
                                  A->csr_mat.csr_row_ptr,
                                  descr,
                                  x,
                                  &beta,
                                  y));
#endif
    }
    else if(A->mat_type == aoclsparse_ellt_csr_hyb_mat)
    {
        return (aoclsparse_dellthybmv(op,
                                      &alpha,
                                      A->m,
                                      A->n,
                                      A->nnz,
                                      (double *)A->ell_csr_hyb_mat.ell_val,
                                      A->ell_csr_hyb_mat.ell_col_ind,
                                      A->ell_csr_hyb_mat.ell_width,
                                      A->ell_csr_hyb_mat.ell_m,
                                      (double *)A->ell_csr_hyb_mat.csr_val,
                                      A->csr_mat.csr_row_ptr,
                                      A->csr_mat.csr_col_ptr,
                                      nullptr,
                                      A->ell_csr_hyb_mat.csr_row_id_map,
                                      descr,
                                      x,
                                      &beta,
                                      y));
    }
    else if(A->mat_type == aoclsparse_csr_mat_br4)
    {
        return (aoclsparse_dcsr_mat_br4(op, alpha, A, descr, x, beta, y));
    }
    else
    {
        return aoclsparse_status_invalid_value;
    }
}

/*
 *===========================================================================
 *   C wrapper
 * ===========================================================================
 */
extern "C" aoclsparse_status aoclsparse_smv(aoclsparse_operation       op,
                                            const float               *alpha,
                                            aoclsparse_matrix          A,
                                            const aoclsparse_mat_descr descr,
                                            const float               *x,
                                            const float               *beta,
                                            float                     *y)
{
    // All input checks are done in the templated version
    return aoclsparse_mv(op, *alpha, A, descr, x, *beta, y);
}

extern "C" aoclsparse_status aoclsparse_dmv(aoclsparse_operation       op,
                                            const double              *alpha,
                                            aoclsparse_matrix          A,
                                            const aoclsparse_mat_descr descr,
                                            const double              *x,
                                            const double              *beta,
                                            double                    *y)
{
    // All input checks are done in the templated version
    return aoclsparse_mv(op, *alpha, A, descr, x, *beta, y);
}
