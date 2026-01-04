/* ************************************************************************
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************
 */
#include "aoclsparse.h"
#include "aoclsparse_context.h"
#include "aoclsparse_kernel_templates.hpp"
#include "aoclsparse_l2_kt.hpp"
#include "aoclsparse_utils.hpp"

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status aoclsparse::csrmv_kt(aoclsparse_index_base base,
                                       const SUF             alpha,
                                       aoclsparse_int        m,
                                       const SUF *__restrict__ aval,
                                       const aoclsparse_int *__restrict__ icol,
                                       const aoclsparse_int *__restrict__ row,
                                       const SUF *__restrict__ x,
                                       const SUF beta,
                                       SUF *__restrict__ y)
{

    using namespace kernel_templates;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *aval_fix = aval - base;
    const SUF            *x_fix    = x - base;
    avxvector_t<SZ, SUF>  va, vx, vb, vc;
    const size_t          k = tsz_v<SZ, SUF>;
    // Perform (beta * y)
    if(beta == static_cast<SUF>(0))
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<SUF>(1))
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads()) private( \
        va, vx, vb, vc)
#endif
    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int j;
        SUF            result  = 0.0;
        vb                     = kt_setzero_p<SZ, SUF>();
        vc                     = kt_setzero_p<SZ, SUF>();
        aoclsparse_int crend   = row[i + 1];
        aoclsparse_int crstart = row[i];
        aoclsparse_int nnz     = crend - crstart;
        aoclsparse_int k_rem   = nnz % k;

        //Loop in multiples of K non-zeroes
        for(j = crstart; j < crend - k_rem; j += k)
        {
            va = kt_loadu_p<SZ, SUF>(&aval_fix[j]);
            vx = kt_set_p<SZ, SUF>(x_fix, &icol_fix[j]);
            kt_fmadd_B<SZ, SUF>(va, vx, vb, vc);
        }

        // Horizontal addition
        result = kt_hsum_B<SZ, SUF>(vb, vc);

        //Remainder loop for nnz%k
        for(j = crend - k_rem; j < crend; j++)
        {
            result += aval_fix[j] * x_fix[icol_fix[j]];
        }

        // Perform alpha * A * x
        result *= alpha;
        y[i] += result;
    }
    return aoclsparse_status_success;
}

template <kernel_templates::bsz SZ, typename SUF>
aoclsparse_status aoclsparse::csrmvt_kt(aoclsparse_index_base base,
                                        const SUF             alpha,
                                        aoclsparse_int        m,
                                        aoclsparse_int        n,
                                        const SUF *__restrict__ aval,
                                        const aoclsparse_int *__restrict__ icol,
                                        const aoclsparse_int *__restrict__ row,
                                        const SUF *__restrict__ x,
                                        const SUF beta,
                                        SUF *__restrict__ y)
{
    using namespace kernel_templates;
    aoclsparse_int        status   = aoclsparse_status_success;
    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *aval_fix = aval - base;

    const size_t tsz = tsz_v<SZ, SUF>;

#ifdef _OPENMP
    aoclsparse_int nthreads = context::get_context()->get_num_threads();
#endif

    // Perform (beta * y)
    if(beta == static_cast<SUF>(0))
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<SUF>(1))
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = beta * y[i];
    }

    // Iterate over each row of the input matrix and
    // Perform matrix-vector product for each non-zero of the ith row
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads) reduction(max : status)
#endif
    {
#ifdef _OPENMP
        aoclsparse_int tid   = omp_get_thread_num();
        aoclsparse_int start = m * tid / nthreads;
        aoclsparse_int end   = m * (tid + 1) / nthreads;
        status               = aoclsparse_status_success;
        // Thread-local buffer
        std::vector<SUF> y_buf;
        try
        {
            //y_buf.resize(n, 0);
            y_buf.resize(n);
            for(aoclsparse_int i = 0; i < n; i++)
                y_buf[i] = 0.;
        }
        catch(std::bad_alloc &)
        {
            status = aoclsparse_status_memory_error;
        }
#else
        aoclsparse_int start = 0;
        aoclsparse_int end   = m;
#endif
        if(status == aoclsparse_status_success)
        {
            aoclsparse_int i;
#ifdef _OPENMP
            SUF *y_buf_fix = y_buf.data() - base;
#else
            SUF *y_buf_fix = y - base;
#endif
            avxvector_t<SZ, SUF> va, vx, vy;
            for(i = start; i < end; i++)
            {
                aoclsparse_int crstart = row[i];
                aoclsparse_int crend   = row[i + 1];
                aoclsparse_int nnz     = crend - crstart;
                aoclsparse_int k_rem   = nnz % tsz;
                vx                     = kt_set1_p<SZ, SUF>(alpha * x[i]);
                aoclsparse_int j;

                for(j = crstart; j < crend - k_rem; j += tsz)
                {
                    va = kt_loadu_p<SZ, SUF>(&aval_fix[j]);
                    vy = kt_mul_p<SZ, SUF>(va, vx);

                    // Scatter vy values into y_local
                    const SUF *vy_cast = reinterpret_cast<const SUF *>(&vy);
                    for(size_t k = 0; k < tsz; k++)
                    {
                        aoclsparse_int idx = icol_fix[j + k];
                        y_buf_fix[idx] += vy_cast[k];
                    }
                }
                for(j = crend - k_rem; j < crend; j++)
                {
                    aoclsparse_int col = icol_fix[j];
                    y_buf_fix[col] += aval_fix[j] * alpha * x[i];
                }
            }
#ifdef _OPENMP
#pragma omp critical
            for(i = 0; i < n; i++)
            {
                // Combine thread-local buffer results
                y[i] += y_buf[i];
            }
#endif
        }
    }
    return (aoclsparse_status)status;
}

// The parameter HERM specifies if the input matrix is hermitian.
template <kernel_templates::bsz SZ, typename SUF, bool HERM = false>
aoclsparse_status aoclsparse::csrmv_symm_kt(aoclsparse_index_base base,
                                            const SUF             alpha,
                                            aoclsparse_int        m,
                                            aoclsparse_diag_type  diag_type,
                                            aoclsparse_fill_mode  fill_mode,
                                            const SUF *__restrict__ aval,
                                            const aoclsparse_int *__restrict__ icol,
                                            const aoclsparse_int *__restrict__ icrow,
                                            const aoclsparse_int *__restrict__ idiag,
                                            const aoclsparse_int *__restrict__ iurow,
                                            const SUF *__restrict__ x,
                                            const SUF beta,
                                            SUF *__restrict__ y)
{
    using namespace kernel_templates;

    const aoclsparse_int *crstart, *crend;
    if(fill_mode == aoclsparse_fill_mode_lower)
    {
        crstart = icrow;
        crend   = idiag;
    }
    else
    {
        crstart = iurow;
        crend   = icrow + 1;
    }

    const aoclsparse_int *icol_fix = icol - base;
    const SUF            *aval_fix = aval - base;
    const SUF            *x_fix    = x - base;
    SUF                  *y_fix    = y - base;
    const size_t          tsz      = tsz_v<SZ, SUF>;
    avxvector_t<SZ, SUF>  va, vx1, vx2, vy1, vy2, vytmp;

    // Perform (beta * y)
    if(beta == static_cast<SUF>(0))
    {
        // if beta==0 and y contains any NaNs, we can zero y directly
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = aoclsparse_numeric::zero<SUF>();
    }
    else if(beta != static_cast<SUF>(1))
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    for(aoclsparse_int i = 0; i < m; i++)
    {
        // strictly (L/U) elements in each row are crstart[i]..crend[i]-1
        aoclsparse_int idxstart = crstart[i];
        aoclsparse_int idxend   = crend[i];
        SUF            result   = aoclsparse_numeric::zero<SUF>();

        vy1   = kt_setzero_p<SZ, SUF>();
        vytmp = kt_setzero_p<SZ, SUF>();
        vx2   = kt_set1_p<SZ, SUF>(alpha * x[i]);

        aoclsparse_int j;
        aoclsparse_int nnz   = idxend - idxstart;
        aoclsparse_int k_rem = nnz % tsz;

        // multiply with all strictly (L/U) triangle elements (and their transpose)
        for(j = idxstart; j < idxend - k_rem; j += tsz)
        {
            va = kt_loadu_p<SZ, SUF>(aval_fix + j);

            vx1 = kt_set_p<SZ, SUF>(x_fix, icol_fix + j);
            kt_fmadd_B<SZ, SUF>(va, vx1, vy1, vytmp);
            if constexpr(HERM)
            {
                va = kt_conj_p<SZ, SUF>(va);
            }
            vy2 = kt_mul_p<SZ, SUF>(va, vx2);

            const SUF *vy_cast = reinterpret_cast<const SUF *>(&vy2);
            for(size_t k = 0; k < tsz; k++)
            {
                aoclsparse_int idx = icol_fix[j + k];
                y_fix[idx] += vy_cast[k];
            }
        }

        result = kt_hsum_B<SZ, SUF>(vy1, vytmp);

        for(j = idxend - k_rem; j < idxend; j++)
        {
            aoclsparse_int idx = icol_fix[j];
            result += aval_fix[j] * x_fix[idx];
            if constexpr(HERM)
            {
                y_fix[idx] += alpha * aoclsparse::conj(aval_fix[j]) * x[i];
            }
            else
            {
                y_fix[idx] += alpha * aval_fix[j] * x[i];
            }
        }

        // multiply with diagonal
        if(diag_type == aoclsparse_diag_type_non_unit)
            result += aval_fix[idiag[i]] * x[i];
        else if(diag_type == aoclsparse_diag_type_unit)
            result += x[i];
        // else zero diagonal
        result *= alpha;
        y[i] += result;
    }

    return aoclsparse_status_success;
}

#define CSRMV_TEMPLATE_DECLARATION(BSZ, SUF)                   \
    template aoclsparse_status aoclsparse::csrmv_kt<BSZ, SUF>( \
        aoclsparse_index_base base,                            \
        const SUF             alpha,                           \
        const aoclsparse_int  m,                               \
        const SUF *__restrict__ aval,                          \
        const aoclsparse_int *__restrict__ icol,               \
        const aoclsparse_int *__restrict__ row,                \
        const SUF *__restrict__ x,                             \
        const SUF beta,                                        \
        SUF *__restrict__ y);

#define CSRMVT_TEMPLATE_DECLARATION(BSZ, SUF)                   \
    template aoclsparse_status aoclsparse::csrmvt_kt<BSZ, SUF>( \
        aoclsparse_index_base base,                             \
        const SUF             alpha,                            \
        aoclsparse_int        m,                                \
        aoclsparse_int        n,                                \
        const SUF *__restrict__ aval,                           \
        const aoclsparse_int *__restrict__ icol,                \
        const aoclsparse_int *__restrict__ row,                 \
        const SUF *__restrict__ x,                              \
        const SUF beta,                                         \
        SUF *__restrict__ y);

#define CSRMV_SYMM_TEMPLATE_DECLARATION(BSZ, SUF, HERM)                   \
    template aoclsparse_status aoclsparse::csrmv_symm_kt<BSZ, SUF, HERM>( \
        aoclsparse_index_base base,                                       \
        const SUF             alpha,                                      \
        aoclsparse_int        m,                                          \
        aoclsparse_diag_type  diag_type,                                  \
        aoclsparse_fill_mode  fill_mode,                                  \
        const SUF *__restrict__ aval,                                     \
        const aoclsparse_int *__restrict__ icol,                          \
        const aoclsparse_int *__restrict__ icrow,                         \
        const aoclsparse_int *__restrict__ idiag,                         \
        const aoclsparse_int *__restrict__ iurow,                         \
        const SUF *__restrict__ x,                                        \
        const SUF beta,                                                   \
        SUF *__restrict__ y);

KT_INSTANTIATE(CSRMV_TEMPLATE_DECLARATION, kernel_templates::get_bsz());
KT_INSTANTIATE(CSRMVT_TEMPLATE_DECLARATION, kernel_templates::get_bsz());
KT_INSTANTIATE_EXT(CSRMV_SYMM_TEMPLATE_DECLARATION, kernel_templates::get_bsz(), true);
KT_INSTANTIATE_EXT(CSRMV_SYMM_TEMPLATE_DECLARATION, kernel_templates::get_bsz(), false);
