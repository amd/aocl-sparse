/* ************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_BSRMV_BUILDER
#define AOCLSPARSE_BSRMV_BUILDER

#include "aoclsparse_kernel_templates.hpp"

namespace bsrmv_builder
{
    using namespace kernel_templates;

    /*
    *  Scales individual elements in an
    *  array by a constant
    *
    *  Template parameters
    *  -------------------
    *  SUF    - datatype
    *  BLK_SZ - BSRMV block size (length of array)
    *
    *  Input parameters
    *  -----------------
    *  Sum   - Accumulation array
    *  Alpha - Alpha constant
    */
    template <typename SUF, size_t BLK_SZ>
    static void alpha_scal(SUF *sum, SUF alpha)
    {
        if constexpr(BLK_SZ > 1)
        {
            bsrmv_builder::alpha_scal<SUF, BLK_SZ - 1>(sum + 1, alpha);
        }

        *sum = (*sum) * alpha;
    }

    template <bsz SZ, typename SUF, size_t BLK_SZ>
    static void alpha_scal_v(avxvector_t<SZ, SUF> *sum, avxvector_t<SZ, SUF> alpha)
    {
        if constexpr((BLK_SZ / tsz_v<SZ, SUF>) > 1)
        {
            bsrmv_builder::alpha_scal_v<SZ, SUF, BLK_SZ - tsz_v<SZ, SUF>>(sum + 1, alpha);
        }

        *sum = kt_mul_p<SZ, SUF>(*sum, alpha);
    }

    /*
    *  Performs sum += beta * y for a block iteration
    *
    *  Template parameters
    *  -------------------
    *  SUF    - datatype
    *  BLK_SZ - Block size
    *
    *  Input parameters
    *  -----------------
    *  Sum        - Accumulation array
    *  Beta       - Beta constant
    *  y          - Out vector of BSRMV
    */
    template <typename SUF, size_t BLK_SZ>
    static void beta_scal(SUF *sum, const SUF beta, SUF *y)
    {
        if constexpr(BLK_SZ > 1)
        {
            bsrmv_builder::beta_scal<SUF, BLK_SZ - 1>(sum, beta, y);
        }

        *(sum + (BLK_SZ - 1)) = *(sum + (BLK_SZ - 1)) + beta * *(y + (BLK_SZ - 1));
    }

    template <bsz SZ, typename SUF, size_t BLK_SZ>
    static void beta_scal_v(avxvector_t<SZ, SUF> *sum, avxvector_t<SZ, SUF> beta, SUF *y)
    {
        if constexpr((BLK_SZ / tsz_v<SZ, SUF>) > 1)
        {
            beta_scal_v<SZ, SUF, BLK_SZ - tsz_v<SZ, SUF>>(sum + 1, beta, y + tsz_v<SZ, SUF>);
        }
        *sum = kt_fmadd_p<SZ, SUF>(kt_loadu_p<SZ, SUF>(y), beta, *sum);
    }

    /*
    *  Performs y = sum for a block iteration
    *
    *  Template parameters
    *  -------------------
    *  SUF    - datatype
    *  BLK_SZ - Block size
    *
    *  Input parameters
    *  -----------------
    *  Sum        - Accumulation array
    *  y          - Out vector of BSRMV
    */
    template <typename T, size_t BLK_SZ>
    static void store_res(T *sum, T *y)
    {
        if constexpr(BLK_SZ > 1)
        {
            bsrmv_builder::store_res<T, BLK_SZ - 1>(sum, y);
        }

        y[BLK_SZ - 1] = *(sum + (BLK_SZ - 1));
    }

    template <bsz SZ, typename SUF, size_t BLK_SZ>
    static void store_res_v(avxvector_t<SZ, SUF> *sum, SUF *y)
    {
        if constexpr(BLK_SZ / tsz_v<SZ, SUF> > 1)
        {
            store_res_v<SZ, SUF, BLK_SZ - tsz_v<SZ, SUF>>(sum + 1, y + tsz_v<SZ, SUF>);
        }

        kt_storeu_p<SZ, SUF>(y, *sum);
    }

    /*
    *  Performs sum += bsr_val_ptr * x_ptr for a row
    *
    *  Template parameters
    *  -------------------
    *  SUF    - datatype
    *  BLK_SZ - Block size
    *  ROW    - Row dimension (Same as block size for square kernels)
    *
    *  Input parameters
    *  -----------------
    *  Sum         - Accumulation array
    *  bsr_val_ptr - Pointer to matrix block
    *  x_ptr       - Pointer to vector
    */
    template <typename T, size_t ROW>
    static void row_compute(T *sum, const T *bsr_val_ptr, const T *x_ptr)
    {
        if constexpr(ROW > 1)
        {
            bsrmv_builder::row_compute<T, ROW - 1>(sum, bsr_val_ptr, x_ptr);
        }
        *(sum + (ROW - 1)) = *(bsr_val_ptr + (ROW - 1)) * (*x_ptr) + *(sum + (ROW - 1));
    };

    template <bsz SZ, typename SUF, size_t ROW>
    static void row_compute_v(avxvector_t<SZ, SUF> *sum,
                              const SUF            *bsr_val_ptr,
                              const SUF            *x_ptr,
                              avxvector_t<SZ, SUF> *mv,
                              avxvector_t<SZ, SUF> *xv)
    {
        if constexpr(ROW > tsz_v<SZ, SUF>)
        {
            bsrmv_builder::row_compute_v<SZ, SUF, ROW - tsz_v<SZ, SUF>>(
                sum, bsr_val_ptr + tsz_v<SZ, SUF>, x_ptr, mv + 1, xv + 1);
        }
        *mv = kt_loadu_p<SZ, SUF>(bsr_val_ptr);
        *xv = kt_set1_p<SZ, SUF>(*x_ptr);

        *mv  = kt_mul_p<SZ, SUF>(*mv, *xv);
        *sum = kt_add_p<SZ, SUF>(*sum, *mv);
    };

    template <bsz SZ, typename SUF, size_t ROW>
    static void row_compute_v_fma(avxvector_t<SZ, SUF> *sum,
                                  const SUF            *bsr_val_ptr,
                                  const SUF            *x_ptr,
                                  avxvector_t<SZ, SUF> *mv,
                                  avxvector_t<SZ, SUF> *xv)
    {
        if constexpr(ROW > 1)
        {
            bsrmv_builder::row_compute_v<SZ, SUF, ROW - 1>(
                sum + 1, bsr_val_ptr + tsz_v<SZ, SUF>, x_ptr + 1, mv + 1, xv + 1);
        }
        *mv = kt_loadu_p<SZ, SUF>(bsr_val_ptr);
        *xv = kt_set1_p<SZ, SUF>(*x_ptr);

        *sum = kt_fmadd_p<SZ, SUF>(*mv, *xv, *sum);
    };

    /*
    *  Performs sum += bsr_val_ptr * x_ptr for a block iteration
    *
    *  Template parameters
    *  -------------------
    *  SUF    - datatype
    *  BLK_SZ - Block size
    *  ROW    - Row dimension (Same as block size for square kernels)
    *
    *  Input parameters
    *  -----------------
    *  Sum         - Accumulation array
    *  bsr_val_ptr - Pointer to matrix block
    *  x_ptr       - Pointer to vector
    */
    template <typename T, size_t BLK_SZ, size_t ROW>
    static void compute(T *sum, const T *bsr_val_ptr, const T *x_ptr)
    {

        if constexpr(BLK_SZ > 1)
        {
            bsrmv_builder::compute<T, BLK_SZ - 1, ROW>(sum, bsr_val_ptr + ROW, x_ptr + 1);
        }

        bsrmv_builder::row_compute<T, ROW>(sum, bsr_val_ptr, x_ptr);
    };

    // Helper template for processing column chunks recursively
    template <bsz SZ, typename SUF, size_t BLK_SZ, size_t CHUNK>
    static void compute_column_chunks(avxvector_t<SZ, SUF> *sum, const SUF *col_ptr, SUF x_val)
    {
        // Will be computed at compile time
        constexpr int vec_idx = (BLK_SZ - CHUNK) / tsz_v<SZ, SUF>;

        if constexpr(CHUNK >= tsz_v<SZ, SUF>)
        {
            // Process remaining chunks
            if constexpr(CHUNK > tsz_v<SZ, SUF>)
            {
                compute_column_chunks<SZ, SUF, BLK_SZ, CHUNK - tsz_v<SZ, SUF>>(sum, col_ptr, x_val);
            }

            // Process current chunk
            avxvector_t<SZ, SUF> col_vec = kt_loadu_p<SZ, SUF>(col_ptr + BLK_SZ - CHUNK);
            avxvector_t<SZ, SUF> x_vec   = kt_set1_p<SZ, SUF>(x_val);
            sum[vec_idx]                 = kt_fmadd_p<SZ, SUF>(col_vec, x_vec, sum[vec_idx]);
        }
    }

    // Currently the blocks are assumed to be square by default. Hence COLS = BLK_SZ
    template <bsz SZ, typename SUF, size_t BLK_SZ, size_t COLS = BLK_SZ>
    static void compute_v(avxvector_t<SZ, SUF> *sum, const SUF *bsr_val_ptr, const SUF *x_ptr)
    {
        // Process all BLK_SZ columns recursively
        if constexpr(COLS > 0)
        {
            constexpr size_t col = BLK_SZ - COLS;

            // Process remaining columns
            if constexpr(COLS > 1)
            {
                compute_v<SZ, SUF, BLK_SZ, COLS - 1>(sum, bsr_val_ptr, x_ptr);
            }

            // Process current column
            compute_column_chunks<SZ, SUF, BLK_SZ, BLK_SZ>(
                sum, bsr_val_ptr + col * BLK_SZ, x_ptr[col]);
        }
    };

    template <bsz SZ, typename SUF, size_t BLK_SZ>
    static void setzero_v(avxvector_t<SZ, SUF> *sum)
    {
        if constexpr(BLK_SZ / tsz_v<SZ, SUF> > 1)
        {
            bsrmv_builder::setzero_v<SZ, SUF, BLK_SZ - tsz_v<SZ, SUF>>(sum + 1);
        }

        *sum = kt_setzero_p<SZ, SUF>();
    };
}
#endif