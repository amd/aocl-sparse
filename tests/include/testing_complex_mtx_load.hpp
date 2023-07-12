/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once
#ifndef TESTING_COMPLEX_LOAD_HPP
#define TESTING_COMPLEX_LOAD_HPP

#include "aoclsparse.hpp"
#include "aoclsparse_arguments.hpp"
#include "aoclsparse_check.hpp"
#include "aoclsparse_flops.hpp"
#include "aoclsparse_gbyte.hpp"
#include "aoclsparse_init.hpp"
#include "aoclsparse_random.hpp"
#include "aoclsparse_reference.hpp"
#include "aoclsparse_test.hpp"
#include "aoclsparse_utility.hpp"

void generate_matrix(aoclsparse_matrix_type                 mat_type,
                     aoclsparse_int                        &m,
                     aoclsparse_int                        &n,
                     aoclsparse_int                        &nnz,
                     std::vector<aoclsparse_int>           &rowid,
                     std::vector<aoclsparse_int>           &colid,
                     std::vector<aoclsparse_float_complex> &val,
                     std::vector<float>                    &real_part,
                     std::vector<float>                    &imag_part)
{
    ofstream mtxf("complex.mtx");
    ofstream mtxf_r("real_part.mtx");
    ofstream mtxf_i("imag_part.mtx");

    if(mat_type == aoclsparse_matrix_type_general)
    {
        aoclsparse_init_coo_matrix(rowid, colid, val, m, n, nnz, aoclsparse_index_base_one);
        mtxf << "%%MatrixMarket matrix coordinate complex general\n";
        mtxf << m << "   " << n << "  " << nnz << std::endl;
        mtxf_r << "%%MatrixMarket matrix coordinate real general\n";
        mtxf_r << m << "   " << n << "  " << nnz << std::endl;
        mtxf_i << "%%MatrixMarket matrix coordinate real general\n";
        mtxf_i << m << "   " << n << "  " << nnz << std::endl;
    }
    else if(mat_type == aoclsparse_matrix_type_symmetric)
    {
        std::cout
            << "Symmetric matrix is currently static, will not use any of the passed parameters\n";
        mtxf << "%%MatrixMarket matrix coordinate complex symmetric\n";
        m = n = 5;
        nnz   = 7;
        mtxf << m << "   " << n << "  " << nnz << std::endl;
        mtxf_r << "%%MatrixMarket matrix coordinate real symmetric\n";
        mtxf_r << m << "   " << n << "  " << nnz << std::endl;
        mtxf_i << "%%MatrixMarket matrix coordinate real symmetric\n";
        mtxf_i << m << "   " << n << "  " << nnz << std::endl;
        rowid.assign({1, 2, 3, 3, 4, 5, 5});
        colid.assign({1, 2, 1, 3, 4, 5, 2});
        val.assign({{1, 0.1},
                    {2.3, -1.1},
                    {2.0, -1.0},
                    {3.0, -2.0},
                    {4.0, 0.0},
                    {5.0, -1.5},
                    {4.0, -3.0}});
    }
    else if(mat_type == aoclsparse_matrix_type_hermitian)
    {
        m = n = 5;
        nnz   = 7;
        std::cout
            << "Hermitian matrix is currently static, will not use any of the passed parameters\n";
        mtxf << "%%MatrixMarket matrix coordinate complex hermitian\n";
        mtxf << m << "   " << n << "  " << nnz << std::endl;
        mtxf_r << "%%MatrixMarket matrix coordinate real symmetric\n";
        mtxf_r << m << "   " << n << "  " << nnz << std::endl;
        mtxf_i << "%%MatrixMarket matrix coordinate real symmetric\n";
        mtxf_i << m << "   " << n << "  " << nnz << std::endl;

        rowid.assign({1, 2, 3, 3, 4, 5, 5});
        colid.assign({1, 2, 1, 3, 4, 5, 2});
        val.assign({{1, 0.1},
                    {2.3, -1.1},
                    {2.0, -1.0},
                    {3.0, -2.0},
                    {4.0, 0.0},
                    {5.0, -1.5},
                    {4.0, -3.0}});
    }
    else
    {
        std::cout << "Error: matrix type not supported\n";
        CHECK_AOCLSPARSE_ERROR(aoclsparse_status_not_implemented);
    }

    aoclsparse_int i;
    for(i = 0; i < nnz; i++)
    {
        real_part[i] = val[i].real;
        imag_part[i] = val[i].imag;
    }

    for(i = 0; i < nnz; i++)
    {
        mtxf << rowid[i] << "   " << colid[i] << "  " << real_part[i] << "  " << imag_part[i]
             << std::endl;
        mtxf_r << rowid[i] << "   " << colid[i] << "  " << real_part[i] << std::endl;
        mtxf_i << rowid[i] << "   " << colid[i] << "  " << imag_part[i] << std::endl;
    }
    mtxf.close();
    mtxf_r.close();
    mtxf_i.close();
}

template <typename T>
void testing_complex_mtx_load(const Arguments &arg)
{
    aoclsparse_int         m        = arg.M;
    aoclsparse_int         n        = arg.N;
    aoclsparse_int         nnz      = arg.nnz;
    aoclsparse_index_base  base     = aoclsparse_index_base_zero;
    aoclsparse_matrix_init mat      = aoclsparse_matrix_file_mtx;
    aoclsparse_matrix_type mat_type = arg.mattypeA;
    bool                   issymm;

    std::vector<aoclsparse_int>           rowid(nnz);
    std::vector<aoclsparse_int>           colid(nnz);
    std::vector<T> val(nnz);
    std::vector<float>                    real_part(nnz);
    std::vector<float>                    imag_part(nnz);

    generate_matrix(mat_type, m, n, nnz, rowid, colid, val, real_part, imag_part);
    std::vector<aoclsparse_int>           csr_row_ptr;
    std::vector<aoclsparse_int>           csr_col_ind;
    std::vector<T> csr_val;

    std::vector<aoclsparse_int> r_csr_row_ptr;
    std::vector<aoclsparse_int> r_csr_col_ind;
    std::vector<float>          r_csr_val;

    std::vector<aoclsparse_int> i_csr_row_ptr;
    std::vector<aoclsparse_int> i_csr_col_ind;
    std::vector<float>          i_csr_val;

    aoclsparse_seedrand();
    std::string filename   = "complex.mtx";
    std::string filename_r = "real_part.mtx";
    std::string filename_i = "imag_part.mtx";
    aoclsparse_init_csr_matrix(
        csr_row_ptr, csr_col_ind, csr_val, m, n, nnz, base, mat, filename.c_str(), issymm, true);
    aoclsparse_init_csr_matrix(r_csr_row_ptr,
                               r_csr_col_ind,
                               r_csr_val,
                               m,
                               n,
                               nnz,
                               base,
                               mat,
                               filename_r.c_str(),
                               issymm,
                               true);
    aoclsparse_init_csr_matrix(i_csr_row_ptr,
                               i_csr_col_ind,
                               i_csr_val,
                               m,
                               n,
                               nnz,
                               base,
                               mat,
                               filename_i.c_str(),
                               issymm,
                               true);

    // validate the correctness of the complex matrix in CSR format
    // compare the real and imaginary parts of complex data with
    // "real" data and "imaginary" data respectively.
    // Assumption: loading mtx and conversion to csr works for the real cases
    aoclsparse_int i, j;
    bool           ok = true;

    for(i = 0; i < m; i++)
    {
        for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++)
        {
            if((mat_type == aoclsparse_matrix_type_hermitian) && (csr_col_ind[j] > i))
            {
                ok = ok && (csr_val[j].real == r_csr_val[j])
                     && (csr_val[j].imag == -(i_csr_val[j]));
            }
            else
            {
                ok = ok && ((csr_val[j].real == r_csr_val[j]) && (csr_val[j].imag == i_csr_val[j]));
            }
        }
    }

    if(!ok)
    {
        std::cout << "Error in either loading complex mtx or conversion to CSR\n";
    }
    else
    {
        std::cout << "Successfully loaded the complex mtx file and converted it to CSR\n";
    }
}

#endif
