# Reusable Kernel Template (KT) Library

<img src="kt_logo_blk.png" width="75" height="75">
<p align="justify"> Reusable kernel templates is a stand alone library implemented in modern C++ as a header only templated framework. The framework is designed for rapid development of high-performant AVX-vectorized kernels
specialized on basic mathematical operations. The main purpose of the framework is to allow for a single abstract implementation of a kernel to be instantiated for any datatype and ISA combination. It leverages
ISA and datatype combination to fully exploit the hardware
resources. KT support 4 different datatypes
(<code>float</code>, <code>double</code>, <code>std::complex&lt;float&gt;</code> and <code>std::complex&lt;double&gt;</code>) for
2 different ISAs (AVX2 and AVX512)</p>

## Purpose

<p align="justify">Numerical libraries such as the
AOCL portfolio are widely used in traditional HPC workloads
and simulation software packages that rely on both real and
complex algebra. In most cases, the implementation of the
mathematical operation changes based on the datatype (input,
output, or intermediate representation types). The number of implementations (E) required for a single mathematical operation amount to the combination of number of supported ISA types (S) along the
number of supported datatypes (D). E = S * D, meaning that if a
new ISA AVX extension is introduced, then E increases by a
factor of D.</p>

<p align="justify">KT provides a paradigm for abstraction
that enables the a library to provide a generalization of
an algorithm without considering the nuances between the
datatype, the ISA instructions, and the mathematical
implementation of the algorithm. This abstraction means that
the algorithm (kernel) is implemented and debugged only once. Highly-performant kernels for the varying datatypes and ISA combinations are instantiated from this single implementation. </p>

# Design

The Kernel Templates are divided into two levels of "μkernels" (micro kernels) and are classified as
 - Level 0 micro kernels, these expand directly to IMM intrinsic instructions, while
 - Level 1 micro kernels, make use of one or more level 0 micro kernels.

## List of operations supported
Table Reusable Kernel Templates μkernels list.
|Level	| Name |	Operation |
|---|---|---|
|0	|kt_setzero_p|	Zero out a vector|
|0	|kt_set1_p|	Fill vector with a scalar value|
|0 |kt_set_p|	Set to register with indirect memory access|
|0	|kt_maskz_set_p|	Unaligned load with zero mask|
|0	|kt_load_p|	Aligned load to register|
|0	|kt_loadu_p|	Unaligned load to register|
|0	|kt_storeu_p|	Store vector to memory location|
|0	|kt_add_p|	Vector addition|
|0	|kt_sub_p|	Vector subtraction|
|0	|kt_mul_p|	Vector product|
|0	|kt_fmadd_p|	Vector fused multiply-add|
|0	|kt_fmsub_p|	Vector fused multiply-subtract|
|0	|kt_hsum_p|	Horizontal sum (reduction)|
|0	|kt_conj_p|	Conjugate the contents register|
|1	|kt_dot_p|	Dot-product of two vectors|
|1	|kt_cdot_p|	Conjugate dot-product|



## Using Kernel Templates

<p align="justify">As an example, taking the Sparse BLAS L1 axpyi operation for  (<a href="https://docs.amd.com/r/en-US/63865-AOCL-sparse/Sparse-BLAS-level-1-2-and-3-functions" target="_blank">AOCL-Sparse documentation for axpyi</a>) has been implemented. </p>

### Kernel source file
```cpp
//FILENAME: kt_example_axpyi.cpp
#include <cstdint>

// Set the kt_int
using kt_int_t = int64_t; // or int32_t

#include "kernel-templates/kernel_templates.hpp"

using namespace kernel_templates;

// Kernel
template <bsz SZ, typename SUF>
void axpyi_kt(int64_t nnz,
              SUF a,
              const SUF *x,
              const int64_t *indx,
              SUF *y)
{
    const size_t tsz = tsz_v<SZ, SUF>;
    size_t       i   = 0;
    avxvector_t<SZ, SUF> xvec, yvec;
    avxvector_t<SZ, SUF> alpha = kt_set1_p<SZ, SUF>(a);

    for(; (i + (tsz - 1)) < nnz; i += tsz)
    {

        xvec = kt_loadu_p<SZ, SUF>(x + i);
        yvec = kt_set_p<SZ, SUF>(y, &indx[i]);

        yvec = kt_fmadd_p<SZ, SUF>(alpha, xvec, yvec);

        kt_scatter_p<SZ, SUF>(yvec, y, indx + i);
    }

    for(; i < nnz; i++)
    {
        y[indx[i]] = a * x[i] + y[indx[i]];
    }
}

#ifdef KT_AVX2_BUILD
// Instantiate for double and 256-bit vector
template void axpyi_kt<bsz::b256, double>(int64_t nnz,
                                          double a,
                                          const double * x,
                                          const int64_t *indx,
                                          double *y);
#else
// Instantiate for double and 512-bit vector
template void axpyi_kt<bsz::b512, double>(int64_t nnz,
                                          double a,
                                          const double * x,
                                          const int64_t *indx,
                                          double *y);
#endif
```

The above `#ifdef` can be removed by using the auxiliary function
`get_bsz()` that return the `bsz::b*` based on the presence of `KT_AVX2_BUILD`
so a single declaration can be used:
```cpp
// Instantiate for double and ISA is based on compilation flags
template void axpyi_kt<get_bsz(), double>(int64_t nnz,
                                          double a,
                                          const double * x,
                                          const int64_t *indx,
                                          double *y);
```

### Compiling the source file

<p align="justify">To generate KT-based kernels for N ISAs, the source needs to be compiled N times. AVX2 and AVX512 variants of the above `axpyi` kernel can be generated as follows:</p>

#### AVX2 object
Compile with AVX2 flags and defile the macro `KT_AVX2_BUILD` (to indicate that the source is for AVX2 ISA). Optionally, to ensure that no AVX512 intrinsics are allowed, add `-mno-avx512f`. This will produce the AVX2 version of kernel. Using GCC on Linux:
```bash
g++ -mavx2 -mfma -mno-avx512f -DKT_AVX2_BUILD -c kt_example_axpyi.cpp -o kt_axpyi_avx2.o # Compile the source with AVX2 flag(s)
```

#### AVX512 object
Compile with AVX512 flags This will produce an object file with AVX512 version of KT-based kernel.
```bash
g++ -mavx512f -mavx512vl -mavx512dq -c kt_example_axpyi.cpp -o kt_axpyi_avx512.o # Compile the source with AVX512
```

#### Example driver
This small example illustrates how to call the AVX2 and AVX512 variants of the kernels.

#### Header containing the kernel's declaration
```cpp
//FILENAME: kt_axpyi.hpp
#include "kernel-templates/kernel_templates.hpp"

using namespace kernel_templates;

// Forward declaration of kernel templates function
template <bsz SZ, typename SUF>
void axpyi_kt(int64_t nnz,
              SUF a,
              const SUF *x,
              const int64_t *indx,
              SUF *y);

```
#### Source containing the kernel calls
```cpp
//FILENAME: compare_kernels.cpp
#include <iostream>
#include <cstdlib>
#include "kt_axpyi.hpp"

// This application calls AVX2 and AVX512 KT-based kernels for axpyi
// and checks if both kernels return the same result
int main()
{
    using namespace kernel_templates;

    double alpha = 2;
    int64_t nnz = 10;
    bool is_equal = true;

    int64_t index[10] = {2, 9, 1, 16, 4, 3, 10, 8, 12, 14};

    double y1[17], y2[17], x[10];

    for(int i = 0; i < nnz; ++i)
    {
        y1[index[i]] = rand();
        x[i] = rand();
    }

    for(int i = 0; i < 17; ++i)
    {
        y2[i] = y1[i];
    }

    axpyi_kt<bsz::b512, double>(nnz, alpha, x, index, y1);
    axpyi_kt<bsz::b256, double>(nnz, alpha, x, index, y2);

    for(int i = 0; i < 17; ++i)
    {
        if(y2[i] != y1[i])
        {
            is_equal = false;
            break;
        }
    }

    if(is_equal)
        std::cout<<"Both kernels return the same result"<<std::endl;
    else
        std::cout<<"Both kernels return different results"<<std::endl;

    return 0;
}

```
#### Compiling and linking the example

The above example can be compiled as follows:

```bash
g++ -o compare_kernels compare_kernels.cpp kt_axpyi_avx2.o kt_axpyi_avx512.o  # Generate the executable for the application
```