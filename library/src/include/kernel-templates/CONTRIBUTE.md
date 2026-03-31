# Contributing to Kernel Templates

This guide explains how to extend the Kernel Templates (KT) library with new datatypes,
Level 0 (L0) micro kernels, Level 1 (L1) micro kernels, and how to write tests for your
contributions. For general usage, compilation, and documentation generation, see
[README.md](README.md).

## Architecture Overview

```
kernel_templates.hpp              <- Top-level include; users include only this file
├── kt_common.hpp                 <- Type traits, enums (bsz, fused_op, kt_avxext), SFINAE helpers
├── kt_common_x86.hpp             <- Declarations of all L0/L1 operations + avxvector type database
│                                    (included by all headers below via #include "kt_common_x86.hpp")
├── kt_l0.hpp                     <- Architecture-independent L0 micro kernel definitions
├── kt_l0_sse.hpp                 <- SSE-specific (bsz::b128) L0 definitions only
├── kt_l0_avx2.hpp                <- AVX2-specific (bsz::b256) L0 definitions only
├── kt_l0_avx512.hpp              <- AVX-512-specific (bsz::b512) L0 definitions only
└── kt_l1.hpp                     <- L1 micro kernel definitions (composed from L0 kernels)
```

`kt_common_x86.hpp` is the central **declaration** file. It contains forward declarations
(with Doxygen documentation) for every L0 and L1 operation, with one declaration per ISA
width (`bsz::b128`, `bsz::b256`, `bsz::b512`). It also defines the `avxvector` type
database and size traits. Every header except `kt_common.hpp` includes
`kt_common_x86.hpp` and provides the corresponding **definitions**. Each ISA-specific
header file must contain definitions for its own vector width only.

### Key types and utilities

Before contributing, familiarize yourself with these core types defined in `kt_common.hpp`
and `kt_common_x86.hpp`:

| Type / Utility | Header | Purpose |
|----------------|--------|---------|
| `avxvector_t<SZ, SUF>`      | `kt_common_x86.hpp` | SIMD vector type alias (e.g., `__m256d` for `<b256, double>`) |
| `avxvector_half_t<SZ, SUF>` | `kt_common_x86.hpp` | Half-width vector type alias |
| `bsz`                       | `kt_common.hpp`     | Enum class for vector widths: `b128`, `b256`, `b512` |
| `get_bsz()`                 | `kt_common.hpp`     | Returns `bsz::b512` or `bsz::b256` based on `KT_AVX2_BUILD` |
| `get_kt_ext()`              | `kt_common.hpp`     | Returns `kt_avxext::AVX512F` or `kt_avxext::AVX2` based on build |
| `kt_avxext`                 | `kt_common.hpp`     | Enum for ISA extension levels: `AVX2`, `AVX512F`, `AVX512VL`, etc. |
| `kt_type_is_real<T>`        | `kt_common.hpp`     | True for `float`/`double`, false for complex types |
| `kt_dt<SUF>::base_type`     | `kt_common.hpp`     | Extracts the real base type (e.g., `double` from `cdouble`) |
| `tsz_v<SZ, SUF>`            | `kt_common_x86.hpp` | Number of scalar elements in a vector |
| `valid_kt_int<IS>`          | `kt_common.hpp`     | SFINAE constraint for valid index types (32/64-bit integers) |

### Include guard convention

All sub-headers use an `#error` guard to prevent direct inclusion. When adding a new
header, follow this pattern:

```cpp
#ifndef KERNEL_TEMPLATES_T_HPP
#error "Never use ``kt_my_new_header.hpp'' directly; include ``kernel_templates.hpp'' instead."
#endif
```

Then add the `#include` for the new header in `kernel_templates.hpp`.

## Adding a New Datatype

Adding a new scalar datatype requires changes in `kt_common.hpp` and `kt_common_x86.hpp`.
The following 5 steps use `bfloat16` as a worked example.

### Step 1: Increment `supported_base_t`

In `kt_common.hpp`, increase the count of supported base types by 1:

```cpp
constexpr int supported_base_t = 4; // was 3
```

### Step 2: Add a type checker struct

Create a new `kt_is_base_t_<type>` struct following the existing pattern:

```cpp
template <typename T>
struct kt_is_base_t_bfloat16
{
    constexpr operator bool() const noexcept
    {
        return std::is_same_v<T, bfloat16>;
    }
};
```

### Step 3: Register the type index

In the `type_idx()` function inside the `generator` struct in `kt_common_x86.hpp`, add a
branch returning a unique index for the new type (equal to `supported_base_t - 1`):

```cpp
template <typename T>
constexpr int type_idx()
{
    if constexpr(kt_is_base_t_float<T>())
         return 0;
    else if constexpr(kt_is_base_t_double<T>())
         return 1;
    else if constexpr(kt_is_base_t_int<T>())
         return 2;
    else if constexpr(kt_is_base_t_bfloat16<T>())
         return 3; // new supported_base_t - 1
}
```

### Step 4: Register the vector types

In `kt_common_x86.hpp`, extend the `get_vec_t` type alias to include the SIMD vector
types that correspond to the new datatype for each ISA width:

```cpp
template <bsz SZ, typename SUF, bool HALF>
using get_vec_t = type_switch<index<SZ, SUF, HALF>(),
    __m64, __m64, __m64, __m64,               // 64-bit: float, double, int, bfloat16
    __m128, __m128d, __m128i, __m128bh,        // 128-bit
    __m256, __m256d, __m256i, __m256bh         // 256-bit
#ifdef __AVX512F__
    , __m512, __m512d, __m512i, __m512bh       // 512-bit
#endif
>;
```

### Step 5: Define the packet sizes

In the `get_sz_v()` function in `kt_common_x86.hpp`, add logic for calculating the packet
size and half-packet size for the new datatype.

 * packet size refers to how many scalar elements fit in the vector;
 * half packet size refers to half the amount of the pack size, re: complex numbers.

```cpp
template <typename T, typename SUF, bool isTSZ = false>
constexpr int get_sz_v()
{
    if constexpr(std::is_floating_point<SUF>::value || isTSZ == true
                 || kt_is_base_t_bfloat16<T>())
         return sizeof(T) / sizeof(SUF);
    else
         return ((sizeof(T) / sizeof(SUF)) * 2);
}
```

## Adding a New L0 Micro Kernel

L0 micro kernels ideally map directly to IMM intrinsic instructions, but can also map to a sequence
of intrinsics or even be a sequence of C++ instructions. Where to place any new L0
kernel depends on whether it is architecture-independent or ISA-specific.

### Architecture-independent L0 kernels

Place these in `kt_l0.hpp`. These kernels must work across all supported ISAs.

See `kt_scatter_p` in `kt_l0.hpp` for a reference implementation.

### ISA-specific L0 kernels

ISA-specific L0 kernels require two steps: adding **declarations** in `kt_common_x86.hpp`
and adding **definitions** in the ISA-specific header. A kernel's definition file is
determined by the intrinsics it uses (i.e., the compilation flags it requires), not by
the vector width it operates on.

| Required intrinsics | Definition file | Compilation flags needed |
|---------------------|-----------------|--------------------------|
| SSE | `kt_l0_sse.hpp` | `-msse` (baseline) |
| AVX2 | `kt_l0_avx2.hpp` | `-mavx2 -mfma` |
| AVX-512 | `kt_l0_avx512.hpp` | `-mavx512f`, ... |

#### Step 1: Add declarations in `kt_common_x86.hpp`

`kt_common_x86.hpp` serves as the central declaration file for all L0 operations. Every
new kernel must have **three forward declarations** here (one per ISA width), along with
Doxygen documentation. All ISA-specific headers include `kt_common_x86.hpp`, so these
declarations are visible to all definition files.

```cpp
/**
 * @brief Short description of the operation
 *
 * Detailed description, supported types, and examples.
 */
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                kt_myop_p(const avxvector_t<SZ, SUF> a) noexcept;

template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                kt_myop_p(const avxvector_t<SZ, SUF> a) noexcept;

template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                kt_myop_p(const avxvector_t<SZ, SUF> a) noexcept;
```

Place the Doxygen `/** ... */` comment block above the first declaration only.

#### Step 2: Add definitions in the ISA-specific headers

Each ISA-specific header file contains **only** definitions for its own vector width.
For example, the AVX2 definition of `kt_myop_p` goes exclusively in `kt_l0_avx2.hpp`,
the SSE definition goes exclusively in `kt_l0_sse.hpp`, and the AVX-512 definition goes
exclusively in `kt_l0_avx512.hpp`. Note that in this last file there can be multiple
definitions of `kt_myop` kernel for 256- and 512-bits-wide vectors. E.g. a
variation for 256-bits-wide function that relies on instrinsics defined in `AVX-512VL`.

### Kernel signature pattern

Every L0 kernel definition follows this template signature (shown here for AVX2):

```cpp
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                kt_myop_p(const avxvector_t<SZ, SUF> a) noexcept
{
    if constexpr(kt_is_base_t_float<SUF>())
        return _mm256_<intrinsic>_ps(a);
    else if constexpr(kt_is_base_t_double<SUF>())
        return _mm256_<intrinsic>_pd(a);
    else if constexpr(kt_is_base_t_int<SUF>())
        return _mm256_<intrinsic>_epi<bits>(a);
};
```

Key conventions:
- The function name follows the pattern `kt_<operation>_p` (e.g., `kt_add_p`, `kt_div_p`).
- `KT_FORCE_INLINE` ensures inlining in optimized builds.
- `std::enable_if_t<SZ == bsz::b*>` constrains each overload to its target ISA width.
- Use `if constexpr` branches to dispatch to the correct intrinsic based on the scalar type.
- All L0 kernel functions are `noexcept`.
- The definition's signature must exactly match the declaration in `kt_common_x86.hpp`.

### Handling complex types

Many operations need distinct intrinsic sequences for complex types. The two common
patterns are:

**Pattern A: Base-type dispatch is sufficient.** Operations like `kt_add_p` and `kt_sub_p`
can use the same intrinsic for both real and complex types because the packed add/sub of
interleaved real and imaginary components produces the correct result:

```cpp
if constexpr(kt_is_base_t_float<SUF>())
    return _mm256_add_ps(a, b);     // works for both float and cfloat
else
    return _mm256_add_pd(a, b);     // works for both double and cdouble
```

**Pattern B: Complex types need separate logic.** Operations like `kt_mul_p` require
multi-step intrinsic sequences for complex multiplication. In this case, use
`std::is_same_v` to dispatch to each concrete type individually:

```cpp
if constexpr(std::is_same_v<SUF, double>)
    return _mm256_mul_pd(a, b);
else if constexpr(std::is_same_v<SUF, float>)
    return _mm256_mul_ps(a, b);
else if constexpr(std::is_same_v<SUF, cdouble>)
{
    // Complex double multiplication using shuffle + fmaddsub
    ...
}
else if constexpr(std::is_same_v<SUF, cfloat>)
{
    // Complex float multiplication using shuffle + fmaddsub
    ...
}
```

Some kernels (like `kt_fmadd_p`) delegate to other L0 kernels for complex types rather
than using raw intrinsics:

```cpp
else if constexpr(std::is_same_v<SUF, cdouble>)
    return kt_add_p<bsz::b256, cdouble>(kt_mul_p<bsz::b256, cdouble>(a, b), c);
```

Use `kt_dt<SUF>::base_type` to extract the underlying real type when needed (e.g.,
`kt_pow2_p` uses `kt_mul_p<SZ, base_t>(a, a)` to square only the real components).

### Real-only operations

Some operations (like `kt_max_p`) only apply to real types. Combine the ISA guard with
`kt_type_is_real<SUF>()` in `std::enable_if_t`:

```cpp
template <bsz SZ, typename SUF>
KT_FORCE_INLINE
    std::enable_if_t<SZ == bsz::b256 && kt_type_is_real<SUF>(), avxvector_t<SZ, SUF>>
    kt_max_p(const avxvector_t<SZ, SUF> a, const avxvector_t<SZ, SUF> b) noexcept
{ ... }
```

### Kernels that accept index types

If your kernel takes an index/indirection array parameter, use the `valid_kt_int` SFINAE
helper to constrain it:

```cpp
template <bsz SZ, typename SUF, typename IS, valid_kt_int<IS> = 0>
KT_FORCE_INLINE avxvector_t<SZ, SUF>
                kt_myop_p(const avxvector_t<SZ, SUF> a,
                          const IS *indices) noexcept
{ ... }
```

### Kernels with ISA extension dispatch (`kt_avxext`)

Some kernels (like `kt_maskz_set_p`) need different implementations depending on the
specific ISA extension available (e.g., AVX2 vs AVX512VL). These take an additional
`kt_avxext` template parameter:

```cpp
template <bsz SZ, typename SUF, kt_avxext EXT, int L, typename IS, valid_kt_int<IS> = 0>
KT_FORCE_INLINE
    std::enable_if_t<(SZ == bsz::b256 && EXT == kt_avxext::AVX2), avxvector_t<SZ, SUF>>
    kt_maskz_set_p(const SUF *v, const IS b) noexcept
{ ... }
```

Use `get_kt_ext()` to select the appropriate extension at instantiation time, similar to
how `get_bsz()` is used for vector width.

### B-type L0 kernels

So far all the kernels have `_p` post fix indicating correct algebraic packing, that is
all returned values correctly represent the requested operation. The B-type kernels,
identified with `_B` postfix, break this rule in favor of increasing performance.

An illustrative example, is the complex pair `kt_fmadd_B`-`kt_hsum_B`, the combination
of these pair provide a more performant complex dot-product for complex numbers that
the `kt_fmadd_p`-`kt_hsum_p` pair. By combining these two the overall amount of
internal operations is reduced at the cost that individually each intrinsic does
not perform correctly its operation.

Block-level variants (e.g., `kt_fmadd_B`, `kt_hsum_B`) use a different signature pattern.
They return `void` and take accumulator parameters by reference, allowing optimized
handling of complex types by storing real and imaginary parts separately:

```cpp
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, void>
                kt_fmadd_B(const avxvector_t<SZ, SUF>             a,
                           const avxvector_t<SZ, SUF>             b,
                           avxvector_t<SZ, SUF>                  &c,
                           [[maybe_unused]] avxvector_t<SZ, SUF> &d) noexcept
{ ... }
```

For real types, the `d` parameter is unused (marked `[[maybe_unused]]`). For complex types,
`c` accumulates one component and `d` accumulates the other. Block-level kernel tests use
the `KT_Block_L0` test suite name (see Writing Tests section).

### Example: adding a hypothetical `kt_neg_p` (negate)

#### 1. Add declarations in `kt_common_x86.hpp`

```cpp
/**
 * @brief Negate all elements in a vector
 *
 * @tparam SZ  Vector size (bsz::b128, bsz::b256, or bsz::b512)
 * @tparam SUF Scalar data type
 * @param a Input vector
 * @return Vector with all elements negated
 */
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                kt_neg_p(const avxvector_t<SZ, SUF> a) noexcept;

template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                kt_neg_p(const avxvector_t<SZ, SUF> a) noexcept;

template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                kt_neg_p(const avxvector_t<SZ, SUF> a) noexcept;
```

#### 2. Add the SSE definition in `kt_l0_sse.hpp`

```cpp
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b128, avxvector_t<SZ, SUF>>
                kt_neg_p(const avxvector_t<SZ, SUF> a) noexcept
{
    avxvector_t<SZ, SUF> zero = kt_setzero_p<SZ, SUF>();
    return kt_sub_p<SZ, SUF>(zero, a);
};
```

#### 3. Add the AVX2 definition in `kt_l0_avx2.hpp`

```cpp
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b256, avxvector_t<SZ, SUF>>
                kt_neg_p(const avxvector_t<SZ, SUF> a) noexcept
{
    avxvector_t<SZ, SUF> zero = kt_setzero_p<SZ, SUF>();
    return kt_sub_p<SZ, SUF>(zero, a);
};
```

#### 4. Add the AVX-512 definition in `kt_l0_avx512.hpp`

```cpp
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<SZ == bsz::b512, avxvector_t<SZ, SUF>>
                kt_neg_p(const avxvector_t<SZ, SUF> a) noexcept
{
    avxvector_t<SZ, SUF> zero = kt_setzero_p<SZ, SUF>();
    return kt_sub_p<SZ, SUF>(zero, a);
};
```

## Adding a New L1 Micro Kernel

L1 micro kernels are composed from one or more L0 kernels. They are placed in `kt_l1.hpp`.

### Pattern

```cpp
template <bsz SZ, typename SUF>
KT_FORCE_INLINE SUF kt_myop_p(const avxvector_t<SZ, SUF> a,
                               const avxvector_t<SZ, SUF> b) noexcept
{
    avxvector_t<SZ, SUF> c = kt_mul_p<SZ, SUF>(a, b);
    return kt_hsum_p<SZ, SUF>(c);
};
```

L1 kernels typically do not need `std::enable_if_t` guards because they delegate entirely
to L0 kernels which are already ISA-constrained. See `kt_dot_p` and `kt_cdot_p` in
`kt_l1.hpp` for reference.

## Dual-Compilation Model

KT-based kernels are compiled **twice** to produce both AVX2 and AVX-512 object code from
the same source file:

| Build | Compiler flags | Macro | `get_bsz()` returns | ISA widths available |
|-------|----------------|-------|---------------------|----------------------|
| AVX2 | `-mavx2 -mfma -mno-avx512f` | `-DKT_AVX2_BUILD` | `bsz::b256` | `bsz::b128` (SSE) and `bsz::b256` (AVX2) |
| AVX-512 | `-mavx512f -mavx512vl -mavx512dq` | (none) | `bsz::b512` | `bsz::b512` (AVX-512) |

SSE (`bsz::b128`) kernels are compiled as part of the AVX2 build since AVX2 is a superset
of SSE. There is no separate third compilation for SSE.

The CMake build system handles this automatically. In `tests/unit_tests/CMakeLists.txt`,
the kernel source files listed in the `KT_KERNELS` variable are compiled as two separate
OBJECT libraries:

```cmake
set(KT_KERNELS
    kt_kernels.cpp
    ktlvl2_kernels.cpp
)

add_library(KT_KERNELS_AVX512 OBJECT ${KT_KERNELS})
target_compile_options(KT_KERNELS_AVX512 PRIVATE ${AOCLSPARSE_AVX512_FLAGS})

add_library(KT_KERNELS_AVX2 OBJECT ${KT_KERNELS})
target_compile_options(KT_KERNELS_AVX2 PUBLIC "-mno-avx512f;" -DKT_AVX2_BUILD ...)
```

Both object libraries are appended to `AOCLSPARSE_TEST_DEPENDENCIES` and linked into every
test executable. At runtime, `can_exec_avx512_tests()` checks whether the CPU and binary
support AVX-512 before running 512-bit test cases.

### Using `get_bsz()` in kernel code

Instead of hard-coding `bsz::b256` or `bsz::b512`, use `get_bsz()` to let the
dual-compilation model select the appropriate width automatically:

```cpp
template void my_kernel<get_bsz(), double>(...);
```

Under AVX2 compilation, `get_bsz()` resolves to `bsz::b256`; under AVX-512 it resolves to
`bsz::b512`. Similarly, `get_kt_ext()` resolves to `kt_avxext::AVX2` or
`kt_avxext::AVX512F` respectively.

## Writing Tests

Tests for kernel templates live in `tests/unit_tests/` and follow a three-layer structure:

| Layer | File | Purpose |
|-------|------|---------|
| GTest surface | `kt_tests.cpp` | Declares `TEST()` macros that call test functions |
| Kernel implementations | `kt_kernels.cpp` | Template test functions with the actual test logic; compiled twice (AVX2 + AVX-512) |
| Shared data | `common_data_utils.{h\|cpp}` | Shared helpers like `can_exec_avx512_tests()`, `expect_eq_vec()` |

### Test suite naming convention

| Suite name | Kernel type |
|------------|-------------|
| `KT_L0` | Standard L0 micro kernels |
| `KT_Block_L0` | Block-level L0 variants (`kt_fmadd_B`, `kt_hsum_B`) |
| `KT_L1` | L1 micro kernels (composed from L0) |

### Test skeleton for a new L0 kernel

The following shows the complete pattern for testing a hypothetical `kt_neg_p` kernel.

#### 1. Implement the test function in `kt_kernels.cpp`

Add the test function inside the `TestsKT` namespace:

```cpp
template <bsz SZ, typename SUF>
void kt_neg_p_test()
{
    const size_t         sz = tsz_v<SZ, SUF>;
    avxvector_t<SZ, SUF> result, input;
    SUF                  refs[sz];

    const SUF *data = D.get_data<SUF>();

    input  = kt_loadu_p<SZ, SUF>(data);
    result = kt_neg_p<SZ, SUF>(input);

    for(size_t i = 0; i < sz; i++)
    {
        refs[i] = -data[i];
    }

    auto res_ptr = reinterpret_cast<SUF *>(&result);
    expect_eq_vec(sz, res_ptr, refs);

    if(::testing::Test::HasFailure())
        std::cerr << __func__ << " failing for type: " << get_typename<SUF>() << std::endl;
}
```

#### 2. Add the explicit template instantiation at the bottom of `kt_kernels.cpp`

Use the appropriate `KT_INSTANTIATE_TEST*` macro. Choose based on which types your kernel
supports:

| Macro | Types instantiated |
|-------|--------------------|
| `KT_INSTANTIATE_TEST` | `float`, `double`, `cfloat`, `cdouble` |
| `KT_INSTANTIATE_TEST_REAL` | `float`, `double` only |
| `KT_INSTANTIATE_TEST_INT` | `int32_t`, `int64_t` |
| `KT_INSTANTIATE_TEST_INDEX` | All types x all index types |

```cpp
KT_INSTANTIATE_TEST(TestsKT::kt_neg_p_test);
```

#### 3. Declare and register the test in `kt_tests.cpp`

Add the forward declaration in the `TestsKT` namespace:

```cpp
template <bsz SZ, typename SUF>
void kt_neg_p_test();
```

Then add `TEST()` entries for each ISA width, using the appropriate `CALL_FOR_*` macro:

```cpp
TEST(KT_L0, kt_neg_p_128)
{
    CALL_FOR_ALL_TYPES(kt_neg_p_test, bsz::b128);
}

TEST(KT_L0, kt_neg_p_256)
{
    CALL_FOR_ALL_TYPES(kt_neg_p_test, bsz::b256);
}

TEST(KT_L0, kt_neg_p_512)
{
    if(can_exec_avx512_tests())
    {
        CALL_FOR_ALL_TYPES(kt_neg_p_test, bsz::b512);
    }
}
```

The available `CALL_FOR_*` macros are:

| Macro | Types invoked |
|-------|---------------|
| `CALL_FOR_REAL_TYPES(func, SZ)` | `float`, `double` |
| `CALL_FOR_COMPLEX_TYPES(func, SZ)` | `cfloat`, `cdouble` |
| `CALL_FOR_INT_TYPES(func, SZ)` | `int64_t`, `int32_t` |
| `CALL_FOR_ALL_TYPES(func, SZ)` | Real + complex types |
| `CALL_FOR_ALL_TYPES_AND_INDEX(func, SZ)` | All types x all index types |

#### 4. CMake: no changes needed

Since `kt_kernels.cpp` is already in the `KT_KERNELS` list in `CMakeLists.txt`, new test
functions added to it are automatically compiled for both AVX2 and AVX-512.

### Test skeleton for a new L1 kernel

L1 tests use the same pattern as L0 tests. The only difference is the test suite name:

```cpp
TEST(KT_L1, kt_myop_p_256)
{
    CALL_FOR_ALL_TYPES(kt_myop_p_test, bsz::b256);
}
```

### Test skeleton for a block-level L0 kernel

Block-level kernels use the `KT_Block_L0` test suite and follow the same three-step
pattern as regular L0 tests. The test function signature matches the block kernel
signature (taking `c` and `d` by reference):

```cpp
TEST(KT_Block_L0, kt_myop_B_256)
{
    CALL_FOR_ALL_TYPES(kt_myop_B_test, bsz::b256);
}
```

### Non-templated ISA-specific tests

Some kernels like `kt_maskz_set_p` have ISA-extension-specific overloads that require
non-templated test functions. These are defined under `#ifdef KT_AVX2_BUILD` / `#else`
guards directly in `kt_kernels.cpp`, without using the `KT_INSTANTIATE_TEST*` macros:

```cpp
#ifdef KT_AVX2_BUILD
void kt_myop_128_avx()
{
    // Test bsz::b128 with AVX/AVX2 extension
}

void kt_myop_256_avx()
{
    // Test bsz::b256 with AVX2 extension
}
#else
void kt_myop_256_AVX512vl()
{
    // Test bsz::b256 with AVX512VL extension
}

void kt_myop_512_AVX512f()
{
    // Test bsz::b512 with AVX512F extension
}
#endif
```

In `kt_tests.cpp`, register these with appropriate runtime guards:

```cpp
TEST(KT_L0, kt_myop_256_AVX512VL)
{
    if(can_exec_avx512_tests())
    {
        kt_myop_256_AVX512vl();
    }
}
```

### Test skeleton for a new datatype

When adding a new datatype, extend the existing test infrastructure:

1. Add the new type's test data to `KTTCommonData` in `kt_kernels.cpp`.
2. Extend `get_data<T>()` with a branch for the new type.
3. Add new `CALL_FOR_*` macros if needed, or extend existing ones.
4. Add type-trait verification tests (similar to `kt_base_t_check()`) that validate
   `kt_is_base_t_<type>` returns correct results.
5. Add `avxvector` type and size tests (similar to `kt_types_128()`, `kt_types_256()`,
   `kt_types_512()`) that verify the vector type mappings.

### Running the tests

Build with `BUILD_UNIT_TESTS` enabled:

```bash
cmake -S . -B build -DBUILD_UNIT_TESTS=ON
cmake --build build
cd build
ctest -j${NPROC} --output-on-failure
```

To run only KT tests:

```bash
./tests/unit_tests/kt_tests
```

## Documentation

When adding new kernels or datatypes, include Doxygen-style documentation using `/**`
comment blocks:

```cpp
/**
 * @brief Short description of the kernel operation
 *
 * Detailed description of what the kernel does, including any
 * mathematical formulation.
 *
 * @tparam SZ  Vector size (bsz::b128, bsz::b256, or bsz::b512)
 * @tparam SUF Scalar data type (float, double, cfloat, cdouble)
 *
 * @param a First input vector
 * @param b Second input vector
 * @return Result vector
 */
```

For instructions on generating, viewing, and building the documentation, see the
[Documentation section in README.md](README.md#documentation).
