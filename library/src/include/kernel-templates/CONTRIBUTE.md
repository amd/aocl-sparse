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
├── kt_l0_avx512fp16.hpp          <- AVX-512 FP16 (`_Float16`/`fp16`) L0 definitions for all
│                                    widths; included only when `__AVX512FP16__` is defined
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
| `kt_avxext`                 | `kt_common.hpp`     | Enum for ISA extension levels: `AVX2`, `AVX512F`, `AVX512VL`, `AVX512FP16`, etc. |
| `fp16`                      | `kt_common.hpp`     | Half-precision scalar type; alias for the native `_Float16` when `__AVX512FP16__` is defined, otherwise a minimal SFINAE-only tag type |
| `kt_is_base_t_fp16<T>`      | `kt_common.hpp`     | True when `T` is `fp16`/`_Float16` |
| `kt_type_is_real<T>`        | `kt_common.hpp`     | True for `float`/`double`/`fp16`, false for complex types |
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
The worked example below is the **actual `fp16` (half-precision) implementation** currently
in the sources, copied verbatim. Lines marked with `// <-- fp16` are the additions required
for the datatype.

Because `fp16` maps to the native `_Float16`, which for the purposes of this library is considered to
only exists when the compiler defines
`__AVX512FP16__`, its additions are guarded by that macro. A datatype that is always
available would omit those `#ifdef __AVX512FP16__` guards.

### Step 0: Define the scalar type

`fp16` is a type alias rather than a plain existing type, so it is first defined in
`kt_common.hpp`. When `__AVX512FP16__` is absent it degrades to a minimal SFINAE-only tag:

```cpp
#ifdef __AVX512FP16__
    using fp16 = _Float16;
#else
    struct fp16
    {
    };
#endif
```

### Step 1: Increment `supported_base_t`

In `kt_common.hpp`, increase the count of supported base types by 1 (verbatim):

```cpp
    /*
     *   Number of supported "base" types: 4
     *
     * 1. float (and cfloat) maps to float intrinsics
     * 2. double (and cdouble) maps to double intrinsics
     * 3. int (int32_t and int64_t) maps to integer intrinsics
     * 4. fp16 maps to half-precision intrinsics                 // <-- fp16
     * Add new type here and update the supported_base_t accordingly.
     */
    constexpr int supported_base_t = 4;
```

### Step 2: Add a type checker struct

Create a new `kt_is_base_t_<type>` struct in `kt_common.hpp` following the existing pattern
(verbatim):

```cpp
    template <typename T>
    struct kt_is_base_t_fp16
    {
        constexpr operator bool() const noexcept
        {
            return std::is_same<T, fp16>::value;
        }
    };
```

### Step 3: Register the type index

In the `index_t()` function inside the `generator` namespace in `kt_common_x86.hpp`, add a
branch returning a unique index for the new type (equal to `supported_base_t - 1`). Verbatim:

```cpp
        template <typename T>
        constexpr int index_t()
        {
            if constexpr(kt_is_base_t_float<T>())
                return 0;
            else if constexpr(kt_is_base_t_double<T>())
                return 1;
            else if constexpr(kt_is_base_t_int<T>())
                return 2;
#ifdef __AVX512FP16__                              // <-- fp16
            else if constexpr(kt_is_base_t_fp16<T>())
                return 3;
#endif
            return -1; // unsupported type
        }
```

### Step 4: Register the vector types

In `kt_common_x86.hpp`, extend the `get_vec_t` type database with the SIMD vector types for
the new datatype at each ISA width. A conditionally-compiled type is appended through a
`KT_IF_xxx` helper macro so the column collapses away when the type is unavailable. Verbatim:

```cpp
// Conditional type-list macros: expand to ", type" when the ISA flag is
// defined, or to nothing otherwise.  Collapses get_vec_t into one definition.
#if defined(__AVX512FP16__) && defined(__AVX512F__)   // <-- fp16
#define KT_IF_FP16(x) , x
#else
#define KT_IF_FP16(x)
#endif
```

```cpp
        // index_t                                            float     double     int            half/fp16
        template <bsz SZ, typename SUF, bool HALF>
        using get_vec_t = type_switch<index<SZ, SUF, HALF>(), __m64,    __m64,   __m64   KT_IF_FP16(  __m64),
                                                              __m128, __m128d, __m128i   KT_IF_FP16(__m128h),
                                                              __m256, __m256d, __m256i   KT_IF_FP16(__m256h)
                                                 KT_IF_AVX512(__m512, __m512d, __m512i   KT_IF_FP16(__m512h))>;
```

The `index()` helper in the same file already adapts the row/column arithmetic for the
compiled-out column via `cols = supported_base_t - 1` on the non-`__AVX512FP16__` path, so
no further change is needed there.

### Step 5: Define the packet sizes

The `get_sz_v()` function in `kt_common_x86.hpp` computes the packet size, half-packet size
and type size. It detects complex types via `kt_dt<SUF>::base_type`, so a new **real** type
like `fp16` requires no change here (it is handled by the `base_type` branch). Verbatim:

 * packet size refers to how many scalar elements fit in the vector;
 * half packet size refers to half the amount of the pack size, re: complex numbers.

```cpp
        template <typename T, typename SUF, bool isTSZ = false>
        constexpr int get_sz_v()
        {
            if constexpr(isTSZ || std::is_same_v<SUF, typename kt_dt<SUF>::base_type>)
                return sizeof(T) / sizeof(SUF);
            else
                return (sizeof(T) / sizeof(SUF)) * 2;
        }
```

> Note: the in-source guide comment block at the top of `kt_common.hpp` still uses a
> hypothetical `bfloat16` and the older `type_idx`/`get_sz_v` shapes; the steps above reflect
> the current code (`index_t`, the `KT_IF_FP16` macro, and `kt_dt`-based `get_sz_v`).

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
| AVX-512 FP16 (`fp16`) | `kt_l0_avx512fp16.hpp` | `-mavx512fp16` |

All `fp16` (`_Float16`) kernels are collected in `kt_l0_avx512fp16.hpp` regardless of vector
width, because every one of them uniformly requires the `__AVX512FP16__` ISA extension. The
file is included by `kernel_templates.hpp` only when `__AVX512FP16__` is defined, and the
non-fp16 headers above exclude `fp16` via SFINAE (`!std::is_same_v<SUF, fp16>`) so there is
no overload ambiguity. See [Handling fp16 (half precision)](#handling-fp16-half-precision).

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

If the kernel also supports `fp16`, the three width-specific declarations above must exclude
it (`!std::is_same_v<SUF, fp16>`) and a single additional declaration must cover all widths
for `fp16`, dispatching internally on `SZ`:

```cpp
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                kt_myop_p(const avxvector_t<SZ, SUF> a) noexcept;
```

#### Step 2: Add definitions in the ISA-specific headers

Each ISA-specific header file contains **only** definitions for its own vector width.
For example, the AVX2 definition of `kt_myop_p` goes exclusively in `kt_l0_avx2.hpp`,
the SSE definition goes exclusively in `kt_l0_sse.hpp`, and the AVX-512 definition goes
exclusively in `kt_l0_avx512.hpp`. Note that in this last file there can be multiple
definitions of `kt_myop` kernel for 256- and 512-bits-wide vectors. E.g. a
variation for 256-bits-wide function that relies on instrinsics defined in `AVX-512VL`.

The `fp16` definition is the exception: all of its widths go together in `kt_l0_avx512fp16.hpp`,
as a single overload selected by `std::is_same_v<SUF, fp16>` that branches on `SZ`
internally (see [Handling fp16 (half precision)](#handling-fp16-half-precision)).

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

### Handling fp16 (half precision)

`fp16` (an alias for the native `_Float16`) is treated as a real floating-point type, but
its intrinsics (`_mm*_ph`, e.g. `_mm512_add_ph`) require the `__AVX512FP16__` ISA extension,
which is independent of the regular `bsz` width split. Consequently `fp16` is handled
differently from the other scalar types:

- **Declarations** (`kt_common_x86.hpp`): the per-width overloads carry
  `!std::is_same_v<SUF, fp16>` so they never match `fp16`, and one extra overload guarded by
  `std::is_same_v<SUF, fp16>` covers all widths.
- **Definitions** (`kt_l0_avx512fp16.hpp`): a single definition handles `b128`/`b256`/`b512` with
  an internal `if constexpr(SZ == bsz::b128/b256/b512)` dispatch, mapping to the
  corresponding `_mm_*_ph` / `_mm256_*_ph` / `_mm512_*_ph` intrinsic.
- **Conditional compilation**: `kt_l0_avx512fp16.hpp` is included only when `__AVX512FP16__` is
  defined. When it is not, `fp16` falls back to a minimal SFINAE-only tag type, `type_idx()`
  maps it to `-1` (unsupported), and no `fp16` kernels are instantiated.
- **Type traits**: use `kt_is_base_t_fp16<SUF>()` (and `kt_type_is_real<SUF>()`, which is
  true for `fp16`) when dispatching. The `_p` real-only guards that must *exclude* `fp16`
  (e.g. integer or complex-only paths) add `&& !std::is_same_v<SUF, fp16>`.

```cpp
// kt_l0_avx512fp16.hpp: one overload, all widths, gated by __AVX512FP16__ at include time
template <bsz SZ, typename SUF>
KT_FORCE_INLINE std::enable_if_t<std::is_same_v<SUF, fp16>, avxvector_t<SZ, SUF>>
                kt_myop_p(const avxvector_t<SZ, SUF> a) noexcept
{
    if constexpr(SZ == bsz::b128)
        return _mm_<intrinsic>_ph(a);
    else if constexpr(SZ == bsz::b256)
        return _mm256_<intrinsic>_ph(a);
    else if constexpr(SZ == bsz::b512)
        return _mm512_<intrinsic>_ph(a);
}
```

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
of these pair provide a more performant complex dot-product for complex numbers than
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

When the compiler supports it (`COMPILER_SUPPORTS_AVX512FP16`), the AVX-512 build also
receives `-mavx512fp16`, which defines `__AVX512FP16__` and enables the `_Float16`
(half-precision) kernels and their tests. When it is absent, the `_Float16` paths are
compiled out and the corresponding tests are registered but skipped at runtime. So the AVX2
build provides the `b128`/`b256` drivers and the AVX-512 build provides the `b512` drivers
plus the `_Float16` rows.

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
test executable. At runtime, `can_exec_avx512_tests()` and / or `can_exec_avx512fp16_test()` checks
whether the CPU and binary support AVX-512 before running 512-bit test cases.

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

Tests for kernel templates live in `tests/unit_tests/` and are spread across four files:

| Layer | File | Purpose |
|-------|------|---------|
| GTest surface | `kt_tests.cpp` | Forward-declares the test drivers and registers one `TEST()` per driver, with runtime ISA guards. Compiled once, with generic flags. |
| Kernel list | `kt_kernels.hpp` | Re-includable list of every `(driver x width x type)` combination, expressed through `KT_TEST_DO_*` macros. Holds no logic of its own. |
| Kernel implementations | `kt_kernels.cpp` | Template test functions with the actual test logic, plus the driver definitions generated from `kt_kernels.hpp`. Compiled twice (AVX2 + AVX-512). |
| Shared data | `common_data_utils.{h\|cpp}` | Shared helpers like `can_exec_avx512_tests()`, `can_exec_avx512fp16_tests()`, `expect_eq_vec()` |

### Test suite naming convention

All generated kernel-driver tests are registered under a single GoogleTest suite:

| Suite name | Tests |
|------------|-------|
| `KT_TEST` | Every kernel driver generated from `kt_kernels.hpp` (`kt_loadu_p_test`, `kt_fmadd_p_test`, ...). The individual test name encodes the width and type, e.g. `KT_TEST.kt_loadu_p_test_b256_double`. |
| `KT_TYPE` | Hand-written type-trait and `avxvector` database tests (`kt_base_t_check`, `kt_types_128`, ...). |

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

#### 2. Register the driver in `kt_kernels.hpp`

Add one line to the list inside the `TestsKT` namespace in `kt_kernels.hpp`, using the
macro that matches the type set your kernel supports:

| Macro | Combinations generated |
|-------|------------------------|
| `KT_TEST_DO_REAL_COMPLEX(func)` | `float`, `double`, `cfloat`, `cdouble` (plus `_Float16`) over the widths of the build |
| `KT_TEST_DO_REAL(func)` | `float`, `double` (plus `_Float16`) over the widths of the build |
| `KT_TEST_DO_INTEGER(func)` | `int32_t`, `int64_t` |
| `KT_TEST_DO_INDEX(func)` | every supported `(type, index-type)` pair, index type in `{int32_t, int64_t, uint32_t, uint64_t}` |

```cpp
KT_TEST_DO_REAL_COMPLEX(kt_neg_p_test);
```

This single line is the only place the combination list is declared. It is consumed by both
`kt_kernels.cpp` (which emits the driver *definitions*) and `kt_tests.cpp` (which emits the
matching forward declarations and `TEST()` registrations) - see
[How the driver list is expanded](#how-the-driver-list-is-expanded). Do not write any
`TEST()` blocks or forward declarations by hand; they are generated for you. The widths and
the `_Float16` rows are selected automatically by the build (AVX2 vs AVX-512); you never
list `b128`/`b256`/`b512` explicitly.

#### 3. `kt_tests.cpp` and CMake: no changes needed

Unless adding a new type (e.g. `bfloat16`) and it's `KT_TYPE` tests, these files don't
need any changes.

`kt_tests.cpp` already includes `kt_kernels.hpp` to generate the registrations, and
`kt_kernels.cpp` is already in the `KT_KERNELS` list in `CMakeLists.txt`, so the new driver
is compiled for both AVX2 and AVX-512 and registered automatically.

### How the driver list is expanded

`kt_kernels.hpp` is not a normal header: it has no include guard and is **designed to be
included several times**, each time with the `KT_TEST_DO3` / `KT_TEST_DO4` macros bound to a
different meaning. It contains only (a) the `KT_TEST_DO_*` convenience macros and (b) the
single authoritative list of drivers to generate. Each includer must define `KT_TEST_DO3`
and `KT_TEST_DO4` *before* including it (the header `#error`s otherwise), and the header
`#undef`s its working macros on the way out so it can be re-included cleanly.

**`KT_TEST_DO3` vs `KT_TEST_DO4`.** Drivers come in two shapes:

- `KT_TEST_DO3(FUNC, SZ, SUF)` - drivers parameterised by width and one scalar type, i.e.
  `FUNC<bsz::SZ, SUF>()`. Used for kernels like `kt_loadu_p_test`, `kt_add_p_test`.
- `KT_TEST_DO4(FUNC, SZ, SUF, IDX)` - drivers that additionally take an index type, i.e.
  `FUNC<bsz::SZ, SUF, IDX>()`. Used for the indirect-access kernels driven by
  `KT_TEST_DO_INDEX` (e.g. `kt_scatter_p_test`, `kt_fmadd_p_test`).

The generated symbol name encodes the parameters: `FUNC_SZ_SUF` for `DO3` and
`FUNC_SZ_SUF_IDX` for `DO4`. Because both the definition (in `kt_kernels.cpp`) and the
declaration/registration (in `kt_tests.cpp`) are pasted from the *same* list with the *same*
naming rule, the two sides cannot drift out of sync.

**Two consumers, two meanings.** The same list is expanded for two different purposes:

| Includer | `KT_TEST_DO3` / `KT_TEST_DO4` expand to | Result |
|----------|------------------------------------------|--------|
| `kt_kernels.cpp` | a function *definition* `void FUNC_SZ_SUF() { FUNC<bsz::SZ, SUF>(); }` | the driver bodies that instantiate the kernel test templates |
| `kt_tests.cpp` | a forward *declaration* plus a `TEST(KT_TEST, FUNC_SZ_SUF){...}` | the GoogleTest registrations that call those drivers |

**`KT_TEST_DO[34]_SKIP` variants.** For example, in the case of tests related to `_Float16` these can only be instantiated
when the compiler defines `__AVX512FP16__`. When it does not, `KT_FP16_E` resolves to `_SKIP` and the
`_Float16` rows are routed through `KT_TEST_DO3_SKIP` / `KT_TEST_DO4_SKIP`. In
`kt_kernels.cpp` those emit a driver body that simply calls `GTEST_SKIP()` instead of the
(uncompilable) kernel instantiation; in `kt_tests.cpp`, where no `_SKIP` form is defined,
the header falls back to the normal form so the test is still registered (and skips at
runtime). Every macro pasted with `KT_FP16_E` must therefore have both a base and a `_SKIP`
form: `KT_TEST_DO3`, `KT_TEST_DO3_SKIP`, `KT_TEST_DO4`, and `KT_TEST_DO4_SKIP`.

**Why `kt_tests.cpp` includes the list twice.** The two builds of `kt_kernels.cpp` define
disjoint sets of drivers (AVX2 build: `b128`/`b256`; AVX-512 build: `b512` and the
`_Float16` rows - see [Dual-Compilation Model](#dual-compilation-model)). To register a
`TEST()` for *every* driver from *both* builds, `kt_tests.cpp` includes `kt_kernels.hpp`
twice: once with `KT_TEST_ADD_ONLY_AVX2` defined (selecting the AVX2 set) and once without
(selecting the AVX-512 set):

```cpp
#define KT_TEST_ADD_ONLY_AVX2
#include "kt_kernels.hpp"   // register the b128/b256 drivers
#undef KT_TEST_ADD_ONLY_AVX2
#include "kt_kernels.hpp"   // register the b512 and _Float16 drivers
```

`kt_kernels.cpp` includes it only once: each of its two compilations already selects the
correct set through `KT_AVX2_BUILD`.

### Test skeleton for a new L1 kernel

L1 tests use the exact same mechanism as L0 tests: implement `kt_myop_p_test<SZ, SUF>()` in
`kt_kernels.cpp` and add one `KT_TEST_DO_REAL_COMPLEX(kt_myop_p_test);` line to
`kt_kernels.hpp`. There is no separate test suite; the registration is generated under
`KT_TEST` like every other driver.

### Test skeleton for a block-level L0 kernel

Block-level kernels (`_B` suffix) follow the same mechanism. The test function signature
matches the block kernel signature (taking `c` and `d` by reference); register it with
`KT_TEST_DO_REAL_COMPLEX(kt_myop_B_test);` in `kt_kernels.hpp`. Existing examples are
`kt_fmadd_B_test` and `kt_hsum_B_test`.

### Non-templated ISA-specific tests

Some kernels like `kt_maskz_set_p` have ISA-extension-specific overloads that require
non-templated test functions. These are defined under `#ifdef KT_AVX2_BUILD` / `#else`
guards directly in `kt_kernels.cpp`, without using the `KT_TEST_DO_REAL_COMPLEX*` macros:

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

These drivers are not part of the `kt_kernels.hpp` list. Register them by hand in
`kt_tests.cpp` with an explicit `TEST()` and the appropriate runtime guard:

```cpp
TEST(KT_TEST, kt_myop_256_AVX512VL)
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
2. Extend `get_data<T>()` (and `get_typename<T>()`) with a branch for the new type.
3. Wire the new type into the `KT_TEST_DO_*` lists in `kt_kernels.hpp`, adding the matching
   `_SKIP` routing if it is conditionally compiled like `_Float16`.
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
 * @tparam SUF Scalar data type (float, double, cfloat, cdouble, fp16)
 *
 * @param a First input vector
 * @param b Second input vector
 * @return Result vector
 */
```

For instructions on generating, viewing, and building the documentation, see the
[Documentation section in README.md](README.md#documentation).
