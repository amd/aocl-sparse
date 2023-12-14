# AOCL-Sparse
AOCL-Sparse exposes a common interface that provides Basic Linear Algebra Subroutines for sparse computation implemented for AMD's CPUs. AOCL-Sparse is built with cmake, using the supported compilers GNU and AOCC on Linux and clang/MSVC on Windows. AOCL-Sparse is dependent on AOCL-BLAS and AOCL-LAPACK.

## Requirements
* Git
* CMake versions 3.11 through 3.25
* Boost library versions 1.65 through 1.77

## Build and Installation

#### Building on Windows
1. Install Visual Studio 2022 (the free version) and cmake 
2. Install AOCL-BLAS, AOCL-LAPACK and AOCL-UTILS
3. Define the environment variable AOCL_ROOT to point to AOCL libs installation
```
set "AOCL_ROOT=C:\Program Files\AMD\AOCL-Windows"	
```
4. Navigate to checkout `aocl-sparse` directory
```
cd aocl-sparse
```
5. Configure the project along with the following options depending on the build that is required:
```
cmake -S . -B <build_directory> -T clangcl -G "Visual Studio 17 2022" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_INSTALL_PREFIX="<aoclsparse_install_path>"

# Configure AOCL-Sparse
# Build options:
#   BUILD_SHARED_LIBS        - build AOCL-Sparse as a shared library (Default: ON)
#   BUILD_ILP64              - ILP64 Support (Default: OFF)
#   SUPPORT_OMP              - OpenMP Support (Default: OFF)
#   BUILD_CLIENTS_SAMPLES    - build examples (Default: ON)
#   BUILD_UNIT_TESTS      	 - build unit tests (Default: OFF)
#   BUILD_CLIENTS_BENCHMARKS - build benchmarks (Default: OFF)
#   BUILD_DOCS               - build the PDF documentation (Default: OFF)
#   USE_AVX512               - AVX512 Support (Default: OFF)
```
6. Export the paths for AOCL-BLAS, AOCL-LAPACK and AOCL-UTILS libraries.
```
set PATH=C:\Users\Program Files\AMD\AOCL-Windows\amd-blis\lib\LP64;%PATH%
set PATH=C:\Users\Program Files\AMD\AOCL-Windows\amd-libflame\lib\LP64;%PATH%
set PATH=C:\Users\Program Files\AMD\AOCL-Windows\amd-utils\lib;%PATH%
```
7. Build the project
```
cmake --build <build_directory> --config Release --target install --verbose
```
8. To run the tests/examples, export the path for Sparse library
```
set PATH=<aoclsparse_install_path>\lib;%PATH%
```
9. Run the examples after exporting the required library/headers paths
```
build_directory\tests\examples\sample_spmv.exe 
```
**Note** Currently multi-threading is not supported on Windows.
#### Building on Linux
1. Install AOCL-BLAS, AOCL-LAPACK and AOCL-UTILS
2. Define the environment variable AOCL_ROOT to point to AOCL libs installation
```
export AOCL_ROOT=/opt/aocl
```
3. Change directory to `aocl-sparse` and create the build directory `build/release`
```
cd aocl-sparse
mkdir -p build/release
cd build/release
```
4. Configure the project by typing cmake ../../ along with (or none) of the following options depending on the build that is required:
```
cmake ../.. -DCMAKE_INSTALL_PREFIX="<aoclsparse_install_path>"

# Configure AOCL-Sparse
# Build options:
#   BUILD_SHARED_LIBS        - build AOCL-Sparse as a shared library (Default: ON)
#   BUILD_ILP64              - ILP64 Support (Default: OFF)
#   SUPPORT_OMP              - OpenMP Support (Default: ON)
#   BUILD_CLIENTS_SAMPLES    - build examples (Default: ON)
#   BUILD_UNIT_TESTS      	 - build unit tests (Default: OFF)
#   BUILD_CLIENTS_BENCHMARKS - build benchmarks (Default: OFF)
#   BUILD_DOCS               - build the PDF documentation (Default: OFF)
#   USE_AVX512               - AVX512 Support (Default: OFF)
```
**Note** If `CMAKE_INSTALL_PREFIX` is not provided then `/opt/aoclsparse/` is chosen as the default installation location. Inside it will contain the `lib`, `include`, and `examples` directories and associated files.
5. Build the project by typing make. To install the binaries, do make install
```
make
[sudo] make install
```
6. Refer the sections below to run examples/tests and to build documentation

#### Running basic examples
To get acquainted on how to use the library, a variety of small illustrative examples
are provided (under the `tests/samples` folder). The purpose of these examples is
to show the intended usage of the offered functionality.

During installation, these basic examples are copied to the destination under the `examples` folder.
This folder also contains instructions on how to compile the examples, view the `README.md` inside
the `examples` folder for further details.

The examples can be found in the installation directory under
`"<aoclsparse_install_path>/examples` folder along with a `README.md`
instructions on how to compile.

#### Testing the installation
The library is equipped with a set of *stringent tests programs* (STP) and *unit tests* that
verify the correct functioning of the built library.

The STP and unit tests are located in the `aocl-sparse/tests/unit_tests` folder and are
intended to be executed from the building directory and are not copied during installation.
Tests can be run using ctest.
```
make test
or
ctest -V -R <string_to_find_particular_set_of_tests>
```

#### Benchmarks
To run the provided benchmarks, AOCL-Sparse has to be built with option `BUILD_CLIENTS_BENCHMARKS` turned on,
```
cmake ../../ -DBUILD_CLIENTS_BENCHMARKS=ON 
make
```

The following example illustrates how to benchmark a Sparse Matrix-Vector Product (SPMV) routine using the `scircuit` MTX matrix.

```
cd aocl-sparse/build/release
# Download and uncompress the matrix data
wget https://sparse.tamu.edu/MM/Hamm/scircuit.tar.gz
tar zxf scircuit.tar.gz && rm -rf scircuit.tar.gz
# Benchmark `csrmv` API, using `double` precision, 1000 times.
# `csrmv` performs v = alpha * A * x + beta * v, where
# v is a `m`-sized vector, A is an `m` times `n` sparse matrix,
# x is a `n`-size vector, `alpha` and `beta` are scalars that can
# (optionally) be defined using `--alpha` and `--beta` parameters.
./tests/staging/aoclsparse-bench --function=csrmv --precision=d --alpha=1 --beta=0 --iters=1000 --mtx=./scircuit/scircuit.mtx
```
The expected output should be similar to
```
M       N       nnz     alpha  beta  GFlop/s  GB/s  msec   iter  verified
170998  170998  958936  1.00   0.00  0.56     4.40  3.40e  1000  no
```

Another example using the same routine over a randomly generated sparse matrix of size 1000 times 1000 with 4000 nonzero entries,
```
./tests/staging/aoclsparse-bench --function=csrmv --precision=d --sizem=1000 --sizen=1000 --sizennz=4000 --verify=1
```
The expected output should be similar to
```
M     N     nnz   alpha  beta  GFlop/s  GB/s  msec      iter  verified
1000  1000  4000  1.00   0.00  0.74     6.28  1.08e-02  10    yes
```

#### Building the documentation
To create Sphinx based AOCL-Sparse documentation, build with the option `BUILD_DOCS` turned on,
and invoke the `doc` target,
```
cmake  ../../ -DBUILD_DOCS=ON
make doc
```
The updated documentation can be found under the `aocl-sparse/docs/sphinx` directory.

**Note** Sphinx based document generation is supported only on Linux
**Note** The document generation process requires a recent copy of `Doxygen`, a
LaTeX distribution with `pdflatex`,`sed`, python with `rocm-docs-core`, `breathe`, `sphinx` and `sphinxcontrib.bibtex`

## License
The [license](LICENSE.md) file can be found in the main `aocl-sparse` directory.
