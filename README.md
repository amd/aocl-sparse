# AOCL-Sparse
AOCL-Sparse exposes a common interface that provides Basic Linear Algebra Subroutines for sparse computation implemented for AMD's CPUs.

## Requirements
* CMake (3.5 or later)

## Quickstart AOCL-Sparse build and install

#### CMake
All compiler specifications are determined automatically.

After cloning AOCL-Sparse, move into the `aocl-sparse` directory and create the build directory `build/release`,
```
cd aocl-sparse
mkdir -p build/release
cd build/release

# Configure AOCL-Sparse
# Build options:
#   BUILD_SHARED_LIBS        - build AOCL-Sparse as a shared library (Default: ON)
#   BUILD_ILP64              - ILP64 Support (Default: OFF)
#   SUPPORT_OMP              - OpenMP Support (Default: ON)
#   BUILD_CLIENTS_SAMPLES    - build examples (Default: ON)
#   BUILD_CLIENTS_TESTS      - build unit tests (Default: OFF)
#   BUILD_CLIENTS_BENCHMARKS - build benchmarks (Default: OFF)
#   BUILD_DOCS               - build the PDF documentation (Default: OFF)
```
**Note** If `CMAKE_INSTALL_PREFIX` is not provided then `/opt/aoclsparse/` is chosen as the default installation location.
Inside it will contain the `lib`, `include`, and `examples` directories and associated files.

#### Building the shared library

```
cmake ../.. -DCMAKE_INSTALL_PREFIX="<aoclsparse_install_path>"
make
[sudo] make install
```

#### Building the static library
The static version of the library can be built by turning off the option `BUILD_SHARED_LIBS`,
```
cmake ../../ -DBUILD_SHARED_LIBS=OFF
make
[sudo] make install
```

#### Running basic examples
To get acquainted on how to use the library, a variety of small illustrative examples
are provided (under the `tests/samples` folder). The purpose of these examples is
to show the intended usage of the offered functionality.

During installation, these basic examples are copied to the destination under the `examples` folder.
This folder also contains instructions on how to compile the examples, view the `README.md` inside
the `examples` folder for further details.

```
cmake ../../
make install
```
then, the examples can be found in the installation directory under
`"<aoclsparse_install_path>/examples` folder along with a `README.md`
instructions on how to compile.

#### Testing the installation
The library is equipped with a set of *stringent tests programs* (STP) and *unit tests* that
verify the correct functioning of the built library.

The STP and unit tests are located in the `aocl-sparse/tests/functionality` folder and are
intended to be executed from the building directory and are not copied during installation.



#### Benchmarks
To run the provided benchmarks, AOCL-Sparse has to be built with option `BUILD_CLIENTS_BENCHMARKS` turned on,
```
cmake -DBUILD_CLIENTS_BENCHMARKS=ON ../../
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
To create the AOCL-Sparse PDF documentation, build with the option `BUILD_DOCS` turned on,
and invoke the `docs` target,
```
cmake  ../../ -DBUILD_DOCS=ON
make docs
```
The updated documentation can be found under the `aocl-sparse/docs` directory.

**Note** The document generation process requires a recent copy of `Doxygen` and a
LaTeX distribution with `pdflatex`.

## License
The [license](LICENSE.md) file can be found in the main `aocl-sparse` directory.
