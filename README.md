# AOCLSPARSE
AOCLSPARSE exposes a common interface that provides Basic Linear Algebra Subroutines for sparse computation implemented for AMD's CPUs.

## Requirements
* Git
* CMake (3.5 or later)

Optional:
* [Boost][]
  * Required for benchmarks.

## Quickstart AOCLSPARSE build and install

#### CMake
All compiler specifications are determined automatically. 
```
# Clone AOCLSPARSE using git
git clone ssh://gerritgit/cpulibraries/er/aocl-sparse

# Go to AOCLSPARSE directory, create and go to the build directory
cd AOCLSPARSE; mkdir -p build/release; cd build/release

# Configure AOCLSPARSE
# Build options:
#   BUILD_CLIENTS_BENCHMARKS - build benchmarks (OFF)
#   BUILD_VERBOSE            - verbose output (OFF)
#   BUILD_SHARED_LIBS        - build AOCLSPARSE as a shared library (ON)
cmake -DBUILD_CLIENTS_BENCHMARKS=ON ../..

# Build
make

# Install
[sudo] make install
```

## Benchmarks
To run benchmarks, AOCLSPARSE has to be built with option -DBUILD_CLIENTS_BENCHMARKS=ON.
```
# Go to AOCLSPARSE build directory
cd AOCLSPARSE/build/release

# Run benchmark, e.g.
./test/staging/AOCLsparse-bench -f csrmv --precision d --alpha 1 --beta 0 --iters 1000 --mtx circuit.mtx
```

## License
The [license file][] can be found in the main repository.

[Boost]: https://www.boost.org/
