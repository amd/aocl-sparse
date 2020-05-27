# aocl-sparse
aocl-sparse exposes a common interface that provides Basic Linear Algebra Subroutines for sparse computation implemented for AMD's CPUs.

## Requirements
* CMake (3.5 or later)

Optional:
* [Boost][]
  * Required for benchmarks.

## Quickstart aocl-sparse build and install

#### CMake
All compiler specifications are determined automatically. 

# After cloning aocl-sparse source to aocl-sparse directory.
# Go to aocl-sparse directory, create and go to the build directory
cd aocl-sparse; mkdir -p build/release; cd build/release

# Configure aocl-sparse
# Build options:
#   BUILD_CLIENTS_BENCHMARKS - build benchmarks (OFF)
#   BUILD_VERBOSE            - verbose output (OFF)
#   BUILD_SHARED_LIBS        - build aocl-sparse as a shared library (ON)
cmake -DBUILD_CLIENTS_BENCHMARKS=ON ../..

# Build
make

# Install
make install

## Benchmarks
To run benchmarks, aocl-sparse has to be built with option -DBUILD_CLIENTS_BENCHMARKS=ON.

# Go to aocl-sparse build directory
cd aocl-sparse/build/

# Get the scircuit.mtx file, e.g.
wget https://sparse.tamu.edu/MM/Hamm/scircuit.tar.gz
tar xf scircuit.tar.gz; mv scircuit ./scircuit.mtx && rm -rf scircuit.tar.gz scircuit

# Run benchmark using scircuit.mtx file, e.g.
./tests/staging/aoclsparse-bench -f csrmv --precision d --alpha 1 --beta 0 --iters 1000 --mtx ./scircuit.mtx/scircuit.mtx

## License
The [license file][] can be found in the main repository.
