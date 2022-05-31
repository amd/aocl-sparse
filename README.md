# AOCL-Sparse
AOCL-Sparse exposes a common interface that provides Basic Linear Algebra Subroutines for sparse computation implemented for AMD's CPUs.

## Requirements
* CMake (3.5 or later)

## Quickstart AOCL-Sparse build and install

#### CMake
All compiler specifications are determined automatically.

After cloning aocl-sparse, Go to aocl-sparse directory, create and go to the build directory
```
cd aocl-sparse;
mkdir -p build/release;
cd build/release;

# Configure AOCL-Sparse
# Build options:
#   BUILD_CLIENTS_BENCHMARKS - build benchmarks (OFF)
#   BUILD_SHARED_LIBS        - build AOCL-Sparse as a shared library (ON)
#   BUILD_ILP64              - ILP64 Support (OFF)
#   SUPPORT_OMP              - OpenMP Support (ON)

# Build shared library (default)
if CMAKE_INSTALL_PREFIX is not provided then /opt/aoclsparse/ is the default location which contains lib/ and include/ directory files.
cmake ../.. -DCMAKE_INSTALL_PREFIX="<Directory_path>"

# Build
make

# Install
[sudo] make install

# Build static library
cmake -DBUILD_SHARED_LIBS=OFF ../../
make install
```

#### Benchmarks
To run benchmarks, AOCL-Sparse has to be built with option -DBUILD_CLIENTS_BENCHMARKS=ON.

cmake -DBUILD_CLIENTS_BENCHMARKS=ON ../../
make install

#### Simple Test
Test the installation by running CSR-SPMV on scircuit.mtx file, e.g.
```
cd aocl-sparse/build/release
wget https://sparse.tamu.edu/MM/Hamm/scircuit.tar.gz
tar xf scircuit.tar.gz; mv scircuit ./scircuit.mtx && rm -rf scircuit.tar.gz scircuit
./tests/staging/aoclsparse-bench --function=csrmv --precision=d --alpha=1 --beta=0 --iters=1000 --mtx=./scircuit.mtx/scircuit.mtx
```

Test the installation by running CSR-SPMV on randomly generated matrix, e.g.
```
./tests/staging/aoclsparse-bench --function=csrmv --precision=d --sizem=1000 --sizen=1000 --sizennz=4000 --verify=1
```

## License
The [license file][LICENSE.md] can be found in the main repository.
