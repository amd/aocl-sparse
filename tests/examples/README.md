# AOCL-Sparse Samples
This directory contains sample source files showing usage of AOCL-Sparse library functions. Use the provided cmake script file "CMakeLists.txt" to compile and run the programs. Same cmake script may be used for both Linux and Windows platforms.

## Requirements
* Git
* CMake versions 3.22 through 3.29
* Boost library versions 1.80 through 1.85

## Build steps on Linux

1. Move to installed examples directory
	```
	cd <install_dir>/examples
	```
2. Define the environment variable AOCL_ROOT to point to AOCL libs installation that has AOCL BLAS, AOCL LAPACK and AOCL UTILS libraries.
	```
	export AOCL_ROOT=/opt/aocl/gcc
	```
3. Define SPARSE_ROOT to aocl-sparse package installation path.
	```
	export SPARSE_ROOT=<path/to/AOCL/Sparse/install/directory>
	```
	eg:
	```
	export SPARSE_ROOT=$HOME/amd/aocl-sparse
	```
	**Note** If in case, AOCL_ROOT contains AOCL Sparse installation along with BLAS, LAPACK and UTILS libraries, then "SPARSE_ROOT" takes precedence when searching for sparse library and headers.
4. Add library paths of Sparse, BLAS, LAPACK and UTILS to environment path variable $LD_LIBRARY_PATH.
	```
	export LD_LIBRARY_PATH=$SPARSE_ROOT/lib:$AOCL_ROOT/lib:$LD_LIBRARY_PATH
	```
	**Note** Adding the library paths to LD_LIBRARY_PATH is required in case of shared library (.so)

5. Configure the build system to compile sample applications. Make sure the build configuration is same as the one used to build aoclsparse library for BUILD_ILP64 and SUPPORT_OMP variables.
	```
	cmake -S . -B out_sparse -DBUILD_SHARED_LIBS=ON -DBUILD_ILP64=OFF -DSUPPORT_OMP=ON
	```
Build options:
	BUILD_SHARED_LIBS        - search for shared libraries of dependent libraries (BLAS, LAPACK, UTILS, SPARSE) and build sparse-example build as a shared library
	BUILD_ILP64              - search for ILP64 libraries of dependent libraries (BLAS, LAPACK, UTILS, SPARSE) and build with ILP64 Support
	SUPPORT_OMP              - search for Multithreaded libraries of dependent libraries (BLAS, LAPACK, UTILS, SPARSE) and build with OpenMP Support
6. Compile the sample applications
	```
	cmake --build out_sparse
	```
7. Run the application
	```
	./out_sparse/sample_mv_cpp
	```
## Build steps on Windows

1. Move to installed examples directory
	```
	cd <install_dir>\examples
	```
2.  Define the environment variable AOCL_ROOT to point to AOCL libs installation that has AOCL BLAS, AOCL LAPACK and AOCL UTILS libraries.
	```
	set "AOCL_ROOT=C:\Program Files\AMD\AOCL-Windows"
	```
3. Define SPARSE_ROOT to aocl-sparse package installation path.
	```
	set SPARSE_ROOT=<path\to\AOCL\Sparse\install\directory>
	```
	eg:
	```
	set SPARSE_ROOT=%HOME%\amd\aocl-sparse
	```
	**Note** If in case, AOCL_ROOT contains AOCL Sparse installation along with BLAS, LAPACK and UTILS libraries, then "SPARSE_ROOT" takes precedence when searching for sparse library and headers.
4. Add library paths of Sparse, BLAS, LAPACK and UTILS to environment path variable
	```
	set PATH=%HOME%\amd\aocl-sparse\lib;%PATH%
	set PATH=%AOCL_ROOT%\amd-blis\lib\ILP64;%PATH%
	set PATH=%AOCL_ROOT%\amd-libflame\lib\ILP64;%PATH%
	set PATH=%AOCL_ROOT%\amd-utils\lib;%PATH%
	```
	**Note** Adding the library and headers paths to LD_LIBRARY_PATH is required in case of shared library (DLL)

5. Configure the build system to compile sample applications. Make sure the build configuration is same as the one used to build aoclsparse library for BUILD_ILP64 and SUPPORT_OMP variables.
	```
	cmake -S . -B out_sparse -G "Visual Studio 17 2022" -A x64 -T "clangcl" -DCMAKE_CXX_COMPILER=clang-cl -DBUILD_SHARED_LIBS=ON -DBUILD_ILP64=OFF -DSUPPORT_OMP=ON
	```
6. Compile the sample applications
	```
	cmake --build out_sparse
	```
7. Run the application
	```
	.\out_sparse\sample_mv_cpp.exe
	```
**Note** The environment variable "SPARSE_ROOT" takes precedence when searching for sparse library and headers. If the environment variable "SPARSE_ROOT" is not defined, then SPARSE_ROOT inherits the path from AOCL_ROOT and looks for AOCL sparse installation in AOCL_ROOT. If Sparse library/headers are not found either in SPARSE_ROOT or AOCL_ROOT, then an error is returned.

## Out-of-Source Tree Build

The examples automatically support both in-source and out-of-source tree builds without requiring user intervention.

### Key Features of Out-of-Source Build Mode

When building examples standalone, the following features are automatically enabled:

- **Independent Project Setup**: Creates a standalone CMake project (`aoclsparse_examples`) with its own build configuration
- **External Dependency Resolution**: Automatically finds and links required AOCL libraries (BLAS, LAPACK, UTILS) through the `Dependencies.cmake` module
- **Library Discovery**: Searches for AOCL-Sparse library and headers in `SPARSE_ROOT` or `AOCL_ROOT` locations with support for both LP64 and ILP64 variants
- **Flexible Installation Path**: Sets default installation to `../bin` relative to the examples directory
- **Build Configuration Inheritance**: Supports the same build options as the main library (`BUILD_SHARED_LIBS`, `BUILD_ILP64`, `SUPPORT_OMP`)
- **Skip Source Installation**: Source files and CMake configurations are not installed when building in out-of-source mode

### Out-of-Source Build Steps on Linux

1. Ensure AOCL-Sparse library is already installed and define environment variables:
	```bash
	export AOCL_ROOT=/opt/aocl/gcc
	export SPARSE_ROOT=/path/to/aocl-sparse/installation
	export LD_LIBRARY_PATH=$SPARSE_ROOT/lib:$AOCL_ROOT/lib:$LD_LIBRARY_PATH
	```

2. Navigate to the examples directory:
	```bash
	cd /path/to/aocl-sparse/tests/examples
	```

3. Configure the build (out-of-source mode is automatically enabled):
	```bash
	cmake -S . -B out_sparse_standalone \
	  -DBUILD_SHARED_LIBS=ON \
	  -DBUILD_ILP64=OFF \
	  -DSUPPORT_OMP=ON
	```

4. Build the examples:
	```bash
	cmake --build out_sparse_standalone
	```

5. Run an example:
	```bash
	./out_sparse_standalone/sample_mv_cpp
	```

### Out-of-Source Build Steps on Windows

1. Ensure AOCL-Sparse library is already installed and define environment variables:
	```cmd
	set "AOCL_ROOT=C:\Program Files\AMD\AOCL-Windows"
	set "SPARSE_ROOT=C:\path\to\aocl-sparse\installation"
	set "PATH=%SPARSE_ROOT%\lib;%AOCL_ROOT%\lib;%PATH%"
	```

2. Navigate to the examples directory:
	```cmd
	cd C:\path\to\aocl-sparse\tests\examples
	```

3. Configure the build (out-of-source mode is automatically enabled):
	```cmd
	cmake -S . -B out_sparse_standalone ^
	  -G "Visual Studio 17 2022" -A x64 -T "clangcl" ^
	  -DCMAKE_CXX_COMPILER=clang-cl ^
	  -DBUILD_SHARED_LIBS=ON ^
	  -DBUILD_ILP64=OFF ^
	  -DSUPPORT_OMP=ON
	```

4. Build the examples:
	```cmd
	cmake --build out_sparse_standalone
	```

5. Run an example:
	```cmd
	.\out_sparse_standalone\Debug\sample_mv_cpp.exe
	```

**Important**: The `SPARSE_ROOT` environment variable must be properly defined to point to the AOCL-Sparse installation directory. If `SPARSE_ROOT` is not set, the build system will fall back to searching in `AOCL_ROOT`.

**Note**: For out-of-source builds, `AOCL_ROOT` is not strictly required to be defined. If you are building with explicit CMake variables for BLAS/LAPACK/UTILS libraries (non-AOCL_ROOT mode), the build system will delegate dependency resolution to the `Dependencies.cmake` module.

### Non-AOCL_ROOT Build Mode

For builds using explicit library paths instead of `AOCL_ROOT`, you can specify the AOCL libraries directly via CMake variables.

**Linux Example**:
```bash
# Define common options with explicit library paths
common_opts="-DAOCL_BLIS_LIB=/path/to/aocl/lib_ILP64/libblis-mt.so \
  -DAOCL_LIBFLAME=/path/to/aocl/lib_ILP64/libflame.so \
  -DAOCL_UTILS_LIB=/path/to/aocl/lib_ILP64/libaoclutils.so \
  -DAOCL_BLIS_INCLUDE_DIR=/path/to/aocl/include_ILP64 \
  -DAOCL_LIBFLAME_INCLUDE_DIR=/path/to/aocl/include_ILP64 \
  -DAOCL_UTILS_INCLUDE_DIR=/path/to/aocl/include_ILP64/alci"

# Configure examples with explicit library paths
cmake -S . -B build $common_opts

# Build examples
cmake --build build -j
```

This mode bypasses the need for `AOCL_ROOT` environment variable and allows fine-grained control over library selection. If AOCL_ROOT is also defined, then non-AOCL_ROOT mode of explicit definition will take precedence.

**Note**: Non-AOCL_ROOT support on Windows is similar to Linux configure/build, using the same explicit CMake library variables with appropriate Windows library paths and extensions.

This mode is particularly useful when:
- Building examples against a pre-installed AOCL-Sparse library
- Developing custom applications based on the example code
- Testing with different build configurations independently
- Using non-AOCL BLAS/LAPACK/UTILS libraries by explicitly setting CMake variables

## Unit Tests Out-of-Source Build

Similar to examples, the unit tests located in `tests/unit_tests` also support out-of-source tree builds with automatic configuration.

The unit tests build includes Google Test framework integration and supports `ctest` for automated test execution. The same environment variable requirements (`SPARSE_ROOT`, `AOCL_ROOT`) and non-AOCL_ROOT mode apply as with examples. Unit tests also support the non-AOCL_ROOT build mode using explicit CMake library variables.

**Example for out-of-source unit tests build**:
```bash
cd /path/to/aocl-sparse/tests/unit_tests
cmake -S . -B build_tests -DBUILD_SHARED_LIBS=ON -DBUILD_ILP64=OFF -DSUPPORT_OMP=ON
cmake --build build_tests
ctest --test-dir build_tests
```
