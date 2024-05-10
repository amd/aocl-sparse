# AOCL-Sparse Samples
This directory contains sample source files showing usage of AOCL-Sparse library functions. Use the provided cmake script file "CMakeLists.txt" to compile and run the programs. Same cmake script may be used for both Linux and Windows platforms.

## Requirements
* Git
* CMake versions 3.21 through 3.29
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
	**Note** If in case, AOCL_ROOT contains AOCL Sparse installation along with BLAS, LAPACK and UTILS libraries,   then "SPARSE_ROOT" takes precedence when searching for sparse library and headers.
4. Add library paths of Sparse, BLAS LAPACK and UTILS to environment path variable $LD_LIBRARY_PATH.
	```
	export LD_LIBRARY_PATH=$SPARSE_ROOT/lib:$AOCL_ROOT/lib:$LD_LIBRARY_PATH
	```
	**Note** Adding the library and headers paths to LD_LIBRARY_PATH is required in case of shared library (.so)

5. Configure the build system to compile sample applications. Make sure the build configuration is same as the one used to build aoclsparse library for BUILD_ILP64 and SUPPORT_OMP variables.
	```
	cmake -S . -B out_sparse -DBUILD_ILP64=ON -DSUPPORT_OMP=OFF
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
	./out_sparse/sample_spmv
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
	cmake -S . -B out_sparse -G "Visual Studio 17 2022" -A x64 -T "clangcl" -DCMAKE_CXX_COMPILER=clang-cl -DBUILD_SHARED_LIBS=ON -DBUILD_ILP64=OFF -DSUPPORT_OMP=OFF
	```
6. Compile the sample applications
	```
	cmake --build out_sparse
	```
7. Run the application
	```
	.\out_sparse\sample_spmv.exe
	```
**Note** The environment variable "SPARSE_ROOT" takes precedence when searching for sparse library and headers. If the environment variable "SPARSE_ROOT" is not defined, then SPARSE_ROOT inherits the path from AOCL_ROOT and looks for AOCL sparse installation in AOCL_ROOT. If Sparse library/headers are not found either in SPARSE_ROOT or AOCL_ROOT, then an error is returned.
**Note** Currently multi-threading is not supported on Windows.
