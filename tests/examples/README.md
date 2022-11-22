# AOCL-Sparse Samples
This directory contains sample source files showing usage of AOCL-Sparse library functions. Use the provided cmake script file "CMakeLists.txt" to compile and run the programs. Same cmake script may be used for both Linux and Windows platforms.

## Build steps on Linux

1. Move to installed examples directory
	```
	cd aocl-sparse/examples
	```
2. Set AOCL_ROOT to aocl-sparse package installation path
	```
	export AOCL_ROOT=<path/to/AOCL/Sparse/install/directory>
	```
	eg:
	```
	export AOCL_ROOT=$HOME/amd/aocl-sparse
	```

2. Add aoclsparse library path to environment path variable $LD_LIBRARY_PATH.
	```
	export LD_LIBRARY_PATH=$AOCL_ROOT/lib:$LD_LIBRARY_PATH
	```
	**Note** Adding the library and headers paths to LD_LIBRARY_PATH is required in case of shared library (.so)

3. Configure the build system to compile sample applications
	```
	cmake S . -B out_sparse
	```
	**Note** The default build configuration supports shared library with LP64. For any other configuration, define the flags "BUILD_SHARED_LIBS" and "BUILD_ILP64" accordingly
	```
	cmake S . -B out_sparse -DBUILD_SHARED_LIBS=ON -DBUILD_ILP64=ON
	```
4. Compile the sample applications
	```
	cmake --build out_sparse
	```
5. Run the application
	```
	./sample_spmv
	```
## Build steps on Windows

1. Move to installed examples directory
	```
	cd aocl-sparse\examples
	```
2. Set AOCL_ROOT to aocl-sparse package installation path
	```
	set AOCL_ROOT=<path\to\AOCL\Sparse\install\directory>
	```
	eg:
	```
	set AOCL_ROOT=%HOME%\amd\aocl-sparse
	```

2. Add aoclsparse library path to environment path variable
	```
	set PATH=%HOME%\amd\aocl-sparse\lib;%HOME%\amd\aocl-sparse\include;%PATH%
	```
	**Note** Adding the library and headers paths to LD_LIBRARY_PATH is required in case of shared library (DLL)

3. Configure the build system to compile sample applications
	```
	cmake S . -B out_sparse -G "Visual Studio 16  2019" -A x64 -T "LLVM" -DCMAKE_CXX_COMPILER=clang-cl
	```
4. Compile the sample applications
	```
	cmake --build out_sparse
	```
5. Run the application
	```
	sample_spmv.exe
	```

**Note** The sample compilation requires a python installation: `python 2.7 or later`.
