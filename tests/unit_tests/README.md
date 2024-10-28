# AOCL-Sparse Unit Tests

## Requirements
* Git
* GCOV/GCC/G++ - same versions
* CMake versions 3.22 through 3.29

## Steps to create unit tests
 1. Make sure BUILD_UNIT_TESTS is enabled while configuring the project
 2. Create 3 categories of source files covering input-data/utils, unit test backbone and the actual google unit tests
		1. input data/utils: contains the input data that is needed for unit testing of sparse API. The files can also have any custom interfaces needed to call the actual APIs. Refer common_data_utils.cpp/hpp
		2. backbone: this should implement the API by consuming the input provided in (1). This will be similar to test code in tests/include or tests/examples. Refer cg_ut_functions in CG tests, gmres_ut_functions in Gmres tests.
		3. gtest files: The test macros provided by Google Testing framework get defined here. We can use either the simple TEST() macros that have no test fixtures or TEST_F() macros which refer to a test fixture class that we are supposed to define. Gmres unit tests use this fixture and CG tests use a simple TEST() macro. Refer cg_tests.cc in CG tests and gmres_tests.cc in GMRES tests
3. Adding CMake commands to include the unit test as part of aocl-sparse's CMake build system
		1. Define a unique api test executable name and include the test source
			```
			add_executable(gmres_utests gmres_tests.cpp)
			```
		2. Link the dependent sources and libraries
			```
			target_link_libraries(gmres_utests aoclsparse GTest::gtest_main)
			```
		3. Add the test to our "make test" environment, which in turn uses CTest to run all the unit tests. gtest_discover_tests() command should be included towards the end of the file so that the specific test gets registered into CTest automatically.
			```
			gtest_discover_tests(gmres_utests)
			```
		4. To add support for ASAN verification for the unit test, link the address sanitizer library using the flag "-fsanitize=address"
			```
			target_compile_options(gmres_utests PRIVATE -fsanitize=address)
			```

## Executing the Unit Tests
Execute ctest from the build directory
1. Run with no verbose logs
	```
    ctest -j <N>
	```
        - N : number of threads
2. Run the failed cases verbosely
	```
    ctest -j <N> --output-on-failure
	```
3. Run the tests by building in Debug mode under Linux. This will also include Address SANitization. Requires either GCC or Clang compiler with Debug build (-DCMAKE_BUILD_TYPE=Debug)
	```
	./gmres_utests
	./cg_utests
	```

## Out-of-source tree Build Unit Tests on Linux
1. Move to installed tests directory
	```
	cd <install_dir>/unit_tests
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
	**Note** Adding the library path to LD_LIBRARY_PATH is required in case of shared library (.so)
5. Configure the build system to compile unit tests. Make sure the build configuration is same as the one used to build aoclsparse library for BUILD_ILP64, SUPPORT_OMP, CMAKE_BUILD_TYPE, ASAN and COVERAGE variables.
	```
	cmake -S . -B out_tests -DBUILD_SHARED_LIBS=ON -DBUILD_ILP64=OFF -DSUPPORT_OMP=ON -DCMAKE_BUILD_TYPE=Debug -DASAN=ON -DCOVERAGE=ON
	```
Build options:
	- **BUILD_SHARED_LIBS**        - search for shared libraries of dependent libraries (BLAS, LAPACK, UTILS, SPARSE) and build sparse-example build as a shared library
	- **BUILD_ILP64**              - search for ILP64 libraries of dependent libraries (BLAS, LAPACK, UTILS, SPARSE) and build with ILP64 Support
	- **SUPPORT_OMP**              - search for Multithreaded libraries of dependent libraries (BLAS, LAPACK, UTILS, SPARSE) and build with OpenMP Support
	- **ASAN**              		 - build with Address Sanitizer support. This is supported only in Linux-Debug configuration, so ASAN variable is disabled for any other configuration
	- **COVERAGE**                 - build with Code Coverage support. This is supported only in Linux-Debug configuration, so COVERAGE variable is disabled for any other configuration. The following targets have to be built after configuration:
		```
		cmake --build out_tests --target coverage-run
		cmake --build out_tests --target coverage-report
		```
**Note** Match GCOV and GCC/G++ versions, otherwise code coverage reporting will fail during CMake configuration

## Out-of-source tree Build Unit Tests on Windows
1. Move to installed tests directory
	```
	cd <install_dir>\unit_tests
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
	**Note** Adding the library paths to LD_LIBRARY_PATH is required in case of shared library (DLL)
5. Configure the build system to compile unit tests. Make sure the build configuration is same as the one used to build aoclsparse library for BUILD_ILP64 and SUPPORT_OMP variables.
	```
	cmake -S . -B out_tests -DBUILD_SHARED_LIBS=ON -DBUILD_ILP64=OFF -DSUPPORT_OMP=ON
	```
6. Compile the sample applications
	```
	cmake --build out_tests
	```
7. Run the application
	```
	.\out_tests\Debug\trsv_tests.exe
	```
**Note**
1. No need to define main() for the google tests, since it is already part of GTest library that is included as source repo dependency using FetchContent cmake command. So every cmake's clean build downloads the google test repo as per the tag provided, builds and the library built is linked against our project
2. Address Sanitizer (ASAN) is enabled in the Linux Debug build for GCC and Clang. It is used to detect possible leaks. Any leaks are considered as test failures.
3. ASAN verification is currently supported by GCC and Clang compilers on Linux. Support on Windows is not available for now.
