# ########################################################################
# Copyright (c) 2020-2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

set(Boost_USE_MULTITHREADED ON)
set(Boost_DETAILED_FAILURE_MSG ON)
set(Boost_ADDITIONAL_VERSIONS 1.65.1 1.65)
set(Boost_USE_STATIC_LIBS OFF)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Boost)

if(NOT Boost_FOUND)
    message(STATUS "Dynamic boost libraries not found. Attempting to find static libraries...")
    set(Boost_USE_STATIC_LIBS ON)
find_package(Boost)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "boost is a required dependency and is not found; try adding boost path to CMAKE_PREFIX_PATH")
    endif()
endif()

set(AOCLSPARSE_BENCHMARK_SOURCES
  aoclsparse_test.cpp
)

set(AOCLSPARSE_CLIENTS_COMMON
  ../common/aoclsparse_utility.cpp
  ../common/aoclsparse_template_specialization.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/library/include
                    ${CMAKE_SOURCE_DIR}/library/src/include
                    ${CMAKE_SOURCE_DIR}/library/src/solvers
                    ${CMAKE_SOURCE_DIR}/library/src/analysis
                    ${CMAKE_SOURCE_DIR}/library/src/level2
                    ${CMAKE_SOURCE_DIR}/library/src/extra
                    ${CMAKE_SOURCE_DIR}/tests/include)

add_executable(aoclsparse-bench ${AOCLSPARSE_BENCHMARK_SOURCES} ${AOCLSPARSE_CLIENTS_COMMON})

# Internal header includes
target_include_directories(aoclsparse-bench
  PRIVATE
    ${LAPACK_INCLUDE_DIRS}
    ${BLIS_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# OpenMP cmake fix for cmake <= 3.9
if(OPENMP_FOUND)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
  endif()
endif()

# Add aoclsparse target
if(NOT TARGET aoclsparse)
  target_link_libraries(aoclsparse-bench PRIVATE ${AOCLSPARSE_LIBRARIES})
else()
  target_link_libraries(aoclsparse-bench PRIVATE aocl::aoclsparse )
endif()


# Add OpenMP
if(OPENMP_FOUND)
  target_link_libraries(aoclsparse-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(aoclsparse-bench PUBLIC ${LAPACK_LIBRARIES})

# Set benchmark output directory
set_target_properties(aoclsparse-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

# Add selected matrix operations on small matrices as test targets

# Run all functions on these sizes
# set common matrix sizes for square matrices "m n nnz"
set(SQMATRIXSIZES
  "1 1 1"
  "2 2 2"
  "47 47 110"    # 5% density, ~2nnz per row
  "47 47 513"    # 23% density, ~11nnz per row
  "100 100 500"
  "103 103 499"
)
# set common matrix sizes for rectangular matrices "m n nnz"
set(RCTMATRIXSIZES
  "1 99 77"
  "103 1 83"
  "4 53 110"     # 5% density, ~27nnz per row, 2 per column
  "53 3 111"     # 7% density, ~2nnz per row, 37 per column
  "29 47 500"    # 3% density
  "50 49 490"    # 20% density
  "20 100 800"   # nnz per row > 30
  "20 105 600"   # nnz per row > 30
)

# set common matrix sizes for rectangular matrices A & B "m k n nnz"
set(RCTMATRIXSIZESMM
  "1 99 53 77"
  "103 1 90 83"
  "4 53 100 110"     # 5% density, ~27nnz per row, 2 per column
  "53 3 100 111"     # 7% density, ~2nnz per row, 37 per column
  "29 47 83 500"    # 3% density
  "50 49 11 490"    # 20% density
  "20 100 50 800"   # nnz per row > 30
  "20 105 80 600"   # nnz per row > 30
)

foreach(FUNCTION "csrmv" "ellmv" "diamv" "optmv") # TODO add back "csrsv", so far failing the tests
  foreach(PREC "d" "s")

    foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
      # split MATSIZE string to individual tokens
      string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
      list(GET MATSIZE_LIST 0 SIZEM)
      list(GET MATSIZE_LIST 1 SIZEN)
      list(GET MATSIZE_LIST 2 SIZENNZ)
      #message(STATUS "PREC=${PREC}: MATSIZE=${MATSIZE} MATSIZE_LIST=${MATSIZE_LIST} ${SIZEM}x${SIZEN}x${SIZENNZ}")

      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ} ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --transposeA=N --verify=1 --iters=1)
      # set_property(TEST FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ} PROPERTY ENVIRONMENT ASAN_OPTIONS="log_path=ASANlogger.FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}")

      if(FUNCTION STREQUAL "csrmv" OR FUNCTION STREQUAL "optmv")
        # Add transpose tests
        add_test(FuncTest.${FUNCTION}T-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ} ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --transposeA=T --verify=1 --iters=1)
      ENDIF()

    endforeach(MATSIZE)

    # Add extra tests to exercise nonstandard multipliers alpha, beta
    add_test(FuncTest.${FUNCTION}-${PREC}-100x99x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=N --verify=1 --iters=1)
    add_test(FuncTest.${FUNCTION}-${PREC}-21x49x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=N --verify=1 --iters=1)

    if(FUNCTION STREQUAL "csrmv" OR FUNCTION STREQUAL "optmv")
      # Add transpose tests
      add_test(FuncTest.${FUNCTION}T-${PREC}-100x99x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=T  --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}T-${PREC}-21x49x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=T  --verify=1 --iters=1)
    ENDIF()

  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "blkcsrmv")
  foreach(PREC "d")
    foreach(BLK "1" "2" "4")

    foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
      # split MATSIZE string to individual tokens
      string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
      list(GET MATSIZE_LIST 0 SIZEM)
      list(GET MATSIZE_LIST 1 SIZEN)
      list(GET MATSIZE_LIST 2 SIZENNZ)
      #message(STATUS "PREC=${PREC}: MATSIZE=${MATSIZE} MATSIZE_LIST=${MATSIZE_LIST} ${SIZEM}x${SIZEN}x${SIZENNZ}")
      if(SIZEN GREATER_EQUAL 8)
         add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-${SIZEM}x${SIZEN}x${SIZENNZ} ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizeblk=${BLK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --verify=1 --iters=1)
      # set_property(TEST FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ} PROPERTY ENVIRONMENT ASAN_OPTIONS="log_path=ASANlogger.FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}")
      endif()
    endforeach(MATSIZE)

    # Add extra tests to exercise nonstandard multipliers alpha, beta
    add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-100x99x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizeblk=${BLK} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
    add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-21x49x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizeblk=${BLK} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)

    endforeach(BLK)

    #adding tests with incorrect block size
    add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-100x99x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizeblk=3 --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
    add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-21x49x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizeblk=7 --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)

  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "bsrmv")
  foreach(PREC "d" "s")
    foreach(BLK "2" "3" "4")

      foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
        # split MATSIZE string to individual tokens
        string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
        list(GET MATSIZE_LIST 0 SIZEM)
        list(GET MATSIZE_LIST 1 SIZEN)
        list(GET MATSIZE_LIST 2 SIZENNZ)

        add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-${SIZEM}x${SIZEN}x${SIZENNZ} ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --blockdim=${BLK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --verify=1 --iters=1)
      endforeach(MATSIZE)

      # Add extra tests to exercise nonstandard multipliers alpha, beta
      add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-100x99x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --blockdim=${BLK} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-21x49x333-mlt ../staging/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --blockdim=${BLK} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)

    endforeach(BLK)
  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "csrmm")
  foreach(PREC "d" "s")

    foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
      # split MATSIZE string to individual tokens
      string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
      list(GET MATSIZE_LIST 0 SIZEM)
      list(GET MATSIZE_LIST 1 SIZEN)
      list(GET MATSIZE_LIST 2 SIZENNZ)

      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZEN}x${SIZENNZ}-co ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEN} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --order=1 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZEN}x${SIZENNZ}-ro ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEN} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --order=0 --verify=1 --iters=1)
    endforeach(MATSIZE)

    foreach(MATSIZE ${RCTMATRIXSIZESMM})
      # split MATSIZE string to individual tokens
      string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
      list(GET MATSIZE_LIST 0 SIZEM)
      list(GET MATSIZE_LIST 1 SIZEK)
      list(GET MATSIZE_LIST 2 SIZEN)
      list(GET MATSIZE_LIST 3 SIZENNZ)

      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEK}x${SIZEN}x${SIZENNZ}-co ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --order=1 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEK}x${SIZEN}x${SIZENNZ}-ro ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --order=0 --verify=1 --iters=1)
    endforeach(MATSIZE)

      # Add extra tests to exercise nonstandard multipliers alpha, beta
      add_test(FuncTest.${FUNCTION}-${PREC}-103x103x103x499-co-mlt ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=103 --sizem=103 --sizen=103 --sizennz=499 --order=1 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-103x103x103x499-ro-mlt ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=103 --sizem=103 --sizen=103 --sizennz=499 --order=0 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-103x103x103x499-co-mlt1 ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=103 --sizem=103 --sizen=103 --sizennz=499 --order=1 --alpha=2 --beta=0 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-103x103x103x499-ro-mlt1 ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=103 --sizem=103 --sizen=103 --sizennz=499 --order=0 --alpha=2 --beta=0 --verify=1 --iters=1)

      add_test(FuncTest.${FUNCTION}-${PREC}-42x42x42x213-co-mlt ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=42 --sizem=42 --sizen=42 --sizennz=213 --order=1 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-42x42x42x213-ro-mlt ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=42 --sizem=42 --sizen=42 --sizennz=213 --order=0 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-42x42x42x213-co-mlt1 ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=42 --sizem=42 --sizen=42 --sizennz=213 --order=1 --alpha=2 --beta=0 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-42x42x42x213-ro-mlt1 ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=42 --sizem=42 --sizen=42 --sizennz=213 --order=0 --alpha=2 --beta=0 --verify=1 --iters=1)

      add_test(FuncTest.${FUNCTION}-${PREC}-81x81x81x815-co-mlt ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=81 --sizem=81 --sizen=81 --sizennz=815 --order=1 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-81x81x81x815-ro-mlt ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=81 --sizem=81 --sizen=81 --sizennz=815 --order=0 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-81x81x81x815-co-mlt1 ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=81 --sizem=81 --sizen=81 --sizennz=815 --order=1 --alpha=2 --beta=0 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-81x81x81x815-ro-mlt1 ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=81 --sizem=81 --sizen=81 --sizennz=815 --order=0 --alpha=2 --beta=0 --verify=1 --iters=1)

  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "csr2m")
  foreach(PREC "d" "s")
    foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
      # split MATSIZE string to individual tokens
      string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
      list(GET MATSIZE_LIST 0 SIZEM)
      list(GET MATSIZE_LIST 1 SIZEN)
      list(GET MATSIZE_LIST 2 SIZENNZ)

      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZEM}x${SIZENNZ}-1s ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEN} --sizem=${SIZEM} --sizen=${SIZEM} --sizennz=${SIZENNZ} --stage=0 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZEM}x${SIZENNZ}-2s ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEN} --sizem=${SIZEM} --sizen=${SIZEM} --sizennz=${SIZENNZ} --stage=1 --verify=1 --iters=1)

    endforeach(MATSIZE)

    foreach(MATSIZE ${RCTMATRIXSIZESMM})
      # split MATSIZE string to individual tokens
      string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
      list(GET MATSIZE_LIST 0 SIZEM)
      list(GET MATSIZE_LIST 1 SIZEK)
      list(GET MATSIZE_LIST 2 SIZEN)
      list(GET MATSIZE_LIST 3 SIZENNZ)

      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEK}x${SIZEN}x${SIZENNZ}-1s ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --stage=0 --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEK}x${SIZEN}x${SIZENNZ}-2s ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --stage=1 --verify=1 --iters=1)
    endforeach(MATSIZE)

  endforeach(PREC)
endforeach(FUNCTION)

# matrix sizes for ILU tests
set(ILU_SQUARE_MATRIX_SIZES
  "1 1 1"
  "4 4 6"
  "47 47 110"
  "47 47 513"
  "100 100 500"
  "103 103 499"
  "1000 1000 2000"
)
foreach(FUNCTION "ilu")
  foreach(PREC "d" "s")
    foreach(MATSIZE ${ILU_SQUARE_MATRIX_SIZES})
      # split MATSIZE string to individual tokens
      string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
      list(GET MATSIZE_LIST 0 SIZEM)
      list(GET MATSIZE_LIST 1 SIZEN)
      list(GET MATSIZE_LIST 2 SIZENNZ)

      add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ} ../staging/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --verify=1 --iters=1)

    endforeach(MATSIZE)
  endforeach(PREC)
endforeach(FUNCTION)

# Add extra function on matrix sizes as needed
# For example, excersice nonstandard multipliers alpha, beta
#add_test(FuncTest.csrmv-d-100x99x333-mlt ../staging/aoclsparse-bench --function=csrmv --precision=d --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)


