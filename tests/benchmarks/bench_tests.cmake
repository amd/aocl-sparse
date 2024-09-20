# ########################################################################
# Copyright (c) 2024 Advanced Micro Devices, Inc.
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

# Add selected matrix operations on small matrices as test targets

# Run all functions on these sizes
# set common matrix sizes for square matrices "m n nnz"
set(SQMATRIXSIZES
  "1 1 1"
  "2 2 2"
  "5 5 12"
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

# set matrix with different sizes for ADD "m n nnz nnzB"
set(ADDMATRIXSIZES
  "1 99 77 21"
  "103 1 83 63"
  "4 53 110 73"     # more than 5% density
  "53 3 111 111"     # more than 7% density
  "48 48 513 432"    # more than 23% density
  "53 43 271 180"    # more than 12% density
)

message(STATUS "Dependencies of bench-tests (libraries and binary)")
message(STATUS "  \$Path to Sparse executable = ${AOCLSPARSE_BENCH_PATH}")

foreach(FUNCTION "csrmv" "ellmv" "diamv" "optmv") # TODO add back "csrsv", so far failing the tests
  foreach(PREC "d" "s")
    foreach(BASE "0" "1")      #test for base-0 and base-1

      foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
        # split MATSIZE string to individual tokens
        string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
        list(GET MATSIZE_LIST 0 SIZEM)
        list(GET MATSIZE_LIST 1 SIZEN)
        list(GET MATSIZE_LIST 2 SIZENNZ)

        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --transposeA=N --verify=1 --iters=1)
        if(FUNCTION STREQUAL "csrmv" OR FUNCTION STREQUAL "optmv")
          # Add transpose tests
          add_test(FuncTest.${FUNCTION}T-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --transposeA=T --verify=1 --iters=1)

          # check full diagonal cases in square matrices
          IF(SIZEM STREQUAL SIZEN)
            add_test(FuncTest.${FUNCTION}-FullDiag-Sorted-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --transposeA=T --matrix=D --sort=F --verify=1 --iters=1)
            add_test(FuncTest.${FUNCTION}-FullDiag-Unsorted-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --transposeA=T --matrix=D --sort=U --verify=1 --iters=1)
            add_test(FuncTest.${FUNCTION}-NonFullDiag-Unsorted-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --transposeA=T --matrix=R --sort=F --verify=1 --iters=1)
            #Partial Sorted
            add_test(FuncTest.${FUNCTION}-FullDiag-PartialSorted-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --transposeA=T --matrix=D --sort=P --verify=1 --iters=1)
          ENDIF()
        ENDIF()

      endforeach(MATSIZE)

      # Add extra tests to exercise nonstandard multipliers alpha, beta
      add_test(FuncTest.${FUNCTION}-${PREC}-100x99x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=N --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-21x49x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=N --verify=1 --iters=1)
      if(FUNCTION STREQUAL "csrmv" OR FUNCTION STREQUAL "optmv")
        # Add transpose tests
        add_test(FuncTest.${FUNCTION}T-${PREC}-100x99x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=T  --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}T-${PREC}-21x49x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=T  --verify=1 --iters=1)
      ENDIF()
    endforeach(BASE)
  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "optmv")
  foreach(PREC "c" "z")
    foreach(BASE "0" "1")      #test for base-0 and base-1

      foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
        # split MATSIZE string to individual tokens
        string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
        list(GET MATSIZE_LIST 0 SIZEM)
        list(GET MATSIZE_LIST 1 SIZEN)
        list(GET MATSIZE_LIST 2 SIZENNZ)

        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --transposeA=N --verify=1 --iters=1)

      endforeach(MATSIZE)

      # Add extra tests to exercise nonstandard multipliers alpha, beta
      add_test(FuncTest.${FUNCTION}-${PREC}-100x99x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=N --verify=1 --iters=1)
      add_test(FuncTest.${FUNCTION}-${PREC}-21x49x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --transposeA=N --verify=1 --iters=1)
    endforeach(BASE)
  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "blkcsrmv")
  foreach(PREC "d")
    foreach(BASE "0" "1")      #test for base-0 and base-1
      foreach(BLK "1" "2" "4")

        foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
          # split MATSIZE string to individual tokens
          string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
          list(GET MATSIZE_LIST 0 SIZEM)
          list(GET MATSIZE_LIST 1 SIZEN)
          list(GET MATSIZE_LIST 2 SIZENNZ)
          if(SIZEN GREATER_EQUAL 8)
            add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizeblk=${BLK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --sort=F --verify=1 --iters=1)
          endif()
        endforeach(MATSIZE)

        # Add extra tests to exercise nonstandard multipliers alpha, beta
        add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-100x99x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizeblk=${BLK} --indexbaseA=${BASE} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --sort=F --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-21x49x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --sizeblk=${BLK} --indexbaseA=${BASE} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --sort=F --verify=1 --iters=1)

        endforeach(BLK)

        #adding tests with incorrect block size
        add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-100x99x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizeblk=3 --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --sort=F --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-21x49x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizeblk=7 --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --sort=F --verify=1 --iters=1)

      endforeach(BASE)
  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "bsrmv")
  foreach(PREC "d" "s")
    foreach(BASE "0" "1")      #test for base-0 and base-1
      foreach(BLK "2" "3" "4")

        foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
          # split MATSIZE string to individual tokens
          string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
          list(GET MATSIZE_LIST 0 SIZEM)
          list(GET MATSIZE_LIST 1 SIZEN)
          list(GET MATSIZE_LIST 2 SIZENNZ)

          add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --blockdim=${BLK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --verify=1 --iters=1)

        endforeach(MATSIZE)

        # Add extra tests to exercise nonstandard multipliers alpha, beta
        add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-100x99x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --blockdim=${BLK} --indexbaseA=${BASE} --sizem=100 --sizen=99 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-${BLK}-21x49x333-mlt-Base-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench --function=${FUNCTION} --precision=${PREC} --blockdim=${BLK} --indexbaseA=${BASE} --sizem=21 --sizen=49 --sizennz=333 --alpha=3 --beta=-1.5 --verify=1 --iters=1)

      endforeach(BLK)
    endforeach(BASE)
  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "csrmm")
  foreach(PREC "d" "s" "c" "z")
    foreach(BASE "0" "1")      #test for base-0 and base-1
      foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
        # split MATSIZE string to individual tokens
        string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
        list(GET MATSIZE_LIST 0 SIZEM)
        list(GET MATSIZE_LIST 1 SIZEN)
        list(GET MATSIZE_LIST 2 SIZENNZ)

        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZEN}x${SIZENNZ}xBase-${BASE}-co ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEN} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --order=1 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZEN}x${SIZENNZ}xBase-${BASE}-ro ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEN} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --order=0 --verify=1 --iters=1)
      endforeach(MATSIZE)

      foreach(MATSIZE ${RCTMATRIXSIZESMM})
        # split MATSIZE string to individual tokens
        string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
        list(GET MATSIZE_LIST 0 SIZEM)
        list(GET MATSIZE_LIST 1 SIZEK)
        list(GET MATSIZE_LIST 2 SIZEN)
        list(GET MATSIZE_LIST 3 SIZENNZ)

        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEK}x${SIZEN}x${SIZENNZ}xBase-${BASE}-co ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --order=1 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEK}x${SIZEN}x${SIZENNZ}xBase-${BASE}-ro ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --order=0 --verify=1 --iters=1)

      endforeach(MATSIZE)

        # Add extra tests to exercise nonstandard multipliers alpha, beta
        add_test(FuncTest.${FUNCTION}-${PREC}-103x103x103x499xBase-${BASE}-co-mlt ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC}  --indexbaseA=${BASE} --sizek=103 --sizem=103 --sizen=103 --sizennz=499 --order=1 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-103x103x103x499xBase-${BASE}-ro-mlt ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=103 --sizem=103 --sizen=103 --sizennz=499 --order=0 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-103x103x103x499xBase-${BASE}-co-mlt1 ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=103 --sizem=103 --sizen=103 --sizennz=499 --order=1 --alpha=2 --beta=0 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-103x103x103x499xBase-${BASE}-ro-mlt1 ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=103 --sizem=103 --sizen=103 --sizennz=499 --order=0 --alpha=2 --beta=0 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-42x42x42x213xBase-${BASE}-co-mlt ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=42 --sizem=42 --sizen=42 --sizennz=213 --order=1 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-42x42x42x213xBase-${BASE}-ro-mlt ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=42 --sizem=42 --sizen=42 --sizennz=213 --order=0 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-42x42x42x213xBase-${BASE}-co-mlt1 ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=42 --sizem=42 --sizen=42 --sizennz=213 --order=1 --alpha=2 --beta=0 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-42x42x42x213xBase-${BASE}-ro-mlt1 ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=42 --sizem=42 --sizen=42 --sizennz=213 --order=0 --alpha=2 --beta=0 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-81x81x81x815xBase-${BASE}-co-mlt ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=81 --sizem=81 --sizen=81 --sizennz=815 --order=1 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-81x81x81x815xBase-${BASE}-ro-mlt ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=81 --sizem=81 --sizen=81 --sizennz=815 --order=0 --alpha=3 --beta=-1.5 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-81x81x81x815xBase-${BASE}-co-mlt1 ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=81 --sizem=81 --sizen=81 --sizennz=815 --order=1 --alpha=2 --beta=0 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-81x81x81x815xBase-${BASE}-ro-mlt1 ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --indexbaseA=${BASE} --sizek=81 --sizem=81 --sizen=81 --sizennz=815 --order=0 --alpha=2 --beta=0 --verify=1 --iters=1)
    endforeach(BASE)
  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "csr2m")
  foreach(PREC "d" "s")
    foreach(BASE "0" "1")      #test for base-0 and base-1
      foreach(MATSIZE ${SQMATRIXSIZES} ${RCTMATRIXSIZES})
        # split MATSIZE string to individual tokens
        string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
        list(GET MATSIZE_LIST 0 SIZEM)
        list(GET MATSIZE_LIST 1 SIZEN)
        list(GET MATSIZE_LIST 2 SIZENNZ)

        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZEM}x${SIZENNZ}xBase-${BASE}-1s ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEN} --sizem=${SIZEM} --sizen=${SIZEM} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --stage=0 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEN}x${SIZEM}x${SIZENNZ}xBase-${BASE}-2s ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEN} --sizem=${SIZEM} --sizen=${SIZEM} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --stage=1 --verify=1 --iters=1)

      endforeach(MATSIZE)

      foreach(MATSIZE ${RCTMATRIXSIZESMM})
        # split MATSIZE string to individual tokens
        string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
        list(GET MATSIZE_LIST 0 SIZEM)
        list(GET MATSIZE_LIST 1 SIZEK)
        list(GET MATSIZE_LIST 2 SIZEN)
        list(GET MATSIZE_LIST 3 SIZENNZ)

        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEK}x${SIZEN}x${SIZENNZ}xBase-${BASE}-1s ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --stage=0 --verify=1 --iters=1)
        add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}x${SIZEK}x${SIZEN}x${SIZENNZ}xBase-${BASE}-2s ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEK} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --stage=1 --verify=1 --iters=1)

      endforeach(MATSIZE)

    endforeach(BASE)
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
      foreach(BASE "0" "1")   #test for base-0 and base-1
        foreach(MATSIZE ${ILU_SQUARE_MATRIX_SIZES})
          # split MATSIZE string to individual tokens
          string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
          list(GET MATSIZE_LIST 0 SIZEM)
          list(GET MATSIZE_LIST 1 SIZEN)
          list(GET MATSIZE_LIST 2 SIZENNZ)

          add_test(FuncTest.${FUNCTION}-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrix=D --sort=F --verify=1 --iters=1)

      endforeach(MATSIZE)
    endforeach(BASE)
  endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "trsv")
  foreach(PREC "d" "s" "c" "z")
      foreach(BASE "0" "1")   #test for base-0 and base-1
        foreach(MATSIZE ${SQMATRIXSIZES})
          # split MATSIZE string to individual tokens
          string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
          list(GET MATSIZE_LIST 0 SIZEM)
          list(GET MATSIZE_LIST 1 SIZEN)
          list(GET MATSIZE_LIST 2 SIZENNZ)
          # TRSV-L
          add_test(FuncTest.${FUNCTION}-L-FullDiag-Sorted-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrixtypeA=T --uplo=L --transposeA=N --matrix=D --sort=F --verify=1 --iters=1)
          add_test(FuncTest.${FUNCTION}-L-FullDiag-Unsorted-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrixtypeA=T --uplo=L --transposeA=N --matrix=D --sort=U --verify=1 --iters=1)
          # TRSV-LT
          add_test(FuncTest.${FUNCTION}-LT-FullDiag-Sorted-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrixtypeA=T --uplo=L --transposeA=T --matrix=D --sort=F --verify=1 --iters=1)
          add_test(FuncTest.${FUNCTION}-LT-FullDiag-Unsorted-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrixtypeA=T --uplo=L --transposeA=T --matrix=D --sort=U --verify=1 --iters=1)
          # TRSV-U
          add_test(FuncTest.${FUNCTION}-U-FullDiag-Sorted-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrixtypeA=T --uplo=U --transposeA=N --matrix=D --sort=F --verify=1 --iters=1)
          add_test(FuncTest.${FUNCTION}-U-FullDiag-Unsorted-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrixtypeA=T --uplo=U --transposeA=N --matrix=D --sort=U --verify=1 --iters=1)
          # TRSV-UT
          add_test(FuncTest.${FUNCTION}-UT-FullDiag-Sorted-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrixtypeA=T --uplo=U --transposeA=T --matrix=D --sort=F --verify=1 --iters=1)
          add_test(FuncTest.${FUNCTION}-UT-FullDiag-Unsorted-${PREC}-${SIZEM}-${SIZEM}x${SIZEN}x${SIZENNZ}xBase-${BASE} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizek=${SIZEM} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --indexbaseA=${BASE} --matrixtypeA=T --uplo=U --transposeA=T --matrix=D --sort=U --verify=1 --iters=1)
     endforeach(MATSIZE)
   endforeach(BASE)
 endforeach(PREC)
endforeach(FUNCTION)

foreach(FUNCTION "add")
  foreach(PREC "d" "s" "c" "z")
    foreach(BASEA "0" "1")      #test for Mat A base-0 and base-1
      foreach(BASEB "0" "1")      #test for Mat B base-0 and base-1
        foreach(OP "N" "T")      # operation none/transpose
          foreach(MATSIZE ${ADDMATRIXSIZES})
            # split MATSIZE string to individual tokens
            string(REGEX MATCHALL "[0-9]+" MATSIZE_LIST ${MATSIZE})
            list(GET MATSIZE_LIST 0 SIZEM)
            list(GET MATSIZE_LIST 1 SIZEN)
            list(GET MATSIZE_LIST 2 SIZENNZ)
            list(GET MATSIZE_LIST 3 SIZENNZB)

            add_test(FuncTest.${FUNCTION}${OP}-${PREC}-${SIZEM}x${SIZEN}x${SIZENNZ}xBaseA-${BASEA}+${SIZENNZB}xBaseB-${BASEB} ${AOCLSPARSE_BENCH_PATH}/aoclsparse-bench  --function=${FUNCTION} --precision=${PREC} --sizem=${SIZEM} --sizen=${SIZEN} --sizennz=${SIZENNZ} --sizennzB=${SIZENNZB} --transposeA=${OP} --indexbaseA=${BASEA} --indexbaseB=${BASEB} --stage=0 --verify=1 --iters=1)

          endforeach(MATSIZE)
        endforeach(OP)
      endforeach(BASEB)
    endforeach(BASEA)
  endforeach(PREC)
endforeach(FUNCTION)