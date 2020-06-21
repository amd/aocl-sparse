#Set AOCLSPARSE_ROOT to library installed path
#export AOCLSPARSE_ROOT=<path-to-library-installaton>
g++ -O3 -DNDEBUG sample_csrmv.cpp -I $AOCLSPARSE_ROOT/include/ -L$AOCLSPARSE_ROOT/lib/ -laoclsparse -o test
