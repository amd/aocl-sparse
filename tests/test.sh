#Set AOCLSPARSE_ROOT to library installed path
#export AOCLSPARSE_ROOT=<path-to-library-installaton>
#Add aoclsparse shared library path to env path variable $LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=<path-to-library-installaton>/lib:$LD_LIBRARY_PATH
g++ -O3 -DNDEBUG sample_csrmv.cpp -I $AOCLSPARSE_ROOT/include/ -L$AOCLSPARSE_ROOT/lib/ -laoclsparse -o test
