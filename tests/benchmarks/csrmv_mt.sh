#!/usr/bin/env bash
# ########################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc.
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

# Helper function
function display_help()
{
    echo "aoclSPARSE benchmark helper script"
    echo "    [-h|--help] prints this help message"
    echo "    [-p|--path] path to aoclsparse-bench"
}

# Check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
    echo "This script uses getopt to parse arguments; try installing the util-linux package";
    exit 1;
fi

path=../../build/release/tests/staging

# Parse command line parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,device:,path: --options hd:p: -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
        -h|--help)
            display_help
            exit 0
            ;;
        -p|--path)
            path=${2}
            shift 2 ;;
        --) shift ; break ;;
        *)  echo "Unexpected command line parameter received; aborting";
            exit 1
            ;;
    esac
done

bench=$path/aoclsparse-bench

# Check if binary is available
if [ ! -f $bench ]; then
    echo $bench not found, exit...
    exit 1
else
    echo ">>" $(realpath $(ldd $bench | grep aoclsparse | awk '{print $3;}'))
fi

# Generate logfile name
logname=mt_dcsrmv_$(date +'%Y%m%d%H%M%S').csv
truncate -s 0 $logname

# Run csrmv for all matrices in ./matrices/matrixlist
while IFS= read -r filename; do
    echo "AOCLSPARSE_NUM_THREADS=4 numactl --physcpubind=4,5,6,7 $bench --function=csrmv --precision=d --alpha=1 --beta=0 --iters=100 --mtx=$filename --verify=1"
    AOCLSPARSE_NUM_THREADS=4 numactl --physcpubind=4,5,6,7 $bench --function=csrmv --precision=d --alpha=1 --beta=0 --iters=100 --mtx=$filename --verify=1 2>&1 | tee -a $logname
done < ./matrices/matrixlist
