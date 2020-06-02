#!/usr/bin/env bash

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

path=../../build/tests/staging

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
logname=dcsrmv_$(date +'%Y%m%d%H%M%S').csv
truncate -s 0 $logname

# Run csrmv for all matrices in ./matrices/matrixlist
while IFS= read -r filename; do
    echo "numactl --physcpubind=4 $bench -f csrmv --precision d --alpha 1 --beta 0 --iters 1 --mtx $filename --verify 1"
    numactl --physcpubind=4 $bench -f csrmv --precision d --alpha 1 --beta 0 --iters 1 --mtx $filename --verify 1 2>&1 | tee -a $logname
done < ./matrices/matrixlist
