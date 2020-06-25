#!/usr/bin/env bash
# Author: Nico Trost

matrices=(ldoor
          bone010
          Rucci1
          rajat31
          crankseg_2
          bibd_22_8
          sls
          Chebyshev4
)

url=(https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/Oberwolfach
     https://sparse.tamu.edu/MM/Rucci
     https://sparse.tamu.edu/MM/Rajat
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/JGD_BIBD
     https://sparse.tamu.edu/MM/Bates
     https://sparse.tamu.edu/MM/Muite
)

for i in {0..7}; do
    m=${matrices[${i}]}
    u=${url[${i}]}
    if [ ! -f ${m}.csr ]; then
        if [ ! -f ${m}.mtx ]; then
            if [ ! -f ${m}.tar.gz ]; then
                echo "Downloading ${m}.tar.gz ..."
                wget ${u}/${m}.tar.gz
            fi
            echo "Extracting ${m}.tar.gz ..."
            tar xf ${m}.tar.gz && mv ${m}/${m}.mtx . && rm -rf ${m}.tar.gz ${m}
        fi
    fi
done
