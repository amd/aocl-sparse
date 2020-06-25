#!/usr/bin/env bash
# Author: Nico Trost

matrices=(boyd2
          Rucci1
          transient
          ASIC_680k
          dc2
          ins2
          cop20k_A
          Si41Ge41H72
)

url=(https://sparse.tamu.edu/MM/GHS_indef
     https://sparse.tamu.edu/MM/Rucci
     https://sparse.tamu.edu/MM/Freescale
     https://sparse.tamu.edu/MM/Sandia
     https://sparse.tamu.edu/MM/IBM_EDA
     https://sparse.tamu.edu/MM/Andrianov
     https://sparse.tamu.edu/MM/Williams
     https://sparse.tamu.edu/MM/PARSEC
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
