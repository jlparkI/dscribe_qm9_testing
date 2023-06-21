#!/bin/bash

mkdir qm9_data
cd qm9_data

if [ -f "qm9_.csv" ]; then
    echo "qm9 csv already downloaded."
else
    wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
    mv qm9.csv qm9_.csv
fi

if [ -d "cleaned_qm9_mols" ]; then
    echo "qm9 mols already downloaded and cleaned."
else
    mkdir qm9_mols
    cd qm9_mols
    wget https://figshare.com/ndownloader/files/3195389
    mv 3195389 3195389.tar.bz2
    tar -xf 3195389.tar.bz2
    rm 3195389.tar.bz2
    cd ..
    python ../scripts/molcleaner.py $PWD
    rm -rf qm9_mols
fi

python ../scripts/soap_maker.py
