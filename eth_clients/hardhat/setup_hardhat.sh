#!/bin/bash
CORES=44
for ((i=0;i<CORES;i++)); do
    PORT=$(( 8601 + $i ))
    dir=$PORT
    mkdir -p $dir
    cd $dir
    rm -rf cache/
    rm -rf node_modules/
    cp ../hardhat.config.js ./
    cp ../package-lock.json ./
    npm install
    cd ..
	# eval $cmd
done
mkdir -p logs