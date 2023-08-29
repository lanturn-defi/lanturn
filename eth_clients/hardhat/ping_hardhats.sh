#!/bin/bash
CORES=44
for ((i=0;i<CORES;i++)); do
	PORT=$(( 8601 + $i ))
	echo $PORT
	curl -X POST localhost:$PORT --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'  --header 'Content-Type: application/json'
	echo ""
done
