#!/bin/bash
CORES=44
for ((i=0;i<CORES;i++)); do
	PORT=$(( 8601 + $i ))
	echo $PORT
	pid=`ps -ef | grep hardhat | grep $PORT | awk {'print $2'}`
	echo $pid
	renice -10 $pid
done
