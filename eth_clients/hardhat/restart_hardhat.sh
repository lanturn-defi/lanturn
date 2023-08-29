#!/bin/bash
PORT=$1
pid=`ps -ef | grep npm | grep hardhat | grep $PORT | awk {'print $2'}`
echo $pid
kill -s SIGTERM $pid
nohup npx hardhat node --port $PORT &> logs/hardhat.$PORT &
