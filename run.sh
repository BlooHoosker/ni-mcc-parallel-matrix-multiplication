#!/bin/bash

make -B

echo "========== Running Mult Simple =========="
./matmul $1 $1 5 1 | tee run.log

for thr in 2 4 8 12
do 
  for alg in 0 1 2 3 4 
  do
    echo "========== Running algorithm $alg =========="
    for limit in 256 512 1024 2048
    do
	echo "======= Limit $limit ======="
  	./matmul $1 $limit $alg $thr | tee -a run_$1_$thr.log
    done
  done
done
