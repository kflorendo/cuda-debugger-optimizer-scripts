#!/bin/bash

while getopts r:c: flag
do
    case "${flag}" in
        r) runCmd=${OPTARG};;
        c) cuPath=${OPTARG};;
    esac
done
echo "Run command: $runCmd";
echo "Path to .cu file: $cuPath"; 

# Compile CUDA code
nvcc -pg -o exe_prog $cuPath

# Run code
$runCmd

# gprof call
gprof exe_prog gmon.out > results.out

# display results
gprof2dot -f pstats results.out | dot -Tsvg -o results.svg

# pie chart?
xdg-open results.svg

