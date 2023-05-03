#!/bin/bash

while getopts r: flag
do
    case "${flag}" in
        r) runCmd=${OPTARG};;
    esac
done

nvprof --print-gpu-trace --csv --log-file output/nvprof_memoryBottleneck.txt $runCmd
