#!/bin/bash

while getopts r: flag
do
    case "${flag}" in
        r) runCmd=${OPTARG};;
    esac
done

nvprof --print-gpu-trace --csv --log-file output/memoryBottleneckGpuTrace.txt --normalized-time-unit us $runCmd

python3 memoryBottleneck.py
