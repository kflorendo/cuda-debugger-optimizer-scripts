#!/bin/bash

while getopts r: flag
do
    case "${flag}" in
        r) runCmd=${OPTARG};;
    esac
done

rm -f output/timestat.png
rm -f output/timedynam.png
rm -f output/timesize.png
rm -f output/timethru.png


nvprof --print-gpu-trace --csv --log-file output/memoryBottleneckGpuTrace.txt --normalized-time-unit us $runCmd

# python3 memoryBottleneck.py
