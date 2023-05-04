#!/bin/bash

while getopts r: flag
do
    case "${flag}" in
        r) runCmd=${OPTARG};;
    esac
done

nvprof --print-gpu-trace --csv --log-file output/timeBottleneckGpuTrace.txt --normalized-time-unit us $runCmd
nvprof --print-api-trace --csv --log-file output/timeBottleneckApiTrace.txt --normalized-time-unit us $runCmd
