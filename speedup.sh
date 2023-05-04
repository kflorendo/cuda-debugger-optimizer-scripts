#!/bin/bash

while getopts s:p: flag
do
    case "${flag}" in
        s) seqRun=${OPTARG};;
        p) paraRun=${OPTARG};;
    esac
done
echo "Sequential File Path: $seqRun";
echo "Parallel File Path: $paraRun";

start_time=$(date +%s.%6N)
$seqRun
end_time=$(date +%s.%6N)
elapsed=$(echo "scale=6; ${end_time} - ${start_time}" | bc)

start_time2=$(date +%s.%6N)
$paraRun
end_time2=$(date +%s.%6N)
elapsed2=$(echo "scale=6; ${end_time2} - ${start_time2}" | bc)

bc -l <<< "${elapsed}/${elapsed2}"