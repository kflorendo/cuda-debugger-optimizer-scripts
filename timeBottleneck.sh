#!/bin/bash

while getopts m:r:c: flag
do
    case "${flag}" in
        m) makeDir=${OPTARG};;
        r) runCmd=${OPTARG};;
        c) cuPath=${OPTARG};;
    esac
done
echo "Make dir: $makeDir";
echo "Run command: $runCmd";
echo "Path to .cu file: $cuPath"; 

(cd $makeDir && make)

# Run code, creates gmon
$runCmd

execute=${runCmd%% *}
#gprof call
gprof ${execute} gmon.out > output/results.txt

# python3 gprofTime.py
