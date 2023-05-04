#!/bin/bash

while getopts m:r:c:l: flag
do
    case "${flag}" in
        m) makeDir=${OPTARG};;
        r) runCmd=${OPTARG};;
        c) cuPath=${OPTARG};;
        l) lineNum=${OPTARG};;
    esac
done
echo "Make dir: $makeDir";
echo "Run command: $runCmd";
echo "Path to .cu file: $cuPath"; 
echo "Insert at line: $lineNum";

# save current contents to temp file
tempCu="temp-$RANDOM.cu"
cat $cuPath > tempCu

# get print prefix
prefix="cuda-debug-optimize$(date +%s)"

# add a print statement

sed -i "${lineNum} i printf(\"${prefix} %d %d %d %d %d %d\\\\n\", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);" $cuPath
# address column needed when var input is array (include index, multiple addresses)


(cd $makeDir && make)
# write to file
$runCmd > threadbp.txt

# filter output file
sed -i "/^${prefix}/!d" threadbp.txt
sed -i "s/${prefix} //" threadbp.txt

python3 sortThreads.py