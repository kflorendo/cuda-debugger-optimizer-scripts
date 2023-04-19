#!/bin/bash

# example of running this script:
# ALL COMMAND LINE ARGUMENTS ARE ABSOLUTE PATHS
# -m = directory that contains the Makefile that compiles the code
# -r = terminal command that runs the code
# -c = path to .cu file
# ./threadOutput.sh -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu" -v "device_data[i+twod1-1]" -t "int" -l 63

while getopts m:r:c:v:t:l: flag
do
    case "${flag}" in
        m) makeDir=${OPTARG};;
        r) runCmd=${OPTARG};;
        c) cuPath=${OPTARG};;
        v) var=${OPTARG};;
        t) varType=${OPTARG};;
        l) insertLine=${OPTARG};;
    esac
done
echo "Make dir: $makeDir";
echo "Run command: $runCmd";
echo "Path to .cu file: $cuPath"; 

# save current contents to temp file
tempCu="temp-$RANDOM.cu"
cat $cuPath > tempCu

# get print prefix
prefix="cuda-debug-optimize$(date +%s)"

# add a print statement
# insertLine=63
# var="device_data[i+twod1-1]"
# varType="int"
case $varType in
    "int") varFormat="%d";;
    "float") varFormat="%f";;
    "string") varFormat="%s";;
esac
insertText="printf(\"${prefix} %d %d ${varFormat}\\\\n\", blockIdx.x, threadIdx.x, ${var});"
echo $insertText
sed -i "${insertLine} i ${insertText}" $cuPath

(cd $makeDir && make)
$runCmd > threadOutput.txt

sed -i "/^${prefix}/!d" threadOutput.txt

# copy back original contents
cp tempCu $cuPath
rm tempCu