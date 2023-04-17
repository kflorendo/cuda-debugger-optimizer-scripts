#!/bin/bash

# example of running this script:
# ALL COMMAND LINE ARGUMENTS ARE ABSOLUTE PATHS
# -m = directory that contains the Makefile that compiles the code
# -r = terminal command that runs the code
# -c = path to .cu file
# ./threadOutput.sh -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu"

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

# save current contents to temp file
tempCu="temp-$RANDOM.cu"
cat $cuPath > tempCu

# add a print statement
insertLine=70
sed -i "${insertLine} i printf(\"hello from bash script!\");" $cuPath
# cat $cuPath

(cd $makeDir && make)
$runCmd

# copy back original contents
cp tempCu $cuPath
rm tempCu