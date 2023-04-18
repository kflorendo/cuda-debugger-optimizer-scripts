#!/bin/bash

# example of running this script:
# ALL COMMAND LINE ARGUMENTS ARE ABSOLUTE PATHS
# -m = directory that contains the Makefile that compiles the code
# -r = terminal command that runs the code
# -c = path to .cu file
# ./threadOutput.sh -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu"

while getopts m:r:c:v:s:e: flag
do
    case "${flag}" in
        m) makeDir=${OPTARG};;
        r) runCmd=${OPTARG};;
        c) cuPath=${OPTARG};;
        v) varName=${OPTARG};;
        s) startLine=${OPTARG};;
        e) endLine=${OPTARG};;
    esac
done
echo "Make dir: $makeDir";
echo "Run command: $runCmd";
echo "Path to .cu file: $cuPath"; 
echo "Variable tracked: $varName";
echo "Starting at line: $startLine";
echo "Ending at line: $endLine";

# save current contents to temp file
tempCu="temp-$RANDOM.cu"
cat $cuPath > tempCu

# add a print statement
insertLine=1
# add print statements into file
sed -i "${insertLine} i printf(\"blockIdx threadIdx address\");" $cuPath
sed -i "${endLine} i printf(\"%d %d %d\", blockIdx, threadIdx, ${varName})" $cuPath
# address column needed when var input is array (include index, multiple addresses)

(cd $makeDir && make)
# write to file
$runCmd > out.txt

# associative array (dict)
declare -A written

while IFS= read -r line; do
    data="$line"
    stringarray=($data)
    written[${stringarray[2]}]="${written[${stringarray[2]}]}${written[${stringarray[2]}]:+,}(${stringarray[0]}, ${stringarray[1]})"
done < out.txt

#source: https://stackoverflow.com/questions/27832452/associate-multiple-values-for-one-key-in-array-in-bash

for key in "${!written[@]}"
do 
    if [ size("$key->${array[$key]}") -gt 1 ]
    then
        echo "address ${varName} written to by ${$key->${array[$key]}}"
    fi
done


# do overwriting for 1 variable and starting/ending line number
# insert at a line "insert _address of variable_ into file"
# dictionary mapping memory address, address __ written to by [(blockIdx b, threadIdx t), (blockIdx b, threadIdx t)]
# printf( %d %d %d, threadIdx.x, etc)
# output file is 3 columns  (blockidx, threadidx, address value)

# copy back original contents
cp tempCu $cuPath
rm tempCu
