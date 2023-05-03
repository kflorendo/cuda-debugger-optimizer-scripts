#!/bin/bash

# test case running example:
# ./threadOverwrite.sh \
#     -m "/afs/andrew.cmu.edu/usr19/huiningl/private/15418-asst2/cuda-debugger-optimizer-scripts" \
#     -r "/afs/andrew.cmu.edu/usr19/huiningl/private/15418-asst2/cuda-debugger-optimizer-scripts/test1" \
#     -c "/afs/andrew.cmu.edu/usr19/huiningl/private/15418-asst2/cuda-debugger-optimizer-scripts/test1.cu" \
#     -v "darr[idx]" \
#     -l 14 \
#     -t "int" \
#     -a y

# example of running this script:
# ALL COMMAND LINE ARGUMENTS ARE ABSOLUTE PATHS
# -m = directory that contains the Makefile that compiles the code
# -r = terminal command that runs the code
# -c = path to .cu file
# ./threadOutput.sh -m "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan" -r "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/cudaScan -m scan -i random -n 100" -c "/afs/andrew.cmu.edu/usr16/kflorend/private/15418/project/15418-asst2/scan/scan.cu"

while getopts m:r:c:v:l:t:a: flag
do
    case "${flag}" in
        m) makeDir=${OPTARG};;
        r) runCmd=${OPTARG};;
        c) cuPath=${OPTARG};;
        v) varName=${OPTARG};;
        l) lineNum=${OPTARG};;
        t) varType=${OPTARG};;
        a) arrayOpt=${OPTARG};;
    esac
done
echo "Make dir: $makeDir";
echo "Run command: $runCmd";
echo "Path to .cu file: $cuPath"; 
echo "Variable tracked: $varName";
echo "Insert at line: $lineNum";
echo "Variable type: $varType";
echo "Is array: $arrayOpt";

# save current contents to temp file
tempCu="temp-$RANDOM.cu"
cat $cuPath > tempCu

# get print prefix
prefix="cuda-debug-optimize$(date +%s)"

# add a print statement
temp=${varName}
# add print statements into file
# sed -i "${insertLine} i printf(\"blockIdx threadIdx address\");" $cuPath
case $varType in
    "int") varFormat="%d";;
    "float") varFormat="%f";;
    "string") varFormat="%s";;
esac

case $arrayOpt in
    "y") 
        index=${varName#*'['} #remove everything before [
        index=${index%']'*};;   #remove everything after ]
    "n") 
        index=0;;
esac

sed -i "${lineNum} i printf(\"${prefix} %d %d %d %d %d %d %p ${varFormat} %d\\\\n\", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, &${temp}, ${temp}, ${index});" $cuPath
# address column needed when var input is array (include index, multiple addresses)


(cd $makeDir && make)
# write to file
$runCmd > out.txt

# filter output file
sed -i "/^${prefix}/!d" out.txt
sed -i "s/${prefix} //" out.txt

# associative array (dict)
declare -A written

while IFS= read -r line; do
    data="$line"
    stringarray=($data)
    written[${stringarray[6]}]="${written[${stringarray[6]}]}${written[${stringarray[6]}]:+ }${stringarray[0]},${stringarray[1]},${stringarray[2]},${stringarray[3]},${stringarray[4]},${stringarray[5]},${stringarray[8]},${stringarray[7]}"
done < out.txt

#source: https://stackoverflow.com/questions/27832452/associate-multiple-values-for-one-key-in-array-in-bash

touch output/threadOverwrite.txt
> output/threadOverwrite.txt
for key in "${!written[@]}"
do 
    count=$(grep -o ',' <<< ${written[$key]} | wc -l)
    count=$(((count + 1) / 2))
    if [ $count -gt 1 ]
    then
        echo "${key} ${written[$key]}" >> output/threadOverwrite.txt
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
rm out.txt
